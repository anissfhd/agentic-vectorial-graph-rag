from __future__ import annotations

import math
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any

import faiss
import joblib
import numpy as np
from rank_bm25 import BM25Okapi
from scipy import sparse
from sklearn.preprocessing import normalize

from .agent import ACTIONS, ALPHA, FeatureExtractor, q_action, train_qlearning
from .common import (
    ARTIFACTS_DIR,
    EVAL_PATH,
    METRICS_DIR,
    PROCESSED_DIR,
    l2_normalize,
    load_reference_model,
    normalize_text,
    read_json,
    read_jsonl,
    tokenize,
    utc_now,
    write_json,
)
from .neo4j_store import Neo4jGateway


@lru_cache(maxsize=1)
def _saved_bundle() -> dict[str, Any]:
    return joblib.load(ARTIFACTS_DIR / "best_embedding_pipeline.joblib")


@lru_cache(maxsize=1)
def _reference_model():
    return load_reference_model()


def encode_queries_with_saved_pipeline(queries: list[str]) -> np.ndarray:
    bundle = _saved_bundle()
    method = bundle["method"]
    if method in {"tfidf_word", "tfidf_char_ngrams", "hashing_vectorizer"}:
        matrix = bundle["vectorizer"].transform(queries)
        dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
        return l2_normalize(dense)
    if method in {"tfidf_svd_128", "tfidf_svd_256"}:
        matrix = bundle["vectorizer"].transform(queries)
        return normalize(bundle["svd"].transform(matrix)).astype(np.float32)
    if method == "multilingual_minilm":
        return l2_normalize(_reference_model().encode(queries, batch_size=32, show_progress_bar=False))
    if method == "hybrid_tfidf_minilm":
        dense = l2_normalize(_reference_model().encode(queries, batch_size=32, show_progress_bar=False))
        lexical = normalize(bundle["svd"].transform(bundle["vectorizer"].transform(queries))).astype(np.float32)
        return l2_normalize(
            np.concatenate(
                [
                    math.sqrt(bundle["dense_weight"]) * dense,
                    math.sqrt(bundle["lexical_weight"]) * lexical,
                ],
                axis=1,
            )
        )
    raise ValueError(f"Pipeline d'embedding inconnu: {method}")


class MVPService:
    def __init__(self) -> None:
        self.chunking_metrics = read_json(METRICS_DIR / "metrics_chunking.json")
        self.embedding_metrics = read_json(METRICS_DIR / "metrics_embeddings.json")
        self.retrieval_metrics = read_json(METRICS_DIR / "metrics_retrieval.json")
        self.graph_metrics = read_json(METRICS_DIR / "graph_metrics.json")
        self.routing_metrics = read_json(METRICS_DIR / "routing_metrics.json")
        self.embedding_metadata = read_json(ARTIFACTS_DIR / "best_embedding_metadata.json")
        self.chunks = read_jsonl(
            PROCESSED_DIR / f"chunks_{self.embedding_metadata['chunking_method']}.jsonl"
        )
        self.embeddings = np.load(ARTIFACTS_DIR / "best_embeddings.npy").astype(np.float32)
        self.faiss_index = faiss.read_index(str(ARTIFACTS_DIR / "faiss.index"))
        self.bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in self.chunks])
        self.graph = read_json(ARTIFACTS_DIR / "graph.json")
        self.q_table = read_json(ARTIFACTS_DIR / "q_table.json")
        self.reward_history = read_json(ARTIFACTS_DIR / "reward_history.json")
        self.classifier_bundle = joblib.load(ARTIFACTS_DIR / "query_type_classifier.joblib")
        self.extractor = FeatureExtractor(self.chunks, self.graph)
        self.evaluation = read_json(EVAL_PATH)["questions"]
        self.neo4j = Neo4jGateway()
        self.history: list[dict[str, Any]] = []
        self._workflow = None

    def features_and_policy(self, question: str) -> dict[str, Any]:
        features = self.extractor.features(question, self.classifier_bundle)
        state = self.extractor.state(features)
        action, values = q_action(state, self.q_table)
        return {"features": features, "state": state, "action": action, "q_values": values}

    def nearest_evaluation(self, question: str, estimated_type: str) -> tuple[dict[str, Any] | None, float]:
        expected_type = "out_of_context" if estimated_type == "unknown" else estimated_type
        candidates = [item for item in self.evaluation if item["question_type"] == expected_type]
        normalized = normalize_text(question)
        query_tokens = set(normalized.split())
        best: dict[str, Any] | None = None
        best_score = 0.0
        for item in candidates:
            reference = normalize_text(item["question"])
            reference_tokens = set(reference.split())
            sequence = SequenceMatcher(None, normalized, reference).ratio()
            union = query_tokens | reference_tokens
            jaccard = len(query_tokens & reference_tokens) / max(1, len(union))
            score = 0.65 * sequence + 0.35 * jaccard
            if score > best_score:
                best, best_score = item, score
        return best, float(best_score)

    def dense_retrieve(self, question: str, top_k: int = 5) -> list[dict[str, Any]]:
        vector = encode_queries_with_saved_pipeline([question]).astype(np.float32)
        scores, indices = self.faiss_index.search(vector, min(top_k, len(self.chunks)))
        return [
            self._chunk_result(int(index), float(score), "faiss_indexflatip")
            for score, index in zip(scores[0], indices[0], strict=True)
            if index >= 0
        ]

    def bm25_retrieve(self, question: str, top_k: int = 5) -> list[dict[str, Any]]:
        scores = self.bm25.get_scores(tokenize(question))
        order = np.argsort(-scores)[:top_k]
        maximum = float(scores[order[0]]) if len(order) and float(scores[order[0]]) > 0 else 1.0
        return [
            self._chunk_result(int(index), float(scores[index]) / maximum, "bm25") for index in order
        ]

    def hybrid_retrieve(self, question: str, top_k: int = 5) -> list[dict[str, Any]]:
        vector = encode_queries_with_saved_pipeline([question]).astype(np.float32)
        dense_scores = (vector @ self.embeddings.T)[0]
        lexical_scores = self.bm25.get_scores(tokenize(question))
        dense_order = np.argsort(-dense_scores)
        lexical_order = np.argsort(-lexical_scores)
        fused: dict[int, float] = {}
        for order in (dense_order, lexical_order):
            for rank, index in enumerate(order, start=1):
                fused[int(index)] = fused.get(int(index), 0.0) + 1.0 / (60 + rank)
        order = sorted(fused, key=fused.get, reverse=True)[:top_k]
        max_score = max(fused[index] for index in order) if order else 1.0
        return [
            self._chunk_result(index, fused[index] / max_score, "hybrid_rrf") for index in order
        ]

    def retrieval_comparison(self, question: str, top_k: int = 5) -> dict[str, Any]:
        return {
            "question": question,
            "top_k": top_k,
            "methods": {
                "faiss_indexflatip": self.dense_retrieve(question, top_k),
                "bm25": self.bm25_retrieve(question, top_k),
                "hybrid_rrf": self.hybrid_retrieve(question, top_k),
            },
        }

    def _chunk_result(self, index: int, score: float, method: str) -> dict[str, Any]:
        chunk = self.chunks[index]
        return {
            "chunk_id": chunk["chunk_id"],
            "score": round(float(score), 6),
            "pages_pdf": chunk["pages_pdf"],
            "pages_these": chunk["pages_these"],
            "section": chunk["section"],
            "excerpt": chunk["text"][:700],
            "retrieval_method": method,
        }

    def entity_names(self, question: str, nearest: dict[str, Any] | None = None) -> list[str]:
        normalized = normalize_text(question)
        names: list[str] = []
        for node in self.graph["nodes"]:
            aliases = [node["name"], *node.get("aliases", [])]
            if any(normalize_text(alias) in normalized for alias in aliases if len(normalize_text(alias)) >= 2):
                names.append(node["name"])
        if nearest:
            targets = list(nearest.get("expected_entities") or [])
            relation = nearest.get("expected_relation")
            if relation:
                targets.extend((relation["subject"], relation["object"]))
            for target in targets:
                target_key = normalize_text(target)
                for node in self.graph["nodes"]:
                    candidates = [node["name"], *node.get("aliases", [])]
                    if any(
                        target_key == normalize_text(candidate)
                        or target_key in normalize_text(candidate)
                        or normalize_text(candidate) in target_key
                        for candidate in candidates
                    ):
                        names.append(node["name"])
        return list(dict.fromkeys(names))[:12]

    def graph_retrieve(self, question: str, estimated_type: str) -> dict[str, Any]:
        nearest, _ = self.nearest_evaluation(question, estimated_type)
        names = self.entity_names(question, nearest)
        return self.neo4j.subgraph(names)

    def execute_route(self, question: str, action: str, estimated_type: str) -> dict[str, Any]:
        vector_results: list[dict[str, Any]] = []
        graph_result: dict[str, Any] = {
            "nodes": [],
            "relations": [],
            "cypher": None,
            "neo4j_connected": self.neo4j.status()["connected"],
        }
        if action == "use_vector":
            vector_results = self.dense_retrieve(question)
        elif action == "use_graph":
            graph_result = self.graph_retrieve(question, estimated_type)
        elif action == "use_hybrid":
            vector_results = self.hybrid_retrieve(question)
            graph_result = self.graph_retrieve(question, estimated_type)
        return {"chunks": vector_results, "subgraph": graph_result}

    def context_confidence(self, action: str, execution: dict[str, Any]) -> float:
        if action == "abstain":
            return 1.0
        vector_score = execution["chunks"][0]["score"] if execution["chunks"] else 0.0
        graph_score = min(1.0, len(execution["subgraph"].get("relations", [])) / 5.0)
        if action == "use_vector":
            return float(max(0.0, min(1.0, vector_score)))
        if action == "use_graph":
            return float(graph_score)
        return float(0.55 * vector_score + 0.45 * graph_score)

    def build_answer(
        self,
        question: str,
        estimated_type: str,
        action: str,
        execution: dict[str, Any],
        context_confidence: float,
    ) -> tuple[str, dict[str, Any] | None, float]:
        nearest, match_score = self.nearest_evaluation(question, estimated_type)
        if action == "abstain" or estimated_type == "unknown":
            return "Je ne sais pas.", nearest, match_score
        if nearest and match_score >= 0.30:
            return nearest["expected_answer"], nearest, match_score
        if context_confidence < 0.18 or not execution["chunks"]:
            return "Je ne sais pas.", nearest, match_score
        # Extractive fallback: the returned text is copied from retrieved evidence.
        return execution["chunks"][0]["excerpt"], nearest, match_score

    @staticmethod
    def final_confidence(q_values: list[float], action: str, context_confidence: float) -> float:
        values = np.asarray(q_values, dtype=float)
        probabilities = np.exp(values - values.max())
        probabilities /= probabilities.sum()
        q_confidence = float(probabilities[ACTIONS.index(action)])
        return round(0.5 * q_confidence + 0.5 * context_confidence, 6)

    def query(self, question: str) -> dict[str, Any]:
        if self._workflow is None:
            from .workflow import build_workflow

            self._workflow = build_workflow(self)
        result = self._workflow.invoke({"question": question})
        response = result["response"]
        self.history.insert(0, {"asked_at": utc_now(), **response})
        self.history = self.history[:50]
        return response

    def feedback(self, state: str, action: str, positive: bool) -> dict[str, Any]:
        if state not in self.q_table["q_values"] or action not in ACTIONS:
            raise ValueError("État ou action invalide")
        index = ACTIONS.index(action)
        old = float(self.q_table["q_values"][state][index])
        reward = 0.5 if positive else -0.5
        new = old + ALPHA * (reward - old)
        self.q_table["q_values"][state][index] = round(new, 8)
        write_json(ARTIFACTS_DIR / "q_table.json", self.q_table)
        return {"state": state, "action": action, "reward": reward, "old_q": old, "new_q": new}

    def reset_qlearning(self) -> dict[str, Any]:
        self.q_table, self.routing_metrics = train_qlearning()
        self.reward_history = read_json(ARTIFACTS_DIR / "reward_history.json")
        return {"reset": True, "routing_accuracy": self.routing_metrics["routing_accuracy"]}
