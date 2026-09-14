from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import faiss
import joblib
import numpy as np
from rank_bm25 import BM25Okapi
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from .common import (
    ARTIFACTS_DIR,
    EVAL_PATH,
    METRICS_DIR,
    PROCESSED_DIR,
    SEED,
    ir_metrics_for_rankings,
    l2_normalize,
    minmax,
    read_json,
    read_jsonl,
    tokenize,
    utc_now,
    write_json,
)


REPRESENTATIONS = (
    "tfidf_word",
    "tfidf_char_ngrams",
    "hashing_vectorizer",
    "tfidf_svd_128",
    "tfidf_svd_256",
    "multilingual_minilm",
    "hybrid_tfidf_minilm",
)


def _scores(doc_matrix: Any, query_matrix: Any) -> np.ndarray:
    product = query_matrix @ doc_matrix.T
    return product.toarray().astype(np.float32) if sparse.issparse(product) else np.asarray(product, dtype=np.float32)


def _fit_tfidf(docs: list[str], queries: list[str], analyzer: str = "word") -> tuple[Any, Any, Any]:
    if analyzer == "word":
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_features=25000,
            sublinear_tf=True,
            norm="l2",
        )
    else:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=30000,
            sublinear_tf=True,
            norm="l2",
        )
    doc_matrix = vectorizer.fit_transform(docs)
    return doc_matrix, vectorizer.transform(queries), vectorizer


def _fit_svd(
    docs: list[str], queries: list[str], components: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    tfidf, query_tfidf, vectorizer = _fit_tfidf(docs, queries, "word")
    actual = min(components, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
    svd = TruncatedSVD(n_components=actual, random_state=SEED)
    doc_matrix = normalize(svd.fit_transform(tfidf)).astype(np.float32)
    query_matrix = normalize(svd.transform(query_tfidf)).astype(np.float32)
    return doc_matrix, query_matrix, {"vectorizer": vectorizer, "svd": svd, "components": actual}


def fit_representation(
    method: str,
    docs: list[str],
    queries: list[str],
    model: Any,
) -> tuple[Any, Any, dict[str, Any], int, float]:
    start = time.perf_counter()
    bundle: dict[str, Any] = {"method": method}
    if method == "tfidf_word":
        doc_matrix, query_matrix, vectorizer = _fit_tfidf(docs, queries, "word")
        bundle["vectorizer"] = vectorizer
    elif method == "tfidf_char_ngrams":
        doc_matrix, query_matrix, vectorizer = _fit_tfidf(docs, queries, "char")
        bundle["vectorizer"] = vectorizer
    elif method == "hashing_vectorizer":
        vectorizer = HashingVectorizer(
            n_features=8192,
            alternate_sign=False,
            ngram_range=(1, 2),
            strip_accents="unicode",
            norm="l2",
        )
        doc_matrix = vectorizer.transform(docs)
        query_matrix = vectorizer.transform(queries)
        bundle["vectorizer"] = vectorizer
    elif method in {"tfidf_svd_128", "tfidf_svd_256"}:
        requested = 128 if method.endswith("128") else 256
        doc_matrix, query_matrix, fitted = _fit_svd(docs, queries, requested)
        bundle.update(fitted)
    elif method == "multilingual_minilm":
        doc_matrix = l2_normalize(model.encode(docs, batch_size=64, show_progress_bar=False))
        query_matrix = l2_normalize(model.encode(queries, batch_size=32, show_progress_bar=False))
    elif method == "hybrid_tfidf_minilm":
        dense_docs = l2_normalize(model.encode(docs, batch_size=64, show_progress_bar=False))
        dense_queries = l2_normalize(model.encode(queries, batch_size=32, show_progress_bar=False))
        lsa_docs, lsa_queries, fitted = _fit_svd(docs, queries, 128)
        doc_matrix = l2_normalize(
            np.concatenate([math.sqrt(0.85) * dense_docs, math.sqrt(0.15) * lsa_docs], axis=1)
        )
        query_matrix = l2_normalize(
            np.concatenate([math.sqrt(0.85) * dense_queries, math.sqrt(0.15) * lsa_queries], axis=1)
        )
        bundle.update(fitted)
        bundle["dense_weight"] = 0.85
        bundle["lexical_weight"] = 0.15
    else:
        raise ValueError(f"Représentation inconnue: {method}")
    elapsed = time.perf_counter() - start
    dimension = int(doc_matrix.shape[1])
    return doc_matrix, query_matrix, bundle, dimension, elapsed


def _positive_similarity(
    score_matrix: np.ndarray,
    chunks: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> float:
    values: list[float] = []
    for row, question in zip(score_matrix, questions, strict=True):
        pages = set(question["relevant_pages_pdf"])
        candidates = [float(row[index]) for index, chunk in enumerate(chunks) if pages.intersection(chunk["pages_pdf"])]
        if candidates:
            values.append(max(candidates))
    return float(np.mean(values)) if values else 0.0


def _inter_similarity(doc_matrix: Any) -> float:
    rng = np.random.default_rng(SEED)
    count = doc_matrix.shape[0]
    pairs = min(500, count * 2)
    left = rng.integers(0, count, pairs)
    right = rng.integers(0, count, pairs)
    valid = left != right
    if sparse.issparse(doc_matrix):
        products = doc_matrix[left[valid]].multiply(doc_matrix[right[valid]]).sum(axis=1)
        return float(np.mean(np.asarray(products).ravel()))
    return float(np.mean(np.sum(doc_matrix[left[valid]] * doc_matrix[right[valid]], axis=1)))


def run_embeddings(model: Any, chunking_report: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    winner = chunking_report["winner"]
    chunks = read_jsonl(PROCESSED_DIR / f"chunks_{winner}.jsonl")
    docs = [chunk["text"] for chunk in chunks]
    evaluation = read_json(EVAL_PATH)
    questions = [question for question in evaluation["questions"] if question["question_type"] != "out_of_context"]
    query_texts = [question["question"] for question in questions]
    results: list[dict[str, Any]] = []
    fitted: dict[str, tuple[Any, dict[str, Any]]] = {}

    for method in REPRESENTATIONS:
        doc_matrix, query_matrix, bundle, dimension, elapsed = fit_representation(method, docs, query_texts, model)
        score_matrix = _scores(doc_matrix, query_matrix)
        rankings = np.argsort(-score_matrix, axis=1).tolist()
        metrics = ir_metrics_for_rankings(rankings, chunks, questions, k=5)
        results.append(
            {
                "method": method,
                **metrics,
                "mean_positive_similarity": round(_positive_similarity(score_matrix, chunks, questions), 6),
                "inter_similarity": round(_inter_similarity(doc_matrix), 6),
                "encoding_seconds": round(elapsed, 6),
                "chunks_per_second": round(len(chunks) / max(elapsed, 1e-9), 3),
                "dimension": dimension,
            }
        )
        fitted[method] = (doc_matrix, bundle)

    speed = minmax([row["encoding_seconds"] for row in results], higher_is_better=False)
    for index, row in enumerate(results):
        row["selection_score"] = round(
            0.30 * row["f1_at_5"]
            + 0.25 * row["map"]
            + 0.20 * row["mrr"]
            + 0.20 * row["ndcg_at_5"]
            + 0.05 * speed[index],
            6,
        )
    winner_method = max(results, key=lambda row: (row["selection_score"], row["f1_at_5"]))["method"]
    winner_matrix, winner_bundle = fitted[winner_method]
    dense_winner = winner_matrix.toarray().astype(np.float32) if sparse.issparse(winner_matrix) else np.asarray(winner_matrix, dtype=np.float32)
    dense_winner = l2_normalize(dense_winner)
    np.save(ARTIFACTS_DIR / "best_embeddings.npy", dense_winner)
    joblib.dump(winner_bundle, ARTIFACTS_DIR / "best_embedding_pipeline.joblib")
    metadata = {
        "computed_at": utc_now(),
        "method": winner_method,
        "chunking_method": winner,
        "dimension": int(dense_winner.shape[1]),
        "chunks": len(chunks),
        "normalized": True,
        "matrix_file": "best_embeddings.npy",
        "pipeline_file": "best_embedding_pipeline.joblib",
    }
    write_json(ARTIFACTS_DIR / "best_embedding_metadata.json", metadata)
    report = {
        "computed_at": utc_now(),
        "chunking_method": winner,
        "evaluation_questions": len(questions),
        "k": 5,
        "selection_formula": "0.30*F1@5 + 0.25*MAP + 0.20*MRR + 0.20*NDCG@5 + 0.05*normalized_speed",
        "methods": results,
        "winner": winner_method,
    }
    write_json(METRICS_DIR / "metrics_embeddings.json", report)
    _plot_embeddings(results, winner_method)
    return report, dense_winner, chunks


def _plot_embeddings(results: list[dict[str, Any]], winner: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["method"].replace("_", "\n") for row in results]
    x = np.arange(len(labels))
    width = 0.25
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(x - width, [row["map"] for row in results], width, label="MAP", color="#4f46e5")
    axis.bar(x, [row["mrr"] for row in results], width, label="MRR", color="#0ea5e9")
    axis.bar(x + width, [row["ndcg_at_5"] for row in results], width, label="NDCG@5", color="#22c55e")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1.05)
    axis.set_title(f"Sept représentations vectorielles — gagnante: {winner}")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(METRICS_DIR / "embedding_comparison.png", dpi=160)
    plt.close(figure)


def build_vector_artifacts(
    embedding_report: dict[str, Any],
    embeddings: np.ndarray,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
    faiss.write_index(index, str(ARTIFACTS_DIR / "faiss.index"))

    tokenized = [tokenize(chunk["text"]) for chunk in chunks]
    write_json(
        ARTIFACTS_DIR / "bm25_data.json",
        {"computed_at": utc_now(), "chunk_count": len(chunks), "tokenized_corpus": tokenized},
    )

    pca = PCA(n_components=2, random_state=SEED)
    coordinates = pca.fit_transform(embeddings)
    cluster_count = min(8, max(2, int(round(math.sqrt(len(chunks) / 2)))))
    labels = KMeans(n_clusters=cluster_count, random_state=SEED, n_init=10).fit_predict(embeddings)
    points = [
        {
            "chunk_id": chunk["chunk_id"],
            "x": round(float(coordinates[index, 0]), 6),
            "y": round(float(coordinates[index, 1]), 6),
            "cluster": int(labels[index]),
            "pages_pdf": chunk["pages_pdf"],
            "section": chunk["section"],
            "excerpt": chunk["text"][:180],
        }
        for index, chunk in enumerate(chunks)
    ]
    write_json(
        ARTIFACTS_DIR / "pca_2d.json",
        {
            "computed_at": utc_now(),
            "explained_variance_ratio": [round(float(value), 6) for value in pca.explained_variance_ratio_],
            "clusters": cluster_count,
            "points": points,
        },
    )
    joblib.dump(pca, ARTIFACTS_DIR / "pca_model.joblib")

    retrieval = evaluate_retrieval(embeddings, chunks)
    write_json(METRICS_DIR / "metrics_retrieval.json", retrieval)
    return retrieval


def _rrf(dense_order: list[int], lexical_order: list[int], constant: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for order in (dense_order, lexical_order):
        for rank, index in enumerate(order, start=1):
            scores[index] = scores.get(index, 0.0) + 1.0 / (constant + rank)
    return sorted(scores, key=scores.get, reverse=True)


def evaluate_retrieval(embeddings: np.ndarray, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    evaluation = read_json(EVAL_PATH)
    questions = [question for question in evaluation["questions"] if question["question_type"] != "out_of_context"]
    from .runtime import encode_queries_with_saved_pipeline

    query_embeddings = encode_queries_with_saved_pipeline([question["question"] for question in questions])
    dense_scores = query_embeddings @ embeddings.T
    dense_rankings = np.argsort(-dense_scores, axis=1).tolist()

    bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])
    lexical_rankings: list[list[int]] = []
    lexical_start = time.perf_counter()
    for question in questions:
        scores = bm25.get_scores(tokenize(question["question"]))
        lexical_rankings.append(np.argsort(-scores).tolist())
    lexical_latency = (time.perf_counter() - lexical_start) * 1000 / len(questions)
    hybrid_start = time.perf_counter()
    hybrid_rankings = [_rrf(dense, lexical) for dense, lexical in zip(dense_rankings, lexical_rankings, strict=True)]
    hybrid_latency = (time.perf_counter() - hybrid_start) * 1000 / len(questions)

    dense_metrics = ir_metrics_for_rankings(dense_rankings, chunks, questions, k=5)
    bm25_metrics = ir_metrics_for_rankings(lexical_rankings, chunks, questions, k=5)
    hybrid_metrics = ir_metrics_for_rankings(hybrid_rankings, chunks, questions, k=5)
    return {
        "computed_at": utc_now(),
        "methods": [
            {"method": "cosine_exact", **dense_metrics, "average_latency_ms": 0.0},
            {"method": "faiss_indexflatip", **dense_metrics, "average_latency_ms": 0.0},
            {"method": "bm25", **bm25_metrics, "average_latency_ms": round(lexical_latency, 6)},
            {"method": "hybrid_rrf", **hybrid_metrics, "average_latency_ms": round(hybrid_latency, 6)},
        ],
        "winner": max(
            [("cosine_exact", dense_metrics), ("bm25", bm25_metrics), ("hybrid_rrf", hybrid_metrics)],
            key=lambda item: (item[1]["f1_at_5"], item[1]["ndcg_at_5"]),
        )[0],
    }


def load_chunks_for_best() -> list[dict[str, Any]]:
    metadata = read_json(ARTIFACTS_DIR / "best_embedding_metadata.json")
    return read_jsonl(PROCESSED_DIR / f"chunks_{metadata['chunking_method']}.jsonl")
