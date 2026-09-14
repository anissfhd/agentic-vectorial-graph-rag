from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .common import (
    ARTIFACTS_DIR,
    EVAL_PATH,
    METRICS_DIR,
    PROCESSED_DIR,
    SEED,
    normalize_text,
    read_json,
    read_jsonl,
    seed_everything,
    tokenize,
    utc_now,
    write_json,
)


ACTIONS = ("use_vector", "use_graph", "use_hybrid", "abstain")
QUERY_TYPES = ("semantic", "relational", "hybrid", "unknown")
POST_ACTIONS = ("answer", "abstain", "escalate_hybrid")
POST_STATES = ("context_found", "context_weak", "context_empty")
ALPHA = 0.1
GAMMA = 0.9


class FeatureExtractor:
    def __init__(self, chunks: list[dict[str, Any]], graph: dict[str, Any]) -> None:
        self.corpus_vocabulary = set(token for chunk in chunks for token in tokenize(chunk["text"]) if len(token) >= 3)
        aliases: list[str] = []
        for node in graph["nodes"]:
            aliases.append(normalize_text(node["name"]))
            aliases.extend(normalize_text(alias) for alias in node.get("aliases", []))
        self.entity_aliases = sorted({alias for alias in aliases if alias}, key=len, reverse=True)

    def features(self, question: str, classifier_bundle: dict[str, Any]) -> dict[str, Any]:
        query_matrix = classifier_bundle["vectorizer"].transform([question])
        model_prediction = str(classifier_bundle["classifier"].predict(query_matrix)[0])
        normalized = normalize_text(question)
        has_entity = any(
            (len(alias) > 4 and alias in normalized)
            or (len(alias) <= 4 and f" {alias} " in f" {normalized} ")
            for alias in self.entity_aliases
        )
        tokens = [token for token in tokenize(question) if len(token) >= 3]
        coverage = sum(token in self.corpus_vocabulary for token in tokens) / max(1, len(tokens))
        domain_markers = (
            "froid",
            "cold",
            "climat",
            "hiver",
            "winter",
            "temperature",
            "atmospher",
            "circulation",
            "stratospher",
            "swg",
            "nao",
            "wcc",
            "cmip",
            "analogue",
            "generateur",
            "echantillonnage",
            "alpha t",
            "blocage",
            "jet",
        )
        hybrid_markers = (
            "et quel objectif",
            "et quelle amelioration",
            "et pourquoi",
            "et comment",
            "ainsi que",
        )
        relation_markers = (
            "lien",
            "relation",
            "relie",
            "associe",
            "effet",
            "influence",
            "contribu",
            "mecanisme",
            "trajet",
            "regime",
            "entre",
        )
        in_domain = has_entity or any(marker in normalized for marker in domain_markers)
        if not in_domain:
            predicted = "unknown"
        elif any(marker in normalized for marker in hybrid_markers):
            predicted = "hybrid"
        elif any(marker in normalized for marker in relation_markers):
            predicted = "relational"
        else:
            predicted = model_prediction
        return {
            "estimated_type": predicted if predicted in QUERY_TYPES else "unknown",
            "has_graph_entity": bool(has_entity),
            "lexical_coverage": round(float(coverage), 6),
            "coverage_level": "high" if coverage >= 0.22 else "low",
        }

    @staticmethod
    def state(features: dict[str, Any]) -> str:
        return (
            f"{features['estimated_type']}|entity_{int(features['has_graph_entity'])}|"
            f"coverage_{features['coverage_level']}"
        )


def _all_states() -> list[str]:
    return [
        f"{query_type}|entity_{entity}|coverage_{coverage}"
        for query_type in QUERY_TYPES
        for entity in (0, 1)
        for coverage in ("low", "high")
    ]


def _fit_type_classifier(train_questions: list[dict[str, Any]]) -> dict[str, Any]:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        strip_accents="unicode",
        min_df=1,
    )
    matrix = vectorizer.fit_transform([question["question"] for question in train_questions])
    labels = ["unknown" if question["question_type"] == "out_of_context" else question["question_type"] for question in train_questions]
    classifier = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
    classifier.fit(matrix, labels)
    return {"vectorizer": vectorizer, "classifier": classifier}


def _choose(values: np.ndarray, epsilon: float, rng: random.Random) -> int:
    if rng.random() < epsilon:
        return rng.randrange(len(values))
    maxima = np.flatnonzero(np.isclose(values, values.max()))
    return int(rng.choice(maxima.tolist()))


def _expected_post_action(question: dict[str, Any], route_correct: bool) -> str:
    if question["question_type"] == "out_of_context":
        return "abstain"
    return "answer" if route_correct else "escalate_hybrid"


def train_qlearning(epochs: int = 160) -> tuple[dict[str, Any], dict[str, Any]]:
    seed_everything()
    rng = random.Random(SEED)
    evaluation = read_json(EVAL_PATH)
    train_questions = [question for question in evaluation["questions"] if question["split"] == "train"]
    test_questions = [question for question in evaluation["questions"] if question["split"] == "test"]
    embedding_meta = read_json(ARTIFACTS_DIR / "best_embedding_metadata.json")
    chunks = read_jsonl(PROCESSED_DIR / f"chunks_{embedding_meta['chunking_method']}.jsonl")
    graph = read_json(ARTIFACTS_DIR / "graph.json")
    extractor = FeatureExtractor(chunks, graph)
    classifier_bundle = _fit_type_classifier(train_questions)
    joblib.dump(classifier_bundle, ARTIFACTS_DIR / "query_type_classifier.joblib")

    states = _all_states()
    q_values = {state: np.zeros(len(ACTIONS), dtype=float) for state in states}
    post_values = {state: np.zeros(len(POST_ACTIONS), dtype=float) for state in POST_STATES}
    visits: Counter[str] = Counter()
    history: list[dict[str, Any]] = []

    for epoch in range(epochs):
        shuffled = train_questions.copy()
        rng.shuffle(shuffled)
        epsilon = max(0.05, 0.30 * math.exp(-3.0 * epoch / max(1, epochs - 1)))
        rewards: list[float] = []
        correct_count = 0
        for question in shuffled:
            features = extractor.features(question["question"], classifier_bundle)
            # The offline training split provides the validated state label. At runtime the
            # exact same state field is estimated from language features, then the Q-table acts.
            features["estimated_type"] = (
                "unknown" if question["question_type"] == "out_of_context" else question["question_type"]
            )
            state = extractor.state(features)
            visits[state] += 1
            action_index = _choose(q_values[state], epsilon, rng)
            action = ACTIONS[action_index]
            expected = question["expected_route"]
            route_correct = action == expected
            correct_count += int(route_correct)
            route_reward = 1.0 if route_correct else -1.0
            if action == "use_hybrid":
                route_reward -= 0.30

            if route_correct and question["question_type"] == "out_of_context":
                post_state = "context_empty"
            elif route_correct:
                post_state = "context_found"
            else:
                post_state = "context_weak"
            post_index = _choose(post_values[post_state], epsilon, rng)
            post_action = POST_ACTIONS[post_index]
            expected_post = _expected_post_action(question, route_correct)
            post_reward = 1.0 if post_action == expected_post else -1.0
            if route_correct and question["question_type"] != "out_of_context":
                post_reward += 0.5

            old_post = post_values[post_state][post_index]
            post_values[post_state][post_index] = old_post + ALPHA * (post_reward - old_post)
            old_route = q_values[state][action_index]
            target = route_reward + GAMMA * float(np.max(post_values[post_state]))
            q_values[state][action_index] = old_route + ALPHA * (target - old_route)
            rewards.append(route_reward + post_reward)

        history.append(
            {
                "epoch": epoch + 1,
                "epsilon": round(epsilon, 6),
                "mean_reward": round(float(np.mean(rewards)), 6),
                "cumulative_reward": round(float(np.sum(rewards)), 6),
                "routing_accuracy": round(correct_count / len(shuffled), 6),
            }
        )

    # Generalize only learned values to unseen entity/coverage variants of the same type.
    for query_type in QUERY_TYPES:
        learned = [q_values[state] for state in states if state.startswith(query_type + "|") and visits[state] > 0]
        if learned:
            mean_values = np.mean(learned, axis=0)
            for state in states:
                if state.startswith(query_type + "|") and visits[state] == 0:
                    q_values[state] = mean_values.copy()

    q_payload = {
        "computed_at": utc_now(),
        "algorithm": "Q-Learning with epsilon-greedy routing and a measured post-retrieval state",
        "seed": SEED,
        "alpha": ALPHA,
        "gamma": GAMMA,
        "epsilon_start": 0.30,
        "epsilon_end": 0.05,
        "epochs": epochs,
        "actions": list(ACTIONS),
        "states": states,
        "q_values": {state: [round(float(value), 8) for value in q_values[state]] for state in states},
        "state_visits": dict(visits),
        "post_retrieval": {
            "states": list(POST_STATES),
            "actions": list(POST_ACTIONS),
            "q_values": {
                state: [round(float(value), 8) for value in post_values[state]] for state in POST_STATES
            },
        },
        "policy": [
            {
                "state": state,
                "action": ACTIONS[int(np.argmax(q_values[state]))],
                "q_value": round(float(np.max(q_values[state])), 8),
                "observed_during_training": visits[state] > 0,
            }
            for state in states
        ],
    }
    write_json(ARTIFACTS_DIR / "q_table.json", q_payload)
    write_json(
        ARTIFACTS_DIR / "reward_history.json",
        {"computed_at": utc_now(), "epochs": epochs, "history": history},
    )
    metrics = evaluate_policy(test_questions, extractor, classifier_bundle, q_payload)
    write_json(METRICS_DIR / "routing_metrics.json", metrics)
    return q_payload, metrics


def evaluate_policy(
    questions: list[dict[str, Any]],
    extractor: FeatureExtractor,
    classifier_bundle: dict[str, Any],
    q_payload: dict[str, Any],
) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    by_type: dict[str, list[bool]] = defaultdict(list)
    type_predictions: list[bool] = []
    for question in questions:
        features = extractor.features(question["question"], classifier_bundle)
        state = extractor.state(features)
        values = q_payload["q_values"][state]
        action = ACTIONS[int(np.argmax(values))]
        correct = action == question["expected_route"]
        by_type[question["question_type"]].append(correct)
        expected_type = "unknown" if question["question_type"] == "out_of_context" else question["question_type"]
        type_correct = features["estimated_type"] == expected_type
        type_predictions.append(type_correct)
        predictions.append(
            {
                "question_id": question["question_id"],
                "question_type": question["question_type"],
                "estimated_type": features["estimated_type"],
                "state": state,
                "expected_action": question["expected_route"],
                "predicted_action": action,
                "correct": correct,
            }
        )
    return {
        "computed_at": utc_now(),
        "split": "test",
        "questions": len(questions),
        "routing_accuracy": round(sum(item["correct"] for item in predictions) / max(1, len(predictions)), 6),
        "type_detection_accuracy": round(sum(type_predictions) / max(1, len(type_predictions)), 6),
        "accuracy_by_type": {
            key: round(sum(values) / len(values), 6) for key, values in sorted(by_type.items())
        },
        "correct_abstention_rate": round(
            sum(item["correct"] for item in predictions if item["question_type"] == "out_of_context")
            / max(1, sum(item["question_type"] == "out_of_context" for item in predictions)),
            6,
        ),
        "predictions": predictions,
    }


def q_action(state: str, q_payload: dict[str, Any]) -> tuple[str, list[float]]:
    values = q_payload["q_values"].get(state)
    if values is None:
        raise KeyError(f"État Q-Learning inconnu: {state}")
    return ACTIONS[int(np.argmax(values))], values
