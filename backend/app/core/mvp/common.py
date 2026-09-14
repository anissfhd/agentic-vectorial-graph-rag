from __future__ import annotations

import json
import math
import random
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


BACKEND_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = BACKEND_ROOT.parent
DATA_ROOT = BACKEND_ROOT / "data"
PROCESSED_DIR = DATA_ROOT / "processed"
ARTIFACTS_DIR = DATA_ROOT / "artifacts"
METRICS_DIR = DATA_ROOT / "metrics"
EVAL_PATH = BACKEND_ROOT / "eval" / "eval_set.json"
CORPUS_PATH = PROCESSED_DIR / "corpus_clean.jsonl"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HF_CACHE = DATA_ROOT / "hf_cache"
SEED = 42


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def ensure_output_dirs() -> None:
    for path in (PROCESSED_DIR, ARTIFACTS_DIR, METRICS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-ZÀ-ÿ0-9][a-zA-ZÀ-ÿ0-9_-]+", text.lower())


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def minmax(values: Sequence[float], higher_is_better: bool = True) -> list[float]:
    array = np.asarray(values, dtype=float)
    low, high = float(array.min()), float(array.max())
    if math.isclose(low, high):
        scaled = np.ones_like(array)
    else:
        scaled = (array - low) / (high - low)
    if not higher_is_better:
        scaled = 1.0 - scaled
    return scaled.tolist()


def relevant(chunk: dict[str, Any], reference_pages: set[int]) -> bool:
    pages = chunk.get("pages_pdf") or [chunk.get("page_pdf")]
    return bool(reference_pages.intersection(page for page in pages if page is not None))


def ir_metrics_for_rankings(
    rankings: Sequence[Sequence[int]],
    chunks: Sequence[dict[str, Any]],
    questions: Sequence[dict[str, Any]],
    k: int = 5,
) -> dict[str, float]:
    precision_values: list[float] = []
    recall_values: list[float] = []
    ap_values: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcg_values: list[float] = []

    for order, question in zip(rankings, questions, strict=True):
        reference_pages = set(question.get("relevant_pages_pdf") or [])
        top = list(order[:k])
        relevance = [1 if relevant(chunks[index], reference_pages) else 0 for index in top]
        precision_values.append(sum(relevance) / k)

        pages_found: set[int] = set()
        for index in top:
            pages_found.update(set(chunks[index].get("pages_pdf") or []) & reference_pages)
        recall_values.append(len(pages_found) / max(1, len(reference_pages)))

        hits = 0
        precision_sum = 0.0
        first_rank = 0
        for rank, index in enumerate(order, start=1):
            if relevant(chunks[index], reference_pages):
                hits += 1
                precision_sum += hits / rank
                if first_rank == 0:
                    first_rank = rank
        ap_values.append(precision_sum / max(1, hits))
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)

        dcg = sum(rel / math.log2(position + 2) for position, rel in enumerate(relevance))
        ideal_hits = min(k, max(1, len(reference_pages)))
        idcg = sum(1.0 / math.log2(position + 2) for position in range(ideal_hits))
        ndcg_values.append(dcg / idcg if idcg else 0.0)

    precision = float(np.mean(precision_values)) if precision_values else 0.0
    recall = float(np.mean(recall_values)) if recall_values else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        f"precision_at_{k}": round(precision, 6),
        f"recall_at_{k}": round(recall, 6),
        f"f1_at_{k}": round(f1, 6),
        "map": round(float(np.mean(ap_values)) if ap_values else 0.0, 6),
        "mrr": round(float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0, 6),
        f"ndcg_at_{k}": round(float(np.mean(ndcg_values)) if ndcg_values else 0.0, 6),
    }


def load_reference_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME, cache_folder=str(HF_CACHE), device="cpu")

