from __future__ import annotations

import re
import time
from collections import defaultdict
from typing import Any, Callable, Iterable

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .common import (
    CORPUS_PATH,
    EVAL_PATH,
    METRICS_DIR,
    MODEL_NAME,
    PROCESSED_DIR,
    SEED,
    ir_metrics_for_rankings,
    l2_normalize,
    minmax,
    read_json,
    read_jsonl,
    seed_everything,
    utc_now,
    write_json,
    write_jsonl,
)


METHODS = (
    "fixed_size",
    "sentence_based",
    "paragraph_based",
    "sliding_window",
    "recursive",
    "semantic",
    "document_structure",
)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ0-9])")


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_RE.split(re.sub(r"\s+", " ", text)) if len(part.strip()) >= 20]


def _page_meta(page: dict[str, Any]) -> dict[str, Any]:
    return {
        "page_pdf": page["page_pdf"],
        "page_these": page.get("page_these"),
        "chapter": page.get("chapter_title") or page.get("chapter_number") or "Front matter",
        "section": page.get("section_title") or page.get("section_number") or "Non spécifiée",
        "language": page.get("language") or "unknown",
    }


def _make_chunk(method: str, index: int, text: str, metas: Iterable[dict[str, Any]]) -> dict[str, Any]:
    meta_list = list(metas)
    pages_pdf = sorted({int(meta["page_pdf"]) for meta in meta_list})
    pages_these = sorted({int(meta["page_these"]) for meta in meta_list if meta.get("page_these") is not None})
    chapters = list(dict.fromkeys(str(meta["chapter"]) for meta in meta_list))
    sections = list(dict.fromkeys(str(meta["section"]) for meta in meta_list))
    languages = list(dict.fromkeys(str(meta["language"]) for meta in meta_list))
    clean = re.sub(r"[ \t]+", " ", text).strip()
    return {
        "chunk_id": f"{method}_{index:05d}",
        "text": clean,
        "pages_pdf": pages_pdf,
        "pages_these": pages_these,
        "chapter": " | ".join(chapters),
        "section": " | ".join(sections),
        "language": languages[0] if len(languages) == 1 else "mixed",
        "method": method,
        "length": len(clean),
        "word_count": len(clean.split()),
    }


def _split_with_spans(text: str, size: int, step: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = max(text.rfind(". ", start + size // 2, end), text.rfind("\n", start + size // 2, end))
            if boundary > start:
                end = boundary + 1
        value = text[start:end].strip()
        if value:
            chunks.append(value)
        if end >= len(text):
            break
        start = max(start + 1, min(end, start + step))
    return chunks


def fixed_size(pages: list[dict[str, Any]], _: Any = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in pages:
        meta = _page_meta(page)
        for text in _split_with_spans(page["clean_text"], 900, 900):
            rows.append(_make_chunk("fixed_size", len(rows), text, [meta]))
    return rows


def sliding_window(pages: list[dict[str, Any]], _: Any = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in pages:
        meta = _page_meta(page)
        for text in _split_with_spans(page["clean_text"], 900, 650):
            rows.append(_make_chunk("sliding_window", len(rows), text, [meta]))
    return rows


def sentence_based(pages: list[dict[str, Any]], _: Any = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in pages:
        sentences = split_sentences(page["clean_text"])
        current: list[str] = []
        for sentence in sentences:
            if current and len(" ".join(current + [sentence])) > 900:
                rows.append(_make_chunk("sentence_based", len(rows), " ".join(current), [_page_meta(page)]))
                current = []
            current.append(sentence)
        if current:
            rows.append(_make_chunk("sentence_based", len(rows), " ".join(current), [_page_meta(page)]))
    return rows


def paragraph_based(pages: list[dict[str, Any]], _: Any = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in pages:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", page["clean_text"]) if len(part.strip()) >= 30]
        for paragraph in paragraphs:
            for text in _split_with_spans(paragraph, 1200, 1200):
                rows.append(_make_chunk("paragraph_based", len(rows), text, [_page_meta(page)]))
    return rows


def recursive(pages: list[dict[str, Any]], _: Any = None) -> list[dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", "; ", ", ", " "],
    )
    rows: list[dict[str, Any]] = []
    for page in pages:
        for text in splitter.split_text(page["clean_text"]):
            rows.append(_make_chunk("recursive", len(rows), text, [_page_meta(page)]))
    return rows


def semantic(pages: list[dict[str, Any]], model: Any) -> list[dict[str, Any]]:
    records: list[tuple[str, dict[str, Any]]] = []
    for page in pages:
        records.extend((sentence, _page_meta(page)) for sentence in split_sentences(page["clean_text"]))
    embeddings = l2_normalize(model.encode([record[0] for record in records], batch_size=64, show_progress_bar=False))
    rows: list[dict[str, Any]] = []
    current_texts: list[str] = []
    current_metas: list[dict[str, Any]] = []
    previous_index: int | None = None
    for index, (sentence, meta) in enumerate(records):
        same_page = bool(current_metas and current_metas[-1]["page_pdf"] == meta["page_pdf"])
        similarity = float(np.dot(embeddings[previous_index], embeddings[index])) if previous_index is not None else 1.0
        proposed_length = len(" ".join(current_texts + [sentence]))
        should_break = current_texts and (not same_page or proposed_length > 1100 or (similarity < 0.30 and len(" ".join(current_texts)) >= 300))
        if should_break:
            rows.append(_make_chunk("semantic", len(rows), " ".join(current_texts), current_metas))
            current_texts, current_metas = [], []
        current_texts.append(sentence)
        current_metas.append(meta)
        previous_index = index
    if current_texts:
        rows.append(_make_chunk("semantic", len(rows), " ".join(current_texts), current_metas))
    return rows


def document_structure(pages: list[dict[str, Any]], _: Any = None) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for page in pages:
        meta = _page_meta(page)
        key = (meta["chapter"], meta["section"])
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", page["clean_text"]) if len(part.strip()) >= 30]
        grouped[key].extend((paragraph, meta) for paragraph in paragraphs)

    rows: list[dict[str, Any]] = []
    for units in grouped.values():
        texts: list[str] = []
        metas: list[dict[str, Any]] = []
        for text, meta in units:
            if texts and len("\n\n".join(texts + [text])) > 1200:
                rows.append(_make_chunk("document_structure", len(rows), "\n\n".join(texts), metas))
                texts, metas = [], []
            texts.append(text)
            metas.append(meta)
        if texts:
            rows.append(_make_chunk("document_structure", len(rows), "\n\n".join(texts), metas))
    return rows


IMPLEMENTATIONS: dict[str, Callable[[list[dict[str, Any]], Any], list[dict[str, Any]]]] = {
    "fixed_size": fixed_size,
    "sentence_based": sentence_based,
    "paragraph_based": paragraph_based,
    "sliding_window": sliding_window,
    "recursive": recursive,
    "semantic": semantic,
    "document_structure": document_structure,
}


def _cohesion_metrics(chunks: list[dict[str, Any]], embeddings: np.ndarray, model: Any) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    sample_indices = rng.choice(len(chunks), size=min(60, len(chunks)), replace=False)
    sentence_groups: list[list[str]] = []
    flattened: list[str] = []
    for index in sample_indices:
        sentences = split_sentences(chunks[int(index)]["text"])[:5]
        if len(sentences) >= 2:
            sentence_groups.append(sentences)
            flattened.extend(sentences)
    intra_values: list[float] = []
    if flattened:
        encoded = l2_normalize(model.encode(flattened, batch_size=64, show_progress_bar=False))
        cursor = 0
        for sentences in sentence_groups:
            group = encoded[cursor : cursor + len(sentences)]
            cursor += len(sentences)
            similarities = group @ group.T
            upper = similarities[np.triu_indices(len(group), 1)]
            intra_values.append(float(np.mean(upper)))

    pair_count = min(500, max(1, len(chunks) * 2))
    left = rng.integers(0, len(chunks), pair_count)
    right = rng.integers(0, len(chunks), pair_count)
    valid = left != right
    inter = float(np.mean(np.sum(embeddings[left[valid]] * embeddings[right[valid]], axis=1))) if np.any(valid) else 0.0
    return float(np.mean(intra_values)) if intra_values else 0.0, inter


def run_chunking(model: Any) -> dict[str, Any]:
    seed_everything()
    pages = read_jsonl(CORPUS_PATH)
    evaluation = read_json(EVAL_PATH)
    questions = [question for question in evaluation["questions"] if question["question_type"] != "out_of_context"]
    query_embeddings = l2_normalize(
        model.encode([question["question"] for question in questions], batch_size=32, show_progress_bar=False)
    )
    results: list[dict[str, Any]] = []

    for method in METHODS:
        start = time.perf_counter()
        chunks = IMPLEMENTATIONS[method](pages, model)
        chunking_seconds = time.perf_counter() - start
        if not chunks:
            raise RuntimeError(f"La méthode {method} n'a produit aucun chunk")
        write_jsonl(PROCESSED_DIR / f"chunks_{method}.jsonl", chunks)

        embeddings = l2_normalize(
            model.encode([chunk["text"] for chunk in chunks], batch_size=64, show_progress_bar=False)
        )
        scores = query_embeddings @ embeddings.T
        rankings = np.argsort(-scores, axis=1).tolist()
        ir = ir_metrics_for_rankings(rankings, chunks, questions, k=5)
        intra, inter = _cohesion_metrics(chunks, embeddings, model)
        lengths = np.asarray([chunk["length"] for chunk in chunks], dtype=float)
        word_lengths = np.asarray([chunk["word_count"] for chunk in chunks], dtype=float)
        results.append(
            {
                "method": method,
                "chunk_count": len(chunks),
                "average_length_chars": round(float(lengths.mean()), 3),
                "std_length_chars": round(float(lengths.std()), 3),
                "min_length_chars": int(lengths.min()),
                "max_length_chars": int(lengths.max()),
                "average_length_words": round(float(word_lengths.mean()), 3),
                "intra_similarity": round(intra, 6),
                "inter_similarity": round(inter, 6),
                **{key: ir[key] for key in ("precision_at_5", "recall_at_5", "f1_at_5")},
                "execution_seconds": round(chunking_seconds, 6),
            }
        )

    f1_scaled = minmax([row["f1_at_5"] for row in results])
    inter_scaled = minmax([row["inter_similarity"] for row in results], higher_is_better=False)
    intra_scaled = minmax([row["intra_similarity"] for row in results])
    regularity = [1.0 / (1.0 + row["std_length_chars"] / max(1.0, row["average_length_chars"])) for row in results]
    regularity_scaled = minmax(regularity)
    speed_scaled = minmax([row["execution_seconds"] for row in results], higher_is_better=False)
    for index, row in enumerate(results):
        row["global_score"] = round(
            0.35 * f1_scaled[index]
            + 0.20 * inter_scaled[index]
            + 0.20 * intra_scaled[index]
            + 0.15 * regularity_scaled[index]
            + 0.10 * speed_scaled[index],
            6,
        )
    winner = max(results, key=lambda row: (row["global_score"], row["f1_at_5"]))["method"]
    report = {
        "computed_at": utc_now(),
        "seed": SEED,
        "reference_model": MODEL_NAME,
        "relevance_granularity": "page_pdf",
        "k": 5,
        "score_formula": "0.35*F1@5 + 0.20*(1-inter) + 0.20*intra + 0.15*length_regularity + 0.10*speed (min-max normalized)",
        "methods": results,
        "winner": winner,
    }
    write_json(METRICS_DIR / "metrics_chunking.json", report)
    _plot_comparison(results, winner)
    return report


def _plot_comparison(results: list[dict[str, Any]], winner: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["method"].replace("_", "\n") for row in results]
    values = [row["global_score"] for row in results]
    colors = ["#22c55e" if row["method"] == winner else "#4f46e5" for row in results]
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.bar(labels, values, color=colors)
    axis.set_ylabel("Score global normalisé")
    axis.set_title("Comparaison réelle des sept méthodes de chunking")
    axis.set_ylim(0, 1.05)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(METRICS_DIR / "chunking_comparison.png", dpi=160)
    plt.close(figure)
