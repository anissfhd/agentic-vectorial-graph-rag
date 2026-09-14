from __future__ import annotations

import argparse
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.mvp.agent import train_qlearning  # noqa: E402
from backend.app.core.mvp.chunking import METHODS, run_chunking  # noqa: E402
from backend.app.core.mvp.common import (  # noqa: E402
    ARTIFACTS_DIR,
    METRICS_DIR,
    PROCESSED_DIR,
    ensure_output_dirs,
    load_reference_model,
    read_json,
)
from backend.app.core.mvp.embeddings import build_vector_artifacts, run_embeddings  # noqa: E402
from backend.app.core.mvp.graph import build_graph  # noqa: E402


def exists_all(paths: list[Path]) -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in paths)


def main(force: bool = False) -> None:
    ensure_output_dirs()
    model = None
    chunk_paths = [PROCESSED_DIR / f"chunks_{method}.jsonl" for method in METHODS]
    chunk_ready = exists_all([METRICS_DIR / "metrics_chunking.json", *chunk_paths])
    if force or not chunk_ready:
        print("[1/6] Calcul réel des sept chunkings")
        model = load_reference_model()
        chunking_report = run_chunking(model)
    else:
        print("[1/6] Chunkings déjà calculés — cache réutilisé")
        chunking_report = read_json(METRICS_DIR / "metrics_chunking.json")

    embedding_paths = [
        METRICS_DIR / "metrics_embeddings.json",
        ARTIFACTS_DIR / "best_embeddings.npy",
        ARTIFACTS_DIR / "best_embedding_metadata.json",
        ARTIFACTS_DIR / "best_embedding_pipeline.joblib",
    ]
    if force or not exists_all(embedding_paths):
        print("[2/6] Calcul réel des sept représentations")
        model = model or load_reference_model()
        embedding_report, embeddings, chunks = run_embeddings(model, chunking_report)
    else:
        print("[2/6] Embeddings déjà calculés — cache réutilisé")
        import numpy as np
        from backend.app.core.mvp.embeddings import load_chunks_for_best

        embedding_report = read_json(METRICS_DIR / "metrics_embeddings.json")
        embeddings = np.load(ARTIFACTS_DIR / "best_embeddings.npy")
        chunks = load_chunks_for_best()

    vector_paths = [
        ARTIFACTS_DIR / "faiss.index",
        ARTIFACTS_DIR / "bm25_data.json",
        ARTIFACTS_DIR / "pca_2d.json",
        METRICS_DIR / "metrics_retrieval.json",
    ]
    if force or not exists_all(vector_paths):
        print("[3/6] FAISS, BM25, RRF et PCA")
        build_vector_artifacts(embedding_report, embeddings, chunks)
    else:
        print("[3/6] Artefacts de retrieval déjà calculés")

    graph_paths = [
        ARTIFACTS_DIR / "graph.json",
        ARTIFACTS_DIR / "graph.graphml",
        ARTIFACTS_DIR / "graph.cypher",
        METRICS_DIR / "graph_metrics.json",
    ]
    if force or not exists_all(graph_paths):
        print("[4/6] Knowledge Graph, Louvain et centralités")
        build_graph()
    else:
        print("[4/6] Graphe déjà calculé — cache réutilisé")

    q_paths = [
        ARTIFACTS_DIR / "q_table.json",
        ARTIFACTS_DIR / "reward_history.json",
        ARTIFACTS_DIR / "query_type_classifier.joblib",
        METRICS_DIR / "routing_metrics.json",
    ]
    if force or not exists_all(q_paths):
        print("[5/6] Q-Learning réel et politique LangGraph")
        train_qlearning()
    else:
        print("[5/6] Q-table déjà entraînée — cache réutilisé")

    print("[6/6] Synchronisation idempotente Neo4j Aura")
    try:
        from backend.scripts.push_to_neo4j import sync

        result = sync()
        print(
            f"Aura OK — {result['nodes_inserted']} nœuds, "
            f"{result['relationships_inserted']} relations"
        )
    except RuntimeError as exc:
        print(f"Aura non synchronisée: {exc}")
    print("MVP artifacts ready")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construit uniquement les artefacts PFA manquants")
    parser.add_argument("--force", action="store_true", help="Recalcule tous les artefacts hors ligne")
    arguments = parser.parse_args()
    main(force=arguments.force)

