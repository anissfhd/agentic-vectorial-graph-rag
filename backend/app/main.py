from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .core.mvp.common import ARTIFACTS_DIR, METRICS_DIR, read_json
from .core.mvp.runtime import MVPService


app = FastAPI(
    title="Agentic Vectorial Graph RAG with Reinforcement Learning",
    version="1.0.0-mvp",
    description="PFA — thèse sur les vagues de froid extrêmes en Europe",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)


class RetrieveRequest(QuestionRequest):
    top_k: int = Field(default=5, ge=1, le=20)


class FeedbackRequest(BaseModel):
    state: str
    action: str
    positive: bool


@lru_cache(maxsize=1)
def get_service() -> MVPService:
    return MVPService()


@app.get("/health")
def health() -> dict[str, Any]:
    service = get_service()
    neo4j = service.neo4j.status()
    return {
        "status": "ok" if neo4j["connected"] else "degraded",
        "artifacts_loaded": True,
        "chunking_winner": service.chunking_metrics["winner"],
        "embedding_winner": service.embedding_metrics["winner"],
        "faiss_vectors": int(service.faiss_index.ntotal),
        "graph_nodes": service.graph_metrics["nodes"],
        "graph_relations": service.graph_metrics["edges"],
        "neo4j": neo4j,
        "langgraph": "active",
    }


@app.get("/vectorial")
def vectorial() -> dict[str, Any]:
    service = get_service()
    return {
        "chunking_winner": service.chunking_metrics["winner"],
        "embedding_winner": service.embedding_metrics["winner"],
        "retrieval_winner": service.retrieval_metrics["winner"],
        "chunks": service.embedding_metadata["chunks"],
        "dimension": service.embedding_metadata["dimension"],
        "computed_at": service.embedding_metrics["computed_at"],
    }


@app.get("/vectorial/chunking")
def vectorial_chunking() -> dict[str, Any]:
    return get_service().chunking_metrics


@app.get("/vectorial/embeddings")
def vectorial_embeddings() -> dict[str, Any]:
    return get_service().embedding_metrics


@app.get("/vectorial/pca")
def vectorial_pca(query: str | None = None, k: int = Query(default=5, ge=1, le=20)) -> dict[str, Any]:
    payload = read_json(ARTIFACTS_DIR / "pca_2d.json")
    if query:
        payload["nearest_chunk_ids"] = [
            item["chunk_id"] for item in get_service().dense_retrieve(query, k)
        ]
        payload["query"] = query
    return payload


@app.post("/vectorial/retrieve")
def vectorial_retrieve(request: RetrieveRequest) -> dict[str, Any]:
    return get_service().retrieval_comparison(request.question, request.top_k)


@app.get("/graph")
def graph() -> dict[str, Any]:
    return get_service().graph


@app.get("/graph/metrics")
def graph_metrics() -> dict[str, Any]:
    return get_service().graph_metrics


@app.get("/graph/communities")
def graph_communities() -> dict[str, Any]:
    metrics = get_service().graph_metrics
    return {
        "computed_at": metrics["computed_at"],
        "community_count": metrics["community_count"],
        "modularity": metrics["modularity"],
        "communities": metrics["communities"],
    }


@app.get("/graph/neo4j/status")
def neo4j_status() -> dict[str, Any]:
    return get_service().neo4j.status()


@app.post("/graph/subgraph")
def graph_subgraph(request: QuestionRequest) -> dict[str, Any]:
    service = get_service()
    features = service.features_and_policy(request.question)["features"]
    try:
        return service.graph_retrieve(request.question, features["estimated_type"])
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Neo4j Aura indisponible ({type(exc).__name__})") from exc


@app.get("/agentic")
def agentic() -> dict[str, Any]:
    service = get_service()
    return {
        "q_table": service.q_table,
        "reward_history": service.reward_history,
        "routing_metrics": service.routing_metrics,
        "langgraph_nodes": [
            "feature_extraction",
            "q_learning_policy",
            "execute_route",
            "context_verification",
            "answer_and_reward",
        ],
    }


@app.post("/query")
def query(request: QuestionRequest) -> dict[str, Any]:
    return get_service().query(request.question)


@app.post("/agentic/feedback")
def agentic_feedback(request: FeedbackRequest) -> dict[str, Any]:
    try:
        return get_service().feedback(request.state, request.action, request.positive)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/agentic/reset")
def agentic_reset() -> dict[str, Any]:
    return get_service().reset_qlearning()


@app.get("/query/history")
def query_history() -> list[dict[str, Any]]:
    return get_service().history


@app.get("/eval/results")
def evaluation_results() -> dict[str, Any]:
    return {
        "chunking": read_json(METRICS_DIR / "metrics_chunking.json"),
        "embeddings": read_json(METRICS_DIR / "metrics_embeddings.json"),
        "retrieval": read_json(METRICS_DIR / "metrics_retrieval.json"),
        "routing": read_json(METRICS_DIR / "routing_metrics.json"),
        "neo4j": read_json(METRICS_DIR / "neo4j_sync_report.json"),
    }
