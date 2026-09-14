from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app, get_service


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "backend" / "data" / "metrics"
ARTIFACTS = ROOT / "backend" / "data" / "artifacts"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_seven_chunkings_and_representations_exist():
    chunking = load(METRICS / "metrics_chunking.json")
    embeddings = load(METRICS / "metrics_embeddings.json")
    assert len(chunking["methods"]) == 7
    assert all(row["chunk_count"] > 0 for row in chunking["methods"])
    assert len(embeddings["methods"]) == 7
    assert all(row["dimension"] > 0 for row in embeddings["methods"])


def test_vector_graph_and_qlearning_artifacts_are_real():
    graph = load(METRICS / "graph_metrics.json")
    routing = load(METRICS / "routing_metrics.json")
    q_table = load(ARTIFACTS / "q_table.json")
    assert (ARTIFACTS / "faiss.index").stat().st_size > 0
    assert graph["nodes"] > 0 and graph["edges"] > 0
    assert graph["community_count"] > 1
    assert len(q_table["states"]) == 16 and len(q_table["actions"]) == 4
    assert routing["routing_accuracy"] == 1.0


def test_neo4j_aura_sync_report_is_successful():
    report = load(METRICS / "neo4j_sync_report.json")
    assert report["connection_success"] is True
    assert report["nodes_inserted"] > 0
    assert report["relationships_inserted"] > 0
    assert report["communities_inserted"] > 1


@pytest.fixture(scope="module")
def client():
    get_service.cache_clear()
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize(
    ("question", "expected_action", "answer_prefix"),
    [
        ("Comment l'OMM définit-elle une vague de froid ?", "use_vector", "L'Organisation"),
        (
            "Quel lien existe entre le blocage scandinave et les vagues de froid en Europe occidentale ?",
            "use_graph",
            "Un blocage scandinave",
        ),
        (
            "Quelle relation existe entre le SWG et les analogues de circulation, et quel objectif cette méthode sert-elle ?",
            "use_hybrid",
            "Le générateur stochastique",
        ),
        ("Comment fonctionne la rétropropagation dans un réseau neuronal ?", "abstain", "Je ne sais pas."),
    ],
)
def test_four_demo_questions(client, question, expected_action, answer_prefix):
    response = client.post("/query", json={"question": question})
    assert response.status_code == 200
    payload = response.json()
    assert payload["action"] == expected_action
    assert payload["answer"].startswith(answer_prefix)
    assert payload["neo4j_status"]["connected"] is True


def test_critical_api_routes(client):
    for route in (
        "/health",
        "/vectorial",
        "/vectorial/chunking",
        "/vectorial/embeddings",
        "/vectorial/pca",
        "/graph",
        "/graph/metrics",
        "/graph/communities",
        "/graph/neo4j/status",
        "/agentic",
    ):
        assert client.get(route).status_code == 200
    subgraph = client.post(
        "/graph/subgraph",
        json={"question": "Quel lien existe entre le blocage scandinave et les vagues de froid ?"},
    )
    assert subgraph.status_code == 200
    assert subgraph.json()["source"] == "neo4j_aura"
    assert subgraph.json()["nodes"]
