from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from neo4j import GraphDatabase


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.mvp.common import ARTIFACTS_DIR, METRICS_DIR, read_json, utc_now, write_json  # noqa: E402


CONSTRAINTS = [
    "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (n:Entity) REQUIRE n.name IS UNIQUE",
]
VERIFICATION_QUERIES = {
    "nodes": "MATCH (n) RETURN count(n) AS nodes",
    "relationships": "MATCH ()-[r]->() RETURN count(r) AS relationships",
    "communities": (
        "MATCH (n) RETURN n.community_id AS community, count(*) AS members "
        "ORDER BY members DESC"
    ),
    "sample_relations": (
        "MATCH (n)-[r]->(m) RETURN n.name AS source, type(r) AS type, "
        "m.name AS target, r.page_pdf AS page_pdf LIMIT 20"
    ),
}


def _load_config() -> dict[str, str]:
    path = BACKEND_ROOT / ".env"
    values = dotenv_values(path) if path.exists() else {}
    required = ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "NEO4J_DATABASE")
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise RuntimeError("Configuration Neo4j incomplète: " + ", ".join(missing))
    return {key: str(values[key]) for key in required}


def _node_rows(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": node["name"],
            "label": node["label"],
            "entity_type": node["entity_type"],
            "aliases": node.get("aliases", []),
            "occurrence_count": node.get("occurrence_count", 0),
            "pages_pdf": node.get("pages_pdf", []),
            "community_id": node["community_id"],
            "degree_centrality": node["degree_centrality"],
            "betweenness_centrality": node["betweenness_centrality"],
            "closeness_centrality": node["closeness_centrality"],
        }
        for node in nodes
    ]


def sync() -> dict[str, Any]:
    started = time.perf_counter()
    config = _load_config()
    payload = read_json(ARTIFACTS_DIR / "graph.json")
    driver = GraphDatabase.driver(
        config["NEO4J_URI"],
        auth=(config["NEO4J_USER"], config["NEO4J_PASSWORD"]),
    )
    report: dict[str, Any] = {
        "connection_success": False,
        "database_name": config["NEO4J_DATABASE"],
        "nodes_inserted": 0,
        "relationships_inserted": 0,
        "communities_inserted": 0,
        "constraints_created": [],
        "verification_results": {},
        "sync_duration_seconds": 0.0,
        "synced_at": None,
    }
    try:
        driver.verify_connectivity()
        with driver.session(database=config["NEO4J_DATABASE"]) as session:
            for statement in CONSTRAINTS:
                session.run(statement).consume()
                report["constraints_created"].append("entity_name_unique")
            session.run(
                """
                UNWIND $rows AS row
                MERGE (n:Entity {name: row.name})
                SET n.label = row.label,
                    n.entity_type = row.entity_type,
                    n.aliases = row.aliases,
                    n.occurrence_count = row.occurrence_count,
                    n.pages_pdf = row.pages_pdf,
                    n.community_id = row.community_id,
                    n.degree_centrality = row.degree_centrality,
                    n.betweenness_centrality = row.betweenness_centrality,
                    n.closeness_centrality = row.closeness_centrality,
                    n.synced_at = datetime()
                """,
                rows=_node_rows(payload["nodes"]),
            ).consume()

            by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for relation in payload["relations"]:
                by_type[relation["type"]].append(relation)
            for relation_type, rows in by_type.items():
                safe_type = re_safe_relation_type(relation_type)
                session.run(
                    f"""
                    UNWIND $rows AS row
                    MATCH (source:Entity {{name: row.source}})
                    MATCH (target:Entity {{name: row.target}})
                    MERGE (source)-[r:{safe_type} {{relation_id: row.relation_id}}]->(target)
                    SET r.confidence = row.confidence,
                        r.extraction_method = row.extraction_method,
                        r.original_predicate = row.original_predicate,
                        r.page_pdf = row.page_pdf,
                        r.page_these = row.page_these,
                        r.chunk_id = row.chunk_id,
                        r.excerpt = row.excerpt,
                        r.occurrence_count = row.occurrence_count,
                        r.synced_at = datetime()
                    """,
                    rows=rows,
                ).consume()

            results: dict[str, Any] = {}
            for name, query in VERIFICATION_QUERIES.items():
                records = session.run(query).data()
                results[name] = records
            report["connection_success"] = True
            report["nodes_inserted"] = int(results["nodes"][0]["nodes"])
            report["relationships_inserted"] = int(results["relationships"][0]["relationships"])
            report["communities_inserted"] = len(
                [row for row in results["communities"] if row.get("community") is not None]
            )
            report["verification_results"] = results
    finally:
        driver.close()
        report["sync_duration_seconds"] = round(time.perf_counter() - started, 6)
        report["synced_at"] = utc_now()
        write_json(METRICS_DIR / "neo4j_sync_report.json", report)
    if not report["connection_success"]:
        raise RuntimeError("La connexion Neo4j n'a pas été validée")
    if report["nodes_inserted"] <= 0 or report["relationships_inserted"] <= 0:
        raise RuntimeError("Aura ne contient pas de nœuds et relations après synchronisation")
    return report


def re_safe_relation_type(value: str) -> str:
    allowed = {
        "MENTIONS",
        "CO_OCCURS_WITH",
        "ASSOCIATED_WITH",
        "INFLUENCES",
        "CAUSES",
        "USES",
        "LOCATED_IN",
        "OCCURRED_DURING",
        "AFFECTS",
    }
    return value if value in allowed else "ASSOCIATED_WITH"


if __name__ == "__main__":
    result = sync()
    print("NEO4J_SYNC=OK")
    print(f"NODES={result['nodes_inserted']}")
    print(f"RELATIONSHIPS={result['relationships_inserted']}")
    print(f"COMMUNITIES={result['communities_inserted']}")
