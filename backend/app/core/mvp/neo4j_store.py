from __future__ import annotations

from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from neo4j import GraphDatabase

from .common import BACKEND_ROOT


SUBGRAPH_CYPHER = """
MATCH (n:Entity)-[r]-(m:Entity)
WHERE size($names) = 0 OR n.name IN $names OR m.name IN $names
WITH n, r, m
ORDER BY coalesce(n.degree_centrality, 0.0) DESC
LIMIT 60
RETURN properties(n) AS source, type(r) AS relation_type,
       properties(r) AS relation, properties(m) AS target
""".strip()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return str(value)


class Neo4jGateway:
    def __init__(self) -> None:
        values = dotenv_values(BACKEND_ROOT / ".env")
        self.uri = values.get("NEO4J_URI")
        self.user = values.get("NEO4J_USER")
        self.password = values.get("NEO4J_PASSWORD")
        self.database = values.get("NEO4J_DATABASE") or "neo4j"

    @property
    def configured(self) -> bool:
        return bool(self.uri and self.user and self.password and self.database)

    def _driver(self):
        if not self.configured:
            raise RuntimeError("Neo4j Aura n'est pas configuré")
        return GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def status(self) -> dict[str, Any]:
        if not self.configured:
            return {
                "configured": False,
                "connected": False,
                "nodes": 0,
                "relationships": 0,
                "communities": 0,
                "message": "backend/.env incomplet",
            }
        driver = self._driver()
        try:
            driver.verify_connectivity()
            records = driver.execute_query(
                "MATCH (n) OPTIONAL MATCH ()-[r]->() "
                "RETURN count(DISTINCT n) AS nodes, count(DISTINCT r) AS relationships, "
                "count(DISTINCT n.community_id) AS communities",
                database_=self.database,
            ).records
            row = records[0]
            return {
                "configured": True,
                "connected": True,
                "database": self.database,
                "nodes": int(row["nodes"]),
                "relationships": int(row["relationships"]),
                "communities": int(row["communities"]),
                "message": "Connecté à Neo4j Aura",
            }
        except Exception as exc:  # Never include connection details in the API response.
            return {
                "configured": True,
                "connected": False,
                "nodes": 0,
                "relationships": 0,
                "communities": 0,
                "message": f"Connexion indisponible ({type(exc).__name__})",
            }
        finally:
            driver.close()

    def subgraph(self, names: list[str]) -> dict[str, Any]:
        driver = self._driver()
        try:
            records = driver.execute_query(
                SUBGRAPH_CYPHER,
                names=names,
                database_=self.database,
            ).records
            nodes: dict[str, dict[str, Any]] = {}
            relations: dict[str, dict[str, Any]] = {}
            for record in records:
                source = _json_safe(dict(record["source"]))
                target = _json_safe(dict(record["target"]))
                relation = _json_safe(dict(record["relation"]))
                nodes[source["name"]] = source
                nodes[target["name"]] = target
                relation_id = relation.get("relation_id") or (
                    f"{source['name']}|{record['relation_type']}|{target['name']}"
                )
                relations[relation_id] = {
                    **relation,
                    "relation_id": relation_id,
                    "source": source["name"],
                    "target": target["name"],
                    "type": record["relation_type"],
                }
            return {
                "source": "neo4j_aura",
                "cypher": SUBGRAPH_CYPHER,
                "parameters": {"names": names},
                "nodes": list(nodes.values()),
                "relations": list(relations.values()),
                "neo4j_connected": True,
            }
        finally:
            driver.close()
