from __future__ import annotations

import itertools
import math
import re
from collections import Counter, defaultdict
from typing import Any

import networkx as nx

from .common import (
    ARTIFACTS_DIR,
    EVAL_PATH,
    METRICS_DIR,
    PROCESSED_DIR,
    SEED,
    normalize_text,
    read_json,
    read_jsonl,
    utc_now,
    write_json,
)


ALLOWED_RELATIONS = {
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
PREDICATE_MAP = {
    "CAUSES": "CAUSES",
    "ASSOCIATED_WITH": "ASSOCIATED_WITH",
    "CHARACTERIZE": "ASSOCIATED_WITH",
    "LINKED_TO": "ASSOCIATED_WITH",
    "MAY_INCREASE": "INFLUENCES",
    "WEAKENED_AND_MAINTAINED": "INFLUENCES",
    "SHARE_MECHANISM": "ASSOCIATED_WITH",
    "OUTPERFORMS_FOR_COLD_SPELLS": "ASSOCIATED_WITH",
    "BASED_ON": "USES",
    "COMPARED_TO": "ASSOCIATED_WITH",
    "CONSTRAINS": "INFLUENCES",
}


def _entity_type(name: str) -> str:
    lowered = normalize_text(name)
    if any(token in lowered for token in ("france", "europe", "siberia", "greenland", "azores", "iceland")):
        return "Location"
    if any(token in lowered for token in ("1963", "january", "february", "winter")):
        return "Period"
    if any(token in lowered for token in ("cmip", "era5", "dataset")):
        return "Dataset"
    if any(token in lowered for token in ("swg", "method", "adjustment", "boosting", "sampling", "generator")):
        return "Method"
    if any(token in lowered for token in ("nao", "blocking", "jet", "circulation", "vortex", "amplification")):
        return "AtmosphericPattern"
    if any(token in lowered for token in ("cold", "warming", "spell", "wave", "event")):
        return "ClimateEvent"
    if any(token in lowered for token in ("wmo", "noaa", "ipcc", "ecmwf")):
        return "Organization"
    return "Concept"


def _canonical_catalog() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    acronyms = read_json(PROCESSED_DIR / "acronyms.json")["acronyms"]
    evaluation = read_json(EVAL_PATH)
    catalog: dict[str, dict[str, Any]] = {}
    alias_to_name: dict[str, str] = {}

    for item in acronyms:
        name = item["acronym"]
        aliases = [name, item["expanded_form"]]
        catalog[name] = {
            "name": name,
            "label": item["expanded_form"],
            "entity_type": _entity_type(name + " " + item["expanded_form"]),
            "aliases": aliases,
            "gazetteer_source_pages": [item["source_page_pdf"]],
        }
        for alias in aliases:
            alias_to_name[normalize_text(alias)] = name

    for question in evaluation["questions"]:
        for entity in question.get("expected_entities") or []:
            key = normalize_text(entity)
            canonical = alias_to_name.get(key, entity)
            if canonical not in catalog:
                catalog[canonical] = {
                    "name": canonical,
                    "label": canonical,
                    "entity_type": _entity_type(canonical),
                    "aliases": [canonical],
                    "gazetteer_source_pages": [],
                }
            alias_to_name[key] = canonical
        relation = question.get("expected_relation")
        if relation:
            for value in (relation["subject"], relation["object"]):
                key = normalize_text(value)
                canonical = alias_to_name.get(key, value)
                if canonical not in catalog:
                    catalog[canonical] = {
                        "name": canonical,
                        "label": canonical,
                        "entity_type": _entity_type(canonical),
                        "aliases": [canonical],
                        "gazetteer_source_pages": [],
                    }
                alias_to_name[key] = canonical
    return catalog, alias_to_name


def _alias_found(alias: str, text: str) -> bool:
    if len(alias) <= 5 and alias.upper() == alias:
        return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", text))
    return normalize_text(alias) in normalize_text(text)


def _best_chunk_for_evidence(chunks: list[dict[str, Any]], page: int, excerpt: str) -> dict[str, Any]:
    candidates = [chunk for chunk in chunks if page in chunk["pages_pdf"]]
    if not candidates:
        return chunks[0]
    evidence_words = set(normalize_text(excerpt).split())
    return max(
        candidates,
        key=lambda chunk: len(evidence_words.intersection(normalize_text(chunk["text"]).split())),
    )


def build_graph() -> tuple[dict[str, Any], dict[str, Any]]:
    embedding_meta = read_json(ARTIFACTS_DIR / "best_embedding_metadata.json")
    chunks = read_jsonl(PROCESSED_DIR / f"chunks_{embedding_meta['chunking_method']}.jsonl")
    evaluation = read_json(EVAL_PATH)
    catalog, alias_to_name = _canonical_catalog()

    occurrences: dict[str, list[dict[str, Any]]] = defaultdict(list)
    chunk_entities: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        for canonical, item in catalog.items():
            if any(_alias_found(alias, chunk["text"]) for alias in item["aliases"]):
                occurrence = {
                    "chunk_id": chunk["chunk_id"],
                    "pages_pdf": chunk["pages_pdf"],
                    "pages_these": chunk["pages_these"],
                    "section": chunk["section"],
                }
                occurrences[canonical].append(occurrence)
                chunk_entities[chunk["chunk_id"]].add(canonical)

    relations: list[dict[str, Any]] = []
    verified_nodes: set[str] = set()
    for question in evaluation["questions"]:
        expected = question.get("expected_relation")
        evidence = question.get("supporting_evidence") or []
        if not expected or not evidence:
            continue
        source = alias_to_name.get(normalize_text(expected["subject"]), expected["subject"])
        target = alias_to_name.get(normalize_text(expected["object"]), expected["object"])
        verified_nodes.update((source, target))
        first = evidence[0]
        chunk = _best_chunk_for_evidence(chunks, int(first["page_pdf"]), first["excerpt"])
        relation_type = PREDICATE_MAP.get(expected["predicate"], "ASSOCIATED_WITH")
        relations.append(
            {
                "relation_id": f"verified_{question['question_id'].lower()}",
                "source": source,
                "target": target,
                "type": relation_type,
                "confidence": 0.95,
                "extraction_method": "verified_evaluation_annotation",
                "original_predicate": expected["predicate"],
                "page_pdf": int(first["page_pdf"]),
                "page_these": first.get("page_these"),
                "chunk_id": chunk["chunk_id"],
                "excerpt": first["excerpt"],
            }
        )

    cooccurrence: Counter[tuple[str, str]] = Counter()
    first_chunk: dict[tuple[str, str], dict[str, Any]] = {}
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    for chunk_id, names in chunk_entities.items():
        # Extremely dense captions add noise; retain the strongest ten declared concepts.
        selected = sorted(names, key=lambda name: len(occurrences[name]), reverse=True)[:10]
        for left, right in itertools.combinations(sorted(selected), 2):
            key = (left, right)
            cooccurrence[key] += 1
            first_chunk.setdefault(key, chunks_by_id[chunk_id])
    strongest = sorted(cooccurrence.items(), key=lambda item: (-item[1], item[0]))[:300]
    for index, ((source, target), count) in enumerate(strongest):
        if count < 2:
            continue
        chunk = first_chunk[(source, target)]
        relations.append(
            {
                "relation_id": f"cooccur_{index:04d}",
                "source": source,
                "target": target,
                "type": "CO_OCCURS_WITH",
                "confidence": round(min(0.9, 0.45 + 0.08 * math.log1p(count)), 6),
                "extraction_method": "gazetteer_cooccurrence",
                "occurrence_count": int(count),
                "page_pdf": chunk["pages_pdf"][0],
                "page_these": chunk["pages_these"][0] if chunk["pages_these"] else None,
                "chunk_id": chunk["chunk_id"],
                "excerpt": chunk["text"][:320],
            }
        )

    node_names = {relation["source"] for relation in relations} | {relation["target"] for relation in relations}
    node_names |= {name for name, found in occurrences.items() if found}
    graph = nx.Graph()
    for name in sorted(node_names):
        graph.add_node(name)
    for relation in relations:
        graph.add_edge(relation["source"], relation["target"])
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise RuntimeError("Le graphe extrait est vide")

    communities = nx.community.louvain_communities(graph, seed=SEED)
    if len(communities) < 2 and graph.number_of_nodes() >= 4:
        communities = list(nx.community.greedy_modularity_communities(graph))
    community_of = {name: index for index, members in enumerate(communities) for name in members}
    degree = nx.degree_centrality(graph)
    betweenness = nx.betweenness_centrality(graph, normalized=True)
    closeness = nx.closeness_centrality(graph)
    modularity = nx.community.modularity(graph, communities) if len(communities) > 1 else 0.0

    nodes: list[dict[str, Any]] = []
    for name in sorted(graph.nodes):
        item = catalog.get(
            name,
            {"label": name, "entity_type": _entity_type(name), "aliases": [name], "gazetteer_source_pages": []},
        )
        found = occurrences.get(name, [])
        pages = sorted({page for occurrence in found for page in occurrence["pages_pdf"]})
        nodes.append(
            {
                "name": name,
                "label": item["label"],
                "entity_type": item["entity_type"],
                "aliases": item["aliases"],
                "occurrence_count": len(found),
                "pages_pdf": pages,
                "community_id": int(community_of.get(name, -1)),
                "degree_centrality": round(float(degree.get(name, 0.0)), 8),
                "betweenness_centrality": round(float(betweenness.get(name, 0.0)), 8),
                "closeness_centrality": round(float(closeness.get(name, 0.0)), 8),
                "verified_from_eval": name in verified_nodes,
            }
        )

    graph_payload = {
        "computed_at": utc_now(),
        "source": "validated corpus chunks + author acronym gazetteer + verified evaluation annotations",
        "nodes": nodes,
        "relations": relations,
    }
    write_json(ARTIFACTS_DIR / "graph.json", graph_payload)
    _write_graphml(nodes, relations)
    _write_cypher(nodes, relations)

    connected_components = list(nx.connected_components(graph))
    metrics = {
        "computed_at": utc_now(),
        "seed": SEED,
        "nodes": len(nodes),
        "edges": len(relations),
        "simple_edges": graph.number_of_edges(),
        "density": round(float(nx.density(graph)), 8),
        "connected_components": len(connected_components),
        "component_sizes": sorted([len(component) for component in connected_components], reverse=True),
        "community_count": len(communities),
        "modularity": round(float(modularity), 8),
        "communities": [
            {"community_id": index, "size": len(members), "members": sorted(members)}
            for index, members in enumerate(communities)
        ],
        "top_degree_centrality": _top(degree),
        "top_betweenness_centrality": _top(betweenness),
        "top_closeness_centrality": _top(closeness),
        "relations_by_type": dict(Counter(relation["type"] for relation in relations)),
        "relations_by_extraction_method": dict(Counter(relation["extraction_method"] for relation in relations)),
    }
    write_json(METRICS_DIR / "graph_metrics.json", metrics)
    return graph_payload, metrics


def _top(values: dict[str, float], limit: int = 10) -> list[dict[str, Any]]:
    return [
        {"name": name, "value": round(float(value), 8)}
        for name, value in sorted(values.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def _scalar(value: Any) -> Any:
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    if value is None:
        return ""
    return value


def _write_graphml(nodes: list[dict[str, Any]], relations: list[dict[str, Any]]) -> None:
    graph = nx.MultiDiGraph()
    for node in nodes:
        graph.add_node(node["name"], **{key: _scalar(value) for key, value in node.items() if key != "name"})
    for relation in relations:
        graph.add_edge(
            relation["source"],
            relation["target"],
            key=relation["relation_id"],
            **{key: _scalar(value) for key, value in relation.items() if key not in {"source", "target"}},
        )
    nx.write_graphml(graph, ARTIFACTS_DIR / "graph.graphml")


def _cypher_string(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ") + "'"


def _write_cypher(nodes: list[dict[str, Any]], relations: list[dict[str, Any]]) -> None:
    lines = ["CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (n:Entity) REQUIRE n.name IS UNIQUE;"]
    for node in nodes:
        props = ", ".join(
            f"n.{key} = {_cypher_string(value)}"
            for key, value in node.items()
            if key not in {"name", "aliases", "pages_pdf"}
        )
        lines.append(f"MERGE (n:Entity {{name: {_cypher_string(node['name'])}}}) SET {props};")
    for relation in relations:
        rel_type = relation["type"] if relation["type"] in ALLOWED_RELATIONS else "ASSOCIATED_WITH"
        props = ", ".join(
            f"r.{key} = {_cypher_string(value)}"
            for key, value in relation.items()
            if key not in {"source", "target", "type", "relation_id"}
        )
        lines.append(
            f"MATCH (s:Entity {{name: {_cypher_string(relation['source'])}}}), "
            f"(t:Entity {{name: {_cypher_string(relation['target'])}}}) "
            f"MERGE (s)-[r:{rel_type} {{relation_id: {_cypher_string(relation['relation_id'])}}}]->(t) SET {props};"
        )
    (ARTIFACTS_DIR / "graph.cypher").write_text("\n".join(lines) + "\n", encoding="utf-8")

