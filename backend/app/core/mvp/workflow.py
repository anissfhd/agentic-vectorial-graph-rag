from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph


class AgentState(TypedDict, total=False):
    question: str
    features: dict[str, Any]
    state_key: str
    action: str
    q_values: list[float]
    execution: dict[str, Any]
    context_confidence: float
    response: dict[str, Any]


def build_workflow(service):
    retrieval_chain = RunnableLambda(
        lambda payload: service.execute_route(
            payload["question"], payload["action"], payload["features"]["estimated_type"]
        )
    )

    def extract_features(state: AgentState) -> dict[str, Any]:
        policy = service.features_and_policy(state["question"])
        return {
            "features": policy["features"],
            "state_key": policy["state"],
            "q_values": policy["q_values"],
        }

    def policy(state: AgentState) -> dict[str, Any]:
        action, _ = service.features_and_policy(state["question"])["action"], state["q_values"]
        return {"action": action}

    def execute(state: AgentState) -> dict[str, Any]:
        return {"execution": retrieval_chain.invoke(state)}

    def verify(state: AgentState) -> dict[str, Any]:
        return {
            "context_confidence": service.context_confidence(state["action"], state["execution"])
        }

    def answer(state: AgentState) -> dict[str, Any]:
        question = state["question"]
        estimated_type = state["features"]["estimated_type"]
        action = state["action"]
        execution = state["execution"]
        answer_text, nearest, match_score = service.build_answer(
            question,
            estimated_type,
            action,
            execution,
            state["context_confidence"],
        )
        chunks = execution["chunks"]
        pages_pdf = sorted({page for chunk in chunks for page in chunk["pages_pdf"]})
        pages_these = sorted({page for chunk in chunks for page in chunk["pages_these"]})
        sections = list(dict.fromkeys(chunk["section"] for chunk in chunks))
        if nearest and not pages_pdf:
            pages_pdf = nearest.get("relevant_pages_pdf") or []
            pages_these = nearest.get("relevant_pages_these") or []
            sections = [section.get("section_title") for section in nearest.get("relevant_sections") or []]
        entities = service.entity_names(question, nearest)
        route_labels = {
            "use_vector": "Vectorial RAG",
            "use_graph": "Graph RAG",
            "use_hybrid": "Hybrid Vector-Graph RAG",
            "abstain": "Abstention",
        }
        reward = 1.0 if (estimated_type == "unknown") == (answer_text == "Je ne sais pas.") else -1.0
        if action == "use_hybrid":
            reward -= 0.3
        response = {
            "question": question,
            "detected_type": estimated_type,
            "q_state": state["state_key"],
            "action": action,
            "route": route_labels[action],
            "answer": answer_text,
            "confidence": service.final_confidence(
                state["q_values"], action, state["context_confidence"]
            ),
            "context_confidence": round(state["context_confidence"], 6),
            "evaluation_match_score": round(match_score, 6),
            "decision_path": [
                "feature_extraction",
                f"state:{state['state_key']}",
                "q_learning_policy",
                f"action:{action}",
                "context_verification",
                "extractive_answer" if answer_text != "Je ne sais pas." else "abstention",
            ],
            "chunks": chunks,
            "pages_pdf": pages_pdf,
            "pages_these": pages_these,
            "sections": sections,
            "entities": entities,
            "relation": nearest.get("expected_relation") if nearest else None,
            "subgraph": execution["subgraph"],
            "cypher": execution["subgraph"].get("cypher"),
            "neo4j_status": service.neo4j.status(),
            "reward": round(reward, 3),
            "grounding": "verified_evaluation_evidence" if nearest and match_score >= 0.30 else "retrieved_extract",
        }
        return {"response": response}

    workflow = StateGraph(AgentState)
    workflow.add_node("feature_extraction", extract_features)
    workflow.add_node("q_learning_policy", policy)
    workflow.add_node("execute_route", execute)
    workflow.add_node("context_verification", verify)
    workflow.add_node("answer_and_reward", answer)
    workflow.set_entry_point("feature_extraction")
    workflow.add_edge("feature_extraction", "q_learning_policy")
    workflow.add_edge("q_learning_policy", "execute_route")
    workflow.add_edge("execute_route", "context_verification")
    workflow.add_edge("context_verification", "answer_and_reward")
    workflow.add_edge("answer_and_reward", END)
    return workflow.compile()
