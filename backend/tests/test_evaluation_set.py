"""Tests du jeu d'evaluation (etape 2).

Ils verifient la STRUCTURE du jeu (comptes, splits, routes) et surtout son
ANCRAGE : chaque page citee doit exister dans le corpus nettoye, chaque extrait de
preuve doit s'y retrouver reellement, et aucune page ne doit provenir de l'annexe B.

    pytest backend/tests -v
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
EVAL_FILE = BACKEND_DIR / "eval" / "eval_set.json"
CORPUS = BACKEND_DIR / "data" / "processed" / "corpus_clean.jsonl"
REPORT = BACKEND_DIR / "data" / "metrics" / "evaluation_set_report.json"

TYPES = ("semantic", "relational", "hybrid", "out_of_context")
ROUTE_BY_TYPE = {
    "semantic": "use_vector",
    "relational": "use_graph",
    "hybrid": "use_hybrid",
    "out_of_context": "abstain",
}
VALID_ROUTES = set(ROUTE_BY_TYPE.values())
IDK = "Je ne sais pas."
ANNEX_B_PAGES = set(range(150, 174))


def _normalise(text: str) -> str:
    """Tolerance a la normalisation Unicode et aux espaces, comme le demande l'etape."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "")).strip()


@pytest.fixture(scope="module")
def questions():
    if not EVAL_FILE.exists():
        pytest.skip("eval_set.json absent : lancer backend/scripts/02_build_eval_set.py")
    return json.loads(EVAL_FILE.read_text(encoding="utf-8"))["questions"]


@pytest.fixture(scope="module")
def corpus():
    if not CORPUS.exists():
        pytest.skip("corpus_clean.jsonl absent : lancer backend/scripts/01_ingest_pdf.py")
    with CORPUS.open(encoding="utf-8") as fh:
        return {json.loads(l)["page_pdf"]: json.loads(l) for l in fh if l.strip()}


@pytest.fixture(scope="module")
def answerable(questions):
    return [q for q in questions if q["question_type"] != "out_of_context"]


# =====================================================================
# 1. Structure du jeu
# =====================================================================
def test_exactement_24_questions(questions):
    assert len(questions) == 24


def test_six_questions_par_type(questions):
    counts = Counter(q["question_type"] for q in questions)
    assert set(counts) == set(TYPES)
    for t in TYPES:
        assert counts[t] == 6, f"{t}: {counts[t]} questions au lieu de 6"


@pytest.mark.parametrize("qtype", TYPES)
def test_difficulte_2_2_2_par_type(questions, qtype):
    counts = Counter(q["difficulty"] for q in questions if q["question_type"] == qtype)
    assert counts == {"easy": 2, "medium": 2, "hard": 2}, f"{qtype}: {dict(counts)}"


@pytest.mark.parametrize("qtype", TYPES)
def test_split_4_train_2_test_par_type(questions, qtype):
    counts = Counter(q["split"] for q in questions if q["question_type"] == qtype)
    assert counts == {"train": 4, "test": 2}, f"{qtype}: {dict(counts)}"


def test_split_global_16_train_8_test(questions):
    counts = Counter(q["split"] for q in questions)
    assert counts["train"] == 16 and counts["test"] == 8


def test_identifiants_uniques(questions):
    ids = [q["question_id"] for q in questions]
    assert len(set(ids)) == len(ids), "identifiants dupliques"


def test_questions_non_vides(questions):
    for q in questions:
        assert q["question"].strip(), f"{q['question_id']} : question vide"
        assert len(q["question"].split()) >= 4
        assert q["expected_answer"].strip()


def test_aucune_question_dupliquee(questions):
    textes = [_normalise(q["question"]).lower() for q in questions]
    doublons = [t for t, n in Counter(textes).items() if n > 1]
    assert not doublons, f"questions dupliquees : {doublons}"


# =====================================================================
# 2. Routes attendues
# =====================================================================
def test_toutes_les_routes_sont_valides(questions):
    for q in questions:
        assert q["expected_route"] in VALID_ROUTES


@pytest.mark.parametrize("qtype,route", list(ROUTE_BY_TYPE.items()))
def test_route_coherente_avec_le_type(questions, qtype, route):
    for q in questions:
        if q["question_type"] == qtype:
            assert q["expected_route"] == route, f"{q['question_id']}"


# =====================================================================
# 3. Ancrage dans le corpus
# =====================================================================
def test_les_pages_pertinentes_existent_dans_le_corpus(answerable, corpus):
    for q in answerable:
        assert q["relevant_pages_pdf"], f"{q['question_id']} : aucune page"
        for page in q["relevant_pages_pdf"]:
            assert page in corpus, (
                f"{q['question_id']} : p{page} absente de corpus_clean.jsonl")
            assert corpus[page]["keep_for_rag"] is True


def test_aucune_page_ne_vient_de_l_annexe_b(questions):
    for q in questions:
        fautives = set(q["relevant_pages_pdf"]) & ANNEX_B_PAGES
        assert not fautives, f"{q['question_id']} : pages d'annexe B {sorted(fautives)}"


def test_chaque_question_repondable_possede_une_preuve(answerable):
    for q in answerable:
        assert q["supporting_evidence"], f"{q['question_id']} : aucune preuve"
        for ev in q["supporting_evidence"]:
            assert ev["excerpt"].strip()
            assert len(ev["excerpt"].split()) >= 5
            assert ev["language"] in ("fr", "en")


def test_les_extraits_de_preuve_existent_reellement(answerable, corpus):
    """Coeur du controle : la preuve doit se retrouver dans la page annoncee."""
    for q in answerable:
        for ev in q["supporting_evidence"]:
            page = corpus[ev["page_pdf"]]
            assert _normalise(ev["excerpt"]) in _normalise(page["clean_text"]), (
                f"{q['question_id']} : extrait absent de p{ev['page_pdf']}")


def test_les_pages_de_preuve_sont_declarees_pertinentes(answerable):
    for q in answerable:
        pages = {ev["page_pdf"] for ev in q["supporting_evidence"]}
        assert pages <= set(q["relevant_pages_pdf"]), f"{q['question_id']}"


def test_sections_pertinentes_renseignees(answerable):
    for q in answerable:
        assert q["relevant_sections"], f"{q['question_id']} : aucune section"


# =====================================================================
# 4. Entites et relations
# =====================================================================
@pytest.mark.parametrize("qtype", ["relational", "hybrid"])
def test_au_moins_deux_entites(questions, qtype):
    for q in questions:
        if q["question_type"] == qtype:
            assert len(q["expected_entities"]) >= 2, (
                f"{q['question_id']} : {q['expected_entities']}")


@pytest.mark.parametrize("qtype", ["relational", "hybrid"])
def test_relation_attendue_renseignee(questions, qtype):
    for q in questions:
        if q["question_type"] == qtype:
            relation = q["expected_relation"]
            assert relation, f"{q['question_id']} : relation manquante"
            for champ in ("subject", "predicate", "object"):
                assert relation.get(champ), f"{q['question_id']} : {champ} vide"


def test_hybrides_appuyees_sur_plusieurs_passages(questions):
    for q in questions:
        if q["question_type"] == "hybrid":
            assert len(q["supporting_evidence"]) >= 2, (
                f"{q['question_id']} : une seule preuve pour une question hybride")


# =====================================================================
# 5. Questions hors contexte
# =====================================================================
def test_hors_contexte_conforme(questions):
    for q in questions:
        if q["question_type"] != "out_of_context":
            continue
        assert q["expected_answer"] == IDK, f"{q['question_id']}"
        assert q["relevant_pages_pdf"] == []
        assert q["relevant_pages_these"] == []
        assert q["relevant_sections"] == []
        assert q["supporting_evidence"] == []
        assert q["expected_entities"] == []
        assert q["expected_relation"] is None
        assert q["expected_route"] == "abstain"
        assert q["out_of_context_origin"] in ("annex_b", "unrelated")


def test_trois_hors_contexte_annexe_b_et_trois_etrangeres(questions):
    origines = Counter(q["out_of_context_origin"] for q in questions
                       if q["question_type"] == "out_of_context")
    assert origines["annex_b"] == 3, f"annexe B : {origines['annex_b']}"
    assert origines["unrelated"] == 3, f"etrangeres : {origines['unrelated']}"


def test_seules_les_hors_contexte_repondent_je_ne_sais_pas(answerable):
    for q in answerable:
        assert q["expected_answer"] != IDK, f"{q['question_id']}"
        assert q["out_of_context_origin"] is None


# =====================================================================
# 6. Chunks et pagination
# =====================================================================
def test_relevant_chunk_ids_vide_partout(questions):
    """Les methodes de chunking ne sont pas encore executees."""
    for q in questions:
        assert q["relevant_chunk_ids"] == [], f"{q['question_id']}"


def test_pagination_pdf_these_coherente(answerable, corpus):
    for q in answerable:
        attendu = sorted(corpus[p]["page_these"] for p in q["relevant_pages_pdf"]
                         if corpus[p]["page_these"] is not None)
        assert q["relevant_pages_these"] == attendu, f"{q['question_id']}"
        for these in q["relevant_pages_these"]:
            assert (these + 15) in q["relevant_pages_pdf"], f"{q['question_id']}"
        for ev in q["supporting_evidence"]:
            page = corpus[ev["page_pdf"]]
            assert ev["page_these"] == page["page_these"], f"{q['question_id']}"


def test_statut_d_annotation(questions):
    for q in questions:
        assert q["annotation_status"] == "verified"


# =====================================================================
# 7. Rapport de metriques
# =====================================================================
def test_rapport_evaluation_coherent(questions):
    if not REPORT.exists():
        pytest.skip("evaluation_set_report.json absent")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["nombre_total_de_questions"] == len(questions) == 24
    assert report["par_type"] == {t: 6 for t in TYPES}
    assert report["par_split"] == {"train": 16, "test": 8}
    assert report["doublons_detectes"] == []
    assert report["erreurs_de_pagination"] == []
    assert report["questions_sans_page_pertinente"] == 6  # les hors contexte
    assert report["nombre_de_chapitres_couverts"] >= 4
    # Le controle cross-lingue est secondaire : il doit exister mais ne conditionne
    # jamais la verite terrain.
    assert 0.0 <= report["average_positive_evidence_similarity"] <= 1.0
    assert isinstance(report["questions_with_similarity_warning"], list)
