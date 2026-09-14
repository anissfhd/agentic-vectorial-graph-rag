"""Tests de l'etape 1 - ingestion, extraction, nettoyage.

Deux familles :
  * des tests UNITAIRES sur les fonctions pures (pagination, Unicode, cesures,
    ordre de lecture en double colonne, politique d'inclusion) ;
  * des tests d'INTEGRATION sur les artefacts reellement produits par
    `backend/scripts/01_ingest_pdf.py`. Ceux-ci sont ignores proprement tant que
    le script n'a pas ete execute, afin que la suite reste lancable sur une copie
    fraiche du depot (les donnees ne sont pas versionnees).

    pytest backend/tests -v
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from app.core.ingestion import (
    build_context,
    clean_page,
    consolidate_captions,
    decide_inclusion,
    detect_language,
    detect_layout,
    normalise_unicode,
    reading_order,
    repair_hyphenation,
    rule_thesis_page,
    zone_of,
)
from app.core.ingestion.cleaner import (
    CleaningContext,
    CleanStats,
    is_page_number_line,
)
from app.core.ingestion.pdf_reader import (
    LAYOUT_DOUBLE,
    LAYOUT_SINGLE,
    Block,
    RawPage,
    _classify_band,
    _classify_column,
    group_paragraphs,
)
from app.core.ingestion.quality import column_alternations, naive_reading_order
from app.core.ingestion.structure import ZONE_APPENDIX_B, ZONE_CHAPTERS

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROCESSED = BACKEND_DIR / "data" / "processed"
METRICS = BACKEND_DIR / "data" / "metrics"
EXPECTED_PAGES = 206
PAGE_W, PAGE_H = 595.0, 842.0


# =====================================================================
# Outils de construction de pages synthetiques
# =====================================================================
def make_block(x0, y0, x1, y1, text) -> Block:
    b = Block(x0=x0, y0=y0, x1=x1, y1=y1, text=text)
    b.band = _classify_band(b, PAGE_H)
    b.column = _classify_column(b, PAGE_W)
    return b


def make_page(blocks, page_pdf=1, layout=None) -> RawPage:
    layout = layout or detect_layout(blocks, PAGE_W)
    ordered = reading_order(blocks, layout)
    return RawPage(
        page_pdf=page_pdf, width=PAGE_W, height=PAGE_H, blocks=ordered,
        layout=layout, raw_text="", groups=group_paragraphs(ordered, PAGE_W),
    )


def empty_context() -> CleaningContext:
    return CleaningContext(running_heads=set(), word_freq=Counter(), hyphen_freq=Counter())


# =====================================================================
# 1. Pagination PDF / these
# =====================================================================
def test_pagination_regle_de_base():
    """page_these = page_pdf - 15, a partir de la page PDF 16."""
    assert rule_thesis_page(16) == 1
    assert rule_thesis_page(19) == 4
    assert rule_thesis_page(130) == 115
    assert rule_thesis_page(206) == 191


def test_pagination_pages_liminaires_sans_numero():
    """Les pages PDF 1 a 15 sont en chiffres romains : aucun numero n'est fabrique."""
    for page in range(1, 16):
        assert rule_thesis_page(page) is None


# =====================================================================
# 2. Normalisation Unicode
# =====================================================================
def test_normalisation_ligatures():
    assert normalise_unicode("Deﬁnition") == "Definition"
    assert normalise_unicode("conﬂict") == "conflict"
    assert normalise_unicode("eﬀect") == "effect"


def test_normalisation_tirets_et_guillemets():
    assert normalise_unicode("−10 °C") == "-10 °C"
    assert normalise_unicode("1950–1999") == "1950-1999"
    assert normalise_unicode("l’hiver") == "l'hiver"


def test_normalisation_preserve_les_unites_scientifiques():
    """NFKC transformerait « km² » en « km2 » : on utilise NFC + tables explicites."""
    for unit in ("km²", "m³", "10⁻³", "°C", "µm"):
        assert normalise_unicode(unit) == unit


# =====================================================================
# 3. Reparation des cesures de fin de ligne
# =====================================================================
def test_cesure_recollee_quand_le_mot_soude_existe():
    ctx = empty_context()
    ctx.word_freq["considered"] = 40
    stats = CleanStats()
    assert repair_hyphenation("con-\nsidered", ctx, stats) == "considered"
    assert stats.hyphens_repaired == 1


def test_cesure_conservee_quand_le_compose_est_plus_frequent():
    """« well-known » doit rester compose : la decision vient des frequences."""
    ctx = empty_context()
    ctx.hyphen_freq["well-known"] = 12
    ctx.word_freq["wellknown"] = 0
    stats = CleanStats()
    assert repair_hyphenation("well-\nknown", ctx, stats) == "well-known"
    assert stats.hyphens_kept == 1


def test_cesure_entre_deux_paragraphes_consecutifs():
    """Cas des listes de references coupees par un changement de colonne."""
    ctx = empty_context()
    ctx.word_freq["extreme"] = 100
    blocks = [
        make_block(46, 100, 287, 130, "Computation of ex-"),
        make_block(307, 100, 548, 130, "treme heat waves in climate models."),
    ]
    page = make_page(blocks)
    text, _ = clean_page(page, ctx)
    assert "extreme heat waves" in text
    assert "ex-" not in text


# =====================================================================
# 4. Numeros de page isoles
# =====================================================================
def test_reconnaissance_des_numeros_de_page():
    assert is_page_number_line("4")
    assert is_page_number_line("115")
    assert is_page_number_line("xii")
    assert not is_page_number_line("1963 was the coldest winter")
    assert not is_page_number_line("")


def test_numero_de_page_isole_supprime_en_pied_de_page():
    ctx = empty_context()
    blocks = [
        make_block(61, 200, 458, 400, "Le contenu scientifique de la page."),
        make_block(256, 762, 262, 774, "4"),          # pied de page
    ]
    page = make_page(blocks)
    text, stats = clean_page(page, ctx, page_number_tokens=frozenset({"4"}))
    assert "contenu scientifique" in text
    assert "\n\n4" not in text and not text.strip().endswith("4")
    assert stats.page_numbers_removed >= 1


def test_numero_etranger_en_bas_de_page_supprime():
    """Les articles inseres impriment leur propre numero, a 0.849 de la hauteur."""
    ctx = empty_context()
    blocks = [
        make_block(46, 200, 548, 400, "Texte de l'article insere."),
        make_block(294, 714, 300, 727, "1"),
    ]
    page = make_page(blocks, page_pdf=88)
    text, stats = clean_page(page, ctx, page_number_tokens=frozenset({"88", "73"}))
    assert "article insere" in text
    assert stats.page_numbers_removed >= 1


# =====================================================================
# 5. Conservation des vraies valeurs scientifiques
# =====================================================================
def test_nombre_scientifique_conserve_dans_une_phrase():
    ctx = empty_context()
    blocks = [make_block(
        61, 200, 458, 300,
        "The winter 1963 reached -10.5 °C over 15 days, i.e. 1000 simulations.",
    )]
    page = make_page(blocks)
    text, _ = clean_page(page, ctx)
    for value in ("1963", "-10.5", "°C", "15", "1000"):
        assert value in text


def test_etiquette_numerique_de_figure_conservee_hors_marge():
    """Un nombre isole au milieu de la page est un label d'axe, pas un numero de page."""
    ctx = empty_context()
    blocks = [
        make_block(61, 150, 458, 240, "Figure 2.3: distribution des temperatures."),
        make_block(143, 380, 150, 392, "500"),   # label d'axe, en plein milieu
    ]
    page = make_page(blocks, page_pdf=39)
    text, _ = clean_page(page, ctx, page_number_tokens=frozenset({"39", "24"}))
    assert "500" in text


# =====================================================================
# 6. Ordre de lecture en double colonne
# =====================================================================
def _two_column_blocks():
    left = [make_block(46, y, 287, y + 60, f"gauche {i}") for i, y in enumerate((100, 200, 300))]
    right = [make_block(307, y, 548, y + 60, f"droite {i}") for i, y in enumerate((100, 200, 300))]
    return left + right


def test_detection_double_colonne():
    assert detect_layout(_two_column_blocks(), PAGE_W) == LAYOUT_DOUBLE


def test_detection_simple_colonne():
    blocks = [make_block(61, y, 458, y + 60, f"paragraphe {i}")
              for i, y in enumerate((100, 200, 300))]
    assert detect_layout(blocks, PAGE_W) == LAYOUT_SINGLE


def test_ordre_de_lecture_ne_melange_pas_les_colonnes():
    """Toute la colonne gauche doit sortir avant la colonne droite."""
    ordered = reading_order(_two_column_blocks(), LAYOUT_DOUBLE)
    colonnes = [b.column for b in ordered if b.band == "body"]
    assert colonnes == ["left"] * 3 + ["right"] * 3
    # Un seul basculement gauche->droite, contre un par ligne en ordre naif.
    assert column_alternations(ordered) == 1
    assert column_alternations(naive_reading_order(_two_column_blocks())) > 1


def test_bloc_pleine_largeur_decoupe_la_page_en_bandes():
    """Un titre pleine largeur au milieu separe deux zones a deux colonnes."""
    blocks = _two_column_blocks() + [make_block(46, 360, 548, 390, "TITRE PLEINE LARGEUR")]
    ordered = reading_order(blocks, LAYOUT_DOUBLE)
    textes = [b.text for b in ordered]
    assert textes.index("gauche 2") < textes.index("TITRE PLEINE LARGEUR")
    assert textes.index("droite 0") < textes.index("TITRE PLEINE LARGEUR")


# =====================================================================
# 7. Reconstruction des paragraphes
# =====================================================================
def test_lignes_justifiees_regroupees_en_un_paragraphe():
    """Une ligne qui atteint la marge droite continue le paragraphe."""
    blocks = [
        make_block(46, 136, 548, 148, "To evaluate the impact of climate change,"),
        make_block(46, 153, 548, 165, "we make SWG simulations of cold events"),
        make_block(46, 170, 300, 182, "for each model."),      # ligne courte : fin
        make_block(46, 200, 548, 212, "A new paragraph starts here and is long"),
    ]
    groups = group_paragraphs(blocks, PAGE_W)
    assert len(groups) == 2
    assert len(groups[0]) == 3


# =====================================================================
# 8. Politique d'inclusion / exclusion
# =====================================================================
@pytest.mark.parametrize("page,zone", [
    (3, "front_matter"), (7, "summaries"), (12, "acknowledgements"),
    (14, "toc"), (60, "chapters"), (128, "acronyms"), (135, "bibliography"),
    (140, "lists_figures_tables"), (146, "appendix_a"), (160, "appendix_b"),
    (190, "appendix_c"),
])
def test_zones_documentaires(page, zone):
    assert zone_of(page) == zone


def test_plages_conservees():
    texte = "Contenu scientifique reel. " * 20
    for page in (7, 60, 146):
        keep, reason = decide_inclusion(page, zone_of(page), texte, "single_column")
        assert keep is True and reason is None


@pytest.mark.parametrize("page", [3, 12, 14, 135, 140, 160, 128])
def test_plages_exclues(page):
    texte = "Contenu quelconque. " * 20
    keep, reason = decide_inclusion(page, zone_of(page), texte, "single_column")
    assert keep is False and reason


def test_page_blanche_exclue_meme_dans_une_plage_conservee():
    keep, reason = decide_inclusion(60, ZONE_CHAPTERS, "", "single_column")
    assert keep is False and reason == "page_blanche"


def test_annexe_b_toujours_exclue_du_corpus_principal():
    texte = "Australian rain bomb. " * 50
    keep, reason = decide_inclusion(160, ZONE_APPENDIX_B, texte, "double_column")
    assert keep is False and reason == "annexe_b_hors_sujet"


# =====================================================================
# 9. Detection de langue
# =====================================================================
def test_detection_langue_francais():
    lang, conf = detect_language(
        "Il est aujourd'hui communement admis par la communaute scientifique que "
        "les emissions de gaz a effet de serre entrainent un rechauffement."
    )
    assert lang == "fr" and conf > 0.5


def test_detection_langue_anglais():
    lang, conf = detect_language(
        "The lack of consistency in the definition of extreme events is also found "
        "in the definition of cold spells, which are studied with this method."
    )
    assert lang == "en" and conf > 0.5


# =====================================================================
# 10. Tests d'integration sur les artefacts produits
# =====================================================================
def _load_jsonl(path: Path):
    if not path.exists():
        pytest.skip(f"{path.name} absent : lancer d'abord backend/scripts/01_ingest_pdf.py")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _load_json(path: Path):
    if not path.exists():
        pytest.skip(f"{path.name} absent : lancer d'abord backend/scripts/01_ingest_pdf.py")
    return json.loads(path.read_text(encoding="utf-8"))


def test_serialisation_jsonl_et_champs_obligatoires():
    """Chaque ligne est un JSON valide portant tous les champs du contrat."""
    records = _load_jsonl(PROCESSED / "corpus_pages_raw.jsonl")
    attendus = {
        "document_id", "source_file", "source_sha256", "page_pdf", "page_these",
        "chapter_number", "chapter_title", "section_number", "section_title",
        "section_inherited", "language", "content_type", "extraction_mode",
        "keep_for_rag", "exclusion_reason", "raw_text", "clean_text",
        "raw_char_count", "clean_char_count", "estimated_word_count", "warnings",
    }
    for rec in records:
        assert attendus <= set(rec), f"champs manquants page {rec.get('page_pdf')}"
        assert isinstance(rec["keep_for_rag"], bool)
        assert isinstance(rec["warnings"], list)
        assert rec["clean_char_count"] == len(rec["clean_text"])


def test_nombre_exact_de_pages_sans_trou_ni_doublon():
    records = _load_jsonl(PROCESSED / "corpus_pages_raw.jsonl")
    assert len(records) == EXPECTED_PAGES
    pages = [r["page_pdf"] for r in records]
    assert sorted(pages) == list(range(1, EXPECTED_PAGES + 1))


def test_pagination_coherente_dans_les_artefacts():
    records = _load_jsonl(PROCESSED / "corpus_pages_raw.jsonl")
    for rec in records:
        if rec["page_pdf"] <= 15:
            assert rec["page_these"] is None
        else:
            assert rec["page_these"] == rec["page_pdf"] - 15


def test_annexe_b_absente_du_corpus_principal():
    kept = _load_jsonl(PROCESSED / "corpus_clean.jsonl")
    fautives = [r["page_pdf"] for r in kept if 150 <= r["page_pdf"] <= 173]
    assert fautives == []


def test_annexe_b_presente_dans_son_fichier_dedie():
    annexe = _load_jsonl(PROCESSED / "annex_b_out_of_context.jsonl")
    assert len(annexe) == 24
    assert {r["page_pdf"] for r in annexe} == set(range(150, 174))
    assert sum(len(r["clean_text"]) for r in annexe) > 10_000


def test_remerciements_et_bibliographie_absents_du_corpus():
    kept = _load_jsonl(PROCESSED / "corpus_clean.jsonl")
    pages = {r["page_pdf"] for r in kept}
    assert not (pages & set(range(12, 14)))      # remerciements
    assert not (pages & set(range(130, 140)))    # bibliographie
    assert not (pages & set(range(14, 16)))      # table des matieres


def test_acronymes_extraits():
    data = _load_json(PROCESSED / "acronyms.json")
    entries = {a["acronym"]: a for a in data["acronyms"]}
    for attendu in ("SWG", "NAO", "CMIP6", "AMOC", "WMO", "SSW", "GKTL"):
        assert attendu in entries, f"acronyme {attendu} absent"
    assert entries["SWG"]["expanded_form"] == "Stochastic Weather Generator"
    assert entries["NAO"]["expanded_form"] == "North Atlantic Oscillation"
    assert entries["SWG"]["referenced_pages"]
    for entry in data["acronyms"]:
        assert entry["source_page_pdf"] in (128, 129)


def test_termes_scientifiques_presents_dans_le_corpus():
    kept = _load_jsonl(PROCESSED / "corpus_clean.jsonl")
    corpus = "\n".join(r["clean_text"] for r in kept).lower()
    for terme in ("swg", "stochastic weather generator", "nao",
                  "north atlantic oscillation", "cmip6", "scandinavian blocking",
                  "polar vortex", "jet stream", "winter 1963", "cold spells",
                  "climate change"):
        assert terme in corpus, f"terme scientifique absent du corpus : {terme}"


def test_corpus_propre_sans_entete_courant_ni_cesure():
    rapport = _load_json(METRICS / "ingestion_report.json")
    nettoyage = rapport["controles"]["C_nettoyage"]
    assert nettoyage["en_tetes_repetes_residuels"] == {}
    assert nettoyage["numeros_de_page_isoles_residuels"] == 0
    assert nettoyage["cesures_non_reparees_dans_la_page"] == 0
    assert nettoyage["pages_conservees_vides"] == []


def test_colonnes_jamais_melangees_sur_les_pages_temoins():
    rapport = _load_json(METRICS / "ingestion_report.json")
    for controle in rapport["controles"]["B_double_colonne"]:
        assert controle["colonnes_non_melangees"], f"page {controle['page_pdf']}"


def test_aucun_ocr_necessaire():
    rapport = _load_json(METRICS / "ingestion_report.json")
    assert rapport["source"]["ocr_necessaire"] is False
    assert rapport["source"]["pages"] == EXPECTED_PAGES


# =====================================================================
# 11. Etape 1B - finalisation du corpus
# =====================================================================
def test_bibliographies_articles_sorties_du_corpus():
    """Les listes de references finales des articles ne sont plus indexees."""
    refs = _load_jsonl(PROCESSED / "article_references.jsonl")
    assert refs, "aucune bibliographie d'article detectee"
    champs = {"page_pdf", "page_these", "chapter", "article", "section",
              "raw_text", "clean_text", "exclusion_reason"}
    for rec in refs:
        assert champs <= set(rec)
        assert rec["exclusion_reason"] == "references_internes_article"
        assert rec["clean_text"].strip()
        assert rec["page_these"] == rec["page_pdf"] - 15

    kept_pages = {r["page_pdf"] for r in _load_jsonl(PROCESSED / "corpus_clean.jsonl")}
    # Les pages entierement bibliographiques ont quitte le corpus principal.
    entieres = {r["page_pdf"] for r in refs if not r["partial_page"]}
    assert not (kept_pages & entieres)


def test_pages_de_references_partielles_gardent_leur_partie_scientifique():
    """Une page mi-science mi-bibliographie garde sa partie scientifique."""
    refs = _load_jsonl(PROCESSED / "article_references.jsonl")
    partielles = [r for r in refs if r["partial_page"]]
    assert partielles, "aucune page coupee detectee"
    corpus = {r["page_pdf"]: r for r in _load_jsonl(PROCESSED / "corpus_clean.jsonl")}
    for rec in partielles:
        page = corpus.get(rec["page_pdf"])
        assert page is not None, f"p{rec['page_pdf']} a disparu du corpus"
        assert page["clean_char_count"] > 0
        # La partie conservee ne doit plus contenir le titre de bibliographie.
        assert "\n\nReferences\n\n" not in page["clean_text"]


def test_citations_normales_preservees():
    """Une citation au fil d'une phrase n'est jamais supprimee."""
    kept = _load_jsonl(PROCESSED / "corpus_clean.jsonl")
    corpus = "\n".join(r["clean_text"] for r in kept)
    assert corpus.count("et al.") > 50, "les citations en ligne ont ete perdues"
    assert "(Hersbach et al., 2020)" in corpus


def test_annexe_c_legendes_consolidees():
    """Les legendes de l'annexe C sont completes et debarrassees des labels."""
    kept = [r for r in _load_jsonl(PROCESSED / "corpus_clean.jsonl")
            if 174 <= r["page_pdf"] <= 206]
    assert kept, "annexe C entierement perdue"

    # La regle des 12 mots ne vaut que pour les PLANCHES de figures. La page de
    # titre du supplement (p175) est de la prose : elle contient legitimement des
    # lignes courtes (auteurs, affiliation).
    planches = [r for r in kept if r["detected_layout"] == "figure_or_sparse"]
    assert len(planches) >= 25, "les planches de l'annexe C ont ete perdues"
    for rec in planches:
        for segment in rec["clean_text"].split("\n\n"):
            assert len(segment.split()) >= 12, (
                f"p{rec['page_pdf']} : segment trop fragmente -> {segment!r}")
        # Aucun label de panneau residuel en tete de legende.
        assert not rec["clean_text"].startswith("ssp")
        assert rec["clean_text"].startswith(("Fig.", "Figure", "Table", "Tab."))

    prose = [r for r in kept if r["detected_layout"] != "figure_or_sparse"]
    assert prose, "la page de titre du supplement a disparu"


def test_reparations_de_frontieres_de_pages():
    """Chaque mot coupe par une fin de page est decrit, sans fusionner les pages."""
    data = _load_json(PROCESSED / "page_boundary_repairs.json")
    repairs = data["repairs"]
    assert repairs, "aucune reparation detectee"
    assert data["count"] == len(repairs)

    raw = {r["page_pdf"]: r for r in _load_jsonl(PROCESSED / "corpus_pages_raw.jsonl")}
    for rep in repairs:
        for champ in ("left_page_pdf", "right_page_pdf", "left_fragment",
                      "right_fragment", "repaired_token", "confidence", "reason"):
            assert champ in rep, f"champ {champ} manquant"
        assert 0.0 <= rep["confidence"] <= 1.0
        assert rep["right_page_pdf"] > rep["left_page_pdf"]
        # Les valeurs proviennent reellement du PDF.
        gauche = raw[rep["left_page_pdf"]]["clean_text"].rstrip()
        assert gauche.endswith(rep["left_fragment"]), (
            f"p{rep['left_page_pdf']} ne se termine pas par {rep['left_fragment']!r}")
        droite = raw[rep["right_page_pdf"]]["clean_text"]
        assert rep["right_fragment"] in droite
        # Le mot reconstitue est bien la concatenation des deux fragments.
        attendu = {rep["left_fragment"].rstrip("-") + rep["right_fragment"],
                   rep["left_fragment"] + rep["right_fragment"]}
        assert rep["repaired_token"] in attendu


def test_les_pages_ne_sont_pas_fusionnees_dans_le_corpus_brut():
    """corpus_pages_raw.jsonl garde bien une ligne par page, sans fusion."""
    records = _load_jsonl(PROCESSED / "corpus_pages_raw.jsonl")
    assert len(records) == EXPECTED_PAGES
    data = _load_json(PROCESSED / "page_boundary_repairs.json")
    for rep in data["repairs"]:
        page = next(r for r in records if r["page_pdf"] == rep["left_page_pdf"])
        # Le mot reconstitue ne doit PAS avoir ete injecte dans la page.
        assert page["clean_text"].rstrip().endswith(rep["left_fragment"])


def test_consolidation_des_legendes_unitaire():
    """Un label de panneau precedant la legende est retire ; les fragments recolles."""
    texte = ("ssp585\n\nFig. S11. Standardized standard deviation of the 500-hPa "
             "geopotential height\n\n(Z500) for the coldest SWG simulations of each "
             "CMIP6 model considered here.")
    out = consolidate_captions(texte)
    assert out.startswith("Fig. S11.")
    assert "ssp585" not in out
    assert "(Z500)" in out
    assert "\n\n" not in out  # une seule legende reconstituee


def test_consolidation_rejette_un_fragment_trop_court():
    assert consolidate_captions("Fig. S1. Trop court.") == ""


def test_manifeste_du_document():
    manifeste = _load_json(PROCESSED / "document_manifest.json")
    assert manifeste["nombre_de_pages"] == EXPECTED_PAGES
    assert manifeste["auteure"] == "Camille Cadiou"
    assert manifeste["universite"]
    assert manifeste["langue_principale"] == "en"
    assert len(manifeste["sha256"]) == 64
