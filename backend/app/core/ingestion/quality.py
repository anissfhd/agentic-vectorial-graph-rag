"""Controle qualite de l'ingestion : couverture, colonnes, nettoyage, statistiques.

Toutes les valeurs produites ici sont MESUREES sur les enregistrements reels.
Aucun chiffre n'est ecrit en dur, aucun seuil n'est ajuste pour faire passer un
controle : un controle qui echoue est rapporte tel quel.
"""

from __future__ import annotations

import re
from collections import Counter
from statistics import median
from typing import Dict, List, Optional, Tuple

# Termes de controle de couverture thematique (issus du sujet de la these).
SCIENTIFIC_TERMS = [
    "SWG", "Stochastic Weather Generator", "NAO", "North Atlantic Oscillation",
    "CMIP6", "Scandinavian Blocking", "polar vortex", "jet stream",
    "winter 1963", "cold spells", "climate change",
]

RE_LONE_NUMBER_PARAGRAPH = re.compile(r"^\s*\d{1,4}\s*$")
RE_BROKEN_HYPHEN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]-\n")


# --- A. Couverture ----------------------------------------------------------
def check_coverage(records: List[dict], expected_pages: int) -> Dict:
    """Verifie qu'aucune page n'est oubliee, dupliquee ou sans statut."""
    pages = [r["page_pdf"] for r in records]
    counts = Counter(pages)
    duplicates = sorted(p for p, n in counts.items() if n > 1)
    missing = sorted(set(range(1, expected_pages + 1)) - set(pages))
    sans_statut = [r["page_pdf"] for r in records if not isinstance(r.get("keep_for_rag"), bool)]
    incoherent = [
        r["page_pdf"] for r in records
        if r.get("keep_for_rag") is False and not r.get("exclusion_reason")
    ]
    return {
        "pages_attendues": expected_pages,
        "pages_presentes": len(records),
        "pages_manquantes": missing,
        "pages_dupliquees": duplicates,
        "pages_sans_statut": sans_statut,
        "pages_exclues_sans_raison": incoherent,
        "ok": (
            len(records) == expected_pages
            and not missing and not duplicates
            and not sans_statut and not incoherent
        ),
    }


# --- B. Double colonne ------------------------------------------------------
def column_alternations(blocks) -> int:
    """Nombre de basculements gauche <-> droite dans une suite de blocs.

    C'est la mesure directe du defaut a eviter : en lecture correcte, on lit toute
    la colonne gauche puis toute la droite, donc un seul basculement par bande.
    Une extraction naive triee par ordonnee alterne a chaque ligne.
    """
    seq = [b.column for b in blocks if b.band == "body" and b.column in ("left", "right")]
    return sum(1 for a, b in zip(seq, seq[1:]) if a != b)


def naive_reading_order(blocks):
    """Ordre naif : tri par ordonnee puis abscisse. Sert de temoin de comparaison."""
    body = [b for b in blocks if b.band == "body"]
    return sorted(body, key=lambda b: (round(b.y0, 1), b.x0))


def check_double_column(raw_page) -> Dict:
    """Compare l'ordre de lecture retenu a l'ordre naif, sur une page reelle."""
    ours = column_alternations(raw_page.blocks)
    naive = column_alternations(naive_reading_order(raw_page.blocks))
    body = [b for b in raw_page.blocks if b.band == "body"]
    n_left = sum(1 for b in body if b.column == "left")
    n_right = sum(1 for b in body if b.column == "right")
    # Les bandes n'ont de sens que sur une page a colonnes : sur une page pleine
    # largeur, chaque bloc est "full" et le compte serait trompeur.
    has_columns = n_left > 0 and n_right > 0
    bands = (sum(1 for b in body if b.column == "full") + 1) if has_columns else 1
    widest = max((b.width for b in body), default=0.0)
    return {
        "page_pdf": raw_page.page_pdf,
        "layout": raw_page.layout,
        "blocs_colonne_gauche": n_left,
        "blocs_colonne_droite": n_right,
        "bandes": bands,
        "basculements_ordre_retenu": ours,
        "basculements_ordre_naif": naive,
        # Preuve geometrique du nombre de colonnes : rapport entre le bloc le plus
        # large et la largeur de page. ~0.85 => pleine largeur (une colonne),
        # ~0.40 => colonne etroite (deux colonnes).
        "largeur_bloc_max_sur_page": round(widest / raw_page.width, 3) if raw_page.width else 0.0,
        "colonnes_non_melangees": ours <= bands,
    }


def pick_layout_samples(raw_pages, ranges: List[Tuple[int, int, str]]) -> List[Tuple]:
    """Choisit une page temoin par plage, en preferant une page a deux colonnes.

    Si une plage n'en contient AUCUNE, on ne force rien : on retient la page la plus
    riche en texte et on signale que la plage est mono-colonne. C'est un resultat de
    mesure, pas un echec — le chapitre 4 de cette these est un preprint compose en
    une seule colonne, contrairement aux articles publies du chapitre 3.
    """
    def is_reference_page(page) -> bool:
        """Une page de bibliographie se reconnait a sa densite de DOI."""
        return page.raw_text.count("doi.org") >= 4

    def prose_pages(pages):
        """Ecarte les pages de references : on veut de la prose scientifique."""
        return [p for p in pages if not is_reference_page(p)] or list(pages)

    chosen: List[Tuple] = []
    for start, end, label in ranges:
        in_range = [p for p in raw_pages if start <= p.page_pdf <= end]
        if not in_range:
            continue
        two_col = [p for p in in_range if p.layout in ("double_column", "mixed_layout")]
        if two_col:
            # On retient la page ou le risque est le PLUS FORT : celle qu'un ordre
            # naif trie par ordonnee melangerait le plus. C'est le temoin le plus
            # demonstratif, et le plus honnete.
            best = max(prose_pages(two_col),
                       key=lambda p: (column_alternations(naive_reading_order(p.blocks)),
                                      len(p.raw_text)))
            chosen.append((label, best, True))
        else:
            textual = [p for p in in_range if p.layout != "figure_or_sparse"] or in_range
            best = max(prose_pages(textual), key=lambda p: len(p.raw_text))
            chosen.append((label, best, False))
    return chosen


# --- C. Nettoyage -----------------------------------------------------------
def check_cleaning(kept_records: List[dict], running_heads: set) -> Dict:
    """Verifie qu'il ne reste ni en-tetes repetes, ni numeros isoles, ni cesures."""
    from .cleaner import _normalised_key

    head_hits: Counter = Counter()
    lone_page_numbers: List[int] = []
    lone_labels: List[int] = []
    broken_hyphens = 0
    trailing_hyphens: List[int] = []
    empty_pages: List[int] = []

    for rec in kept_records:
        text = rec.get("clean_text", "")
        if not text.strip():
            empty_pages.append(rec["page_pdf"])
        broken_hyphens += len(RE_BROKEN_HYPHEN.findall(text))
        # Un mot coupe par une fin de PAGE ne peut pas etre recolle a l'interieur
        # de la page : il est compte a part, pas comme un defaut de nettoyage.
        if text.rstrip().endswith("-"):
            trailing_hyphens.append(rec["page_pdf"])
        # Un nombre isole n'est un NUMERO DE PAGE que s'il vaut le numero de cette
        # page-la. Les autres sont des labels d'axes de figures : ils sont comptes
        # a part et ne constituent pas un defaut de nettoyage.
        own = {str(rec["page_pdf"])}
        if rec.get("page_these") is not None:
            own.add(str(rec["page_these"]))
        for para in text.split("\n\n"):
            if RE_LONE_NUMBER_PARAGRAPH.match(para):
                if para.strip() in own:
                    lone_page_numbers.append(rec["page_pdf"])
                else:
                    lone_labels.append(rec["page_pdf"])
            key = _normalised_key(para)
            if key in running_heads:
                head_hits[key] += 1

    # Doublons massifs : deux pages conservees au texte strictement identique.
    seen: Dict[str, int] = {}
    duplicates: List[Tuple[int, int]] = []
    for rec in kept_records:
        text = rec.get("clean_text", "").strip()
        if len(text) < 200:
            continue
        if text in seen:
            duplicates.append((seen[text], rec["page_pdf"]))
        else:
            seen[text] = rec["page_pdf"]

    return {
        "en_tetes_repetes_residuels": dict(head_hits),
        "numeros_de_page_isoles_residuels": len(lone_page_numbers),
        "pages_avec_numero_isole_residuel": sorted(set(lone_page_numbers)),
        "etiquettes_numeriques_de_figures": len(lone_labels),
        "pages_avec_etiquettes_numeriques": sorted(set(lone_labels)),
        "cesures_non_reparees_dans_la_page": broken_hyphens,
        "cesures_en_fin_de_page": len(trailing_hyphens),
        "pages_a_cesure_de_fin_de_page": trailing_hyphens,
        "pages_conservees_vides": empty_pages,
        "pages_dupliquees": duplicates,
        "ok": (
            not head_hits and not lone_page_numbers
            and broken_hyphens == 0 and not empty_pages and not duplicates
        ),
    }


# --- D. Couverture semantique ----------------------------------------------
def check_scientific_terms(kept_records: List[dict],
                           terms: Optional[List[str]] = None) -> Dict:
    """Compte les occurrences reelles des termes attendus dans le corpus nettoye."""
    terms = terms or SCIENTIFIC_TERMS
    corpus = "\n".join(r.get("clean_text", "") for r in kept_records).lower()
    counts = {t: corpus.count(t.lower()) for t in terms}
    return {
        "occurrences": counts,
        "termes_absents": [t for t, n in counts.items() if n == 0],
        "ok": all(n > 0 for n in counts.values()),
    }


# --- E. Plages exclues ------------------------------------------------------
def check_exclusions(kept_records: List[dict], annex_b_records: List[dict],
                     acronyms: List[dict]) -> Dict:
    """Verifie que les zones interdites sont absentes du corpus principal."""
    kept_pages = {r["page_pdf"] for r in kept_records}
    forbidden = {
        "remerciements": set(range(12, 14)),
        "bibliographie": set(range(130, 140)),
        "annexe_b": set(range(150, 174)),
        "table_des_matieres": set(range(14, 16)),
        "listes_figures_tableaux": set(range(140, 142)),
    }
    fuites = {name: sorted(kept_pages & pages) for name, pages in forbidden.items()}
    fuites = {k: v for k, v in fuites.items() if v}
    annex_b_pages = {r["page_pdf"] for r in annex_b_records}
    return {
        "fuites_dans_corpus_principal": fuites,
        "annexe_b_pages_dans_fichier_dedie": len(annex_b_pages),
        "annexe_b_presente_ailleurs": bool(annex_b_records),
        "acronymes_extraits": len(acronyms),
        "ok": (not fuites) and bool(annex_b_records) and len(acronyms) > 0,
    }


# --- F. Statistiques --------------------------------------------------------
def compute_statistics(records: List[dict], kept_records: List[dict],
                       clean_stats_total: Dict[str, int]) -> Dict:
    """Calcule toutes les statistiques demandees, uniquement par mesure."""
    by_language = Counter(r.get("language") for r in records)
    by_language_kept = Counter(r.get("language") for r in kept_records)
    by_mode = Counter(r.get("extraction_mode") for r in records)
    by_detected = Counter(r.get("detected_layout") for r in records)
    by_content = Counter(r.get("content_type") for r in kept_records)
    by_reason = Counter(
        r.get("exclusion_reason") for r in records if not r.get("keep_for_rag")
    )

    raw_chars = sum(r.get("raw_char_count", 0) for r in records)
    clean_chars_all = sum(r.get("clean_char_count", 0) for r in records)
    clean_chars_kept = sum(r.get("clean_char_count", 0) for r in kept_records)
    words_kept = sum(r.get("estimated_word_count", 0) for r in kept_records)
    empty = [r["page_pdf"] for r in records if r.get("clean_char_count", 0) == 0]

    return {
        "pages_total": len(records),
        "pages_incluses": len(kept_records),
        "pages_exclues": len(records) - len(kept_records),
        "pages_par_langue_total": dict(by_language),
        "pages_par_langue_incluses": dict(by_language_kept),
        "pages_par_extraction_mode": dict(by_mode),
        "pages_par_layout_detecte": dict(by_detected),
        "pages_par_content_type_incluses": dict(by_content),
        "pages_par_raison_exclusion": dict(by_reason),
        "caracteres_bruts_total": raw_chars,
        "caracteres_nettoyes_total": clean_chars_all,
        "caracteres_nettoyes_corpus_rag": clean_chars_kept,
        "mots_estimes_corpus_rag": words_kept,
        "pages_vides_apres_nettoyage": len(empty),
        "liste_pages_vides": empty,
        "cesures_reparees": clean_stats_total.get("hyphens_repaired", 0),
        "cesures_conservees_avec_trait_union": clean_stats_total.get("hyphens_kept", 0),
        "lignes_entete_pied_supprimees": clean_stats_total.get("headers_removed", 0),
        "numeros_de_page_supprimes": clean_stats_total.get("page_numbers_removed", 0),
        "numeros_de_ligne_de_marge_supprimes":
            clean_stats_total.get("margin_line_numbers_removed", 0),
    }
