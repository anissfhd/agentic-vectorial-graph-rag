"""Structure du document : pagination, zones, chapitres, sections, acronymes.

Trois responsabilites :

  * **Pagination** — le manuscrit imprime sa propre numerotation, decalee de 15 pages
    par rapport au PDF. Certains articles inseres (chapitres 3 et 4, annexes B et C)
    portent EN PLUS leur propre numerotation de revue, qui n'a rien a voir avec celle
    de la these. On distingue donc les deux cas.
  * **Zones** — politique d'inclusion / exclusion du corpus RAG, page par page.
  * **Titres** — chapitres et sections, issus de la table des matieres du PDF et des
    titres reellement detectes dans le texte. Aucun titre n'est invente.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# --- Pagination -------------------------------------------------------------
PAGE_OFFSET = 15            # page_these = page_pdf - 15
FIRST_NUMBERED_PDF_PAGE = 16  # avant cela, la these est en chiffres romains

# Ecart tolere entre le numero imprime et la regle avant de considerer que le
# numero appartient a une numerotation etrangere (article de revue insere).
PAGINATION_TOLERANCE = 2

# --- Zones du document (bornes en pages PDF, incluses) ----------------------
ZONE_FRONT_MATTER = "front_matter"
ZONE_SUMMARIES = "summaries"
ZONE_ACKNOWLEDGEMENTS = "acknowledgements"
ZONE_TOC = "toc"
ZONE_CHAPTERS = "chapters"
ZONE_ACRONYMS = "acronyms"
ZONE_BIBLIOGRAPHY = "bibliography"
ZONE_LISTS = "lists_figures_tables"
ZONE_APPENDIX_DIVIDER = "appendices_divider"
ZONE_APPENDIX_A = "appendix_a"
ZONE_APPENDIX_B = "appendix_b"
ZONE_APPENDIX_C = "appendix_c"

ZONES: List[Tuple[int, int, str]] = [
    (1, 5, ZONE_FRONT_MATTER),
    (6, 11, ZONE_SUMMARIES),
    (12, 13, ZONE_ACKNOWLEDGEMENTS),
    (14, 15, ZONE_TOC),
    (16, 127, ZONE_CHAPTERS),
    (128, 129, ZONE_ACRONYMS),
    (130, 139, ZONE_BIBLIOGRAPHY),
    (140, 141, ZONE_LISTS),
    (142, 143, ZONE_APPENDIX_DIVIDER),
    (144, 149, ZONE_APPENDIX_A),
    (150, 173, ZONE_APPENDIX_B),
    (174, 206, ZONE_APPENDIX_C),
]

# Zones dont le contenu alimente le corpus RAG principal.
KEEP_ZONES = {ZONE_SUMMARIES, ZONE_CHAPTERS, ZONE_APPENDIX_A, ZONE_APPENDIX_C}

EXCLUSION_REASONS: Dict[str, str] = {
    ZONE_FRONT_MATTER: "couverture_et_metadonnees",
    ZONE_ACKNOWLEDGEMENTS: "remerciements",
    ZONE_TOC: "table_des_matieres",
    ZONE_ACRONYMS: "page_acronymes_traitee_separement",
    ZONE_BIBLIOGRAPHY: "bibliographie",
    ZONE_LISTS: "listes_figures_et_tableaux",
    ZONE_APPENDIX_DIVIDER: "page_structurelle_sans_contenu_scientifique",
    ZONE_APPENDIX_B: "annexe_b_hors_sujet",
}

# Seuils de contenu utile (mesures, pas devines : voir ingestion_report.json)
MIN_USEFUL_CHARS = 120        # en dessous, la page conservee serait vide de sens
MIN_APPENDIX_C_CHARS = 150    # l'annexe C est majoritairement figurative

# --- Chapitres : numero attribue a partir de la page d'ouverture reelle -----
CHAPTER_START_PAGES: Dict[int, str] = {
    16: "1", 34: "2", 50: "3", 86: "4", 122: "5",
    144: "A", 150: "B", 174: "C",
}

RE_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+){0,3})\s+(\S.{2,120})$")
# Une ligne de sommaire porte des points de conduite (« Conclusions . . . . 107 ») :
# ce n'est pas un titre de section, c'est le sommaire du chapitre.
RE_DOT_LEADER = re.compile(r"\.\s*\.\s*\.|…")
RE_LONE_NUMBER = re.compile(r"^(\d+(?:\.\d+){0,3})$")
RE_FIGURE_CAPTION = re.compile(r"^(Figure|Fig\.)\s*[A-Z]?\.?\d", re.IGNORECASE)
RE_TABLE_CAPTION = re.compile(r"^(Table|Tableau)\s*[A-Z]?\.?\d", re.IGNORECASE)


def zone_of(page_pdf: int) -> str:
    """Renvoie la zone documentaire d'une page PDF."""
    for start, end, name in ZONES:
        if start <= page_pdf <= end:
            return name
    return ZONE_FRONT_MATTER


def rule_thesis_page(page_pdf: int) -> Optional[int]:
    """Numero de page these attendu par la regle, ou None pour les pages liminaires."""
    if page_pdf < FIRST_NUMBERED_PDF_PAGE:
        return None
    return page_pdf - PAGE_OFFSET


def detect_printed_page_number(raw_page) -> Optional[int]:
    """Cherche un numero de page arabe imprime dans les bandes d'en-tete/pied."""
    candidates: List[int] = []
    for block in raw_page.blocks:
        if block.band not in ("header", "footer"):
            continue
        for line in block.text.split("\n"):
            s = line.strip()
            if s.isdigit() and 1 <= int(s) <= 999:
                candidates.append(int(s))
    if not candidates:
        return None
    return candidates[-1]


def resolve_pagination(page_pdf: int, raw_page) -> Tuple[Optional[int], Optional[int], List[str]]:
    """Concilie la regle de pagination et le numero reellement imprime.

    Renvoie `(page_these, page_printed_detected, warnings)`.

    Trois cas :
      * aucun numero imprime -> la regle s'applique telle quelle ;
      * numero imprime proche de la regle -> le numero **detecte** fait foi
        (la regle peut deriver localement) et un avertissement est pose ;
      * numero imprime tres eloigne -> il s'agit de la numerotation propre d'un
        article de revue insere (chapitres 3-4, annexes B-C). La regle de la these
        est conservee pour que les citations restent verifiables, le numero
        etranger est archive dans `page_printed_detected`, et un avertissement
        explicite est pose.
    """
    warnings: List[str] = []
    expected = rule_thesis_page(page_pdf)
    printed = detect_printed_page_number(raw_page)

    if printed is None:
        return expected, None, warnings

    if expected is None:
        # Page liminaire : la these y utilise des chiffres romains. On ne fabrique
        # aucun numero arabe.
        warnings.append(
            f"numero_imprime_{printed}_sur_page_liminaire_ignore_pagination_romaine"
        )
        return None, printed, warnings

    if printed == expected:
        return expected, printed, warnings

    if abs(printed - expected) <= PAGINATION_TOLERANCE:
        warnings.append(
            f"pagination_detectee_{printed}_differe_de_la_regle_{expected}_valeur_detectee_conservee"
        )
        return printed, printed, warnings

    warnings.append(
        f"numerotation_etrangere_{printed}_article_insere_regle_these_{expected}_conservee"
    )
    return expected, printed, warnings


# --- Table des matieres -----------------------------------------------------
@dataclass
class TocEntry:
    level: int
    title: str
    page_pdf: int
    chapter_number: Optional[str] = None


@dataclass
class DocumentStructure:
    """Index des chapitres et sections, construit a partir de la TOC du PDF."""

    entries: List[TocEntry] = field(default_factory=list)

    def chapter_for(self, page_pdf: int) -> Optional[TocEntry]:
        best = None
        for e in self.entries:
            if e.level == 1 and e.page_pdf <= page_pdf:
                best = e
        return best

    def section_for(self, page_pdf: int) -> Optional[TocEntry]:
        best = None
        for e in self.entries:
            if e.level >= 2 and e.page_pdf <= page_pdf:
                best = e
        return best

    def starts_on(self, page_pdf: int) -> bool:
        """Vrai si un titre de la TOC commence exactement sur cette page."""
        return any(e.page_pdf == page_pdf for e in self.entries)


def build_structure(doc) -> DocumentStructure:
    """Construit l'index a partir de la table des matieres embarquee du PDF."""
    from .cleaner import normalise_unicode

    entries: List[TocEntry] = []
    for level, title, page in doc.get_toc() or []:
        if page is None or page < 1:
            continue
        entry = TocEntry(level=int(level),
                         title=normalise_unicode(str(title)).strip(),
                         page_pdf=int(page))
        if entry.level == 1:
            entry.chapter_number = CHAPTER_START_PAGES.get(entry.page_pdf)
        entries.append(entry)
    entries.sort(key=lambda e: (e.page_pdf, e.level))
    return DocumentStructure(entries=entries)


def detect_heading_in_page(raw_page) -> Optional[Tuple[str, str]]:
    """Cherche un titre de section numerote reellement present dans la page.

    Gere les deux mises en forme rencontrees dans ce PDF :
      * un bloc « 1.1.2\\nDefinition of extreme cold events » ;
      * un bloc « 3.1 Statistical Attribution » sur une seule ligne.
    Renvoie (numero, titre) ou None. Aucun titre n'est fabrique.
    """
    from .cleaner import normalise_unicode

    for block in raw_page.blocks:
        if block.band != "body":
            continue
        # Normalise avant analyse : sans cela le titre stocke en metadonnee
        # conserverait les ligatures du PDF (« Deﬁnition of extreme cold events »).
        lines = [l.strip() for l in normalise_unicode(block.text).split("\n") if l.strip()]
        if not lines:
            continue
        first = lines[0]
        if RE_DOT_LEADER.search(first):
            continue
        m = RE_NUMBERED_HEADING.match(first)
        if m and len(first) < 130:
            return m.group(1), m.group(2).strip()
        if (RE_LONE_NUMBER.match(first) and len(lines) > 1
                and len(lines[1]) < 130 and not RE_DOT_LEADER.search(lines[1])):
            return first, lines[1].strip()
    return None


def classify_content_type(zone: str, section_title: Optional[str], clean_text: str,
                          layout: str) -> str:
    """Deduit le type de contenu de la page, sans jamais inventer d'etiquette."""
    head = (clean_text or "").lstrip()[:80]
    if RE_FIGURE_CAPTION.match(head):
        return "figure_caption"
    if RE_TABLE_CAPTION.match(head):
        return "table_caption"
    if layout == "figure_or_sparse":
        return "figure_caption"

    if zone == ZONE_SUMMARIES:
        return "abstract"
    if head.lower().startswith("abstract"):
        return "abstract"

    title = (section_title or "").lower()
    if "introduction" in title:
        return "introduction"
    if any(k in title for k in ("conclusion", "perspective", "resume", "résumé", "summary")):
        return "conclusion"
    if any(k in title for k in ("result", "résultat", "resultat")):
        return "result"
    if any(k in title for k in ("discussion", "caveat")):
        return "discussion"
    if any(k in title for k in ("method", "méthode", "methode", "data", "generator",
                                "analogues", "algorithm", "approach")):
        return "methodology"
    if any(k in title for k in ("definition", "définition")):
        return "definition"

    if zone in (ZONE_APPENDIX_A, ZONE_APPENDIX_B, ZONE_APPENDIX_C):
        return "appendix"
    return "other"


def decide_inclusion(page_pdf: int, zone: str, clean_text: str,
                     layout: str) -> Tuple[bool, Optional[str]]:
    """Applique la politique d'inclusion / exclusion. Renvoie (keep, raison)."""
    n = len(clean_text.strip())

    if zone not in KEEP_ZONES:
        return False, EXCLUSION_REASONS.get(zone, "hors_perimetre")

    if n == 0:
        return False, "page_blanche"
    if zone == ZONE_APPENDIX_C:
        # L'annexe C est majoritairement figurative : on ne garde que les pages
        # portant du texte scientifique ou des legendes reellement exploitables.
        if n < MIN_APPENDIX_C_CHARS:
            return False, "annexe_c_sans_texte_exploitable"
        return True, None
    if n < MIN_USEFUL_CHARS:
        return False, "page_quasi_vide_ou_titre_seul"
    return True, None


# --- Listes de references internes aux articles inseres --------------------
# Un titre de section de bibliographie occupe un paragraphe a lui seul. On ne
# cherche donc PAS ces mots n'importe ou dans le texte : une citation normale
# au fil d'une phrase scientifique doit etre conservee.
REFERENCE_HEADINGS = {
    "references", "bibliography", "literature cited", "bibliographie",
    "references cited", "reference list",
}


def find_article_reference_ranges(records: List[dict],
                                  structure: "DocumentStructure") -> List[dict]:
    """Localise les listes bibliographiques FINALES des articles inseres.

    Le debut est un paragraphe reduit au seul mot « References » (ou variante).
    La fin est donnee par la structure du document : la prochaine entree de la
    table des matieres apres ce debut. C'est une borne factuelle, pas une
    heuristique de densite de DOI.

    Renvoie une liste de plages `{start_page_pdf, start_paragraph, end_page_pdf,
    chapter_number, article_title}`.
    """
    ranges: List[dict] = []
    for rec in records:
        if not rec.get("keep_for_rag"):
            continue
        paragraphs = rec["clean_text"].split("\n\n")
        for index, para in enumerate(paragraphs):
            if para.strip().lower().rstrip(".:") not in REFERENCE_HEADINGS:
                continue
            start = rec["page_pdf"]
            following = [e.page_pdf for e in structure.entries if e.page_pdf > start]
            end = (min(following) - 1) if following else start
            article = structure.section_for(start)
            ranges.append({
                "start_page_pdf": start,
                "start_paragraph": index,
                "end_page_pdf": end,
                "chapter_number": rec.get("chapter_number"),
                "chapter_title": rec.get("chapter_title"),
                "article_title": article.title if article else None,
                "heading": para.strip(),
            })
            break  # un seul debut de bibliographie par page
    return ranges


# --- Acronymes (pages 128-129) ---------------------------------------------
# L'acronyme est en capitales, eventuellement suivi d'un chiffre (CMIP6, AR6) et
# d'un signe (« NAO- » pour la phase negative de la NAO). Autoriser les minuscules
# rendait la capture gloutonne : « NAO-Negative Phase of the NAO » etait lu comme
# l'acronyme « NAO-Negative ».
#
# Le separateur est une espace OU rien du tout : dans ce PDF l'entree de la phase
# negative est composee « NAO-Negative Phase of the NAO. » sans espace apres le
# signe. La lookbehind autorise ce seul cas.
#
# Deux caracteres minimum : une ligne de continuation commencant par un mot
# capitalise (« Forecasts. 132 ») ne doit pas etre prise pour un nouvel acronyme.
RE_ACRONYM_ENTRY = re.compile(
    r"^(?P<acr>[A-Z][A-Z0-9]{1,9}[-+]?)(?:\s+|(?<=[-+]))(?P<rest>\S.*)$"
)
RE_REFS_TAIL = re.compile(r"^(?P<expanded>.*?)\.\s*(?P<refs>[0-9ivxlcdm,\s-]*)$")
RE_ONLY_REFS = re.compile(r"^[0-9ivxlcdm,\s-]+$")


def _parse_refs(raw: str) -> Tuple[List[int], List[str], List[str]]:
    """Transforme « 9, 10, 108-110, ii » en pages arabes, romaines et anomalies."""
    arabic: List[int] = []
    roman: List[str] = []
    problems: List[str] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", token)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a <= b and b - a < 200:
                arabic.extend(range(a, b + 1))
            else:
                problems.append(f"plage_invalide:{token}")
            continue
        if token.isdigit():
            arabic.append(int(token))
        elif re.fullmatch(r"[ivxlcdm]+", token, re.IGNORECASE):
            roman.append(token)
        else:
            problems.append(f"reference_non_analysee:{token}")
    return arabic, roman, problems


def extract_acronyms(raw_pages: List) -> List[dict]:
    """Extrait le glossaire officiel des pages 128-129 en enregistrements structures.

    Cette page est le gazetteer valide par l'auteure : elle fonde l'ontologie du
    futur graphe de connaissances. Elle n'est donc PAS traitee comme de la prose.
    """
    from .cleaner import normalise_unicode

    acronyms: List[dict] = []
    for raw_page in raw_pages:
        if zone_of(raw_page.page_pdf) != ZONE_ACRONYMS:
            continue
        # On travaille LIGNE PAR LIGNE et non bloc par bloc : PyMuPDF regroupe
        # parfois deux entrees voisines dans un meme bloc (« QBO ... 129 » et
        # « SB Scandinavian Blocking. 7 »), ce qui en aurait fait perdre une.
        for block in raw_page.blocks:
            if block.band != "body":
                continue
            for raw_line in normalise_unicode(block.text).split("\n"):
                flat = raw_line.strip()
                if not flat or flat.lower() == "acronyms":
                    continue

                # Ligne de continuation : uniquement des numeros de page.
                if RE_ONLY_REFS.fullmatch(flat) and acronyms:
                    a, r, p = _parse_refs(flat)
                    acronyms[-1]["referenced_pages"].extend(a)
                    acronyms[-1]["referenced_pages_roman"].extend(r)
                    acronyms[-1]["warnings"].extend(p)
                    continue

                m = RE_ACRONYM_ENTRY.match(flat)
                if not m:
                    # Suite d'une forme developpee coupee en fin de ligne.
                    if acronyms:
                        acronyms[-1]["expanded_form"] = (
                            f"{acronyms[-1]['expanded_form']} {flat}".strip().rstrip(".")
                        )
                    continue
                acr, rest = m.group("acr"), m.group("rest")

                warnings: List[str] = []
                tail = RE_REFS_TAIL.match(rest)
                if tail:
                    expanded = tail.group("expanded").strip()
                    arabic, roman, problems = _parse_refs(tail.group("refs"))
                    warnings.extend(problems)
                else:
                    expanded = rest.strip().rstrip(".")
                    arabic, roman = [], []
                    warnings.append("aucune_page_de_reference_detectee")
                if not expanded:
                    warnings.append("forme_developpee_vide")
                acronyms.append({
                    "acronym": acr,
                    "expanded_form": expanded,
                    "referenced_pages": arabic,
                    "referenced_pages_roman": roman,
                    "source_page_pdf": raw_page.page_pdf,
                    "warnings": warnings,
                })

    # Deduplication defensive : un acronyme ne doit apparaitre qu'une fois.
    seen: Dict[str, dict] = {}
    for entry in acronyms:
        key = entry["acronym"]
        if key in seen:
            seen[key]["referenced_pages"].extend(entry["referenced_pages"])
            seen[key]["warnings"].append("entree_dupliquee_fusionnee")
        else:
            seen[key] = entry
    for entry in seen.values():
        entry["referenced_pages"] = sorted(set(entry["referenced_pages"]))
        entry["referenced_pages_roman"] = sorted(set(entry["referenced_pages_roman"]))
    return list(seen.values())
