"""Lecture du PDF page par page, bloc par bloc, en ordre de lecture correct.

Ce module resout le risque R1 du document d'architecture : les chapitres 3 et 4 et
l'annexe B contiennent des articles de revue composes en DOUBLE COLONNE. Une
extraction naive (`page.get_text()` ou un tri par ordonnee seule) alterne une ligne
de la colonne gauche et une ligne de la colonne droite, ce qui detruit les phrases.

La strategie retenue est purement geometrique, sans OCR :

1. on recupere les blocs de texte avec leurs coordonnees (`get_text("blocks")`) ;
2. on isole les bandes d'en-tete et de pied de page par leur ordonnee ;
3. on classe chaque bloc du corps en `left`, `right` ou `full` selon sa position par
   rapport a la gouttiere centrale ;
4. les blocs `full` (titres, resumes pleine largeur, larges figures) decoupent la page
   en bandes horizontales ;
5. dans chaque bande on lit TOUTE la colonne gauche de haut en bas, PUIS toute la
   colonne droite. Deux colonnes ne peuvent donc jamais etre fusionnees sur une
   meme phrase.

Aucune valeur n'est codee en dur : les seuils sont des fractions de la largeur et de
la hauteur reelles de chaque page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import median
from typing import List

import fitz  # PyMuPDF

# --- Seuils geometriques, exprimes en fraction de la page --------------------
# 0.11 et non 0.09 : l'en-tete courant des articles inseres du chapitre 3
# (« C. Cadiou and P. Yiou: ... ») se termine a y=77.9 sur une page de 842 pt,
# soit 0.0925. Un seuil a 0.09 le laissait dans le corps et il polluait alors
# 14 pages du corpus. Les garde-fous de hauteur et de longueur ci-dessous
# empechent qu'un vrai paragraphe soit pris pour un en-tete.
HEADER_BAND = 0.11          # un bloc dont le bas est au-dessus est un en-tete
FOOTER_BAND = 0.88          # un bloc dont le haut est en dessous est un pied de page
BAND_MAX_HEIGHT = 0.06      # un en-tete/pied ne peut pas etre plus haut que cela
BAND_MAX_CHARS = 220        # ni plus long que cela (sinon c'est du corps de texte)
GUTTER_TOL = 0.04           # tolerance autour de la gouttiere centrale
MIN_COLUMN_BLOCKS = 2       # nb minimal de blocs par colonne pour parler de 2 colonnes
COLUMN_CHAR_SHARE = 0.55    # part du texte qui doit vivre dans les colonnes
SPARSE_PAGE_CHARS = 30      # en dessous, la page est consideree vide
FIGURE_MIN_BLOCKS = 20      # beaucoup de blocs...
FIGURE_MEDIAN_CHARS = 25    # ...tous tres courts => page de figures (labels d'axes)

LAYOUT_SINGLE = "single_column"
LAYOUT_DOUBLE = "double_column"
LAYOUT_MIXED = "mixed_layout"
LAYOUT_FIGURE = "figure_or_sparse"
LAYOUT_EXCLUDED = "excluded"


@dataclass
class Block:
    """Un bloc de texte PyMuPDF, enrichi de sa position logique dans la page."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    band: str = "body"      # header | body | footer
    column: str = "full"    # left | right | full

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass
class RawPage:
    """Resultat brut de l'extraction d'une page, avant tout nettoyage."""

    page_pdf: int
    width: float
    height: float
    blocks: List[Block]
    layout: str
    raw_text: str
    n_images: int = 0
    warnings: List[str] = field(default_factory=list)
    groups: List[List[Block]] = field(default_factory=list)

    @property
    def body_blocks(self) -> List[Block]:
        return [b for b in self.blocks if b.band == "body"]


def open_document(pdf_path) -> fitz.Document:
    """Ouvre le PDF et verifie qu'il est exploitable sans mot de passe."""
    doc = fitz.open(str(pdf_path))
    if doc.needs_pass:
        raise RuntimeError(f"PDF chiffre, mot de passe requis : {pdf_path}")
    return doc


def _classify_band(block: Block, height: float) -> str:
    """Distingue en-tete / pied de page / corps a partir de l'ordonnee du bloc."""
    short = block.height <= height * BAND_MAX_HEIGHT and len(block.text) <= BAND_MAX_CHARS
    if short and block.y1 <= height * HEADER_BAND:
        return "header"
    if short and block.y0 >= height * FOOTER_BAND:
        return "footer"
    return "body"


def _classify_column(block: Block, width: float) -> str:
    """Place le bloc a gauche, a droite, ou a cheval sur la gouttiere centrale."""
    mid = width / 2.0
    tol = width * GUTTER_TOL
    if block.x1 <= mid + tol:
        return "left"
    if block.x0 >= mid - tol:
        return "right"
    return "full"


def detect_layout(blocks: List[Block], width: float) -> str:
    """Determine la mise en page reelle a partir de la geometrie des blocs.

    Renvoie l'une des valeurs LAYOUT_*. La detection de figure passe AVANT celle
    des colonnes : une page de figures produit beaucoup de minuscules blocs
    (labels d'axes, legendes de couleurs) repartis a gauche et a droite, qui
    ressembleraient a tort a deux colonnes de texte.
    """
    body = [b for b in blocks if b.band == "body" and b.text.strip()]
    total_chars = sum(len(b.text.strip()) for b in body)

    if total_chars < SPARSE_PAGE_CHARS:
        return LAYOUT_FIGURE

    lengths = [len(b.text.strip()) for b in body]
    if len(body) >= FIGURE_MIN_BLOCKS and median(lengths) < FIGURE_MEDIAN_CHARS:
        return LAYOUT_FIGURE

    left = [b for b in body if b.column == "left"]
    right = [b for b in body if b.column == "right"]
    full = [b for b in body if b.column == "full"]

    column_chars = sum(len(b.text.strip()) for b in left + right)
    two_columns = (
        len(left) >= MIN_COLUMN_BLOCKS
        and len(right) >= MIN_COLUMN_BLOCKS
        and column_chars >= total_chars * COLUMN_CHAR_SHARE
    )
    if not two_columns:
        return LAYOUT_SINGLE
    # Un bloc pleine largeur au milieu du corps (titre d'article, resume, grande
    # figure) signale une page mixte : la lecture par bandes reste necessaire.
    if any(len(b.text.strip()) > 0 for b in full):
        return LAYOUT_MIXED
    return LAYOUT_DOUBLE


def reading_order(blocks: List[Block], layout: str) -> List[Block]:
    """Ordonne les blocs dans l'ordre ou un humain les lirait.

    En une colonne : simple tri par (y, x).
    En deux colonnes : les blocs pleine largeur decoupent la page en bandes ;
    dans chaque bande on lit la colonne gauche entiere, puis la droite entiere.
    """
    header = sorted([b for b in blocks if b.band == "header"], key=lambda b: (b.y0, b.x0))
    footer = sorted([b for b in blocks if b.band == "footer"], key=lambda b: (b.y0, b.x0))
    body = [b for b in blocks if b.band == "body"]

    if layout not in (LAYOUT_DOUBLE, LAYOUT_MIXED):
        return header + sorted(body, key=lambda b: (round(b.y0, 1), b.x0)) + footer

    spanning = sorted([b for b in body if b.column == "full"], key=lambda b: b.y0)
    columns = [b for b in body if b.column in ("left", "right")]

    # Frontieres horizontales des bandes : le bas de chaque bloc pleine largeur.
    boundaries = [b.y1 for b in spanning]

    def band_index(block: Block) -> int:
        """Numero de bande : combien de blocs pleine largeur sont au-dessus."""
        centre = (block.y0 + block.y1) / 2.0
        return sum(1 for y in boundaries if y <= centre)

    ordered: List[Block] = []
    n_bands = len(spanning) + 1
    for band in range(n_bands):
        in_band = [b for b in columns if band_index(b) == band]
        left = sorted([b for b in in_band if b.column == "left"], key=lambda b: b.y0)
        right = sorted([b for b in in_band if b.column == "right"], key=lambda b: b.y0)
        ordered.extend(left)
        ordered.extend(right)
        if band < len(spanning):
            ordered.append(spanning[band])

    # Filet de securite : aucun bloc ne doit disparaitre de la sortie.
    seen = {id(b) for b in ordered}
    ordered.extend(b for b in body if id(b) not in seen)
    return header + ordered + footer


def is_multiline(block: Block) -> bool:
    """Vrai si le bloc contient deja plusieurs lignes de TEXTE, donc un paragraphe.

    Les lignes purement numeriques sont ignorees : dans l'article du chapitre 4,
    PyMuPDF agrege le numero de ligne de marge (« 180 ») a la ligne de texte
    voisine. Sans cette exception, ces blocs passeraient pour des paragraphes
    complets et la phrase serait coupee a chaque multiple de cinq.
    """
    lines = [l.strip() for l in block.text.strip("\n").split("\n") if l.strip()]
    real_lines = [l for l in lines if not re.fullmatch(r"\d{1,4}", l)]
    return len(real_lines) > 1


def group_paragraphs(ordered: List[Block], width: float) -> List[List[Block]]:
    """Regroupe les blocs en paragraphes logiques.

    PyMuPDF ne segmente pas uniformement ce PDF : dans les chapitres 1 a 3 un bloc
    est deja un paragraphe entier (plusieurs lignes), tandis que dans le chapitre 4
    et l'annexe C **chaque ligne forme son propre bloc**. Sans regroupement, chaque
    ligne deviendrait un faux paragraphe : les mots coupes en fin de ligne ne
    seraient jamais recolles et le decoupage par paragraphes (methode 3 des sept
    chunkings) serait vide de sens.

    Le critere est celui du texte justifie : une ligne qui atteint la marge droite
    de sa colonne continue le paragraphe ; une ligne qui s'arrete avant le termine.
    """
    if not ordered:
        return []

    # Marges droites reelles de la page : les abscisses de fin de ligne qui
    # REVIENNENT. Sur une page a deux colonnes il y en a deux (~287 et ~548) ;
    # sur une page pleine largeur une seule (~548). Prendre le simple maximum
    # serait fragile : un unique bloc deborde parfois la marge et decalerait le
    # seuil pour toute la page.
    body = [b for b in ordered if b.band == "body"]
    edge_counts: dict = {}
    for b in body:
        key = round(b.x1)
        edge_counts[key] = edge_counts.get(key, 0) + 1
    margins = [x for x, n in edge_counts.items() if n >= 2]
    if not margins:
        margins = [max((b.x1 for b in body), default=width)]

    tol = width * 0.02
    tol_left = width * 0.06  # tolerance sur l'alignement a gauche

    def reaches_margin(block: Block) -> bool:
        return any(abs(block.x1 - m) <= tol for m in margins)

    groups: List[List[Block]] = []
    current: List[Block] = []
    previous: Block = None  # type: ignore[assignment]

    for block in ordered:
        start_new = True
        if previous is not None and not is_multiline(previous) and not is_multiline(block):
            # Meme paragraphe : meme bande, meme marge gauche. En deux colonnes,
            # l'ecart des marges gauches (46 contre 307) empeche tout regroupement
            # accidentel entre colonnes.
            same_zone = (previous.band == block.band
                         and abs(previous.x0 - block.x0) <= tol_left)
            gap = block.y0 - previous.y1
            close = gap < max(previous.height, 1.0) * 1.6
            ends_hyphen = previous.text.strip().endswith("-")
            if same_zone and close and (reaches_margin(previous) or ends_hyphen):
                start_new = False

        if start_new and current:
            groups.append(current)
            current = []
        current.append(block)
        previous = block

    if current:
        groups.append(current)
    return groups


def extract_page(doc: fitz.Document, index: int) -> RawPage:
    """Extrait une page (index 0-based) et renvoie ses blocs en ordre de lecture."""
    page = doc[index]
    width, height = page.rect.width, page.rect.height
    warnings: List[str] = []

    try:
        raw_blocks = page.get_text("blocks")
    except Exception as exc:  # page corrompue : on continue sur les autres
        return RawPage(
            page_pdf=index + 1,
            width=width,
            height=height,
            blocks=[],
            layout=LAYOUT_FIGURE,
            raw_text="",
            n_images=0,
            warnings=[f"extraction_impossible: {exc}"],
        )

    blocks: List[Block] = []
    for b in raw_blocks:
        # b = (x0, y0, x1, y1, text, block_no, block_type) ; type 0 = texte
        if len(b) < 7 or b[6] != 0:
            continue
        text = b[4]
        if not text.strip():
            continue
        blk = Block(x0=b[0], y0=b[1], x1=b[2], y1=b[3], text=text)
        blk.band = _classify_band(blk, height)
        blk.column = _classify_column(blk, width)
        blocks.append(blk)

    layout = detect_layout(blocks, width)
    ordered = reading_order(blocks, layout)
    groups = group_paragraphs(ordered, width)
    raw_text = "\n\n".join(
        "\n".join(b.text.strip("\n") for b in grp) for grp in groups
    )

    try:
        n_images = len(page.get_images(full=True))
    except Exception:
        n_images = 0

    if not blocks and n_images == 0:
        warnings.append("page_sans_texte_ni_image")
    elif not blocks and n_images > 0:
        warnings.append("page_sans_couche_texte_mais_avec_image")

    return RawPage(
        page_pdf=index + 1,
        width=width,
        height=height,
        blocks=ordered,
        layout=layout,
        raw_text=raw_text,
        n_images=n_images,
        warnings=warnings,
        groups=groups,
    )


def extract_document(doc: fitz.Document) -> List[RawPage]:
    """Extrait toutes les pages. Une page en erreur n'interrompt pas les autres."""
    pages: List[RawPage] = []
    for i in range(doc.page_count):
        try:
            pages.append(extract_page(doc, i))
        except Exception as exc:
            pages.append(
                RawPage(
                    page_pdf=i + 1,
                    width=0.0,
                    height=0.0,
                    blocks=[],
                    layout=LAYOUT_FIGURE,
                    raw_text="",
                    warnings=[f"echec_extraction_page: {exc}"],
                )
            )
    return pages
