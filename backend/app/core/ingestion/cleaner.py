"""Nettoyage du texte extrait, sans jamais alterer le sens scientifique.

Regles absolues :
  * aucune traduction, aucun resume, aucune reformulation ;
  * aucune suppression d'un passage au motif qu'il est en anglais ;
  * unites, valeurs numeriques, symboles, acronymes et renvois aux figures,
    tableaux et equations sont conserves tels quels.

Deux traitements ont besoin d'une vision GLOBALE du document et sont donc calcules
une fois pour toutes dans `CleaningContext` :

  1. les en-tetes et pieds de page courants, reperes par leur REPETITION reelle a
     travers les pages (approche statistique, aucune liste ecrite a la main) ;
  2. la reparation des cesures de fin de ligne : pour `con-\\nsidered`, on compare la
     frequence reelle de « considered » a celle de « con-sidered » dans le reste du
     document, et on tranche par les donnees plutot que par une regle arbitraire.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

# --- Tables de normalisation ------------------------------------------------
# On evite volontairement unicodedata.normalize("NFKC") : NFKC transformerait
# « km² » en « km2 » et casserait les unites scientifiques. On normalise en NFC
# puis on traite explicitement ligatures, tirets, guillemets et espaces.
LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
    "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
    "œ": "oe", "Œ": "OE", "æ": "ae", "Æ": "AE",
}

# Tirets, traits d'union et signe moins ramenes au trait d'union ASCII :
# « -10 °C » reste « -10 °C », la valeur numerique est preservee.
DASHES = {
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-", "⁃": "-",
}

QUOTES = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "«": '"', "»": '"', "′": "'", "″": '"',
}

SPACES = {
    " ": " ", " ": " ", " ": " ", " ": " ",
    " ": " ", " ": " ", " ": " ", " ": " ",
    " ": " ", " ": " ", " ": " ", "　": " ",
}

INVISIBLES = ["­", "​", "‌", "‍", "﻿"]

_TRANSLATION = {ord(k): v for k, v in {**LIGATURES, **DASHES, **QUOTES, **SPACES}.items()}

# --- Expressions reperant ce qui doit disparaitre ---------------------------
RE_ARABIC_PAGE = re.compile(r"^\d{1,4}$")
RE_ROMAN_PAGE = re.compile(r"^[ivxlcdm]{1,8}$", re.IGNORECASE)
RE_PAGE_WORD = re.compile(r"^(page|p\.)\s*\d{1,4}$", re.IGNORECASE)

# Un tiret de cesure : fragment + '-' + saut de ligne + suite en minuscules.
RE_HYPHEN_BREAK = re.compile(r"([A-Za-zÀ-ÖØ-öø-ÿ]{2,})-\n([A-Za-zÀ-ÖØ-öø-ÿ]+)")
RE_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]{2,}")
RE_INLINE_HYPHEN = re.compile(r"([A-Za-zÀ-ÖØ-öø-ÿ]{2,})-([A-Za-zÀ-ÖØ-öø-ÿ]{2,})")

MIN_REPEAT_FOR_RUNNING_HEAD = 4  # une ligne vue sur >= 4 pages est un en-tete courant

# Zone ou un nombre isole est un numero de page et non une donnee. Les articles
# inseres impriment leur propre numero legerement plus haut que la these : sur les
# pages du chapitre 4 il se trouve a 0.849 de la hauteur. Les labels d'axes de
# figures, eux, vivent entre 0.30 et 0.50 : ils ne sont pas concernes.
PAGE_NUMBER_TOP_BAND = 0.12
PAGE_NUMBER_BOTTOM_BAND = 0.82


def normalise_unicode(text: str) -> str:
    """NFC + ligatures + tirets + guillemets + espaces, sans casser les unites."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    for ch in INVISIBLES:
        text = text.replace(ch, "")
    return text.translate(_TRANSLATION)


def is_page_number_line(line: str) -> bool:
    """Vrai si la ligne n'est qu'un numero de page (arabe, romain ou « page N »)."""
    s = line.strip()
    if not s:
        return False
    return bool(RE_ARABIC_PAGE.match(s) or RE_ROMAN_PAGE.match(s) or RE_PAGE_WORD.match(s))


@dataclass
class CleaningContext:
    """Connaissance globale du document, calculee avant le nettoyage page par page."""

    running_heads: set = field(default_factory=set)
    word_freq: Counter = field(default_factory=Counter)
    hyphen_freq: Counter = field(default_factory=Counter)


@dataclass
class CleanStats:
    """Compteurs reels de ce que le nettoyage a fait sur une page."""

    hyphens_repaired: int = 0
    hyphens_kept: int = 0
    headers_removed: int = 0
    page_numbers_removed: int = 0
    margin_line_numbers_removed: int = 0


# Un bloc qui commence nettement a gauche de la marge dominante de la page a
# absorbe un element de marge. Dans l'article du chapitre 4 (preprint Earth
# System Dynamics), ce sont les NUMEROS DE LIGNE imprimes tous les cinq vers
# x/largeur = 0.037, alors que la colonne de texte commence a 0.078.
MARGIN_TOLERANCE = 12.0


def _normalised_key(line: str) -> str:
    """Cle de comparaison d'une ligne d'en-tete : casse et chiffres neutralises."""
    s = normalise_unicode(line).strip().lower()
    s = re.sub(r"\d+", "#", s)
    return re.sub(r"\s+", " ", s)


def build_context(raw_pages: Iterable) -> CleaningContext:
    """Analyse tout le document pour reperer en-tetes courants et vocabulaire.

    `raw_pages` est une sequence de `pdf_reader.RawPage`.
    """
    head_counter: Counter = Counter()
    word_freq: Counter = Counter()
    hyphen_freq: Counter = Counter()

    pages = list(raw_pages)
    for page in pages:
        seen_on_page = set()
        for block in page.blocks:
            text = normalise_unicode(block.text)
            for raw_line in text.split("\n"):
                line = raw_line.strip()
                if not line:
                    continue
                # 1. candidats en-tete / pied : uniquement dans les bandes hautes/basses
                if block.band in ("header", "footer") and not is_page_number_line(line):
                    key = _normalised_key(line)
                    if key and key not in seen_on_page:
                        head_counter[key] += 1
                        seen_on_page.add(key)
                # 2. vocabulaire : on ignore le dernier mot d'une ligne cesuree
                body_line = line[:-1] if line.endswith("-") else line
                for w in RE_WORD.findall(body_line):
                    word_freq[w.lower()] += 1
                for a, b in RE_INLINE_HYPHEN.findall(line):
                    hyphen_freq[f"{a.lower()}-{b.lower()}"] += 1

    running = {k for k, n in head_counter.items() if n >= MIN_REPEAT_FOR_RUNNING_HEAD}
    return CleaningContext(running_heads=running, word_freq=word_freq, hyphen_freq=hyphen_freq)


def repair_hyphenation(text: str, ctx: CleaningContext, stats: CleanStats) -> str:
    """Recolle les mots coupes en fin de ligne, en tranchant par les frequences.

    `con-\\nsidered` devient `considered` si « considered » est atteste dans le
    document ; `well-\\nknown` reste `well-known` si la forme avec trait d'union est
    plus frequente. C'est une decision fondee sur le corpus, pas sur une heuristique.
    """

    def decide(match: re.Match) -> str:
        head, tail = match.group(1), match.group(2)
        joined = f"{head}{tail}".lower()
        hyphenated = f"{head}-{tail}".lower()
        if ctx.hyphen_freq.get(hyphenated, 0) > ctx.word_freq.get(joined, 0):
            stats.hyphens_kept += 1
            return f"{head}-{tail}"
        stats.hyphens_repaired += 1
        return f"{head}{tail}"

    return RE_HYPHEN_BREAK.sub(decide, text)


def _clean_block_text(text: str, ctx: CleaningContext, stats: CleanStats, band: str,
                      strip_margin_numbers: bool = False) -> str:
    """Nettoie un bloc : lignes parasites retirees, cesures reparees, lignes recollees."""
    text = normalise_unicode(text)

    kept_lines: List[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        # Une ligne purement numerique n'est supprimee que dans les bandes haute et
        # basse. Dans le corps, « 1963 » ou « 500 » est une donnee scientifique
        # (cellule de tableau, valeur, annee) : la supprimer serait une perte de
        # donnees. Les numeros de page egares dans le corps sont traites en amont,
        # dans clean_page, par comparaison avec le numero reel de la page.
        if band in ("header", "footer") and is_page_number_line(line):
            stats.page_numbers_removed += 1
            continue
        # Numero de ligne de marge absorbe par le bloc de texte : sans cela on
        # obtiendrait « Then we initialize the 180 » au milieu d'une phrase.
        if strip_margin_numbers and RE_ARABIC_PAGE.match(line):
            stats.margin_line_numbers_removed += 1
            continue
        # Les en-tetes courants ne sont retires que dans les bandes haute et basse :
        # un vrai titre de chapitre, situe dans le corps, est ainsi preserve.
        if band in ("header", "footer") and _normalised_key(line) in ctx.running_heads:
            stats.headers_removed += 1
            continue
        kept_lines.append(line)

    if not kept_lines:
        return ""

    block = "\n".join(kept_lines)
    block = repair_hyphenation(block, ctx, stats)
    # Les retours de ligne restants sont des retours typographiques, pas des
    # fins de paragraphe : ils deviennent des espaces.
    block = block.replace("\n", " ")
    block = re.sub(r"[ \t]+", " ", block)
    return block.strip()


RE_ENDS_HYPHEN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]-$")
RE_STARTS_WORD = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ0-9]")


def merge_hyphenated_paragraphs(paragraphs: List[str], ctx: CleaningContext,
                                stats: CleanStats) -> List[str]:
    """Recolle un mot coupe entre DEUX paragraphes consecutifs.

    Cas reel du corpus : dans les listes de references en deux colonnes, une entree
    commencee au bas de la colonne gauche se poursuit en haut de la colonne droite
    (« ... Computation of ex- » puis « treme heat waves ... »). Les deux fragments
    sont deux blocs distincts, dans deux colonnes distinctes, mais consecutifs dans
    l'ordre de lecture.

    En prose scientifique, un paragraphe ne se termine jamais par un trait d'union :
    c'est toujours une coupure typographique. La suite peut commencer par une
    majuscule quand le trait d'union est reel (« control- » + « SWG », « IPSL- » +
    « CM6A-LR ») ; `repair_hyphenation` tranche alors par les frequences du corpus
    et conserve le trait d'union au lieu de souder les deux fragments.
    """
    merged: List[str] = []
    for para in paragraphs:
        if (merged and RE_ENDS_HYPHEN.search(merged[-1])
                and RE_STARTS_WORD.match(para)):
            joined = repair_hyphenation(f"{merged[-1]}\n{para}", ctx, stats)
            merged[-1] = re.sub(r"[ \t]+", " ", joined.replace("\n", " ")).strip()
        else:
            merged.append(para)
    return merged


def clean_page(raw_page, ctx: CleaningContext, min_block_chars: int = 0,
               page_number_tokens: frozenset = frozenset()) -> Tuple[str, CleanStats]:
    """Nettoie une page entiere et renvoie (texte_propre, statistiques_reelles).

    `min_block_chars` sert aux pages de figures : une page de graphiques produit des
    dizaines de blocs minuscules (« 5000 », « -5 », « ssp126 ») qui sont des labels
    d'axes, sans valeur pour le RAG, melanges a une VRAIE legende de figure de
    plusieurs centaines de caracteres. Filtrer sur la longueur du bloc supprime le
    bruit tout en conservant la legende scientifique.
    """
    stats = CleanStats()
    paragraphs: List[str] = []

    # On travaille sur les paragraphes reconstruits, pas sur les blocs bruts : dans
    # certaines parties du PDF un bloc PyMuPDF ne vaut qu'une seule ligne. Les
    # lignes d'un meme paragraphe sont reunies par "\n" pour que la reparation des
    # cesures de fin de ligne s'applique aussi entre deux blocs consecutifs.
    groups = raw_page.groups or [[b] for b in raw_page.blocks]

    # Marge gauche dominante de la page, mesuree sur les blocs du corps.
    left_counts: dict = {}
    for b in raw_page.blocks:
        if b.band == "body":
            left_counts[round(b.x0)] = left_counts.get(round(b.x0), 0) + 1
    dominant_left = max(left_counts, key=left_counts.get) if left_counts else None

    for group in groups:
        merged = "\n".join(b.text.strip("\n") for b in group)
        band = group[0].band
        # Numero de page isole tombe dans le corps : on ne le supprime que s'il
        # correspond reellement au numero de CETTE page, jamais sur sa seule forme.
        flat = normalise_unicode(merged).strip()
        if band == "body" and is_page_number_line(flat):
            height = raw_page.height or 1.0
            top = min(b.y0 for b in group)
            bottom = max(b.y1 for b in group)
            in_margin = (bottom <= height * PAGE_NUMBER_TOP_BAND
                         or top >= height * PAGE_NUMBER_BOTTOM_BAND)
            if flat in page_number_tokens or in_margin:
                stats.page_numbers_removed += 1
                continue
        in_outer_margin = (
            dominant_left is not None
            and any(b.x0 < dominant_left - MARGIN_TOLERANCE for b in group)
        )
        cleaned = _clean_block_text(merged, ctx, stats, band,
                                    strip_margin_numbers=in_outer_margin)
        if cleaned and len(cleaned) >= min_block_chars:
            paragraphs.append(cleaned)

    paragraphs = merge_hyphenated_paragraphs(paragraphs, ctx, stats)
    text = "\n\n".join(paragraphs)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), stats


# --- Consolidation des legendes de figures ---------------------------------
RE_CAPTION_MARKER = re.compile(r"(?:Fig\.|Figure|Tab\.|Table)\s*(?:S\s*)?\d+")
MIN_CAPTION_WORDS = 12


def consolidate_captions(text: str, min_words: int = MIN_CAPTION_WORDS) -> str:
    """Reconstitue les legendes d'une page de figures et jette le reste.

    Sur les planches de l'annexe C, la legende est coupee en plusieurs
    paragraphes par la figure elle-meme (« ... 500-hPa geopotential height » puis
    « (Z500) for the ... ») et precedee de labels de panneaux (« ssp585 »). On
    reconstruit donc chaque legende a partir de son marqueur (« Fig. S11. »),
    en lui rattachant les fragments suivants, et on ecarte tout ce qui precede
    la premiere legende ainsi que les legendes trop courtes pour avoir un sens
    sans la figure.
    """
    if not text.strip():
        return ""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    captions: List[str] = []
    current: List[str] = []

    for para in paragraphs:
        marker = RE_CAPTION_MARKER.search(para)
        if marker and marker.start() <= 40:
            # Nouvelle legende : on coupe l'eventuel label de panneau qui precede.
            if current:
                captions.append(" ".join(current))
            current = [para[marker.start():].strip()]
        elif current:
            current.append(para)
        # Un fragment situe avant toute legende est un artefact graphique : ignore.
    if current:
        captions.append(" ".join(current))

    kept = [c for c in captions if len(c.split()) >= min_words]
    return "\n\n".join(re.sub(r"\s+", " ", c).strip() for c in kept)


# --- Mots coupes entre deux pages ------------------------------------------
RE_TRAILING_HYPHEN = re.compile(r"([A-Za-zÀ-ÖØ-öø-ÿ]{2,})-$")
RE_LOWER_START = re.compile(r"^([a-zà-öø-ÿ]+)")


def detect_page_boundary_repairs(records: List[dict], ctx: CleaningContext,
                                 max_lookahead: int = 3) -> List[dict]:
    """Repere les mots coupes par une fin de page et reconstitue le mot entier.

    Un enregistrement par page ne permet pas de recoller ces mots sans detruire
    la provenance : les reparations sont donc DECRITES ici et appliquees plus
    tard par le chunker, uniquement quand un chunk traverse la frontiere.

    Le fragment de continuation est toujours en minuscules : cela ecarte d'office
    les legendes de figures (« Figure 2. ... ») qui ouvrent parfois la page
    suivante. Si la page suivante est une pleine page de figure, on poursuit la
    recherche sur la suivante, en tracant les pages sautees.
    """
    kept = [r for r in records if r.get("keep_for_rag")]
    by_page = {r["page_pdf"]: r for r in kept}
    repairs: List[dict] = []

    for rec in kept:
        text = rec["clean_text"].rstrip()
        m = RE_TRAILING_HYPHEN.search(text)
        if not m:
            continue
        stem = m.group(1)
        skipped: List[int] = []
        found = None

        for offset in range(1, max_lookahead + 1):
            nxt = by_page.get(rec["page_pdf"] + offset)
            if nxt is None:
                break
            candidate = None
            for para in (p.strip() for p in nxt["clean_text"].split("\n\n") if p.strip()):
                tok_match = RE_LOWER_START.match(para)
                if not tok_match:
                    continue
                tok = tok_match.group(1)
                joined = f"{stem}{tok}".lower()
                hyphenated = f"{stem}-{tok}".lower()
                freq_joined = ctx.word_freq.get(joined, 0)
                freq_hyphen = ctx.hyphen_freq.get(hyphenated, 0)
                if freq_joined or freq_hyphen:
                    candidate = (tok, freq_joined, freq_hyphen, para)
                    break
                if candidate is None:
                    candidate = (tok, 0, 0, para)      # premier fragment plausible
            if candidate is not None:
                found = (nxt, *candidate)
                break
            skipped.append(nxt["page_pdf"])

        if found is None:
            repairs.append({
                "left_page_pdf": rec["page_pdf"],
                "right_page_pdf": None,
                "left_fragment": stem + "-",
                "right_fragment": None,
                "repaired_token": None,
                "confidence": 0.0,
                "reason": "continuation_introuvable",
                "needs_human_review": True,
                "skipped_pages_pdf": skipped,
            })
            continue

        nxt, tok, freq_joined, freq_hyphen, para = found
        if freq_joined >= freq_hyphen and freq_joined > 0:
            token, confidence = f"{stem}{tok}", (1.0 if freq_joined >= 5 else 0.9)
            reason = "hyphenated_word_continues_on_next_page"
        elif freq_hyphen > 0:
            token, confidence = f"{stem}-{tok}", 0.9
            reason = "compound_word_continues_on_next_page"
        else:
            # Aucune des deux formes n'est attestee ailleurs dans le corpus. On
            # tranche alors sur la nature des fragments : un VRAI mot compose a
            # ses deux membres attestes isolement (« cold » et « wave ») ; une
            # simple coupure syllabique, non (« placement » n'apparait jamais
            # seul, donc « re- » + « placement » = « replacement »).
            left_is_word = ctx.word_freq.get(stem.lower(), 0) >= 3
            right_is_word = ctx.word_freq.get(tok.lower(), 0) >= 3
            if left_is_word and right_is_word:
                token = f"{stem}-{tok}"
                reason = "compound_word_continues_on_next_page_unattested"
            else:
                token = f"{stem}{tok}"
                reason = "hyphenated_word_continues_on_next_page_unattested"
            confidence = 0.75

        repairs.append({
            "left_page_pdf": rec["page_pdf"],
            "right_page_pdf": nxt["page_pdf"],
            "left_fragment": stem + "-",
            "right_fragment": tok,
            "repaired_token": token,
            "confidence": round(confidence, 2),
            "reason": reason,
            "needs_human_review": confidence < 0.9,
            "skipped_pages_pdf": skipped,
            "frequence_forme_soudee": freq_joined,
            "frequence_forme_composee": freq_hyphen,
            "left_context": text[-90:],
            "right_context": para[:90],
        })
    return repairs


def count_scientific_terms(text: str, terms: Iterable[str]) -> Dict[str, int]:
    """Compte les occurrences reelles de termes de controle (insensible a la casse)."""
    lowered = text.lower()
    return {t: lowered.count(t.lower()) for t in terms}
