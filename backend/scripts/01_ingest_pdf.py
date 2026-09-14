"""ETAPE 1 - Ingestion, extraction, nettoyage et controle qualite de la these.

    conda activate pfa-rag
    python backend/scripts/01_ingest_pdf.py

Le script est IDEMPOTENT : chaque execution reecrit integralement ses sorties, sans
jamais dupliquer de donnees et sans jamais modifier le PDF source, qui est ouvert
en lecture seule.

Il ne fait ni chunking, ni embedding, n'appelle aucun LLM et ne contacte aucune API.

Sorties (chemins relatifs a la racine du projet) :
    backend/data/processed/corpus_pages_raw.jsonl        1 ligne par page PDF
    backend/data/processed/corpus_clean.jsonl            pages conservees pour le RAG
    backend/data/processed/excluded_pages.jsonl          pages exclues + raison
    backend/data/processed/annex_b_out_of_context.jsonl  annexe B, hors sujet
    backend/data/processed/acronyms.json                 glossaire officiel p.128-129
    backend/data/processed/section_index.json            chapitres et sections
    backend/data/processed/document_manifest.json        identite du document
    backend/data/metrics/ingestion_report.json           toutes les statistiques
    backend/data/metrics/ingestion_quality_samples.md    echantillons avant/apres
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# --- Racine du projet : aucun chemin absolu machine n'est ecrit dans le code --
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.core.ingestion import (  # noqa: E402
    PIPELINE_VERSION,
    build_context,
    build_structure,
    check_cleaning,
    check_coverage,
    check_double_column,
    check_exclusions,
    check_scientific_terms,
    classify_content_type,
    clean_page,
    compute_statistics,
    consolidate_captions,
    decide_inclusion,
    detect_heading_in_page,
    detect_language,
    detect_page_boundary_repairs,
    extract_acronyms,
    extract_document,
    find_article_reference_ranges,
    open_document,
    resolve_pagination,
    zone_of,
)
from app.core.ingestion.pdf_reader import LAYOUT_EXCLUDED, LAYOUT_FIGURE  # noqa: E402
from app.core.ingestion.quality import pick_layout_samples  # noqa: E402
from app.core.ingestion.structure import (  # noqa: E402
    ZONE_ACRONYMS,
    ZONE_APPENDIX_B,
    ZONES,
    KEEP_ZONES,
    PAGE_OFFSET,
)

PDF_PATH = BACKEND_DIR / "data" / "raw" / "these_vagues_froid.pdf"
PROCESSED_DIR = BACKEND_DIR / "data" / "processed"
METRICS_DIR = BACKEND_DIR / "data" / "metrics"

EXPECTED_PAGES = 206
DOCUMENT_ID = "these_cadiou_2025_vagues_froid"
# Sur une page de figures, seuls les blocs assez longs sont de vraies legendes :
# le reste (« 5000 », « -5 », « ssp126 ») sont des labels d'axes.
FIGURE_CAPTION_MIN_CHARS = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingestion")


def sha256_of(path: Path) -> str:
    """Empreinte du fichier, lue par blocs (le PDF pese 45 Mo)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def write_jsonl(path: Path, records) -> None:
    """Ecriture JSONL atomique du point de vue du contenu : le fichier est reecrit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_records(doc, raw_pages, ctx, structure, sha):
    """Construit un enregistrement complet par page PDF."""
    records = []
    totals = Counter()
    current_chapter = None
    current_section = (None, None)

    for raw in raw_pages:
        page_pdf = raw.page_pdf
        zone = zone_of(page_pdf)
        warnings = list(raw.warnings)

        page_these, printed, pag_warnings = resolve_pagination(page_pdf, raw)
        warnings.extend(pag_warnings)

        # --- chapitre / section ------------------------------------------
        chapter_entry = structure.chapter_for(page_pdf)
        if chapter_entry is not None and chapter_entry is not current_chapter:
            current_chapter = chapter_entry
            current_section = (None, None)  # nouveau chapitre : section remise a zero

        section_inherited = True
        heading = detect_heading_in_page(raw)
        if heading is not None:
            current_section = heading
            section_inherited = False
        else:
            toc_section = structure.section_for(page_pdf)
            if toc_section is not None and toc_section.page_pdf == page_pdf:
                current_section = (None, toc_section.title)
                section_inherited = False
        if current_section == (None, None):
            section_inherited = False

        # --- nettoyage -----------------------------------------------------
        min_block = FIGURE_CAPTION_MIN_CHARS if raw.layout == LAYOUT_FIGURE else 0
        # Numeros reellement attribues a cette page : eux seuls peuvent etre
        # supprimes du corps. Toute autre valeur numerique est une donnee.
        tokens = {str(page_pdf)}
        if page_these is not None:
            tokens.add(str(page_these))
        clean_text, stats = clean_page(raw, ctx, min_block_chars=min_block,
                                       page_number_tokens=frozenset(tokens))
        # Sur une planche de figures, on reconstitue les legendes et on ecarte
        # les fragments graphiques qui n'ont aucun sens sans l'image.
        if raw.layout == LAYOUT_FIGURE and clean_text:
            clean_text = consolidate_captions(clean_text)
        totals["hyphens_repaired"] += stats.hyphens_repaired
        totals["hyphens_kept"] += stats.hyphens_kept
        totals["headers_removed"] += stats.headers_removed
        totals["page_numbers_removed"] += stats.page_numbers_removed
        totals["margin_line_numbers_removed"] += stats.margin_line_numbers_removed

        language, confidence = detect_language(clean_text)
        keep, reason = decide_inclusion(page_pdf, zone, clean_text, raw.layout)
        content_type = classify_content_type(
            zone, current_section[1], clean_text, raw.layout
        )

        if raw.layout == LAYOUT_FIGURE and clean_text:
            warnings.append("page_de_figures_seules_les_legendes_longues_sont_conservees")
        if keep and not clean_text.strip():
            warnings.append("page_conservee_mais_texte_vide")

        records.append({
            "document_id": DOCUMENT_ID,
            "source_file": PDF_PATH.name,
            "source_sha256": sha,
            "page_pdf": page_pdf,
            "page_these": page_these,
            "page_printed_detected": printed,
            "zone": zone,
            "chapter_number": current_chapter.chapter_number if current_chapter else None,
            "chapter_title": current_chapter.title if current_chapter else None,
            "section_number": current_section[0],
            "section_title": current_section[1],
            "section_inherited": section_inherited,
            "language": language,
            "language_confidence": confidence,
            "content_type": content_type,
            "extraction_mode": raw.layout if keep else LAYOUT_EXCLUDED,
            "detected_layout": raw.layout,
            "keep_for_rag": keep,
            "exclusion_reason": reason,
            "raw_text": raw.raw_text,
            "clean_text": clean_text,
            "raw_char_count": len(raw.raw_text),
            "clean_char_count": len(clean_text),
            "estimated_word_count": len(clean_text.split()),
            "n_images": raw.n_images,
            "warnings": warnings,
        })
    return records, totals


def annexe_c_stats(records) -> dict:
    """Bilan chiffre du traitement de l'annexe C (pages PDF 174 a 206)."""
    pages = [r for r in records if 174 <= r["page_pdf"] <= 206]
    kept = [r for r in pages if r["keep_for_rag"]]
    excluded = [r for r in pages if not r["keep_for_rag"]]
    segments = sum(len([p for p in r["clean_text"].split("\n\n") if p.strip()])
                   for r in kept)
    return {
        "pages_inspectees": len(pages),
        "pages_conservees": len(kept),
        "pages_exclues": len(excluded),
        "segments_conserves": segments,
        "caracteres_conserves": sum(r["clean_char_count"] for r in kept),
        "pages_conservees_liste": [r["page_pdf"] for r in kept],
        "pages_exclues_detail": [
            {"page_pdf": r["page_pdf"], "raison": r["exclusion_reason"]}
            for r in excluded
        ],
        "regle": "legendes reconstituees a partir de leur marqueur (« Fig. Sxx. »), "
                 "labels de panneaux et fragments graphiques ecartes, legendes de "
                 "moins de 12 mots rejetees car incomprehensibles sans la figure",
    }


def split_article_references(records, structure):
    """Sort les bibliographies finales des articles inseres hors du corpus RAG.

    Une page entierement bibliographique bascule en exclusion ; une page qui
    commence par du texte scientifique et se termine par la bibliographie est
    COUPEE au paragraphe du titre « References », son debut restant indexe.
    Les citations au fil des phrases ne sont jamais touchees.
    """
    ranges = find_article_reference_ranges(records, structure)
    by_page = {r["page_pdf"]: r for r in records}
    reference_records = []

    for rng in ranges:
        for page in range(rng["start_page_pdf"], rng["end_page_pdf"] + 1):
            rec = by_page.get(page)
            if rec is None or not rec["keep_for_rag"]:
                continue
            paragraphs = rec["clean_text"].split("\n\n")
            cut = rng["start_paragraph"] if page == rng["start_page_pdf"] else 0
            head = "\n\n".join(paragraphs[:cut]).strip()
            tail = "\n\n".join(paragraphs[cut:]).strip()
            if not tail:
                continue

            # Partie correspondante du texte BRUT, coupee au meme titre.
            raw_text = rec["raw_text"]
            if page == rng["start_page_pdf"] and cut > 0:
                marker = raw_text.find(rng["heading"])
                raw_part = raw_text[marker:] if marker >= 0 else raw_text
            else:
                raw_part = raw_text

            reference_records.append({
                "document_id": rec["document_id"],
                "source_file": rec["source_file"],
                "source_sha256": rec["source_sha256"],
                "page_pdf": page,
                "page_these": rec["page_these"],
                "chapter": rng["chapter_number"],
                "chapter_title": rng["chapter_title"],
                "article": rng["article_title"],
                "section": rng["heading"],
                "partial_page": bool(cut > 0),
                "raw_text": raw_part,
                "clean_text": tail,
                "clean_char_count": len(tail),
                "exclusion_reason": "references_internes_article",
            })

            # Mise a jour de la page dans le corpus principal.
            rec["clean_text"] = head
            rec["clean_char_count"] = len(head)
            rec["estimated_word_count"] = len(head.split())
            rec["warnings"] = rec["warnings"] + ["bibliographie_article_retiree"]
            if not head:
                rec["keep_for_rag"] = False
                rec["exclusion_reason"] = "references_internes_article"
                rec["extraction_mode"] = LAYOUT_EXCLUDED
    return ranges, reference_records


def extract_identity(records, doc):
    """Recupere titre, auteure, universite et annee, par lecture reelle du document."""
    meta = doc.metadata or {}
    front = "\n".join(r["clean_text"] for r in records if r["page_pdf"] <= 5)
    university = None
    m = re.search(r"Universit[eé]\s+Paris-Saclay", front, re.IGNORECASE)
    if m:
        university = m.group(0)
    year = None
    m = re.search(r"NNT\s*:\s*(\d{4})", front)
    if m:
        year = int(m.group(1))
    return {
        "titre": (meta.get("title") or "").strip() or None,
        "auteure": (meta.get("author") or "").strip() or None,
        "universite": university,
        "annee": year,
        "sujet": (meta.get("subject") or "").strip() or None,
    }


def build_step_1b_section(records, ref_ranges, reference_records,
                          boundary_repairs) -> list:
    """Section du rapport qualite consacree a la finalisation du corpus (etape 1B)."""
    by_page = {r["page_pdf"]: r for r in records}
    lines = [
        "---",
        "",
        "## 6. Etape 1B - finalisation du corpus",
        "",
        "### 6.1 Bibliographies internes aux articles inseres",
        "",
        "Le debut est un paragraphe reduit au seul mot « References » ; la fin est la",
        "prochaine entree de la table des matieres. Les citations au fil des phrases",
        "scientifiques ne sont jamais touchees.",
        "",
        "| Debut | Fin | Chapitre | Article | Page coupee |",
        "|---|---|---|---|---|",
    ]
    for rng in ref_ranges:
        lines.append(
            f"| p{rng['start_page_pdf']} (par. {rng['start_paragraph']}) "
            f"| p{rng['end_page_pdf']} | {rng['chapter_number']} "
            f"| {(rng['article_title'] or '')[:56]} "
            f"| {'oui' if rng['start_paragraph'] > 0 else 'non'} |"
        )
    total_chars = sum(r["clean_char_count"] for r in reference_records)
    lines += [
        "",
        f"**{len(reference_records)} pages** sorties du corpus RAG "
        f"({sum(1 for r in reference_records if not r['partial_page'])} entieres, "
        f"{sum(1 for r in reference_records if r['partial_page'])} partielles), "
        f"soit **{total_chars:,} caracteres**, conservees dans "
        "`backend/data/processed/article_references.jsonl`.",
        "",
        "Sur les deux pages coupees, la partie scientifique reste indexee :",
        "",
    ]
    for rng in ref_ranges:
        if rng["start_paragraph"] == 0:
            continue
        rec = by_page[rng["start_page_pdf"]]
        lines += [
            f"- **p{rec['page_pdf']}** conserve {rec['clean_char_count']} caracteres :",
            "",
            "```",
            rec["clean_text"][:280].strip(),
            "```",
            "",
        ]

    # --- 6.2 Annexe C ----------------------------------------------------
    annexe = [r for r in records if 174 <= r["page_pdf"] <= 206]
    kept = [r for r in annexe if r["keep_for_rag"]]
    lines += [
        "### 6.2 Annexe C - legendes reconstituees",
        "",
        f"{len(annexe)} pages inspectees, **{len(kept)} conservees**, "
        f"{len(annexe) - len(kept)} exclues. Chaque legende est reconstruite a partir",
        "de son marqueur (« Fig. Sxx. ») ; les labels de panneaux qui la precedent",
        "(« ssp585 ») et les fragments de moins de 12 mots sont ecartes.",
        "",
    ]
    for page in (176, 186, 196):
        rec = by_page.get(page)
        if rec is None or not rec["clean_text"]:
            continue
        lines += [
            f"**p{page}** ({len(rec['clean_text'].split())} mots) :",
            "",
            "```",
            rec["clean_text"][:340].strip(),
            "```",
            "",
        ]
    excluded = [r for r in annexe if not r["keep_for_rag"]]
    if excluded:
        lines.append("Pages exclues : " + ", ".join(
            f"p{r['page_pdf']} ({r['exclusion_reason']})" for r in excluded) + ".")
        lines.append("")

    # --- 6.3 Mots coupes entre deux pages --------------------------------
    lines += [
        "### 6.3 Mots coupes entre deux pages",
        "",
        "Ces reparations ne sont **pas** appliquees au corpus : la provenance page par",
        "page est preservee. Elles seront appliquees par le chunker quand un chunk",
        "traversera la frontiere. Voir `page_boundary_repairs.json`.",
        "",
        "| De | Vers | Fragment gauche | Fragment droit | Mot reconstitue | Confiance | A relire |",
        "|---|---|---|---|---|---|---|",
    ]
    for rep in boundary_repairs:
        lines.append(
            f"| p{rep['left_page_pdf']} | p{rep['right_page_pdf']} "
            f"| `{rep['left_fragment']}` | `{rep['right_fragment']}` "
            f"| **{rep['repaired_token']}** | {rep['confidence']} "
            f"| {'oui' if rep['needs_human_review'] else 'non'} |"
        )
    lines += [
        "",
        "La decision vient des frequences reelles du corpus. Quand aucune des deux",
        "formes n'est attestee, on tranche sur la nature des fragments : « cold » et",
        "« wave » existent isolement, c'est donc un mot compose (`cold-wave`) ;",
        "« placement » n'existe jamais seul, c'est donc une coupure syllabique",
        "(`replacement`). Ces deux cas sont signales pour relecture humaine.",
        "",
    ]
    return lines


def build_quality_samples(raw_pages, records, ctx, samples_path: Path,
                          double_column_checks, step_1b_lines=None) -> None:
    """Redige le rapport lisible avant/apres, avec des extraits reels."""
    by_page = {r["page_pdf"]: r for r in records}
    lines = [
        "# Controle qualite de l'ingestion - echantillons reels",
        "",
        f"Genere le {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
        f"par le pipeline d'ingestion v{PIPELINE_VERSION}.",
        "",
        "Tous les extraits ci-dessous sont copies tels quels depuis le corpus produit.",
        "",
        "---",
        "",
        "## 1. Pages en double colonne",
        "",
        "Le defaut redoute est l'alternation d'une ligne de la colonne gauche et d'une",
        "ligne de la colonne droite. La mesure objective est le **nombre de basculements",
        "gauche/droite** dans l'ordre de lecture : en lecture correcte il vaut au plus le",
        "nombre de bandes horizontales de la page (typiquement 1), alors que l'ordre naif",
        "trie par ordonnee bascule a chaque paragraphe.",
        "",
    ]

    for label, raw, check in double_column_checks:
        rec = by_page[raw.page_pdf]
        two_col = check["plage_contient_du_double_colonne"]
        lines += [
            f"### {label} - page PDF {raw.page_pdf} (page these {rec['page_these']})",
            "",
            f"- Mode detecte : `{rec['detected_layout']}`",
            f"- Blocs colonne gauche / droite : {check['blocs_colonne_gauche']} / "
            f"{check['blocs_colonne_droite']}",
            f"- Bandes horizontales : {check['bandes']}",
            f"- Largeur du bloc le plus large / largeur de page : "
            f"**{check['largeur_bloc_max_sur_page']}**",
            f"- **Basculements avec l'ordre retenu : {check['basculements_ordre_retenu']}**",
            f"- Basculements avec l'ordre naif (temoin) : "
            f"{check['basculements_ordre_naif']}",
            f"- Colonnes non melangees : "
            f"{'OUI' if check['colonnes_non_melangees'] else 'NON'}",
            "",
        ]
        if not two_col:
            lines += [
                "> **Constat mesure : cette plage ne contient aucune page en double "
                "colonne.** Le bloc le plus large occupe "
                f"{check['largeur_bloc_max_sur_page']} de la largeur de page, ce qui "
                "correspond a une composition pleine largeur. Le risque de melange de "
                "colonnes ne s'applique donc pas ici. Aucun echantillon n'a ete force.",
                "",
            ]
        lines += [
            "**Extrait brut (ordre de lecture reconstruit, avant nettoyage) :**",
            "",
            "```",
            rec["raw_text"][:900].strip(),
            "```",
            "",
            "**Extrait nettoye :**",
            "",
            "```",
            rec["clean_text"][:900].strip(),
            "```",
            "",
        ]
        if two_col:
            lines += [
                "**Justification :** l'ordre retenu emet l'integralite de la colonne "
                "gauche avant la colonne droite dans chaque bande, d'ou un nombre de "
                f"basculements egal a {check['basculements_ordre_retenu']} pour "
                f"{check['bandes']} bande(s). L'ordre naif en produirait "
                f"{check['basculements_ordre_naif']}, soit autant de phrases coupees "
                "par un fragment de l'autre colonne.",
                "",
            ]
        else:
            lines += [
                "**Justification :** page mono-colonne, lue de haut en bas. Les "
                f"{check['basculements_ordre_retenu']} basculements mesures confirment "
                "l'absence de structure en colonnes.",
                "",
            ]

    # --- 2. Reparation des cesures ---------------------------------------
    lines += ["---", "", "## 2. Reparation des cesures de fin de ligne", ""]
    shown = 0
    for raw in raw_pages:
        if shown >= 3:
            break
        rec = by_page[raw.page_pdf]
        if not rec["keep_for_rag"]:
            continue
        m = re.search(r"([A-Za-zÀ-ÿ]{3,})-\n([a-zà-ÿ]{2,})", raw.raw_text)
        if not m:
            continue
        fragment = m.group(0).replace("\n", "\\n")
        joined = f"{m.group(1)}{m.group(2)}"
        if joined.lower() not in rec["clean_text"].lower():
            continue
        idx = rec["clean_text"].lower().find(joined.lower())
        lines += [
            f"### Page PDF {raw.page_pdf}",
            "",
            f"- Avant : `{fragment}`",
            f"- Apres : `{joined}`",
            f"- Contexte nettoye : `...{rec['clean_text'][max(0, idx - 70):idx + 70]}...`",
            "",
        ]
        shown += 1
    if shown == 0:
        lines += ["_Aucune cesure de fin de ligne detectee dans le corpus conserve._", ""]

    # --- 3. En-tetes courants supprimes -----------------------------------
    lines += [
        "---",
        "",
        "## 3. En-tetes et pieds de page courants supprimes",
        "",
        "Ils ont ete reperes par leur repetition mesuree a travers les pages, et non "
        "par une liste ecrite a la main.",
        "",
    ]
    for head in sorted(ctx.running_heads)[:18]:
        lines.append(f"- `{head}`")
    lines += ["", f"_Total : {len(ctx.running_heads)} motifs repetitifs identifies._", ""]

    # --- 4. Normalisation Unicode -----------------------------------------
    lines += ["---", "", "## 4. Normalisation Unicode et ligatures", ""]
    shown = 0
    for raw in raw_pages:
        if shown >= 3:
            break
        rec = by_page[raw.page_pdf]
        if not rec["keep_for_rag"]:
            continue
        m = re.search(r"\S*[ﬀ-ﬆ−–—’]\S*", raw.raw_text)
        if not m:
            continue
        lines += [
            f"- Page PDF {raw.page_pdf} : `{m.group(0)}` "
            f"(U+{ord([c for c in m.group(0) if c in chr(0xfb01) + chr(0xfb02) + chr(0x2212) + chr(0x2013) + chr(0x2014) + chr(0x2019)][0]):04X})",
        ]
        shown += 1
    lines.append("")

    # --- 5. Pages a pagination conflictuelle -------------------------------
    conflicts = [
        r for r in records
        if any(w.startswith("numerotation_etrangere") for w in r["warnings"])
    ]
    lines += [
        "---",
        "",
        "## 5. Pages portant une numerotation etrangere",
        "",
        "Les articles de revue inseres impriment leur PROPRE numero de page. La "
        "numerotation de la these est conservee dans `page_these` pour que les "
        "citations restent verifiables, et le numero etranger est archive dans "
        "`page_printed_detected`.",
        "",
        f"Nombre de pages concernees : **{len(conflicts)}**",
        "",
    ]
    if conflicts:
        lines += ["| page_pdf | page_these (regle) | numero imprime | chapitre |",
                  "|---|---|---|---|"]
        for r in conflicts[:15]:
            lines.append(
                f"| {r['page_pdf']} | {r['page_these']} | "
                f"{r['page_printed_detected']} | {r['chapter_number']} |"
            )
        lines.append("")

    if step_1b_lines:
        lines.extend(step_1b_lines)

    samples_path.parent.mkdir(parents=True, exist_ok=True)
    samples_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    t0 = time.perf_counter()
    log.info("=" * 66)
    log.info("ETAPE 1 - INGESTION DE LA THESE")
    log.info("=" * 66)

    # --- 1. Verification du corpus ---------------------------------------
    if not PDF_PATH.exists():
        log.error("Corpus introuvable : %s", PDF_PATH)
        return 1
    size_mb = PDF_PATH.stat().st_size / (1024 * 1024)
    sha = sha256_of(PDF_PATH)
    log.info("Corpus     : %s (%.1f Mo)", PDF_PATH.relative_to(PROJECT_ROOT), size_mb)
    log.info("SHA-256    : %s", sha)

    doc = open_document(PDF_PATH)
    log.info("Pages      : %d (attendu %d)", doc.page_count, EXPECTED_PAGES)
    if doc.page_count != EXPECTED_PAGES:
        log.warning("Nombre de pages inattendu : %d", doc.page_count)

    # --- 2. Extraction ----------------------------------------------------
    log.info("Extraction page par page, bloc par bloc...")
    raw_pages = extract_document(doc)

    no_text = [p.page_pdf for p in raw_pages if not p.raw_text.strip()]
    ocr_candidates = [
        p.page_pdf for p in raw_pages if not p.raw_text.strip() and p.n_images > 0
    ]
    log.info("Couche texte : %d/%d pages porteuses de texte",
             doc.page_count - len(no_text), doc.page_count)
    log.info("Pages sans texte : %s", no_text or "aucune")
    if ocr_candidates:
        log.warning("Pages sans texte MAIS avec image (OCR eventuel, non lance) : %s",
                    ocr_candidates)
    else:
        log.info("Aucun OCR necessaire : aucune page n'est une image sans couche texte.")

    layout_counts = Counter(p.layout for p in raw_pages)
    log.info("Mises en page detectees : %s", dict(layout_counts))

    # --- 3. Contexte global puis nettoyage --------------------------------
    log.info("Analyse globale (en-tetes courants, vocabulaire de cesure)...")
    ctx = build_context(raw_pages)
    log.info("En-tetes/pieds courants identifies : %d motifs", len(ctx.running_heads))

    structure = build_structure(doc)
    log.info("Table des matieres : %d entrees", len(structure.entries))

    log.info("Nettoyage et structuration...")
    records, totals = build_records(doc, raw_pages, ctx, structure, sha)

    # --- 3bis. Etape 1B : bibliographies internes hors du corpus RAG -------
    ref_ranges, reference_records = split_article_references(records, structure)
    log.info("Bibliographies d'articles detectees : %d plage(s)", len(ref_ranges))
    for rng in ref_ranges:
        log.info("   p%d(par.%d) -> p%d | %s",
                 rng["start_page_pdf"], rng["start_paragraph"],
                 rng["end_page_pdf"], (rng["article_title"] or "")[:52])
    log.info("Pages de references sorties du corpus : %d", len(reference_records))

    # --- 3ter. Mots coupes entre deux pages --------------------------------
    boundary_repairs = detect_page_boundary_repairs(records, ctx)
    log.info("Reparations de frontieres de pages : %d (a relire : %d)",
             len(boundary_repairs),
             sum(1 for r in boundary_repairs if r["needs_human_review"]))

    kept = [r for r in records if r["keep_for_rag"]]
    excluded = [r for r in records if not r["keep_for_rag"]]
    annex_b = [r for r in records if r["zone"] == ZONE_APPENDIX_B]

    log.info("Pages conservees : %d | exclues : %d", len(kept), len(excluded))

    # --- 4. Acronymes -----------------------------------------------------
    acronyms = extract_acronyms(raw_pages)
    log.info("Acronymes extraits : %d", len(acronyms))

    # --- 5. Ecriture des sorties ------------------------------------------
    write_jsonl(PROCESSED_DIR / "corpus_pages_raw.jsonl", records)
    write_jsonl(PROCESSED_DIR / "corpus_clean.jsonl", kept)
    write_jsonl(PROCESSED_DIR / "excluded_pages.jsonl", excluded)
    write_jsonl(PROCESSED_DIR / "annex_b_out_of_context.jsonl", annex_b)
    write_jsonl(PROCESSED_DIR / "article_references.jsonl", reference_records)
    write_json(PROCESSED_DIR / "page_boundary_repairs.json", {
        "note": "Mots coupes par une fin de page. Ce fichier N'EST PAS applique a "
                "corpus_pages_raw.jsonl : la provenance page par page est preservee. "
                "Le chunker appliquera ces reparations lorsqu'un chunk traversera la "
                "frontiere entre left_page_pdf et right_page_pdf.",
        "count": len(boundary_repairs),
        "a_relire": sum(1 for r in boundary_repairs if r["needs_human_review"]),
        "repairs": boundary_repairs,
    })
    write_json(PROCESSED_DIR / "acronyms.json", {
        "source_pages_pdf": [128, 129],
        "note": "Glossaire officiel de l'auteure. Les pages de reference sont des "
                "numeros de page THESE (page_pdf = page_these + 15).",
        "count": len(acronyms),
        "acronyms": acronyms,
    })

    section_index = [{
        "level": e.level,
        "title": e.title,
        "page_pdf": e.page_pdf,
        "page_these": (e.page_pdf - PAGE_OFFSET) if e.page_pdf >= 16 else None,
        "chapter_number": e.chapter_number,
    } for e in structure.entries]
    for i, item in enumerate(section_index):
        nxt = section_index[i + 1]["page_pdf"] - 1 if i + 1 < len(section_index) else EXPECTED_PAGES
        item["page_pdf_end"] = max(item["page_pdf"], nxt)
    write_json(PROCESSED_DIR / "section_index.json", {
        "count": len(section_index),
        "sections": section_index,
    })

    identity = extract_identity(records, doc)
    lang_counter = Counter(r["language"] for r in kept)
    write_json(PROCESSED_DIR / "document_manifest.json", {
        **identity,
        "document_id": DOCUMENT_ID,
        "source_file": PDF_PATH.name,
        "nombre_de_pages": doc.page_count,
        "sha256": sha,
        "taille_octets": PDF_PATH.stat().st_size,
        "langue_principale": lang_counter.most_common(1)[0][0] if lang_counter else None,
        "repartition_langues_corpus_rag": dict(lang_counter),
        "plages_incluses": [
            {"debut_pdf": s, "fin_pdf": e, "zone": z}
            for s, e, z in ZONES if z in KEEP_ZONES
        ],
        "plages_exclues": [
            {"debut_pdf": s, "fin_pdf": e, "zone": z}
            for s, e, z in ZONES if z not in KEEP_ZONES
        ],
        "zone_speciale_acronymes": {"debut_pdf": 128, "fin_pdf": 129,
                                    "sortie": "acronyms.json"},
        "bibliographies_articles_exclues": [
            {"debut_pdf": r["start_page_pdf"], "fin_pdf": r["end_page_pdf"],
             "article": r["article_title"], "chapitre": r["chapter_number"],
             "page_partielle": r["start_paragraph"] > 0,
             "sortie": "article_references.jsonl"}
            for r in ref_ranges
        ],
        "regles_pagination": {
            "formule": "page_these = page_pdf - 15",
            "premiere_page_numerotee_pdf": 16,
            "pages_liminaires_pdf": "1-15 (page_these = null, numerotation romaine)",
            "tolerance_ecart": 2,
            "conflit": "un numero imprime tres eloigne de la regle provient d'un "
                       "article de revue insere ; la regle these est conservee et le "
                       "numero etranger archive dans page_printed_detected",
        },
        "date_traitement": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version_pipeline": PIPELINE_VERSION,
    })

    # --- 6. Controle qualite ----------------------------------------------
    log.info("Controle qualite...")
    coverage = check_coverage(records, EXPECTED_PAGES)
    cleaning = check_cleaning(kept, ctx.running_heads)
    terms = check_scientific_terms(kept)
    exclusions = check_exclusions(kept, annex_b, acronyms)

    samples = pick_layout_samples(raw_pages, [
        (50, 85, "Chapitre 3 - article publie"),
        (86, 121, "Chapitre 4 - article en revision"),
        (174, 206, "Annexe C - supplement"),
        (150, 173, "Annexe B - hors sujet (temoin)"),
    ])
    double_checks = []
    for label, raw, is_two_col in samples:
        chk = check_double_column(raw)
        chk["plage_contient_du_double_colonne"] = is_two_col
        chk["libelle"] = label
        double_checks.append((label, raw, chk))

    stats = compute_statistics(records, kept, totals)
    stats["temps_execution_s"] = round(time.perf_counter() - t0, 2)
    stats["acronymes_extraits"] = len(acronyms)

    all_warnings = Counter(w for r in records for w in r["warnings"])

    report = {
        "genere_le": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "etape": "1 - ingestion, extraction, nettoyage",
        "version_pipeline": PIPELINE_VERSION,
        "source": {
            "fichier": PDF_PATH.name,
            "sha256": sha,
            "taille_octets": PDF_PATH.stat().st_size,
            "pages": doc.page_count,
            "pages_attendues": EXPECTED_PAGES,
            "pages_sans_couche_texte": no_text,
            "pages_candidates_ocr": ocr_candidates,
            "ocr_necessaire": bool(ocr_candidates),
        },
        "statistiques": stats,
        "controles": {
            "A_couverture": coverage,
            "B_double_colonne": [c for _, _, c in double_checks],
            "C_nettoyage": cleaning,
            "D_termes_scientifiques": terms,
            "E_plages_exclues": exclusions,
        },
        "etape_1B": {
            "bibliographies_articles": {
                "plages_detectees": ref_ranges,
                "pages_sorties_du_corpus": len(reference_records),
                "pages_entieres": sum(1 for r in reference_records if not r["partial_page"]),
                "pages_partielles": sum(1 for r in reference_records if r["partial_page"]),
                "caracteres_retires": sum(r["clean_char_count"] for r in reference_records),
                "sortie": "backend/data/processed/article_references.jsonl",
            },
            "annexe_c": annexe_c_stats(records),
            "reparations_frontieres_pages": {
                "total": len(boundary_repairs),
                "a_relire": sum(1 for r in boundary_repairs if r["needs_human_review"]),
                "sortie": "backend/data/processed/page_boundary_repairs.json",
                "detail": [
                    {"de": r["left_page_pdf"], "vers": r["right_page_pdf"],
                     "mot": r["repaired_token"], "confiance": r["confidence"]}
                    for r in boundary_repairs
                ],
            },
        },
        "avertissements": dict(all_warnings),
        "volume_reference_historique": {
            "caracteres_utiles_attendus_ordre_de_grandeur": 389000,
            "caracteres_utiles_mesures": stats["caracteres_nettoyes_corpus_rag"],
            "ecart_relatif": round(
                (stats["caracteres_nettoyes_corpus_rag"] - 389000) / 389000, 4
            ),
            "note": "Ordre de grandeur historique, jamais utilise pour ajuster les "
                    "donnees. L'ecart est documente, pas corrige.",
        },
    }
    write_json(METRICS_DIR / "ingestion_report.json", report)
    build_quality_samples(
        raw_pages, records, ctx,
        METRICS_DIR / "ingestion_quality_samples.md", double_checks,
        step_1b_lines=build_step_1b_section(records, ref_ranges, reference_records,
                                            boundary_repairs),
    )

    doc.close()

    # --- 7. Bilan console --------------------------------------------------
    log.info("-" * 66)
    for name, res in (("A couverture", coverage), ("C nettoyage", cleaning),
                      ("D termes", terms), ("E exclusions", exclusions)):
        log.info("Controle %-14s : %s", name, "OK" if res["ok"] else "ECHEC")
    for label, raw, chk in double_checks:
        log.info("  p%-4d %-34s %-16s retenu=%d naif=%d bandes=%d largeur=%.2f -> %s",
                 raw.page_pdf, label, chk["layout"],
                 chk["basculements_ordre_retenu"], chk["basculements_ordre_naif"],
                 chk["bandes"], chk["largeur_bloc_max_sur_page"],
                 "OK" if chk["colonnes_non_melangees"] else "ECHEC")
    log.info("Caracteres corpus RAG : %d | mots : %d",
             stats["caracteres_nettoyes_corpus_rag"], stats["mots_estimes_corpus_rag"])
    log.info("Duree : %.2f s", stats["temps_execution_s"])
    log.info("Rapport : %s", (METRICS_DIR / "ingestion_report.json").relative_to(PROJECT_ROOT))

    ok = all(r["ok"] for r in (coverage, cleaning, terms, exclusions)) and all(
        c["colonnes_non_melangees"] for _, _, c in double_checks
    )
    log.info("RESULTAT GLOBAL : %s", "OK" if ok else "ECHEC - voir le rapport")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
