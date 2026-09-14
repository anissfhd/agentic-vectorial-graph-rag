"""Couche d'ingestion : PDF -> pages nettoyees, structurees et tracees.

Etape 1 du plan d'implementation (documentation/00_COMPREHENSION_ET_ARCHITECTURE.md).
Ce paquet ne fait AUCUN chunking, aucun embedding et n'appelle aucun LLM : il
produit uniquement le corpus propre persistant sur lequel toutes les etapes
suivantes s'appuieront.

Modules :
    pdf_reader  extraction PyMuPDF bloc par bloc, ordre de lecture, double colonne
    cleaner     normalisation Unicode, cesures, en-tetes courants, numeros de page
    structure   pagination these/PDF, zones, chapitres, sections, acronymes
    language    detection FR / EN deterministe par mots outils
    quality     controles de couverture, de colonnes, de nettoyage et statistiques
"""

from .cleaner import (
    CleaningContext,
    CleanStats,
    build_context,
    clean_page,
    consolidate_captions,
    detect_page_boundary_repairs,
    normalise_unicode,
    repair_hyphenation,
)
from .language import detect_language
from .pdf_reader import (
    Block,
    RawPage,
    detect_layout,
    extract_document,
    extract_page,
    open_document,
    reading_order,
)
from .quality import (
    check_cleaning,
    check_coverage,
    check_double_column,
    check_exclusions,
    check_scientific_terms,
    compute_statistics,
)
from .structure import (
    DocumentStructure,
    build_structure,
    classify_content_type,
    decide_inclusion,
    detect_heading_in_page,
    extract_acronyms,
    find_article_reference_ranges,
    resolve_pagination,
    rule_thesis_page,
    zone_of,
)

PIPELINE_VERSION = "1.1.0"

__all__ = [
    "PIPELINE_VERSION",
    "Block", "RawPage", "open_document", "extract_page", "extract_document",
    "detect_layout", "reading_order",
    "CleaningContext", "CleanStats", "build_context", "clean_page",
    "normalise_unicode", "repair_hyphenation", "consolidate_captions",
    "detect_page_boundary_repairs",
    "detect_language",
    "DocumentStructure", "build_structure", "zone_of", "rule_thesis_page",
    "resolve_pagination", "detect_heading_in_page", "classify_content_type",
    "decide_inclusion", "extract_acronyms", "find_article_reference_ranges",
    "check_coverage", "check_double_column", "check_cleaning",
    "check_scientific_terms", "check_exclusions", "compute_statistics",
]
