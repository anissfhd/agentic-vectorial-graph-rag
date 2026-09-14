"""Detection de la langue d'une page (francais / anglais), par mots outils.

Le corpus est a environ 90 % en anglais ; le francais se limite au resume long,
aux remerciements et aux « Resume » de fin de chapitre. Une detection par mots
outils est suffisante et surtout deterministe : elle ne depend d'aucun modele
telecharge, d'aucune graine aleatoire, et donne exactement le meme resultat a
chaque execution — ce qui est exige par la contrainte de reproductibilite.

`langdetect` n'est volontairement pas utilise : son resultat varie d'une execution
a l'autre si la graine n'est pas fixee, et il est instable sur les pages courtes
riches en formules.
"""

from __future__ import annotations

import re
from typing import Tuple

FR_STOPWORDS = {
    "le", "la", "les", "des", "une", "un", "du", "de", "et", "est", "dans",
    "pour", "que", "qui", "sur", "par", "avec", "plus", "cette", "ces", "aux",
    "ont", "ete", "etre", "nous", "je", "au", "en", "il", "elle", "sont",
    "ainsi", "alors", "mais", "ou", "si", "se", "sa", "son", "leur", "leurs",
    "comme", "entre", "tout", "tous", "chapitre", "resume", "aussi", "donc",
    "peut", "cela", "afin", "lors", "sans", "meme", "deux", "trois",
}

EN_STOPWORDS = {
    "the", "of", "and", "to", "in", "a", "is", "that", "for", "with", "on",
    "as", "are", "by", "this", "be", "from", "at", "it", "we", "which", "an",
    "or", "has", "have", "was", "were", "can", "not", "but", "their", "these",
    "its", "also", "more", "than", "such", "been", "there", "when", "however",
    "between", "during", "chapter", "figure", "table", "each", "both", "using",
}

# Une page doit contenir un minimum de mots outils pour qu'un verdict ait du sens.
MIN_SIGNALS = 5

RE_TOKEN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")


def _tokens(text: str):
    for tok in RE_TOKEN.findall(text.lower()):
        # On retire les accents des mots outils francais pour comparer a la liste.
        yield tok.replace("é", "e").replace("è", "e").replace("ê", "e").replace("à", "a")


def detect_language(text: str, default: str = "en") -> Tuple[str, float]:
    """Renvoie (code_langue, confiance) ou code_langue vaut 'fr', 'en' ou 'unknown'.

    La confiance est la part du signal majoritaire : 1.0 signifie que tous les
    mots outils reconnus appartiennent a une seule langue.
    """
    if not text or not text.strip():
        return "unknown", 0.0

    fr = en = 0
    for tok in _tokens(text):
        if tok in FR_STOPWORDS:
            fr += 1
        if tok in EN_STOPWORDS:
            en += 1

    total = fr + en
    if total < MIN_SIGNALS:
        # Trop peu de signal (page de figures, de formules ou de tableaux) :
        # on retombe sur la langue dominante du document plutot que d'inventer.
        return default, 0.0

    if fr == en:
        return default, 0.5
    winner = "fr" if fr > en else "en"
    return winner, round(max(fr, en) / total, 3)
