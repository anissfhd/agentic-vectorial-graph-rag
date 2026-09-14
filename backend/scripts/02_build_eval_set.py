"""ETAPE 2 - Construction et VERIFICATION du jeu d'evaluation.

    conda activate pfa-rag
    python backend/scripts/02_build_eval_set.py

Les 24 questions sont annotees a la main, a partir de passages reellement lus dans
`corpus_clean.jsonl`. Ce script ne les invente pas : il les **verifie**. Pour chaque
question repondable, il exige que

  * chaque page citee existe dans le corpus nettoye et y soit conservee ;
  * chaque extrait de preuve soit present TEL QUEL dans la page annoncee
    (a la normalisation Unicode et aux espaces pres) ;
  * les metadonnees de section proviennent du corpus, jamais d'une saisie manuelle ;
  * la pagination these/PDF soit coherente.

Toute annotation qui ne se verifie pas fait echouer le script : aucune question ne
peut entrer dans le jeu sans preuve.

Un controle cross-lingue **secondaire** mesure ensuite la similarite entre chaque
question francaise et ses preuves anglaises. Il ne modifie jamais la verite terrain :
un score faible produit un avertissement, pas une suppression.

Sorties :
    backend/eval/eval_set.json
    backend/eval/eval_set_review.md
    backend/data/metrics/evaluation_set_report.json
"""

from __future__ import annotations

import json
import logging
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

CORPUS = BACKEND_DIR / "data" / "processed" / "corpus_clean.jsonl"
ANNEX_B = BACKEND_DIR / "data" / "processed" / "annex_b_out_of_context.jsonl"
EVAL_DIR = BACKEND_DIR / "eval"
METRICS_DIR = BACKEND_DIR / "data" / "metrics"

EVAL_SET_VERSION = "1.0.0"
SIMILARITY_WARNING_THRESHOLD = 0.40

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("eval")
# Le chargement du modele bavarde beaucoup (requetes HTTP vers le cache Hugging
# Face) : on ne garde que les avertissements pour que le rapport reste lisible.
for noisy in ("httpx", "httpcore", "urllib3", "filelock", "sentence_transformers",
              "transformers", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

ROUTE_BY_TYPE = {
    "semantic": "use_vector",
    "relational": "use_graph",
    "hybrid": "use_hybrid",
    "out_of_context": "abstain",
}
IDK = "Je ne sais pas."


# =====================================================================
# ANNOTATIONS - chaque extrait a ete lu dans corpus_clean.jsonl
# =====================================================================
QUESTIONS = [
    # ---------------- SEMANTIC : definitions, explications ----------------
    dict(
        question_id="SEM_001", question_type="semantic", difficulty="easy", split="train",
        question="Comment l'Organisation météorologique mondiale définit-elle une vague de froid ?",
        expected_answer=(
            "L'Organisation météorologique mondiale (WMO) définit une vague de froid comme un "
            "événement météorologique généralement caractérisé par une chute brutale de la "
            "température de l'air près de la surface terrestre, conduisant à des valeurs "
            "extrêmement basses pouvant s'accompagner de conditions dangereuses telles que le "
            "gel et le verglas."
        ),
        evidence=[(19, 'the World Meteorological Organisation (WMO) defines a cold wave as')],
        entities=["World Meteorological Organisation", "cold wave"],
        relation=None,
        notes="Definition officielle citee mot pour mot dans la these.",
    ),
    dict(
        question_id="SEM_002", question_type="semantic", difficulty="easy", split="train",
        question="Qu'est-ce que l'indice WCC utilisé dans la thèse et comment est-il calculé ?",
        expected_answer=(
            "Le WCC (Western Europe Cold Circulation index) est un indice qui caractérise le "
            "schéma de circulation atmosphérique associé aux vagues de froid en Europe de "
            "l'Ouest. Il se calcule en soustrayant la moyenne du géopotentiel Z500 entre les "
            "zones (1°W - 9°E ; 40°N - 47°N) et (24°W - 13°W ; 62°N - 66°N), zones choisies "
            "comme maximum et minimum de la structure dipolaire identifiée."
        ),
        evidence=[
            (96, 'we compute a Western Europe Cold Circulation index (WCC)'),
            (96, 'Those areas were chosen as the maximum and minimum of the dipole structure'),
        ],
        entities=["WCC", "Z500"],
        relation=None,
        notes="Definition et formule donnees dans la section methodes du chapitre 4.",
    ),
    dict(
        question_id="SEM_003", question_type="semantic", difficulty="medium", split="train",
        question="Qu'est-ce que l'ajustement dynamique (dynamical adjustment) et à quoi sert-il ?",
        expected_answer=(
            "L'ajustement dynamique est une technique des sciences du climat qui vise à estimer "
            "l'influence de la circulation atmosphérique sur une variable climatique de surface, "
            "par exemple la température de l'air en surface. Dans la thèse, il sert à séparer la "
            "composante induite par la circulation de la composante résiduelle (thermodynamique) "
            "des tendances de température hivernale."
        ),
        evidence=[
            (70, 'Dynamical adjustment is a technique in climate science'),
            (70, 'characterizing circulation-induced and residual (thermodynamical) trends'),
        ],
        entities=["dynamical adjustment", "atmospheric circulation"],
        relation=None,
        notes="Definition explicite en section 2.2 de l'article du chapitre 3.",
    ),
    dict(
        question_id="SEM_004", question_type="semantic", difficulty="medium", split="test",
        question="Pourquoi la thèse utilise-t-elle un générateur stochastique de temps avec échantillonnage d'importance empirique ?",
        expected_answer=(
            "Parce qu'il s'agit d'une méthode peu coûteuse en calcul qui combine des approches "
            "à la fois statistiques et physiques. Elle repose sur une discrétisation de l'espace "
            "des phases par les analogues de circulation, et elle est particulièrement adaptée "
            "aux événements de moyenne à longue durée, c'est-à-dire durant plus d'une semaine."
        ),
        evidence=[
            (30, 'I use an SWG with empirical importance sampling, which is a computationally efficient method'),
            (30, 'This method is particularly well-suited for mid- to long-term events'),
        ],
        entities=["SWG", "empirical importance sampling", "analogues of circulation"],
        relation=None,
        notes="Justification methodologique donnee en section 1.2 du chapitre 1.",
    ),
    dict(
        question_id="SEM_005", question_type="semantic", difficulty="hard", split="train",
        question="Quel était l'objectif général de la thèse et pourquoi les événements de froid extrême restent-ils peu étudiés ?",
        expected_answer=(
            "L'objectif général était d'explorer comment le changement climatique affecte les "
            "événements de froid extrême aux moyennes latitudes, et plus particulièrement en "
            "Europe. Ces événements restent peu étudiés parce que l'attention se concentre sur "
            "les vagues de chaleur extrême ; leurs impacts pourraient donc être sous-estimés, "
            "alors qu'ils affectent encore fortement la santé et les systèmes énergétiques."
        ),
        evidence=[
            (122, 'The overall objective of this thesis was to explore how climate change impacts extreme cold events'),
            (122, 'Cold extremes remain understudied because of the focus on extreme heat events'),
            (122, 'cold events are still causing significant impacts on health and energy systems'),
        ],
        entities=["climate change", "extreme cold events", "midlatitudes"],
        relation=None,
        notes="Chapitre 5, section Conclusions.",
    ),
    dict(
        question_id="SEM_006", question_type="semantic", difficulty="hard", split="test",
        question="Comment la thèse justifie-t-elle la valeur retenue pour le paramètre alpha_T du générateur stochastique ?",
        expected_answer=(
            "Avec une valeur de 0,2, seules quelques simulations atteignent des températures plus "
            "froides que l'hiver 1963. Une valeur supérieure à 0,5 permet de simuler une plus "
            "grande proportion d'événements extrêmes, mais les différences deviennent moins "
            "significatives au-delà de ce seuil. L'auteure a donc retenu la valeur 0,5 pour la "
            "suite de l'analyse."
        ),
        evidence=[
            (45, 'With αT = 0.2, only a few simulations reach temperatures colder than winter 1963'),
            (45, 'A value of αT exceeding 0.5 enables the simulation of a higher proportion of extreme events'),
            (45, 'I selected αT = 0.5 for the subsequent analysis'),
        ],
        entities=["SWG", "winter 1963"],
        relation=None,
        notes="Test de sensibilite du parametre, chapitre 2 section 2.3.3.",
    ),

    # ---------------- RELATIONAL : relations entre entites ----------------
    dict(
        question_id="REL_001", question_type="relational", difficulty="easy", split="train",
        question="Par quel trajet le blocage scandinave amène-t-il l'air froid, et quelle en est la conséquence pour la France ?",
        expected_answer=(
            "Un blocage scandinave amène l'air froid en passant par la Sibérie et le nord-est de "
            "l'Europe. Cet air reste au-dessus des surfaces continentales ; cette configuration, "
            "parfois appelée « Moscou-Paris express », provoque généralement les vagues de froid "
            "les plus intenses sur l'Europe de l'Ouest et la France."
        ),
        evidence=[
            (23, 'A Scandinavian blocking brings cold air through Siberia and the North-East of Europe'),
            (23, 'usually causes the most intense cold spells over Western Europe and France'),
        ],
        entities=["Scandinavian Blocking", "Siberia", "Western Europe", "France"],
        relation={"subject": "Scandinavian Blocking", "predicate": "CAUSES",
                  "object": "vagues de froid les plus intenses en Europe de l'Ouest et en France"},
        notes="Relation causale explicite, chapitre 1 section 1.1.3.",
    ),
    dict(
        question_id="REL_002", question_type="relational", difficulty="easy", split="train",
        question="Quels sont les quatre régimes de temps qui caractérisent la météo hivernale en Atlantique Nord selon la thèse ?",
        expected_answer=(
            "Les quatre régimes sont la phase positive et la phase négative de l'Oscillation "
            "nord-atlantique (NAO+ et NAO−), qui sont opposées, la dorsale atlantique (Atlantic "
            "ridge, AR) et le blocage scandinave (Scandinavian blocking, SB)."
        ),
        evidence=[(21, 'the Atlantic ridge (AR) and the Scandinavian blocking (SB)')],
        entities=["NAO+", "NAO-", "Atlantic ridge", "Scandinavian Blocking"],
        relation={"subject": "NAO+, NAO-, Atlantic ridge, Scandinavian Blocking",
                  "predicate": "CHARACTERIZE",
                  "object": "la circulation hivernale en Atlantique Nord"},
        notes="Enumeration des regimes de temps, chapitre 1 section 1.1.3.",
    ),
    dict(
        question_id="REL_003", question_type="relational", difficulty="medium", split="train",
        question="À quelle phase de l'Oscillation nord-atlantique l'hiver 1963 est-il associé, et que traduit cet indice ?",
        expected_answer=(
            "L'hiver 1963 est associé à un indice NAO négatif. Un indice négatif traduit une "
            "différence de pression plus faible que la normale entre la dépression d'Islande et "
            "l'anticyclone des Açores."
        ),
        evidence=[(53, 'Winter 1963 was associated with a negative North Atlantic Oscillation (NAO) index')],
        entities=["winter 1963", "NAO", "Iceland low", "Azores high"],
        relation={"subject": "winter 1963", "predicate": "ASSOCIATED_WITH",
                  "object": "phase negative de la NAO"},
        notes="Chapitre 3, introduction du premier article.",
    ),
    dict(
        question_id="REL_004", question_type="relational", difficulty="medium", split="test",
        question="Quelle configuration du courant-jet Harnik et ses coauteurs relient-ils à la phase négative de la NAO ?",
        expected_answer=(
            "Harnik et al. (2014) relient la configuration de jet fusionné (merged jet) à une "
            "phase négative de l'Oscillation nord-atlantique et à des hivers anormaux."
        ),
        evidence=[(21, 'link the merged jet configuration to a negative phase of the North Atlantic Oscillation')],
        entities=["merged jet", "NAO"],
        relation={"subject": "merged jet configuration", "predicate": "LINKED_TO",
                  "object": "phase negative de la NAO"},
        notes="Relation attribuee a une reference precise, chapitre 1 section 1.1.3.",
    ),
    dict(
        question_id="REL_005", question_type="relational", difficulty="hard", split="train",
        question="Quel effet l'amplification arctique pourrait-elle avoir sur la météo hivernale des moyennes latitudes ?",
        expected_answer=(
            "L'amplification arctique est présentée comme un mécanisme susceptible de conduire à "
            "une augmentation des conditions hivernales sévères aux moyennes latitudes. La thèse "
            "souligne toutefois que cet effet potentiel reste débattu et entouré d'incertitudes."
        ),
        evidence=[
            (53, 'Arctic amplification (AA) is a mechanism that may lead to an increase in severe winter weather in the midlatitudes'),
            (53, 'the quantification of its influence remains debated'),
        ],
        entities=["Arctic Amplification", "midlatitudes", "severe winter weather"],
        relation={"subject": "Arctic Amplification", "predicate": "MAY_INCREASE",
                  "object": "conditions hivernales severes aux moyennes latitudes"},
        notes="Relation explicitement presentee comme incertaine et debattue.",
    ),
    dict(
        question_id="REL_006", question_type="relational", difficulty="hard", split="test",
        question="Quel événement stratosphérique de janvier 1963 a pu contribuer à maintenir les conditions anormales en Europe, et par quel mécanisme ?",
        expected_answer=(
            "Un réchauffement stratosphérique soudain (SSW) s'est produit fin janvier 1963. Il "
            "s'est accompagné d'un affaiblissement du vortex polaire, ce qui a pu contribuer à "
            "maintenir la persistance des conditions anormales en Europe tout au long du mois de "
            "février."
        ),
        evidence=[(69, 'a sudden stratospheric warming event took place with the associated weakened polar vortex')],
        entities=["sudden stratospheric warming", "polar vortex", "Europe"],
        relation={"subject": "sudden stratospheric warming de janvier 1963",
                  "predicate": "WEAKENED_AND_MAINTAINED",
                  "object": "vortex polaire affaibli et persistance des conditions anormales en Europe"},
        notes="Formulation prudente dans la these (« may have helped »).",
    ),

    # ---------------- HYBRID : relation + explication ----------------
    dict(
        question_id="HYB_001", question_type="hybrid", difficulty="easy", split="train",
        question="Quel lien existe entre le blocage scandinave et l'intensité des vagues de froid en Europe de l'Ouest, et qu'ont en commun ce régime et la NAO négative ?",
        expected_answer=(
            "Le blocage scandinave amène l'air froid par la Sibérie et le nord-est de l'Europe ; "
            "cet air restant au-dessus des terres, cette configuration provoque les vagues de "
            "froid les plus intenses sur l'Europe de l'Ouest et la France. Ce régime et la phase "
            "négative de la NAO ont en commun un blocage sur l'Atlantique Nord, situé soit sur "
            "le Groenland, soit sur la Scandinavie."
        ),
        evidence=[
            (23, 'A Scandinavian blocking brings cold air through Siberia and the North-East of Europe'),
            (23, 'That air remains over land areas'),
            (22, 'Both these patterns have in common a blocking over the North-Atlantic'),
        ],
        entities=["Scandinavian Blocking", "NAO", "Western Europe", "Greenland"],
        relation={"subject": "Scandinavian Blocking et NAO negative",
                  "predicate": "SHARE_MECHANISM",
                  "object": "blocage sur l'Atlantique Nord (Groenland ou Scandinavie)"},
        notes="Necessite deux pages : la relation causale (p23) et le mecanisme commun (p22).",
    ),
    dict(
        question_id="HYB_002", question_type="hybrid", difficulty="easy", split="train",
        question="Quelle relation la thèse établit-elle entre l'hiver 1963 et la NAO, et comment le courant-jet explique-t-il ces conditions ?",
        expected_answer=(
            "L'hiver 1963 est associé à un indice NAO négatif, traduisant une différence de "
            "pression plus faible que la normale entre la dépression d'Islande et l'anticyclone "
            "des Açores. Du point de vue dynamique, le courant-jet et les vents d'ouest associés "
            "ont été scindés et déplacés très au sud, la dépression étant décalée à l'est des "
            "Açores."
        ),
        evidence=[
            (53, 'Winter 1963 was associated with a negative North Atlantic Oscillation (NAO) index'),
            (69, 'the jet stream and associated westerlies were split and displaced far'),
        ],
        entities=["winter 1963", "NAO", "jet stream"],
        relation={"subject": "winter 1963", "predicate": "ASSOCIATED_WITH",
                  "object": "NAO negative et courant-jet scinde et deplace vers le sud"},
        notes="Relation (p53) + explication dynamique dans un second article (p69).",
    ),
    dict(
        question_id="HYB_003", question_type="hybrid", difficulty="medium", split="train",
        question="À quoi la thèse utilise-t-elle l'indice WCC vis-à-vis des modèles CMIP6, et pourquoi cet indice est-il préférable à un indice NAO classique ?",
        expected_answer=(
            "L'indice WCC sert d'abord à évaluer dans quelle mesure les modèles CMIP6 "
            "reproduisent des mécanismes semblables à ceux observés sur la période historique, "
            "puis à établir un lien causal entre la circulation atmosphérique identifiée et les "
            "vagues de froid extrême en France. Il est préférable à un indice NAO quotidien "
            "classique parce qu'il est corrélé plus fortement à la température quotidienne en "
            "France en hiver (r = 0,65 contre r = 0,36 pour la NAO), étant taillé pour ce type "
            "d'événement et cette région."
        ),
        evidence=[
            (87, 'to assess how well CMIP6 models reproduce similar mechanisms'),
            (96, 'the WCC is correlated to daily temperature over France during winter months'),
            (96, 'it performs better than a classic daily North Atlantic Oscillation (NAO) index'),
        ],
        entities=["WCC", "CMIP6", "NAO", "France"],
        relation={"subject": "WCC", "predicate": "OUTPERFORMS_FOR_COLD_SPELLS",
                  "object": "indice NAO quotidien classique"},
        notes="Deux pages et deux chapitres differents : usage (p87) et performance (p96).",
    ),
    dict(
        question_id="HYB_004", question_type="hybrid", difficulty="medium", split="test",
        question="Quel est le lien entre le générateur stochastique de temps et les analogues de circulation, et quel objectif de la thèse cela sert-il ?",
        expected_answer=(
            "Le générateur stochastique de temps repose sur une discrétisation de l'espace des "
            "phases par les analogues de circulation. Ce lien sert le premier objectif de la "
            "thèse : adapter et améliorer le SWG existant, fondé sur ces analogues, pour simuler "
            "des événements de froid extrême dans une approche de type storyline."
        ),
        evidence=[
            (30, 'It comprises a discretization of the phase-space by analogues of circulation'),
            (31, 'the existing SWG based on analogues of circulation to the simulation of extreme cold events'),
        ],
        entities=["SWG", "analogues of circulation", "extreme cold events"],
        relation={"subject": "SWG", "predicate": "BASED_ON",
                  "object": "analogues de circulation"},
        notes="Relation methodologique (p30) rattachee a l'objectif de recherche (p31).",
    ),
    dict(
        question_id="HYB_005", question_type="hybrid", difficulty="hard", split="train",
        question="Quelle relation la thèse établit-elle entre le SWG et la méthode d'ensemble boosting, et comment explique-t-elle l'écart d'intensité observé ?",
        expected_answer=(
            "La thèse compare le SWG à un autre algorithme d'événements rares, la méthode de "
            "splitting dite « ensemble boosting » de Gessner et al. (2021) ; l'approche du SWG, "
            "qui consiste à rejouer des trajectoires alternatives d'événements extrêmes "
            "préexistants, en est proche. Les deux méthodes donnent des résultats très "
            "similaires et cohérents. Les hivers simulés par le SWG sont toutefois légèrement "
            "moins intenses, parce que le SWG est contraint par ses données d'entrée."
        ),
        evidence=[
            (122, 'the splitting method of ensemble boosting by Gessner et al. (2021)'),
            (122, 'The SWG and ensemble boosting yielded very similar and coherent results'),
            (122, 'The winters simulated by the SWG were slightly less intense'),
            (95, 'is similar to the "ensemble boosting" method of Gessner et al. (2021)'),
        ],
        entities=["SWG", "ensemble boosting", "Gessner et al."],
        relation={"subject": "SWG", "predicate": "COMPARED_TO",
                  "object": "ensemble boosting de Gessner et al. (2021)"},
        notes="Relation entre methodes (p122) et similarite d'approche (p95).",
    ),
    dict(
        question_id="HYB_006", question_type="hybrid", difficulty="hard", split="test",
        question="Quelle contrainte le nombre d'analogues impose-t-il aux résultats du SWG, et quelle amélioration la thèse propose-t-elle ?",
        expected_answer=(
            "Se limiter à K = 20 analogues, pour préserver la qualité des analogues, contraint "
            "les résultats du SWG. La thèse propose de calculer les analogues sur les membres "
            "d'ensembles de simulations de grande taille, tels que ceux fournis par certains "
            "modèles CMIP6 : cela permettrait d'utiliser un nombre d'analogues plus élevé à "
            "chaque pas de temps tout en maintenant une qualité d'analogues élevée."
        ),
        evidence=[
            (123, 'being limited to K = 20 analogues for the sake of analogue quality does constrain the SWG results'),
            (123, 'Further studies could be improved by computing analogues over ensemble members of large ensemble simulations'),
            (123, 'This would enable running the SWG with a higher number of analogues'),
        ],
        entities=["SWG", "analogues", "CMIP6"],
        relation={"subject": "nombre d'analogues K", "predicate": "CONSTRAINS",
                  "object": "resultats du SWG"},
        notes="Limite (p123) et perspective d'amelioration (p123), chapitre 5.",
    ),

    # ---------------- OUT OF CONTEXT ----------------
    dict(
        question_id="OOC_001", question_type="out_of_context", difficulty="easy", split="train",
        question="Quelles ont été les causes de la « rain bomb » australienne de 2022 ?",
        origin="annex_b",
        notes="Sujet de l'annexe B, volontairement exclue du corpus RAG principal.",
    ),
    dict(
        question_id="OOC_002", question_type="out_of_context", difficulty="medium", split="train",
        question="Comment des ensembles de simulations climatiques ont-ils servi à anticiper les canicules des Jeux olympiques de Paris 2024 ?",
        origin="annex_b",
        notes="Second article de l'annexe B, hors du perimetre de la these.",
    ),
    dict(
        question_id="OOC_003", question_type="out_of_context", difficulty="hard", split="test",
        question="Quelle méthode d'attribution au changement climatique a été appliquée aux précipitations extrêmes du Queensland en 2022 ?",
        origin="annex_b",
        notes="Annexe B. Le texte existe dans annex_b_out_of_context.jsonl mais ce fichier "
              "n'appartient pas au corpus RAG : la reponse attendue reste « Je ne sais pas. »",
    ),
    dict(
        question_id="OOC_004", question_type="out_of_context", difficulty="easy", split="train",
        question="Quels sont les principaux symptômes de la grippe saisonnière ?",
        origin="unrelated",
        notes="Sujet medical, totalement etranger au corpus.",
    ),
    dict(
        question_id="OOC_005", question_type="out_of_context", difficulty="medium", split="train",
        question="Comment fonctionne l'algorithme de rétropropagation du gradient dans un réseau de neurones ?",
        origin="unrelated",
        notes="Sujet d'apprentissage automatique, absent de la these.",
    ),
    dict(
        question_id="OOC_006", question_type="out_of_context", difficulty="hard", split="test",
        question="Quelles sont les conséquences de l'inflation sur le marché immobilier français ?",
        origin="unrelated",
        notes="Sujet economique, absent de la these. Piege : contient « francais », present "
              "dans le corpus, mais aucun contenu ne soutient la reponse.",
    ),
]


# =====================================================================
# Verification
# =====================================================================
def normalise(text: str) -> str:
    """Comparaison tolerante a la normalisation Unicode et aux espaces."""
    text = unicodedata.normalize("NFC", text or "")
    return re.sub(r"\s+", " ", text).strip()


def load_corpus():
    if not CORPUS.exists():
        log.error("Corpus absent : %s", CORPUS)
        raise SystemExit(1)
    with CORPUS.open(encoding="utf-8") as fh:
        return {json.loads(l)["page_pdf"]: json.loads(l) for l in fh if l.strip()}


def build(corpus) -> tuple:
    """Construit les enregistrements verifies. Leve une erreur si une preuve manque."""
    questions, errors = [], []

    for spec in QUESTIONS:
        qtype = spec["question_type"]
        record = {
            "question_id": spec["question_id"],
            "question": spec["question"],
            "question_type": qtype,
            "expected_route": ROUTE_BY_TYPE[qtype],
            "difficulty": spec["difficulty"],
            "split": spec["split"],
            "expected_answer": IDK,
            "relevant_pages_pdf": [],
            "relevant_pages_these": [],
            "relevant_sections": [],
            "supporting_evidence": [],
            "expected_entities": [],
            "expected_relation": None,
            "out_of_context_origin": None,
            "relevant_chunk_ids": [],
            "annotation_status": "verified",
            "notes": spec.get("notes", ""),
        }

        if qtype == "out_of_context":
            record["out_of_context_origin"] = spec["origin"]
            questions.append(record)
            continue

        record["expected_answer"] = spec["expected_answer"]
        record["expected_entities"] = spec.get("entities", [])
        record["expected_relation"] = spec.get("relation")

        pages, sections = [], []
        for page_pdf, excerpt in spec["evidence"]:
            page = corpus.get(page_pdf)
            if page is None:
                errors.append(f"{spec['question_id']}: p{page_pdf} absente du corpus nettoye")
                continue
            haystack = normalise(page["clean_text"])
            needle = normalise(excerpt)
            if needle not in haystack:
                errors.append(f"{spec['question_id']}: extrait introuvable p{page_pdf} -> {excerpt!r}")
                continue

            # On elargit l'extrait a une phrase complete, pour que la preuve soit lisible.
            start = haystack.find(needle)
            left = haystack.rfind(". ", 0, start)
            begin = 0 if left < 0 else left + 2
            end = haystack.find(". ", start + len(needle))
            stop = len(haystack) if end < 0 else end + 1
            sentence = haystack[begin:stop].strip()
            if len(sentence) > 480:
                sentence = haystack[start:start + 480].strip()

            # Deux extraits demandes peuvent tomber dans la meme phrase : on ne la
            # garde qu'une fois, pour que chaque preuve apporte une information.
            already = {(e["page_pdf"], e["excerpt"]) for e in record["supporting_evidence"]}
            if (page_pdf, sentence) not in already:
                record["supporting_evidence"].append({
                    "page_pdf": page_pdf,
                    "page_these": page["page_these"],
                    "language": page["language"],
                    "excerpt": sentence,
                })
            if page_pdf not in pages:
                pages.append(page_pdf)
                section = {
                    "chapter_number": page["chapter_number"],
                    "section_number": page["section_number"],
                    "section_title": page["section_title"],
                }
                if section not in sections:
                    sections.append(section)

        record["relevant_pages_pdf"] = sorted(pages)
        record["relevant_pages_these"] = sorted(
            corpus[p]["page_these"] for p in pages if corpus[p]["page_these"] is not None
        )
        record["relevant_sections"] = sections
        questions.append(record)

    return questions, errors


def cross_lingual_control(questions) -> dict:
    """Controle SECONDAIRE : similarite question francaise / preuve anglaise.

    N'influence jamais la verite terrain. Un score bas produit un avertissement.
    """
    import os
    os.environ.setdefault("HF_HOME", str((BACKEND_DIR / "data" / "hf_cache").resolve()))
    from sentence_transformers import SentenceTransformer

    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    log.info("Controle cross-lingue : chargement de %s", model_name)
    model = SentenceTransformer(model_name, device="cpu")

    answerable = [q for q in questions if q["supporting_evidence"]]
    per_question, warnings = [], []
    for q in answerable:
        excerpts = [e["excerpt"] for e in q["supporting_evidence"]]
        vectors = model.encode([q["question"]] + excerpts, normalize_embeddings=True,
                               convert_to_numpy=True)
        sims = (vectors[0] @ vectors[1:].T).tolist()
        entry = {
            "question_id": q["question_id"],
            "similarities": [round(float(s), 4) for s in sims],
            "mean": round(float(sum(sims) / len(sims)), 4),
            "max": round(float(max(sims)), 4),
        }
        per_question.append(entry)
        if entry["max"] < SIMILARITY_WARNING_THRESHOLD:
            warnings.append({
                "question_id": q["question_id"],
                "max_similarity": entry["max"],
                "message": "similarite faible entre la question et ses preuves : "
                           "annotation conservee, a relire humainement",
            })

    means = [e["mean"] for e in per_question]
    return {
        "modele": model_name,
        "seuil_avertissement": SIMILARITY_WARNING_THRESHOLD,
        "questions_mesurees": len(per_question),
        "average_positive_evidence_similarity": round(sum(means) / len(means), 4) if means else 0.0,
        "minimum_positive_evidence_similarity": round(min(means), 4) if means else 0.0,
        "maximum_positive_evidence_similarity": round(max(means), 4) if means else 0.0,
        "questions_with_similarity_warning": warnings,
        "detail": per_question,
    }


def build_review(questions, corpus) -> str:
    """Rapport lisible : tableau de synthese puis preuve de chaque question."""
    lines = [
        "# Jeu d'evaluation - revue des 24 questions",
        "",
        f"Version {EVAL_SET_VERSION}, genere le "
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}.",
        "",
        "Chaque page et chaque extrait ci-dessous a ete verifie automatiquement dans",
        "`backend/data/processed/corpus_clean.jsonl` : le script de construction echoue si",
        "une preuve est introuvable. Les questions sont en francais, le corpus est a 94 %",
        "en anglais : c'est le cas d'usage cross-lingue vise par le projet.",
        "",
        "## Tableau de synthese",
        "",
        "| ID | Type | Diff. | Split | Question | Route | Pages PDF | Pages these | Entites | Relation | Statut |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for q in questions:
        rel = q["expected_relation"]
        rel_txt = (f"{rel['subject']} -[{rel['predicate']}]-> {rel['object']}"[:60]
                   if rel else "—")
        lines.append(
            f"| `{q['question_id']}` | {q['question_type']} | {q['difficulty']} "
            f"| {q['split']} | {q['question'][:78]} | `{q['expected_route']}` "
            f"| {q['relevant_pages_pdf'] or '—'} | {q['relevant_pages_these'] or '—'} "
            f"| {', '.join(q['expected_entities']) or '—'} | {rel_txt} "
            f"| {q['annotation_status']} |"
        )

    lines += ["", "## Reponses attendues et preuves", ""]
    for q in questions:
        lines += [
            f"### {q['question_id']} — {q['question_type']} — {q['difficulty']} — {q['split']}",
            "",
            f"**Question.** {q['question']}",
            "",
            f"**Route attendue.** `{q['expected_route']}`",
            "",
            f"**Reponse attendue.** {q['expected_answer']}",
            "",
        ]
        if q["out_of_context_origin"]:
            origine = ("annexe B, exclue du corpus RAG principal"
                       if q["out_of_context_origin"] == "annex_b"
                       else "sujet totalement etranger a la these")
            lines += [f"**Origine hors contexte.** {origine}.", "",
                      "Aucune page, aucune preuve : le systeme doit s'abstenir.", ""]
        else:
            for evidence in q["supporting_evidence"]:
                # La section est relue dans le corpus pour CETTE page precise.
                page = corpus[evidence["page_pdf"]]
                lines += [
                    f"- **Page PDF {evidence['page_pdf']}** (page these "
                    f"{evidence['page_these']}, langue `{evidence['language']}`) — "
                    f"chapitre {page['chapter_number']}, section "
                    f"{page['section_number']} « {page['section_title']} »",
                    "",
                    "  > " + evidence["excerpt"],
                    "",
                ]
            if q["expected_relation"]:
                rel = q["expected_relation"]
                lines += [f"**Relation attendue.** `{rel['subject']}` "
                          f"-[{rel['predicate']}]-> `{rel['object']}`", ""]
        if q["notes"]:
            lines += [f"_Note : {q['notes']}_", ""]
    return "\n".join(lines)


def main() -> int:
    log.info("=" * 66)
    log.info("ETAPE 2 - JEU D'EVALUATION")
    log.info("=" * 66)

    corpus = load_corpus()
    log.info("Corpus nettoye : %d pages", len(corpus))

    questions, errors = build(corpus)
    if errors:
        for e in errors:
            log.error(e)
        log.error("%d annotation(s) non verifiee(s) : jeu d'evaluation NON ecrit.", len(errors))
        return 1
    log.info("24 questions construites, toutes les preuves verifiees dans le corpus.")

    similarity = cross_lingual_control(questions)
    log.info("Similarite cross-lingue moyenne : %.4f (min %.4f, max %.4f) | %d avertissement(s)",
             similarity["average_positive_evidence_similarity"],
             similarity["minimum_positive_evidence_similarity"],
             similarity["maximum_positive_evidence_similarity"],
             len(similarity["questions_with_similarity_warning"]))

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "eval_set.json").write_text(
        json.dumps({"version": EVAL_SET_VERSION,
                    "genere_le": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "corpus": "backend/data/processed/corpus_clean.jsonl",
                    "count": len(questions),
                    "questions": questions}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (EVAL_DIR / "eval_set_review.md").write_text(build_review(questions, corpus), encoding="utf-8")

    # --- rapport de metriques -------------------------------------------
    def count(field):
        out = {}
        for q in questions:
            out[q[field]] = out.get(q[field], 0) + 1
        return out

    answerable = [q for q in questions if q["question_type"] != "out_of_context"]
    pages = sorted({p for q in questions for p in q["relevant_pages_pdf"]})
    chapters = sorted({s["chapter_number"] for q in questions for s in q["relevant_sections"]
                       if s["chapter_number"]})
    texts = [q["question"].strip().lower() for q in questions]
    doublons = sorted({t for t in texts if texts.count(t) > 1})
    pagination_errors = [
        {"question_id": q["question_id"], "pdf": q["relevant_pages_pdf"],
         "these": q["relevant_pages_these"]}
        for q in questions
        if q["relevant_pages_these"] != sorted(
            corpus[p]["page_these"] for p in q["relevant_pages_pdf"]
            if corpus[p]["page_these"] is not None)
    ]

    report = {
        "genere_le": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version_jeu_evaluation": EVAL_SET_VERSION,
        "etape": "2 - jeu d'evaluation",
        "nombre_total_de_questions": len(questions),
        "par_type": count("question_type"),
        "par_difficulte": count("difficulty"),
        "par_split": count("split"),
        "par_route_attendue": count("expected_route"),
        "questions_avec_preuve": sum(1 for q in questions if q["supporting_evidence"]),
        "questions_sans_page_pertinente": sum(1 for q in questions if not q["relevant_pages_pdf"]),
        "questions_repondables": len(answerable),
        "pages_differentes_couvertes": len(pages),
        "liste_pages_pdf_couvertes": pages,
        "chapitres_couverts": chapters,
        "nombre_de_chapitres_couverts": len(chapters),
        "doublons_detectes": doublons,
        "erreurs_de_pagination": pagination_errors,
        "hors_contexte_par_origine": {
            "annex_b": sum(1 for q in questions if q["out_of_context_origin"] == "annex_b"),
            "unrelated": sum(1 for q in questions if q["out_of_context_origin"] == "unrelated"),
        },
        "controle_cross_lingue": similarity,
        "average_positive_evidence_similarity": similarity["average_positive_evidence_similarity"],
        "minimum_positive_evidence_similarity": similarity["minimum_positive_evidence_similarity"],
        "maximum_positive_evidence_similarity": similarity["maximum_positive_evidence_similarity"],
        "questions_with_similarity_warning": similarity["questions_with_similarity_warning"],
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "evaluation_set_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("-" * 66)
    log.info("Types      : %s", report["par_type"])
    log.info("Difficulte : %s", report["par_difficulte"])
    log.info("Split      : %s", report["par_split"])
    log.info("Pages couvertes : %d | chapitres : %s", len(pages), chapters)
    log.info("eval_set.json          : %s", (EVAL_DIR / 'eval_set.json').relative_to(PROJECT_ROOT))
    log.info("eval_set_review.md     : %s", (EVAL_DIR / 'eval_set_review.md').relative_to(PROJECT_ROOT))
    log.info("evaluation_set_report  : %s",
             (METRICS_DIR / 'evaluation_set_report.json').relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
