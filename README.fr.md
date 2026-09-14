# Agentic Vectorial Graph RAG avec apprentissage par renforcement

> Un assistant documentaire spécialisé qui répond à des questions sur une thèse scientifique de 206 pages — et qui **décide lui-même** s'il doit répondre depuis un index vectoriel, un graphe de connaissances, les deux, ou dire « Je ne sais pas ».

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Neo4j](https://img.shields.io/badge/Neo4j%20Aura-008CC1?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![FAISS](https://img.shields.io/badge/FAISS-0467DF)](https://faiss.ai/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C)](https://langchain-ai.github.io/langgraph/)
[![Licence : MIT](https://img.shields.io/badge/Licence-MIT-yellow.svg)](LICENSE)

🇬🇧 [Read this document in English](README.md)

---

## De quoi il s'agit

Projet de fin d'année — 4ème année, spécialité Intelligence Artificielle. Le corpus est un document unique : *Une approche statistique pour l'étude de l'intensité et de la dynamique des vagues de froid extrêmes en Europe*, thèse de doctorat de Camille Cadiou (Université Paris-Saclay / LSCE, 2025, 206 pages).

**Ce n'est pas un modèle de prévision météorologique.** C'est un système de recherche documentaire construit pour démontrer et arbitrer trois paradigmes RAG sur un même corpus :

| Paradigme | Ce qu'il fait ici |
|---|---|
| **Vectorial RAG** | 7 méthodes de chunking et 7 représentations vectorielles réellement comparées, puis FAISS + BM25 + fusion RRF |
| **Graph RAG** | Entités et relations scientifiques extraites en graphe de connaissances, persistées dans Neo4j Aura, analysées avec NetworkX + Louvain |
| **Agentic RAG + RL** | Un agent Q-Learning orchestré par LangGraph route chaque question vers le bon store — ou s'abstient |

La règle qui traverse tout le projet : **aucune métrique inventée, aucune valeur codée en dur.** Tous les chiffres ci-dessous proviennent d'une exécution réelle sur la thèse.

<p align="center">
  <img src="report/assets/interface_query.png" alt="Onglet Query montrant le pipeline agentique complet avec routage, confiance, sources et provenance des pages" width="900">
</p>

---

## Résultats

Le système ingère les 206 pages sans OCR et en conserve 132. Le reste — articles de revue insérés, annexe B, pages liminaires — est exclu volontairement et réutilisé comme source de questions hors contexte.

| Ingestion | Valeur |
|---|---|
| Pages PDF / retenues / exclues | 206 / 132 / 74 |
| Caractères et mots utiles | 289 941 et 45 769 |
| Acronymes extraits | 25 |
| Réparations de frontières de page | 7 |
| OCR nécessaire | Non |

### Chunking — 7 méthodes comparées

`paragraph_based` l'emporte au score global et devient la méthode de production, ce qui est cohérent avec un corpus rédigé en paragraphes argumentatifs.

| Méthode | Chunks | F1@5 | Recall@5 | Intra | Score global |
|---|---|---|---|---|---|
| fixed_size | 416 | 0,266 | 0,528 | 0,417 | 0,622 |
| sentence_based | 413 | 0,234 | 0,472 | 0,394 | 0,302 |
| **paragraph_based** | **680** | **0,282** | 0,556 | 0,393 | **0,693** |
| sliding_window | 478 | 0,278 | 0,528 | 0,384 | 0,573 |
| recursive | 504 | 0,237 | 0,500 | **0,429** | 0,492 |
| semantic | 456 | 0,256 | 0,556 | 0,396 | 0,400 |
| document_structure | 299 | 0,234 | **0,611** | 0,411 | 0,329 |

`document_structure` obtient le meilleur Recall@5, mais son score global est pénalisé par les autres critères.

### Embeddings — 7 représentations comparées

Le vrai enjeu est le **retrieval cross-lingue** : les questions sont posées en français, le manuscrit est à ~90 % en anglais. C'est ce qui disqualifie les méthodes lexicales comme modèle principal.

| Représentation | Dim. | F1@5 | MAP | MRR | NDCG@5 | Temps (s) |
|---|---|---|---|---|---|---|
| tfidf_word | 7 901 | 0,051 | 0,061 | 0,136 | 0,135 | 0,12 |
| tfidf_char_ngrams | 22 176 | 0,126 | 0,087 | 0,282 | 0,289 | 0,34 |
| hashing_vectorizer | 8 192 | 0,032 | 0,037 | 0,080 | 0,077 | 0,05 |
| tfidf_svd_128 | 128 | 0,037 | 0,059 | 0,132 | 0,111 | 0,43 |
| tfidf_svd_256 | 256 | 0,037 | 0,059 | 0,134 | 0,111 | 0,91 |
| **multilingual_minilm** | **384** | **0,282** | **0,133** | **0,402** | **0,500** | 24,21 |
| hybrid_tfidf_minilm | 512 | 0,159 | 0,118 | 0,282 | 0,327 | 25,59 |

MiniLM est le plus lent à encoder, mais ce coût n'est payé qu'une fois à la construction — une requête ne réencode que la question.

<p align="center">
  <img src="report/assets/chart_chunking.png" alt="Comparaison de F1@5 et du score global par méthode de chunking" width="46%">
  <img src="report/assets/chart_embeddings.png" alt="Comparaison de MAP, MRR et NDCG@5 par représentation vectorielle" width="46%">
</p>

### Graphe de connaissances

| Métrique | Valeur |
|---|---|
| Nœuds et relations | 66 et 171 |
| Densité | 0,0788 |
| Communautés Louvain | 16 |
| Modularité | 0,2812 |
| Types de relations | 5 principaux |

Entités les plus centrales (degree centrality) : `Europe` (0,477), `SWG` (0,338), `France` (0,308), `Atmospheric circulation` (0,277), `Winter 1963` (0,262), `CMIP6` (0,231).

<p align="center">
  <img src="report/assets/interface_graph.png" alt="Onglet Graph RAG avec graphe interactif, communautés Louvain et métriques structurelles" width="900">
</p>

Le graphe lui-même est versionné — [`backend/data/artifacts/graph.json`](backend/data/artifacts/graph.json), `graph.cypher`, `graph.graphml` — ainsi qu'un export Aura réel dans [`data/exports/neo4j_graph_export.csv`](data/exports/neo4j_graph_export.csv). Chaque relation porte sa page, sa section, son identifiant de chunk et l'extrait source.

### Méthodes de retrieval — et un résultat qui ne s'est pas passé comme prévu

| Méthode | F1@5 | MAP | MRR | NDCG@5 | Latence (ms) |
|---|---|---|---|---|---|
| **cosine_exact** | **0,282** | **0,133** | **0,402** | **0,500** | ~0 |
| faiss_indexflatip | 0,282 | 0,133 | 0,402 | 0,500 | ~0 |
| hybrid_rrf | 0,183 | 0,113 | 0,280 | 0,342 | 0,35 |
| bm25 | 0,032 | 0,042 | 0,059 | 0,063 | 1,70 |

Deux points à lire attentivement :

- **FAISS `IndexFlatIP` égale le cosinus exact à six décimales.** C'est le résultat attendu pour un index plat, et c'est le contrôle qui prouve que l'index est correctement câblé plutôt que de renvoyer silencieusement autre chose.
- **La fusion RRF a dégradé le retrieval, pas amélioré.** L'hybride obtient 0,183 de F1@5 contre 0,282 pour le vectoriel seul. BM25 est si faible sur ce corpus — 0,032 — que le fusionner fait descendre les bons résultats vectoriels dans le classement. La cause est l'écart cross-lingue : les questions sont en français, les passages en anglais, et un matcher lexical n'a presque rien à apparier. Le retrieval hybride est un bon réglage par défaut, mais **pas sur un corpus où l'un des deux retrievers est quasi aveugle**. La mesure est conservée et rapportée plutôt qu'écartée.

Chiffres bruts : [`backend/data/metrics/`](backend/data/metrics/).

### Routage par Q-Learning

16 états, 4 actions (`use_vector`, `use_graph`, `use_hybrid`, `abstain`), epsilon décroissant de 0,30 à 0,05.

| État | Vector | Graph | Hybrid | Abstain | Action |
|---|---|---|---|---|---|
| `semantic / entity=0 / coverage=high` | **2,350** | -0,234 | -0,315 | -0,252 | `use_vector` |
| `relational / entity=1 / coverage=high` | -0,194 | **2,350** | -0,497 | -0,242 | `use_graph` |
| `hybrid / entity=1 / coverage=high` | -0,308 | -0,400 | **2,050** | -0,484 | `use_hybrid` |
| `unknown / entity=0 / coverage=low` | négatif | négatif | négatif | **maximum** | `abstain` |

Sur les 8 questions de test annotées, le routage et l'abstention correcte atteignent tous deux 100 %. **Ce chiffre doit être lu dans le périmètre d'un petit jeu de test** — ce n'est pas une garantie universelle sur toute formulation possible.

<p align="center">
  <img src="report/assets/interface_agentic.png" alt="Onglet Agentic RAG avec Q-table, reward monitor et tests de routage" width="900">
</p>

---

## Décisions de conception

| Sujet | Décision | Pourquoi |
|---|---|---|
| Synthèse finale | **Extractive**, aucun LLM génératif | Supprime le risque d'hallucination et le coût d'API ; toute phrase renvoyée existe dans la thèse |
| Modèle d'embedding principal | `paraphrase-multilingual-MiniLM-L12-v2` | Le corpus est en anglais, les questions en français — le retrieval cross-lingue est obligatoire |
| `all-MiniLM-L6-v2` | Baseline uniquement | Anglais seul, inadapté comme modèle principal mais utile comme point de comparaison |
| Backend graphe | NetworkX **et** Neo4j Aura | NetworkX calcule et sert de secours ; Aura est le livrable qui prouve la persistance réelle |
| Annexe B (p. 150-173) | Exclue des deux stores | Réutilisée comme source de questions réellement hors contexte |
| Jeu d'évaluation | 24 questions : 6 sémantiques, 6 relationnelles, 6 hybrides, 6 hors contexte | Couvre les quatre branches de routage |
| Reproductibilité | Seed 42 partout | |
| Secrets | Dans `.env` uniquement, jamais dans le code | |

### Pourquoi l'abstention compte

Sans l'action `abstain`, l'agent serait obligé de choisir un retriever même pour une question sur la rétropropagation ou la grippe. Un passage faiblement similaire serait alors transformé en réponse trompeuse. L'abstention est une **fonction de sûreté**, pas une absence de performance.

---

## Architecture

```
Question (FR)
     |
     v
+-----------------+   classification : sémantique / relationnelle / hybride / inconnue
|   LangGraph     |
|  orchestrateur  |   +------------------------------------------+
+--------+--------+   | Politique Q-Learning (Q-table, eps-greedy)|
         |            +------------------------------------------+
         |-- use_vector --> FAISS IndexFlatIP + BM25 --> fusion RRF
         |-- use_graph  --> Cypher sur Neo4j Aura --> sous-graphe
         |-- use_hybrid --> les deux, fusionnés
         +-- abstain    --> « Je ne sais pas »
                                  |
                                  v
                     Réponse extractive + provenance
              (page PDF, page thèse, section, id de chunk, score)
```

La pagination de la thèse est décalée de 15 pages par rapport à celle du PDF : **les deux numéros sont stockés et affichés**, sinon une citation ne peut pas être vérifiée sur le document imprimé.

<p align="center">
  <img src="report/assets/uml_architecture.png" alt="Diagramme d'architecture du système" width="820">
</p>

### Surface de l'API

| Méthode | Route | Rôle |
|---|---|---|
| `POST` | `/query` | Pipeline agentique complet |
| `GET` | `/vectorial` | Résumé Vector RAG |
| `GET` | `/vectorial/chunking` | Métriques des 7 chunkings |
| `GET` | `/vectorial/embeddings` | Métriques des 7 représentations |
| `POST` | `/vectorial/retrieve` | FAISS, BM25 et RRF côte à côte |
| `GET` | `/graph` | Nœuds, arêtes et statistiques |
| `GET` | `/graph/communities` | Communautés Louvain et membres |
| `POST` | `/graph/subgraph` | Requête Cypher et sous-graphe |
| `GET` | `/agentic` | Q-table et rewards |
| `GET` | `/query/history` | Historique de session |

---

## Contenu du dépôt

```
.
├── run_mvp.bat                   une commande : contrôle des artefacts, API, frontend, navigateur
├── rebuild_artifacts.bat         régénère tout depuis le PDF
├── backend/
│   ├── app/
│   │   ├── main.py               application FastAPI, 10 routes
│   │   └── core/
│   │       ├── ingestion/        pdf_reader · cleaner · structure · language · quality
│   │       └── mvp/              chunking · embeddings · graph · agent · workflow
│   │                             · neo4j_store · runtime · common
│   ├── scripts/
│   │   ├── 01_ingest_pdf.py      206 pages vers corpus_clean.jsonl
│   │   ├── 02_build_eval_set.py  le jeu d'évaluation de 24 questions
│   │   ├── fast_build_mvp.py     contrôle et reconstruction des artefacts
│   │   ├── push_to_neo4j.py      synchronisation du graphe vers Aura
│   │   ├── 00_env_check.py       imports + test d'embedding cross-lingue
│   │   ├── check_neo4j.py        diagnostic Neo4j sans écriture
│   │   ├── audit_env.ps1         audit système Windows
│   │   └── setup_env.bat         création de l'environnement conda
│   ├── tests/                    ingestion · mvp · jeu d'évaluation · environnement
│   ├── eval/eval_set.json        24 questions annotées + notes de relecture
│   └── data/
│       ├── artifacts/            faiss.index · graph.{json,cypher,graphml}
│       │                         q_table.json · reward_history.json · pca_2d.json
│       └── metrics/              tous les résultats mesurés, tels que produits par le pipeline
├── frontend/                     React + Vite, 4 onglets
├── data/exports/
│   └── neo4j_graph_export.csv    export Aura réel : entités, relations, provenance
├── docs/
│   ├── ARCHITECTURE.md           document de conception consolidé, rédigé avant tout codage
│   ├── SETUP.fr.md               mise en route pas à pas
│   └── assets/                   captures d'interface, graphiques, diagrammes UML
└── report/
    ├── Rapport_PFA_*.tex         rapport LaTeX complet
    ├── assets/                   20 figures
    └── make_diagrams.py          script de génération des diagrammes
```

### Ce qui est délibérément **exclu** du dépôt

Le corpus est une thèse de doctorat tierce et n'est jamais redistribué — ni le PDF, ni rien qui en reproduirait le texte intégral :

| Exclu | Pourquoi |
|---|---|
| `backend/data/raw/*.pdf` | La thèse elle-même (47 Mo) |
| `backend/data/processed/*` | Corpus nettoyé et les 7 jeux de chunks — le texte complet |
| `backend/data/artifacts/bm25_data.json` | Le corpus intégral tokenisé (680 chunks) |
| `documentation/specifications/` | PDF de référence fournis par l'encadrement |
| `backend/data/hf_cache/` | 916 Mo de cache de modèles HuggingFace |

Placez votre propre copie de la thèse dans `backend/data/raw/these_vagues_froid.pdf`, puis lancez `rebuild_artifacts.bat` pour régénérer ce qui manque.

**L'index FAISS, le graphe de connaissances et la Q-table sont versionnés**, donc la démo tourne sans rien reconstruire — seul BM25 nécessite de récupérer le corpus.

---

## Mise en route

```bash
# 1. Environnement Python
conda create -y -n pfa-rag python=3.11
conda activate pfa-rag
pip install -r backend/requirements.txt      # ou requirements.lock.txt pour les versions exactes

# 2. Configuration
cp backend/.env.example backend/.env         # puis le remplir

# 3. Contrôles — ne retourne 0 que si le retrieval cross-lingue FR vers EN est validé
python backend/scripts/00_env_check.py
python backend/scripts/check_neo4j.py
pytest backend/tests -v

# 4. Frontend
cd frontend && npm install && cd ..
```

Ensuite, sous Windows, une seule commande lance tout — contrôle des artefacts, FastAPI, Vite, navigateur :

```bat
run_mvp.bat
```

Ou manuellement :

```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
cd frontend && npm run dev
```

Pour reconstruire tous les artefacts depuis le PDF de la thèse (ingestion, 7 chunkings, 7 embeddings, FAISS, graphe, Q-Learning) :

```bat
rebuild_artifacts.bat
```

| Variable | Rôle | Exemple |
|---|---|---|
| `NEO4J_URI` | Endpoint Aura ou Bolt local | `neo4j+s://xxxx.databases.neo4j.io` |
| `NEO4J_USER` | Utilisateur de la base | `neo4j` |
| `NEO4J_PASSWORD` | Mot de passe — **affiché une seule fois à la création de l'instance** | |
| `HF_HOME` | Cache Hugging Face, 5 Go libres minimum | `D:\hf_cache` |
| `API_HOST` / `API_PORT` | Binding FastAPI | `127.0.0.1` / `8000` |

> Les instances Aura gratuites se mettent en pause après quelques jours d'inactivité. **Réveiller l'instance la veille d'une démonstration**, jamais le matin même.

Une fois lancé : frontend sur `:5173`, API sur `:8000`, Swagger sur `:8000/docs`.

Mise en route détaillée : [`docs/SETUP.fr.md`](docs/SETUP.fr.md). Justification complète des choix : [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## État et limites

**L'implémentation complète est ici** — ingestion, les sept chunkings, les sept embeddings, FAISS, le graphe de connaissances, le Q-Learning, le backend FastAPI et l'interface React, plus les tests et tous les artefacts dont proviennent les chiffres ci-dessus. Environ 6 200 lignes de Python, 978 lignes de tests, et un frontend React.

Chaque chiffre de ce README est traçable jusqu'à un fichier versionné dans `backend/data/metrics/` — aucun n'a été saisi à la main.

Limites connues, telles qu'énoncées dans le rapport :

- **Jeu d'évaluation restreint.** 24 questions, un seul annotateur. Les pourcentages de routage doivent toujours être présentés avec ce contexte.
- **La synthèse extractive** évite l'hallucination mais produit un style moins naturel qu'une génération.
- **Les chunks réduits à un titre** peuvent apparaître dans le top-k (par exemple `paragraph_based_00045`) ; les filtrer ou les sous-pondérer est une amélioration identifiée.
- **Pas de reranker cross-encoder** — il améliorerait probablement le top-k au prix de la latence.
- Le graphe ne porte pas encore de relations temporelles ni bibliographiques.
- **La fusion RRF dégrade actuellement les résultats.** Voir le tableau de retrieval plus haut — elle est conservée dans le code et rapportée, pas masquée, mais le vectoriel seul est la meilleure route sur ce corpus.
- **BM25 ne peut pas tourner depuis un clone neuf** : son artefact contient le corpus complet et est exclu pour des raisons de droit d'auteur. Tout le reste fonctionne en l'état.

---

## Crédits

Corpus : Camille Cadiou, *Une approche statistique pour l'étude de l'intensité et de la dynamique des vagues de froid extrêmes en Europe*, thèse de doctorat, Université Paris-Saclay (LSCE), soutenue le 2 avril 2025. Utilisée ici uniquement comme corpus d'étude, et non redistribuée.

## Licence

[MIT](LICENSE)
