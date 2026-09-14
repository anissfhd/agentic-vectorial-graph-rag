# Agentic Vectorial Graph RAG with Reinforcement Learning

> A specialized document assistant that answers questions about a 206-page scientific thesis — and **decides for itself** whether to answer from a vector store, a knowledge graph, both, or to say "I don't know".

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Neo4j](https://img.shields.io/badge/Neo4j%20Aura-008CC1?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![FAISS](https://img.shields.io/badge/FAISS-0467DF)](https://faiss.ai/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

🇫🇷 [Lire ce document en français](README.fr.md)

---

## What this project is

Final-year project (4th year, AI specialization). The corpus is a single document: *A statistical approach to the study of the intensity and dynamics of extreme cold spells in Europe* — Camille Cadiou's PhD thesis (Université Paris-Saclay / LSCE, 2025, 206 pages).

**This is not a weather forecasting model.** It is a retrieval system built to demonstrate and arbitrate between three RAG paradigms on the same corpus:

| Paradigm | What it does here |
|---|---|
| **Vectorial RAG** | 7 chunking methods and 7 vector representations actually benchmarked, then FAISS + BM25 + RRF fusion |
| **Graph RAG** | Scientific entities and relations extracted into a knowledge graph, persisted in Neo4j Aura, analyzed with NetworkX + Louvain |
| **Agentic RAG + RL** | A Q-Learning agent orchestrated by LangGraph routes each question to the right store — or abstains |

The guiding constraint of the whole project: **no invented metric, no hard-coded number.** Every figure below comes from a real run over the thesis.

<p align="center">
  <img src="report/assets/interface_query.png" alt="Query tab showing the full agentic pipeline with routing, confidence, sources and page provenance" width="900">
</p>

---

## Key results

The system ingests 206 pages without OCR and keeps 132 of them. The rest — inserted journal articles, appendix B, front matter — is excluded on purpose and reused as a source of out-of-scope questions.

| Ingestion | Value |
|---|---|
| PDF pages / kept / excluded | 206 / 132 / 74 |
| Useful characters, words | 289,941 and 45,769 |
| Acronyms extracted | 25 |
| Page-boundary repairs | 7 |
| OCR needed | No |

### Chunking — 7 methods compared

`paragraph_based` wins on overall score and becomes the production method, which matches a corpus written as argumentative paragraphs.

| Method | Chunks | F1@5 | Recall@5 | Intra-sim | Overall |
|---|---|---|---|---|---|
| fixed_size | 416 | 0.266 | 0.528 | 0.417 | 0.622 |
| sentence_based | 413 | 0.234 | 0.472 | 0.394 | 0.302 |
| **paragraph_based** | **680** | **0.282** | 0.556 | 0.393 | **0.693** |
| sliding_window | 478 | 0.278 | 0.528 | 0.384 | 0.573 |
| recursive | 504 | 0.237 | 0.500 | **0.429** | 0.492 |
| semantic | 456 | 0.256 | 0.556 | 0.396 | 0.400 |
| document_structure | 299 | 0.234 | **0.611** | 0.411 | 0.329 |

`document_structure` has the best Recall@5, but its overall score is penalized on the other criteria.

### Embeddings — 7 representations compared

The real problem is **cross-lingual retrieval**: questions are asked in French, roughly 90% of the manuscript is in English. That is what disqualifies lexical methods as the main model.

| Representation | Dim. | F1@5 | MAP | MRR | NDCG@5 | Time (s) |
|---|---|---|---|---|---|---|
| tfidf_word | 7,901 | 0.051 | 0.061 | 0.136 | 0.135 | 0.12 |
| tfidf_char_ngrams | 22,176 | 0.126 | 0.087 | 0.282 | 0.289 | 0.34 |
| hashing_vectorizer | 8,192 | 0.032 | 0.037 | 0.080 | 0.077 | 0.05 |
| tfidf_svd_128 | 128 | 0.037 | 0.059 | 0.132 | 0.111 | 0.43 |
| tfidf_svd_256 | 256 | 0.037 | 0.059 | 0.134 | 0.111 | 0.91 |
| **multilingual_minilm** | **384** | **0.282** | **0.133** | **0.402** | **0.500** | 24.21 |
| hybrid_tfidf_minilm | 512 | 0.159 | 0.118 | 0.282 | 0.327 | 25.59 |

MiniLM is the slowest to encode, but that cost is paid once at build time — a query only re-encodes the question.

<p align="center">
  <img src="report/assets/chart_chunking.png" alt="Bar chart of F1@5 and overall score per chunking method" width="46%">
  <img src="report/assets/chart_embeddings.png" alt="Bar chart of MAP, MRR and NDCG@5 per vector representation" width="46%">
</p>

### Knowledge graph

| Metric | Value |
|---|---|
| Nodes, relationships | 66 and 171 |
| Density | 0.0788 |
| Louvain communities | 16 |
| Modularity | 0.2812 |
| Relation types | 5 main |

Most central entities by degree centrality: `Europe` (0.477), `SWG` (0.338), `France` (0.308), `Atmospheric circulation` (0.277), `Winter 1963` (0.262), `CMIP6` (0.231).

<p align="center">
  <img src="report/assets/interface_graph.png" alt="Graph RAG tab with interactive graph, Louvain communities and structural metrics" width="900">
</p>

The graph itself is committed — [`backend/data/artifacts/graph.json`](backend/data/artifacts/graph.json), `graph.cypher`, `graph.graphml` — along with a real Aura export at [`data/exports/neo4j_graph_export.csv`](data/exports/neo4j_graph_export.csv). Every relation carries its page, section, chunk id and source excerpt.

### Retrieval methods — and a result that did not go as planned

| Method | F1@5 | MAP | MRR | NDCG@5 | Latency (ms) |
|---|---|---|---|---|---|
| **cosine_exact** | **0.282** | **0.133** | **0.402** | **0.500** | ~0 |
| faiss_indexflatip | 0.282 | 0.133 | 0.402 | 0.500 | ~0 |
| hybrid_rrf | 0.183 | 0.113 | 0.280 | 0.342 | 0.35 |
| bm25 | 0.032 | 0.042 | 0.059 | 0.063 | 1.70 |

Two things worth reading carefully:

- **FAISS `IndexFlatIP` matches exact cosine to six decimals.** That is the expected result for a flat index, and it is the check that proves the index is wired correctly rather than silently returning something else.
- **RRF fusion made retrieval worse, not better.** Hybrid scores 0.183 F1@5 against 0.282 for pure vector search. BM25 is so weak on this corpus — 0.032 — that fusing it drags good vector results down the ranking. The cause is the cross-lingual gap: questions are French, the passages are English, and a lexical matcher has almost nothing to match on. Hybrid retrieval is a good default, but **not on a corpus where one of the two retrievers is near-blind**. The measurement is kept and reported rather than dropped.

Raw numbers: [`backend/data/metrics/`](backend/data/metrics/).

### Q-Learning routing

16 states, 4 actions (`use_vector`, `use_graph`, `use_hybrid`, `abstain`), epsilon decaying from 0.30 to 0.05.

| State | Vector | Graph | Hybrid | Abstain | Chosen |
|---|---|---|---|---|---|
| `semantic / entity=0 / coverage=high` | **2.350** | -0.234 | -0.315 | -0.252 | `use_vector` |
| `relational / entity=1 / coverage=high` | -0.194 | **2.350** | -0.497 | -0.242 | `use_graph` |
| `hybrid / entity=1 / coverage=high` | -0.308 | -0.400 | **2.050** | -0.484 | `use_hybrid` |
| `unknown / entity=0 / coverage=low` | neg. | neg. | neg. | **max** | `abstain` |

On the 8 annotated test questions, routing accuracy and correct abstention both reach 100%. **That figure must be read within the scope of a small test set** — it is not a universal guarantee over every possible phrasing.

<p align="center">
  <img src="report/assets/interface_agentic.png" alt="Agentic RAG tab with Q-table, reward monitor and routing tests" width="900">
</p>

---

## Design decisions

| Topic | Decision | Why |
|---|---|---|
| Final synthesis | **Extractive**, no generative LLM | Removes hallucination risk and API cost; every sentence returned exists in the thesis |
| Main embedding model | `paraphrase-multilingual-MiniLM-L12-v2` | The corpus is English, the questions are French — cross-lingual retrieval is mandatory |
| `all-MiniLM-L6-v2` | Baseline only | English-only, unfit as the main model but useful as a comparison point |
| Graph backend | NetworkX **and** Neo4j Aura | NetworkX computes and acts as fallback; Aura is the deliverable proving real persistence |
| Appendix B (p. 150-173) | Excluded from both stores | Reused as a source of genuinely out-of-scope questions |
| Evaluation set | 24 questions: 6 semantic, 6 relational, 6 hybrid, 6 out-of-scope | Covers all four routing branches |
| Reproducibility | Seed 42 everywhere | |
| Secrets | `.env` only, never in code | |

### Why abstention matters

Without the `abstain` action the agent would be forced to pick a retriever even for a question about backpropagation or influenza. A weakly similar passage would then be turned into a misleading answer. Abstention is a **safety function**, not a performance gap.

---

## Architecture

```
Question (FR)
     |
     v
+-----------------+   classify: semantic / relational / hybrid / unknown
|   LangGraph     |
|  orchestrator   |   +----------------------------------------+
+--------+--------+   | Q-Learning policy (Q-table, eps-greedy) |
         |            +----------------------------------------+
         |-- use_vector --> FAISS IndexFlatIP + BM25 --> RRF fusion
         |-- use_graph  --> Cypher on Neo4j Aura --> subgraph
         |-- use_hybrid --> both, merged
         +-- abstain    --> "Je ne sais pas"
                                  |
                                  v
                    Extractive answer + provenance
                (PDF page, thesis page, section, chunk id, score)
```

The thesis pagination is offset by 15 pages from the PDF pagination, so **both numbers are stored and displayed** — otherwise a citation cannot be verified against the printed document.

<p align="center">
  <img src="report/assets/uml_architecture.png" alt="System architecture diagram" width="820">
</p>

### API surface

| Method | Route | Role |
|---|---|---|
| `POST` | `/query` | Full agentic pipeline |
| `GET` | `/vectorial` | Vector RAG summary |
| `GET` | `/vectorial/chunking` | Metrics for the 7 chunking methods |
| `GET` | `/vectorial/embeddings` | Metrics for the 7 representations |
| `POST` | `/vectorial/retrieve` | FAISS, BM25 and RRF side by side |
| `GET` | `/graph` | Nodes, edges and statistics |
| `GET` | `/graph/communities` | Louvain communities and members |
| `POST` | `/graph/subgraph` | Cypher query and resulting subgraph |
| `GET` | `/agentic` | Q-table and rewards |
| `GET` | `/query/history` | Session history |

---

## Repository contents

```
.
├── run_mvp.bat                   one command: artifacts check, API, frontend, browser
├── rebuild_artifacts.bat         regenerate everything from the PDF
├── backend/
│   ├── app/
│   │   ├── main.py               FastAPI application, 10 routes
│   │   └── core/
│   │       ├── ingestion/        pdf_reader · cleaner · structure · language · quality
│   │       └── mvp/              chunking · embeddings · graph · agent · workflow
│   │                             · neo4j_store · runtime · common
│   ├── scripts/
│   │   ├── 01_ingest_pdf.py      206 pages to corpus_clean.jsonl
│   │   ├── 02_build_eval_set.py  the 24-question evaluation set
│   │   ├── fast_build_mvp.py     artifact check and rebuild
│   │   ├── push_to_neo4j.py      graph sync to Aura
│   │   ├── 00_env_check.py       imports + cross-lingual embedding test
│   │   ├── check_neo4j.py        read-only Neo4j diagnostic
│   │   ├── audit_env.ps1         Windows system audit
│   │   └── setup_env.bat         conda environment creation
│   ├── tests/                    ingestion · mvp · evaluation set · environment
│   ├── eval/eval_set.json        24 annotated questions + review notes
│   └── data/
│       ├── artifacts/            faiss.index · graph.{json,cypher,graphml}
│       │                         q_table.json · reward_history.json · pca_2d.json
│       └── metrics/              every measured result, as produced by the pipeline
├── frontend/                     React + Vite, 4 tabs
├── data/exports/
│   └── neo4j_graph_export.csv    real Aura export: entities, relations, provenance
├── docs/
│   ├── ARCHITECTURE.md           consolidated design document, written before coding
│   ├── SETUP.fr.md               step-by-step environment setup
│   └── assets/                   interface captures, charts, UML diagrams
└── report/
    ├── Rapport_PFA_*.tex         full LaTeX report
    ├── assets/                   20 figures
    └── make_diagrams.py          diagram generation script
```

### What is deliberately **not** committed

The corpus is a third-party doctoral thesis and is never redistributed — neither the PDF nor anything that would reproduce its full text:

| Excluded | Why |
|---|---|
| `backend/data/raw/*.pdf` | The thesis itself (47 MB) |
| `backend/data/processed/*` | Cleaned corpus and all 7 chunk sets — the complete text |
| `backend/data/artifacts/bm25_data.json` | The full tokenised corpus (680 chunks) |
| `documentation/specifications/` | Supervisor-provided reference PDFs |
| `backend/data/hf_cache/` | 916 MB HuggingFace model cache |

Place your own copy of the thesis at `backend/data/raw/these_vagues_froid.pdf`, then run `rebuild_artifacts.bat` to regenerate what is missing.

**The FAISS index, the knowledge graph and the Q-table are committed**, so the demo runs without rebuilding anything — only BM25 needs the corpus back.

---

## Getting started

```bash
# 1. Python environment
conda create -y -n pfa-rag python=3.11
conda activate pfa-rag
pip install -r backend/requirements.txt      # or requirements.lock.txt for exact pins

# 2. Configuration
cp backend/.env.example backend/.env         # then fill it in

# 3. Checks — exits 0 only if cross-lingual FR to EN retrieval is validated
python backend/scripts/00_env_check.py
python backend/scripts/check_neo4j.py
pytest backend/tests -v

# 4. Frontend
cd frontend && npm install && cd ..
```

Then, on Windows, a single command starts everything — artifact check, FastAPI, Vite, browser:

```bat
run_mvp.bat
```

Or manually:

```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
cd frontend && npm run dev
```

To rebuild every artifact from the thesis PDF (ingestion, 7 chunkings, 7 embeddings, FAISS, graph, Q-Learning):

```bat
rebuild_artifacts.bat
```

| Variable | Role | Example |
|---|---|---|
| `NEO4J_URI` | Aura or local Bolt endpoint | `neo4j+s://xxxx.databases.neo4j.io` |
| `NEO4J_USER` | Database user | `neo4j` |
| `NEO4J_PASSWORD` | Password — **shown once at instance creation** | |
| `HF_HOME` | Hugging Face cache, 5 GB free minimum | `D:\hf_cache` |
| `API_HOST` / `API_PORT` | FastAPI binding | `127.0.0.1` / `8000` |

> Free Aura instances pause after a few days of inactivity. **Wake the instance the day before a demo**, never the morning of.

Once running: frontend on `:5173`, API on `:8000`, Swagger on `:8000/docs`.

Full step-by-step setup: [`docs/SETUP.fr.md`](docs/SETUP.fr.md). Full design rationale: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Status and limitations

**The full implementation is here** — ingestion, the seven chunkings, the seven embeddings, FAISS, the knowledge graph, Q-Learning, the FastAPI backend and the React interface, plus the tests and every artifact the numbers above come from. Roughly 6,200 lines of Python, 978 lines of tests, and a React frontend.

Every figure in this README can be traced back to a committed file in `backend/data/metrics/` — none of them were typed in by hand.

Known limitations, as stated in the report:

- **Small evaluation set.** 24 questions, one annotator. Routing percentages must always be presented with that context.
- **Extractive synthesis** avoids hallucination but reads less naturally than generated prose.
- **Title-only chunks** can surface in the top-k (for example `paragraph_based_00045`); filtering or down-weighting them is a known improvement.
- **No cross-encoder reranker** — it would likely improve top-k at the cost of latency.
- The graph carries no temporal or bibliographic relations yet.
- **RRF fusion currently hurts.** See the retrieval table above — it is kept in the codebase and reported, not hidden, but vector-only is the better route on this corpus.
- **BM25 cannot run from a fresh clone**: its artifact holds the full corpus and is excluded for copyright reasons. Everything else runs as committed.

---

## Credits

Corpus: Camille Cadiou, *Une approche statistique pour l'étude de l'intensité et de la dynamique des vagues de froid extrêmes en Europe*, PhD thesis, Université Paris-Saclay (LSCE), defended 2 April 2025. Used here as a study corpus only, and not redistributed.

## License

[MIT](LICENSE)
