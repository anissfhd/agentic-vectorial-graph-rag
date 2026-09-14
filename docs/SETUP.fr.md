> **Note** — Ce document a été rédigé à l'étape 0 du projet, avant l'implémentation.
> Il reste utile pour la mise en place détaillée de l'environnement et de Neo4j Aura,
> mais la phrase « aucun algorithme n'est encore implémenté » n'est plus vraie :
> le pipeline complet est dans `backend/app/`. Voir le README pour la marche à suivre actuelle.

---

# Agentic Vectorial Graph RAG with Reinforcement Learning

Projet de fin d'année — 4ème année Spécialité Intelligence Artificielle.
Corpus : *Une approche statistique pour l'étude de l'intensité et de la dynamique des vagues de froid extrêmes en Europe* (Camille Cadiou, Université Paris-Saclay, 206 pages).

> **État actuel : étape 0 — squelette et environnement.**
> Aucun algorithme n'est encore implémenté : ni ingestion, ni chunking, ni embeddings, ni graphe, ni Q-Learning, ni API.

---

## 1. Mise en route (dans cet ordre)

### 1.1 Audit système
```powershell
powershell -ExecutionPolicy Bypass -File backend\scripts\audit_env.ps1
```
Écrit `backend\data\metrics\audit_systeme.txt`. **Contrôler avant tout : Python 3.11 disponible, Node.js présent, ports 7474/7687, espace disque ≥ 10 Go, accès à `huggingface.co`.**

### 1.2 Placer le corpus
```powershell
Copy-Item "C:\chemin\vers\01_DATA_These_Vagues_Froid.pdf" "backend\data\raw\these_vagues_froid.pdf"
```
Le PDF original ne doit jamais être modifié.

### 1.3 Environnement Python dédié
Depuis un **Anaconda Prompt** :
```bat
backend\scripts\setup_env.bat
```
Équivalent manuel :
```bat
conda create -y -n pfa-rag python=3.11
conda activate pfa-rag
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
pip freeze > backend\requirements.lock.txt
```

### 1.4 Configurer les variables d'environnement
```powershell
Copy-Item backend\.env.example backend\.env
```
Puis remplir `backend\.env`. **Ce fichier n'est jamais versionné.**

### 1.5 Contrôle technique
```bat
conda activate pfa-rag
python backend\scripts\00_env_check.py
pytest backend\tests -v
```
Le premier script vérifie les 7 imports critiques, télécharge le modèle multilingue léger, encode trois phrases de contrôle et écrit `backend\data\metrics\environment_embedding_test.json`. Il retourne le code 0 uniquement si le retrieval cross-lingue FR→EN est validé.

### 1.6 Diagnostic Neo4j
```bat
python backend\scripts\check_neo4j.py
```

---

## 2. Décisions de conception figées

| Sujet | Décision |
|---|---|
| Périmètre | MVP strict |
| Synthèse finale | **Extractive** — aucun LLM génératif local installé |
| Langue | Questions et réponses en français, corpus à ~90 % en anglais → **retrieval cross-lingue obligatoire** |
| Modèle principal | `paraphrase-multilingual-MiniLM-L12-v2` |
| Graphe | NetworkX pour les calculs et le mode de secours ; **Neo4j réellement alimenté** = livrable obligatoire |
| Annexe B (p. 150-173) | **Exclue** du Vector Store et du Knowledge Graph ; sert de source de questions hors contexte |
| Jeu d'évaluation | 24 questions : 6 sémantiques, 6 relationnelles, 6 hybrides, 6 hors contexte |
| Reproductibilité | Seed fixe = 42 partout |
| Secrets | Exclusivement dans `.env`, jamais dans le code |

---

## 3. Neo4j — chemin le plus rapide

Neo4j **doit réellement être alimenté** (nœuds, relations, Cypher exécutée, sous-graphe affiché, captures vérifiables). Générer un fichier `.cql` sans insertion ne suffit pas.

**Option recommandée pour une démonstration imminente — Neo4j Aura Free** (≈ 5 min, aucune installation) :
1. Créer un compte sur `console.neo4j.io` → *Create instance* → *AuraDB Free*
2. **Télécharger le fichier d'identifiants au moment de la création** — le mot de passe n'est affiché qu'une seule fois
3. Reporter `NEO4J_URI` (`neo4j+s://xxxx.databases.neo4j.io`), `NEO4J_USER`, `NEO4J_PASSWORD` dans `backend\.env`
4. Attendre le statut *Running* (2 à 3 min), puis `python backend\scripts\check_neo4j.py`

> Les instances gratuites se mettent en pause après plusieurs jours d'inactivité : **réveiller l'instance la veille de la soutenance**, jamais le matin même.

**Option de repli — Neo4j Desktop** (fonctionne sans Internet, ~500 Mo) : installer depuis `neo4j.com/download`, créer un projet puis une base locale, définir le mot de passe, démarrer, et utiliser `NEO4J_URI=bolt://localhost:7687`, `NEO4J_USER=neo4j`.

Le driver Python `neo4j` sera ajouté aux dépendances lors de l'étape « Graph Store », pas avant.

---

## 4. Arborescence

```
pfa-agentic-graph-rag/
├── README.md
├── .gitignore
├── documentation/              9 PDF de référence + document d'architecture
├── frontend/                   React (étape ultérieure)
└── backend/
    ├── requirements.txt        phase 1 uniquement
    ├── .env.example            noms de variables, sans aucun secret
    ├── app/                    code applicatif (vide à ce stade)
    ├── scripts/
    │   ├── audit_env.ps1       audit système Windows
    │   ├── setup_env.bat       création de l'environnement conda
    │   ├── 00_env_check.py     imports + test d'embedding cross-lingue
    │   └── check_neo4j.py      diagnostic Neo4j sans écriture
    ├── tests/test_env.py       tests d'environnement
    ├── eval/                   jeu d'évaluation (24 questions, à venir)
    └── data/
        ├── raw/                these_vagues_froid.pdf (non versionné)
        ├── processed/          corpus nettoyé, chunks
        ├── artifacts/          index FAISS, embeddings, graphe, Q-table
        └── metrics/            métriques calculées (versionnées : preuves)
```

---

## 5. Étape suivante

Étape 1 — ingestion du PDF : extraction en ordre de lecture (obligatoire pour les pages en double colonne des chapitres 3 et 4), nettoyage des en-têtes et césures, rattachement chapitre/section, exclusions, écriture de `corpus_clean.jsonl`.
