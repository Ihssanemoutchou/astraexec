<div align="center">

# 🚀 AstraExec

### Intelligent Action Module for Agentic RAG Systems

Execution Engine • Hybrid Retrieval • ChromaDB • Ethical Filtering • Evaluation Framework

---

Projet réalisé dans le cadre d'un **Stage Ingénieur** à l'**EMSI**.

</div>

---

# 📖 Table des matières

- Présentation
- Fonctionnalités
- Architecture
- Workflow
- Structure du projet
- Technologies
- Installation
- Utilisation
- API REST
- Recherche hybride
- Base documentaire
- Évaluation
- Sécurité
- Livrables
- Perspectives

---

# 🎯 Présentation

AstraExec est un **Action Module** destiné aux architectures **Agentic RAG**.

Contrairement à un moteur de recherche documentaire classique, AstraExec est responsable de :

- l'exécution des actions,
- l'orchestration des outils,
- la validation des requêtes,
- la récupération des connaissances,
- la sécurisation de l'exécution,
- l'évaluation des performances.

Le module reçoit un plan d'action provenant du **Reasoning Module**, sélectionne les outils adaptés puis retourne une réponse fiable.

---

# ✨ Fonctionnalités

## ⚙️ Action Engine

- Tool Registry
- Action Executor
- Validation des actions
- Routage intelligent

---

## 🔎 Recherche Hybride

- Recherche lexicale (LexiRank)
- Recherche vectorielle
- Fusion des résultats
- EvidenceRank

---

## 📚 Gestion documentaire

- Import TXT
- Import PDF
- SmartSeg
- Génération automatique des chunks
- Base documentaire ChromaDB

---

## 🛡 Sécurité

- Validation des paramètres
- Ethical Filter
- Détection de Prompt Injection
- Journalisation complète

---

## 📊 Évaluation

Campagne complète d'évaluation :

- Recall@5
- Recall@10
- Mean Reciprocal Rank
- Robustesse
- Sécurité
- Performance

---

# 🏗 Architecture

```text
Utilisateur

        │

        ▼

    FastAPI

        │

        ▼

    Validator

        │

        ▼

 Ethical Filter

        │

        ▼

    Executor

        │

        ▼

 Tool Registry

        │

        ▼

 Hybrid Search

   ├── LexiRank
   ├── Vector Search
   └── EvidenceRank

        │

        ▼

   ChromaDB

        │

        ▼

Réponse
```

---

# 🔄 Workflow

1. Réception d'une requête.
2. Validation.
3. Filtrage éthique.
4. Sélection automatique du bon outil.
5. Recherche hybride.
6. Réordonnancement des résultats.
7. Génération de la réponse.
8. Journalisation.

---

# 📂 Structure du projet

```text
astra-exec/

app/
│
├── api/
├── executor/
├── retrieval/
├── registry/
├── storage/
├── guardrails/
├── telemetry/
├── evaluation/
├── security/
└── utils/

docs/

logs/

exports/
```

---

# ⚙️ Technologies

| Technologie | Utilisation |
|-------------|-------------|
| Python 3.11 | Langage principal |
| FastAPI | API REST |
| ChromaDB | Base documentaire vectorielle |
| Sentence Transformers | Embeddings |
| all-MiniLM-L6-v2 | Modèle d'embedding |
| NumPy | Calcul scientifique |
| Pydantic | Validation |
| Pytest | Tests |

---

# 🚀 Installation

```bash
git clone https://github.com/...

cd astra-exec

python -m venv .venv

pip install -r requirements.txt
```

---

# ▶ Utilisation

Lancer l'API

```bash
uvicorn app.api.main:app --reload
```

Créer la base documentaire

```bash
python demo_database.py
```

Lancer la campagne d'évaluation

```bash
python demo_evaluation.py
```

---

# 📡 API REST

Principaux endpoints :

```
POST /execute

POST /documents

GET /health
```

---

# 🔎 Recherche hybride

Le moteur combine :

- recherche lexicale,
- recherche vectorielle,
- fusion des résultats,
- reranking via EvidenceRank.

Cette approche améliore la pertinence des réponses par rapport à une recherche purement lexicale ou vectorielle.

---

# 📚 Base documentaire

Le Livrable 4 introduit une base documentaire vectorielle reposant sur ChromaDB.

Elle permet :

- l'import de documents TXT et PDF ;
- la génération automatique des embeddings ;
- la persistance des données ;
- l'export de la base pour le module Reasoning.

---

# 📊 Campagne d'évaluation

Le projet intègre une campagne complète d'évaluation mesurant :

- Recall@5
- Recall@10
- Mean Reciprocal Rank (MRR)

Ces métriques permettent d'évaluer la qualité du moteur de recherche hybride.

---

# 🛡 Sécurité

AstraExec applique plusieurs mécanismes de sécurité :

- validation des entrées ;
- filtre éthique ;
- protection contre les Prompt Injections ;
- journalisation des actions.

---

# 📦 Livrables

| Livrable | Description |
|----------|-------------|
| Livrable 1 | Architecture |
| Livrable 2 | Action Engine |
| Livrable 3 | Recherche hybride |
| Livrable 4 | Base documentaire ChromaDB |
| Livrable 5 | Campagne d'évaluation |
| Livrable 6 | Sécurité, filtres et finalisation |

---

# 🔮 Perspectives

- Ajout de nouveaux outils.
- Extension du moteur d'exécution.
- Optimisation des performances.
- Support de nouveaux modèles d'embedding.
- Monitoring avancé.

---

# 👨‍💻 Auteur

**Ihssane Moutchou**

Stage Ingénieur — EMSI

Encadrante : **Pr. Zineb Hidila**

2026
