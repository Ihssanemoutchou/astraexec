# 🚀 AstraExec — Intelligent Action Module for RAG Agents

<p align="center">

**Custom Action Module for Retrieval-Augmented Generation (RAG) Agents**

Python • FastAPI • ChromaDB • SentenceTransformers • Hybrid Retrieval • BM25 • Security • Evaluation

</p>

---

# 📖 Overview

AstraExec est un **module d'action intelligent** développé dans le cadre d'un stage d'ingénieur à l'EMSI.

Il constitue la partie **Action** d'un agent RAG et est responsable de l'exécution sécurisée des actions générées par le **Module Reasoning**.

Contrairement à un moteur de recherche classique, AstraExec orchestre plusieurs composants afin de :

- exécuter des outils spécialisés ;
- effectuer une recherche hybride (sémantique + lexicale) ;
- protéger le système contre les attaques de Prompt Injection ;
- filtrer les requêtes sensibles ;
- produire des résultats fiables et interprétables ;
- fournir une base documentaire vectorielle réutilisable par le Module Reasoning.

---

# 🎯 Objectifs du projet

Le projet répond aux objectifs suivants :

- Développer un moteur d'exécution d'actions pour un agent RAG.
- Concevoir des outils personnalisés inspirés des techniques de recherche modernes.
- Fournir une base documentaire vectorielle exploitable par le Module Reasoning.
- Garantir la sécurité des interactions grâce à plusieurs niveaux de validation.
- Évaluer objectivement les performances du système.

---

# ✨ Fonctionnalités principales

## ⚙️ Moteur d'exécution

- Exécution d'actions
- Validation des paramètres
- Gestion des erreurs
- Journalisation des exécutions

---

## 🔎 Recherche hybride personnalisée

AstraExec implémente plusieurs outils développés spécifiquement pour ce projet :

### FusionSearch

Recherche hybride combinant :

- recherche vectorielle
- recherche lexicale BM25

---

### LexiRank

Recherche lexicale personnalisée inspirée de BM25.

Fonctions :

- normalisation
- tokenisation
- indexation
- calcul lexical

---

### EvidenceRank

Système de re-ranking utilisant :

- score sémantique
- score lexical
- qualité du document
- position du chunk

---

### SmartSeg

Segmentation intelligente des documents :

- TXT
- PDF

avec gestion :

- overlap
- taille minimale
- métadonnées

---

## 🗄 Base documentaire (Livrable 4)

Le module génère automatiquement une base documentaire vectorielle.

Fonctionnalités :

- génération des embeddings
- stockage ChromaDB
- export ZIP
- livraison au Module Reasoning

Composants :

- EmbeddingGenerator
- ChromaManager
- BaseExport

---

## 🛡 Sécurité

Le projet intègre plusieurs couches de sécurité :

### Validator

Validation des actions.

### InjectionGuard

Détection des Prompt Injections.

### EthicalFilter

Blocage des requêtes dangereuses :

- prompt injection
- bypass
- hidden instructions
- contenus malveillants

---

## 📊 Campagne d'évaluation (Livrable 5)

Le projet contient une campagne complète d'évaluation.

### Recherche

- Recall@5
- Recall@10
- Mean Reciprocal Rank

### Sécurité

- Prompt Injection
- Validation
- Robustesse

### Performance

- Temps moyen
- Débit
- Charge

---

# 🏗 Architecture

```
                    Module Reasoning
                           │
                           ▼
                   AstraExec API
                           │
          ┌────────────────────────────────┐
          │           Executor             │
          └────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   Tool Registry      Guardrails        Telemetry
        │                  │
        ▼                  ▼
   FusionSearch      Ethical Filter
        │
        ▼
 Hybrid Retrieval
        │
        ▼
 EvidenceRank
        │
        ▼
 ChromaDB Storage
```

---

# 📂 Structure du projet

```
astra-exec/

app/
│
├── api/
├── executor/
├── registry/
├── retrieval/
├── storage/
├── guardrails/
├── telemetry/
├── schemas/
├── utils/
│
docs/
tests/
logs/
exports/

demo.py
demo_database.py
demo_evaluation.py
```

---

# 🔄 Workflow

```
Question utilisateur

↓

Module Reasoning

↓

Action JSON

↓

Executor

↓

Validator

↓

Ethical Filter

↓

FusionSearch

↓

LexiRank

↓

Recherche vectorielle

↓

EvidenceRank

↓

Résultat final

↓

Module Reasoning
```

---

# 🗄 Base documentaire

La base documentaire est construite automatiquement.

Pipeline :

Documents

↓

SmartSeg

↓

Embeddings

↓

ChromaDB

↓

Export ZIP

↓

Module Reasoning

---

# 📡 API

## Health

GET

```
/health
```

---

## Execute

POST

```
/execute
```

Exemple :

```json
{
  "tool": "fusion_search",
  "parameters": {
    "query": "machine learning"
  }
}
```

---

# 📈 Évaluation

Les campagnes d'évaluation couvrent :

## Recherche

- Recall@5
- Recall@10
- MRR

## Robustesse

- erreurs
- récupération

## Sécurité

- Prompt Injection
- Ethical Filter

## Performance

- temps moyen
- benchmark

---

# 📚 Documentation

Le projet contient une documentation complète :

```
docs/

livrable4.md

livrable5.md

contrat_integration.md

campagne_evaluation.md
```

---

# 🚀 Installation

Créer un environnement virtuel

```bash
python -m venv .venv
```

Activation

Windows

```bash
.\.venv\Scripts\activate
```

Installation

```bash
pip install -r requirements.txt
```

Lancer l'API

```bash
uvicorn app.api.main:app --reload
```

---

# ▶ Démonstrations

Exécution principale

```bash
python demo.py
```

Base documentaire

```bash
python demo_database.py
```

Campagne d'évaluation

```bash
python demo_evaluation.py
```

---

# 🛠 Technologies

| Technologie | Utilisation |
|-------------|------------|
| Python 3.11 | Langage principal |
| FastAPI | API REST |
| ChromaDB | Base documentaire |
| SentenceTransformers | Embeddings |
| NumPy | Calcul scientifique |
| PyMuPDF | Lecture PDF |
| Pytest | Tests |
| BM25 | Recherche lexicale |
| FAISS (prototype) | Recherche vectorielle |
| Git | Versionning |

---

# 🤝 Intégration avec le Module Reasoning

AstraExec fournit au Module Reasoning :

- Base documentaire ChromaDB
- Contrat d'intégration
- Documentation technique
- API REST
- Outils disponibles
- Schéma des actions
- Guide de fusion

La communication entre les deux modules repose sur un contrat d'intégration garantissant une fusion sans conflit.

---

# 🗺 Roadmap

- ✅ Livrable 1 — Architecture
- ✅ Livrable 2 — Executor
- ✅ Livrable 3 — Recherche Hybride
- ✅ Livrable 4 — Base documentaire ChromaDB
- ✅ Livrable 5 — Sécurité & Évaluation
- ✅ Livrable 6 — Intégration complète

---

# 👨‍💻 Auteur

**Ihssane MOUTCHOU**

Stage Ingénieur — EMSI

Module Action pour Agent RAG

2026
