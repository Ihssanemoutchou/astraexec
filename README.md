# AstraExec

## Description

AstraExec est un module d'action intelligent conçu pour un agent RAG.

Son rôle est d'exécuter les plans générés par le module de raisonnement, d'orchestrer les outils disponibles et de retourner des résultats fiables.

## Objectifs

- Exécuter les actions demandées par le module Reasoning.
- Récupérer des informations à partir de différentes sources.
- Assurer la sécurité des actions exécutées.
- Produire des journaux d'exécution.

## Architecture

Le module est composé des sous-modules suivants :

- API
- Executor
- Retrieval
- Registry
- Guardrails
- Telemetry
- Storage (Livrable 4 : base documentaire ChromaDB)

## Livrable 4 — Base Documentaire

Le Livrable 4 ajoute la génération et la livraison de la base documentaire
vectorielle (ChromaDB) au module Raisonnement (binôme).

- `app/storage/embedding_generator.py` — vectorisation `all-MiniLM-L6-v2` (384 dims)
- `app/storage/chroma_manager.py` — unique accès à ChromaDB (collection `astra_docs`, cosine)
- `app/storage/base_export.py` — archive ZIP brute de la base
- `demo_database.py` — génération complète : `python demo_database.py`

La base livrée (`exports/base_documentaire_v1.zip`) se dézippe directement :
le binôme ouvre `chromadb.PersistentClient(path="storage/chroma")` puis
interroge avec le même modèle `all-MiniLM-L6-v2` (version `chromadb==1.5.9`
requise).

Documentation complète : [`docs/livrable4.md`](docs/livrable4.md).

## Technologies

- Python 3.11
- FastAPI
- NumPy
- Pickle
- Pydantic
- Pytest
- ChromaDB (moteur de stockage de la base documentaire — Livrable 4)
- Sentence Transformers (modèle d'embedding `all-MiniLM-L6-v2` — Livrable 4)


## Auteur

Ihssane MOUTCHOU - EMSI
