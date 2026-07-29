# Note de cadrage

## 1. Contexte

Le projet consiste à développer un module d'action intelligent pour un agent RAG.

Ce module reçoit un plan d'exécution provenant du module de raisonnement et exécute les différentes actions demandées.

---

## 2. Problématique

Les systèmes RAG classiques disposent souvent d'un moteur d'exécution peu flexible.

L'objectif est de proposer un module capable d'orchestrer plusieurs outils, de sécuriser les actions et d'améliorer la qualité des résultats.

---

## 3. Objectifs

### Objectif général

Développer un module d'action modulaire, sécurisé et extensible.

### Objectifs spécifiques

- Exécuter les plans générés par le module Reasoning.
- Développer des outils personnalisés de recherche.
- Gérer plusieurs types d'actions.
- Assurer la sécurité des exécutions.
- Produire des journaux d'exécution.

---

## 4. Périmètre

Le projet couvre :

- Gestion des outils
- Recherche documentaire
- Exécution des actions
- Journalisation
- Validation des entrées

Le projet ne couvre pas :

- Génération des réponses
- Raisonnement de l'agent

---

## 5. Architecture

Le module est composé de :

- API
- Executor
- Retrieval
- Registry
- Guardrails
- Telemetry

---

## 6. Technologies

- Python
- FastAPI
- Pydantic
- FAISS
- Rank-BM25
- SentenceTransformers
- Git

---

## 7. Livrables

S1
- Note de cadrage
- Interface JSON

S2
- Développement des outils de recherche

S3
- Développement de l'exécution

---

## 8. Planning

Semaine 1
Conception

Semaine 2
Développement Retrieval

Semaine 3
Développement Executor

---

## 9. Conclusion

Le module AstraExec constitue la couche d'exécution de l'agent RAG. Il a été conçu pour être modulaire, sécurisé et facilement extensible.