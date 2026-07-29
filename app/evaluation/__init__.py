"""
AstraExec — Module d'évaluation
================================

Métriques d'évaluation maison pour la recherche d'information.

Composants :
  - recall_at_k      : Rappel au rang K
  - mean_reciprocal_rank : MRR
  - Evaluator        : Évaluation multi-requêtes

Aucune bibliothèque externe d'évaluation IR utilisée.
"""

from app.evaluation.metrics import recall_at_k, mean_reciprocal_rank, Evaluator

__all__ = ["recall_at_k", "mean_reciprocal_rank", "Evaluator"]
