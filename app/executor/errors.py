"""
AstraExec — Taxonomie des erreurs
==================================

Catégories d'exceptions du moteur d'action (extension production).

L'Executor n'est JAMAIS censé propager ces exceptions à l'appelant :
il les capture et les convertit en objet de réponse
``{"status": "error", "execution_time": ..., "message": ...}``.

La hiérarchie permet :
- un traitement catégorisé (timeout, réseau, outil indisponible, ...) ;
- des stratégies de retry ciblées (voir RetryPolicy) ;
- des journaux d'audit structurés avec le type d'erreur exact.
"""


class ActionError(Exception):
    """
    Erreur de base du moteur d'action.

    Toutes les erreurs « métier » du moteur en héritent.
    """


class ToolTimeoutError(ActionError):
    """L'outil a dépassé son délai d'exécution configuré."""


class ToolNetworkError(ActionError):
    """Erreur réseau lors de l'exécution d'un outil (appel externe)."""


class ToolUnavailableError(ActionError):
    """L'outil est indisponible (circuit breaker ouvert, dépendance absente)."""


class PermissionDeniedError(ActionError):
    """Les permissions accordées ne couvrent pas l'outil demandé."""


class BudgetExhaustedError(ActionError):
    """Le budget d'actions maximum du moteur est épuisé."""


class InvalidOutputError(ActionError):
    """La sortie de l'outil ne respecte pas son output_schema."""


class InvalidSchemaError(ActionError):
    """Les paramètres de l'action violent le schéma de l'outil."""


class PlanValidationError(ActionError):
    """Un plan multi-étapes est invalide (structure, cycle, dépendance)."""
