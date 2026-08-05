"""
AstraExec — Résilience (extension production)
==============================================

Deux primitives de résilience réutilisables :

- RetryPolicy   : stratégie de nouvelle tentative avec backoff exponentiel.
- CircuitBreaker: coupe-circuit par outil (closed → open → half_open).

Ces primitives sont **injectables** dans l'Executor via son constructeur :
elles sont désactivées par défaut, donc le comportement historique du
moteur reste strictement identique si elles ne sont pas fournies.
"""

import threading
import time
from typing import Optional, Tuple, Type

from app.executor.errors import (
    ToolNetworkError,
    ToolTimeoutError,
    ToolUnavailableError,
)


class RetryPolicy:
    """
    Stratégie de nouvelle tentative avec backoff exponentiel.

    Délai après la tentative n : base_delay * multiplier ** n,
    plafonné à max_delay (évite les attentes déraisonnables).

    Seules les exceptions listées dans `retryable` déclenchent une
    nouvelle tentative (par défaut : timeout, réseau, outil indisponible).
    Une erreur de validation, d'injection ou de permissions n'est
    JAMAIS retentée.
    """

    DEFAULT_RETRYABLE: Tuple[Type[Exception], ...] = (
        ToolTimeoutError,
        ToolNetworkError,
        ToolUnavailableError,
    )

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 0.1,
        max_delay: float = 2.0,
        multiplier: float = 2.0,
        retryable: Optional[Tuple[Type[Exception], ...]] = None,
    ):
        if max_retries < 0:
            raise ValueError("max_retries doit être >= 0.")
        if base_delay < 0 or max_delay < 0 or multiplier <= 0:
            raise ValueError("Délais invalides (base_delay, max_delay >= 0, multiplier > 0).")

        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.retryable = (
            tuple(retryable) if retryable is not None else self.DEFAULT_RETRYABLE
        )

    def delay_for(self, attempt: int) -> float:
        """
        Délai d'attente avant la tentative `attempt` (0 = première retry).

        Backoff exponentiel plafonné :
            min(base_delay * multiplier ** attempt, max_delay)
        """
        attempt = max(0, attempt)
        return min(
            self.base_delay * (self.multiplier ** attempt),
            self.max_delay,
        )

    def should_retry(self, error: Exception) -> bool:
        """True si l'exception appartient aux types retentables."""
        return isinstance(error, self.retryable)

    def info(self) -> dict:
        return {
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
            "multiplier": self.multiplier,
            "retryable": [e.__name__ for e in self.retryable],
        }


class CircuitBreaker:
    """
    Coupe-circuit par outil.

    États :
      closed    : tout passe, les échecs sont comptés ;
      open      : les appels sont refusés immédiatement (économie de la
                  dépendance en panne) pendant `reset_timeout` secondes ;
      half_open : après `reset_timeout`, une requête-sonde est autorisée ;
                  succès → closed, échec → open à nouveau.

    Thread-safe : transitions protégées par un verrou (exécution parallèle).
    """

    def __init__(
        self,
        name: str = "circuit",
        failure_threshold: int = 5,
        reset_timeout: float = 10.0,
    ):
        if failure_threshold < 1:
            raise ValueError("failure_threshold doit être >= 1.")
        if reset_timeout <= 0:
            raise ValueError("reset_timeout doit être > 0.")

        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout

        self._failure_count = 0
        self._state = "closed"
        self._opened_at: Optional[float] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # État observable
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failure_count

    def is_open(self) -> bool:
        return self.state == "open"

    # ------------------------------------------------------------------
    # Cycle de vie
    # ------------------------------------------------------------------

    def allow_request(self) -> bool:
        """
        True si l'appel peut partir, False s'il doit être refusé (open).

        Le passage open → half_open se fait ici : dès que le délai de
        récupération est écoulé, une requête-sonde est autorisée.
        """
        with self._lock:
            if self._state == "closed":
                return True

            if self._state == "open":
                if time.monotonic() - self._opened_at >= self.reset_timeout:
                    self._state = "half_open"
                    return True
                return False

            # half_open : laisse passer la sonde unique
            return True

    def record_success(self) -> None:
        """Appel réussi : compteur remis à zéro, retour à closed."""
        with self._lock:
            self._failure_count = 0
            if self._state == "half_open":
                self._state = "closed"

    def record_failure(self) -> None:
        """Appel en échec : compteur incrémenté, ouverture éventuelle."""
        with self._lock:
            self._failure_count += 1

            # Une sonde en échec rouvre immédiatement le circuit.
            if self._state == "half_open":
                self._state = "open"
                self._opened_at = time.monotonic()
                return

            if self._failure_count >= self.failure_threshold:
                self._state = "open"
                self._opened_at = time.monotonic()

    def reset(self) -> None:
        """Remet le coupe-circuit à zéro (fermé, aucun échec compté)."""
        with self._lock:
            self._failure_count = 0
            self._state = "closed"
            self._opened_at = None

    def info(self) -> dict:
        with self._lock:
            return {
                "name": self.name,
                "state": self._state,
                "failure_threshold": self.failure_threshold,
                "reset_timeout": self.reset_timeout,
                "failure_count": self._failure_count,
            }
