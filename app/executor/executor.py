import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

from app.registry.tool_registry import ToolRegistry
from app.guardrails.validator import Validator
from app.guardrails.injection_guard import InjectionGuard
from app.telemetry.logger import Logger
from app.executor.errors import (
    BudgetExhaustedError,
    InvalidOutputError,
    InvalidSchemaError,
    PermissionDeniedError,
    ToolTimeoutError,
    ToolUnavailableError,
)
from app.executor.resilience import CircuitBreaker, RetryPolicy
from app.telemetry.audit import AuditLogger


class Executor:
    """
    AstraExec Executor

    Responsable de :
    - Validation des actions
    - Sélection de l'outil
    - Exécution
    - Journalisation
    """

    def __init__(
        self,
        max_actions: Optional[int] = None,
        retry_policy: Optional[RetryPolicy] = None,
        circuit_breakers: Optional[Dict[str, CircuitBreaker]] = None,
        default_timeout: Optional[float] = None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        """
        Extensions production (toutes optionnelles, désactivées par défaut) :

        - max_actions      : budget d'actions (nombre max d'appels à run()).
        - retry_policy     : stratégie de nouvelle tentative (backoff exponentiel).
        - circuit_breakers : coupe-circuit par outil {nom_outil: CircuitBreaker}.
        - default_timeout  : délai max d'exécution par défaut (s), surchargé par
                             le timeout déclaré dans les métadonnées de l'outil.
        - audit_logger     : journal d'audit structuré JSON Lines.

        Aucun de ces paramètres n'altère le comportement historique du moteur.
        """

        self.registry = ToolRegistry()

        self.validator = Validator()

        self.guard = InjectionGuard()

        self.logger = Logger()

        # -------------------------------------------------------
        # Extensions production (optionnelles)
        # -------------------------------------------------------
        self.max_actions = max_actions
        self.retry_policy = retry_policy
        self.circuit_breakers = dict(circuit_breakers or {})
        self.default_timeout = default_timeout
        self.audit_logger = audit_logger

        self._actions_used = 0
        self._lock = threading.Lock()

    # =======================================================
    # Enregistrer un outil
    # =======================================================

    def register_tool(self, tool):

        self.registry.register(tool)

    # =======================================================
    # Vérifier si l'outil existe
    # =======================================================

    def has_tool(self, tool_name: str):

        return self.registry.exists(tool_name)

    # =======================================================
    # Liste des outils
    # =======================================================

    def available_tools(self):

        return self.registry.list_tools()
        # =======================================================
    # Préparation d'une action
    # =======================================================

    def prepare_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Vérifie que l'action est valide avant son exécution.
        """

        if not isinstance(action, dict):
            raise TypeError("Une action doit être un dictionnaire.")

        required = ["tool", "parameters"]

        for field in required:
            if field not in action:
                raise ValueError(f"Champ obligatoire manquant : {field}")

        return action

    # =======================================================
    # Validation complète
    # =======================================================

    def validate_action(self, action: Dict[str, Any]):

        action = self.prepare_action(action)

        self.validator.validate(action)

        self.guard.inspect(action)

        return True

    # =======================================================
    # Récupération de l'outil
    # =======================================================

    def resolve_tool(self, action):

        tool_name = action["tool"]

        if not self.registry.exists(tool_name):

            raise ValueError(
                f"L'outil '{tool_name}' n'existe pas."
            )

        return self.registry.get(tool_name)
        # =======================================================
    # Exécution interne
    # =======================================================

    def execute_tool(self, tool, parameters):

        return tool.execute(**parameters)

    # =======================================================
    # Budget d'actions
    # =======================================================

    @property
    def actions_used(self) -> int:
        """Nombre d'actions déjà consommées."""
        return self._actions_used

    @property
    def actions_remaining(self) -> Optional[int]:
        """Actions restantes (None si aucun budget configuré)."""
        if self.max_actions is None:
            return None
        return max(0, self.max_actions - self._actions_used)

    def _consume_budget(self) -> bool:
        """Réserve une unité du budget. False si le budget est épuisé."""
        if self.max_actions is None:
            return True
        with self._lock:
            if self._actions_used >= self.max_actions:
                return False
            self._actions_used += 1
            return True

    # =======================================================
    # Permissions
    # =======================================================

    def _check_permissions(self, tool, permissions) -> None:
        """Vérifie que les permissions accordées couvrent l'outil demandé."""
        if permissions is None:
            return
        # Accepte aussi une chaîne unique (ex. "retrieval").
        if isinstance(permissions, str):
            permissions = [permissions]
        required = list(getattr(tool, "permissions", None) or [])
        if not required:
            return
        allowed = set(permissions or [])
        missing = [perm for perm in required if perm not in allowed]
        if missing:
            raise PermissionDeniedError(
                f"Permission refusée : l'outil '{tool.name}' requiert {missing}, "
                f"non inclus dans les permissions accordées : {sorted(allowed)}."
            )

    # =======================================================
    # Exécution résiliente (retry + timeout + circuit breaker)
    # =======================================================

    def _execute_with_resilience(self, tool, parameters):

        breaker = self.circuit_breakers.get(tool.name)

        if breaker is not None and not breaker.allow_request():
            raise ToolUnavailableError(
                f"L'outil '{tool.name}' est indisponible : circuit breaker ouvert."
            )

        timeout = getattr(tool, "timeout", None)
        if timeout is None:
            timeout = self.default_timeout

        retry = self.retry_policy
        attempt = 0

        while True:
            try:
                result = self._run_with_timeout(tool, parameters, timeout)

                if breaker is not None:
                    breaker.record_success()

                return result

            except Exception as error:

                # Backoff exponentiel : nouvelle tentative si l'erreur est
                # retentable et que le budget de retries n'est pas épuisé.
                if (
                    retry is not None
                    and retry.should_retry(error)
                    and attempt < retry.max_retries
                ):
                    time.sleep(retry.delay_for(attempt))
                    attempt += 1
                    continue

                if breaker is not None:
                    breaker.record_failure()

                raise

    def _run_with_timeout(self, tool, parameters, timeout):

        if timeout is None or timeout <= 0:
            return tool.execute(**parameters)

        # Timeout portable (Windows inclus) : exécution dans un thread de
        # travail et attente bornée. Si le délai expire, une ToolTimeoutError
        # est levée ; le thread orphelin est laissé en arrière-plan.
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(tool.execute, **parameters)

        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            pool.shutdown(wait=False)
            raise ToolTimeoutError(
                f"L'outil '{tool.name}' a dépassé le délai de {timeout}s."
            ) from None
        except BaseException:
            pool.shutdown(wait=True)
            raise
        else:
            pool.shutdown(wait=True)

    # =======================================================
    # Validation de la sortie (output_schema)
    # =======================================================

    def _validate_output(self, tool, result) -> None:

        output_schema = getattr(tool, "output_schema", None) or {}

        if not output_schema:
            return

        if not isinstance(result, dict):
            raise InvalidOutputError(
                f"Sortie invalide pour l'outil '{tool.name}' : un dictionnaire "
                f"est attendu, reçu {type(result).__name__}."
            )

        try:
            self.validator.validate_schema(result, output_schema)
        except (TypeError, ValueError) as error:
            raise InvalidOutputError(str(error)) from error

    # =======================================================
    # Journalisation d'audit structurée (JSON Lines)
    # =======================================================

    def _audit(
        self,
        execution_id,
        plan_id,
        tool_name,
        parameters,
        latency,
        status,
        error=None,
        error_type=None,
    ):

        if self.audit_logger is None:
            return

        self.audit_logger.record(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            execution_id=execution_id,
            plan_id=plan_id,
            tool=tool_name,
            arguments=parameters,
            latency=round(latency, 4),
            status=status,
            error=error,
            error_type=error_type,
        )

    # =======================================================
    # Pipeline principal
    # =======================================================

    def run(
        self,
        action: Dict[str, Any],
        permissions: Optional[Iterable[str]] = None,
        plan_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Exécute une action complète.

        Extensions production (optionnelles, rétrocompatibles) :

        - permissions : permissions accordées à l'appelant ; si fournies,
          l'outil ne s'exécute que si ses permissions requises y figurent.
        - plan_id     : identifiant du plan multi-étapes (journalisé dans
          l'audit JSON), si l'action est exécutée dans le cadre d'un plan.

        Contrat de réponse STRICTEMENT inchangé :
          succès → {"status": "success", "tool", "execution_time", "result"}
          erreur → {"status": "error", "execution_time", "message"}
        Le moteur ne lève JAMAIS d'exception vers l'appelant.
        """

        start_time = time.perf_counter()

        execution_id = uuid.uuid4().hex

        tool_name = None
        parameters = {}

        try:

            # Budget d'actions (extension production)
            if not self._consume_budget():
                raise BudgetExhaustedError(
                    f"Budget d'actions épuisé : maximum {self.max_actions} "
                    f"action(s) autorisée(s)."
                )

            # Validation
            self.validate_action(action)

            # Sélection de l'outil
            tool = self.resolve_tool(action)

            tool_name = tool.name

            # Vérification des permissions (extension production)
            self._check_permissions(tool, permissions)

            # Paramètres
            parameters = action.get("parameters", {})

            # Validation du schéma des paramètres (si l'outil en définit un)
            schema = getattr(tool, "parameter_schema", None)
            if schema is not None:
                try:
                    self.validator.validate_schema(parameters, schema)
                except (TypeError, ValueError) as error:
                    raise InvalidSchemaError(str(error)) from error

            # Exécution résiliente (retry + timeout + circuit breaker)
            result = self._execute_with_resilience(tool, parameters)

            # Validation de la sortie (output_schema, extension production)
            self._validate_output(tool, result)

            elapsed = time.perf_counter() - start_time

            # Journalisation
            self.logger.log_success(
                tool.name,
                elapsed,
            )

            # Audit structuré (extension production)
            self._audit(
                execution_id,
                plan_id,
                tool_name,
                parameters,
                elapsed,
                "success",
            )

            return {

                "status": "success",

                "tool": tool.name,

                "execution_time": round(elapsed, 4),

                "result": result,

            }

        except Exception as error:

            elapsed = time.perf_counter() - start_time

            self.logger.log_error(
                str(error),
                elapsed,
            )

            # Audit structuré (extension production)
            self._audit(
                execution_id,
                plan_id,
                tool_name,
                parameters,
                elapsed,
                "error",
                str(error),
                type(error).__name__,
            )

            return {

                "status": "error",

                "execution_time": round(elapsed, 4),

                "message": str(error),

            }