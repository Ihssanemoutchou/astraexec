"""
AstraExec — Exécution de plans multi-étapes (extension production)
==================================================================

Exécute un plan JSON composé d'étapes (steps) liées par des dépendances
(formant un DAG), en réutilisant l'Executor existant — aucun composant
métier n'est modifié.

Format du plan :

    {
        "plan_id": "PLAN-001",
        "steps": [
            {
                "id": "s1",
                "tool": "fusion_search",
                "parameters": {"query": "machine learning"},
                "dependencies": []
            },
            {
                "id": "s2",
                "tool": "fusion_search",
                "parameters": {"query": "$s1"},   # référence au résultat de s1
                "dependencies": ["s1"]
            }
        ]
    }

Fonctionnalités :
- résolution topologique des dépendances (tri par vagues) ;
- détection des cycles, dépendances inconnues, ids dupliqués ;
- exécution parallèle optionnelle des étapes indépendantes ;
- fail_fast : arrêt du plan dès la première étape en échec ;
- passage de données entre étapes via les références "$step_id" ;
- audit structuré : chaque étape est journalisée avec plan_id.

Convention (identique à l'Executor) : execute() ne lève jamais
d'exception — un plan invalide retourne un objet d'erreur structuré.
"""

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.executor.errors import PlanValidationError
from app.executor.executor import Executor


@dataclass
class PlanStep:
    """Étape d'un plan : un appel d'outil avec ses dépendances."""

    step_id: str
    tool: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanStep":
        if not isinstance(data, dict):
            raise PlanValidationError(
                "Chaque étape du plan doit être un dictionnaire."
            )

        step_id = data.get("id")
        if not isinstance(step_id, str) or not step_id.strip():
            raise PlanValidationError(
                "Chaque étape doit avoir un identifiant 'id' non vide."
            )

        tool = data.get("tool")
        if not isinstance(tool, str) or not tool.strip():
            raise PlanValidationError(
                f"L'étape '{step_id}' doit définir un 'tool' non vide."
            )

        parameters = data.get("parameters", {})
        if not isinstance(parameters, dict):
            raise PlanValidationError(
                f"L'étape '{step_id}' doit avoir des 'parameters' de type dict."
            )

        dependencies = data.get("dependencies", [])
        if (
            not isinstance(dependencies, list)
            or not all(isinstance(d, str) for d in dependencies)
        ):
            raise PlanValidationError(
                f"L'étape '{step_id}' doit avoir des 'dependencies' "
                f"de type list de chaînes."
            )

        return cls(
            step_id=step_id,
            tool=tool,
            parameters=parameters,
            dependencies=list(dependencies),
        )


class PlanExecutor:
    """
    Exécuteur de plans multi-étapes (DAG).

    - executor  : l'Executor du projet (réutilisé tel quel).
    - parallel  : exécution parallèle des étapes indépendantes (True par défaut).
    - max_workers : nombre max de threads pour l'exécution parallèle.
    - fail_fast : stoppe le plan à la première étape en échec (True par défaut).
    - permissions : permissions accordées, transmises à chaque étape
                    (None = pas de vérification de permissions).
    """

    def __init__(
        self,
        executor: Executor,
        *,
        parallel: bool = True,
        max_workers: int = 4,
        fail_fast: bool = True,
        permissions: Optional[List[str]] = None,
    ):
        self.executor = executor
        self.parallel = parallel
        self.max_workers = max(1, int(max_workers))
        self.fail_fast = fail_fast
        self.permissions = permissions

    # ==================================================================
    # Point d'entrée
    # ==================================================================

    def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Exécute un plan complet et retourne un objet structuré.

        Ne lève jamais : les erreurs de validation de plan sont retournées
        sous forme d'objet d'erreur (même convention que Executor.run).
        """
        start_time = time.perf_counter()

        try:
            steps = self._parse_steps(plan)
            plan_id = plan.get("plan_id") if isinstance(plan, dict) else None
            waves = self._topological_waves(steps)

            by_id = {step.step_id: step for step in steps}
            flat_order = [sid for wave in waves for sid in wave]

            step_results: Dict[str, Dict[str, Any]] = {}

            for wave in waves:
                # fail_fast : une étape en échec interrompt le plan.
                if self.fail_fast and any(
                    step_results[sid]["status"] == "error"
                    for sid in step_results
                ):
                    for sid in flat_order:
                        if sid not in step_results:
                            step_results[sid] = self._skipped(by_id[sid])
                    break

                if self.parallel and len(wave) > 1:
                    self._run_wave_parallel(by_id, step_results, wave, plan_id)
                else:
                    for sid in wave:
                        step_results[sid] = self._execute_step(
                            by_id, step_results, sid, plan_id
                        )

            return self._build_report(plan_id, steps, step_results, start_time)

        except PlanValidationError as error:
            return {
                "status": "error",
                "execution_time": round(time.perf_counter() - start_time, 4),
                "message": str(error),
            }

    # ==================================================================
    # Validation et ordonnancement (DAG)
    # ==================================================================

    def _parse_steps(self, plan: Dict[str, Any]) -> List[PlanStep]:
        if not isinstance(plan, dict):
            raise PlanValidationError("Le plan doit être un dictionnaire.")

        raw_steps = plan.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise PlanValidationError(
                "Le plan doit contenir une liste non vide 'steps'."
            )

        return [PlanStep.from_dict(raw) for raw in raw_steps]

    def _topological_waves(self, steps: List[PlanStep]) -> List[List[str]]:
        """
        Tri topologique par vagues : chaque vague ne contient que des
        étapes dont toutes les dépendances sont déjà terminées.
        Détecte les ids dupliqués, les dépendances inconnues et les cycles.
        """
        by_id = {step.step_id: step for step in steps}

        if len(by_id) != len(steps):
            raise PlanValidationError(
                "Identifiants d'étapes dupliqués dans le plan."
            )

        for step in steps:
            for dep in step.dependencies:
                if dep not in by_id:
                    raise PlanValidationError(
                        f"L'étape '{step.step_id}' référence une dépendance "
                        f"inconnue : '{dep}'."
                    )

        remaining = set(by_id)
        waves: List[List[str]] = []

        while remaining:
            ready = [
                sid
                for sid in remaining
                if all(dep not in remaining for dep in by_id[sid].dependencies)
            ]
            if not ready:
                raise PlanValidationError(
                    "Détection de cycle dans les dépendances du plan."
                )
            waves.append(sorted(ready))
            remaining -= set(ready)

        return waves

    # ==================================================================
    # Exécution
    # ==================================================================

    def _run_wave_parallel(
        self,
        by_id: Dict[str, PlanStep],
        step_results: Dict[str, Dict[str, Any]],
        wave: List[str],
        plan_id: Optional[str],
    ) -> None:
        """Exécute une vague d'étapes indépendantes en parallèle."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(
                    self._execute_step, by_id, step_results, sid, plan_id
                ): sid
                for sid in wave
            }
            for future in as_completed(futures):
                sid = futures[future]
                try:
                    step_results[sid] = future.result()
                except Exception as error:  # jamais attendu (run ne lève pas)
                    step_results[sid] = {
                        "step_id": sid,
                        "tool": by_id[sid].tool,
                        "status": "error",
                        "message": str(error),
                        "duration": 0.0,
                    }

    def _execute_step(
        self,
        by_id: Dict[str, PlanStep],
        step_results: Dict[str, Dict[str, Any]],
        sid: str,
        plan_id: Optional[str],
    ) -> Dict[str, Any]:
        step = by_id[sid]
        parameters = self._resolve_parameters(step.parameters, step_results)
        action = {"tool": step.tool, "parameters": parameters}

        step_start = time.perf_counter()
        result = self.executor.run(
            action,
            permissions=self.permissions,
            plan_id=plan_id,
        )
        result["step_id"] = sid
        result["duration"] = round(time.perf_counter() - step_start, 4)
        return result

    def _resolve_parameters(
        self,
        parameters: Dict[str, Any],
        step_results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Résout les références "$step_id" dans les paramètres :
        la valeur est remplacée par le résultat de l'étape référencée.
        """

        def resolve(value: Any) -> Any:
            if isinstance(value, str) and value.startswith("$"):
                ref = value[1:]
                if ref in step_results:
                    return step_results[ref].get("result")
            if isinstance(value, dict):
                return {k: resolve(v) for k, v in value.items()}
            if isinstance(value, list):
                return [resolve(v) for v in value]
            return value

        return resolve(parameters)

    def _skipped(self, step: PlanStep) -> Dict[str, Any]:
        return {
            "step_id": step.step_id,
            "tool": step.tool,
            "status": "skipped",
            "reason": "Étape précédente en échec (fail_fast).",
        }

    # ==================================================================
    # Rapport
    # ==================================================================

    def _build_report(
        self,
        plan_id: Optional[str],
        steps: List[PlanStep],
        step_results: Dict[str, Dict[str, Any]],
        start_time: float,
    ) -> Dict[str, Any]:
        statuses = [step_results[sid]["status"] for sid in step_results]
        errors = [s for s in statuses if s == "error"]

        if errors:
            overall = (
                "error"
                if len(errors) == len(step_results)
                else "partial"
            )
        else:
            overall = "success"

        execution_id = uuid.uuid4().hex
        elapsed = round(time.perf_counter() - start_time, 4)

        # Audit structuré au niveau du plan (si un AuditLogger est branché).
        audit_logger = getattr(self.executor, "audit_logger", None)
        if audit_logger is not None:
            audit_logger.record(
                timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                execution_id=execution_id,
                plan_id=plan_id,
                tool=None,
                arguments={"steps": [step.step_id for step in steps]},
                latency=elapsed,
                status=overall,
                error=None,
            )

        return {
            "plan_id": plan_id,
            "execution_id": execution_id,
            "status": overall,
            "total_steps": len(steps),
            "successful_steps": sum(
                1 for r in step_results.values() if r["status"] == "success"
            ),
            "failed_steps": sum(
                1 for r in step_results.values() if r["status"] == "error"
            ),
            "skipped_steps": sum(
                1 for r in step_results.values() if r["status"] == "skipped"
            ),
            "execution_time": elapsed,
            "steps": step_results,
        }
