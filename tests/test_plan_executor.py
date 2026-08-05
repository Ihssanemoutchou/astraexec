"""
Tests du PlanExecutor : plans JSON multi-étapes avec dépendances (DAG).

Couvre :
- exécution d'un plan simple (toutes étapes en succès) ;
- respect de l'ordre imposé par les dépendances ;
- exécution parallèle optionnelle des étapes indépendantes ;
- mode séquentiel (parallel=False) ;
- détection des cycles, dépendances inconnues, ids dupliqués ;
- plans mal formés (pas un dict, steps vides, outil manquant) ;
- fail_fast (étapes restantes marquées skipped) vs mode continue ;
- passage de données entre étapes (références "$step_id") ;
- audit structuré des étapes avec plan_id.

L'Executor existant est réutilisé tel quel : aucun composant métier
n'est modifié.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.executor.executor import Executor
from app.executor.plan_executor import PlanExecutor
from app.registry.base_tool import BaseTool
from app.telemetry.audit import AuditLogger


class EchoTool(BaseTool):
    """Outil de test qui journalise ses appels et renvoie ses paramètres."""

    def __init__(self, name="echo", events=None, delay=0.0):
        super().__init__(name, "Outil echo de test")
        self.events = events if events is not None else []
        self.delay = delay

    def execute(self, **kwargs):
        self.events.append(("start", self.name, time.perf_counter()))
        if self.delay:
            time.sleep(self.delay)
        self.events.append(("end", self.name, time.perf_counter()))
        return {
            "tool": self.name,
            "echo": kwargs.get("value", None),
            "query": kwargs.get("query", None),
        }


def make_executor(events=None):
    executor = Executor()
    executor.register_tool(EchoTool("echo", events))
    executor.register_tool(EchoTool("echo2", events))
    executor.register_tool(EchoTool("echo3", events))
    return executor


PLAN_SIMPLE = {
    "plan_id": "PLAN-SIMPLE",
    "steps": [
        {"id": "a", "tool": "echo", "parameters": {"value": 1}},
        {"id": "b", "tool": "echo2", "parameters": {"value": 2}},
    ],
}

PLAN_DAG = {
    "plan_id": "PLAN-DAG",
    "steps": [
        {"id": "a", "tool": "echo", "parameters": {"value": "racine"}, "dependencies": []},
        {"id": "b", "tool": "echo2", "parameters": {"value": "enfant-a"}, "dependencies": ["a"]},
        {"id": "c", "tool": "echo3", "parameters": {"value": "enfant-a-aussi"}, "dependencies": ["a"]},
        {"id": "d", "tool": "echo", "parameters": {"value": "enfant-bc"}, "dependencies": ["b", "c"]},
    ],
}


class TestPlanBasics:
    def test_simple_plan_all_success(self):
        report = PlanExecutor(make_executor()).execute(PLAN_SIMPLE)
        assert report["status"] == "success"
        assert report["plan_id"] == "PLAN-SIMPLE"
        assert report["total_steps"] == 2
        assert report["successful_steps"] == 2
        assert report["failed_steps"] == 0
        assert report["skipped_steps"] == 0
        assert report["steps"]["a"]["status"] == "success"
        assert report["steps"]["b"]["status"] == "success"

    def test_each_step_keeps_executor_contract(self):
        report = PlanExecutor(make_executor()).execute(PLAN_SIMPLE)
        step = report["steps"]["a"]
        assert step["tool"] == "echo"
        assert step["execution_time"] >= 0
        assert step["result"]["echo"] == 1
        assert step["step_id"] == "a"
        assert step["duration"] >= 0

    def test_execution_id_present(self):
        report = PlanExecutor(make_executor()).execute(PLAN_SIMPLE)
        assert report["execution_id"]


class TestDependencies:
    def test_dependencies_impose_lor(self):
        events = []
        report = PlanExecutor(make_executor(events)).execute(PLAN_DAG)
        assert report["status"] == "success"

        starts = {
            name: [t for (e, n, t) in events if e == "start" and n == name]
            for name in ("echo", "echo2", "echo3")
        }
        ends = {
            name: [t for (e, n, t) in events if e == "end" and n == name]
            for name in ("echo", "echo2", "echo3")
        }

        # b et c démarrent après la fin de a (dépendances directes).
        assert starts["echo2"][0] >= ends["echo"][0]
        assert starts["echo3"][0] >= ends["echo"][0]
        # d (2e exécution de echo) démarre après la fin de b et c.
        assert starts["echo"][1] >= ends["echo2"][0]
        assert starts["echo"][1] >= ends["echo3"][0]

    def test_dependency_order_in_report(self):
        report = PlanExecutor(make_executor()).execute(PLAN_DAG)
        assert report["steps"]["a"]["status"] == "success"
        assert report["steps"]["b"]["status"] == "success"
        assert report["steps"]["c"]["status"] == "success"
        assert report["steps"]["d"]["status"] == "success"


class TestParallelExecution:
    def test_parallel_independent_steps_overlap(self):
        executor = Executor()
        executor.register_tool(EchoTool("echo", None, delay=0.3))
        executor.register_tool(EchoTool("echo2", None, delay=0.3))
        plan = {
            "plan_id": "P-PAR",
            "steps": [
                {"id": "a", "tool": "echo", "parameters": {}},
                {"id": "b", "tool": "echo2", "parameters": {}},
            ],
        }
        start = time.perf_counter()
        report = PlanExecutor(executor, parallel=True).execute(plan)
        wall = time.perf_counter() - start
        assert report["status"] == "success"
        # 2x0.3s en parallèle ≈ 0.3s ; en série ≈ 0.6s.
        assert wall < 0.5

    def test_serial_mode_no_overlap(self):
        executor = Executor()
        executor.register_tool(EchoTool("echo", None, delay=0.1))
        executor.register_tool(EchoTool("echo2", None, delay=0.1))
        plan = {
            "plan_id": "P-SER",
            "steps": [
                {"id": "a", "tool": "echo", "parameters": {}},
                {"id": "b", "tool": "echo2", "parameters": {}},
            ],
        }
        start = time.perf_counter()
        report = PlanExecutor(executor, parallel=False).execute(plan)
        wall = time.perf_counter() - start
        assert report["status"] == "success"
        # 2x0.1s en série ≈ 0.2s.
        assert wall >= 0.18


class TestPlanValidation:
    def test_cycle_detected(self):
        plan = {
            "plan_id": "P-CYC",
            "steps": [
                {"id": "a", "tool": "echo", "parameters": {}, "dependencies": ["b"]},
                {"id": "b", "tool": "echo", "parameters": {}, "dependencies": ["a"]},
            ],
        }
        report = PlanExecutor(make_executor()).execute(plan)
        assert report["status"] == "error"
        assert "cycle" in report["message"].lower()

    def test_self_dependency_detected(self):
        plan = {
            "plan_id": "P-SELF",
            "steps": [
                {"id": "a", "tool": "echo", "parameters": {}, "dependencies": ["a"]},
            ],
        }
        report = PlanExecutor(make_executor()).execute(plan)
        assert report["status"] == "error"
        assert "cycle" in report["message"].lower()

    def test_unknown_dependency(self):
        plan = {
            "plan_id": "P-UNK",
            "steps": [
                {"id": "a", "tool": "echo", "parameters": {}, "dependencies": ["zz"]},
            ],
        }
        report = PlanExecutor(make_executor()).execute(plan)
        assert report["status"] == "error"
        assert "dépendance inconnue" in report["message"].lower()

    def test_duplicate_ids(self):
        plan = {
            "plan_id": "P-DUP",
            "steps": [
                {"id": "a", "tool": "echo", "parameters": {}},
                {"id": "a", "tool": "echo2", "parameters": {}},
            ],
        }
        report = PlanExecutor(make_executor()).execute(plan)
        assert report["status"] == "error"
        assert "dupliqués" in report["message"].lower()

    def test_plan_not_dict(self):
        report = PlanExecutor(make_executor()).execute("pas un plan")  # type: ignore
        assert report["status"] == "error"

    def test_empty_steps(self):
        report = PlanExecutor(make_executor()).execute({"plan_id": "P", "steps": []})
        assert report["status"] == "error"

    def test_step_missing_tool(self):
        plan = {"plan_id": "P", "steps": [{"id": "a", "parameters": {}}]}
        report = PlanExecutor(make_executor()).execute(plan)
        assert report["status"] == "error"


class TestFailFast:
    PLAN = {
        "plan_id": "P-FF",
        "steps": [
            {"id": "a", "tool": "echo", "parameters": {}},
            {"id": "b", "tool": "ghost", "parameters": {}},  # outil inconnu
            {"id": "c", "tool": "echo3", "parameters": {}, "dependencies": ["a"]},
        ],
    }

    def test_fail_fast_skips_remaining_steps(self):
        report = PlanExecutor(make_executor(), fail_fast=True).execute(self.PLAN)
        assert report["status"] == "partial"
        assert report["steps"]["a"]["status"] == "success"
        assert report["steps"]["b"]["status"] == "error"
        assert report["steps"]["c"]["status"] == "skipped"
        assert report["skipped_steps"] == 1

    def test_no_fail_fast_continues(self):
        report = PlanExecutor(make_executor(), fail_fast=False).execute(self.PLAN)
        assert report["status"] == "partial"
        assert report["steps"]["a"]["status"] == "success"
        assert report["steps"]["b"]["status"] == "error"
        assert report["steps"]["c"]["status"] == "success"

    def test_all_steps_failed_status_error(self):
        plan = {
            "plan_id": "P-ERR",
            "steps": [
                {"id": "a", "tool": "ghost", "parameters": {}},
                {"id": "b", "tool": "ghost2", "parameters": {}},
            ],
        }
        report = PlanExecutor(make_executor()).execute(plan)
        assert report["status"] == "error"
        assert report["failed_steps"] == 2


class TestDataPassing:
    def test_step_reference_resolved(self):
        plan = {
            "plan_id": "P-REF",
            "steps": [
                {"id": "a", "tool": "echo", "parameters": {"value": 42}},
                {
                    "id": "b",
                    "tool": "echo2",
                    "parameters": {"value": "$a"},
                    "dependencies": ["a"],
                },
            ],
        }
        report = PlanExecutor(make_executor()).execute(plan)
        assert report["status"] == "success"
        result_b = report["steps"]["b"]["result"]
        # La valeur de b est le résultat complet de l'étape a.
        assert result_b["echo"]["echo"] == 42

    def test_unresolvable_reference_left_untouched(self):
        plan = {
            "plan_id": "P-NOREF",
            "steps": [
                {"id": "a", "tool": "echo", "parameters": {"value": "$ghost"}},
            ],
        }
        report = PlanExecutor(make_executor()).execute(plan)
        assert report["status"] == "success"
        assert report["steps"]["a"]["result"]["echo"] == "$ghost"


class TestPlanAudit:
    def test_steps_audited_with_plan_id(self, tmp_path):
        logger = AuditLogger(log_dir=str(tmp_path))
        executor = Executor(audit_logger=logger)
        executor.register_tool(EchoTool("echo"))
        plan = {
            "plan_id": "PLAN-AUDIT",
            "steps": [
                {"id": "a", "tool": "echo", "parameters": {"value": 1}},
            ],
        }
        report = PlanExecutor(executor).execute(plan)
        assert report["status"] == "success"

        records = logger.read()
        step_records = [
            r for r in records
            if r["plan_id"] == "PLAN-AUDIT" and r["tool"] == "echo"
        ]
        plan_records = [
            r for r in records
            if r["plan_id"] == "PLAN-AUDIT" and r["tool"] is None
        ]
        assert len(step_records) == 1
        assert step_records[0]["status"] == "success"
        assert step_records[0]["arguments"] == {"value": 1}
        assert len(plan_records) == 1
        assert plan_records[0]["status"] == "success"
