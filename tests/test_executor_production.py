"""
Tests des extensions production de l'Executor :
- métadonnées d'outil (BaseTool.metadata : input/output schema, timeout,
  permissions, estimated_cost, version) ;
- vérification des permissions avant exécution ;
- budget d'actions (max_actions) ;
- journal d'audit structuré JSON Lines ;
- validation de la sortie (output_schema) ;
- classification des erreurs (audit error_type) ;
- contrat de réponse STRICTEMENT inchangé (non-régression).

Aucun composant métier existant n'est modifié.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.executor.executor import Executor
from app.registry.base_tool import BaseTool
from app.telemetry.audit import AuditLogger


class SimpleTool(BaseTool):
    """Outil de test simple, acceptant des métadonnées via kwargs."""

    def __init__(self, name="simple", **metadata):
        super().__init__(name, f"Outil {name}", **metadata)

    def execute(self, **kwargs):
        return {"ok": True, "value": kwargs.get("value")}


class SchemaOutputTool(BaseTool):
    """Outil avec output_schema (extension production)."""

    def __init__(self, valid=True):
        super().__init__(
            "schema_out",
            "Outil avec sortie validée",
            output_schema={
                "status": {"type": "string", "required": True},
                "count": {"type": "integer", "required": True},
            },
        )
        self.valid = valid

    def execute(self, **kwargs):
        if self.valid:
            return {"status": "ok", "count": 3}
        return {"status": "ok"}  # "count" manquant


class TestToolMetadata:
    def test_defaults(self):
        tool = SimpleTool("meta_default")
        assert tool.timeout is None
        assert tool.permissions == []
        assert tool.estimated_cost == 0.0
        assert tool.version == "1.0.0"
        assert tool.input_schema == {}
        assert tool.output_schema == {}

    def test_custom_metadata(self):
        tool = SimpleTool(
            "meta_custom",
            timeout=2.5,
            permissions=["retrieval", "write"],
            estimated_cost=0.01,
            version="2.1.0",
            input_schema={"query": {"type": "string", "required": True}},
            output_schema={"ok": {"type": "boolean", "required": True}},
        )
        assert tool.timeout == 2.5
        assert tool.permissions == ["retrieval", "write"]
        assert tool.estimated_cost == 0.01
        assert tool.version == "2.1.0"
        assert tool.input_schema["query"]["required"] is True
        assert tool.output_schema["ok"]["required"] is True

    def test_metadata_returns_all_fields(self):
        tool = SimpleTool(
            "meta_all",
            timeout=1.0,
            permissions=["r"],
            estimated_cost=0.5,
            version="3.0.0",
        )
        metadata = tool.metadata()
        assert metadata["name"] == "meta_all"
        assert metadata["description"] == "Outil meta_all"
        assert metadata["timeout"] == 1.0
        assert metadata["permissions"] == ["r"]
        assert metadata["estimated_cost"] == 0.5
        assert metadata["version"] == "3.0.0"
        assert "input_schema" in metadata
        assert "output_schema" in metadata

    def test_input_schema_fallback_to_parameter_schema(self):
        class LegacyTool(BaseTool):
            """Outil historique : parameter_schema, pas d'input_schema."""

            def __init__(self):
                super().__init__("legacy", "Outil avec parameter_schema")

            def execute(self, **kwargs):
                return {"ok": True}

            @property
            def parameter_schema(self):
                return {"query": {"type": "string", "required": True}}

        tool = LegacyTool()
        assert tool.metadata()["input_schema"]["query"]["required"] is True

    def test_info_unchanged(self):
        tool = SimpleTool("meta_info")
        assert set(tool.info().keys()) == {"name", "description"}


class TestPermissions:
    def test_no_permissions_arg_skips_check(self):
        executor = Executor()
        executor.register_tool(SimpleTool("sec", permissions=["admin"]))
        result = executor.run({"tool": "sec", "parameters": {}})
        assert result["status"] == "success"

    def test_allowed_permission(self):
        executor = Executor()
        executor.register_tool(SimpleTool("sec", permissions=["retrieval"]))
        result = executor.run(
            {"tool": "sec", "parameters": {}},
            permissions=["retrieval"],
        )
        assert result["status"] == "success"

    def test_denied_permission(self):
        executor = Executor()
        executor.register_tool(SimpleTool("sec", permissions=["retrieval"]))
        result = executor.run(
            {"tool": "sec", "parameters": {}},
            permissions=["write"],
        )
        assert result["status"] == "error"
        assert "Permission refusée" in result["message"]

    def test_tool_without_required_permissions_always_allowed(self):
        executor = Executor()
        executor.register_tool(SimpleTool("open"))
        result = executor.run(
            {"tool": "open", "parameters": {}},
            permissions=["write"],
        )
        assert result["status"] == "success"

    def test_permissions_as_single_string(self):
        executor = Executor()
        executor.register_tool(SimpleTool("sec_str", permissions=["retrieval"]))
        result = executor.run(
            {"tool": "sec_str", "parameters": {}},
            permissions="retrieval",  # chaîne unique acceptée
        )
        assert result["status"] == "success"


class TestActionBudget:
    def test_budget_exhausted(self):
        executor = Executor(max_actions=2)
        executor.register_tool(SimpleTool("budget_tool"))
        assert executor.actions_remaining == 2

        assert executor.run({"tool": "budget_tool", "parameters": {}})["status"] == "success"
        assert executor.actions_remaining == 1

        assert executor.run({"tool": "budget_tool", "parameters": {}})["status"] == "success"
        assert executor.actions_remaining == 0

        result = executor.run({"tool": "budget_tool", "parameters": {}})
        assert result["status"] == "error"
        assert "Budget" in result["message"]
        assert executor.actions_used == 2

    def test_unlimited_budget_by_default(self):
        executor = Executor()
        assert executor.actions_remaining is None
        executor.register_tool(SimpleTool("unlimited"))
        for _ in range(5):
            result = executor.run({"tool": "unlimited", "parameters": {}})
            assert result["status"] == "success"


class TestAuditLog:
    def test_success_record_has_all_required_fields(self, tmp_path):
        logger = AuditLogger(log_dir=str(tmp_path))
        executor = Executor(audit_logger=logger)
        executor.register_tool(SimpleTool("audit_tool"))
        result = executor.run({"tool": "audit_tool", "parameters": {"value": 7}})
        assert result["status"] == "success"

        records = logger.read()
        assert len(records) == 1
        record = records[0]
        for field in (
            "timestamp",
            "execution_id",
            "plan_id",
            "tool",
            "arguments",
            "latency",
            "status",
            "error",
        ):
            assert field in record, f"Champ manquant : {field}"
        assert record["tool"] == "audit_tool"
        assert record["arguments"] == {"value": 7}
        assert record["status"] == "success"
        assert record["error"] is None
        assert record["plan_id"] is None
        assert isinstance(record["latency"], (int, float))

    def test_error_record_captures_message_and_type(self, tmp_path):
        logger = AuditLogger(log_dir=str(tmp_path))
        executor = Executor(audit_logger=logger)
        result = executor.run({"tool": "ghost", "parameters": {}})
        assert result["status"] == "error"

        records = logger.read()
        assert len(records) == 1
        assert records[0]["status"] == "error"
        assert "ghost" in records[0]["error"]
        assert records[0]["error_type"] == "ValueError"
        assert records[0]["tool"] is None

    def test_execution_ids_unique(self, tmp_path):
        logger = AuditLogger(log_dir=str(tmp_path))
        executor = Executor(audit_logger=logger)
        executor.register_tool(SimpleTool("uid_tool"))
        executor.run({"tool": "uid_tool", "parameters": {}})
        executor.run({"tool": "uid_tool", "parameters": {}})
        records = logger.read()
        assert len(records) == 2
        assert records[0]["execution_id"]
        assert records[0]["execution_id"] != records[1]["execution_id"]

    def test_plan_id_propagated(self, tmp_path):
        logger = AuditLogger(log_dir=str(tmp_path))
        executor = Executor(audit_logger=logger)
        executor.register_tool(SimpleTool("plan_tool"))
        executor.run(
            {"tool": "plan_tool", "parameters": {}},
            plan_id="PLAN-42",
        )
        record = logger.read()[0]
        assert record["plan_id"] == "PLAN-42"

    def test_json_lines_format(self, tmp_path):
        logger = AuditLogger(log_dir=str(tmp_path))
        executor = Executor(audit_logger=logger)
        executor.register_tool(SimpleTool("json_tool"))
        executor.run({"tool": "json_tool", "parameters": {}})
        with open(logger.path, "r", encoding="utf-8") as fh:
            lines = [line for line in fh if line.strip()]
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["tool"] == "json_tool"

    def test_audit_disabled_by_default(self, tmp_path):
        logger = AuditLogger(log_dir=str(tmp_path))
        executor = Executor()  # pas d'audit_logger
        executor.register_tool(SimpleTool("no_audit"))
        executor.run({"tool": "no_audit", "parameters": {}})
        assert logger.read() == []


class TestOutputValidation:
    def test_valid_output_passes(self):
        executor = Executor()
        executor.register_tool(SchemaOutputTool(valid=True))
        result = executor.run({"tool": "schema_out", "parameters": {}})
        assert result["status"] == "success"

    def test_invalid_output_rejected(self):
        executor = Executor()
        executor.register_tool(SchemaOutputTool(valid=False))
        result = executor.run({"tool": "schema_out", "parameters": {}})
        assert result["status"] == "error"
        assert "count" in result["message"]

    def test_no_output_schema_no_check(self):
        executor = Executor()
        executor.register_tool(SimpleTool("no_schema"))
        result = executor.run({"tool": "no_schema", "parameters": {}})
        assert result["status"] == "success"


class TestErrorContractAndCategorization:
    def test_success_contract_unchanged(self):
        executor = Executor()
        executor.register_tool(SimpleTool("contract"))
        result = executor.run({"tool": "contract", "parameters": {}})
        assert set(result.keys()) == {"status", "tool", "execution_time", "result"}

    def test_error_contract_unchanged(self):
        executor = Executor()
        result = executor.run({"tool": "ghost", "parameters": {}})
        assert set(result.keys()) == {"status", "execution_time", "message"}

    def test_schema_violation_categorized_in_audit(self, tmp_path):
        logger = AuditLogger(log_dir=str(tmp_path))
        executor = Executor(audit_logger=logger)

        class StrictTool(BaseTool):
            def __init__(self):
                super().__init__("strict", "Outil strict")

            def execute(self, **kwargs):
                return {"ok": True}

            @property
            def parameter_schema(self):
                return {"query": {"type": "string", "required": True}}

        executor.register_tool(StrictTool())
        result = executor.run({"tool": "strict", "parameters": {}})
        assert result["status"] == "error"
        assert "query" in result["message"]

        record = logger.read()[0]
        assert record["error_type"] == "InvalidSchemaError"

    def test_invalid_output_categorized_in_audit(self, tmp_path):
        logger = AuditLogger(log_dir=str(tmp_path))
        executor = Executor(audit_logger=logger)
        executor.register_tool(SchemaOutputTool(valid=False))
        result = executor.run({"tool": "schema_out", "parameters": {}})
        assert result["status"] == "error"
        record = logger.read()[0]
        assert record["error_type"] == "InvalidOutputError"

    def test_permission_denied_categorized_in_audit(self, tmp_path):
        logger = AuditLogger(log_dir=str(tmp_path))
        executor = Executor(audit_logger=logger)
        executor.register_tool(SimpleTool("guarded", permissions=["admin"]))
        result = executor.run(
            {"tool": "guarded", "parameters": {}},
            permissions=["user"],
        )
        assert result["status"] == "error"
        record = logger.read()[0]
        assert record["error_type"] == "PermissionDeniedError"
