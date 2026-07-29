"""
Tests pour Executor, ToolRegistry, BaseTool.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.executor.executor import Executor
from app.registry.base_tool import BaseTool
from app.registry.tool_registry import ToolRegistry
from app.telemetry.logger import Logger


class TestTool(BaseTool):
    def __init__(self):
        super().__init__("test_tool", "Un outil de test")

    def execute(self, **kwargs):
        return {"result": f"Executed: {kwargs.get('input', '')}"}


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = TestTool()
        registry.register(tool)
        assert registry.exists("test_tool") is True
        retrieved = registry.get("test_tool")
        assert retrieved.name == "test_tool"

    def test_get_nonexistent(self):
        registry = ToolRegistry()
        try:
            registry.get("nonexistent")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_list_tools(self):
        registry = ToolRegistry()
        tool = TestTool()
        registry.register(tool)
        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "test_tool"

    def test_exists(self):
        registry = ToolRegistry()
        assert registry.exists("test_tool") is False
        registry.register(TestTool())
        assert registry.exists("test_tool") is True


class TestBaseTool:
    def test_info(self):
        tool = TestTool()
        info = tool.info()
        assert info["name"] == "test_tool"
        assert info["description"] == "Un outil de test"

    def test_execute(self):
        tool = TestTool()
        result = tool.execute(input="hello")
        assert result["result"] == "Executed: hello"


class TestExecutor:
    def test_register_and_run(self):
        executor = Executor()
        executor.register_tool(TestTool())
        action = {"tool": "test_tool", "parameters": {"input": "hello"}}
        result = executor.run(action)
        assert result["status"] == "success"
        assert result["tool"] == "test_tool"
        assert "execution_time" in result

    def test_run_nonexistent_tool(self):
        executor = Executor()
        action = {"tool": "nonexistent", "parameters": {}}
        result = executor.run(action)
        assert result["status"] == "error"

    def test_run_invalid_action(self):
        executor = Executor()
        action = {"tool": "test_tool"}  # missing parameters
        result = executor.run(action)
        assert result["status"] == "error"

    def test_run_not_dict(self):
        executor = Executor()
        action = "not_a_dict"
        result = executor.run(action)  # type: ignore
        assert result["status"] == "error"

    def test_has_tool(self):
        executor = Executor()
        executor.register_tool(TestTool())
        assert executor.has_tool("test_tool") is True
        assert executor.has_tool("nonexistent") is False

    def test_available_tools(self):
        executor = Executor()
        executor.register_tool(TestTool())
        tools = executor.available_tools()
        assert len(tools) == 1

    def test_prepare_action(self):
        executor = Executor()
        action = {"tool": "test", "parameters": {}}
        prepared = executor.prepare_action(action)
        assert prepared == action

    def test_validate_action(self):
        executor = Executor()
        executor.register_tool(TestTool())
        action = {"tool": "test_tool", "parameters": {"input": "test"}}
        assert executor.validate_action(action) is True


class TestLogger:
    def test_log_success(self):
        logger = Logger(log_dir="logs_test")
        logger.log_success("test_tool", 0.123)
        assert True

    def test_log_error(self):
        logger = Logger(log_dir="logs_test")
        logger.log_error("Test error", 0.04)
        assert True

    def test_log_info(self):
        logger = Logger(log_dir="logs_test")
        logger.log_info("Info message")
        assert True

    def test_log_warning(self):
        logger = Logger(log_dir="logs_test")
        logger.log_warning("Warning message")
        assert True

    def test_log_action_start_end(self):
        logger = Logger(log_dir="logs_test")
        logger.log_action_start("FusionSearch")
        logger.log_action_end("FusionSearch")
        assert True

    def test_log_event(self):
        logger = Logger(log_dir="logs_test")
        logger.log_event("TEST", "Message")
        assert True
