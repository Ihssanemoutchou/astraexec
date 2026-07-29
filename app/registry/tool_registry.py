from typing import Dict

from app.registry.base_tool import BaseTool


class ToolRegistry:

    def __init__(self):

        self.tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):

        self.tools[tool.name] = tool

    def get(self, name: str):

        if name not in self.tools:
            raise ValueError(f"Outil '{name}' introuvable.")

        return self.tools[name]

    def exists(self, name: str):

        return name in self.tools

    def list_tools(self):

        return [
            tool.info()
            for tool in self.tools.values()
        ]