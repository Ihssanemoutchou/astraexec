from typing import Dict, Any
import time

from app.registry.tool_registry import ToolRegistry
from app.guardrails.validator import Validator
from app.guardrails.injection_guard import InjectionGuard
from app.telemetry.logger import Logger


class Executor:
    """
    AstraExec Executor

    Responsable de :
    - Validation des actions
    - Sélection de l'outil
    - Exécution
    - Journalisation
    """

    def __init__(self):

        self.registry = ToolRegistry()

        self.validator = Validator()

        self.guard = InjectionGuard()

        self.logger = Logger()

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
    # Pipeline principal
    # =======================================================

    def run(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Exécute une action complète.
        """

        start_time = time.perf_counter()

        try:

            # Validation
            self.validate_action(action)

            # Sélection de l'outil
            tool = self.resolve_tool(action)

            # Paramètres
            parameters = action.get("parameters", {})

            # Validation du schéma des paramètres (si l'outil en définit un)
            schema = getattr(tool, "parameter_schema", None)
            if schema is not None:
                self.validator.validate_schema(parameters, schema)

            # Exécution
            result = self.execute_tool(
                tool,
                parameters,
            )

            elapsed = time.perf_counter() - start_time

            # Journalisation
            self.logger.log_success(
                tool.name,
                elapsed,
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

            return {

                "status": "error",

                "execution_time": round(elapsed, 4),

                "message": str(error),

            }