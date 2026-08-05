from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseTool(ABC):
    """
    Classe abstraite représentant un outil AstraExec.
    """

    def __init__(
        self,
        name: str,
        description: str,
        *,
        timeout: Optional[float] = None,
        permissions: Optional[List[str]] = None,
        estimated_cost: float = 0.0,
        version: str = "1.0.0",
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
    ):

        self.name = name
        self.description = description

        # Métadonnées de production (extension) — optionnelles par défaut.
        self.timeout = timeout
        self.permissions = list(permissions or [])
        self.estimated_cost = estimated_cost
        self.version = version
        self.input_schema = dict(input_schema or {})
        self.output_schema = dict(output_schema or {})

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Exécute l'outil.
        """
        pass

    def info(self):

        return {
            "name": self.name,
            "description": self.description,
        }

    def metadata(self):
        """
        Métadonnées complètes de l'outil (extension production).

        input_schema est complété par parameter_schema si celui-ci existe
        (rétrocompatibilité avec les outils existants).
        """

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self._resolved_input_schema(),
            "output_schema": self.output_schema,
            "timeout": self.timeout,
            "permissions": self.permissions,
            "estimated_cost": self.estimated_cost,
            "version": self.version,
        }

    def _resolved_input_schema(self):

        if self.input_schema:
            return self.input_schema

        try:
            return dict(getattr(self, "parameter_schema", {}) or {})
        except (AttributeError, TypeError):
            return {}