from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    """
    Classe abstraite représentant un outil AstraExec.
    """

    def __init__(self, name: str, description: str):

        self.name = name
        self.description = description

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