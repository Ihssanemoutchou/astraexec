from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class ActionInterface:
    """
    Interface standard d'une action AstraExec.

    Toute action envoyée à l'Executor
    doit respecter cette structure.
    """

    tool: str

    parameters: Dict[str, Any]

    priority: str = "normal"

    confidence: float = 1.0

    request_id: str = ""

    metadata: Dict[str, Any] = field(default_factory=dict)

    # ==========================================================
    # Validation
    # ==========================================================

    def validate(self):

        if not self.tool:

            raise ValueError("Tool obligatoire.")

        if not isinstance(self.parameters, dict):

            raise TypeError(
                "parameters doit être un dictionnaire."
            )

        if self.priority not in [

            "low",

            "normal",

            "high",

        ]:

            raise ValueError(

                "Priority invalide."

            )

        if not (0 <= self.confidence <= 1):

            raise ValueError(

                "Confidence doit être comprise entre 0 et 1."

            )

        return True

    # ==========================================================
    # Conversion JSON
    # ==========================================================

    def to_dict(self):

        return {

            "tool": self.tool,

            "parameters": self.parameters,

            "priority": self.priority,

            "confidence": self.confidence,

            "request_id": self.request_id,

            "metadata": self.metadata,

        }

    # ==========================================================
    # Construction depuis un dictionnaire
    # ==========================================================

    @classmethod
    def from_dict(cls, data):

        return cls(

            tool=data.get("tool"),

            parameters=data.get("parameters", {}),

            priority=data.get(

                "priority",

                "normal",

            ),

            confidence=data.get(

                "confidence",

                1.0,

            ),

            request_id=data.get(

                "request_id",

                "",

            ),

            metadata=data.get(

                "metadata",

                {},

            ),

        )


if __name__ == "__main__":

    action = ActionInterface(

        tool="fusion_search",

        parameters={

            "query": "Qu'est-ce que BM25 ?"

        },

        priority="high",

        confidence=0.94,

        request_id="REQ-001",

        metadata={

            "user": "demo"

        }

    )

    action.validate()

    print(action.to_dict())