from typing import Dict, Any


class Validator:
    """
    Validator

    Vérifie qu'une action respecte
    la structure attendue par AstraExec.
    """

    REQUIRED_FIELDS = [
        "tool",
        "parameters",
    ]

    def validate_schema(self, parameters: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """
        Valide les paramètres selon un schéma de type.

        Le schéma est un dict de la forme :
        {
            "param_name": {
                "type": "string",       # type Python attendu
                "required": True,        # champ obligatoire
                "allowed": ["a", "b"],  # valeurs autorisées (optionnel)
                "min_length": 1,         # longueur minimale (string)
                "max_length": 500,       # longueur maximale (string)
                "description": "...",   # description
            }
        }
        """
        if not isinstance(parameters, dict):
            raise TypeError("Les paramètres doivent être un dictionnaire.")

        for param_name, rules in schema.items():
            is_required = rules.get("required", False)
            param_type = rules.get("type", "string")

            if is_required and param_name not in parameters:
                raise ValueError(
                    f"Paramètre obligatoire manquant : {param_name}"
                )

            if param_name in parameters:
                value = parameters[param_name]

                # Validation du type
                type_map = {
                    "string": str,
                    "integer": int,
                    "number": (int, float),
                    "boolean": bool,
                    "list": list,
                    "dict": dict,
                }
                expected_type = type_map.get(param_type, str)
                if not isinstance(value, expected_type):
                    raise TypeError(
                        f"Le paramètre '{param_name}' doit être de type {param_type}, "
                        f"reçu : {type(value).__name__}"
                    )

                # Validation des valeurs autorisées
                allowed = rules.get("allowed")
                if allowed is not None and value not in allowed:
                    raise ValueError(
                        f"Le paramètre '{param_name}' doit être l'une des valeurs : {allowed}, "
                        f"reçu : {value}"
                    )

                # Validation de la longueur (uniquement pour les chaînes)
                if param_type == "string" and isinstance(value, str):
                    min_len = rules.get("min_length", 0)
                    max_len = rules.get("max_length", 99999)
                    if len(value) < min_len:
                        raise ValueError(
                            f"Le paramètre '{param_name}' doit contenir au moins {min_len} caractère(s)"
                        )
                    if len(value) > max_len:
                        raise ValueError(
                            f"Le paramètre '{param_name}' doit contenir au maximum {max_len} caractère(s)"
                        )

        return True

    def validate(self, action: Dict[str, Any]) -> bool:

        if not isinstance(action, dict):
            raise TypeError(
                "Une action doit être un dictionnaire."
            )

        for field in self.REQUIRED_FIELDS:

            if field not in action:

                raise ValueError(
                    f"Champ obligatoire manquant : {field}"
                )

        if not isinstance(action["tool"], str):

            raise TypeError(
                "Le nom de l'outil doit être une chaîne."
            )

        if not isinstance(action["parameters"], dict):

            raise TypeError(
                "Les paramètres doivent être un dictionnaire."
            )

        return True

    # =======================================================
    # Vérification des paramètres
    # =======================================================

    def validate_parameters(
        self,
        parameters: Dict[str, Any],
        required=None,
    ):

        if required is None:
            required = []

        for name in required:

            if name not in parameters:

                raise ValueError(
                    f"Paramètre obligatoire manquant : {name}"
                )

        return True

    # =======================================================
    # Vérification d'un outil
    # =======================================================

    def validate_tool_name(
        self,
        tool_name: str,
    ):

        if len(tool_name.strip()) == 0:

            raise ValueError(
                "Nom d'outil vide."
            )

        return True

    # =======================================================
    # Validation complète
    # =======================================================

    def full_validation(
        self,
        action,
        required_params=None,
        schema=None,
    ):

        self.validate(action)

        self.validate_tool_name(
            action["tool"]
        )

        self.validate_parameters(
            action["parameters"],
            required_params,
        )

        if schema is not None:
            self.validate_schema(
                action["parameters"],
                schema,
            )

        return True


if __name__ == "__main__":

    validator = Validator()

    action = {

        "tool": "fusion_search",

        "parameters": {

            "query": "BM25"

        }

    }

    validator.full_validation(
        action,
        ["query"]
    )

    print("Validation réussie.")