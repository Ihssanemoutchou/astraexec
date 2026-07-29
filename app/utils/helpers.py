import uuid
import json
import time
from typing import Any, Dict


class Helpers:
    """
    Helpers

    Fonctions utilitaires communes utilisées
    dans AstraExec.
    """

    # =====================================================
    # Génération Request ID
    # =====================================================

    @staticmethod
    def generate_request_id():

        return str(uuid.uuid4())

    # =====================================================
    # Temps actuel
    # =====================================================

    @staticmethod
    def current_time():

        return time.time()

    # =====================================================
    # Calcul durée
    # =====================================================

    @staticmethod
    def elapsed(start):

        return round(time.time() - start, 4)

    # =====================================================
    # Nettoyage paramètres
    # =====================================================

    @staticmethod
    def clean_parameters(parameters: Dict[str, Any]):

        cleaned = {}

        for key, value in parameters.items():

            if isinstance(value, str):

                cleaned[key] = value.strip()

            else:

                cleaned[key] = value

        return cleaned

    # =====================================================
    # JSON
    # =====================================================

    @staticmethod
    def to_json(data):

        return json.dumps(

            data,

            indent=4,

            ensure_ascii=False,

        )

    # =====================================================
    # Depuis JSON
    # =====================================================

    @staticmethod
    def from_json(text):

        return json.loads(text)

    # =====================================================
    # Réponse standard
    # =====================================================

    @staticmethod
    def response(

        status,

        message,

        data=None,

    ):

        return {

            "status": status,

            "message": message,

            "data": data,

        }


if __name__ == "__main__":

    rid = Helpers.generate_request_id()

    print(rid)

    params = {

        "query": "   BM25   "

    }

    print(

        Helpers.clean_parameters(params)

    )

    print(

        Helpers.response(

            "success",

            "OK",

            {

                "query": "BM25"

            }

        )

    )