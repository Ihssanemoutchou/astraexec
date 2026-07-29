import re


class QueryProfiler:
    """
    Analyse une requête utilisateur afin d'adapter
    automatiquement la stratégie de recherche.
    """

    def profile(self, query: str) -> dict:

        query = query.strip()

        words = query.split()

        length = len(words)

        profile = {
            "query": query,
            "length": length,
            "type": "semantic"
        }

        # -----------------------------
        # Requête courte
        # -----------------------------
        if length <= 2:
            profile["type"] = "keyword"

        # -----------------------------
        # Comparaison
        # -----------------------------
        elif re.search(
            r"\b(compare|comparaison|différence|vs|versus)\b",
            query.lower()
        ):
            profile["type"] = "comparative"

        # -----------------------------
        # Définition
        # -----------------------------
        elif re.search(
            r"\b(qu'est-ce|définition|define|what is)\b",
            query.lower()
        ):
            profile["type"] = "definition"

        # -----------------------------
        # Explication
        # -----------------------------
        elif re.search(
            r"\b(pourquoi|comment|explique)\b",
            query.lower()
        ):
            profile["type"] = "explanatory"

        return profile