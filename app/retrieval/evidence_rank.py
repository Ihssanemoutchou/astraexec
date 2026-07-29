from typing import List, Dict


class EvidenceRank:
    """
    EvidenceRank

    Classement final des résultats récupérés.

    Combine plusieurs critères afin de produire
    un score plus robuste qu'un simple score BM25
    ou un simple score vectoriel.
    """

    def __init__(
        self,
        semantic_weight=0.50,
        lexical_weight=0.30,
        quality_weight=0.10,
        position_weight=0.10,
    ):

        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.quality_weight = quality_weight
        self.position_weight = position_weight

    # =======================================================
    # Qualité du chunk
    # =======================================================

    def quality_score(self, chunk):

        length = chunk["length"]

        if 300 <= length <= 800:
            return 1.0

        if 200 <= length < 300:
            return 0.80

        if length > 800:
            return 0.75

        return 0.50

    # =======================================================
    # Position du chunk
    # =======================================================

    def position_score(self, chunk):

        cid = chunk["chunk_id"]

        if cid <= 3:
            return 1.0

        if cid <= 8:
            return 0.80

        return 0.60

    # =======================================================
    # Score Final
    # =======================================================

    def compute_score(self, result):

        semantic = result.get("semantic", 0)

        lexical = result.get("lexical", 0)

        quality = self.quality_score(
            result["chunk"]
        )

        position = self.position_score(
            result["chunk"]
        )

        score = (

            semantic * self.semantic_weight

            +

            lexical * self.lexical_weight

            +

            quality * self.quality_weight

            +

            position * self.position_weight

        )

        return score

    # =======================================================
    # Ranking
    # =======================================================

    def rerank(
        self,
        results: List[Dict],
    ):

        ranked = []

        for result in results:

            result["final_score"] = self.compute_score(
                result
            )

            ranked.append(result)

        ranked.sort(

            key=lambda x: x["final_score"],

            reverse=True,

        )

        return ranked


# =======================================================
# Test
# =======================================================

if __name__ == "__main__":

    fake_results = [

        {

            "chunk": {
                "chunk_id": 1,
                "length": 520,
                "content": "Python..."
            },

            "semantic": 0.82,

            "lexical": 0.71,

        },

        {

            "chunk": {
                "chunk_id": 6,
                "length": 420,
                "content": "FAISS..."
            },

            "semantic": 0.91,

            "lexical": 0.35,

        }

    ]

    ranker = EvidenceRank()

    ranked = ranker.rerank(fake_results)

    for r in ranked:

        print("=" * 50)

        print(r["final_score"])

        print(r["chunk"]["content"])