"""
Tests pour le module d'évaluation AstraExec.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.evaluation.metrics import (
    recall_at_k,
    mean_reciprocal_rank,
    reciprocal_rank,
    Evaluator,
)


class TestRecallAtK:
    """Tests pour Recall@K."""

    def test_recall_at_1_found(self):
        """Le document pertinent est en première position."""
        assert recall_at_k([1, 2, 3], [1], 1) == 1.0

    def test_recall_at_1_not_found(self):
        """Aucun document pertinent dans le top 1."""
        assert recall_at_k([2, 3, 4], [1], 1) == 0.0

    def test_recall_at_3_half(self):
        """1 pertinent sur 4 dans le top 3 = 0.25 (seul 1 est dans retrieved[:3])."""
        assert recall_at_k([1, 2, 3, 4, 5], [1, 4, 6, 7], 3) == 0.25

    def test_recall_at_5_full(self):
        """Tous les pertinents sont dans le top 5."""
        assert recall_at_k([1, 2, 3, 4, 5], [1, 3], 5) == 1.0

    def test_recall_at_k_zero(self):
        """K=0 retourne 0.0."""
        assert recall_at_k([1, 2, 3], [1], 0) == 0.0

    def test_recall_at_k_negative(self):
        """K négatif retourne 0.0."""
        assert recall_at_k([1, 2, 3], [1], -1) == 0.0

    def test_recall_no_relevant(self):
        """Aucun pertinent attendu → 0.0."""
        assert recall_at_k([1, 2, 3], [], 3) == 0.0

    def test_recall_empty_retrieved(self):
        """Liste retournée vide → 0.0."""
        assert recall_at_k([], [1, 2], 3) == 0.0

    def test_recall_k_greater_than_retrieved(self):
        """K plus grand que le nombre de résultats retournés."""
        assert recall_at_k([1, 2], [1, 3], 10) == 0.5

    def test_recall_duplicates(self):
        """IDs en double dans retrieved (ne doit pas planter)."""
        assert recall_at_k([1, 1, 2, 3], [1], 2) == 1.0

    def test_recall_with_chunk_ids(self):
        """Vérifie que ça marche avec des IDs non entiers (strings)."""
        assert recall_at_k(
            ["doc_a", "doc_b", "doc_c"], ["doc_a", "doc_d"], 1
        ) == 0.5


class TestReciprocalRank:
    """Tests pour Reciprocal Rank."""

    def test_rr_first_position(self):
        """Premier pertinent en position 1 → RR = 1.0."""
        assert reciprocal_rank([1, 2, 3], [1]) == 1.0

    def test_rr_second_position(self):
        """Premier pertinent en position 2 → RR = 0.5."""
        assert reciprocal_rank([2, 1, 3], [1]) == 0.5

    def test_rr_third_position(self):
        """Premier pertinent en position 3 → RR = 1/3."""
        assert reciprocal_rank([2, 3, 1], [1]) == 1.0 / 3.0

    def test_rr_not_found(self):
        """Aucun pertinent trouvé → RR = 0.0."""
        assert reciprocal_rank([2, 3, 4], [1]) == 0.0

    def test_rr_empty_retrieved(self):
        """Liste retournée vide → 0.0."""
        assert reciprocal_rank([], [1]) == 0.0

    def test_rr_multiple_relevant(self):
        """Plusieurs pertinents, on prend le premier rang."""
        assert reciprocal_rank([3, 1, 2], [1, 2]) == 0.5  # 1 est à l'index 1 → rang 2


class TestMeanReciprocalRank:
    """Tests pour MRR."""

    def test_mrr_all_first(self):
        """Tous les pertinents sont en première position → MRR = 1.0."""
        results = [
            {"retrieved_ids": [1, 2, 3], "relevant_ids": [1]},
            {"retrieved_ids": [2, 1, 3], "relevant_ids": [2]},
        ]
        assert mean_reciprocal_rank(results) == 1.0

    def test_mrr_mixed(self):
        """Mélange de positions : (1 + 0.5) / 2 = 0.75."""
        results = [
            {"retrieved_ids": [1, 2, 3], "relevant_ids": [1]},    # RR = 1.0
            {"retrieved_ids": [2, 1, 3], "relevant_ids": [1]},    # RR = 0.5
        ]
        assert mean_reciprocal_rank(results) == 0.75

    def test_mrr_one_not_found(self):
        """Un trouvé, un pas trouvé : (1 + 0) / 2 = 0.5."""
        results = [
            {"retrieved_ids": [1, 2, 3], "relevant_ids": [1]},    # RR = 1.0
            {"retrieved_ids": [2, 3, 4], "relevant_ids": [1]},    # RR = 0.0
        ]
        assert mean_reciprocal_rank(results) == 0.5

    def test_mrr_empty_list(self):
        """Liste de résultats vide → MRR = 0.0."""
        assert mean_reciprocal_rank([]) == 0.0

    def test_mrr_no_relevant(self):
        """Aucun pertinent pour toutes les requêtes → MRR = 0.0."""
        results = [
            {"retrieved_ids": [1, 2, 3], "relevant_ids": []},
            {"retrieved_ids": [4, 5, 6], "relevant_ids": []},
        ]
        assert mean_reciprocal_rank(results) == 0.0


class TestEvaluator:
    """Tests pour la classe Evaluator."""

    def test_add_query(self):
        evaluator = Evaluator()
        evaluator.add_query("test", [1, 2, 3], [1])
        assert len(evaluator.queries) == 1
        assert evaluator.queries[0]["query"] == "test"

    def test_evaluate_single_query(self):
        evaluator = Evaluator()
        evaluator.add_query("test", [1, 2, 3], [1])
        results = evaluator.evaluate(ks=[1, 3])
        assert results["total_queries"] == 1
        assert results["recall_at_k"][1] == 1.0
        assert results["recall_at_k"][3] == 1.0
        assert results["mrr"] == 1.0

    def test_evaluate_multiple_queries(self):
        evaluator = Evaluator()
        evaluator.add_query("q1", [1, 2, 3], [1])          # Recall@1=1.0, RR=1.0
        evaluator.add_query("q2", [2, 1, 3], [1])          # Recall@1=0.0, RR=0.5
        evaluator.add_query("q3", [1, 2, 3], [4])          # Recall@1=0.0, RR=0.0
        results = evaluator.evaluate(ks=[1, 3])
        assert results["total_queries"] == 3
        # Recall@1 moyen : (1.0 + 0.0 + 0.0) / 3 = 0.3333
        assert results["recall_at_k"][1] == 0.3333
        # Recall@3 moyen : (1.0 + 1.0 + 0.0) / 3 = 0.6667
        assert results["recall_at_k"][3] == 0.6667
        # MRR : (1.0 + 0.5 + 0.0) / 3 = 0.5
        assert results["mrr"] == 0.5

    def test_evaluate_empty(self):
        evaluator = Evaluator()
        results = evaluator.evaluate(ks=[1, 3, 5])
        assert results["total_queries"] == 0
        assert results["recall_at_k"] == {1: 0.0, 3: 0.0, 5: 0.0}
        assert results["mrr"] == 0.0

    def test_reset(self):
        evaluator = Evaluator()
        evaluator.add_query("test", [1, 2], [1])
        assert len(evaluator.queries) == 1
        evaluator.reset()
        assert len(evaluator.queries) == 0

    def test_summary_string(self):
        evaluator = Evaluator()
        evaluator.add_query("test", [1, 2, 3], [1])
        summary = evaluator.summary(ks=[1, 3])
        assert isinstance(summary, str)
        assert "Rapport" in summary
        assert "test" in summary
        assert "Recall@1" in summary

    def test_details_structure(self):
        evaluator = Evaluator()
        evaluator.add_query("my query", [5, 4, 3, 2, 1], [1, 5])
        results = evaluator.evaluate(ks=[1, 5])
        detail = results["details"][0]
        assert detail["query"] == "my query"
        assert detail["retrieved_count"] == 5
        assert detail["relevant_count"] == 2
        assert detail["reciprocal_rank"] == 1.0  # 5 en position 0 → rang 1
