"""
Tests pour EvidenceRank.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.retrieval.evidence_rank import EvidenceRank


def make_result(semantic, lexical, chunk_id, length, content):
    return {
        "chunk": {"chunk_id": chunk_id, "length": length, "content": content},
        "semantic": semantic,
        "lexical": lexical,
    }


class TestEvidenceRank:
    def test_quality_score_ideal(self):
        ranker = EvidenceRank()
        score = ranker.quality_score({"length": 500, "chunk_id": 0})
        assert score == 1.0

    def test_quality_score_short(self):
        ranker = EvidenceRank()
        score = ranker.quality_score({"length": 250, "chunk_id": 0})
        assert score == 0.80

    def test_quality_score_long(self):
        ranker = EvidenceRank()
        score = ranker.quality_score({"length": 900, "chunk_id": 0})
        assert score == 0.75

    def test_quality_score_too_short(self):
        ranker = EvidenceRank()
        score = ranker.quality_score({"length": 50, "chunk_id": 0})
        assert score == 0.50

    def test_position_score_first(self):
        ranker = EvidenceRank()
        score = ranker.position_score({"chunk_id": 0})
        assert score == 1.0

    def test_position_score_mid(self):
        ranker = EvidenceRank()
        score = ranker.position_score({"chunk_id": 5})
        assert score == 0.80

    def test_position_score_late(self):
        ranker = EvidenceRank()
        score = ranker.position_score({"chunk_id": 10})
        assert score == 0.60

    def test_compute_score(self):
        ranker = EvidenceRank()
        result = make_result(0.8, 0.6, 1, 500, "Test content")
        score = ranker.compute_score(result)
        assert score > 0
        assert score < 2

    def test_rerank(self):
        ranker = EvidenceRank()
        results = [
            make_result(0.9, 0.3, 1, 500, "Premier document"),
            make_result(0.7, 0.8, 2, 500, "Second document"),
        ]
        ranked = ranker.rerank(results)
        assert len(ranked) == 2
        assert "final_score" in ranked[0]
        # Doit être trié par score descendant
        assert ranked[0]["final_score"] >= ranked[1]["final_score"]

    def test_rerank_empty(self):
        ranker = EvidenceRank()
        ranked = ranker.rerank([])
        assert ranked == []

    def test_custom_weights(self):
        ranker = EvidenceRank(semantic_weight=0.7, lexical_weight=0.2,
                               quality_weight=0.05, position_weight=0.05)
        assert ranker.semantic_weight == 0.7
        assert ranker.lexical_weight == 0.2
