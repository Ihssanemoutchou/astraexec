"""
Tests pour LexiRank (BM25 custom).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.retrieval.lexi_rank import LexiRank


def make_chunks(texts):
    return [
        {"chunk_id": i, "content": t, "length": len(t), "source": "test.txt", "word_count": len(t.split())}
        for i, t in enumerate(texts)
    ]


class TestLexiRank:
    def test_build_index(self):
        chunks = make_chunks(["Python est un langage", "FAISS est un outil"])
        lr = LexiRank()
        lr.build_index(chunks)
        assert len(lr.tokenized_docs) == 2
        assert len(lr.idf) > 0
        assert lr.avg_doc_length > 0

    def test_search_normal(self):
        chunks = make_chunks([
            "Python est un langage de programmation",
            "FAISS est un outil de recherche vectorielle",
        ])
        lr = LexiRank()
        lr.build_index(chunks)
        results = lr.search("python", top_k=2)
        assert len(results) == 2
        assert results[0]["score"] >= results[1]["score"]

    def test_search_empty_index(self):
        lr = LexiRank()
        lr.build_index([])
        # build_index sur liste vide → tokenized_docs = [] → ValueError
        import traceback
        try:
            results = lr.search("python")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_search_empty_query(self):
        chunks = make_chunks(["Python est un langage"])
        lr = LexiRank()
        lr.build_index(chunks)
        results = lr.search("")
        assert results == []

    def test_tokenize(self):
        lr = LexiRank()
        tokens = lr.tokenize("Bonjour le monde!")
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    def test_vocabulary_size(self):
        chunks = make_chunks(["Python est un langage", "Java est un langage"])
        lr = LexiRank()
        lr.build_index(chunks)
        assert lr.vocabulary_size() > 0

    def test_custom_bonus(self):
        lr = LexiRank()
        chunk = {"content": "Python est un langage", "length": 500, "chunk_id": 0}
        bonus = lr.custom_bonus(chunk, ["python"])
        assert bonus > 0

    def test_info(self):
        lr = LexiRank()
        lr.build_index(make_chunks(["Test"]))
        info = lr.info()
        assert "engine" in info
        assert "vocabulary" in info
        assert "documents" in info

    def test_default_params(self):
        lr = LexiRank()
        assert lr.k1 == 1.5
        assert lr.b == 0.75

    def test_custom_params(self):
        lr = LexiRank(k1=2.0, b=0.5)
        assert lr.k1 == 2.0
        assert lr.b == 0.5
