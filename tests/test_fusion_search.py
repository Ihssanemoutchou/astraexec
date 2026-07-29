"""
Tests pour FusionSearch.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.retrieval.fusion_search import FusionSearch, TermVectorizer


def make_chunks(texts):
    return [
        {"chunk_id": i, "content": t, "length": len(t), "source": "test.txt", "word_count": len(t.split())}
        for i, t in enumerate(texts)
    ]


class TestTermVectorizer:
    def test_tokenize(self):
        tv = TermVectorizer()
        tokens = tv.tokenize("Bonjour, je suis un test!")
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    def test_build_vocabulary(self):
        tv = TermVectorizer()
        tv.build_vocabulary(["hello world", "world of python"])
        assert len(tv.vocabulary) > 0
        assert "hello" in tv.vocabulary
        assert "world" in tv.vocabulary

    def test_transform_dimension(self):
        tv = TermVectorizer()
        tv.build_vocabulary(["hello world"])
        tv.compute_idf(["hello world"])
        vec = tv.transform("hello")
        assert vec.shape[0] == tv.dimension

    def test_empty_text(self):
        tv = TermVectorizer()
        tv.build_vocabulary(["hello"])
        tv.compute_idf(["hello"])
        vec = tv.transform("")
        assert vec.shape[0] == tv.dimension
        assert (vec == 0).all()


class TestFusionSearch:
    def test_build_index(self):
        chunks = make_chunks(["Python est un langage", "FAISS est un outil"])
        fs = FusionSearch()
        fs.build_index(chunks)
        assert fs.index_built is True
        assert len(fs.documents) == 2

    def test_search_normal(self):
        chunks = make_chunks([
            "Python est un langage de programmation",
            "FAISS permet la recherche vectorielle",
            "BM25 est utilisé pour la recherche lexicale",
            "L'apprentissage automatique est lié au machine learning",
        ])
        fs = FusionSearch()
        fs.build_index(chunks)
        results = fs.search("python", top_k=2)
        assert len(results) <= 2
        assert all("score" in r for r in results)
        assert all("chunk" in r for r in results)
        # Le meilleur résultat doit être celui avec "python"
        assert "python" in results[0]["chunk"]["content"].lower()

    def test_search_empty_query(self):
        chunks = make_chunks(["Python est un langage"])
        fs = FusionSearch()
        fs.build_index(chunks)
        # Une requête vide retourne des résultats avec score de base
        results = fs.search("", top_k=5)
        # Le moteur retourne les résultats (score neutre)
        assert isinstance(results, list)
        # Et pas d'erreur
        fs.search("", top_k=5)

    def test_search_no_match(self):
        chunks = make_chunks(["Python est un langage"])
        fs = FusionSearch()
        fs.build_index(chunks)
        results = fs.search("xyznonexistent123", top_k=5)
        # Doit retourner des résultats même sans match parfait
        assert isinstance(results, list)

    def test_normalize_scores(self):
        fs = FusionSearch()
        normalized = fs.normalize_scores([0.0, 0.5, 1.0])
        assert normalized == [0.0, 0.5, 1.0]

        normalized = fs.normalize_scores([5.0, 10.0, 15.0])
        assert normalized == [0.0, 0.5, 1.0]

        # Tous égaux
        normalized = fs.normalize_scores([3.0, 3.0, 3.0])
        assert normalized == [0.5, 0.5, 0.5]

        # Liste vide
        normalized = fs.normalize_scores([])
        assert normalized == []

    def test_set_weights(self):
        fs = FusionSearch()
        sw, lw = fs.set_weights({"type": "keyword"})
        assert sw < lw  # keyword → plus de lexical

        sw, lw = fs.set_weights({"type": "definition"})
        assert sw > lw  # definition → plus de vectoriel

        sw, lw = fs.set_weights({"type": "explanatory"})
        assert sw > lw

        sw, lw = fs.set_weights(None)
        assert sw == 0.50 and lw == 0.50

    def test_search_with_profile(self):
        chunks = make_chunks([
            "Python est un langage de programmation",
            "La programmation Python permet le machine learning",
        ])
        fs = FusionSearch()
        fs.build_index(chunks)
        results = fs.search("python", top_k=2, profile={"type": "keyword"})
        assert len(results) > 0
        assert "score" in results[0]

    def test_index_not_built(self):
        fs = FusionSearch()
        try:
            fs.semantic_search("test")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_save_load(self, tmp_path):
        chunks = make_chunks(["Python est un langage"])
        fs = FusionSearch()
        fs.build_index(chunks)

        save_path = str(tmp_path / "test_index")
        fs.save(save_path)

        fs2 = FusionSearch()
        fs2.load(save_path)
        assert fs2.index_built is True
        assert len(fs2.documents) == 1
        assert fs2.vectorizer.dimension == fs.vectorizer.dimension

    def test_rerank_integration(self):
        from app.retrieval.evidence_rank import EvidenceRank
        chunks = make_chunks([
            "Python est un langage de programmation",
            "Machine learning avec Python",
        ])
        fs = FusionSearch()
        fs.build_index(chunks)

        results = fs.search("python", top_k=5)
        ranker = EvidenceRank()
        ranked = ranker.rerank(results)
        assert len(ranked) > 0
        assert "final_score" in ranked[0]
