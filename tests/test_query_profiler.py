"""
Tests pour QueryProfiler.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.retrieval.query_profiler import QueryProfiler


class TestQueryProfiler:
    def test_profile_keyword_short(self):
        profiler = QueryProfiler()
        profile = profiler.profile("machine learning")
        assert profile["type"] == "keyword"

    def test_profile_keyword_one_word(self):
        profiler = QueryProfiler()
        profile = profiler.profile("BM25")
        assert profile["type"] == "keyword"

    def test_profile_definition(self):
        profiler = QueryProfiler()
        profile = profiler.profile("Qu'est-ce que le deep learning ?")
        assert profile["type"] == "definition"

    def test_profile_definition_english(self):
        profiler = QueryProfiler()
        profile = profiler.profile("What is machine learning?")
        assert profile["type"] == "definition"

    def test_profile_comparative(self):
        profiler = QueryProfiler()
        profile = profiler.profile("Comparaison entre BM25 et TF-IDF")
        assert profile["type"] == "comparative"

    def test_profile_comparative_vs(self):
        profiler = QueryProfiler()
        profile = profiler.profile("Python vs Java")
        assert profile["type"] == "comparative"

    def test_profile_explanatory(self):
        profiler = QueryProfiler()
        profile = profiler.profile("Comment fonctionne le machine learning ?")
        assert profile["type"] == "explanatory"

    def test_profile_explanatory_pourquoi(self):
        profiler = QueryProfiler()
        profile = profiler.profile("Pourquoi utiliser Python ?")
        assert profile["type"] == "explanatory"

    def test_profile_auto_semantic(self):
        """Les requêtes longues sans mot-clé spécial sont 'semantic'."""
        profiler = QueryProfiler()
        profile = profiler.profile("Les algorithmes de machine learning supervisé")
        # 5 mots → pas keyword (<=2), pas de mot-clé spécial → semantic
        assert profile["type"] == "semantic"

    def test_profile_length(self):
        profiler = QueryProfiler()
        profile = profiler.profile("machine learning")
        assert profile["length"] == 2

    def test_profile_query_preserved(self):
        profiler = QueryProfiler()
        profile = profiler.profile("  Test Query  ")
        assert profile["query"] == "Test Query"
