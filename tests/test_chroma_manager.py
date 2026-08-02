"""
Tests pour ChromaManager (client éphémère injecté).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from app.storage.chroma_manager import ChromaManager, DEFAULT_COLLECTION


def _ephemeral_factory(path):
    """Client ChromaDB en mémoire pour les tests.

    NB : chromadb n'est importé ici que pour construire le client de test.
    La règle d'encapsulation (seul chroma_manager.py importe chromadb dans
    le code applicatif app/) reste respectée.
    """
    import chromadb
    from chromadb.config import Settings

    return chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))


def _chunk(source, chunk_id, content, length=50, word_count=10):
    return {
        "source": source,
        "chunk_id": chunk_id,
        "length": length,
        "word_count": word_count,
        "content": content,
    }


def _sample_chunks(n=3, source="machine_learning.txt"):
    return [
        _chunk(source, i, f"Contenu du chunk {i} sur le machine learning.")
        for i in range(n)
    ]


def _unit_vectors(n, dim=384):
    """Vecteurs unitaires sur des axes distincts (similarité cosine nette)."""
    vectors = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        vectors[i, i] = 1.0
    return vectors


@pytest.fixture()
def manager():
    chroma = ChromaManager(
        path=":memory:",
        collection_name=DEFAULT_COLLECTION,
        client_factory=_ephemeral_factory,
    )
    yield chroma
    chroma.close()


class TestChromaManagerInitialisation:
    def test_attributs_par_defaut(self):
        chroma = ChromaManager()
        assert chroma.path == "storage/chroma"
        assert chroma.collection_name == "astra_docs"
        assert chroma._client is None  # création paresseuse
        chroma.close()

    def test_count_sans_collection(self, manager):
        assert manager.count() == 0

    def test_search_sans_collection(self, manager):
        query = np.zeros(384, dtype=np.float32)
        assert manager.search(query, top_k=3) == []


class TestChromaManagerBuild:
    def test_build_et_count(self, manager):
        chunks = _sample_chunks(3)
        manager.build(chunks, _unit_vectors(3))
        assert manager.count() == 3

    def test_build_idempotent(self, manager):
        chunks = _sample_chunks(3)
        manager.build(chunks, _unit_vectors(3))
        manager.build(chunks, _unit_vectors(3))  # même lot → pas de doublon
        assert manager.count() == 3

    def test_build_erreur_chunks_vides(self, manager):
        with pytest.raises(ValueError):
            manager.build([], _unit_vectors(0))

    def test_build_erreur_tailles_incoherentes(self, manager):
        with pytest.raises(ValueError):
            manager.build(_sample_chunks(3), _unit_vectors(2))

    def test_build_erreur_embedding_1d(self, manager):
        with pytest.raises(ValueError):
            manager.build(_sample_chunks(3), np.zeros(384, dtype=np.float32))


class TestChromaManagerAddChunks:
    def test_add_chunks_incremental(self, manager):
        manager.build(_sample_chunks(2), _unit_vectors(2))
        manager.add_chunks([_chunk("autre.txt", 0, "Nouveau document.")], _unit_vectors(1))
        assert manager.count() == 3

    def test_add_chunks_upsert_meme_id(self, manager):
        """Même id source::chunk_id → mise à jour, pas de doublon."""
        chunks = _sample_chunks(1)
        manager.build(chunks, _unit_vectors(1))
        manager.add_chunks(
            [_chunk("machine_learning.txt", 0, "Contenu mis à jour.")],
            _unit_vectors(1),
        )
        assert manager.count() == 1

        query = _unit_vectors(1)[0]
        results = manager.search(query, top_k=1)
        assert results[0]["content"] == "Contenu mis à jour."


class TestChromaManagerSearch:
    def test_search_pertinence(self, manager):
        """Le chunk le plus proche du vecteur de requête est renvoyé en tête."""
        chunks = _sample_chunks(3)
        manager.build(chunks, _unit_vectors(3))

        query = _unit_vectors(3)[0]  # s'aligne sur le chunk 0
        results = manager.search(query, top_k=3)

        assert len(results) == 3
        assert results[0]["chunk_id"] == 0
        assert results[0]["source"] == "machine_learning.txt"
        assert results[0]["content"] == "Contenu du chunk 0 sur le machine learning."
        assert results[0]["distance"] == pytest.approx(0.0, abs=1e-6)

    def test_search_top_k(self, manager):
        manager.build(_sample_chunks(3), _unit_vectors(3))
        query = _unit_vectors(3)[0]
        results = manager.search(query, top_k=1)
        assert len(results) == 1

    def test_search_top_k_invalide(self, manager):
        query = np.zeros(384, dtype=np.float32)
        with pytest.raises(ValueError):
            manager.search(query, top_k=0)

    def test_search_metadonnees_preservees(self, manager):
        chunks = _sample_chunks(2)
        manager.build(chunks, _unit_vectors(2))
        query = _unit_vectors(2)[0]
        results = manager.search(query, top_k=1)
        assert results[0]["length"] == 50
        assert results[0]["word_count"] == 10

    def test_search_embedding_invalide(self, manager):
        with pytest.raises(ValueError):
            manager.search(np.zeros((1, 384), dtype=np.float32))  # 2D


class TestChromaManagerControle:
    def test_delete_collection(self, manager):
        manager.build(_sample_chunks(2), _unit_vectors(2))
        assert manager.count() == 2
        manager.delete_collection()
        assert manager.count() == 0

    def test_delete_collection_sans_collection(self, manager):
        manager.delete_collection()  # ne doit pas lever

    def test_info(self, manager):
        info = manager.info()
        assert info["engine"] == "ChromaManager (ChromaDB)"
        assert info["collection_name"] == "astra_docs"
        assert info["space"] == "cosine"
        assert info["count"] == 0

        manager.build(_sample_chunks(2), _unit_vectors(2))
        info = manager.info()
        assert info["count"] == 2

    def test_close_et_reutilisation(self, manager):
        manager.build(_sample_chunks(2), _unit_vectors(2))
        manager.close()
        # Le client est recréé paresseusement au prochain usage.
        assert manager.count() == 0
        manager.build(_sample_chunks(1), _unit_vectors(1))
        assert manager.count() == 1
