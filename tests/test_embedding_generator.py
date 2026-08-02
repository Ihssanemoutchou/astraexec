"""
Tests pour EmbeddingGenerator.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from app.storage.embedding_generator import EmbeddingGenerator


@pytest.fixture(scope="module")
def generator():
    """Le modèle (≈ 90 Mo) est chargé une seule fois pour toute la session."""
    return EmbeddingGenerator()


class TestEmbeddingGenerator:
    def test_attributs(self, generator):
        assert generator.model_name == "all-MiniLM-L6-v2"
        assert generator.dimension == 384

    def test_embed_text_shape_et_dtype(self, generator):
        vector = generator.embed_text("Python est un langage de programmation.")
        assert isinstance(vector, np.ndarray)
        assert vector.shape == (384,)
        assert vector.dtype == np.float32

    def test_embed_texts_shape_et_dtype(self, generator):
        vectors = generator.embed_texts(["un", "deux", "trois"])
        assert isinstance(vectors, np.ndarray)
        assert vectors.shape == (3, 384)
        assert vectors.dtype == np.float32

    def test_determinisme_bit_a_bit(self, generator):
        """Contrat binôme : deux encodages du même texte sont identiques."""
        text = "La recherche lexicale utilise BM25."
        v1 = generator.embed_text(text)
        v2 = generator.embed_text(text)
        assert np.array_equal(v1, v2)

    def test_embed_text_equivaut_a_batch_singleton(self, generator):
        text = "La recherche vectorielle utilise le cosinus."
        single = generator.embed_text(text)
        batch = generator.embed_texts([text])
        assert np.allclose(single, batch[0])

    def test_info(self, generator):
        info = generator.info()
        assert info["engine"] == "EmbeddingGenerator (sentence-transformers)"
        assert info["model_name"] == "all-MiniLM-L6-v2"
        assert info["dimension"] == 384

    def test_liste_vide_raises(self, generator):
        with pytest.raises(ValueError):
            generator.embed_texts([])

    def test_texte_vide_raises(self, generator):
        with pytest.raises(ValueError):
            generator.embed_text("   ")

    def test_texte_vide_dans_batch_raises(self, generator):
        with pytest.raises(ValueError):
            generator.embed_texts(["ok", "   "])

    def test_mauvais_type_raises(self, generator):
        with pytest.raises(ValueError):
            generator.embed_text(42)
