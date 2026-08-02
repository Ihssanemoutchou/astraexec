"""
EmbeddingGenerator — Vectorisation des chunks
===============================================

MON outil de vectorisation, adossé à `sentence-transformers`.

Encapsule `SentenceTransformer("all-MiniLM-L6-v2")` (384 dimensions)
afin que les embeddings soient identiques entre l'étudiante et le binôme
(même bibliothèque, même modèle, mêmes vecteurs).

Pipeline :
    chunks (List[str])
        ↓
    embed_texts()
        ↓
    np.ndarray (N × 384, float32)

Ce composant est un transformateur pur :
  - il ne lit pas de fichiers    (→ ReaderFactory)
  - il ne segmente pas           (→ SmartSeg)
  - il ne persiste pas           (→ ChromaManager)
  - il ne recherche ni ne classe (→ ChromaManager / EvidenceRank)

Règles d'encapsulation :
  - seul fichier du projet autorisé à importer `sentence_transformers`
  - import différé et chargement paresseux du modèle (premier appel)
    pour préserver le démarrage de l'API et des démos existantes.
"""

from typing import Dict, List

import numpy as np


class EmbeddingGenerator:
    """
    EmbeddingGenerator

    Encapsule SentenceTransformer("all-MiniLM-L6-v2") (384 dimensions).

    API :
      - embed_text(text)    -> np.ndarray (384, float32)
      - embed_texts(texts)  -> np.ndarray (N, 384, float32)
      - info()              -> Dict

    Attributs :
      - model_name : nom du modèle (contrat partagé avec le binôme)
      - dimension  : taille des vecteurs produits
    """

    model_name = "all-MiniLM-L6-v2"
    dimension = 384

    def __init__(self):
        self._model = None

    # ------------------------------------------------------------------
    # Chargement différé du modèle
    # ------------------------------------------------------------------

    def _load_model(self):
        """Charge le modèle au premier appel (import différé)."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence_transformers est requis pour EmbeddingGenerator. "
                    "Installez-le : pip install sentence-transformers"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    # ------------------------------------------------------------------
    # Vectorisation
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> np.ndarray:
        """
        Encode un texte unique en un vecteur (384, float32).
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Le texte à encoder ne doit pas être vide.")
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode une liste de textes en une matrice (N, 384, float32).

        Le même modèle est utilisé pour l'indexation et les requêtes :
        les vecteurs sont identiques à ceux du binôme.
        """
        texts = list(texts)
        if not texts:
            raise ValueError("La liste de textes ne doit pas être vide.")
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    "Chaque texte à encoder doit être une chaîne non vide."
                )

        model = self._load_model()

        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def info(self) -> Dict:
        return {
            "engine": "EmbeddingGenerator (sentence-transformers)",
            "model_name": self.model_name,
            "dimension": self.dimension,
            "model_loaded": self._model is not None,
        }


# =====================================================
# Test
# =====================================================

if __name__ == "__main__":
    generator = EmbeddingGenerator()
    print(generator.info())

    vector = generator.embed_text("Python est un langage de programmation.")
    print(f"embed_text  : {vector.shape}, {vector.dtype}")

    batch = generator.embed_texts(["a", "b", "c"])
    print(f"embed_texts : {batch.shape}, {batch.dtype}")
