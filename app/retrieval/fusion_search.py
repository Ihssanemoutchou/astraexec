"""
FusionSearch — Recherche hybride maison
========================================

Remplace FAISS + SentenceTransformers par un système vectoriel 100% custom.

Pipeline :
  1. Vectorisation TF-IDF maison (matrice terme-document)
  2. Index vectoriel via cosinus (numpy)
  3. Recherche lexicale via LexiRank (BM25 custom)
  4. Fusion pondérée des scores
"""

import math
import os
import re
import pickle
from typing import List, Dict, Tuple, Optional

import numpy as np

from app.retrieval.lexi_rank import LexiRank


class TermVectorizer:
    """
    Vectoriseur TF-IDF maison.

    Transforme une collection de textes en vecteurs TF-IDF
    sans utiliser scikit-learn.
    """

    def __init__(self):
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.dimension: int = 0

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9À-ÿ'èéêëàâùûüôöîïç]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip().split()

    # ------------------------------------------------------------------
    # Construction du vocabulaire
    # ------------------------------------------------------------------

    def build_vocabulary(self, texts: List[str]):
        token_set = set()
        for text in texts:
            tokens = self.tokenize(text)
            token_set.update(tokens)

        self.vocabulary = {word: idx for idx, word in enumerate(sorted(token_set))}
        self.dimension = len(self.vocabulary)

    # ------------------------------------------------------------------
    # Calcul IDF
    # ------------------------------------------------------------------

    def compute_idf(self, texts: List[str]):
        n = len(texts)
        doc_freq = {word: 0 for word in self.vocabulary}

        for text in texts:
            tokens = set(self.tokenize(text))
            for token in tokens:
                if token in doc_freq:
                    doc_freq[token] += 1

        self.idf = {
            word: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for word, freq in doc_freq.items()
        }

    # ------------------------------------------------------------------
    # Transformation d'un texte en vecteur TF-IDF
    # ------------------------------------------------------------------

    def transform(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = self.tokenize(text)

        # Comptage des TF
        tf_counter = {}
        for token in tokens:
            if token in self.vocabulary:
                tf_counter[token] = tf_counter.get(token, 0) + 1

        max_tf = max(tf_counter.values()) if tf_counter else 1

        # TF-IDF normalisé
        for token, tf in tf_counter.items():
            idx = self.vocabulary[token]
            tf_norm = tf / max_tf  # normalisation par le max
            idf_val = self.idf.get(token, 0)
            vector[idx] = tf_norm * idf_val

        return vector

    # ------------------------------------------------------------------
    # Transformation batch
    # ------------------------------------------------------------------

    def transform_batch(self, texts: List[str]) -> np.ndarray:
        vectors = [self.transform(text) for text in texts]
        return np.array(vectors, dtype=np.float32)

    # ------------------------------------------------------------------
    # Fit complet
    # ------------------------------------------------------------------

    def fit(self, texts: List[str]):
        self.build_vocabulary(texts)
        self.compute_idf(texts)
        return self.transform_batch(texts)


class FusionSearch:
    """
    Moteur de recherche hybride.

    Combine :
    - Recherche vectorielle (TF-IDF + cosinus via numpy)
    - Recherche lexicale (BM25 custom via LexiRank)
    """

    def __init__(self):
        self.vectorizer = TermVectorizer()
        self.index_vectors: np.ndarray = np.array([])
        self.documents: List[Dict] = []
        self.lexical = LexiRank()
        self.index_built = False

    # ------------------------------------------------------------------
    # Construction des index
    # ------------------------------------------------------------------

    def build_index(self, chunks: List[Dict]):
        self.documents = chunks

        # Index lexical
        self.lexical.build_index(chunks)

        # Index vectoriel
        texts = [chunk["content"] for chunk in chunks]
        self.index_vectors = self.vectorizer.fit(texts)
        self.index_built = True

        print(f"[FusionSearch] Index construit : {len(chunks)} chunks, "
              f"vocabulaire: {self.vectorizer.dimension} termes")

    # ------------------------------------------------------------------
    # Recherche vectorielle (cosinus)
    # ------------------------------------------------------------------

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.index_built:
            raise ValueError("Index non construit.")

        query_vector = self.vectorizer.transform(query)
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector_norm = query_vector / query_norm
        else:
            query_vector_norm = query_vector

        doc_norms = np.linalg.norm(self.index_vectors, axis=1)
        doc_norms = np.maximum(doc_norms, 1e-10)

        cos_similarities = (self.index_vectors @ query_vector_norm) / doc_norms

        # Top-k indices
        top_indices = np.argsort(cos_similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.documents[idx],
                "semantic_score": float(cos_similarities[idx]),
            })

        return results

    # ------------------------------------------------------------------
    # Fusion pondérée
    # ------------------------------------------------------------------

    def normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalisation min-max en [0, 1].
        Si tous les scores sont égaux, retourne 0.5 pour chacun.
        """
        if not scores:
            return []
        mn = min(scores)
        mx = max(scores)
        if mx == mn:
            return [0.5] * len(scores)
        return [(s - mn) / (mx - mn) for s in scores]

    def fusion_score(self, semantic_score: float, lexical_score: float,
                     semantic_weight: float = 0.50,
                     lexical_weight: float = 0.50) -> float:
        """
        Fusion pondérée avec poids ajustables.
        Les scores doivent être normalisés avant l'appel.
        """
        return semantic_score * semantic_weight + lexical_score * lexical_weight

    def set_weights(self, profile: Optional[Dict] = None):
        """
        Ajuste les poids selon le profil de requête.
        Retourne (semantic_weight, lexical_weight).
        """
        if profile is None:
            return 0.50, 0.50

        qtype = profile.get("type", "")
        if qtype == "keyword":
            # Mots-clés → poids fort sur le lexical
            return 0.25, 0.75
        elif qtype == "definition":
            # Définition → poids fort sur le vectoriel
            return 0.75, 0.25
        elif qtype == "comparative":
            # Comparaison → équilibré
            return 0.60, 0.40
        elif qtype == "explanatory":
            # Explication → plutôt vectoriel
            return 0.65, 0.35
        else:
            # Par défaut : équilibré
            return 0.50, 0.50

    # ------------------------------------------------------------------
    # Recherche hybride complète
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5,
               profile: Optional[Dict] = None) -> List[Dict]:
        """
        Recherche hybride complète.

        Paramètres :
            query   : requête utilisateur
            top_k   : nombre de résultats à retourner
            profile : profil de requête (QueryProfiler output)
                      Si fourni, les poids sont adaptés automatiquement.
        """
        # Ajustement des poids selon le profil
        sem_weight, lex_weight = self.set_weights(profile)

        # Recherche vectorielle (top_k * 2 pour avoir de la marge)
        semantic = self.semantic_search(query, top_k * 2)

        # Recherche lexicale
        lexical = self.lexical.search(query, top_k * 2)

        # Fusion des résultats
        merged: Dict[int, Dict] = {}

        # Résultats vectoriels
        for item in semantic:
            cid = item["chunk"]["chunk_id"]
            merged[cid] = {
                "chunk": item["chunk"],
                "semantic": item["semantic_score"],
                "lexical": 0.0,
            }

        # Résultats lexicaux
        for item in lexical:
            cid = item["chunk"]["chunk_id"]
            if cid not in merged:
                merged[cid] = {
                    "chunk": item["chunk"],
                    "semantic": 0.0,
                    "lexical": item["score"],
                }
            else:
                merged[cid]["lexical"] = item["score"]

        # Collecte des scores pour normalisation
        sem_scores = [v["semantic"] for v in merged.values()]
        lex_scores = [v["lexical"] for v in merged.values()]

        # Normalisation min-max des scores
        sem_norm = self.normalize_scores(sem_scores)
        lex_norm = self.normalize_scores(lex_scores)

        # Calcul du score final avec poids ajustés
        results = []
        for idx, (cid, value) in enumerate(merged.items()):
            score = self.fusion_score(
                sem_norm[idx], lex_norm[idx],
                semantic_weight=sem_weight,
                lexical_weight=lex_weight
            )
            results.append({
                "chunk": value["chunk"],
                "score": round(score, 4),
                "semantic": round(sem_norm[idx], 4),
                "lexical": round(lex_norm[idx], 4),
                "semantic_raw": round(value["semantic"], 4),
                "lexical_raw": round(value["lexical"], 4),
            })

        # Tri par score descendant
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    # ------------------------------------------------------------------
    # Persistance (NumPy + pickle)
    # ------------------------------------------------------------------

    def save(self, path: str = "index/fusion_search"):
        """
        Sauvegarde l'index complet sur disque.
        Utilise uniquement NumPy + pickle (pas de base vectorielle externe).
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        state = {
            "vectorizer_vocab": self.vectorizer.vocabulary,
            "vectorizer_idf": self.vectorizer.idf,
            "vectorizer_dim": self.vectorizer.dimension,
            "index_vectors": self.index_vectors,
            "documents": self.documents,
            "lexical_k1": self.lexical.k1,
            "lexical_b": self.lexical.b,
            "lexical_idf": self.lexical.idf,
            "lexical_avg_doc_length": self.lexical.avg_doc_length,
            "lexical_tokenized_docs": self.lexical.tokenized_docs,
        }

        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(state, f)
        print(f"[FusionSearch] Index sauvegardé : {path}.pkl")

    def load(self, path: str = "index/fusion_search"):
        """
        Charge l'index depuis le disque.
        """
        pkl_path = f"{path}.pkl"
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Index introuvable : {pkl_path}")

        with open(pkl_path, "rb") as f:
            state = pickle.load(f)

        # Restauration du vectorizer
        self.vectorizer.vocabulary = state["vectorizer_vocab"]
        self.vectorizer.idf = state["vectorizer_idf"]
        self.vectorizer.dimension = state["vectorizer_dim"]

        # Restauration de l'index vectoriel
        self.index_vectors = state["index_vectors"]
        self.documents = state["documents"]

        # Restauration du LexiRank
        self.lexical.k1 = state["lexical_k1"]
        self.lexical.b = state["lexical_b"]
        self.lexical.idf = state["lexical_idf"]
        self.lexical.avg_doc_length = state["lexical_avg_doc_length"]
        self.lexical.tokenized_docs = state["lexical_tokenized_docs"]
        self.lexical.documents = self.documents

        self.index_built = True
        print(f"[FusionSearch] Index chargé : {len(self.documents)} chunks, "
              f"vocabulaire: {self.vectorizer.dimension} termes")

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def info(self) -> Dict:
        return {
            "engine": "FusionSearch (vectoriel custom + BM25 custom)",
            "vocabulary": self.vectorizer.dimension,
            "documents": len(self.documents),
            "index_built": self.index_built,
        }


# =====================================================
# Test
# =====================================================

if __name__ == "__main__":

    chunks = [
        {
            "chunk_id": 0,
            "length": 520,
            "content": "Python est un langage de programmation."
        },
        {
            "chunk_id": 1,
            "length": 430,
            "content": "FAISS permet une recherche vectorielle rapide."
        },
        {
            "chunk_id": 2,
            "length": 410,
            "content": "BM25 est utilisé pour la recherche lexicale."
        }
    ]

    fusion = FusionSearch()

    fusion.build_index(chunks)

    results = fusion.search("python")

    for r in results:

        print("=" * 60)
        print("Score :", r["score"])
        print("Semantic :", r["semantic"])
        print("Lexical :", r["lexical"])
        print(r["chunk"]["content"])