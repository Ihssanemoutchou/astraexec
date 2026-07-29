"""
LexiRank — Custom BM25 search engine
=====================================

Implémentation maison de BM25Okapi.
Aucune dépendance externe (ni rank_bm25, ni sklearn).

Pipeline :
  - Tokenisation maison
  - Calcul des IDF
  - Scoring BM25
  - Bonus de proximité personnalisé
"""

import math
import re
from typing import List, Dict, Tuple


class LexiRank:
    """
    LexiRank

    Moteur de recherche lexicale développé pour AstraExec.

    Pipeline :
    - Prétraitement
    - Tokenisation
    - Index BM25
    - Score personnalisé
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        self.documents: List[Dict] = []
        self.tokenized_docs: List[List[str]] = []
        self.idf: Dict[str, float] = {}
        self.avg_doc_length: float = 0.0

    # ------------------------------------------------------------------
    # Tokenisation maison
    # ------------------------------------------------------------------

    def normalize(self, text: str) -> str:
        text = text.lower()
        # Conserve les lettres latines, les accents français et les chiffres
        text = re.sub(r"[^a-zA-Z0-9À-ÿ'èéêëàâùûüôöîïç]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def tokenize(self, text: str) -> List[str]:
        return self.normalize(text).split()

    # ------------------------------------------------------------------
    # Calcul des IDF (formule BM25 classique)
    # ------------------------------------------------------------------

    def compute_idf(self, doc_freq: int, total_docs: int) -> float:
        """
        IDF selon Robertson / Sparck Jones.
        """
        return math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))

    # ------------------------------------------------------------------
    # Construction de l'index
    # ------------------------------------------------------------------

    def build_index(self, chunks: List[Dict]):
        self.documents = chunks

        # Tokenisation
        self.tokenized_docs = [
            self.tokenize(chunk["content"]) for chunk in chunks
        ]

        # Longueur moyenne
        total_length = sum(len(tokens) for tokens in self.tokenized_docs)
        self.avg_doc_length = total_length / len(self.tokenized_docs) if self.tokenized_docs else 1.0

        # Fréquence documentaire (df)
        doc_freq: Dict[str, int] = {}
        for tokens in self.tokenized_docs:
            seen = set()
            for token in tokens:
                if token not in seen:
                    doc_freq[token] = doc_freq.get(token, 0) + 1
                    seen.add(token)

        # Calcul IDF pour chaque terme
        n = len(self.tokenized_docs)
        self.idf = {
            term: self.compute_idf(freq, n)
            for term, freq in doc_freq.items()
        }

    # ------------------------------------------------------------------
    # Score BM25 pour un (terme, document)
    # ------------------------------------------------------------------

    def bm25_term_score(self, term: str, doc_tokens: List[str], doc_length: int) -> float:
        if term not in self.idf:
            return 0.0

        tf = doc_tokens.count(term)
        if tf == 0:
            return 0.0

        idf_val = self.idf[term]
        numerator = tf * (self.k1 + 1)
        denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))

        return idf_val * (numerator / denominator)

    # ------------------------------------------------------------------
    # Bonus personnalisé (exact match bonus)
    # ------------------------------------------------------------------

    def custom_bonus(self, chunk: Dict, query_words: List[str]) -> float:
        """
        Récompense les chunks qui contiennent exactement les mots de la requête.
        """
        bonus = 0.0
        text = chunk["content"].lower()

        for word in query_words:
            # Bonus pour présence exacte
            if word in text:
                bonus += 0.15

        # Bonus pour chunks de bonne taille
        if 300 <= chunk.get("length", 0) <= 800:
            bonus += 0.10

        return bonus

    # ------------------------------------------------------------------
    # Recherche
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.tokenized_docs:
            raise ValueError("Index non construit. Appelez build_index() d'abord.")

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        results = []

        for idx, doc_tokens in enumerate(self.tokenized_docs):
            doc_length = len(doc_tokens)

            # Score BM25 pour chaque terme de la requête
            score = sum(
                self.bm25_term_score(term, doc_tokens, doc_length)
                for term in query_tokens
            )

            # Bonus maison
            bonus = self.custom_bonus(self.documents[idx], query_tokens)
            final_score = score + bonus

            results.append({
                "chunk": self.documents[idx],
                "score": final_score,
                "raw_bm25": score,
                "bonus": bonus,
            })

        # Tri descendant
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def vocabulary_size(self) -> int:
        return len(self.idf)

    def info(self) -> Dict:
        return {
            "engine": "LexiRank (BM25 custom)",
            "k1": self.k1,
            "b": self.b,
            "vocabulary": self.vocabulary_size(),
            "documents": len(self.documents),
        }


# ======================================================
# Test
# ======================================================

if __name__ == "__main__":

    fake_chunks = [
        {
            "chunk_id": 0,
            "length": 520,
            "content": "Python est un langage de programmation.",
        },
        {
            "chunk_id": 1,
            "length": 410,
            "content": "FAISS est utilisé pour la recherche vectorielle.",
        },
        {
            "chunk_id": 2,
            "length": 480,
            "content": "BM25 est une méthode de recherche lexicale.",
        },
    ]

    retriever = LexiRank()

    retriever.build_index(fake_chunks)

    results = retriever.search("python")

    for result in results:

        print("=" * 50)

        print(result["score"])

        print(result["chunk"]["content"])