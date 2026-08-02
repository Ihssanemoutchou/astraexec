"""
ChromaManager — Accès unique à ChromaDB
========================================

MON outil de persistance et de recherche vectorielle, adossé à ChromaDB.

ChromaDB est UNIQUEMENT le moteur de stockage (comme SQLite ou PostgreSQL) :
ce fichier est le SEUL endroit du projet autorisé à importer `chromadb`.
Toute la logique autour (embeddings, tri, fusion, export) reste dans mes
composants.

Responsabilités :
  - créer la collection (espace cosine)
  - ajouter des chunks vectorisés (build / add_chunks)
  - rechercher par similarité (query_embeddings, jamais query_texts)
  - compter, supprimer, fermer proprement

Il ne fait jamais :
  - d'embedding            (→ EmbeddingGenerator)
  - de reranking / fusion  (→ EvidenceRank / FusionSearch)
  - de logique métier      (→ DocumentManager / SmartSeg)

Pipeline :
    chunks + embeddings
        ↓
    build() / add_chunks()
        ↓
    collection "astra_docs" (espace cosine, ids "source::chunk_id")

Export : `close()` libère les verrous de fichiers (indispensable sur Windows
avant de zipper `storage/chroma/` avec BaseExporter).
"""

from typing import Callable, Dict, List, Optional

import numpy as np

DEFAULT_COLLECTION = "astra_docs"
DEFAULT_SPACE = "cosine"


class ChromaManager:
    """
    ChromaManager

    Encapsule ChromaDB : persistance (build, add_chunks) et recherche
    vectorielle (search) sur une collection unique.

    API :
      - build(chunks, embeddings)       : construction initiale idempotente
      - add_chunks(chunks, embeddings)  : ingestion incrémentale (upsert)
      - search(embedding, top_k)        : similarité par vecteur uniquement
      - count()                         : nombre de chunks indexés
      - delete_collection()             : reset de la collection
      - info()                          : état du manager
      - close()                         : libère le client (avant export zip)

    Chaque chunk est identifié par "source::chunk_id" (upsert = pas de
    doublon lors d'une ré-indexation).
    """

    def __init__(
        self,
        path: str = "storage/chroma",
        collection_name: str = DEFAULT_COLLECTION,
        client_factory: Optional[Callable] = None,
    ):
        """
        Paramètres :
          - path           : dossier de persistance ChromaDB
          - collection_name: nom de la collection
          - client_factory : fabrique de client injectable (tests → client
            éphémère en mémoire). Par défaut : PersistentClient + télémétrie
            désactivée.
        """
        self.path = path
        self.collection_name = collection_name
        self._client_factory = client_factory
        self._client = None
        self._collection = None

    # ------------------------------------------------------------------
    # Initialisation différée (client + collection)
    # ------------------------------------------------------------------

    @staticmethod
    def _default_client_factory(path: str):
        """Client ChromaDB persistant par défaut (télémétrie désactivée)."""
        import chromadb
        from chromadb.config import Settings

        return chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False),
        )

    def _ensure_client(self):
        """Crée le client au premier usage (import différé de chromadb)."""
        if self._client is None:
            factory = self._client_factory or self._default_client_factory
            self._client = factory(self.path)
        return self._client

    def _ensure_collection(self, create: bool = False):
        """
        Récupère la collection existante ou la crée (espace cosine) si
        `create` est vrai. Retourne None si elle n'existe pas.
        """
        if self._collection is not None:
            return self._collection

        client = self._ensure_client()
        try:
            self._collection = client.get_collection(self.collection_name)
        except Exception:
            if not create:
                return None
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": DEFAULT_SPACE},
            )
        return self._collection

    # ------------------------------------------------------------------
    # Validation des entrées
    # ------------------------------------------------------------------

    @staticmethod
    def _to_metadata(chunk: Dict) -> Dict:
        """Métadonnées ChromaDB : seules les clés présentes et non nulles."""
        metadata = {}
        for key in ("source", "chunk_id", "length", "word_count"):
            value = chunk.get(key)
            if value is not None:
                metadata[key] = value
        return metadata

    @staticmethod
    def _validate_chunks_embeddings(chunks: List[Dict], embeddings) -> None:
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("chunks doit être une liste non vide de dictionnaires.")

        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise ValueError("Chaque chunk doit être un dictionnaire.")
            missing = [
                key for key in ("content", "source", "chunk_id") if key not in chunk
            ]
            if missing:
                raise ValueError(f"Chunk invalide, clés manquantes : {missing}")

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise ValueError(
                "embeddings doit être une matrice (N, dimension) avec N = len(chunks)."
            )

    # ------------------------------------------------------------------
    # Indexation
    # ------------------------------------------------------------------

    def build(self, chunks: List[Dict], embeddings) -> None:
        """
        Construction initiale IDEMPOTENTE : supprime la collection existante,
        la recrée (espace cosine) puis indexe tout le lot.
        """
        self._validate_chunks_embeddings(chunks, embeddings)

        client = self._ensure_client()
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass  # rien à supprimer : première construction

        self._collection = client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": DEFAULT_SPACE},
        )
        self._add(self._collection, chunks, embeddings)

    def add_chunks(self, chunks: List[Dict], embeddings) -> None:
        """
        Ingestion incrémentale : ajoute ou met à jour (upsert sur
        "source::chunk_id") sans reconstruire la base.
        """
        self._validate_chunks_embeddings(chunks, embeddings)
        collection = self._ensure_collection(create=True)
        self._add(collection, chunks, embeddings)

    @staticmethod
    def _add(collection, chunks: List[Dict], embeddings) -> None:
        ids = [f"{chunk['source']}::{chunk['chunk_id']}" for chunk in chunks]
        documents = [chunk["content"] for chunk in chunks]
        metadatas = [ChromaManager._to_metadata(chunk) for chunk in chunks]
        vectors = np.asarray(embeddings, dtype=np.float32).tolist()
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=vectors,
        )

    # ------------------------------------------------------------------
    # Recherche (uniquement par vecteur, jamais query_texts)
    # ------------------------------------------------------------------

    def search(self, embedding, top_k: int = 5) -> List[Dict]:
        """
        Recherche par similarité cosine. L'entrée est un VECTEUR (produit par
        EmbeddingGenerator) : le binôme utilise le même modèle, donc la même
        requête donnera les mêmes résultats.

        Retourne une liste de chunks (schéma projet + distance), du plus
        proche au plus éloigné.
        """
        if top_k < 1:
            raise ValueError("top_k doit être supérieur ou égal à 1.")

        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.ndim != 1:
            raise ValueError("embedding doit être un vecteur 1D.")

        collection = self._ensure_collection(create=False)
        if collection is None:
            return []

        n_results = min(top_k, collection.count())
        if n_results == 0:
            return []

        result = collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for i in range(n_results):
            metadata = result["metadatas"][0][i] or {}
            chunks.append(
                {
                    "source": metadata.get("source"),
                    "chunk_id": metadata.get("chunk_id"),
                    "length": metadata.get("length"),
                    "word_count": metadata.get("word_count"),
                    "content": result["documents"][0][i],
                    "distance": result["distances"][0][i],
                }
            )
        return chunks

    # ------------------------------------------------------------------
    # Contrôle
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Nombre de chunks indexés (0 si la collection n'existe pas)."""
        collection = self._ensure_collection(create=False)
        if collection is None:
            return 0
        return collection.count()

    def delete_collection(self) -> None:
        """Supprime la collection (reset). Idempotent."""
        client = self._ensure_client()
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = None

    def close(self) -> None:
        """Ferme le client et libère les verrous (avant export zip)."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        self._client = None
        self._collection = None

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def info(self) -> Dict:
        """État du manager (introspection pure, sans effet de bord)."""
        count = 0
        if self._client is not None:
            collection = self._ensure_collection(create=False)
            count = collection.count() if collection is not None else 0

        return {
            "engine": "ChromaManager (ChromaDB)",
            "path": self.path,
            "collection_name": self.collection_name,
            "space": DEFAULT_SPACE,
            "count": count,
            "client_loaded": self._client is not None,
        }


# =====================================================
# Test
# =====================================================

if __name__ == "__main__":
    import chromadb
    from chromadb.config import Settings

    # Démo avec un client éphémère (aucun fichier créé sur disque).
    manager = ChromaManager(
        path="storage/chroma",
        client_factory=lambda path: chromadb.EphemeralClient(
            settings=Settings(anonymized_telemetry=False)
        ),
    )

    demo_chunks = [
        {
            "chunk_id": 0,
            "source": "demo.txt",
            "length": 42,
            "word_count": 8,
            "content": "Le machine learning permet de prédire à partir de données.",
        },
        {
            "chunk_id": 1,
            "source": "demo.txt",
            "length": 38,
            "word_count": 7,
            "content": "La recherche lexicale utilise le BM25.",
        },
    ]
    demo_vectors = np.random.rand(2, 384).astype(np.float32)

    manager.build(demo_chunks, demo_vectors)
    print(manager.info())

    results = manager.search(demo_vectors[0], top_k=2)
    for r in results:
        print(r["distance"], "|", r["content"])

    manager.close()
