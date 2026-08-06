"""
RetrievalAdapter — Façade d'intégration AstraExec ↔ Module REASONING
=====================================================================

Rôle unique : faire le pont entre le contrat Pydantic du module
REASONING (`RetrievalRequest` / `RetrievalResponse`, voir
`app/schemas/retrieval_contract.py`) et le moteur AstraExec existant
(`Executor` + outil `fusion_search`).

Règles d'architecture respectées (100% additif, aucun composant métier
modifié) :

- Executor ne connaît PAS RetrievalRequest : l'adaptateur construit une
  action `{"tool": "fusion_search", "parameters": {"query": ...}}`,
  identique en forme à celle envoyée par `/execute`.
- FusionSearch ne connaît PAS RetrievalResponse : le mapping
  (`final_score` → `relevance_score`) et l'aplatissement de la structure
  imbriquée `{"results": [{"chunk": {...}, "final_score": ...}]}` en une
  liste plate de `RetrievedChunk` vivent ici.
- Le `query_id` de la requête est réémis tel quel dans la réponse
  (corrélation multi-hop / appels parallèles).
- Le moteur ne lève jamais d'exception : l'adaptateur traduit un statut
  `error` de l'Executor en `RetrievalError` (→ HTTP 400 au niveau API).

Limite connue (documentée) : l'outil `fusion_search` enregistré dans
`app/api/main.py` appelle le moteur avec son `top_k` par défaut (5).
L'adaptateur tronque donc à `request.top_k`, mais ne peut pas retourner
plus de `ENGINE_MAX_RESULTS` (5) résultats. Ne pas modifier le moteur
pour lever cette limite : toute évolution doit passer par l'outil
enregistré, sans changer le comportement de `/execute`.
"""

from typing import Any, Dict, List, Optional

from app.executor.executor import Executor
from app.schemas.retrieval_contract import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunk,
)


class RetrievalError(Exception):
    """Erreur de retrieval, levée par l'adaptateur (jamais par le moteur).

    Porte le `query_id` d'origine pour permettre au module REASONING de
    corréler l'erreur avec sa requête (appels parallèles multi-hop).
    """

    def __init__(self, message: str, query_id: str = ""):
        super().__init__(message)
        self.query_id = query_id


class RetrievalAdapter:
    """
    Façade entre le contrat REASONING et le moteur AstraExec.

    Usage :
        adapter = RetrievalAdapter(executor)   # executor doit avoir fusion_search
        response = adapter.retrieve(request)   # request: RetrievalRequest
    """

    TOOL_NAME = "fusion_search"

    # Nombre maximal de résultats que le moteur fusion_search retourne
    # (top_k par défaut de FusionTool.execute). Voir docstring de module.
    ENGINE_MAX_RESULTS = 5

    def __init__(self, executor: Executor):
        if not executor.has_tool(self.TOOL_NAME):
            raise ValueError(
                f"L'outil '{self.TOOL_NAME}' doit être enregistré dans "
                "l'Executor avant d'instancier RetrievalAdapter."
            )
        self.executor = executor

    # ========================================================
    # Point d'entrée
    # ========================================================

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """
        Exécute la sous-requête via le moteur existant et construit la
        réponse conforme au contrat REASONING.

        Lève RetrievalError si le moteur retourne un statut error
        (ex. query vide, injection détectée, outil indisponible).
        """
        if isinstance(request, dict):
            request = RetrievalRequest(**request)

        if not isinstance(request, RetrievalRequest):
            raise TypeError("RetrievalRequest (ou dict équivalent) attendu.")

        action: Dict[str, Any] = {
            "tool": self.TOOL_NAME,
            "parameters": {"query": request.sub_query},
        }

        outcome = self.executor.run(action)

        if outcome.get("status") != "success":
            message = outcome.get(
                "message", "Échec inconnu du retrieval (statut non-success)."
            )
            raise RetrievalError(message, query_id=request.query_id)

        return self._build_response(request, outcome.get("result") or {})

    # ========================================================
    # Transformation de la réponse moteur → contrat REASONING
    # ========================================================

    def _build_response(
        self,
        request: RetrievalRequest,
        result: Dict[str, Any],
    ) -> RetrievalResponse:
        raw_results = result.get("results", [])

        # Aplatissement : {"chunk": {...}, "final_score": ...} → RetrievedChunk
        chunks: List[RetrievedChunk] = []
        for item in raw_results:
            chunk = item.get("chunk", {}) if isinstance(item, dict) else {}
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(chunk.get("chunk_id", "")),
                    content=str(chunk.get("content", "")),
                    source=str(chunk.get("source", "")),
                    relevance_score=float(item.get("final_score") or 0.0),
                )
            )

        # Filtres optionnels (uniquement "source", seul filtre que le moteur
        # actuel peut honorer sans modification ; les autres clés sont ignorées).
        chunks = self._apply_filters(chunks, request.filters)

        # top_k : le moteur retourne au plus ENGINE_MAX_RESULTS résultats.
        chunks = chunks[: request.top_k]

        # Métadonnées : celles du demandeur + contexte de l'exécution.
        metadata: Dict[str, Any] = dict(request.metadata or {})
        metadata.update(
            {
                "hop_index": request.hop_index,
                "n_chunks": len(chunks),
                "top_k": request.top_k,
            }
        )

        return RetrievalResponse(
            query_id=request.query_id,
            chunks=chunks,
            retrieval_score=self._retrieval_score(chunks),
            metadata=metadata,
        )

    # ========================================================
    # Helpers
    # ========================================================

    @staticmethod
    def _apply_filters(
        chunks: List[RetrievedChunk],
        filters: Optional[Dict[str, Any]],
    ) -> List[RetrievedChunk]:
        """Applique le filtre optionnel `source` (chaîne ou liste)."""
        if not filters:
            return chunks

        source_filter = filters.get("source")
        if source_filter is None:
            return chunks

        if isinstance(source_filter, (list, tuple, set)):
            allowed = set(source_filter)
            return [c for c in chunks if c.source in allowed]

        return [c for c in chunks if c.source == source_filter]

    @staticmethod
    def _retrieval_score(chunks: List[RetrievedChunk]) -> Optional[float]:
        """Score global de confiance : moyenne des pertinences (ou None)."""
        if not chunks:
            return None
        mean = sum(c.relevance_score for c in chunks) / len(chunks)
        return round(mean, 4)
