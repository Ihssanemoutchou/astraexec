"""
app.integration — Couche d'intégration AstraExec ↔ Module REASONING.

Couche STRICTEMENT additive : elle ne modifie aucun composant métier
(Executor, FusionSearch, ToolRegistry, Guardrails, Retrieval, Storage).
Elle expose le contrat RetrievalRequest/RetrievalResponse du module
REASONING en façade sur le moteur existant, sans que le moteur ne
connaisse ce contrat.

Composant principal : app.integration.retrieval_adapter.RetrievalAdapter
"""
