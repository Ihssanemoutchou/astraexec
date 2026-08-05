"""
AstraExec — Journal d'audit structuré (extension production)
=============================================================

Enregistre chaque action exécutée sous forme de **JSON Lines**
(logs/audit.jsonl par défaut), une ligne JSON par événement.

Champs garantis :
    timestamp    : horodatage ISO-8601 UTC (millisecondes)
    execution_id : identifiant unique de l'exécution (uuid hex)
    plan_id      : identifiant du plan multi-étapes, si exécuté dans un plan
    tool         : nom de l'outil exécuté
    arguments    : paramètres de l'action
    latency      : durée d'exécution en secondes
    status       : "success" | "error"
    error        : message d'erreur (None en cas de succès)

Des champs supplémentaires peuvent être ajoutés via **extra.
Le format JSONL est le même que celui déjà utilisé par le projet
(logs/ethical_filter.jsonl) : une ligne = un événement, exploitable
sans parser maison.
"""

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional


class AuditLogger:
    """
    Journal d'audit structuré (JSON Lines), thread-safe.

    Utilisation :
        logger = AuditLogger(log_dir="logs")
        logger.record(
            timestamp="2026-08-06T10:00:00.000+00:00",
            execution_id="...",
            plan_id=None,
            tool="fusion_search",
            arguments={"query": "BM25"},
            latency=0.0213,
            status="success",
            error=None,
        )
    """

    def __init__(self, log_dir: str = "logs", filename: str = "audit.jsonl"):
        Path(log_dir).mkdir(exist_ok=True)
        self.path = Path(log_dir) / filename
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Écriture
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        timestamp: str,
        execution_id: str,
        plan_id: Optional[str] = None,
        tool: Optional[str] = None,
        arguments: Optional[Any] = None,
        latency: Optional[float] = None,
        status: Optional[str] = None,
        error: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """
        Écrit un enregistrement d'audit (une ligne JSON).

        Retourne le dictionnaire écrit (utile pour les tests et la
        corrélation). Les valeurs non sérialisables sont converties en
        chaîne (default=str) pour ne jamais casser la journalisation.
        """
        entry: Dict[str, Any] = {
            "timestamp": timestamp,
            "execution_id": execution_id,
            "plan_id": plan_id,
            "tool": tool,
            "arguments": arguments,
            "latency": latency,
            "status": status,
            "error": error,
        }
        entry.update(extra)

        line = json.dumps(entry, ensure_ascii=False, default=str)

        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

        return entry

    # ------------------------------------------------------------------
    # Lecture (tests, débogage)
    # ------------------------------------------------------------------

    def read(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Recharge les enregistrements écrits (JSON Lines → listes de dicts).

        `limit` borne le nombre d'enregistrements retournés.
        """
        if not self.path.exists():
            return []

        records: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if limit is not None:
            return records[:limit]
        return records

    def clear(self) -> None:
        """Supprime le fichier d'audit (réinitialisation de la piste)."""
        if self.path.exists():
            self.path.unlink()
