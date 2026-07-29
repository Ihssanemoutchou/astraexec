"""
Métriques d'évaluation pour la recherche d'information.
=======================================================

Implémentations 100% personnelles de métriques IR classiques.
Aucune dépendance externe (ni pytrec_eval, ni ir_measures, ni ranx).

Métriques disponibles :
  - Recall@K    : Proportion de documents pertinents retrouvés dans les K premiers
  - MRR         : Mean Reciprocal Rank — rang moyen du premier document pertinent
  - Evaluator   : Classe pour évaluer plusieurs requêtes en une passe
"""

from typing import List, Set, Dict, Any, Union


# ---------------------------------------------------------------------------
# Recall@K
# ---------------------------------------------------------------------------

def recall_at_k(
    retrieved_ids: List[Any],
    relevant_ids: List[Any],
    k: int
) -> float:
    """
    Calcule le Recall@K.

    Paramètres
    ----------
    retrieved_ids : liste des IDs retournés par le moteur (dans l'ordre)
    relevant_ids  : liste des IDs pertinents attendus
    k             : rang jusqu'auquel calculer le rappel

    Retourne
    -------
    float : Recall@K entre 0.0 et 1.0

    Exemple
    -------
    >>> recall_at_k([1, 2, 3], [1, 4], 2)
    0.5
    """
    if not relevant_ids:
        return 0.0

    if k <= 0:
        return 0.0

    top_k_set = set(retrieved_ids[:k])  # set pour éviter de compter les doublons
    relevant_set = set(relevant_ids)
    retrieved_relevant = len(top_k_set & relevant_set)  # intersection

    return retrieved_relevant / len(relevant_set)


# ---------------------------------------------------------------------------
# MRR — Mean Reciprocal Rank
# ---------------------------------------------------------------------------

def mean_reciprocal_rank(results: List[Dict[str, Any]]) -> float:
    """
    Calcule le Mean Reciprocal Rank (MRR).

    Paramètres
    ----------
    results : liste de dictionnaires, chacun contenant au minimum :
        - "retrieved_ids" : liste des IDs retournés (ordonnés)
        - "relevant_ids"  : liste des IDs pertinents attendus

    Retourne
    -------
    float : MRR entre 0.0 et 1.0

    Exemple
    -------
    >>> results = [
    ...     {"retrieved_ids": [1, 2, 3], "relevant_ids": [1]},
    ...     {"retrieved_ids": [3, 2, 1], "relevant_ids": [1]},
    ... ]
    >>> mean_reciprocal_rank(results)
    1.0
    """
    if not results:
        return 0.0

    reciprocal_sum = 0.0
    query_count = 0

    for result in results:
        retrieved = result.get("retrieved_ids", [])
        relevant = result.get("relevant_ids", [])

        if not relevant:
            continue

        rank = _first_relevant_rank(retrieved, relevant)
        if rank is not None:
            reciprocal_sum += 1.0 / rank

        query_count += 1

    if query_count == 0:
        return 0.0

    return reciprocal_sum / query_count


def reciprocal_rank(
    retrieved_ids: List[Any],
    relevant_ids: List[Any]
) -> float:
    """
    Calcule le Reciprocal Rank pour une seule requête.

    Retourne 0.0 si aucun document pertinent n'est trouvé.
    """
    rank = _first_relevant_rank(retrieved_ids, relevant_ids)
    if rank is None:
        return 0.0
    return 1.0 / rank


def _first_relevant_rank(
    retrieved_ids: List[Any],
    relevant_ids: List[Any]
) -> Union[int, None]:
    """
    Trouve le rang (1-indexé) du premier document pertinent dans les résultats.
    Retourne None si aucun pertinent n'est trouvé.
    """
    relevant_set = set(relevant_ids)
    for idx, rid in enumerate(retrieved_ids):
        if rid in relevant_set:
            return idx + 1  # rang 1-indexé
    return None


# ---------------------------------------------------------------------------
# Evaluator — Évaluation multi-requêtes
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Évaluateur de moteur de recherche.

    Permet d'évaluer plusieurs requêtes en une seule passe,
    en calculant Recall@K (pour plusieurs K) et MRR.

    Utilisation typique :
    --------------------
    >>> evaluator = Evaluator()
    >>> evaluator.add_query(
    ...     query="machine learning",
    ...     retrieved_ids=[1, 2, 3, 4, 5],
    ...     relevant_ids=[1, 4, 7]
    ... )
    >>> evaluator.add_query(
    ...     query="deep learning",
    ...     retrieved_ids=[3, 1, 2],
    ...     relevant_ids=[1, 2]
    ... )
    >>> rapport = evaluator.evaluate(ks=[1, 3, 5])
    >>> print(rapport["mrr"])
    """

    def __init__(self):
        self.queries: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Ajout d'une requête
    # ------------------------------------------------------------------

    def add_query(
        self,
        query: str,
        retrieved_ids: List[Any],
        relevant_ids: List[Any],
    ):
        """
        Ajoute une requête à la session d'évaluation.

        Paramètres
        ----------
        query         : texte de la requête (pour traçabilité)
        retrieved_ids : IDs retournés par le moteur (dans l'ordre)
        relevant_ids  : IDs pertinents attendus
        """
        self.queries.append({
            "query": query,
            "retrieved_ids": list(retrieved_ids),
            "relevant_ids": list(relevant_ids),
        })

    # ------------------------------------------------------------------
    # Évaluation complète
    # ------------------------------------------------------------------

    def evaluate(self, ks: List[int] = None) -> Dict[str, Any]:
        """
        Exécute l'évaluation sur toutes les requêtes ajoutées.

        Paramètres
        ----------
        ks : liste des rangs K pour Recall@K (défaut : [1, 3, 5])

        Retourne
        -------
        dict : {
            "total_queries": int,
            "recall_at_k": {k: moyenne_Recall@K},
            "mrr": float,
            "details": [
                {
                    "query": str,
                    "recall_at_k": {k: Recall@K},
                    "reciprocal_rank": float,
                },
                ...
            ]
        }
        """
        if ks is None:
            ks = [1, 3, 5]

        if not self.queries:
            return {
                "total_queries": 0,
                "recall_at_k": {k: 0.0 for k in ks},
                "mrr": 0.0,
                "details": [],
            }

        total = len(self.queries)
        recall_sums = {k: 0.0 for k in ks}
        rr_sum = 0.0
        details = []

        for q in self.queries:
            retrieved = q["retrieved_ids"]
            relevant = q["relevant_ids"]

            # Recall@K pour chaque K
            recall_per_k = {}
            for k in ks:
                recall_per_k[k] = recall_at_k(retrieved, relevant, k)
                recall_sums[k] += recall_per_k[k]

            # Reciprocal Rank
            rr = reciprocal_rank(retrieved, relevant)
            rr_sum += rr

            details.append({
                "query": q["query"],
                "retrieved_count": len(retrieved),
                "relevant_count": len(relevant),
                "recall_at_k": recall_per_k,
                "reciprocal_rank": round(rr, 4),
            })

        # Moyennes
        return {
            "total_queries": total,
            "recall_at_k": {
                k: round(recall_sums[k] / total, 4) for k in ks
            },
            "mrr": round(rr_sum / total, 4),
            "details": details,
        }

    # ------------------------------------------------------------------
    # Réinitialisation
    # ------------------------------------------------------------------

    def reset(self):
        """Vide la session d'évaluation."""
        self.queries.clear()

    # ------------------------------------------------------------------
    # Résumé textuel
    # ------------------------------------------------------------------

    def summary(self, ks: List[int] = None) -> str:
        """
        Retourne un résumé textuel de l'évaluation.
        Utile pour l'affichage console / démo.
        """
        results = self.evaluate(ks)
        lines = []
        lines.append("=" * 50)
        lines.append("Rapport d'évaluation AstraExec")
        lines.append("=" * 50)
        lines.append(f"Requêtes évaluées : {results['total_queries']}")
        lines.append("")

        for k, v in results["recall_at_k"].items():
            lines.append(f"  Recall@{k} : {v:.4f}")

        lines.append(f"  MRR        : {results['mrr']:.4f}")
        lines.append("")
        lines.append("Détail par requête :")
        lines.append("-" * 50)

        for d in results["details"]:
            lines.append(f"  Requête : \"{d['query']}\"")
            lines.append(f"    Retournés : {d['retrieved_count']}, "
                         f"Pertinents : {d['relevant_count']}")
            r_at_k = ", ".join(
                f"R@{k}={v:.4f}" for k, v in d["recall_at_k"].items()
            )
            lines.append(f"    {r_at_k}")
            lines.append(f"    RR : {d['reciprocal_rank']:.4f}")
            lines.append("")

        lines.append("=" * 50)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Démonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Données de démonstration
    evaluator = Evaluator()

    evaluator.add_query(
        query="machine learning",
        retrieved_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        relevant_ids=[1, 4, 7, 10],
    )
    evaluator.add_query(
        query="deep learning",
        retrieved_ids=[3, 1, 5, 2, 4],
        relevant_ids=[1, 2, 9],
    )
    evaluator.add_query(
        query="recherche lexicale",
        retrieved_ids=[2, 4, 6, 8, 10],
        relevant_ids=[2, 6],
    )
    evaluator.add_query(
        query="TF-IDF",
        retrieved_ids=[5, 3, 1, 7, 9, 2, 4, 6, 8, 10],
        relevant_ids=[1, 3, 5],
    )
    evaluator.add_query(
        query="BM25",
        retrieved_ids=[2, 4, 1, 3],
        relevant_ids=[1, 2, 6],
    )

    print(evaluator.summary(ks=[1, 3, 5]))
