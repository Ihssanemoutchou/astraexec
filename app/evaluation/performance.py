"""
Mesure de performance — Livrable 5, Phase 4
============================================

Module maison de mesure de performance, 100 % bibliothèque standard
(comme metrics.py : aucune dépendance externe).

Fonctionnalités :
  - time_call(fn, ...)            : mesure unitaire d'un appel
  - measure(fn, iterations, ...)  : série de mesures répétées
  - summarize(times)              : statistiques de la série :
      mean (moyenne), min, max, median (médiane), p95 (percentile 95),
      throughput (débit en opérations/seconde)

Utilisation typique :
    times = measure(searcher.search, 20, "machine learning")
    report = summarize(times)
    print(f"Moyenne : {report['mean']:.4f}s — p95 : {report['p95']:.4f}s")

Le module ne fait que MESURER des fonctions existantes : il ne modifie
aucun composant et n'ajoute aucune dépendance.
"""

import time
from typing import Callable, Dict, List


# ---------------------------------------------------------------------------
# Mesure unitaire
# ---------------------------------------------------------------------------

def time_call(fn: Callable, *args, **kwargs) -> float:
    """
    Mesure la durée d'UN appel de fn(*args, **kwargs).

    Paramètres
    ----------
    fn      : fonction à mesurer
    *args   : arguments positionnels transmis à fn
    **kwargs: arguments nommés transmis à fn

    Retourne
    --------
    float : durée de l'appel en secondes
    """
    start = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - start


# ---------------------------------------------------------------------------
# Série de mesures
# ---------------------------------------------------------------------------

def measure(
    fn: Callable,
    iterations: int = 10,
    *args,
    **kwargs,
) -> List[float]:
    """
    Répète `iterations` appels de fn(*args, **kwargs)
    et retourne la liste des durées en secondes.

    Paramètres
    ----------
    fn         : fonction à mesurer
    iterations : nombre d'exécutions (<= 0 -> liste vide)
    *args      : arguments positionnels transmis à fn
    **kwargs   : arguments nommés transmis à fn

    Retourne
    --------
    list[float] : une durée par itération
    """
    if iterations <= 0:
        return []
    return [time_call(fn, *args, **kwargs) for _ in range(iterations)]


# ---------------------------------------------------------------------------
# Statistiques
# ---------------------------------------------------------------------------

def _percentile(sorted_values: List[float], p: float) -> float:
    """
    Percentile p (0.0 - 1.0) par la méthode « nearest-rank ».

    Le calcul se fait en ARITHMÉTIQUE ENTIÈRE (scale = round(p * 100))
    pour être totalement déterministe, sans dépendre des imprécisions
    flottantes de ceil(p * n).
    """
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    scale = int(round(p * 100))
    idx = max(0, min(n - 1, (scale * n + 99) // 100 - 1))
    return sorted_values[idx]


def _median(sorted_values: List[float]) -> float:
    """Médiane d'une liste déjà triée."""
    n = len(sorted_values)
    mid = n // 2
    if n % 2 == 1:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def summarize(times: List[float]) -> Dict[str, float]:
    """
    Calcule les statistiques d'une série de durées (secondes).

    Paramètres
    ----------
    times : liste des durées mesurées

    Retourne
    --------
    dict :
        count      : nombre de mesures
        mean       : moyenne arithmétique (s)
        min / max  : valeurs extrêmes (s)
        median     : médiane (s)
        p95        : percentile 95, méthode nearest-rank (s)
        total      : somme des durées (s)
        throughput : débit = count / total (opérations/seconde)

    Une liste vide retourne toutes les valeurs à 0.0 (dégradation propre).
    """
    if not times:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "total": 0.0,
            "throughput": 0.0,
        }

    total = sum(times)
    count = len(times)
    sorted_values = sorted(times)

    return {
        "count": count,
        "mean": total / count,
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "median": _median(sorted_values),
        "p95": _percentile(sorted_values, 0.95),
        "total": total,
        "throughput": (count / total) if total > 0 else 0.0,
    }
