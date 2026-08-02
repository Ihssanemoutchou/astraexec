"""
Tests du module de mesure de performance — Livrable 5, Phase 4
===============================================================

Vérifie la logique DÉTERMINISTE de app/evaluation/performance.py
(mean, min, max, médiane, p95, débit, measure, time_call).

IMPORTANT : aucun test n'asserte de durée réelle (non-fragile,
machine-indépendant). Les statistiques sont vérifiées sur des jeux
de données CONNUS.

Composants mesurés : uniquement app/evaluation/performance.py
(module de mesure maison). Aucun composant métier n'est modifié.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.evaluation.performance import time_call, measure, summarize


class TestSummarize:
    """Statistiques sur des séries de durées connues."""

    def test_empty_list_returns_zeros(self):
        report = summarize([])
        assert report["count"] == 0
        assert report["mean"] == 0.0
        assert report["min"] == 0.0
        assert report["max"] == 0.0
        assert report["median"] == 0.0
        assert report["p95"] == 0.0
        assert report["throughput"] == 0.0

    def test_single_sample(self):
        report = summarize([2.5])
        assert report["count"] == 1
        assert report["mean"] == 2.5
        assert report["min"] == 2.5
        assert report["max"] == 2.5
        assert report["median"] == 2.5
        assert report["p95"] == 2.5

    def test_mean_min_max(self):
        report = summarize([1.0, 2.0, 3.0, 4.0])
        assert report["mean"] == 2.5
        assert report["min"] == 1.0
        assert report["max"] == 4.0

    def test_median_odd_count(self):
        report = summarize([3.0, 1.0, 2.0])
        assert report["median"] == 2.0

    def test_median_even_count(self):
        report = summarize([4.0, 1.0, 3.0, 2.0])
        assert report["median"] == 2.5

    def test_p95_nearest_rank_ten_samples(self):
        # p95 nearest-rank : index = ceil(0.95 * 10) - 1 = 9 -> 10.0
        times = [float(i) for i in range(1, 11)]
        report = summarize(times)
        assert report["p95"] == 10.0

    def test_p95_nearest_rank_twenty_samples(self):
        # index = ceil(0.95 * 20) - 1 = 18 -> valeur 19.0
        times = [float(i) for i in range(1, 21)]
        report = summarize(times)
        assert report["p95"] == 19.0

    def test_p95_small_sample(self):
        # 3 échantillons : index = ceil(0.95 * 3) - 1 = 2 -> 3.0
        report = summarize([1.0, 2.0, 3.0])
        assert report["p95"] == 3.0

    def test_total(self):
        report = summarize([1.0, 2.0, 3.0])
        assert report["total"] == 6.0

    def test_throughput(self):
        report = summarize([1.0, 2.0, 3.0])
        # 3 opérations en 6 s -> 0.5 op/s
        assert abs(report["throughput"] - 0.5) < 1e-9

    def test_throughput_zero_total(self):
        # Durée totale nulle -> débit 0 (pas de division par zéro)
        report = summarize([0.0, 0.0])
        assert report["throughput"] == 0.0

    def test_unsorted_input(self):
        report = summarize([4.0, 1.0, 3.0, 2.0])
        assert report["min"] == 1.0
        assert report["max"] == 4.0
        assert report["median"] == 2.5


class TestMeasure:
    """Série de mesures répétées."""

    def test_iteration_count(self):
        calls = []

        def fn():
            calls.append(1)

        times = measure(fn, iterations=5)
        assert len(times) == 5
        assert len(calls) == 5

    def test_default_iterations(self):
        calls = []

        def fn():
            calls.append(1)

        measure(fn)
        assert len(calls) == 10

    def test_zero_iterations_returns_empty(self):
        def fn():
            pass

        assert measure(fn, iterations=0) == []

    def test_negative_iterations_returns_empty(self):
        def fn():
            pass

        assert measure(fn, iterations=-3) == []

    def test_propagates_positional_args(self):
        seen = []

        def fn(value):
            seen.append(value)

        measure(fn, 3, "x")
        assert seen == ["x", "x", "x"]

    def test_propagates_keyword_args(self):
        seen = []

        def fn(value, **kwargs):
            seen.append((value, kwargs["flag"]))

        measure(fn, 2, "y", flag=True)
        assert seen == [("y", True), ("y", True)]

    def test_returns_positive_durations(self):
        # Durées réelles : on vérifie uniquement le type et la positivité
        # (aucune assertion de magnitude -> non-fragile).
        times = measure(lambda: None, iterations=4)
        assert all(isinstance(t, float) for t in times)
        assert all(t >= 0 for t in times)


class TestTimeCall:
    """Mesure unitaire."""

    def test_returns_float(self):
        duration = time_call(lambda: None)
        assert isinstance(duration, float)
        assert duration >= 0

    def test_propagates_arguments(self):
        seen = []

        def fn(a, b):
            seen.append((a, b))

        time_call(fn, 1, 2)
        assert seen == [(1, 2)]

    def test_return_value_not_required(self):
        # time_call mesure la durée, pas la valeur de retour.
        duration = time_call(lambda: "resultat")
        assert duration >= 0


class TestModuleIntegration:
    """Chaîne complète measure() -> summarize() sur une fonction réelle."""

    def test_measure_then_summarize_consistency(self):
        times = measure(lambda: None, iterations=6)
        report = summarize(times)
        assert report["count"] == 6
        # Cohérence interne : la moyenne est entre min et max.
        assert report["min"] <= report["mean"] <= report["max"]

    def test_summarize_does_not_mutate_input(self):
        times = [3.0, 1.0, 2.0]
        original = list(times)
        summarize(times)
        assert times == original
