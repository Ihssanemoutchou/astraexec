"""
Tests des extensions de résilience (production) :
- RetryPolicy : backoff exponentiel et types retentables.
- CircuitBreaker : transitions closed → open → half_open.
- Timeout par outil (métadonnée outil ou défaut Executor).
- Intégration retry / circuit breaker dans Executor.run().

Aucun composant métier existant n'est modifié : ces tests mesurent
uniquement les extensions ajoutées (app/executor/resilience.py et
les paramètres optionnels d'Executor).
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.executor.executor import Executor
from app.executor.errors import ToolNetworkError, ToolTimeoutError, ToolUnavailableError
from app.executor.resilience import CircuitBreaker, RetryPolicy
from app.registry.base_tool import BaseTool
from app.telemetry.audit import AuditLogger


class CountingTool(BaseTool):
    """Outil qui compte ses exécutions (vérification des retries)."""

    def __init__(
        self,
        name="counting",
        failures_before_success=0,
        failure_type=RuntimeError,
        delay=0.0,
    ):
        super().__init__(name, "Outil compteur de test")
        self.calls = 0
        self.failures_before_success = failures_before_success
        self.failure_type = failure_type
        self.delay = delay

    def execute(self, **kwargs):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.calls <= self.failures_before_success:
            raise self.failure_type("transient failure")
        return {"calls": self.calls}


class SlowTool(BaseTool):
    """Outil lent (pour tester le timeout), avec ou sans métadonnée timeout."""

    def __init__(self, name="slow", timeout=None):
        super().__init__(name, "Outil lent de test", timeout=timeout)

    def execute(self, **kwargs):
        time.sleep(0.5)
        return {"ok": True}


class TestRetryPolicy:
    def test_exponential_delays(self):
        policy = RetryPolicy(max_retries=3, base_delay=0.1, max_delay=1.0, multiplier=2.0)
        assert policy.delay_for(0) == 0.1
        assert policy.delay_for(1) == 0.2
        assert policy.delay_for(2) == 0.4
        assert policy.delay_for(3) == 0.8

    def test_delay_capped_at_max(self):
        policy = RetryPolicy(max_retries=5, base_delay=0.1, max_delay=1.0, multiplier=2.0)
        assert policy.delay_for(4) == 1.0  # 1.6 plafonné à 1.0
        assert policy.delay_for(10) == 1.0

    def test_defaults(self):
        policy = RetryPolicy()
        assert policy.max_retries == 3
        assert policy.delay_for(0) == 0.1
        assert policy.delay_for(1) == 0.2

    def test_should_retry_default_types(self):
        policy = RetryPolicy()
        assert policy.should_retry(ToolTimeoutError("x")) is True
        assert policy.should_retry(ToolNetworkError("x")) is True
        assert policy.should_retry(ToolUnavailableError("x")) is True
        assert policy.should_retry(ValueError("x")) is False
        assert policy.should_retry(RuntimeError("x")) is False

    def test_should_retry_custom_types(self):
        policy = RetryPolicy(retryable=(ValueError,))
        assert policy.should_retry(ValueError("x")) is True
        assert policy.should_retry(ToolTimeoutError("x")) is False

    def test_invalid_parameters(self):
        try:
            RetryPolicy(max_retries=-1)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_info(self):
        policy = RetryPolicy()
        info = policy.info()
        assert info["max_retries"] == 3
        assert "ToolTimeoutError" in info["retryable"]


class TestCircuitBreaker:
    def test_initial_state_closed(self):
        breaker = CircuitBreaker(name="cb", failure_threshold=3, reset_timeout=10.0)
        assert breaker.state == "closed"
        assert breaker.allow_request() is True
        assert breaker.failure_count == 0

    def test_opens_after_threshold(self):
        breaker = CircuitBreaker(failure_threshold=3, reset_timeout=10.0)
        for _ in range(3):
            breaker.record_failure()
        assert breaker.state == "open"
        assert breaker.allow_request() is False

    def test_below_threshold_stays_closed(self):
        breaker = CircuitBreaker(failure_threshold=3, reset_timeout=10.0)
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == "closed"
        assert breaker.allow_request() is True

    def test_success_resets_counter(self):
        breaker = CircuitBreaker(failure_threshold=3, reset_timeout=10.0)
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()
        assert breaker.failure_count == 0
        assert breaker.state == "closed"

    def test_half_open_probe_success_closes(self):
        breaker = CircuitBreaker(failure_threshold=2, reset_timeout=0.05)
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == "open"
        assert breaker.allow_request() is False
        time.sleep(0.06)
        assert breaker.allow_request() is True  # transition half_open
        assert breaker.state == "half_open"
        breaker.record_success()
        assert breaker.state == "closed"

    def test_half_open_probe_failure_reopens(self):
        breaker = CircuitBreaker(failure_threshold=2, reset_timeout=0.05)
        breaker.record_failure()
        breaker.record_failure()
        time.sleep(0.06)
        breaker.allow_request()  # half_open
        breaker.record_failure()
        assert breaker.state == "open"
        assert breaker.allow_request() is False

    def test_reset(self):
        breaker = CircuitBreaker(failure_threshold=2, reset_timeout=10.0)
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == "open"
        breaker.reset()
        assert breaker.state == "closed"
        assert breaker.failure_count == 0
        assert breaker.allow_request() is True

    def test_info(self):
        breaker = CircuitBreaker(name="cb", failure_threshold=3, reset_timeout=10.0)
        info = breaker.info()
        assert info["name"] == "cb"
        assert info["state"] == "closed"
        assert info["failure_threshold"] == 3


class TestTimeout:
    def test_no_timeout_by_default(self):
        executor = Executor()
        tool = SlowTool()
        executor.register_tool(tool)
        result = executor.run({"tool": "slow", "parameters": {}})
        assert result["status"] == "success"

    def test_timeout_from_tool_metadata(self):
        executor = Executor()
        executor.register_tool(SlowTool("slow_timeout", timeout=0.05))
        result = executor.run({"tool": "slow_timeout", "parameters": {}})
        assert result["status"] == "error"
        assert "dépassé" in result["message"]

    def test_timeout_from_executor_default(self):
        executor = Executor(default_timeout=0.05)
        executor.register_tool(SlowTool("slow_default"))
        result = executor.run({"tool": "slow_default", "parameters": {}})
        assert result["status"] == "error"
        assert "délai" in result["message"] or "dépassé" in result["message"]

    def test_timeout_error_contract(self):
        executor = Executor()
        executor.register_tool(SlowTool("slow_contract", timeout=0.05))
        result = executor.run({"tool": "slow_contract", "parameters": {}})
        assert set(result.keys()) == {"status", "execution_time", "message"}

    def test_timeout_error_type_in_audit(self, tmp_path):
        logger = AuditLogger(log_dir=str(tmp_path))
        executor = Executor(audit_logger=logger)
        executor.register_tool(SlowTool("slow_audit", timeout=0.05))
        result = executor.run({"tool": "slow_audit", "parameters": {}})
        assert result["status"] == "error"
        record = logger.read()[0]
        assert record["error_type"] == "ToolTimeoutError"


class TestRetryIntegration:
    def test_flaky_tool_succeeds_after_retries(self):
        executor = Executor(
            retry_policy=RetryPolicy(max_retries=3, base_delay=0.01)
        )
        tool = CountingTool(
            name="flaky",
            failures_before_success=2,
            failure_type=ToolNetworkError,
        )
        executor.register_tool(tool)
        result = executor.run({"tool": "flaky", "parameters": {}})
        assert result["status"] == "success"
        assert tool.calls == 3  # 1 essai + 2 retries

    def test_gives_up_after_max_retries(self):
        executor = Executor(
            retry_policy=RetryPolicy(max_retries=2, base_delay=0.01)
        )
        tool = CountingTool(
            name="broken",
            failures_before_success=99,
            failure_type=ToolNetworkError,
        )
        executor.register_tool(tool)
        result = executor.run({"tool": "broken", "parameters": {}})
        assert result["status"] == "error"
        assert tool.calls == 3  # 1 essai + 2 retries

    def test_no_retry_on_non_retryable_error(self):
        executor = Executor(
            retry_policy=RetryPolicy(max_retries=3, base_delay=0.01)
        )
        tool = CountingTool(
            name="valerr",
            failures_before_success=1,
            failure_type=ValueError,
        )
        executor.register_tool(tool)
        result = executor.run({"tool": "valerr", "parameters": {}})
        assert result["status"] == "error"
        assert tool.calls == 1  # aucune retry sur ValueError


class TestCircuitBreakerIntegration:
    def test_breaker_blocks_failing_tool(self):
        breaker = CircuitBreaker(name="fragile", failure_threshold=2, reset_timeout=10.0)
        executor = Executor(circuit_breakers={"fragile": breaker})
        tool = CountingTool(
            name="fragile",
            failures_before_success=99,
            failure_type=RuntimeError,
        )
        executor.register_tool(tool)

        first = executor.run({"tool": "fragile", "parameters": {}})
        assert first["status"] == "error"
        assert breaker.state == "closed"  # 1 échec < seuil

        second = executor.run({"tool": "fragile", "parameters": {}})
        assert second["status"] == "error"
        assert breaker.state == "open"

        third = executor.run({"tool": "fragile", "parameters": {}})
        assert third["status"] == "error"
        assert "indisponible" in third["message"]
        assert tool.calls == 2  # le 3e appel n'a pas atteint l'outil

    def test_breaker_recovers_after_reset_timeout(self):
        breaker = CircuitBreaker(name="flaky2", failure_threshold=1, reset_timeout=0.05)
        executor = Executor(circuit_breakers={"flaky2": breaker})
        tool = CountingTool(
            name="flaky2",
            failures_before_success=1,
            failure_type=RuntimeError,
        )
        executor.register_tool(tool)

        executor.run({"tool": "flaky2", "parameters": {}})
        assert breaker.state == "open"

        # Bloqué tant que le délai de récupération n'est pas écoulé.
        blocked = executor.run({"tool": "flaky2", "parameters": {}})
        assert blocked["status"] == "error"
        assert tool.calls == 1

        time.sleep(0.06)

        # Sonde autorisée → succès → circuit refermé.
        recovered = executor.run({"tool": "flaky2", "parameters": {}})
        assert recovered["status"] == "success"
        assert breaker.state == "closed"
        assert tool.calls == 2
