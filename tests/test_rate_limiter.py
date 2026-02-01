"""Tests for the smart rate limiter."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock


from acmeow._internal.rate_limiter import (
    CompositeRateLimiter,
    RateLimitConfig,
    RateLimitMetrics,
    SmartRateLimiter,
)


class TestRateLimitConfig:
    """Tests for RateLimitConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = RateLimitConfig()

        assert config.requests_per_second == 10.0
        assert config.burst_size == 20
        assert config.auto_wait is True
        assert config.max_wait == 300.0
        assert config.backoff_multiplier == 2.0
        assert config.initial_backoff == 1.0
        assert config.max_backoff == 120.0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = RateLimitConfig(
            requests_per_second=5.0,
            burst_size=10,
            auto_wait=False,
            max_wait=60.0,
        )

        assert config.requests_per_second == 5.0
        assert config.burst_size == 10
        assert config.auto_wait is False
        assert config.max_wait == 60.0


class TestSmartRateLimiter:
    """Tests for SmartRateLimiter class."""

    def test_acquire_basic(self) -> None:
        """Test basic acquire."""
        limiter = SmartRateLimiter(RateLimitConfig(
            requests_per_second=100,
            burst_size=100,
        ))

        # Should acquire immediately for first requests
        assert limiter.acquire(timeout=0.1) is True
        assert limiter.acquire(timeout=0.1) is True

    def test_rate_limiting(self) -> None:
        """Test that rate limiting kicks in."""
        config = RateLimitConfig(
            requests_per_second=2,
            burst_size=2,
            auto_wait=True,
            max_wait=5.0,
        )
        limiter = SmartRateLimiter(config)

        # First two should be instant (burst)
        start = time.monotonic()
        limiter.acquire()
        limiter.acquire()

        # Third should wait
        limiter.acquire()
        elapsed = time.monotonic() - start

        # Should have waited some time
        assert elapsed >= 0.4  # At least 0.5s for 2 req/s, with some margin

    def test_record_rate_limit(self) -> None:
        """Test recording rate limit from server."""
        limiter = SmartRateLimiter()

        limiter.record_rate_limit(retry_after=1.0)

        assert limiter.is_rate_limited()
        assert limiter.get_wait_time() > 0

    def test_record_success(self) -> None:
        """Test recording successful request."""
        limiter = SmartRateLimiter()

        # Record some rate limits
        limiter.record_rate_limit()
        limiter.record_rate_limit()

        metrics = limiter.get_metrics()
        assert metrics.total_rate_limits == 2

        # Record success should reset consecutive counter
        limiter.record_success()
        # Internal state should be reset (tested via behavior)

    def test_get_metrics(self) -> None:
        """Test getting metrics."""
        limiter = SmartRateLimiter()

        # Make some requests
        limiter.acquire(timeout=0.1)
        limiter.acquire(timeout=0.1)

        metrics = limiter.get_metrics()

        assert isinstance(metrics, RateLimitMetrics)
        assert metrics.total_requests == 2
        assert metrics.total_rate_limits == 0
        assert metrics.rate_limit_ratio == 0.0

    def test_reset(self) -> None:
        """Test resetting the limiter."""
        limiter = SmartRateLimiter()

        limiter.acquire(timeout=0.1)
        limiter.record_rate_limit()

        limiter.reset()

        metrics = limiter.get_metrics()
        assert metrics.total_requests == 0
        assert metrics.total_rate_limits == 0
        assert not limiter.is_rate_limited()

    def test_callbacks(self) -> None:
        """Test rate limit and wait callbacks."""
        on_rate_limit = MagicMock()
        on_wait = MagicMock()

        limiter = SmartRateLimiter(
            on_rate_limit=on_rate_limit,
            on_wait=on_wait,
        )

        limiter.record_rate_limit(retry_after=1.0, reason="test")

        on_rate_limit.assert_called_once()
        args = on_rate_limit.call_args[0]
        assert args[0] == "test"
        assert args[1] == 1.0

    def test_exponential_backoff(self) -> None:
        """Test exponential backoff on repeated rate limits."""
        config = RateLimitConfig(
            initial_backoff=1.0,
            backoff_multiplier=2.0,
            max_backoff=10.0,
        )
        limiter = SmartRateLimiter(config)

        # Each rate limit should increase backoff
        limiter.record_rate_limit()
        wait1 = limiter.get_wait_time()

        # Wait for backoff to expire
        time.sleep(wait1 + 0.1)

        limiter.record_rate_limit()
        wait2 = limiter.get_wait_time()

        # Second backoff should be larger (approximately 2x)
        assert wait2 > wait1 * 1.5  # Allow some margin

    def test_max_backoff(self) -> None:
        """Test that backoff is capped at max_backoff."""
        config = RateLimitConfig(
            initial_backoff=1.0,
            backoff_multiplier=10.0,
            max_backoff=5.0,
        )
        limiter = SmartRateLimiter(config)

        # Many rate limits
        for _ in range(10):
            limiter.record_rate_limit()

        # Should not exceed max_backoff
        assert limiter.get_wait_time() <= 5.0

    def test_thread_safety(self) -> None:
        """Test thread-safe operations."""
        config = RateLimitConfig(
            requests_per_second=1000,
            burst_size=1000,
        )
        limiter = SmartRateLimiter(config)
        errors: list[Exception] = []

        def make_requests() -> None:
            try:
                for _ in range(100):
                    limiter.acquire(timeout=1.0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=make_requests) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        metrics = limiter.get_metrics()
        assert metrics.total_requests == 500


class TestCompositeRateLimiter:
    """Tests for CompositeRateLimiter class."""

    def test_add_limiter(self) -> None:
        """Test adding a named limiter."""
        composite = CompositeRateLimiter()
        limiter = SmartRateLimiter()

        composite.add_limiter("test", limiter)

        assert composite.get_limiter("test") is limiter

    def test_auto_create_limiter(self) -> None:
        """Test auto-creating limiter on first access."""
        composite = CompositeRateLimiter()

        limiter = composite.get_limiter("new")

        assert isinstance(limiter, SmartRateLimiter)
        # Should return same instance on second access
        assert composite.get_limiter("new") is limiter

    def test_acquire(self) -> None:
        """Test acquiring from named limiter."""
        composite = CompositeRateLimiter(RateLimitConfig(
            requests_per_second=100,
            burst_size=100,
        ))

        assert composite.acquire("test", timeout=0.1) is True

    def test_record_rate_limit(self) -> None:
        """Test recording rate limit for named limiter."""
        composite = CompositeRateLimiter()

        composite.record_rate_limit("api", retry_after=1.0)

        limiter = composite.get_limiter("api")
        assert limiter.is_rate_limited()

    def test_record_success(self) -> None:
        """Test recording success for named limiter."""
        composite = CompositeRateLimiter()

        composite.record_rate_limit("api")
        composite.record_success("api")

        # Should not raise
        composite.get_limiter("api")

    def test_get_all_metrics(self) -> None:
        """Test getting metrics from all limiters."""
        composite = CompositeRateLimiter()

        composite.acquire("api1")
        composite.acquire("api2")
        composite.acquire("api2")

        metrics = composite.get_all_metrics()

        assert "api1" in metrics
        assert "api2" in metrics
        assert metrics["api1"].total_requests == 1
        assert metrics["api2"].total_requests == 2

    def test_independent_limiters(self) -> None:
        """Test that limiters are independent."""
        composite = CompositeRateLimiter()

        composite.record_rate_limit("api1")

        # api2 should not be affected
        limiter2 = composite.get_limiter("api2")
        assert not limiter2.is_rate_limited()
