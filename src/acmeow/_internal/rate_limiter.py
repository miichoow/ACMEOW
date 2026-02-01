"""Smart rate limiting for ACME API requests.

Provides intelligent rate limiting that respects ACME server limits
and adapts to rate limit responses.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for smart rate limiting.

    Args:
        requests_per_second: Maximum requests per second. Default 10.
        burst_size: Number of requests allowed in a burst. Default 20.
        auto_wait: Whether to automatically wait when rate limited. Default True.
        max_wait: Maximum time to wait for rate limit in seconds. Default 300.
        backoff_multiplier: Multiplier for exponential backoff. Default 2.0.
        initial_backoff: Initial backoff time in seconds. Default 1.0.
        max_backoff: Maximum backoff time in seconds. Default 120.0.
    """

    requests_per_second: float = 10.0
    burst_size: int = 20
    auto_wait: bool = True
    max_wait: float = 300.0
    backoff_multiplier: float = 2.0
    initial_backoff: float = 1.0
    max_backoff: float = 120.0


@dataclass
class RateLimitState:
    """Current state of the rate limiter.

    Attributes:
        request_times: Timestamps of recent requests.
        backoff_until: Time until which requests should be delayed.
        consecutive_rate_limits: Number of consecutive rate limit responses.
        total_requests: Total requests made.
        total_rate_limits: Total rate limit responses received.
        total_wait_time: Total time spent waiting due to rate limits.
    """

    request_times: deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    backoff_until: float = 0.0
    consecutive_rate_limits: int = 0
    total_requests: int = 0
    total_rate_limits: int = 0
    total_wait_time: float = 0.0


@dataclass
class RateLimitMetrics:
    """Metrics from the rate limiter.

    Attributes:
        total_requests: Total requests made.
        total_rate_limits: Total rate limit responses received.
        total_wait_time: Total time spent waiting (seconds).
        requests_per_second: Current requests per second.
        rate_limit_ratio: Ratio of rate limited to total requests.
        current_backoff: Current backoff time (if any).
    """

    total_requests: int
    total_rate_limits: int
    total_wait_time: float
    requests_per_second: float
    rate_limit_ratio: float
    current_backoff: float


# Type alias for rate limit callback
RateLimitCallback = Callable[[str, float], None]


class SmartRateLimiter:
    """Intelligent rate limiter for ACME API requests.

    Implements token bucket algorithm with adaptive backoff when
    rate limits are encountered. Thread-safe for concurrent use.

    Example:
        >>> limiter = SmartRateLimiter(RateLimitConfig())
        >>> limiter.acquire()  # Wait if necessary
        >>> # Make request...
        >>> if response.status_code == 429:
        ...     limiter.record_rate_limit(retry_after=60)
    """

    def __init__(
        self,
        config: RateLimitConfig | None = None,
        on_rate_limit: RateLimitCallback | None = None,
        on_wait: RateLimitCallback | None = None,
    ) -> None:
        """Initialize the rate limiter.

        Args:
            config: Rate limit configuration.
            on_rate_limit: Callback when rate limit is recorded.
                Signature: (reason: str, wait_time: float) -> None
            on_wait: Callback when waiting before request.
                Signature: (reason: str, wait_time: float) -> None
        """
        self._config = config or RateLimitConfig()
        self._state = RateLimitState()
        self._lock = threading.Lock()
        self._on_rate_limit = on_rate_limit
        self._on_wait = on_wait

        # Calculate minimum interval between requests
        self._min_interval = 1.0 / self._config.requests_per_second

    @property
    def config(self) -> RateLimitConfig:
        """Current rate limit configuration."""
        return self._config

    def acquire(self, timeout: float | None = None) -> bool:
        """Acquire permission to make a request.

        Blocks until a request can be made without exceeding rate limits.
        If auto_wait is disabled or timeout is exceeded, returns False.

        Args:
            timeout: Maximum time to wait in seconds. Uses config.max_wait if None.

        Returns:
            True if request can proceed, False if timed out.
        """
        max_wait = timeout if timeout is not None else self._config.max_wait
        start_time = time.monotonic()

        while True:
            wait_time = self._calculate_wait_time()

            if wait_time <= 0:
                # Can proceed immediately
                with self._lock:
                    self._state.request_times.append(time.monotonic())
                    self._state.total_requests += 1
                return True

            # Check if we would exceed timeout
            elapsed = time.monotonic() - start_time
            if elapsed + wait_time > max_wait:
                if not self._config.auto_wait:
                    return False
                # Reduce wait to fit within timeout
                wait_time = max(0, max_wait - elapsed)
                if wait_time <= 0:
                    return False

            # Wait
            if self._on_wait:
                self._on_wait("rate_limit", wait_time)

            logger.debug("Rate limiter waiting %.2f seconds", wait_time)

            with self._lock:
                self._state.total_wait_time += wait_time

            time.sleep(wait_time)

    def _calculate_wait_time(self) -> float:
        """Calculate how long to wait before the next request.

        Returns:
            Wait time in seconds, or 0 if no wait needed.
        """
        now = time.monotonic()

        with self._lock:
            # Check if we're in a backoff period
            if self._state.backoff_until > now:
                return self._state.backoff_until - now

            # Check token bucket (sliding window)
            if len(self._state.request_times) < self._config.burst_size:
                return 0.0

            # Check rate limit
            window_start = now - 1.0  # 1 second window
            recent_requests = sum(
                1 for t in self._state.request_times if t > window_start
            )

            if recent_requests < self._config.requests_per_second:
                return 0.0

            # Need to wait for oldest request to expire from window
            oldest_in_window = min(
                (t for t in self._state.request_times if t > window_start),
                default=now,
            )
            return max(0, oldest_in_window + 1.0 - now + 0.01)  # Small buffer

    def record_rate_limit(
        self,
        retry_after: float | None = None,
        reason: str = "rate_limited",
    ) -> None:
        """Record a rate limit response from the server.

        Updates the limiter state to back off from making requests.

        Args:
            retry_after: Server-provided retry delay in seconds.
            reason: Reason for the rate limit (for logging/callbacks).
        """
        with self._lock:
            self._state.total_rate_limits += 1
            self._state.consecutive_rate_limits += 1

            # Calculate backoff time
            if retry_after is not None and retry_after > 0:
                backoff = retry_after
            else:
                # Exponential backoff
                backoff = min(
                    self._config.initial_backoff
                    * (self._config.backoff_multiplier ** (self._state.consecutive_rate_limits - 1)),
                    self._config.max_backoff,
                )

            self._state.backoff_until = time.monotonic() + backoff

            logger.warning(
                "Rate limit recorded: %s, backing off for %.1f seconds",
                reason,
                backoff,
            )

        if self._on_rate_limit:
            self._on_rate_limit(reason, backoff)

    def record_success(self) -> None:
        """Record a successful request.

        Resets the consecutive rate limit counter.
        """
        with self._lock:
            self._state.consecutive_rate_limits = 0

    def get_metrics(self) -> RateLimitMetrics:
        """Get current rate limiter metrics.

        Returns:
            RateLimitMetrics with current statistics.
        """
        now = time.monotonic()

        with self._lock:
            # Calculate current requests per second
            window_start = now - 1.0
            recent = sum(1 for t in self._state.request_times if t > window_start)

            rate_limit_ratio = (
                self._state.total_rate_limits / self._state.total_requests
                if self._state.total_requests > 0
                else 0.0
            )

            current_backoff = max(0, self._state.backoff_until - now)

            return RateLimitMetrics(
                total_requests=self._state.total_requests,
                total_rate_limits=self._state.total_rate_limits,
                total_wait_time=self._state.total_wait_time,
                requests_per_second=float(recent),
                rate_limit_ratio=rate_limit_ratio,
                current_backoff=current_backoff,
            )

    def reset(self) -> None:
        """Reset the rate limiter state."""
        with self._lock:
            self._state = RateLimitState()
            logger.debug("Rate limiter state reset")

    def is_rate_limited(self) -> bool:
        """Check if currently in a rate-limited state.

        Returns:
            True if in backoff period.
        """
        with self._lock:
            return self._state.backoff_until > time.monotonic()

    def get_wait_time(self) -> float:
        """Get the current wait time without blocking.

        Returns:
            Seconds until a request can be made.
        """
        return self._calculate_wait_time()


class CompositeRateLimiter:
    """Combines multiple rate limiters for different endpoints.

    Useful when different ACME endpoints have different rate limits.

    Example:
        >>> limiter = CompositeRateLimiter()
        >>> limiter.add_limiter("newOrder", SmartRateLimiter(
        ...     RateLimitConfig(requests_per_second=5)
        ... ))
        >>> limiter.acquire("newOrder")
    """

    def __init__(self, default_config: RateLimitConfig | None = None) -> None:
        """Initialize the composite rate limiter.

        Args:
            default_config: Default config for auto-created limiters.
        """
        self._default_config = default_config or RateLimitConfig()
        self._limiters: dict[str, SmartRateLimiter] = {}
        self._lock = threading.Lock()

    def add_limiter(self, name: str, limiter: SmartRateLimiter) -> None:
        """Add a named rate limiter.

        Args:
            name: Name/identifier for this limiter.
            limiter: The rate limiter instance.
        """
        with self._lock:
            self._limiters[name] = limiter

    def get_limiter(self, name: str) -> SmartRateLimiter:
        """Get or create a rate limiter by name.

        Args:
            name: Name/identifier for the limiter.

        Returns:
            The rate limiter for this name.
        """
        with self._lock:
            if name not in self._limiters:
                self._limiters[name] = SmartRateLimiter(self._default_config)
            return self._limiters[name]

    def acquire(self, name: str, timeout: float | None = None) -> bool:
        """Acquire permission from a named limiter.

        Args:
            name: Name of the limiter to acquire from.
            timeout: Maximum time to wait.

        Returns:
            True if request can proceed.
        """
        return self.get_limiter(name).acquire(timeout)

    def record_rate_limit(
        self,
        name: str,
        retry_after: float | None = None,
        reason: str = "rate_limited",
    ) -> None:
        """Record a rate limit for a named limiter.

        Args:
            name: Name of the limiter.
            retry_after: Server-provided retry delay.
            reason: Reason for the rate limit.
        """
        self.get_limiter(name).record_rate_limit(retry_after, reason)

    def record_success(self, name: str) -> None:
        """Record a successful request for a named limiter.

        Args:
            name: Name of the limiter.
        """
        self.get_limiter(name).record_success()

    def get_all_metrics(self) -> dict[str, RateLimitMetrics]:
        """Get metrics from all limiters.

        Returns:
            Dictionary mapping limiter names to their metrics.
        """
        with self._lock:
            return {name: limiter.get_metrics() for name, limiter in self._limiters.items()}
