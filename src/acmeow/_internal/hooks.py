"""Event hook system for ACME client observability.

Provides an event-driven hook system for monitoring and reacting to
events during certificate issuance.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class HookEvent(str, Enum):
    """Events that can be hooked into during ACME operations.

    These events are emitted at key points in the certificate issuance
    process, allowing external code to monitor, log, or react to the
    client's actions.
    """

    # Account events
    ACCOUNT_CREATED = "account_created"
    """Emitted when a new ACME account is created."""

    ACCOUNT_LOADED = "account_loaded"
    """Emitted when an existing account is loaded from storage."""

    ACCOUNT_UPDATED = "account_updated"
    """Emitted when account contact info is updated."""

    ACCOUNT_KEY_ROLLED = "account_key_rolled"
    """Emitted when the account key is rolled over."""

    ACCOUNT_DEACTIVATED = "account_deactivated"
    """Emitted when the account is deactivated."""

    # Order events
    ORDER_CREATED = "order_created"
    """Emitted when a new certificate order is created."""

    ORDER_LOADED = "order_loaded"
    """Emitted when a saved order is loaded from storage."""

    ORDER_READY = "order_ready"
    """Emitted when all authorizations are complete and order is ready."""

    ORDER_FINALIZED = "order_finalized"
    """Emitted when order finalization (CSR submission) is complete."""

    ORDER_FAILED = "order_failed"
    """Emitted when an order enters an invalid state."""

    # Challenge events
    CHALLENGE_SETUP = "challenge_setup"
    """Emitted when a challenge response is being set up."""

    CHALLENGE_READY = "challenge_ready"
    """Emitted when the server is notified a challenge is ready."""

    CHALLENGE_COMPLETED = "challenge_completed"
    """Emitted when a challenge is validated successfully."""

    CHALLENGE_FAILED = "challenge_failed"
    """Emitted when a challenge fails validation."""

    CHALLENGE_CLEANUP = "challenge_cleanup"
    """Emitted when challenge resources are cleaned up."""

    # Certificate events
    CERTIFICATE_ISSUED = "certificate_issued"
    """Emitted when a certificate is successfully issued."""

    CERTIFICATE_DOWNLOADED = "certificate_downloaded"
    """Emitted when a certificate is downloaded from the server."""

    CERTIFICATE_SAVED = "certificate_saved"
    """Emitted when a certificate is saved to disk."""

    CERTIFICATE_REVOKED = "certificate_revoked"
    """Emitted when a certificate is revoked."""

    # Network/operational events
    REQUEST_STARTED = "request_started"
    """Emitted before an HTTP request is made."""

    REQUEST_COMPLETED = "request_completed"
    """Emitted after an HTTP request completes."""

    RATE_LIMITED = "rate_limited"
    """Emitted when a rate limit is encountered."""

    RETRY = "retry"
    """Emitted when a request is being retried."""

    DNS_PROPAGATION_CHECK = "dns_propagation_check"
    """Emitted when checking DNS propagation."""

    DNS_PROPAGATION_VERIFIED = "dns_propagation_verified"
    """Emitted when DNS propagation is verified."""

    # General events
    ERROR = "error"
    """Emitted when an error occurs (in addition to specific error events)."""

    WARNING = "warning"
    """Emitted for non-fatal warnings."""


@dataclass
class HookContext:
    """Context passed to hook callbacks.

    Contains information about the event that triggered the hook,
    including event type, timestamp, and event-specific data.

    Attributes:
        event: The event type that triggered this callback.
        timestamp: When the event occurred (UTC).
        data: Event-specific data dictionary.
        client: Reference to the AcmeClient instance (if available).
    """

    event: HookEvent
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict[str, Any] = field(default_factory=dict)
    client: Any = None

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return f"HookContext({self.event.value}, {self.timestamp.isoformat()})"


# Type alias for hook callback functions
HookCallback = Callable[[HookContext], None]


class HookManager:
    """Manages event hooks for the ACME client.

    Thread-safe manager for registering, removing, and emitting events
    to callback functions. Hooks can be registered for specific events
    or for all events.

    Example:
        >>> manager = HookManager()
        >>> def on_cert_issued(ctx):
        ...     print(f"Certificate issued for {ctx.data['domains']}")
        >>> manager.add_hook(HookEvent.CERTIFICATE_ISSUED, on_cert_issued)
        >>> manager.emit(HookEvent.CERTIFICATE_ISSUED, {"domains": ["example.com"]})
    """

    def __init__(self) -> None:
        """Initialize the hook manager."""
        self._hooks: dict[HookEvent | None, list[HookCallback]] = {}
        self._lock = threading.Lock()

    def add_hook(
        self,
        event: HookEvent | None,
        callback: HookCallback,
    ) -> None:
        """Register a callback for an event.

        Args:
            event: The event to hook into, or None for all events.
            callback: Function to call when the event occurs.
                Signature: (context: HookContext) -> None
        """
        with self._lock:
            if event not in self._hooks:
                self._hooks[event] = []
            self._hooks[event].append(callback)
            logger.debug("Added hook for event: %s", event)

    def remove_hook(
        self,
        event: HookEvent | None,
        callback: HookCallback,
    ) -> bool:
        """Remove a callback from an event.

        Args:
            event: The event the callback was registered for.
            callback: The callback function to remove.

        Returns:
            True if the callback was removed, False if not found.
        """
        with self._lock:
            if event in self._hooks:
                try:
                    self._hooks[event].remove(callback)
                    logger.debug("Removed hook for event: %s", event)
                    return True
                except ValueError:
                    pass
            return False

    def emit(
        self,
        event: HookEvent,
        data: dict[str, Any] | None = None,
        client: Any = None,
    ) -> None:
        """Emit an event to all registered callbacks.

        Callbacks are called synchronously in the order they were registered.
        If a callback raises an exception, it is logged and the remaining
        callbacks are still called.

        Args:
            event: The event to emit.
            data: Optional event-specific data.
            client: Optional reference to the AcmeClient instance.
        """
        context = HookContext(
            event=event,
            data=data or {},
            client=client,
        )

        callbacks: list[HookCallback] = []

        with self._lock:
            # Get event-specific hooks
            if event in self._hooks:
                callbacks.extend(self._hooks[event])
            # Get global hooks (None = all events)
            if None in self._hooks:
                callbacks.extend(self._hooks[None])

        if not callbacks:
            logger.debug("No hooks registered for event: %s", event)
            return

        logger.debug("Emitting event %s to %d callbacks", event, len(callbacks))

        for callback in callbacks:
            try:
                callback(context)
            except Exception as e:
                logger.warning(
                    "Hook callback error for event %s: %s",
                    event,
                    e,
                    exc_info=True,
                )

    def clear_hooks(self, event: HookEvent | None = None) -> None:
        """Clear all hooks for an event.

        Args:
            event: The event to clear hooks for, or None to clear all hooks.
        """
        with self._lock:
            if event is None:
                self._hooks.clear()
                logger.debug("Cleared all hooks")
            elif event in self._hooks:
                del self._hooks[event]
                logger.debug("Cleared hooks for event: %s", event)

    def get_hook_count(self, event: HookEvent | None = None) -> int:
        """Get the number of hooks registered for an event.

        Args:
            event: The event to count hooks for, or None for total count.

        Returns:
            Number of registered hooks.
        """
        with self._lock:
            if event is None:
                return sum(len(callbacks) for callbacks in self._hooks.values())
            return len(self._hooks.get(event, []))

    def has_hooks(self, event: HookEvent | None = None) -> bool:
        """Check if any hooks are registered.

        Args:
            event: The event to check, or None for any event.

        Returns:
            True if hooks are registered.
        """
        return self.get_hook_count(event) > 0


# Global hook manager instance for convenience
_global_hook_manager: HookManager | None = None


def get_hook_manager() -> HookManager:
    """Get the global hook manager instance.

    Creates the instance on first call (lazy initialization).

    Returns:
        The global HookManager instance.
    """
    global _global_hook_manager
    if _global_hook_manager is None:
        _global_hook_manager = HookManager()
    return _global_hook_manager


def add_hook(event: HookEvent | None, callback: HookCallback) -> None:
    """Register a callback with the global hook manager.

    Convenience function that delegates to get_hook_manager().add_hook().

    Args:
        event: The event to hook into, or None for all events.
        callback: Function to call when the event occurs.
    """
    get_hook_manager().add_hook(event, callback)


def remove_hook(event: HookEvent | None, callback: HookCallback) -> bool:
    """Remove a callback from the global hook manager.

    Convenience function that delegates to get_hook_manager().remove_hook().

    Args:
        event: The event the callback was registered for.
        callback: The callback function to remove.

    Returns:
        True if the callback was removed.
    """
    return get_hook_manager().remove_hook(event, callback)


def emit_hook(
    event: HookEvent,
    data: dict[str, Any] | None = None,
    client: Any = None,
) -> None:
    """Emit an event to the global hook manager.

    Convenience function that delegates to get_hook_manager().emit().

    Args:
        event: The event to emit.
        data: Optional event-specific data.
        client: Optional reference to the AcmeClient instance.
    """
    get_hook_manager().emit(event, data, client)
