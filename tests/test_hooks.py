"""Tests for the observability hooks system."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock


from acmeow._internal.hooks import (
    HookContext,
    HookEvent,
    HookManager,
    add_hook,
    emit_hook,
    get_hook_manager,
    remove_hook,
)


class TestHookEvent:
    """Tests for HookEvent enum."""

    def test_event_values(self) -> None:
        """Test that event values are strings."""
        assert HookEvent.ACCOUNT_CREATED.value == "account_created"
        assert HookEvent.CERTIFICATE_ISSUED.value == "certificate_issued"
        assert HookEvent.CHALLENGE_SETUP.value == "challenge_setup"

    def test_all_events_are_strings(self) -> None:
        """Test that all events have string values."""
        for event in HookEvent:
            assert isinstance(event.value, str)


class TestHookContext:
    """Tests for HookContext dataclass."""

    def test_create_context(self) -> None:
        """Test creating a hook context."""
        ctx = HookContext(
            event=HookEvent.ACCOUNT_CREATED,
            data={"email": "test@example.com"},
        )

        assert ctx.event == HookEvent.ACCOUNT_CREATED
        assert ctx.data["email"] == "test@example.com"
        assert ctx.timestamp is not None
        assert ctx.client is None

    def test_context_with_client(self) -> None:
        """Test context with client reference."""
        mock_client = MagicMock()
        ctx = HookContext(
            event=HookEvent.ORDER_CREATED,
            data={},
            client=mock_client,
        )

        assert ctx.client is mock_client

    def test_context_str(self) -> None:
        """Test context string representation."""
        ctx = HookContext(event=HookEvent.ERROR)
        assert "ERROR" in str(ctx).upper() or "error" in str(ctx)


class TestHookManager:
    """Tests for HookManager class."""

    def test_add_and_emit_hook(self) -> None:
        """Test adding a hook and emitting an event."""
        manager = HookManager()
        callback = MagicMock()

        manager.add_hook(HookEvent.ACCOUNT_CREATED, callback)
        manager.emit(HookEvent.ACCOUNT_CREATED, {"test": True})

        callback.assert_called_once()
        ctx = callback.call_args[0][0]
        assert isinstance(ctx, HookContext)
        assert ctx.event == HookEvent.ACCOUNT_CREATED
        assert ctx.data["test"] is True

    def test_multiple_hooks_same_event(self) -> None:
        """Test multiple callbacks for the same event."""
        manager = HookManager()
        callback1 = MagicMock()
        callback2 = MagicMock()

        manager.add_hook(HookEvent.CERTIFICATE_ISSUED, callback1)
        manager.add_hook(HookEvent.CERTIFICATE_ISSUED, callback2)
        manager.emit(HookEvent.CERTIFICATE_ISSUED)

        callback1.assert_called_once()
        callback2.assert_called_once()

    def test_global_hook(self) -> None:
        """Test hook that receives all events."""
        manager = HookManager()
        callback = MagicMock()

        manager.add_hook(None, callback)  # Global hook
        manager.emit(HookEvent.ORDER_CREATED)
        manager.emit(HookEvent.CHALLENGE_SETUP)

        assert callback.call_count == 2

    def test_remove_hook(self) -> None:
        """Test removing a hook."""
        manager = HookManager()
        callback = MagicMock()

        manager.add_hook(HookEvent.ERROR, callback)
        result = manager.remove_hook(HookEvent.ERROR, callback)

        assert result is True
        manager.emit(HookEvent.ERROR)
        callback.assert_not_called()

    def test_remove_nonexistent_hook(self) -> None:
        """Test removing a hook that doesn't exist."""
        manager = HookManager()
        callback = MagicMock()

        result = manager.remove_hook(HookEvent.ERROR, callback)
        assert result is False

    def test_clear_hooks(self) -> None:
        """Test clearing hooks."""
        manager = HookManager()
        callback = MagicMock()

        manager.add_hook(HookEvent.WARNING, callback)
        manager.clear_hooks(HookEvent.WARNING)
        manager.emit(HookEvent.WARNING)

        callback.assert_not_called()

    def test_clear_all_hooks(self) -> None:
        """Test clearing all hooks."""
        manager = HookManager()
        callback = MagicMock()

        manager.add_hook(HookEvent.ERROR, callback)
        manager.add_hook(HookEvent.WARNING, callback)
        manager.clear_hooks()

        assert manager.get_hook_count() == 0

    def test_get_hook_count(self) -> None:
        """Test getting hook count."""
        manager = HookManager()
        callback = MagicMock()

        assert manager.get_hook_count() == 0
        manager.add_hook(HookEvent.ERROR, callback)
        assert manager.get_hook_count() == 1
        assert manager.get_hook_count(HookEvent.ERROR) == 1
        assert manager.get_hook_count(HookEvent.WARNING) == 0

    def test_has_hooks(self) -> None:
        """Test checking if hooks exist."""
        manager = HookManager()
        callback = MagicMock()

        assert not manager.has_hooks()
        manager.add_hook(HookEvent.ERROR, callback)
        assert manager.has_hooks()
        assert manager.has_hooks(HookEvent.ERROR)
        assert not manager.has_hooks(HookEvent.WARNING)

    def test_callback_exception_handling(self) -> None:
        """Test that callback exceptions don't break other callbacks."""
        manager = HookManager()

        def bad_callback(ctx: HookContext) -> None:
            raise ValueError("Test error")

        good_callback = MagicMock()

        manager.add_hook(HookEvent.ERROR, bad_callback)
        manager.add_hook(HookEvent.ERROR, good_callback)

        # Should not raise
        manager.emit(HookEvent.ERROR)

        # Good callback should still be called
        good_callback.assert_called_once()

    def test_thread_safety(self) -> None:
        """Test thread-safe operations."""
        manager = HookManager()
        results: list[int] = []

        def callback(ctx: HookContext) -> None:
            results.append(1)

        def add_hooks() -> None:
            for _ in range(100):
                manager.add_hook(HookEvent.REQUEST_STARTED, callback)

        def emit_events() -> None:
            for _ in range(100):
                manager.emit(HookEvent.REQUEST_STARTED)

        threads = [
            threading.Thread(target=add_hooks),
            threading.Thread(target=emit_events),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors
        assert len(results) > 0


class TestGlobalHookFunctions:
    """Tests for global hook convenience functions."""

    def test_get_hook_manager(self) -> None:
        """Test getting global hook manager."""
        manager = get_hook_manager()
        assert isinstance(manager, HookManager)

        # Should return same instance
        manager2 = get_hook_manager()
        assert manager is manager2

    def test_add_and_remove_hook(self) -> None:
        """Test add_hook and remove_hook convenience functions."""
        callback = MagicMock()

        add_hook(HookEvent.RETRY, callback)
        emit_hook(HookEvent.RETRY)
        callback.assert_called_once()

        remove_hook(HookEvent.RETRY, callback)
        callback.reset_mock()
        emit_hook(HookEvent.RETRY)
        callback.assert_not_called()

    def test_emit_hook(self) -> None:
        """Test emit_hook convenience function."""
        callback = MagicMock()
        add_hook(HookEvent.DNS_PROPAGATION_CHECK, callback)

        emit_hook(HookEvent.DNS_PROPAGATION_CHECK, {"domain": "example.com"})

        callback.assert_called_once()
        ctx = callback.call_args[0][0]
        assert ctx.data["domain"] == "example.com"

        # Cleanup
        remove_hook(HookEvent.DNS_PROPAGATION_CHECK, callback)
