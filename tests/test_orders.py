"""Tests for order management utilities."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path


from acmeow.enums import OrderStatus
from acmeow.orders import OrderInfo, OrderManager


class TestOrderInfo:
    """Tests for OrderInfo dataclass."""

    def test_create_order_info(self) -> None:
        """Test creating an OrderInfo."""
        info = OrderInfo(
            url="https://acme.example.com/order/123",
            status=OrderStatus.PENDING,
            domains=["example.com", "www.example.com"],
            created=datetime.now(timezone.utc),
            expires=datetime.now(timezone.utc) + timedelta(days=7),
            path=Path("/tmp/order.json"),
        )

        assert info.url == "https://acme.example.com/order/123"
        assert info.status == OrderStatus.PENDING
        assert len(info.domains) == 2

    def test_is_terminal(self) -> None:
        """Test is_terminal property."""
        valid = OrderInfo(
            url="", status=OrderStatus.VALID, domains=[], created=None,
            expires=None, path=Path("/tmp/test"),
        )
        invalid = OrderInfo(
            url="", status=OrderStatus.INVALID, domains=[], created=None,
            expires=None, path=Path("/tmp/test"),
        )
        pending = OrderInfo(
            url="", status=OrderStatus.PENDING, domains=[], created=None,
            expires=None, path=Path("/tmp/test"),
        )

        assert valid.is_terminal is True
        assert invalid.is_terminal is True
        assert pending.is_terminal is False

    def test_is_expired(self) -> None:
        """Test is_expired property."""
        expired = OrderInfo(
            url="", status=OrderStatus.PENDING, domains=[],
            created=datetime.now(timezone.utc) - timedelta(days=10),
            expires=datetime.now(timezone.utc) - timedelta(days=1),
            path=Path("/tmp/test"),
        )

        not_expired = OrderInfo(
            url="", status=OrderStatus.PENDING, domains=[],
            created=datetime.now(timezone.utc),
            expires=datetime.now(timezone.utc) + timedelta(days=7),
            path=Path("/tmp/test"),
        )

        no_expiry = OrderInfo(
            url="", status=OrderStatus.PENDING, domains=[],
            created=None, expires=None,
            path=Path("/tmp/test"),
        )

        assert expired.is_expired is True
        assert not_expired.is_expired is False
        assert no_expiry.is_expired is False

    def test_age_days(self) -> None:
        """Test age_days property."""
        info = OrderInfo(
            url="", status=OrderStatus.PENDING, domains=[],
            created=datetime.now(timezone.utc) - timedelta(days=5),
            expires=None,
            path=Path("/tmp/test"),
        )

        assert info.age_days is not None
        assert info.age_days >= 4.9
        assert info.age_days <= 5.1

    def test_age_days_no_created(self) -> None:
        """Test age_days when created is None."""
        info = OrderInfo(
            url="", status=OrderStatus.PENDING, domains=[],
            created=None, expires=None,
            path=Path("/tmp/test"),
        )

        assert info.age_days is None

    def test_str(self) -> None:
        """Test string representation."""
        info = OrderInfo(
            url="", status=OrderStatus.READY,
            domains=["example.com", "www.example.com"],
            created=None, expires=None,
            path=Path("/tmp/test"),
        )

        s = str(info)
        assert "example.com" in s
        assert "ready" in s.lower()


class TestOrderManager:
    """Tests for OrderManager class."""

    def test_list_orders_empty(self) -> None:
        """Test listing orders when none exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OrderManager(tmpdir)
            orders = manager.list_orders()

            assert orders == []

    def test_save_and_list_order(self) -> None:
        """Test saving and listing an order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OrderManager(tmpdir)

            manager.save_order(
                url="https://acme.example.com/order/1",
                status=OrderStatus.PENDING,
                domains=["example.com"],
                finalize_url="https://acme.example.com/finalize/1",
            )

            orders = manager.list_orders()

            assert len(orders) == 1
            assert orders[0].url == "https://acme.example.com/order/1"
            assert orders[0].status == OrderStatus.PENDING
            assert "example.com" in orders[0].domains

    def test_list_orders_by_status(self) -> None:
        """Test filtering orders by status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OrderManager(tmpdir)

            manager.save_order(
                url="https://acme.example.com/order/1",
                status=OrderStatus.PENDING,
                domains=["pending.com"],
                finalize_url="",
                order_id="order1",
            )

            manager.save_order(
                url="https://acme.example.com/order/2",
                status=OrderStatus.VALID,
                domains=["valid.com"],
                finalize_url="",
                order_id="order2",
            )

            pending_orders = manager.list_orders(status=OrderStatus.PENDING)
            valid_orders = manager.list_orders(status=OrderStatus.VALID)

            assert len(pending_orders) == 1
            assert pending_orders[0].status == OrderStatus.PENDING

            assert len(valid_orders) == 1
            assert valid_orders[0].status == OrderStatus.VALID

    def test_get_order_status(self) -> None:
        """Test getting a specific order's status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OrderManager(tmpdir)

            manager.save_order(
                url="https://acme.example.com/order/1",
                status=OrderStatus.READY,
                domains=["example.com"],
                finalize_url="",
                order_id="test_order",
            )

            info = manager.get_order_status("test_order")

            assert info is not None
            assert info.status == OrderStatus.READY

    def test_get_order_status_not_found(self) -> None:
        """Test getting status of nonexistent order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OrderManager(tmpdir)

            info = manager.get_order_status("nonexistent")

            assert info is None

    def test_cleanup_orders(self) -> None:
        """Test cleaning up orders."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OrderManager(tmpdir)

            # Create an old valid order
            orders_path = Path(tmpdir) / "orders"
            orders_path.mkdir(parents=True, exist_ok=True)

            old_order = {
                "url": "https://acme.example.com/order/old",
                "status": "valid",
                "identifiers": [{"type": "dns", "value": "old.com"}],
                "finalize_url": "",
                "created": (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
            }
            (orders_path / "old_order.json").write_text(json.dumps(old_order))

            # Create a recent pending order
            manager.save_order(
                url="https://acme.example.com/order/new",
                status=OrderStatus.PENDING,
                domains=["new.com"],
                finalize_url="",
                order_id="new_order",
            )

            # Cleanup old and terminal orders
            deleted = manager.cleanup_orders(max_age_days=7)

            assert len(deleted) == 1
            assert deleted[0].status == OrderStatus.VALID

            # Pending order should still exist
            remaining = manager.list_orders()
            assert len(remaining) == 1
            assert remaining[0].status == OrderStatus.PENDING

    def test_cleanup_orders_dry_run(self) -> None:
        """Test dry run cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OrderManager(tmpdir)

            manager.save_order(
                url="https://acme.example.com/order/1",
                status=OrderStatus.VALID,
                domains=["example.com"],
                finalize_url="",
            )

            deleted = manager.cleanup_orders(dry_run=True)

            assert len(deleted) == 1

            # Order should still exist
            orders = manager.list_orders()
            assert len(orders) == 1

    def test_auto_cleanup(self) -> None:
        """Test auto cleanup after successful issuance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OrderManager(tmpdir)

            order_path = manager.save_order(
                url="https://acme.example.com/order/1",
                status=OrderStatus.VALID,
                domains=["example.com"],
                finalize_url="",
            )

            assert order_path.exists()

            manager.auto_cleanup(order_path)

            assert not order_path.exists()

    def test_get_statistics(self) -> None:
        """Test getting order statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OrderManager(tmpdir)

            manager.save_order(
                url="", status=OrderStatus.PENDING,
                domains=["a.com"], finalize_url="",
                order_id="order1",
            )
            manager.save_order(
                url="", status=OrderStatus.PENDING,
                domains=["b.com"], finalize_url="",
                order_id="order2",
            )
            manager.save_order(
                url="", status=OrderStatus.VALID,
                domains=["c.com"], finalize_url="",
                order_id="order3",
            )

            stats = manager.get_statistics()

            assert stats["total"] == 3
            assert stats["by_status"]["pending"] == 2
            assert stats["by_status"]["valid"] == 1
            assert stats["terminal"] == 1

    def test_invalid_order_file(self) -> None:
        """Test handling of invalid order files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orders_path = Path(tmpdir) / "orders"
            orders_path.mkdir(parents=True, exist_ok=True)

            # Create an invalid JSON file
            (orders_path / "invalid.json").write_text("not json")

            # Create a valid order
            manager = OrderManager(tmpdir)
            manager.save_order(
                url="", status=OrderStatus.PENDING,
                domains=["valid.com"], finalize_url="",
                order_id="valid_order",
            )

            orders = manager.list_orders()

            # Should only return the valid order
            assert len(orders) == 1
            assert "valid.com" in orders[0].domains
