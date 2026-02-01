"""Order management utilities for ACME certificates.

Provides utilities for listing, cleaning up, and managing certificate orders.
"""

from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from acmeow.enums import OrderStatus

logger = logging.getLogger(__name__)


@dataclass
class OrderInfo:
    """Information about a certificate order.

    Attributes:
        url: The ACME order URL.
        status: Current order status.
        domains: List of domain names in the order.
        created: When the order was created (if known).
        expires: When the order expires (if known).
        path: Path to the order file on disk.
        certificate_url: URL for downloading the certificate (if issued).
    """

    url: str
    status: OrderStatus
    domains: list[str]
    created: datetime | None
    expires: datetime | None
    path: Path
    certificate_url: str | None = None

    @property
    def is_terminal(self) -> bool:
        """Check if the order is in a terminal state (valid or invalid)."""
        return self.status in (OrderStatus.VALID, OrderStatus.INVALID)

    @property
    def is_expired(self) -> bool:
        """Check if the order has expired."""
        if self.expires is None:
            return False
        return datetime.now(timezone.utc) > self.expires

    @property
    def age_days(self) -> float | None:
        """Get the age of the order in days."""
        if self.created is None:
            return None
        delta = datetime.now(timezone.utc) - self.created
        return delta.total_seconds() / 86400

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        domains = ", ".join(self.domains[:2])
        if len(self.domains) > 2:
            domains += f", ... ({len(self.domains)} total)"
        return f"OrderInfo({domains}, status={self.status.value})"


class OrderManager:
    """Manages certificate orders stored on disk.

    Provides utilities for listing, cleaning up, and managing orders
    that have been saved to the storage path.

    Args:
        storage_path: Path to the ACME storage directory.

    Example:
        >>> manager = OrderManager(Path("./acme_data"))
        >>> orders = manager.list_orders()
        >>> manager.cleanup_orders(max_age_days=30)
    """

    def __init__(self, storage_path: Path | str) -> None:
        self._storage_path = Path(storage_path)
        self._orders_path = self._storage_path / "orders"

    @property
    def storage_path(self) -> Path:
        """The base storage path."""
        return self._storage_path

    @property
    def orders_path(self) -> Path:
        """The orders directory path."""
        return self._orders_path

    def list_orders(
        self,
        status: OrderStatus | None = None,
        include_expired: bool = True,
    ) -> list[OrderInfo]:
        """List all saved orders.

        Args:
            status: Filter by order status. None returns all.
            include_expired: Whether to include expired orders. Default True.

        Returns:
            List of OrderInfo objects, sorted by creation time (newest first).
        """
        orders: list[OrderInfo] = []

        if not self._orders_path.exists():
            return orders

        # Find all order files
        for order_file in self._orders_path.glob("*.json"):
            try:
                order_info = self._load_order_info(order_file)
                if order_info is None:
                    continue

                # Apply filters
                if status is not None and order_info.status != status:
                    continue
                if not include_expired and order_info.is_expired:
                    continue

                orders.append(order_info)

            except Exception as e:
                logger.warning("Failed to load order file %s: %s", order_file, e)

        # Sort by creation time (newest first), with None dates last
        orders.sort(
            key=lambda o: o.created or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

        return orders

    def _load_order_info(self, order_file: Path) -> OrderInfo | None:
        """Load order info from a file.

        Args:
            order_file: Path to the order JSON file.

        Returns:
            OrderInfo or None if the file is invalid.
        """
        try:
            data = json.loads(order_file.read_text())

            # Parse status
            status_str = data.get("status", "pending")
            try:
                status = OrderStatus(status_str)
            except ValueError:
                status = OrderStatus.PENDING

            # Parse identifiers to domains
            domains = []
            for identifier in data.get("identifiers", []):
                domains.append(identifier.get("value", ""))

            # Parse dates
            created = None
            if "created" in data:
                with contextlib.suppress(ValueError, TypeError):
                    created = datetime.fromisoformat(data["created"])

            # Try to get created time from file mtime
            if created is None:
                try:
                    mtime = order_file.stat().st_mtime
                    created = datetime.fromtimestamp(mtime, tz=timezone.utc)
                except OSError:
                    pass

            expires = None
            if "expires" in data and data["expires"]:
                try:
                    expires_str = data["expires"]
                    # Handle both ISO format and RFC 3339
                    if expires_str.endswith("Z"):
                        expires_str = expires_str[:-1] + "+00:00"
                    expires = datetime.fromisoformat(expires_str)
                except (ValueError, TypeError):
                    pass

            return OrderInfo(
                url=data.get("url", ""),
                status=status,
                domains=domains,
                created=created,
                expires=expires,
                path=order_file,
                certificate_url=data.get("certificate_url"),
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.debug("Invalid order file %s: %s", order_file, e)
            return None

    def get_order_status(self, order_id: str) -> OrderInfo | None:
        """Get the status of a specific order.

        Args:
            order_id: The order identifier (filename without extension).

        Returns:
            OrderInfo or None if not found.
        """
        order_file = self._orders_path / f"{order_id}.json"
        if not order_file.exists():
            return None

        return self._load_order_info(order_file)

    def cleanup_orders(
        self,
        max_age_days: int | None = None,
        status: list[OrderStatus] | None = None,
        dry_run: bool = False,
    ) -> list[OrderInfo]:
        """Clean up old or terminal orders.

        Removes order files that match the specified criteria.

        Args:
            max_age_days: Remove orders older than this many days.
                If None, age is not considered.
            status: Remove orders with these statuses.
                Default is [VALID, INVALID] (terminal states).
            dry_run: If True, don't actually delete files, just return
                what would be deleted.

        Returns:
            List of OrderInfo that were (or would be) deleted.
        """
        if status is None:
            status = [OrderStatus.VALID, OrderStatus.INVALID]

        deleted: list[OrderInfo] = []
        now = datetime.now(timezone.utc)

        for order_info in self.list_orders():
            should_delete = False

            # Check status
            if order_info.status in status:
                should_delete = True

            # Check age
            if max_age_days is not None and order_info.created is not None:
                age = now - order_info.created
                if age > timedelta(days=max_age_days):
                    should_delete = True

            # Check expiration
            if order_info.is_expired:
                should_delete = True

            if should_delete:
                if not dry_run:
                    try:
                        order_info.path.unlink()
                        logger.info("Deleted order: %s", order_info.path)
                    except OSError as e:
                        logger.warning("Failed to delete %s: %s", order_info.path, e)
                        continue

                deleted.append(order_info)

        return deleted

    def auto_cleanup(self, order_path: Path | None = None) -> None:
        """Automatically clean up after successful certificate issuance.

        Called after a certificate is successfully obtained to clean up
        the completed order file.

        Args:
            order_path: Specific order file to clean up.
                If None, cleans up the current order file.
        """
        if order_path is None:
            order_path = self._orders_path / "current_order.json"

        if order_path.exists():
            try:
                order_path.unlink()
                logger.debug("Auto-cleaned order: %s", order_path)
            except OSError as e:
                logger.warning("Failed to auto-clean %s: %s", order_path, e)

    def save_order(
        self,
        url: str,
        status: OrderStatus,
        domains: list[str],
        finalize_url: str,
        expires: str | None = None,
        certificate_url: str | None = None,
        order_id: str = "current_order",
    ) -> Path:
        """Save an order to disk.

        Args:
            url: The ACME order URL.
            status: Current order status.
            domains: List of domain names.
            finalize_url: URL for finalizing the order.
            expires: Order expiration timestamp.
            certificate_url: URL for downloading the certificate.
            order_id: Order identifier for the filename.

        Returns:
            Path to the saved order file.
        """
        self._orders_path.mkdir(parents=True, exist_ok=True)

        order_data = {
            "url": url,
            "status": status.value,
            "identifiers": [{"type": "dns", "value": d} for d in domains],
            "finalize_url": finalize_url,
            "expires": expires,
            "certificate_url": certificate_url,
            "created": datetime.now(timezone.utc).isoformat(),
        }

        order_path = self._orders_path / f"{order_id}.json"
        order_path.write_text(json.dumps(order_data, indent=2))

        logger.debug("Saved order to %s", order_path)
        return order_path

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about stored orders.

        Returns:
            Dictionary with order statistics.
        """
        orders = self.list_orders()

        stats: dict[str, Any] = {
            "total": len(orders),
            "by_status": {},
            "expired": 0,
            "terminal": 0,
        }

        for order in orders:
            status_name = order.status.value
            stats["by_status"][status_name] = stats["by_status"].get(status_name, 0) + 1

            if order.is_expired:
                stats["expired"] += 1
            if order.is_terminal:
                stats["terminal"] += 1

        return stats
