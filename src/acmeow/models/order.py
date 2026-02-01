"""Order model for ACME protocol.

Represents ACME certificate orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from acmeow.enums import OrderStatus
from acmeow.models.authorization import Authorization
from acmeow.models.identifier import Identifier


@dataclass(slots=True)
class Order:
    """An ACME certificate order.

    Orders represent a request for a certificate covering one or more
    identifiers. They progress through states as authorizations are
    completed and the certificate is issued.

    Note: This class is mutable because orders are updated as their
    status changes during the certificate issuance process.

    Args:
        status: Current order status.
        url: URL for polling order status.
        identifiers: List of identifiers the certificate will cover.
        finalize_url: URL for submitting the CSR.
        expires: Expiration timestamp for the order.
        certificate_url: URL for downloading the certificate (when ready).
        authorizations: List of authorizations for this order.
        error: Error details if the order failed.
    """

    status: OrderStatus
    url: str
    identifiers: tuple[Identifier, ...]
    finalize_url: str
    expires: str | None = None
    certificate_url: str | None = None
    authorizations: list[Authorization] = field(default_factory=list)
    error: dict[str, str] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], url: str) -> Order:
        """Create an Order from an ACME response dictionary.

        Args:
            data: Order object from ACME server response.
            url: The order URL.

        Returns:
            New Order instance.
        """
        identifiers = tuple(
            Identifier.from_dict(i)
            for i in data.get("identifiers", [])
        )

        return cls(
            status=OrderStatus(data.get("status", "pending")),
            url=url,
            identifiers=identifiers,
            finalize_url=data["finalize"],
            expires=data.get("expires"),
            certificate_url=data.get("certificate"),
            error=data.get("error"),
        )

    def update_from_dict(self, data: dict[str, Any]) -> None:
        """Update the order from an ACME response dictionary.

        Args:
            data: Order object from ACME server response.
        """
        self.status = OrderStatus(data.get("status", self.status.value))
        if "certificate" in data:
            self.certificate_url = data["certificate"]
        if "error" in data:
            self.error = data["error"]

    @property
    def is_pending(self) -> bool:
        """Check if the order is pending authorization completion."""
        return self.status == OrderStatus.PENDING

    @property
    def is_ready(self) -> bool:
        """Check if the order is ready for finalization."""
        return self.status == OrderStatus.READY

    @property
    def is_processing(self) -> bool:
        """Check if the order is being processed."""
        return self.status == OrderStatus.PROCESSING

    @property
    def is_valid(self) -> bool:
        """Check if the order is complete and certificate is ready."""
        return self.status == OrderStatus.VALID

    @property
    def is_invalid(self) -> bool:
        """Check if the order failed."""
        return self.status == OrderStatus.INVALID

    @property
    def is_finalized(self) -> bool:
        """Check if the order has been finalized (processing or valid)."""
        return self.status in (OrderStatus.PROCESSING, OrderStatus.VALID)

    @property
    def domains(self) -> list[str]:
        """Get the list of domain names in this order.

        Returns:
            List of domain/identifier values.
        """
        return [i.value for i in self.identifiers]

    @property
    def common_name(self) -> str:
        """Get the common name (first identifier) for this order.

        Returns:
            The first identifier value.
        """
        return self.identifiers[0].value if self.identifiers else ""

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        domains = ", ".join(self.domains[:3])
        if len(self.domains) > 3:
            domains += f", ... ({len(self.domains)} total)"
        return f"Order({domains}, status={self.status.value})"
