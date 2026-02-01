"""Identifier model for ACME protocol.

Represents identifiers (domains, IPs) that certificates are issued for.
"""

from __future__ import annotations

from dataclasses import dataclass

from acmeow.enums import IdentifierType


@dataclass(frozen=True, slots=True)
class Identifier:
    """An ACME identifier representing a domain or IP address.

    Identifiers specify what the certificate will be issued for.
    They are immutable to ensure consistency throughout the certificate
    issuance process.

    Args:
        type: The type of identifier (DNS or IP).
        value: The identifier value (e.g., "example.com" or "192.168.1.1").

    Examples:
        >>> domain = Identifier(IdentifierType.DNS, "example.com")
        >>> ip = Identifier(IdentifierType.IP, "192.168.1.1")
    """

    type: IdentifierType
    value: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary for ACME requests.

        Returns:
            Dictionary with "type" and "value" keys.
        """
        return {"type": self.type.value, "value": self.value}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> Identifier:
        """Create an Identifier from an ACME response dictionary.

        Args:
            data: Dictionary with "type" and "value" keys.

        Returns:
            New Identifier instance.
        """
        return cls(
            type=IdentifierType(data["type"]),
            value=data["value"],
        )

    @classmethod
    def dns(cls, domain: str) -> Identifier:
        """Create a DNS identifier for a domain name.

        Args:
            domain: Domain name (e.g., "example.com").

        Returns:
            DNS Identifier for the domain.
        """
        return cls(IdentifierType.DNS, domain)

    @classmethod
    def ip(cls, address: str) -> Identifier:
        """Create an IP identifier for an IP address.

        Args:
            address: IP address (e.g., "192.168.1.1").

        Returns:
            IP Identifier for the address.
        """
        return cls(IdentifierType.IP, address)

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return f"{self.type.value}:{self.value}"
