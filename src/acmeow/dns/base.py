"""Base classes for DNS provider system.

Provides abstract base class and data models for DNS providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DnsRecord:
    """Represents a DNS record.

    Attributes:
        name: The full DNS record name (e.g., "_acme-challenge.example.com").
        type: The record type (e.g., "TXT", "A", "CNAME").
        value: The record value.
        ttl: Time to live in seconds.
        id: Provider-specific record identifier (for deletion).
    """

    name: str
    type: str
    value: str
    ttl: int = 300
    id: str | None = None

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return f"{self.name} {self.ttl} IN {self.type} {self.value}"


class DnsProvider(ABC):
    """Abstract base class for DNS providers.

    DNS providers implement the interface for creating and deleting
    DNS records with a specific DNS service. They are used by
    DnsProviderHandler to automate DNS-01 challenge completion.

    Subclasses must implement create_record(), delete_record(), and
    optionally list_records().

    Attributes:
        propagation_delay: Seconds to wait after record changes for propagation.

    Example:
        >>> class MyDnsProvider(DnsProvider):
        ...     def create_record(self, domain, name, value, ttl):
        ...         # Create TXT record via API
        ...         return DnsRecord(name, "TXT", value, ttl, id="123")
        ...
        ...     def delete_record(self, domain, record):
        ...         # Delete record via API
        ...         pass
    """

    propagation_delay: int = 60
    """Seconds to wait after record changes for DNS propagation."""

    @abstractmethod
    def create_record(
        self,
        domain: str,
        name: str,
        value: str,
        ttl: int = 300,
    ) -> DnsRecord:
        """Create a DNS TXT record.

        Args:
            domain: The base domain (e.g., "example.com").
            name: The full record name (e.g., "_acme-challenge.example.com").
            value: The TXT record value.
            ttl: Time to live in seconds. Default 300.

        Returns:
            The created DnsRecord with provider-assigned ID.

        Raises:
            Exception: If record creation fails.
        """

    @abstractmethod
    def delete_record(self, domain: str, record: DnsRecord) -> None:
        """Delete a DNS record.

        Args:
            domain: The base domain (e.g., "example.com").
            record: The record to delete (with provider ID).

        Raises:
            Exception: If record deletion fails.
        """

    def list_records(
        self,
        domain: str,
        record_type: str | None = None,
    ) -> list[DnsRecord]:
        """List DNS records for a domain.

        Optional method to list existing records. Useful for debugging
        and verification.

        Args:
            domain: The domain to list records for.
            record_type: Filter by record type (e.g., "TXT").

        Returns:
            List of DnsRecord objects.
        """
        raise NotImplementedError("list_records not implemented for this provider")

    def get_zone_for_domain(self, domain: str) -> str:
        """Get the DNS zone name for a domain.

        Override this method if the provider requires zone lookup.
        By default, returns the domain as-is.

        Args:
            domain: The domain name.

        Returns:
            The zone name (e.g., "example.com" for "sub.example.com").
        """
        return domain

    def supports_record_type(self, record_type: str) -> bool:
        """Check if the provider supports a record type.

        Args:
            record_type: The record type to check.

        Returns:
            True if the record type is supported.
        """
        return record_type == "TXT"

    def close(self) -> None:  # noqa: B027
        """Close any resources held by the provider.

        Override this method if the provider holds resources that
        need to be cleaned up (e.g., API sessions).
        """

    def __enter__(self) -> DnsProvider:
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.close()

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return f"{self.__class__.__name__}()"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DnsProvider:
        """Create a provider instance from configuration.

        Override this method to support configuration-based instantiation.

        Args:
            config: Provider configuration dictionary.

        Returns:
            Configured provider instance.
        """
        return cls(**config)
