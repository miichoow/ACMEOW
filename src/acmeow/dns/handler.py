"""DNS provider handler for ACME DNS-01 challenges.

Bridges DNS providers to the ChallengeHandler interface.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from acmeow._internal.encoding import base64url_encode
from acmeow.dns.base import DnsRecord
from acmeow.handlers.base import ChallengeHandler

if TYPE_CHECKING:
    from acmeow.dns.base import DnsProvider

logger = logging.getLogger(__name__)


class DnsProviderHandler(ChallengeHandler):
    """Challenge handler that uses a DNS provider for DNS-01 challenges.

    This handler bridges the DnsProvider interface to the ChallengeHandler
    interface, managing the full lifecycle of DNS TXT records for ACME
    DNS-01 challenges.

    Args:
        provider: The DNS provider to use for record management.
        ttl: Time to live for created records. Default 300.

    Example:
        >>> from acmeow.dns import get_dns_provider
        >>> provider = get_dns_provider("cloudflare", api_token="...")
        >>> handler = DnsProviderHandler(provider)
        >>> client.complete_challenges(handler, ChallengeType.DNS)
    """

    def __init__(
        self,
        provider: DnsProvider,
        ttl: int = 300,
    ) -> None:
        self._provider = provider
        self._ttl = ttl
        self._records: dict[str, DnsRecord] = {}  # domain -> record

    @property
    def propagation_delay(self) -> int:
        """Seconds to wait after creating DNS record for propagation."""
        return self._provider.propagation_delay

    @property
    def provider(self) -> DnsProvider:
        """The underlying DNS provider."""
        return self._provider

    def setup(self, domain: str, token: str, key_authorization: str) -> None:
        """Create DNS TXT record for the challenge.

        Computes the TXT record value from the key authorization and
        creates the _acme-challenge record using the provider.

        Args:
            domain: The domain being validated.
            token: The challenge token (not used for DNS record value).
            key_authorization: The key authorization string to hash.
        """
        record_name = self._get_record_name(domain)
        record_value = self._compute_record_value(key_authorization)

        logger.info(
            "Creating DNS TXT record via %s: %s = %s",
            self._provider.__class__.__name__,
            record_name,
            record_value,
        )

        # Get the base domain for the provider
        base_domain = self._get_base_domain(domain)

        record = self._provider.create_record(
            domain=base_domain,
            name=record_name,
            value=record_value,
            ttl=self._ttl,
        )

        # Store record for cleanup
        self._records[domain] = record

        logger.debug("Created DNS record: %s (id=%s)", record, record.id)

    def cleanup(self, domain: str, token: str) -> None:
        """Remove DNS TXT record.

        Args:
            domain: The domain that was validated.
            token: The challenge token (not used).
        """
        record = self._records.pop(domain, None)
        if record is None:
            logger.warning("No record found for cleanup: %s", domain)
            return

        record_name = self._get_record_name(domain)
        logger.info(
            "Removing DNS TXT record via %s: %s",
            self._provider.__class__.__name__,
            record_name,
        )

        try:
            base_domain = self._get_base_domain(domain)
            self._provider.delete_record(base_domain, record)
            logger.debug("Deleted DNS record: %s", record)
        except Exception as e:
            logger.warning("Failed to cleanup DNS record %s: %s", record_name, e)

    @staticmethod
    def _get_record_name(domain: str) -> str:
        """Get the DNS record name for a domain.

        Args:
            domain: The domain being validated.

        Returns:
            The full record name (e.g., "_acme-challenge.example.com").
        """
        return f"_acme-challenge.{domain}"

    @staticmethod
    def _compute_record_value(key_authorization: str) -> str:
        """Compute the DNS TXT record value.

        Args:
            key_authorization: The key authorization string.

        Returns:
            Base64url-encoded SHA-256 hash of the key authorization.
        """
        digest = hashlib.sha256(key_authorization.encode("utf-8")).digest()
        return base64url_encode(digest)

    def _get_base_domain(self, domain: str) -> str:
        """Extract the base domain from a full domain.

        For subdomains like "sub.example.com", returns "example.com".
        For wildcards like "*.example.com", returns "example.com".

        This is a simple heuristic; providers may override get_zone_for_domain
        for more accurate zone detection.

        Args:
            domain: The full domain name.

        Returns:
            The base domain.
        """
        # Remove wildcard prefix
        if domain.startswith("*."):
            domain = domain[2:]

        # Use provider's zone detection if available
        try:
            return self._provider.get_zone_for_domain(domain)
        except NotImplementedError:
            pass

        # Simple heuristic: take last two parts
        parts = domain.split(".")
        if len(parts) > 2:
            return ".".join(parts[-2:])
        return domain
