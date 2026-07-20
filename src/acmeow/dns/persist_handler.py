"""DNS provider handler for ACME DNS-PERSIST-01 challenges.

Bridges DNS providers to the DnsPersistHandler interface.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from acmeow.dns.base import DnsRecord
from acmeow.handlers.dns_persist import DnsPersistHandler

if TYPE_CHECKING:
    from acmeow.dns.base import DnsProvider

logger = logging.getLogger(__name__)


class DnsProviderPersistHandler(DnsPersistHandler):
    """DNS-PERSIST-01 handler that publishes records through a DNS provider.

    Args:
        provider: The DNS provider used to manage records.
        ttl: Time to live for the published record. Default 3600. This is
            higher than the DNS-01 default because the record is long-lived.
        persist: Whether to keep the record after validation. Default True.
        replace_existing: Whether to delete a pre-existing record at the
            validation name before publishing. Default True, so that a record
            left by an earlier issuance does not accumulate duplicates.
            Requires the provider to implement ``list_records``.

    Example:
        >>> from acmeow.dns import get_dns_provider
        >>> provider = get_dns_provider("cloudflare", api_token="...")
        >>> handler = DnsProviderPersistHandler(provider)
        >>> client.complete_dns_persist_challenges(handler)
    """

    def __init__(
        self,
        provider: DnsProvider,
        ttl: int = 3600,
        persist: bool = True,
        replace_existing: bool = True,
    ) -> None:
        self._provider = provider
        self._ttl = ttl
        self.persist = persist
        self._replace_existing = replace_existing
        self._records: dict[str, DnsRecord] = {}  # domain -> record

    @property
    def propagation_delay(self) -> int:  # type: ignore[override]
        """Seconds to wait after publishing for DNS propagation."""
        return self._provider.propagation_delay

    @property
    def provider(self) -> DnsProvider:
        """The underlying DNS provider."""
        return self._provider

    def setup(self, domain: str, record_name: str, record_value: str) -> None:
        """Publish the persistent validation record via the provider.

        Args:
            domain: The domain being validated.
            record_name: The Validation Domain Name.
            record_value: The issue-value formatted TXT record value.
        """
        base_domain = self._get_base_domain(domain)

        if self._replace_existing:
            self._remove_existing(base_domain, record_name)

        logger.info(
            "Publishing DNS-PERSIST TXT record via %s: %s = %s",
            self._provider.__class__.__name__,
            record_name,
            record_value,
        )

        record = self._provider.create_record(
            domain=base_domain,
            name=record_name,
            value=record_value,
            ttl=self._ttl,
        )

        self._records[domain] = record
        logger.debug("Published DNS record: %s (id=%s)", record, record.id)

    def cleanup(self, domain: str, record_name: str) -> None:
        """Remove the record, if ``persist`` is False.

        Args:
            domain: The domain that was validated.
            record_name: The Validation Domain Name.
        """
        record = self._records.pop(domain, None)

        if self.persist:
            logger.debug("Keeping persistent record %s", record_name)
            return

        if record is None:
            logger.warning("No record found for cleanup: %s", domain)
            return

        logger.info(
            "Removing DNS-PERSIST TXT record via %s: %s",
            self._provider.__class__.__name__,
            record_name,
        )

        try:
            self._provider.delete_record(self._get_base_domain(domain), record)
        except Exception as e:
            logger.warning("Failed to remove record %s: %s", record_name, e)

    def _remove_existing(self, base_domain: str, record_name: str) -> None:
        """Delete any pre-existing TXT record at the validation name.

        A stale record from an earlier issuance would otherwise sit alongside
        the new one. Providers that cannot list records are skipped, since
        creating the new record is still the useful outcome.
        """
        try:
            existing = self._provider.list_records(base_domain, record_type="TXT")
        except NotImplementedError:
            logger.debug(
                "%s cannot list records; skipping replacement of %s",
                self._provider.__class__.__name__,
                record_name,
            )
            return
        except Exception as e:
            logger.warning("Failed to list records for %s: %s", base_domain, e)
            return

        wanted = record_name.rstrip(".")
        for record in existing:
            if record.name.rstrip(".") != wanted:
                continue
            try:
                self._provider.delete_record(base_domain, record)
                logger.info("Replaced stale DNS-PERSIST record: %s", record.name)
            except Exception as e:
                logger.warning("Failed to remove stale record %s: %s", record.name, e)

    def _get_base_domain(self, domain: str) -> str:
        """Extract the base domain from a full domain.

        Mirrors :class:`~acmeow.dns.handler.DnsProviderHandler` so both
        challenge types resolve zones the same way.

        Args:
            domain: The full domain name.

        Returns:
            The base domain.
        """
        if domain.startswith("*."):
            domain = domain[2:]

        try:
            return self._provider.get_zone_for_domain(domain)
        except NotImplementedError:
            pass

        parts = domain.split(".")
        if len(parts) > 2:
            return ".".join(parts[-2:])
        return domain
