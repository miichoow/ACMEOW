"""DigitalOcean DNS provider.

Provides DNS record management via the DigitalOcean API.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from acmeow.dns.base import DnsProvider, DnsRecord

logger = logging.getLogger(__name__)

DIGITALOCEAN_API_BASE = "https://api.digitalocean.com/v2"


class DigitalOceanDnsProvider(DnsProvider):
    """DNS provider for DigitalOcean.

    Uses the DigitalOcean API to create and delete DNS records.
    Requires an API token with DNS write permissions.

    Args:
        api_token: DigitalOcean API token.
        timeout: Request timeout in seconds. Default 30.

    Example:
        >>> provider = DigitalOceanDnsProvider(api_token="your-api-token")
        >>> handler = DnsProviderHandler(provider)
        >>> client.complete_challenges(handler, ChallengeType.DNS)
    """

    propagation_delay = 60

    def __init__(
        self,
        api_token: str,
        timeout: int = 30,
    ) -> None:
        self._api_token = api_token
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        })
        self._domain_cache: dict[str, str] = {}  # domain -> DO domain name

    def _get_domain_name(self, domain: str) -> str:
        """Get the DigitalOcean domain name for a domain.

        DigitalOcean requires the apex domain name for API calls.

        Args:
            domain: The full domain name.

        Returns:
            The apex domain name.

        Raises:
            ValueError: If the domain is not found in DigitalOcean.
        """
        # Check cache
        if domain in self._domain_cache:
            return self._domain_cache[domain]

        logger.debug("Looking up DigitalOcean domain for %s", domain)

        # Try progressively shorter domain suffixes
        parts = domain.split(".")
        for i in range(len(parts) - 1):
            test_domain = ".".join(parts[i:])

            response = self._session.get(
                f"{DIGITALOCEAN_API_BASE}/domains/{test_domain}",
                timeout=self._timeout,
            )

            if response.status_code == 200:
                data = response.json()
                do_domain: str | None = data.get("domain", {}).get("name")
                if do_domain:
                    self._domain_cache[domain] = do_domain
                    logger.debug("Found DigitalOcean domain %s for %s", do_domain, domain)
                    return do_domain

        raise ValueError(f"Could not find DigitalOcean domain for: {domain}")

    def _get_record_name(self, full_name: str, domain_name: str) -> str:
        """Get the relative record name for DigitalOcean.

        DigitalOcean wants just the subdomain part, not the full name.

        Args:
            full_name: The full record name (e.g., "_acme-challenge.sub.example.com")
            domain_name: The domain name (e.g., "example.com")

        Returns:
            The relative name (e.g., "_acme-challenge.sub")
        """
        suffix = f".{domain_name}"
        if full_name.endswith(suffix):
            return full_name[:-len(suffix)]
        return full_name

    def create_record(
        self,
        domain: str,
        name: str,
        value: str,
        ttl: int = 300,
    ) -> DnsRecord:
        """Create a DNS TXT record in DigitalOcean.

        Args:
            domain: The base domain.
            name: The full record name.
            value: The TXT record value.
            ttl: Time to live in seconds.

        Returns:
            The created DnsRecord with DigitalOcean record ID.
        """
        domain_name = self._get_domain_name(domain)
        record_name = self._get_record_name(name, domain_name)

        logger.debug("Creating DigitalOcean TXT record: %s = %s", name, value)

        response = self._session.post(
            f"{DIGITALOCEAN_API_BASE}/domains/{domain_name}/records",
            json={
                "type": "TXT",
                "name": record_name,
                "data": value,
                "ttl": ttl,
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()

        record_id = str(data.get("domain_record", {}).get("id", ""))
        if not record_id:
            raise RuntimeError("DigitalOcean did not return record ID")

        logger.info("Created DigitalOcean TXT record: %s (id=%s)", name, record_id)

        return DnsRecord(
            name=name,
            type="TXT",
            value=value,
            ttl=ttl,
            id=record_id,
        )

    def delete_record(self, domain: str, record: DnsRecord) -> None:
        """Delete a DNS record from DigitalOcean.

        Args:
            domain: The base domain.
            record: The record to delete (must have DigitalOcean record ID).
        """
        if not record.id:
            raise ValueError("Record does not have a DigitalOcean ID")

        domain_name = self._get_domain_name(domain)

        logger.debug("Deleting DigitalOcean TXT record: %s (id=%s)", record.name, record.id)

        response = self._session.delete(
            f"{DIGITALOCEAN_API_BASE}/domains/{domain_name}/records/{record.id}",
            timeout=self._timeout,
        )
        response.raise_for_status()

        logger.info("Deleted DigitalOcean TXT record: %s", record.name)

    def list_records(
        self,
        domain: str,
        record_type: str | None = None,
    ) -> list[DnsRecord]:
        """List DNS records in DigitalOcean.

        Args:
            domain: The domain to list records for.
            record_type: Filter by record type.

        Returns:
            List of DnsRecord objects.
        """
        domain_name = self._get_domain_name(domain)

        logger.debug("Listing DigitalOcean DNS records for %s", domain)

        params: dict[str, Any] = {"per_page": 200}
        if record_type:
            params["type"] = record_type

        response = self._session.get(
            f"{DIGITALOCEAN_API_BASE}/domains/{domain_name}/records",
            params=params,
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()

        records = []
        for r in data.get("domain_records", []):
            rtype = r.get("type")
            if record_type and rtype != record_type:
                continue

            # Reconstruct full name
            record_name = r.get("name", "")
            full_name = domain_name if record_name == "@" else f"{record_name}.{domain_name}"

            records.append(DnsRecord(
                name=full_name,
                type=rtype,
                value=r.get("data", ""),
                ttl=r.get("ttl", 300),
                id=str(r.get("id", "")),
            ))

        return records

    def get_zone_for_domain(self, domain: str) -> str:
        """Get the DigitalOcean domain name for a domain.

        Args:
            domain: The domain name.

        Returns:
            The DigitalOcean domain name.
        """
        return self._get_domain_name(domain)

    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return "DigitalOceanDnsProvider()"
