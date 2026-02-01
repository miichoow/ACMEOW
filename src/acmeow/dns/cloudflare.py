"""Cloudflare DNS provider.

Provides DNS record management via the Cloudflare API.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from acmeow.dns.base import DnsProvider, DnsRecord

logger = logging.getLogger(__name__)

CLOUDFLARE_API_BASE = "https://api.cloudflare.com/client/v4"


class CloudflareDnsProvider(DnsProvider):
    """DNS provider for Cloudflare.

    Uses the Cloudflare API to create and delete DNS records.
    Requires an API token with DNS edit permissions.

    Args:
        api_token: Cloudflare API token with DNS edit permissions.
        zone_id: Optional zone ID. If not provided, will be looked up by domain.
        timeout: Request timeout in seconds. Default 30.

    Example:
        >>> provider = CloudflareDnsProvider(api_token="your-api-token")
        >>> handler = DnsProviderHandler(provider)
        >>> client.complete_challenges(handler, ChallengeType.DNS)
    """

    propagation_delay = 30  # Cloudflare is typically fast

    def __init__(
        self,
        api_token: str,
        zone_id: str | None = None,
        timeout: int = 30,
    ) -> None:
        self._api_token = api_token
        self._zone_id = zone_id
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        })
        self._zone_cache: dict[str, str] = {}  # domain -> zone_id

    def _get_zone_id(self, domain: str) -> str:
        """Get the Cloudflare zone ID for a domain.

        Args:
            domain: The domain to look up.

        Returns:
            The zone ID.

        Raises:
            ValueError: If the zone cannot be found.
        """
        if self._zone_id:
            return self._zone_id

        # Check cache
        if domain in self._zone_cache:
            return self._zone_cache[domain]

        # Look up zone
        logger.debug("Looking up Cloudflare zone for %s", domain)

        # Try progressively shorter domain suffixes
        parts = domain.split(".")
        for i in range(len(parts) - 1):
            test_domain = ".".join(parts[i:])

            response = self._session.get(
                f"{CLOUDFLARE_API_BASE}/zones",
                params={"name": test_domain},
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("success") and data.get("result"):
                zone_id: str = data["result"][0]["id"]
                self._zone_cache[domain] = zone_id
                logger.debug("Found zone %s for %s", zone_id, domain)
                return zone_id

        raise ValueError(f"Could not find Cloudflare zone for domain: {domain}")

    def create_record(
        self,
        domain: str,
        name: str,
        value: str,
        ttl: int = 300,
    ) -> DnsRecord:
        """Create a DNS TXT record in Cloudflare.

        Args:
            domain: The base domain.
            name: The full record name.
            value: The TXT record value.
            ttl: Time to live in seconds. Use 1 for automatic.

        Returns:
            The created DnsRecord with Cloudflare record ID.
        """
        zone_id = self._get_zone_id(domain)

        logger.debug("Creating Cloudflare TXT record: %s = %s", name, value)

        response = self._session.post(
            f"{CLOUDFLARE_API_BASE}/zones/{zone_id}/dns_records",
            json={
                "type": "TXT",
                "name": name,
                "content": value,
                "ttl": ttl if ttl > 1 else 1,  # 1 = automatic
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            errors = data.get("errors", [])
            error_msg = "; ".join(e.get("message", str(e)) for e in errors)
            raise RuntimeError(f"Cloudflare API error: {error_msg}")

        record_id = data["result"]["id"]
        logger.info("Created Cloudflare TXT record: %s (id=%s)", name, record_id)

        return DnsRecord(
            name=name,
            type="TXT",
            value=value,
            ttl=ttl,
            id=record_id,
        )

    def delete_record(self, domain: str, record: DnsRecord) -> None:
        """Delete a DNS record from Cloudflare.

        Args:
            domain: The base domain.
            record: The record to delete (must have Cloudflare record ID).
        """
        if not record.id:
            raise ValueError("Record does not have a Cloudflare ID")

        zone_id = self._get_zone_id(domain)

        logger.debug("Deleting Cloudflare TXT record: %s (id=%s)", record.name, record.id)

        response = self._session.delete(
            f"{CLOUDFLARE_API_BASE}/zones/{zone_id}/dns_records/{record.id}",
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            errors = data.get("errors", [])
            error_msg = "; ".join(e.get("message", str(e)) for e in errors)
            raise RuntimeError(f"Cloudflare API error: {error_msg}")

        logger.info("Deleted Cloudflare TXT record: %s", record.name)

    def list_records(
        self,
        domain: str,
        record_type: str | None = None,
    ) -> list[DnsRecord]:
        """List DNS records in Cloudflare.

        Args:
            domain: The domain to list records for.
            record_type: Filter by record type.

        Returns:
            List of DnsRecord objects.
        """
        zone_id = self._get_zone_id(domain)

        params: dict[str, Any] = {"per_page": 100}
        if record_type:
            params["type"] = record_type

        logger.debug("Listing Cloudflare DNS records for %s", domain)

        response = self._session.get(
            f"{CLOUDFLARE_API_BASE}/zones/{zone_id}/dns_records",
            params=params,
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            errors = data.get("errors", [])
            error_msg = "; ".join(e.get("message", str(e)) for e in errors)
            raise RuntimeError(f"Cloudflare API error: {error_msg}")

        records = []
        for r in data.get("result", []):
            records.append(DnsRecord(
                name=r["name"],
                type=r["type"],
                value=r["content"],
                ttl=r["ttl"],
                id=r["id"],
            ))

        return records

    def get_zone_for_domain(self, domain: str) -> str:
        """Get the Cloudflare zone name for a domain.

        Args:
            domain: The domain name.

        Returns:
            The zone name.
        """
        zone_id = self._get_zone_id(domain)

        # Get zone details
        response = self._session.get(
            f"{CLOUDFLARE_API_BASE}/zones/{zone_id}",
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("success") and data.get("result"):
            zone_name: str = data["result"]["name"]
            return zone_name

        return domain

    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return f"CloudflareDnsProvider(zone_id={self._zone_id})"
