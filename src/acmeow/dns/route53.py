"""AWS Route53 DNS provider.

Provides DNS record management via the AWS Route53 API using boto3.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from acmeow.dns.base import DnsProvider, DnsRecord

if TYPE_CHECKING:
    from mypy_boto3_route53 import Route53Client

logger = logging.getLogger(__name__)


class Route53DnsProvider(DnsProvider):
    """DNS provider for AWS Route53.

    Uses boto3 to interact with the Route53 API. Requires boto3 to be
    installed and AWS credentials to be configured.

    Args:
        hosted_zone_id: Optional hosted zone ID. If not provided, will be
            looked up by domain.
        wait_for_change: Whether to wait for Route53 changes to propagate.
            Default True.
        aws_access_key_id: Optional AWS access key ID.
        aws_secret_access_key: Optional AWS secret access key.
        region_name: AWS region. Default "us-east-1".

    Example:
        >>> provider = Route53DnsProvider()  # Uses default credentials
        >>> handler = DnsProviderHandler(provider)
        >>> client.complete_challenges(handler, ChallengeType.DNS)

    Note:
        Requires boto3 to be installed: pip install boto3
    """

    propagation_delay = 60  # Route53 can be slow

    def __init__(
        self,
        hosted_zone_id: str | None = None,
        wait_for_change: bool = True,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        region_name: str = "us-east-1",
    ) -> None:
        try:
            import boto3
        except ImportError as e:
            raise ImportError(
                "Route53DnsProvider requires boto3. "
                "Install it with: pip install boto3"
            ) from e

        self._hosted_zone_id = hosted_zone_id
        self._wait_for_change = wait_for_change
        self._zone_cache: dict[str, str] = {}  # domain -> zone_id

        # Create boto3 client
        kwargs: dict[str, Any] = {"region_name": region_name}
        if aws_access_key_id and aws_secret_access_key:
            kwargs["aws_access_key_id"] = aws_access_key_id
            kwargs["aws_secret_access_key"] = aws_secret_access_key

        self._client: Route53Client = boto3.client("route53", **kwargs)

    def _get_hosted_zone_id(self, domain: str) -> str:
        """Get the Route53 hosted zone ID for a domain.

        Args:
            domain: The domain to look up.

        Returns:
            The hosted zone ID.

        Raises:
            ValueError: If the zone cannot be found.
        """
        if self._hosted_zone_id:
            return self._hosted_zone_id

        # Check cache
        if domain in self._zone_cache:
            return self._zone_cache[domain]

        logger.debug("Looking up Route53 hosted zone for %s", domain)

        # List hosted zones and find matching one
        paginator = self._client.get_paginator("list_hosted_zones")

        # Try progressively shorter domain suffixes
        parts = domain.split(".")
        for i in range(len(parts) - 1):
            test_domain = ".".join(parts[i:])
            # Route53 zone names end with a dot
            zone_name = f"{test_domain}."

            for page in paginator.paginate():
                for zone in page.get("HostedZones", []):
                    if zone["Name"] == zone_name:
                        # Extract zone ID (format: /hostedzone/XXXXX)
                        zone_id = zone["Id"].split("/")[-1]
                        self._zone_cache[domain] = zone_id
                        logger.debug("Found hosted zone %s for %s", zone_id, domain)
                        return str(zone_id)

        raise ValueError(f"Could not find Route53 hosted zone for domain: {domain}")

    def _wait_for_change_sync(self, change_id: str, max_wait: int = 120) -> None:
        """Wait for a Route53 change to complete.

        Args:
            change_id: The change ID to wait for.
            max_wait: Maximum wait time in seconds.
        """
        logger.debug("Waiting for Route53 change %s to sync", change_id)
        start_time = time.time()

        while time.time() - start_time < max_wait:
            response = self._client.get_change(Id=change_id)
            status = response.get("ChangeInfo", {}).get("Status")

            if status == "INSYNC":
                logger.debug("Route53 change %s is now INSYNC", change_id)
                return

            logger.debug("Route53 change %s status: %s", change_id, status)
            time.sleep(5)

        logger.warning("Route53 change %s did not sync within %ds", change_id, max_wait)

    def create_record(
        self,
        domain: str,
        name: str,
        value: str,
        ttl: int = 300,
    ) -> DnsRecord:
        """Create a DNS TXT record in Route53.

        Args:
            domain: The base domain.
            name: The full record name.
            value: The TXT record value.
            ttl: Time to live in seconds.

        Returns:
            The created DnsRecord.
        """
        zone_id = self._get_hosted_zone_id(domain)

        # Route53 record names must end with a dot
        record_name = f"{name}." if not name.endswith(".") else name

        logger.debug("Creating Route53 TXT record: %s = %s", name, value)

        response = self._client.change_resource_record_sets(
            HostedZoneId=zone_id,
            ChangeBatch={
                "Comment": f"ACME challenge for {domain}",
                "Changes": [
                    {
                        "Action": "UPSERT",
                        "ResourceRecordSet": {
                            "Name": record_name,
                            "Type": "TXT",
                            "TTL": ttl,
                            "ResourceRecords": [{"Value": f'"{value}"'}],
                        },
                    }
                ],
            },
        )

        change_id = response.get("ChangeInfo", {}).get("Id", "")
        logger.info("Created Route53 TXT record: %s (change_id=%s)", name, change_id)

        if self._wait_for_change and change_id:
            self._wait_for_change_sync(change_id)

        return DnsRecord(
            name=name,
            type="TXT",
            value=value,
            ttl=ttl,
            id=change_id,
        )

    def delete_record(self, domain: str, record: DnsRecord) -> None:
        """Delete a DNS record from Route53.

        Args:
            domain: The base domain.
            record: The record to delete.
        """
        zone_id = self._get_hosted_zone_id(domain)

        # Route53 record names must end with a dot
        record_name = f"{record.name}." if not record.name.endswith(".") else record.name

        logger.debug("Deleting Route53 TXT record: %s", record.name)

        response = self._client.change_resource_record_sets(
            HostedZoneId=zone_id,
            ChangeBatch={
                "Comment": f"ACME challenge cleanup for {domain}",
                "Changes": [
                    {
                        "Action": "DELETE",
                        "ResourceRecordSet": {
                            "Name": record_name,
                            "Type": "TXT",
                            "TTL": record.ttl,
                            "ResourceRecords": [{"Value": f'"{record.value}"'}],
                        },
                    }
                ],
            },
        )

        change_id = response.get("ChangeInfo", {}).get("Id", "")
        logger.info("Deleted Route53 TXT record: %s (change_id=%s)", record.name, change_id)

    def list_records(
        self,
        domain: str,
        record_type: str | None = None,
    ) -> list[DnsRecord]:
        """List DNS records in Route53.

        Args:
            domain: The domain to list records for.
            record_type: Filter by record type.

        Returns:
            List of DnsRecord objects.
        """
        zone_id = self._get_hosted_zone_id(domain)

        logger.debug("Listing Route53 DNS records for %s", domain)

        records = []
        paginator = self._client.get_paginator("list_resource_record_sets")

        for page in paginator.paginate(HostedZoneId=zone_id):
            for rrs in page.get("ResourceRecordSets", []):
                rtype = rrs.get("Type")
                if record_type and rtype != record_type:
                    continue

                name = rrs.get("Name", "").rstrip(".")

                for rr in rrs.get("ResourceRecords", []):
                    value = rr.get("Value", "")
                    # Strip quotes from TXT records
                    if rtype == "TXT" and value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]

                    records.append(DnsRecord(
                        name=name,
                        type=rtype,
                        value=value,
                        ttl=rrs.get("TTL", 300),
                    ))

        return records

    def get_zone_for_domain(self, domain: str) -> str:
        """Get the Route53 zone name for a domain.

        Args:
            domain: The domain name.

        Returns:
            The zone name.
        """
        zone_id = self._get_hosted_zone_id(domain)

        response = self._client.get_hosted_zone(Id=zone_id)
        zone_name: str = response.get("HostedZone", {}).get("Name", "")

        # Remove trailing dot
        return zone_name.rstrip(".")

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return f"Route53DnsProvider(hosted_zone_id={self._hosted_zone_id})"
