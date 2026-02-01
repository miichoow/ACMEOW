"""Manual DNS provider for interactive use.

Provides a DNS provider that prompts the user to manually create
and delete DNS records.
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

from acmeow.dns.base import DnsProvider, DnsRecord

logger = logging.getLogger(__name__)


class ManualDnsProvider(DnsProvider):
    """Interactive DNS provider that prompts for manual record management.

    This provider prints instructions for creating DNS TXT records and
    waits for user confirmation. Useful for testing, one-off certificates,
    or when API access to DNS is not available.

    Args:
        input_stream: Input stream for reading user confirmation.
            Default is sys.stdin.
        output_stream: Output stream for printing instructions.
            Default is sys.stderr.
        prompt_delete: Whether to prompt before record deletion. Default False.

    Example:
        >>> provider = ManualDnsProvider()
        >>> handler = DnsProviderHandler(provider)
        >>> client.complete_challenges(handler, ChallengeType.DNS)
        # Provider will print instructions and wait for user input
    """

    propagation_delay = 120  # Manual records may take longer to propagate

    def __init__(
        self,
        input_stream: TextIO | None = None,
        output_stream: TextIO | None = None,
        prompt_delete: bool = False,
    ) -> None:
        self._input = input_stream or sys.stdin
        self._output = output_stream or sys.stderr
        self._prompt_delete = prompt_delete
        self._record_counter = 0

    def _print(self, message: str) -> None:
        """Print a message to the output stream."""
        print(message, file=self._output)

    def _prompt(self, message: str) -> str:
        """Prompt the user for input."""
        self._print(message)
        self._output.flush()
        return self._input.readline().strip()

    def create_record(
        self,
        domain: str,
        name: str,
        value: str,
        ttl: int = 300,
    ) -> DnsRecord:
        """Display instructions for creating a DNS TXT record.

        Prints the record details and waits for user confirmation
        that the record has been created.

        Args:
            domain: The base domain.
            name: The full record name.
            value: The TXT record value.
            ttl: Time to live in seconds.

        Returns:
            DnsRecord with a generated ID.
        """
        self._record_counter += 1
        record_id = f"manual-{self._record_counter}"

        self._print("")
        self._print("=" * 60)
        self._print("MANUAL DNS RECORD CREATION REQUIRED")
        self._print("=" * 60)
        self._print("")
        self._print(f"Domain: {domain}")
        self._print(f"Record name: {name}")
        self._print("Record type: TXT")
        self._print(f"Record value: {value}")
        self._print(f"TTL: {ttl}")
        self._print("")
        self._print("Please create this TXT record in your DNS provider's control panel.")
        self._print("")

        # Wait for confirmation
        while True:
            response = self._prompt("Press Enter when the record has been created (or 'q' to quit): ")
            if response.lower() == "q":
                raise KeyboardInterrupt("User cancelled DNS record creation")
            if response == "" or response.lower() in ("y", "yes", "done"):
                break

        logger.info("User confirmed DNS record creation: %s", name)

        return DnsRecord(
            name=name,
            type="TXT",
            value=value,
            ttl=ttl,
            id=record_id,
        )

    def delete_record(self, domain: str, record: DnsRecord) -> None:
        """Display instructions for deleting a DNS TXT record.

        Optionally prompts for confirmation that the record has been deleted.

        Args:
            domain: The base domain.
            record: The record to delete.
        """
        self._print("")
        self._print("-" * 60)
        self._print("DNS RECORD CLEANUP")
        self._print("-" * 60)
        self._print("")
        self._print("Please delete this DNS record:")
        self._print(f"  Name: {record.name}")
        self._print(f"  Type: {record.type}")
        self._print(f"  Value: {record.value}")
        self._print("")

        if self._prompt_delete:
            self._prompt("Press Enter when the record has been deleted: ")
            logger.info("User confirmed DNS record deletion: %s", record.name)
        else:
            self._print("(You can delete this record at your convenience)")
            logger.info("Instructed user to delete DNS record: %s", record.name)

    def list_records(
        self,
        domain: str,
        record_type: str | None = None,
    ) -> list[DnsRecord]:
        """Not supported for manual provider.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Manual provider does not support listing records")

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return "ManualDnsProvider(interactive)"
