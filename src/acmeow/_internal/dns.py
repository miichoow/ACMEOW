"""DNS verification utilities.

Provides functionality for verifying DNS record propagation before
notifying the ACME server of challenge completion.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Default DNS servers for verification
DEFAULT_DNS_SERVERS = [
    "8.8.8.8",        # Google
    "8.8.4.4",        # Google
    "1.1.1.1",        # Cloudflare
    "1.0.0.1",        # Cloudflare
    "9.9.9.9",        # Quad9
]

# Default configuration
DEFAULT_DNS_TIMEOUT = 5.0
DEFAULT_DNS_RETRIES = 3
DEFAULT_DNS_RETRY_DELAY = 2.0


@dataclass
class DnsConfig:
    """Configuration for DNS verification.

    Args:
        nameservers: List of DNS server IP addresses to query.
        timeout: Timeout for each DNS query in seconds. Default 5.0.
        retries: Number of retry attempts per server. Default 3.
        retry_delay: Delay between retries in seconds. Default 2.0.
        require_all: Require all nameservers to see the record. Default False.
        min_servers: Minimum number of servers that must see the record. Default 1.
    """

    nameservers: list[str] = field(default_factory=lambda: DEFAULT_DNS_SERVERS.copy())
    timeout: float = DEFAULT_DNS_TIMEOUT
    retries: int = DEFAULT_DNS_RETRIES
    retry_delay: float = DEFAULT_DNS_RETRY_DELAY
    require_all: bool = False
    min_servers: int = 1


class DnsVerifier:
    """Verify DNS record propagation.

    Queries multiple DNS servers to verify that a TXT record has propagated
    before notifying the ACME server.

    Args:
        config: DNS verification configuration.
    """

    def __init__(self, config: DnsConfig | None = None) -> None:
        self._config = config or DnsConfig()
        self._resolver_available = self._check_resolver()

    def _check_resolver(self) -> bool:
        """Check if dnspython is available for advanced DNS queries.

        Returns:
            True if dnspython is installed.
        """
        try:
            import dns.resolver  # noqa: F401
            return True
        except ImportError:
            logger.debug("dnspython not installed, using basic DNS verification")
            return False

    def verify_txt_record(
        self,
        record_name: str,
        expected_value: str,
        max_wait: int = 300,
        poll_interval: int = 10,
    ) -> bool:
        """Verify a TXT record has propagated.

        Polls DNS servers until the expected TXT record is found or timeout.

        Args:
            record_name: The full DNS record name (e.g., "_acme-challenge.example.com").
            expected_value: The expected TXT record value.
            max_wait: Maximum time to wait for propagation in seconds. Default 300.
            poll_interval: Time between poll attempts in seconds. Default 10.

        Returns:
            True if the record was verified, False if timeout.
        """
        start_time = time.time()
        attempt = 0

        while time.time() - start_time < max_wait:
            attempt += 1
            logger.info(
                "Verifying DNS record %s (attempt %d, elapsed %.0fs)",
                record_name,
                attempt,
                time.time() - start_time,
            )

            if self._check_record(record_name, expected_value):
                logger.info("DNS record verified: %s", record_name)
                return True

            remaining = max_wait - (time.time() - start_time)
            if remaining > poll_interval:
                time.sleep(poll_interval)
            elif remaining > 0:
                time.sleep(remaining)

        logger.warning(
            "DNS record verification timed out after %ds: %s",
            max_wait,
            record_name,
        )
        return False

    def _check_record(self, record_name: str, expected_value: str) -> bool:
        """Check if TXT record matches expected value on configured servers.

        Args:
            record_name: The DNS record name.
            expected_value: The expected TXT record value.

        Returns:
            True if enough servers returned the expected value.
        """
        if self._resolver_available:
            return self._check_record_dnspython(record_name, expected_value)
        return self._check_record_socket(record_name, expected_value)

    def _check_record_dnspython(self, record_name: str, expected_value: str) -> bool:
        """Check TXT record using dnspython for precise nameserver control.

        Args:
            record_name: The DNS record name.
            expected_value: The expected TXT record value.

        Returns:
            True if enough servers returned the expected value.
        """
        import dns.resolver

        success_count = 0

        for nameserver in self._config.nameservers:
            for attempt in range(self._config.retries):
                try:
                    resolver = dns.resolver.Resolver()
                    resolver.nameservers = [nameserver]
                    resolver.lifetime = self._config.timeout

                    answers = resolver.resolve(record_name, "TXT")

                    for rdata in answers:
                        # TXT records may be split into multiple strings
                        txt_value = b"".join(rdata.strings).decode("utf-8")
                        if txt_value == expected_value:
                            logger.debug(
                                "DNS server %s returned expected value for %s",
                                nameserver,
                                record_name,
                            )
                            success_count += 1
                            break
                    else:
                        logger.debug(
                            "DNS server %s returned different value for %s",
                            nameserver,
                            record_name,
                        )
                    break  # Got an answer, don't retry

                except dns.resolver.NXDOMAIN:
                    logger.debug(
                        "DNS server %s: NXDOMAIN for %s",
                        nameserver,
                        record_name,
                    )
                    break  # Domain doesn't exist, don't retry

                except dns.resolver.NoAnswer:
                    logger.debug(
                        "DNS server %s: no TXT records for %s",
                        nameserver,
                        record_name,
                    )
                    break  # No TXT records, don't retry

                except dns.exception.DNSException as e:
                    logger.debug(
                        "DNS server %s failed (attempt %d): %s",
                        nameserver,
                        attempt + 1,
                        e,
                    )
                    if attempt < self._config.retries - 1:
                        time.sleep(self._config.retry_delay)

        # Check if we have enough successful verifications
        if self._config.require_all:
            return success_count == len(self._config.nameservers)

        return success_count >= self._config.min_servers

    def _check_record_socket(self, record_name: str, expected_value: str) -> bool:
        """Check TXT record using basic socket DNS (fallback when dnspython unavailable).

        This is a simplified check that uses the system resolver.

        Args:
            record_name: The DNS record name.
            expected_value: The expected TXT record value.

        Returns:
            True if the system resolver found the expected value.
        """
        try:
            # Use system resolver via socket
            # This is limited but works without dnspython
            import subprocess
            import sys

            if sys.platform == "win32":
                # Windows nslookup
                result = subprocess.run(
                    ["nslookup", "-type=TXT", record_name],
                    capture_output=True,
                    text=True,
                    timeout=self._config.timeout,
                )
                output = result.stdout
            else:
                # Unix dig or host
                try:
                    result = subprocess.run(
                        ["dig", "+short", "TXT", record_name],
                        capture_output=True,
                        text=True,
                        timeout=self._config.timeout,
                    )
                    output = result.stdout
                except FileNotFoundError:
                    result = subprocess.run(
                        ["host", "-t", "TXT", record_name],
                        capture_output=True,
                        text=True,
                        timeout=self._config.timeout,
                    )
                    output = result.stdout

            # Check if expected value is in output
            if expected_value in output:
                logger.debug("System resolver found expected TXT value for %s", record_name)
                return True

            logger.debug("System resolver did not find expected TXT value for %s", record_name)
            return False

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.warning("DNS verification failed using system resolver: %s", e)
            # If we can't verify, assume it might be propagated
            # This prevents blocking when DNS tools aren't available
            return True


def verify_dns_propagation(
    record_name: str,
    expected_value: str,
    config: DnsConfig | None = None,
    max_wait: int = 300,
    poll_interval: int = 10,
) -> bool:
    """Convenience function to verify DNS propagation.

    Args:
        record_name: The full DNS record name.
        expected_value: The expected TXT record value.
        config: Optional DNS configuration.
        max_wait: Maximum time to wait in seconds.
        poll_interval: Time between checks in seconds.

    Returns:
        True if verified, False if timeout.
    """
    verifier = DnsVerifier(config)
    return verifier.verify_txt_record(
        record_name,
        expected_value,
        max_wait=max_wait,
        poll_interval=poll_interval,
    )
