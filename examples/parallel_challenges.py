#!/usr/bin/env python3
"""Example: Parallel challenge completion for multi-domain certificates.

This example demonstrates how to use parallel challenge completion
to speed up certificate issuance for certificates with many domains.

When issuing certificates with many Subject Alternative Names (SANs),
challenges can be set up and cleaned up in parallel to reduce total time.

Usage:
    python parallel_challenges.py
"""

import time
from typing import Any



class TimingDnsProvider:
    """A DNS provider wrapper that tracks timing."""

    def __init__(self, provider: Any) -> None:
        self._provider = provider
        self.setup_times: list[float] = []
        self.cleanup_times: list[float] = []

    @property
    def propagation_delay(self) -> int:
        return self._provider.propagation_delay

    def create_record(
        self,
        domain: str,
        name: str,
        value: str,
        ttl: int = 300,
    ) -> Any:
        start = time.time()
        # Simulate API delay
        time.sleep(0.5)
        result = self._provider.create_record(domain, name, value, ttl)
        elapsed = time.time() - start
        self.setup_times.append(elapsed)
        print(f"  Setup {name}: {elapsed:.2f}s")
        return result

    def delete_record(self, domain: str, record: Any) -> None:
        start = time.time()
        # Simulate API delay
        time.sleep(0.3)
        self._provider.delete_record(domain, record)
        elapsed = time.time() - start
        self.cleanup_times.append(elapsed)
        print(f"  Cleanup {record.name}: {elapsed:.2f}s")

    def get_zone_for_domain(self, domain: str) -> str:
        return self._provider.get_zone_for_domain(domain)


def demonstrate_timing_difference() -> None:
    """Demonstrate the timing difference between sequential and parallel."""
    print("=== Parallel vs Sequential Challenge Completion ===\n")

    # Simulated domains
    domains = [
        "example.com",
        "www.example.com",
        "api.example.com",
        "cdn.example.com",
        "mail.example.com",
    ]

    print(f"Domains: {len(domains)}")
    print()

    # Simulate timing for sequential
    setup_time_per_domain = 0.5  # seconds
    cleanup_time_per_domain = 0.3  # seconds

    sequential_setup = len(domains) * setup_time_per_domain
    sequential_cleanup = len(domains) * cleanup_time_per_domain
    sequential_total = sequential_setup + sequential_cleanup

    print("Sequential timing (simulated):")
    print(f"  Setup: {len(domains)} x {setup_time_per_domain}s = {sequential_setup:.1f}s")
    print(f"  Cleanup: {len(domains)} x {cleanup_time_per_domain}s = {sequential_cleanup:.1f}s")
    print(f"  Total: {sequential_total:.1f}s")
    print()

    # Simulate timing for parallel (4 workers)
    workers = 4
    parallel_setup = (len(domains) / workers) * setup_time_per_domain
    parallel_cleanup = (len(domains) / workers) * cleanup_time_per_domain
    parallel_total = parallel_setup + parallel_cleanup

    print(f"Parallel timing ({workers} workers, simulated):")
    print(f"  Setup: ~{parallel_setup:.1f}s")
    print(f"  Cleanup: ~{parallel_cleanup:.1f}s")
    print(f"  Total: ~{parallel_total:.1f}s")
    print()

    speedup = sequential_total / parallel_total
    print(f"Speedup: {speedup:.1f}x faster")
    print()


def show_usage_example() -> None:
    """Show the code for using parallel challenges."""
    print("=== Usage Example ===\n")

    print("Enable parallel challenge completion by passing parallel=True:")
    print()
    print('''from pathlib import Path
from acmeow import (
    AcmeClient,
    ChallengeType,
    DnsProviderHandler,
    Identifier,
    KeyType,
    get_dns_provider,
)

# Get DNS provider
provider = get_dns_provider("cloudflare", api_token="your-token")
handler = DnsProviderHandler(provider)

# Create client
client = AcmeClient(
    server_url="https://acme-v02.api.letsencrypt.org/directory",
    email="admin@example.com",
    storage_path=Path("./acme_data"),
)

client.create_account()

# Create order with multiple domains
order = client.create_order([
    Identifier.dns("example.com"),
    Identifier.dns("www.example.com"),
    Identifier.dns("api.example.com"),
    Identifier.dns("cdn.example.com"),
])

# Complete challenges IN PARALLEL
client.complete_challenges(
    handler,
    challenge_type=ChallengeType.DNS,
    parallel=True,           # Enable parallel execution
    max_workers=4,           # Optional: limit workers (default: auto)
)

# Finalize and get certificate
client.finalize_order(KeyType.EC256)
cert_pem, key_pem = client.get_certificate()''')
    print()


def show_when_to_use() -> None:
    """Show when parallel execution is beneficial."""
    print("=== When to Use Parallel Challenges ===\n")

    print("Parallel challenge completion is beneficial when:")
    print()
    print("1. Issuing certificates with MANY domains (5+ SANs)")
    print("   - SAN certificates for multiple subdomains")
    print("   - Wildcard + specific domain combinations")
    print()
    print("2. DNS provider API has significant latency")
    print("   - API calls take 500ms+ each")
    print("   - Rate limits allow concurrent requests")
    print()
    print("3. NOT recommended when:")
    print("   - Single domain certificates")
    print("   - DNS provider has strict rate limits")
    print("   - Sequential order matters for your setup")
    print()

    print("Note: Server notification and polling remain SEQUENTIAL")
    print("(ACME servers expect ordered challenge responses)")
    print()


def show_max_workers_guidance() -> None:
    """Show guidance for choosing max_workers."""
    print("=== Choosing max_workers ===\n")

    print("The max_workers parameter controls parallelism:")
    print()
    print("  max_workers=None (default)")
    print("    Python automatically chooses based on CPUs")
    print("    Usually min(32, cpu_count + 4)")
    print()
    print("  max_workers=4")
    print("    Good default for most DNS providers")
    print("    Balances speed vs API rate limits")
    print()
    print("  max_workers=1")
    print("    Effectively sequential (same as parallel=False)")
    print()
    print("  max_workers=10+")
    print("    Only if DNS provider allows high request rates")
    print("    May trigger rate limiting")
    print()

    print("Provider-specific recommendations:")
    print("  Cloudflare: 4-8 workers (5 req/sec by default)")
    print("  Route53: 2-4 workers (5 req/sec per API)")
    print("  DigitalOcean: 4-6 workers (varies)")
    print()


def main() -> None:
    """Run the parallel challenges examples."""
    demonstrate_timing_difference()
    show_usage_example()
    show_when_to_use()
    show_max_workers_guidance()

    print("=== Summary ===")
    print("Parallel challenge completion can significantly speed up")
    print("multi-domain certificate issuance by setting up and cleaning")
    print("up DNS records concurrently. Use parallel=True when issuing")
    print("certificates with many SANs.")


if __name__ == "__main__":
    main()
