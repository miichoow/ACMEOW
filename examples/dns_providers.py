#!/usr/bin/env python3
"""Example: Using DNS providers for automated DNS-01 challenges.

This example demonstrates how to use the DNS provider system for
automated DNS-01 challenge completion. The system supports multiple
DNS providers and allows custom provider registration.

Available providers:
- manual: Interactive provider for manual DNS management
- cloudflare: Cloudflare DNS API
- route53: AWS Route53 (requires boto3)
- digitalocean: DigitalOcean DNS API

Usage:
    python dns_providers.py
"""


from acmeow import (
    DnsProvider,
    DnsRecord,
    get_dns_provider,
    list_dns_providers,
    register_dns_provider,
)


def show_available_providers() -> None:
    """Show all available DNS providers."""
    print("=== Available DNS Providers ===\n")

    providers = list_dns_providers()
    for name in providers:
        print(f"  - {name}")

    print()


def example_manual_provider() -> None:
    """Example using the manual DNS provider."""
    print("=== Manual DNS Provider ===\n")
    print("The manual provider prompts you to create DNS records manually.")
    print("Useful for testing or when no API access is available.\n")

    # Get the manual provider
    provider = get_dns_provider("manual")
    print(f"Provider: {provider}")
    print(f"Propagation delay: {provider.propagation_delay} seconds")
    print()


def example_cloudflare_provider() -> None:
    """Example using Cloudflare DNS provider."""
    print("=== Cloudflare DNS Provider ===\n")

    # To use Cloudflare, you need an API token with DNS edit permissions
    # Get one from: https://dash.cloudflare.com/profile/api-tokens

    print("To use Cloudflare:")
    print("1. Create an API token at https://dash.cloudflare.com/profile/api-tokens")
    print("2. Grant 'Zone.DNS' edit permission")
    print("3. Use the token:")
    print()
    print('   from acmeow import get_dns_provider, DnsProviderHandler')
    print('   ')
    print('   provider = get_dns_provider("cloudflare", api_token="your-token")')
    print('   handler = DnsProviderHandler(provider)')
    print('   ')
    print('   # Or specify a zone ID directly:')
    print('   provider = get_dns_provider(')
    print('       "cloudflare",')
    print('       api_token="your-token",')
    print('       zone_id="your-zone-id",')
    print('   )')
    print()


def example_route53_provider() -> None:
    """Example using AWS Route53 DNS provider."""
    print("=== AWS Route53 DNS Provider ===\n")

    print("To use Route53:")
    print("1. Install boto3: pip install boto3")
    print("2. Configure AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
    print("   Or use IAM roles if running on AWS")
    print("3. Use the provider:")
    print()
    print('   from acmeow import get_dns_provider, DnsProviderHandler')
    print('   ')
    print('   # Using default credentials:')
    print('   provider = get_dns_provider("route53")')
    print('   ')
    print('   # Or specify credentials directly:')
    print('   provider = get_dns_provider(')
    print('       "route53",')
    print('       aws_access_key_id="AKIAXXXXXXXX",')
    print('       aws_secret_access_key="xxxxxxxxxxxxx",')
    print('       hosted_zone_id="Z1234567890ABC",  # Optional')
    print('   )')
    print()


def example_digitalocean_provider() -> None:
    """Example using DigitalOcean DNS provider."""
    print("=== DigitalOcean DNS Provider ===\n")

    print("To use DigitalOcean:")
    print("1. Create an API token at https://cloud.digitalocean.com/account/api/tokens")
    print("2. Grant read+write access")
    print("3. Use the provider:")
    print()
    print('   from acmeow import get_dns_provider, DnsProviderHandler')
    print('   ')
    print('   provider = get_dns_provider("digitalocean", api_token="your-token")')
    print('   handler = DnsProviderHandler(provider)')
    print()


def example_custom_provider() -> None:
    """Example of creating and registering a custom DNS provider."""
    print("=== Custom DNS Provider ===\n")

    # Define a custom DNS provider
    class MyDnsProvider(DnsProvider):
        """Example custom DNS provider."""

        propagation_delay = 45  # Seconds to wait after record creation

        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.records: dict[str, DnsRecord] = {}

        def create_record(
            self,
            domain: str,
            name: str,
            value: str,
            ttl: int = 300,
        ) -> DnsRecord:
            """Create a TXT record."""
            record_id = f"rec-{len(self.records) + 1}"
            record = DnsRecord(
                name=name,
                type="TXT",
                value=value,
                ttl=ttl,
                id=record_id,
            )
            self.records[name] = record
            print(f"  Created record: {name} = {value}")
            return record

        def delete_record(self, domain: str, record: DnsRecord) -> None:
            """Delete a TXT record."""
            if record.name in self.records:
                del self.records[record.name]
                print(f"  Deleted record: {record.name}")

        def list_records(
            self,
            domain: str,
            record_type: str | None = None,
        ) -> list[DnsRecord]:
            """List all records."""
            records = list(self.records.values())
            if record_type:
                records = [r for r in records if r.type == record_type]
            return records

    # Register the custom provider
    register_dns_provider("myprovider", MyDnsProvider)

    print("Custom provider registered as 'myprovider'")
    print()
    print("Usage:")
    print('   provider = get_dns_provider("myprovider", api_key="xxx")')
    print('   handler = DnsProviderHandler(provider)')
    print()

    # Demonstrate usage
    provider = get_dns_provider("myprovider", api_key="demo-key")
    print(f"Created provider: {provider}")

    # Create a test record
    record = provider.create_record(
        domain="example.com",
        name="_acme-challenge.example.com",
        value="test-value-123",
    )

    # List records
    records = provider.list_records("example.com")
    print(f"Records: {len(records)}")

    # Delete record
    provider.delete_record("example.com", record)
    print()


def example_full_workflow() -> None:
    """Example of a complete workflow with DNS provider."""
    print("=== Full Workflow Example ===\n")

    print("Complete code for certificate issuance with DNS provider:")
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

# Get DNS provider (using Cloudflare as example)
provider = get_dns_provider("cloudflare", api_token="your-api-token")
handler = DnsProviderHandler(provider)

# Create ACME client
client = AcmeClient(
    server_url="https://acme-v02.api.letsencrypt.org/directory",
    email="admin@example.com",
    storage_path=Path("./acme_data"),
)

# Issue certificate
client.create_account()
order = client.create_order([
    Identifier.dns("example.com"),
    Identifier.dns("www.example.com"),
])

# Complete DNS-01 challenges
# Handler will automatically create/delete TXT records
client.complete_challenges(handler, ChallengeType.DNS)

# Finalize and get certificate
client.finalize_order(KeyType.EC256)
cert_pem, key_pem = client.get_certificate()

# Save certificate
Path("cert.pem").write_text(cert_pem)
Path("key.pem").write_text(key_pem)
print("Certificate saved!")''')
    print()


def main() -> None:
    """Run the DNS provider examples."""
    show_available_providers()
    example_manual_provider()
    example_cloudflare_provider()
    example_route53_provider()
    example_digitalocean_provider()
    example_custom_provider()
    example_full_workflow()

    print("=== Summary ===")
    print("DNS providers automate DNS-01 challenge completion by:")
    print("1. Creating _acme-challenge TXT records via API")
    print("2. Waiting for DNS propagation")
    print("3. Automatically cleaning up records after validation")


if __name__ == "__main__":
    main()
