"""DNS-01 challenge example.

This example demonstrates obtaining a certificate using DNS-01 challenges
with the CallbackDnsHandler.

Requirements:
- Access to your DNS provider's API
- Ability to create/delete TXT records

Note: DNS-01 is the only challenge type that supports wildcard certificates.
"""

from __future__ import annotations

from pathlib import Path

from acmeow import (
    AcmeClient,
    AcmeError,
    CallbackDnsHandler,
    ChallengeType,
    Identifier,
    KeyType,
)

# =============================================================================
# Configuration
# =============================================================================

EMAIL = "admin@example.com"
DOMAIN = "example.com"
STORAGE_PATH = Path("./acme_data")

# Use staging server for testing (no rate limits)
SERVER_URL = "https://acme-staging-v02.api.letsencrypt.org/directory"
# For production, use:
# SERVER_URL = "https://acme-v02.api.letsencrypt.org/directory"

# DNS propagation delay in seconds
PROPAGATION_DELAY = 60


# =============================================================================
# DNS Record Callbacks
# =============================================================================


def create_txt_record(domain: str, record_name: str, record_value: str) -> None:
    """Create a DNS TXT record.

    Replace this implementation with your DNS provider's API.

    Args:
        domain: The domain being validated (e.g., "example.com").
        record_name: Full record name (e.g., "_acme-challenge.example.com").
        record_value: The TXT record value (base64url SHA-256 hash).
    """
    print(f"    Create TXT: {record_name} = {record_value}")

    # TODO: Replace with your DNS provider API call
    # Example for Cloudflare:
    #   cf.dns.records.create(zone_id="...", type="TXT",
    #                         name=record_name, content=record_value)


def delete_txt_record(domain: str, record_name: str) -> None:
    """Delete a DNS TXT record.

    Replace this implementation with your DNS provider's API.

    Args:
        domain: The domain that was validated.
        record_name: Full record name to delete.
    """
    print(f"    Delete TXT: {record_name}")

    # TODO: Replace with your DNS provider API call


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Obtain a certificate using DNS-01 challenge."""
    print("=" * 60)
    print("DNS-01 Challenge Example")
    print("=" * 60)
    print(f"Domain: {DOMAIN}")
    print(f"Email:  {EMAIL}")
    print(f"Server: {SERVER_URL}")
    print()

    # Create DNS handler with callbacks
    handler = CallbackDnsHandler(
        create_record=create_txt_record,
        delete_record=delete_txt_record,
        propagation_delay=PROPAGATION_DELAY,
    )

    with AcmeClient(
        server_url=SERVER_URL,
        email=EMAIL,
        storage_path=STORAGE_PATH,
    ) as client:
        try:
            # Step 1: Create account
            print("[1/5] Creating account...")
            account = client.create_account()
            print(f"    Account: {account.uri}")

            # Step 2: Create order
            print(f"\n[2/5] Creating order for {DOMAIN}...")
            order = client.create_order([Identifier.dns(DOMAIN)])
            print(f"    Order: {order.url}")

            # Step 3: Show challenge info
            print("\n[3/5] Challenge information:")
            for auth in order.authorizations:
                challenge = auth.get_dns_challenge()
                if challenge:
                    print(f"    Domain: {auth.domain}")
                    print(f"    Record: _acme-challenge.{auth.domain}")

            # Step 4: Complete challenges
            print(f"\n[4/5] Completing DNS-01 challenges (waiting {PROPAGATION_DELAY}s)...")
            client.complete_challenges(handler, ChallengeType.DNS)
            print("    Challenges completed!")

            # Step 5: Finalize and get certificate
            print("\n[5/5] Finalizing order...")
            client.finalize_order(KeyType.EC256)
            cert_pem, key_pem = client.get_certificate()

            # Success
            print("\n" + "=" * 60)
            print("SUCCESS!")
            print("=" * 60)
            print(f"Certificate: {STORAGE_PATH}/certificates/{DOMAIN}.crt")
            print(f"Private Key: {STORAGE_PATH}/certificates/{DOMAIN}.key")

        except AcmeError as e:
            print(f"\nError: {e.message}")
            raise


if __name__ == "__main__":
    main()
