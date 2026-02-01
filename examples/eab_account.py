"""External Account Binding (EAB) example.

This example demonstrates creating an ACME account with External Account
Binding, which is required by some Certificate Authorities (CAs) to link
the ACME account to an existing account in their system.

Requirements:
- EAB credentials from your CA (Key ID and HMAC key)
- A valid email address

CAs that require EAB include:
- ZeroSSL (https://zerossl.com)
- Sectigo (https://sectigo.com)
- Google Trust Services
- Some enterprise CAs

To get EAB credentials:
- ZeroSSL: Create an account at zerossl.com, go to Developer section
- Sectigo: Contact your account manager
- Google: Use the Google Cloud Certificate Manager API
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

# EAB credentials from your CA
# These are example values - replace with your actual credentials
EAB_KEY_ID = "your-eab-key-id"
EAB_HMAC_KEY = "your-base64url-encoded-hmac-key"

# ZeroSSL ACME server (requires EAB)
SERVER_URL = "https://acme.zerossl.com/v2/DV90"

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
    """Obtain a certificate using EAB authentication."""
    print("=" * 60)
    print("External Account Binding (EAB) Example")
    print("=" * 60)
    print(f"Domain: {DOMAIN}")
    print(f"Email:  {EMAIL}")
    print(f"Server: {SERVER_URL}")
    print(f"EAB ID: {EAB_KEY_ID}")
    print()

    # Validate EAB credentials
    if EAB_KEY_ID == "your-eab-key-id" or EAB_HMAC_KEY == "your-base64url-encoded-hmac-key":
        print("ERROR: Please update EAB_KEY_ID and EAB_HMAC_KEY with your")
        print("       actual credentials from your CA.")
        print()
        print("To get EAB credentials:")
        print("  - ZeroSSL: Sign up at https://zerossl.com, go to Developer section")
        print("  - Sectigo: Contact your account manager")
        print("  - Google:  Use Google Cloud Certificate Manager API")
        return

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
            # Step 1: Set EAB credentials
            print("[1/6] Setting EAB credentials...")
            client.set_external_account_binding(
                kid=EAB_KEY_ID,
                hmac_key=EAB_HMAC_KEY,
            )
            print("    EAB credentials configured")

            # Step 2: Create account with EAB
            print("\n[2/6] Creating account with EAB...")
            account = client.create_account()
            print(f"    Account: {account.uri}")
            print("    EAB linked successfully!")

            # Step 3: Create order
            print(f"\n[3/6] Creating order for {DOMAIN}...")
            order = client.create_order([Identifier.dns(DOMAIN)])
            print(f"    Order: {order.url}")

            # Step 4: Show challenge info
            print("\n[4/6] Challenge information:")
            for auth in order.authorizations:
                challenge = auth.get_dns_challenge()
                if challenge:
                    print(f"    Domain: {auth.domain}")
                    print(f"    Record: _acme-challenge.{auth.domain}")

            # Step 5: Complete challenges
            print(f"\n[5/6] Completing DNS-01 challenges (waiting {PROPAGATION_DELAY}s)...")
            client.complete_challenges(handler, ChallengeType.DNS)
            print("    Challenges completed!")

            # Step 6: Finalize and get certificate
            print("\n[6/6] Finalizing order...")
            client.finalize_order(KeyType.EC256)
            cert_pem, key_pem = client.get_certificate()

            # Success
            print("\n" + "=" * 60)
            print("SUCCESS!")
            print("=" * 60)
            print(f"Certificate: {STORAGE_PATH}/certificates/{DOMAIN}.crt")
            print(f"Private Key: {STORAGE_PATH}/certificates/{DOMAIN}.key")
            print()
            print("The certificate was issued using EAB authentication.")

        except AcmeError as e:
            print(f"\nError: {e.message}")
            raise


if __name__ == "__main__":
    main()
