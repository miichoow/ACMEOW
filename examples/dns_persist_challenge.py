"""DNS-PERSIST-01 challenge example.

This example demonstrates obtaining a certificate using the persistent DNS
validation method from draft-ietf-acme-dns-persist.

Unlike DNS-01, the TXT record:
- lives at _validation-persist.<domain>, not _acme-challenge.<domain>
- contains no token or key authorization, but binds an Issuer Domain Name to
  your ACME account URI using the issue-value syntax of RFC 8659
- stays published after validation, authorizing future issuance

Requirements:
- A CA that offers dns-persist-01 challenges. Let's Encrypt staging does
  (verified July 2026); production is unverified.
- Access to your DNS provider's API

Warning: draft-ietf-acme-dns-persist is an Internet-Draft, not an RFC. The
record format may change before publication.

Note: an order covering both example.com and *.example.com produces two
authorizations that share one validation record. ACMEOW merges them into a
single record with policy=wildcard, so this callback fires once, not twice.
"""

from __future__ import annotations

from pathlib import Path

from acmeow import (
    AcmeClient,
    AcmeError,
    CallbackDnsPersistHandler,
    Identifier,
    KeyType,
    parse_record_value,
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
# DNS Record Callback
# =============================================================================


def upsert_txt_record(domain: str, record_name: str, record_value: str) -> None:
    """Publish the persistent validation TXT record.

    Replace this implementation with your DNS provider's API.

    Treat this as an upsert rather than a create: because the record is
    long-lived, one may already exist from a previous issuance. Creating a
    second record at the same name would leave duplicates behind.

    Args:
        domain: The domain being validated (e.g., "example.com").
        record_name: Full record name (e.g., "_validation-persist.example.com").
        record_value: The issue-value formatted TXT record value, e.g.
            "ca.example; accounturi=https://ca.example/acct/1".
    """
    print(f"    Publish TXT: {record_name} = {record_value}")

    # Show what the CA will read back out of the record.
    parsed = parse_record_value(record_value)
    print(f"      issuer:     {parsed.issuer_domain_name}")
    print(f"      accounturi: {parsed.accounturi}")
    if parsed.allows_wildcard:
        print("      wildcard:   authorized")

    # TODO: Replace with your DNS provider API call
    # Example for Cloudflare:
    #   cf.dns.records.create(zone_id="...", type="TXT",
    #                         name=record_name, content=record_value)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Obtain a certificate using the DNS-PERSIST-01 challenge."""
    print("=" * 60)
    print("DNS-PERSIST-01 Challenge Example")
    print("=" * 60)
    print(f"Domain: {DOMAIN}")
    print(f"Email:  {EMAIL}")
    print(f"Server: {SERVER_URL}")
    print()

    # persist=True (the default) leaves the record published so it can
    # authorize future issuance without another DNS change.
    handler = CallbackDnsPersistHandler(
        create_record=upsert_txt_record,
        propagation_delay=PROPAGATION_DELAY,
        persist=True,
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
                challenge = auth.get_dns_persist_challenge()
                if challenge is None:
                    print(f"    {auth.domain}: no dns-persist-01 challenge offered")
                    print("    This CA does not support the draft challenge type.")
                    return

                print(f"    Domain:  {auth.domain}")
                print(f"    Record:  _validation-persist.{auth.domain}")
                print(f"    Issuers: {', '.join(challenge.issuer_domain_names)}")

            # Step 4: Complete challenges
            print(f"\n[4/5] Completing challenges (waiting {PROPAGATION_DELAY}s)...")
            client.complete_dns_persist_challenges(handler)
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
            print()
            print("The validation record is still published. Keep it in place")
            print("to renew without touching DNS again.")

        except AcmeError as e:
            print(f"\nError: {e.message}")
            raise


if __name__ == "__main__":
    main()
