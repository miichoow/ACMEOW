"""HTTP-01 challenge example.

This example demonstrates obtaining a certificate using HTTP-01 challenges
with the FileHttpHandler.

Requirements:
- A web server (nginx, Apache, etc.) serving the webroot directory
- Port 80 accessible from the internet
- DNS A/AAAA records pointing to your server

Note: HTTP-01 does not support wildcard certificates.
"""

from __future__ import annotations

from pathlib import Path

from acmeow import (
    AcmeClient,
    AcmeError,
    ChallengeType,
    FileHttpHandler,
    Identifier,
    KeyType,
)

# =============================================================================
# Configuration
# =============================================================================

EMAIL = "admin@example.com"
DOMAIN = "example.com"
STORAGE_PATH = Path("./acme_data")

# Web server document root
# Challenge files will be created at: {WEBROOT}/.well-known/acme-challenge/
WEBROOT = Path("/var/www/html")

# Use staging server for testing (no rate limits)
SERVER_URL = "https://acme-staging-v02.api.letsencrypt.org/directory"
# For production, use:
# SERVER_URL = "https://acme-v02.api.letsencrypt.org/directory"


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Obtain a certificate using HTTP-01 challenge."""
    print("=" * 60)
    print("HTTP-01 Challenge Example")
    print("=" * 60)
    print(f"Domain:  {DOMAIN}")
    print(f"Email:   {EMAIL}")
    print(f"Server:  {SERVER_URL}")
    print(f"Webroot: {WEBROOT}")
    print()

    # Ensure webroot exists
    if not WEBROOT.exists():
        print(f"Warning: Creating webroot directory {WEBROOT}")
        WEBROOT.mkdir(parents=True, exist_ok=True)

    # Create HTTP handler
    handler = FileHttpHandler(webroot=WEBROOT)

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
                challenge = auth.get_http_challenge()
                if challenge:
                    print(f"    Domain: {auth.domain}")
                    print(f"    URL: http://{auth.domain}/.well-known/acme-challenge/{challenge.token}")

            # Step 4: Complete challenges
            print("\n[4/5] Completing HTTP-01 challenges...")
            print(f"    Files written to: {handler.challenge_dir}")
            client.complete_challenges(handler, ChallengeType.HTTP)
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
