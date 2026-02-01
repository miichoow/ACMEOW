"""Account deactivation example.

This example demonstrates deactivating an ACME account.

Requirements:
- An existing ACME account

WARNING: Account deactivation is PERMANENT and CANNOT be undone!
A deactivated account can no longer be used to issue or manage certificates.
"""

from __future__ import annotations

from pathlib import Path

from acmeow import (
    AcmeClient,
    AcmeError,
)

# =============================================================================
# Configuration
# =============================================================================

EMAIL = "admin@example.com"
STORAGE_PATH = Path("./acme_data")

# Use staging server for testing (no rate limits)
SERVER_URL = "https://acme-staging-v02.api.letsencrypt.org/directory"
# For production, use:
# SERVER_URL = "https://acme-v02.api.letsencrypt.org/directory"

# Safety flag - set to True to actually deactivate
CONFIRM_DEACTIVATION = False


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Deactivate an ACME account."""
    print("=" * 60)
    print("Account Deactivation Example")
    print("=" * 60)
    print(f"Email:  {EMAIL}")
    print(f"Server: {SERVER_URL}")
    print()

    print("WARNING: Account deactivation is PERMANENT!")
    print("         This action CANNOT be undone.")
    print("         A deactivated account cannot issue new certificates.")
    print()

    if not CONFIRM_DEACTIVATION:
        print("Safety flag CONFIRM_DEACTIVATION is False.")
        print("Set it to True to actually deactivate the account.")
        print()

    with AcmeClient(
        server_url=SERVER_URL,
        email=EMAIL,
        storage_path=STORAGE_PATH,
    ) as client:
        try:
            # Step 1: Load or create account
            print("[1/3] Loading account...")
            account = client.create_account()
            print(f"    Account URI: {account.uri}")
            print(f"    Status:      {account.status}")

            # Step 2: Verify account is active
            print("\n[2/3] Verifying account status...")
            if not account.is_valid:
                print(f"    Account is not active (status: {account.status})")
                print("    Cannot deactivate an inactive account.")
                return
            print("    Account is active and can be deactivated.")

            # Step 3: Deactivate account
            print("\n[3/3] Deactivating account...")
            if CONFIRM_DEACTIVATION:
                # DANGER: This will permanently deactivate the account!
                client.deactivate_account()
                print("    Account DEACTIVATED!")
                print()
                print("    The account can no longer be used to:")
                print("    - Issue new certificates")
                print("    - Renew existing certificates")
                print("    - Manage authorizations")
            else:
                print("    (Skipped - set CONFIRM_DEACTIVATION = True to enable)")

            # Status
            print("\n" + "=" * 60)
            if CONFIRM_DEACTIVATION:
                print("ACCOUNT DEACTIVATED")
            else:
                print("DEACTIVATION NOT PERFORMED")
            print("=" * 60)

        except AcmeError as e:
            print(f"\nError: {e.message}")
            raise


if __name__ == "__main__":
    main()
