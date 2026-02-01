"""Account management example.

This example demonstrates creating and managing an ACME account,
including account creation, contact updates, and key rollover.

Requirements:
- A valid email address
- Network access to the ACME server

Note: Account management operations can be done without owning a domain.
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
NEW_EMAIL = "newadmin@example.com"  # For contact update example
STORAGE_PATH = Path("./acme_data")

# Use staging server for testing (no rate limits)
SERVER_URL = "https://acme-staging-v02.api.letsencrypt.org/directory"
# For production, use:
# SERVER_URL = "https://acme-v02.api.letsencrypt.org/directory"


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Demonstrate account management operations."""
    print("=" * 60)
    print("Account Management Example")
    print("=" * 60)
    print(f"Email:  {EMAIL}")
    print(f"Server: {SERVER_URL}")
    print()

    with AcmeClient(
        server_url=SERVER_URL,
        email=EMAIL,
        storage_path=STORAGE_PATH,
    ) as client:
        try:
            # Step 1: Create account
            print("[1/4] Creating account...")
            account = client.create_account()
            print(f"    Account URI: {account.uri}")
            print(f"    Status:      {account.status}")
            print(f"    Contact:     {EMAIL}")

            # Step 2: Verify account properties
            print("\n[2/4] Account properties:")
            print(f"    Is valid:  {account.is_valid}")
            print(f"    Has key:   {account.key is not None}")

            # Step 3: Update contact email (optional)
            print(f"\n[3/4] Updating contact email to {NEW_EMAIL}...")
            # Note: Uncomment the following line to actually update
            # client.update_account(email=NEW_EMAIL)
            print("    (Skipped - uncomment to enable)")

            # Step 4: Key rollover (optional)
            print("\n[4/4] Performing key rollover...")
            # Note: Uncomment the following line to actually roll over keys
            # client.key_rollover()
            print("    (Skipped - uncomment to enable)")

            # Success
            print("\n" + "=" * 60)
            print("SUCCESS!")
            print("=" * 60)
            print(f"Account created and ready for use")
            print(f"Account data stored in: {STORAGE_PATH}")

        except AcmeError as e:
            print(f"\nError: {e.message}")
            raise


if __name__ == "__main__":
    main()
