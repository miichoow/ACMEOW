"""Certificate revocation example.

This example demonstrates revoking a certificate using the ACME protocol.

Requirements:
- An existing ACME account
- A certificate to revoke (PEM format)

Note: Certificate revocation is permanent and cannot be undone.
"""

from __future__ import annotations

from pathlib import Path

from acmeow import (
    AcmeClient,
    AcmeError,
    RevocationReason,
)

# =============================================================================
# Configuration
# =============================================================================

EMAIL = "admin@example.com"
STORAGE_PATH = Path("./acme_data")

# Path to the certificate to revoke
CERTIFICATE_PATH = Path("./acme_data/certificates/example.com.crt")

# Revocation reason (optional)
# Available reasons:
#   RevocationReason.UNSPECIFIED          - No specific reason
#   RevocationReason.KEY_COMPROMISE       - Private key compromised
#   RevocationReason.CA_COMPROMISE        - CA compromised
#   RevocationReason.AFFILIATION_CHANGED  - Affiliation changed
#   RevocationReason.SUPERSEDED           - Certificate superseded
#   RevocationReason.CESSATION_OF_OPERATION - Operations ceased
#   RevocationReason.CERTIFICATE_HOLD     - Temporarily on hold
#   RevocationReason.PRIVILEGE_WITHDRAWN  - Privileges withdrawn
REVOCATION_REASON = RevocationReason.SUPERSEDED

# Use staging server for testing (no rate limits)
SERVER_URL = "https://acme-staging-v02.api.letsencrypt.org/directory"
# For production, use:
# SERVER_URL = "https://acme-v02.api.letsencrypt.org/directory"


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Revoke a certificate."""
    print("=" * 60)
    print("Certificate Revocation Example")
    print("=" * 60)
    print(f"Email:       {EMAIL}")
    print(f"Server:      {SERVER_URL}")
    print(f"Certificate: {CERTIFICATE_PATH}")
    print(f"Reason:      {REVOCATION_REASON.name}")
    print()

    # Check if certificate exists
    if not CERTIFICATE_PATH.exists():
        print(f"Error: Certificate not found at {CERTIFICATE_PATH}")
        print("Please update CERTIFICATE_PATH to point to a valid certificate.")
        return

    with AcmeClient(
        server_url=SERVER_URL,
        email=EMAIL,
        storage_path=STORAGE_PATH,
    ) as client:
        try:
            # Step 1: Create or load account
            print("[1/3] Loading account...")
            account = client.create_account()
            print(f"    Account: {account.uri}")

            # Step 2: Read certificate
            print(f"\n[2/3] Reading certificate from {CERTIFICATE_PATH}...")
            cert_pem = CERTIFICATE_PATH.read_text()
            print(f"    Certificate loaded ({len(cert_pem)} bytes)")

            # Step 3: Revoke certificate
            print("\n[3/3] Revoking certificate...")
            print(f"    Reason: {REVOCATION_REASON.name}")

            # IMPORTANT: Uncomment the following lines to actually revoke
            # client.revoke_certificate(cert_pem, reason=REVOCATION_REASON)
            # print("    Certificate revoked!")

            print("    (Skipped - uncomment to enable revocation)")
            print()
            print("    WARNING: Certificate revocation is PERMANENT!")
            print("    Once revoked, the certificate will be added to CRLs")
            print("    and OCSP responses will indicate it is revoked.")

            # Success
            print("\n" + "=" * 60)
            print("READY TO REVOKE")
            print("=" * 60)
            print("Uncomment the revoke_certificate() call to perform revocation.")

        except AcmeError as e:
            print(f"\nError: {e.message}")
            raise


if __name__ == "__main__":
    main()
