#!/usr/bin/env python3
"""Example: Using convenience methods for quick certificate issuance.

This example demonstrates the high-level convenience methods for
common certificate operations:

- quick_issue(): Issue a certificate in one call
- issue_batch(): Issue multiple certificates
- renew_if_needed(): Conditionally renew expiring certificates
- get_certificate_info(): Extract certificate details

Usage:
    python quick_issue.py
"""




def show_quick_issue() -> None:
    """Show the quick_issue() function."""
    print("=== quick_issue() - One-Line Certificate Issuance ===\n")

    print("The quick_issue() function handles the entire certificate")
    print("issuance process in a single call:")
    print()
    print('''from acmeow import quick_issue, ChallengeType, KeyType
from acmeow import CallbackDnsHandler

# Define DNS callbacks
def create_txt(domain, name, value):
    # Create TXT record via your DNS API
    dns_api.create_record(name, "TXT", value)

def delete_txt(domain, name):
    # Delete TXT record
    dns_api.delete_record(name, "TXT")

# Create handler
handler = CallbackDnsHandler(create_txt, delete_txt)

# Issue certificate in ONE call
cert_pem, key_pem = quick_issue(
    domain="example.com",
    email="admin@example.com",
    handler=handler,
    # Optional parameters:
    additional_domains=["www.example.com", "api.example.com"],
    key_type=KeyType.EC256,
    challenge_type=ChallengeType.DNS,
    parallel=True,  # For faster multi-domain issuance
)

# Save certificate
Path("cert.pem").write_text(cert_pem)
Path("key.pem").write_text(key_pem)''')
    print()


def show_issue_batch() -> None:
    """Show the issue_batch() function."""
    print("=== issue_batch() - Multiple Separate Certificates ===\n")

    print("The issue_batch() function issues multiple certificates,")
    print("each domain getting its own separate certificate:")
    print()
    print('''from acmeow import issue_batch, BatchResult

# Each entry creates ONE certificate
# Use a list for multi-domain certs
results = issue_batch(
    domains=[
        "site1.com",                           # Single domain cert
        "site2.com",                           # Single domain cert
        ["api.example.com", "admin.example.com"],  # Multi-domain cert
    ],
    email="admin@example.com",
    handler=handler,
    stop_on_error=False,  # Continue even if one fails
)

# Process results
for result in results:
    if result.success:
        print(f"[OK] {result.domain}")
        # Save certificate
        Path(f"{result.domain}.crt").write_text(result.cert_pem)
        Path(f"{result.domain}.key").write_text(result.key_pem)
    else:
        print(f"[FAIL] {result.domain}: {result.error}")''')
    print()


def show_renew_if_needed() -> None:
    """Show the renew_if_needed() function."""
    print("=== renew_if_needed() - Automatic Certificate Renewal ===\n")

    print("The renew_if_needed() function checks expiration and renews")
    print("only if the certificate is expiring soon:")
    print()
    print('''from acmeow import renew_if_needed

# Check and renew if needed
renewed, cert_pem, key_pem = renew_if_needed(
    cert_path="certs/example.com.crt",
    key_path="certs/example.com.key",
    email="admin@example.com",
    handler=handler,
    days_before_expiry=30,  # Renew if expiring within 30 days
    force=False,            # Set True to force renewal
)

if renewed:
    print("Certificate was renewed!")
    # Files are automatically updated
else:
    print("Certificate still valid, no renewal needed")''')
    print()

    print("Use in a cron job for automatic renewal:")
    print()
    print('''#!/bin/bash
# Add to crontab: 0 3 * * * /path/to/renew.py

python3 -c "
from acmeow import renew_if_needed
from my_dns import handler

renew_if_needed(
    '/etc/ssl/certs/example.com.crt',
    '/etc/ssl/private/example.com.key',
    'admin@example.com',
    handler,
    days_before_expiry=30,
)
"''')
    print()


def show_certificate_info() -> None:
    """Show certificate info functions."""
    print("=== get_certificate_info() - Extract Certificate Details ===\n")

    print("The get_certificate_info() function parses certificate details:")
    print()
    print('''from acmeow import get_certificate_info, CertificateInfo

# Read certificate
cert_pem = Path("cert.pem").read_text()

# Get info
info = get_certificate_info(cert_pem)

print(f"Subject: {info.subject}")
print(f"Issuer: {info.issuer}")
print(f"Domains: {info.domains}")
print(f"Serial: {info.serial_number}")
print(f"Valid from: {info.not_before}")
print(f"Valid until: {info.not_after}")
print(f"Days until expiry: {info.days_until_expiry:.1f}")
print(f"Key type: {info.key_type} ({info.key_size})")
print(f"SHA-256: {info.fingerprint_sha256}")

# Check status
if info.is_expired:
    print("Certificate is EXPIRED!")
elif info.is_expiring_soon:  # Within 30 days
    print("Certificate expires soon!")
else:
    print("Certificate is valid")''')
    print()


def show_check_expiry() -> None:
    """Show the check_certificate_expiry() function."""
    print("=== check_certificate_expiry() - Quick Status Check ===\n")

    print("Quick function to check certificate status:")
    print()
    print('''from acmeow import check_certificate_expiry

info, status = check_certificate_expiry(
    "cert.pem",
    warn_days=30,   # Warn if expiring within 30 days
    error_days=7,   # Error if expiring within 7 days
)

# status is one of: "ok", "warning", "error", "expired"
print(f"Status: {status}")
print(f"Expires in: {info.days_until_expiry:.1f} days")

if status == "ok":
    print("Certificate is healthy")
elif status == "warning":
    print("Certificate expires soon - consider renewing")
elif status == "error":
    print("Certificate critically close to expiry!")
elif status == "expired":
    print("Certificate has EXPIRED!")''')
    print()


def show_dns_provider_integration() -> None:
    """Show integration with DNS providers."""
    print("=== Using with DNS Providers ===\n")

    print("Combine convenience methods with DNS providers:")
    print()
    print('''from acmeow import (
    quick_issue,
    get_dns_provider,
    DnsProviderHandler,
    ChallengeType,
)

# Get provider
provider = get_dns_provider("cloudflare", api_token="your-token")
handler = DnsProviderHandler(provider)

# Issue certificate
cert, key = quick_issue(
    domain="example.com",
    email="admin@example.com",
    handler=handler,
    additional_domains=["www.example.com"],
    parallel=True,  # Parallel DNS record creation
)

# For batch with provider
results = issue_batch(
    domains=["site1.com", "site2.com"],
    email="admin@example.com",
    handler=handler,
    parallel_challenges=True,
)''')
    print()


def main() -> None:
    """Run the quick issue examples."""
    show_quick_issue()
    show_issue_batch()
    show_renew_if_needed()
    show_certificate_info()
    show_check_expiry()
    show_dns_provider_integration()

    print("=== Summary ===")
    print("Convenience methods simplify common operations:")
    print()
    print("  quick_issue()        - Full issuance in one call")
    print("  issue_batch()        - Multiple separate certificates")
    print("  renew_if_needed()    - Conditional renewal")
    print("  get_certificate_info() - Parse certificate details")
    print("  check_certificate_expiry() - Quick status check")
    print()
    print("These methods handle account creation, order management,")
    print("challenge completion, and error handling internally.")


if __name__ == "__main__":
    main()
