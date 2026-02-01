#!/usr/bin/env python3
"""Example: TLS-ALPN-01 challenge workflow.

This example demonstrates how to use the TLS-ALPN-01 challenge type
to obtain a certificate. TLS-ALPN-01 proves control over a domain by
serving a specially crafted TLS certificate on port 443.

Requirements:
- Control of a TLS server on port 443
- Ability to configure the server to use ALPN protocol "acme-tls/1"

Usage:
    python tls_alpn_challenge.py
"""

from pathlib import Path

from acmeow.handlers.tls_alpn import CallbackTlsAlpnHandler, FileTlsAlpnHandler


def example_callback_handler() -> None:
    """Example using CallbackTlsAlpnHandler with custom callbacks."""
    print("=== TLS-ALPN-01 with Callback Handler ===\n")

    # Define callbacks for deploying TLS certificates
    # In a real scenario, these would configure your TLS server

    deployed_certs: dict[str, tuple[bytes, bytes]] = {}

    def deploy_certificate(domain: str, cert_pem: bytes, key_pem: bytes) -> None:
        """Deploy the validation certificate to the TLS server."""
        print(f"Deploying TLS-ALPN certificate for {domain}")
        # In production, you would:
        # 1. Save cert and key to a location your TLS server can read
        # 2. Configure the server to use ALPN protocol "acme-tls/1"
        # 3. Reload the TLS server configuration
        deployed_certs[domain] = (cert_pem, key_pem)
        print("  Certificate deployed (in memory for demo)")

    def cleanup_certificate(domain: str) -> None:
        """Remove the validation certificate."""
        print(f"Removing TLS-ALPN certificate for {domain}")
        if domain in deployed_certs:
            del deployed_certs[domain]
        print("  Certificate removed")

    # Create the handler
    handler = CallbackTlsAlpnHandler(
        deploy_callback=deploy_certificate,
        cleanup_callback=cleanup_certificate,
    )

    # Use with AcmeClient (staging server for testing)
    print("\nNote: Using Let's Encrypt staging server")
    print("In production, remove 'staging' from the URL\n")

    # This would work with a real domain and TLS server
    # Commented out to prevent accidental runs
    """
    client = AcmeClient(
        server_url="https://acme-staging-v02.api.letsencrypt.org/directory",
        email="admin@example.com",
        storage_path=Path("./acme_data"),
    )

    client.create_account()
    order = client.create_order([Identifier.dns("example.com")])
    client.complete_challenges(handler, ChallengeType.TLS_ALPN)
    client.finalize_order(KeyType.EC256)
    cert, key = client.get_certificate()
    """

    print("Handler created successfully!")
    print("To use this in production:")
    print("1. Configure your TLS server to use ALPN 'acme-tls/1'")
    print("2. Implement deploy_certificate to save certs and reload server")
    print("3. Run the ACME workflow with ChallengeType.TLS_ALPN")


def example_file_handler() -> None:
    """Example using FileTlsAlpnHandler for file-based deployment."""
    print("\n=== TLS-ALPN-01 with File Handler ===\n")

    # Define a directory for certificates
    cert_dir = Path("./tls_alpn_certs")
    cert_dir.mkdir(exist_ok=True)

    # Optional: Define a reload callback
    def reload_nginx() -> None:
        """Reload nginx to pick up new certificates."""
        # subprocess.run(["nginx", "-s", "reload"], check=True)
        print("  Would reload nginx here")

    # Create the handler
    handler = FileTlsAlpnHandler(
        cert_dir=cert_dir,
        cert_pattern="{domain}.alpn.crt",
        key_pattern="{domain}.alpn.key",
        reload_callback=reload_nginx,
    )

    print(f"Certificate directory: {cert_dir.absolute()}")
    print(f"Certificate pattern: {{domain}}.alpn.crt")
    print(f"Key pattern: {{domain}}.alpn.key")

    # Demonstrate certificate generation (without actual ACME)
    print("\nGenerating sample TLS-ALPN certificate...")

    from acmeow.handlers.tls_alpn import generate_tls_alpn_certificate

    cert_pem, key_pem = generate_tls_alpn_certificate(
        domain="demo.example.com",
        key_authorization="sample-token.sample-thumbprint",
    )

    # Save to files
    sample_cert = cert_dir / "demo.example.com.alpn.crt"
    sample_key = cert_dir / "demo.example.com.alpn.key"

    sample_cert.write_bytes(cert_pem)
    sample_key.write_bytes(key_pem)

    print(f"Saved sample certificate to: {sample_cert}")
    print(f"Saved sample key to: {sample_key}")

    # Show certificate info
    from cryptography import x509
    cert = x509.load_pem_x509_certificate(cert_pem)
    print(f"\nCertificate subject: {cert.subject}")
    print(f"Certificate validity: {cert.not_valid_before_utc} to {cert.not_valid_after_utc}")

    # Cleanup
    sample_cert.unlink()
    sample_key.unlink()
    cert_dir.rmdir()
    print("\nCleaned up sample files")


def main() -> None:
    """Run the TLS-ALPN examples."""
    example_callback_handler()
    example_file_handler()

    print("\n=== Summary ===")
    print("TLS-ALPN-01 is useful when:")
    print("- You control a TLS server on port 443")
    print("- DNS-01 is not available (no DNS API access)")
    print("- HTTP-01 is not suitable (e.g., can't serve on port 80)")
    print("\nRequirements:")
    print("- TLS server must support ALPN protocol negotiation")
    print("- Server must be configured to advertise 'acme-tls/1'")
    print("- The validation certificate must be served for the domain")


if __name__ == "__main__":
    main()
