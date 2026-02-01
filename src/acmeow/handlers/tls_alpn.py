"""TLS-ALPN-01 challenge handlers.

Provides handlers for TLS-ALPN-01 challenge validation as specified
in RFC 8737. This challenge type proves control over a domain by
serving a specially crafted TLS certificate with the acmeIdentifier
extension.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, cast

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.types import CertificateIssuerPrivateKeyTypes
from cryptography.x509.oid import ExtensionOID, NameOID, ObjectIdentifier

from acmeow.handlers.base import ChallengeHandler

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric.types import PrivateKeyTypes

logger = logging.getLogger(__name__)

# OID for the acmeIdentifier extension (1.3.6.1.5.5.7.1.31)
# See RFC 8737 Section 3
ACME_IDENTIFIER_OID = ObjectIdentifier("1.3.6.1.5.5.7.1.31")

# ALPN protocol identifier for ACME TLS-ALPN-01
ACME_TLS_ALPN_PROTOCOL = b"acme-tls/1"


def generate_tls_alpn_certificate(
    domain: str,
    key_authorization: str,
    key: PrivateKeyTypes | None = None,
    validity_days: int = 1,
) -> tuple[bytes, bytes]:
    """Generate a TLS-ALPN-01 validation certificate.

    Creates a self-signed certificate with the acmeIdentifier extension
    containing the SHA-256 hash of the key authorization, as required
    by RFC 8737.

    Args:
        domain: The domain name to validate.
        key_authorization: The key authorization string (token.thumbprint).
        key: Private key to use. If None, generates an EC P-256 key.
        validity_days: Certificate validity period in days. Default 1.

    Returns:
        Tuple of (certificate_pem, private_key_pem) as bytes.

    Example:
        >>> cert_pem, key_pem = generate_tls_alpn_certificate(
        ...     "example.com",
        ...     "token.thumbprint",
        ... )
    """
    # Generate key if not provided
    if key is None:
        key = ec.generate_private_key(ec.SECP256R1())

    # Compute the authorization hash
    auth_hash = hashlib.sha256(key_authorization.encode("utf-8")).digest()

    # Build the acmeIdentifier extension value
    # This is an OCTET STRING containing the 32-byte SHA-256 hash
    # Encoded as ASN.1 OCTET STRING: 04 20 <32 bytes>
    extension_value = bytes([0x04, 0x20]) + auth_hash

    # Create certificate subject
    subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, domain),
    ])

    # Calculate validity period
    now = datetime.now(timezone.utc)
    not_before = now - timedelta(minutes=5)  # Allow for clock skew
    not_after = now + timedelta(days=validity_days)

    # Build certificate
    # Cast key to the type expected by CertificateBuilder
    signing_key = cast(CertificateIssuerPrivateKeyTypes, key)
    builder = x509.CertificateBuilder()
    builder = builder.subject_name(subject)
    builder = builder.issuer_name(subject)  # Self-signed
    builder = builder.public_key(signing_key.public_key())
    builder = builder.serial_number(x509.random_serial_number())
    builder = builder.not_valid_before(not_before)
    builder = builder.not_valid_after(not_after)

    # Add Subject Alternative Name with the domain
    builder = builder.add_extension(
        x509.SubjectAlternativeName([x509.DNSName(domain)]),
        critical=False,
    )

    # Add the acmeIdentifier extension (MUST be critical per RFC 8737)
    builder = builder.add_extension(
        x509.UnrecognizedExtension(ACME_IDENTIFIER_OID, extension_value),
        critical=True,
    )

    # Sign the certificate
    certificate = builder.sign(signing_key, hashes.SHA256())

    # Serialize to PEM
    cert_pem = certificate.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    return cert_pem, key_pem


def validate_tls_alpn_certificate(
    cert_pem: bytes,
    expected_domain: str,
    expected_key_auth: str,
) -> bool:
    """Validate a TLS-ALPN-01 certificate.

    Checks that the certificate has the correct acmeIdentifier extension
    with the expected authorization hash.

    Args:
        cert_pem: PEM-encoded certificate bytes.
        expected_domain: The domain name that should be in the certificate.
        expected_key_auth: The expected key authorization string.

    Returns:
        True if the certificate is valid for the challenge.
    """
    try:
        cert = x509.load_pem_x509_certificate(cert_pem)

        # Check domain in SAN
        try:
            san_ext = cert.extensions.get_extension_for_oid(
                ExtensionOID.SUBJECT_ALTERNATIVE_NAME
            )
            san_value = cast(x509.SubjectAlternativeName, san_ext.value)
            # Handle both old and new cryptography versions
            dns_names: list[str] = []
            for name in san_value.get_values_for_type(x509.DNSName):
                dns_names.append(str(name))
            if expected_domain not in dns_names:
                logger.warning("Domain %s not in certificate SANs: %s", expected_domain, dns_names)
                return False
        except x509.ExtensionNotFound:
            logger.warning("Certificate missing SAN extension")
            return False

        # Check acmeIdentifier extension
        try:
            acme_ext = cert.extensions.get_extension_for_oid(ACME_IDENTIFIER_OID)

            # Extension must be critical
            if not acme_ext.critical:
                logger.warning("acmeIdentifier extension is not critical")
                return False

            # Extract the hash from the extension value
            # Value is ASN.1 OCTET STRING: 04 20 <32 bytes>
            unrecognized_ext = cast(x509.UnrecognizedExtension, acme_ext.value)
            ext_value = unrecognized_ext.value
            if len(ext_value) < 34 or ext_value[0] != 0x04 or ext_value[1] != 0x20:
                logger.warning("Invalid acmeIdentifier extension format")
                return False

            cert_hash = ext_value[2:34]

            # Compute expected hash
            expected_hash = hashlib.sha256(expected_key_auth.encode("utf-8")).digest()

            if cert_hash != expected_hash:
                logger.warning("acmeIdentifier hash mismatch")
                return False

        except x509.ExtensionNotFound:
            logger.warning("Certificate missing acmeIdentifier extension")
            return False

        return True

    except Exception as e:
        logger.warning("Failed to validate TLS-ALPN certificate: %s", e)
        return False


class CallbackTlsAlpnHandler(ChallengeHandler):
    """TLS-ALPN-01 handler using user-provided callbacks.

    This handler delegates certificate deployment to user-provided
    callback functions, allowing integration with any TLS server.

    Args:
        deploy_callback: Callback to deploy the certificate.
            Signature: (domain: str, cert_pem: bytes, key_pem: bytes) -> None
        cleanup_callback: Callback to remove the certificate.
            Signature: (domain: str) -> None

    Example:
        >>> def deploy_cert(domain, cert_pem, key_pem):
        ...     # Configure TLS server with certificate
        ...     server.set_certificate(domain, cert_pem, key_pem)
        >>> def cleanup_cert(domain):
        ...     server.remove_certificate(domain)
        >>> handler = CallbackTlsAlpnHandler(deploy_cert, cleanup_cert)
    """

    def __init__(
        self,
        deploy_callback: Callable[[str, bytes, bytes], None],
        cleanup_callback: Callable[[str], None],
    ) -> None:
        self._deploy_callback = deploy_callback
        self._cleanup_callback = cleanup_callback

    def setup(self, domain: str, token: str, key_authorization: str) -> None:
        """Generate and deploy TLS-ALPN-01 certificate.

        Args:
            domain: The domain being validated.
            token: The challenge token (not used for TLS-ALPN-01).
            key_authorization: The key authorization string.
        """
        logger.info("Generating TLS-ALPN-01 certificate for %s", domain)

        cert_pem, key_pem = generate_tls_alpn_certificate(domain, key_authorization)

        logger.info("Deploying TLS-ALPN-01 certificate for %s", domain)
        self._deploy_callback(domain, cert_pem, key_pem)

    def cleanup(self, domain: str, token: str) -> None:
        """Remove TLS-ALPN-01 certificate.

        Args:
            domain: The domain that was validated.
            token: The challenge token (not used).
        """
        logger.info("Cleaning up TLS-ALPN-01 certificate for %s", domain)
        try:
            self._cleanup_callback(domain)
        except Exception as e:
            logger.warning(
                "Failed to cleanup TLS-ALPN certificate for %s: %s",
                domain,
                e,
            )


class FileTlsAlpnHandler(ChallengeHandler):
    """TLS-ALPN-01 handler that writes certificates to files.

    This handler writes the validation certificate and key to files,
    which can then be loaded by a TLS server. Optionally calls a reload
    callback to signal the server to reload certificates.

    Args:
        cert_dir: Directory to write certificate files.
        cert_pattern: Pattern for certificate filename. {domain} is replaced.
        key_pattern: Pattern for key filename. {domain} is replaced.
        reload_callback: Optional callback to reload the TLS server.
            Signature: () -> None

    Example:
        >>> handler = FileTlsAlpnHandler(
        ...     cert_dir=Path("/etc/tls/acme"),
        ...     cert_pattern="{domain}.crt",
        ...     key_pattern="{domain}.key",
        ...     reload_callback=lambda: subprocess.run(["nginx", "-s", "reload"]),
        ... )
    """

    def __init__(
        self,
        cert_dir: Path,
        cert_pattern: str = "{domain}.alpn.crt",
        key_pattern: str = "{domain}.alpn.key",
        reload_callback: Callable[[], None] | None = None,
    ) -> None:
        self._cert_dir = Path(cert_dir)
        self._cert_pattern = cert_pattern
        self._key_pattern = key_pattern
        self._reload_callback = reload_callback

    def _get_cert_path(self, domain: str) -> Path:
        """Get the certificate file path for a domain."""
        filename = self._cert_pattern.format(domain=domain.replace("*", "_wildcard"))
        return self._cert_dir / filename

    def _get_key_path(self, domain: str) -> Path:
        """Get the key file path for a domain."""
        filename = self._key_pattern.format(domain=domain.replace("*", "_wildcard"))
        return self._cert_dir / filename

    def setup(self, domain: str, token: str, key_authorization: str) -> None:
        """Generate and write TLS-ALPN-01 certificate files.

        Args:
            domain: The domain being validated.
            token: The challenge token (not used for TLS-ALPN-01).
            key_authorization: The key authorization string.
        """
        logger.info("Generating TLS-ALPN-01 certificate for %s", domain)

        cert_pem, key_pem = generate_tls_alpn_certificate(domain, key_authorization)

        # Ensure directory exists
        self._cert_dir.mkdir(parents=True, exist_ok=True)

        # Write files
        cert_path = self._get_cert_path(domain)
        key_path = self._get_key_path(domain)

        cert_path.write_bytes(cert_pem)
        key_path.write_bytes(key_pem)

        # Set restrictive permissions on key file
        with contextlib.suppress(OSError):
            key_path.chmod(0o600)  # Windows doesn't support chmod

        logger.info(
            "TLS-ALPN-01 certificate written to %s and %s",
            cert_path,
            key_path,
        )

        # Reload server if callback provided
        if self._reload_callback:
            logger.info("Reloading TLS server")
            try:
                self._reload_callback()
            except Exception as e:
                logger.warning("Failed to reload TLS server: %s", e)

    def cleanup(self, domain: str, token: str) -> None:
        """Remove TLS-ALPN-01 certificate files.

        Args:
            domain: The domain that was validated.
            token: The challenge token (not used).
        """
        cert_path = self._get_cert_path(domain)
        key_path = self._get_key_path(domain)

        for path in [cert_path, key_path]:
            try:
                if path.exists():
                    path.unlink()
                    logger.info("Removed TLS-ALPN file: %s", path)
            except Exception as e:
                logger.warning("Failed to remove %s: %s", path, e)

        # Reload server if callback provided
        if self._reload_callback:
            try:
                self._reload_callback()
            except Exception as e:
                logger.warning("Failed to reload TLS server after cleanup: %s", e)
