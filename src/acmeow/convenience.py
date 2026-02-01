"""Convenience methods for common certificate operations.

Provides high-level functions for quick certificate issuance, batch
processing, and automatic renewal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from cryptography import x509
from cryptography.x509.oid import ExtensionOID, NameOID

from acmeow.client import AcmeClient
from acmeow.enums import ChallengeType, KeyType
from acmeow.handlers.base import ChallengeHandler
from acmeow.models.identifier import Identifier

logger = logging.getLogger(__name__)

# Default Let's Encrypt staging URL
LETSENCRYPT_STAGING = "https://acme-staging-v02.api.letsencrypt.org/directory"
# Default Let's Encrypt production URL
LETSENCRYPT_PRODUCTION = "https://acme-v02.api.letsencrypt.org/directory"


@dataclass
class CertificateInfo:
    """Information extracted from a certificate.

    Attributes:
        subject: Certificate subject (typically CN).
        issuer: Certificate issuer.
        domains: List of domain names from Subject Alternative Names.
        serial_number: Certificate serial number (hex).
        not_before: Certificate validity start.
        not_after: Certificate validity end (expiration).
        days_until_expiry: Days until certificate expires.
        is_expired: Whether the certificate has expired.
        key_type: Type of public key (RSA or EC).
        key_size: Key size in bits (for RSA) or curve name (for EC).
        fingerprint_sha256: SHA-256 fingerprint of the certificate.
    """

    subject: str
    issuer: str
    domains: list[str]
    serial_number: str
    not_before: datetime
    not_after: datetime
    days_until_expiry: float
    is_expired: bool
    key_type: str
    key_size: str
    fingerprint_sha256: str

    @property
    def is_expiring_soon(self) -> bool:
        """Check if certificate expires within 30 days."""
        return self.days_until_expiry <= 30

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return (
            f"CertificateInfo({self.subject}, "
            f"expires={self.not_after.isoformat()}, "
            f"days_left={self.days_until_expiry:.1f})"
        )


def get_certificate_info(cert_pem: str | bytes) -> CertificateInfo:
    """Extract information from a PEM certificate.

    Args:
        cert_pem: PEM-encoded certificate as string or bytes.

    Returns:
        CertificateInfo with parsed certificate details.

    Example:
        >>> info = get_certificate_info(cert_pem)
        >>> if info.days_until_expiry < 30:
        ...     print("Certificate expires soon!")
    """
    if isinstance(cert_pem, str):
        cert_pem = cert_pem.encode()

    cert = x509.load_pem_x509_certificate(cert_pem)

    # Extract subject
    subject_attrs = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
    subject: str = str(subject_attrs[0].value) if subject_attrs else "Unknown"

    # Extract issuer
    issuer_attrs = cert.issuer.get_attributes_for_oid(NameOID.COMMON_NAME)
    issuer: str = str(issuer_attrs[0].value) if issuer_attrs else "Unknown"

    # Extract domains from SAN
    domains: list[str] = []
    try:
        san_ext = cert.extensions.get_extension_for_oid(ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
        san_value = cast(x509.SubjectAlternativeName, san_ext.value)
        for name in san_value.get_values_for_type(x509.DNSName):
            domains.append(str(name))
    except x509.ExtensionNotFound:
        # Use CN as fallback
        if subject != "Unknown":
            domains.append(subject)

    # Calculate expiry
    now = datetime.now(timezone.utc)
    not_before = cert.not_valid_before_utc
    not_after = cert.not_valid_after_utc
    days_until_expiry = (not_after - now).total_seconds() / 86400
    is_expired = now > not_after

    # Get key info
    public_key = cert.public_key()
    key_type = type(public_key).__name__.replace("PublicKey", "")

    from cryptography.hazmat.primitives.asymmetric import ec, rsa
    if isinstance(public_key, rsa.RSAPublicKey):
        key_size = str(public_key.key_size)
    elif isinstance(public_key, ec.EllipticCurvePublicKey):
        key_size = public_key.curve.name
    else:
        key_size = "Unknown"

    # Calculate fingerprint
    from cryptography.hazmat.primitives import hashes
    fingerprint = cert.fingerprint(hashes.SHA256())
    fingerprint_hex = fingerprint.hex().upper()
    fingerprint_formatted = ":".join(
        fingerprint_hex[i:i+2] for i in range(0, len(fingerprint_hex), 2)
    )

    return CertificateInfo(
        subject=str(subject),
        issuer=str(issuer),
        domains=domains,
        serial_number=format(cert.serial_number, "X"),
        not_before=not_before,
        not_after=not_after,
        days_until_expiry=days_until_expiry,
        is_expired=is_expired,
        key_type=key_type,
        key_size=key_size,
        fingerprint_sha256=fingerprint_formatted,
    )


def quick_issue(
    domain: str,
    email: str,
    handler: ChallengeHandler,
    challenge_type: ChallengeType = ChallengeType.DNS,
    key_type: KeyType = KeyType.EC256,
    storage_path: Path | str | None = None,
    server_url: str = LETSENCRYPT_PRODUCTION,
    additional_domains: list[str] | None = None,
    parallel: bool = False,
    **client_kwargs: Any,
) -> tuple[str, str]:
    """Issue a certificate in one call.

    High-level convenience function that handles the entire certificate
    issuance process: account creation, order creation, challenge completion,
    and certificate retrieval.

    Args:
        domain: Primary domain name for the certificate.
        email: Account email address.
        handler: Challenge handler for completing challenges.
        challenge_type: Type of challenge to use. Default DNS-01.
        key_type: Key type for the certificate. Default EC256.
        storage_path: Directory for storing account/certificate data.
            Default is ".acme" in current directory.
        server_url: ACME server URL. Default is Let's Encrypt production.
        additional_domains: Additional domain names for the certificate.
        parallel: Whether to complete challenges in parallel.
        **client_kwargs: Additional arguments for AcmeClient.

    Returns:
        Tuple of (certificate_pem, private_key_pem).

    Example:
        >>> from acmeow import quick_issue, CallbackDnsHandler
        >>> handler = CallbackDnsHandler(create_record, delete_record)
        >>> cert, key = quick_issue("example.com", "admin@example.com", handler)
    """
    if storage_path is None:
        storage_path = Path(".acme")

    # Build identifier list
    identifiers = [Identifier.dns(domain)]
    if additional_domains:
        for d in additional_domains:
            identifiers.append(Identifier.dns(d))

    logger.info("Starting certificate issuance for %s", domain)

    with AcmeClient(
        server_url=server_url,
        email=email,
        storage_path=storage_path,
        **client_kwargs,
    ) as client:
        # Create or retrieve account
        client.create_account()

        # Create order
        client.create_order(identifiers)

        # Complete challenges
        client.complete_challenges(
            handler,
            challenge_type=challenge_type,
            parallel=parallel,
        )

        # Finalize order
        client.finalize_order(key_type)

        # Get certificate
        cert_pem, key_pem = client.get_certificate()

    logger.info("Certificate issued successfully for %s", domain)
    return cert_pem, key_pem


@dataclass
class BatchResult:
    """Result of a batch certificate issuance.

    Attributes:
        domain: The primary domain.
        success: Whether issuance succeeded.
        cert_pem: Certificate PEM (if successful).
        key_pem: Private key PEM (if successful).
        error: Error message (if failed).
    """

    domain: str
    success: bool
    cert_pem: str | None = None
    key_pem: str | None = None
    error: str | None = None


def issue_batch(
    domains: list[str | list[str]],
    email: str,
    handler: ChallengeHandler,
    challenge_type: ChallengeType = ChallengeType.DNS,
    key_type: KeyType = KeyType.EC256,
    storage_path: Path | str | None = None,
    server_url: str = LETSENCRYPT_PRODUCTION,
    stop_on_error: bool = False,
    parallel_challenges: bool = False,
    **client_kwargs: Any,
) -> list[BatchResult]:
    """Issue multiple certificates in batch.

    Each domain (or domain list) gets its own separate certificate.
    Useful for issuing certificates for multiple unrelated domains.

    Args:
        domains: List of domains or domain lists. Each entry creates
            one certificate. Use a list for multi-domain certs.
        email: Account email address.
        handler: Challenge handler for completing challenges.
        challenge_type: Type of challenge to use. Default DNS-01.
        key_type: Key type for certificates. Default EC256.
        storage_path: Directory for storing account/certificate data.
        server_url: ACME server URL. Default is Let's Encrypt production.
        stop_on_error: Whether to stop on first error. Default False.
        parallel_challenges: Whether to complete challenges in parallel.
        **client_kwargs: Additional arguments for AcmeClient.

    Returns:
        List of BatchResult for each domain.

    Example:
        >>> results = issue_batch(
        ...     ["example.com", ["api.example.com", "www.example.com"]],
        ...     "admin@example.com",
        ...     handler,
        ... )
        >>> for r in results:
        ...     if r.success:
        ...         save_certificate(r.domain, r.cert_pem, r.key_pem)
    """
    if storage_path is None:
        storage_path = Path(".acme")

    results: list[BatchResult] = []

    with AcmeClient(
        server_url=server_url,
        email=email,
        storage_path=storage_path,
        **client_kwargs,
    ) as client:
        # Create or retrieve account (once for all certificates)
        client.create_account()

        for domain_entry in domains:
            # Normalize to list
            domain_list = [domain_entry] if isinstance(domain_entry, str) else list(domain_entry)

            primary_domain = domain_list[0]
            logger.info("Issuing certificate for %s", primary_domain)

            try:
                # Create order
                identifiers = [Identifier.dns(d) for d in domain_list]
                client.create_order(identifiers)

                # Complete challenges
                client.complete_challenges(
                    handler,
                    challenge_type=challenge_type,
                    parallel=parallel_challenges,
                )

                # Finalize order
                client.finalize_order(key_type)

                # Get certificate
                cert_pem, key_pem = client.get_certificate()

                results.append(BatchResult(
                    domain=primary_domain,
                    success=True,
                    cert_pem=cert_pem,
                    key_pem=key_pem,
                ))

                logger.info("Certificate issued for %s", primary_domain)

            except Exception as e:
                logger.error("Failed to issue certificate for %s: %s", primary_domain, e)
                results.append(BatchResult(
                    domain=primary_domain,
                    success=False,
                    error=str(e),
                ))

                if stop_on_error:
                    break

    return results


def renew_if_needed(
    cert_path: Path | str,
    key_path: Path | str,
    email: str,
    handler: ChallengeHandler,
    days_before_expiry: int = 30,
    challenge_type: ChallengeType = ChallengeType.DNS,
    key_type: KeyType = KeyType.EC256,
    storage_path: Path | str | None = None,
    server_url: str = LETSENCRYPT_PRODUCTION,
    force: bool = False,
    parallel: bool = False,
    **client_kwargs: Any,
) -> tuple[bool, str | None, str | None]:
    """Renew a certificate if it's expiring soon.

    Checks the expiration date of an existing certificate and renews it
    if it expires within the specified number of days.

    Args:
        cert_path: Path to the existing certificate PEM file.
        key_path: Path to the existing private key PEM file.
        email: Account email address.
        handler: Challenge handler for completing challenges.
        days_before_expiry: Renew if expiring within this many days. Default 30.
        challenge_type: Type of challenge to use. Default DNS-01.
        key_type: Key type for the new certificate. Default EC256.
        storage_path: Directory for storing account/certificate data.
        server_url: ACME server URL. Default is Let's Encrypt production.
        force: Force renewal regardless of expiration. Default False.
        parallel: Whether to complete challenges in parallel.
        **client_kwargs: Additional arguments for AcmeClient.

    Returns:
        Tuple of (renewed: bool, cert_pem: str | None, key_pem: str | None).
        If renewed, the new certificate and key are returned and also
        written to the original paths.

    Example:
        >>> renewed, cert, key = renew_if_needed(
        ...     "certs/example.com.crt",
        ...     "certs/example.com.key",
        ...     "admin@example.com",
        ...     handler,
        ... )
        >>> if renewed:
        ...     print("Certificate renewed!")
    """
    cert_path = Path(cert_path)
    key_path = Path(key_path)

    if not cert_path.exists():
        logger.warning("Certificate file not found: %s", cert_path)
        return False, None, None

    # Check current certificate
    cert_pem = cert_path.read_text()
    info = get_certificate_info(cert_pem)

    logger.info(
        "Certificate for %s expires in %.1f days",
        info.subject,
        info.days_until_expiry,
    )

    # Check if renewal is needed
    if not force and info.days_until_expiry > days_before_expiry:
        logger.info("Certificate does not need renewal yet")
        return False, None, None

    logger.info("Renewing certificate for %s", info.subject)

    # Issue new certificate
    if storage_path is None:
        storage_path = cert_path.parent / ".acme"

    new_cert, new_key = quick_issue(
        domain=info.domains[0] if info.domains else info.subject,
        email=email,
        handler=handler,
        challenge_type=challenge_type,
        key_type=key_type,
        storage_path=storage_path,
        server_url=server_url,
        additional_domains=info.domains[1:] if len(info.domains) > 1 else None,
        parallel=parallel,
        **client_kwargs,
    )

    # Save new certificate
    cert_path.write_text(new_cert)
    key_path.write_text(new_key)

    logger.info("Certificate renewed and saved")
    return True, new_cert, new_key


def check_certificate_expiry(
    cert_path: Path | str,
    warn_days: int = 30,
    error_days: int = 7,
) -> tuple[CertificateInfo, str]:
    """Check a certificate's expiration status.

    Args:
        cert_path: Path to the certificate PEM file.
        warn_days: Days threshold for warning status.
        error_days: Days threshold for error status.

    Returns:
        Tuple of (CertificateInfo, status) where status is one of:
        "ok", "warning", "error", "expired".

    Example:
        >>> info, status = check_certificate_expiry("cert.pem")
        >>> if status != "ok":
        ...     print(f"Certificate status: {status}")
    """
    cert_path = Path(cert_path)
    cert_pem = cert_path.read_text()
    info = get_certificate_info(cert_pem)

    if info.is_expired:
        status = "expired"
    elif info.days_until_expiry <= error_days:
        status = "error"
    elif info.days_until_expiry <= warn_days:
        status = "warning"
    else:
        status = "ok"

    return info, status
