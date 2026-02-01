"""Enumerations for the ACME client library.

Provides type-safe enumerations for challenge types, identifier types,
and cryptographic key types used throughout the library.
"""

from __future__ import annotations

import sys
from enum import Enum

# StrEnum was added in Python 3.11; for 3.10 compatibility, use str + Enum
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    class StrEnum(str, Enum):
        """String enumeration for Python 3.10 compatibility."""

        def __str__(self) -> str:
            return str(self.value)


class ChallengeType(StrEnum):
    """ACME challenge types for domain validation.

    The ACME protocol supports multiple challenge types for proving
    control over a domain. This library supports DNS-01, HTTP-01,
    and TLS-ALPN-01.
    """

    DNS = "dns-01"
    """DNS-01 challenge: Prove control by creating a DNS TXT record."""

    HTTP = "http-01"
    """HTTP-01 challenge: Prove control by serving a file over HTTP."""

    TLS_ALPN = "tls-alpn-01"
    """TLS-ALPN-01 challenge: Prove control by serving a special TLS certificate."""


class IdentifierType(StrEnum):
    """ACME identifier types.

    Identifiers specify what the certificate will be issued for.
    DNS identifiers (domain names) are the most common.
    """

    DNS = "dns"
    """DNS identifier: A domain name (e.g., example.com)."""

    IP = "ip"
    """IP identifier: An IP address (requires ACME server support)."""


class KeyType(StrEnum):
    """Cryptographic key types for certificate private keys.

    Specifies the algorithm and key size for the certificate's
    private key. RSA keys offer broader compatibility, while EC
    keys provide equivalent security with smaller key sizes.
    """

    RSA2048 = "rsa2048"
    """RSA with 2048-bit key (minimum recommended size)."""

    RSA3072 = "rsa3072"
    """RSA with 3072-bit key (good balance of security and performance)."""

    RSA4096 = "rsa4096"
    """RSA with 4096-bit key (high security)."""

    RSA8192 = "rsa8192"
    """RSA with 8192-bit key (maximum security, slow)."""

    EC256 = "ec256"
    """ECDSA with P-256 curve (equivalent to RSA 3072)."""

    EC384 = "ec384"
    """ECDSA with P-384 curve (equivalent to RSA 7680)."""

    @classmethod
    def get_all(cls) -> list[KeyType]:
        """Return all supported key types.

        Returns:
            List of all KeyType enumeration values.
        """
        return list(cls)

    @classmethod
    def get_rsa_types(cls) -> list[KeyType]:
        """Return all RSA key types.

        Returns:
            List of RSA KeyType enumeration values.
        """
        return [cls.RSA2048, cls.RSA3072, cls.RSA4096, cls.RSA8192]

    @classmethod
    def get_ec_types(cls) -> list[KeyType]:
        """Return all EC key types.

        Returns:
            List of EC KeyType enumeration values.
        """
        return [cls.EC256, cls.EC384]


class OrderStatus(StrEnum):
    """ACME order status values.

    Orders progress through these states during the certificate
    issuance process.
    """

    PENDING = "pending"
    """Order created, authorizations not yet completed."""

    READY = "ready"
    """All authorizations completed, ready for finalization."""

    PROCESSING = "processing"
    """CSR submitted, certificate being issued."""

    VALID = "valid"
    """Certificate issued and available for download."""

    INVALID = "invalid"
    """Order failed and cannot be completed."""


class AuthorizationStatus(StrEnum):
    """ACME authorization status values.

    Authorizations prove control over an identifier and progress
    through these states.
    """

    PENDING = "pending"
    """Authorization created, no challenge completed yet."""

    VALID = "valid"
    """Challenge completed successfully."""

    INVALID = "invalid"
    """Challenge failed."""

    DEACTIVATED = "deactivated"
    """Authorization deactivated by the client."""

    EXPIRED = "expired"
    """Authorization expired."""

    REVOKED = "revoked"
    """Authorization revoked by the server."""


class ChallengeStatus(StrEnum):
    """ACME challenge status values.

    Challenges prove control over a specific identifier and progress
    through these states.
    """

    PENDING = "pending"
    """Challenge created, waiting for client response."""

    PROCESSING = "processing"
    """Challenge response submitted, server is validating."""

    VALID = "valid"
    """Challenge completed successfully."""

    INVALID = "invalid"
    """Challenge failed validation."""


class AccountStatus(StrEnum):
    """ACME account status values."""

    VALID = "valid"
    """Account is active and can be used."""

    DEACTIVATED = "deactivated"
    """Account has been deactivated."""

    REVOKED = "revoked"
    """Account has been revoked by the server."""


class RevocationReason(int, Enum):
    """Certificate revocation reasons per RFC 5280.

    These reason codes indicate why a certificate is being revoked.
    The ACME protocol uses these codes when revoking certificates.
    """

    UNSPECIFIED = 0
    """No specific reason provided."""

    KEY_COMPROMISE = 1
    """The certificate's private key has been compromised."""

    CA_COMPROMISE = 2
    """The CA that issued the certificate has been compromised."""

    AFFILIATION_CHANGED = 3
    """The subject's affiliation has changed."""

    SUPERSEDED = 4
    """The certificate has been superseded by a new one."""

    CESSATION_OF_OPERATION = 5
    """The subject has ceased operations."""

    CERTIFICATE_HOLD = 6
    """The certificate is temporarily on hold."""

    # Value 7 is unused

    REMOVE_FROM_CRL = 8
    """Remove a previously held certificate from CRL."""

    PRIVILEGE_WITHDRAWN = 9
    """The subject's privileges have been withdrawn."""

    AA_COMPROMISE = 10
    """The attribute authority has been compromised."""
