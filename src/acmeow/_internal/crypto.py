"""Cryptographic utilities for the ACME client library.

Provides key generation, CSR creation, JWS signing, and related
cryptographic operations required by the ACME protocol.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.x509.oid import NameOID

from acmeow._internal.encoding import base64url_encode, json_bytes
from acmeow.enums import KeyType
from acmeow.exceptions import AcmeConfigurationError

if TYPE_CHECKING:
    from acmeow.models.identifier import Identifier

logger = logging.getLogger(__name__)

# Type alias for keys that can be used for certificate generation
CertKeyTypes = rsa.RSAPrivateKey | ec.EllipticCurvePrivateKey
# Type alias for keys that can sign certificates/CSRs
SigningKeyTypes = rsa.RSAPrivateKey | ec.EllipticCurvePrivateKey


def generate_private_key(key_type: KeyType) -> CertKeyTypes:
    """Generate a private key of the specified type.

    Args:
        key_type: The type and size of key to generate.

    Returns:
        Generated private key (RSA or EC).

    Raises:
        AcmeConfigurationError: If the key type is not supported.
    """
    match key_type:
        case KeyType.RSA2048:
            return rsa.generate_private_key(public_exponent=65537, key_size=2048)
        case KeyType.RSA3072:
            return rsa.generate_private_key(public_exponent=65537, key_size=3072)
        case KeyType.RSA4096:
            return rsa.generate_private_key(public_exponent=65537, key_size=4096)
        case KeyType.RSA8192:
            return rsa.generate_private_key(public_exponent=65537, key_size=8192)
        case KeyType.EC256:
            return ec.generate_private_key(ec.SECP256R1())
        case KeyType.EC384:
            return ec.generate_private_key(ec.SECP384R1())
        case _:
            raise AcmeConfigurationError(f"Unsupported key type: {key_type}")


def generate_account_key() -> ec.EllipticCurvePrivateKey:
    """Generate an EC P-256 private key for ACME account.

    Account keys use EC P-256 (SECP256R1) as recommended by the ACME
    specification for smaller signatures and faster verification.

    Returns:
        EC P-256 private key.
    """
    return ec.generate_private_key(ec.SECP256R1())


def serialize_private_key(key: CertKeyTypes, password: bytes | None = None) -> bytes:
    """Serialize a private key to PEM format.

    Args:
        key: Private key to serialize.
        password: Optional password for encryption.

    Returns:
        PEM-encoded private key bytes.
    """
    encryption = (
        serialization.BestAvailableEncryption(password)
        if password
        else serialization.NoEncryption()
    )
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=encryption,
    )


def load_private_key(pem_data: bytes, password: bytes | None = None) -> SigningKeyTypes:
    """Load a private key from PEM format.

    Args:
        pem_data: PEM-encoded private key bytes.
        password: Optional password for decryption.

    Returns:
        Loaded private key (RSA or EC).

    Raises:
        ValueError: If the key type is not supported.
    """
    key = serialization.load_pem_private_key(pem_data, password=password)
    if not isinstance(key, (rsa.RSAPrivateKey, ec.EllipticCurvePrivateKey)):
        raise ValueError(f"Unsupported key type: {type(key)}")
    return key


def get_jwk(key: ec.EllipticCurvePrivateKey) -> dict[str, str]:
    """Get the JWK (JSON Web Key) representation of an EC public key.

    Creates a JWK with the public key components in the format required
    by the ACME protocol. Keys are sorted alphabetically as required
    for thumbprint calculation.

    Args:
        key: EC private key.

    Returns:
        JWK dictionary with kty, crv, x, and y fields.
    """
    public_key = key.public_key()
    public_numbers = public_key.public_numbers()

    # Pad coordinates to 32 bytes (256 bits) for P-256
    x_bytes = public_numbers.x.to_bytes(32, byteorder="big")
    y_bytes = public_numbers.y.to_bytes(32, byteorder="big")

    return {
        "crv": "P-256",
        "kty": "EC",
        "x": base64url_encode(x_bytes),
        "y": base64url_encode(y_bytes),
    }


def get_jwk_thumbprint(jwk: dict[str, str]) -> str:
    """Calculate the JWK thumbprint.

    Computes the SHA-256 hash of the canonical JWK representation
    as defined in RFC 7638.

    Args:
        jwk: JWK dictionary.

    Returns:
        Base64url-encoded SHA-256 thumbprint.
    """
    # Create canonical JWK with required members in lexicographic order
    canonical = {"crv": jwk["crv"], "kty": jwk["kty"], "x": jwk["x"], "y": jwk["y"]}
    thumbprint_input = json_bytes(canonical)
    digest = hashlib.sha256(thumbprint_input).digest()
    return base64url_encode(digest)


def create_csr(
    identifiers: list[Identifier],
    key: CertKeyTypes,
    common_name: str | None = None,
) -> bytes:
    """Create a Certificate Signing Request (CSR).

    Creates a CSR with the first identifier as the Common Name (CN)
    and all identifiers as Subject Alternative Names (SANs).

    Args:
        identifiers: List of identifiers (domains/IPs) for the certificate.
        key: Private key for signing the CSR.
        common_name: Optional CN override (defaults to first identifier).

    Returns:
        DER-encoded CSR bytes.

    Raises:
        AcmeConfigurationError: If no identifiers provided.
    """
    if not identifiers:
        raise AcmeConfigurationError("At least one identifier is required for CSR")

    cn = common_name or identifiers[0].value

    # Build subject
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])

    # Build SANs
    san_entries: list[x509.GeneralName] = []
    for identifier in identifiers:
        if identifier.type.value == "dns":
            san_entries.append(x509.DNSName(identifier.value))
        elif identifier.type.value == "ip":
            import ipaddress

            san_entries.append(x509.IPAddress(ipaddress.ip_address(identifier.value)))

    # Create CSR builder
    builder = x509.CertificateSigningRequestBuilder()
    builder = builder.subject_name(subject)

    if san_entries:
        builder = builder.add_extension(
            x509.SubjectAlternativeName(san_entries),
            critical=False,
        )

    # Sign and return DER encoding
    csr = builder.sign(key, hashes.SHA256())
    return csr.public_bytes(serialization.Encoding.DER)


def sign_es256(key: ec.EllipticCurvePrivateKey, data: bytes) -> bytes:
    """Sign data using ES256 (ECDSA with SHA-256).

    Produces a signature in the R||S format (64 bytes) required by JWS,
    rather than the DER format used by default.

    Args:
        key: EC P-256 private key.
        data: Data to sign.

    Returns:
        64-byte signature in R||S format.
    """
    from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

    # Sign with ECDSA SHA-256
    der_signature = key.sign(data, ec.ECDSA(hashes.SHA256()))

    # Decode DER signature to get r and s integers
    r, s = decode_dss_signature(der_signature)

    # Convert to fixed-width byte arrays (32 bytes each for P-256)
    r_bytes = r.to_bytes(32, byteorder="big")
    s_bytes = s.to_bytes(32, byteorder="big")

    return r_bytes + s_bytes


def get_key_authorization(token: str, thumbprint: str) -> str:
    """Compute the key authorization string for a challenge.

    The key authorization is used to prove control over the account key
    during challenge validation.

    Args:
        token: Challenge token from the ACME server.
        thumbprint: JWK thumbprint of the account key.

    Returns:
        Key authorization string (token.thumbprint).
    """
    return f"{token}.{thumbprint}"


def get_dns_challenge_value(key_authorization: str) -> str:
    """Compute the DNS TXT record value for a DNS-01 challenge.

    The DNS record value is the base64url-encoded SHA-256 hash
    of the key authorization.

    Args:
        key_authorization: Key authorization string.

    Returns:
        Base64url-encoded SHA-256 hash for DNS TXT record.
    """
    digest = hashlib.sha256(key_authorization.encode("utf-8")).digest()
    return base64url_encode(digest)
