"""Tests for cryptographic utilities."""

from __future__ import annotations

import hashlib

import pytest
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.x509 import load_der_x509_csr

from acmeow._internal.crypto import (
    create_csr,
    generate_account_key,
    generate_private_key,
    get_dns_challenge_value,
    get_jwk,
    get_jwk_thumbprint,
    get_key_authorization,
    load_private_key,
    serialize_private_key,
    sign_es256,
)
from acmeow._internal.encoding import base64url_encode
from acmeow.enums import KeyType
from acmeow.exceptions import AcmeConfigurationError
from acmeow.models.identifier import Identifier


class TestGeneratePrivateKey:
    """Tests for generate_private_key function."""

    def test_generate_rsa2048(self):
        """Test generating RSA 2048-bit key."""
        key = generate_private_key(KeyType.RSA2048)
        assert isinstance(key, rsa.RSAPrivateKey)
        assert key.key_size == 2048

    def test_generate_rsa3072(self):
        """Test generating RSA 3072-bit key."""
        key = generate_private_key(KeyType.RSA3072)
        assert isinstance(key, rsa.RSAPrivateKey)
        assert key.key_size == 3072

    def test_generate_rsa4096(self):
        """Test generating RSA 4096-bit key."""
        key = generate_private_key(KeyType.RSA4096)
        assert isinstance(key, rsa.RSAPrivateKey)
        assert key.key_size == 4096

    def test_generate_ec256(self):
        """Test generating EC P-256 key."""
        key = generate_private_key(KeyType.EC256)
        assert isinstance(key, ec.EllipticCurvePrivateKey)
        assert key.curve.name == "secp256r1"

    def test_generate_ec384(self):
        """Test generating EC P-384 key."""
        key = generate_private_key(KeyType.EC384)
        assert isinstance(key, ec.EllipticCurvePrivateKey)
        assert key.curve.name == "secp384r1"

    def test_keys_are_unique(self):
        """Test that generated keys are unique."""
        key1 = generate_private_key(KeyType.EC256)
        key2 = generate_private_key(KeyType.EC256)
        # Different keys should have different public numbers
        pub1 = key1.public_key().public_numbers()
        pub2 = key2.public_key().public_numbers()
        assert pub1.x != pub2.x or pub1.y != pub2.y


class TestGenerateAccountKey:
    """Tests for generate_account_key function."""

    def test_returns_ec_key(self):
        """Test that account key is EC."""
        key = generate_account_key()
        assert isinstance(key, ec.EllipticCurvePrivateKey)

    def test_uses_p256_curve(self):
        """Test that account key uses P-256 curve."""
        key = generate_account_key()
        assert key.curve.name == "secp256r1"

    def test_keys_are_unique(self):
        """Test that generated account keys are unique."""
        key1 = generate_account_key()
        key2 = generate_account_key()
        assert key1.public_key().public_numbers() != key2.public_key().public_numbers()


class TestSerializePrivateKey:
    """Tests for serialize_private_key function."""

    def test_serialize_ec_key(self):
        """Test serializing EC key."""
        key = generate_account_key()
        pem = serialize_private_key(key)
        assert b"-----BEGIN PRIVATE KEY-----" in pem

    def test_serialize_rsa_key(self):
        """Test serializing RSA key."""
        key = generate_private_key(KeyType.RSA2048)
        pem = serialize_private_key(key)
        assert b"-----BEGIN PRIVATE KEY-----" in pem

    def test_serialize_with_password(self):
        """Test serializing with password encryption."""
        key = generate_account_key()
        pem = serialize_private_key(key, password=b"secret")
        assert b"-----BEGIN ENCRYPTED PRIVATE KEY-----" in pem

    def test_serialize_roundtrip(self):
        """Test serialize and load roundtrip."""
        original = generate_account_key()
        pem = serialize_private_key(original)
        loaded = load_private_key(pem)
        # Compare public keys
        assert (
            original.public_key().public_numbers() ==
            loaded.public_key().public_numbers()  # type: ignore[union-attr]
        )

    def test_serialize_roundtrip_with_password(self):
        """Test serialize and load with password."""
        original = generate_account_key()
        password = b"test-password"
        pem = serialize_private_key(original, password=password)
        loaded = load_private_key(pem, password=password)
        assert (
            original.public_key().public_numbers() ==
            loaded.public_key().public_numbers()  # type: ignore[union-attr]
        )


class TestLoadPrivateKey:
    """Tests for load_private_key function."""

    def test_load_ec_key(self):
        """Test loading EC key."""
        key = generate_account_key()
        pem = serialize_private_key(key)
        loaded = load_private_key(pem)
        assert isinstance(loaded, ec.EllipticCurvePrivateKey)

    def test_load_rsa_key(self):
        """Test loading RSA key."""
        key = generate_private_key(KeyType.RSA2048)
        pem = serialize_private_key(key)
        loaded = load_private_key(pem)
        assert isinstance(loaded, rsa.RSAPrivateKey)

    def test_load_with_wrong_password_fails(self):
        """Test that wrong password fails."""
        key = generate_account_key()
        pem = serialize_private_key(key, password=b"correct")
        with pytest.raises(Exception):
            load_private_key(pem, password=b"wrong")


class TestGetJwk:
    """Tests for get_jwk function."""

    def test_jwk_structure(self):
        """Test JWK has required fields."""
        key = generate_account_key()
        jwk = get_jwk(key)
        assert "kty" in jwk
        assert "crv" in jwk
        assert "x" in jwk
        assert "y" in jwk

    def test_jwk_values(self):
        """Test JWK values are correct."""
        key = generate_account_key()
        jwk = get_jwk(key)
        assert jwk["kty"] == "EC"
        assert jwk["crv"] == "P-256"
        # x and y should be base64url encoded
        assert isinstance(jwk["x"], str)
        assert isinstance(jwk["y"], str)

    def test_jwk_deterministic(self):
        """Test JWK is deterministic for same key."""
        key = generate_account_key()
        jwk1 = get_jwk(key)
        jwk2 = get_jwk(key)
        assert jwk1 == jwk2

    def test_jwk_different_for_different_keys(self):
        """Test JWK is different for different keys."""
        key1 = generate_account_key()
        key2 = generate_account_key()
        jwk1 = get_jwk(key1)
        jwk2 = get_jwk(key2)
        assert jwk1 != jwk2


class TestGetJwkThumbprint:
    """Tests for get_jwk_thumbprint function."""

    def test_thumbprint_format(self):
        """Test thumbprint is base64url string."""
        key = generate_account_key()
        jwk = get_jwk(key)
        thumbprint = get_jwk_thumbprint(jwk)
        assert isinstance(thumbprint, str)
        # Should not contain padding
        assert "=" not in thumbprint

    def test_thumbprint_length(self):
        """Test thumbprint has correct length."""
        key = generate_account_key()
        jwk = get_jwk(key)
        thumbprint = get_jwk_thumbprint(jwk)
        # SHA-256 produces 32 bytes, base64url encodes to ~43 chars
        assert 40 <= len(thumbprint) <= 50

    def test_thumbprint_deterministic(self):
        """Test thumbprint is deterministic."""
        key = generate_account_key()
        jwk = get_jwk(key)
        tp1 = get_jwk_thumbprint(jwk)
        tp2 = get_jwk_thumbprint(jwk)
        assert tp1 == tp2

    def test_thumbprint_different_for_different_keys(self):
        """Test thumbprint differs for different keys."""
        key1 = generate_account_key()
        key2 = generate_account_key()
        tp1 = get_jwk_thumbprint(get_jwk(key1))
        tp2 = get_jwk_thumbprint(get_jwk(key2))
        assert tp1 != tp2


class TestCreateCsr:
    """Tests for create_csr function."""

    def test_csr_single_domain(self):
        """Test CSR with single domain."""
        key = generate_private_key(KeyType.EC256)
        identifiers = [Identifier.dns("example.com")]
        csr_der = create_csr(identifiers, key)
        assert isinstance(csr_der, bytes)
        # Verify it's valid DER
        csr = load_der_x509_csr(csr_der)
        assert csr is not None

    def test_csr_multiple_domains(self):
        """Test CSR with multiple domains."""
        from cryptography.x509 import SubjectAlternativeName

        key = generate_private_key(KeyType.EC256)
        identifiers = [
            Identifier.dns("example.com"),
            Identifier.dns("www.example.com"),
            Identifier.dns("api.example.com"),
        ]
        csr_der = create_csr(identifiers, key)
        csr = load_der_x509_csr(csr_der)
        # Check SANs extension exists
        san_ext = csr.extensions.get_extension_for_class(SubjectAlternativeName)
        assert san_ext is not None
        # Verify all domains are in SAN
        san_values = [name.value for name in san_ext.value]
        assert "example.com" in san_values
        assert "www.example.com" in san_values
        assert "api.example.com" in san_values

    def test_csr_common_name(self):
        """Test CSR has correct common name."""
        from cryptography.x509.oid import NameOID

        key = generate_private_key(KeyType.EC256)
        identifiers = [Identifier.dns("example.com")]
        csr_der = create_csr(identifiers, key)
        csr = load_der_x509_csr(csr_der)
        # Get CN from subject
        cn_attrs = csr.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
        assert len(cn_attrs) == 1
        assert cn_attrs[0].value == "example.com"

    def test_csr_custom_common_name(self):
        """Test CSR with custom common name."""
        key = generate_private_key(KeyType.EC256)
        identifiers = [Identifier.dns("example.com")]
        csr_der = create_csr(identifiers, key, common_name="custom.example.com")
        csr = load_der_x509_csr(csr_der)
        assert csr is not None

    def test_csr_with_rsa_key(self):
        """Test CSR with RSA key."""
        key = generate_private_key(KeyType.RSA2048)
        identifiers = [Identifier.dns("example.com")]
        csr_der = create_csr(identifiers, key)
        csr = load_der_x509_csr(csr_der)
        assert csr is not None

    def test_csr_empty_identifiers_raises(self):
        """Test CSR with empty identifiers raises error."""
        key = generate_private_key(KeyType.EC256)
        with pytest.raises(AcmeConfigurationError, match="identifier"):
            create_csr([], key)

    def test_csr_with_ip(self):
        """Test CSR with IP identifier."""
        key = generate_private_key(KeyType.EC256)
        identifiers = [
            Identifier.dns("example.com"),
            Identifier.ip("192.168.1.1"),
        ]
        csr_der = create_csr(identifiers, key)
        csr = load_der_x509_csr(csr_der)
        assert csr is not None


class TestSignEs256:
    """Tests for sign_es256 function."""

    def test_signature_length(self):
        """Test signature is 64 bytes."""
        key = generate_account_key()
        data = b"test data to sign"
        signature = sign_es256(key, data)
        assert len(signature) == 64

    def test_signature_deterministic_structure(self):
        """Test signature has R||S structure."""
        key = generate_account_key()
        data = b"test data"
        sig = sign_es256(key, data)
        # R and S are each 32 bytes
        r_bytes = sig[:32]
        s_bytes = sig[32:]
        assert len(r_bytes) == 32
        assert len(s_bytes) == 32

    def test_different_data_different_signature(self):
        """Test different data produces different signatures."""
        key = generate_account_key()
        sig1 = sign_es256(key, b"data1")
        sig2 = sign_es256(key, b"data2")
        assert sig1 != sig2

    def test_different_keys_different_signature(self):
        """Test different keys produce different signatures."""
        key1 = generate_account_key()
        key2 = generate_account_key()
        data = b"same data"
        sig1 = sign_es256(key1, data)
        sig2 = sign_es256(key2, data)
        assert sig1 != sig2


class TestGetKeyAuthorization:
    """Tests for get_key_authorization function."""

    def test_format(self):
        """Test key authorization format."""
        result = get_key_authorization("token123", "thumbprint456")
        assert result == "token123.thumbprint456"

    def test_with_actual_values(self):
        """Test with actual generated values."""
        key = generate_account_key()
        jwk = get_jwk(key)
        thumbprint = get_jwk_thumbprint(jwk)
        token = "abcdef123456"
        key_auth = get_key_authorization(token, thumbprint)
        assert key_auth.startswith(token + ".")
        assert thumbprint in key_auth


class TestGetDnsChallengeValue:
    """Tests for get_dns_challenge_value function."""

    def test_returns_base64url(self):
        """Test that value is base64url encoded."""
        result = get_dns_challenge_value("token.thumbprint")
        # Should not contain padding
        assert "=" not in result
        # Should not contain + or /
        assert "+" not in result
        assert "/" not in result

    def test_correct_computation(self):
        """Test DNS challenge value is correctly computed."""
        key_auth = "test-token.test-thumbprint"
        result = get_dns_challenge_value(key_auth)
        # Manually compute expected value
        expected_digest = hashlib.sha256(key_auth.encode("utf-8")).digest()
        expected = base64url_encode(expected_digest)
        assert result == expected

    def test_deterministic(self):
        """Test computation is deterministic."""
        key_auth = "token.thumbprint"
        result1 = get_dns_challenge_value(key_auth)
        result2 = get_dns_challenge_value(key_auth)
        assert result1 == result2

    def test_different_input_different_output(self):
        """Test different inputs produce different outputs."""
        result1 = get_dns_challenge_value("key.auth1")
        result2 = get_dns_challenge_value("key.auth2")
        assert result1 != result2


class TestCryptoIntegration:
    """Integration tests for cryptographic workflow."""

    def test_full_key_authorization_workflow(self):
        """Test complete key authorization workflow."""
        # Generate account key
        key = generate_account_key()

        # Get JWK and thumbprint
        jwk = get_jwk(key)
        thumbprint = get_jwk_thumbprint(jwk)

        # Create key authorization
        token = "challenge-token-abc123"
        key_auth = get_key_authorization(token, thumbprint)

        # Get DNS challenge value
        dns_value = get_dns_challenge_value(key_auth)

        # Verify format
        assert "." in key_auth
        assert token in key_auth
        assert len(dns_value) > 20  # Base64url encoded SHA-256

    def test_csr_signing_workflow(self):
        """Test complete CSR generation workflow."""
        # Generate certificate key
        cert_key = generate_private_key(KeyType.EC256)

        # Create identifiers
        identifiers = [
            Identifier.dns("example.com"),
            Identifier.dns("www.example.com"),
        ]

        # Generate CSR
        csr_der = create_csr(identifiers, cert_key)

        # Verify CSR is valid
        csr = load_der_x509_csr(csr_der)
        assert csr is not None
        assert csr.is_signature_valid

    def test_jws_signing_workflow(self):
        """Test JWS signing workflow."""
        # Generate account key
        key = generate_account_key()

        # Create protected header and payload
        header = b'{"alg":"ES256","nonce":"test-nonce","url":"https://acme.test/resource"}'
        payload = b'{"identifiers":[{"type":"dns","value":"example.com"}]}'

        # Sign
        signing_input = header + b"." + payload
        signature = sign_es256(key, signing_input)

        # Verify signature format
        assert len(signature) == 64
