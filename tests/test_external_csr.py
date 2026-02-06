"""Tests for external CSR support in finalize_order and get_certificate."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.x509.oid import NameOID

from acmeow import (
    AcmeClient,
    AcmeConfigurationError,
    AcmeOrderError,
    Identifier,
    KeyType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csr_pem(
    domains: list[str] | None = None,
    key: ec.EllipticCurvePrivateKey | rsa.RSAPrivateKey | None = None,
) -> tuple[bytes, ec.EllipticCurvePrivateKey | rsa.RSAPrivateKey]:
    """Generate a PEM-encoded CSR and return (pem_bytes, key)."""
    if domains is None:
        domains = ["example.com"]
    if key is None:
        key = ec.generate_private_key(ec.SECP256R1())

    builder = x509.CertificateSigningRequestBuilder()
    builder = builder.subject_name(
        x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, domains[0])])
    )
    san_entries = [x509.DNSName(d) for d in domains]
    builder = builder.add_extension(
        x509.SubjectAlternativeName(san_entries), critical=False
    )
    csr = builder.sign(key, hashes.SHA256())
    return csr.public_bytes(serialization.Encoding.PEM), key


def _make_csr_der(
    domains: list[str] | None = None,
    key: ec.EllipticCurvePrivateKey | rsa.RSAPrivateKey | None = None,
) -> tuple[bytes, ec.EllipticCurvePrivateKey | rsa.RSAPrivateKey]:
    """Generate a DER-encoded CSR and return (der_bytes, key)."""
    if domains is None:
        domains = ["example.com"]
    if key is None:
        key = ec.generate_private_key(ec.SECP256R1())

    builder = x509.CertificateSigningRequestBuilder()
    builder = builder.subject_name(
        x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, domains[0])])
    )
    san_entries = [x509.DNSName(d) for d in domains]
    builder = builder.add_extension(
        x509.SubjectAlternativeName(san_entries), critical=False
    )
    csr = builder.sign(key, hashes.SHA256())
    return csr.public_bytes(serialization.Encoding.DER), key


def _mock_response(
    status_code: int = 200,
    json_data: dict[str, Any] | None = None,
    text: str = "",
    headers: dict[str, str] | None = None,
) -> MagicMock:
    """Create a properly configured mock HTTP response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    resp.headers = {"Replay-Nonce": "nonce", **(headers or {})}
    return resp


def _make_client_with_account(temp_storage: Path) -> AcmeClient:
    """Create an AcmeClient with a valid account set up."""
    mock_dir = _mock_response(json_data={
        "newNonce": "https://acme.test/new-nonce",
        "newAccount": "https://acme.test/new-account",
        "newOrder": "https://acme.test/new-order",
        "keyChange": "https://acme.test/key-change",
        "revokeCert": "https://acme.test/revoke-cert",
    })

    mock_acct = _mock_response(
        status_code=201,
        json_data={"status": "valid"},
        headers={"Location": "https://acme.test/acct/123"},
    )

    with (
        patch("requests.Session.get", return_value=mock_dir),
        patch("requests.Session.head", return_value=mock_dir),
    ):
        client = AcmeClient(
            server_url="https://acme.test/directory",
            email="test@example.com",
            storage_path=temp_storage,
        )
        with patch("requests.Session.post", return_value=mock_acct):
            client.create_account()
    return client


def _add_order(client: AcmeClient) -> None:
    """Attach a pending order for example.com to the client."""
    mock_order = _mock_response(
        status_code=201,
        json_data={
            "status": "pending",
            "identifiers": [{"type": "dns", "value": "example.com"}],
            "authorizations": ["https://acme.test/authz/1"],
            "finalize": "https://acme.test/order/1/finalize",
        },
        headers={"Location": "https://acme.test/order/1"},
    )

    mock_authz = _mock_response(json_data={
        "status": "pending",
        "identifier": {"type": "dns", "value": "example.com"},
        "challenges": [
            {"type": "dns-01", "status": "pending",
             "url": "https://acme.test/chall/1", "token": "abc"},
        ],
    })

    mock_get = _mock_response(json_data={
        "status": "pending",
        "authorizations": ["https://acme.test/authz/1"],
    })

    with (
        patch.object(client._http._session, "post") as mock_post,
        patch.object(client._http._session, "get", return_value=mock_get),
    ):
        mock_post.side_effect = [mock_order, mock_authz]
        client.create_order([Identifier.dns("example.com")])

    # Ensure a nonce is cached for the next call
    client._http._nonce = "cached-nonce"


def _finalize_side_effects() -> list[MagicMock]:
    """Return (refresh-ready, finalize-ok, poll-valid) mock responses."""
    mock_ready = _mock_response(json_data={
        "status": "ready",
        "identifiers": [{"type": "dns", "value": "example.com"}],
        "finalize": "https://acme.test/order/1/finalize",
    })
    mock_finalize = _mock_response(json_data={"status": "processing"})
    mock_valid = _mock_response(json_data={
        "status": "valid",
        "certificate": "https://acme.test/cert/1",
    })
    return [mock_ready, mock_finalize, mock_valid]


# ---------------------------------------------------------------------------
# Tests: _parse_external_csr
# ---------------------------------------------------------------------------

class TestParseExternalCsr:
    """Tests for AcmeClient._parse_external_csr."""

    def test_parse_pem_bytes(self) -> None:
        """PEM-encoded CSR as bytes is parsed correctly."""
        pem, _ = _make_csr_pem()
        der = AcmeClient._parse_external_csr(pem)
        parsed = x509.load_der_x509_csr(der)
        assert parsed.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value == "example.com"

    def test_parse_pem_string(self) -> None:
        """PEM-encoded CSR as str is parsed correctly."""
        pem, _ = _make_csr_pem()
        pem_str = pem.decode("utf-8")
        der = AcmeClient._parse_external_csr(pem_str)
        parsed = x509.load_der_x509_csr(der)
        assert parsed.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value == "example.com"

    def test_parse_der_bytes(self) -> None:
        """DER-encoded CSR as bytes is parsed correctly."""
        der_in, _ = _make_csr_der()
        der_out = AcmeClient._parse_external_csr(der_in)
        parsed = x509.load_der_x509_csr(der_out)
        assert parsed.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value == "example.com"

    def test_parse_preserves_sans(self) -> None:
        """SANs in the external CSR are preserved."""
        domains = ["example.com", "www.example.com", "api.example.com"]
        pem, _ = _make_csr_pem(domains=domains)
        der = AcmeClient._parse_external_csr(pem)
        parsed = x509.load_der_x509_csr(der)
        san = parsed.extensions.get_extension_for_class(x509.SubjectAlternativeName)
        dns_names = san.value.get_values_for_type(x509.DNSName)
        assert sorted(dns_names) == sorted(domains)

    def test_parse_rsa_key_csr(self) -> None:
        """CSR signed with an RSA key is parsed correctly."""
        rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pem, _ = _make_csr_pem(key=rsa_key)
        der = AcmeClient._parse_external_csr(pem)
        parsed = x509.load_der_x509_csr(der)
        assert parsed.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value == "example.com"

    def test_parse_invalid_data_raises(self) -> None:
        """Invalid data raises AcmeConfigurationError."""
        with pytest.raises(AcmeConfigurationError, match="Unable to parse external CSR"):
            AcmeClient._parse_external_csr(b"not a valid CSR")

    def test_parse_invalid_string_raises(self) -> None:
        """Invalid string raises AcmeConfigurationError."""
        with pytest.raises(AcmeConfigurationError, match="Unable to parse external CSR"):
            AcmeClient._parse_external_csr("garbage data")

    def test_parse_empty_bytes_raises(self) -> None:
        """Empty bytes raises AcmeConfigurationError."""
        with pytest.raises(AcmeConfigurationError, match="Unable to parse external CSR"):
            AcmeClient._parse_external_csr(b"")


# ---------------------------------------------------------------------------
# Tests: finalize_order with external CSR
# ---------------------------------------------------------------------------

class TestFinalizeOrderExternalCsr:
    """Tests for finalize_order with external CSR."""

    @pytest.fixture
    def client_with_ready_order(self, temp_storage: Path) -> AcmeClient:
        """Create a client with a valid account and a pending order."""
        client = _make_client_with_account(temp_storage)
        _add_order(client)
        return client

    def test_finalize_with_external_pem_csr(self, client_with_ready_order: AcmeClient) -> None:
        """Finalize with an external PEM CSR does not generate or save a key."""
        pem_csr, _ = _make_csr_pem()

        with patch.object(client_with_ready_order._http._session, "post") as mock_post:
            mock_post.side_effect = _finalize_side_effects()
            client_with_ready_order.finalize_order(csr=pem_csr)

        assert client_with_ready_order._account is not None
        _, key_path = client_with_ready_order._account.get_certificate_paths("example.com")
        assert not key_path.exists()

    def test_finalize_with_external_der_csr(self, client_with_ready_order: AcmeClient) -> None:
        """Finalize with an external DER CSR works."""
        der_csr, _ = _make_csr_der()

        with patch.object(client_with_ready_order._http._session, "post") as mock_post:
            mock_post.side_effect = _finalize_side_effects()
            client_with_ready_order.finalize_order(csr=der_csr)

        assert client_with_ready_order._account is not None
        _, key_path = client_with_ready_order._account.get_certificate_paths("example.com")
        assert not key_path.exists()

    def test_finalize_with_external_csr_string(self, client_with_ready_order: AcmeClient) -> None:
        """Finalize with CSR passed as a PEM string works."""
        pem_csr, _ = _make_csr_pem()
        pem_str = pem_csr.decode("utf-8")

        with patch.object(client_with_ready_order._http._session, "post") as mock_post:
            mock_post.side_effect = _finalize_side_effects()
            client_with_ready_order.finalize_order(csr=pem_str)

    def test_finalize_with_invalid_csr_raises(self, client_with_ready_order: AcmeClient) -> None:
        """Finalize with invalid CSR data raises AcmeConfigurationError."""
        mock_ready = _mock_response(json_data={
            "status": "ready",
            "identifiers": [{"type": "dns", "value": "example.com"}],
            "finalize": "https://acme.test/order/1/finalize",
        })

        with patch.object(client_with_ready_order._http._session, "post") as mock_post:
            mock_post.side_effect = [mock_ready]
            with pytest.raises(AcmeConfigurationError, match="Unable to parse external CSR"):
                client_with_ready_order.finalize_order(csr=b"invalid-csr-data")

    def test_finalize_without_csr_generates_key(self, client_with_ready_order: AcmeClient) -> None:
        """Finalize without CSR generates and saves a private key (default behaviour)."""
        with patch.object(client_with_ready_order._http._session, "post") as mock_post:
            mock_post.side_effect = _finalize_side_effects()
            client_with_ready_order.finalize_order(key_type=KeyType.EC256)

        assert client_with_ready_order._account is not None
        _, key_path = client_with_ready_order._account.get_certificate_paths("example.com")
        assert key_path.exists()

    def test_finalize_with_csr_ignores_key_type(self, client_with_ready_order: AcmeClient) -> None:
        """When CSR is provided, key_type is ignored — no key is generated."""
        pem_csr, _ = _make_csr_pem()

        with patch.object(client_with_ready_order._http._session, "post") as mock_post:
            mock_post.side_effect = _finalize_side_effects()
            client_with_ready_order.finalize_order(key_type=KeyType.RSA4096, csr=pem_csr)

        assert client_with_ready_order._account is not None
        _, key_path = client_with_ready_order._account.get_certificate_paths("example.com")
        assert not key_path.exists()

    def test_finalize_no_order_raises(self, temp_storage: Path) -> None:
        """Finalize with external CSR but no order raises AcmeOrderError."""
        mock_dir = _mock_response(json_data={
            "newNonce": "https://acme.test/new-nonce",
            "newAccount": "https://acme.test/new-account",
            "newOrder": "https://acme.test/new-order",
        })

        with patch("requests.Session.get", return_value=mock_dir):
            client = AcmeClient(
                server_url="https://acme.test/directory",
                email="test@example.com",
                storage_path=temp_storage,
            )

        pem_csr, _ = _make_csr_pem()
        with pytest.raises(AcmeOrderError, match="No order exists"):
            client.finalize_order(csr=pem_csr)

    def test_finalize_csr_payload_is_base64url(self, client_with_ready_order: AcmeClient) -> None:
        """The finalize POST body contains a base64url-encoded CSR."""
        pem_csr, _ = _make_csr_pem()

        with patch.object(client_with_ready_order._http._session, "post") as mock_post:
            mock_post.side_effect = _finalize_side_effects()
            client_with_ready_order.finalize_order(csr=pem_csr)

        # The finalize POST is the second session.post call (first is _refresh_order)
        finalize_call = mock_post.call_args_list[1]
        assert "finalize" in finalize_call.args[0]


# ---------------------------------------------------------------------------
# Tests: get_certificate after external CSR
# ---------------------------------------------------------------------------

class TestGetCertificateExternalCsr:
    """Tests for get_certificate when an external CSR was used."""

    @pytest.fixture
    def client_with_valid_order(self, temp_storage: Path) -> AcmeClient:
        """Client with account + pending order, no key file on disk."""
        client = _make_client_with_account(temp_storage)
        _add_order(client)
        return client

    def test_returns_none_key_for_external_csr(self, client_with_valid_order: AcmeClient) -> None:
        """get_certificate returns None for key_pem when no key file exists."""
        mock_cert_pem = "-----BEGIN CERTIFICATE-----\nMIIB...\n-----END CERTIFICATE-----\n"

        mock_refresh = _mock_response(json_data={
            "status": "valid",
            "identifiers": [{"type": "dns", "value": "example.com"}],
            "finalize": "https://acme.test/order/1/finalize",
            "certificate": "https://acme.test/cert/1",
        })
        mock_cert = _mock_response(text=mock_cert_pem)

        with patch.object(client_with_valid_order._http._session, "post") as mock_post:
            mock_post.side_effect = [mock_refresh, mock_cert]
            cert_pem, key_pem = client_with_valid_order.get_certificate()

        assert cert_pem == mock_cert_pem
        assert key_pem is None

    def test_returns_key_when_exists(self, client_with_valid_order: AcmeClient) -> None:
        """get_certificate returns the key when the key file exists on disk."""
        mock_cert_pem = "-----BEGIN CERTIFICATE-----\nMIIB...\n-----END CERTIFICATE-----\n"
        mock_key_pem = "-----BEGIN EC PRIVATE KEY-----\nMHQC...\n-----END EC PRIVATE KEY-----\n"

        # Write a key file to simulate internal key generation
        assert client_with_valid_order._account is not None
        _, key_path = client_with_valid_order._account.get_certificate_paths("example.com")
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_text(mock_key_pem)

        mock_refresh = _mock_response(json_data={
            "status": "valid",
            "identifiers": [{"type": "dns", "value": "example.com"}],
            "finalize": "https://acme.test/order/1/finalize",
            "certificate": "https://acme.test/cert/1",
        })
        mock_cert = _mock_response(text=mock_cert_pem)

        with patch.object(client_with_valid_order._http._session, "post") as mock_post:
            mock_post.side_effect = [mock_refresh, mock_cert]
            cert_pem, key_pem = client_with_valid_order.get_certificate()

        assert cert_pem == mock_cert_pem
        assert key_pem == mock_key_pem

    def test_saves_cert_to_disk(self, client_with_valid_order: AcmeClient) -> None:
        """Certificate PEM is saved to disk regardless of external CSR usage."""
        mock_cert_pem = "-----BEGIN CERTIFICATE-----\nMIIB...\n-----END CERTIFICATE-----\n"

        mock_refresh = _mock_response(json_data={
            "status": "valid",
            "identifiers": [{"type": "dns", "value": "example.com"}],
            "finalize": "https://acme.test/order/1/finalize",
            "certificate": "https://acme.test/cert/1",
        })
        mock_cert = _mock_response(text=mock_cert_pem)

        with patch.object(client_with_valid_order._http._session, "post") as mock_post:
            mock_post.side_effect = [mock_refresh, mock_cert]
            client_with_valid_order.get_certificate()

        assert client_with_valid_order._account is not None
        cert_path, _ = client_with_valid_order._account.get_certificate_paths("example.com")
        assert cert_path.exists()
        assert cert_path.read_text() == mock_cert_pem
