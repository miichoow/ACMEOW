"""Tests for convenience methods."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.x509.oid import NameOID

from acmeow.convenience import (
    BatchResult,
    CertificateInfo,
    check_certificate_expiry,
    get_certificate_info,
    issue_batch,
    quick_issue,
    renew_if_needed,
    LETSENCRYPT_PRODUCTION,
    LETSENCRYPT_STAGING,
)


def generate_test_certificate(
    domain: str = "example.com",
    days_valid: int = 30,
    domains: list[str] | None = None,
) -> bytes:
    """Generate a test certificate for testing."""
    key = ec.generate_private_key(ec.SECP256R1())

    now = datetime.now(timezone.utc)
    not_before = now - timedelta(days=1)
    not_after = now + timedelta(days=days_valid)

    subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, domain),
    ])

    issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "Test CA"),
    ])

    builder = x509.CertificateBuilder()
    builder = builder.subject_name(subject)
    builder = builder.issuer_name(issuer)
    builder = builder.public_key(key.public_key())
    builder = builder.serial_number(x509.random_serial_number())
    builder = builder.not_valid_before(not_before)
    builder = builder.not_valid_after(not_after)

    # Add SAN
    san_domains = domains or [domain]
    san_entries = [x509.DNSName(d) for d in san_domains]
    builder = builder.add_extension(
        x509.SubjectAlternativeName(san_entries),
        critical=False,
    )

    cert = builder.sign(key, hashes.SHA256())
    return cert.public_bytes(serialization.Encoding.PEM)


class TestCertificateInfo:
    """Tests for CertificateInfo dataclass."""

    def test_is_expiring_soon(self) -> None:
        """Test is_expiring_soon property."""
        info = CertificateInfo(
            subject="example.com",
            issuer="Test CA",
            domains=["example.com"],
            serial_number="ABC123",
            not_before=datetime.now(timezone.utc),
            not_after=datetime.now(timezone.utc) + timedelta(days=20),
            days_until_expiry=20,
            is_expired=False,
            key_type="EC",
            key_size="P-256",
            fingerprint_sha256="AA:BB:CC",
        )

        assert info.is_expiring_soon is True

        info2 = CertificateInfo(
            subject="example.com",
            issuer="Test CA",
            domains=["example.com"],
            serial_number="ABC123",
            not_before=datetime.now(timezone.utc),
            not_after=datetime.now(timezone.utc) + timedelta(days=60),
            days_until_expiry=60,
            is_expired=False,
            key_type="EC",
            key_size="P-256",
            fingerprint_sha256="AA:BB:CC",
        )

        assert info2.is_expiring_soon is False

    def test_str(self) -> None:
        """Test string representation."""
        info = CertificateInfo(
            subject="example.com",
            issuer="Test CA",
            domains=["example.com"],
            serial_number="ABC123",
            not_before=datetime.now(timezone.utc),
            not_after=datetime.now(timezone.utc) + timedelta(days=30),
            days_until_expiry=30,
            is_expired=False,
            key_type="EC",
            key_size="P-256",
            fingerprint_sha256="AA:BB:CC",
        )

        s = str(info)
        assert "example.com" in s


class TestGetCertificateInfo:
    """Tests for get_certificate_info function."""

    def test_basic_info(self) -> None:
        """Test extracting basic certificate info."""
        cert_pem = generate_test_certificate("example.com", days_valid=30)
        info = get_certificate_info(cert_pem)

        assert info.subject == "example.com"
        assert info.issuer == "Test CA"
        assert "example.com" in info.domains
        assert info.is_expired is False
        assert info.days_until_expiry > 0

    def test_multiple_domains(self) -> None:
        """Test certificate with multiple domains."""
        domains = ["example.com", "www.example.com", "api.example.com"]
        cert_pem = generate_test_certificate("example.com", domains=domains)
        info = get_certificate_info(cert_pem)

        assert len(info.domains) == 3
        for domain in domains:
            assert domain in info.domains

    def test_expired_certificate(self) -> None:
        """Test detecting expired certificate."""
        # Generate a certificate that expired yesterday
        key = ec.generate_private_key(ec.SECP256R1())
        now = datetime.now(timezone.utc)

        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "expired.com")])
        issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Test CA")])

        builder = x509.CertificateBuilder()
        builder = builder.subject_name(subject)
        builder = builder.issuer_name(issuer)
        builder = builder.public_key(key.public_key())
        builder = builder.serial_number(x509.random_serial_number())
        builder = builder.not_valid_before(now - timedelta(days=30))
        builder = builder.not_valid_after(now - timedelta(days=1))

        cert = builder.sign(key, hashes.SHA256())
        cert_pem = cert.public_bytes(serialization.Encoding.PEM)

        info = get_certificate_info(cert_pem)

        assert info.is_expired is True
        assert info.days_until_expiry < 0

    def test_key_type_detection(self) -> None:
        """Test detecting key type."""
        cert_pem = generate_test_certificate("example.com")
        info = get_certificate_info(cert_pem)

        assert "EC" in info.key_type or "Elliptic" in info.key_type

    def test_fingerprint(self) -> None:
        """Test fingerprint calculation."""
        cert_pem = generate_test_certificate("example.com")
        info = get_certificate_info(cert_pem)

        # SHA-256 fingerprint should be 64 hex chars + colons
        assert len(info.fingerprint_sha256) > 60
        assert ":" in info.fingerprint_sha256

    def test_string_input(self) -> None:
        """Test with string input instead of bytes."""
        cert_pem = generate_test_certificate("example.com")
        cert_str = cert_pem.decode()

        info = get_certificate_info(cert_str)
        assert info.subject == "example.com"


class TestCheckCertificateExpiry:
    """Tests for check_certificate_expiry function."""

    def test_ok_status(self) -> None:
        """Test certificate with OK status."""
        with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as f:
            cert_pem = generate_test_certificate("example.com", days_valid=60)
            f.write(cert_pem)
            f.flush()

            info, status = check_certificate_expiry(f.name)

            assert status == "ok"
            assert info.subject == "example.com"

    def test_warning_status(self) -> None:
        """Test certificate with warning status."""
        with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as f:
            cert_pem = generate_test_certificate("example.com", days_valid=20)
            f.write(cert_pem)
            f.flush()

            info, status = check_certificate_expiry(f.name, warn_days=30)

            assert status == "warning"

    def test_error_status(self) -> None:
        """Test certificate with error status."""
        with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as f:
            cert_pem = generate_test_certificate("example.com", days_valid=5)
            f.write(cert_pem)
            f.flush()

            info, status = check_certificate_expiry(f.name, warn_days=30, error_days=7)

            assert status == "error"

    def test_expired_status(self) -> None:
        """Test expired certificate status."""
        # Create an expired certificate
        key = ec.generate_private_key(ec.SECP256R1())
        now = datetime.now(timezone.utc)

        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "expired.com")])
        issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Test CA")])

        builder = x509.CertificateBuilder()
        builder = builder.subject_name(subject)
        builder = builder.issuer_name(issuer)
        builder = builder.public_key(key.public_key())
        builder = builder.serial_number(x509.random_serial_number())
        builder = builder.not_valid_before(now - timedelta(days=30))
        builder = builder.not_valid_after(now - timedelta(days=1))

        cert = builder.sign(key, hashes.SHA256())
        cert_pem = cert.public_bytes(serialization.Encoding.PEM)

        with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as f:
            f.write(cert_pem)
            f.flush()

            info, status = check_certificate_expiry(f.name)

            assert status == "expired"


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_success_result(self) -> None:
        """Test successful batch result."""
        result = BatchResult(
            domain="example.com",
            success=True,
            cert_pem="-----BEGIN CERTIFICATE-----",
            key_pem="-----BEGIN PRIVATE KEY-----",
        )

        assert result.success is True
        assert result.cert_pem is not None
        assert result.error is None

    def test_failure_result(self) -> None:
        """Test failed batch result."""
        result = BatchResult(
            domain="example.com",
            success=False,
            error="Connection failed",
        )

        assert result.success is False
        assert result.cert_pem is None
        assert result.error == "Connection failed"


class TestQuickIssue:
    """Tests for quick_issue function."""

    def test_quick_issue_basic(self) -> None:
        """Test basic certificate issuance."""
        mock_handler = MagicMock()
        mock_client = MagicMock()
        mock_client.get_certificate.return_value = ("cert-pem", "key-pem")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("acmeow.convenience.AcmeClient", return_value=mock_client):
            cert, key = quick_issue(
                domain="example.com",
                email="admin@example.com",
                handler=mock_handler,
            )

            assert cert == "cert-pem"
            assert key == "key-pem"
            mock_client.create_account.assert_called_once()
            mock_client.create_order.assert_called_once()
            mock_client.complete_challenges.assert_called_once()
            mock_client.finalize_order.assert_called_once()

    def test_quick_issue_with_additional_domains(self) -> None:
        """Test issuance with additional domains."""
        mock_handler = MagicMock()
        mock_client = MagicMock()
        mock_client.get_certificate.return_value = ("cert", "key")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("acmeow.convenience.AcmeClient", return_value=mock_client):
            quick_issue(
                domain="example.com",
                email="admin@example.com",
                handler=mock_handler,
                additional_domains=["www.example.com", "api.example.com"],
            )

            # Should create order with 3 identifiers
            call_args = mock_client.create_order.call_args[0][0]
            assert len(call_args) == 3

    def test_quick_issue_with_parallel(self) -> None:
        """Test issuance with parallel challenges."""
        mock_handler = MagicMock()
        mock_client = MagicMock()
        mock_client.get_certificate.return_value = ("cert", "key")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("acmeow.convenience.AcmeClient", return_value=mock_client):
            quick_issue(
                domain="example.com",
                email="admin@example.com",
                handler=mock_handler,
                parallel=True,
            )

            call_kwargs = mock_client.complete_challenges.call_args[1]
            assert call_kwargs["parallel"] is True

    def test_quick_issue_custom_server(self) -> None:
        """Test issuance with custom server URL."""
        mock_handler = MagicMock()
        mock_client = MagicMock()
        mock_client.get_certificate.return_value = ("cert", "key")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("acmeow.convenience.AcmeClient", return_value=mock_client) as MockClient:
            quick_issue(
                domain="example.com",
                email="admin@example.com",
                handler=mock_handler,
                server_url=LETSENCRYPT_STAGING,
            )

            call_kwargs = MockClient.call_args[1]
            assert call_kwargs["server_url"] == LETSENCRYPT_STAGING


class TestIssueBatch:
    """Tests for issue_batch function."""

    def test_issue_batch_single_domains(self) -> None:
        """Test batch issuance with single domains."""
        mock_handler = MagicMock()
        mock_client = MagicMock()
        mock_client.get_certificate.return_value = ("cert", "key")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("acmeow.convenience.AcmeClient", return_value=mock_client):
            results = issue_batch(
                domains=["site1.com", "site2.com"],
                email="admin@example.com",
                handler=mock_handler,
            )

            assert len(results) == 2
            assert all(r.success for r in results)
            assert results[0].domain == "site1.com"
            assert results[1].domain == "site2.com"

    def test_issue_batch_multi_domain_cert(self) -> None:
        """Test batch issuance with multi-domain certificate."""
        mock_handler = MagicMock()
        mock_client = MagicMock()
        mock_client.get_certificate.return_value = ("cert", "key")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("acmeow.convenience.AcmeClient", return_value=mock_client):
            results = issue_batch(
                domains=[["example.com", "www.example.com"]],
                email="admin@example.com",
                handler=mock_handler,
            )

            assert len(results) == 1
            assert results[0].domain == "example.com"
            # Check identifiers created
            call_args = mock_client.create_order.call_args[0][0]
            assert len(call_args) == 2

    def test_issue_batch_with_error(self) -> None:
        """Test batch issuance handles errors."""
        mock_handler = MagicMock()
        mock_client = MagicMock()
        mock_client.create_order.side_effect = [None, Exception("API error"), None]
        mock_client.get_certificate.return_value = ("cert", "key")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("acmeow.convenience.AcmeClient", return_value=mock_client):
            results = issue_batch(
                domains=["site1.com", "site2.com", "site3.com"],
                email="admin@example.com",
                handler=mock_handler,
                stop_on_error=False,
            )

            assert len(results) == 3
            assert results[0].success is True
            assert results[1].success is False
            assert "API error" in results[1].error
            assert results[2].success is True

    def test_issue_batch_stop_on_error(self) -> None:
        """Test batch issuance stops on error when requested."""
        mock_handler = MagicMock()
        mock_client = MagicMock()
        mock_client.create_order.side_effect = [None, Exception("API error")]
        mock_client.get_certificate.return_value = ("cert", "key")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("acmeow.convenience.AcmeClient", return_value=mock_client):
            results = issue_batch(
                domains=["site1.com", "site2.com", "site3.com"],
                email="admin@example.com",
                handler=mock_handler,
                stop_on_error=True,
            )

            # Should stop after the error
            assert len(results) == 2
            assert results[0].success is True
            assert results[1].success is False


class TestRenewIfNeeded:
    """Tests for renew_if_needed function."""

    def test_renew_not_needed(self) -> None:
        """Test when renewal is not needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cert_path = Path(tmpdir) / "cert.pem"
            key_path = Path(tmpdir) / "key.pem"

            # Create a certificate valid for 60 days
            cert_pem = generate_test_certificate("example.com", days_valid=60)
            cert_path.write_bytes(cert_pem)
            key_path.write_text("fake-key")

            mock_handler = MagicMock()

            renewed, cert, key = renew_if_needed(
                cert_path=cert_path,
                key_path=key_path,
                email="admin@example.com",
                handler=mock_handler,
                days_before_expiry=30,
            )

            assert renewed is False
            assert cert is None
            assert key is None

    def test_renew_needed(self) -> None:
        """Test when renewal is needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cert_path = Path(tmpdir) / "cert.pem"
            key_path = Path(tmpdir) / "key.pem"

            # Create a certificate expiring in 10 days
            cert_pem = generate_test_certificate("example.com", days_valid=10)
            cert_path.write_bytes(cert_pem)
            key_path.write_text("fake-key")

            mock_handler = MagicMock()

            with patch("acmeow.convenience.quick_issue") as mock_quick_issue:
                mock_quick_issue.return_value = ("new-cert", "new-key")

                renewed, cert, key = renew_if_needed(
                    cert_path=cert_path,
                    key_path=key_path,
                    email="admin@example.com",
                    handler=mock_handler,
                    days_before_expiry=30,
                )

                assert renewed is True
                assert cert == "new-cert"
                assert key == "new-key"
                mock_quick_issue.assert_called_once()

    def test_renew_force(self) -> None:
        """Test forced renewal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cert_path = Path(tmpdir) / "cert.pem"
            key_path = Path(tmpdir) / "key.pem"

            # Create a certificate valid for 60 days
            cert_pem = generate_test_certificate("example.com", days_valid=60)
            cert_path.write_bytes(cert_pem)
            key_path.write_text("fake-key")

            mock_handler = MagicMock()

            with patch("acmeow.convenience.quick_issue") as mock_quick_issue:
                mock_quick_issue.return_value = ("new-cert", "new-key")

                renewed, cert, key = renew_if_needed(
                    cert_path=cert_path,
                    key_path=key_path,
                    email="admin@example.com",
                    handler=mock_handler,
                    force=True,
                )

                assert renewed is True
                mock_quick_issue.assert_called_once()

    def test_renew_cert_not_found(self) -> None:
        """Test when certificate file doesn't exist."""
        mock_handler = MagicMock()

        renewed, cert, key = renew_if_needed(
            cert_path="/nonexistent/cert.pem",
            key_path="/nonexistent/key.pem",
            email="admin@example.com",
            handler=mock_handler,
        )

        assert renewed is False
        assert cert is None
        assert key is None

    def test_renew_with_multiple_domains(self) -> None:
        """Test renewal preserves multiple domains."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cert_path = Path(tmpdir) / "cert.pem"
            key_path = Path(tmpdir) / "key.pem"

            # Create a certificate with multiple domains
            domains = ["example.com", "www.example.com", "api.example.com"]
            cert_pem = generate_test_certificate("example.com", days_valid=10, domains=domains)
            cert_path.write_bytes(cert_pem)
            key_path.write_text("fake-key")

            mock_handler = MagicMock()

            with patch("acmeow.convenience.quick_issue") as mock_quick_issue:
                mock_quick_issue.return_value = ("new-cert", "new-key")

                renew_if_needed(
                    cert_path=cert_path,
                    key_path=key_path,
                    email="admin@example.com",
                    handler=mock_handler,
                    days_before_expiry=30,
                )

                call_kwargs = mock_quick_issue.call_args[1]
                # Should pass additional domains
                assert call_kwargs.get("additional_domains") == ["www.example.com", "api.example.com"]


class TestGetCertificateInfoRSA:
    """Tests for get_certificate_info with RSA certificates."""

    def test_rsa_key_detection(self) -> None:
        """Test detecting RSA key type."""
        # Generate RSA certificate
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        now = datetime.now(timezone.utc)

        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "rsa.example.com")])
        issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Test CA")])

        builder = x509.CertificateBuilder()
        builder = builder.subject_name(subject)
        builder = builder.issuer_name(issuer)
        builder = builder.public_key(key.public_key())
        builder = builder.serial_number(x509.random_serial_number())
        builder = builder.not_valid_before(now - timedelta(days=1))
        builder = builder.not_valid_after(now + timedelta(days=30))

        cert = builder.sign(key, hashes.SHA256())
        cert_pem = cert.public_bytes(serialization.Encoding.PEM)

        info = get_certificate_info(cert_pem)

        assert "RSA" in info.key_type
        assert info.key_size == "2048"


class TestConstants:
    """Tests for module constants."""

    def test_letsencrypt_urls(self) -> None:
        """Test Let's Encrypt URL constants."""
        assert "acme-v02.api.letsencrypt.org" in LETSENCRYPT_PRODUCTION
        assert "staging" in LETSENCRYPT_STAGING
