"""Tests for TLS-ALPN-01 challenge support."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import ec

from acmeow.enums import ChallengeType
from acmeow.handlers.tls_alpn import (
    ACME_IDENTIFIER_OID,
    CallbackTlsAlpnHandler,
    FileTlsAlpnHandler,
    generate_tls_alpn_certificate,
    validate_tls_alpn_certificate,
)


class TestGenerateTlsAlpnCertificate:
    """Tests for generate_tls_alpn_certificate function."""

    def test_basic_generation(self) -> None:
        """Test basic certificate generation."""
        cert_pem, key_pem = generate_tls_alpn_certificate(
            domain="example.com",
            key_authorization="test-token.thumbprint",
        )

        assert b"-----BEGIN CERTIFICATE-----" in cert_pem
        assert b"-----END CERTIFICATE-----" in cert_pem
        assert b"-----BEGIN PRIVATE KEY-----" in key_pem
        assert b"-----END PRIVATE KEY-----" in key_pem

    def test_certificate_has_acme_extension(self) -> None:
        """Test that certificate has the acmeIdentifier extension."""
        cert_pem, _ = generate_tls_alpn_certificate(
            domain="example.com",
            key_authorization="test.auth",
        )

        cert = x509.load_pem_x509_certificate(cert_pem)

        # Find the acmeIdentifier extension
        acme_ext = cert.extensions.get_extension_for_oid(ACME_IDENTIFIER_OID)

        assert acme_ext is not None
        assert acme_ext.critical is True

    def test_certificate_has_san(self) -> None:
        """Test that certificate has Subject Alternative Name."""
        domain = "test.example.com"
        cert_pem, _ = generate_tls_alpn_certificate(
            domain=domain,
            key_authorization="test.auth",
        )

        cert = x509.load_pem_x509_certificate(cert_pem)
        san = cert.extensions.get_extension_for_oid(
            x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME
        )

        # Handle both old and new cryptography versions
        dns_names = []
        for name in san.value.get_values_for_type(x509.DNSName):
            if hasattr(name, 'value'):
                dns_names.append(name.value)
            else:
                dns_names.append(str(name))
        assert domain in dns_names

    def test_certificate_has_correct_cn(self) -> None:
        """Test that certificate has correct Common Name."""
        domain = "cn.example.com"
        cert_pem, _ = generate_tls_alpn_certificate(
            domain=domain,
            key_authorization="test.auth",
        )

        cert = x509.load_pem_x509_certificate(cert_pem)
        cn_attrs = cert.subject.get_attributes_for_oid(x509.oid.NameOID.COMMON_NAME)

        assert len(cn_attrs) == 1
        assert cn_attrs[0].value == domain

    def test_custom_key(self) -> None:
        """Test using a custom key."""
        custom_key = ec.generate_private_key(ec.SECP256R1())

        cert_pem, key_pem = generate_tls_alpn_certificate(
            domain="example.com",
            key_authorization="test.auth",
            key=custom_key,
        )

        # Should still work
        cert = x509.load_pem_x509_certificate(cert_pem)
        assert cert is not None

    def test_validity_period(self) -> None:
        """Test certificate validity period."""
        cert_pem, _ = generate_tls_alpn_certificate(
            domain="example.com",
            key_authorization="test.auth",
            validity_days=7,
        )

        cert = x509.load_pem_x509_certificate(cert_pem)

        # Certificate should be valid now
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        assert cert.not_valid_before_utc <= now
        assert cert.not_valid_after_utc > now


class TestValidateTlsAlpnCertificate:
    """Tests for validate_tls_alpn_certificate function."""

    def test_valid_certificate(self) -> None:
        """Test validating a correct certificate."""
        domain = "example.com"
        key_auth = "token.thumbprint"

        cert_pem, _ = generate_tls_alpn_certificate(domain, key_auth)

        assert validate_tls_alpn_certificate(cert_pem, domain, key_auth) is True

    def test_wrong_domain(self) -> None:
        """Test validation fails with wrong domain."""
        cert_pem, _ = generate_tls_alpn_certificate("example.com", "test.auth")

        assert validate_tls_alpn_certificate(cert_pem, "other.com", "test.auth") is False

    def test_wrong_key_auth(self) -> None:
        """Test validation fails with wrong key authorization."""
        cert_pem, _ = generate_tls_alpn_certificate("example.com", "correct.auth")

        assert validate_tls_alpn_certificate(cert_pem, "example.com", "wrong.auth") is False

    def test_invalid_certificate(self) -> None:
        """Test validation fails with invalid certificate."""
        result = validate_tls_alpn_certificate(
            b"not a certificate",
            "example.com",
            "test.auth",
        )
        assert result is False


class TestCallbackTlsAlpnHandler:
    """Tests for CallbackTlsAlpnHandler class."""

    def test_setup_calls_deploy(self) -> None:
        """Test that setup calls the deploy callback."""
        deploy = MagicMock()
        cleanup = MagicMock()
        handler = CallbackTlsAlpnHandler(deploy, cleanup)

        handler.setup("example.com", "token123", "token.thumbprint")

        deploy.assert_called_once()
        args = deploy.call_args[0]
        assert args[0] == "example.com"
        assert b"-----BEGIN CERTIFICATE-----" in args[1]
        assert b"-----BEGIN PRIVATE KEY-----" in args[2]

    def test_cleanup_calls_callback(self) -> None:
        """Test that cleanup calls the cleanup callback."""
        deploy = MagicMock()
        cleanup = MagicMock()
        handler = CallbackTlsAlpnHandler(deploy, cleanup)

        handler.cleanup("example.com", "token123")

        cleanup.assert_called_once_with("example.com")

    def test_cleanup_handles_exception(self) -> None:
        """Test that cleanup handles exceptions gracefully."""
        deploy = MagicMock()
        cleanup = MagicMock(side_effect=RuntimeError("Cleanup failed"))
        handler = CallbackTlsAlpnHandler(deploy, cleanup)

        # Should not raise
        handler.cleanup("example.com", "token")


class TestFileTlsAlpnHandler:
    """Tests for FileTlsAlpnHandler class."""

    def test_setup_writes_files(self) -> None:
        """Test that setup writes certificate files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cert_dir = Path(tmpdir)
            handler = FileTlsAlpnHandler(cert_dir)

            handler.setup("example.com", "token", "key.auth")

            cert_path = cert_dir / "example.com.alpn.crt"
            key_path = cert_dir / "example.com.alpn.key"

            assert cert_path.exists()
            assert key_path.exists()
            assert b"-----BEGIN CERTIFICATE-----" in cert_path.read_bytes()
            assert b"-----BEGIN PRIVATE KEY-----" in key_path.read_bytes()

    def test_cleanup_removes_files(self) -> None:
        """Test that cleanup removes certificate files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cert_dir = Path(tmpdir)
            handler = FileTlsAlpnHandler(cert_dir)

            handler.setup("example.com", "token", "key.auth")
            handler.cleanup("example.com", "token")

            cert_path = cert_dir / "example.com.alpn.crt"
            key_path = cert_dir / "example.com.alpn.key"

            assert not cert_path.exists()
            assert not key_path.exists()

    def test_custom_patterns(self) -> None:
        """Test custom filename patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cert_dir = Path(tmpdir)
            handler = FileTlsAlpnHandler(
                cert_dir,
                cert_pattern="{domain}-validation.pem",
                key_pattern="{domain}-validation.key",
            )

            handler.setup("example.com", "token", "key.auth")

            assert (cert_dir / "example.com-validation.pem").exists()
            assert (cert_dir / "example.com-validation.key").exists()

    def test_wildcard_domain(self) -> None:
        """Test handling of wildcard domains."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cert_dir = Path(tmpdir)
            handler = FileTlsAlpnHandler(cert_dir)

            handler.setup("*.example.com", "token", "key.auth")

            # Wildcard should be replaced
            cert_path = cert_dir / "_wildcard.example.com.alpn.crt"
            assert cert_path.exists()

    def test_reload_callback(self) -> None:
        """Test reload callback is called."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cert_dir = Path(tmpdir)
            reload_cb = MagicMock()
            handler = FileTlsAlpnHandler(cert_dir, reload_callback=reload_cb)

            handler.setup("example.com", "token", "key.auth")

            reload_cb.assert_called_once()

    def test_reload_callback_on_cleanup(self) -> None:
        """Test reload callback is called on cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cert_dir = Path(tmpdir)
            reload_cb = MagicMock()
            handler = FileTlsAlpnHandler(cert_dir, reload_callback=reload_cb)

            handler.setup("example.com", "token", "key.auth")
            reload_cb.reset_mock()

            handler.cleanup("example.com", "token")

            reload_cb.assert_called_once()


class TestChallengeTypeEnum:
    """Tests for TLS_ALPN challenge type in enum."""

    def test_tls_alpn_value(self) -> None:
        """Test TLS_ALPN enum value."""
        assert ChallengeType.TLS_ALPN.value == "tls-alpn-01"

    def test_tls_alpn_is_string(self) -> None:
        """Test TLS_ALPN can be used as string."""
        assert str(ChallengeType.TLS_ALPN) == "tls-alpn-01"
