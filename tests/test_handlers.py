"""Tests for challenge handlers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, call


from acmeow._internal.encoding import base64url_encode
from acmeow.handlers.dns import CallbackDnsHandler
from acmeow.handlers.http import CallbackHttpHandler, FileHttpHandler


class TestCallbackDnsHandler:
    """Tests for CallbackDnsHandler."""

    def test_setup_calls_create_record(self):
        """Test that setup calls the create_record callback."""
        create_mock = MagicMock()
        delete_mock = MagicMock()
        handler = CallbackDnsHandler(create_mock, delete_mock)

        handler.setup("example.com", "test-token", "key.auth")

        create_mock.assert_called_once()
        args = create_mock.call_args[0]
        assert args[0] == "example.com"
        assert args[1] == "_acme-challenge.example.com"
        # Third arg is the hashed key authorization
        assert len(args[2]) > 0

    def test_setup_computes_correct_record_value(self):
        """Test that record value is correctly computed."""
        create_mock = MagicMock()
        delete_mock = MagicMock()
        handler = CallbackDnsHandler(create_mock, delete_mock)

        key_auth = "test-token.thumbprint123"
        handler.setup("example.com", "test-token", key_auth)

        # Verify the record value is base64url SHA-256
        expected_digest = hashlib.sha256(key_auth.encode("utf-8")).digest()
        expected_value = base64url_encode(expected_digest)

        args = create_mock.call_args[0]
        assert args[2] == expected_value

    def test_cleanup_calls_delete_record(self):
        """Test that cleanup calls the delete_record callback."""
        create_mock = MagicMock()
        delete_mock = MagicMock()
        handler = CallbackDnsHandler(create_mock, delete_mock)

        handler.cleanup("example.com", "test-token")

        delete_mock.assert_called_once_with("example.com", "_acme-challenge.example.com")

    def test_cleanup_handles_exception(self):
        """Test that cleanup doesn't raise if delete fails."""
        create_mock = MagicMock()
        delete_mock = MagicMock(side_effect=Exception("Delete failed"))
        handler = CallbackDnsHandler(create_mock, delete_mock)

        # Should not raise
        handler.cleanup("example.com", "test-token")

    def test_propagation_delay_default(self):
        """Test default propagation delay."""
        handler = CallbackDnsHandler(MagicMock(), MagicMock())
        assert handler.propagation_delay == 60

    def test_propagation_delay_custom(self):
        """Test custom propagation delay."""
        handler = CallbackDnsHandler(MagicMock(), MagicMock(), propagation_delay=120)
        assert handler.propagation_delay == 120

    def test_get_record_name_simple(self):
        """Test record name for simple domain."""
        assert CallbackDnsHandler._get_record_name("example.com") == "_acme-challenge.example.com"

    def test_get_record_name_subdomain(self):
        """Test record name for subdomain."""
        assert CallbackDnsHandler._get_record_name("www.example.com") == "_acme-challenge.www.example.com"

    def test_compute_record_value_deterministic(self):
        """Test that record value computation is deterministic."""
        value1 = CallbackDnsHandler._compute_record_value("key.auth")
        value2 = CallbackDnsHandler._compute_record_value("key.auth")
        assert value1 == value2

    def test_compute_record_value_different_input(self):
        """Test that different inputs produce different values."""
        value1 = CallbackDnsHandler._compute_record_value("key.auth1")
        value2 = CallbackDnsHandler._compute_record_value("key.auth2")
        assert value1 != value2


class TestFileHttpHandler:
    """Tests for FileHttpHandler."""

    def test_setup_creates_challenge_file(self, tmp_path: Path):
        """Test that setup creates the challenge file."""
        handler = FileHttpHandler(tmp_path)

        handler.setup("example.com", "test-token", "key.authorization")

        challenge_file = tmp_path / ".well-known" / "acme-challenge" / "test-token"
        assert challenge_file.exists()
        assert challenge_file.read_text() == "key.authorization"

    def test_setup_creates_directory(self, tmp_path: Path):
        """Test that setup creates the directory structure."""
        handler = FileHttpHandler(tmp_path)

        # Directory shouldn't exist yet
        challenge_dir = tmp_path / ".well-known" / "acme-challenge"
        assert not challenge_dir.exists()

        handler.setup("example.com", "token1", "content1")

        assert challenge_dir.exists()
        assert challenge_dir.is_dir()

    def test_cleanup_removes_file(self, tmp_path: Path):
        """Test that cleanup removes the challenge file."""
        handler = FileHttpHandler(tmp_path)

        handler.setup("example.com", "test-token", "key.authorization")
        challenge_file = tmp_path / ".well-known" / "acme-challenge" / "test-token"
        assert challenge_file.exists()

        handler.cleanup("example.com", "test-token")
        assert not challenge_file.exists()

    def test_cleanup_nonexistent_file(self, tmp_path: Path):
        """Test that cleanup handles nonexistent file gracefully."""
        handler = FileHttpHandler(tmp_path)

        # Should not raise even if file doesn't exist
        handler.cleanup("example.com", "nonexistent-token")

    def test_challenge_dir_property(self, tmp_path: Path):
        """Test challenge_dir property."""
        handler = FileHttpHandler(tmp_path)
        expected = tmp_path / ".well-known" / "acme-challenge"
        assert handler.challenge_dir == expected

    def test_multiple_challenges(self, tmp_path: Path):
        """Test handling multiple challenges."""
        handler = FileHttpHandler(tmp_path)

        handler.setup("example.com", "token1", "auth1")
        handler.setup("www.example.com", "token2", "auth2")
        handler.setup("api.example.com", "token3", "auth3")

        challenge_dir = tmp_path / ".well-known" / "acme-challenge"
        assert (challenge_dir / "token1").read_text() == "auth1"
        assert (challenge_dir / "token2").read_text() == "auth2"
        assert (challenge_dir / "token3").read_text() == "auth3"

    def test_overwrite_existing_file(self, tmp_path: Path):
        """Test that setup overwrites existing challenge file."""
        handler = FileHttpHandler(tmp_path)

        handler.setup("example.com", "token", "old-content")
        handler.setup("example.com", "token", "new-content")

        challenge_file = tmp_path / ".well-known" / "acme-challenge" / "token"
        assert challenge_file.read_text() == "new-content"


class TestCallbackHttpHandler:
    """Tests for CallbackHttpHandler."""

    def test_setup_calls_callback(self):
        """Test that setup calls the setup callback."""
        setup_mock = MagicMock()
        cleanup_mock = MagicMock()
        handler = CallbackHttpHandler(setup_mock, cleanup_mock)

        handler.setup("example.com", "test-token", "key.auth")

        setup_mock.assert_called_once_with("example.com", "test-token", "key.auth")

    def test_cleanup_calls_callback(self):
        """Test that cleanup calls the cleanup callback."""
        setup_mock = MagicMock()
        cleanup_mock = MagicMock()
        handler = CallbackHttpHandler(setup_mock, cleanup_mock)

        handler.cleanup("example.com", "test-token")

        cleanup_mock.assert_called_once_with("example.com", "test-token")

    def test_cleanup_handles_exception(self):
        """Test that cleanup doesn't raise if callback fails."""
        setup_mock = MagicMock()
        cleanup_mock = MagicMock(side_effect=Exception("Cleanup failed"))
        handler = CallbackHttpHandler(setup_mock, cleanup_mock)

        # Should not raise
        handler.cleanup("example.com", "test-token")

    def test_multiple_setups(self):
        """Test multiple setup calls."""
        setup_mock = MagicMock()
        cleanup_mock = MagicMock()
        handler = CallbackHttpHandler(setup_mock, cleanup_mock)

        handler.setup("example.com", "t1", "a1")
        handler.setup("www.example.com", "t2", "a2")

        assert setup_mock.call_count == 2
        setup_mock.assert_has_calls([
            call("example.com", "t1", "a1"),
            call("www.example.com", "t2", "a2"),
        ])

    def test_setup_with_complex_token(self):
        """Test setup with complex token containing special characters."""
        setup_mock = MagicMock()
        cleanup_mock = MagicMock()
        handler = CallbackHttpHandler(setup_mock, cleanup_mock)

        complex_token = "abc123_-XYZ789"
        handler.setup("example.com", complex_token, "auth")

        setup_mock.assert_called_once_with("example.com", complex_token, "auth")


class TestHandlerIntegration:
    """Integration tests for handler behavior."""

    def test_dns_handler_full_workflow(self):
        """Test full DNS handler workflow."""
        records: dict[str, str] = {}

        def create_record(domain: str, name: str, value: str) -> None:
            records[name] = value

        def delete_record(domain: str, name: str) -> None:
            records.pop(name, None)

        handler = CallbackDnsHandler(create_record, delete_record)

        # Setup challenge
        handler.setup("example.com", "token", "key.auth")
        assert "_acme-challenge.example.com" in records

        # Cleanup
        handler.cleanup("example.com", "token")
        assert "_acme-challenge.example.com" not in records

    def test_http_handler_full_workflow(self, tmp_path: Path):
        """Test full HTTP handler workflow."""
        handler = FileHttpHandler(tmp_path)

        # Setup multiple challenges
        domains = ["example.com", "www.example.com", "api.example.com"]
        for i, domain in enumerate(domains):
            handler.setup(domain, f"token{i}", f"auth{i}")

        # Verify all files exist
        challenge_dir = handler.challenge_dir
        for i, _ in enumerate(domains):
            assert (challenge_dir / f"token{i}").exists()

        # Cleanup all
        for i, domain in enumerate(domains):
            handler.cleanup(domain, f"token{i}")

        # Verify all files removed
        for i, _ in enumerate(domains):
            assert not (challenge_dir / f"token{i}").exists()

    def test_handler_cleanup_after_failure(self, tmp_path: Path):
        """Test that cleanup works after partial failures."""
        handler = FileHttpHandler(tmp_path)

        # Setup
        handler.setup("example.com", "token", "auth")

        # Simulate failure by deleting file externally
        challenge_file = handler.challenge_dir / "token"
        challenge_file.unlink()

        # Cleanup should still work without raising
        handler.cleanup("example.com", "token")
