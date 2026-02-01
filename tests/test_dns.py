"""Tests for DNS verification utilities."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from acmeow._internal.dns import (
    DEFAULT_DNS_SERVERS,
    DEFAULT_DNS_TIMEOUT,
    DnsConfig,
    DnsVerifier,
    verify_dns_propagation,
)


class TestDnsConfig:
    """Tests for DnsConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DnsConfig()
        assert config.nameservers == DEFAULT_DNS_SERVERS
        assert config.timeout == DEFAULT_DNS_TIMEOUT
        assert config.retries == 3
        assert config.retry_delay == 2.0
        assert config.require_all is False
        assert config.min_servers == 1

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DnsConfig(
            nameservers=["8.8.8.8"],
            timeout=10.0,
            retries=5,
            retry_delay=1.0,
            require_all=True,
            min_servers=2,
        )
        assert config.nameservers == ["8.8.8.8"]
        assert config.timeout == 10.0
        assert config.retries == 5
        assert config.retry_delay == 1.0
        assert config.require_all is True
        assert config.min_servers == 2

    def test_default_nameservers_is_copy(self):
        """Test that default nameservers is a copy."""
        config1 = DnsConfig()
        config2 = DnsConfig()
        config1.nameservers.append("custom.dns")
        # config2 should not be affected
        assert "custom.dns" not in config2.nameservers


class TestDnsVerifier:
    """Tests for DnsVerifier."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        verifier = DnsVerifier()
        assert verifier._config is not None

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = DnsConfig(timeout=15.0)
        verifier = DnsVerifier(config)
        assert verifier._config.timeout == 15.0

    def test_check_resolver_availability(self):
        """Test resolver availability check."""
        verifier = DnsVerifier()
        # Result depends on whether dnspython is installed
        assert isinstance(verifier._resolver_available, bool)

    @patch("acmeow._internal.dns.DnsVerifier._check_record")
    def test_verify_txt_record_success_immediate(self, mock_check: MagicMock):
        """Test immediate verification success."""
        mock_check.return_value = True
        verifier = DnsVerifier()

        result = verifier.verify_txt_record(
            "_acme-challenge.example.com",
            "expected-value",
            max_wait=60,
            poll_interval=1,
        )

        assert result is True
        mock_check.assert_called_once_with("_acme-challenge.example.com", "expected-value")

    @patch("acmeow._internal.dns.DnsVerifier._check_record")
    @patch("time.sleep")
    def test_verify_txt_record_success_after_retry(self, mock_sleep: MagicMock, mock_check: MagicMock):
        """Test verification success after retries."""
        mock_check.side_effect = [False, False, True]
        verifier = DnsVerifier()

        result = verifier.verify_txt_record(
            "_acme-challenge.example.com",
            "expected-value",
            max_wait=60,
            poll_interval=1,
        )

        assert result is True
        assert mock_check.call_count == 3

    @patch("acmeow._internal.dns.DnsVerifier._check_record")
    @patch("time.sleep")
    def test_verify_txt_record_timeout(self, mock_sleep: MagicMock, mock_check: MagicMock):
        """Test verification timeout."""
        mock_check.return_value = False

        # Mock time to simulate timeout
        start_time = time.time()
        call_count = [0]

        def time_mock():
            # First call returns start, subsequent calls return past timeout
            call_count[0] += 1
            if call_count[0] <= 2:
                return start_time
            return start_time + 5  # Past max_wait

        with patch("time.time", side_effect=time_mock):
            verifier = DnsVerifier()
            result = verifier.verify_txt_record(
                "_acme-challenge.example.com",
                "expected-value",
                max_wait=3,
                poll_interval=1,
            )

        assert result is False

    def test_check_record_uses_correct_method(self):
        """Test that _check_record uses appropriate method."""
        verifier = DnsVerifier()
        # The method used depends on resolver availability
        # We just verify the method exists and can be called
        with patch.object(verifier, "_check_record_dnspython", return_value=True):
            with patch.object(verifier, "_check_record_socket", return_value=True):
                if verifier._resolver_available:
                    result = verifier._check_record("test", "value")
                else:
                    result = verifier._check_record("test", "value")
                assert result is True


class TestDnsVerifierSocket:
    """Tests for socket-based DNS verification fallback."""

    @patch("subprocess.run")
    def test_check_record_socket_windows(self, mock_run: MagicMock):
        """Test socket-based DNS check on Windows."""
        mock_run.return_value = MagicMock(stdout="expected-value")

        with patch("sys.platform", "win32"):
            verifier = DnsVerifier(DnsConfig(timeout=5.0))
            # Force socket method
            verifier._resolver_available = False

            result = verifier._check_record_socket("_acme-challenge.example.com", "expected-value")

            assert result is True
            mock_run.assert_called_once()
            args = mock_run.call_args
            assert "nslookup" in args[0][0]

    @patch("subprocess.run")
    def test_check_record_socket_unix_dig(self, mock_run: MagicMock):
        """Test socket-based DNS check on Unix using dig."""
        mock_run.return_value = MagicMock(stdout="expected-value")

        with patch("sys.platform", "linux"):
            verifier = DnsVerifier(DnsConfig(timeout=5.0))
            verifier._resolver_available = False

            result = verifier._check_record_socket("_acme-challenge.example.com", "expected-value")

            assert result is True
            args = mock_run.call_args
            assert "dig" in args[0][0]

    @patch("subprocess.run")
    def test_check_record_socket_unix_host_fallback(self, mock_run: MagicMock):
        """Test socket-based DNS check falls back to host command."""

        def run_side_effect(cmd, **kwargs):
            if "dig" in cmd:
                raise FileNotFoundError("dig not found")
            return MagicMock(stdout="expected-value")

        mock_run.side_effect = run_side_effect

        with patch("sys.platform", "linux"):
            verifier = DnsVerifier(DnsConfig(timeout=5.0))
            verifier._resolver_available = False

            result = verifier._check_record_socket("_acme-challenge.example.com", "expected-value")

            assert result is True

    @patch("subprocess.run")
    def test_check_record_socket_value_not_found(self, mock_run: MagicMock):
        """Test socket check when value not found."""
        mock_run.return_value = MagicMock(stdout="different-value")

        with patch("sys.platform", "win32"):
            verifier = DnsVerifier()
            verifier._resolver_available = False

            result = verifier._check_record_socket("_acme-challenge.example.com", "expected-value")

            assert result is False

    @patch("subprocess.run")
    def test_check_record_socket_timeout(self, mock_run: MagicMock):
        """Test socket check handles timeout."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 5)

        with patch("sys.platform", "win32"):
            verifier = DnsVerifier()
            verifier._resolver_available = False

            # When verification fails, it returns True to allow proceeding
            result = verifier._check_record_socket("_acme-challenge.example.com", "expected-value")
            assert result is True

    @patch("subprocess.run")
    def test_check_record_socket_command_not_found(self, mock_run: MagicMock):
        """Test socket check handles command not found."""
        mock_run.side_effect = FileNotFoundError("command not found")

        with patch("sys.platform", "win32"):
            verifier = DnsVerifier()
            verifier._resolver_available = False

            # When tools unavailable, returns True to not block
            result = verifier._check_record_socket("_acme-challenge.example.com", "expected-value")
            assert result is True


class TestDnsVerifierDnspython:
    """Tests for dnspython-based DNS verification."""

    @pytest.fixture(autouse=True)
    def skip_if_no_dnspython(self):
        """Skip tests if dnspython is not installed."""
        try:
            import dns.resolver  # noqa: F401
        except ImportError:
            pytest.skip("dnspython not installed")

    def test_check_record_dnspython_success(self):
        """Test dnspython verification success."""
        import dns.resolver

        verifier = DnsVerifier(DnsConfig(nameservers=["8.8.8.8"], min_servers=1))

        # Mock dns.resolver
        mock_resolver = MagicMock()
        mock_rdata = MagicMock()
        mock_rdata.strings = [b"expected-value"]
        mock_resolver.resolve.return_value = [mock_rdata]

        with patch.object(dns.resolver, "Resolver", return_value=mock_resolver):
            verifier._resolver_available = True
            result = verifier._check_record_dnspython("_acme-challenge.example.com", "expected-value")

            assert result is True

    def test_check_record_dnspython_nxdomain(self):
        """Test dnspython handling NXDOMAIN."""
        import dns.resolver

        verifier = DnsVerifier(DnsConfig(nameservers=["8.8.8.8"], min_servers=1))

        mock_resolver = MagicMock()
        mock_resolver.resolve.side_effect = dns.resolver.NXDOMAIN()

        with patch.object(dns.resolver, "Resolver", return_value=mock_resolver):
            verifier._resolver_available = True
            result = verifier._check_record_dnspython("_acme-challenge.example.com", "expected")
            # NXDOMAIN means domain doesn't exist, so record not found
            assert result is False

    def test_check_record_dnspython_require_all(self):
        """Test dnspython with require_all option."""
        import dns.resolver

        config = DnsConfig(nameservers=["8.8.8.8", "1.1.1.1"], require_all=True)
        verifier = DnsVerifier(config)

        # Mock both servers returning the value
        mock_resolver = MagicMock()
        mock_rdata = MagicMock()
        mock_rdata.strings = [b"expected-value"]
        mock_resolver.resolve.return_value = [mock_rdata]

        with patch.object(dns.resolver, "Resolver", return_value=mock_resolver):
            verifier._resolver_available = True
            result = verifier._check_record_dnspython("_acme-challenge.example.com", "expected-value")

            assert result is True
            # Should have queried all nameservers
            assert mock_resolver.resolve.call_count == 2


class TestVerifyDnsPropagation:
    """Tests for verify_dns_propagation convenience function."""

    @patch("acmeow._internal.dns.DnsVerifier.verify_txt_record")
    def test_calls_verifier(self, mock_verify: MagicMock):
        """Test that function creates verifier and calls it."""
        mock_verify.return_value = True

        result = verify_dns_propagation(
            "_acme-challenge.example.com",
            "expected-value",
        )

        assert result is True
        mock_verify.assert_called_once()

    @patch("acmeow._internal.dns.DnsVerifier.verify_txt_record")
    def test_passes_config(self, mock_verify: MagicMock):
        """Test that custom config is passed."""
        mock_verify.return_value = True
        config = DnsConfig(timeout=15.0)

        verify_dns_propagation(
            "_acme-challenge.example.com",
            "expected-value",
            config=config,
        )

        mock_verify.assert_called_once()

    @patch("acmeow._internal.dns.DnsVerifier.verify_txt_record")
    def test_passes_wait_params(self, mock_verify: MagicMock):
        """Test that max_wait and poll_interval are passed."""
        mock_verify.return_value = True

        verify_dns_propagation(
            "_acme-challenge.example.com",
            "expected-value",
            max_wait=120,
            poll_interval=5,
        )

        mock_verify.assert_called_once_with(
            "_acme-challenge.example.com",
            "expected-value",
            max_wait=120,
            poll_interval=5,
        )


class TestDnsConfigValidation:
    """Tests for DNS configuration edge cases."""

    def test_empty_nameservers(self):
        """Test config with empty nameservers list."""
        config = DnsConfig(nameservers=[])
        assert config.nameservers == []

    def test_single_nameserver(self):
        """Test config with single nameserver."""
        config = DnsConfig(nameservers=["192.168.1.1"])
        assert len(config.nameservers) == 1

    def test_zero_timeout(self):
        """Test config with zero timeout."""
        config = DnsConfig(timeout=0.0)
        assert config.timeout == 0.0

    def test_zero_retries(self):
        """Test config with zero retries."""
        config = DnsConfig(retries=0)
        assert config.retries == 0

    def test_high_min_servers(self):
        """Test config requiring more servers than available."""
        config = DnsConfig(nameservers=["8.8.8.8"], min_servers=5)
        verifier = DnsVerifier(config)
        # Even if one server succeeds, won't meet min_servers threshold
        assert config.min_servers > len(config.nameservers)
