"""Tests for the HTTP client with retry functionality."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from acmeow import (
    AcmeNetworkError,
    AcmeRateLimitError,
    AcmeServerError,
    RetryConfig,
)
from acmeow._internal.http import AcmeHttpClient, RETRYABLE_STATUS_CODES


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RetryConfig()
        assert config.max_retries == 5
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.multiplier == 2.0
        assert config.jitter is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RetryConfig(
            max_retries=3,
            initial_delay=0.5,
            max_delay=30.0,
            multiplier=1.5,
            jitter=False,
        )
        assert config.max_retries == 3
        assert config.initial_delay == 0.5
        assert config.max_delay == 30.0
        assert config.multiplier == 1.5
        assert config.jitter is False

    def test_get_delay_exponential(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(
            initial_delay=1.0,
            multiplier=2.0,
            max_delay=100.0,
            jitter=False,
        )
        assert config.get_delay(0) == 1.0
        assert config.get_delay(1) == 2.0
        assert config.get_delay(2) == 4.0
        assert config.get_delay(3) == 8.0

    def test_get_delay_max_cap(self):
        """Test that delay is capped at max_delay."""
        config = RetryConfig(
            initial_delay=1.0,
            multiplier=10.0,
            max_delay=5.0,
            jitter=False,
        )
        assert config.get_delay(0) == 1.0
        assert config.get_delay(1) == 5.0  # Capped
        assert config.get_delay(2) == 5.0  # Capped

    def test_get_delay_with_jitter(self):
        """Test that jitter adds randomness."""
        config = RetryConfig(
            initial_delay=1.0,
            multiplier=2.0,
            jitter=True,
        )
        # With jitter, delay should be between 0.75 and 1.25 of base
        delays = [config.get_delay(0) for _ in range(100)]
        assert min(delays) >= 0.75
        assert max(delays) <= 1.25
        # Should have some variance
        assert len(set(delays)) > 1


class TestAcmeHttpClient:
    """Tests for AcmeHttpClient."""

    @pytest.fixture
    def client(self):
        """Create an HTTP client for testing."""
        return AcmeHttpClient(
            verify_ssl=True,
            timeout=30,
            retry_config=RetryConfig(max_retries=2, initial_delay=0.01, jitter=False),
        )

    def test_init(self, client: AcmeHttpClient):
        """Test client initialization."""
        assert client._timeout == 30
        assert client._retry_config.max_retries == 2

    def test_set_nonce_url(self, client: AcmeHttpClient):
        """Test setting nonce URL."""
        client.set_nonce_url("https://acme.test/new-nonce")
        assert client._new_nonce_url == "https://acme.test/new-nonce"


class TestRetryBehavior:
    """Tests for retry behavior."""

    def test_get_retries_on_503(self):
        """Test that GET retries on 503 status."""
        client = AcmeHttpClient(
            retry_config=RetryConfig(max_retries=2, initial_delay=0.01, jitter=False)
        )
        client.set_nonce_url("https://acme.test/new-nonce")

        responses = [
            MagicMock(status_code=503, headers={"Replay-Nonce": "n1"}),
            MagicMock(status_code=503, headers={"Replay-Nonce": "n2"}),
            MagicMock(status_code=200, headers={"Replay-Nonce": "n3"}),
        ]

        with patch.object(client._session, "get", side_effect=responses):
            response = client.get("https://acme.test/resource")
            assert response.status_code == 200

    def test_get_retries_on_connection_error(self):
        """Test that GET retries on connection error."""
        client = AcmeHttpClient(
            retry_config=RetryConfig(max_retries=2, initial_delay=0.01, jitter=False)
        )
        client.set_nonce_url("https://acme.test/new-nonce")

        success_response = MagicMock(status_code=200, headers={"Replay-Nonce": "n1"})

        with patch.object(client._session, "get") as mock_get:
            mock_get.side_effect = [
                requests.ConnectionError("Connection failed"),
                success_response,
            ]
            response = client.get("https://acme.test/resource")
            assert response.status_code == 200

    def test_get_exhausts_retries(self):
        """Test that GET raises after exhausting retries."""
        client = AcmeHttpClient(
            retry_config=RetryConfig(max_retries=2, initial_delay=0.01, jitter=False)
        )
        client.set_nonce_url("https://acme.test/new-nonce")

        with patch.object(client._session, "get") as mock_get:
            mock_get.side_effect = requests.ConnectionError("Connection failed")
            with pytest.raises(AcmeNetworkError, match="after 3 attempts"):
                client.get("https://acme.test/resource")
            assert mock_get.call_count == 3

    def test_post_retries_on_429(self):
        """Test that POST retries on rate limit."""
        client = AcmeHttpClient(
            retry_config=RetryConfig(max_retries=2, initial_delay=0.01, jitter=False)
        )
        client.set_nonce_url("https://acme.test/new-nonce")

        # Setup mock key
        from acmeow._internal.crypto import generate_account_key

        key = generate_account_key()

        nonce_response = MagicMock(headers={"Replay-Nonce": "nonce"})
        rate_limit_response = MagicMock(
            status_code=429,
            headers={"Replay-Nonce": "nonce", "Retry-After": "1"},
        )
        rate_limit_response.json.return_value = {
            "type": "urn:ietf:params:acme:error:rateLimited",
            "detail": "Rate limited",
        }
        success_response = MagicMock(
            status_code=200,
            headers={"Replay-Nonce": "nonce"},
        )

        with patch.object(client._session, "head", return_value=nonce_response):
            with patch.object(client._session, "post") as mock_post:
                mock_post.side_effect = [
                    rate_limit_response,
                    rate_limit_response,
                    success_response,
                ]
                response = client.post(
                    "https://acme.test/resource",
                    {"data": "test"},
                    key,
                    kid="https://acme.test/acct/1",
                )
                assert response.status_code == 200

    def test_post_respects_retry_after(self):
        """Test that POST respects Retry-After header."""
        client = AcmeHttpClient(
            retry_config=RetryConfig(max_retries=1, initial_delay=0.01, jitter=False)
        )
        client.set_nonce_url("https://acme.test/new-nonce")

        from acmeow._internal.crypto import generate_account_key

        key = generate_account_key()

        nonce_response = MagicMock(headers={"Replay-Nonce": "nonce"})
        rate_limit_response = MagicMock(
            status_code=429,
            headers={"Replay-Nonce": "nonce", "Retry-After": "0.05"},
        )
        success_response = MagicMock(
            status_code=200,
            headers={"Replay-Nonce": "nonce"},
        )

        with patch.object(client._session, "head", return_value=nonce_response):
            with patch.object(client._session, "post") as mock_post:
                mock_post.side_effect = [rate_limit_response, success_response]

                start = time.time()
                response = client.post(
                    "https://acme.test/resource",
                    {"data": "test"},
                    key,
                    kid="https://acme.test/acct/1",
                )
                elapsed = time.time() - start

                # Should have waited at least the Retry-After time
                assert elapsed >= 0.05
                assert response.status_code == 200


class TestNonceManagement:
    """Tests for nonce management."""

    def test_fetch_nonce_success(self):
        """Test successful nonce fetch."""
        client = AcmeHttpClient()
        client.set_nonce_url("https://acme.test/new-nonce")

        mock_response = MagicMock()
        mock_response.headers = {"Replay-Nonce": "test-nonce"}

        with patch.object(client._session, "head", return_value=mock_response):
            nonce = client._fetch_nonce()
            assert nonce == "test-nonce"

    def test_fetch_nonce_missing_header(self):
        """Test error when nonce header is missing."""
        client = AcmeHttpClient()
        client.set_nonce_url("https://acme.test/new-nonce")

        mock_response = MagicMock()
        mock_response.headers = {}

        with patch.object(client._session, "head", return_value=mock_response):
            with pytest.raises(AcmeNetworkError, match="Replay-Nonce"):
                client._fetch_nonce()

    def test_nonce_cached_and_consumed(self):
        """Test that nonce is cached and consumed."""
        client = AcmeHttpClient()
        client.set_nonce_url("https://acme.test/new-nonce")

        # Manually set cached nonce
        client._nonce = "cached-nonce"

        nonce = client._get_nonce()
        assert nonce == "cached-nonce"
        assert client._nonce is None  # Consumed

    def test_nonce_updated_from_response(self):
        """Test that nonce is updated from response headers."""
        client = AcmeHttpClient()

        mock_response = MagicMock()
        mock_response.headers = {"Replay-Nonce": "new-nonce"}

        client._update_nonce(mock_response)
        assert client._nonce == "new-nonce"


class TestErrorHandling:
    """Tests for error handling."""

    def test_server_error_response(self):
        """Test handling of server error response."""
        client = AcmeHttpClient(
            retry_config=RetryConfig(max_retries=0)  # No retries
        )
        client.set_nonce_url("https://acme.test/new-nonce")

        from acmeow._internal.crypto import generate_account_key

        key = generate_account_key()

        nonce_response = MagicMock(headers={"Replay-Nonce": "nonce"})
        error_response = MagicMock(
            status_code=400,
            headers={"Replay-Nonce": "nonce"},
        )
        error_response.json.return_value = {
            "type": "urn:ietf:params:acme:error:malformed",
            "detail": "Malformed request",
        }

        with patch.object(client._session, "head", return_value=nonce_response):
            with patch.object(client._session, "post", return_value=error_response):
                with pytest.raises(AcmeServerError) as exc_info:
                    client.post(
                        "https://acme.test/resource",
                        {"data": "test"},
                        key,
                        kid="https://acme.test/acct/1",
                    )
                assert exc_info.value.status_code == 400
                assert "malformed" in exc_info.value.error_type

    def test_rate_limit_error(self):
        """Test handling of rate limit error."""
        client = AcmeHttpClient(
            retry_config=RetryConfig(max_retries=0)
        )
        client.set_nonce_url("https://acme.test/new-nonce")

        from acmeow._internal.crypto import generate_account_key

        key = generate_account_key()

        nonce_response = MagicMock(headers={"Replay-Nonce": "nonce"})
        rate_limit_response = MagicMock(
            status_code=429,
            headers={"Replay-Nonce": "nonce", "Retry-After": "60"},
        )
        rate_limit_response.json.return_value = {
            "type": "urn:ietf:params:acme:error:rateLimited",
            "detail": "Too many requests",
        }

        with patch.object(client._session, "head", return_value=nonce_response):
            with patch.object(client._session, "post", return_value=rate_limit_response):
                with pytest.raises(AcmeRateLimitError) as exc_info:
                    client.post(
                        "https://acme.test/resource",
                        {"data": "test"},
                        key,
                        kid="https://acme.test/acct/1",
                    )
                # Rate limit error is raised when retries exhausted
                assert "Rate limited" in str(exc_info.value)


class TestRetryableStatusCodes:
    """Tests for retryable status codes."""

    def test_retryable_codes(self):
        """Test that correct status codes are retryable."""
        assert 408 in RETRYABLE_STATUS_CODES  # Request Timeout
        assert 429 in RETRYABLE_STATUS_CODES  # Too Many Requests
        assert 500 in RETRYABLE_STATUS_CODES  # Internal Server Error
        assert 502 in RETRYABLE_STATUS_CODES  # Bad Gateway
        assert 503 in RETRYABLE_STATUS_CODES  # Service Unavailable
        assert 504 in RETRYABLE_STATUS_CODES  # Gateway Timeout

    def test_non_retryable_codes(self):
        """Test that certain codes are not retryable."""
        assert 400 not in RETRYABLE_STATUS_CODES  # Bad Request
        assert 401 not in RETRYABLE_STATUS_CODES  # Unauthorized
        assert 403 not in RETRYABLE_STATUS_CODES  # Forbidden
        assert 404 not in RETRYABLE_STATUS_CODES  # Not Found
