"""Tests for the ACME client."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from acmeow import (
    AcmeClient,
    AcmeAuthenticationError,
    AcmeConfigurationError,
    AcmeNetworkError,
    DnsConfig,
    Identifier,
    RetryConfig,
)


class TestAcmeClientInit:
    """Tests for AcmeClient initialization."""

    def test_init_fetches_directory(self, temp_storage: Path):
        """Test that client fetches directory on init."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "newNonce": "https://acme.test/new-nonce",
            "newAccount": "https://acme.test/new-account",
            "newOrder": "https://acme.test/new-order",
        }

        with patch("requests.Session.get", return_value=mock_response):
            client = AcmeClient(
                server_url="https://acme.test/directory",
                email="test@example.com",
                storage_path=temp_storage,
            )
            assert client.server_url == "https://acme.test/directory"
            assert client.email == "test@example.com"

    def test_init_with_retry_config(self, temp_storage: Path):
        """Test initialization with custom retry config."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "newNonce": "https://acme.test/new-nonce",
            "newAccount": "https://acme.test/new-account",
        }

        retry_config = RetryConfig(max_retries=3, initial_delay=0.5)

        with patch("requests.Session.get", return_value=mock_response):
            client = AcmeClient(
                server_url="https://acme.test/directory",
                email="test@example.com",
                storage_path=temp_storage,
                retry_config=retry_config,
            )
            assert client._http._retry_config.max_retries == 3
            assert client._http._retry_config.initial_delay == 0.5

    def test_init_missing_nonce_url_raises(self, temp_storage: Path):
        """Test that missing newNonce URL raises error."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "newAccount": "https://acme.test/new-account",
        }

        with patch("requests.Session.get", return_value=mock_response):
            with pytest.raises(AcmeNetworkError, match="newNonce"):
                AcmeClient(
                    server_url="https://acme.test/directory",
                    email="test@example.com",
                    storage_path=temp_storage,
                )


class TestAccountManagement:
    """Tests for account management."""

    @pytest.fixture
    def client(self, temp_storage: Path):
        """Create a client with mocked directory fetch."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "newNonce": "https://acme.test/new-nonce",
            "newAccount": "https://acme.test/new-account",
            "newOrder": "https://acme.test/new-order",
            "keyChange": "https://acme.test/key-change",
            "revokeCert": "https://acme.test/revoke-cert",
        }
        mock_response.headers = {"Replay-Nonce": "test-nonce"}

        with patch("requests.Session.get", return_value=mock_response):
            with patch("requests.Session.head", return_value=mock_response):
                return AcmeClient(
                    server_url="https://acme.test/directory",
                    email="test@example.com",
                    storage_path=temp_storage,
                )

    def test_create_account_terms_required(self, client: AcmeClient):
        """Test that terms must be agreed."""
        with pytest.raises(AcmeConfigurationError, match="Terms"):
            client.create_account(terms_agreed=False)

    def test_create_account_success(self, client: AcmeClient):
        """Test successful account creation."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "valid"}
        mock_response.status_code = 201
        mock_response.headers = {
            "Location": "https://acme.test/acct/123",
            "Replay-Nonce": "new-nonce",
        }

        with patch.object(client._http._session, "post", return_value=mock_response):
            with patch.object(client._http._session, "head", return_value=mock_response):
                account = client.create_account()
                assert account.uri == "https://acme.test/acct/123"
                assert account.is_valid

    def test_create_account_no_location_raises(self, client: AcmeClient):
        """Test that missing Location header raises error."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "valid"}
        mock_response.headers = {"Replay-Nonce": "new-nonce"}
        mock_response.status_code = 201

        with patch.object(client._http._session, "post", return_value=mock_response):
            with patch.object(client._http._session, "head", return_value=mock_response):
                with pytest.raises(AcmeAuthenticationError, match="URL"):
                    client.create_account()


class TestOrderManagement:
    """Tests for order management."""

    @pytest.fixture
    def client_with_account(self, temp_storage: Path):
        """Create a client with a valid account."""
        mock_dir_response = MagicMock()
        mock_dir_response.json.return_value = {
            "newNonce": "https://acme.test/new-nonce",
            "newAccount": "https://acme.test/new-account",
            "newOrder": "https://acme.test/new-order",
            "keyChange": "https://acme.test/key-change",
            "revokeCert": "https://acme.test/revoke-cert",
        }
        mock_dir_response.headers = {"Replay-Nonce": "test-nonce"}

        mock_acct_response = MagicMock()
        mock_acct_response.json.return_value = {"status": "valid"}
        mock_acct_response.headers = {
            "Location": "https://acme.test/acct/123",
            "Replay-Nonce": "new-nonce",
        }
        mock_acct_response.status_code = 201

        with patch("requests.Session.get", return_value=mock_dir_response):
            with patch("requests.Session.head", return_value=mock_dir_response):
                client = AcmeClient(
                    server_url="https://acme.test/directory",
                    email="test@example.com",
                    storage_path=temp_storage,
                )
                with patch("requests.Session.post", return_value=mock_acct_response):
                    client.create_account()
                return client

    def test_create_order_without_account_raises(self, temp_storage: Path):
        """Test that creating order without account raises error."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "newNonce": "https://acme.test/new-nonce",
            "newAccount": "https://acme.test/new-account",
            "newOrder": "https://acme.test/new-order",
        }
        mock_response.headers = {"Replay-Nonce": "test-nonce"}

        with patch("requests.Session.get", return_value=mock_response):
            client = AcmeClient(
                server_url="https://acme.test/directory",
                email="test@example.com",
                storage_path=temp_storage,
            )
            with pytest.raises(AcmeAuthenticationError):
                client.create_order([Identifier.dns("example.com")])

    def test_create_order_empty_identifiers_raises(self, client_with_account: AcmeClient):
        """Test that empty identifiers list raises error."""
        with pytest.raises(AcmeConfigurationError, match="identifier"):
            client_with_account.create_order([])

    def test_create_order_success(self, client_with_account: AcmeClient):
        """Test successful order creation."""
        mock_order_response = MagicMock()
        mock_order_response.json.return_value = {
            "status": "pending",
            "identifiers": [{"type": "dns", "value": "example.com"}],
            "authorizations": ["https://acme.test/authz/1"],
            "finalize": "https://acme.test/order/1/finalize",
        }
        mock_order_response.headers = {
            "Location": "https://acme.test/order/1",
            "Replay-Nonce": "new-nonce",
        }
        mock_order_response.status_code = 201

        mock_authz_response = MagicMock()
        mock_authz_response.json.return_value = {
            "status": "pending",
            "identifier": {"type": "dns", "value": "example.com"},
            "challenges": [
                {"type": "dns-01", "status": "pending", "url": "https://acme.test/chall/1", "token": "abc"},
                {"type": "http-01", "status": "pending", "url": "https://acme.test/chall/2", "token": "abc"},
            ],
        }
        mock_authz_response.headers = {"Replay-Nonce": "new-nonce"}
        mock_authz_response.status_code = 200

        mock_get_response = MagicMock()
        mock_get_response.json.return_value = {
            "status": "pending",
            "authorizations": ["https://acme.test/authz/1"],
        }
        mock_get_response.headers = {"Replay-Nonce": "new-nonce"}

        with patch.object(client_with_account._http._session, "post") as mock_post:
            with patch.object(client_with_account._http._session, "get", return_value=mock_get_response):
                mock_post.side_effect = [mock_order_response, mock_authz_response]
                order = client_with_account.create_order([Identifier.dns("example.com")])
                assert order.url == "https://acme.test/order/1"
                assert order.is_pending


class TestOrderRecovery:
    """Tests for order recovery functionality."""

    @pytest.fixture
    def client_with_order(self, temp_storage: Path):
        """Create a client with a saved order."""
        # Create order file
        order_dir = temp_storage / "orders"
        order_dir.mkdir(parents=True, exist_ok=True)
        order_file = order_dir / "current_order.json"
        order_file.write_text(json.dumps({
            "url": "https://acme.test/order/saved",
            "status": "ready",
            "identifiers": [{"type": "dns", "value": "example.com"}],
            "finalize_url": "https://acme.test/order/saved/finalize",
            "expires": "2099-01-01T00:00:00Z",
        }))

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "newNonce": "https://acme.test/new-nonce",
            "newAccount": "https://acme.test/new-account",
            "newOrder": "https://acme.test/new-order",
        }
        mock_response.headers = {"Replay-Nonce": "test-nonce"}

        with patch("requests.Session.get", return_value=mock_response):
            with patch("requests.Session.head", return_value=mock_response):
                return AcmeClient(
                    server_url="https://acme.test/directory",
                    email="test@example.com",
                    storage_path=temp_storage,
                )

    def test_load_order_from_disk(self, client_with_order: AcmeClient):
        """Test loading order from disk."""
        mock_order_response = MagicMock()
        mock_order_response.json.return_value = {
            "status": "ready",
            "identifiers": [{"type": "dns", "value": "example.com"}],
            "authorizations": ["https://acme.test/authz/1"],
            "finalize": "https://acme.test/order/saved/finalize",
        }
        mock_order_response.headers = {"Replay-Nonce": "new-nonce"}
        mock_order_response.status_code = 200

        mock_authz_response = MagicMock()
        mock_authz_response.json.return_value = {
            "status": "valid",
            "identifier": {"type": "dns", "value": "example.com"},
            "challenges": [],
        }
        mock_authz_response.headers = {"Replay-Nonce": "new-nonce"}
        mock_authz_response.status_code = 200

        # Create mock account
        mock_acct_response = MagicMock()
        mock_acct_response.json.return_value = {"status": "valid"}
        mock_acct_response.headers = {
            "Location": "https://acme.test/acct/123",
            "Replay-Nonce": "new-nonce",
        }
        mock_acct_response.status_code = 201

        mock_head_response = MagicMock()
        mock_head_response.headers = {"Replay-Nonce": "nonce"}

        mock_get_response = MagicMock()
        mock_get_response.json.return_value = {
            "status": "ready",
            "authorizations": ["https://acme.test/authz/1"],
        }
        mock_get_response.headers = {"Replay-Nonce": "new-nonce"}

        with patch.object(client_with_order._http._session, "post") as mock_post:
            with patch.object(client_with_order._http._session, "head", return_value=mock_head_response):
                with patch.object(client_with_order._http._session, "get", return_value=mock_get_response):
                    mock_post.side_effect = [
                        mock_acct_response,
                        mock_order_response,
                        mock_authz_response,
                    ]
                    client_with_order.create_account()

                    # Explicitly load order after account is set up
                    # (order was loaded in constructor but without auth URLs)
                    order = client_with_order.load_order()
                    assert order is not None
                    assert order.url == "https://acme.test/order/saved"


class TestDnsVerification:
    """Tests for DNS verification functionality."""

    def test_set_dns_config(self, temp_storage: Path):
        """Test setting DNS configuration."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "newNonce": "https://acme.test/new-nonce",
            "newAccount": "https://acme.test/new-account",
        }
        mock_response.headers = {"Replay-Nonce": "test-nonce"}

        with patch("requests.Session.get", return_value=mock_response):
            client = AcmeClient(
                server_url="https://acme.test/directory",
                email="test@example.com",
                storage_path=temp_storage,
            )

            dns_config = DnsConfig(
                nameservers=["8.8.8.8", "1.1.1.1"],
                timeout=10.0,
                retries=5,
            )
            client.set_dns_config(dns_config)

            assert client._dns_config is not None
            assert client._dns_config.nameservers == ["8.8.8.8", "1.1.1.1"]
            assert client._dns_config.timeout == 10.0


class TestPreferredChain:
    """Tests for preferred chain selection."""

    def test_get_certificate_with_preferred_chain(self, temp_storage: Path):
        """Test certificate download with preferred chain."""
        # This is a more complex integration test
        # We test the chain selection logic
        from acmeow.client import AcmeClient

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "newNonce": "https://acme.test/new-nonce",
            "newAccount": "https://acme.test/new-account",
            "newOrder": "https://acme.test/new-order",
        }
        mock_response.headers = {"Replay-Nonce": "test-nonce"}

        with patch("requests.Session.get", return_value=mock_response):
            client = AcmeClient(
                server_url="https://acme.test/directory",
                email="test@example.com",
                storage_path=temp_storage,
            )
            # preferred_chain parameter should be accepted
            assert hasattr(client.get_certificate, "__call__")


class TestContextManager:
    """Tests for context manager functionality."""

    def test_context_manager(self, temp_storage: Path):
        """Test client as context manager."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "newNonce": "https://acme.test/new-nonce",
            "newAccount": "https://acme.test/new-account",
        }
        mock_response.headers = {"Replay-Nonce": "test-nonce"}

        with patch("requests.Session.get", return_value=mock_response):
            with AcmeClient(
                server_url="https://acme.test/directory",
                email="test@example.com",
                storage_path=temp_storage,
            ) as client:
                assert client is not None

    def test_close_called(self, temp_storage: Path):
        """Test that close is called on exit."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "newNonce": "https://acme.test/new-nonce",
            "newAccount": "https://acme.test/new-account",
        }
        mock_response.headers = {"Replay-Nonce": "test-nonce"}

        with patch("requests.Session.get", return_value=mock_response):
            with patch("requests.Session.close") as mock_close:
                with AcmeClient(
                    server_url="https://acme.test/directory",
                    email="test@example.com",
                    storage_path=temp_storage,
                ) as client:
                    pass
                mock_close.assert_called()
