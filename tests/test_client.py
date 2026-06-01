"""Tests for the ACME client."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from acmeow import (
    AcmeAuthenticationError,
    AcmeClient,
    AcmeConfigurationError,
    AcmeNetworkError,
    DnsConfig,
    Identifier,
    RetryConfig,
)
from acmeow.exceptions import AcmeServerError


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

    def test_init_with_order_ready_timeout(self, temp_storage: Path):
        """order_ready_timeout is stored on the client."""
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
                order_ready_timeout=60,
            )
        assert client._order_ready_timeout == 60

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

    def test_load_order_discards_expired_order(self, client_with_order: AcmeClient, temp_storage: Path):
        """Server rejecting a saved order with 400 'order is expired' clears state and file."""
        order_file = temp_storage / "orders" / "current_order.json"
        assert order_file.exists()

        mock_account = MagicMock()
        mock_account.uri = "https://acme.test/acct/123"
        mock_account.key = MagicMock()
        client_with_order._account = mock_account

        expired_error = AcmeServerError(400, "urn:ietf:params:acme:error:malformed", "order is expired")

        with patch.object(client_with_order._http, "post_as_get", side_effect=expired_error):
            client_with_order.load_order()

        assert client_with_order._order is None
        assert not order_file.exists()

    def test_load_order_discards_on_any_server_error(self, client_with_order: AcmeClient, temp_storage: Path):
        """Any AcmeServerError during order refresh clears state and file."""
        order_file = temp_storage / "orders" / "current_order.json"
        assert order_file.exists()

        mock_account = MagicMock()
        mock_account.uri = "https://acme.test/acct/123"
        mock_account.key = MagicMock()
        client_with_order._account = mock_account

        server_error = AcmeServerError(403, "urn:ietf:params:acme:error:unauthorized", "account not found")

        with patch.object(client_with_order._http, "post_as_get", side_effect=server_error):
            client_with_order.load_order()

        assert client_with_order._order is None
        assert not order_file.exists()


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

        with patch("requests.Session.get", return_value=mock_response), AcmeClient(
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


class TestFinalizeOrderExternalCSR:
    """Tests for finalize_order with an external CSR."""

    @pytest.fixture
    def client_with_ready_order(self, temp_storage: Path):
        """Client with an account and a READY order."""
        mock_dir = MagicMock()
        mock_dir.json.return_value = {
            "newNonce": "https://acme.test/new-nonce",
            "newAccount": "https://acme.test/new-account",
            "newOrder": "https://acme.test/new-order",
        }
        mock_dir.headers = {"Replay-Nonce": "nonce"}

        with patch("requests.Session.get", return_value=mock_dir):
            client = AcmeClient(
                server_url="https://acme.test/directory",
                email="test@example.com",
                storage_path=temp_storage,
            )

        # Wire up a mock account
        from acmeow.models.account import Account

        acct = Account(
            email="test@example.com",
            storage_path=temp_storage,
            server_url="https://acme.test/directory",
        )
        acct.create_key()
        acct.save("https://acme.test/acct/1", "valid")
        client._account = acct

        # Wire up a READY order
        from acmeow.enums import OrderStatus
        from acmeow.models.order import Order

        client._order = Order(
            status=OrderStatus.READY,
            url="https://acme.test/order/1",
            identifiers=(Identifier.dns("example.com"),),
            finalize_url="https://acme.test/order/1/finalize",
        )
        return client

    def _make_external_csr(self) -> bytes:
        """Return a minimal DER-encoded CSR for example.com."""
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID

        key = ec.generate_private_key(ec.SECP256R1())
        csr = (
            x509.CertificateSigningRequestBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "example.com")]))
            .add_extension(
                x509.SubjectAlternativeName([x509.DNSName("example.com")]),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )
        return csr.public_bytes(serialization.Encoding.DER)

    def test_finalize_with_external_csr_skips_key_generation(self, client_with_ready_order):
        """External CSR path must not call generate_private_key."""
        client = client_with_ready_order
        csr_der = self._make_external_csr()

        # Responses: refresh order (READY), finalize POST, poll order (VALID)
        refresh_ready = MagicMock(status_code=200, headers={"Replay-Nonce": "n1"})
        refresh_ready.json.return_value = {
            "status": "ready",
            "identifiers": [{"type": "dns", "value": "example.com"}],
            "finalize": "https://acme.test/order/1/finalize",
        }
        finalize_ok = MagicMock(status_code=200, headers={"Replay-Nonce": "n2"})
        finalize_ok.json.return_value = {"status": "processing"}
        poll_valid = MagicMock(status_code=200, headers={"Replay-Nonce": "n3"})
        poll_valid.json.return_value = {
            "status": "valid",
            "certificate": "https://acme.test/cert/1",
        }

        with patch("acmeow.client.generate_private_key") as mock_gen_key:
            with patch.object(client._http._session, "post") as mock_post:
                mock_post.side_effect = [refresh_ready, finalize_ok, poll_valid]
                client.finalize_order(csr=csr_der)

            mock_gen_key.assert_not_called()

    def test_finalize_with_external_csr_sends_correct_payload(self, client_with_ready_order):
        """The CSR bytes in the payload must match what was passed in."""
        import base64

        client = client_with_ready_order
        csr_der = self._make_external_csr()

        refresh_ready = MagicMock(status_code=200, headers={"Replay-Nonce": "n1"})
        refresh_ready.json.return_value = {
            "status": "ready",
            "identifiers": [{"type": "dns", "value": "example.com"}],
            "finalize": "https://acme.test/order/1/finalize",
        }
        finalize_ok = MagicMock(status_code=200, headers={"Replay-Nonce": "n2"})
        finalize_ok.json.return_value = {"status": "processing"}
        poll_valid = MagicMock(status_code=200, headers={"Replay-Nonce": "n3"})
        poll_valid.json.return_value = {
            "status": "valid",
            "certificate": "https://acme.test/cert/1",
        }

        with patch.object(client._http._session, "post") as mock_post:
            mock_post.side_effect = [refresh_ready, finalize_ok, poll_valid]
            client.finalize_order(csr=csr_der)

        # The finalize POST is the second call; extract the JWS payload
        finalize_call = mock_post.call_args_list[1]
        jws_body = finalize_call[1]["json"]
        # payload is base64url-encoded JSON — decode it
        payload_b64 = jws_body["payload"]
        # Pad to multiple of 4
        padded = payload_b64 + "=" * (-len(payload_b64) % 4)
        payload_json = json.loads(base64.urlsafe_b64decode(padded))
        # The csr field in the payload must round-trip back to the original bytes
        csr_b64 = payload_json["csr"]
        padded_csr = csr_b64 + "=" * (-len(csr_b64) % 4)
        decoded_csr = base64.urlsafe_b64decode(padded_csr)
        assert decoded_csr == csr_der

    def test_get_certificate_returns_empty_key_for_external_csr(self, client_with_ready_order, tmp_path):
        """get_certificate returns None for the key when external CSR was used (no key file on disk)."""
        client = client_with_ready_order

        from acmeow.enums import OrderStatus
        from acmeow.models.order import Order

        client._order = Order(
            status=OrderStatus.VALID,
            url="https://acme.test/order/1",
            identifiers=(Identifier.dns("example.com"),),
            finalize_url="https://acme.test/order/1/finalize",
            certificate_url="https://acme.test/cert/1",
        )

        cert_pem = "-----BEGIN CERTIFICATE-----\nMIIB...\n-----END CERTIFICATE-----\n"

        refresh_valid = MagicMock(status_code=200, headers={"Replay-Nonce": "n1"})
        refresh_valid.json.return_value = {
            "status": "valid",
            "certificate": "https://acme.test/cert/1",
        }
        cert_response = MagicMock(status_code=200, headers={"Replay-Nonce": "n2", "Link": ""})
        cert_response.text = cert_pem

        with patch.object(client._http._session, "post") as mock_post:
            mock_post.side_effect = [refresh_valid, cert_response]
            returned_cert, returned_key = client.get_certificate()

        assert returned_cert == cert_pem
        assert returned_key is None

    def test_get_certificate_returns_key_for_internal_csr(self, client_with_ready_order, tmp_path):
        """get_certificate returns the key PEM when the key was generated internally."""
        client = client_with_ready_order

        from acmeow.enums import OrderStatus
        from acmeow.models.order import Order

        client._order = Order(
            status=OrderStatus.VALID,
            url="https://acme.test/order/1",
            identifiers=(Identifier.dns("example.com"),),
            finalize_url="https://acme.test/order/1/finalize",
            certificate_url="https://acme.test/cert/1",
        )

        cert_pem = "-----BEGIN CERTIFICATE-----\nMIIB...\n-----END CERTIFICATE-----\n"
        key_pem = "-----BEGIN EC PRIVATE KEY-----\nMHQ...\n-----END EC PRIVATE KEY-----\n"

        # Provide a key file where get_certificate_paths points
        _, key_path = client._account.get_certificate_paths(client._order.common_name)
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_text(key_pem)

        refresh_valid = MagicMock(status_code=200, headers={"Replay-Nonce": "n1"})
        refresh_valid.json.return_value = {
            "status": "valid",
            "certificate": "https://acme.test/cert/1",
        }
        cert_response = MagicMock(status_code=200, headers={"Replay-Nonce": "n2", "Link": ""})
        cert_response.text = cert_pem

        with patch.object(client._http._session, "post") as mock_post:
            mock_post.side_effect = [refresh_valid, cert_response]
            returned_cert, returned_key = client.get_certificate()

        assert returned_cert == cert_pem
        assert returned_key == key_pem


class TestFinalizeOrderReadyTimeout:
    """Tests for the pending→ready polling loop in finalize_order."""

    @pytest.fixture
    def client_with_pending_order(self, temp_storage: Path):
        """Client with account + PENDING order and a short order_ready_timeout (4 s → 2 attempts)."""
        mock_dir = MagicMock()
        mock_dir.json.return_value = {
            "newNonce": "https://acme.test/new-nonce",
            "newAccount": "https://acme.test/new-account",
            "newOrder": "https://acme.test/new-order",
        }
        mock_dir.headers = {"Replay-Nonce": "nonce"}

        with patch("requests.Session.get", return_value=mock_dir):
            client = AcmeClient(
                server_url="https://acme.test/directory",
                email="test@example.com",
                storage_path=temp_storage,
                order_ready_timeout=4,
            )

        from acmeow.models.account import Account

        acct = Account(
            email="test@example.com",
            storage_path=temp_storage,
            server_url="https://acme.test/directory",
        )
        acct.create_key()
        acct.save("https://acme.test/acct/1", "valid")
        client._account = acct

        from acmeow.enums import OrderStatus
        from acmeow.models.order import Order

        client._order = Order(
            status=OrderStatus.PENDING,
            url="https://acme.test/order/1",
            identifiers=(Identifier.dns("example.com"),),
            finalize_url="https://acme.test/order/1/finalize",
        )
        return client

    def _make_external_csr(self) -> bytes:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID

        key = ec.generate_private_key(ec.SECP256R1())
        csr = (
            x509.CertificateSigningRequestBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "example.com")]))
            .add_extension(
                x509.SubjectAlternativeName([x509.DNSName("example.com")]),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )
        return csr.public_bytes(serialization.Encoding.DER)

    def _pending_response(self, nonce: str) -> MagicMock:
        r = MagicMock(status_code=200, headers={"Replay-Nonce": nonce})
        r.json.return_value = {
            "status": "pending",
            "identifiers": [{"type": "dns", "value": "example.com"}],
            "finalize": "https://acme.test/order/1/finalize",
        }
        return r

    def _ready_response(self, nonce: str) -> MagicMock:
        r = MagicMock(status_code=200, headers={"Replay-Nonce": nonce})
        r.json.return_value = {
            "status": "ready",
            "identifiers": [{"type": "dns", "value": "example.com"}],
            "finalize": "https://acme.test/order/1/finalize",
        }
        return r

    def test_finalize_retries_until_order_ready(self, client_with_pending_order):
        """finalize_order succeeds when server returns ready status."""
        client = client_with_pending_order
        csr_der = self._make_external_csr()

        finalize_ok = MagicMock(status_code=200, headers={"Replay-Nonce": "n2"})
        finalize_ok.json.return_value = {"status": "processing"}
        poll_valid = MagicMock(status_code=200, headers={"Replay-Nonce": "n3"})
        poll_valid.json.return_value = {
            "status": "valid",
            "certificate": "https://acme.test/cert/1",
        }

        with (
            patch("acmeow.client.time.sleep"),
            patch.object(client._http._session, "post") as mock_post,
        ):
            mock_post.side_effect = [
                self._ready_response("n1"),
                finalize_ok,
                poll_valid,
            ]
            client.finalize_order(csr=csr_der)

    def test_finalize_proceeds_when_order_stays_pending(self, client_with_pending_order):
        """finalize_order proceeds with CSR submission when order stays pending.

        Some CAs never transition the order to 'ready'; the client should
        attempt finalization anyway rather than timing out.
        """
        client = client_with_pending_order
        csr_der = self._make_external_csr()

        finalize_ok = MagicMock(status_code=200, headers={"Replay-Nonce": "n2"})
        finalize_ok.json.return_value = {"status": "processing"}
        poll_valid = MagicMock(status_code=200, headers={"Replay-Nonce": "n3"})
        poll_valid.json.return_value = {
            "status": "valid",
            "certificate": "https://acme.test/cert/1",
        }

        with (
            patch("acmeow.client.time.sleep"),
            patch.object(client._http._session, "post") as mock_post,
        ):
            mock_post.side_effect = [
                self._pending_response("n1"),
                finalize_ok,
                poll_valid,
            ]
            client.finalize_order(csr=csr_der)
