"""Pytest configuration and fixtures for acmeow tests."""

from __future__ import annotations

import base64
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurvePrivateKey


@dataclass
class MockAccount:
    """Mock ACME account state."""

    uri: str = ""
    status: str = "valid"
    contact: list[str] = field(default_factory=list)
    key_thumbprint: str = ""


@dataclass
class MockOrder:
    """Mock ACME order state."""

    url: str = ""
    status: str = "pending"
    identifiers: list[dict[str, str]] = field(default_factory=list)
    authorizations: list[str] = field(default_factory=list)
    finalize: str = ""
    certificate: str | None = None
    expires: str = "2099-01-01T00:00:00Z"


@dataclass
class MockAuthorization:
    """Mock ACME authorization state."""

    url: str = ""
    status: str = "pending"
    identifier: dict[str, str] = field(default_factory=dict)
    challenges: list[dict[str, Any]] = field(default_factory=list)


class MockAcmeServer:
    """Mock ACME server for testing.

    Simulates ACME server responses for testing without network.
    """

    def __init__(self, base_url: str = "https://acme.test") -> None:
        self.base_url = base_url
        self.directory = {
            "newNonce": f"{base_url}/acme/new-nonce",
            "newAccount": f"{base_url}/acme/new-acct",
            "newOrder": f"{base_url}/acme/new-order",
            "revokeCert": f"{base_url}/acme/revoke-cert",
            "keyChange": f"{base_url}/acme/key-change",
            "meta": {
                "termsOfService": f"{base_url}/acme/terms",
                "website": base_url,
                "externalAccountRequired": False,
            },
        }
        self._nonce_counter = 0
        self._accounts: dict[str, MockAccount] = {}
        self._orders: dict[str, MockOrder] = {}
        self._authorizations: dict[str, MockAuthorization] = {}
        self._certificates: dict[str, str] = {}

        # Error simulation flags
        self.simulate_rate_limit = False
        self.simulate_server_error = False
        self.simulate_timeout = False
        self.simulate_bad_nonce = False
        self.rate_limit_retry_after: float | None = None

    def get_nonce(self) -> str:
        """Generate a new nonce."""
        self._nonce_counter += 1
        return f"nonce-{self._nonce_counter}"

    def create_account(self, email: str, key_thumbprint: str) -> MockAccount:
        """Create a new mock account."""
        account_id = str(uuid.uuid4())
        account = MockAccount(
            uri=f"{self.base_url}/acme/acct/{account_id}",
            status="valid",
            contact=[f"mailto:{email}"],
            key_thumbprint=key_thumbprint,
        )
        self._accounts[account.uri] = account
        return account

    def create_order(self, identifiers: list[dict[str, str]]) -> MockOrder:
        """Create a new mock order."""
        order_id = str(uuid.uuid4())
        order_url = f"{self.base_url}/acme/order/{order_id}"

        # Create authorizations
        auth_urls = []
        for ident in identifiers:
            auth_id = str(uuid.uuid4())
            auth_url = f"{self.base_url}/acme/authz/{auth_id}"
            auth_urls.append(auth_url)

            challenge_token = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode().rstrip("=")
            auth = MockAuthorization(
                url=auth_url,
                status="pending",
                identifier=ident,
                challenges=[
                    {
                        "type": "dns-01",
                        "status": "pending",
                        "url": f"{self.base_url}/acme/chall/{uuid.uuid4()}",
                        "token": challenge_token,
                    },
                    {
                        "type": "http-01",
                        "status": "pending",
                        "url": f"{self.base_url}/acme/chall/{uuid.uuid4()}",
                        "token": challenge_token,
                    },
                ],
            )
            self._authorizations[auth_url] = auth

        order = MockOrder(
            url=order_url,
            status="pending",
            identifiers=identifiers,
            authorizations=auth_urls,
            finalize=f"{self.base_url}/acme/order/{order_id}/finalize",
        )
        self._orders[order_url] = order
        return order

    def complete_authorization(self, auth_url: str) -> None:
        """Mark an authorization as valid."""
        if auth_url in self._authorizations:
            self._authorizations[auth_url].status = "valid"
            for challenge in self._authorizations[auth_url].challenges:
                challenge["status"] = "valid"

    def complete_order(self, order_url: str) -> None:
        """Mark an order as ready for finalization."""
        if order_url in self._orders:
            order = self._orders[order_url]
            order.status = "ready"
            # Mark all authorizations as valid
            for auth_url in order.authorizations:
                self.complete_authorization(auth_url)

    def finalize_order(self, order_url: str) -> None:
        """Finalize an order and generate certificate."""
        if order_url in self._orders:
            order = self._orders[order_url]
            order.status = "valid"
            cert_id = str(uuid.uuid4())
            order.certificate = f"{self.base_url}/acme/cert/{cert_id}"
            # Store a mock certificate
            self._certificates[order.certificate] = self._generate_mock_cert()

    def _generate_mock_cert(self) -> str:
        """Generate a mock PEM certificate."""
        return """-----BEGIN CERTIFICATE-----
MIIBkTCB+wIJAKHBfUzJT+FFMA0GCSqGSIb3DQEBCwUAMBExDzANBgNVBAMMBnVu
dXNlZDAeFw0yNDAxMDEwMDAwMDBaFw0yNTAxMDEwMDAwMDBaMBExDzANBgNVBAMM
BnVudXNlZDBcMA0GCSqGSIb3DQEBAQUAA0sAMEgCQQC7o96FCPCVVhBNfLwaWy0B
c2k5q6qNpSLa5xVEMGe1/FbGVEY5xgU8k7j5TuBDe3I3D1n2DLsN7VlWvzqq3B1L
AgMBAAGjUzBRMB0GA1UdDgQWBBQk6NuMYZSyJ/TP7j6h3bm1fqFXjzAfBgNVHSME
GDAWgBQk6NuMYZSyJ/TP7j6h3bm1fqFXjzAPBgNVHRMBAf8EBTADAQH/MA0GCSqG
SIb3DQEBCwUAA0EALgPxYv9r0QL0Z3qPKTLBkxNA6ZYvND5GKfmkYJPCiPsP7j2D
ZjCVPmBKSANNCVHnGJV8YPXE7D7xKjHYOJXm4g==
-----END CERTIFICATE-----
"""

    def get_order(self, url: str) -> MockOrder | None:
        """Get an order by URL."""
        return self._orders.get(url)

    def get_authorization(self, url: str) -> MockAuthorization | None:
        """Get an authorization by URL."""
        return self._authorizations.get(url)


class MockResponse:
    """Mock HTTP response."""

    def __init__(
        self,
        status_code: int = 200,
        json_data: dict[str, Any] | None = None,
        text: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.text = text if text else (json.dumps(json_data) if json_data else "")
        self.headers = headers or {}

    def json(self) -> dict[str, Any]:
        if self._json_data is None:
            raise ValueError("No JSON data")
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise Exception(f"HTTP Error: {self.status_code}")


@pytest.fixture
def mock_server() -> MockAcmeServer:
    """Create a mock ACME server."""
    return MockAcmeServer()


@pytest.fixture
def temp_storage(tmp_path: Path) -> Path:
    """Create a temporary storage directory."""
    storage = tmp_path / "acme_data"
    storage.mkdir(parents=True, exist_ok=True)
    return storage


@pytest.fixture
def mock_account_key() -> EllipticCurvePrivateKey:
    """Generate a mock account key."""
    from acmeow._internal.crypto import generate_account_key
    return generate_account_key()


@pytest.fixture
def mock_http_client(mock_server: MockAcmeServer):
    """Create a mock HTTP client."""

    def mock_request(method: str, url: str, **kwargs) -> MockResponse:
        """Handle mock requests."""
        # Check error simulation
        if mock_server.simulate_rate_limit:
            return MockResponse(
                status_code=429,
                json_data={
                    "type": "urn:ietf:params:acme:error:rateLimited",
                    "detail": "Rate limit exceeded",
                },
                headers={"Retry-After": str(mock_server.rate_limit_retry_after or 60)},
            )

        if mock_server.simulate_server_error:
            return MockResponse(
                status_code=503,
                json_data={
                    "type": "urn:ietf:params:acme:error:serverInternal",
                    "detail": "Server error",
                },
            )

        # Handle directory
        if url == mock_server.base_url:
            return MockResponse(
                status_code=200,
                json_data=mock_server.directory,
                headers={"Replay-Nonce": mock_server.get_nonce()},
            )

        # Handle nonce
        if url == mock_server.directory["newNonce"]:
            return MockResponse(
                status_code=200,
                headers={"Replay-Nonce": mock_server.get_nonce()},
            )

        # Handle new account
        if url == mock_server.directory["newAccount"]:
            account = mock_server.create_account("test@example.com", "thumbprint")
            return MockResponse(
                status_code=201,
                json_data={"status": account.status, "contact": account.contact},
                headers={
                    "Location": account.uri,
                    "Replay-Nonce": mock_server.get_nonce(),
                },
            )

        # Handle new order
        if url == mock_server.directory["newOrder"]:
            payload = kwargs.get("json", {})
            identifiers = payload.get("payload", {})
            # Parse identifiers from JWS payload
            order = mock_server.create_order([{"type": "dns", "value": "example.com"}])
            return MockResponse(
                status_code=201,
                json_data={
                    "status": order.status,
                    "identifiers": order.identifiers,
                    "authorizations": order.authorizations,
                    "finalize": order.finalize,
                    "expires": order.expires,
                },
                headers={
                    "Location": order.url,
                    "Replay-Nonce": mock_server.get_nonce(),
                },
            )

        # Handle authorization fetch
        if "/acme/authz/" in url:
            auth = mock_server.get_authorization(url)
            if auth:
                return MockResponse(
                    status_code=200,
                    json_data={
                        "status": auth.status,
                        "identifier": auth.identifier,
                        "challenges": auth.challenges,
                    },
                    headers={"Replay-Nonce": mock_server.get_nonce()},
                )

        # Handle order fetch
        if "/acme/order/" in url and "/finalize" not in url:
            order = mock_server.get_order(url)
            if order:
                return MockResponse(
                    status_code=200,
                    json_data={
                        "status": order.status,
                        "identifiers": order.identifiers,
                        "authorizations": order.authorizations,
                        "finalize": order.finalize,
                        "certificate": order.certificate,
                        "expires": order.expires,
                    },
                    headers={"Replay-Nonce": mock_server.get_nonce()},
                )

        # Handle finalize
        if "/finalize" in url:
            order_url = url.replace("/finalize", "")
            mock_server.finalize_order(order_url)
            order = mock_server.get_order(order_url)
            if order:
                return MockResponse(
                    status_code=200,
                    json_data={
                        "status": order.status,
                        "certificate": order.certificate,
                    },
                    headers={"Replay-Nonce": mock_server.get_nonce()},
                )

        # Handle certificate download
        if "/acme/cert/" in url:
            cert = mock_server._certificates.get(url)
            if cert:
                return MockResponse(
                    status_code=200,
                    text=cert,
                    headers={"Replay-Nonce": mock_server.get_nonce()},
                )

        # Handle challenge response
        if "/acme/chall/" in url:
            return MockResponse(
                status_code=200,
                json_data={"status": "processing"},
                headers={"Replay-Nonce": mock_server.get_nonce()},
            )

        # Default response
        return MockResponse(
            status_code=404,
            json_data={"type": "urn:ietf:params:acme:error:malformed", "detail": "Not found"},
        )

    return mock_request
