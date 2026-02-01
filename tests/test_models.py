"""Tests for ACME models."""

from __future__ import annotations

import pytest

from acmeow.enums import (
    AuthorizationStatus,
    ChallengeStatus,
    ChallengeType,
    IdentifierType,
    OrderStatus,
)
from acmeow.models import Authorization, Challenge, Identifier, Order


class TestIdentifier:
    """Tests for Identifier model."""

    def test_create_dns_identifier(self):
        """Test creating a DNS identifier."""
        ident = Identifier.dns("example.com")
        assert ident.type == IdentifierType.DNS
        assert ident.value == "example.com"

    def test_create_ip_identifier(self):
        """Test creating an IP identifier."""
        ident = Identifier.ip("192.168.1.1")
        assert ident.type == IdentifierType.IP
        assert ident.value == "192.168.1.1"

    def test_to_dict(self):
        """Test converting identifier to dictionary."""
        ident = Identifier.dns("example.com")
        result = ident.to_dict()
        assert result == {"type": "dns", "value": "example.com"}

    def test_from_dict(self):
        """Test creating identifier from dictionary."""
        data = {"type": "dns", "value": "test.example.com"}
        ident = Identifier.from_dict(data)
        assert ident.type == IdentifierType.DNS
        assert ident.value == "test.example.com"

    def test_from_dict_ip(self):
        """Test creating IP identifier from dictionary."""
        data = {"type": "ip", "value": "10.0.0.1"}
        ident = Identifier.from_dict(data)
        assert ident.type == IdentifierType.IP
        assert ident.value == "10.0.0.1"

    def test_str_representation(self):
        """Test string representation."""
        ident = Identifier.dns("example.com")
        assert str(ident) == "dns:example.com"

    def test_frozen_immutable(self):
        """Test that Identifier is immutable."""
        ident = Identifier.dns("example.com")
        with pytest.raises(AttributeError):
            ident.value = "other.com"  # type: ignore[misc]

    def test_equality(self):
        """Test identifier equality."""
        ident1 = Identifier.dns("example.com")
        ident2 = Identifier.dns("example.com")
        ident3 = Identifier.dns("other.com")
        assert ident1 == ident2
        assert ident1 != ident3

    def test_hash(self):
        """Test identifier hashing."""
        ident1 = Identifier.dns("example.com")
        ident2 = Identifier.dns("example.com")
        # Should be usable in sets/dicts
        assert hash(ident1) == hash(ident2)
        assert len({ident1, ident2}) == 1


class TestChallenge:
    """Tests for Challenge model."""

    def test_from_dict_dns(self):
        """Test creating DNS challenge from dictionary."""
        data = {
            "type": "dns-01",
            "status": "pending",
            "url": "https://acme.test/chall/1",
            "token": "abc123",
        }
        challenge = Challenge.from_dict(data)
        assert challenge.type == ChallengeType.DNS
        assert challenge.status == ChallengeStatus.PENDING
        assert challenge.url == "https://acme.test/chall/1"
        assert challenge.token == "abc123"

    def test_from_dict_http(self):
        """Test creating HTTP challenge from dictionary."""
        data = {
            "type": "http-01",
            "status": "valid",
            "url": "https://acme.test/chall/2",
            "token": "xyz789",
            "validated": "2024-01-01T00:00:00Z",
        }
        challenge = Challenge.from_dict(data)
        assert challenge.type == ChallengeType.HTTP
        assert challenge.status == ChallengeStatus.VALID
        assert challenge.validated == "2024-01-01T00:00:00Z"

    def test_from_dict_unknown_type(self):
        """Test handling of unknown challenge type."""
        data = {
            "type": "unknown-01",
            "status": "pending",
            "url": "https://acme.test/chall/3",
            "token": "def456",
        }
        challenge = Challenge.from_dict(data)
        # Unknown types default to DNS
        assert challenge.type == ChallengeType.DNS

    def test_from_dict_with_error(self):
        """Test challenge with error data."""
        data = {
            "type": "dns-01",
            "status": "invalid",
            "url": "https://acme.test/chall/4",
            "token": "err123",
            "error": {"type": "urn:ietf:params:acme:error:dns", "detail": "DNS lookup failed"},
        }
        challenge = Challenge.from_dict(data)
        assert challenge.status == ChallengeStatus.INVALID
        assert challenge.error is not None
        assert "dns" in challenge.error["type"]

    def test_status_properties(self):
        """Test challenge status property methods."""
        pending = Challenge.from_dict({
            "type": "dns-01", "status": "pending",
            "url": "https://acme.test/c/1", "token": "t1"
        })
        processing = Challenge.from_dict({
            "type": "dns-01", "status": "processing",
            "url": "https://acme.test/c/2", "token": "t2"
        })
        valid = Challenge.from_dict({
            "type": "dns-01", "status": "valid",
            "url": "https://acme.test/c/3", "token": "t3"
        })
        invalid = Challenge.from_dict({
            "type": "dns-01", "status": "invalid",
            "url": "https://acme.test/c/4", "token": "t4"
        })

        assert pending.is_pending and not pending.is_valid
        assert processing.is_processing and not processing.is_pending
        assert valid.is_valid and not valid.is_invalid
        assert invalid.is_invalid and not invalid.is_valid

    def test_str_representation(self):
        """Test string representation."""
        challenge = Challenge.from_dict({
            "type": "dns-01", "status": "pending",
            "url": "https://acme.test/c/1", "token": "t1"
        })
        assert "dns-01" in str(challenge)
        assert "pending" in str(challenge)

    def test_frozen_immutable(self):
        """Test that Challenge is immutable."""
        challenge = Challenge.from_dict({
            "type": "dns-01", "status": "pending",
            "url": "https://acme.test/c/1", "token": "t1"
        })
        with pytest.raises(AttributeError):
            challenge.status = ChallengeStatus.VALID  # type: ignore[misc]


class TestOrder:
    """Tests for Order model."""

    def test_from_dict(self):
        """Test creating order from dictionary."""
        data = {
            "status": "pending",
            "identifiers": [
                {"type": "dns", "value": "example.com"},
                {"type": "dns", "value": "www.example.com"},
            ],
            "finalize": "https://acme.test/order/1/finalize",
            "expires": "2024-12-31T23:59:59Z",
        }
        order = Order.from_dict(data, "https://acme.test/order/1")
        assert order.status == OrderStatus.PENDING
        assert order.url == "https://acme.test/order/1"
        assert len(order.identifiers) == 2
        assert order.identifiers[0].value == "example.com"
        assert order.finalize_url == "https://acme.test/order/1/finalize"

    def test_update_from_dict(self):
        """Test updating order from dictionary."""
        data = {
            "status": "pending",
            "identifiers": [{"type": "dns", "value": "example.com"}],
            "finalize": "https://acme.test/order/1/finalize",
        }
        order = Order.from_dict(data, "https://acme.test/order/1")

        # Update to valid status with certificate
        order.update_from_dict({
            "status": "valid",
            "certificate": "https://acme.test/cert/1",
        })
        assert order.status == OrderStatus.VALID
        assert order.certificate_url == "https://acme.test/cert/1"

    def test_status_properties(self):
        """Test order status property methods."""
        base = {
            "identifiers": [{"type": "dns", "value": "example.com"}],
            "finalize": "https://acme.test/order/1/finalize",
        }

        pending = Order.from_dict({**base, "status": "pending"}, "url1")
        ready = Order.from_dict({**base, "status": "ready"}, "url2")
        processing = Order.from_dict({**base, "status": "processing"}, "url3")
        valid = Order.from_dict({**base, "status": "valid"}, "url4")
        invalid = Order.from_dict({**base, "status": "invalid"}, "url5")

        assert pending.is_pending
        assert ready.is_ready
        assert processing.is_processing
        assert valid.is_valid
        assert invalid.is_invalid

    def test_is_finalized(self):
        """Test is_finalized property."""
        base = {
            "identifiers": [{"type": "dns", "value": "example.com"}],
            "finalize": "https://acme.test/order/1/finalize",
        }

        processing = Order.from_dict({**base, "status": "processing"}, "url1")
        valid = Order.from_dict({**base, "status": "valid"}, "url2")
        pending = Order.from_dict({**base, "status": "pending"}, "url3")

        assert processing.is_finalized
        assert valid.is_finalized
        assert not pending.is_finalized

    def test_domains_property(self):
        """Test domains property."""
        data = {
            "status": "pending",
            "identifiers": [
                {"type": "dns", "value": "example.com"},
                {"type": "dns", "value": "www.example.com"},
                {"type": "dns", "value": "api.example.com"},
            ],
            "finalize": "https://acme.test/order/1/finalize",
        }
        order = Order.from_dict(data, "https://acme.test/order/1")
        assert order.domains == ["example.com", "www.example.com", "api.example.com"]

    def test_common_name_property(self):
        """Test common_name property."""
        data = {
            "status": "pending",
            "identifiers": [
                {"type": "dns", "value": "www.example.com"},
                {"type": "dns", "value": "example.com"},
            ],
            "finalize": "https://acme.test/order/1/finalize",
        }
        order = Order.from_dict(data, "https://acme.test/order/1")
        assert order.common_name == "www.example.com"

    def test_common_name_empty(self):
        """Test common_name when no identifiers."""
        data = {
            "status": "pending",
            "identifiers": [],
            "finalize": "https://acme.test/order/1/finalize",
        }
        order = Order.from_dict(data, "https://acme.test/order/1")
        assert order.common_name == ""

    def test_str_representation(self):
        """Test string representation."""
        data = {
            "status": "pending",
            "identifiers": [{"type": "dns", "value": "example.com"}],
            "finalize": "https://acme.test/order/1/finalize",
        }
        order = Order.from_dict(data, "https://acme.test/order/1")
        result = str(order)
        assert "example.com" in result
        assert "pending" in result

    def test_str_representation_many_domains(self):
        """Test string representation with many domains."""
        data = {
            "status": "pending",
            "identifiers": [
                {"type": "dns", "value": f"domain{i}.example.com"}
                for i in range(10)
            ],
            "finalize": "https://acme.test/order/1/finalize",
        }
        order = Order.from_dict(data, "https://acme.test/order/1")
        result = str(order)
        assert "10 total" in result

    def test_order_with_error(self):
        """Test order with error data."""
        data = {
            "status": "invalid",
            "identifiers": [{"type": "dns", "value": "example.com"}],
            "finalize": "https://acme.test/order/1/finalize",
            "error": {"type": "urn:ietf:params:acme:error:unauthorized", "detail": "Unauthorized"},
        }
        order = Order.from_dict(data, "https://acme.test/order/1")
        assert order.is_invalid
        assert order.error is not None


class TestAuthorization:
    """Tests for Authorization model."""

    def test_from_dict(self):
        """Test creating authorization from dictionary."""
        data = {
            "status": "pending",
            "identifier": {"type": "dns", "value": "example.com"},
            "challenges": [
                {"type": "dns-01", "status": "pending", "url": "https://acme.test/c/1", "token": "t1"},
                {"type": "http-01", "status": "pending", "url": "https://acme.test/c/2", "token": "t2"},
            ],
        }
        auth = Authorization.from_dict(data, "https://acme.test/authz/1")
        assert auth.status == AuthorizationStatus.PENDING
        assert auth.identifier.value == "example.com"
        assert len(auth.challenges) == 2

    def test_get_challenge_by_type_dns(self):
        """Test getting DNS challenge."""
        data = {
            "status": "pending",
            "identifier": {"type": "dns", "value": "example.com"},
            "challenges": [
                {"type": "dns-01", "status": "pending", "url": "https://acme.test/c/1", "token": "dns-token"},
                {"type": "http-01", "status": "pending", "url": "https://acme.test/c/2", "token": "http-token"},
            ],
        }
        auth = Authorization.from_dict(data, "https://acme.test/authz/1")
        challenge = auth.get_challenge(ChallengeType.DNS)
        assert challenge is not None
        assert challenge.token == "dns-token"

    def test_get_challenge_by_type_http(self):
        """Test getting HTTP challenge."""
        data = {
            "status": "pending",
            "identifier": {"type": "dns", "value": "example.com"},
            "challenges": [
                {"type": "dns-01", "status": "pending", "url": "https://acme.test/c/1", "token": "dns-token"},
                {"type": "http-01", "status": "pending", "url": "https://acme.test/c/2", "token": "http-token"},
            ],
        }
        auth = Authorization.from_dict(data, "https://acme.test/authz/1")
        challenge = auth.get_challenge(ChallengeType.HTTP)
        assert challenge is not None
        assert challenge.token == "http-token"

    def test_get_challenge_not_found(self):
        """Test getting challenge that doesn't exist."""
        data = {
            "status": "pending",
            "identifier": {"type": "dns", "value": "example.com"},
            "challenges": [
                {"type": "dns-01", "status": "pending", "url": "https://acme.test/c/1", "token": "t1"},
            ],
        }
        auth = Authorization.from_dict(data, "https://acme.test/authz/1")
        challenge = auth.get_challenge(ChallengeType.HTTP)
        assert challenge is None

    def test_status_properties(self):
        """Test authorization status properties."""
        base_data = {
            "identifier": {"type": "dns", "value": "example.com"},
            "challenges": [],
        }

        pending = Authorization.from_dict({**base_data, "status": "pending"}, "url1")
        valid = Authorization.from_dict({**base_data, "status": "valid"}, "url2")
        invalid = Authorization.from_dict({**base_data, "status": "invalid"}, "url3")

        assert pending.is_pending
        assert valid.is_valid
        assert invalid.is_invalid

    def test_domain_property(self):
        """Test domain property."""
        data = {
            "status": "pending",
            "identifier": {"type": "dns", "value": "test.example.com"},
            "challenges": [],
        }
        auth = Authorization.from_dict(data, "https://acme.test/authz/1")
        assert auth.domain == "test.example.com"

    def test_wildcard_authorization(self):
        """Test authorization for wildcard domain."""
        data = {
            "status": "pending",
            "identifier": {"type": "dns", "value": "example.com"},
            "wildcard": True,
            "challenges": [
                {"type": "dns-01", "status": "pending", "url": "https://acme.test/c/1", "token": "t1"},
            ],
        }
        auth = Authorization.from_dict(data, "https://acme.test/authz/1")
        assert auth.wildcard is True
