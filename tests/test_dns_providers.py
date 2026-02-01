"""Tests for the DNS provider system."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest

from acmeow.dns import (
    DnsProvider,
    DnsProviderHandler,
    DnsRecord,
    get_dns_provider,
    is_dns_provider_available,
    list_dns_providers,
    register_dns_provider,
    unregister_dns_provider,
)
from acmeow.dns.manual import ManualDnsProvider


class TestDnsRecord:
    """Tests for DnsRecord dataclass."""

    def test_create_record(self) -> None:
        """Test creating a DNS record."""
        record = DnsRecord(
            name="_acme-challenge.example.com",
            type="TXT",
            value="test-value",
            ttl=300,
            id="record-123",
        )

        assert record.name == "_acme-challenge.example.com"
        assert record.type == "TXT"
        assert record.value == "test-value"
        assert record.ttl == 300
        assert record.id == "record-123"

    def test_record_defaults(self) -> None:
        """Test record default values."""
        record = DnsRecord(
            name="test.example.com",
            type="TXT",
            value="value",
        )

        assert record.ttl == 300
        assert record.id is None

    def test_record_str(self) -> None:
        """Test record string representation."""
        record = DnsRecord(
            name="_acme-challenge.example.com",
            type="TXT",
            value="test-value",
            ttl=60,
        )

        s = str(record)
        assert "_acme-challenge.example.com" in s
        assert "TXT" in s
        assert "test-value" in s

    def test_record_is_frozen(self) -> None:
        """Test that record is immutable."""
        record = DnsRecord(name="test", type="TXT", value="val")

        with pytest.raises(AttributeError):
            record.name = "other"  # type: ignore


class TestDnsProviderRegistry:
    """Tests for the DNS provider registry."""

    def test_list_providers(self) -> None:
        """Test listing available providers."""
        providers = list_dns_providers()

        assert isinstance(providers, list)
        assert "manual" in providers
        assert "cloudflare" in providers
        assert "route53" in providers
        assert "digitalocean" in providers

    def test_is_provider_available(self) -> None:
        """Test checking provider availability."""
        assert is_dns_provider_available("manual") is True
        assert is_dns_provider_available("cloudflare") is True
        assert is_dns_provider_available("nonexistent") is False

    def test_get_manual_provider(self) -> None:
        """Test getting manual provider."""
        provider = get_dns_provider("manual")

        assert isinstance(provider, ManualDnsProvider)

    def test_get_provider_case_insensitive(self) -> None:
        """Test that provider names are case-insensitive."""
        provider = get_dns_provider("MANUAL")
        assert isinstance(provider, ManualDnsProvider)

        provider = get_dns_provider("Manual")
        assert isinstance(provider, ManualDnsProvider)

    def test_get_unknown_provider(self) -> None:
        """Test getting an unknown provider raises error."""
        with pytest.raises(ValueError, match="Unknown DNS provider"):
            get_dns_provider("nonexistent")

    def test_register_custom_provider(self) -> None:
        """Test registering a custom provider."""

        class CustomProvider(DnsProvider):
            def create_record(self, domain: str, name: str, value: str, ttl: int = 300) -> DnsRecord:
                return DnsRecord(name, "TXT", value, ttl)

            def delete_record(self, domain: str, record: DnsRecord) -> None:
                pass

        register_dns_provider("custom", CustomProvider)

        try:
            provider = get_dns_provider("custom")
            assert isinstance(provider, CustomProvider)
        finally:
            unregister_dns_provider("custom")

    def test_unregister_provider(self) -> None:
        """Test unregistering a provider."""

        class TempProvider(DnsProvider):
            def create_record(self, domain: str, name: str, value: str, ttl: int = 300) -> DnsRecord:
                return DnsRecord(name, "TXT", value, ttl)

            def delete_record(self, domain: str, record: DnsRecord) -> None:
                pass

        register_dns_provider("temp", TempProvider)
        result = unregister_dns_provider("temp")

        assert result is True
        assert is_dns_provider_available("temp") is False

    def test_unregister_nonexistent(self) -> None:
        """Test unregistering a provider that doesn't exist."""
        result = unregister_dns_provider("nonexistent_xyz")
        assert result is False


class TestManualDnsProvider:
    """Tests for ManualDnsProvider."""

    def test_create_record(self) -> None:
        """Test creating a record with manual provider."""
        input_stream = io.StringIO("\n")  # Simulate Enter key
        output_stream = io.StringIO()

        provider = ManualDnsProvider(
            input_stream=input_stream,
            output_stream=output_stream,
        )

        record = provider.create_record(
            domain="example.com",
            name="_acme-challenge.example.com",
            value="test-value",
            ttl=300,
        )

        assert record.name == "_acme-challenge.example.com"
        assert record.value == "test-value"
        assert record.id is not None

        # Check output
        output = output_stream.getvalue()
        assert "example.com" in output
        assert "_acme-challenge.example.com" in output
        assert "test-value" in output

    def test_delete_record(self) -> None:
        """Test deleting a record with manual provider."""
        input_stream = io.StringIO("")
        output_stream = io.StringIO()

        provider = ManualDnsProvider(
            input_stream=input_stream,
            output_stream=output_stream,
            prompt_delete=False,
        )

        record = DnsRecord(
            name="_acme-challenge.example.com",
            type="TXT",
            value="test-value",
            id="123",
        )

        # Should not raise
        provider.delete_record("example.com", record)

        output = output_stream.getvalue()
        assert "_acme-challenge.example.com" in output

    def test_propagation_delay(self) -> None:
        """Test manual provider has longer propagation delay."""
        provider = ManualDnsProvider()
        assert provider.propagation_delay == 120

    def test_list_records_not_supported(self) -> None:
        """Test that list_records raises NotImplementedError."""
        provider = ManualDnsProvider()

        with pytest.raises(NotImplementedError):
            provider.list_records("example.com")

    def test_quit_on_q(self) -> None:
        """Test quitting with 'q' raises KeyboardInterrupt."""
        input_stream = io.StringIO("q\n")
        output_stream = io.StringIO()

        provider = ManualDnsProvider(
            input_stream=input_stream,
            output_stream=output_stream,
        )

        with pytest.raises(KeyboardInterrupt):
            provider.create_record("example.com", "_acme-challenge.example.com", "value")


class TestDnsProviderHandler:
    """Tests for DnsProviderHandler."""

    def test_setup_calls_provider(self) -> None:
        """Test that setup calls provider's create_record."""
        mock_provider = MagicMock(spec=DnsProvider)
        mock_provider.create_record.return_value = DnsRecord(
            name="_acme-challenge.example.com",
            type="TXT",
            value="computed-value",
            id="123",
        )
        mock_provider.propagation_delay = 30
        mock_provider.get_zone_for_domain.return_value = "example.com"

        handler = DnsProviderHandler(mock_provider)

        handler.setup("example.com", "token", "key.auth")

        mock_provider.create_record.assert_called_once()
        call_args = mock_provider.create_record.call_args
        assert call_args[1]["name"] == "_acme-challenge.example.com"
        assert call_args[1]["domain"] == "example.com"

    def test_cleanup_calls_provider(self) -> None:
        """Test that cleanup calls provider's delete_record."""
        mock_provider = MagicMock(spec=DnsProvider)
        record = DnsRecord(
            name="_acme-challenge.example.com",
            type="TXT",
            value="value",
            id="123",
        )
        mock_provider.create_record.return_value = record
        mock_provider.propagation_delay = 30

        handler = DnsProviderHandler(mock_provider)

        handler.setup("example.com", "token", "key.auth")
        handler.cleanup("example.com", "token")

        mock_provider.delete_record.assert_called_once()

    def test_propagation_delay(self) -> None:
        """Test that propagation delay comes from provider."""
        mock_provider = MagicMock(spec=DnsProvider)
        mock_provider.propagation_delay = 120

        handler = DnsProviderHandler(mock_provider)

        assert handler.propagation_delay == 120

    def test_record_value_is_hashed(self) -> None:
        """Test that the TXT record value is properly hashed."""
        mock_provider = MagicMock(spec=DnsProvider)
        mock_provider.create_record.return_value = DnsRecord(
            name="_acme-challenge.example.com",
            type="TXT",
            value="hashed",
            id="123",
        )

        handler = DnsProviderHandler(mock_provider)
        handler.setup("example.com", "token123", "token123.thumbprint")

        # The value should be base64url-encoded SHA-256 hash
        call_args = mock_provider.create_record.call_args
        value = call_args[1]["value"]

        # Base64url encoded SHA-256 is 43 characters
        assert len(value) == 43

    def test_subdomain_handling(self) -> None:
        """Test handling of subdomains."""
        mock_provider = MagicMock(spec=DnsProvider)
        mock_provider.create_record.return_value = DnsRecord(
            name="_acme-challenge.sub.example.com",
            type="TXT",
            value="value",
            id="123",
        )
        mock_provider.get_zone_for_domain.side_effect = NotImplementedError()

        handler = DnsProviderHandler(mock_provider)
        handler.setup("sub.example.com", "token", "key.auth")

        call_args = mock_provider.create_record.call_args
        # Should pass base domain to provider
        assert call_args[1]["domain"] == "example.com"

    def test_wildcard_handling(self) -> None:
        """Test handling of wildcard domains."""
        mock_provider = MagicMock(spec=DnsProvider)
        mock_provider.create_record.return_value = DnsRecord(
            name="_acme-challenge.example.com",
            type="TXT",
            value="value",
            id="123",
        )
        mock_provider.get_zone_for_domain.side_effect = NotImplementedError()

        handler = DnsProviderHandler(mock_provider)
        handler.setup("*.example.com", "token", "key.auth")

        call_args = mock_provider.create_record.call_args
        # Should strip wildcard for base domain
        assert call_args[1]["domain"] == "example.com"


class TestCloudflareDnsProvider:
    """Tests for CloudflareDnsProvider."""

    def test_init(self) -> None:
        """Test provider initialization."""
        from acmeow.dns.cloudflare import CloudflareDnsProvider

        provider = CloudflareDnsProvider(api_token="test-token", zone_id="zone123")
        assert provider._api_token == "test-token"
        assert provider._zone_id == "zone123"
        assert provider.propagation_delay == 30

    def test_str(self) -> None:
        """Test string representation."""
        from acmeow.dns.cloudflare import CloudflareDnsProvider

        provider = CloudflareDnsProvider(api_token="test", zone_id="zone123")
        assert "zone123" in str(provider)

    def test_get_zone_id_with_explicit_zone(self) -> None:
        """Test getting zone ID when explicitly provided."""
        from acmeow.dns.cloudflare import CloudflareDnsProvider

        provider = CloudflareDnsProvider(api_token="test", zone_id="explicit-zone")
        assert provider._get_zone_id("example.com") == "explicit-zone"

    def test_get_zone_id_from_cache(self) -> None:
        """Test getting zone ID from cache."""
        from acmeow.dns.cloudflare import CloudflareDnsProvider

        provider = CloudflareDnsProvider(api_token="test")
        provider._zone_cache["example.com"] = "cached-zone"
        assert provider._get_zone_id("example.com") == "cached-zone"

    def test_get_zone_id_lookup(self) -> None:
        """Test looking up zone ID from API."""
        from unittest.mock import patch, MagicMock
        from acmeow.dns.cloudflare import CloudflareDnsProvider

        provider = CloudflareDnsProvider(api_token="test")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "result": [{"id": "found-zone-id"}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider._session, "get", return_value=mock_response):
            zone_id = provider._get_zone_id("sub.example.com")
            assert zone_id == "found-zone-id"

    def test_get_zone_id_not_found(self) -> None:
        """Test zone ID lookup failure."""
        from unittest.mock import patch, MagicMock
        from acmeow.dns.cloudflare import CloudflareDnsProvider

        provider = CloudflareDnsProvider(api_token="test")

        mock_response = MagicMock()
        mock_response.json.return_value = {"success": True, "result": []}
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider._session, "get", return_value=mock_response):
            with pytest.raises(ValueError, match="Could not find Cloudflare zone"):
                provider._get_zone_id("example.com")

    def test_create_record(self) -> None:
        """Test creating a DNS record."""
        from unittest.mock import patch, MagicMock
        from acmeow.dns.cloudflare import CloudflareDnsProvider

        provider = CloudflareDnsProvider(api_token="test", zone_id="zone123")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "result": {"id": "record-id-123"}
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider._session, "post", return_value=mock_response):
            record = provider.create_record(
                domain="example.com",
                name="_acme-challenge.example.com",
                value="test-value",
                ttl=300,
            )

            assert record.name == "_acme-challenge.example.com"
            assert record.value == "test-value"
            assert record.id == "record-id-123"

    def test_create_record_api_error(self) -> None:
        """Test handling API error on create."""
        from unittest.mock import patch, MagicMock
        from acmeow.dns.cloudflare import CloudflareDnsProvider

        provider = CloudflareDnsProvider(api_token="test", zone_id="zone123")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": False,
            "errors": [{"message": "Invalid API token"}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider._session, "post", return_value=mock_response):
            with pytest.raises(RuntimeError, match="Cloudflare API error"):
                provider.create_record("example.com", "_acme.example.com", "value")

    def test_delete_record(self) -> None:
        """Test deleting a DNS record."""
        from unittest.mock import patch, MagicMock
        from acmeow.dns.cloudflare import CloudflareDnsProvider

        provider = CloudflareDnsProvider(api_token="test", zone_id="zone123")

        mock_response = MagicMock()
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status = MagicMock()

        record = DnsRecord(name="_acme.example.com", type="TXT", value="val", id="rec123")

        with patch.object(provider._session, "delete", return_value=mock_response):
            provider.delete_record("example.com", record)

    def test_delete_record_no_id(self) -> None:
        """Test deleting record without ID raises error."""
        from acmeow.dns.cloudflare import CloudflareDnsProvider

        provider = CloudflareDnsProvider(api_token="test", zone_id="zone123")
        record = DnsRecord(name="_acme.example.com", type="TXT", value="val")

        with pytest.raises(ValueError, match="does not have a Cloudflare ID"):
            provider.delete_record("example.com", record)

    def test_delete_record_api_error(self) -> None:
        """Test handling API error on delete."""
        from unittest.mock import patch, MagicMock
        from acmeow.dns.cloudflare import CloudflareDnsProvider

        provider = CloudflareDnsProvider(api_token="test", zone_id="zone123")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": False,
            "errors": [{"message": "Record not found"}]
        }
        mock_response.raise_for_status = MagicMock()

        record = DnsRecord(name="_acme.example.com", type="TXT", value="val", id="rec123")

        with patch.object(provider._session, "delete", return_value=mock_response):
            with pytest.raises(RuntimeError, match="Cloudflare API error"):
                provider.delete_record("example.com", record)

    def test_list_records(self) -> None:
        """Test listing DNS records."""
        from unittest.mock import patch, MagicMock
        from acmeow.dns.cloudflare import CloudflareDnsProvider

        provider = CloudflareDnsProvider(api_token="test", zone_id="zone123")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "result": [
                {"name": "example.com", "type": "A", "content": "1.2.3.4", "ttl": 300, "id": "r1"},
                {"name": "_acme.example.com", "type": "TXT", "content": "val", "ttl": 60, "id": "r2"},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider._session, "get", return_value=mock_response):
            records = provider.list_records("example.com", record_type="TXT")
            assert len(records) == 2

    def test_list_records_api_error(self) -> None:
        """Test handling API error on list."""
        from unittest.mock import patch, MagicMock
        from acmeow.dns.cloudflare import CloudflareDnsProvider

        provider = CloudflareDnsProvider(api_token="test", zone_id="zone123")

        mock_response = MagicMock()
        mock_response.json.return_value = {"success": False, "errors": [{"message": "Error"}]}
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider._session, "get", return_value=mock_response):
            with pytest.raises(RuntimeError, match="Cloudflare API error"):
                provider.list_records("example.com")

    def test_get_zone_for_domain(self) -> None:
        """Test getting zone name for domain."""
        from unittest.mock import patch, MagicMock
        from acmeow.dns.cloudflare import CloudflareDnsProvider

        provider = CloudflareDnsProvider(api_token="test", zone_id="zone123")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "result": {"name": "example.com"}
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider._session, "get", return_value=mock_response):
            zone = provider.get_zone_for_domain("sub.example.com")
            assert zone == "example.com"

    def test_close(self) -> None:
        """Test closing the provider."""
        from unittest.mock import patch
        from acmeow.dns.cloudflare import CloudflareDnsProvider

        provider = CloudflareDnsProvider(api_token="test")
        with patch.object(provider._session, "close") as mock_close:
            provider.close()
            mock_close.assert_called_once()


class TestDigitalOceanDnsProvider:
    """Tests for DigitalOceanDnsProvider."""

    def test_init(self) -> None:
        """Test provider initialization."""
        from acmeow.dns.digitalocean import DigitalOceanDnsProvider

        provider = DigitalOceanDnsProvider(api_token="test-token")
        assert provider._api_token == "test-token"
        assert provider.propagation_delay == 60

    def test_str(self) -> None:
        """Test string representation."""
        from acmeow.dns.digitalocean import DigitalOceanDnsProvider

        provider = DigitalOceanDnsProvider(api_token="test")
        assert "DigitalOceanDnsProvider" in str(provider)

    def test_get_domain_name_from_cache(self) -> None:
        """Test getting domain name from cache."""
        from acmeow.dns.digitalocean import DigitalOceanDnsProvider

        provider = DigitalOceanDnsProvider(api_token="test")
        provider._domain_cache["example.com"] = "example.com"
        assert provider._get_domain_name("example.com") == "example.com"

    def test_get_domain_name_lookup(self) -> None:
        """Test looking up domain name from API."""
        from unittest.mock import patch, MagicMock
        from acmeow.dns.digitalocean import DigitalOceanDnsProvider

        provider = DigitalOceanDnsProvider(api_token="test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"domain": {"name": "example.com"}}

        with patch.object(provider._session, "get", return_value=mock_response):
            domain = provider._get_domain_name("sub.example.com")
            assert domain == "example.com"

    def test_get_domain_name_not_found(self) -> None:
        """Test domain name lookup failure."""
        from unittest.mock import patch, MagicMock
        from acmeow.dns.digitalocean import DigitalOceanDnsProvider

        provider = DigitalOceanDnsProvider(api_token="test")

        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch.object(provider._session, "get", return_value=mock_response):
            with pytest.raises(ValueError, match="Could not find DigitalOcean domain"):
                provider._get_domain_name("example.com")

    def test_get_record_name(self) -> None:
        """Test getting relative record name."""
        from acmeow.dns.digitalocean import DigitalOceanDnsProvider

        provider = DigitalOceanDnsProvider(api_token="test")
        assert provider._get_record_name("_acme.sub.example.com", "example.com") == "_acme.sub"
        assert provider._get_record_name("other", "example.com") == "other"

    def test_create_record(self) -> None:
        """Test creating a DNS record."""
        from unittest.mock import patch, MagicMock
        from acmeow.dns.digitalocean import DigitalOceanDnsProvider

        provider = DigitalOceanDnsProvider(api_token="test")
        provider._domain_cache["example.com"] = "example.com"

        mock_response = MagicMock()
        mock_response.json.return_value = {"domain_record": {"id": 12345}}
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider._session, "post", return_value=mock_response):
            record = provider.create_record(
                domain="example.com",
                name="_acme-challenge.example.com",
                value="test-value",
            )

            assert record.name == "_acme-challenge.example.com"
            assert record.id == "12345"

    def test_create_record_no_id(self) -> None:
        """Test handling missing record ID on create."""
        from unittest.mock import patch, MagicMock
        from acmeow.dns.digitalocean import DigitalOceanDnsProvider

        provider = DigitalOceanDnsProvider(api_token="test")
        provider._domain_cache["example.com"] = "example.com"

        mock_response = MagicMock()
        mock_response.json.return_value = {"domain_record": {}}
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider._session, "post", return_value=mock_response):
            with pytest.raises(RuntimeError, match="did not return record ID"):
                provider.create_record("example.com", "_acme.example.com", "value")

    def test_delete_record(self) -> None:
        """Test deleting a DNS record."""
        from unittest.mock import patch, MagicMock
        from acmeow.dns.digitalocean import DigitalOceanDnsProvider

        provider = DigitalOceanDnsProvider(api_token="test")
        provider._domain_cache["example.com"] = "example.com"

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        record = DnsRecord(name="_acme.example.com", type="TXT", value="val", id="123")

        with patch.object(provider._session, "delete", return_value=mock_response):
            provider.delete_record("example.com", record)

    def test_delete_record_no_id(self) -> None:
        """Test deleting record without ID raises error."""
        from acmeow.dns.digitalocean import DigitalOceanDnsProvider

        provider = DigitalOceanDnsProvider(api_token="test")
        record = DnsRecord(name="_acme.example.com", type="TXT", value="val")

        with pytest.raises(ValueError, match="does not have a DigitalOcean ID"):
            provider.delete_record("example.com", record)

    def test_list_records(self) -> None:
        """Test listing DNS records."""
        from unittest.mock import patch, MagicMock
        from acmeow.dns.digitalocean import DigitalOceanDnsProvider

        provider = DigitalOceanDnsProvider(api_token="test")
        provider._domain_cache["example.com"] = "example.com"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "domain_records": [
                {"name": "_acme", "type": "TXT", "data": "val", "ttl": 60, "id": 1},
                {"name": "@", "type": "A", "data": "1.2.3.4", "ttl": 300, "id": 2},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider._session, "get", return_value=mock_response):
            records = provider.list_records("example.com")
            assert len(records) == 2
            assert records[0].name == "_acme.example.com"
            assert records[1].name == "example.com"

    def test_list_records_with_type_filter(self) -> None:
        """Test listing records with type filter."""
        from unittest.mock import patch, MagicMock
        from acmeow.dns.digitalocean import DigitalOceanDnsProvider

        provider = DigitalOceanDnsProvider(api_token="test")
        provider._domain_cache["example.com"] = "example.com"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "domain_records": [
                {"name": "_acme", "type": "TXT", "data": "val", "ttl": 60, "id": 1},
                {"name": "@", "type": "A", "data": "1.2.3.4", "ttl": 300, "id": 2},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider._session, "get", return_value=mock_response):
            records = provider.list_records("example.com", record_type="TXT")
            assert len(records) == 1

    def test_get_zone_for_domain(self) -> None:
        """Test getting zone for domain."""
        from acmeow.dns.digitalocean import DigitalOceanDnsProvider

        provider = DigitalOceanDnsProvider(api_token="test")
        provider._domain_cache["example.com"] = "example.com"
        assert provider.get_zone_for_domain("example.com") == "example.com"

    def test_close(self) -> None:
        """Test closing the provider."""
        from unittest.mock import patch
        from acmeow.dns.digitalocean import DigitalOceanDnsProvider

        provider = DigitalOceanDnsProvider(api_token="test")
        with patch.object(provider._session, "close") as mock_close:
            provider.close()
            mock_close.assert_called_once()


class TestRoute53DnsProvider:
    """Tests for Route53DnsProvider.

    These tests require boto3 to be installed. They are skipped if boto3
    is not available.
    """

    def test_init_with_boto3(self) -> None:
        """Test provider initialization with boto3."""
        pytest.importorskip("boto3")
        from acmeow.dns.route53 import Route53DnsProvider

        mock_client = MagicMock()
        with patch("boto3.client", return_value=mock_client):
            provider = Route53DnsProvider(hosted_zone_id="Z123")
            assert provider._hosted_zone_id == "Z123"
            assert provider.propagation_delay == 60

    def test_str(self) -> None:
        """Test string representation."""
        pytest.importorskip("boto3")
        from acmeow.dns.route53 import Route53DnsProvider

        mock_client = MagicMock()
        with patch("boto3.client", return_value=mock_client):
            provider = Route53DnsProvider(hosted_zone_id="Z123")
            assert "Z123" in str(provider)

    def test_get_hosted_zone_id_explicit(self) -> None:
        """Test getting zone ID when explicitly provided."""
        pytest.importorskip("boto3")
        from acmeow.dns.route53 import Route53DnsProvider

        mock_client = MagicMock()
        with patch("boto3.client", return_value=mock_client):
            provider = Route53DnsProvider(hosted_zone_id="explicit-zone")
            assert provider._get_hosted_zone_id("example.com") == "explicit-zone"

    def test_get_hosted_zone_id_from_cache(self) -> None:
        """Test getting zone ID from cache."""
        pytest.importorskip("boto3")
        from acmeow.dns.route53 import Route53DnsProvider

        mock_client = MagicMock()
        with patch("boto3.client", return_value=mock_client):
            provider = Route53DnsProvider()
            provider._zone_cache["example.com"] = "cached-zone"
            assert provider._get_hosted_zone_id("example.com") == "cached-zone"

    def test_get_hosted_zone_id_lookup(self) -> None:
        """Test looking up zone ID from API."""
        pytest.importorskip("boto3")
        from acmeow.dns.route53 import Route53DnsProvider

        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {"HostedZones": [{"Id": "/hostedzone/Z123ABC", "Name": "example.com."}]}
        ]
        mock_client.get_paginator.return_value = mock_paginator

        with patch("boto3.client", return_value=mock_client):
            provider = Route53DnsProvider()
            zone_id = provider._get_hosted_zone_id("sub.example.com")
            assert zone_id == "Z123ABC"

    def test_get_hosted_zone_id_not_found(self) -> None:
        """Test zone ID lookup failure."""
        pytest.importorskip("boto3")
        from acmeow.dns.route53 import Route53DnsProvider

        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"HostedZones": []}]
        mock_client.get_paginator.return_value = mock_paginator

        with patch("boto3.client", return_value=mock_client):
            provider = Route53DnsProvider()
            with pytest.raises(ValueError, match="Could not find Route53 hosted zone"):
                provider._get_hosted_zone_id("example.com")

    def test_create_record(self) -> None:
        """Test creating a DNS record."""
        pytest.importorskip("boto3")
        from acmeow.dns.route53 import Route53DnsProvider

        mock_client = MagicMock()
        mock_client.change_resource_record_sets.return_value = {
            "ChangeInfo": {"Id": "change-123", "Status": "PENDING"}
        }
        mock_client.get_change.return_value = {"ChangeInfo": {"Status": "INSYNC"}}

        with patch("boto3.client", return_value=mock_client):
            provider = Route53DnsProvider(hosted_zone_id="Z123", wait_for_change=True)
            record = provider.create_record(
                domain="example.com",
                name="_acme-challenge.example.com",
                value="test-value",
            )

            assert record.name == "_acme-challenge.example.com"
            assert record.value == "test-value"
            mock_client.change_resource_record_sets.assert_called_once()

    def test_create_record_no_wait(self) -> None:
        """Test creating a record without waiting."""
        pytest.importorskip("boto3")
        from acmeow.dns.route53 import Route53DnsProvider

        mock_client = MagicMock()
        mock_client.change_resource_record_sets.return_value = {
            "ChangeInfo": {"Id": "change-123"}
        }

        with patch("boto3.client", return_value=mock_client):
            provider = Route53DnsProvider(hosted_zone_id="Z123", wait_for_change=False)
            provider.create_record("example.com", "_acme.example.com", "value")
            mock_client.get_change.assert_not_called()

    def test_delete_record(self) -> None:
        """Test deleting a DNS record."""
        pytest.importorskip("boto3")
        from acmeow.dns.route53 import Route53DnsProvider

        mock_client = MagicMock()
        mock_client.change_resource_record_sets.return_value = {
            "ChangeInfo": {"Id": "change-456"}
        }

        with patch("boto3.client", return_value=mock_client):
            provider = Route53DnsProvider(hosted_zone_id="Z123")
            record = DnsRecord(name="_acme.example.com", type="TXT", value="val", ttl=300)
            provider.delete_record("example.com", record)
            mock_client.change_resource_record_sets.assert_called_once()

    def test_list_records(self) -> None:
        """Test listing DNS records."""
        pytest.importorskip("boto3")
        from acmeow.dns.route53 import Route53DnsProvider

        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "ResourceRecordSets": [
                    {
                        "Name": "_acme.example.com.",
                        "Type": "TXT",
                        "TTL": 60,
                        "ResourceRecords": [{"Value": '"test-value"'}]
                    },
                    {
                        "Name": "example.com.",
                        "Type": "A",
                        "TTL": 300,
                        "ResourceRecords": [{"Value": "1.2.3.4"}]
                    },
                ]
            }
        ]
        mock_client.get_paginator.return_value = mock_paginator

        with patch("boto3.client", return_value=mock_client):
            provider = Route53DnsProvider(hosted_zone_id="Z123")
            records = provider.list_records("example.com")
            assert len(records) == 2
            # TXT value should have quotes stripped
            assert records[0].value == "test-value"

    def test_list_records_with_type_filter(self) -> None:
        """Test listing records with type filter."""
        pytest.importorskip("boto3")
        from acmeow.dns.route53 import Route53DnsProvider

        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "ResourceRecordSets": [
                    {"Name": "_acme.example.com.", "Type": "TXT", "TTL": 60, "ResourceRecords": [{"Value": '"v"'}]},
                    {"Name": "example.com.", "Type": "A", "TTL": 300, "ResourceRecords": [{"Value": "1.2.3.4"}]},
                ]
            }
        ]
        mock_client.get_paginator.return_value = mock_paginator

        with patch("boto3.client", return_value=mock_client):
            provider = Route53DnsProvider(hosted_zone_id="Z123")
            records = provider.list_records("example.com", record_type="TXT")
            assert len(records) == 1

    def test_get_zone_for_domain(self) -> None:
        """Test getting zone name for domain."""
        pytest.importorskip("boto3")
        from acmeow.dns.route53 import Route53DnsProvider

        mock_client = MagicMock()
        mock_client.get_hosted_zone.return_value = {
            "HostedZone": {"Name": "example.com."}
        }

        with patch("boto3.client", return_value=mock_client):
            provider = Route53DnsProvider(hosted_zone_id="Z123")
            zone = provider.get_zone_for_domain("sub.example.com")
            assert zone == "example.com"

    def test_wait_for_change_sync(self) -> None:
        """Test waiting for change to sync."""
        pytest.importorskip("boto3")
        from acmeow.dns.route53 import Route53DnsProvider

        mock_client = MagicMock()
        mock_client.get_change.side_effect = [
            {"ChangeInfo": {"Status": "PENDING"}},
            {"ChangeInfo": {"Status": "INSYNC"}},
        ]

        with patch("boto3.client", return_value=mock_client):
            with patch("time.sleep"):
                provider = Route53DnsProvider(hosted_zone_id="Z123")
                provider._wait_for_change_sync("change-123")
                assert mock_client.get_change.call_count == 2

    def test_init_with_credentials(self) -> None:
        """Test initialization with explicit credentials."""
        pytest.importorskip("boto3")
        from acmeow.dns.route53 import Route53DnsProvider

        mock_client = MagicMock()
        with patch("boto3.client", return_value=mock_client) as mock_boto:
            Route53DnsProvider(
                aws_access_key_id="AKIATEST",
                aws_secret_access_key="secret123",
                region_name="eu-west-1",
            )
            mock_boto.assert_called_once()
            call_kwargs = mock_boto.call_args[1]
            assert call_kwargs["aws_access_key_id"] == "AKIATEST"
            assert call_kwargs["aws_secret_access_key"] == "secret123"
            assert call_kwargs["region_name"] == "eu-west-1"
