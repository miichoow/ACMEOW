"""Tests for exception classes."""

from __future__ import annotations

import pytest

from acmeow.exceptions import (
    AcmeAuthenticationError,
    AcmeAuthorizationError,
    AcmeCertificateError,
    AcmeConfigurationError,
    AcmeDnsError,
    AcmeError,
    AcmeNetworkError,
    AcmeOrderError,
    AcmeRateLimitError,
    AcmeServerError,
    AcmeTimeoutError,
)


class TestAcmeError:
    """Tests for base AcmeError."""

    def test_message_attribute(self):
        """Test that message is stored as attribute."""
        error = AcmeError("Test error message")
        assert error.message == "Test error message"

    def test_str_representation(self):
        """Test string representation."""
        error = AcmeError("Test error")
        assert str(error) == "Test error"

    def test_inheritance(self):
        """Test that AcmeError inherits from Exception."""
        assert issubclass(AcmeError, Exception)

    def test_catchable_as_exception(self):
        """Test that AcmeError can be caught as Exception."""
        with pytest.raises(Exception):
            raise AcmeError("test")


class TestAcmeServerError:
    """Tests for AcmeServerError."""

    def test_attributes(self):
        """Test that all attributes are stored."""
        error = AcmeServerError(
            status_code=400,
            error_type="urn:ietf:params:acme:error:malformed",
            detail="The request was malformed",
        )
        assert error.status_code == 400
        assert error.error_type == "urn:ietf:params:acme:error:malformed"
        assert error.detail == "The request was malformed"

    def test_message_format(self):
        """Test message format includes all info."""
        error = AcmeServerError(403, "unauthorized", "Not authorized")
        assert "403" in str(error)
        assert "unauthorized" in str(error)
        assert "Not authorized" in str(error)

    def test_inheritance(self):
        """Test inheritance from AcmeError."""
        assert issubclass(AcmeServerError, AcmeError)

    def test_catchable_as_acme_error(self):
        """Test that AcmeServerError can be caught as AcmeError."""
        with pytest.raises(AcmeError):
            raise AcmeServerError(500, "internal", "Internal error")


class TestAcmeAuthenticationError:
    """Tests for AcmeAuthenticationError."""

    def test_basic_error(self):
        """Test basic authentication error."""
        error = AcmeAuthenticationError("Invalid account key")
        assert error.message == "Invalid account key"

    def test_inheritance(self):
        """Test inheritance from AcmeError."""
        assert issubclass(AcmeAuthenticationError, AcmeError)


class TestAcmeAuthorizationError:
    """Tests for AcmeAuthorizationError."""

    def test_domain_attribute(self):
        """Test that domain is stored."""
        error = AcmeAuthorizationError("example.com", "DNS validation failed")
        assert error.domain == "example.com"

    def test_message_includes_domain(self):
        """Test that message includes domain."""
        error = AcmeAuthorizationError("test.example.com", "Challenge failed")
        assert "test.example.com" in str(error)
        assert "Challenge failed" in str(error)

    def test_inheritance(self):
        """Test inheritance from AcmeError."""
        assert issubclass(AcmeAuthorizationError, AcmeError)


class TestAcmeOrderError:
    """Tests for AcmeOrderError."""

    def test_basic_error(self):
        """Test basic order error."""
        error = AcmeOrderError("Order creation failed")
        assert error.message == "Order creation failed"

    def test_inheritance(self):
        """Test inheritance from AcmeError."""
        assert issubclass(AcmeOrderError, AcmeError)


class TestAcmeCertificateError:
    """Tests for AcmeCertificateError."""

    def test_basic_error(self):
        """Test basic certificate error."""
        error = AcmeCertificateError("Certificate download failed")
        assert error.message == "Certificate download failed"

    def test_inheritance(self):
        """Test inheritance from AcmeError."""
        assert issubclass(AcmeCertificateError, AcmeError)


class TestAcmeConfigurationError:
    """Tests for AcmeConfigurationError."""

    def test_basic_error(self):
        """Test basic configuration error."""
        error = AcmeConfigurationError("Invalid email address")
        assert error.message == "Invalid email address"

    def test_inheritance(self):
        """Test inheritance from AcmeError."""
        assert issubclass(AcmeConfigurationError, AcmeError)


class TestAcmeNetworkError:
    """Tests for AcmeNetworkError."""

    def test_basic_error(self):
        """Test basic network error."""
        error = AcmeNetworkError("Connection refused")
        assert error.message == "Connection refused"
        assert error.original_error is None

    def test_with_original_error(self):
        """Test network error with original exception."""
        original = ConnectionError("Socket closed")
        error = AcmeNetworkError("Connection failed", original)
        assert error.original_error is original

    def test_original_error_preserved(self):
        """Test that original error type is preserved."""
        original = TimeoutError("Read timed out")
        error = AcmeNetworkError("Timeout", original)
        assert isinstance(error.original_error, TimeoutError)

    def test_inheritance(self):
        """Test inheritance from AcmeError."""
        assert issubclass(AcmeNetworkError, AcmeError)


class TestAcmeTimeoutError:
    """Tests for AcmeTimeoutError."""

    def test_basic_error(self):
        """Test basic timeout error."""
        error = AcmeTimeoutError("Operation timed out after 300s")
        assert error.message == "Operation timed out after 300s"

    def test_inheritance(self):
        """Test inheritance from AcmeError."""
        assert issubclass(AcmeTimeoutError, AcmeError)


class TestAcmeRateLimitError:
    """Tests for AcmeRateLimitError."""

    def test_basic_error(self):
        """Test basic rate limit error."""
        error = AcmeRateLimitError("Too many requests")
        assert error.message == "Too many requests"
        assert error.retry_after is None

    def test_with_retry_after(self):
        """Test rate limit error with retry_after."""
        error = AcmeRateLimitError("Rate limited", retry_after=60.0)
        assert error.retry_after == 60.0

    def test_retry_after_float(self):
        """Test retry_after as float."""
        error = AcmeRateLimitError("Rate limited", retry_after=30.5)
        assert error.retry_after == 30.5

    def test_inheritance(self):
        """Test inheritance from AcmeError."""
        assert issubclass(AcmeRateLimitError, AcmeError)


class TestAcmeDnsError:
    """Tests for AcmeDnsError."""

    def test_domain_attribute(self):
        """Test that domain is stored."""
        error = AcmeDnsError("example.com", "Record not found")
        assert error.domain == "example.com"

    def test_message_includes_domain(self):
        """Test that message includes domain."""
        error = AcmeDnsError("test.example.com", "Propagation timeout")
        assert "test.example.com" in str(error)
        assert "Propagation timeout" in str(error)

    def test_inheritance(self):
        """Test inheritance from AcmeError."""
        assert issubclass(AcmeDnsError, AcmeError)


class TestExceptionHierarchy:
    """Tests for exception hierarchy behavior."""

    def test_catch_all_with_acme_error(self):
        """Test catching all ACME errors with base class."""
        exceptions = [
            AcmeError("base"),
            AcmeServerError(400, "type", "detail"),
            AcmeAuthenticationError("auth"),
            AcmeAuthorizationError("dom", "msg"),
            AcmeOrderError("order"),
            AcmeCertificateError("cert"),
            AcmeConfigurationError("config"),
            AcmeNetworkError("network"),
            AcmeTimeoutError("timeout"),
            AcmeRateLimitError("rate"),
            AcmeDnsError("dom", "dns"),
        ]

        for exc in exceptions:
            with pytest.raises(AcmeError):
                raise exc

    def test_specific_catch(self):
        """Test catching specific exception types."""
        with pytest.raises(AcmeServerError):
            raise AcmeServerError(400, "type", "detail")

        with pytest.raises(AcmeRateLimitError):
            raise AcmeRateLimitError("rate")

    def test_exception_chaining(self):
        """Test exception chaining with 'from'."""
        original = ValueError("Original error")
        try:
            try:
                raise original
            except ValueError as e:
                raise AcmeNetworkError("Wrapped error", e) from e
        except AcmeNetworkError as e:
            assert e.__cause__ is original
            assert e.original_error is original

    def test_exception_message_encoding(self):
        """Test exceptions handle unicode messages."""
        error = AcmeError("Error with unicode: éàü")
        assert "éàü" in str(error)

    def test_exception_repr(self):
        """Test exception repr for debugging."""
        error = AcmeServerError(400, "malformed", "Bad request")
        repr_str = repr(error)
        assert "AcmeServerError" in repr_str
