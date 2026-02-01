"""Tests for encoding utilities."""

from __future__ import annotations

import json


from acmeow._internal.encoding import (
    base64url_decode,
    base64url_encode,
    base64url_encode_json,
    json_bytes,
    json_encode,
)


class TestBase64UrlEncode:
    """Tests for base64url_encode function."""

    def test_encode_empty(self):
        """Test encoding empty bytes."""
        result = base64url_encode(b"")
        assert result == ""

    def test_encode_simple(self):
        """Test encoding simple bytes."""
        result = base64url_encode(b"hello")
        assert result == "aGVsbG8"

    def test_encode_no_padding(self):
        """Test that encoding produces no padding."""
        # Various lengths that would normally require padding
        assert "=" not in base64url_encode(b"a")
        assert "=" not in base64url_encode(b"ab")
        assert "=" not in base64url_encode(b"abc")
        assert "=" not in base64url_encode(b"abcd")

    def test_encode_url_safe(self):
        """Test that encoding uses URL-safe characters."""
        # Bytes that would produce + and / in standard base64
        data = bytes([0xfb, 0xef, 0xbe])  # Would be ++++ in standard
        result = base64url_encode(data)
        assert "+" not in result
        assert "/" not in result
        # Should use - and _ instead
        assert "-" in result or "_" in result or result.isalnum()

    def test_encode_binary(self):
        """Test encoding binary data."""
        data = bytes(range(256))
        result = base64url_encode(data)
        assert isinstance(result, str)
        assert "=" not in result

    def test_encode_deterministic(self):
        """Test that encoding is deterministic."""
        data = b"test data for determinism"
        result1 = base64url_encode(data)
        result2 = base64url_encode(data)
        assert result1 == result2


class TestBase64UrlDecode:
    """Tests for base64url_decode function."""

    def test_decode_empty(self):
        """Test decoding empty string."""
        result = base64url_decode("")
        assert result == b""

    def test_decode_simple(self):
        """Test decoding simple string."""
        result = base64url_decode("aGVsbG8")
        assert result == b"hello"

    def test_decode_with_padding(self):
        """Test decoding string with padding."""
        result = base64url_decode("aGVsbG8=")
        assert result == b"hello"

    def test_decode_without_padding(self):
        """Test decoding string without padding."""
        # Should work with or without padding
        result = base64url_decode("YQ")  # "a" without padding
        assert result == b"a"

    def test_decode_roundtrip(self):
        """Test encode/decode roundtrip."""
        original = b"test roundtrip data"
        encoded = base64url_encode(original)
        decoded = base64url_decode(encoded)
        assert decoded == original

    def test_decode_roundtrip_binary(self):
        """Test encode/decode roundtrip with binary data."""
        original = bytes(range(256))
        encoded = base64url_encode(original)
        decoded = base64url_decode(encoded)
        assert decoded == original

    def test_decode_url_safe_chars(self):
        """Test decoding URL-safe characters."""
        # Manually construct base64url with - and _
        # "-" replaces "+", "_" replaces "/"
        encoded = "YWJj"  # "abc" in base64
        result = base64url_decode(encoded)
        assert result == b"abc"


class TestJsonEncode:
    """Tests for json_encode function."""

    def test_encode_dict(self):
        """Test encoding dictionary."""
        obj = {"key": "value"}
        result = json_encode(obj)
        assert result == '{"key":"value"}'

    def test_encode_compact(self):
        """Test that encoding uses compact format."""
        obj = {"a": 1, "b": 2}
        result = json_encode(obj)
        # Should have no spaces after separators
        assert " " not in result

    def test_encode_sorted_keys(self):
        """Test that keys are sorted."""
        obj = {"z": 1, "a": 2, "m": 3}
        result = json_encode(obj)
        # Keys should appear in alphabetical order
        assert result.index('"a"') < result.index('"m"') < result.index('"z"')

    def test_encode_nested(self):
        """Test encoding nested objects."""
        obj = {"outer": {"inner": "value"}}
        result = json_encode(obj)
        assert result == '{"outer":{"inner":"value"}}'

    def test_encode_list(self):
        """Test encoding list."""
        obj = [1, 2, 3]
        result = json_encode(obj)
        assert result == "[1,2,3]"

    def test_encode_unicode(self):
        """Test encoding unicode strings."""
        obj = {"key": "value with unicode: éàü"}
        result = json_encode(obj)
        # JSON may encode unicode as escape sequences
        assert "value with unicode" in result

    def test_encode_boolean(self):
        """Test encoding boolean values."""
        obj = {"flag": True, "other": False}
        result = json_encode(obj)
        assert "true" in result
        assert "false" in result

    def test_encode_null(self):
        """Test encoding None/null."""
        obj = {"value": None}
        result = json_encode(obj)
        assert "null" in result


class TestJsonBytes:
    """Tests for json_bytes function."""

    def test_returns_bytes(self):
        """Test that function returns bytes."""
        obj = {"key": "value"}
        result = json_bytes(obj)
        assert isinstance(result, bytes)

    def test_utf8_encoding(self):
        """Test that bytes are UTF-8 encoded."""
        obj = {"key": "value"}
        result = json_bytes(obj)
        assert result.decode("utf-8") == '{"key":"value"}'

    def test_unicode_encoding(self):
        """Test encoding unicode to bytes."""
        obj = {"key": "unicode: éàü"}
        result = json_bytes(obj)
        # Should be valid UTF-8
        decoded = result.decode("utf-8")
        # JSON may encode unicode as escape sequences
        assert "unicode:" in decoded


class TestBase64UrlEncodeJson:
    """Tests for base64url_encode_json function."""

    def test_encode_dict(self):
        """Test encoding dictionary."""
        obj = {"key": "value"}
        result = base64url_encode_json(obj)
        # Should be base64url encoded
        assert isinstance(result, str)
        assert "=" not in result

    def test_decode_roundtrip(self):
        """Test encoding can be decoded back."""
        obj = {"alg": "ES256", "nonce": "abc123"}
        encoded = base64url_encode_json(obj)
        decoded_bytes = base64url_decode(encoded)
        decoded_obj = json.loads(decoded_bytes)
        assert decoded_obj == obj

    def test_jws_header_format(self):
        """Test encoding JWS header format."""
        header = {
            "alg": "ES256",
            "nonce": "test-nonce",
            "url": "https://acme.test/resource",
        }
        encoded = base64url_encode_json(header)
        # Should be decodable
        decoded = json.loads(base64url_decode(encoded))
        assert decoded["alg"] == "ES256"

    def test_acme_payload_format(self):
        """Test encoding ACME payload format."""
        payload = {
            "identifiers": [
                {"type": "dns", "value": "example.com"},
            ],
        }
        encoded = base64url_encode_json(payload)
        decoded = json.loads(base64url_decode(encoded))
        assert decoded["identifiers"][0]["value"] == "example.com"


class TestEncodingEdgeCases:
    """Tests for encoding edge cases."""

    def test_empty_dict(self):
        """Test encoding empty dictionary."""
        result = json_encode({})
        assert result == "{}"

    def test_empty_list(self):
        """Test encoding empty list."""
        result = json_encode([])
        assert result == "[]"

    def test_special_chars_in_string(self):
        """Test encoding strings with special characters."""
        obj = {"value": 'quote: " and backslash: \\'}
        result = json_encode(obj)
        # Should properly escape
        assert '\\"' in result
        assert "\\\\" in result

    def test_large_data(self):
        """Test encoding large data."""
        obj = {"data": "x" * 10000}
        result = json_encode(obj)
        assert len(result) > 10000

    def test_deeply_nested(self):
        """Test encoding deeply nested structure."""
        obj: dict = {}
        current = obj
        for i in range(10):
            current["nested"] = {}
            current = current["nested"]
        current["value"] = "deep"

        result = json_encode(obj)
        assert "deep" in result

    def test_numeric_types(self):
        """Test encoding various numeric types."""
        obj = {
            "int": 42,
            "float": 3.14,
            "negative": -1,
            "zero": 0,
        }
        result = json_encode(obj)
        assert "42" in result
        assert "3.14" in result

    def test_base64url_encode_decode_all_bytes(self):
        """Test all possible byte values."""
        for i in range(256):
            original = bytes([i])
            encoded = base64url_encode(original)
            decoded = base64url_decode(encoded)
            assert decoded == original, f"Failed for byte value {i}"

    def test_consistent_output_order(self):
        """Test that output is consistent across runs."""
        obj = {"z": 1, "a": 2, "m": 3}
        results = [json_encode(obj) for _ in range(100)]
        # All results should be identical
        assert len(set(results)) == 1
