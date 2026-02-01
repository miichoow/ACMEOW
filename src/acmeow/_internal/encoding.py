"""Base64URL encoding utilities for ACME protocol.

Provides RFC 4648 compliant base64url encoding/decoding functions
used throughout the ACME protocol for JWS and other cryptographic payloads.
"""

from __future__ import annotations

import base64
import json
from typing import Any


def base64url_encode(data: bytes) -> str:
    """Encode bytes to base64url string without padding.

    Implements RFC 4648 Section 5 base64url encoding with padding removed,
    as required by the ACME protocol and JWS specification.

    Args:
        data: Bytes to encode.

    Returns:
        Base64url encoded string without padding.
    """
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def base64url_decode(data: str) -> bytes:
    """Decode base64url string to bytes.

    Handles strings with or without padding, adding padding as needed
    before decoding.

    Args:
        data: Base64url encoded string.

    Returns:
        Decoded bytes.
    """
    # Add padding if necessary
    padding = 4 - (len(data) % 4)
    if padding != 4:
        data += "=" * padding
    return base64.urlsafe_b64decode(data)


def json_encode(obj: Any) -> str:
    """Encode object to JSON string with compact formatting.

    Uses compact separators (no spaces) as required by ACME protocol.

    Args:
        obj: Object to encode.

    Returns:
        JSON string with no extra whitespace.
    """
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


def json_bytes(obj: Any) -> bytes:
    """Encode object to JSON bytes with compact formatting.

    Args:
        obj: Object to encode.

    Returns:
        JSON string encoded as UTF-8 bytes.
    """
    return json_encode(obj).encode("utf-8")


def base64url_encode_json(obj: Any) -> str:
    """Encode object to base64url-encoded JSON string.

    Combines JSON encoding and base64url encoding in one step,
    commonly used for JWS protected headers and payloads.

    Args:
        obj: Object to encode.

    Returns:
        Base64url encoded JSON string.
    """
    return base64url_encode(json_bytes(obj))
