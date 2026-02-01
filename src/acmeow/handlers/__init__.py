"""Challenge handlers for the ACME client library.

This package provides handler implementations for ACME challenges,
allowing automated certificate issuance with different validation methods.
"""

from __future__ import annotations

from acmeow.handlers.base import ChallengeHandler
from acmeow.handlers.dns import CallbackDnsHandler
from acmeow.handlers.http import CallbackHttpHandler, FileHttpHandler
from acmeow.handlers.tls_alpn import (
    CallbackTlsAlpnHandler,
    FileTlsAlpnHandler,
    generate_tls_alpn_certificate,
)

__all__ = [
    "ChallengeHandler",
    "CallbackDnsHandler",
    "CallbackHttpHandler",
    "FileHttpHandler",
    "CallbackTlsAlpnHandler",
    "FileTlsAlpnHandler",
    "generate_tls_alpn_certificate",
]
