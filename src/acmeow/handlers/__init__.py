"""Challenge handlers for the ACME client library.

This package provides handler implementations for ACME challenges,
allowing automated certificate issuance with different validation methods.
"""

from __future__ import annotations

from acmeow.handlers.base import ChallengeHandler
from acmeow.handlers.dns import CallbackDnsHandler
from acmeow.handlers.dns_persist import (
    PERSIST_LABEL,
    WILDCARD_POLICY,
    CallbackDnsPersistHandler,
    DnsPersistError,
    DnsPersistHandler,
    ManualDnsPersistHandler,
    PersistRecordValue,
    build_record_value,
    parse_record_value,
    select_issuer_domain_name,
    validation_domain_name,
)
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
    # DNS-PERSIST-01 (draft-ietf-acme-dns-persist)
    "DnsPersistHandler",
    "CallbackDnsPersistHandler",
    "ManualDnsPersistHandler",
    "PersistRecordValue",
    "DnsPersistError",
    "build_record_value",
    "parse_record_value",
    "select_issuer_domain_name",
    "validation_domain_name",
    "PERSIST_LABEL",
    "WILDCARD_POLICY",
]
