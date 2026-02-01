"""ACMEOW - Production-grade ACME protocol client library.

A Python library for automated SSL/TLS certificate management using the
ACME protocol (RFC 8555). Supports DNS-01, HTTP-01, and TLS-ALPN-01 challenges.

Basic Usage:
    >>> from acmeow import AcmeClient, Identifier, KeyType
    >>> from pathlib import Path
    >>>
    >>> client = AcmeClient(
    ...     server_url="https://acme-staging-v02.api.letsencrypt.org/directory",
    ...     email="admin@example.com",
    ...     storage_path=Path("./acme_data"),
    ... )
    >>> client.create_account()
    >>> order = client.create_order([Identifier.dns("example.com")])
    >>> # Complete challenges with a handler...
    >>> client.finalize_order(KeyType.EC256)
    >>> cert_pem, key_pem = client.get_certificate()

Quick Certificate Issuance:
    >>> from acmeow import quick_issue, CallbackDnsHandler
    >>> handler = CallbackDnsHandler(create_record, delete_record)
    >>> cert, key = quick_issue("example.com", "admin@example.com", handler)

DNS Provider System:
    >>> from acmeow import get_dns_provider, DnsProviderHandler, ChallengeType
    >>> provider = get_dns_provider("cloudflare", api_token="...")
    >>> handler = DnsProviderHandler(provider)
    >>> client.complete_challenges(handler, ChallengeType.DNS)
"""

from __future__ import annotations

from acmeow._internal.dns import DnsConfig
from acmeow._internal.hooks import (
    HookContext,
    HookEvent,
    HookManager,
    add_hook,
    emit_hook,
    get_hook_manager,
    remove_hook,
)
from acmeow._internal.http import RetryConfig
from acmeow._internal.rate_limiter import (
    RateLimitConfig,
    RateLimitMetrics,
    SmartRateLimiter,
)
from acmeow.client import AcmeClient
from acmeow.convenience import (
    BatchResult,
    CertificateInfo,
    check_certificate_expiry,
    get_certificate_info,
    issue_batch,
    quick_issue,
    renew_if_needed,
)
from acmeow.dns import (
    DnsProvider,
    DnsProviderHandler,
    DnsRecord,
    get_dns_provider,
    is_dns_provider_available,
    list_dns_providers,
    register_dns_provider,
)
from acmeow.enums import (
    AccountStatus,
    AuthorizationStatus,
    ChallengeStatus,
    ChallengeType,
    IdentifierType,
    KeyType,
    OrderStatus,
    RevocationReason,
)
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
from acmeow.handlers import (
    CallbackDnsHandler,
    CallbackHttpHandler,
    CallbackTlsAlpnHandler,
    ChallengeHandler,
    FileHttpHandler,
    FileTlsAlpnHandler,
    generate_tls_alpn_certificate,
)
from acmeow.models import (
    Account,
    Authorization,
    Challenge,
    Identifier,
    Order,
)
from acmeow.orders import OrderInfo, OrderManager

__version__ = "1.0.0"

__all__ = [
    # Version
    "__version__",
    # Main client
    "AcmeClient",
    # Configuration
    "RetryConfig",
    "DnsConfig",
    "RateLimitConfig",
    # Models
    "Account",
    "Authorization",
    "Challenge",
    "Identifier",
    "Order",
    # Enums
    "AccountStatus",
    "AuthorizationStatus",
    "ChallengeStatus",
    "ChallengeType",
    "IdentifierType",
    "KeyType",
    "OrderStatus",
    "RevocationReason",
    # Handlers - DNS
    "ChallengeHandler",
    "CallbackDnsHandler",
    # Handlers - HTTP
    "CallbackHttpHandler",
    "FileHttpHandler",
    # Handlers - TLS-ALPN
    "CallbackTlsAlpnHandler",
    "FileTlsAlpnHandler",
    "generate_tls_alpn_certificate",
    # DNS Provider System
    "DnsProvider",
    "DnsProviderHandler",
    "DnsRecord",
    "get_dns_provider",
    "register_dns_provider",
    "list_dns_providers",
    "is_dns_provider_available",
    # Observability Hooks
    "HookEvent",
    "HookContext",
    "HookManager",
    "get_hook_manager",
    "add_hook",
    "remove_hook",
    "emit_hook",
    # Rate Limiting
    "SmartRateLimiter",
    "RateLimitMetrics",
    # Order Management
    "OrderManager",
    "OrderInfo",
    # Convenience Methods
    "quick_issue",
    "issue_batch",
    "renew_if_needed",
    "get_certificate_info",
    "check_certificate_expiry",
    "CertificateInfo",
    "BatchResult",
    # Exceptions
    "AcmeError",
    "AcmeServerError",
    "AcmeAuthenticationError",
    "AcmeAuthorizationError",
    "AcmeCertificateError",
    "AcmeConfigurationError",
    "AcmeDnsError",
    "AcmeNetworkError",
    "AcmeOrderError",
    "AcmeRateLimitError",
    "AcmeTimeoutError",
]
