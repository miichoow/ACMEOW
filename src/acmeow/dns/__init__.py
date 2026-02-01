"""DNS provider system for ACME DNS-01 challenges.

This package provides a pluggable DNS provider system for automating
DNS-01 challenge completion. It includes several built-in providers
and supports custom provider registration.

Basic Usage:
    >>> from acmeow.dns import get_dns_provider, DnsProviderHandler
    >>> from acmeow import AcmeClient, ChallengeType
    >>>
    >>> # Get a provider by name
    >>> provider = get_dns_provider("cloudflare", api_token="your-token")
    >>> handler = DnsProviderHandler(provider)
    >>>
    >>> # Use with AcmeClient
    >>> client.complete_challenges(handler, ChallengeType.DNS)

Available Providers:
    - "manual" - Interactive provider for manual DNS record management
    - "cloudflare" - Cloudflare DNS API
    - "route53" - AWS Route53 (requires boto3)
    - "digitalocean" - DigitalOcean DNS API

Custom Provider Registration:
    >>> from acmeow.dns import register_dns_provider, DnsProvider
    >>>
    >>> class MyProvider(DnsProvider):
    ...     def create_record(self, domain, name, value, ttl):
    ...         # Implementation
    ...         pass
    ...     def delete_record(self, domain, record):
    ...         # Implementation
    ...         pass
    >>>
    >>> register_dns_provider("myprovider", MyProvider)
    >>> provider = get_dns_provider("myprovider", **kwargs)
"""

from __future__ import annotations

import logging
from typing import Any

from acmeow.dns.base import DnsProvider, DnsRecord
from acmeow.dns.handler import DnsProviderHandler

logger = logging.getLogger(__name__)

# Registry of DNS providers
_PROVIDERS: dict[str, type[DnsProvider]] = {}

# Lazy-loaded providers (loaded on first use)
_LAZY_PROVIDERS: dict[str, tuple[str, str]] = {
    "manual": ("acmeow.dns.manual", "ManualDnsProvider"),
    "cloudflare": ("acmeow.dns.cloudflare", "CloudflareDnsProvider"),
    "route53": ("acmeow.dns.route53", "Route53DnsProvider"),
    "digitalocean": ("acmeow.dns.digitalocean", "DigitalOceanDnsProvider"),
}


def register_dns_provider(name: str, provider_class: type[DnsProvider]) -> None:
    """Register a DNS provider class.

    Registered providers can be retrieved by name using get_dns_provider().

    Args:
        name: Name to register the provider under (case-insensitive).
        provider_class: The DnsProvider subclass to register.

    Example:
        >>> class MyProvider(DnsProvider):
        ...     pass
        >>> register_dns_provider("myprovider", MyProvider)
    """
    name_lower = name.lower()
    _PROVIDERS[name_lower] = provider_class
    logger.debug("Registered DNS provider: %s", name)


def unregister_dns_provider(name: str) -> bool:
    """Unregister a DNS provider.

    Args:
        name: Name of the provider to unregister.

    Returns:
        True if the provider was unregistered, False if not found.
    """
    name_lower = name.lower()
    if name_lower in _PROVIDERS:
        del _PROVIDERS[name_lower]
        logger.debug("Unregistered DNS provider: %s", name)
        return True
    return False


def get_dns_provider(name: str, **kwargs: Any) -> DnsProvider:
    """Get a DNS provider instance by name.

    Creates and returns a provider instance. The kwargs are passed
    to the provider's constructor.

    Args:
        name: Provider name (case-insensitive).
        **kwargs: Arguments to pass to the provider constructor.

    Returns:
        Configured DnsProvider instance.

    Raises:
        ValueError: If the provider is not found.
        ImportError: If the provider's dependencies are not installed.

    Example:
        >>> provider = get_dns_provider("cloudflare", api_token="xxx")
        >>> provider = get_dns_provider("manual")
    """
    name_lower = name.lower()

    # Check registered providers first
    if name_lower in _PROVIDERS:
        return _PROVIDERS[name_lower](**kwargs)

    # Try lazy-loading
    if name_lower in _LAZY_PROVIDERS:
        module_name, class_name = _LAZY_PROVIDERS[name_lower]
        try:
            import importlib
            module = importlib.import_module(module_name)
            provider_class = getattr(module, class_name)
            # Register for future use
            _PROVIDERS[name_lower] = provider_class
            result: DnsProvider = provider_class(**kwargs)
            return result
        except ImportError as e:
            raise ImportError(
                f"DNS provider '{name}' requires additional dependencies: {e}"
            ) from e

    raise ValueError(
        f"Unknown DNS provider: {name}. "
        f"Available providers: {', '.join(list_dns_providers())}"
    )


def list_dns_providers() -> list[str]:
    """List all available DNS provider names.

    Returns:
        List of provider names that can be used with get_dns_provider().
    """
    providers = set(_PROVIDERS.keys())
    providers.update(_LAZY_PROVIDERS.keys())
    return sorted(providers)


def is_dns_provider_available(name: str) -> bool:
    """Check if a DNS provider is available.

    Args:
        name: Provider name to check.

    Returns:
        True if the provider is available.
    """
    name_lower = name.lower()
    return name_lower in _PROVIDERS or name_lower in _LAZY_PROVIDERS


__all__ = [
    # Base classes
    "DnsProvider",
    "DnsRecord",
    "DnsProviderHandler",
    # Registry functions
    "register_dns_provider",
    "unregister_dns_provider",
    "get_dns_provider",
    "list_dns_providers",
    "is_dns_provider_available",
]
