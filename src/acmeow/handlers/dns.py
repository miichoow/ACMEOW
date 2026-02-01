"""DNS-01 challenge handlers.

Provides handlers for DNS-01 challenge validation.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable

from acmeow._internal.encoding import base64url_encode
from acmeow.handlers.base import ChallengeHandler

logger = logging.getLogger(__name__)


class CallbackDnsHandler(ChallengeHandler):
    """DNS-01 handler using user-provided callbacks.

    This handler delegates DNS record management to user-provided
    callback functions, allowing integration with any DNS provider.

    Args:
        create_record: Callback to create a DNS TXT record.
            Signature: (domain: str, record_name: str, record_value: str) -> None
            - domain: The domain being validated (e.g., "example.com")
            - record_name: The full record name (e.g., "_acme-challenge.example.com")
            - record_value: The TXT record value (base64url SHA-256 hash)
        delete_record: Callback to delete a DNS TXT record.
            Signature: (domain: str, record_name: str) -> None
        propagation_delay: Seconds to wait after creating record. Default 60.

    Example:
        >>> def create_txt(domain, name, value):
        ...     dns_api.create_record(name, "TXT", value)
        >>> def delete_txt(domain, name):
        ...     dns_api.delete_record(name, "TXT")
        >>> handler = CallbackDnsHandler(create_txt, delete_txt)
    """

    def __init__(
        self,
        create_record: Callable[[str, str, str], None],
        delete_record: Callable[[str, str], None],
        propagation_delay: int = 60,
    ) -> None:
        self._create_record = create_record
        self._delete_record = delete_record
        self._propagation_delay = propagation_delay

    @property
    def propagation_delay(self) -> int:
        """Seconds to wait after creating DNS record for propagation."""
        return self._propagation_delay

    def setup(self, domain: str, token: str, key_authorization: str) -> None:
        """Create DNS TXT record for the challenge.

        The TXT record value is the base64url-encoded SHA-256 hash of
        the key authorization, as specified by RFC 8555.

        Args:
            domain: The domain being validated.
            token: The challenge token (not used for DNS record value).
            key_authorization: The key authorization string to hash.
        """
        record_name = self._get_record_name(domain)
        record_value = self._compute_record_value(key_authorization)

        logger.info(
            "Creating DNS TXT record: %s = %s",
            record_name,
            record_value,
        )
        self._create_record(domain, record_name, record_value)

    def cleanup(self, domain: str, token: str) -> None:
        """Remove DNS TXT record.

        Args:
            domain: The domain that was validated.
            token: The challenge token (not used).
        """
        record_name = self._get_record_name(domain)

        logger.info("Removing DNS TXT record: %s", record_name)
        try:
            self._delete_record(domain, record_name)
        except Exception as e:
            logger.warning("Failed to cleanup DNS record %s: %s", record_name, e)

    @staticmethod
    def _get_record_name(domain: str) -> str:
        """Get the DNS record name for a domain.

        Args:
            domain: The domain being validated.

        Returns:
            The full record name (e.g., "_acme-challenge.example.com").
        """
        return f"_acme-challenge.{domain}"

    @staticmethod
    def _compute_record_value(key_authorization: str) -> str:
        """Compute the DNS TXT record value.

        Args:
            key_authorization: The key authorization string.

        Returns:
            Base64url-encoded SHA-256 hash of the key authorization.
        """
        digest = hashlib.sha256(key_authorization.encode("utf-8")).digest()
        return base64url_encode(digest)
