"""Base challenge handler interface.

Defines the abstract interface for all challenge handlers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ChallengeHandler(ABC):
    """Abstract base class for ACME challenge handlers.

    Challenge handlers are responsible for deploying and cleaning up
    challenge responses. Implementations must handle both setup and
    cleanup to ensure proper resource management.

    Example:
        >>> class MyDnsHandler(ChallengeHandler):
        ...     def setup(self, domain, token, key_authorization):
        ...         # Create DNS TXT record
        ...         pass
        ...     def cleanup(self, domain, token):
        ...         # Remove DNS TXT record
        ...         pass
    """

    @abstractmethod
    def setup(self, domain: str, token: str, key_authorization: str) -> None:
        """Deploy the challenge response.

        This method is called before notifying the ACME server that the
        challenge is ready for validation. It should set up whatever
        resource is needed for the challenge type (DNS record, HTTP file, etc.).

        Args:
            domain: The domain being validated.
            token: The challenge token from the ACME server.
            key_authorization: The key authorization string (token.thumbprint).
                For DNS-01, this should be hashed with SHA-256 and base64url encoded.
                For HTTP-01, this is served directly.

        Raises:
            Exception: If setup fails and the challenge cannot proceed.
        """

    @abstractmethod
    def cleanup(self, domain: str, token: str) -> None:
        """Remove the challenge response.

        This method is called after challenge validation completes
        (whether successful or not) to clean up deployed resources.

        Args:
            domain: The domain that was validated.
            token: The challenge token.

        Note:
            This method should not raise exceptions even if cleanup fails,
            as it's called during error handling paths.
        """
