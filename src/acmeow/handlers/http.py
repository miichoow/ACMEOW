"""HTTP-01 challenge handlers.

Provides handlers for HTTP-01 challenge validation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from acmeow.handlers.base import ChallengeHandler

logger = logging.getLogger(__name__)


class FileHttpHandler(ChallengeHandler):
    """HTTP-01 handler that writes challenge files to a webroot directory.

    This handler writes challenge response files to the standard
    `.well-known/acme-challenge/` directory structure. The web server
    must be configured to serve this directory.

    Args:
        webroot: Path to the web server's document root.
            Files will be written to {webroot}/.well-known/acme-challenge/

    Example:
        >>> handler = FileHttpHandler(Path("/var/www/html"))
        >>> # Challenge file will be at:
        >>> # /var/www/html/.well-known/acme-challenge/{token}
    """

    def __init__(self, webroot: Path) -> None:
        self._webroot = webroot
        self._challenge_dir = webroot / ".well-known" / "acme-challenge"

    @property
    def challenge_dir(self) -> Path:
        """Directory where challenge files are written."""
        return self._challenge_dir

    def setup(self, domain: str, token: str, key_authorization: str) -> None:
        """Write challenge response file.

        Creates the challenge file containing the key authorization
        at the path expected by the ACME server:
        {webroot}/.well-known/acme-challenge/{token}

        Args:
            domain: The domain being validated (for logging).
            token: The challenge token (becomes the filename).
            key_authorization: The key authorization (file content).
        """
        # Ensure challenge directory exists
        self._challenge_dir.mkdir(parents=True, exist_ok=True)

        # Write challenge file
        challenge_path = self._challenge_dir / token
        challenge_path.write_text(key_authorization)

        logger.info(
            "Created HTTP challenge file for %s at %s",
            domain,
            challenge_path,
        )

    def cleanup(self, domain: str, token: str) -> None:
        """Remove challenge response file.

        Args:
            domain: The domain that was validated.
            token: The challenge token (the filename).
        """
        challenge_path = self._challenge_dir / token

        try:
            if challenge_path.exists():
                challenge_path.unlink()
                logger.info(
                    "Removed HTTP challenge file for %s at %s",
                    domain,
                    challenge_path,
                )
        except Exception as e:
            logger.warning(
                "Failed to cleanup HTTP challenge file %s: %s",
                challenge_path,
                e,
            )


class CallbackHttpHandler(ChallengeHandler):
    """HTTP-01 handler using user-provided callbacks.

    This handler delegates challenge file management to user-provided
    callback functions, allowing integration with any HTTP serving method.

    Args:
        setup_callback: Callback to deploy the challenge response.
            Signature: (domain: str, token: str, key_authorization: str) -> None
        cleanup_callback: Callback to remove the challenge response.
            Signature: (domain: str, token: str) -> None

    Example:
        >>> def setup_challenge(domain, token, key_auth):
        ...     redis.set(f"acme:{token}", key_auth)
        >>> def cleanup_challenge(domain, token):
        ...     redis.delete(f"acme:{token}")
        >>> handler = CallbackHttpHandler(setup_challenge, cleanup_challenge)
    """

    def __init__(
        self,
        setup_callback: Callable[[str, str, str], None],
        cleanup_callback: Callable[[str, str], None],
    ) -> None:
        self._setup_callback = setup_callback
        self._cleanup_callback = cleanup_callback

    def setup(self, domain: str, token: str, key_authorization: str) -> None:
        """Deploy challenge response using callback.

        Args:
            domain: The domain being validated.
            token: The challenge token.
            key_authorization: The key authorization string.
        """
        logger.info("Setting up HTTP challenge for %s (token: %s)", domain, token)
        self._setup_callback(domain, token, key_authorization)

    def cleanup(self, domain: str, token: str) -> None:
        """Remove challenge response using callback.

        Args:
            domain: The domain that was validated.
            token: The challenge token.
        """
        logger.info("Cleaning up HTTP challenge for %s (token: %s)", domain, token)
        try:
            self._cleanup_callback(domain, token)
        except Exception as e:
            logger.warning(
                "Failed to cleanup HTTP challenge for %s: %s",
                domain,
                e,
            )
