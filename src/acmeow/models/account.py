"""Account model for ACME protocol.

Manages ACME account state, keys, and persistence.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from urllib.parse import urlparse

from cryptography.hazmat.primitives.asymmetric import ec

from acmeow._internal.crypto import (
    generate_account_key,
    get_jwk,
    get_jwk_thumbprint,
    load_private_key,
    serialize_private_key,
)
from acmeow.enums import AccountStatus
from acmeow.exceptions import AcmeAuthenticationError, AcmeConfigurationError

logger = logging.getLogger(__name__)


class Account:
    """ACME account management.

    Handles account creation, key management, and persistence of account
    data to disk. Account keys use EC P-256 (SECP256R1) as recommended
    by the ACME specification.

    Args:
        email: Account email address.
        storage_path: Base directory for storing account data.
        server_url: ACME server URL (used to organize storage by server).

    Raises:
        AcmeConfigurationError: If email is invalid.
    """

    def __init__(
        self,
        email: str,
        storage_path: Path,
        server_url: str,
    ) -> None:
        if not self._is_valid_email(email):
            raise AcmeConfigurationError(f"Invalid email address: {email}")

        self._email = email
        self._storage_path = storage_path
        self._server_host = urlparse(server_url).hostname or "unknown"

        self._key: ec.EllipticCurvePrivateKey | None = None
        self._jwk: dict[str, str] | None = None
        self._thumbprint: str | None = None
        self._uri: str | None = None
        self._status: AccountStatus | None = None

        # Set up paths
        self._account_dir = (
            self._storage_path / "accounts" / self._server_host / self._email
        )
        self._key_path = self._account_dir / "keys" / f"{self._email}.key"
        self._account_path = self._account_dir / "account.json"

        # Try to load existing account
        self._load_existing_account()

    @property
    def email(self) -> str:
        """Account email address."""
        return self._email

    @property
    def key(self) -> ec.EllipticCurvePrivateKey:
        """Account private key.

        Raises:
            AcmeAuthenticationError: If key is not initialized.
        """
        if self._key is None:
            raise AcmeAuthenticationError("Account key not initialized")
        return self._key

    @property
    def jwk(self) -> dict[str, str]:
        """JSON Web Key representation of public key.

        Raises:
            AcmeAuthenticationError: If key is not initialized.
        """
        if self._jwk is None:
            raise AcmeAuthenticationError("Account JWK not initialized")
        return self._jwk

    @property
    def thumbprint(self) -> str:
        """JWK thumbprint for key authorization.

        Raises:
            AcmeAuthenticationError: If key is not initialized.
        """
        if self._thumbprint is None:
            raise AcmeAuthenticationError("Account thumbprint not initialized")
        return self._thumbprint

    @property
    def uri(self) -> str | None:
        """Account URI from the ACME server."""
        return self._uri

    @property
    def status(self) -> AccountStatus | None:
        """Account status."""
        return self._status

    @property
    def exists(self) -> bool:
        """Check if account data exists on disk."""
        return self._account_path.exists() and self._key_path.exists()

    @property
    def is_valid(self) -> bool:
        """Check if account is valid and ready for use."""
        return (
            self._key is not None
            and self._uri is not None
            and self._status == AccountStatus.VALID
        )

    @property
    def contact(self) -> str:
        """Contact URL for the account (mailto: format)."""
        return f"mailto:{self._email}"

    def create_key(self) -> None:
        """Generate a new account key.

        Creates a new EC P-256 private key and saves it to disk.

        Raises:
            AcmeAuthenticationError: If key already exists.
        """
        if self._key is not None:
            raise AcmeAuthenticationError("Account key already exists")

        logger.info("Generating new account key for %s", self._email)
        self._key = generate_account_key()
        self._jwk = get_jwk(self._key)
        self._thumbprint = get_jwk_thumbprint(self._jwk)

        # Save key to disk
        self._key_path.parent.mkdir(parents=True, exist_ok=True)
        self._key_path.write_bytes(serialize_private_key(self._key))
        logger.debug("Account key saved to %s", self._key_path)

    def save(self, uri: str, status: str) -> None:
        """Save account data after registration.

        Args:
            uri: Account URI from the ACME server.
            status: Account status string.
        """
        self._uri = uri
        self._status = AccountStatus(status)

        account_data = {
            "email": self._email,
            "uri": self._uri,
            "status": self._status.value,
        }

        self._account_dir.mkdir(parents=True, exist_ok=True)
        self._account_path.write_text(json.dumps(account_data, indent=2))
        logger.info("Account saved: %s", self._uri)

    def update_key(self, new_key: ec.EllipticCurvePrivateKey) -> None:
        """Update the account key after a key rollover.

        Args:
            new_key: The new EC P-256 private key.
        """
        self._key = new_key
        self._jwk = get_jwk(self._key)
        self._thumbprint = get_jwk_thumbprint(self._jwk)

        # Save new key to disk
        self._key_path.parent.mkdir(parents=True, exist_ok=True)
        self._key_path.write_bytes(serialize_private_key(self._key))
        logger.info("Account key updated and saved to %s", self._key_path)

    def _load_existing_account(self) -> None:
        """Load existing account data and key from disk."""
        if not self.exists:
            logger.debug("No existing account found for %s", self._email)
            return

        try:
            # Load key
            key_data = self._key_path.read_bytes()
            loaded_key = load_private_key(key_data)
            if not isinstance(loaded_key, ec.EllipticCurvePrivateKey):
                raise AcmeAuthenticationError("Invalid account key type")
            self._key = loaded_key
            self._jwk = get_jwk(self._key)
            self._thumbprint = get_jwk_thumbprint(self._jwk)

            # Load account data
            account_data = json.loads(self._account_path.read_text())
            self._uri = account_data.get("uri")
            status = account_data.get("status")
            if status:
                self._status = AccountStatus(status)

            logger.info("Loaded existing account: %s", self._uri)

        except Exception as e:
            logger.warning("Failed to load existing account: %s", e)
            self._reset()

    def _reset(self) -> None:
        """Reset account state."""
        self._key = None
        self._jwk = None
        self._thumbprint = None
        self._uri = None
        self._status = None

    @staticmethod
    def _is_valid_email(email: str) -> bool:
        """Validate email address format.

        Args:
            email: Email address to validate.

        Returns:
            True if email appears valid.
        """
        from email.utils import parseaddr

        _, addr = parseaddr(email)
        return "@" in addr and len(addr) > 3

    def get_certificate_paths(self, common_name: str) -> tuple[Path, Path]:
        """Get the paths for storing a certificate and its key.

        Args:
            common_name: Certificate common name.

        Returns:
            Tuple of (certificate_path, key_path).
        """
        # Sanitize filename
        safe_name = "".join(
            c if c.isalnum() or c in ".-_" else "_"
            for c in common_name
        )
        cert_dir = self._storage_path / "certificates"
        return (
            cert_dir / f"{safe_name}.crt",
            cert_dir / f"{safe_name}.key",
        )
