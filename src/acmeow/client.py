"""Main ACME client implementation.

Provides the AcmeClient class for automated certificate management
using the ACME protocol (RFC 8555).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization

from acmeow._internal.crypto import (
    create_csr,
    generate_private_key,
    get_key_authorization,
    serialize_private_key,
)
from acmeow._internal.dns import DnsConfig, DnsVerifier
from acmeow._internal.encoding import base64url_encode, base64url_encode_json
from acmeow._internal.http import AcmeHttpClient, RetryConfig
from acmeow.enums import (
    AuthorizationStatus,
    ChallengeType,
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
    AcmeNetworkError,
    AcmeOrderError,
    AcmeTimeoutError,
)
from acmeow.handlers.base import ChallengeHandler
from acmeow.models.account import Account
from acmeow.models.authorization import Authorization
from acmeow.models.identifier import Identifier
from acmeow.models.order import Order

logger = logging.getLogger(__name__)


class AcmeClient:
    """ACME protocol client for automated certificate management.

    Implements the ACME protocol (RFC 8555) for obtaining SSL/TLS
    certificates from ACME-compliant certificate authorities like
    Let's Encrypt.

    Args:
        server_url: ACME server directory URL.
        email: Account email address.
        storage_path: Directory for storing account and certificate data.
        verify_ssl: Whether to verify SSL certificates. Default True.
        timeout: Request timeout in seconds. Default 30.
        retry_config: Retry configuration for transient failures.

    Example:
        >>> client = AcmeClient(
        ...     server_url="https://acme-staging-v02.api.letsencrypt.org/directory",
        ...     email="admin@example.com",
        ...     storage_path=Path("./acme_data"),
        ... )
        >>> client.create_account()
        >>> order = client.create_order([Identifier.dns("example.com")])
        >>> # Complete challenges...
        >>> client.finalize_order(KeyType.EC256)
        >>> cert, key = client.get_certificate()

    Raises:
        AcmeConfigurationError: If configuration is invalid.
        AcmeNetworkError: If server communication fails.
    """

    def __init__(
        self,
        server_url: str,
        email: str,
        storage_path: Path | str,
        verify_ssl: bool = True,
        timeout: int = 30,
        retry_config: RetryConfig | None = None,
    ) -> None:
        self._server_url = server_url.rstrip("/")
        self._email = email
        self._storage_path = Path(storage_path)
        self._verify_ssl = verify_ssl

        # HTTP client with retry support
        self._http = AcmeHttpClient(
            verify_ssl=verify_ssl,
            timeout=timeout,
            retry_config=retry_config,
        )

        # ACME directory endpoints
        self._directory: dict[str, Any] = {}

        # Account
        self._account: Account | None = None

        # Current order
        self._order: Order | None = None

        # External Account Binding (EAB)
        self._eab_kid: str | None = None
        self._eab_hmac_key: bytes | None = None

        # DNS verification
        self._dns_config: DnsConfig | None = None

        # Fetch directory on initialization
        self._fetch_directory()

    @property
    def server_url(self) -> str:
        """ACME server directory URL."""
        return self._server_url

    @property
    def email(self) -> str:
        """Account email address."""
        return self._email

    @property
    def account(self) -> Account | None:
        """Current account, or None if not created."""
        return self._account

    @property
    def order(self) -> Order | None:
        """Current order, or None if not created."""
        return self._order

    def set_dns_config(self, config: DnsConfig) -> None:
        """Set DNS verification configuration.

        When configured, DNS propagation is verified before notifying the
        ACME server during DNS-01 challenges.

        Args:
            config: DNS verification configuration.
        """
        self._dns_config = config
        logger.info("DNS verification configured with %d nameservers", len(config.nameservers))

    def set_external_account_binding(self, kid: str, hmac_key: str) -> None:
        """Set External Account Binding (EAB) credentials.

        Some ACME servers require EAB to link the ACME account to an
        existing account in an external system.

        Args:
            kid: Key identifier from the CA.
            hmac_key: Base64url-encoded HMAC key from the CA.
        """
        from acmeow._internal.encoding import base64url_decode

        self._eab_kid = kid
        self._eab_hmac_key = base64url_decode(hmac_key)
        logger.info("External Account Binding configured with kid: %s", kid)

    def _fetch_directory(self) -> None:
        """Fetch ACME directory from server.

        Raises:
            AcmeNetworkError: If directory cannot be fetched.
        """
        logger.info("Fetching ACME directory from %s", self._server_url)
        response = self._http.get(self._server_url)

        try:
            self._directory = response.json()
        except ValueError as e:
            raise AcmeNetworkError(f"Invalid directory response: {e}", e) from e

        # Configure nonce URL
        new_nonce_url = self._directory.get("newNonce")
        if not new_nonce_url:
            raise AcmeNetworkError("Directory missing newNonce URL")
        self._http.set_nonce_url(new_nonce_url)

        logger.debug("ACME directory: %s", self._directory)

    def create_account(self, terms_agreed: bool = True) -> Account:
        """Create or retrieve an ACME account.

        Creates a new account if one doesn't exist, or retrieves the
        existing account if already registered.

        Args:
            terms_agreed: Whether the user agrees to the CA's terms of service.
                Must be True to create an account.

        Returns:
            The Account object.

        Raises:
            AcmeAuthenticationError: If account creation fails.
            AcmeConfigurationError: If terms not agreed.
        """
        if not terms_agreed:
            raise AcmeConfigurationError("Terms of service must be agreed to")

        # Initialize account
        self._account = Account(
            email=self._email,
            storage_path=self._storage_path,
            server_url=self._server_url,
        )

        # Check if account already exists and is valid
        if self._account.is_valid:
            logger.info("Using existing account: %s", self._account.uri)
            # Try to load any saved order
            self._load_order()
            return self._account

        # Create new key if needed
        if not self._account.exists:
            self._account.create_key()

        # Build registration payload
        payload: dict[str, Any] = {
            "termsOfServiceAgreed": True,
            "contact": [self._account.contact],
        }

        # Add EAB if required
        if self._is_eab_required():
            if not self._eab_kid or not self._eab_hmac_key:
                raise AcmeConfigurationError(
                    "External Account Binding required but not configured"
                )
            payload["externalAccountBinding"] = self._create_eab_payload()

        # Register account
        new_account_url = self._directory.get("newAccount")
        if not new_account_url:
            raise AcmeNetworkError("Directory missing newAccount URL")

        logger.info("Creating account for %s", self._email)
        response = self._http.post(
            new_account_url,
            payload,
            self._account.key,
            jwk=self._account.jwk,
        )

        # Get account URL from Location header
        account_uri = response.headers.get("Location")
        if not account_uri:
            raise AcmeAuthenticationError("Server did not return account URL")

        # Parse response
        data = response.json()
        status = data.get("status", "valid")

        # Save account
        self._account.save(account_uri, status)
        logger.info("Account created: %s (status: %s)", account_uri, status)

        return self._account

    def _is_eab_required(self) -> bool:
        """Check if External Account Binding is required.

        Returns:
            True if EAB is required by the server.
        """
        meta = self._directory.get("meta", {})
        return bool(meta.get("externalAccountRequired", False))

    def _create_eab_payload(self) -> dict[str, str]:
        """Create External Account Binding JWS payload.

        Returns:
            EAB JWS structure.
        """
        if not self._account or not self._eab_kid or not self._eab_hmac_key:
            raise AcmeConfigurationError("EAB not properly configured")

        # Protected header for EAB (HS256)
        protected = {
            "alg": "HS256",
            "kid": self._eab_kid,
            "url": self._directory["newAccount"],
        }
        protected_b64 = base64url_encode_json(protected)

        # Payload is the account JWK
        payload_b64 = base64url_encode_json(self._account.jwk)

        # Sign with HMAC-SHA256
        signing_input = f"{protected_b64}.{payload_b64}".encode("ascii")
        signature = hmac.new(self._eab_hmac_key, signing_input, hashlib.sha256).digest()
        signature_b64 = base64url_encode(signature)

        return {
            "protected": protected_b64,
            "payload": payload_b64,
            "signature": signature_b64,
        }

    def create_order(
        self,
        identifiers: list[Identifier],
        save: bool = True,
    ) -> Order:
        """Create a new certificate order.

        Args:
            identifiers: List of identifiers (domains/IPs) for the certificate.
            save: Whether to save order state for recovery. Default True.

        Returns:
            The Order object.

        Raises:
            AcmeOrderError: If order creation fails.
            AcmeAuthenticationError: If not authenticated.
        """
        if not self._account or not self._account.is_valid:
            raise AcmeAuthenticationError("Account not created or invalid")

        if not identifiers:
            raise AcmeConfigurationError("At least one identifier required")

        new_order_url = self._directory.get("newOrder")
        if not new_order_url:
            raise AcmeNetworkError("Directory missing newOrder URL")

        payload = {
            "identifiers": [i.to_dict() for i in identifiers],
        }

        logger.info("Creating order for: %s", [i.value for i in identifiers])
        response = self._http.post(
            new_order_url,
            payload,
            self._account.key,
            kid=self._account.uri,
        )

        # Get order URL from Location header
        order_url = response.headers.get("Location")
        if not order_url:
            raise AcmeOrderError("Server did not return order URL")

        # Parse response
        data = response.json()
        self._order = Order.from_dict(data, order_url)

        # Fetch authorizations
        self._fetch_authorizations()

        # Save order for recovery
        if save:
            self._save_order()

        logger.info("Order created: %s (status: %s)", order_url, self._order.status)
        return self._order

    def load_order(self) -> Order | None:
        """Load a previously saved order.

        Attempts to load and resume an incomplete order that was saved
        to disk. Useful for recovering from interruptions.

        Returns:
            The Order if found and still valid, None otherwise.
        """
        self._load_order()
        return self._order

    def _get_order_path(self) -> Path:
        """Get the path for storing order state."""
        return self._storage_path / "orders" / "current_order.json"

    def _save_order(self) -> None:
        """Save current order state to disk for recovery."""
        if not self._order:
            return

        order_path = self._get_order_path()
        order_path.parent.mkdir(parents=True, exist_ok=True)

        order_data = {
            "url": self._order.url,
            "status": self._order.status.value,
            "identifiers": [i.to_dict() for i in self._order.identifiers],
            "finalize_url": self._order.finalize_url,
            "expires": self._order.expires,
            "certificate_url": self._order.certificate_url,
        }

        order_path.write_text(json.dumps(order_data, indent=2))
        logger.debug("Order saved to %s", order_path)

    def _load_order(self) -> None:
        """Load order state from disk."""
        order_path = self._get_order_path()
        if not order_path.exists():
            return

        try:
            order_data = json.loads(order_path.read_text())

            # Check if order is still relevant
            status = OrderStatus(order_data.get("status", "invalid"))
            if status in (OrderStatus.INVALID, OrderStatus.VALID):
                # Order is terminal, delete it
                order_path.unlink()
                return

            # Recreate order
            self._order = Order(
                status=status,
                url=order_data["url"],
                identifiers=tuple(
                    Identifier.from_dict(i) for i in order_data["identifiers"]
                ),
                finalize_url=order_data["finalize_url"],
                expires=order_data.get("expires"),
                certificate_url=order_data.get("certificate_url"),
            )

            # Refresh from server to get current status
            self._refresh_order()
            self._fetch_authorizations()

            logger.info("Loaded order from disk: %s (status: %s)", self._order.url, self._order.status)

        except (KeyError, ValueError, json.JSONDecodeError) as e:
            logger.warning("Failed to load order: %s", e)
            order_path.unlink(missing_ok=True)

    def _clear_order(self) -> None:
        """Clear saved order state."""
        order_path = self._get_order_path()
        order_path.unlink(missing_ok=True)

    def _fetch_authorizations(self) -> None:
        """Fetch all authorizations for the current order."""
        if not self._order or not self._account or not self._account.uri:
            return

        data = self._http.get(self._order.url).json()
        auth_urls = data.get("authorizations", [])

        self._order.authorizations.clear()
        for auth_url in auth_urls:
            response = self._http.post_as_get(
                auth_url,
                self._account.key,
                self._account.uri,
            )
            auth_data = response.json()
            auth = Authorization.from_dict(auth_data, auth_url)
            self._order.authorizations.append(auth)

        logger.debug("Fetched %d authorizations", len(self._order.authorizations))

    def complete_challenges(
        self,
        handler: ChallengeHandler,
        challenge_type: ChallengeType = ChallengeType.DNS,
        propagation_delay: int = 0,
        verify_dns: bool = True,
        dns_timeout: int = 300,
        parallel: bool = False,
        max_workers: int | None = None,
    ) -> None:
        """Complete all pending challenges using the provided handler.

        This method sets up challenge responses, optionally verifies DNS
        propagation (for DNS-01), notifies the ACME server, and waits for
        validation to complete.

        Args:
            handler: Challenge handler for deploying/cleaning up responses.
            challenge_type: Type of challenge to complete. Default DNS-01.
            propagation_delay: Seconds to wait after setup before notifying server.
                If handler has propagation_delay attribute, that value is used.
            verify_dns: Whether to verify DNS propagation before notifying server.
                Only applies to DNS-01 challenges. Default True.
            dns_timeout: Maximum time to wait for DNS propagation in seconds.
                Default 300 (5 minutes).
            parallel: Whether to set up and clean up challenges in parallel.
                Default False. Server notification and polling remain sequential.
            max_workers: Maximum number of parallel workers. Default None (auto).
                Only used when parallel=True.

        Raises:
            AcmeAuthorizationError: If challenge validation fails.
            AcmeOrderError: If no order exists.
            AcmeDnsError: If DNS propagation verification fails.
        """
        if not self._order:
            raise AcmeOrderError("No order exists")
        if not self._account:
            raise AcmeAuthenticationError("Account not created")

        # Get propagation delay from handler if available
        delay = getattr(handler, "propagation_delay", propagation_delay)

        # Track challenges we set up for cleanup
        setup_challenges: list[tuple[str, str, str]] = []  # domain, token, key_auth

        try:
            # Collect challenges to set up
            challenges_to_setup: list[tuple[str, str, str]] = []
            for auth in self._order.authorizations:
                if auth.is_valid:
                    logger.debug("Authorization already valid: %s", auth.domain)
                    continue

                challenge = auth.get_challenge(challenge_type)
                if not challenge:
                    raise AcmeAuthorizationError(
                        auth.domain,
                        f"No {challenge_type.value} challenge available",
                    )

                # Compute key authorization
                key_auth = get_key_authorization(
                    challenge.token,
                    self._account.thumbprint,
                )
                challenges_to_setup.append((auth.domain, challenge.token, key_auth))

            # Set up challenges (parallel or sequential)
            if parallel and len(challenges_to_setup) > 1:
                setup_challenges = self._setup_challenges_parallel(
                    handler, challenges_to_setup, challenge_type, max_workers
                )
            else:
                setup_challenges = self._setup_challenges_sequential(
                    handler, challenges_to_setup, challenge_type
                )

            # Wait for propagation (simple delay)
            if delay > 0 and setup_challenges:
                logger.info("Waiting %d seconds for propagation", delay)
                time.sleep(delay)

            # Verify DNS propagation if configured
            if (
                challenge_type == ChallengeType.DNS
                and verify_dns
                and self._dns_config
                and setup_challenges
            ):
                self._verify_dns_propagation(setup_challenges, dns_timeout)

            # Respond to all challenges (sequential - server expects order)
            for auth in self._order.authorizations:
                if auth.is_valid:
                    continue

                challenge = auth.get_challenge(challenge_type)
                if not challenge:
                    continue

                logger.info("Responding to challenge for %s", auth.domain)
                self._respond_to_challenge(challenge.url)

            # Poll for validation
            self._poll_authorizations()

            # Update saved order
            self._save_order()

        finally:
            # Clean up challenges (parallel or sequential)
            if parallel and len(setup_challenges) > 1:
                self._cleanup_challenges_parallel(handler, setup_challenges, max_workers)
            else:
                self._cleanup_challenges_sequential(handler, setup_challenges)

    def _setup_challenges_sequential(
        self,
        handler: ChallengeHandler,
        challenges: list[tuple[str, str, str]],
        challenge_type: ChallengeType,
    ) -> list[tuple[str, str, str]]:
        """Set up challenges sequentially.

        Args:
            handler: Challenge handler.
            challenges: List of (domain, token, key_auth) tuples.
            challenge_type: The challenge type.

        Returns:
            List of successfully set up challenges.
        """
        setup_challenges: list[tuple[str, str, str]] = []

        for domain, token, key_auth in challenges:
            logger.info(
                "Setting up %s challenge for %s",
                challenge_type.value,
                domain,
            )
            handler.setup(domain, token, key_auth)
            setup_challenges.append((domain, token, key_auth))

        return setup_challenges

    def _setup_challenges_parallel(
        self,
        handler: ChallengeHandler,
        challenges: list[tuple[str, str, str]],
        challenge_type: ChallengeType,
        max_workers: int | None,
    ) -> list[tuple[str, str, str]]:
        """Set up challenges in parallel using ThreadPoolExecutor.

        Args:
            handler: Challenge handler.
            challenges: List of (domain, token, key_auth) tuples.
            challenge_type: The challenge type.
            max_workers: Maximum number of workers.

        Returns:
            List of successfully set up challenges.
        """
        setup_challenges: list[tuple[str, str, str]] = []
        errors: list[tuple[str, Exception]] = []

        logger.info(
            "Setting up %d %s challenges in parallel",
            len(challenges),
            challenge_type.value,
        )

        def setup_one(challenge_info: tuple[str, str, str]) -> tuple[str, str, str]:
            domain, token, key_auth = challenge_info
            logger.info(
                "Setting up %s challenge for %s",
                challenge_type.value,
                domain,
            )
            handler.setup(domain, token, key_auth)
            return challenge_info

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(setup_one, c): c[0] for c in challenges
            }

            for future in as_completed(futures):
                domain = futures[future]
                try:
                    result = future.result()
                    setup_challenges.append(result)
                except Exception as e:
                    logger.error("Failed to setup challenge for %s: %s", domain, e)
                    errors.append((domain, e))

        if errors:
            # Raise the first error, but log all
            domain, error = errors[0]
            raise AcmeAuthorizationError(
                domain,
                f"Challenge setup failed: {error}",
            )

        return setup_challenges

    def _cleanup_challenges_sequential(
        self,
        handler: ChallengeHandler,
        challenges: list[tuple[str, str, str]],
    ) -> None:
        """Clean up challenges sequentially.

        Args:
            handler: Challenge handler.
            challenges: List of (domain, token, key_auth) tuples.
        """
        for domain, token, _ in challenges:
            try:
                handler.cleanup(domain, token)
            except Exception as e:
                logger.warning("Challenge cleanup failed for %s: %s", domain, e)

    def _cleanup_challenges_parallel(
        self,
        handler: ChallengeHandler,
        challenges: list[tuple[str, str, str]],
        max_workers: int | None,
    ) -> None:
        """Clean up challenges in parallel.

        Args:
            handler: Challenge handler.
            challenges: List of (domain, token, key_auth) tuples.
            max_workers: Maximum number of workers.
        """
        logger.debug("Cleaning up %d challenges in parallel", len(challenges))

        def cleanup_one(challenge_info: tuple[str, str, str]) -> None:
            domain, token, _ = challenge_info
            handler.cleanup(domain, token)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(cleanup_one, c): c[0] for c in challenges
            }

            for future in as_completed(futures):
                domain = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.warning("Challenge cleanup failed for %s: %s", domain, e)

    def _verify_dns_propagation(
        self,
        challenges: list[tuple[str, str, str]],
        timeout: int,
    ) -> None:
        """Verify DNS propagation for all challenges.

        Args:
            challenges: List of (domain, token, key_auth) tuples.
            timeout: Maximum time to wait for propagation.

        Raises:
            AcmeDnsError: If any DNS record fails to propagate.
        """
        if not self._dns_config:
            return

        verifier = DnsVerifier(self._dns_config)

        for domain, _, key_auth in challenges:
            # Compute expected TXT value
            digest = hashlib.sha256(key_auth.encode("utf-8")).digest()
            expected_value = base64url_encode(digest)
            record_name = f"_acme-challenge.{domain}"

            logger.info("Verifying DNS propagation for %s", record_name)

            if not verifier.verify_txt_record(record_name, expected_value, max_wait=timeout):
                raise AcmeDnsError(
                    domain,
                    f"TXT record {record_name} did not propagate within {timeout}s",
                )

    def _respond_to_challenge(self, challenge_url: str) -> None:
        """Notify the server that a challenge is ready.

        Args:
            challenge_url: URL of the challenge to respond to.
        """
        if not self._account:
            raise AcmeAuthenticationError("Account not created")

        self._http.post(
            challenge_url,
            {},  # Empty payload signals ready
            self._account.key,
            kid=self._account.uri,
        )

    def _poll_authorizations(
        self,
        max_attempts: int = 30,
        interval: int = 2,
    ) -> None:
        """Poll authorizations until all are valid or failed.

        Args:
            max_attempts: Maximum polling attempts.
            interval: Seconds between polls.

        Raises:
            AcmeAuthorizationError: If any authorization fails.
            AcmeTimeoutError: If polling times out.
        """
        if not self._order or not self._account or not self._account.uri:
            return

        for attempt in range(max_attempts):
            all_valid = True
            any_failed = False
            failed_domain = ""
            failed_error = ""

            for auth in self._order.authorizations:
                # Fetch current status
                response = self._http.post_as_get(
                    auth.url,
                    self._account.key,
                    self._account.uri,
                )
                data = response.json()
                status = AuthorizationStatus(data.get("status", "pending"))

                if status == AuthorizationStatus.INVALID:
                    any_failed = True
                    failed_domain = auth.domain
                    # Try to get error details from challenge
                    for c in data.get("challenges", []):
                        if c.get("error"):
                            failed_error = c["error"].get("detail", "Unknown error")
                            break
                    break
                elif status != AuthorizationStatus.VALID:
                    all_valid = False

            if any_failed:
                raise AcmeAuthorizationError(failed_domain, failed_error)

            if all_valid:
                logger.info("All authorizations valid")
                return

            logger.debug(
                "Polling authorizations (attempt %d/%d)",
                attempt + 1,
                max_attempts,
            )
            time.sleep(interval)

        raise AcmeTimeoutError("Authorization polling timed out")

    def finalize_order(
        self,
        key_type: KeyType = KeyType.EC256,
        common_name: str | None = None,
    ) -> None:
        """Finalize the order by submitting a CSR.

        Generates a private key and CSR, then submits them to finalize
        the order. The private key is saved to disk.

        Args:
            key_type: Key type for the certificate. Default EC256.
            common_name: Common name override (defaults to first identifier).

        Raises:
            AcmeOrderError: If finalization fails.
        """
        if not self._order:
            raise AcmeOrderError("No order exists")
        if not self._account:
            raise AcmeAuthenticationError("Account not created")

        # Refresh order status
        self._refresh_order()

        if not self._order.is_ready:
            raise AcmeOrderError(
                f"Order not ready for finalization (status: {self._order.status})"
            )

        # Generate key and CSR
        logger.info("Generating %s key and CSR", key_type.value)
        cert_key = generate_private_key(key_type)
        csr_der = create_csr(
            list(self._order.identifiers),
            cert_key,
            common_name=common_name,
        )

        # Save the private key
        cert_path, key_path = self._account.get_certificate_paths(
            self._order.common_name
        )
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_bytes(serialize_private_key(cert_key))
        logger.info("Certificate key saved to %s", key_path)

        # Submit CSR
        payload = {"csr": base64url_encode(csr_der)}

        logger.info("Finalizing order")
        self._http.post(
            self._order.finalize_url,
            payload,
            self._account.key,
            kid=self._account.uri,
        )

        # Poll for certificate
        self._poll_order()

        # Update saved order
        self._save_order()

    def _refresh_order(self) -> None:
        """Refresh the current order from the server."""
        if not self._order or not self._account or not self._account.uri:
            return

        response = self._http.post_as_get(
            self._order.url,
            self._account.key,
            self._account.uri,
        )
        data = response.json()
        self._order.update_from_dict(data)

    def _poll_order(
        self,
        max_attempts: int = 30,
        interval: int = 2,
    ) -> None:
        """Poll order until certificate is ready.

        Args:
            max_attempts: Maximum polling attempts.
            interval: Seconds between polls.

        Raises:
            AcmeOrderError: If order fails.
            AcmeTimeoutError: If polling times out.
        """
        if not self._order or not self._account:
            return

        for attempt in range(max_attempts):
            self._refresh_order()

            if self._order.is_valid:
                logger.info("Order valid, certificate ready")
                return

            if self._order.is_invalid:
                error_detail = ""
                if self._order.error:
                    error_detail = self._order.error.get("detail", "")
                raise AcmeOrderError(f"Order failed: {error_detail}")

            logger.debug(
                "Polling order (attempt %d/%d, status: %s)",
                attempt + 1,
                max_attempts,
                self._order.status,
            )
            time.sleep(interval)

        raise AcmeTimeoutError("Order polling timed out")

    def get_certificate(
        self,
        preferred_chain: str | None = None,
    ) -> tuple[str, str]:
        """Download the issued certificate.

        Args:
            preferred_chain: Preferred certificate chain issuer CN.
                If the server provides multiple chains, select the one
                whose issuer CN contains this string. Default None (use default chain).

        Returns:
            Tuple of (certificate_pem, private_key_pem).
            The certificate includes the full chain.

        Raises:
            AcmeCertificateError: If certificate download fails.
        """
        if not self._order:
            raise AcmeOrderError("No order exists")
        if not self._account or not self._account.uri:
            raise AcmeAuthenticationError("Account not created")

        # Refresh order to get certificate URL
        self._refresh_order()

        if not self._order.certificate_url:
            raise AcmeCertificateError("Certificate URL not available")

        # Download certificate
        logger.info("Downloading certificate")
        response = self._http.post_as_get(
            self._order.certificate_url,
            self._account.key,
            self._account.uri,
        )

        cert_pem = response.text

        # Check for alternate chains if preferred chain is specified
        if preferred_chain:
            cert_pem = self._select_preferred_chain(response, preferred_chain) or cert_pem

        # Save certificate
        cert_path, key_path = self._account.get_certificate_paths(
            self._order.common_name
        )
        cert_path.parent.mkdir(parents=True, exist_ok=True)
        cert_path.write_text(cert_pem)
        logger.info("Certificate saved to %s", cert_path)

        # Read private key
        key_pem = key_path.read_text()

        # Clear saved order (completed)
        self._clear_order()

        return cert_pem, key_pem

    def _select_preferred_chain(
        self,
        response: Any,
        preferred_chain: str,
    ) -> str | None:
        """Select preferred certificate chain from Link headers.

        Args:
            response: HTTP response with potential Link headers.
            preferred_chain: Preferred issuer CN substring.

        Returns:
            PEM certificate if preferred chain found, None otherwise.
        """
        from cryptography import x509

        # Parse Link headers for alternate chains
        link_header = response.headers.get("Link", "")
        alternate_urls: list[str] = []

        for link in link_header.split(","):
            link = link.strip()
            if 'rel="alternate"' in link:
                # Extract URL from <url>
                start = link.find("<")
                end = link.find(">")
                if start != -1 and end != -1:
                    alternate_urls.append(link[start + 1 : end])

        if not alternate_urls:
            logger.debug("No alternate certificate chains available")
            return None

        logger.info("Found %d alternate certificate chain(s)", len(alternate_urls))

        # Check default chain first
        try:
            default_cert = x509.load_pem_x509_certificate(response.text.encode())
            issuer_cn = default_cert.issuer.get_attributes_for_oid(
                x509.oid.NameOID.COMMON_NAME
            )
            if issuer_cn and preferred_chain.lower() in str(issuer_cn[0].value).lower():
                logger.info("Default chain matches preferred issuer: %s", issuer_cn[0].value)
                return None  # Use default
        except Exception as e:
            logger.debug("Failed to parse default certificate: %s", e)

        # Check alternate chains
        if not self._account or not self._account.uri:
            return None

        for alt_url in alternate_urls:
            try:
                alt_response = self._http.post_as_get(
                    alt_url,
                    self._account.key,
                    self._account.uri,
                )
                alt_cert = x509.load_pem_x509_certificate(alt_response.text.encode())
                issuer_cn = alt_cert.issuer.get_attributes_for_oid(
                    x509.oid.NameOID.COMMON_NAME
                )

                if issuer_cn and preferred_chain.lower() in str(issuer_cn[0].value).lower():
                    logger.info("Selected alternate chain with issuer: %s", issuer_cn[0].value)
                    return alt_response.text

            except Exception as e:
                logger.debug("Failed to fetch/parse alternate chain %s: %s", alt_url, e)

        logger.warning("Preferred chain '%s' not found, using default", preferred_chain)
        return None

    def deactivate_account(self) -> None:
        """Deactivate the current ACME account.

        This permanently deactivates the account. Once deactivated,
        the account cannot be used for any further operations and
        cannot be reactivated.

        Per RFC 8555 Section 7.3.7.

        Raises:
            AcmeAuthenticationError: If not authenticated or deactivation fails.
        """
        if not self._account or not self._account.uri:
            raise AcmeAuthenticationError("Account not created")

        logger.info("Deactivating account: %s", self._account.uri)

        payload = {"status": "deactivated"}

        response = self._http.post(
            self._account.uri,
            payload,
            self._account.key,
            kid=self._account.uri,
        )

        # Update account status
        data = response.json()
        new_status = data.get("status", "deactivated")
        self._account.save(self._account.uri, new_status)

        logger.info("Account deactivated: %s", self._account.uri)

    def update_account(self, email: str | None = None) -> Account:
        """Update the account contact information.

        Updates the account's contact email address with the ACME server.

        Per RFC 8555 Section 7.3.2.

        Args:
            email: New email address for the account.
                If None, keeps the current email.

        Returns:
            Updated Account object.

        Raises:
            AcmeAuthenticationError: If not authenticated or update fails.
        """
        if not self._account or not self._account.uri:
            raise AcmeAuthenticationError("Account not created")

        if email is None:
            email = self._email

        logger.info("Updating account contact to: %s", email)

        payload = {"contact": [f"mailto:{email}"]}

        response = self._http.post(
            self._account.uri,
            payload,
            self._account.key,
            kid=self._account.uri,
        )

        data = response.json()
        status = data.get("status", "valid")
        self._account.save(self._account.uri, status)
        self._email = email

        logger.info("Account contact updated: %s", email)
        return self._account

    def key_rollover(self) -> None:
        """Roll over the account key to a new key.

        Generates a new account key and updates the ACME server to use it.
        The old key is replaced and can no longer be used.

        Per RFC 8555 Section 7.3.5.

        Raises:
            AcmeAuthenticationError: If not authenticated or rollover fails.
            AcmeNetworkError: If key-change URL is not available.
        """
        from acmeow._internal.crypto import generate_account_key, get_jwk

        if not self._account or not self._account.uri:
            raise AcmeAuthenticationError("Account not created")

        key_change_url = self._directory.get("keyChange")
        if not key_change_url:
            raise AcmeNetworkError("Directory missing keyChange URL")

        logger.info("Rolling over account key for: %s", self._account.uri)

        # Generate new key
        new_key = generate_account_key()
        new_jwk = get_jwk(new_key)

        # Create inner JWS payload
        inner_payload = {
            "account": self._account.uri,
            "oldKey": self._account.jwk,
        }

        # Create inner JWS protected header
        inner_protected = {
            "alg": "ES256",
            "jwk": new_jwk,
            "url": key_change_url,
        }

        # Sign inner JWS with new key
        from acmeow._internal.crypto import sign_es256

        inner_protected_b64 = base64url_encode_json(inner_protected)
        inner_payload_b64 = base64url_encode_json(inner_payload)
        inner_signing_input = f"{inner_protected_b64}.{inner_payload_b64}".encode("ascii")
        inner_signature = sign_es256(new_key, inner_signing_input)
        inner_signature_b64 = base64url_encode(inner_signature)

        # The inner JWS becomes the outer payload
        outer_payload = {
            "protected": inner_protected_b64,
            "payload": inner_payload_b64,
            "signature": inner_signature_b64,
        }

        # Send outer JWS signed with old key
        self._http.post(
            key_change_url,
            outer_payload,
            self._account.key,
            kid=self._account.uri,
        )

        # Update account with new key
        self._account.update_key(new_key)
        logger.info("Account key rollover complete")

    def revoke_certificate(
        self,
        certificate: str | bytes,
        reason: RevocationReason | None = None,
    ) -> None:
        """Revoke a certificate.

        Revokes the specified certificate with the ACME server.

        Per RFC 8555 Section 7.6.

        Args:
            certificate: PEM-encoded certificate string or DER-encoded bytes.
            reason: Optional revocation reason code.

        Raises:
            AcmeAuthenticationError: If not authenticated.
            AcmeCertificateError: If revocation fails.
            AcmeNetworkError: If revokeCert URL is not available.
        """
        import base64

        from cryptography import x509

        if not self._account or not self._account.uri:
            raise AcmeAuthenticationError("Account not created")

        revoke_cert_url = self._directory.get("revokeCert")
        if not revoke_cert_url:
            raise AcmeNetworkError("Directory missing revokeCert URL")

        # Convert PEM to DER if needed
        if isinstance(certificate, str):
            cert = x509.load_pem_x509_certificate(certificate.encode())
            cert_der = cert.public_bytes(serialization.Encoding.DER)
        else:
            cert_der = certificate

        logger.info("Revoking certificate")

        # Build payload
        payload: dict[str, Any] = {
            "certificate": base64.urlsafe_b64encode(cert_der).rstrip(b"=").decode("ascii"),
        }
        if reason is not None:
            payload["reason"] = reason.value

        self._http.post(
            revoke_cert_url,
            payload,
            self._account.key,
            kid=self._account.uri,
        )

        logger.info("Certificate revoked successfully")

    def close(self) -> None:
        """Close the client and release resources."""
        self._http.close()

    def __enter__(self) -> AcmeClient:
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.close()
