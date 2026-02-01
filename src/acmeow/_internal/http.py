"""HTTP client wrapper for ACME protocol communication.

Provides a configured requests session with JWS signing support
for authenticated ACME requests, including retry with exponential backoff.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any

import requests
from cryptography.hazmat.primitives.asymmetric import ec

from acmeow._internal.crypto import sign_es256
from acmeow._internal.encoding import base64url_encode, base64url_encode_json
from acmeow.exceptions import AcmeNetworkError, AcmeRateLimitError, AcmeServerError

logger = logging.getLogger(__name__)

# HTTP status codes that trigger retry
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

# Default retry configuration
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_DELAY = 1.0  # seconds
DEFAULT_RETRY_MAX_DELAY = 60.0  # seconds
DEFAULT_RETRY_MULTIPLIER = 2.0


class RetryConfig:
    """Configuration for retry behavior with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts. Default 5.
        initial_delay: Initial delay between retries in seconds. Default 1.0.
        max_delay: Maximum delay between retries in seconds. Default 60.0.
        multiplier: Multiplier for exponential backoff. Default 2.0.
        jitter: Whether to add random jitter to delays. Default True.
    """

    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_delay: float = DEFAULT_RETRY_DELAY,
        max_delay: float = DEFAULT_RETRY_MAX_DELAY,
        multiplier: float = DEFAULT_RETRY_MULTIPLIER,
        jitter: bool = True,
    ) -> None:
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given retry attempt.

        Args:
            attempt: The retry attempt number (0-based).

        Returns:
            Delay in seconds before the next retry.
        """
        delay = self.initial_delay * (self.multiplier ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            # Add up to 25% random jitter
            delay = delay * (0.75 + random.random() * 0.5)

        return delay


class AcmeHttpClient:
    """HTTP client for ACME protocol communication.

    Handles nonce management, JWS signing, error handling, and automatic
    retry with exponential backoff for all ACME server communication.

    Args:
        verify_ssl: Whether to verify SSL certificates. Default True.
        timeout: Request timeout in seconds. Default 30.
        retry_config: Retry configuration. Default uses standard backoff.
    """

    def __init__(
        self,
        verify_ssl: bool = True,
        timeout: int = 30,
        retry_config: RetryConfig | None = None,
    ) -> None:
        self._session = requests.Session()
        self._session.verify = verify_ssl
        self._timeout = timeout
        self._retry_config = retry_config or RetryConfig()
        self._nonce: str | None = None
        self._nonce_lock = threading.Lock()
        self._new_nonce_url: str | None = None

    @property
    def retry_config(self) -> RetryConfig:
        """Current retry configuration."""
        return self._retry_config

    def set_nonce_url(self, url: str) -> None:
        """Set the URL for fetching new nonces.

        Args:
            url: The newNonce endpoint URL from the ACME directory.
        """
        self._new_nonce_url = url

    def _get_nonce(self) -> str:
        """Get a nonce for the next request.

        Thread-safe nonce management. Returns cached nonce if available,
        otherwise fetches a new one from the server.

        Returns:
            Nonce string for replay protection.

        Raises:
            AcmeNetworkError: If nonce cannot be fetched.
        """
        with self._nonce_lock:
            if self._nonce is not None:
                nonce = self._nonce
                self._nonce = None
                return nonce

        return self._fetch_nonce()

    def _fetch_nonce(self) -> str:
        """Fetch a fresh nonce from the ACME server.

        Returns:
            New nonce string.

        Raises:
            AcmeNetworkError: If nonce cannot be fetched.
        """
        if not self._new_nonce_url:
            raise AcmeNetworkError("Nonce URL not configured")

        try:
            response = self._session.head(self._new_nonce_url, timeout=self._timeout)
            response.raise_for_status()
            nonce = response.headers.get("Replay-Nonce")
            if not nonce:
                raise AcmeNetworkError("Server did not return Replay-Nonce header")
            return nonce
        except requests.RequestException as e:
            raise AcmeNetworkError(f"Failed to fetch nonce: {e}", e) from e

    def _update_nonce(self, response: requests.Response) -> None:
        """Update cached nonce from response headers.

        Args:
            response: HTTP response from ACME server.
        """
        nonce = response.headers.get("Replay-Nonce")
        if nonce:
            with self._nonce_lock:
                self._nonce = nonce

    def _should_retry(self, response: requests.Response | None, exception: Exception | None) -> bool:
        """Determine if a request should be retried.

        Args:
            response: HTTP response if available.
            exception: Exception if request failed.

        Returns:
            True if the request should be retried.
        """
        if exception is not None:
            # Retry on connection errors
            return isinstance(exception, (
                requests.ConnectionError,
                requests.Timeout,
            ))

        if response is not None:
            # Retry on specific status codes
            return response.status_code in RETRYABLE_STATUS_CODES

        return False

    def _get_retry_after(self, response: requests.Response) -> float | None:
        """Extract Retry-After header value.

        Args:
            response: HTTP response.

        Returns:
            Delay in seconds, or None if header not present.
        """
        retry_after = response.headers.get("Retry-After")
        if retry_after is None:
            return None

        try:
            return float(retry_after)
        except ValueError:
            # Retry-After might be a date, ignore it
            return None

    def get(self, url: str) -> requests.Response:
        """Make an unauthenticated GET request with retry.

        Args:
            url: URL to fetch.

        Returns:
            HTTP response.

        Raises:
            AcmeNetworkError: If request fails after all retries.
        """
        last_exception: Exception | None = None

        for attempt in range(self._retry_config.max_retries + 1):
            try:
                response = self._session.get(url, timeout=self._timeout)
                self._update_nonce(response)

                if not self._should_retry(response, None):
                    return response

                # Check for Retry-After header
                retry_after = self._get_retry_after(response)
                delay = retry_after or self._retry_config.get_delay(attempt)

                logger.warning(
                    "GET %s returned %d, retrying in %.1fs (attempt %d/%d)",
                    url,
                    response.status_code,
                    delay,
                    attempt + 1,
                    self._retry_config.max_retries + 1,
                )
                time.sleep(delay)

            except requests.RequestException as e:
                last_exception = e

                if not self._should_retry(None, e):
                    raise AcmeNetworkError(f"GET request failed: {e}", e) from e

                if attempt < self._retry_config.max_retries:
                    delay = self._retry_config.get_delay(attempt)
                    logger.warning(
                        "GET %s failed: %s, retrying in %.1fs (attempt %d/%d)",
                        url,
                        e,
                        delay,
                        attempt + 1,
                        self._retry_config.max_retries + 1,
                    )
                    time.sleep(delay)

        if last_exception:
            raise AcmeNetworkError(
                f"GET request failed after {self._retry_config.max_retries + 1} attempts: {last_exception}",
                last_exception,
            ) from last_exception

        raise AcmeNetworkError(f"GET request failed after {self._retry_config.max_retries + 1} attempts")

    def post(
        self,
        url: str,
        payload: dict[str, Any] | str,
        key: ec.EllipticCurvePrivateKey,
        jwk: dict[str, str] | None = None,
        kid: str | None = None,
    ) -> requests.Response:
        """Make a signed POST request to the ACME server with retry.

        Creates a JWS-signed request with the provided payload and
        either a JWK (for new account) or KID (for existing account).

        Args:
            url: URL to post to.
            payload: Request payload (dict or empty string for POST-as-GET).
            key: Account private key for signing.
            jwk: JWK for new account requests (mutually exclusive with kid).
            kid: Account URL for existing account requests.

        Returns:
            HTTP response.

        Raises:
            AcmeNetworkError: If request fails after all retries.
            AcmeServerError: If server returns an error response.
            AcmeRateLimitError: If rate limited and retries exhausted.
        """
        if not jwk and not kid:
            raise AcmeNetworkError("Either jwk or kid must be provided")

        last_exception: Exception | None = None
        last_response: requests.Response | None = None

        for attempt in range(self._retry_config.max_retries + 1):
            try:
                # Get fresh nonce for each attempt
                nonce = self._get_nonce()

                # Build protected header
                protected: dict[str, Any] = {
                    "alg": "ES256",
                    "nonce": nonce,
                    "url": url,
                }

                if jwk:
                    protected["jwk"] = jwk
                else:
                    protected["kid"] = kid

                # Encode protected header and payload
                protected_b64 = base64url_encode_json(protected)
                payload_b64 = "" if payload == "" else base64url_encode_json(payload)

                # Sign
                signing_input = f"{protected_b64}.{payload_b64}".encode("ascii")
                signature = sign_es256(key, signing_input)
                signature_b64 = base64url_encode(signature)

                # Build JWS
                jws = {
                    "protected": protected_b64,
                    "payload": payload_b64,
                    "signature": signature_b64,
                }

                response = self._session.post(
                    url,
                    json=jws,
                    headers={"Content-Type": "application/jose+json"},
                    timeout=self._timeout,
                )
                self._update_nonce(response)
                last_response = response

                # Check if we should retry
                if self._should_retry(response, None):
                    retry_after = self._get_retry_after(response)
                    delay = retry_after or self._retry_config.get_delay(attempt)

                    logger.warning(
                        "POST %s returned %d, retrying in %.1fs (attempt %d/%d)",
                        url,
                        response.status_code,
                        delay,
                        attempt + 1,
                        self._retry_config.max_retries + 1,
                    )
                    time.sleep(delay)
                    continue

                # Check for error response (non-retryable)
                if response.status_code >= 400:
                    self._handle_error_response(response)

                return response

            except (AcmeServerError, AcmeRateLimitError):
                # Don't retry ACME errors (except rate limits handled above)
                raise

            except requests.RequestException as e:
                last_exception = e

                if not self._should_retry(None, e):
                    raise AcmeNetworkError(f"POST request failed: {e}", e) from e

                if attempt < self._retry_config.max_retries:
                    delay = self._retry_config.get_delay(attempt)
                    logger.warning(
                        "POST %s failed: %s, retrying in %.1fs (attempt %d/%d)",
                        url,
                        e,
                        delay,
                        attempt + 1,
                        self._retry_config.max_retries + 1,
                    )
                    time.sleep(delay)

        # All retries exhausted
        if last_response is not None and last_response.status_code == 429:
            raise AcmeRateLimitError(
                f"Rate limited after {self._retry_config.max_retries + 1} attempts"
            )

        if last_exception:
            raise AcmeNetworkError(
                f"POST request failed after {self._retry_config.max_retries + 1} attempts: {last_exception}",
                last_exception,
            ) from last_exception

        raise AcmeNetworkError(f"POST request failed after {self._retry_config.max_retries + 1} attempts")

    def post_as_get(
        self,
        url: str,
        key: ec.EllipticCurvePrivateKey,
        kid: str,
    ) -> requests.Response:
        """Make a POST-as-GET request to the ACME server.

        POST-as-GET is used to fetch resources that require authentication
        but don't require a payload (e.g., fetching authorizations).

        Args:
            url: URL to fetch.
            key: Account private key for signing.
            kid: Account URL.

        Returns:
            HTTP response.
        """
        return self.post(url, "", key, kid=kid)

    def _handle_error_response(self, response: requests.Response) -> None:
        """Parse and raise an exception for ACME error responses.

        Args:
            response: HTTP response with error status.

        Raises:
            AcmeServerError: With parsed error details.
            AcmeRateLimitError: If rate limited (429).
        """
        try:
            error_data = response.json()
            error_type = error_data.get("type", "unknown")
            detail = error_data.get("detail", "No details provided")

            logger.error(
                "ACME server error: status=%d type=%s detail=%s",
                response.status_code,
                error_type,
                detail,
            )

            # Check for rate limit error
            if response.status_code == 429 or "rateLimited" in error_type:
                retry_after = self._get_retry_after(response)
                raise AcmeRateLimitError(detail, retry_after)

            raise AcmeServerError(response.status_code, error_type, detail)

        except ValueError as e:
            # Response is not JSON
            if response.status_code == 429:
                raise AcmeRateLimitError(
                    response.text or "Rate limited",
                    self._get_retry_after(response),
                ) from e

            raise AcmeServerError(
                response.status_code,
                "unknown",
                response.text or "Unknown error",
            ) from e

    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()
