"""Exception hierarchy for the ACME client library.

All exceptions inherit from AcmeError, allowing callers to catch all
ACME-related errors with a single except clause if desired.
"""

from __future__ import annotations


class AcmeError(Exception):
    """Base exception for all ACME errors.

    Args:
        message: Human-readable error description.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class AcmeServerError(AcmeError):
    """ACME server returned an error response.

    This exception is raised when the ACME server responds with an error
    status code and provides error details in the RFC 8555 format.

    Args:
        status_code: HTTP status code from the server.
        error_type: ACME error type URN (e.g., "urn:ietf:params:acme:error:malformed").
        detail: Human-readable error description from the server.
    """

    def __init__(self, status_code: int, error_type: str, detail: str) -> None:
        self.status_code = status_code
        self.error_type = error_type
        self.detail = detail
        message = f"ACME server error {status_code}: {error_type} - {detail}"
        super().__init__(message)


class AcmeAuthenticationError(AcmeError):
    """Account authentication failed.

    Raised when account creation, account key verification, or
    request signing fails.
    """


class AcmeAuthorizationError(AcmeError):
    """Challenge authorization failed.

    Raised when a challenge fails validation or an authorization
    enters an invalid state.

    Args:
        domain: The domain that failed authorization.
        message: Human-readable error description.
    """

    def __init__(self, domain: str, message: str) -> None:
        self.domain = domain
        super().__init__(f"Authorization failed for {domain}: {message}")


class AcmeOrderError(AcmeError):
    """Order creation or finalization failed.

    Raised when an order cannot be created, enters an invalid state,
    or finalization fails.
    """


class AcmeCertificateError(AcmeError):
    """Certificate download or validation failed.

    Raised when the certificate cannot be downloaded, parsed, or
    saved to disk.
    """


class AcmeConfigurationError(AcmeError):
    """Invalid client configuration.

    Raised when the client is configured with invalid parameters,
    such as an invalid email address or unsupported key type.
    """


class AcmeNetworkError(AcmeError):
    """Network communication error.

    Raised when network communication with the ACME server fails,
    including connection errors, timeouts, and TLS errors.

    Args:
        message: Human-readable error description.
        original_error: The underlying exception that caused this error.
    """

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        self.original_error = original_error
        super().__init__(message)


class AcmeTimeoutError(AcmeError):
    """Operation timed out.

    Raised when an operation exceeds its timeout, such as waiting
    for an order to become ready or a challenge to be validated.
    """


class AcmeRateLimitError(AcmeError):
    """Rate limit exceeded.

    Raised when the ACME server returns a 429 status code or
    a rateLimited error type.

    Args:
        message: Human-readable error description.
        retry_after: Suggested wait time in seconds, if provided by server.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        super().__init__(message)


class AcmeDnsError(AcmeError):
    """DNS verification failed.

    Raised when DNS propagation check fails or DNS records
    cannot be verified.

    Args:
        domain: The domain that failed DNS verification.
        message: Human-readable error description.
    """

    def __init__(self, domain: str, message: str) -> None:
        self.domain = domain
        super().__init__(f"DNS verification failed for {domain}: {message}")
