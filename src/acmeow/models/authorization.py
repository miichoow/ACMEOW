"""Authorization model for ACME protocol.

Represents ACME authorizations that prove control over identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from acmeow.enums import AuthorizationStatus, ChallengeType
from acmeow.models.challenge import Challenge
from acmeow.models.identifier import Identifier


@dataclass(frozen=True, slots=True)
class Authorization:
    """An ACME authorization for an identifier.

    Authorizations represent the server's acknowledgment that the
    account holder has proven control over an identifier. They contain
    one or more challenges that can be completed to validate control.

    Args:
        identifier: The identifier this authorization is for.
        status: Current authorization status.
        url: URL for polling authorization status.
        expires: Expiration timestamp for the authorization.
        challenges: List of available challenges.
        wildcard: Whether this is a wildcard authorization.
    """

    identifier: Identifier
    status: AuthorizationStatus
    url: str
    expires: str | None = None
    challenges: tuple[Challenge, ...] = field(default_factory=tuple)
    wildcard: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any], url: str) -> Authorization:
        """Create an Authorization from an ACME response dictionary.

        Args:
            data: Authorization object from ACME server response.
            url: The authorization URL.

        Returns:
            New Authorization instance.
        """
        challenges = tuple(
            Challenge.from_dict(c)
            for c in data.get("challenges", [])
        )

        return cls(
            identifier=Identifier.from_dict(data["identifier"]),
            status=AuthorizationStatus(data.get("status", "pending")),
            url=url,
            expires=data.get("expires"),
            challenges=challenges,
            wildcard=data.get("wildcard", False),
        )

    def get_challenge(self, challenge_type: ChallengeType) -> Challenge | None:
        """Get a challenge of the specified type.

        Args:
            challenge_type: The type of challenge to find.

        Returns:
            The matching challenge, or None if not found.
        """
        for challenge in self.challenges:
            if challenge.type == challenge_type:
                return challenge
        return None

    def get_dns_challenge(self) -> Challenge | None:
        """Get the DNS-01 challenge if available.

        Returns:
            The DNS-01 challenge, or None if not available.
        """
        return self.get_challenge(ChallengeType.DNS)

    def get_http_challenge(self) -> Challenge | None:
        """Get the HTTP-01 challenge if available.

        Returns:
            The HTTP-01 challenge, or None if not available.
        """
        return self.get_challenge(ChallengeType.HTTP)

    def get_tls_alpn_challenge(self) -> Challenge | None:
        """Get the TLS-ALPN-01 challenge if available.

        Returns:
            The TLS-ALPN-01 challenge, or None if not available.
        """
        return self.get_challenge(ChallengeType.TLS_ALPN)

    @property
    def is_pending(self) -> bool:
        """Check if the authorization is pending completion."""
        return self.status == AuthorizationStatus.PENDING

    @property
    def is_valid(self) -> bool:
        """Check if the authorization has been validated."""
        return self.status == AuthorizationStatus.VALID

    @property
    def is_invalid(self) -> bool:
        """Check if the authorization failed."""
        return self.status == AuthorizationStatus.INVALID

    @property
    def domain(self) -> str:
        """Get the domain name for this authorization.

        Returns:
            The identifier value (domain name).
        """
        return self.identifier.value

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return f"Authorization({self.identifier}, status={self.status.value})"
