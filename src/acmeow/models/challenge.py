"""Challenge model for ACME protocol.

Represents ACME challenges used to prove control over identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from acmeow.enums import ChallengeStatus, ChallengeType


@dataclass(frozen=True, slots=True)
class Challenge:
    """An ACME challenge for domain validation.

    Challenges are used to prove control over an identifier.
    They are immutable; status changes require fetching updated
    challenge data from the server.

    Args:
        type: The challenge type (DNS-01 or HTTP-01).
        status: Current challenge status.
        url: URL for responding to and polling the challenge.
        token: Challenge token provided by the server.
        validated: Timestamp when the challenge was validated (if valid).
        error: Error details if the challenge failed.
    """

    type: ChallengeType
    status: ChallengeStatus
    url: str
    token: str
    validated: str | None = None
    error: dict[str, str] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Challenge:
        """Create a Challenge from an ACME response dictionary.

        Args:
            data: Challenge object from ACME server response.

        Returns:
            New Challenge instance.
        """
        challenge_type = data["type"]
        # Map ACME challenge type strings to enum
        if challenge_type == "dns-01":
            ctype = ChallengeType.DNS
        elif challenge_type == "http-01":
            ctype = ChallengeType.HTTP
        elif challenge_type == "tls-alpn-01":
            ctype = ChallengeType.TLS_ALPN
        else:
            # Default to DNS for unknown types
            ctype = ChallengeType.DNS

        return cls(
            type=ctype,
            status=ChallengeStatus(data.get("status", "pending")),
            url=data["url"],
            token=data["token"],
            validated=data.get("validated"),
            error=data.get("error"),
        )

    @property
    def is_pending(self) -> bool:
        """Check if the challenge is pending response."""
        return self.status == ChallengeStatus.PENDING

    @property
    def is_processing(self) -> bool:
        """Check if the challenge is being validated."""
        return self.status == ChallengeStatus.PROCESSING

    @property
    def is_valid(self) -> bool:
        """Check if the challenge was successfully validated."""
        return self.status == ChallengeStatus.VALID

    @property
    def is_invalid(self) -> bool:
        """Check if the challenge failed validation."""
        return self.status == ChallengeStatus.INVALID

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return f"Challenge({self.type.value}, status={self.status.value})"
