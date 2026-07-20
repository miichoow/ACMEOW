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
        type: The challenge type.
        status: Current challenge status.
        url: URL for responding to and polling the challenge.
        token: Challenge token provided by the server. ``None`` for challenge
            types that do not use one, such as DNS-PERSIST-01.
        validated: Timestamp when the challenge was validated (if valid).
        error: Error details if the challenge failed.
        accounturi: DNS-PERSIST-01 only. URI identifying the ACME account that
            must appear in the ``accounturi`` parameter of the TXT record.
        issuer_domain_names: DNS-PERSIST-01 only. Issuer Domain Names the CA
            will accept; the client picks one for the record's issue-value.
    """

    type: ChallengeType
    status: ChallengeStatus
    url: str
    token: str | None = None
    validated: str | None = None
    error: dict[str, str] | None = None
    accounturi: str | None = None
    issuer_domain_names: tuple[str, ...] = ()

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
        elif challenge_type == "dns-persist-01":
            ctype = ChallengeType.DNS_PERSIST
        else:
            # Default to DNS for unknown types
            ctype = ChallengeType.DNS

        # DNS-PERSIST-01 challenges carry no token; every other type does.
        return cls(
            type=ctype,
            status=ChallengeStatus(data.get("status", "pending")),
            url=data["url"],
            token=data.get("token"),
            validated=data.get("validated"),
            error=data.get("error"),
            accounturi=data.get("accounturi"),
            issuer_domain_names=tuple(data.get("issuer-domain-names", ())),
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
