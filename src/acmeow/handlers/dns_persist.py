"""DNS-PERSIST-01 challenge handlers.

Implements the persistent DNS validation method defined by
draft-ietf-acme-dns-persist.

Unlike DNS-01, this method does not use a per-challenge token or key
authorization. The client publishes a long-lived TXT record at
``_validation-persist.<domain>`` whose value follows the ``issue-value``
syntax of RFC 8659 (the CAA record format) and binds an Issuer Domain Name
to the validating ACME account::

    _validation-persist.example.com. IN TXT "ca.example; accounturi=https://ca.example/acct/1"

Because the record authorizes future issuance, it is **not** removed after
validation by default. Removing it would defeat the purpose of the method.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

PERSIST_LABEL = "_validation-persist"
"""DNS label prepended to the domain to form the Validation Domain Name."""

WILDCARD_POLICY = "wildcard"
"""``policy`` parameter value that extends authorization to wildcard names."""

# RFC 8659 section 4.2: a parameter tag is one or more alphanumerics.
_TAG_RE = re.compile(r"^[A-Za-z0-9]+$")

# RFC 8659 section 4.2: a parameter value is printable US-ASCII excluding
# ";" (%x3B), which terminates the parameter. Whitespace is excluded too so a
# value cannot silently absorb the separator between parameters.
_VALUE_RE = re.compile(r"^[\x21-\x3A\x3C-\x7E]+$")

# An issuer-domain-name is a DNS name; allow the usual LDH label set. The empty
# string is deliberately rejected here -- ";" alone means "no issuer permitted"
# in CAA, which is never a valid dns-persist-01 record.
_ISSUER_RE = re.compile(
    r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)(\.(?!-)[A-Za-z0-9-]{1,63}(?<!-))*\.?$"
)


class DnsPersistError(ValueError):
    """Raised when a DNS-PERSIST-01 record value is malformed."""


@dataclass(frozen=True, slots=True)
class PersistRecordValue:
    """A parsed DNS-PERSIST-01 TXT record value.

    Attributes:
        issuer_domain_name: The Issuer Domain Name authorized by this record.
        accounturi: URI of the ACME account authorized to request issuance.
        policy: Optional ``policy`` parameter (e.g. ``"wildcard"``).
        persist_until: Optional expiry as a UNIX timestamp, after which the
            record should no longer be honoured.
        parameters: Any additional parameters, preserved in encounter order.
    """

    issuer_domain_name: str
    accounturi: str
    policy: str | None = None
    persist_until: int | None = None
    parameters: tuple[tuple[str, str], ...] = ()

    @property
    def allows_wildcard(self) -> bool:
        """Whether this record authorizes wildcard issuance."""
        return self.policy == WILDCARD_POLICY

    @property
    def is_expired(self) -> bool:
        """Whether ``persist_until`` is in the past.

        Records without a ``persist_until`` parameter never expire.
        """
        if self.persist_until is None:
            return False
        return datetime.now(timezone.utc).timestamp() > self.persist_until

    def __str__(self) -> str:
        """Render the record value in RFC 8659 issue-value syntax."""
        parts = [self.issuer_domain_name, f"accounturi={self.accounturi}"]
        if self.policy is not None:
            parts.append(f"policy={self.policy}")
        if self.persist_until is not None:
            parts.append(f"persistUntil={self.persist_until}")
        parts.extend(f"{tag}={value}" for tag, value in self.parameters)
        return "; ".join(parts)


def validation_domain_name(domain: str) -> str:
    """Build the Validation Domain Name for a domain.

    A wildcard identifier is validated through the record for its base domain;
    the wildcard itself is authorized by ``policy=wildcard`` in the record
    value rather than by a separate record name.

    Args:
        domain: The domain being validated, optionally a ``*.`` wildcard.

    Returns:
        The Validation Domain Name (e.g. ``_validation-persist.example.com``).

    Example:
        >>> validation_domain_name("example.com")
        '_validation-persist.example.com'
        >>> validation_domain_name("*.example.com")
        '_validation-persist.example.com'
    """
    base = domain[2:] if domain.startswith("*.") else domain
    return f"{PERSIST_LABEL}.{base}"


def build_record_value(
    issuer_domain_name: str,
    accounturi: str,
    policy: str | None = None,
    persist_until: int | datetime | None = None,
    parameters: dict[str, str] | None = None,
) -> str:
    """Build a DNS-PERSIST-01 TXT record value.

    The result follows the ``issue-value`` syntax of RFC 8659 section 4.2.

    Args:
        issuer_domain_name: An Issuer Domain Name from the challenge's
            ``issuer-domain-names`` list.
        accounturi: The challenge's ``accounturi`` value.
        policy: Optional policy, e.g. :data:`WILDCARD_POLICY`.
        persist_until: Optional expiry, as a UNIX timestamp or a datetime.
            Naive datetimes are interpreted as UTC.
        parameters: Optional additional parameters.

    Returns:
        The record value, e.g. ``"ca.example; accounturi=https://ca.example/acct/1"``.

    Raises:
        DnsPersistError: If any component is not representable in issue-value
            syntax. Both ``issuer_domain_name`` and ``accounturi`` originate
            from the CA, so they are validated rather than trusted: a value
            containing ``";"`` would otherwise inject extra parameters into
            the published record.
    """
    if not _ISSUER_RE.match(issuer_domain_name):
        raise DnsPersistError(
            f"Invalid issuer domain name: {issuer_domain_name!r}"
        )

    parts = [issuer_domain_name, f"accounturi={_check_value('accounturi', accounturi)}"]

    if policy is not None:
        parts.append(f"policy={_check_value('policy', policy)}")

    if persist_until is not None:
        parts.append(f"persistUntil={_normalize_persist_until(persist_until)}")

    for tag, value in (parameters or {}).items():
        if not _TAG_RE.match(tag):
            raise DnsPersistError(f"Invalid parameter tag: {tag!r}")
        parts.append(f"{tag}={_check_value(tag, value)}")

    return "; ".join(parts)


def parse_record_value(value: str) -> PersistRecordValue:
    """Parse a DNS-PERSIST-01 TXT record value.

    Args:
        value: The raw TXT record value.

    Returns:
        The parsed record.

    Raises:
        DnsPersistError: If the value is not a well-formed issue-value or is
            missing the mandatory ``accounturi`` parameter.

    Example:
        >>> parsed = parse_record_value("ca.example; accounturi=https://ca.example/a/1")
        >>> parsed.issuer_domain_name
        'ca.example'
    """
    segments = value.split(";")
    issuer = segments[0].strip()

    if not _ISSUER_RE.match(issuer):
        raise DnsPersistError(f"Invalid issuer domain name: {issuer!r}")

    accounturi: str | None = None
    policy: str | None = None
    persist_until: int | None = None
    extra: list[tuple[str, str]] = []

    for segment in segments[1:]:
        segment = segment.strip()
        if not segment:
            # RFC 8659 permits trailing/empty separators.
            continue

        tag, sep, raw = segment.partition("=")
        if not sep:
            raise DnsPersistError(f"Parameter is not a tag=value pair: {segment!r}")

        tag = tag.strip()
        raw = raw.strip()
        if not _TAG_RE.match(tag):
            raise DnsPersistError(f"Invalid parameter tag: {tag!r}")

        if tag == "accounturi":
            accounturi = _check_value(tag, raw)
        elif tag == "policy":
            policy = _check_value(tag, raw)
        elif tag == "persistUntil":
            try:
                persist_until = int(raw)
            except ValueError:
                raise DnsPersistError(
                    f"persistUntil is not an integer timestamp: {raw!r}"
                ) from None
        else:
            extra.append((tag, _check_value(tag, raw)))

    if accounturi is None:
        raise DnsPersistError("Record is missing the mandatory accounturi parameter")

    return PersistRecordValue(
        issuer_domain_name=issuer,
        accounturi=accounturi,
        policy=policy,
        persist_until=persist_until,
        parameters=tuple(extra),
    )


def select_issuer_domain_name(
    issuer_domain_names: Sequence[str],
    preferred: str | None = None,
) -> str:
    """Choose which Issuer Domain Name to publish.

    Args:
        issuer_domain_names: The list offered by the challenge.
        preferred: An Issuer Domain Name to prefer, if the CA offers it.

    Returns:
        ``preferred`` when the CA offers it, otherwise the first offered name.

    Raises:
        DnsPersistError: If the CA offered no names, or if ``preferred`` was
            given but is not among them. Publishing a name the CA did not
            offer would fail validation, so this is reported rather than
            silently falling back.
    """
    if not issuer_domain_names:
        raise DnsPersistError("Challenge offered no issuer-domain-names")

    if preferred is None:
        return issuer_domain_names[0]

    if preferred not in issuer_domain_names:
        raise DnsPersistError(
            f"Preferred issuer domain name {preferred!r} is not offered by the CA; "
            f"available: {', '.join(issuer_domain_names)}"
        )

    return preferred


def _check_value(tag: str, value: str) -> str:
    """Validate a parameter value against RFC 8659 issue-value syntax."""
    if not _VALUE_RE.match(value):
        raise DnsPersistError(
            f"Invalid value for parameter {tag!r}: {value!r} "
            "(must be printable US-ASCII without ';' or whitespace)"
        )
    return value


def _normalize_persist_until(persist_until: int | datetime) -> int:
    """Coerce a persistUntil value to an integer UNIX timestamp."""
    if isinstance(persist_until, datetime):
        moment = persist_until
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return int(moment.timestamp())
    return int(persist_until)


class DnsPersistHandler(ABC):
    """Abstract base class for DNS-PERSIST-01 challenge handlers.

    This is deliberately a separate interface from
    :class:`~acmeow.handlers.base.ChallengeHandler`: DNS-PERSIST-01 has no
    token and no key authorization, and its record is meant to outlive the
    validation, so the ``setup``/``cleanup`` contract of the other challenge
    types does not apply.

    Example:
        >>> class MyPersistHandler(DnsPersistHandler):
        ...     def setup(self, domain, record_name, record_value):
        ...         dns_api.upsert(record_name, "TXT", record_value)
    """

    propagation_delay: int = 60
    """Seconds to wait after publishing the record before notifying the CA."""

    persist: bool = True
    """Whether the record is kept after validation. See :meth:`cleanup`."""

    @abstractmethod
    def setup(self, domain: str, record_name: str, record_value: str) -> None:
        """Publish the persistent validation record.

        Implementations should treat this as an upsert: the record name is
        stable per domain, so a record left over from an earlier issuance may
        already exist and should be replaced rather than duplicated.

        Args:
            domain: The domain being validated (may be a ``*.`` wildcard).
            record_name: The Validation Domain Name to publish at.
            record_value: The issue-value formatted TXT record value.

        Raises:
            Exception: If the record cannot be published.
        """

    def cleanup(self, domain: str, record_name: str) -> None:  # noqa: B027
        """Remove the validation record.

        The default implementation does nothing. The record authorizes future
        issuance, so removing it after a successful validation would discard
        the benefit of using this challenge type at all. Override only when
        the record is genuinely meant to be single-use.

        Args:
            domain: The domain that was validated.
            record_name: The Validation Domain Name that was published.
        """


class CallbackDnsPersistHandler(DnsPersistHandler):
    """DNS-PERSIST-01 handler backed by user-provided callbacks.

    Args:
        create_record: Callback publishing the TXT record.
            Signature: ``(domain, record_name, record_value) -> None``.
        delete_record: Optional callback removing the TXT record.
            Signature: ``(domain, record_name) -> None``. Only invoked when
            ``persist`` is False.
        propagation_delay: Seconds to wait after publishing. Default 60.
        persist: Whether to keep the record after validation. Default True,
            matching the intent of the challenge type.

    Example:
        >>> def upsert(domain, name, value):
        ...     dns_api.upsert(name, "TXT", value)
        >>> handler = CallbackDnsPersistHandler(upsert)
    """

    def __init__(
        self,
        create_record: Callable[[str, str, str], None],
        delete_record: Callable[[str, str], None] | None = None,
        propagation_delay: int = 60,
        persist: bool = True,
    ) -> None:
        self._create_record = create_record
        self._delete_record = delete_record
        self.propagation_delay = propagation_delay
        self.persist = persist

    def setup(self, domain: str, record_name: str, record_value: str) -> None:
        """Publish the persistent validation record via the callback."""
        logger.info(
            "Publishing DNS-PERSIST TXT record: %s = %s",
            record_name,
            record_value,
        )
        self._create_record(domain, record_name, record_value)

    def cleanup(self, domain: str, record_name: str) -> None:
        """Remove the record, if ``persist`` is False and a callback was given."""
        if self.persist:
            logger.debug("Keeping persistent record %s", record_name)
            return

        if self._delete_record is None:
            logger.warning(
                "persist=False but no delete_record callback was provided; "
                "leaving %s in place",
                record_name,
            )
            return

        logger.info("Removing DNS-PERSIST TXT record: %s", record_name)
        try:
            self._delete_record(domain, record_name)
        except Exception as e:
            logger.warning("Failed to remove record %s: %s", record_name, e)


class ManualDnsPersistHandler(DnsPersistHandler):
    """DNS-PERSIST-01 handler that prints the record and waits for confirmation.

    Intended for domains whose DNS is managed by hand or by another team.

    Args:
        propagation_delay: Seconds to wait after confirmation. Default 0,
            since the operator confirms the record is already live.
        prompt: Whether to block on input() until the operator confirms.
            Default True. Set False for unattended runs that only need the
            record printed to the log.

    Example:
        >>> client.complete_dns_persist_challenges(ManualDnsPersistHandler())
    """

    def __init__(self, propagation_delay: int = 0, prompt: bool = True) -> None:
        self.propagation_delay = propagation_delay
        self._prompt = prompt

    def setup(self, domain: str, record_name: str, record_value: str) -> None:
        """Print the record and optionally wait for the operator."""
        message = (
            f"\nPublish the following DNS record to validate {domain}:\n\n"
            f"  {record_name}. IN TXT \"{record_value}\"\n\n"
            "Keep this record published -- it authorizes future issuance.\n"
        )
        logger.info(
            "Manual DNS-PERSIST record required: %s = %s", record_name, record_value
        )
        print(message)

        if self._prompt:
            input("Press Enter once the record is live... ")
