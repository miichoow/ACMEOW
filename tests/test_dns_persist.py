"""Tests for DNS-PERSIST-01 challenge support (draft-ietf-acme-dns-persist)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from acmeow.dns.base import DnsRecord
from acmeow.dns.persist_handler import DnsProviderPersistHandler
from acmeow.enums import ChallengeType
from acmeow.handlers.dns_persist import (
    PERSIST_LABEL,
    WILDCARD_POLICY,
    CallbackDnsPersistHandler,
    DnsPersistError,
    ManualDnsPersistHandler,
    PersistRecordValue,
    build_record_value,
    parse_record_value,
    select_issuer_domain_name,
    validation_domain_name,
)
from acmeow.models.authorization import Authorization
from acmeow.models.challenge import Challenge


class TestValidationDomainName:
    """Tests for validation_domain_name()."""

    def test_prepends_persist_label(self):
        """The label defined by the draft is prepended to the domain."""
        assert validation_domain_name("example.com") == "_validation-persist.example.com"

    def test_wildcard_uses_base_domain(self):
        """A wildcard is validated via the base domain's record."""
        assert validation_domain_name("*.example.com") == "_validation-persist.example.com"

    def test_subdomain(self):
        """Subdomains get their own validation name."""
        assert (
            validation_domain_name("a.b.example.com")
            == "_validation-persist.a.b.example.com"
        )

    def test_label_constant(self):
        """The exported label matches the draft."""
        assert PERSIST_LABEL == "_validation-persist"


class TestBuildRecordValue:
    """Tests for build_record_value()."""

    def test_minimal_record(self):
        """Issuer domain name plus the mandatory accounturi."""
        value = build_record_value("ca.example", "https://ca.example/acct/1")
        assert value == "ca.example; accounturi=https://ca.example/acct/1"

    def test_all_parameters(self):
        """Optional policy and persistUntil are appended in order."""
        value = build_record_value(
            "ca.example",
            "https://ca.example/acct/1",
            policy=WILDCARD_POLICY,
            persist_until=1721952000,
        )
        assert value == (
            "ca.example; accounturi=https://ca.example/acct/1; "
            "policy=wildcard; persistUntil=1721952000"
        )

    def test_extra_parameters(self):
        """Additional parameters are included."""
        value = build_record_value(
            "ca.example",
            "https://ca.example/acct/1",
            parameters={"validationmethods": "dns-persist-01"},
        )
        assert value.endswith("; validationmethods=dns-persist-01")

    def test_persist_until_accepts_datetime(self):
        """A datetime is converted to a UNIX timestamp."""
        moment = datetime(2024, 7, 26, 0, 0, 0, tzinfo=timezone.utc)
        value = build_record_value("ca.example", "https://ca/1", persist_until=moment)
        assert f"persistUntil={int(moment.timestamp())}" in value

    def test_naive_datetime_treated_as_utc(self):
        """A naive datetime is interpreted as UTC rather than local time."""
        naive = datetime(2024, 7, 26, 0, 0, 0)
        aware = datetime(2024, 7, 26, 0, 0, 0, tzinfo=timezone.utc)
        assert build_record_value("ca.example", "https://ca/1", persist_until=naive) == (
            build_record_value("ca.example", "https://ca/1", persist_until=aware)
        )

    def test_rejects_semicolon_in_accounturi(self):
        """A ';' would inject extra parameters into the published record."""
        with pytest.raises(DnsPersistError, match="accounturi"):
            build_record_value("ca.example", "https://ca/1; policy=wildcard")

    def test_rejects_whitespace_in_accounturi(self):
        """Whitespace is outside the issue-value character set."""
        with pytest.raises(DnsPersistError):
            build_record_value("ca.example", "https://ca/1 extra")

    def test_rejects_empty_issuer(self):
        """An empty issuer means 'no issuance permitted' in CAA syntax."""
        with pytest.raises(DnsPersistError, match="issuer domain name"):
            build_record_value("", "https://ca/1")

    def test_rejects_malformed_issuer(self):
        """Issuer domain names must be valid DNS names."""
        with pytest.raises(DnsPersistError, match="issuer domain name"):
            build_record_value("ca example", "https://ca/1")

    def test_rejects_bad_parameter_tag(self):
        """Parameter tags are alphanumeric per RFC 8659."""
        with pytest.raises(DnsPersistError, match="tag"):
            build_record_value("ca.example", "https://ca/1", parameters={"bad tag": "x"})


class TestParseRecordValue:
    """Tests for parse_record_value()."""

    def test_parses_minimal_record(self):
        """Issuer and accounturi are extracted."""
        parsed = parse_record_value("ca.example; accounturi=https://ca.example/acct/1")
        assert parsed.issuer_domain_name == "ca.example"
        assert parsed.accounturi == "https://ca.example/acct/1"
        assert parsed.policy is None
        assert parsed.persist_until is None

    def test_parses_all_parameters(self):
        """Policy and persistUntil are extracted and typed."""
        parsed = parse_record_value(
            "ca.example; accounturi=https://ca/1; policy=wildcard; persistUntil=1721952000"
        )
        assert parsed.policy == "wildcard"
        assert parsed.persist_until == 1721952000
        assert parsed.allows_wildcard is True

    def test_tolerates_extra_whitespace(self):
        """Whitespace around separators is permitted by RFC 8659."""
        parsed = parse_record_value("  ca.example ;  accounturi = https://ca/1  ")
        assert parsed.issuer_domain_name == "ca.example"
        assert parsed.accounturi == "https://ca/1"

    def test_tolerates_trailing_semicolon(self):
        """An empty trailing segment is ignored."""
        parsed = parse_record_value("ca.example; accounturi=https://ca/1;")
        assert parsed.accounturi == "https://ca/1"

    def test_preserves_unknown_parameters(self):
        """Unrecognized parameters survive the round trip."""
        parsed = parse_record_value("ca.example; accounturi=https://ca/1; future=yes")
        assert parsed.parameters == (("future", "yes"),)

    def test_requires_accounturi(self):
        """accounturi is mandatory for this challenge type."""
        with pytest.raises(DnsPersistError, match="accounturi"):
            parse_record_value("ca.example; policy=wildcard")

    def test_rejects_parameter_without_value(self):
        """Every parameter must be a tag=value pair."""
        with pytest.raises(DnsPersistError, match="tag=value"):
            parse_record_value("ca.example; accounturi=https://ca/1; broken")

    def test_rejects_non_integer_persist_until(self):
        """persistUntil is a UNIX timestamp."""
        with pytest.raises(DnsPersistError, match="persistUntil"):
            parse_record_value("ca.example; accounturi=https://ca/1; persistUntil=soon")

    def test_round_trip(self):
        """build -> parse -> str reproduces the original value."""
        value = build_record_value(
            "ca.example",
            "https://ca.example/acct/1",
            policy=WILDCARD_POLICY,
            persist_until=1721952000,
        )
        assert str(parse_record_value(value)) == value


class TestPersistRecordValue:
    """Tests for the PersistRecordValue model."""

    def test_no_persist_until_never_expires(self):
        """Records without persistUntil are open-ended."""
        record = PersistRecordValue("ca.example", "https://ca/1")
        assert record.is_expired is False

    def test_past_persist_until_is_expired(self):
        """A timestamp in the past marks the record expired."""
        past = datetime.now(timezone.utc) - timedelta(days=1)
        record = PersistRecordValue("ca.example", "https://ca/1", persist_until=int(past.timestamp()))
        assert record.is_expired is True

    def test_future_persist_until_is_not_expired(self):
        """A timestamp in the future leaves the record valid."""
        future = datetime.now(timezone.utc) + timedelta(days=1)
        record = PersistRecordValue("ca.example", "https://ca/1", persist_until=int(future.timestamp()))
        assert record.is_expired is False

    def test_allows_wildcard_only_for_wildcard_policy(self):
        """Only policy=wildcard authorizes wildcard issuance."""
        assert PersistRecordValue("ca.example", "https://ca/1").allows_wildcard is False
        assert (
            PersistRecordValue("ca.example", "https://ca/1", policy="wildcard").allows_wildcard
            is True
        )


class TestSelectIssuerDomainName:
    """Tests for select_issuer_domain_name()."""

    def test_defaults_to_first_offered(self):
        """Without a preference the CA's first name is used."""
        assert select_issuer_domain_name(["a.example", "b.example"]) == "a.example"

    def test_honours_preference(self):
        """A preferred name that the CA offers is selected."""
        assert (
            select_issuer_domain_name(["a.example", "b.example"], "b.example")
            == "b.example"
        )

    def test_rejects_unoffered_preference(self):
        """Publishing an unoffered name would fail validation, so it errors."""
        with pytest.raises(DnsPersistError, match="not offered"):
            select_issuer_domain_name(["a.example"], "other.example")

    def test_rejects_empty_list(self):
        """A challenge with no issuer names is unusable."""
        with pytest.raises(DnsPersistError, match="no issuer-domain-names"):
            select_issuer_domain_name([])


class TestChallengeModel:
    """Tests for parsing dns-persist-01 challenge objects."""

    def _challenge_data(self, **overrides):
        data = {
            "type": "dns-persist-01",
            "url": "https://ca.example/chall/1",
            "status": "pending",
            "accounturi": "https://ca.example/acct/1",
            "issuer-domain-names": ["ca.example"],
        }
        data.update(overrides)
        return data

    def test_parses_type(self):
        """The challenge type maps to the enum."""
        challenge = Challenge.from_dict(self._challenge_data())
        assert challenge.type == ChallengeType.DNS_PERSIST

    def test_tokenless_challenge_does_not_raise(self):
        """The draft defines no token; parsing must not require one."""
        challenge = Challenge.from_dict(self._challenge_data())
        assert challenge.token is None

    def test_parses_accounturi_and_issuers(self):
        """CA-supplied fields are carried on the model."""
        challenge = Challenge.from_dict(
            self._challenge_data(**{"issuer-domain-names": ["a.example", "b.example"]})
        )
        assert challenge.accounturi == "https://ca.example/acct/1"
        assert challenge.issuer_domain_names == ("a.example", "b.example")

    def test_missing_issuers_defaults_to_empty(self):
        """A challenge without issuer names parses to an empty tuple."""
        data = self._challenge_data()
        del data["issuer-domain-names"]
        assert Challenge.from_dict(data).issuer_domain_names == ()

    def test_dns01_challenge_still_has_token(self):
        """Existing challenge types are unaffected by the optional token."""
        challenge = Challenge.from_dict(
            {
                "type": "dns-01",
                "url": "https://ca.example/chall/2",
                "status": "pending",
                "token": "tok",
            }
        )
        assert challenge.type == ChallengeType.DNS
        assert challenge.token == "tok"

    def test_authorization_accessor(self):
        """Authorization exposes the challenge by type."""
        auth = Authorization.from_dict(
            {
                "identifier": {"type": "dns", "value": "example.com"},
                "status": "pending",
                "challenges": [self._challenge_data()],
            },
            "https://ca.example/authz/1",
        )
        challenge = auth.get_dns_persist_challenge()
        assert challenge is not None
        assert challenge.type == ChallengeType.DNS_PERSIST
        assert auth.get_dns_challenge() is None


class TestCallbackDnsPersistHandler:
    """Tests for CallbackDnsPersistHandler."""

    def test_setup_invokes_callback(self):
        """setup passes the domain, record name and value through."""
        create = MagicMock()
        handler = CallbackDnsPersistHandler(create)

        handler.setup("example.com", "_validation-persist.example.com", "ca.example; accounturi=x")

        create.assert_called_once_with(
            "example.com", "_validation-persist.example.com", "ca.example; accounturi=x"
        )

    def test_cleanup_keeps_record_by_default(self):
        """The record is persistent, so cleanup must not delete it."""
        delete = MagicMock()
        handler = CallbackDnsPersistHandler(MagicMock(), delete)

        handler.cleanup("example.com", "_validation-persist.example.com")

        delete.assert_not_called()

    def test_cleanup_deletes_when_persist_disabled(self):
        """Opting out of persistence removes the record."""
        delete = MagicMock()
        handler = CallbackDnsPersistHandler(MagicMock(), delete, persist=False)

        handler.cleanup("example.com", "_validation-persist.example.com")

        delete.assert_called_once_with("example.com", "_validation-persist.example.com")

    def test_cleanup_without_delete_callback_is_safe(self):
        """persist=False with no callback warns rather than crashing."""
        handler = CallbackDnsPersistHandler(MagicMock(), persist=False)
        handler.cleanup("example.com", "_validation-persist.example.com")

    def test_cleanup_swallows_callback_errors(self):
        """Cleanup runs in a finally block, so it must not raise."""
        delete = MagicMock(side_effect=RuntimeError("boom"))
        handler = CallbackDnsPersistHandler(MagicMock(), delete, persist=False)

        handler.cleanup("example.com", "_validation-persist.example.com")

    def test_propagation_delay_default(self):
        """Default propagation delay matches the DNS-01 handlers."""
        assert CallbackDnsPersistHandler(MagicMock()).propagation_delay == 60


class TestManualDnsPersistHandler:
    """Tests for ManualDnsPersistHandler."""

    def test_setup_prints_record(self, capsys):
        """The operator is shown the record to publish."""
        handler = ManualDnsPersistHandler(prompt=False)

        handler.setup("example.com", "_validation-persist.example.com", "ca.example; accounturi=x")

        out = capsys.readouterr().out
        assert "_validation-persist.example.com" in out
        assert "ca.example; accounturi=x" in out

    def test_cleanup_is_noop(self):
        """Manual records are left in place."""
        ManualDnsPersistHandler(prompt=False).cleanup("example.com", "_x.example.com")


class TestDnsProviderPersistHandler:
    """Tests for DnsProviderPersistHandler."""

    def _provider(self, existing=None):
        provider = MagicMock()
        provider.propagation_delay = 30
        provider.get_zone_for_domain.side_effect = lambda d: d
        provider.create_record.return_value = DnsRecord(
            "_validation-persist.example.com", "TXT", "value", 3600, id="rec-1"
        )
        provider.list_records.return_value = existing if existing is not None else []
        return provider

    def test_setup_creates_record(self):
        """The record is created through the provider with a long TTL."""
        provider = self._provider()
        handler = DnsProviderPersistHandler(provider)

        handler.setup("example.com", "_validation-persist.example.com", "ca.example; accounturi=x")

        provider.create_record.assert_called_once_with(
            domain="example.com",
            name="_validation-persist.example.com",
            value="ca.example; accounturi=x",
            ttl=3600,
        )

    def test_setup_replaces_stale_record(self):
        """A leftover record at the same name is removed first."""
        stale = DnsRecord("_validation-persist.example.com", "TXT", "old", 3600, id="old-1")
        provider = self._provider(existing=[stale])
        handler = DnsProviderPersistHandler(provider)

        handler.setup("example.com", "_validation-persist.example.com", "new")

        provider.delete_record.assert_called_once_with("example.com", stale)

    def test_setup_ignores_unrelated_records(self):
        """Records at other names are left alone."""
        other = DnsRecord("_acme-challenge.example.com", "TXT", "x", 300, id="o-1")
        provider = self._provider(existing=[other])
        handler = DnsProviderPersistHandler(provider)

        handler.setup("example.com", "_validation-persist.example.com", "new")

        provider.delete_record.assert_not_called()

    def test_setup_survives_providers_without_list_records(self):
        """Providers that cannot list records still publish successfully."""
        provider = self._provider()
        provider.list_records.side_effect = NotImplementedError
        handler = DnsProviderPersistHandler(provider)

        handler.setup("example.com", "_validation-persist.example.com", "new")

        provider.create_record.assert_called_once()

    def test_replace_existing_can_be_disabled(self):
        """Opting out skips the lookup entirely."""
        provider = self._provider()
        handler = DnsProviderPersistHandler(provider, replace_existing=False)

        handler.setup("example.com", "_validation-persist.example.com", "new")

        provider.list_records.assert_not_called()

    def test_cleanup_keeps_record_by_default(self):
        """Persistence is the point of the challenge type."""
        provider = self._provider()
        handler = DnsProviderPersistHandler(provider)
        handler.setup("example.com", "_validation-persist.example.com", "v")
        provider.delete_record.reset_mock()

        handler.cleanup("example.com", "_validation-persist.example.com")

        provider.delete_record.assert_not_called()

    def test_cleanup_deletes_when_persist_disabled(self):
        """Opting out removes the created record."""
        provider = self._provider()
        handler = DnsProviderPersistHandler(provider, persist=False)
        handler.setup("example.com", "_validation-persist.example.com", "v")
        provider.delete_record.reset_mock()

        handler.cleanup("example.com", "_validation-persist.example.com")

        provider.delete_record.assert_called_once()

    def test_propagation_delay_comes_from_provider(self):
        """The provider's delay is used."""
        assert DnsProviderPersistHandler(self._provider()).propagation_delay == 30

    def test_wildcard_resolves_to_base_zone(self):
        """Wildcard domains publish into the base zone."""
        provider = self._provider()
        handler = DnsProviderPersistHandler(provider)

        handler.setup("*.example.com", "_validation-persist.example.com", "v")

        assert provider.create_record.call_args.kwargs["domain"] == "example.com"


class TestTokenlessChallengeGuard:
    """Making Challenge.token optional must not weaken the token-based types."""

    def test_tokenless_dns01_challenge_is_rejected(self):
        """A dns-01 challenge without a token is malformed, not a None key auth."""
        from acmeow.client import AcmeClient
        from acmeow.enums import OrderStatus
        from acmeow.exceptions import AcmeAuthorizationError
        from acmeow.handlers.dns import CallbackDnsHandler
        from acmeow.models.identifier import Identifier
        from acmeow.models.order import Order

        client = AcmeClient.__new__(AcmeClient)
        client._account = MagicMock()
        client._dns_config = None
        client._order = Order(
            status=OrderStatus.PENDING,
            url="https://ca.example/order/1",
            identifiers=(Identifier.dns("example.com"),),
            finalize_url="https://ca.example/finalize/1",
            authorizations=[
                Authorization.from_dict(
                    {
                        "identifier": {"type": "dns", "value": "example.com"},
                        "status": "pending",
                        "challenges": [
                            {
                                "type": "dns-01",
                                "url": "https://ca.example/chall/1",
                                "status": "pending",
                            }
                        ],
                    },
                    "https://ca.example/authz/1",
                )
            ],
        )

        with pytest.raises(AcmeAuthorizationError, match="without a token"):
            client.complete_challenges(
                CallbackDnsHandler(MagicMock(), MagicMock()), ChallengeType.DNS
            )


class TestVerifyPersistPropagation:
    """Tests for AcmeClient._verify_persist_propagation()."""

    def _client(self, dns_config=True):
        from acmeow._internal.dns import DnsConfig
        from acmeow.client import AcmeClient

        client = AcmeClient.__new__(AcmeClient)
        client._dns_config = DnsConfig() if dns_config else None
        return client

    def test_no_dns_config_skips_verification(self, monkeypatch):
        """Without DNS configuration there is nothing to verify against."""
        import acmeow.client as client_module

        verifier_cls = MagicMock()
        monkeypatch.setattr(client_module, "DnsVerifier", verifier_cls)
        client = self._client(dns_config=False)

        client._verify_persist_propagation(
            [("example.com", "_validation-persist.example.com", "v")], 30
        )

        verifier_cls.assert_not_called()

    def test_verifies_each_record(self, monkeypatch):
        """Every published record is checked."""
        import acmeow.client as client_module

        verifier = MagicMock()
        verifier.verify_txt_record.return_value = True
        monkeypatch.setattr(client_module, "DnsVerifier", MagicMock(return_value=verifier))
        client = self._client()

        client._verify_persist_propagation(
            [
                ("example.com", "_validation-persist.example.com", "v1"),
                ("other.example", "_validation-persist.other.example", "v2"),
            ],
            30,
        )

        assert verifier.verify_txt_record.call_count == 2

    def test_checks_full_record_value_not_a_hash(self, monkeypatch):
        """DNS-PERSIST publishes the issue-value verbatim, unlike DNS-01.

        DNS-01 verifies a base64url SHA-256 digest; this method must compare
        against the literal record value instead.
        """
        import acmeow.client as client_module

        verifier = MagicMock()
        verifier.verify_txt_record.return_value = True
        monkeypatch.setattr(client_module, "DnsVerifier", MagicMock(return_value=verifier))
        client = self._client()

        value = "ca.example; accounturi=https://ca.example/acct/1"
        client._verify_persist_propagation(
            [("example.com", "_validation-persist.example.com", value)], 45
        )

        args, kwargs = verifier.verify_txt_record.call_args
        assert args[0] == "_validation-persist.example.com"
        assert args[1] == value
        assert kwargs["max_wait"] == 45

    def test_raises_when_record_does_not_propagate(self, monkeypatch):
        """A record that never appears aborts before notifying the CA."""
        import acmeow.client as client_module
        from acmeow.exceptions import AcmeDnsError

        verifier = MagicMock()
        verifier.verify_txt_record.return_value = False
        monkeypatch.setattr(client_module, "DnsVerifier", MagicMock(return_value=verifier))
        client = self._client()

        with pytest.raises(AcmeDnsError, match="did not propagate"):
            client._verify_persist_propagation(
                [("example.com", "_validation-persist.example.com", "v")], 30
            )

    def test_stops_at_first_failure(self, monkeypatch):
        """Verification is abandoned once a record fails."""
        import acmeow.client as client_module
        from acmeow.exceptions import AcmeDnsError

        verifier = MagicMock()
        verifier.verify_txt_record.side_effect = [False, True]
        monkeypatch.setattr(client_module, "DnsVerifier", MagicMock(return_value=verifier))
        client = self._client()

        with pytest.raises(AcmeDnsError):
            client._verify_persist_propagation(
                [
                    ("example.com", "_validation-persist.example.com", "v1"),
                    ("other.example", "_validation-persist.other.example", "v2"),
                ],
                30,
            )

        assert verifier.verify_txt_record.call_count == 1

    def test_empty_list_is_a_noop(self, monkeypatch):
        """Nothing published means nothing to verify."""
        import acmeow.client as client_module

        verifier = MagicMock()
        monkeypatch.setattr(client_module, "DnsVerifier", MagicMock(return_value=verifier))

        self._client()._verify_persist_propagation([], 30)

        verifier.verify_txt_record.assert_not_called()


class RecordingHandler(CallbackDnsPersistHandler):
    """Handler capturing what was published, for orchestration tests."""

    def __init__(self, **kwargs):
        self.published: list[tuple[str, str, str]] = []
        self.cleaned: list[tuple[str, str]] = []
        super().__init__(self._record, propagation_delay=0, **kwargs)

    def _record(self, domain, record_name, record_value):
        self.published.append((domain, record_name, record_value))

    def cleanup(self, domain, record_name):
        self.cleaned.append((domain, record_name))
        super().cleanup(domain, record_name)


class TestCompleteDnsPersistChallenges:
    """Tests for AcmeClient.complete_dns_persist_challenges()."""

    def _client(self, domains, account_uri="https://ca.example/acct/1", **challenge_overrides):
        """Build a client whose order covers ``domains``.

        A ``*.`` prefix is translated the way RFC 8555 section 7.1.4 requires:
        the authorization identifier carries the *base* domain and the wildcard
        flag is set. Real servers never leave the prefix in the identifier.
        """
        from acmeow.client import AcmeClient
        from acmeow.enums import OrderStatus
        from acmeow.models.identifier import Identifier
        from acmeow.models.order import Order

        # Bypass __init__, which fetches the ACME directory over the network.
        client = AcmeClient.__new__(AcmeClient)

        account = MagicMock()
        account.uri = account_uri
        client._account = account
        client._dns_config = None

        authorizations = []
        for index, domain in enumerate(domains):
            is_wildcard = domain.startswith("*.")
            base = domain.removeprefix("*.")
            data = {
                "type": "dns-persist-01",
                "url": f"https://ca.example/chall/{index}",
                "status": "pending",
                "accounturi": "https://ca.example/acct/1",
                "issuer-domain-names": ["a.example", "b.example"],
            }
            data.update(challenge_overrides)
            authorizations.append(
                Authorization.from_dict(
                    {
                        "identifier": {"type": "dns", "value": base},
                        "status": "pending",
                        "wildcard": is_wildcard,
                        "challenges": [data],
                    },
                    f"https://ca.example/authz/{index}",
                )
            )

        client._order = Order(
            status=OrderStatus.PENDING,
            url="https://ca.example/order/1",
            identifiers=tuple(Identifier.dns(d) for d in domains),
            finalize_url="https://ca.example/finalize/1",
            authorizations=authorizations,
        )

        client._respond_to_challenge = MagicMock()
        client._poll_authorizations = MagicMock()
        client._save_order = MagicMock()
        return client

    def test_publishes_record_and_responds(self):
        """A record is published and the challenge is answered."""
        client = self._client(["example.com"])
        handler = RecordingHandler()

        client.complete_dns_persist_challenges(handler)

        assert handler.published == [
            (
                "example.com",
                "_validation-persist.example.com",
                "a.example; accounturi=https://ca.example/acct/1",
            )
        ]
        client._respond_to_challenge.assert_called_once_with(
            "https://ca.example/chall/0"
        )
        client._poll_authorizations.assert_called_once()

    def test_wildcard_gets_wildcard_policy(self):
        """A wildcard authorization is authorized via policy=wildcard.

        The server signals the wildcard through the authorization's wildcard
        flag, not a "*." prefix on the identifier, so the flag is what counts.
        """
        client = self._client(["*.example.com"])
        handler = RecordingHandler()

        client.complete_dns_persist_challenges(handler)

        domain, record_name, record_value = handler.published[0]
        assert record_name == "_validation-persist.example.com"
        assert "policy=wildcard" in record_value

    def test_wildcard_and_base_share_one_record(self):
        """Both authorizations map to one validation name, so one record.

        policy=wildcard authorizes the base name too, so publishing twice
        would be redundant and the second write could drop the policy.
        """
        client = self._client(["example.com", "*.example.com"])
        handler = RecordingHandler()

        client.complete_dns_persist_challenges(handler)

        assert len(handler.published) == 1
        _, record_name, record_value = handler.published[0]
        assert record_name == "_validation-persist.example.com"
        assert "policy=wildcard" in record_value
        # Both authorizations still get answered.
        assert client._respond_to_challenge.call_count == 2

    def test_wildcard_first_still_merges(self):
        """Merge order does not affect the result."""
        client = self._client(["*.example.com", "example.com"])
        handler = RecordingHandler()

        client.complete_dns_persist_challenges(handler)

        assert len(handler.published) == 1
        assert "policy=wildcard" in handler.published[0][2]

    def test_published_domain_strips_wildcard_prefix(self):
        """Handlers receive the base domain, matching the record name."""
        client = self._client(["*.example.com"])
        handler = RecordingHandler()

        client.complete_dns_persist_challenges(handler)

        assert handler.published[0][0] == "example.com"

    def test_non_wildcard_has_no_policy(self):
        """Plain domains do not get a policy parameter."""
        client = self._client(["example.com"])
        handler = RecordingHandler()

        client.complete_dns_persist_challenges(handler)

        assert "policy=" not in handler.published[0][2]

    def test_preferred_issuer_is_used(self):
        """An offered preferred issuer is published."""
        client = self._client(["example.com"])
        handler = RecordingHandler()

        client.complete_dns_persist_challenges(handler, preferred_issuer="b.example")

        assert handler.published[0][2].startswith("b.example;")

    def test_unoffered_preferred_issuer_raises(self):
        """Publishing an issuer the CA did not offer is refused."""
        client = self._client(["example.com"])

        with pytest.raises(DnsPersistError, match="not offered"):
            client.complete_dns_persist_challenges(
                RecordingHandler(), preferred_issuer="evil.example"
            )

    def test_persist_until_is_included(self):
        """An explicit expiry reaches the record."""
        client = self._client(["example.com"])
        handler = RecordingHandler()

        client.complete_dns_persist_challenges(handler, persist_until=1721952000)

        assert "persistUntil=1721952000" in handler.published[0][2]

    def test_falls_back_to_account_uri(self):
        """If the CA omits accounturi, the client's own account URI is used."""
        client = self._client(["example.com"], accounturi=None)
        handler = RecordingHandler()

        client.complete_dns_persist_challenges(handler)

        assert "accounturi=https://ca.example/acct/1" in handler.published[0][2]

    def test_multiple_domains(self):
        """Every pending authorization gets a record."""
        client = self._client(["example.com", "other.example"])
        handler = RecordingHandler()

        client.complete_dns_persist_challenges(handler)

        assert len(handler.published) == 2
        assert client._respond_to_challenge.call_count == 2

    def test_missing_challenge_raises(self):
        """An authorization without the challenge type is an error."""
        from acmeow.exceptions import AcmeAuthorizationError

        client = self._client(["example.com"])
        client._order.authorizations[0] = Authorization.from_dict(
            {
                "identifier": {"type": "dns", "value": "example.com"},
                "status": "pending",
                "challenges": [
                    {
                        "type": "http-01",
                        "url": "https://ca.example/chall/h",
                        "status": "pending",
                        "token": "tok",
                    }
                ],
            },
            "https://ca.example/authz/1",
        )

        with pytest.raises(AcmeAuthorizationError):
            client.complete_dns_persist_challenges(RecordingHandler())

    def test_cleanup_runs_even_when_polling_fails(self):
        """Cleanup is in a finally block, so failures still reach the handler."""
        client = self._client(["example.com"])
        client._poll_authorizations.side_effect = RuntimeError("validation failed")
        handler = RecordingHandler()

        with pytest.raises(RuntimeError):
            client.complete_dns_persist_challenges(handler)

        assert handler.cleaned == [("example.com", "_validation-persist.example.com")]

    def test_record_survives_by_default(self):
        """The published record is not deleted after a successful run."""
        client = self._client(["example.com"])
        delete = MagicMock()
        handler = RecordingHandler()
        handler._delete_record = delete

        client.complete_dns_persist_challenges(handler)

        delete.assert_not_called()

    def test_verify_dns_runs_before_responding(self, monkeypatch):
        """Propagation is confirmed before the CA is told to look."""
        import acmeow.client as client_module
        from acmeow._internal.dns import DnsConfig

        order: list[str] = []
        verifier = MagicMock()
        verifier.verify_txt_record.side_effect = lambda *a, **k: order.append("verify") or True
        monkeypatch.setattr(client_module, "DnsVerifier", MagicMock(return_value=verifier))

        client = self._client(["example.com"])
        client._dns_config = DnsConfig()
        client._respond_to_challenge.side_effect = lambda url: order.append("respond")

        client.complete_dns_persist_challenges(RecordingHandler(), verify_dns=True)

        assert order == ["verify", "respond"]

    def test_verify_dns_false_skips_verification(self, monkeypatch):
        """Opting out bypasses DNS checks entirely."""
        import acmeow.client as client_module
        from acmeow._internal.dns import DnsConfig

        verifier = MagicMock()
        monkeypatch.setattr(client_module, "DnsVerifier", MagicMock(return_value=verifier))

        client = self._client(["example.com"])
        client._dns_config = DnsConfig()

        client.complete_dns_persist_challenges(RecordingHandler(), verify_dns=False)

        verifier.verify_txt_record.assert_not_called()
        client._respond_to_challenge.assert_called_once()

    def test_failed_propagation_prevents_responding(self, monkeypatch):
        """If the record never appears, the CA is never notified."""
        import acmeow.client as client_module
        from acmeow._internal.dns import DnsConfig
        from acmeow.exceptions import AcmeDnsError

        verifier = MagicMock()
        verifier.verify_txt_record.return_value = False
        monkeypatch.setattr(client_module, "DnsVerifier", MagicMock(return_value=verifier))

        client = self._client(["example.com"])
        client._dns_config = DnsConfig()
        handler = RecordingHandler()

        with pytest.raises(AcmeDnsError):
            client.complete_dns_persist_challenges(handler, verify_dns=True)

        client._respond_to_challenge.assert_not_called()
        # Cleanup still runs, so a persist=False handler would not leak records.
        assert handler.cleaned == [("example.com", "_validation-persist.example.com")]

    def test_valid_authorization_is_skipped(self):
        """Already-valid authorizations need no record."""
        client = self._client(["example.com"])
        client._order.authorizations[0] = Authorization.from_dict(
            {
                "identifier": {"type": "dns", "value": "example.com"},
                "status": "valid",
                "challenges": [],
            },
            "https://ca.example/authz/1",
        )
        handler = RecordingHandler()

        client.complete_dns_persist_challenges(handler)

        assert handler.published == []
        client._respond_to_challenge.assert_not_called()
