Changelog
=========

All notable changes to ACMEOW are documented here.

Version 1.1.0
-------------

*Unreleased*

Features
~~~~~~~~

- **DNS-PERSIST-01 challenge** (``draft-ietf-acme-dns-persist``, experimental):
  proves domain control with a long-lived TXT record at
  ``_validation-persist.<domain>`` that stays published and authorizes future
  issuance. Adds ``ChallengeType.DNS_PERSIST``,
  ``AcmeClient.complete_dns_persist_challenges()``, the ``DnsPersistHandler``
  interface with ``CallbackDnsPersistHandler``, ``ManualDnsPersistHandler`` and
  ``DnsProviderPersistHandler`` implementations, and the record helpers
  ``build_record_value()``, ``parse_record_value()``,
  ``validation_domain_name()`` and ``select_issuer_domain_name()``.
  Verified against Let's Encrypt staging, which offers the challenge as of
  July 2026. The draft is not yet an RFC and the record format may still
  change.

  Wildcard handling follows :rfc:`8555` section 7.1.4: the server strips the
  ``*.`` prefix and sets a ``wildcard`` flag, so an order covering both a
  domain and its wildcard produces two authorizations sharing one validation
  name. These are merged into a single record carrying ``policy=wildcard``.

  Let's Encrypt omits the ``accounturi`` field the draft specifies on the
  challenge object; the client falls back to its own account URI.

- **External CSR support**: ``finalize_order()`` now accepts an optional ``csr``
  parameter (PEM or DER encoded) so that users can supply a CSR generated
  outside of ACMEOW. When an external CSR is provided, no private key is
  generated or stored by the library. ``get_certificate()`` returns ``None``
  for the key PEM in this case.

Changes
~~~~~~~

- ``Challenge.token`` is now ``str | None``, since DNS-PERSIST-01 challenges
  carry no token. Parsing a token-based challenge (dns-01, http-01,
  tls-alpn-01) that lacks a token now raises ``AcmeAuthorizationError`` with a
  clear message instead of failing later during key authorization.
- ``Challenge`` gains the ``accounturi`` and ``issuer_domain_names`` fields,
  populated for DNS-PERSIST-01 challenges.

Version 1.0.0
-------------

*Released: 2024*

Initial release with full RFC 8555 support.

Features
~~~~~~~~

- ACME account creation and management
- Account update, key rollover, and deactivation
- Certificate ordering with DNS-01 and HTTP-01 challenges
- Certificate revocation with reason codes
- Pluggable DNS provider system
- Callback-based challenge handlers
- External Account Binding (EAB) support
- Full type annotations
- Thread-safe nonce management
