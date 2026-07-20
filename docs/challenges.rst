Challenge Types
===============

The ACME protocol supports multiple challenge types to prove domain control.
ACMEOW supports DNS-01, HTTP-01, and TLS-ALPN-01 challenges, plus the draft
DNS-PERSIST-01 method.

DNS-01 Challenge
----------------

DNS-01 challenges prove domain control by creating a DNS TXT record.
This is the only challenge type that supports wildcard certificates.

Using CallbackDnsHandler
~~~~~~~~~~~~~~~~~~~~~~~~

The ``CallbackDnsHandler`` uses callback functions you provide:

.. code-block:: python

   from acmeow import CallbackDnsHandler, ChallengeType

   def create_txt(domain: str, record_name: str, value: str) -> None:
       # record_name: "_acme-challenge.example.com"
       # value: base64url SHA-256 hash to put in TXT record
       your_dns_api.create_record(record_name, "TXT", value)

   def delete_txt(domain: str, record_name: str) -> None:
       your_dns_api.delete_record(record_name, "TXT")

   handler = CallbackDnsHandler(
       create_record=create_txt,
       delete_record=delete_txt,
       propagation_delay=120,  # Seconds to wait for DNS propagation
   )

   client.complete_challenges(handler, ChallengeType.DNS)

HTTP-01 Challenge
-----------------

HTTP-01 challenges prove domain control by serving a file over HTTP.
This requires the domain to point to a web server you control.

.. note::
   HTTP-01 does **not** support wildcard certificates.

Using FileHttpHandler
~~~~~~~~~~~~~~~~~~~~~

Writes challenge files to a webroot directory:

.. code-block:: python

   from pathlib import Path
   from acmeow import FileHttpHandler, ChallengeType

   # Files are written to {webroot}/.well-known/acme-challenge/
   handler = FileHttpHandler(webroot=Path("/var/www/html"))

   client.complete_challenges(handler, ChallengeType.HTTP)

Your web server must serve the ``.well-known/acme-challenge/`` directory.

Using CallbackHttpHandler
~~~~~~~~~~~~~~~~~~~~~~~~~

For custom HTTP challenge handling:

.. code-block:: python

   from acmeow import CallbackHttpHandler, ChallengeType

   def setup(domain: str, token: str, key_authorization: str) -> None:
       # Serve key_authorization at:
       # http://{domain}/.well-known/acme-challenge/{token}
       pass

   def cleanup(domain: str, token: str) -> None:
       # Remove the challenge response
       pass

   handler = CallbackHttpHandler(setup, cleanup)

   client.complete_challenges(handler, ChallengeType.HTTP)

TLS-ALPN-01 Challenge
---------------------

TLS-ALPN-01 challenges prove domain control by serving a specially crafted
TLS certificate with the ACME identifier extension (RFC 8737). This is useful
when you have direct control over the TLS termination but cannot easily modify
DNS records or HTTP responses.

.. note::
   TLS-ALPN-01 does **not** support wildcard certificates.
   The server must support the ``acme-tls/1`` ALPN protocol.

Using CallbackTlsAlpnHandler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``CallbackTlsAlpnHandler`` uses callback functions to deploy and remove
the validation certificate:

.. code-block:: python

   from acmeow import CallbackTlsAlpnHandler, ChallengeType

   def deploy_cert(domain: str, cert_pem: bytes, key_pem: bytes) -> None:
       # Configure your TLS server with the validation certificate
       # The certificate contains the acmeIdentifier extension
       your_tls_server.set_certificate(domain, cert_pem, key_pem)

   def cleanup_cert(domain: str) -> None:
       # Remove the validation certificate
       your_tls_server.remove_certificate(domain)

   handler = CallbackTlsAlpnHandler(deploy_cert, cleanup_cert)

   client.complete_challenges(handler, ChallengeType.TLS_ALPN)

Using FileTlsAlpnHandler
~~~~~~~~~~~~~~~~~~~~~~~~

Writes validation certificates to files, with optional server reload:

.. code-block:: python

   from pathlib import Path
   import subprocess
   from acmeow import FileTlsAlpnHandler, ChallengeType

   def reload_nginx():
       subprocess.run(["nginx", "-s", "reload"])

   handler = FileTlsAlpnHandler(
       cert_dir=Path("/etc/tls/acme"),
       cert_pattern="{domain}.alpn.crt",
       key_pattern="{domain}.alpn.key",
       reload_callback=reload_nginx,  # Optional
   )

   client.complete_challenges(handler, ChallengeType.TLS_ALPN)

Helper Functions
~~~~~~~~~~~~~~~~

ACMEOW provides helper functions for working with TLS-ALPN-01 certificates:

.. code-block:: python

   from acmeow.handlers.tls_alpn import (
       generate_tls_alpn_certificate,
       validate_tls_alpn_certificate,
   )

   # Generate a validation certificate manually
   cert_pem, key_pem = generate_tls_alpn_certificate(
       domain="example.com",
       key_authorization="token.thumbprint",
   )

   # Validate a certificate has the correct acmeIdentifier
   is_valid = validate_tls_alpn_certificate(
       cert_pem=cert_pem,
       expected_domain="example.com",
       expected_key_auth="token.thumbprint",
   )

DNS-PERSIST-01 Challenge
------------------------

DNS-PERSIST-01 proves domain control with a **long-lived** DNS TXT record
instead of a per-challenge one. The record stays published and authorizes
repeated issuance, so renewals need no DNS changes at all.

.. warning::
   This challenge type is defined by `draft-ietf-acme-dns-persist
   <https://datatracker.ietf.org/doc/draft-ietf-acme-dns-persist/>`_, which is
   an Internet-Draft and **not yet an RFC**, so the record format may change
   before publication.

   Let's Encrypt **staging** offers ``dns-persist-01`` (verified July 2026);
   support in Let's Encrypt production and other CAs is unverified. Check that
   your CA advertises the challenge before relying on it -- use
   ``auth.get_dns_persist_challenge()``, which returns ``None`` when it is not
   offered.

How it differs from DNS-01
~~~~~~~~~~~~~~~~~~~~~~~~~~

The two DNS methods are not variations on each other:

* The record lives at ``_validation-persist.example.com``, not
  ``_acme-challenge.example.com``.
* There is **no token and no key authorization**. The record value instead
  binds an *Issuer Domain Name* to your ACME account URI, using the
  ``issue-value`` syntax of :rfc:`8659` (the CAA record format).
* The record is **not deleted** after validation. Removing it would defeat
  the purpose of the method.

A published record looks like this::

   _validation-persist.example.com. IN TXT "ca.example; accounturi=https://ca.example/acct/1"

Because the response has a different shape, DNS-PERSIST-01 uses its own
handler interface (``DnsPersistHandler``) and its own client method
(``complete_dns_persist_challenges``) rather than ``complete_challenges``.

Using CallbackDnsPersistHandler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from acmeow import CallbackDnsPersistHandler

   def upsert_txt(domain: str, record_name: str, value: str) -> None:
       # record_name: "_validation-persist.example.com"
       # value: "ca.example; accounturi=https://ca.example/acct/1"
       # Treat this as an upsert -- a record may already exist from last time.
       your_dns_api.upsert_record(record_name, "TXT", value)

   handler = CallbackDnsPersistHandler(upsert_txt, propagation_delay=120)

   client.complete_dns_persist_challenges(handler)

Using a DNS provider
~~~~~~~~~~~~~~~~~~~~

The built-in DNS providers work with this challenge type too:

.. code-block:: python

   from acmeow import get_dns_provider, DnsProviderPersistHandler

   provider = get_dns_provider("cloudflare", api_token="...")
   handler = DnsProviderPersistHandler(provider, ttl=3600)

   client.complete_dns_persist_challenges(handler)

By default the handler replaces any stale record already sitting at the
validation name, so repeated issuance does not accumulate duplicates.

Managing the record
~~~~~~~~~~~~~~~~~~~

Choose which Issuer Domain Name to publish when the CA offers several, and
optionally set an expiry:

.. code-block:: python

   from datetime import datetime, timedelta, timezone

   client.complete_dns_persist_challenges(
       handler,
       preferred_issuer="ca.example",
       persist_until=datetime.now(timezone.utc) + timedelta(days=365),
   )

If ``preferred_issuer`` is not among the names the CA offers, an error is
raised rather than silently falling back -- publishing an unoffered name
would fail validation.

To opt out of persistence and delete the record after validation, pass
``persist=False`` to the handler.

Wildcards
~~~~~~~~~

A wildcard is authorized by the ``policy=wildcard`` parameter on the **base
domain's** record, not by a separate record name, and that policy covers the
base name as well as wildcards.

Per :rfc:`8555` section 7.1.4 the server strips the ``*.`` prefix from a
wildcard authorization's identifier and sets a ``wildcard`` flag instead, so
ACMEOW keys off that flag. An order covering both ``example.com`` and
``*.example.com`` therefore produces two authorizations that share one
validation name; ACMEOW merges them into a **single** record carrying
``policy=wildcard``, and answers both challenges::

   _validation-persist.example.com. IN TXT "ca.example; accounturi=https://ca.example/acct/1; policy=wildcard"

.. note::
   The draft has the CA supply ``accounturi`` in the challenge object, but
   Let's Encrypt omits it. ACMEOW falls back to the client's own account URI
   in that case.

Building and parsing records
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The record helpers are exported for auditing existing DNS records:

.. code-block:: python

   from acmeow import build_record_value, parse_record_value

   value = build_record_value(
       issuer_domain_name="ca.example",
       accounturi="https://ca.example/acct/1",
       policy="wildcard",
   )

   parsed = parse_record_value(value)
   parsed.accounturi        # "https://ca.example/acct/1"
   parsed.allows_wildcard   # True
   parsed.is_expired        # False (no persistUntil set)

Challenge Comparison
--------------------

+----------+-------------------+-------------------+-------------------+-------------------+
| Feature  | DNS-01            | HTTP-01           | TLS-ALPN-01       | DNS-PERSIST-01    |
+==========+===================+===================+===================+===================+
| Wildcards| Yes               | No                | No                | Yes (policy)      |
+----------+-------------------+-------------------+-------------------+-------------------+
| Port     | 53 (DNS)          | 80 (HTTP)         | 443 (HTTPS)       | 53 (DNS)          |
+----------+-------------------+-------------------+-------------------+-------------------+
| Setup    | DNS API access    | Web server access | TLS server access | DNS API access    |
+----------+-------------------+-------------------+-------------------+-------------------+
| Record   | Per-challenge     | Per-challenge     | Per-challenge     | Long-lived        |
+----------+-------------------+-------------------+-------------------+-------------------+
| Status   | RFC 8555          | RFC 8555          | RFC 8737          | Internet-Draft    |
+----------+-------------------+-------------------+-------------------+-------------------+
| Use case | Wildcard certs,   | Simple web apps   | TLS termination   | Frequent renewals,|
|          | internal servers  |                   | proxies, CDNs     | hands-off DNS     |
+----------+-------------------+-------------------+-------------------+-------------------+
