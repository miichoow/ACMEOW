Challenge Types
===============

The ACME protocol supports multiple challenge types to prove domain control.
ACMEOW supports DNS-01, HTTP-01, and TLS-ALPN-01 challenges.

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

Challenge Comparison
--------------------

+----------+-------------------+-------------------+-------------------+
| Feature  | DNS-01            | HTTP-01           | TLS-ALPN-01       |
+==========+===================+===================+===================+
| Wildcards| Yes               | No                | No                |
+----------+-------------------+-------------------+-------------------+
| Port     | 53 (DNS)          | 80 (HTTP)         | 443 (HTTPS)       |
+----------+-------------------+-------------------+-------------------+
| Setup    | DNS API access    | Web server access | TLS server access |
+----------+-------------------+-------------------+-------------------+
| Use case | Wildcard certs,   | Simple web apps   | TLS termination   |
|          | internal servers  |                   | proxies, CDNs     |
+----------+-------------------+-------------------+-------------------+
