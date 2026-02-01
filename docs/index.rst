ACMEOW Documentation
====================

**ACMEOW** is a production-grade Python library for automated SSL/TLS
certificate management using the ACME protocol (RFC 8555).

.. note::
   This library requires Python 3.10 or later.

Features
--------

- **Full ACME Protocol Support**: Complete RFC 8555 implementation including
  account management, certificate issuance, and revocation
- **Multiple Challenge Types**: DNS-01, HTTP-01, and TLS-ALPN-01 challenge validation
- **Flexible Handlers**: Built-in handlers and callback-based custom handlers
- **Thread-Safe**: Safe for use in multi-threaded applications
- **Type Hints**: Full type annotations for better IDE support

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install acmeow

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path
   from acmeow import AcmeClient, Identifier, KeyType, CallbackDnsHandler

   # Create client
   client = AcmeClient(
       server_url="https://acme-v02.api.letsencrypt.org/directory",
       email="admin@example.com",
       storage_path=Path("./acme_data"),
   )

   # Create account
   client.create_account()

   # Create order
   order = client.create_order([Identifier.dns("example.com")])

   # Define DNS record handlers
   def create_record(domain, name, value):
       # Create TXT record using your DNS provider API
       pass

   def delete_record(domain, name):
       # Delete TXT record
       pass

   # Complete challenges
   handler = CallbackDnsHandler(create_record, delete_record)
   client.complete_challenges(handler)

   # Finalize and get certificate
   client.finalize_order(KeyType.EC256)
   cert_pem, key_pem = client.get_certificate()

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   challenges
   account-management

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/client
   api/models
   api/handlers
   api/exceptions
   api/enums

.. toctree::
   :maxdepth: 1
   :caption: Project

   changelog
   contributing

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
