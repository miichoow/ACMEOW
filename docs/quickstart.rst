Quick Start Guide
=================

This guide walks through obtaining a certificate using ACMEOW.

Prerequisites
-------------

Before starting, you need:

1. A domain name you control
2. Access to create DNS TXT records for your domain
3. Python 3.10 or later

Basic Workflow
--------------

The certificate issuance process follows these steps:

1. Create an ACME client and account
2. Create a certificate order
3. Complete domain validation challenges
4. Finalize the order and download the certificate

Step 1: Create Client and Account
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path
   from acmeow import AcmeClient

   # Create client (use staging for testing!)
   client = AcmeClient(
       server_url="https://acme-staging-v02.api.letsencrypt.org/directory",
       email="admin@example.com",
       storage_path=Path("./acme_data"),
   )

   # Create or retrieve account
   client.create_account()

.. warning::
   Use the **staging** server for testing to avoid rate limits.
   Switch to the production server only when ready.

Step 2: Create Certificate Order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from acmeow import Identifier

   # Order certificate for one or more domains
   order = client.create_order([
       Identifier.dns("example.com"),
       Identifier.dns("www.example.com"),
   ])

Step 3: Complete Challenges
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For DNS-01 challenges (recommended for wildcards):

.. code-block:: python

   from acmeow import CallbackDnsHandler, ChallengeType

   def create_txt_record(domain: str, record_name: str, value: str) -> None:
       # record_name is "_acme-challenge.example.com"
       # value is the TXT record content
       print(f"Create TXT record: {record_name} = {value}")
       # Use your DNS provider's API here

   def delete_txt_record(domain: str, record_name: str) -> None:
       print(f"Delete TXT record: {record_name}")
       # Use your DNS provider's API here

   handler = CallbackDnsHandler(
       create_record=create_txt_record,
       delete_record=delete_txt_record,
       propagation_delay=120,  # Wait for DNS propagation
   )

   client.complete_challenges(handler, ChallengeType.DNS)

Step 4: Finalize and Get Certificate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from acmeow import KeyType

   # Generate key and submit CSR
   client.finalize_order(KeyType.EC256)

   # Download certificate
   cert_pem, key_pem = client.get_certificate()

   # Save to files
   Path("certificate.pem").write_text(cert_pem)
   Path("private_key.pem").write_text(key_pem)

Complete Example
----------------

.. code-block:: python

   from pathlib import Path
   from acmeow import (
       AcmeClient, Identifier, KeyType,
       CallbackDnsHandler, ChallengeType,
   )

   # Initialize
   client = AcmeClient(
       server_url="https://acme-staging-v02.api.letsencrypt.org/directory",
       email="admin@example.com",
       storage_path=Path("./acme_data"),
   )

   # Account
   client.create_account()

   # Order
   order = client.create_order([Identifier.dns("example.com")])

   # Challenges
   def create_record(domain, name, value):
       print(f"CREATE: {name} TXT {value}")
       input("Press Enter when record is created...")

   def delete_record(domain, name):
       print(f"DELETE: {name}")

   handler = CallbackDnsHandler(create_record, delete_record)
   client.complete_challenges(handler, ChallengeType.DNS)

   # Finalize
   client.finalize_order(KeyType.EC256)
   cert_pem, key_pem = client.get_certificate()

   print("Certificate obtained successfully!")

ACME Servers
------------

+---------------+----------------------------------------------------------+
| CA            | Server URL                                               |
+===============+==========================================================+
| Let's Encrypt | ``https://acme-v02.api.letsencrypt.org/directory``       |
| (Production)  |                                                          |
+---------------+----------------------------------------------------------+
| Let's Encrypt | ``https://acme-staging-v02.api.letsencrypt.org/directory``|
| (Staging)     |                                                          |
+---------------+----------------------------------------------------------+
| ZeroSSL       | ``https://acme.zerossl.com/v2/DV90``                     |
+---------------+----------------------------------------------------------+
| Buypass       | ``https://api.buypass.com/acme/directory``               |
+---------------+----------------------------------------------------------+
