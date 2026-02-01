Account Management
==================

ACMEOW provides full account lifecycle management per RFC 8555.

Creating an Account
-------------------

.. code-block:: python

   from acmeow import AcmeClient
   from pathlib import Path

   client = AcmeClient(
       server_url="https://acme-v02.api.letsencrypt.org/directory",
       email="admin@example.com",
       storage_path=Path("./acme_data"),
   )

   # Create new account or load existing
   account = client.create_account()

   print(f"Account URI: {account.uri}")
   print(f"Status: {account.status}")

External Account Binding (EAB)
------------------------------

Some CAs require External Account Binding:

.. code-block:: python

   client.set_external_account_binding(
       kid="your-key-id",
       hmac_key="your-base64url-hmac-key",
   )
   client.create_account()

Updating Contact Information
----------------------------

Update the account's email address:

.. code-block:: python

   client.update_account(email="new-email@example.com")

Account Key Rollover
--------------------

Roll over to a new account key (e.g., for key rotation):

.. code-block:: python

   # Generate new key and update server
   client.key_rollover()

.. warning::
   After key rollover, the old key can no longer be used.

Deactivating an Account
-----------------------

Permanently deactivate an account:

.. code-block:: python

   client.deactivate_account()

.. danger::
   Account deactivation is **permanent** and cannot be undone.
   The account will no longer be usable for any operations.

Certificate Revocation
----------------------

Revoke a previously issued certificate:

.. code-block:: python

   from acmeow import RevocationReason

   # Load certificate
   cert_pem = Path("certificate.pem").read_text()

   # Revoke without reason
   client.revoke_certificate(cert_pem)

   # Or with a specific reason
   client.revoke_certificate(cert_pem, reason=RevocationReason.KEY_COMPROMISE)

Revocation Reasons
~~~~~~~~~~~~~~~~~~

+-----------------------------------+------------------------------------------+
| Reason                            | Description                              |
+===================================+==========================================+
| ``UNSPECIFIED``                   | No specific reason                       |
+-----------------------------------+------------------------------------------+
| ``KEY_COMPROMISE``                | Private key compromised                  |
+-----------------------------------+------------------------------------------+
| ``CA_COMPROMISE``                 | CA compromised                           |
+-----------------------------------+------------------------------------------+
| ``AFFILIATION_CHANGED``           | Subject's affiliation changed            |
+-----------------------------------+------------------------------------------+
| ``SUPERSEDED``                    | Certificate superseded                   |
+-----------------------------------+------------------------------------------+
| ``CESSATION_OF_OPERATION``        | Operations ceased                        |
+-----------------------------------+------------------------------------------+
| ``CERTIFICATE_HOLD``              | Temporarily on hold                      |
+-----------------------------------+------------------------------------------+
| ``PRIVILEGE_WITHDRAWN``           | Privileges withdrawn                     |
+-----------------------------------+------------------------------------------+
