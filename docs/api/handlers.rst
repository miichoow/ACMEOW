Challenge Handlers
==================

.. module:: acmeow.handlers

Handlers for completing ACME challenges. The library supports three challenge
types: DNS-01, HTTP-01, and TLS-ALPN-01.

ChallengeHandler (Base)
-----------------------

.. autoclass:: acmeow.ChallengeHandler
   :members:
   :undoc-members:
   :show-inheritance:

DNS-01 Handlers
---------------

CallbackDnsHandler
~~~~~~~~~~~~~~~~~~

.. autoclass:: acmeow.CallbackDnsHandler
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

HTTP-01 Handlers
----------------

CallbackHttpHandler
~~~~~~~~~~~~~~~~~~~

.. autoclass:: acmeow.CallbackHttpHandler
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

FileHttpHandler
~~~~~~~~~~~~~~~

.. autoclass:: acmeow.FileHttpHandler
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

TLS-ALPN-01 Handlers
--------------------

CallbackTlsAlpnHandler
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: acmeow.CallbackTlsAlpnHandler
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

FileTlsAlpnHandler
~~~~~~~~~~~~~~~~~~

.. autoclass:: acmeow.FileTlsAlpnHandler
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Helper Functions
----------------

generate_tls_alpn_certificate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: acmeow.generate_tls_alpn_certificate
