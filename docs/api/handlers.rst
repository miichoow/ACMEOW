Challenge Handlers
==================

.. module:: acmeow.handlers

Handlers for completing ACME challenges. The library supports four challenge
types: DNS-01, HTTP-01, TLS-ALPN-01, and the draft DNS-PERSIST-01.

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

DNS-PERSIST-01 Handlers
-----------------------

Handlers for the persistent DNS validation method defined by
draft-ietf-acme-dns-persist. These implement ``DnsPersistHandler`` rather than
``ChallengeHandler``, because the challenge uses no token or key authorization
and its record is meant to outlive the order.

DnsPersistHandler (Base)
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: acmeow.DnsPersistHandler
   :members:
   :undoc-members:
   :show-inheritance:

CallbackDnsPersistHandler
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: acmeow.CallbackDnsPersistHandler
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

ManualDnsPersistHandler
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: acmeow.ManualDnsPersistHandler
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

DnsProviderPersistHandler
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: acmeow.DnsProviderPersistHandler
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

PersistRecordValue
~~~~~~~~~~~~~~~~~~

.. autoclass:: acmeow.PersistRecordValue
   :members:
   :undoc-members:
   :show-inheritance:

Helper Functions
----------------

generate_tls_alpn_certificate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: acmeow.generate_tls_alpn_certificate

validation_domain_name
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: acmeow.validation_domain_name

build_record_value
~~~~~~~~~~~~~~~~~~

.. autofunction:: acmeow.build_record_value

parse_record_value
~~~~~~~~~~~~~~~~~~

.. autofunction:: acmeow.parse_record_value

select_issuer_domain_name
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: acmeow.select_issuer_domain_name
