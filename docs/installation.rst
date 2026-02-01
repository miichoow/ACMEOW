Installation
============

Requirements
------------

- Python 3.10 or later
- ``cryptography`` >= 41.0.0
- ``requests`` >= 2.31.0

Installing from PyPI
--------------------

The recommended way to install ACMEOW is from PyPI:

.. code-block:: bash

   pip install acmeow

Installing from Source
----------------------

To install from source:

.. code-block:: bash

   git clone https://github.com/miichoow/ACMEOW.git
   cd ACMEOW
   pip install -e .

Development Installation
------------------------

For development, install with extra dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

This installs additional tools for testing and linting:

- ``pytest`` - Testing framework
- ``pytest-cov`` - Coverage reporting
- ``mypy`` - Static type checking
- ``ruff`` - Linting
- ``types-requests`` - Type stubs for requests

Verifying Installation
----------------------

You can verify the installation by importing the library:

.. code-block:: python

   from acmeow import AcmeClient, Identifier, KeyType
   print("ACMEOW installed successfully!")
