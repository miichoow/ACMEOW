Contributing
============

Contributions to ACMEOW are welcome!

Development Setup
-----------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/miichoow/ACMEOW.git
      cd ACMEOW

2. Create a virtual environment:

   .. code-block:: bash

      python -m venv .venv
      source .venv/bin/activate  # Linux/macOS
      .venv\Scripts\activate     # Windows

3. Install in development mode:

   .. code-block:: bash

      pip install -e ".[dev]"

Running Tests
-------------

.. code-block:: bash

   pytest

Type Checking
-------------

.. code-block:: bash

   mypy src/acmeow/

Linting
-------

.. code-block:: bash

   ruff check src/acmeow/

Building Documentation
----------------------

.. code-block:: bash

   cd docs
   pip install -r requirements.txt
   make html

Code Style
----------

- Follow PEP 8
- Use type annotations
- Write docstrings in Google style
- Keep line length under 100 characters

Pull Requests
-------------

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

Issues
------

Report bugs and request features at:
https://github.com/miichoow/ACMEOW/issues
