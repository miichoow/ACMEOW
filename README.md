# ACMEOW

[![Tests](https://github.com/miichoow/ACMEOW/actions/workflows/tests.yml/badge.svg)](https://github.com/miichoow/ACMEOW/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/acmeow.svg)](https://pypi.org/project/acmeow/)
[![Python versions](https://img.shields.io/pypi/pyversions/acmeow.svg)](https://pypi.org/project/acmeow/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![codecov](https://codecov.io/gh/miichoow/ACMEOW/branch/main/graph/badge.svg)](https://codecov.io/gh/miichoow/ACMEOW)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](https://miichoow.github.io/ACMEOW/)

A production-grade Python library for automated SSL/TLS certificate management using the ACME protocol (RFC 8555).

**[Documentation](https://miichoow.github.io/ACMEOW/)** | **[PyPI](https://pypi.org/project/acmeow/)** | **[GitHub](https://github.com/miichoow/ACMEOW)**

## Features

- **Full ACME Protocol Support**: Complete RFC 8555 implementation including:
  - Account creation, update, key rollover, and deactivation
  - Certificate ordering, issuance, and revocation
  - DNS-01, HTTP-01, and TLS-ALPN-01 challenge validation
- **Multiple Challenge Types**: Supports DNS-01, HTTP-01, and TLS-ALPN-01 challenges
- **Flexible Challenge Handlers**: Built-in handlers and callback-based custom handlers
- **Automatic Retry with Backoff**: Configurable retry logic for transient failures
- **DNS Propagation Verification**: Optional verification that DNS records are visible before challenge completion
- **Order Recovery**: Resume interrupted certificate orders automatically
- **Preferred Chain Selection**: Choose alternate certificate chains (e.g., for compatibility)
- **Thread-Safe**: Safe for use in multi-threaded applications
- **Type Hints**: Full type annotations for better IDE support
- **Modern Python**: Requires Python 3.10+

## Installation

```bash
pip install acmeow
```

Or install from source:

```bash
git clone https://github.com/miichoow/ACMEOW.git
cd ACMEOW
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
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

# Define DNS record handlers (implement with your DNS provider)
def create_record(domain, name, value):
    # Create TXT record using your DNS provider API
    pass

def delete_record(domain, name):
    # Delete TXT record
    pass

# Complete challenges
handler = CallbackDnsHandler(create_record, delete_record, propagation_delay=60)
client.complete_challenges(handler)

# Finalize and get certificate
client.finalize_order(KeyType.EC256)
cert_pem, key_pem = client.get_certificate()
```

## Challenge Handlers

### DNS-01 Challenge (recommended for wildcards)

```python
from acmeow import CallbackDnsHandler

def create_txt(domain: str, record_name: str, value: str) -> None:
    # record_name is "_acme-challenge.example.com"
    # value is the base64url SHA-256 hash to put in the TXT record
    your_dns_api.create_record(record_name, "TXT", value)

def delete_txt(domain: str, record_name: str) -> None:
    your_dns_api.delete_record(record_name, "TXT")

handler = CallbackDnsHandler(
    create_record=create_txt,
    delete_record=delete_txt,
    propagation_delay=120,  # Wait for DNS propagation
)
```

### HTTP-01 Challenge (simpler setup, no wildcards)

```python
from pathlib import Path
from acmeow import FileHttpHandler, ChallengeType

# Files written to {webroot}/.well-known/acme-challenge/
handler = FileHttpHandler(webroot=Path("/var/www/html"))

client.complete_challenges(handler, challenge_type=ChallengeType.HTTP)
```

Or with callbacks:

```python
from acmeow import CallbackHttpHandler

def setup(domain: str, token: str, key_authorization: str) -> None:
    # Serve key_authorization at http://{domain}/.well-known/acme-challenge/{token}
    pass

def cleanup(domain: str, token: str) -> None:
    # Remove the challenge response
    pass

handler = CallbackHttpHandler(setup, cleanup)
```

### TLS-ALPN-01 Challenge (TLS termination control)

TLS-ALPN-01 proves domain control by serving a validation certificate with the ACME identifier extension (RFC 8737).

```python
from acmeow import CallbackTlsAlpnHandler, ChallengeType

def deploy_cert(domain: str, cert_pem: bytes, key_pem: bytes) -> None:
    # Configure your TLS server with the validation certificate
    your_tls_server.set_certificate(domain, cert_pem, key_pem)

def cleanup_cert(domain: str) -> None:
    your_tls_server.remove_certificate(domain)

handler = CallbackTlsAlpnHandler(deploy_cert, cleanup_cert)

client.complete_challenges(handler, challenge_type=ChallengeType.TLS_ALPN)
```

Or write certificates to files:

```python
from pathlib import Path
from acmeow import FileTlsAlpnHandler, ChallengeType

handler = FileTlsAlpnHandler(
    cert_dir=Path("/etc/tls/acme"),
    cert_pattern="{domain}.alpn.crt",
    key_pattern="{domain}.alpn.key",
    reload_callback=lambda: subprocess.run(["nginx", "-s", "reload"]),
)

client.complete_challenges(handler, challenge_type=ChallengeType.TLS_ALPN)
```

## Key Types

```python
from acmeow import KeyType

# Available key types for certificate private keys:
KeyType.RSA2048  # RSA 2048-bit (minimum recommended)
KeyType.RSA3072  # RSA 3072-bit
KeyType.RSA4096  # RSA 4096-bit
KeyType.RSA8192  # RSA 8192-bit (slow)
KeyType.EC256    # ECDSA P-256 (recommended)
KeyType.EC384    # ECDSA P-384
```

## Configuration Options

```python
client = AcmeClient(
    server_url="https://acme-v02.api.letsencrypt.org/directory",
    email="admin@example.com",
    storage_path=Path("./acme_data"),
    verify_ssl=True,   # Verify SSL certificates (default: True)
    timeout=30,        # Request timeout in seconds (default: 30)
)
```

### Retry Configuration

Configure automatic retry with exponential backoff for transient failures:

```python
from acmeow import AcmeClient, RetryConfig

# Custom retry settings
retry_config = RetryConfig(
    max_retries=5,      # Maximum retry attempts (default: 5)
    initial_delay=1.0,  # Initial delay in seconds (default: 1.0)
    max_delay=60.0,     # Maximum delay between retries (default: 60.0)
    multiplier=2.0,     # Exponential backoff multiplier (default: 2.0)
    jitter=True,        # Add randomness to prevent thundering herd (default: True)
)

client = AcmeClient(
    server_url="https://acme-v02.api.letsencrypt.org/directory",
    email="admin@example.com",
    storage_path=Path("./acme_data"),
    retry_config=retry_config,
)
```

The client automatically retries on:
- Rate limits (HTTP 429)
- Server errors (HTTP 500, 502, 503, 504)
- Connection errors and timeouts

### DNS Propagation Verification

Verify DNS records are visible before completing DNS-01 challenges:

```python
from acmeow import AcmeClient, DnsConfig

# Configure DNS verification
dns_config = DnsConfig(
    nameservers=["8.8.8.8", "1.1.1.1"],  # DNS servers to query
    timeout=5.0,                          # Query timeout in seconds
    retries=3,                            # Retries per server
    min_servers=1,                        # Minimum servers that must see the record
    require_all=False,                    # Require all servers to see the record
)

client = AcmeClient(...)
client.set_dns_config(dns_config)

# DNS propagation will be verified before notifying the ACME server
client.complete_challenges(handler, verify_dns=True, dns_timeout=300)
```

### Order Recovery

Orders are automatically saved and can be resumed after interruption:

```python
# Orders are automatically saved during creation
order = client.create_order([Identifier.dns("example.com")])

# If the process is interrupted, the order can be loaded later
client = AcmeClient(...)
client.create_account()  # Must create/load account first

# Load the saved order (returns None if no saved order exists)
order = client.load_order()
if order:
    print(f"Resumed order: {order.url} (status: {order.status})")
```

### Preferred Certificate Chain

Select an alternate certificate chain when downloading the certificate:

```python
# Get certificate with preferred chain (e.g., for older client compatibility)
cert_pem, key_pem = client.get_certificate(preferred_chain="ISRG Root X1")

# The preferred_chain parameter matches against the issuer CN in alternate chains
# If not found, the default chain is returned
```

### External Account Binding (EAB)

Some CAs require EAB to link ACME accounts to existing accounts:

```python
client.set_external_account_binding(
    kid="your-key-id",
    hmac_key="your-base64url-hmac-key",
)
client.create_account()
```

## ACME Servers

| CA | Production URL | Staging URL |
|----|---------------|-------------|
| Let's Encrypt | `https://acme-v02.api.letsencrypt.org/directory` | `https://acme-staging-v02.api.letsencrypt.org/directory` |
| ZeroSSL | `https://acme.zerossl.com/v2/DV90` | - |
| Buypass | `https://api.buypass.com/acme/directory` | `https://api.test4.buypass.no/acme/directory` |

## Storage Structure

```
acme_data/
├── accounts/
│   └── acme-v02.api.letsencrypt.org/
│       └── admin@example.com/
│           ├── account.json
│           └── keys/
│               └── admin@example.com.key
├── orders/
│   └── current_order.json    # Saved order for recovery
└── certificates/
    ├── example.com.crt
    └── example.com.key
```

## Exception Handling

```python
from acmeow import (
    AcmeError,              # Base exception
    AcmeServerError,        # Server returned an error
    AcmeAuthenticationError,# Account authentication failed
    AcmeAuthorizationError, # Challenge validation failed
    AcmeOrderError,         # Order creation/finalization failed
    AcmeCertificateError,   # Certificate download failed
    AcmeConfigurationError, # Invalid configuration
    AcmeNetworkError,       # Network communication failed
    AcmeTimeoutError,       # Operation timed out
    AcmeRateLimitError,     # Rate limit exceeded
    AcmeDnsError,           # DNS verification failed
)

try:
    client.complete_challenges(handler)
except AcmeRateLimitError as e:
    print(f"Rate limited, retry after {e.retry_after}s")
except AcmeDnsError as e:
    print(f"DNS verification failed for {e.domain}")
except AcmeAuthorizationError as e:
    print(f"Challenge failed for {e.domain}: {e.message}")
except AcmeError as e:
    print(f"ACME error: {e.message}")
```

## Context Manager

```python
with AcmeClient(...) as client:
    client.create_account()
    # ... operations
# Client is automatically closed
```

## Account Management

### Update Contact Email

```python
# Update account contact information
client.update_account(email="new-email@example.com")
```

### Account Key Rollover

```python
# Roll over to a new account key (RFC 8555 Section 7.3.5)
client.key_rollover()
```

### Deactivate Account

```python
# Permanently deactivate the account (RFC 8555 Section 7.3.7)
# Warning: This cannot be undone!
client.deactivate_account()
```

## Certificate Revocation

```python
from acmeow import RevocationReason

# Revoke a certificate (RFC 8555 Section 7.6)
with open("certificate.pem") as f:
    cert_pem = f.read()

# Revoke without reason
client.revoke_certificate(cert_pem)

# Or specify a revocation reason
client.revoke_certificate(cert_pem, reason=RevocationReason.KEY_COMPROMISE)
```

Available revocation reasons:
- `RevocationReason.UNSPECIFIED` - No specific reason
- `RevocationReason.KEY_COMPROMISE` - Private key compromised
- `RevocationReason.CA_COMPROMISE` - CA compromised
- `RevocationReason.AFFILIATION_CHANGED` - Affiliation changed
- `RevocationReason.SUPERSEDED` - Certificate superseded
- `RevocationReason.CESSATION_OF_OPERATION` - Operations ceased
- `RevocationReason.CERTIFICATE_HOLD` - Temporarily on hold
- `RevocationReason.REMOVE_FROM_CRL` - Remove from CRL
- `RevocationReason.PRIVILEGE_WITHDRAWN` - Privileges withdrawn
- `RevocationReason.AA_COMPROMISE` - Attribute authority compromised

## Development

### Installation

```bash
# Clone the repository
git clone https://github.com/miichoow/ACMEOW.git
cd ACMEOW

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

# Install in development mode with all dependencies
pip install -e ".[dev,docs]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=acmeow --cov-report=html

# Run specific test file
pytest tests/test_client.py

# Run specific test class
pytest tests/test_client.py::TestAcmeClientInit

# Run specific test
pytest tests/test_client.py::TestAcmeClientInit::test_init_fetches_directory
```

### Code Quality

```bash
# Run linter
ruff check src/acmeow/

# Run type checker
mypy src/acmeow/

# Format code (check only)
ruff format --check src/acmeow/

# Format code (apply changes)
ruff format src/acmeow/
```

### Building Documentation

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Build HTML documentation
cd docs
make html

# View documentation
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux
start _build/html/index.html  # Windows
```

## Examples

The `examples/` directory contains complete working examples:

| Example | Description |
|---------|-------------|
| `dns_challenge.py` | DNS-01 challenge workflow for certificate issuance |
| `http_challenge.py` | HTTP-01 challenge workflow for certificate issuance |
| `account_management.py` | Account creation, updates, and key rollover |
| `revoke_certificate.py` | Certificate revocation |
| `deactivate_account.py` | Permanent account deactivation |
| `eab_account.py` | External Account Binding for CAs like ZeroSSL |

Run examples:

```bash
python examples/dns_challenge.py
```

## License

Apache License 2.0

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
