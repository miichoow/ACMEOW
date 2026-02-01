#!/usr/bin/env python3
"""Example: Observability hooks for monitoring certificate issuance.

This example demonstrates how to use the hook system to monitor and
react to events during ACME certificate operations. Hooks are useful
for logging, metrics collection, alerting, and debugging.

Available Events:
- Account events: ACCOUNT_CREATED, ACCOUNT_LOADED, ACCOUNT_UPDATED, etc.
- Order events: ORDER_CREATED, ORDER_READY, ORDER_FINALIZED, ORDER_FAILED
- Challenge events: CHALLENGE_SETUP, CHALLENGE_COMPLETED, CHALLENGE_FAILED
- Certificate events: CERTIFICATE_ISSUED, CERTIFICATE_REVOKED
- Network events: REQUEST_STARTED, RATE_LIMITED, RETRY
- General events: ERROR, WARNING

Usage:
    python observability_hooks.py
"""

import json
from datetime import datetime, timezone

from acmeow import (
    HookContext,
    HookEvent,
    HookManager,
    add_hook,
    emit_hook,
    get_hook_manager,
    remove_hook,
)


def example_basic_logging() -> None:
    """Example: Basic logging hook."""
    print("=== Basic Logging Hook ===\n")

    manager = HookManager()

    def log_event(ctx: HookContext) -> None:
        """Log all events."""
        print(f"[{ctx.timestamp.isoformat()}] {ctx.event.value}: {ctx.data}")

    # Register for all events (event=None)
    manager.add_hook(None, log_event)

    # Emit some test events
    manager.emit(HookEvent.ACCOUNT_CREATED, {"email": "test@example.com"})
    manager.emit(HookEvent.ORDER_CREATED, {"domains": ["example.com"]})
    manager.emit(HookEvent.CHALLENGE_SETUP, {"domain": "example.com", "type": "dns-01"})

    print()


def example_specific_events() -> None:
    """Example: Hooking into specific events."""
    print("=== Specific Event Hooks ===\n")

    manager = HookManager()

    def on_certificate_issued(ctx: HookContext) -> None:
        """Handle certificate issuance."""
        domains = ctx.data.get("domains", [])
        print(f"Certificate issued for: {', '.join(domains)}")

    def on_error(ctx: HookContext) -> None:
        """Handle errors."""
        error = ctx.data.get("error", "Unknown error")
        print(f"ERROR: {error}")

    def on_rate_limit(ctx: HookContext) -> None:
        """Handle rate limits."""
        retry_after = ctx.data.get("retry_after", 0)
        print(f"Rate limited! Retry after {retry_after}s")

    manager.add_hook(HookEvent.CERTIFICATE_ISSUED, on_certificate_issued)
    manager.add_hook(HookEvent.ERROR, on_error)
    manager.add_hook(HookEvent.RATE_LIMITED, on_rate_limit)

    # Emit test events
    manager.emit(HookEvent.CERTIFICATE_ISSUED, {"domains": ["example.com", "www.example.com"]})
    manager.emit(HookEvent.ERROR, {"error": "Connection timeout"})
    manager.emit(HookEvent.RATE_LIMITED, {"retry_after": 30})

    # This won't trigger any hooks (no hook registered)
    manager.emit(HookEvent.ORDER_CREATED, {"domains": ["example.com"]})

    print()


def example_metrics_collection() -> None:
    """Example: Collecting metrics with hooks."""
    print("=== Metrics Collection ===\n")

    # Simple metrics collector
    metrics = {
        "accounts_created": 0,
        "orders_created": 0,
        "challenges_completed": 0,
        "certificates_issued": 0,
        "errors": 0,
        "rate_limits": 0,
        "total_events": 0,
    }

    manager = HookManager()

    def collect_metrics(ctx: HookContext) -> None:
        """Collect metrics from events."""
        metrics["total_events"] += 1

        if ctx.event == HookEvent.ACCOUNT_CREATED:
            metrics["accounts_created"] += 1
        elif ctx.event == HookEvent.ORDER_CREATED:
            metrics["orders_created"] += 1
        elif ctx.event == HookEvent.CHALLENGE_COMPLETED:
            metrics["challenges_completed"] += 1
        elif ctx.event == HookEvent.CERTIFICATE_ISSUED:
            metrics["certificates_issued"] += 1
        elif ctx.event == HookEvent.ERROR:
            metrics["errors"] += 1
        elif ctx.event == HookEvent.RATE_LIMITED:
            metrics["rate_limits"] += 1

    # Register for all events
    manager.add_hook(None, collect_metrics)

    # Simulate some operations
    manager.emit(HookEvent.ACCOUNT_CREATED, {})
    manager.emit(HookEvent.ORDER_CREATED, {})
    manager.emit(HookEvent.CHALLENGE_SETUP, {})
    manager.emit(HookEvent.CHALLENGE_COMPLETED, {})
    manager.emit(HookEvent.CHALLENGE_SETUP, {})
    manager.emit(HookEvent.CHALLENGE_COMPLETED, {})
    manager.emit(HookEvent.CERTIFICATE_ISSUED, {})
    manager.emit(HookEvent.RATE_LIMITED, {})

    print("Collected metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print()


def example_json_logging() -> None:
    """Example: JSON-formatted event logging."""
    print("=== JSON Event Logging ===\n")

    manager = HookManager()

    def log_json(ctx: HookContext) -> None:
        """Log events in JSON format for log aggregators."""
        log_entry = {
            "timestamp": ctx.timestamp.isoformat(),
            "event": ctx.event.value,
            "data": ctx.data,
            "level": "ERROR" if ctx.event in (HookEvent.ERROR, HookEvent.CHALLENGE_FAILED) else "INFO",
        }
        print(json.dumps(log_entry))

    manager.add_hook(None, log_json)

    manager.emit(HookEvent.ACCOUNT_CREATED, {"email": "admin@example.com"})
    manager.emit(HookEvent.ERROR, {"message": "Connection failed", "code": 500})

    print()


def example_alerting() -> None:
    """Example: Alerting on critical events."""
    print("=== Alerting Example ===\n")

    manager = HookManager()
    alerts_sent: list[dict] = []

    def send_alert(title: str, message: str, severity: str) -> None:
        """Simulate sending an alert (would use email, Slack, PagerDuty, etc.)."""
        alert = {
            "title": title,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        alerts_sent.append(alert)
        print(f"ALERT [{severity}]: {title}")
        print(f"  {message}")

    def on_error(ctx: HookContext) -> None:
        """Alert on errors."""
        error = ctx.data.get("error", "Unknown error")
        send_alert(
            "ACME Error",
            f"Error during certificate operation: {error}",
            "ERROR",
        )

    def on_challenge_failed(ctx: HookContext) -> None:
        """Alert on failed challenges."""
        domain = ctx.data.get("domain", "unknown")
        reason = ctx.data.get("reason", "Unknown reason")
        send_alert(
            "Challenge Failed",
            f"Challenge failed for {domain}: {reason}",
            "WARNING",
        )

    def on_certificate_issued(ctx: HookContext) -> None:
        """Notify on successful issuance."""
        domains = ctx.data.get("domains", [])
        send_alert(
            "Certificate Issued",
            f"New certificate issued for: {', '.join(domains)}",
            "INFO",
        )

    manager.add_hook(HookEvent.ERROR, on_error)
    manager.add_hook(HookEvent.CHALLENGE_FAILED, on_challenge_failed)
    manager.add_hook(HookEvent.CERTIFICATE_ISSUED, on_certificate_issued)

    # Simulate events
    manager.emit(HookEvent.CERTIFICATE_ISSUED, {"domains": ["example.com"]})
    manager.emit(HookEvent.CHALLENGE_FAILED, {"domain": "test.com", "reason": "DNS timeout"})
    manager.emit(HookEvent.ERROR, {"error": "Network unreachable"})

    print(f"\nTotal alerts sent: {len(alerts_sent)}")
    print()


def example_global_hooks() -> None:
    """Example: Using global hook functions."""
    print("=== Global Hook Functions ===\n")

    # Use the global hook manager for convenience
    def my_callback(ctx: HookContext) -> None:
        print(f"Global hook received: {ctx.event.value}")

    # Add hook using convenience function
    add_hook(HookEvent.DNS_PROPAGATION_CHECK, my_callback)

    # Emit event using convenience function
    emit_hook(HookEvent.DNS_PROPAGATION_CHECK, {"domain": "example.com"})

    # Remove hook
    result = remove_hook(HookEvent.DNS_PROPAGATION_CHECK, my_callback)
    print(f"Hook removed: {result}")

    # Get the manager if needed
    manager = get_hook_manager()
    print(f"Total hooks registered: {manager.get_hook_count()}")
    print()


def example_integration_code() -> None:
    """Show how to integrate hooks with AcmeClient."""
    print("=== Integration with AcmeClient ===\n")

    print("To integrate hooks with certificate issuance:")
    print()
    print('''from acmeow import (
    AcmeClient,
    HookEvent,
    add_hook,
    ChallengeType,
    Identifier,
)

# Set up hooks before creating client
def log_progress(ctx):
    print(f"[{ctx.event.value}] {ctx.data}")

def on_cert_issued(ctx):
    # Send notification, update dashboard, etc.
    domains = ctx.data.get("domains", [])
    print(f"Certificate ready for: {domains}")

# Register hooks
add_hook(None, log_progress)  # All events
add_hook(HookEvent.CERTIFICATE_ISSUED, on_cert_issued)

# Create client and issue certificate
client = AcmeClient(
    server_url="https://acme-v02.api.letsencrypt.org/directory",
    email="admin@example.com",
    storage_path="./acme_data",
)

# Operations will trigger registered hooks
client.create_account()  # -> ACCOUNT_CREATED or ACCOUNT_LOADED
order = client.create_order([Identifier.dns("example.com")])  # -> ORDER_CREATED
# ... complete challenges, finalize, etc.''')
    print()


def main() -> None:
    """Run the observability hooks examples."""
    example_basic_logging()
    example_specific_events()
    example_metrics_collection()
    example_json_logging()
    example_alerting()
    example_global_hooks()
    example_integration_code()

    print("=== Summary ===")
    print("The hook system allows you to:")
    print("- Monitor certificate issuance in real-time")
    print("- Collect metrics for dashboards")
    print("- Send alerts on errors or failures")
    print("- Log events in custom formats")
    print("- Integrate with external systems")


if __name__ == "__main__":
    main()
