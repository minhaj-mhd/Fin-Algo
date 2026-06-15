"""
Network connectivity monitor for the Vanguard engine.

Provides a lightweight liveness probe (`is_online`) and a blocking
"halt until the network is back" helper (`wait_for_network`). The main
orchestrator loop, the shadow-tracker loop, and startup all call
`wait_for_network()` so the engine pauses cleanly during an internet/broker
outage instead of spamming API errors, then resumes automatically the moment
connectivity returns.
"""
import socket
import time
from datetime import datetime

from scripts.vanguard import config
from scripts.terminal_utils import log

# (host, port) pairs probed to decide whether we are online. The first two are
# raw IPs (no DNS dependency) of public DNS resolvers; the third validates DNS
# resolution plus reachability of the broker API itself. We are considered
# online if ANY probe succeeds — a single working route is enough to trade.
_PROBE_TARGETS = [
    ("8.8.8.8", 53),          # Google DNS
    ("1.1.1.1", 53),          # Cloudflare DNS
    ("api.upstox.com", 443),  # Broker API (also exercises DNS resolution)
]


def is_online(timeout=None):
    """Return True if any probe target is reachable within `timeout`, else False."""
    if timeout is None:
        timeout = getattr(config, "NETWORK_PROBE_TIMEOUT", 3.0)
    for host, port in _PROBE_TARGETS:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            continue
    return False


def _fmt_duration(start):
    secs = int((datetime.now() - start).total_seconds())
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def wait_for_network(label="SYSTEM", check_interval=None, max_backoff=60):
    """
    Block until the network is reachable.

    Returns immediately with False if already online (the normal hot path,
    costing one fast TCP connect). If the network is down, logs the outage,
    polls until connectivity is restored, logs the recovery, and returns True.
    Honours the NETWORK_MONITOR_ENABLED config flag (returns False when off).
    """
    if not getattr(config, "NETWORK_MONITOR_ENABLED", True):
        return False
    if is_online():
        return False

    if check_interval is None:
        check_interval = getattr(config, "NETWORK_CHECK_INTERVAL", 15)
    down_since = datetime.now()
    interval = check_interval
    log(f"[NETWORK] {label}: connection DOWN — halting engine and waiting for the network to come back...")

    while True:
        time.sleep(interval)
        if is_online():
            log(f"[NETWORK] {label}: connection RESTORED after {_fmt_duration(down_since)} — resuming.")
            return True
        log(f"[NETWORK] {label}: still offline ({_fmt_duration(down_since)} down) — retrying in {interval}s...")
        # Gentle linear backoff so logs don't flood during a long outage.
        interval = min(max_backoff, interval + check_interval)
