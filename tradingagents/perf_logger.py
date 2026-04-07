"""Lightweight performance logger for tracking agent and data-fetch timings.

Collects timing entries during an analysis run and prints a summary table.
Thread-safe for parallel analyst execution.
"""

import time
import threading
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

_lock = threading.Lock()

# List of (agent_name, event, duration_seconds, extra_info)
_entries: list[tuple[str, str, float, str]] = []


def reset():
    """Clear all recorded entries (call before each analysis run)."""
    with _lock:
        _entries.clear()


def log_time(agent: str, event: str, duration: float, info: str = ""):
    """Record a timing entry."""
    with _lock:
        _entries.append((agent, event, duration, info))
    # Also emit to standard logging for the message_tool.log
    logger.info(f"[PERF] {agent} | {event} | {duration:.2f}s | {info}")


def get_entries() -> list[tuple[str, str, float, str]]:
    """Return a copy of all recorded entries."""
    with _lock:
        return list(_entries)


def format_summary() -> str:
    """Return a formatted performance summary string."""
    entries = get_entries()
    if not entries:
        return "No performance data recorded."

    lines = []
    lines.append("")
    lines.append("=" * 90)
    lines.append("  PERFORMANCE SUMMARY")
    lines.append("=" * 90)
    lines.append(f"  {'Agent':<25} {'Event':<25} {'Duration':>10}  {'Details'}")
    lines.append("-" * 90)

    # Group by agent to show subtotals
    agent_totals: dict[str, float] = defaultdict(float)
    for agent, event, duration, info in entries:
        agent_totals[agent] += duration
        dur_str = f"{duration:.2f}s"
        lines.append(f"  {agent:<25} {event:<25} {dur_str:>10}  {info}")

    lines.append("-" * 90)
    lines.append(f"  {'AGENT TOTALS':<25}")
    lines.append("-" * 90)
    for agent, total in sorted(agent_totals.items(), key=lambda x: -x[1]):
        dur_str = f"{total:.2f}s"
        lines.append(f"  {agent:<25} {'total':<25} {dur_str:>10}")

    grand_total = sum(agent_totals.values())
    lines.append("-" * 90)
    lines.append(f"  {'GRAND TOTAL':<25} {'':<25} {grand_total:.2f}s")
    lines.append("=" * 90)
    lines.append("")

    return "\n".join(lines)


def print_summary():
    """Print the performance summary to stdout."""
    print(format_summary())
