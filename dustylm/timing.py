"""Small timing helpers for command-line pipeline steps."""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter


def format_duration(seconds: float) -> str:
    """Format elapsed seconds for human-readable CLI output."""
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes, remaining_seconds = divmod(round(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {remaining_seconds}s"

    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h {remaining_minutes}m {remaining_seconds}s"


@contextmanager
def timed_step(title: str, stage: str | None = None):
    """Print a stage title and elapsed time around a CLI step."""
    prefix = f"[{stage}] " if stage else ""
    print()
    print(f"{prefix}{title}")
    started_at = perf_counter()
    try:
        yield
    except Exception:
        elapsed = format_duration(perf_counter() - started_at)
        print(f"✖ Failed after {elapsed}")
        raise
    else:
        elapsed = format_duration(perf_counter() - started_at)
        print(f"✔ Completed in {elapsed}")
