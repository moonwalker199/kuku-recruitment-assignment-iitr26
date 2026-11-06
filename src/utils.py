"""
src/utils.py — Shared utilities: logging setup, time formatting, banner.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> None:
    """Configure root logger to stderr with a clean format."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(
        stream=sys.stderr,
        level=level,
        format=fmt,
        datefmt=datefmt,
    )
    # Silence noisy third-party loggers
    for noisy in ("urllib3", "httpcore", "httpx", "google", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------

def seconds_to_timestamp(seconds: float) -> str:
    """Convert fractional seconds to MM:SS.cc display string."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


def format_duration(seconds: float) -> str:
    """Human-readable duration, e.g. '1m 23s' or '45s'."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs:02d}s"


# ---------------------------------------------------------------------------
# Output dir helpers
# ---------------------------------------------------------------------------

def ensure_output_dirs(cfg) -> None:
    """Create all configured output directories if they don't exist."""
    dirs = [
        cfg.output.candidates_dir,
        cfg.output.ranked_promos_dir,
        cfg.output.rejected_candidates_dir,
        cfg.output.reports_dir,
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# CLI banner
# ---------------------------------------------------------------------------

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║   🎙  Kuku FM — Audio Drama Promo Recommendation System      ║
║   End-to-end explainable promotional moment ranking          ║
╚══════════════════════════════════════════════════════════════╝
"""


def print_banner() -> None:
    print(BANNER, flush=True)


# ---------------------------------------------------------------------------
# Simple elapsed-time context manager
# ---------------------------------------------------------------------------

class Timer:
    """Context manager that records wall-clock elapsed time."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start

    def __str__(self):
        return format_duration(self.elapsed)
