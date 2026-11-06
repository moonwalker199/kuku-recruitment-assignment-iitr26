"""
src/ingestion/episode_loader.py — Load parsed episode JSON files.

Responsible for discovering and reading the structured JSON outputs produced
by src/preprocessing/screenplay_parser.py.  Returns a list of episode dicts
in the same format the parser writes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_episodes(processed_dir: str | Path) -> list[dict]:
    """
    Load all *_parsed.json files from *processed_dir*.

    Returns a list of episode dicts sorted by episode number (ascending).
    Logs a warning for any file that cannot be parsed; never raises.
    """
    processed_path = Path(processed_dir)

    if not processed_path.is_dir():
        raise FileNotFoundError(
            f"Processed data directory not found: {processed_path.resolve()}"
        )

    json_files = sorted(processed_path.glob("*_parsed.json"))

    if not json_files:
        raise FileNotFoundError(
            f"No *_parsed.json files found in {processed_path.resolve()}"
        )

    episodes: list[dict] = []

    for path in json_files:
        try:
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            # Minimal sanity check
            if "segments" not in data or "episode" not in data:
                logger.warning("Skipping %s — missing required keys.", path.name)
                continue
            episodes.append(data)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s — %s", path.name, exc)

    logger.info("Loaded %d episode(s) from %s", len(episodes), processed_path)
    return episodes


def episode_id_from_name(episode_str: str) -> str:
    """
    Extract the numeric episode ID from an episode string.

    'Episode 7'  → '7'
    'Episode 42' → '42'
    'UNK'        → 'UNK'
    """
    import re
    m = re.search(r"\d+", episode_str or "")
    return m.group() if m else "UNK"
