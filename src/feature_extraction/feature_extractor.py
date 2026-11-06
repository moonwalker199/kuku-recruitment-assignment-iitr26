"""
src/feature_extraction/feature_extractor.py — Deterministic feature computation.

Computes 12 numerical features (all in [0.0, 1.0]) for every CandidateWindow.
No ML model, no external API — purely rule-based, fully reproducible.

Feature catalogue
-----------------
F01  emotional_intensity    Delivery-tag weighted emotion across dialogue
F02  cliffhanger_score      Hook / unresolved tension at window end
F03  conflict_score         Confrontation / antagonism between speakers
F04  tension_arc            Emotion escalation across the window timeline
F05  lexical_hook_score     Promo-keyword density in dialogue + action text
F06  revelation_score       Surprising disclosure or key twist
F07  dialogue_density       Lines per unit time (dramatic pacing)
F08  protagonist_presence   Juliana / Alexander speaking in the window
F09  speaker_variety        Number of unique speakers (dialogue exchange)
F10  position_in_episode    Slight recency bias toward episode endings
F11  sound_fx_density       Cinematic atmosphere markers
F12  scene_boundary_bonus   Scene transitions within the window
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.candidate_generation.window_generator import CandidateWindow
from src.feature_extraction.lexicons import (
    CLIFFHANGER_KEYWORDS,
    CLIFFHANGER_TAGS,
    CONFLICT_TAGS,
    HOOK_KEYWORDS,
    PROTAGONIST_NAMES,
    REVELATION_TAGS,
    score_delivery_tag,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Dialogue rate normaliser: 5 lines / 10 s → 1.0
_DIALOGUE_DENSITY_NORM = 5.0 / 10.0  # lines per second at which score = 1.0

# Speaker count at which speaker_variety = 1.0
_MAX_SPEAKERS = 4

# sound_fx count per 10 s at which score = 1.0
_SFX_DENSITY_NORM = 2.0 / 10.0

# Fraction of window length to treat as "tail" for cliffhanger analysis
_CLIFFHANGER_TAIL = 0.75


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """Stateless feature extractor; thread-safe."""

    def extract(self, window: CandidateWindow) -> dict[str, float]:
        """
        Compute all 12 deterministic features for *window*.

        Returns a dict mapping feature name → float in [0.0, 1.0].
        """
        dialogue = window.dialogue_segments
        actions = window.action_segments
        sound_fx = window.sound_fx_segments
        dur = window.duration_seconds

        return {
            "emotional_intensity":  _emotional_intensity(dialogue),
            "cliffhanger_score":    _cliffhanger_score(window),
            "conflict_score":       _conflict_score(dialogue),
            "tension_arc":          _tension_arc(dialogue, window),
            "lexical_hook_score":   _lexical_hook_score(window),
            "revelation_score":     _revelation_score(dialogue),
            "dialogue_density":     _dialogue_density(dialogue, dur),
            "protagonist_presence": _protagonist_presence(dialogue),
            "speaker_variety":      _speaker_variety(dialogue),
            "position_in_episode":  _position_in_episode(window),
            "sound_fx_density":     _sound_fx_density(sound_fx, dur),
            "scene_boundary_bonus": _scene_boundary_bonus(window),
        }


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------

def _emotional_intensity(dialogue: list[dict]) -> float:
    """
    F01 — Average delivery-tag intensity across all dialogue segments.

    Each segment's score is the max over its delivery tags. Segments with
    no delivery tag receive 0.1 (non-zero because silence is still acting).
    """
    if not dialogue:
        return 0.0

    scores: list[float] = []
    for seg in dialogue:
        raw_delivery = seg.get("delivery") or ""
        if raw_delivery:
            tags = [t.strip() for t in raw_delivery.split(",") if t.strip()]
            seg_score = max(score_delivery_tag(t) for t in tags)
        else:
            seg_score = 0.10
        scores.append(seg_score)

    return min(sum(scores) / len(scores), 1.0)


def _cliffhanger_score(window: CandidateWindow) -> float:
    """
    F02 — Hook / unresolved-tension score focused on the tail of the window.

    Two sub-signals:
      (a) Cliffhanger delivery tags on dialogue in the last 25% of the window.
      (b) Cliffhanger keywords in ACTION / SOUND FX text in the last 25%.
    """
    cutoff = window.start_seconds + window.duration_seconds * _CLIFFHANGER_TAIL

    # (a) Late dialogue with cliffhanger delivery
    late_dialogue = [
        s for s in window.dialogue_segments
        if s["start_seconds"] >= cutoff
    ]
    tag_hits = sum(
        1 for s in late_dialogue
        if _has_tag_in_group(s.get("delivery") or "", CLIFFHANGER_TAGS)
    )
    tag_score = min(tag_hits / 2.0, 1.0) if late_dialogue else 0.0

    # (b) Late non-dialogue text with cliffhanger keywords
    late_other = [
        s for s in window.all_segments
        if s["kind"] in ("action", "sound_fx") and s["start_seconds"] >= cutoff
    ]
    late_text = " ".join(s.get("text", "").lower() for s in late_other)
    kw_hits = sum(1 for kw in CLIFFHANGER_KEYWORDS if kw in late_text)
    kw_score = min(kw_hits / 3.0, 1.0)

    # Weighted combination (dialogue signal weighted higher)
    combined = 0.65 * tag_score + 0.35 * kw_score

    # Bonus: last segment is a sound_fx (ominous audio cue)
    last_seg = window.all_segments[-1] if window.all_segments else None
    if last_seg and last_seg["kind"] == "sound_fx":
        combined = min(combined + 0.10, 1.0)

    return combined


def _conflict_score(dialogue: list[dict]) -> float:
    """
    F03 — Fraction of dialogue with conflict-class delivery tags.

    Also rewards multi-speaker exchanges (back-and-forth confrontation).
    """
    if not dialogue:
        return 0.0

    conflict_count = sum(
        1 for s in dialogue
        if _has_tag_in_group(s.get("delivery") or "", CONFLICT_TAGS)
    )
    conflict_fraction = conflict_count / len(dialogue)

    # Multi-speaker diversity bonus (capped at 0.30 of total score)
    speakers = {s["speaker"] for s in dialogue if s.get("speaker")}
    speaker_bonus = min(len(speakers) / 3.0, 1.0) * 0.30

    return min(conflict_fraction * 0.70 + speaker_bonus, 1.0)


def _tension_arc(dialogue: list[dict], window: CandidateWindow) -> float:
    """
    F04 — Measures whether emotional intensity escalates toward the window end.

    The window is split into thirds; score rewards windows where the late
    third has higher average intensity than the early third.
    """
    if len(dialogue) < 3:
        return 0.0

    span = window.duration_seconds
    t0 = window.start_seconds
    early_cut = t0 + span * 0.33
    mid_cut = t0 + span * 0.67

    early = [s for s in dialogue if s["start_seconds"] < early_cut]
    late = [s for s in dialogue if s["start_seconds"] >= mid_cut]

    def _avg(segs: list[dict]) -> float:
        if not segs:
            return 0.3
        intensities = []
        for s in segs:
            raw = s.get("delivery") or ""
            if raw:
                tags = [t.strip() for t in raw.split(",") if t.strip()]
                intensities.append(max(score_delivery_tag(t) for t in tags))
            else:
                intensities.append(0.1)
        return sum(intensities) / len(intensities)

    early_score = _avg(early)
    late_score = _avg(late)

    # Normalised improvement: how much late exceeds early
    arc = (late_score - early_score) / (early_score + 0.01)
    return max(0.0, min(arc, 1.0))


def _lexical_hook_score(window: CandidateWindow) -> float:
    """
    F05 — Promotional keyword density across dialogue and action text.

    Score = number of unique keywords hit / 5 (saturates at 5 hits).
    """
    all_text = " ".join(
        (s.get("text") or "").lower()
        for s in window.all_segments
        if s["kind"] in ("dialogue", "action")
    )
    hits = sum(1 for kw in HOOK_KEYWORDS if kw in all_text)
    return min(hits / 5.0, 1.0)


def _revelation_score(dialogue: list[dict]) -> float:
    """
    F06 — Fraction of dialogue carrying revelation / shock delivery tags.
    """
    if not dialogue:
        return 0.0
    hits = sum(
        1 for s in dialogue
        if _has_tag_in_group(s.get("delivery") or "", REVELATION_TAGS)
    )
    return min(hits / max(len(dialogue) * 0.3, 1.0), 1.0)


def _dialogue_density(dialogue: list[dict], duration: float) -> float:
    """
    F07 — Dialogue lines per unit time (pacing).

    Normalised so that 5 lines / 10 s → 1.0 (high dramatic pacing).
    """
    if duration <= 0:
        return 0.0
    rate = len(dialogue) / duration  # lines per second
    return min(rate / _DIALOGUE_DENSITY_NORM, 1.0)


def _protagonist_presence(dialogue: list[dict]) -> float:
    """
    F08 — Fraction of dialogue lines spoken by the protagonist (Juliana / Alexander).

    Ranges 0.0 (protagonists absent) → 1.0 (all lines from protagonists).
    """
    if not dialogue:
        return 0.0
    proto_lines = sum(
        1 for s in dialogue
        if (s.get("speaker") or "").lower() in PROTAGONIST_NAMES
    )
    return proto_lines / len(dialogue)


def _speaker_variety(dialogue: list[dict]) -> float:
    """
    F09 — Unique speaker count, normalised so 4+ speakers → 1.0.
    """
    if not dialogue:
        return 0.0
    speakers = {s["speaker"] for s in dialogue if s.get("speaker")}
    return min(len(speakers) / _MAX_SPEAKERS, 1.0)


def _position_in_episode(window: CandidateWindow) -> float:
    """
    F10 — Normalised temporal position (0 = episode start, 1 = end).

    Uses square root to soften the bias — mid-episode hooks score ~0.7
    rather than a sharp cliff at the very end.
    """
    ep_dur = window.episode_duration_seconds
    if ep_dur <= 0:
        return 0.0
    centre = (window.start_seconds + window.end_seconds) / 2.0
    return (centre / ep_dur) ** 0.5


def _sound_fx_density(sound_fx: list[dict], duration: float) -> float:
    """
    F11 — Sound FX segments per 10 s, normalised so 2 / 10 s → 1.0.
    """
    if duration <= 0:
        return 0.0
    rate = len(sound_fx) / (duration / 10.0)
    return min(rate / _SFX_DENSITY_NORM, 1.0)


def _scene_boundary_bonus(window: CandidateWindow) -> float:
    """
    F12 — Normalised scene-boundary count inside the window.

    0 crossings → 0.0, 1 → 0.5, 2+ → 1.0.
    """
    return min(window.scene_boundary_count / 2.0, 1.0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _has_tag_in_group(delivery_str: str, tag_set: frozenset[str]) -> bool:
    """Return True if any tag in *delivery_str* belongs to *tag_set*."""
    if not delivery_str:
        return False
    for tag in delivery_str.split(","):
        if tag.strip().lower() in tag_set:
            return True
    return False
