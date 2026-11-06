
"""
src/candidate_generation/window_generator.py

Generate continuous promotional clip candidates from parsed screenplay episodes.

Design principles
-----------------
1. Promo duration must comply with the assignment: 20 to 60 seconds.
2. Candidate generation must prioritize recall, not emotional keyword matching.
3. Windows are based on wall-clock time, not sequential segment indices.
4. Overlapping screenplay elements (camera, dialogue, action, sound FX)
   are preserved as separate source segments.
5. Multiple generation strategies improve coverage around scene boundaries.
6. Every candidate retains exact source segment IDs and timestamps.
7. Candidate generation remains deterministic and independent of an LLM.

Candidate generation strategies
--------------------------------
1. Sliding windows:
   Explore the episode using configurable time steps and durations.

2. Scene-aligned windows:
   Generate windows beginning at scene starts and ending at scene ends,
   including duration-constrained windows anchored to those boundaries.

3. Consecutive scene-pair windows:
   Explore moments spanning adjacent scenes, provided the resulting
   candidate satisfies the promotional duration constraint.

This module generates continuous candidates. Non-contiguous composite
promos will be handled separately so that their individual source
segments and gaps are represented accurately.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from typing import Iterator

from src.config import CandidateConfig
from src.ingestion.episode_loader import episode_id_from_name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PS_MIN_DURATION_SECONDS = 20.0
PS_MAX_DURATION_SECONDS = 60.0

VALID_SEGMENT_KINDS = {
    "dialogue",
    "action",
    "sound_fx",
    "camera",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CandidateWindow:
    """A continuous wall-clock candidate within a single episode."""

    window_id: str
    episode: str
    episode_id: str
    source_file: str
    logline: str
    continuity_notes: str
    episode_duration_seconds: float
    scene_numbers: list[int]
    segment_ids: list[str]
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    all_segments: list[dict]
    dialogue_segments: list[dict]
    action_segments: list[dict]
    sound_fx_segments: list[dict]
    camera_segments: list[dict]
    scene_boundary_count: int

    # Additional metadata with defaults to preserve compatibility.
    generation_method: str = "sliding"
    candidate_type: str = "continuous"

    def to_dict(self) -> dict:
        """
        Serialize to a JSON-compatible dictionary.

        The complete all_segments collection is represented through
        its categorized segment lists to avoid unnecessary duplication.
        Camera segments are included to preserve visual-direction evidence.
        """

        return {
            "window_id": self.window_id,
            "candidate_type": self.candidate_type,
            "generation_method": self.generation_method,
            "episode": self.episode,
            "episode_id": self.episode_id,
            "source_file": self.source_file,
            "logline": self.logline,
            "continuity_notes": self.continuity_notes,
            "episode_duration_seconds": self.episode_duration_seconds,
            "scene_numbers": self.scene_numbers,
            "segment_ids": self.segment_ids,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "duration_seconds": self.duration_seconds,
            "dialogue_count": len(self.dialogue_segments),
            "action_count": len(self.action_segments),
            "sound_fx_count": len(self.sound_fx_segments),
            "camera_count": len(self.camera_segments),
            "scene_boundary_count": self.scene_boundary_count,
            "dialogue_segments": self.dialogue_segments,
            "action_segments": self.action_segments,
            "sound_fx_segments": self.sound_fx_segments,
            "camera_segments": self.camera_segments,
        }


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class WindowGenerator:
    """Generate continuous promotional candidates from parsed episodes."""

    def __init__(self, config: CandidateConfig) -> None:
        self.cfg = config

        # Respect the assignment's actual duration constraints even if
        # the old configuration contains a larger maximum.
        configured_min = float(
            getattr(config, "min_duration_seconds", PS_MIN_DURATION_SECONDS)
        )
        configured_max = float(
            getattr(config, "max_duration_seconds", PS_MAX_DURATION_SECONDS)
        )

        self.min_duration = max(
            PS_MIN_DURATION_SECONDS,
            configured_min,
        )
        self.max_duration = min(
            PS_MAX_DURATION_SECONDS,
            configured_max,
        )

        self.window_step = float(
            getattr(config, "window_step_seconds", 5.0)
        )

        configured_durations = getattr(
            config,
            "window_durations",
            [20, 30, 40, 50, 60],
        )

        self.window_durations = sorted({
            float(duration)
            for duration in configured_durations
            if self._is_valid_number(duration)
            and self.min_duration <= float(duration) <= self.max_duration
        })

        # Ensure the generator still explores the permitted duration
        # range if the configured duration list is outdated.
        if not self.window_durations:
            self.window_durations = [
                self.min_duration,
                min(30.0, self.max_duration),
                min(40.0, self.max_duration),
                min(50.0, self.max_duration),
                self.max_duration,
            ]

            self.window_durations = sorted({
                duration
                for duration in self.window_durations
                if self.min_duration <= duration <= self.max_duration
            })

        self.include_scene_aligned = bool(
            getattr(config, "include_scene_aligned_windows", True)
        )

        self.include_scene_pairs = bool(
            getattr(config, "include_scene_pair_windows", True)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, episode: dict) -> list[CandidateWindow]:
        """
        Generate valid, deduplicated continuous candidates for an episode.

        Candidates are sorted deterministically by start time, end time
        and window ID.

        Parameters
        ----------
        episode:
            Parsed episode dictionary.

        Returns
        -------
        list[CandidateWindow]
        """

        if not isinstance(episode, dict):
            logger.warning("Invalid episode input. Expected a dictionary.")
            return []

        ep_name = str(episode.get("episode") or "UNK")
        ep_id = episode_id_from_name(ep_name)

        raw_segments = episode.get("segments", [])
        raw_scenes = episode.get("scenes", [])

        if not isinstance(raw_segments, list):
            logger.warning("Episode %s has invalid segments.", ep_name)
            return []

        if not isinstance(raw_scenes, list):
            raw_scenes = []

        segments = self._prepare_segments(raw_segments)

        if not segments:
            logger.warning(
                "Episode '%s' has no valid segments. Skipping.",
                ep_name,
            )
            return []

        scenes = self._prepare_scenes(raw_scenes)

        episode_duration = self._get_episode_duration(
            episode,
            segments,
            scenes,
        )

        if episode_duration <= 0:
            logger.warning(
                "Episode '%s' has an invalid duration. Skipping.",
                ep_name,
            )
            return []

        if self.max_duration < self.min_duration:
            logger.error(
                "Invalid duration configuration: minimum %.2f exceeds maximum %.2f.",
                self.min_duration,
                self.max_duration,
            )
            return []

        # Each range carries its generation strategy.
        candidate_ranges: list[tuple[float, float, str]] = []

        candidate_ranges.extend(
            (start, end, "sliding")
            for start, end in self._sliding_ranges(episode_duration)
        )

        if self.include_scene_aligned:
            candidate_ranges.extend(
                (start, end, "scene_aligned")
                for start, end in self._scene_ranges(
                    scenes,
                    episode_duration,
                )
            )

        if self.include_scene_pairs:
            candidate_ranges.extend(
                (start, end, "scene_pair")
                for start, end in self._scene_pair_ranges(
                    scenes,
                    episode_duration,
                )
            )

        candidates: list[CandidateWindow] = []
        seen_signatures: set[frozenset[str]] = set()

        for start, end, method in candidate_ranges:
            candidate = self._make_window(
                episode=episode,
                ep_name=ep_name,
                ep_id=ep_id,
                segments=segments,
                scenes=scenes,
                ep_duration=episode_duration,
                t_start=start,
                t_end=end,
                generation_method=method,
                seen_signatures=seen_signatures,
            )

            if candidate is not None:
                candidates.append(candidate)

        candidates.sort(
            key=lambda candidate: (
                candidate.start_seconds,
                candidate.end_seconds,
                candidate.window_id,
            )
        )

        logger.info(
            "Episode %s: generated %d continuous candidates "
            "from %d valid segments.",
            ep_name,
            len(candidates),
            len(segments),
        )

        return candidates

    # ------------------------------------------------------------------
    # Segment and scene preparation
    # ------------------------------------------------------------------

    def _prepare_segments(self, segments: list[dict]) -> list[dict]:
        """
        Validate and chronologically sort screenplay segments.

        Invalid records are skipped rather than crashing the pipeline.
        Original segment dictionaries are preserved.
        """

        valid_segments = []

        for segment in segments:
            if not isinstance(segment, dict):
                continue

            start = self._to_float(segment.get("start_seconds"))
            end = self._to_float(segment.get("end_seconds"))

            if start is None or end is None:
                continue

            if end <= start or start < 0:
                continue

            if not segment.get("segment_id"):
                continue

            kind = str(segment.get("kind", "")).strip().lower()

            if kind not in VALID_SEGMENT_KINDS:
                continue

            valid_segments.append(segment)

        valid_segments.sort(
            key=lambda segment: (
                float(segment["start_seconds"]),
                float(segment["end_seconds"]),
                str(segment.get("segment_id", "")),
            )
        )

        return valid_segments

    def _prepare_scenes(self, scenes: list[dict]) -> list[dict]:
        """Validate and chronologically sort scene metadata."""

        valid_scenes = []

        for scene in scenes:
            if not isinstance(scene, dict):
                continue

            start = self._to_float(scene.get("start_seconds"))
            end = self._to_float(scene.get("end_seconds"))

            if start is None or end is None or end <= start:
                continue

            valid_scenes.append(scene)

        valid_scenes.sort(
            key=lambda scene: (
                float(scene["start_seconds"]),
                float(scene["end_seconds"]),
            )
        )

        return valid_scenes

    @staticmethod
    def _get_episode_duration(
        episode: dict,
        segments: list[dict],
        scenes: list[dict],
    ) -> float:
        """Determine episode duration, with a fallback to source timestamps."""

        duration = WindowGenerator._to_float(
            episode.get("episode_duration_seconds")
        )

        if duration is not None and duration > 0:
            return duration

        segment_end = max(
            float(segment["end_seconds"])
            for segment in segments
        )

        scene_end = max(
            (
                float(scene["end_seconds"])
                for scene in scenes
            ),
            default=0.0,
        )

        return max(segment_end, scene_end)

    # ------------------------------------------------------------------
    # Time-range generators
    # ------------------------------------------------------------------

    def _sliding_ranges(
        self,
        episode_duration: float,
    ) -> Iterator[tuple[float, float]]:
        """
        Generate overlapping windows across the episode.

        Windows are anchored at the beginning of the episode and advance
        using the configured step size.
        """

        if self.window_step <= 0:
            raise ValueError("window_step_seconds must be positive.")

        for duration in self.window_durations:
            if duration > episode_duration:
                continue

            start = 0.0

            while start + duration <= episode_duration + 1e-9:
                yield start, start + duration
                start += self.window_step

            # Include the final possible window ending exactly at the
            # episode boundary, even when the step does not align.
            final_start = episode_duration - duration

            if final_start >= 0:
                yield final_start, episode_duration

    def _scene_ranges(
        self,
        scenes: list[dict],
        episode_duration: float,
    ) -> Iterator[tuple[float, float]]:
        """
        Generate scene-aware candidate ranges.

        For each scene:
        - Consider the full scene.
        - Anchor configured-duration windows at its beginning.
        - Anchor configured-duration windows at its end.

        This helps discover moments that a fixed sliding grid might miss.
        """

        for scene in scenes:
            scene_start = float(scene["start_seconds"])
            scene_end = float(scene["end_seconds"])

            scene_start = max(0.0, scene_start)
            scene_end = min(episode_duration, scene_end)

            if scene_end <= scene_start:
                continue

            # Full scene candidate. The duration filter is applied later.
            yield scene_start, scene_end

            for duration in self.window_durations:
                if duration > episode_duration:
                    continue

                # Window beginning at scene start.
                end = scene_start + duration

                if end <= episode_duration:
                    yield scene_start, end

                # Window ending at scene end.
                start = scene_end - duration

                if start >= 0:
                    yield start, scene_end

    def _scene_pair_ranges(
        self,
        scenes: list[dict],
        episode_duration: float,
    ) -> Iterator[tuple[float, float]]:
        """
        Generate ranges spanning consecutive scenes.

        These candidates are useful when a dramatic beat begins in one
        scene and resolves or escalates in the next.

        Overlong scene pairs are rejected by the common duration gate.
        """

        for index in range(len(scenes) - 1):
            start = float(scenes[index]["start_seconds"])
            end = float(scenes[index + 1]["end_seconds"])

            start = max(0.0, start)
            end = min(episode_duration, end)

            if end > start:
                yield start, end

    # ------------------------------------------------------------------
    # Window construction
    # ------------------------------------------------------------------

    def _make_window(
        self,
        episode: dict,
        ep_name: str,
        ep_id: str,
        segments: list[dict],
        scenes: list[dict],
        ep_duration: float,
        t_start: float,
        t_end: float,
        seen_signatures: set[frozenset[str]],
        generation_method: str = "sliding",
    ) -> CandidateWindow | None:
        """
        Construct and validate a continuous promotional candidate.

        A segment is included if its time interval overlaps the requested
        candidate range.

        Actual candidate boundaries align with included source segment
        boundaries, preserving the ability to trace the clip to the script.
        """

        if not self._is_valid_number(t_start):
            return None

        if not self._is_valid_number(t_end):
            return None

        t_start = max(0.0, float(t_start))
        t_end = min(ep_duration, float(t_end))

        if t_end <= t_start:
            return None

        included = _segments_in_range(segments, t_start, t_end)

        if not included:
            return None

        # Use actual source segment boundaries.
        actual_start = min(
            float(segment["start_seconds"])
            for segment in included
        )

        actual_end = max(
            float(segment["end_seconds"])
            for segment in included
        )

        actual_duration = actual_end - actual_start

        # Assignment-level duration constraint.
        if not (
            self.min_duration - 1e-9
            <= actual_duration
            <= self.max_duration + 1e-9
        ):
            return None

        # Do not include segments extending beyond the episode.
        if actual_end > ep_duration + 1e-6:
            return None

        dialogue = [
            segment
            for segment in included
            if segment["kind"] == "dialogue"
        ]

        min_dialogue = int(
            getattr(self.cfg, "min_dialogue_segments", 3)
        )

        if len(dialogue) < min_dialogue:
            return None

        # Exact source segment signature.
        signature = frozenset(
            str(segment["segment_id"])
            for segment in included
        )

        if not signature or signature in seen_signatures:
            return None

        seen_signatures.add(signature)

        actions = [
            segment
            for segment in included
            if segment["kind"] == "action"
        ]

        sound_fx = [
            segment
            for segment in included
            if segment["kind"] == "sound_fx"
        ]

        cameras = [
            segment
            for segment in included
            if segment["kind"] == "camera"
        ]

        scene_numbers = sorted({
            segment.get("scene_number")
            for segment in included
            if segment.get("scene_number") is not None
        })

        scene_boundaries = _count_scene_boundaries(
            scenes,
            actual_start,
            actual_end,
        )

        # Stable ID based on episode and actual clip boundaries.
        start_cs = round(actual_start * 100)
        end_cs = round(actual_end * 100)

        window_id = (
            f"W_E{ep_id}_{start_cs:06d}_{end_cs:06d}"
        )

        return CandidateWindow(
            window_id=window_id,
            episode=ep_name,
            episode_id=ep_id,
            source_file=episode.get("source_file", ""),
            logline=episode.get("logline") or "",
            continuity_notes=episode.get("continuity_notes") or "",
            episode_duration_seconds=ep_duration,
            scene_numbers=scene_numbers,
            segment_ids=[
                str(segment["segment_id"])
                for segment in included
            ],
            start_seconds=round(actual_start, 3),
            end_seconds=round(actual_end, 3),
            duration_seconds=round(actual_duration, 3),
            all_segments=included,
            dialogue_segments=dialogue,
            action_segments=actions,
            sound_fx_segments=sound_fx,
            camera_segments=cameras,
            scene_boundary_count=scene_boundaries,
            generation_method=generation_method,
        )

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _to_float(value) -> float | None:
        """Safely convert a value to a finite float."""

        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return None

        if not math.isfinite(result):
            return None

        return result

    @staticmethod
    def _is_valid_number(value) -> bool:
        """Check whether a value can be represented as a finite number."""

        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _segments_in_range(
    segments: list[dict],
    t_start: float,
    t_end: float,
) -> list[dict]:
    """
    Return chronologically ordered segments overlapping [t_start, t_end].

    Overlap condition:
        segment.start < t_end AND segment.end > t_start

    Camera, dialogue, action and sound-effect segments can overlap
    in wall-clock time. They are retained independently.
    """

    included = [
        segment
        for segment in segments
        if (
            float(segment["start_seconds"]) < t_end
            and float(segment["end_seconds"]) > t_start
        )
    ]

    return sorted(
        included,
        key=lambda segment: (
            float(segment["start_seconds"]),
            float(segment["end_seconds"]),
            str(segment.get("segment_id", "")),
        ),
    )


def _count_scene_boundaries(
    scenes: list[dict],
    window_start: float,
    window_end: float,
) -> int:
    """
    Count scene-start boundaries strictly inside the candidate window.

    The first scene's start is not counted as a transition.
    """

    count = 0

    for index, scene in enumerate(scenes):
        if index == 0:
            continue

        boundary = scene.get("start_seconds")

        if boundary is None:
            continue

        try:
            boundary = float(boundary)
        except (TypeError, ValueError):
            continue

        if window_start < boundary < window_end:
            count += 1

    return count
