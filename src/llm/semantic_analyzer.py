
"""
src/llm/semantic_analyzer.py

Episode-level LLM analysis and AI-assisted promo boundary optimization.

Pipeline:
1. Process episodes chronologically.
2. Make one LLM call per episode for story understanding and candidate discovery.
3. Pass a compact running story summary between episodes.
4. Validate candidates against original screenplay segments.
5. Generate boundary alternatives using Python.
6. Use a separate LLM stage to select the best boundary alternative.
7. Cache successful results and track API usage.

Legacy window-level analysis remains available during migration.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from src.candidate_generation.window_generator import CandidateWindow
from src.config import Config
from src.llm.client import LLMClient, LLMUnavailable
from src.llm.prompts import (
    build_analysis_prompt,
    build_episode_analysis_prompt,
    build_boundary_optimization_prompt,
    generate_boundary_options,
    validate_episode_analysis_response,
    validate_boundary_optimization_response,
    validate_llm_response,
)

logger = logging.getLogger(__name__)

LLM_FEATURE_KEYS = (
    "emotional_intensity",
    "conflict_score",
    "cliffhanger_score",
    "revelation_score",
    "hook_potential",
    "context_independence",
    "promotional_suitability",
)

_ANALYSIS_VERSION = "episode_analysis_v1"
_BOUNDARY_VERSION = "boundary_optimization_v1"
_MAX_RUNNING_SUMMARY_CHARS = 6000


class SemanticAnalyzer:
    """
    Orchestrates episode-level analysis and boundary optimization.

    New public methods:
        analyze_episodes(episodes)
        optimize_boundaries(episodes, candidates)

    Legacy public method:
        analyze_batch(windows, det_scores)
    """

    def __init__(
        self,
        cfg: Config,
        cache_dir: str | Path = "outputs/cache/episode_analysis",
        boundary_cache_dir: str | Path = "outputs/cache/boundary_optimization",
    ) -> None:
        self.cfg = cfg

        self.cache_dir = Path(cache_dir)
        self.boundary_cache_dir = Path(boundary_cache_dir)

        self._client: LLMClient | None = None
        self._boundary_client: LLMClient | None = None

        self._available = False
        self._boundary_available = False

        self.stats: dict[str, Any] = {
            "episodes_received": 0,
            "episodes_analyzed": 0,
            "episodes_cached": 0,
            "episodes_failed": 0,
            "api_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "candidates_discovered": 0,

            "boundary_candidates_received": 0,
            "boundary_candidates_optimized": 0,
            "boundary_candidates_cached": 0,
            "boundary_candidates_failed": 0,
            "boundary_api_calls": 0,
            "boundary_successful_calls": 0,
            "boundary_failed_calls": 0,
        }

        if not cfg.llm_enabled:
            logger.info("LLM analysis disabled in configuration.")
            return

        try:
            self._client = LLMClient.from_config(
                cfg,
                stage="episode_analysis",
            )
            self._available = True

            logger.info(
                "Episode analysis client initialized: provider=%s, model=%s",
                self._client.provider,
                self._client.model,
            )

        except LLMUnavailable as exc:
            logger.warning(
                "Episode analysis LLM unavailable: %s",
                exc,
            )

    @property
    def is_available(self) -> bool:
        """Whether the episode-analysis LLM client initialized."""
        return self._available

    # ========================================================
    # EPISODE-LEVEL ANALYSIS
    # ========================================================

    def analyze_episodes(
        self,
        episodes: list[dict],
        max_candidates_per_episode: int = 5,
    ) -> list[dict]:
        """
        Analyze complete episodes sequentially.

        Each episode receives the running story summary from previous
        successfully analyzed episodes.

        Failed episodes are included with analysis_status="failed".
        """

        self.stats["episodes_received"] = len(episodes)

        if not self._available or self._client is None:
            logger.warning(
                "Episode analysis skipped because the LLM is unavailable."
            )
            return []

        ordered_episodes = sorted(
            episodes,
            key=self._episode_sort_key,
        )

        running_summary = ""
        results: list[dict] = []

        logger.info(
            "Starting episode-level analysis for %d episodes.",
            len(ordered_episodes),
        )

        for index, episode in enumerate(ordered_episodes):
            episode_name = str(
                episode.get("episode", f"Episode {index + 1}")
            )

            logger.info(
                "Analyzing %s (%d/%d).",
                episode_name,
                index + 1,
                len(ordered_episodes),
            )

            try:
                analysis = self._analyze_episode(
                    episode=episode,
                    running_summary=running_summary,
                    max_candidates=max_candidates_per_episode,
                )

                if analysis is None:
                    self.stats["episodes_failed"] += 1

                    results.append({
                        "episode": episode_name,
                        "episode_summary": "",
                        "story_updates": [],
                        "unresolved_storylines": [],
                        "candidate_moments": [],
                        "analysis_status": "failed",
                    })
                    continue

                analysis["episode"] = episode_name
                analysis["analysis_status"] = "success"

                results.append(analysis)

                self.stats["episodes_analyzed"] += 1
                self.stats["candidates_discovered"] += len(
                    analysis["candidate_moments"]
                )

                running_summary = self._update_running_summary(
                    running_summary,
                    analysis,
                )

                logger.info(
                    "%s: discovered %d candidate moments.",
                    episode_name,
                    len(analysis["candidate_moments"]),
                )

            except Exception:
                self.stats["episodes_failed"] += 1

                logger.exception(
                    "Unexpected episode analysis failure: %s",
                    episode_name,
                )

                results.append({
                    "episode": episode_name,
                    "episode_summary": "",
                    "story_updates": [],
                    "unresolved_storylines": [],
                    "candidate_moments": [],
                    "analysis_status": "failed",
                })

        logger.info(
            "Episode analysis complete: %d successful, %d failed, "
            "%d candidates, %d API calls, %d cache hits.",
            self.stats["episodes_analyzed"],
            self.stats["episodes_failed"],
            self.stats["candidates_discovered"],
            self.stats["api_calls"],
            self.stats["episodes_cached"],
        )

        return results

    def _analyze_episode(
        self,
        episode: dict,
        running_summary: str,
        max_candidates: int,
    ) -> dict | None:
        """Build, call, validate and cache one episode analysis."""

        episode_name = str(episode.get("episode", "UNKNOWN"))

        cache_path = self._get_cache_path(
            episode,
            running_summary,
            max_candidates,
        )

        cached = self._read_cache(cache_path)

        if cached is not None:
            validated = validate_episode_analysis_response(
                cached,
                episode,
            )

            if validated is not None:
                self.stats["episodes_cached"] += 1
                logger.info("Using cached analysis for %s.", episode_name)
                return validated

            logger.warning(
                "Cached analysis for %s failed validation. Reanalyzing.",
                episode_name,
            )

        try:
            prompt = build_episode_analysis_prompt(
                episode=episode,
                running_summary=running_summary,
                episode_number=episode_name,
                max_candidates=max_candidates,
            )

        except Exception as exc:
            logger.error(
                "Could not build episode prompt for %s: %s",
                episode_name,
                exc,
            )
            return None

        self.stats["api_calls"] += 1

        try:
            raw_response = self._client.call_json(prompt)

        except Exception as exc:
            self.stats["failed_calls"] += 1
            logger.error(
                "LLM API call failed for %s: %s",
                episode_name,
                exc,
            )
            return None

        if raw_response is None:
            self.stats["failed_calls"] += 1
            logger.warning("LLM returned no response for %s.", episode_name)
            return None

        validated = validate_episode_analysis_response(
            raw_response,
            episode,
        )

        if validated is None:
            self.stats["failed_calls"] += 1
            logger.warning(
                "Episode-level response validation failed for %s.",
                episode_name,
            )
            return None

        self.stats["successful_calls"] += 1
        self._write_cache(cache_path, raw_response)

        return validated

    # ========================================================
    # BOUNDARY OPTIMIZATION
    # ========================================================

    def optimize_boundaries(
        self,
        episodes: list[dict],
        candidates: list[dict],
    ) -> list[dict]:
        """
        Optimize candidate start/end boundaries using a separate LLM stage.

        Python first generates valid boundary alternatives. The LLM selects
        one of those alternatives. It cannot invent timestamps or segment IDs.

        Returns copies of all input candidates. Successful candidates receive
        optimized boundary fields. Failed candidates retain their original
        boundaries and receive boundary_status="failed".
        """

        self.stats["boundary_candidates_received"] = len(candidates)

        if not candidates:
            logger.info("No candidates supplied for boundary optimization.")
            return []

        if not self.cfg.llm_enabled:
            logger.warning("Boundary optimization skipped: LLM is disabled.")
            return [
                {**candidate, "boundary_status": "skipped"}
                for candidate in candidates
            ]

        episode_map = {
            str(episode.get("episode", "")): episode
            for episode in episodes
        }

        # Initialize this client only when the boundary stage is actually used.
        if self._boundary_client is None and not self._boundary_available:
            try:
                self._boundary_client = LLMClient.from_config(
                    self.cfg,
                    stage="boundary_optimization",
                )
                self._boundary_available = True

                logger.info(
                    "Boundary optimization client initialized: "
                    "provider=%s, model=%s",
                    self._boundary_client.provider,
                    self._boundary_client.model,
                )

            except LLMUnavailable as exc:
                logger.warning(
                    "Boundary optimization client unavailable: %s",
                    exc,
                )

        if not self._boundary_available or self._boundary_client is None:
            logger.warning(
                "Boundary optimization skipped because its LLM is unavailable."
            )
            return [
                {**candidate, "boundary_status": "skipped"}
                for candidate in candidates
            ]

        optimized_candidates: list[dict] = []

        for index, candidate in enumerate(candidates):
            result = dict(candidate)
            episode_name = str(candidate.get("episode", ""))

            episode = episode_map.get(episode_name)

            if episode is None:
                logger.warning(
                    "Cannot optimize candidate %s: episode %s not found.",
                    candidate.get("candidate_id", index),
                    episode_name,
                )
                result["boundary_status"] = "failed"
                result["boundary_error"] = "source_episode_not_found"
                self.stats["boundary_candidates_failed"] += 1
                optimized_candidates.append(result)
                continue

            try:
                optimized = self._optimize_single_candidate(
                    episode=episode,
                    candidate=candidate,
                )

                if optimized is None:
                    result["boundary_status"] = "failed"
                    self.stats["boundary_candidates_failed"] += 1
                else:
                    result.update(optimized)
                    self.stats["boundary_candidates_optimized"] += 1

                optimized_candidates.append(result)

            except Exception:
                logger.exception(
                    "Unexpected boundary optimization failure for candidate %s.",
                    candidate.get("candidate_id", index),
                )

                result["boundary_status"] = "failed"
                self.stats["boundary_candidates_failed"] += 1
                optimized_candidates.append(result)

        logger.info(
            "Boundary optimization complete: %d optimized, %d failed, "
            "%d cached, %d API calls.",
            self.stats["boundary_candidates_optimized"],
            self.stats["boundary_candidates_failed"],
            self.stats["boundary_candidates_cached"],
            self.stats["boundary_api_calls"],
        )

        return optimized_candidates

    def _optimize_single_candidate(
        self,
        episode: dict,
        candidate: dict,
    ) -> dict | None:
        """Optimize one candidate using Python-generated boundary options."""

        options = generate_boundary_options(
            episode=episode,
            candidate=candidate,
        )

        if not options:
            logger.warning(
                "No valid boundary options for candidate %s.",
                candidate.get("candidate_id", "UNKNOWN"),
            )
            return None

        cache_path = self._get_boundary_cache_path(
            episode,
            candidate,
            options,
        )

        cached = self._read_cache(cache_path)

        if cached is not None:
            validated = validate_boundary_optimization_response(
                cached,
                options,
                episode,
            )

            if validated is not None:
                self.stats["boundary_candidates_cached"] += 1
                logger.info(
                    "Using cached boundary selection for %s.",
                    candidate.get("candidate_id", "UNKNOWN"),
                )
                return self._format_boundary_result(validated, cached)

            logger.warning(
                "Cached boundary selection failed validation for %s.",
                candidate.get("candidate_id", "UNKNOWN"),
            )

        try:
            prompt = build_boundary_optimization_prompt(
                episode=episode,
                candidate=candidate,
                options=options,
            )

        except Exception as exc:
            logger.error(
                "Could not build boundary prompt for %s: %s",
                candidate.get("candidate_id", "UNKNOWN"),
                exc,
            )
            return None

        self.stats["boundary_api_calls"] += 1

        try:
            raw_response = self._boundary_client.call_json(prompt)

        except Exception as exc:
            self.stats["boundary_failed_calls"] += 1
            logger.error(
                "Boundary API call failed for %s: %s",
                candidate.get("candidate_id", "UNKNOWN"),
                exc,
            )
            return None

        if raw_response is None:
            self.stats["boundary_failed_calls"] += 1
            logger.warning(
                "Boundary LLM returned no response for %s.",
                candidate.get("candidate_id", "UNKNOWN"),
            )
            return None

        validated = validate_boundary_optimization_response(
            raw_response,
            options,
            episode,
        )

        if validated is None:
            self.stats["boundary_failed_calls"] += 1
            logger.warning(
                "Boundary response validation failed for %s.",
                candidate.get("candidate_id", "UNKNOWN"),
            )
            return None

        self.stats["boundary_successful_calls"] += 1
        self._write_cache(cache_path, raw_response)

        return self._format_boundary_result(validated, raw_response)

    @staticmethod
    def _format_boundary_result(
        validated: dict,
        raw_response: dict,
    ) -> dict:
        """
        Convert a validated boundary selection into candidate fields.

        The original candidate's narrative information is preserved.
        """

        return {
            "start_segment_id": validated["start_segment_id"],
            "end_segment_id": validated["end_segment_id"],
            "segment_ids": validated["segment_ids"],
            "start_seconds": validated["start_seconds"],
            "end_seconds": validated["end_seconds"],
            "start_timestamp": validated["start_timestamp"],
            "end_timestamp": validated["end_timestamp"],
            "duration_seconds": validated["duration_seconds"],
            "boundary_status": "success",
            "boundary_option_id": validated["option_id"],
            "boundary_reasoning": validated["reasoning"],
            "boundary_confidence": validated["confidence"],
        }

    def _get_boundary_cache_path(
        self,
        episode: dict,
        candidate: dict,
        options: list[dict],
    ) -> Path:
        """Generate a cache key for a candidate's boundary decision."""

        payload = {
            "version": _BOUNDARY_VERSION,
            "episode": episode,
            "candidate": candidate,
            "options": options,
        }

        serialized = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )

        digest = hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

        candidate_id = str(
            candidate.get("candidate_id", "unknown")
        )

        safe_candidate_id = "".join(
            char if char.isalnum() or char in ("-", "_") else "_"
            for char in candidate_id
        )

        return (
            self.boundary_cache_dir
            / f"{safe_candidate_id}_{digest[:16]}.json"
        )

    # ========================================================
    # RUNNING STORY SUMMARY
    # ========================================================

    @staticmethod
    def _update_running_summary(
        previous_summary: str,
        analysis: dict,
    ) -> str:
        """Carry forward concise story information."""

        current_episode = analysis.get("episode", "Unknown episode")
        episode_summary = analysis.get("episode_summary", "")

        updates = analysis.get("story_updates", [])
        unresolved = analysis.get("unresolved_storylines", [])

        parts = []

        if previous_summary:
            parts.append(previous_summary.strip())

        parts.append(
            f"{current_episode}: {episode_summary.strip()}"
        )

        if updates:
            parts.append(
                "Important developments: "
                + "; ".join(str(item) for item in updates)
            )

        if unresolved:
            parts.append(
                "Unresolved storylines: "
                + "; ".join(str(item) for item in unresolved)
            )

        combined = "\n".join(parts)

        if len(combined) > _MAX_RUNNING_SUMMARY_CHARS:
            combined = combined[-_MAX_RUNNING_SUMMARY_CHARS:]

            first_newline = combined.find("\n")

            if first_newline >= 0:
                combined = combined[first_newline + 1:]

        return combined

    # ========================================================
    # CACHE
    # ========================================================

    @staticmethod
    def _episode_sort_key(episode: dict) -> int:
        """Extract numeric episode order."""

        import re

        episode_name = str(episode.get("episode", ""))
        match = re.search(r"\d+", episode_name)

        return int(match.group()) if match else 0

    def _get_cache_path(
        self,
        episode: dict,
        running_summary: str,
        max_candidates: int,
    ) -> Path:
        """Generate a cache key for an episode analysis."""

        payload = {
            "version": _ANALYSIS_VERSION,
            "episode": episode,
            "running_summary": running_summary,
            "max_candidates": max_candidates,
        }

        serialized = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )

        digest = hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

        episode_id = str(episode.get("episode", "unknown"))

        safe_episode_id = "".join(
            char if char.isalnum() or char in ("-", "_") else "_"
            for char in episode_id
        )

        return self.cache_dir / f"{safe_episode_id}_{digest[:16]}.json"

    @staticmethod
    def _read_cache(path: Path) -> dict | None:
        """Read a cached raw LLM response."""

        if not path.is_file():
            return None

        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            return data if isinstance(data, dict) else None

        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _write_cache(path: Path, data: dict) -> None:
        """Write a successful raw LLM response to disk."""

        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            temporary_path = path.with_suffix(".tmp")

            with temporary_path.open("w", encoding="utf-8") as file:
                json.dump(
                    data,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            temporary_path.replace(path)

        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Could not write LLM cache: %s", exc)

    # ========================================================
    # LEGACY WINDOW-LEVEL ANALYSIS
    # ========================================================

    def analyze_batch(
        self,
        windows: list[CandidateWindow],
        det_scores: list[float],
    ) -> list[dict | None]:
        """Legacy candidate-window analysis."""

        results: list[dict | None] = [None] * len(windows)

        if not self._available or self._client is None:
            return results

        if len(windows) != len(det_scores):
            raise ValueError(
                "windows and det_scores must have equal lengths."
            )

        threshold = self.cfg.llm.llm_score_threshold
        sent = 0

        for index, (window, score) in enumerate(zip(windows, det_scores)):
            if score < threshold:
                continue

            results[index] = self._analyze_single(window)
            sent += 1

        logger.info(
            "Legacy LLM window analysis: %d/%d windows submitted.",
            sent,
            len(windows),
        )

        return results

    def _analyze_single(
        self,
        window: CandidateWindow,
    ) -> dict | None:
        """Legacy analysis of one candidate window."""

        try:
            prompt = build_analysis_prompt(window)
            raw_response = self._client.call_json(prompt)

            if raw_response is None:
                return None

            return validate_llm_response(raw_response)

        except Exception as exc:
            logger.warning(
                "Legacy window analysis failed for %s: %s",
                window.window_id,
                exc,
            )
            return None


# ============================================================
# LEGACY FEATURE MERGING
# ============================================================

def merge_llm_features(
    features: dict,
    llm_analysis: dict | None,
) -> dict:
    """Merge legacy window-level LLM features."""

    merged = dict(features)

    if llm_analysis is None:
        return merged

    merged["llm_promotional_suitability"] = (
        llm_analysis.get("promotional_suitability", 0.0)
    )

    merged["llm_hook_potential"] = (
        llm_analysis.get("hook_potential", 0.0)
    )

    merged["llm_context_independence"] = (
        llm_analysis.get("context_independence", 0.0)
    )

    return merged



