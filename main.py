
#!/usr/bin/env python3
"""
main.py - PromoFinder AI

Episode-level promotional moment discovery pipeline.

Stages:
1. Load and validate parsed screenplay episodes.
2. Analyze each complete episode once using the LLM.
3. Carry a running story summary between episodes.
4. Discover candidate promotional moments with source segment IDs.
5. Calculate generic structural signals.
6. Pre-rank candidates while preserving episode diversity.
7. Optimize shortlisted candidate boundaries.
8. Evaluate optimized promos using five semantic dimensions.
9. Rank, separate and save final promotional candidates.
10. Save execution statistics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Load environment variables before importing the LLM client.
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(".env"), override=False)
except ImportError:
    pass

from src.config import load_config, apply_overrides
from src.utils import setup_logging, print_banner, Timer
from src.ingestion.episode_loader import load_episodes
from src.llm.semantic_analyzer import SemanticAnalyzer

logger = logging.getLogger(__name__)


# ============================================================
# CLI
# ============================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="PromoFinder AI: Drama promotional moment discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Configuration file.",
    )

    parser.add_argument(
        "--input",
        default="data/processed/",
        help="Directory containing parsed episode JSON files.",
    )

    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM analysis.",
    )

    parser.add_argument(
        "--max-candidates-per-episode",
        type=int,
        default=5,
        help="Maximum candidate moments requested per episode.",
    )

    parser.add_argument(
        "--shortlist-size",
        type=int,
        default=30,
        help="Number of candidates retained for AI evaluation.",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )

    return parser


# ============================================================
# GENERIC STRUCTURAL ANALYSIS
# ============================================================

def calculate_structural_signals(
    candidate: dict,
    source_segments: list[dict],
) -> dict:
    """
    Calculate show-agnostic structural signals.

    These support preliminary selection. They do not attempt
    to understand emotions or narrative meaning.
    """

    selected_ids = {
        str(item)
        for item in candidate.get("segment_ids", [])
    }

    segments = [
        segment
        for segment in source_segments
        if str(segment.get("segment_id")) in selected_ids
    ]

    if not segments:
        return {
            "dialogue_share": 0.0,
            "speaker_diversity": 0.0,
            "scene_diversity": 0.0,
            "kind_transition_ratio": 0.0,
            "segment_density": 0.0,
            "structural_score": 0.0,
        }

    segments.sort(key=lambda item: item["start_seconds"])

    total = len(segments)

    dialogue_segments = [
        segment for segment in segments
        if segment.get("kind") == "dialogue"
    ]

    speakers = {
        str(segment.get("speaker")).strip().lower()
        for segment in dialogue_segments
        if segment.get("speaker")
    }

    scenes = {
        segment.get("scene_number")
        for segment in segments
        if segment.get("scene_number") is not None
    }

    kinds = [
        segment.get("kind", "unknown")
        for segment in segments
    ]

    transitions = sum(
        1
        for index in range(1, len(kinds))
        if kinds[index] != kinds[index - 1]
    )

    dialogue_share = len(dialogue_segments) / total
    speaker_diversity = min(len(speakers) / 4.0, 1.0)
    scene_diversity = min(len(scenes) / 3.0, 1.0)

    transition_ratio = (
        transitions / (total - 1)
        if total > 1 else 0.0
    )

    duration = max(
        float(candidate.get("duration_seconds", 0.0)),
        1.0,
    )

    segment_density = min(total / duration, 1.0)

    structural_score = (
        0.25 * dialogue_share
        + 0.25 * speaker_diversity
        + 0.15 * scene_diversity
        + 0.20 * transition_ratio
        + 0.15 * segment_density
    )

    return {
        "dialogue_share": round(dialogue_share, 4),
        "speaker_diversity": round(speaker_diversity, 4),
        "scene_diversity": round(scene_diversity, 4),
        "kind_transition_ratio": round(transition_ratio, 4),
        "segment_density": round(segment_density, 4),
        "structural_score": round(structural_score, 4),
    }


# ============================================================
# CANDIDATE PREPARATION
# ============================================================

def prepare_discovered_candidates(
    episodes: list[dict],
    episode_analyses: list[dict],
) -> list[dict]:
    """Attach source segments and structural signals to candidates."""

    episode_map = {
        str(episode.get("episode")): episode
        for episode in episodes
    }

    prepared = []

    for analysis in episode_analyses:
        episode_name = str(analysis.get("episode", ""))

        if analysis.get("analysis_status") != "success":
            continue

        source_episode = episode_map.get(episode_name)

        if source_episode is None:
            logger.warning(
                "Could not match analysis to source episode: %s",
                episode_name,
            )
            continue

        source_segments = source_episode.get("segments", [])

        segment_map = {
            str(segment.get("segment_id")): segment
            for segment in source_segments
            if segment.get("segment_id") is not None
        }

        for candidate in analysis.get("candidate_moments", []):
            segment_ids = [
                str(item)
                for item in candidate.get("segment_ids", [])
            ]

            source_excerpt = [
                segment_map[segment_id]
                for segment_id in segment_ids
                if segment_id in segment_map
            ]

            if len(source_excerpt) != len(segment_ids):
                logger.warning(
                    "Candidate %s has missing source segments. Skipping.",
                    candidate.get("candidate_id"),
                )
                continue

            structural = calculate_structural_signals(
                candidate,
                source_segments,
            )

            prepared_candidate = dict(candidate)

            prepared_candidate["episode"] = episode_name
            prepared_candidate["logline"] = source_episode.get("logline", "")
            prepared_candidate["source_file"] = source_episode.get("source_file")
            prepared_candidate["source_segments"] = source_excerpt
            prepared_candidate["structural_signals"] = structural

            confidence = float(
                candidate.get("candidate_confidence", 0.0)
            )

            structural_score = structural["structural_score"]

            prepared_candidate["preliminary_score"] = round(
                0.30 * confidence + 0.70 * structural_score,
                4,
            )

            prepared.append(prepared_candidate)

    return prepared


# ============================================================
# DIVERSITY-AWARE SHORTLIST
# ============================================================

def select_shortlist(
    candidates: list[dict],
    shortlist_size: int,
    max_per_episode_first_pass: int = 2,
) -> list[dict]:
    """Select a preliminary shortlist with episode diversity."""

    if shortlist_size <= 0:
        return []

    ordered = sorted(
        candidates,
        key=lambda item: (
            -item.get("preliminary_score", 0.0),
            str(item.get("episode", "")),
            str(item.get("candidate_id", "")),
        ),
    )

    selected = []
    selected_ids = set()
    episode_counts = defaultdict(int)

    for candidate in ordered:
        if len(selected) >= shortlist_size:
            break

        episode = str(candidate.get("episode", "UNKNOWN"))
        identity = (
            episode,
            str(candidate.get("candidate_id", "")),
        )

        if episode_counts[episode] >= max_per_episode_first_pass:
            continue

        selected.append(candidate)
        selected_ids.add(identity)
        episode_counts[episode] += 1

    if len(selected) < shortlist_size:
        for candidate in ordered:
            if len(selected) >= shortlist_size:
                break

            identity = (
                str(candidate.get("episode", "UNKNOWN")),
                str(candidate.get("candidate_id", "")),
            )

            if identity in selected_ids:
                continue

            selected.append(candidate)
            selected_ids.add(identity)

    return selected


# ============================================================
# BOUNDARY RESULT PREPARATION
# ============================================================

def refresh_candidate_source_segments(
    candidates: list[dict],
    episodes: list[dict],
) -> list[dict]:
    """Refresh source excerpts after boundary optimization."""

    episode_map = {
        str(episode.get("episode")): episode
        for episode in episodes
    }

    refreshed = []

    for candidate in candidates:
        result = dict(candidate)

        episode_name = str(result.get("episode", ""))
        episode = episode_map.get(episode_name)

        if episode is None:
            refreshed.append(result)
            continue

        segment_map = {
            str(segment.get("segment_id")): segment
            for segment in episode.get("segments", [])
            if segment.get("segment_id") is not None
        }

        segment_ids = [
            str(item)
            for item in result.get("segment_ids", [])
        ]

        source_excerpt = [
            segment_map[segment_id]
            for segment_id in segment_ids
            if segment_id in segment_map
        ]

        if len(source_excerpt) != len(segment_ids):
            logger.warning(
                "Candidate %s has missing segments after boundary "
                "optimization. Keeping its existing source excerpt.",
                result.get("candidate_id"),
            )
        else:
            result["source_segments"] = source_excerpt

        refreshed.append(result)

    return refreshed


# ============================================================
# FINAL RANKING
# ============================================================
def deduplicate_evaluated_candidates(
    candidates: list[dict],
    iou_threshold: float = 0.5,
    enabled: bool = True,
) -> tuple[list[dict], list[dict]]:
    """
    Suppress highly overlapping evaluated promos within the same episode.

    Only successfully evaluated accept/review candidates compete.
    The higher-scoring candidate is retained.

    Returns:
        retained_candidates, suppressed_candidates
    """

    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in the range (0, 1].")

    retained = []
    suppressed = []

    if not enabled:
        for candidate in candidates:
            result = dict(candidate)
            result["deduplication_status"] = "disabled"
            retained.append(result)

        return retained, suppressed

    eligible_by_episode = defaultdict(list)
    ineligible = []

    for candidate in candidates:
        result = dict(candidate)

        is_eligible = (
            result.get("evaluation_status") == "success"
            and result.get("recommendation") in {"accept", "review"}
        )

        if is_eligible:
            eligible_by_episode[
                str(result.get("episode", "UNKNOWN"))
            ].append(result)
        else:
            result["deduplication_status"] = "not_applicable"
            ineligible.append(result)

    def score(candidate: dict) -> float:
        try:
            return float(candidate.get("overall_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def hook_score(candidate: dict) -> float:
        try:
            return float(
                candidate.get("evaluation_scores", {}).get(
                    "hook_first_3_seconds", 0.0
                ) or 0.0
            )
        except (TypeError, ValueError):
            return 0.0

    for episode_name, episode_candidates in eligible_by_episode.items():
        ordered = sorted(
            episode_candidates,
            key=lambda item: (
                -score(item),
                -hook_score(item),
                str(item.get("candidate_id", "")),
            ),
        )

        kept_for_episode = []

        for candidate in ordered:
            try:
                start = float(candidate["start_seconds"])
                end = float(candidate["end_seconds"])
            except (KeyError, TypeError, ValueError):
                candidate["deduplication_status"] = "invalid_timestamps"
                retained.append(candidate)
                continue

            if end <= start:
                candidate["deduplication_status"] = "invalid_timestamps"
                retained.append(candidate)
                continue

            duplicate_of = None

            for kept in kept_for_episode:
                kept_start = float(kept["start_seconds"])
                kept_end = float(kept["end_seconds"])

                intersection = max(
                    0.0,
                    min(end, kept_end) - max(start, kept_start),
                )

                union = (
                    max(end, kept_end) - min(start, kept_start)
                )

                iou = intersection / union if union > 0 else 0.0

                if iou >= iou_threshold:
                    duplicate_of = kept
                    break

            if duplicate_of is not None:
                candidate["deduplication_status"] = "suppressed"
                candidate["duplicate_of_candidate_id"] = (
                    duplicate_of.get("candidate_id")
                )
                candidate["rejection_reason"] = (
                    "overlap_with_higher_scoring_candidate"
                )
                candidate["overlap_iou_threshold"] = iou_threshold
                suppressed.append(candidate)
            else:
                candidate["deduplication_status"] = "retained"
                kept_for_episode.append(candidate)
                retained.append(candidate)

    retained.extend(ineligible)

    return retained, suppressed



def rank_evaluated_candidates(
    candidates: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Rank candidates by semantic score.

    Recommendation labels determine output categories, not ranking order.
    """

    def safe_score(value: Any) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    # ranked = sorted(
    #     candidates,
    #     key=lambda item: (
    #         -float(item.get("overall_score", 0.0) or 0.0),
    #         -float(
    #             item.get("evaluation_scores", {}).get(
    #                 "hook_first_3_seconds", 0.0
    #             ) or 0.0
    #         ),
    #         str(item.get("episode", "")),
    #         str(item.get("candidate_id", "")),
    #     ),
    # )
    ranked = sorted(
        candidates,
        key=lambda item: (
            -safe_score(item.get("overall_score")),
            -safe_score(
                item.get("evaluation_scores", {}).get(
                    "hook_first_3_seconds", 0.0
                )
            ),
            str(item.get("episode", "")),
            str(item.get("candidate_id", "")),
        ),
    )

    for position, candidate in enumerate(ranked, start=1):
        candidate["final_rank"] = position

    rejected = [
        candidate
        for candidate in ranked
        if candidate.get("recommendation") == "reject"
    ]

    return ranked, rejected


# ============================================================
# OUTPUT HELPERS
# ============================================================

def write_json(data: Any, path: str | Path) -> None:
    """Write JSON atomically."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    temporary_path.replace(path)


def save_episode_candidate_files(
    candidates: list[dict],
    output_dir: str | Path,
) -> None:
    """Save discovered candidates grouped by episode."""

    grouped = defaultdict(list)

    for candidate in candidates:
        episode = str(candidate.get("episode", "UNKNOWN"))
        grouped[episode].append(candidate)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for episode, items in grouped.items():
        safe_name = episode.lower().replace(" ", "_")

        write_json(
            items,
            output_dir / f"{safe_name}_discovered_candidates.json",
        )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run(
    cfg,
    input_dir: str,
    max_candidates_per_episode: int = 5,
    shortlist_size: int = 30,
    output_root: str | Path = "outputs",
    cache_root: str | Path | None = None,
    api_key: str | None = None,
    progress_callback=None,
) -> dict:
    """Execute discovery, optimization, evaluation and ranking."""

    stats = {
        "episodes_loaded": 0,
        "episodes_analyzed": 0,
        "episodes_failed": 0,
        "total_discovered_candidates": 0,
        "shortlisted_candidates": 0,
        "api_calls": 0,
        "successful_calls": 0,
        "failed_calls": 0,
        "cache_hits": 0,
        "llm_enabled": False,

        "boundary_candidates_received": 0,
        "boundary_candidates_optimized": 0,
        "boundary_candidates_failed": 0,
        "boundary_candidates_cached": 0,
        "boundary_api_calls": 0,
        "boundary_successful_calls": 0,
        "boundary_failed_calls": 0,

        "evaluation_candidates_received": 0,
        "evaluation_candidates_evaluated": 0,
        "evaluation_candidates_cached": 0,
        "evaluation_candidates_failed": 0,
        "evaluation_api_calls": 0,
        "evaluation_successful_calls": 0,
        "evaluation_failed_calls": 0,
        "evaluation_accepted": 0,
        "evaluation_review": 0,
        "evaluation_rejected": 0,

        "deduplication_enabled": False,
        "deduplication_iou_threshold": 0.5,
        "deduplication_candidates_received": 0,
        "duplicate_promos_suppressed": 0,
    }

    def report_progress(
        stage: str,
        current: int,
        total: int,
        message: str,
        status: str = "running",
    ) -> None:
        if progress_callback is None:
            return

        try:
            progress_callback({
                "stage": stage,
                "current": current,
                "total": total,
                "message": message,
                "status": status,
            })
        except Exception:
            logger.exception("Progress callback failed.")


    # --------------------------------------------------------
    # Stage 1: Load episodes
    # --------------------------------------------------------

    report_progress("loading", 0, 1, "Loading uploaded episodes")

    with Timer() as load_timer:
        episodes = load_episodes(input_dir)

    stats["episodes_loaded"] = len(episodes)

    report_progress(
        "loading",
        1,
        1,
        f"Loaded {len(episodes)} episodes",
        "complete",
    )

    logger.info(
        "Loaded %d episodes in %s.",
        len(episodes),
        load_timer,
    )

    if not episodes:
        raise ValueError("No valid episodes were loaded.")

    # --------------------------------------------------------
    # Stage 2: Episode-level LLM analysis
    # --------------------------------------------------------
    # --------------------------------------------------------

    # --------------------------------------------------------

    analyzer_kwargs = {}

    if cache_root is not None:
        cache_root = Path(cache_root)
        analyzer_kwargs = {
            "cache_dir": cache_root / "episode_analysis",
            "boundary_cache_dir": cache_root / "boundary_optimization",
            "evaluation_cache_dir": cache_root / "promo_evaluation",
        }

    if api_key:
        analyzer_kwargs["api_key"] = api_key

    analyzer = SemanticAnalyzer(
        cfg,
        progress_callback=progress_callback,
        **analyzer_kwargs,
    )
    stats["llm_enabled"] = analyzer.is_available
    with Timer() as analysis_timer:
        episode_analyses = analyzer.analyze_episodes(
            episodes,
            max_candidates_per_episode=max_candidates_per_episode,
        )

    report_progress(
        "episode_analysis",
        len(episodes),
        len(episodes),
        f"Analyzed {analyzer.stats.get('episodes_analyzed', len(episodes))} episodes; "
        f"{analyzer.stats.get('candidates_discovered', 0)} candidates discovered",
        "complete",
    )

    logger.info(
        "Episode analysis completed: %d episodes analysed in %s.",
        len(episode_analyses),
        analysis_timer,
    )
    
    # --------------------------------------------------------
    # Stage 3: Prepare discovered candidates
    # --------------------------------------------------------

    with Timer() as preparation_timer:
        candidates = prepare_discovered_candidates(
            episodes,
            episode_analyses,
        )

    report_progress(
        "preparation",
        1,
        1,
        f"Prepared {len(candidates)} candidates",
        "complete",
    )

    logger.info(
        "Prepared %d candidates in %s.",
        len(candidates),
        preparation_timer,
    )

    # --------------------------------------------------------
    # Stage 4: Generic structural pre-ranking
    # --------------------------------------------------------

    shortlist = select_shortlist(
        candidates,
        shortlist_size=shortlist_size,
    )

    report_progress(
        "shortlisting",
        1,
        1,
        f"Selected {len(shortlist)} candidates for optimization",
        "complete",
    )

    logger.info(
        "Shortlist selected: %d / %d discovered candidates.",
        len(shortlist),
        len(candidates),
    )

    pre_boundary_shortlist = [
        dict(candidate)
        for candidate in shortlist
    ]

    # --------------------------------------------------------
    # Stage 5: AI-assisted boundary optimization
    # --------------------------------------------------------

    with Timer() as boundary_timer:
        optimized_shortlist = analyzer.optimize_boundaries(
            episodes=episodes,
            candidates=shortlist,
        )

    optimized_shortlist = refresh_candidate_source_segments(
        optimized_shortlist,
        episodes,
    )

    report_progress(
        "boundary_complete",
        1,
        1,
        f"Boundary optimization processed {len(optimized_shortlist)} candidates",
        "complete",
    )

    logger.info(
        "Boundary optimization completed in %s.",
        boundary_timer,
    )

    # --------------------------------------------------------
    # Stage 6: Semantic promo evaluation
    # --------------------------------------------------------

    with Timer() as evaluation_timer:
        evaluated_candidates = analyzer.evaluate_promos(
            episodes=episodes,
            candidates=optimized_shortlist,
        )

    report_progress(
        "evaluation_complete",
        1,
        1,
        f"Evaluated {len(evaluated_candidates)} candidates",
        "complete",
    )

    logger.info(
        "Promo evaluation completed in %s.",
        evaluation_timer,
    )

    # --------------------------------------------------------
    # Stage 7: Deduplication, final ranking and separation
    # --------------------------------------------------------

    diversity_cfg = getattr(
        getattr(cfg, "ranking", None),
        "diversity",
        None,
    )

    dedup_enabled = getattr(diversity_cfg, "enabled", True)
    dedup_iou_threshold = getattr(
        diversity_cfg,
        "iou_threshold",
        0.5,
    )

    deduplicated_candidates, suppressed_candidates = (
        deduplicate_evaluated_candidates(
            evaluated_candidates,
            iou_threshold=dedup_iou_threshold,
            enabled=dedup_enabled,
        )
    )

    ranked_candidates, rejected_candidates = (
        rank_evaluated_candidates(deduplicated_candidates)
    )

    # Suppressed duplicates are also retained in the rejected output
    # so the human reviewer can inspect what was removed.
    rejected_candidates.extend(suppressed_candidates)

    accepted_candidates = [
        candidate
        for candidate in ranked_candidates
        if candidate.get("recommendation") == "accept"
    ]

    review_candidates = [
        candidate
        for candidate in ranked_candidates
        if candidate.get("recommendation") == "review"
    ]

    report_progress(
        "ranking",
        1,
        1,
        f"Accepted: {len(accepted_candidates)} | "
        f"Review: {len(review_candidates)} | "
        f"Rejected: {len(rejected_candidates)}",
        "complete",
    )

    logger.info(
        "Final ranking: %d accepted, %d review, %d rejected, "
        "%d duplicates suppressed.",
        len(accepted_candidates),
        len(review_candidates),
        len(rejected_candidates),
        len(suppressed_candidates),
    )

    # --------------------------------------------------------
    # Stage 8: Save outputs
    # --------------------------------------------------------

    # output_root = Path("outputs")
    output_root = Path(output_root)

    analysis_path = (
        output_root / "episode_analysis" / "episode_analysis.json"
    )

    discovered_path = (
        output_root / "candidates" / "discovered_candidates.json"
    )

    pre_boundary_path = (
        output_root / "candidates" / "shortlist_pre_boundary.json"
    )

    shortlist_path = (
        output_root / "candidates" / "shortlist.json"
    )

    ranked_path = (
        output_root / "ranked_promos" / "ranked_promos.json"
    )

    accepted_path = (
        output_root / "ranked_promos" / "accepted_promos.json"
    )

    review_path = (
        output_root / "ranked_promos" / "review_promos.json"
    )

    rejected_path = (
        output_root / "rejected_candidates" / "rejected_promos.json"
    )

    stats_path = (
        output_root / "reports" / "discovery_stats.json"
    )

    write_json(episode_analyses, analysis_path)
    write_json(candidates, discovered_path)
    write_json(pre_boundary_shortlist, pre_boundary_path)
    write_json(optimized_shortlist, shortlist_path)
    write_json(ranked_candidates, ranked_path)
    write_json(accepted_candidates, accepted_path)
    write_json(review_candidates, review_path)
    write_json(rejected_candidates, rejected_path)

    save_episode_candidate_files(
        candidates,
        output_root / "candidates" / "by_episode",
    )

    # --------------------------------------------------------
    # Stage 9: Collect execution statistics
    # --------------------------------------------------------

    analyzer_stats = analyzer.stats

    stats.update({
        "episodes_analyzed": analyzer_stats.get(
            "episodes_analyzed", 0
        ),
        "episodes_failed": analyzer_stats.get(
            "episodes_failed", 0
        ),
        "total_discovered_candidates": len(candidates),
        "shortlisted_candidates": len(pre_boundary_shortlist),

        "api_calls": analyzer_stats.get("api_calls", 0),
        "successful_calls": analyzer_stats.get(
            "successful_calls", 0
        ),
        "failed_calls": analyzer_stats.get("failed_calls", 0),
        "cache_hits": analyzer_stats.get("episodes_cached", 0),

        "boundary_candidates_received": analyzer_stats.get(
            "boundary_candidates_received", 0
        ),
        "boundary_candidates_optimized": analyzer_stats.get(
            "boundary_candidates_optimized", 0
        ),
        "boundary_candidates_failed": analyzer_stats.get(
            "boundary_candidates_failed", 0
        ),
        "boundary_candidates_cached": analyzer_stats.get(
            "boundary_candidates_cached", 0
        ),
        "boundary_api_calls": analyzer_stats.get(
            "boundary_api_calls", 0
        ),
        "boundary_successful_calls": analyzer_stats.get(
            "boundary_successful_calls", 0
        ),
        "boundary_failed_calls": analyzer_stats.get(
            "boundary_failed_calls", 0
        ),

        "evaluation_candidates_received": analyzer_stats.get(
            "evaluation_candidates_received", 0
        ),
        "evaluation_candidates_evaluated": analyzer_stats.get(
            "evaluation_candidates_evaluated", 0
        ),
        "evaluation_candidates_cached": analyzer_stats.get(
            "evaluation_candidates_cached", 0
        ),
        "evaluation_candidates_failed": analyzer_stats.get(
            "evaluation_candidates_failed", 0
        ),
        "evaluation_api_calls": analyzer_stats.get(
            "evaluation_api_calls", 0
        ),
        "evaluation_successful_calls": analyzer_stats.get(
            "evaluation_successful_calls", 0
        ),
        "evaluation_failed_calls": analyzer_stats.get(
            "evaluation_failed_calls", 0
        ),
        "evaluation_accepted": analyzer_stats.get(
            "evaluation_accepted", 0
        ),
        "evaluation_review": analyzer_stats.get(
            "evaluation_review", 0
        ),
        "evaluation_rejected": analyzer_stats.get(
            "evaluation_rejected", 0
        ),

        "accepted_promos": len(accepted_candidates),
        "review_promos": len(review_candidates),
       "rejected_promos": sum(  
            1
            for candidate in ranked_candidates
            if candidate.get("recommendation") == "reject"
),
        "rejected_output_items": len(rejected_candidates),

        "analysis_runtime": str(analysis_timer),
        "boundary_runtime": str(boundary_timer),
        "evaluation_runtime": str(evaluation_timer),
        "preparation_runtime": str(preparation_timer),

        "deduplication_enabled": dedup_enabled,
        "deduplication_iou_threshold": dedup_iou_threshold,
        "deduplication_candidates_received": len(evaluated_candidates),
        "duplicate_promos_suppressed": len(suppressed_candidates),
    })

    write_json(stats, stats_path)

    report_progress(
        "saving",
        1,
        1,
        "All output files saved",
        "complete",
    )

    logger.info("Episode analyses: %s", analysis_path)
    logger.info("Discovered candidates: %s", discovered_path)
    logger.info("Pre-boundary shortlist: %s", pre_boundary_path)
    logger.info("Optimized shortlist: %s", shortlist_path)
    logger.info("Final ranking: %s", ranked_path)
    logger.info("Accepted promos: %s", accepted_path)
    logger.info("Review promos: %s", review_path)
    logger.info("Rejected promos: %s", rejected_path)
    logger.info("Statistics: %s", stats_path)

    return stats


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.max_candidates_per_episode < 1:
        parser.error("--max-candidates-per-episode must be at least 1.")

    if args.shortlist_size < 1:
        parser.error("--shortlist-size must be at least 1.")

    setup_logging(verbose=args.verbose)
    print_banner()

    try:
        cfg = load_config(args.config)

    except (FileNotFoundError, ValueError) as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    apply_overrides(
        cfg,
        {
            "llm_enabled": False if args.no_llm else None,
        },
    )

    logger.info(
        "PromoFinder AI starting: input=%s, llm_enabled=%s",
        args.input,
        cfg.llm_enabled,
    )

    with Timer() as total_timer:
        try:
            stats = run(
                cfg=cfg,
                input_dir=args.input,
                max_candidates_per_episode=args.max_candidates_per_episode,
                shortlist_size=args.shortlist_size,
            )

        except Exception as exc:
            logger.exception("Pipeline failed: %s", exc)
            sys.exit(1)

    print("\n" + "=" * 60)
    print("  PROMOFINDER AI: PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Runtime                 : {total_timer}")
    print(f"  Episodes loaded         : {stats['episodes_loaded']}")
    print(f"  Episodes analyzed       : {stats['episodes_analyzed']}")
    print(f"  Episodes failed         : {stats['episodes_failed']}")
    print(f"  Episode API calls       : {stats['api_calls']}")
    print(f"  Episode cache hits      : {stats['cache_hits']}")
    print(f"  Candidates discovered   : {stats['total_discovered_candidates']}")
    print(f"  Candidates shortlisted  : {stats['shortlisted_candidates']}")
    print("-" * 60)
    print(f"  Boundary optimized      : {stats['boundary_candidates_optimized']}")
    print(f"  Boundary failures       : {stats['boundary_candidates_failed']}")
    print(f"  Boundary API calls     : {stats['boundary_api_calls']}")
    print("-" * 60)
    print(f"  Evaluation candidates   : {stats['evaluation_candidates_received']}")
    print(f"  Evaluation successes    : {stats['evaluation_candidates_evaluated']}")
    print(f"  Evaluation failures     : {stats['evaluation_candidates_failed']}")
    print(f"  Evaluation API calls    : {stats['evaluation_api_calls']}")
    print(f"  Evaluation cache hits   : {stats['evaluation_candidates_cached']}")
    print("-" * 60)
    print(f"  Accepted promos         : {stats['accepted_promos']}")
    print(f"  Needs human review      : {stats['review_promos']}")
    print(f"  Semantically rejected   : {stats['rejected_promos']}")
    print(f"  Duplicates suppressed   : {stats['duplicate_promos_suppressed']}")
    print(f"  Rejected output entries : {stats['rejected_output_items']}")
    print("=" * 60)
    print("  Outputs:")
    print("  outputs/candidates/discovered_candidates.json")
    print("  outputs/candidates/shortlist_pre_boundary.json")
    print("  outputs/candidates/shortlist.json")
    print("  outputs/ranked_promos/ranked_promos.json")
    print("  outputs/ranked_promos/accepted_promos.json")
    print("  outputs/ranked_promos/review_promos.json")
    print("  outputs/rejected_candidates/rejected_promos.json")
    print("  outputs/reports/discovery_stats.json")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
