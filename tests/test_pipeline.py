
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from main import (
    deduplicate_evaluated_candidates,
    rank_evaluated_candidates,
    run,
)


# ------------------------------------------------------------
# DEDUPLICATION AND RANKING TEST
# ------------------------------------------------------------

def test_deduplication_and_ranking():
    def make_candidate(
        candidate_id,
        episode_name,
        start,
        end,
        score,
    ):
        return {
            "candidate_id": candidate_id,
            "episode": episode_name,
            "start_seconds": start,
            "end_seconds": end,
            "overall_score": score,
            "evaluation_status": "success",
            "recommendation": "accept",
            "evaluation_scores": {
                "hook_first_3_seconds": 0.90,
            },
        }

    candidates = [
        # A and B overlap heavily. A has the higher score.
        make_candidate("A", "Episode 1", 0, 30, 0.90),
        make_candidate("B", "Episode 1", 5, 35, 0.80),

        # C does not overlap with A.
        make_candidate("C", "Episode 1", 40, 70, 0.85),

        # D overlaps with A and B in time, but belongs to Episode 2.
        make_candidate("D", "Episode 2", 5, 35, 0.82),
    ]

    retained, suppressed = deduplicate_evaluated_candidates(
        candidates,
        iou_threshold=0.50,
        enabled=True,
    )

    retained_ids = {
        item["candidate_id"]
        for item in retained
    }

    suppressed_ids = {
        item["candidate_id"]
        for item in suppressed
    }

    assert retained_ids == {"A", "C", "D"}, (
        f"Unexpected retained candidates: {retained_ids}"
    )

    assert suppressed_ids == {"B"}, (
        f"Unexpected suppressed candidates: {suppressed_ids}"
    )

    suppressed_b = next(
        item for item in suppressed
        if item["candidate_id"] == "B"
    )

    assert suppressed_b["duplicate_of_candidate_id"] == "A"
    assert suppressed_b["deduplication_status"] == "suppressed"

    print("Overlapping candidate suppression: PASS")
    print("Higher-scoring candidate retained: PASS")
    print("Non-overlapping candidate retained: PASS")
    print("Cross-episode candidates kept separate: PASS")

    ranked, rejected = rank_evaluated_candidates(retained)

    ranked_ids = [
        item["candidate_id"]
        for item in ranked
    ]

    assert ranked_ids == ["A", "C", "D"], (
        f"Unexpected ranking order: {ranked_ids}"
    )

    assert [
        item["final_rank"]
        for item in ranked
    ] == [1, 2, 3]

    assert rejected == []

    print("Final score-based ranking: PASS")


test_deduplication_and_ranking()


# ------------------------------------------------------------
# LOAD A REAL PARSED EPISODE
# ------------------------------------------------------------

DATA_DIR = Path("data/processed")
episode_files = sorted(DATA_DIR.glob("*.json"))

if not episode_files:
    raise RuntimeError("No processed episode JSON files found.")

episode = json.loads(
    episode_files[0].read_text(encoding="utf-8")
)

segments = episode["segments"]

candidate_segments = []
start_seconds = None

for segment in segments:
    if start_seconds is None:
        start_seconds = float(segment["start_seconds"])

    candidate_segments.append(segment)

    duration = (
        float(candidate_segments[-1]["end_seconds"])
        - start_seconds
    )

    if duration >= 20:
        break

duration = (
    float(candidate_segments[-1]["end_seconds"])
    - float(candidate_segments[0]["start_seconds"])
)

if not 20 <= duration <= 60:
    raise RuntimeError(
        f"Invalid sample candidate duration: {duration}"
    )

candidate = {
    "candidate_id": "PIPELINE_TEST_01",
    "episode": episode["episode"],
    "segment_ids": [
        segment["segment_id"]
        for segment in candidate_segments
    ],
    "start_segment_id": candidate_segments[0]["segment_id"],
    "end_segment_id": candidate_segments[-1]["segment_id"],
    "start_seconds": float(candidate_segments[0]["start_seconds"]),
    "end_seconds": float(candidate_segments[-1]["end_seconds"]),
    "start_timestamp": candidate_segments[0]["start"],
    "end_timestamp": candidate_segments[-1]["end"],
    "duration_seconds": duration,
    "candidate_confidence": 0.90,
    "hook": "A compelling opening hook.",
    "cliffhanger": "An unresolved ending.",
    "standalone_clarity": "The situation is understandable.",
    "narrative_context": "Test narrative context.",
    "promotional_strength": "Strong promotional potential.",
}


# ------------------------------------------------------------
# MOCK SEMANTIC ANALYZER
# ------------------------------------------------------------

class MockSemanticAnalyzer:

    def __init__(self, cfg):
        self.is_available = True

        self.stats = {
            "episodes_analyzed": 1,
            "episodes_failed": 0,
            "episodes_cached": 0,
            "api_calls": 1,
            "successful_calls": 1,
            "failed_calls": 0,

            "boundary_candidates_received": 1,
            "boundary_candidates_optimized": 1,
            "boundary_candidates_failed": 0,
            "boundary_candidates_cached": 0,
            "boundary_api_calls": 1,
            "boundary_successful_calls": 1,
            "boundary_failed_calls": 0,

            "evaluation_candidates_received": 1,
            "evaluation_candidates_evaluated": 1,
            "evaluation_candidates_cached": 0,
            "evaluation_candidates_failed": 0,
            "evaluation_api_calls": 1,
            "evaluation_successful_calls": 1,
            "evaluation_failed_calls": 0,
            "evaluation_accepted": 1,
            "evaluation_review": 0,
            "evaluation_rejected": 0,
        }

    def analyze_episodes(
        self,
        episodes,
        max_candidates_per_episode=5,
    ):
        return [
            {
                "episode": episode["episode"],
                "analysis_status": "success",
                "candidate_moments": [candidate],
            }
        ]

    def optimize_boundaries(self, episodes, candidates):
        optimized = []

        for item in candidates:
            result = dict(item)
            result["boundary_status"] = "success"
            optimized.append(result)

        return optimized

    def evaluate_promos(self, episodes, candidates):
        evaluated = []

        for item in candidates:
            result = dict(item)

            result.update({
                "evaluation_status": "success",
                "evaluation_scores": {
                    "hook_first_3_seconds": 0.90,
                    "cliffhanger": 0.85,
                    "standalone_clarity": 0.85,
                    "emotional_impact": 0.80,
                    "promo_suitability": 0.90,
                },
                "overall_score": 0.86,
                "recommendation": "accept",
                "primary_weakness": "",
                "evaluation_weights": {
                    "hook_first_3_seconds": 0.20,
                    "cliffhanger": 0.25,
                    "standalone_clarity": 0.20,
                    "emotional_impact": 0.15,
                    "promo_suitability": 0.20,
                },
            })

            evaluated.append(result)

        return evaluated


# ------------------------------------------------------------
# RUN PIPELINE IN AN ISOLATED OUTPUT DIRECTORY
# ------------------------------------------------------------

original_cwd = Path.cwd()
temp_dir = Path(tempfile.mkdtemp(prefix="kuku_pipeline_test_"))

try:
    os.chdir(temp_dir)

    with patch(
        "main.load_episodes",
        return_value=[episode],
    ), patch(
        "main.SemanticAnalyzer",
        MockSemanticAnalyzer,
    ):
        stats = run(
            cfg=object(),
            input_dir="mock_input",
            max_candidates_per_episode=5,
            shortlist_size=30,
        )

    output_root = Path("outputs")

    expected_files = [
        "episode_analysis/episode_analysis.json",
        "candidates/discovered_candidates.json",
        "candidates/shortlist_pre_boundary.json",
        "candidates/shortlist.json",
        "ranked_promos/ranked_promos.json",
        "ranked_promos/accepted_promos.json",
        "ranked_promos/review_promos.json",
        "rejected_candidates/rejected_promos.json",
        "reports/discovery_stats.json",
    ]

    for relative_path in expected_files:
        file_path = output_root / relative_path

        assert file_path.exists(), (
            f"Missing output file: {relative_path}"
        )

        json.loads(
            file_path.read_text(encoding="utf-8")
        )

    print("Pipeline execution: PASS")
    print("Output file generation: PASS")
    print("Output JSON validity: PASS")

    # Candidate flow
    discovered = json.loads(
        (output_root / "candidates/discovered_candidates.json")
        .read_text(encoding="utf-8")
    )

    ranked = json.loads(
        (output_root / "ranked_promos/ranked_promos.json")
        .read_text(encoding="utf-8")
    )

    accepted = json.loads(
        (output_root / "ranked_promos/accepted_promos.json")
        .read_text(encoding="utf-8")
    )

    rejected = json.loads(
        (output_root / "rejected_candidates/rejected_promos.json")
        .read_text(encoding="utf-8")
    )

    assert len(discovered) == 1
    assert len(ranked) == 1
    assert len(accepted) == 1
    assert len(rejected) == 0

    assert ranked[0]["candidate_id"] == "PIPELINE_TEST_01"
    assert ranked[0]["final_rank"] == 1
    assert ranked[0]["recommendation"] == "accept"

    print("Candidate flow across stages: PASS")
    print("Final ranking: PASS")
    print("Accepted promo output: PASS")

    # Statistics
    assert stats["episodes_loaded"] == 1
    assert stats["episodes_analyzed"] == 1
    assert stats["total_discovered_candidates"] == 1
    assert stats["shortlisted_candidates"] == 1
    assert stats["accepted_promos"] == 1
    assert stats["evaluation_api_calls"] == 1
    assert stats["duplicate_promos_suppressed"] == 0

    saved_stats = json.loads(
        (output_root / "reports/discovery_stats.json")
        .read_text(encoding="utf-8")
    )

    assert saved_stats["accepted_promos"] == 1
    assert saved_stats["evaluation_api_calls"] == 1
    assert saved_stats["duplicate_promos_suppressed"] == 0

    print("Execution statistics: PASS")
    print("Statistics file consistency: PASS")

finally:
    os.chdir(original_cwd)


print("=" * 55)
print("MOCKED END-TO-END PIPELINE TEST: PASS")
print("=" * 55)
