
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock

from src.llm.semantic_analyzer import SemanticAnalyzer


DATA_DIR = Path("data/processed")
episode_files = sorted(DATA_DIR.glob("*.json"))

if not episode_files:
    raise RuntimeError("No processed episode JSON files found.")

episode = json.loads(
    episode_files[0].read_text(encoding="utf-8")
)

segments = episode["segments"]

# Build a contiguous candidate between 20 and 60 seconds.
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
    raise RuntimeError(f"Invalid sample duration: {duration}")

candidate = {
    "candidate_id": "INTEGRATION_TEST_01",
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
    "boundary_status": "success",
    "hook": "Test hook",
    "cliffhanger": "Test cliffhanger",
    "standalone_clarity": "Test clarity",
    "narrative_context": "Test context",
    "promotional_strength": "Test strength",
}

# A candidate without successful boundary optimization must be skipped.
failed_boundary_candidate = {
    **candidate,
    "candidate_id": "INTEGRATION_TEST_FAILED_BOUNDARY",
    "boundary_status": "failed",
}

first_id = candidate_segments[0]["segment_id"]
last_id = candidate_segments[-1]["segment_id"]

mock_response = {
    "candidate_id": candidate["candidate_id"],
    "assessments": {
        "hook_first_3_seconds": {
            "score": 0.85,
            "reasoning": "The opening creates curiosity.",
            "evidence_segment_ids": [first_id],
        },
        "cliffhanger": {
            "score": 0.80,
            "reasoning": "The ending leaves an unresolved question.",
            "evidence_segment_ids": [last_id],
        },
        "standalone_clarity": {
            "score": 0.75,
            "reasoning": "The situation is understandable.",
            "evidence_segment_ids": [first_id],
        },
        "emotional_impact": {
            "score": 0.80,
            "reasoning": "The moment has emotional stakes.",
            "evidence_segment_ids": [first_id],
        },
        "promo_suitability": {
            "score": 0.85,
            "reasoning": "The excerpt has a coherent progression.",
            "evidence_segment_ids": [last_id],
        },
    },
    "major_weakness": "",
}


class MockLLMClient:
    provider = "mock"
    model = "mock-evaluation-model"

    def __init__(self):
        self.call_count = 0

    def call_json(self, prompt):
        self.call_count += 1
        return mock_response


mock_client = MockLLMClient()
initialized_stages = []


def fake_from_config(cfg, stage=None):
    initialized_stages.append(stage)

    if stage == "episode_analysis":
        return Mock()

    if stage == "promo_evaluation":
        return mock_client

    raise AssertionError(f"Unexpected LLM stage: {stage}")


# A minimal config is sufficient because the LLM factory is mocked.
cfg = SimpleNamespace(llm_enabled=True)

with tempfile.TemporaryDirectory() as temp_dir:
    temp_path = Path(temp_dir)

    with patch(
        "src.llm.semantic_analyzer.LLMClient.from_config",
        side_effect=fake_from_config,
    ):
        analyzer = SemanticAnalyzer(
            cfg,
            cache_dir=temp_path / "episode",
            boundary_cache_dir=temp_path / "boundary",
            evaluation_cache_dir=temp_path / "evaluation",
        )

        assert analyzer.is_available
        assert "episode_analysis" in initialized_stages

        # First evaluation: one actual mocked call.
        first_results = analyzer.evaluate_promos(
            episodes=[episode],
            candidates=[
                candidate,
                failed_boundary_candidate,
            ],
        )

        assert len(first_results) == 2

        evaluated = first_results[0]
        skipped = first_results[1]

        assert evaluated["evaluation_status"] == "success"
        assert evaluated["recommendation"] == "accept"
        assert 0 <= evaluated["overall_score"] <= 1

        assert skipped["evaluation_status"] == "skipped"
        assert skipped["recommendation"] == "review"

        assert mock_client.call_count == 1
        assert analyzer.stats["evaluation_api_calls"] == 1
        assert analyzer.stats["evaluation_successful_calls"] == 1
        assert analyzer.stats["evaluation_accepted"] == 1

        print("Evaluation integration: PASS")
        print("Overall score:", evaluated["overall_score"])
        print("Recommendation:", evaluated["recommendation"])
        print("Failed boundary candidate skipped: PASS")

        # Second evaluation: same candidate should be served from cache.
        second_results = analyzer.evaluate_promos(
            episodes=[episode],
            candidates=[candidate],
        )

        assert second_results[0]["evaluation_status"] == "success"
        assert second_results[0]["recommendation"] == "accept"

        # No additional LLM call should have occurred.
        assert mock_client.call_count == 1
        assert analyzer.stats["evaluation_api_calls"] == 1
        assert analyzer.stats["evaluation_candidates_cached"] == 1

        print("Evaluation cache reuse: PASS")
        print("Mock API calls:", mock_client.call_count)

        assert "promo_evaluation" in initialized_stages

        print("Separate evaluation client initialization: PASS")
        print("MOCKED SEMANTIC ANALYZER EVALUATION: PASS")
