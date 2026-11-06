
import json
from pathlib import Path

from src.llm.prompts import (
    build_promo_evaluation_prompt,
    validate_promo_evaluation_response,
)


# ------------------------------------------------------------
# LOAD A REAL EPISODE AND BUILD A VALID SAMPLE CANDIDATE
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
        f"Invalid sample duration: {duration}"
    )

candidate = {
    "candidate_id": "UNIT_TEST_01",
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

first_id = candidate_segments[0]["segment_id"]
last_id = candidate_segments[-1]["segment_id"]


# ------------------------------------------------------------
# MOCK EVALUATION RESPONSE
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# TEST 1: PROMPT CONSTRUCTION
# ------------------------------------------------------------

prompt = build_promo_evaluation_prompt(
    episode=episode,
    candidate=candidate,
)

assert isinstance(prompt, str)
assert len(prompt) > 0
assert candidate["candidate_id"] in prompt
assert first_id in prompt
assert last_id in prompt

print("Prompt construction: PASS")
print("Prompt length:", len(prompt))


# ------------------------------------------------------------
# TEST 2: VALID RESPONSE AND WEIGHTED SCORE
# ------------------------------------------------------------

validated = validate_promo_evaluation_response(
    data=mock_response,
    candidate=candidate,
    episode=episode,
)

assert validated["candidate_id"] == candidate["candidate_id"]
assert validated["recommendation"] == "accept"
assert 0 <= validated["overall_score"] <= 1
assert len(validated["assessments"]) == 5

# Independently verify the weighted score.
weights = validated["evaluation_weights"]
scores = validated["dimension_scores"]

expected_score = sum(
    weights[key] * scores[key]
    for key in weights
)

assert abs(
    validated["overall_score"] - expected_score
) < 1e-9

print("Valid response validation: PASS")
print("Weighted score calculation: PASS")
print("Overall score:", round(validated["overall_score"], 4))
print("Recommendation:", validated["recommendation"])



# ------------------------------------------------------------
# TEST 3: INVALID EVIDENCE ID MUST BE REJECTED
# ------------------------------------------------------------

invalid_evidence_response = json.loads(
    json.dumps(mock_response)
)

invalid_evidence_response["assessments"]["cliffhanger"][
    "evidence_segment_ids"
] = ["NON_EXISTENT_SEGMENT"]

result = validate_promo_evaluation_response(
    data=invalid_evidence_response,
    candidate=candidate,
    episode=episode,
)

assert result is None
print("Invalid evidence rejection: PASS")


# ------------------------------------------------------------
# TEST 4: HOOK EVIDENCE MUST COME FROM FIRST THREE SECONDS
# ------------------------------------------------------------

invalid_hook_response = json.loads(
    json.dumps(mock_response)
)

invalid_hook_response["assessments"]["hook_first_3_seconds"][
    "evidence_segment_ids"
] = [last_id]

result = validate_promo_evaluation_response(
    data=invalid_hook_response,
    candidate=candidate,
    episode=episode,
)

assert result is None
print("Invalid hook evidence rejection: PASS")


print("STANDALONE PROMO EVALUATION UNIT TESTS: PASS")
