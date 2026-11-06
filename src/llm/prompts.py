
"""
src/llm/prompts.py

Episode-level story understanding and promotional candidate discovery.

New architecture:
1. One LLM call per episode.
2. Pass a compact running story summary from previous episodes.
3. Identify narrative developments and potential promotional moments.
4. Return exact source segment IDs and evidence.
5. Validate the response against the original episode segments.

No show-specific emotional keyword dictionaries are used.
"""

from __future__ import annotations

import json
import math
from typing import Any

from src.utils import seconds_to_timestamp


# ============================================================
# EPISODE-LEVEL ANALYSIS PROMPT
# ============================================================

_EPISODE_ANALYSIS_TEMPLATE = """\
You are an expert narrative analyst helping identify promotional clips \
from short-form drama screenplays.

Your task is to understand this episode in the context of the story so far \
and identify moments that could work as short promotional videos.

IMPORTANT:
- This is a new show-independent task. Do not assume a particular genre.
- Do not rely on predefined emotional keywords or genre-specific rules.
- Interpret the actual narrative, characters, relationships and events.
- Do not invent events, dialogue, character motivations or consequences.
- Every proposed moment must be grounded in the supplied source segments.
- Preserve the exact source segment IDs.
- Prefer moments that can interest someone who has never watched the show.
- A moment does not need to be dramatic or loud to be promotional.
- Do not force a candidate if the episode contains no suitable moment.

## Episode information

Episode: {episode_number}

Episode logline:
{logline}

Continuity notes:
{continuity_notes}

## Story summary from previous episodes

Use this only as background context. Do not repeat it unnecessarily.

{running_summary}

## Complete episode screenplay

Each source segment has an exact ID and timestamp.
The screenplay is in chronological order.

{episode_text}

## Your tasks

### Task 1: Understand the episode

Identify the important developments in this episode.

Summarise:
- What happens in this episode.
- Important changes in character relationships.
- New information or revelations.
- Unresolved conflicts or questions.
- How this episode changes the broader story.

Only report developments supported by the screenplay.

### Task 2: Update the running story understanding

Provide concise updates that should be carried forward to the next episode.

Focus on:
- Character identities and relationships.
- Important events and revelations.
- Changes in motivations or alliances.
- Unresolved storylines.

Do not reproduce the entire episode summary.

### Task 3: Discover potential promotional moments

Identify up to {max_candidates} promising moments.

For each candidate:
- Select a continuous stretch of source segments.
- The total duration must be between 20 and 60 seconds.
- The opening should have the potential to capture attention within its first 3 seconds.
- The ending should ideally leave an unresolved question, reveal, threat or emotional uncertainty.
- The moment should be understandable to a new viewer, or have enough context to become understandable with minimal editing.
- Prefer moments with a clear narrative progression rather than disconnected dialogue.
- Include exact source segment IDs and supporting evidence.
- Do not invent or paraphrase dialogue as if it were a direct quote.

Do not select moments simply because they contain emotional language.
Evaluate what is actually happening in the story.

## Required output

Return ONLY valid JSON. Do not include markdown fences or explanatory text.

Use this exact schema:

{{
  "episode_number": "{episode_number}",
  "episode_summary": "<concise summary of the episode>",
  "story_updates": [
    "<important story fact to carry forward>"
  ],
  "unresolved_storylines": [
    "<important unresolved question or conflict>"
  ],
  "candidate_moments": [
    {{
      "candidate_id": "EP{episode_number}_C01",
      "start_segment_id": "<exact source segment ID>",
      "end_segment_id": "<exact source segment ID>",
      "segment_ids": [
        "<all included source segment IDs in chronological order>"
      ],
      "hook": "<what happens in the opening and why it may attract attention>",
      "cliffhanger": "<what remains unresolved at the end>",
      "standalone_clarity": "<what a new viewer can understand and what context is missing>",
      "narrative_context": "<brief explanation of the moment in the story>",
      "promotional_strength": "<why this moment may attract a new viewer>",
      "evidence": [
        {{
          "segment_id": "<exact source segment ID>",
          "timestamp": "<source timestamp>",
          "observation": "<what this segment establishes>"
        }}
      ],
      "candidate_confidence": <float from 0.0 to 1.0>
    }}
  ]
}}

If no suitable promotional moment exists, return an empty candidate_moments array.

Keep summaries and explanations concise.
"""


# ============================================================
# SOURCE SEGMENT NORMALIZATION
# ============================================================

def _get_episode_segments(episode: dict) -> list[dict]:
    """
    Retrieve the normalized source segments from an episode.

    Supports common container key names. The actual segment objects
    must contain a source ID, kind and start timestamp.
    """

    if not isinstance(episode, dict):
        raise ValueError("Episode must be a dictionary.")

    for key in ("segments", "all_segments", "source_segments", "blocks"):
        value = episode.get(key)

        if isinstance(value, list):
            return value

    raise ValueError(
        "Could not find episode segments. Expected one of: "
        "'segments', 'all_segments', 'source_segments', or 'blocks'."
    )


def _get_segment_id(segment: dict) -> str | None:
    """Return the source ID of a segment."""

    for key in ("segment_id", "id", "source_id"):
        value = segment.get(key)

        if value is not None:
            return str(value)

    return None


def _format_episode_segment(segment: dict) -> str:
    """Format one source segment while preserving its original ID."""

    segment_id = _get_segment_id(segment)

    if not segment_id:
        raise ValueError("A source segment is missing its ID.")

    kind = str(segment.get("kind", "unknown")).lower()
    start = float(segment.get("start_seconds", 0.0))
    end = float(segment.get("end_seconds", start))

    timestamp = seconds_to_timestamp(start)
    end_timestamp = seconds_to_timestamp(end)

    text = str(segment.get("text") or "").strip()
    speaker = str(segment.get("speaker") or "UNKNOWN").strip()
    delivery = str(segment.get("delivery") or "").strip()

    if kind == "dialogue":
        delivery_text = f" ({delivery})" if delivery else ""
        content = f'{speaker.upper()}{delivery_text}: "{text}"'

    elif kind == "action":
        content = f"ACTION: {text}"

    elif kind in ("sound_fx", "sfx"):
        content = f"SOUND EFFECT: {text}"

    else:
        content = f"{kind.upper()}: {text}"

    return (
        f"[ID: {segment_id}] "
        f"[{timestamp}-{end_timestamp}] "
        f"{content}"
    )


# ============================================================
# EPISODE PROMPT BUILDER
# ============================================================

def build_episode_analysis_prompt(
    episode: dict,
    running_summary: str = "",
    episode_number: int | str | None = None,
    max_candidates: int = 5,
) -> str:
    """
    Build one complete episode-level analysis prompt.

    The full episode is passed once, along with a compact summary
    of previous episodes.

    Returns a prompt requesting:
    - Episode understanding
    - Story updates
    - Unresolved storylines
    - Promotional candidate moments
    """

    segments = _get_episode_segments(episode)

    if not segments:
        raise ValueError("Episode contains no source segments.")

    formatted_segments = [
        _format_episode_segment(segment)
        for segment in segments
    ]

    episode_text = "\n".join(formatted_segments)

    number = (
        episode_number
        if episode_number is not None
        else episode.get("episode_number", episode.get("episode", "UNKNOWN"))
    )

    logline = str(episode.get("logline") or "Not provided.")
    continuity_notes = str(
        episode.get("continuity_notes") or "Not provided."
    )

    if not running_summary.strip():
        running_summary = "This is the first episode. No previous story context."

    if max_candidates < 1:
        raise ValueError("max_candidates must be at least 1.")

    return _EPISODE_ANALYSIS_TEMPLATE.format(
        episode_number=number,
        logline=logline,
        continuity_notes=continuity_notes,
        running_summary=running_summary.strip(),
        episode_text=episode_text,
        max_candidates=max_candidates,
    )


# ============================================================
# RESPONSE VALIDATION
# ============================================================

def _is_valid_score(value: Any) -> bool:
    """Check that a value is a finite numeric score in [0, 1]."""

    if isinstance(value, bool):
        return False

    try:
        number = float(value)
    except (TypeError, ValueError):
        return False

    return math.isfinite(number) and 0.0 <= number <= 1.0


def validate_episode_analysis_response(
    data: dict,
    episode: dict,
) -> dict | None:
    """
    Validate an episode-level LLM response.

    Checks:
    - Required response fields.
    - Candidate segment IDs exist in the original episode.
    - Candidate segment IDs are chronological and contiguous.
    - Candidate duration is between 20 and 60 seconds.
    - Evidence references valid source segments.
    - Confidence scores are valid.

    Invalid candidates are discarded individually.
    A malformed episode-level response returns None.
    """

    if not isinstance(data, dict) or not isinstance(episode, dict):
        return None

    required_fields = (
        "episode_summary",
        "story_updates",
        "unresolved_storylines",
        "candidate_moments",
    )

    if any(field not in data for field in required_fields):
        return None

    if not isinstance(data["episode_summary"], str):
        return None

    if not isinstance(data["story_updates"], list):
        return None

    if not isinstance(data["unresolved_storylines"], list):
        return None

    if not isinstance(data["candidate_moments"], list):
        return None

    try:
        source_segments = _get_episode_segments(episode)
    except ValueError:
        return None

    segment_map = {}
    segment_positions = {}

    for position, segment in enumerate(source_segments):
        segment_id = _get_segment_id(segment)

        if not segment_id:
            continue

        segment_map[segment_id] = segment
        segment_positions[segment_id] = position

    cleaned_candidates = []

    for candidate in data["candidate_moments"]:
        if not isinstance(candidate, dict):
            continue

        required_candidate_fields = (
            "candidate_id",
            "start_segment_id",
            "end_segment_id",
            "segment_ids",
            "hook",
            "cliffhanger",
            "standalone_clarity",
            "narrative_context",
            "promotional_strength",
            "evidence",
            "candidate_confidence",
        )

        if any(field not in candidate for field in required_candidate_fields):
            continue

        segment_ids = candidate["segment_ids"]

        if not isinstance(segment_ids, list) or not segment_ids:
            continue

        segment_ids = [str(segment_id) for segment_id in segment_ids]

        if len(set(segment_ids)) != len(segment_ids):
            continue

        if any(segment_id not in segment_map for segment_id in segment_ids):
            continue

        if candidate["start_segment_id"] != segment_ids[0]:
            continue

        if candidate["end_segment_id"] != segment_ids[-1]:
            continue

        positions = [segment_positions[sid] for sid in segment_ids]

        # Require the selected segments to be consecutive in the
        # original screenplay. This prevents invented or disconnected clips.
        if positions != list(range(positions[0], positions[-1] + 1)):
            continue

        first_segment = segment_map[segment_ids[0]]
        last_segment = segment_map[segment_ids[-1]]

        try:
            start = float(first_segment["start_seconds"])
            end = float(last_segment["end_seconds"])
        except (KeyError, TypeError, ValueError):
            continue

        duration = end - start

        if not math.isfinite(duration) or not 20.0 <= duration <= 60.0:
            continue

        confidence = candidate["candidate_confidence"]

        if not _is_valid_score(confidence):
            continue

        text_fields = (
            "hook",
            "cliffhanger",
            "standalone_clarity",
            "narrative_context",
            "promotional_strength",
        )

        if any(
            not isinstance(candidate[field], str) or not candidate[field].strip()
            for field in text_fields
        ):
            continue

        evidence = candidate["evidence"]

        if not isinstance(evidence, list):
            continue

        valid_evidence = []

        for item in evidence:
            if not isinstance(item, dict):
                continue

            evidence_id = str(item.get("segment_id", ""))

            if evidence_id not in segment_map:
                continue

            if evidence_id not in segment_ids:
                continue

            valid_evidence.append({
                "segment_id": evidence_id,
                "timestamp": str(item.get("timestamp", "")),
                "observation": str(item.get("observation", ""))[:300],
            })

        if not valid_evidence:
            continue

        cleaned_candidates.append({
            "candidate_id": str(candidate["candidate_id"]),
            "start_segment_id": segment_ids[0],
            "end_segment_id": segment_ids[-1],
            "segment_ids": segment_ids,
            "start_seconds": start,
            "end_seconds": end,
            "duration_seconds": duration,
            "hook": candidate["hook"].strip()[:500],
            "cliffhanger": candidate["cliffhanger"].strip()[:500],
            "standalone_clarity": candidate["standalone_clarity"].strip()[:500],
            "narrative_context": candidate["narrative_context"].strip()[:500],
            "promotional_strength": candidate["promotional_strength"].strip()[:500],
            "evidence": valid_evidence,
            "candidate_confidence": float(confidence),
        })

    return {
        "episode_number": str(data.get("episode_number", "")),
        "episode_summary": data["episode_summary"].strip()[:2000],
        "story_updates": [
            str(item).strip()[:500]
            for item in data["story_updates"]
            if isinstance(item, str) and item.strip()
        ],
        "unresolved_storylines": [
            str(item).strip()[:500]
            for item in data["unresolved_storylines"]
            if isinstance(item, str) and item.strip()
        ],
        "candidate_moments": cleaned_candidates,
    }


# ============================================================
# LEGACY WINDOW ANALYSIS COMPATIBILITY
# ============================================================
# Kept temporarily so the existing semantic_analyzer.py does not
# break before we update that file in the next implementation step.
# The new pipeline will use build_episode_analysis_prompt().

_WINDOW_ANALYSIS_TEMPLATE = """\
You are an expert promotional consultant evaluating a {duration:.0f}-second \
excerpt from an audio drama episode.

## Episode Context
{logline}

## Excerpt
{excerpt}

Return ONLY valid JSON:

{{
  "emotional_intensity": <float 0.0-1.0>,
  "conflict_score": <float 0.0-1.0>,
  "cliffhanger_score": <float 0.0-1.0>,
  "revelation_score": <float 0.0-1.0>,
  "hook_potential": <float 0.0-1.0>,
  "context_independence": <float 0.0-1.0>,
  "promotional_suitability": <float 0.0-1.0>,
  "reasoning": "<single sentence>"
}}
"""

_REQUIRED_FLOAT_KEYS = (
    "emotional_intensity",
    "conflict_score",
    "cliffhanger_score",
    "revelation_score",
    "hook_potential",
    "context_independence",
    "promotional_suitability",
)

_REQUIRED_STR_KEYS = ("reasoning",)


def build_analysis_prompt(window) -> str:
    """Legacy candidate-window prompt. Retained during migration."""

    relevant = [
        segment
        for segment in window.all_segments
        if segment.get("kind") in ("dialogue", "action", "sound_fx")
    ]

    relevant.sort(key=lambda segment: segment["start_seconds"])

    lines = []

    for segment in relevant:
        timestamp = seconds_to_timestamp(segment["start_seconds"])
        kind = segment.get("kind")
        text = str(segment.get("text") or "").strip()

        if kind == "dialogue":
            speaker = segment.get("speaker") or "UNKNOWN"
            delivery = segment.get("delivery") or ""
            delivery_part = f" ({delivery})" if delivery else ""
            lines.append(
                f'[{timestamp}] {speaker.upper()}{delivery_part}: "{text}"'
            )
        else:
            lines.append(f"[{timestamp}] *{kind.upper()}* {text}")

    return _WINDOW_ANALYSIS_TEMPLATE.format(
        duration=window.duration_seconds,
        logline=window.logline or "No logline available.",
        excerpt="\n".join(lines),
    )


def validate_llm_response(data: dict) -> dict | None:
    """Legacy response validator. Retained during migration."""

    if not isinstance(data, dict):
        return None

    cleaned = {}

    for key in _REQUIRED_FLOAT_KEYS:
        value = data.get(key)

        if not _is_valid_score(value):
            return None

        cleaned[key] = float(value)

    for key in _REQUIRED_STR_KEYS:
        value = data.get(key, "")

        if not isinstance(value, str) or not value.strip():
            return None

        cleaned[key] = value.strip()[:300]

    return cleaned

# ============================================================
# BOUNDARY OPTIMIZATION
# ============================================================

_BOUNDARY_OPTIMIZATION_TEMPLATE = """\
You are an expert promotional video editor evaluating alternative
start and end boundaries for a short-form drama promotional clip.

Your task is to SELECT the best boundary option from the options
generated by Python.

IMPORTANT RULES:
- You MUST select an option from the supplied list.
- Never invent timestamps, segment IDs or new boundaries.
- Do not combine different options.
- Every option is a continuous stretch of the original screenplay.
- The selected duration must remain between 20 and 60 seconds.
- Do not automatically prefer the longest or shortest option.
- Preserve enough context for a new viewer to understand the moment.
- Prefer an opening that creates curiosity or tension quickly.
- Prefer an ending that leaves a question, threat, revelation or
  emotional uncertainty unresolved.
- Avoid cutting dialogue in a way that destroys its meaning.
- Consider the complete narrative progression, not just individual
  dramatic lines.
- Do not change the actual story or invent events.

## Episode

{episode_name}

## Original candidate

Candidate ID: {candidate_id}

Original hook:
{hook}

Original cliffhanger:
{cliffhanger}

Standalone clarity:
{standalone_clarity}

Narrative context:
{narrative_context}

Promotional strength:
{promotional_strength}

## Original screenplay segments

The following segments are the complete original candidate.
Segment IDs and timestamps are authoritative.

{screenplay}

## Available boundary options

Each option contains:
- An exact option ID.
- Its start and end segment IDs.
- Its complete ordered segment ID list.
- Its exact duration.

{options}

## Evaluation criteria

Evaluate each option based on:

1. Opening hook:
   Does the opening create attention or curiosity quickly?

2. Narrative continuity:
   Does the selected stretch retain a coherent progression?

3. Standalone clarity:
   Can a new viewer understand enough without watching earlier episodes?

4. Ending strength:
   Does the ending leave an unresolved question or compelling
   reason to continue watching?

5. Clip completeness:
   Does the option avoid cutting dialogue or action at an awkward point?

Select the strongest overall promotional excerpt.

## Required output

Return ONLY valid JSON:

{{
  "selected_option_id": "<one exact option ID from the supplied list>",
  "reasoning": "<concise explanation of why this option has the strongest boundaries>",
  "confidence": <float from 0.0 to 1.0>
}}
"""


def generate_boundary_options(
    episode: dict,
    candidate: dict,
    max_trim_segments: int = 2,
    min_duration_seconds: float = 20.0,
    max_duration_seconds: float = 60.0,
) -> list[dict]:
    """
    Generate valid alternative boundaries for a candidate.

    The original candidate is retained as an option when valid.

    Alternatives are created by trimming zero, one or two source
    segments from either end. No new segments are introduced.

    Returns a list of option dictionaries containing exact source IDs,
    timestamps and durations.
    """

    if not isinstance(candidate, dict):
        return []

    if max_trim_segments < 0:
        raise ValueError("max_trim_segments cannot be negative.")

    if min_duration_seconds <= 0:
        raise ValueError("Minimum duration must be positive.")

    if max_duration_seconds < min_duration_seconds:
        raise ValueError("Maximum duration must be >= minimum duration.")

    try:
        source_segments = _get_episode_segments(episode)
    except ValueError:
        return []

    segment_map = {}
    segment_positions = {}

    for position, segment in enumerate(source_segments):
        segment_id = _get_segment_id(segment)

        if segment_id:
            segment_map[segment_id] = segment
            segment_positions[segment_id] = position

    original_ids = candidate.get("segment_ids")

    if not isinstance(original_ids, list) or not original_ids:
        return []

    original_ids = [str(item) for item in original_ids]

    if len(set(original_ids)) != len(original_ids):
        return []

    if any(item not in segment_map for item in original_ids):
        return []

    positions = [segment_positions[item] for item in original_ids]

    if positions != list(range(positions[0], positions[-1] + 1)):
        return []

    options = []
    seen_boundaries = set()

    option_number = 1

    # Generate all combinations of left and right trimming.
    for trim_left in range(max_trim_segments + 1):
        for trim_right in range(max_trim_segments + 1):

            if trim_left + trim_right >= len(original_ids):
                continue

            selected_ids = original_ids[
                trim_left:len(original_ids) - trim_right
                if trim_right > 0
                else len(original_ids)
            ]

            if not selected_ids:
                continue

            first_segment = segment_map[selected_ids[0]]
            last_segment = segment_map[selected_ids[-1]]

            try:
                start = float(first_segment["start_seconds"])
                end = float(last_segment["end_seconds"])
            except (KeyError, TypeError, ValueError):
                continue

            duration = end - start

            if not math.isfinite(duration):
                continue

            if not min_duration_seconds <= duration <= max_duration_seconds:
                continue

            boundary_key = (
                selected_ids[0],
                selected_ids[-1],
            )

            if boundary_key in seen_boundaries:
                continue

            seen_boundaries.add(boundary_key)

            options.append({
                "option_id": f"B{option_number:02d}",
                "start_segment_id": selected_ids[0],
                "end_segment_id": selected_ids[-1],
                "segment_ids": selected_ids,
                "start_seconds": start,
                "end_seconds": end,
                "start_timestamp": str(first_segment.get("start", "")),
                "end_timestamp": str(last_segment.get("end", "")),
                "duration_seconds": round(duration, 3),
                "trimmed_start_segments": trim_left,
                "trimmed_end_segments": trim_right,
            })

            option_number += 1

    return options


def build_boundary_optimization_prompt(
    episode: dict,
    candidate: dict,
    options: list[dict],
) -> str:
    """
    Build a boundary optimization prompt.

    The LLM receives the original candidate screenplay and a set of
    Python-generated boundary options. It must select one option.
    """

    if not isinstance(candidate, dict):
        raise ValueError("Candidate must be a dictionary.")

    if not isinstance(options, list) or not options:
        raise ValueError("At least one valid boundary option is required.")

    segments = _get_episode_segments(episode)

    segment_map = {}

    for segment in segments:
        segment_id = _get_segment_id(segment)

        if segment_id:
            segment_map[segment_id] = segment

    original_ids = candidate.get("segment_ids")

    if not isinstance(original_ids, list) or not original_ids:
        raise ValueError("Candidate has no valid segment_ids.")

    original_ids = [str(item) for item in original_ids]

    if any(item not in segment_map for item in original_ids):
        raise ValueError("Candidate references unknown source segments.")

    screenplay_lines = [
        _format_episode_segment(segment_map[segment_id])
        for segment_id in original_ids
    ]

    option_lines = []

    for option in options:
        option_lines.append(
            json.dumps(
                {
                    "option_id": option["option_id"],
                    "start_segment_id": option["start_segment_id"],
                    "end_segment_id": option["end_segment_id"],
                    "segment_ids": option["segment_ids"],
                    "start_timestamp": option["start_timestamp"],
                    "end_timestamp": option["end_timestamp"],
                    "duration_seconds": option["duration_seconds"],
                    "trimmed_start_segments": option[
                        "trimmed_start_segments"
                    ],
                    "trimmed_end_segments": option[
                        "trimmed_end_segments"
                    ],
                },
                ensure_ascii=False,
            )
        )

    episode_name = str(
        episode.get("episode", episode.get("episode_number", "UNKNOWN"))
    )

    return _BOUNDARY_OPTIMIZATION_TEMPLATE.format(
        episode_name=episode_name,
        candidate_id=str(candidate.get("candidate_id", "UNKNOWN")),
        hook=str(candidate.get("hook", "Not provided.")),
        cliffhanger=str(candidate.get("cliffhanger", "Not provided.")),
        standalone_clarity=str(
            candidate.get("standalone_clarity", "Not provided.")
        ),
        narrative_context=str(
            candidate.get("narrative_context", "Not provided.")
        ),
        promotional_strength=str(
            candidate.get("promotional_strength", "Not provided.")
        ),
        screenplay="\n".join(screenplay_lines),
        options="\n".join(option_lines),
    )


def validate_boundary_optimization_response(
    data: dict,
    options: list[dict],
    episode: dict,
) -> dict | None:
    """
    Validate the LLM's selected boundary option.

    The LLM can only select an option generated by Python.

    The selected option is independently checked against the original
    screenplay for:
    - Valid source segment IDs.
    - Chronological continuity.
    - Correct start and end segments.
    - Exact source timestamps.
    - Duration between 20 and 60 seconds.

    Returns the selected option and LLM reasoning, or None.
    """

    if not isinstance(data, dict):
        return None

    if not isinstance(options, list) or not options:
        return None

    selected_id = data.get("selected_option_id")

    if not isinstance(selected_id, str):
        return None

    selected_option = next(
        (
            option for option in options
            if option.get("option_id") == selected_id
        ),
        None,
    )

    if selected_option is None:
        return None

    reasoning = data.get("reasoning")
    confidence = data.get("confidence")

    if not isinstance(reasoning, str) or not reasoning.strip():
        return None

    if not _is_valid_score(confidence):
        return None

    try:
        source_segments = _get_episode_segments(episode)
    except ValueError:
        return None

    segment_map = {}
    segment_positions = {}

    for position, segment in enumerate(source_segments):
        segment_id = _get_segment_id(segment)

        if segment_id:
            segment_map[segment_id] = segment
            segment_positions[segment_id] = position

    segment_ids = selected_option.get("segment_ids")

    if not isinstance(segment_ids, list) or not segment_ids:
        return None

    segment_ids = [str(item) for item in segment_ids]

    if len(set(segment_ids)) != len(segment_ids):
        return None

    if any(item not in segment_map for item in segment_ids):
        return None

    if selected_option.get("start_segment_id") != segment_ids[0]:
        return None

    if selected_option.get("end_segment_id") != segment_ids[-1]:
        return None

    positions = [segment_positions[item] for item in segment_ids]

    if positions != list(range(positions[0], positions[-1] + 1)):
        return None

    first_segment = segment_map[segment_ids[0]]
    last_segment = segment_map[segment_ids[-1]]

    try:
        start = float(first_segment["start_seconds"])
        end = float(last_segment["end_seconds"])
    except (KeyError, TypeError, ValueError):
        return None

    duration = end - start

    if not math.isfinite(duration) or not 20.0 <= duration <= 60.0:
        return None

    # Verify that the option's stored values still match the source.
    try:
        option_start = float(selected_option["start_seconds"])
        option_end = float(selected_option["end_seconds"])
        option_duration = float(selected_option["duration_seconds"])
    except (KeyError, TypeError, ValueError):
        return None

    if not (
        math.isclose(start, option_start, abs_tol=0.001)
        and math.isclose(end, option_end, abs_tol=0.001)
        and math.isclose(duration, option_duration, abs_tol=0.001)
    ):
        return None

    return {
        "option_id": selected_id,
        "start_segment_id": segment_ids[0],
        "end_segment_id": segment_ids[-1],
        "segment_ids": segment_ids,
        "start_seconds": start,
        "end_seconds": end,
        "start_timestamp": str(first_segment.get("start", "")),
        "end_timestamp": str(last_segment.get("end", "")),
        "duration_seconds": round(duration, 3),
        "reasoning": reasoning.strip()[:1000],
        "confidence": float(confidence),
    }


# ============================================================
# FINAL PROMO EVALUATION
# ============================================================

_PROMO_EVALUATION_TEMPLATE = """\
You are an expert short-form drama promotional editor.

Your task is to evaluate whether a proposed screenplay excerpt would
work as an effective promotional video for a microdrama.

Evaluate the actual content of the supplied excerpt. Do not assume
that a candidate is good merely because it was discovered by an
earlier stage of the pipeline.

IMPORTANT RULES:
- Evaluate only what is supported by the supplied screenplay.
- Do not invent dialogue, actions, motivations or consequences.
- Do not reward dramatic language without meaningful narrative impact.
- Do not assume that a new viewer knows the characters or story.
- A visually compelling moment can be effective even without dialogue.
- Distinguish visual evidence from dialogue and sound evidence.
- Do not assume that music, captions, editing or external footage
  will improve the clip. The assignment evaluates the source moments.
- Be critical. Weak candidates should receive low scores.
- Every assessment must cite exact source segment IDs.

## Episode and candidate

Episode: {episode_name}

Candidate ID: {candidate_id}

Clip start: {start_timestamp}
Clip end: {end_timestamp}
Clip duration: {duration_seconds} seconds

Original candidate context:

Hook:
{hook}

Cliffhanger:
{cliffhanger}

Standalone clarity:
{standalone_clarity}

Narrative context:
{narrative_context}

Promotional strength:
{promotional_strength}

## Complete optimized screenplay excerpt

The source segments below are in chronological order.
Their IDs and timestamps are authoritative.

{screenplay}

## Evaluation dimensions

Score each dimension from 0.0 to 1.0.

### 1. Hook within the first 3 seconds

Evaluate ONLY what happens within the first three seconds of the clip.

Consider:
- Does something immediately create curiosity, tension, surprise
  or an emotional reaction?
- Is there a compelling visual or narrative development?
- Would someone scrolling through social media have a reason
  to stop?

A dramatic event occurring later in the clip must not improve
this score.

### 2. Cliffhanger and ending strength

Evaluate the ending of the complete excerpt.

Consider:
- Does the clip end at a compelling unresolved moment?
- Is there a question, threat, revelation or emotional uncertainty?
- Would a viewer want to know what happens next?

A clip that simply ends after completing an event should not
automatically receive a high score.

### 3. Standalone clarity

Evaluate whether someone who has never watched the show can
understand the essential situation.

Consider:
- Are the important people and their immediate situation
  understandable?
- Is there enough context to understand the conflict or event?
- Does the clip depend too heavily on knowledge from earlier episodes?

Do not require the viewer to understand the entire story.

### 4. Emotional impact

Evaluate the emotional strength of the complete excerpt.

Consider:
- Does the moment create a meaningful emotional response?
- Are the stakes, conflict or character reactions compelling?
- Does the emotional progression feel earned and understandable?

Do not confuse shouting, insults or emotional vocabulary with
genuine emotional impact.

### 5. Overall promotional suitability

Evaluate whether the excerpt would make an effective short-form
advertisement for the show.

Consider:
- Is there a compelling narrative progression?
- Does the excerpt create interest in the wider story?
- Are its opening, progression and ending effective together?
- Is the content distinctive and engaging enough to promote?

Consider all five dimensions, not just dramatic intensity.

## Evidence requirements

For every dimension:
- Provide a concise explanation.
- Provide one or more exact source segment IDs supporting the score.
- Cite only segments that actually support the assessment.
- Do not invent segment IDs.
- The hook evidence must come from segments overlapping the
  first three seconds of the clip.

Be honest about weaknesses. Do not inflate scores to be encouraging.

## Required output

Return ONLY valid JSON. Do not include markdown fences.

Use this exact schema:

{{
  "candidate_id": "{candidate_id}",
  "assessments": {{
    "hook_first_3_seconds": {{
      "score": <float from 0.0 to 1.0>,
      "reasoning": "<concise explanation>",
      "evidence_segment_ids": ["<exact source segment ID>"]
    }},
    "cliffhanger": {{
      "score": <float from 0.0 to 1.0>,
      "reasoning": "<concise explanation>",
      "evidence_segment_ids": ["<exact source segment ID>"]
    }},
    "standalone_clarity": {{
      "score": <float from 0.0 to 1.0>,
      "reasoning": "<concise explanation>",
      "evidence_segment_ids": ["<exact source segment ID>"]
    }},
    "emotional_impact": {{
      "score": <float from 0.0 to 1.0>,
      "reasoning": "<concise explanation>",
      "evidence_segment_ids": ["<exact source segment ID>"]
    }},
    "promo_suitability": {{
      "score": <float from 0.0 to 1.0>,
      "reasoning": "<concise explanation>",
      "evidence_segment_ids": ["<exact source segment ID>"]
    }}
  }},
  "major_weakness": "<most important weakness, or empty string if none>"
}}

Do not calculate an overall score or make an accept/reject decision.
Python will calculate those consistently.
"""


_PROMO_EVALUATION_WEIGHTS = {
    "hook_first_3_seconds": 0.20,
    "cliffhanger": 0.25,
    "standalone_clarity": 0.20,
    "emotional_impact": 0.15,
    "promo_suitability": 0.20,
}


def build_promo_evaluation_prompt(
    episode: dict,
    candidate: dict,
) -> str:
    """
    Build the final promotional quality evaluation prompt.

    The LLM evaluates an already optimized candidate using its
    complete original screenplay excerpt.

    No new boundaries or timestamps are generated here.
    """

    if not isinstance(candidate, dict):
        raise ValueError("Candidate must be a dictionary.")

    try:
        source_segments = _get_episode_segments(episode)
    except ValueError as exc:
        raise ValueError(
            "Could not retrieve episode source segments."
        ) from exc

    segment_map = {}

    for segment in source_segments:
        segment_id = _get_segment_id(segment)

        if segment_id:
            segment_map[segment_id] = segment

    segment_ids = candidate.get("segment_ids")

    if not isinstance(segment_ids, list) or not segment_ids:
        raise ValueError("Candidate has no segment_ids.")

    segment_ids = [str(item) for item in segment_ids]

    if len(set(segment_ids)) != len(segment_ids):
        raise ValueError("Candidate contains duplicate segment IDs.")

    if any(item not in segment_map for item in segment_ids):
        raise ValueError("Candidate references unknown source segments.")

    positions = {
        _get_segment_id(segment): index
        for index, segment in enumerate(source_segments)
        if _get_segment_id(segment)
    }

    selected_positions = [positions[item] for item in segment_ids]

    if selected_positions != list(
        range(selected_positions[0], selected_positions[-1] + 1)
    ):
        raise ValueError("Candidate segments are not contiguous.")

    first_segment = segment_map[segment_ids[0]]
    last_segment = segment_map[segment_ids[-1]]

    start = float(first_segment["start_seconds"])
    end = float(last_segment["end_seconds"])
    duration = end - start

    if not math.isfinite(duration) or not 20.0 <= duration <= 60.0:
        raise ValueError("Candidate duration must be between 20 and 60 seconds.")

    screenplay_lines = [
        _format_episode_segment(segment_map[item])
        for item in segment_ids
    ]

    episode_name = str(
        episode.get(
            "episode",
            episode.get("episode_number", "UNKNOWN"),
        )
    )

    return _PROMO_EVALUATION_TEMPLATE.format(
        episode_name=episode_name,
        candidate_id=str(candidate.get("candidate_id", "UNKNOWN")),
        start_timestamp=seconds_to_timestamp(start),
        end_timestamp=seconds_to_timestamp(end),
        duration_seconds=round(duration, 3),
        hook=str(candidate.get("hook", "Not provided.")),
        cliffhanger=str(candidate.get("cliffhanger", "Not provided.")),
        standalone_clarity=str(
            candidate.get("standalone_clarity", "Not provided.")
        ),
        narrative_context=str(
            candidate.get("narrative_context", "Not provided.")
        ),
        promotional_strength=str(
            candidate.get("promotional_strength", "Not provided.")
        ),
        screenplay="\n".join(screenplay_lines),
    )


def validate_promo_evaluation_response(
    data: dict,
    candidate: dict,
    episode: dict,
) -> dict | None:
    """
    Validate the final LLM promotional evaluation.

    Checks:
    - All five evaluation dimensions exist.
    - Scores are finite values between 0 and 1.
    - Each dimension has reasoning and source evidence.
    - Evidence IDs belong to the optimized candidate.
    - Hook evidence overlaps the first three seconds.
    - Candidate boundaries and duration match the original episode.

    Python calculates the weighted score and recommendation.

    Returns a normalized evaluation or None if invalid.
    """

    if not isinstance(data, dict):
        return None

    if not isinstance(candidate, dict) or not isinstance(episode, dict):
        return None

    candidate_id = str(candidate.get("candidate_id", ""))

    if not candidate_id:
        return None

    if str(data.get("candidate_id", "")) != candidate_id:
        return None

    assessments = data.get("assessments")

    if not isinstance(assessments, dict):
        return None

    required_dimensions = tuple(_PROMO_EVALUATION_WEIGHTS.keys())

    if any(
        dimension not in assessments
        for dimension in required_dimensions
    ):
        return None

    try:
        source_segments = _get_episode_segments(episode)
    except ValueError:
        return None

    segment_map = {}
    segment_positions = {}

    for position, segment in enumerate(source_segments):
        segment_id = _get_segment_id(segment)

        if segment_id:
            segment_map[segment_id] = segment
            segment_positions[segment_id] = position

    segment_ids = candidate.get("segment_ids")

    if not isinstance(segment_ids, list) or not segment_ids:
        return None

    segment_ids = [str(item) for item in segment_ids]

    if len(set(segment_ids)) != len(segment_ids):
        return None

    if any(item not in segment_map for item in segment_ids):
        return None

    positions = [segment_positions[item] for item in segment_ids]

    if positions != list(range(positions[0], positions[-1] + 1)):
        return None

    first_segment = segment_map[segment_ids[0]]
    last_segment = segment_map[segment_ids[-1]]

    try:
        clip_start = float(first_segment["start_seconds"])
        clip_end = float(last_segment["end_seconds"])
    except (KeyError, TypeError, ValueError):
        return None

    duration = clip_end - clip_start

    if not math.isfinite(duration) or not 20.0 <= duration <= 60.0:
        return None

    cleaned_assessments = {}

    for dimension in required_dimensions:
        assessment = assessments[dimension]

        if not isinstance(assessment, dict):
            return None

        score = assessment.get("score")
        reasoning = assessment.get("reasoning")
        evidence_ids = assessment.get("evidence_segment_ids")

        if not _is_valid_score(score):
            return None

        if not isinstance(reasoning, str) or not reasoning.strip():
            return None

        if not isinstance(evidence_ids, list) or not evidence_ids:
            return None

        evidence_ids = [str(item) for item in evidence_ids]

        if len(set(evidence_ids)) != len(evidence_ids):
            return None

        # Evidence must belong to the actual optimized clip.
        if any(item not in segment_ids for item in evidence_ids):
            return None

        # The hook assessment must cite evidence from the first
        # three seconds, not merely from somewhere in the clip.
        if dimension == "hook_first_3_seconds":
            hook_window_end = clip_start + 3.0

            valid_hook_evidence = []

            for evidence_id in evidence_ids:
                segment = segment_map[evidence_id]

                try:
                    segment_start = float(segment["start_seconds"])
                    segment_end = float(segment["end_seconds"])
                except (KeyError, TypeError, ValueError):
                    continue

                overlaps_hook_window = (
                    segment_start < hook_window_end
                    and segment_end > clip_start
                )

                if overlaps_hook_window:
                    valid_hook_evidence.append(evidence_id)

            if not valid_hook_evidence:
                return None

        cleaned_assessments[dimension] = {
            "score": float(score),
            "reasoning": reasoning.strip()[:1000],
            "evidence_segment_ids": evidence_ids,
        }

    # Weighted score is calculated in Python, not by the LLM.
    overall_score = sum(
        cleaned_assessments[dimension]["score"] * weight
        for dimension, weight in _PROMO_EVALUATION_WEIGHTS.items()
    )

    # Recommendation thresholds are deliberately conservative.
    # A high average cannot compensate for a fundamentally weak
    # hook, cliffhanger or standalone clarity.
    hook_score = cleaned_assessments["hook_first_3_seconds"]["score"]
    cliffhanger_score = cleaned_assessments["cliffhanger"]["score"]
    clarity_score = cleaned_assessments["standalone_clarity"]["score"]

    if (
        overall_score >= 0.72
        and hook_score >= 0.50
        and cliffhanger_score >= 0.50
        and clarity_score >= 0.50
    ):
        recommendation = "accept"

    elif overall_score >= 0.50:
        recommendation = "review"

    else:
        recommendation = "reject"

    major_weakness = data.get("major_weakness", "")

    if not isinstance(major_weakness, str):
        major_weakness = ""

    return {
        "candidate_id": candidate_id,
        "dimension_scores": {
            dimension: assessment["score"]
            for dimension, assessment in cleaned_assessments.items()
        },
        "assessments": cleaned_assessments,
        "overall_score": round(overall_score, 4),
        "recommendation": recommendation,
        "major_weakness": major_weakness.strip()[:1000],
        "evaluation_weights": dict(_PROMO_EVALUATION_WEIGHTS),
        "duration_seconds": round(duration, 3),
    }
