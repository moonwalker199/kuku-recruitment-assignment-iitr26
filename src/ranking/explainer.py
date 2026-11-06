"""
src/ranking/explainer.py — Human-readable explanation generation.

Produces two artefacts per recommendation:
  1. A structured dict with score breakdown, evidence segments, and reasoning.
  2. A per-episode Markdown section (assembled by the report writer in main.py).

The 'reasoning' field is generated programmatically from the actual screenplay
evidence — delivery tags, quoted lines, and score signals — so every explanation
is fully grounded in the text, not hallucinated.

Markdown reports are written as:
  outputs/reports/episode-NNN_report.md   (per episode)
  outputs/reports/global_report.md        (all episodes)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.candidate_generation.window_generator import CandidateWindow
from src.feature_extraction.lexicons import (
    CLIFFHANGER_TAGS,
    DISTRESS_TAGS,
    HIGH_CONFLICT_TAGS,
    score_delivery_tag,
)
from src.utils import seconds_to_timestamp, format_duration

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Explanation builder
# ---------------------------------------------------------------------------

class Explainer:
    """
    Generates reasoning strings and evidence segments for top recommendations.

    Parameters
    ----------
    max_evidence : int
        Maximum number of evidence segments to include per recommendation.
    """

    def __init__(self, max_evidence: int = 5) -> None:
        self.max_evidence = max_evidence

    def enrich(self, ranked_candidates: list[dict]) -> list[dict]:
        """
        Add 'reasoning', 'top_evidence_segments', 'start_time', 'end_time'
        to each candidate dict.  Operates in-place, returns the list.
        """
        for cand in ranked_candidates:
            cand["start_time"] = seconds_to_timestamp(cand["start_seconds"])
            cand["end_time"] = seconds_to_timestamp(cand["end_seconds"])
            evidence = _select_evidence(cand, self.max_evidence)
            cand["top_evidence_segments"] = evidence
            cand["reasoning"] = _build_reasoning(cand, evidence)
        return ranked_candidates


# ---------------------------------------------------------------------------
# Evidence selection
# ---------------------------------------------------------------------------

def _select_evidence(cand: dict, max_ev: int) -> list[dict]:
    """
    Choose the most emotionally intense dialogue segments as evidence.

    Strategy:
      1. Score each dialogue segment by its delivery tags.
      2. Add the last non-camera segment unconditionally (cliffhanger anchor).
      3. Deduplicate by segment_id; sort by start_seconds.
    """
    dialogue = cand.get("dialogue_segments", [])

    scored = []
    for seg in dialogue:
        raw = seg.get("delivery") or ""
        tags = [t.strip() for t in raw.split(",") if t.strip()]
        intensity = max((score_delivery_tag(t) for t in tags), default=0.1)
        scored.append((intensity, seg))

    scored.sort(key=lambda x: -x[0])
    top = [seg for _, seg in scored[:max_ev]]

    # Always include the last action/dialogue segment (hook anchor)
    all_segs = cand.get("dialogue_segments", []) + cand.get("action_segments", [])
    all_segs = sorted(all_segs, key=lambda s: s["start_seconds"])
    if all_segs:
        last = all_segs[-1]
        if last.get("segment_id") not in {s.get("segment_id") for s in top}:
            if len(top) >= max_ev:
                top[-1] = last
            else:
                top.append(last)

    # Sort by time for readability
    top.sort(key=lambda s: s.get("start_seconds", 0))

    return [
        {
            "segment_id": s.get("segment_id", ""),
            "kind": s.get("kind", ""),
            "speaker": s.get("speaker"),
            "delivery": s.get("delivery"),
            "text": s.get("text", ""),
            "start_seconds": s.get("start_seconds"),
            "end_seconds": s.get("end_seconds"),
            "start_time": seconds_to_timestamp(s.get("start_seconds", 0)),
        }
        for s in top
    ]


# ---------------------------------------------------------------------------
# Reasoning text generation
# ---------------------------------------------------------------------------

def _build_reasoning(cand: dict, evidence: list[dict]) -> str:
    """
    Generate a concise, evidence-grounded reasoning paragraph.

    Rules:
    • Reference actual speaker names and delivery tags.
    • Include a direct quote from the strongest evidence segment.
    • Comment on the duration fit for short-form promotional formats.
    • Mention LLM assessment if present.
    """
    parts: list[str] = []
    features = cand.get("features", {})
    breakdown = cand.get("score_breakdown", {})
    duration = cand.get("duration_seconds", 0)
    llm = cand.get("llm_analysis")

    # Identify dominant dramatic signal
    top_feat = max(breakdown.items(), key=lambda x: x[1], default=("unknown", 0.0))

    # Conflict / confrontation
    speakers = list({
        s["speaker"].title() for s in cand.get("dialogue_segments", [])
        if s.get("speaker")
    })
    if len(speakers) >= 2 and features.get("conflict_score", 0) > 0.5:
        parts.append(
            f"Direct confrontation between {_speaker_list(speakers)} "
            f"drives a conflict score of {features.get('conflict_score', 0):.2f}."
        )
    elif speakers:
        parts.append(f"Features {_speaker_list(speakers)}.")

    # Emotional intensity — quote the best delivery tag
    best_ev = _highest_intensity_segment(evidence)
    if best_ev:
        speaker = (best_ev.get("speaker") or "UNKNOWN").title()
        delivery = best_ev.get("delivery") or ""
        text = (best_ev.get("text") or "").strip()[:80]
        if delivery:
            parts.append(
                f"Peak emotional beat: {speaker} ({delivery}): \"{text}\""
            )

    # Cliffhanger / ending signal
    if features.get("cliffhanger_score", 0) > 0.6:
        last_action = _last_action_text(cand)
        if last_action:
            parts.append(f"Cliffhanger ending: {last_action[:100]}")
        else:
            parts.append("Strong cliffhanger hook at window end.")

    # Revelation
    if features.get("revelation_score", 0) > 0.5:
        parts.append("Contains a key revelation or unexpected disclosure.")

    # LLM assessment
    if llm and llm.get("reasoning"):
        parts.append(f"LLM: {llm['reasoning'].strip()}")

    # Duration comment
    dur_str = format_duration(duration)
    if duration <= 60:
        parts.append(f"{dur_str} fits a standard short-form promotional slot.")
    else:
        parts.append(f"{dur_str} — suitable for mid-length platform previews.")

    return " ".join(parts) if parts else "High promotional suitability based on combined scoring."


def _speaker_list(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def _highest_intensity_segment(evidence: list[dict]) -> dict | None:
    if not evidence:
        return None
    dialogue_ev = [e for e in evidence if e.get("kind") == "dialogue"]
    if not dialogue_ev:
        return None
    return max(
        dialogue_ev,
        key=lambda s: max(
            (score_delivery_tag(t.strip()) for t in (s.get("delivery") or "").split(",")),
            default=0.0,
        ),
    )


def _last_action_text(cand: dict) -> str:
    action_segs = sorted(cand.get("action_segments", []), key=lambda s: s.get("start_seconds", 0))
    sfx_segs = sorted(cand.get("sound_fx_segments", []), key=lambda s: s.get("start_seconds", 0))
    all_late = action_segs + sfx_segs
    all_late.sort(key=lambda s: s.get("start_seconds", 0))
    return all_late[-1].get("text", "") if all_late else ""


# ---------------------------------------------------------------------------
# Markdown report writers
# ---------------------------------------------------------------------------

def write_episode_report(
    episode_name: str,
    episode_candidates: list[dict],
    all_episode_candidates: list[dict],
    output_dir: str | Path,
) -> Path:
    """
    Write a per-episode Markdown report.

    Parameters
    ----------
    episode_name : str
        e.g. 'Episode 7'
    episode_candidates : list[dict]
        The final kept recommendations for this episode.
    all_episode_candidates : list[dict]
        All candidates generated for this episode (for statistics).
    output_dir : Path
        Directory to write the .md file into.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Safe filename
    safe_name = episode_name.lower().replace(" ", "_")
    out_path = out_dir / f"{safe_name}_report.md"

    logline = ""
    if episode_candidates:
        logline = episode_candidates[0].get("logline", "")
    elif all_episode_candidates:
        logline = all_episode_candidates[0].get("logline", "")

    lines: list[str] = [
        f"# {episode_name} — Promotional Moment Report\n",
        f"**Logline**: {logline}\n" if logline else "",
        f"**Candidates generated**: {len(all_episode_candidates)}  \n",
        f"**Recommendations**: {len(episode_candidates)}\n",
        "",
    ]

    if not episode_candidates:
        lines.append(
            "> No promotional moments above the score threshold were found "
            "for this episode.\n"
        )
    else:
        lines.append("## Recommended Promotional Moments\n")
        for idx, cand in enumerate(episode_candidates, start=1):
            lines.extend(_format_candidate_md(cand, idx))

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def write_global_report(
    top_k: list[dict],
    stats: dict,
    output_dir: str | Path,
) -> Path:
    """Write the global summary Markdown report."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "global_report.md"

    lines: list[str] = [
        "# Kuku FM — Global Promotional Moment Rankings\n",
        "## Pipeline Statistics\n",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Episodes analysed | {stats.get('episodes', 0)} |",
        f"| Total candidates | {stats.get('total_candidates', 0)} |",
        f"| Candidates passing filters | {stats.get('passing_candidates', 0)} |",
        f"| Final recommendations | {len(top_k)} |",
        f"| LLM analysis enabled | {stats.get('llm_enabled', False)} |",
        "",
        "## Top Recommendations\n",
        "| Rank | Episode | Time Range | Duration | Score | Key Hook |",
        "|---|---|---|---|---|---|",
    ]

    for cand in top_k:
        rank = cand.get("rank", "?")
        ep = cand.get("episode", "?")
        t_start = cand.get("start_time", "?")
        t_end = cand.get("end_time", "?")
        dur = format_duration(cand.get("duration_seconds", 0))
        score = f"{cand.get('promo_score', 0):.4f}"
        # Pull a short quote from evidence
        ev = cand.get("top_evidence_segments", [])
        hook = ""
        for e in ev:
            if e.get("kind") == "dialogue" and e.get("text"):
                hook = f"\"{e['text'][:50]}\"" if len(e["text"]) > 50 else f"\"{e['text']}\""
                break
        lines.append(f"| {rank} | {ep} | {t_start}–{t_end} | {dur} | {score} | {hook} |")

    lines += [
        "",
        "---",
        "*Generated by the Kuku FM Promo Recommendation System.*",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _format_candidate_md(cand: dict, idx: int) -> list[str]:
    """Format a single recommendation as a Markdown section."""
    score = cand.get("promo_score", 0)
    t_start = cand.get("start_time", "?")
    t_end = cand.get("end_time", "?")
    dur = format_duration(cand.get("duration_seconds", 0))
    scenes = cand.get("scene_numbers", [])
    scene_str = ", ".join(f"Scene {s}" for s in scenes) if scenes else "Unknown"
    reasoning = cand.get("reasoning", "")

    lines: list[str] = [
        f"### 🎬 Clip #{idx} — Score: {score:.4f}\n",
        f"**Time Range**: {t_start} → {t_end} ({dur})  ",
        f"**Scene(s)**: {scene_str}  ",
        f"**Window ID**: `{cand.get('window_id', '')}`\n",
        f"**Why This Works**: {reasoning}\n",
    ]

    evidence = cand.get("top_evidence_segments", [])
    if evidence:
        lines.append("**Key Lines**:\n")
        for ev in evidence:
            if ev.get("kind") == "dialogue":
                sp = (ev.get("speaker") or "?").title()
                dl = ev.get("delivery") or ""
                txt = ev.get("text", "")
                ts = ev.get("start_time", "")
                dl_part = f" *({dl})*" if dl else ""
                lines.append(f"- [{ts}] **{sp}**{dl_part}: \"{txt}\"")
            elif ev.get("kind") == "action":
                ts = ev.get("start_time", "")
                txt = ev.get("text", "")[:80]
                lines.append(f"- [{ts}] *{txt}*")

    # Score breakdown table
    breakdown = cand.get("score_breakdown", {})
    if breakdown:
        lines += [
            "",
            "<details><summary>Score Breakdown</summary>\n",
            "| Feature | Contribution |",
            "|---|---|",
        ]
        for feat, val in sorted(breakdown.items(), key=lambda x: -x[1]):
            if val > 0:
                raw_feat = cand.get("features", {}).get(feat, 0)
                lines.append(f"| {feat} | {val:.4f} (raw={raw_feat:.3f}) |")
        lines.append("</details>")

    lines += ["", "---", ""]
    return lines
