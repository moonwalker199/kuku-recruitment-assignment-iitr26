"""
src/ranking/deduplicator.py — IoU-based Non-Maximum Suppression (NMS).

Applies per-episode NMS: given a list of scored candidates from the same episode,
suppress any window that heavily overlaps (IoU > threshold) with a higher-scored
window.

Design constraints
------------------
• An episode may produce 0, 1, or multiple final recommendations.
• No quota is imposed per episode.
• IoU is computed on wall-clock seconds (not segment indices).

Returns
-------
(kept, suppressed) — both are lists of scored candidate dicts.
Suppressed items include a 'rejection_reason' key for the report.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _iou(a: dict, b: dict) -> float:
    """Compute Intersection-over-Union on wall-clock time ranges."""
    a_start, a_end = a["start_seconds"], a["end_seconds"]
    b_start, b_end = b["start_seconds"], b["end_seconds"]
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return intersection / union if union > 0.0 else 0.0


class Deduplicator:
    """
    Per-episode Non-Maximum Suppression on ranked candidates.

    Parameters
    ----------
    iou_threshold : float
        Windows whose IoU with a higher-scored window exceeds this are
        suppressed.  Recommended range: 0.40 – 0.65.
    enabled : bool
        When False, all windows are kept and nothing is suppressed.
    """

    def __init__(self, iou_threshold: float = 0.50, enabled: bool = True) -> None:
        self.iou_threshold = iou_threshold
        self.enabled = enabled

    def run(self, candidates: list[dict]) -> tuple[list[dict], list[dict]]:
        """
        Apply NMS to *candidates* from a single episode.

        Parameters
        ----------
        candidates : list[dict]
            Scored candidate dicts (must have keys: start_seconds, end_seconds,
            promo_score, window_id).  Need not be sorted beforehand.

        Returns
        -------
        kept : list[dict]
            Non-suppressed candidates, sorted by promo_score descending.
        suppressed : list[dict]
            Suppressed candidates with 'rejection_reason' set.
        """
        if not candidates:
            return [], []

        if not self.enabled:
            return sorted(candidates, key=lambda w: w["promo_score"], reverse=True), []

        # Sort by score descending (greedy NMS)
        ranked = sorted(candidates, key=lambda w: w["promo_score"], reverse=True)
        kept: list[dict] = []
        suppressed: list[dict] = []
        suppressed_indices: set[int] = set()

        for i, candidate in enumerate(ranked):
            if i in suppressed_indices:
                continue
            kept.append(candidate)

            for j in range(i + 1, len(ranked)):
                if j in suppressed_indices:
                    continue
                iou_val = _iou(candidate, ranked[j])
                if iou_val > self.iou_threshold:
                    suppressed_indices.add(j)
                    ranked[j]["rejection_reason"] = (
                        f"IoU {iou_val:.2f} > {self.iou_threshold:.2f} "
                        f"with higher-ranked window {candidate['window_id']} "
                        f"(score={candidate['promo_score']:.4f})"
                    )
                    suppressed.append(ranked[j])

        logger.debug(
            "NMS: %d kept, %d suppressed (threshold=%.2f).",
            len(kept), len(suppressed), self.iou_threshold,
        )
        return kept, suppressed


def group_by_episode(candidates: list[dict]) -> dict[str, list[dict]]:
    """Group a flat list of candidates by episode string."""
    groups: dict[str, list[dict]] = {}
    for cand in candidates:
        ep = cand.get("episode", "UNK")
        groups.setdefault(ep, []).append(cand)
    return groups
