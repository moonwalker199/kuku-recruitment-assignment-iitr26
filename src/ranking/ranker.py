"""
src/ranking/ranker.py — Global ranking across all episodes.

After per-episode NMS, all surviving candidates are merged into a single
global list sorted by promo_score descending.  The top-K are selected for
the final output.

Also provides the hard-filter step that removes windows below the minimum
score threshold before NMS.
"""

from __future__ import annotations

import logging

from src.config import Config

logger = logging.getLogger(__name__)


class Ranker:
    """
    Global ranker: applies score threshold, runs per-episode NMS,
    then returns a globally sorted top-K list.

    Parameters
    ----------
    cfg : Config
        Pipeline configuration; reads top_k and ranking settings.
    deduplicator
        A Deduplicator instance (injected for testability).
    """

    def __init__(self, cfg: Config, deduplicator) -> None:
        self.cfg = cfg
        self.deduplicator = deduplicator

    def rank(
        self,
        all_candidates: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        """
        Apply threshold, per-episode NMS, global sort, and top-K selection.

        Parameters
        ----------
        all_candidates : list[dict]
            All scored candidates across all episodes (must have 'promo_score').

        Returns
        -------
        top_k : list[dict]
            Globally ranked top-K recommendations, sorted descending by score.
            Each dict has a 'rank' key (1-indexed).
        rejected : list[dict]
            All windows that were filtered out or suppressed, with
            'rejection_reason' set.
        """
        from src.ranking.deduplicator import group_by_episode

        threshold = self.cfg.ranking.min_score_threshold
        rejected: list[dict] = []

        # --- Hard score threshold filter ---------------------------------
        passing, below = [], []
        for c in all_candidates:
            if c.get("promo_score", 0.0) >= threshold:
                passing.append(c)
            else:
                c["rejection_reason"] = (
                    f"Score {c.get('promo_score', 0.0):.4f} below threshold {threshold:.2f}"
                )
                below.append(c)

        rejected.extend(below)
        logger.info(
            "Score threshold (%.2f): %d passing, %d rejected.",
            threshold, len(passing), len(below),
        )

        # --- Per-episode NMS ----------------------------------------------
        by_episode = group_by_episode(passing)
        kept: list[dict] = []

        for ep, ep_candidates in sorted(by_episode.items()):
            ep_kept, ep_suppressed = self.deduplicator.run(ep_candidates)
            kept.extend(ep_kept)
            rejected.extend(ep_suppressed)

        # --- Global sort + top-K selection --------------------------------
        ranked = sorted(kept, key=lambda w: w["promo_score"], reverse=True)

        top_k = ranked[: self.cfg.top_k]
        below_top_k = ranked[self.cfg.top_k :]
        for w in below_top_k:
            w["rejection_reason"] = (
                f"Outside top-{self.cfg.top_k} globally (score={w['promo_score']:.4f})"
            )
        rejected.extend(below_top_k)

        # Assign global ranks
        for rank_i, window in enumerate(top_k, start=1):
            window["rank"] = rank_i

        logger.info(
            "Global ranking: %d total kept after NMS → top-%d selected.",
            len(kept), len(top_k),
        )
        return top_k, rejected

    @staticmethod
    def apply_score_to_candidates(
        candidates: list[dict],
        scorer,
    ) -> list[dict]:
        """
        Compute and attach 'promo_score' and 'score_breakdown' to each candidate.

        Modifies dicts in-place and returns the same list for chaining.
        """
        for cand in candidates:
            features = cand.get("features", {})
            cand["promo_score"] = scorer.score(features)
            cand["score_breakdown"] = scorer.score_breakdown(features)
        return candidates
