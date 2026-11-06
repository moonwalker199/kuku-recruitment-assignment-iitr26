"""
src/ranking/scorer.py — Weighted promo score computation.

Scoring logic
-------------
1. Receives a feature dict (deterministic keys + optional 'llm_*' keys).
2. Looks up per-feature weights from Config, normalised to sum to 1.0.
3. If LLM features are absent, their weight is redistributed to deterministic
   features (handled by Config.effective_weights).
4. Returns a single float in [0.0, 1.0].

This module is intentionally free of ML models. The weighted-sum formula is
transparent, auditable, and tunable via config/config.yaml alone.
"""

from __future__ import annotations

import logging

from src.config import Config

logger = logging.getLogger(__name__)


class Scorer:
    """
    Computes a promotional suitability score from a feature dict.

    Parameters
    ----------
    cfg : Config
        Full pipeline configuration; weights read from cfg.feature_weights.
    llm_available : bool
        Whether LLM-derived features are present.  If False, their weight
        mass is redistributed to deterministic features automatically.
    """

    def __init__(self, cfg: Config, llm_available: bool) -> None:
        self.weights = cfg.effective_weights(llm_available)
        logger.debug(
            "Scorer initialised with %d active features (llm=%s).",
            len(self.weights), llm_available,
        )

    def score(self, features: dict[str, float]) -> float:
        """
        Compute weighted sum over available features.

        Missing features default to 0.0 (conservative — no inflation).
        Returns a float in [0.0, 1.0].
        """
        total = 0.0
        for feature, weight in self.weights.items():
            value = features.get(feature, 0.0)
            total += weight * value
        return min(max(total, 0.0), 1.0)

    def score_breakdown(self, features: dict[str, float]) -> dict[str, float]:
        """
        Return per-feature weighted contributions for explainability.

        Each value is weight_i × feature_i (un-normalised contribution).
        """
        return {
            feature: round(self.weights.get(feature, 0.0) * features.get(feature, 0.0), 4)
            for feature in set(list(self.weights.keys()) + list(features.keys()))
        }
