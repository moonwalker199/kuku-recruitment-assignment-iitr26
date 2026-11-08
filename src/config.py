
"""
src/config.py

Central configuration loading and validation.

Supports:
- Stage-specific LLM configurations.
- Shared LLM defaults.
- Backward compatibility with the previous global LLM config.
- Typed configuration dataclasses.
- CLI overrides.

LLM stages:
1. episode_analysis
2. boundary_optimization
3. promo_evaluation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------

# @dataclass
# class LLMConfig:
#     provider: str = "google"
#     model: str = "gemini-3.8-flash"
#     temperature: float = 0.1
#     max_output_tokens: int = 4096
#     timeout_seconds: float = 120.0
#     max_retries: int = 3
#     retry_delay_seconds: float = 2.0
#     llm_score_threshold: float = 0.25


@dataclass
class LLMConfig:
    provider: str = "google"
    model: str = "gemini-3.5-flash"
    base_url: str | None = None
    temperature: float = 0.1
    max_output_tokens: int = 4096
    timeout_seconds: float = 120.0
    max_retries: int = 3
    retry_delay_seconds: float = 2.0
    llm_score_threshold: float = 0.25
    thinking_level: str | None = None


@dataclass
class LLMStagesConfig:
    """
    Independent LLM configurations for each pipeline stage.

    Each stage can use a different provider, model and generation settings.
    """

    episode_analysis: LLMConfig = field(default_factory=LLMConfig)

    boundary_optimization: LLMConfig = field(
        default_factory=lambda: LLMConfig(
            max_output_tokens=2048,
            timeout_seconds=60.0,
        )
    )

    promo_evaluation: LLMConfig = field(
        default_factory=lambda: LLMConfig(
            max_output_tokens=4096,
            timeout_seconds=120.0,
        )
    )


# ---------------------------------------------------------------------------
# Candidate generation configuration
# ---------------------------------------------------------------------------

@dataclass
class CandidateConfig:
    window_step_seconds: int = 5

    window_durations: list = field(
        default_factory=lambda: [15, 25, 35, 50, 65, 80]
    )

    min_duration_seconds: float = 25.0
    max_duration_seconds: float = 60.0

    min_dialogue_segments: int = 3
    include_scene_aligned_windows: bool = True
    include_scene_pair_windows: bool = True


# ---------------------------------------------------------------------------
# Ranking configuration
# ---------------------------------------------------------------------------

@dataclass
class DiversityConfig:
    enabled: bool = True
    iou_threshold: float = 0.5


@dataclass
class RankingConfig:
    min_score_threshold: float = 0.30
    diversity: DiversityConfig = field(
        default_factory=DiversityConfig
    )


# ---------------------------------------------------------------------------
# Output configuration
# ---------------------------------------------------------------------------

@dataclass
class OutputConfig:
    candidates_dir: str = "outputs/candidates"
    ranked_promos_dir: str = "outputs/ranked_promos"
    rejected_candidates_dir: str = "outputs/rejected_candidates"
    reports_dir: str = "outputs/reports"

    save_all_candidates: bool = True
    generate_markdown: bool = True
    generate_global_report: bool = True

    max_evidence_segments: int = 5


# ---------------------------------------------------------------------------
# Main configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:

    top_k: int = 50
    llm_enabled: bool = True

    llm: LLMConfig = field(default_factory=LLMConfig)

    llm_stages: LLMStagesConfig = field(
        default_factory=LLMStagesConfig
    )

    candidate_generation: CandidateConfig = field(
        default_factory=CandidateConfig
    )

    feature_weights: dict = field(default_factory=dict)

    ranking: RankingConfig = field(
        default_factory=RankingConfig
    )

    output: OutputConfig = field(
        default_factory=OutputConfig
    )

    DEFAULT_WEIGHTS: dict = field(
        default_factory=lambda: {
            "emotional_intensity": 0.15,
            "cliffhanger_score": 0.15,
            "conflict_score": 0.12,
            "tension_arc": 0.10,
            "lexical_hook_score": 0.08,
            "revelation_score": 0.07,
            "dialogue_density": 0.05,
            "protagonist_presence": 0.04,
            "speaker_variety": 0.03,
            "position_in_episode": 0.02,
            "sound_fx_density": 0.01,
            "scene_boundary_bonus": 0.01,
            "llm_promotional_suitability": 0.10,
            "llm_hook_potential": 0.05,
            "llm_context_independence": 0.02,
        }
    )

    def get_llm_config(self, stage: str) -> LLMConfig:
        """
        Retrieve the configuration for a specific LLM stage.

        Supported stages:
            episode_analysis
            boundary_optimization
            promo_evaluation
        """

        if not hasattr(self.llm_stages, stage):
            raise ValueError(
                f"Unknown LLM stage: {stage!r}. "
                "Expected episode_analysis, boundary_optimization "
                "or promo_evaluation."
            )

        return getattr(self.llm_stages, stage)

    def effective_weights(
        self,
        llm_available: bool,
    ) -> dict[str, float]:
        """
        Return normalized feature weights.

        If the LLM is unavailable, LLM-specific weights are
        redistributed proportionally across deterministic features.
        """

        raw = (
            self.feature_weights
            if self.feature_weights
            else self.DEFAULT_WEIGHTS
        )

        if llm_available:

            weights = {
                key: value
                for key, value in raw.items()
                if value > 0
            }

        else:

            det_weights = {
                key: value
                for key, value in raw.items()
                if not key.startswith("llm_") and value > 0
            }

            llm_total = sum(
                value
                for key, value in raw.items()
                if key.startswith("llm_")
            )

            det_total = sum(det_weights.values())

            if det_total > 0 and llm_total > 0:

                scale = (
                    det_total + llm_total
                ) / det_total

                det_weights = {
                    key: value * scale
                    for key, value in det_weights.items()
                }

            weights = det_weights

        total = sum(weights.values())

        if total <= 0:
            raise ValueError(
                "All feature weights are zero; check config.yaml."
            )

        return {
            key: value / total
            for key, value in weights.items()
        }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_nested(
    data: dict,
    *keys: str,
    default=None,
):
    """Safely retrieve nested dictionary values."""

    current = data

    for key in keys:

        if not isinstance(current, dict):
            return default

        current = current.get(key, default)

    return current



def _build_llm_config(
    raw: dict | None,
    defaults: dict | None = None,
) -> LLMConfig:
    """
    Construct an LLMConfig.

    Values in raw override defaults.
    Missing values fall back to dataclass defaults.
    """

    raw = raw or {}
    defaults = defaults or {}

    merged = {
        **defaults,
        **raw,
    }

    return LLMConfig(
        provider=str(merged.get("provider", "google")),
        model=str(merged.get("model", "gemini-3.5-flash")),
        base_url=(
            str(merged["base_url"]).strip()
            if merged.get("base_url")
            else None
        ),
        temperature=float(merged.get("temperature", 0.1)),
        max_output_tokens=int(merged.get("max_output_tokens", 4096)),
        timeout_seconds=float(merged.get("timeout_seconds", 120.0)),
        max_retries=int(merged.get("max_retries", 3)),
        retry_delay_seconds=float(merged.get("retry_delay_seconds", 2.0)),
        llm_score_threshold=float(merged.get("llm_score_threshold", 0.25)),
        thinking_level=(
            str(merged["thinking_level"])
            if merged.get("thinking_level") is not None
            else None
        ),

    )





# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------

def load_config(
    path: str | Path = "config/config.yaml",
) -> Config:
    """
    Load YAML configuration and return a validated Config object.

    Supports both:

    Old format:
        llm:
          provider: google
          model: ...

    New format:
        llm:
          defaults:
            provider: google
            model: ...
          episode_analysis:
            model: ...
          boundary_optimization:
            model: ...
          promo_evaluation:
            model: ...

    Old configurations remain supported.
    """

    config_path = Path(path)

    raw: dict[str, Any] = {}

    if config_path.exists():

        with config_path.open(
            encoding="utf-8"
        ) as file:

            raw = yaml.safe_load(file) or {}

        logger.debug(
            "Loaded configuration from %s",
            config_path,
        )

    else:

        logger.warning(
            "Configuration file not found at %s. "
            "Using defaults.",
            config_path,
        )

    pipeline = raw.get("pipeline", {}) or {}

    llm_raw = raw.get("llm", {}) or {}

    candidate_raw = (
        raw.get("candidate_generation", {}) or {}
    )

    ranking_raw = raw.get("ranking", {}) or {}

    output_raw = raw.get("output", {}) or {}

    diversity_raw = (
        ranking_raw.get("diversity", {}) or {}
    )

    # -------------------------------------------------------
    # LLM configurations
    # -------------------------------------------------------

    # Backward compatibility:
    # If the old flat llm configuration is used, treat it
    # as the shared default for all stages.

    if "defaults" in llm_raw:

        shared_defaults = (
            llm_raw.get("defaults", {}) or {}
        )

    else:

        shared_defaults = {
            key: value
            for key, value in llm_raw.items()
            if key not in (
                "episode_analysis",
                "boundary_optimization",
                "promo_evaluation",
            )
        }

    episode_raw = (
        llm_raw.get("episode_analysis", {}) or {}
    )

    boundary_raw = (
        llm_raw.get("boundary_optimization", {}) or {}
    )

    evaluation_raw = (
        llm_raw.get("promo_evaluation", {}) or {}
    )

    episode_cfg = _build_llm_config(
        episode_raw,
        shared_defaults,
    )

    boundary_cfg = _build_llm_config(
        boundary_raw,
        shared_defaults,
    )

    evaluation_cfg = _build_llm_config(
        evaluation_raw,
        shared_defaults,
    )

    stages_cfg = LLMStagesConfig(
        episode_analysis=episode_cfg,
        boundary_optimization=boundary_cfg,
        promo_evaluation=evaluation_cfg,
    )

    # Backward compatibility:
    # Existing code using cfg.llm continues to work.
    legacy_llm_cfg = episode_cfg

    # -------------------------------------------------------
    # Candidate generation
    # -------------------------------------------------------

    candidate_cfg = CandidateConfig(

        window_step_seconds=int(
            candidate_raw.get(
                "window_step_seconds",
                5,
            )
        ),

        window_durations=list(
            candidate_raw.get(
                "window_durations",
                [15, 25, 35, 50, 65, 80],
            )
        ),

        min_duration_seconds=float(
            candidate_raw.get(
                "min_duration_seconds",
                25.0,
            )
        ),

        max_duration_seconds=float(
            candidate_raw.get(
                "max_duration_seconds",
                60.0,
            )
        ),

        min_dialogue_segments=int(
            candidate_raw.get(
                "min_dialogue_segments",
                3,
            )
        ),

        include_scene_aligned_windows=bool(
            candidate_raw.get(
                "include_scene_aligned_windows",
                True,
            )
        ),

        include_scene_pair_windows=bool(
            candidate_raw.get(
                "include_scene_pair_windows",
                True,
            )
        ),
    )

    # -------------------------------------------------------
    # Ranking
    # -------------------------------------------------------

    ranking_cfg = RankingConfig(

        min_score_threshold=float(
            ranking_raw.get(
                "min_score_threshold",
                0.30,
            )
        ),

        diversity=DiversityConfig(

            enabled=bool(
                diversity_raw.get(
                    "enabled",
                    True,
                )
            ),

            iou_threshold=float(
                diversity_raw.get(
                    "iou_threshold",
                    0.50,
                )
            ),
        ),
    )

    # -------------------------------------------------------
    # Output
    # -------------------------------------------------------

    output_cfg = OutputConfig(

        candidates_dir=output_raw.get(
            "candidates_dir",
            "outputs/candidates",
        ),

        ranked_promos_dir=output_raw.get(
            "ranked_promos_dir",
            "outputs/ranked_promos",
        ),

        rejected_candidates_dir=output_raw.get(
            "rejected_candidates_dir",
            "outputs/rejected_candidates",
        ),

        reports_dir=output_raw.get(
            "reports_dir",
            "outputs/reports",
        ),

        save_all_candidates=bool(
            output_raw.get(
                "save_all_candidates",
                True,
            )
        ),

        generate_markdown=bool(
            output_raw.get(
                "generate_markdown",
                True,
            )
        ),

        generate_global_report=bool(
            output_raw.get(
                "generate_global_report",
                True,
            )
        ),

        max_evidence_segments=int(
            output_raw.get(
                "max_evidence_segments",
                5,
            )
        ),
    )

    # -------------------------------------------------------
    # Main configuration
    # -------------------------------------------------------

    cfg = Config(

        top_k=int(
            pipeline.get(
                "top_k",
                50,
            )
        ),

        llm_enabled=bool(
            pipeline.get(
                "llm_enabled",
                True,
            )
        ),

        llm=legacy_llm_cfg,

        llm_stages=stages_cfg,

        candidate_generation=candidate_cfg,

        feature_weights=raw.get(
            "feature_weights",
            {},
        ) or {},

        ranking=ranking_cfg,

        output=output_cfg,
    )

    _validate(cfg)

    return cfg


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_llm_config(
    config: LLMConfig,
    stage_name: str,
    errors: list[str],
) -> None:

    # if config.provider not in (
    #     "google",
    #     "openai",
    # ):
    #     errors.append(
    #         f"{stage_name}.provider must be "
    #         "'google' or 'openai'."
    #     )

    # if not config.model.strip():
    #     errors.append(
    #         f"{stage_name}.model cannot be empty."
    #     )

    # if not 0.0 <= config.temperature <= 2.0:
    #     errors.append(
    #         f"{stage_name}.temperature must be in [0, 2]."
    #     )

    # if config.max_output_tokens < 1:
    #     errors.append(
    #         f"{stage_name}.max_output_tokens must be >= 1."
    #     )

    # if config.timeout_seconds <= 0:
    #     errors.append(
    #         f"{stage_name}.timeout_seconds must be positive."
    #     )

    # if config.max_retries < 1:
    #     errors.append(
    #         f"{stage_name}.max_retries must be >= 1."
    #     )

    # if config.retry_delay_seconds < 0:
    #     errors.append(
    #         f"{stage_name}.retry_delay_seconds cannot be negative."
    #     )

    # if not 0.0 <= config.llm_score_threshold <= 1.0:
    #     errors.append(
    #         f"{stage_name}.llm_score_threshold must be in [0, 1]."
    #     )
    if config.provider not in (
    "google",
    "openai",
    "openai_compatible",
):
        errors.append(
            f"{stage_name}.provider must be "
            "'google', 'openai' or 'openai_compatible'."
    )

    if config.provider == "google" and config.base_url:
        errors.append(
            f"{stage_name}.base_url is not supported "
            "by the current native Gemini adapter."
    )

    if config.base_url and not config.base_url.startswith(
        ("http://", "https://")
):
        errors.append(
            f"{stage_name}.base_url must be a valid HTTP(S) URL."
    )


def _validate(cfg: Config) -> None:
    """Validate all configuration constraints."""

    errors: list[str] = []

    candidate = cfg.candidate_generation

    if cfg.top_k < 1:
        errors.append(
            "pipeline.top_k must be >= 1."
        )

    if candidate.window_step_seconds < 1:
        errors.append(
            "candidate_generation.window_step_seconds must be >= 1."
        )

    if candidate.min_duration_seconds <= 0:
        errors.append(
            "candidate_generation.min_duration_seconds must be positive."
        )

    if candidate.min_duration_seconds >= candidate.max_duration_seconds:
        errors.append(
            "candidate_generation.min_duration_seconds must be "
            "less than max_duration_seconds."
        )

    if candidate.max_duration_seconds > 60:
        errors.append(
            "candidate_generation.max_duration_seconds cannot exceed "
            "60 seconds for this assignment."
        )

    if candidate.min_dialogue_segments < 1:
        errors.append(
            "candidate_generation.min_dialogue_segments must be >= 1."
        )

    if not candidate.window_durations:
        errors.append(
            "candidate_generation.window_durations cannot be empty."
        )

    if any(
        float(duration) <= 0
        for duration in candidate.window_durations
    ):
        errors.append(
            "All candidate window durations must be positive."
        )

    if not 0.0 <= cfg.ranking.min_score_threshold <= 1.0:
        errors.append(
            "ranking.min_score_threshold must be in [0, 1]."
        )

    if not 0.0 < cfg.ranking.diversity.iou_threshold <= 1.0:
        errors.append(
            "ranking.diversity.iou_threshold must be in (0, 1]."
        )

    if cfg.output.max_evidence_segments < 0:
        errors.append(
            "output.max_evidence_segments cannot be negative."
        )

    # Validate every LLM stage.
    _validate_llm_config(
        cfg.llm_stages.episode_analysis,
        "llm.episode_analysis",
        errors,
    )

    _validate_llm_config(
        cfg.llm_stages.boundary_optimization,
        "llm.boundary_optimization",
        errors,
    )

    _validate_llm_config(
        cfg.llm_stages.promo_evaluation,
        "llm.promo_evaluation",
        errors,
    )

    # Validate feature weights.
    if cfg.feature_weights:

        for name, value in cfg.feature_weights.items():

            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                errors.append(
                    f"feature_weights.{name} must be numeric."
                )
                continue

            if numeric_value < 0:
                errors.append(
                    f"feature_weights.{name} cannot be negative."
                )

    if errors:

        raise ValueError(
            "Configuration errors:\n  "
            + "\n  ".join(errors)
        )


# ---------------------------------------------------------------------------
# CLI overrides
# ---------------------------------------------------------------------------

def apply_overrides(
    cfg: Config,
    overrides: dict,
) -> None:
    """
    Apply CLI-level overrides in place.

    Supported:
        top_k
        llm_enabled
    """

    if (
        "top_k" in overrides
        and overrides["top_k"] is not None
    ):
        cfg.top_k = int(overrides["top_k"])

        if cfg.top_k < 1:
            raise ValueError(
                "top_k override must be >= 1."
            )

    if (
        "llm_enabled" in overrides
        and overrides["llm_enabled"] is not None
    ):
        cfg.llm_enabled = bool(
            overrides["llm_enabled"]
        )





