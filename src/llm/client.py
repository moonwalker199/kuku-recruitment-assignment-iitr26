
"""
src/llm/client.py

Provider-independent LLM API client with retries and graceful fallback.

Supports:
  - Google Gemini via google-genai
  - OpenAI via openai

Supports independent configuration for:
  - episode_analysis
  - boundary_optimization
  - promo_evaluation

API keys are loaded from environment variables.
Keys are never logged or written to output files.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional provider imports
# ---------------------------------------------------------------------------

try:
    from google import genai as _genai
    from google.genai import types as _genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _genai = None
    _genai_types = None
    _GENAI_AVAILABLE = False


try:
    from openai import OpenAI as _OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OpenAI = None
    _OPENAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LLMUnavailable(Exception):
    """Raised when the LLM cannot be initialised."""


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class _BaseClient:

    def call(self, prompt: str) -> str | None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Google Gemini client
# ---------------------------------------------------------------------------

class _GeminiClient(_BaseClient):

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
        max_retries: int,
        retry_delay: float,
    ) -> None:

        if not _GENAI_AVAILABLE:
            raise LLMUnavailable(
                "google-genai is not installed. "
                "Run: pip install -U google-genai"
            )

        api_key = (
            os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )

        if not api_key:
            raise LLMUnavailable(
                "No Google API key found. "
                "Set GOOGLE_API_KEY or GEMINI_API_KEY in .env."
            )

        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        try:
            self._client = _genai.Client(
                api_key=api_key,
                http_options=_genai_types.HttpOptions(
                    timeout=int(timeout * 1000)
                ),
            )
        except Exception as exc:
            raise LLMUnavailable(
                f"Could not initialise Google Gemini client: {exc}"
            ) from exc

        logger.info(
            "Gemini client initialised (model=%s).",
            model,
        )

    def call(self, prompt: str) -> str | None:

        for attempt in range(1, self._max_retries + 1):

            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=_genai_types.GenerateContentConfig(
                        temperature=self._temperature,
                        max_output_tokens=self._max_tokens,
                    ),
                )

                text = response.text

                if not text:
                    logger.error(
                        "Gemini returned an empty text response."
                    )
                    return None

                return text

            except Exception as exc:

                err_str = str(exc).lower()

                if (
                    attempt < self._max_retries
                    and _is_transient(err_str)
                ):
                    delay = self._retry_delay * (
                        2 ** (attempt - 1)
                    )

                    logger.warning(
                        "Gemini request failed "
                        "(attempt %d/%d). Retrying in %.1fs. Error: %s",
                        attempt,
                        self._max_retries,
                        delay,
                        exc,
                    )

                    time.sleep(delay)

                else:
                    logger.error(
                        "Gemini request failed after %d attempt(s): %s",
                        attempt,
                        exc,
                    )
                    return None

        return None


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

class _OpenAIClient(_BaseClient):

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
        max_retries: int,
        retry_delay: float,
    ) -> None:

        if not _OPENAI_AVAILABLE:
            raise LLMUnavailable(
                "openai is not installed. Run: pip install openai"
            )

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise LLMUnavailable(
                "No OpenAI API key found. "
                "Set OPENAI_API_KEY in .env."
            )

        self._client = _OpenAI(
            api_key=api_key,
            timeout=timeout,
        )

        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        logger.info(
            "OpenAI client initialised (model=%s).",
            model,
        )

    def call(self, prompt: str) -> str | None:

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert promotional consultant "
                    "for short-form drama content."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        for attempt in range(1, self._max_retries + 1):

            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )

                content = response.choices[0].message.content

                return content

            except Exception as exc:

                err_str = str(exc).lower()

                if (
                    attempt < self._max_retries
                    and _is_transient(err_str)
                ):
                    delay = self._retry_delay * (
                        2 ** (attempt - 1)
                    )

                    logger.warning(
                        "OpenAI request failed "
                        "(attempt %d/%d). Retrying in %.1fs. Error: %s",
                        attempt,
                        self._max_retries,
                        delay,
                        exc,
                    )

                    time.sleep(delay)

                else:
                    logger.error(
                        "OpenAI request failed after %d attempt(s): %s",
                        attempt,
                        exc,
                    )
                    return None

        return None


# ---------------------------------------------------------------------------
# Factory and public interface
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Provider-independent LLM wrapper.

    Usage:
        # Backward-compatible default
        client = LLMClient.from_config(cfg)

        # Explicit stage configuration
        client = LLMClient.from_config(
            cfg,
            stage="episode_analysis",
        )

        raw = client.call(prompt)
        parsed = client.call_json(prompt)
    """

    VALID_STAGES = (
        "episode_analysis",
        "boundary_optimization",
        "promo_evaluation",
    )

    def __init__(
        self,
        inner: _BaseClient,
        stage: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:

        self._inner = inner
        self.stage = stage
        self.provider = provider
        self.model = model

    @classmethod
    def from_config(
        cls,
        cfg,
        stage: str | None = None,
    ) -> "LLMClient":
        """
        Build a client using the requested stage configuration.

        If stage is omitted, episode_analysis is used.
        Older configurations exposing only cfg.llm remain supported.
        """

        selected_stage = stage or "episode_analysis"

        if selected_stage not in cls.VALID_STAGES:
            raise ValueError(
                f"Unknown LLM stage: {selected_stage!r}. "
                f"Valid stages: {', '.join(cls.VALID_STAGES)}"
            )

        # Prefer the stage-aware configuration interface.
        if hasattr(cfg, "get_llm_config"):
            llm = cfg.get_llm_config(selected_stage)
        else:
            # Backward compatibility with older Config objects.
            llm = cfg.llm

        kwargs = {
            "model": llm.model,
            "temperature": llm.temperature,
            "max_tokens": llm.max_output_tokens,
            "timeout": llm.timeout_seconds,
            "max_retries": llm.max_retries,
            "retry_delay": llm.retry_delay_seconds,
        }

        if llm.provider == "google":
            inner = _GeminiClient(**kwargs)

        elif llm.provider == "openai":
            inner = _OpenAIClient(**kwargs)

        else:
            raise LLMUnavailable(
                f"Unknown LLM provider: {llm.provider!r}"
            )

        logger.info(
            "LLM client created for stage=%s, provider=%s, model=%s.",
            selected_stage,
            llm.provider,
            llm.model,
        )

        return cls(
            inner=inner,
            stage=selected_stage,
            provider=llm.provider,
            model=llm.model,
        )

    def call(self, prompt: str) -> str | None:
        """Send a prompt and return raw response text."""
        return self._inner.call(prompt)

    def call_json(self, prompt: str) -> dict | None:
        """Send a prompt and parse a JSON object response."""

        raw = self.call(prompt)

        if raw is None:
            return None

        return _parse_json_response(raw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TRANSIENT_INDICATORS = (
    "429",
    "rate limit",
    "quota",
    "500",
    "502",
    "503",
    "504",
    "timeout",
    "connection",
    "unavailable",
    "overloaded",
    "try again",
)


def _is_transient(error_str: str) -> bool:
    return any(
        indicator in error_str
        for indicator in _TRANSIENT_INDICATORS
    )


def _parse_json_response(text: str) -> dict | None:
    """Extract a JSON object from a model response."""

    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
        else:
            text = "\n".join(lines[1:])

        text = text.strip()

    try:
        result = json.loads(text)

        if not isinstance(result, dict):
            logger.warning(
                "LLM returned JSON but not a dictionary."
            )
            return None

        return result

    except json.JSONDecodeError as exc:
        logger.warning(
            "LLM response is not valid JSON: %s",
            exc,
        )
        return None



