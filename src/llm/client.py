
"""
Provider-independent LLM API client with retries and graceful fallback.

Supports Google Gemini via google-genai and OpenAI via openai.
Supports independent episode_analysis, boundary_optimization,
and promo_evaluation configurations.
"""

from __future__ import annotations

import re
import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)


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


class LLMUnavailable(Exception):
    """Raised when the LLM cannot be initialised."""


class _BaseClient:

    def call(self, prompt: str) -> str | None:
        raise NotImplementedError


class _GeminiClient(_BaseClient):

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
        max_retries: int,
        retry_delay: float,
        thinking_level: str | None = None,
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
        self._thinking_level = thinking_level

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

        logger.info("Gemini client initialised (model=%s).", model)

    def call(self, prompt: str) -> str | None:

        for attempt in range(1, self._max_retries + 1):

            try:
                # response = self._client.models.generate_content(
                #     model=self._model,
                #     contents=prompt,
                #     config=_genai_types.GenerateContentConfig(
                #         temperature=self._temperature,
                #         max_output_tokens=self._max_tokens,
                #         response_mime_type="application/json",
                #     ),
                # )

                config_kwargs = {
                    "temperature": self._temperature,
                    "max_output_tokens": self._max_tokens,
                    "response_mime_type": "application/json",
                }

                if self._thinking_level:
                    config_kwargs["thinking_config"] = (
                        _genai_types.ThinkingConfig(
                            thinking_level=self._thinking_level
                        )
                    )

                response = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=_genai_types.GenerateContentConfig(
                        **config_kwargs
                    ),
                )

                candidates = getattr(response, "candidates", None)

                if candidates:
                    finish_reason = getattr(
                        candidates[0], "finish_reason", None
                    )
                    logger.info(
                        "Gemini finish reason: %s",
                        finish_reason,
                    )

                usage = getattr(response, "usage_metadata", None)

                # if usage:
                #     logger.info(
                #         "Gemini token usage: prompt=%s, response=%s, total=%s",
                #         getattr(usage, "prompt_token_count", None),
                #         getattr(usage, "candidates_token_count", None),
                #         getattr(usage, "total_token_count", None),
                #     )
                if usage:
                    logger.info(
                        "Gemini token usage: prompt=%s, response=%s, "
                        "thinking=%s, total=%s",
                        getattr(usage, "prompt_token_count", None),
                        getattr(usage, "candidates_token_count", None),
                        getattr(usage, "thoughts_token_count", None),
                        getattr(usage, "total_token_count", None),
                    )

                text = response.text

                if not text:
                    logger.error(
                        "Gemini returned an empty response. "
                        "Finish reason: %s",
                        finish_reason if candidates else "unknown",
                    )
                    return None

                return text.strip()

            # except Exception as exc:

            #     err_str = str(exc).lower()

            #     if (
            #         attempt < self._max_retries
            #         and _is_transient(err_str)
            #     ):
            #         delay = self._retry_delay * (
            #             2 ** (attempt - 1)
            #         )

            #         logger.warning(
            #             "Gemini request failed "
            #             "(attempt %d/%d). Retrying in %.1fs. Error: %s",
            #             attempt,
            #             self._max_retries,
            #             delay,
            #             exc,
            #         )

            #         time.sleep(delay)

            #     else:
            #         logger.error(
            #             "Gemini request failed after %d attempt(s): %s",
            #             attempt,
            #             exc,
            #         )
            #         return None
            except Exception as exc:

                err_str = str(exc).lower()

                if (
                    attempt < self._max_retries
                    and _is_transient(err_str)
                ):
                    delay = _get_retry_delay(
                        exc,
                        self._retry_delay,
                        attempt,
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

        logger.info("OpenAI client initialised (model=%s).", model)

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

                if not content:
                    logger.error("OpenAI returned an empty response.")
                    return None

                return content.strip()

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


class LLMClient:
    """Provider-independent LLM wrapper."""

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

        selected_stage = stage or "episode_analysis"

        if selected_stage not in cls.VALID_STAGES:
            raise ValueError(
                f"Unknown LLM stage: {selected_stage!r}. "
                f"Valid stages: {', '.join(cls.VALID_STAGES)}"
            )

        if hasattr(cfg, "get_llm_config"):
            llm = cfg.get_llm_config(selected_stage)
        else:
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
            inner = _GeminiClient(
                **kwargs,
                thinking_level=getattr(llm, "thinking_level", None),
            )


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
        return self._inner.call(prompt)

    def call_json(self, prompt: str) -> dict | None:
        raw = self.call(prompt)

        if raw is None:
            return None

        return _parse_json_response(raw)


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


def _get_retry_delay(exc: Exception, fallback: float, attempt: int) -> float:
    """Extract Google's suggested retry delay, or use exponential backoff."""

    error_text = str(exc).lower()

    patterns = [
        r"retry in\s+(\d+(?:\.\d+)?)\s*s",
        r"retry_delay\s*\{\s*seconds:\s*(\d+)",
        r"retrydelay.*?(\d+(?:\.\d+)?)\s*s",
    ]

    for pattern in patterns:
        match = re.search(pattern, error_text, re.DOTALL)

        if match:
            return max(float(match.group(1)), 1.0)

    # Rate limits need a longer fallback than ordinary transient errors.
    if "429" in error_text or "resource_exhausted" in error_text:
        return max(30.0, fallback * (2 ** (attempt - 1)))

    return fallback * (2 ** (attempt - 1))

def _is_transient(error_str: str) -> bool:
    error_str = error_str.lower()

    # Daily quota exhaustion cannot be resolved by short retries.
    permanent_quota_indicators = (
        "per day limit exceeded",
        "requests per day limit exceeded",
        "generate_content_free_tier_requests",
        "generaterequestsperday",
    )

    if any(
        indicator in error_str
        for indicator in permanent_quota_indicators
    ):
        return False

    return any(
        indicator in error_str
        for indicator in _TRANSIENT_INDICATORS
    )


def _parse_json_response(text: str) -> dict | None:
    """Parse a JSON object without accepting arbitrary trailing text."""

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

    except json.JSONDecodeError as exc:
        logger.warning(
            "LLM response is not valid JSON: %s",
            exc,
        )
        logger.debug(
            "Invalid LLM response preview: %r",
            text[:1000],
        )
        return None

    if not isinstance(result, dict):
        logger.warning("LLM returned JSON but not a dictionary.")
        return None

    return result
