"""
Provider-independent LLM API client with retries and graceful fallback.

Supports Google Gemini via google-genai and OpenAI via openai.
Supports independent episode_analysis, boundary_optimization,
and promo_evaluation configurations.

Gemini rate limits:
    Hard daily cap:       500 requests
    Operating RPM target: 10 requests per rolling minute
    Operating TPM target: 200,000 estimated input tokens per rolling minute

Daily usage is persisted across program restarts. Google's daily quota
resets at midnight Pacific Time.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time

from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Optional provider imports
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------

class LLMUnavailable(Exception):
    """Raised when the LLM cannot be initialised."""


class LLMRateLimitExceeded(Exception):
    """Raised when the configured Gemini request budget is exhausted."""


# ---------------------------------------------------------------------
# Gemini rate limiter
# ---------------------------------------------------------------------

class SlidingWindowRateLimiter:
    """
    Shared, persistent Gemini API rate limiter.

    Limits are enforced before every actual API attempt, including retries.

    Input token usage is estimated locally using UTF-8 prompt length.
    Actual usage reported by Gemini updates the reservation afterward.
    """

    def __init__(
        self,
        max_requests_per_minute: int = 10,
        max_input_tokens_per_minute: int = 200_000,
        max_requests_per_day: int = 500,
        window_seconds: int = 60,
        state_file: str = "outputs/cache/gemini_rate_limit.json",
    ) -> None:

        self.max_requests_per_minute = max_requests_per_minute
        self.max_input_tokens_per_minute = max_input_tokens_per_minute
        self.max_requests_per_day = max_requests_per_day
        self.window_seconds = window_seconds

        self.state_file = Path(state_file)
        self.lock = threading.Lock()

        # Entries contain:
        # {"timestamp": float, "tokens": int}
        self.requests = deque()

        self.daily_date = self._get_pacific_date()
        self.daily_requests = 0

        self._load_state()

        logger.info(
            "Gemini rate limiter initialised: "
            "%d RPM, %d estimated TPM, %d RPD.",
            self.max_requests_per_minute,
            self.max_input_tokens_per_minute,
            self.max_requests_per_day,
        )

    @staticmethod
    def _get_pacific_date() -> str:
        """Get the date used by Google's Pacific Time daily quota reset."""

        return datetime.now(
            ZoneInfo("America/Los_Angeles")
        ).date().isoformat()

    @staticmethod
    def estimate_tokens(prompt: str) -> int:
        """
        Estimate input tokens conservatively.

        This is an estimate, not an exact Gemini tokenizer count.
        Actual prompt token usage is logged and used to update the
        rolling-window reservation after a successful response.
        """

        prompt_bytes = len(prompt.encode("utf-8"))

        return max(1, math.ceil(prompt_bytes / 3))

    def _load_state(self) -> None:
        """Load persisted daily usage and rolling-window reservations."""

        if not self.state_file.exists():
            return

        try:
            with self.state_file.open("r", encoding="utf-8") as file:
                state = json.load(file)

            stored_date = state.get("daily_date")
            current_date = self._get_pacific_date()

            if stored_date == current_date:
                self.daily_date = stored_date
                self.daily_requests = int(
                    state.get("daily_requests", 0)
                )

            now = time.time()

            for item in state.get("requests", []):
                timestamp = float(item["timestamp"])
                tokens = int(item["tokens"])

                if (
                    now - timestamp < self.window_seconds
                    and tokens >= 0
                ):
                    self.requests.append(
                        {
                            "timestamp": timestamp,
                            "tokens": tokens,
                        }
                    )

            logger.info(
                "Restored Gemini API usage: %d/%d daily requests.",
                self.daily_requests,
                self.max_requests_per_day,
            )

        except Exception as exc:
            # Fail closed rather than accidentally resetting a daily cap.
            raise LLMUnavailable(
                "Could not read Gemini rate-limit state. "
                f"Check {self.state_file}: {exc}"
            ) from exc

    def _save_state(self) -> None:
        """Persist daily usage and recent rolling-window reservations."""

        self.state_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        state = {
            "daily_date": self.daily_date,
            "daily_requests": self.daily_requests,
            "requests": list(self.requests),
        }

        temporary_file = self.state_file.with_suffix(".tmp")

        with temporary_file.open("w", encoding="utf-8") as file:
            json.dump(state, file, indent=2)

        os.replace(temporary_file, self.state_file)

    def _refresh_daily_counter(self) -> None:
        """Reset the daily counter when the Pacific calendar date changes."""

        current_date = self._get_pacific_date()

        if current_date != self.daily_date:
            logger.info(
                "Gemini daily quota date changed from %s to %s.",
                self.daily_date,
                current_date,
            )

            self.daily_date = current_date
            self.daily_requests = 0

    def _remove_expired_requests(self, now: float) -> None:
        """Remove requests outside the rolling 60-second window."""

        while self.requests:
            oldest = self.requests[0]

            if now - oldest["timestamp"] >= self.window_seconds:
                self.requests.popleft()
            else:
                break

    def acquire(self, prompt: str) -> dict[str, Any]:
        """
        Wait until a request fits the configured limits.

        The daily request is reserved before the API call. Failed attempts
        and retries therefore count toward the daily request budget.
        """

        estimated_tokens = self.estimate_tokens(prompt)

        if estimated_tokens > self.max_input_tokens_per_minute:
            raise LLMRateLimitExceeded(
                f"Prompt estimate ({estimated_tokens:,} tokens) exceeds "
                f"the configured rolling TPM budget "
                f"({self.max_input_tokens_per_minute:,})."
            )

        with self.lock:

            while True:
                now = time.time()

                self._refresh_daily_counter()
                self._remove_expired_requests(now)

                # Hard daily request cap.
                if self.daily_requests >= self.max_requests_per_day:
                    self._save_state()

                    raise LLMRateLimitExceeded(
                        "Gemini daily request budget exhausted: "
                        f"{self.daily_requests}/"
                        f"{self.max_requests_per_day}. "
                        "Wait for the next Pacific Time quota reset."
                    )

                requests_in_window = len(self.requests)

                tokens_in_window = sum(
                    item["tokens"]
                    for item in self.requests
                )

                rpm_reached = (
                    requests_in_window
                    >= self.max_requests_per_minute
                )

                tpm_reached = (
                    tokens_in_window + estimated_tokens
                    > self.max_input_tokens_per_minute
                )

                if not rpm_reached and not tpm_reached:
                    reservation = {
                        "timestamp": now,
                        "tokens": estimated_tokens,
                    }

                    self.requests.append(reservation)
                    self.daily_requests += 1

                    self._save_state()

                    logger.info(
                        "Gemini request reserved: daily=%d/%d, "
                        "rolling RPM=%d/%d, estimated TPM=%d/%d.",
                        self.daily_requests,
                        self.max_requests_per_day,
                        len(self.requests),
                        self.max_requests_per_minute,
                        tokens_in_window + estimated_tokens,
                        self.max_input_tokens_per_minute,
                    )

                    return reservation

                wait_times = []

                if rpm_reached:
                    oldest_timestamp = self.requests[0]["timestamp"]

                    wait_times.append(
                        self.window_seconds
                        - (now - oldest_timestamp)
                    )

                if tpm_reached:
                    remaining_tokens = tokens_in_window

                    for item in self.requests:
                        remaining_tokens -= item["tokens"]

                        if (
                            remaining_tokens + estimated_tokens
                            <= self.max_input_tokens_per_minute
                        ):
                            wait_times.append(
                                self.window_seconds
                                - (now - item["timestamp"])
                            )
                            break

                wait_seconds = max(
                    0.1,
                    max(wait_times, default=1.0),
                )

                logger.info(
                    "Gemini rolling rate limit reached. "
                    "Waiting %.1f seconds.",
                    wait_seconds,
                )

                # Release the lock while waiting.
                self.lock.release()

                try:
                    time.sleep(wait_seconds)
                finally:
                    self.lock.acquire()

    def record_actual_usage(
        self,
        reservation: dict[str, Any],
        actual_tokens: int | None,
    ) -> None:
        """Replace the estimated token reservation with reported usage."""

        if actual_tokens is None:
            return

        try:
            actual_tokens = int(actual_tokens)

            if actual_tokens <= 0:
                return

            with self.lock:
                target_timestamp = reservation["timestamp"]

                for item in self.requests:
                    if item["timestamp"] == target_timestamp:
                        item["tokens"] = actual_tokens
                        break

                self._save_state()

            logger.info(
                "Gemini actual input usage: %d tokens.",
                actual_tokens,
            )

        except Exception:
            logger.exception(
                "Could not update Gemini actual token usage."
            )


# One shared limiter for all Gemini clients and pipeline stages.
_GEMINI_RATE_LIMITER = SlidingWindowRateLimiter()


# ---------------------------------------------------------------------
# Base provider client
# ---------------------------------------------------------------------

class _BaseClient:

    def call(self, prompt: str) -> str | None:
        raise NotImplementedError


# ---------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------

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
        api_key: str | None = None,
        
    ) -> None:

        if not _GENAI_AVAILABLE:
            raise LLMUnavailable(
                "google-genai is not installed. "
                "Run: pip install -U google-genai"
            )

        # api_key = (
        #     os.getenv("GOOGLE_API_KEY")
        #     or os.getenv("GEMINI_API_KEY")
        # )

        api_key = (
            api_key
            or os.getenv("GOOGLE_API_KEY")
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

        logger.info(
            "Gemini client initialised (model=%s).",
            model,
        )

    def call(self, prompt: str) -> str | None:

        for attempt in range(1, self._max_retries + 1):

            try:
                # Every attempt, including retries, passes through
                # the shared RPM, TPM and RPD limiter.
                reservation = _GEMINI_RATE_LIMITER.acquire(prompt)

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

                candidates = getattr(
                    response,
                    "candidates",
                    None,
                )

                finish_reason = "unknown"

                if candidates:
                    finish_reason = getattr(
                        candidates[0],
                        "finish_reason",
                        "unknown",
                    )

                    logger.info(
                        "Gemini finish reason: %s",
                        finish_reason,
                    )

                usage = getattr(
                    response,
                    "usage_metadata",
                    None,
                )

                if usage:
                    prompt_tokens = getattr(
                        usage,
                        "prompt_token_count",
                        None,
                    )

                    logger.info(
                        "Gemini token usage: prompt=%s, response=%s, "
                        "thinking=%s, total=%s",
                        prompt_tokens,
                        getattr(
                            usage,
                            "candidates_token_count",
                            None,
                        ),
                        getattr(
                            usage,
                            "thoughts_token_count",
                            None,
                        ),
                        getattr(
                            usage,
                            "total_token_count",
                            None,
                        ),
                    )

                    _GEMINI_RATE_LIMITER.record_actual_usage(
                        reservation,
                        prompt_tokens,
                    )

                text = response.text

                if not text:
                    logger.error(
                        "Gemini returned an empty response. "
                        "Finish reason: %s",
                        finish_reason,
                    )
                    return None

                return text.strip()

            except LLMRateLimitExceeded as exc:
                logger.error("Gemini request blocked by local limiter: %s", exc)
                return None

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


# ---------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------
class _OpenAIClient(_BaseClient):

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
        max_retries: int,
        retry_delay: float,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:

        if not _OPENAI_AVAILABLE:
            raise LLMUnavailable(
                "openai is not installed. Run: pip install openai"
            )

        resolved_api_key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
        )

        if not resolved_api_key:
            raise LLMUnavailable(
                "No API key provided. Enter your API key "
                "or configure OPENAI_API_KEY."
            )

        client_kwargs = {
            "api_key": resolved_api_key,
            "timeout": timeout,
        }

        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = _OpenAI(**client_kwargs)

        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        logger.info(
            "OpenAI-compatible client initialised (model=%s, custom_endpoint=%s).",
            model,
            bool(base_url),
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

                if not content:
                    logger.error(
                        "OpenAI-compatible API returned an empty response."
                    )
                    return None

                return content.strip()

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
                        "API request failed (attempt %d/%d). "
                        "Retrying in %.1fs. Error: %s",
                        attempt,
                        self._max_retries,
                        delay,
                        exc,
                    )

                    time.sleep(delay)

                else:
                    logger.error(
                        "API request failed after %d attempt(s): %s",
                        attempt,
                        exc,
                    )
                    return None

        return None


# ---------------------------------------------------------------------
# Provider-independent wrapper
# ---------------------------------------------------------------------

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
        api_key: str | None = None,
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
                thinking_level=getattr(
                    llm,
                    "thinking_level",
                    None,
                ),
                api_key=api_key,
            )

        elif llm.provider in ("openai", "openai_compatible"):

            inner = _OpenAIClient(
                **kwargs,
                api_key=api_key,
                base_url=getattr(llm, "base_url", None),
            )

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


# ---------------------------------------------------------------------
# Retry handling
# ---------------------------------------------------------------------

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


def _get_retry_delay(
    exc: Exception,
    fallback: float,
    attempt: int,
) -> float:
    """Extract Google's suggested retry delay, or use exponential backoff."""

    error_text = str(exc).lower()

    patterns = [
        r"retry in\s+(\d+(?:\.\d+)?)\s*s",
        r"retry_delay\s*\{\s*seconds:\s*(\d+)",
        r"retrydelay.*?(\d+(?:\.\d+)?)\s*s",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            error_text,
            re.DOTALL,
        )

        if match:
            return max(float(match.group(1)), 1.0)

    # Rate limits need a longer fallback than ordinary transient errors.
    if "429" in error_text or "resource_exhausted" in error_text:
        return max(
            30.0,
            fallback * (2 ** (attempt - 1)),
        )

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


# ---------------------------------------------------------------------
# JSON response parsing
# ---------------------------------------------------------------------

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
        logger.warning(
            "LLM returned JSON but not a dictionary."
        )
        return None

    return result