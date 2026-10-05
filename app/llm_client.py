from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, TypeVar

import openai
from openai import OpenAI
from pydantic import BaseModel

from .config import Settings
from .models import SupportResponse
from .prompts import build_system_prompt, build_user_prompt

try:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - dependency is installed in project env
    genai = None
    genai_errors = None
    genai_types = None

ModelType = TypeVar("ModelType", bound=BaseModel)


class LLMError(RuntimeError):
    """Base application error for LLM provider failures."""


class LLMPermanentError(LLMError):
    """Raised for persistent client/configuration failures that should not be retried."""


class LLMTransientError(LLMError):
    """Raised when a transient provider error exhausts the retry budget."""


def calculate_backoff(
    attempt_number: int,
    *,
    initial_backoff_seconds: float = 1.0,
    max_backoff_seconds: float = 8.0,
) -> float:
    """Return the exponential backoff delay for a given retry attempt."""
    if attempt_number < 1:
        return 0.0
    delay = float(initial_backoff_seconds) * (2 ** (attempt_number - 1))
    return min(delay, float(max_backoff_seconds))


def is_transient_error(exc: Exception) -> bool:
    """Return True when the exception represents a retryable transient provider failure."""
    if isinstance(exc, (TimeoutError, ConnectionError, openai.APIConnectionError, openai.APITimeoutError)):
        return True

    if isinstance(exc, openai.RateLimitError):
        return True

    if isinstance(exc, openai.APIStatusError):
        status_code = getattr(exc, "status_code", None)
        return status_code in {429, 500, 502, 503, 504}

    if genai_errors is not None and isinstance(exc, genai_errors.APIError):
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            status_code = getattr(exc, "code", None)
        return status_code in {429, 500, 502, 503, 504}

    status_code = getattr(exc, "status_code", None)
    if status_code is None and hasattr(exc, "response"):
        status_code = getattr(exc.response, "status_code", None)
    if status_code in {429, 500, 502, 503, 504}:
        return True

    message = str(exc).lower()
    if any(token in message for token in ("rate limit", "timed out", "timeout", "temporarily unavailable", "503", "502", "504")):
        return True

    return False


class LLMClient:
    """Thin abstraction around the supported LLM SDKs and future retry/fallback logic."""

    def __init__(
        self,
        *,
        provider: str = "openai",
        api_key: str | None = None,
        model: str | None = None,
        fallback_model: str | None = None,
        max_retries: int | None = None,
        initial_backoff_seconds: float | None = None,
        max_backoff_seconds: float | None = None,
        max_output_tokens: int | None = None,
        client: Any | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.provider = (provider or "openai").lower()
        self.api_key = api_key or ""
        self.model = model or ("gemini-2.0-flash" if self.provider == "gemini" else "gpt-4o-mini")
        self.fallback_model = fallback_model or self.model
        self.max_retries = max_retries if max_retries is not None else 3
        self.initial_backoff_seconds = float(initial_backoff_seconds if initial_backoff_seconds is not None else 1.0)
        self.max_backoff_seconds = float(max_backoff_seconds if max_backoff_seconds is not None else 8.0)
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            self.max_backoff_seconds = self.initial_backoff_seconds
        self.max_output_tokens = max_output_tokens if max_output_tokens is not None else 512
        self.client = client or self._build_client()
        self.logger = logger or logging.getLogger(__name__)

    def _build_client(self) -> Any:
        if self.provider == "openai":
            return OpenAI(api_key=self.api_key)
        if self.provider == "gemini":
            if genai is None:
                raise RuntimeError("The google-genai package is required when LLM_PROVIDER is set to 'gemini'.")
            return genai.Client(api_key=self.api_key)
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    @classmethod
    def from_settings(cls, settings: Settings) -> "LLMClient":
        """Build an LLM client from the application settings object."""
        max_retries = settings.LLM_MAX_RETRIES if settings.LLM_MAX_RETRIES is not None else settings.OPENAI_MAX_RETRIES
        if settings.LLM_PROVIDER == "gemini":
            return cls(
                provider="gemini",
                api_key=settings.GEMINI_API_KEY,
                model=settings.GEMINI_MODEL,
                max_retries=max_retries,
                initial_backoff_seconds=settings.LLM_INITIAL_BACKOFF_SECONDS,
                max_backoff_seconds=settings.LLM_MAX_BACKOFF_SECONDS,
                max_output_tokens=settings.OPENAI_MAX_OUTPUT_TOKENS,
            )

        return cls(
            provider="openai",
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            fallback_model=settings.OPENAI_FALLBACK_MODEL,
            max_retries=max_retries,
            initial_backoff_seconds=settings.LLM_INITIAL_BACKOFF_SECONDS,
            max_backoff_seconds=settings.LLM_MAX_BACKOFF_SECONDS,
            max_output_tokens=settings.OPENAI_MAX_OUTPUT_TOKENS,
        )

    def _sanitize_exception(self, message: str) -> str:
        return message.replace(self.api_key, "[REDACTED]") if self.api_key else message

    def _execute_with_retry(self, operation: Callable[[], Any], *, provider_name: str) -> Any:
        """Retry transient provider errors using exponential backoff."""
        for attempt in range(self.max_retries + 1):
            try:
                return operation()
            except Exception as exc:
                if not is_transient_error(exc):
                    if isinstance(exc, openai.OpenAIError):
                        raise
                    if genai_errors is not None and isinstance(exc, (genai_errors.APIError, genai_errors.ClientError)):
                        raise
                    raise LLMPermanentError(
                        f"{provider_name} request failed: {self._sanitize_exception(str(exc))}"
                    ) from exc
                if attempt >= self.max_retries:
                    raise LLMTransientError(
                        f"{provider_name} request failed: max retries reached ({self.max_retries}) - "
                        f"{self._sanitize_exception(str(exc))}"
                    ) from exc

                delay = calculate_backoff(
                    attempt + 1,
                    initial_backoff_seconds=self.initial_backoff_seconds,
                    max_backoff_seconds=self.max_backoff_seconds,
                )
                self.logger.warning(
                    "%s transient failure on attempt %s/%s. Retrying in %.1f seconds.",
                    provider_name,
                    attempt + 1,
                    self.max_retries + 1,
                    delay,
                )
                time.sleep(delay)

        raise LLMTransientError(f"{provider_name} request failed: max retries reached.")

    def _validate_structured_response(self, payload: Any, *, provider_name: str) -> SupportResponse:
        try:
            if isinstance(payload, SupportResponse):
                return payload
            return SupportResponse.model_validate(payload)
        except Exception as exc:  # pragma: no cover - defensive validation guard
            raise ValueError(f"invalid structured data from {provider_name} API: {exc}") from exc

    def _process_openai_message(self, customer_message: str) -> SupportResponse:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required to use the OpenAI provider")
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(customer_message)

        def request() -> Any:
            return self.client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                text_format=SupportResponse,
                max_output_tokens=self.max_output_tokens,
            )

        response = self._execute_with_retry(request, provider_name="OpenAI")
        parsed_response = getattr(response, "output_parsed", None)
        if parsed_response is None:
            raise ValueError("The OpenAI API returned no parsed structured data.")

        return self._validate_structured_response(parsed_response, provider_name="OpenAI")

    def _process_gemini_message(self, customer_message: str) -> SupportResponse:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required to use the Gemini provider")

        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(customer_message)
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"

        def request() -> Any:
            if genai_types is not None:
                return self.client.models.generate_content(
                    model=self.model,
                    contents=combined_prompt,
                    config=genai_types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=SupportResponse,
                    ),
                )
            return self.client.models.generate_content(
                model=self.model,
                contents=combined_prompt,
            )

        response = self._execute_with_retry(request, provider_name="Gemini")
        parsed_response = getattr(response, "parsed", None)
        if parsed_response is None:
            raw_text = getattr(response, "text", None)
            if raw_text:
                try:
                    parsed_response = json.loads(raw_text)
                except json.JSONDecodeError as exc:
                    raise ValueError("The Gemini API returned invalid JSON structured data.") from exc
            else:
                raise ValueError("The Gemini API returned no parsed structured data.")

        return self._validate_structured_response(parsed_response, provider_name="Gemini")

    def process_message(self, customer_message: str) -> SupportResponse:
        """Send a customer message to the configured provider and validate the structured result."""
        cleaned_message = (customer_message or "").strip()
        if not cleaned_message:
            raise ValueError("customer_message must not be empty")

        if self.provider == "openai":
            return self._process_openai_message(cleaned_message)
        if self.provider == "gemini":
            return self._process_gemini_message(cleaned_message)

        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ModelType] | None = None,
    ) -> dict[str, Any] | ModelType:
        """Send a structured request to the configured SDK and return validated output."""
        if response_model is None:
            response_model = SupportResponse

        if self.provider == "openai":
            def request() -> Any:
                return self.client.responses.parse(
                    model=self.model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    text_format=response_model,
                    max_output_tokens=self.max_output_tokens,
                )

            response = self._execute_with_retry(request, provider_name="OpenAI")
            parsed_response = getattr(response, "output_parsed", None)
            if parsed_response is None:
                raise ValueError("The OpenAI API returned no parsed structured data.")
            return self._validate_structured_response(parsed_response, provider_name="OpenAI")

        if self.provider == "gemini":
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"

            def request() -> Any:
                if genai_types is not None:
                    return self.client.models.generate_content(
                        model=self.model,
                        contents=combined_prompt,
                        config=genai_types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=response_model,
                        ),
                    )
                return self.client.models.generate_content(
                    model=self.model,
                    contents=combined_prompt,
                )

            response = self._execute_with_retry(request, provider_name="Gemini")
            parsed_response = getattr(response, "parsed", None)
            if parsed_response is None:
                raw_text = getattr(response, "text", None)
                if raw_text:
                    try:
                        parsed_response = json.loads(raw_text)
                    except json.JSONDecodeError as exc:
                        raise ValueError("The Gemini API returned invalid JSON structured data.") from exc
                else:
                    raise ValueError("The Gemini API returned no parsed structured data.")

            return self._validate_structured_response(parsed_response, provider_name="Gemini")

        raise ValueError(f"Unsupported LLM provider: {self.provider}")
