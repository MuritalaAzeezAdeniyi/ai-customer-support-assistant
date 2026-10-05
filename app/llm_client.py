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
        fallback_provider: str | None = None,
        fallback_model: str | None = None,
        fallback_api_key: str | None = None,
        max_retries: int | None = None,
        initial_backoff_seconds: float | None = None,
        max_backoff_seconds: float | None = None,
        max_output_tokens: int | None = None,
        client: Any | None = None,
        fallback_client: Any | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.provider = (provider or "openai").lower()
        self.api_key = api_key or ""
        self.model = model or ("gemini-2.0-flash" if self.provider == "gemini" else "gpt-4o-mini")
        self.fallback_provider = (fallback_provider or None)
        self.fallback_model = fallback_model or None
        self.fallback_api_key = fallback_api_key or None
        self.fallback_client = fallback_client
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
        primary_model = settings.LLM_MODEL or (settings.GEMINI_MODEL if settings.LLM_PROVIDER == "gemini" else settings.OPENAI_MODEL)
        fallback_provider = settings.LLM_FALLBACK_PROVIDER
        fallback_model = settings.LLM_FALLBACK_MODEL or settings.OPENAI_FALLBACK_MODEL
        if settings.LLM_PROVIDER == "gemini":
            provider_specific_fallback = settings.LLM_FALLBACK_PROVIDER or None
            if fallback_provider is None and settings.LLM_FALLBACK_MODEL is None and settings.OPENAI_FALLBACK_MODEL is not None:
                fallback_provider = settings.LLM_PROVIDER
            if fallback_provider is None and fallback_model is None:
                fallback_provider = None
                fallback_model = None
            return cls(
                provider="gemini",
                api_key=settings.GEMINI_API_KEY,
                model=primary_model,
                fallback_provider=fallback_provider,
                fallback_model=fallback_model,
                fallback_api_key=settings.GEMINI_API_KEY,
                max_retries=max_retries,
                initial_backoff_seconds=settings.LLM_INITIAL_BACKOFF_SECONDS,
                max_backoff_seconds=settings.LLM_MAX_BACKOFF_SECONDS,
                max_output_tokens=settings.OPENAI_MAX_OUTPUT_TOKENS,
            )

        fallback_api_key = settings.OPENAI_API_KEY
        if fallback_provider == "gemini":
            fallback_api_key = settings.GEMINI_API_KEY
        if fallback_provider is None and settings.LLM_FALLBACK_MODEL is None and settings.OPENAI_FALLBACK_MODEL is not None:
            fallback_provider = settings.LLM_PROVIDER
        if fallback_provider is None and fallback_model is None:
            fallback_provider = None
            fallback_model = None
        return cls(
            provider="openai",
            api_key=settings.OPENAI_API_KEY,
            model=primary_model,
            fallback_provider=fallback_provider,
            fallback_model=fallback_model,
            fallback_api_key=fallback_api_key,
            max_retries=max_retries,
            initial_backoff_seconds=settings.LLM_INITIAL_BACKOFF_SECONDS,
            max_backoff_seconds=settings.LLM_MAX_BACKOFF_SECONDS,
            max_output_tokens=settings.OPENAI_MAX_OUTPUT_TOKENS,
        )

    def _sanitize_exception(self, message: str) -> str:
        sanitized = message
        for secret in {self.api_key, self.fallback_api_key}:
            if secret:
                sanitized = sanitized.replace(secret, "[REDACTED]")
        return sanitized

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

    def _has_fallback(self) -> bool:
        if self.fallback_client is not None:
            return True
        if not self.fallback_provider and not self.fallback_model and not self.fallback_api_key:
            return False
        fallback_provider = (self.fallback_provider or self.provider).lower()
        fallback_model = self.fallback_model or self.model
        return fallback_provider != self.provider or fallback_model != self.model

    def _build_fallback_client(self) -> "LLMClient | None":
        if not self._has_fallback():
            return None
        fallback_provider = (self.fallback_provider or self.provider).lower()
        fallback_model = self.fallback_model or self.model
        fallback_api_key = self.fallback_api_key or self.api_key
        return LLMClient(
            provider=fallback_provider,
            api_key=fallback_api_key,
            model=fallback_model,
            max_retries=self.max_retries,
            initial_backoff_seconds=self.initial_backoff_seconds,
            max_backoff_seconds=self.max_backoff_seconds,
            max_output_tokens=self.max_output_tokens,
            client=self.fallback_client,
            logger=self.logger,
        )

    def _execute_with_fallback(self, *, primary_action: Callable[[], Any], fallback_action: Callable[[], Any] | None) -> Any:
        try:
            return primary_action()
        except LLMTransientError:
            if fallback_action is None:
                raise
            try:
                return fallback_action()
            except LLMTransientError as fallback_exc:
                raise LLMTransientError(
                    f"Primary and fallback providers exhausted retries: {self._sanitize_exception(str(fallback_exc))}"
                ) from fallback_exc

    def _iter_stream_chunks(self, stream: Any) -> Any:
        try:
            iterator = iter(stream)
        except TypeError as exc:
            raise ValueError("The provider stream did not return an iterable payload.") from exc

        for item in iterator:
            if item is None:
                continue
            if isinstance(item, str):
                yield item
                continue
            if hasattr(item, "delta"):
                delta = getattr(item, "delta")
                if delta is not None:
                    if isinstance(delta, str):
                        yield delta
                        continue
                    if isinstance(delta, dict):
                        text = delta.get("content") or delta.get("text")
                        if text:
                            yield str(text)
                            continue
            if hasattr(item, "text"):
                text = getattr(item, "text")
                if text is not None:
                    yield str(text)
                    continue
            if hasattr(item, "content"):
                content = getattr(item, "content")
                if content is not None:
                    yield str(content)
                    continue
            if hasattr(item, "output_text"):
                output_text = getattr(item, "output_text")
                if output_text is not None:
                    yield str(output_text)
                    continue

    def _stream_openai_response(self, customer_message: str) -> Any:
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(customer_message)

        def request() -> Any:
            return self.client.responses.stream(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_output_tokens=self.max_output_tokens,
            )

        stream = self._execute_with_retry(request, provider_name="OpenAI")
        yield from self._iter_stream_chunks(stream)

    def _stream_gemini_response(self, customer_message: str) -> Any:
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(customer_message)
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"

        def request() -> Any:
            return self.client.models.generate_content_stream(
                model=self.model,
                contents=combined_prompt,
            )

        stream = self._execute_with_retry(request, provider_name="Gemini")
        yield from self._iter_stream_chunks(stream)

    def _stream_with_fallback(
        self,
        customer_message: str,
        *,
        primary_stream: Callable[[str], Any],
        allow_fallback: bool = True,
    ) -> Any:
        try:
            yield from primary_stream(customer_message)
            return
        except Exception as exc:
            if not is_transient_error(exc):
                raise
            if not allow_fallback:
                raise LLMTransientError(
                    f"{self.provider.capitalize()} request failed: max retries reached - {self._sanitize_exception(str(exc))}"
                ) from exc
            fallback_client = self._build_fallback_client()
            if fallback_client is None:
                raise LLMTransientError(
                    f"{self.provider.capitalize()} request failed: max retries reached - {self._sanitize_exception(str(exc))}"
                ) from exc
            yield from fallback_client.stream_response(customer_message, allow_fallback=False)

    def stream_response(self, customer_message: str, *, allow_fallback: bool = True) -> Any:
        """Stream plain-text output from the configured provider without exposing SDK objects."""
        cleaned_message = (customer_message or "").strip()
        if not cleaned_message:
            raise ValueError("customer_message must not be empty")

        if self.provider == "openai":
            return self._stream_with_fallback(
                cleaned_message,
                primary_stream=self._stream_openai_response,
                allow_fallback=allow_fallback,
            )
        if self.provider == "gemini":
            return self._stream_with_fallback(
                cleaned_message,
                primary_stream=self._stream_gemini_response,
                allow_fallback=allow_fallback,
            )

        raise ValueError(f"Unsupported LLM provider: {self.provider}")

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
            fallback_client = self._build_fallback_client()
            fallback_action = None if fallback_client is None else lambda: fallback_client.process_message(cleaned_message)
            return self._execute_with_fallback(
                primary_action=lambda: self._process_openai_message(cleaned_message),
                fallback_action=fallback_action,
            )
        if self.provider == "gemini":
            fallback_client = self._build_fallback_client()
            fallback_action = None if fallback_client is None else lambda: fallback_client.process_message(cleaned_message)
            return self._execute_with_fallback(
                primary_action=lambda: self._process_gemini_message(cleaned_message),
                fallback_action=fallback_action,
            )

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
            fallback_client = self._build_fallback_client()
            fallback_action = None if fallback_client is None else lambda: fallback_client.generate(system_prompt, user_prompt, response_model)
            return self._execute_with_fallback(
                primary_action=lambda: self._generate_openai_response(system_prompt, user_prompt, response_model),
                fallback_action=fallback_action,
            )

        if self.provider == "gemini":
            fallback_client = self._build_fallback_client()
            fallback_action = None if fallback_client is None else lambda: fallback_client.generate(system_prompt, user_prompt, response_model)
            return self._execute_with_fallback(
                primary_action=lambda: self._generate_gemini_response(system_prompt, user_prompt, response_model),
                fallback_action=fallback_action,
            )

        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _generate_openai_response(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ModelType],
    ) -> dict[str, Any] | ModelType:
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

    def _generate_gemini_response(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ModelType],
    ) -> dict[str, Any] | ModelType:
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
