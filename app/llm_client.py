from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from .config import Settings
from .models import SupportResponse
from .prompts import build_system_prompt, build_user_prompt

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - dependency is installed in project env
    genai = None
    genai_types = None

ModelType = TypeVar("ModelType", bound=BaseModel)


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
        max_output_tokens: int | None = None,
        client: Any | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.provider = (provider or "openai").lower()
        self.api_key = api_key or ""
        self.model = model or ("gemini-2.0-flash" if self.provider == "gemini" else "gpt-4o-mini")
        self.fallback_model = fallback_model or self.model
        self.max_retries = max_retries if max_retries is not None else 2
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
        if settings.LLM_PROVIDER == "gemini":
            return cls(
                provider="gemini",
                api_key=settings.GEMINI_API_KEY,
                model=settings.GEMINI_MODEL,
                max_output_tokens=settings.OPENAI_MAX_OUTPUT_TOKENS,
            )

        return cls(
            provider="openai",
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            fallback_model=settings.OPENAI_FALLBACK_MODEL,
            max_retries=settings.OPENAI_MAX_RETRIES,
            max_output_tokens=settings.OPENAI_MAX_OUTPUT_TOKENS,
        )

    def _sanitize_exception(self, message: str) -> str:
        return message.replace(self.api_key, "[REDACTED]") if self.api_key else message

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

        try:
            response = self.client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                text_format=SupportResponse,
                max_output_tokens=self.max_output_tokens,
            )
        except Exception as exc:
            raise RuntimeError(f"OpenAI request failed: {self._sanitize_exception(str(exc))}") from exc

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

        try:
            if genai_types is not None:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=combined_prompt,
                    config=genai_types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=SupportResponse,
                    ),
                )
            else:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=combined_prompt,
                )
        except Exception as exc:
            raise RuntimeError(f"Gemini request failed: {self._sanitize_exception(str(exc))}") from exc

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
            try:
                response = self.client.responses.parse(
                    model=self.model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    text_format=response_model,
                    max_output_tokens=self.max_output_tokens,
                )
            except Exception as exc:
                raise RuntimeError(f"OpenAI request failed: {self._sanitize_exception(str(exc))}") from exc

            parsed_response = getattr(response, "output_parsed", None)
            if parsed_response is None:
                raise ValueError("The OpenAI API returned no parsed structured data.")
            return self._validate_structured_response(parsed_response, provider_name="OpenAI")

        if self.provider == "gemini":
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            try:
                if genai_types is not None:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=combined_prompt,
                        config=genai_types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=response_model,
                        ),
                    )
                else:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=combined_prompt,
                    )
            except Exception as exc:
                raise RuntimeError(f"Gemini request failed: {self._sanitize_exception(str(exc))}") from exc

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
