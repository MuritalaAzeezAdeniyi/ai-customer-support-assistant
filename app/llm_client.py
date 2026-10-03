from __future__ import annotations

import logging
from typing import Any, TypeVar

from pydantic import BaseModel

from .config import Settings

ModelType = TypeVar("ModelType", bound=BaseModel)


class LLMClient:
    """Thin abstraction around the OpenAI SDK and future retry/fallback logic."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        fallback_model: str | None = None,
        max_retries: int | None = None,
        max_output_tokens: int | None = None,
        client: Any | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.api_key = api_key or ""
        self.model = model or "gpt-4o-mini"
        self.fallback_model = fallback_model or self.model
        self.max_retries = max_retries if max_retries is not None else 2
        self.max_output_tokens = max_output_tokens if max_output_tokens is not None else 512
        self.client = client
        self.logger = logger or logging.getLogger(__name__)

    @classmethod
    def from_settings(cls, settings: Settings) -> "LLMClient":
        """Build an LLM client from the application settings object."""
        return cls(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            fallback_model=settings.OPENAI_FALLBACK_MODEL,
            max_retries=settings.OPENAI_MAX_RETRIES,
            max_output_tokens=settings.OPENAI_MAX_OUTPUT_TOKENS,
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ModelType] | None = None,
    ) -> dict[str, Any] | ModelType:
        """Placeholder method for later OpenAI request execution.

        This initial version intentionally avoids network calls and retry logic.
        """
        raise NotImplementedError("LLM call implementation is intentionally deferred for this setup phase.")
