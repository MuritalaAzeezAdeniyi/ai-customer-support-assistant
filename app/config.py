from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env files."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    LLM_PROVIDER: Literal["openai", "gemini"] = Field(default="openai", description="LLM provider to use.")
    OPENAI_API_KEY: str | None = Field(default=None, min_length=1, description="API key for the OpenAI service.")
    OPENAI_MODEL: str = Field(default="gpt-4o-mini", min_length=1, description="Primary OpenAI model to use.")
    OPENAI_FALLBACK_MODEL: str = Field(
        default="gpt-4o-mini",
        min_length=1,
        description="Fallback model for resiliency during outages or rate limits.",
    )
    GEMINI_API_KEY: str | None = Field(default=None, min_length=1, description="API key for the Google Gemini service.")
    GEMINI_MODEL: str = Field(default="gemini-2.0-flash", min_length=1, description="Primary Gemini model to use.")
    OPENAI_MAX_RETRIES: int = Field(default=2, ge=0, le=10, description="Maximum retry attempts.")
    OPENAI_MAX_OUTPUT_TOKENS: int = Field(
        default=512,
        ge=1,
        le=4096,
        description="Maximum number of output tokens to request from the model.",
    )

    @model_validator(mode="after")
    def validate_provider_configuration(self) -> "Settings":
        if self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER is set to 'openai'.")
        if self.LLM_PROVIDER == "gemini" and not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER is set to 'gemini'.")
        return self
