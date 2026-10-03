from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env files."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    OPENAI_API_KEY: str = Field(..., min_length=1, description="API key for the OpenAI service.")
    OPENAI_MODEL: str = Field(default="gpt-4o-mini", min_length=1, description="Primary model to use.")
    OPENAI_FALLBACK_MODEL: str = Field(
        default="gpt-4o-mini",
        min_length=1,
        description="Fallback model for resiliency during outages or rate limits.",
    )
    OPENAI_MAX_RETRIES: int = Field(default=2, ge=0, le=10, description="Maximum retry attempts.")
    OPENAI_MAX_OUTPUT_TOKENS: int = Field(
        default=512,
        ge=1,
        le=4096,
        description="Maximum number of output tokens to request from the model.",
    )
