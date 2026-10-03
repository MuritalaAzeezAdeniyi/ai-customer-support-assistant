import pytest
from pydantic import ValidationError

from app.config import Settings


def test_settings_accepts_valid_configuration() -> None:
    settings = Settings(
        OPENAI_API_KEY="test-key",
        OPENAI_MODEL="gpt-4o-mini",
        OPENAI_FALLBACK_MODEL="gpt-4o-mini",
        OPENAI_MAX_RETRIES=3,
        OPENAI_MAX_OUTPUT_TOKENS=1024,
    )

    assert settings.OPENAI_API_KEY == "test-key"
    assert settings.OPENAI_MODEL == "gpt-4o-mini"
    assert settings.OPENAI_MAX_RETRIES == 3
    assert settings.OPENAI_MAX_OUTPUT_TOKENS == 1024


def test_settings_rejects_invalid_configuration() -> None:
    with pytest.raises(ValidationError):
        Settings(
            OPENAI_API_KEY="",
            OPENAI_MODEL="",
            OPENAI_FALLBACK_MODEL="",
            OPENAI_MAX_RETRIES=0,
            OPENAI_MAX_OUTPUT_TOKENS=0,
        )
