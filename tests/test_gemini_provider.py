from types import SimpleNamespace

import pytest

from app.config import Settings
from app.llm_client import LLMClient
from app.models import SupportResponse


def test_gemini_successful_response_is_converted_to_support_response() -> None:
    payload = SupportResponse(
        category="TRANSFER_ISSUE",
        priority="HIGH",
        sentiment="NEGATIVE",
        requires_human=True,
        suggested_response="We are examining your pending transfer.",
    )

    class FakeGeminiClient:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                return SimpleNamespace(parsed=payload)

    client = LLMClient(
        provider="gemini",
        api_key="gemini-test-key",
        model="gemini-2.0-flash",
        max_output_tokens=256,
        client=FakeGeminiClient(),
    )

    result = client.process_message("My transfer is stuck.")

    assert isinstance(result, SupportResponse)
    assert result.category == "TRANSFER_ISSUE"
    assert result.priority == "HIGH"


def test_provider_configuration_selects_gemini() -> None:
    settings = Settings(
        OPENAI_API_KEY="openai-key",
        OPENAI_MODEL="gpt-4o-mini",
        LLM_PROVIDER="gemini",
        GEMINI_API_KEY="gemini-key",
        GEMINI_MODEL="gemini-2.0-flash",
    )

    client = LLMClient.from_settings(settings)

    assert client.provider == "gemini"
    assert client.model == "gemini-2.0-flash"
    assert client.api_key == "gemini-key"


def test_gemini_api_failure_is_handled_without_exposing_api_key() -> None:
    class FakeGeminiClient:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                raise RuntimeError("Gemini API failed: gemini-secret-123")

    client = LLMClient(
        provider="gemini",
        api_key="gemini-secret-123",
        model="gemini-2.0-flash",
        max_output_tokens=256,
        client=FakeGeminiClient(),
    )

    with pytest.raises(RuntimeError) as exc_info:
        client.process_message("My card was charged twice.")

    assert "Gemini request failed" in str(exc_info.value)
    assert "gemini-secret-123" not in str(exc_info.value)
