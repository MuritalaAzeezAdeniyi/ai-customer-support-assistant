from types import SimpleNamespace

import pytest

from app.llm_client import LLMClient
from app.models import SupportResponse


def test_process_message_returns_support_response() -> None:
    payload = SupportResponse(
        category="TRANSFER_ISSUE",
        priority="HIGH",
        sentiment="NEGATIVE",
        requires_human=True,
        suggested_response="We are investigating your transfer delay.",
    )

    class FakeResponses:
        @staticmethod
        def parse(**kwargs):
            return SimpleNamespace(output_parsed=payload)

    client = LLMClient(
        api_key="test-key",
        model="gpt-4o-mini",
        max_output_tokens=256,
        client=SimpleNamespace(responses=FakeResponses()),
    )

    result = client.process_message("My transfer has been pending since yesterday.")

    assert isinstance(result, SupportResponse)
    assert result.category == "TRANSFER_ISSUE"
    assert result.priority == "HIGH"


def test_process_message_passes_customer_message_to_model() -> None:
    captured: dict[str, object] = {}

    class FakeResponses:
        @staticmethod
        def parse(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                output_parsed=SupportResponse(
                    category="CARD_ISSUE",
                    priority="MEDIUM",
                    sentiment="NEGATIVE",
                    requires_human=False,
                    suggested_response="We are reviewing your card issue.",
                )
            )

    client = LLMClient(
        api_key="test-key",
        model="gpt-4o-mini",
        max_output_tokens=256,
        client=SimpleNamespace(responses=FakeResponses()),
    )

    client.process_message("My card payment failed this morning.")

    assert captured["model"] == "gpt-4o-mini"
    assert captured["text_format"] is SupportResponse
    assert any(
        item["role"] == "user" and "My card payment failed this morning." in item["content"]
        for item in captured["input"]
    )


def test_process_message_passes_openai_max_output_tokens_to_sdk() -> None:
    captured: dict[str, object] = {}

    class FakeResponses:
        @staticmethod
        def parse(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                output_parsed=SupportResponse(
                    category="ACCOUNT_ISSUE",
                    priority="HIGH",
                    sentiment="NEGATIVE",
                    requires_human=True,
                    suggested_response="We are reviewing the account issue.",
                )
            )

    client = LLMClient(
        api_key="test-key",
        model="gpt-4o-mini",
        max_output_tokens=768,
        client=SimpleNamespace(responses=FakeResponses()),
    )

    client.process_message("My account was locked after a failed login.")

    assert captured["max_output_tokens"] == 768


def test_process_message_passes_gemini_max_output_tokens_to_sdk() -> None:
    captured: dict[str, object] = {}

    class FakeGeminiClient:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                captured.update(kwargs)
                return SimpleNamespace(parsed=SupportResponse(
                    category="TRANSFER_ISSUE",
                    priority="MEDIUM",
                    sentiment="NEGATIVE",
                    requires_human=False,
                    suggested_response="We are checking the transfer delay.",
                ))

    client = LLMClient(
        provider="gemini",
        api_key="gemini-test-key",
        model="gemini-2.0-flash",
        max_output_tokens=512,
        client=FakeGeminiClient(),
    )

    client.process_message("My transfer is stuck.")

    assert captured["config"].max_output_tokens == 512
    assert captured["config"].response_mime_type == "application/json"


def test_process_message_handles_api_failure_without_exposing_key() -> None:
    class FakeResponses:
        @staticmethod
        def parse(**kwargs):
            raise RuntimeError("OpenAI request failed: sk-secret-123")

    client = LLMClient(
        api_key="sk-secret-123",
        model="gpt-4o-mini",
        max_output_tokens=256,
        client=SimpleNamespace(responses=FakeResponses()),
    )

    with pytest.raises(RuntimeError) as exc_info:
        client.process_message("My payment was declined.")

    assert "OpenAI request failed" in str(exc_info.value)
    assert "sk-secret-123" not in str(exc_info.value)


def test_process_message_rejects_invalid_structured_data() -> None:
    class FakeResponses:
        @staticmethod
        def parse(**kwargs):
            return SimpleNamespace(
                output_parsed={
                    "category": "INVALID",
                    "priority": "SEVERE",
                    "sentiment": "UNKNOWN",
                    "requires_human": "yes",
                    "suggested_response": "",
                }
            )

    client = LLMClient(
        api_key="test-key",
        model="gpt-4o-mini",
        max_output_tokens=256,
        client=SimpleNamespace(responses=FakeResponses()),
    )

    with pytest.raises(ValueError, match="invalid structured data"):
        client.process_message("This is a test message.")
