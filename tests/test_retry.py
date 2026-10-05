from types import SimpleNamespace

import httpx
import pytest

import openai

from app.llm_client import LLMClient, LLMTransientError, calculate_backoff
from app.models import SupportResponse


def _status_error(status_code: int, message: str) -> Exception:
    response = httpx.Response(status_code, request=httpx.Request("POST", "https://example.com/v1/chat/completions"))
    return openai.APIStatusError(message, response=response, body=None)


def test_transient_failure_then_success(monkeypatch) -> None:
    attempts = {"count": 0}
    sleep_calls: list[float] = []

    class FakeResponses:
        @staticmethod
        def parse(**kwargs):
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise _status_error(503, "temporary outage")
            return SimpleNamespace(output_parsed=SupportResponse(
                category="TRANSFER_ISSUE",
                priority="HIGH",
                sentiment="NEGATIVE",
                requires_human=True,
                suggested_response="We are investigating the delay.",
            ))

    monkeypatch.setattr("app.llm_client.time.sleep", lambda delay: sleep_calls.append(delay))

    client = LLMClient(
        provider="openai",
        api_key="test-key",
        model="gpt-4o-mini",
        max_retries=3,
        initial_backoff_seconds=1.0,
        max_backoff_seconds=8.0,
        client=SimpleNamespace(responses=FakeResponses()),
    )

    response = client.process_message("My transfer is delayed.")

    assert isinstance(response, SupportResponse)
    assert attempts["count"] == 3
    assert sleep_calls == [1.0, 2.0]


def test_persistent_error_fails_immediately() -> None:
    calls = {"count": 0}

    class FakeResponses:
        @staticmethod
        def parse(**kwargs):
            calls["count"] += 1
            raise openai.AuthenticationError("invalid API key", response=httpx.Response(401, request=httpx.Request("POST", "https://example.com")), body=None)

    client = LLMClient(
        provider="openai",
        api_key="bad-key",
        model="gpt-4o-mini",
        max_retries=3,
        initial_backoff_seconds=1.0,
        max_backoff_seconds=8.0,
        client=SimpleNamespace(responses=FakeResponses()),
    )

    with pytest.raises(openai.AuthenticationError):
        client.process_message("My account is locked.")

    assert calls["count"] == 1


def test_max_retries_reached_raises_controlled_error(monkeypatch) -> None:
    calls = {"count": 0}
    monkeypatch.setattr("app.llm_client.time.sleep", lambda delay: None)

    class FakeResponses:
        @staticmethod
        def parse(**kwargs):
            calls["count"] += 1
            raise _status_error(503, "still down")

    client = LLMClient(
        provider="openai",
        api_key="test-key",
        model="gpt-4o-mini",
        max_retries=3,
        initial_backoff_seconds=1.0,
        max_backoff_seconds=8.0,
        client=SimpleNamespace(responses=FakeResponses()),
    )

    with pytest.raises(LLMTransientError, match="max retries"):
        client.process_message("My payment failed again.")

    assert calls["count"] == 4


def test_calculate_backoff_matches_expected_delays() -> None:
    assert calculate_backoff(1, initial_backoff_seconds=1.0, max_backoff_seconds=8.0) == 1.0
    assert calculate_backoff(2, initial_backoff_seconds=1.0, max_backoff_seconds=8.0) == 2.0
    assert calculate_backoff(3, initial_backoff_seconds=1.0, max_backoff_seconds=8.0) == 4.0
    assert calculate_backoff(5, initial_backoff_seconds=1.0, max_backoff_seconds=8.0) == 8.0


def test_rate_limit_error_is_retried(monkeypatch) -> None:
    attempts = {"count": 0}
    sleep_calls: list[float] = []
    monkeypatch.setattr("app.llm_client.time.sleep", lambda delay: sleep_calls.append(delay))

    class FakeResponses:
        @staticmethod
        def parse(**kwargs):
            attempts["count"] += 1
            if attempts["count"] < 2:
                raise openai.RateLimitError("too many requests", response=httpx.Response(429, request=httpx.Request("POST", "https://example.com")), body=None)
            return SimpleNamespace(output_parsed=SupportResponse(
                category="PAYMENT_ISSUE",
                priority="MEDIUM",
                sentiment="NEGATIVE",
                requires_human=False,
                suggested_response="We are reviewing the payment issue.",
            ))

    client = LLMClient(
        provider="openai",
        api_key="test-key",
        model="gpt-4o-mini",
        max_retries=3,
        initial_backoff_seconds=1.0,
        max_backoff_seconds=8.0,
        client=SimpleNamespace(responses=FakeResponses()),
    )

    response = client.process_message("My payment is stuck.")

    assert isinstance(response, SupportResponse)
    assert attempts["count"] == 2
    assert sleep_calls == [1.0]
