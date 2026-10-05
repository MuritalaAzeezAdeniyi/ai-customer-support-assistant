from __future__ import annotations

from .config import Settings
from .llm_client import LLMClient


def _sanitize_message(message: str, api_key: str | None = None) -> str:
    cleaned = message
    if api_key:
        cleaned = cleaned.replace(api_key, "[REDACTED]")
    return cleaned


def _format_response(response: object) -> str:
    """Return a readable representation of the parsed support response."""
    data = response.model_dump() if hasattr(response, "model_dump") else response
    lines = ["Structured response:"]
    for key, value in data.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def main() -> None:
    """Interactive CLI for submitting customer support requests to the model."""
    settings = Settings()
    client = LLMClient.from_settings(settings)
    provider_key = settings.GEMINI_API_KEY if settings.LLM_PROVIDER == "gemini" else settings.OPENAI_API_KEY

    print("AI Customer Support Assistant")
    print(f"Provider: {client.provider}")
    print(f"Model: {client.model}")
    print("Type a customer support message below. Enter 'quit' to exit.")

    while True:
        message = input("Customer message: ").strip()

        if not message:
            print("Please enter a customer message.")
            continue

        if message.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        try:
            response = client.process_message(message)
            print(_format_response(response))
        except ValueError as exc:
            print(f"Validation error: {_sanitize_message(str(exc), provider_key)}")
        except Exception as exc:  # pragma: no cover - CLI safety envelope
            print(f"Request failed: {_sanitize_message(str(exc), provider_key)}")


if __name__ == "__main__":
    main()
