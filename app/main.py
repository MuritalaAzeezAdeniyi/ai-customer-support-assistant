from __future__ import annotations

import argparse

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
    parser = argparse.ArgumentParser(description="AI Customer Support Assistant")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the assistant's response incrementally instead of waiting for a full structured output.",
    )
    args = parser.parse_args()

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
            if args.stream:
                print("Streaming response:")
                for chunk in client.stream_response(message):
                    print(chunk, end="", flush=True)
                print()
            else:
                response = client.process_message(message)
                print(_format_response(response))
        except ValueError as exc:
            print(f"Validation error: {_sanitize_message(str(exc), provider_key)}")
        except Exception as exc:  # pragma: no cover - CLI safety envelope
            print(f"Request failed: {_sanitize_message(str(exc), provider_key)}")


if __name__ == "__main__":
    main()
