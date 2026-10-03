from __future__ import annotations

from .config import Settings
from .llm_client import LLMClient
from .prompts import build_system_prompt, build_user_prompt


def main() -> None:
    """CLI entry point for the project in its initial setup stage."""
    settings = Settings()
    client = LLMClient.from_settings(settings)

    print("AI Customer Support Assistant")
    print(f"Primary model: {client.model}")
    print(f"Fallback model: {client.fallback_model}")
    print(f"Max retries: {client.max_retries}")
    print("Current status: configuration and validation scaffolding is in place.")

    sample_message = "My transfer of ₦50,000 has been pending since yesterday."
    print("System prompt ready:", bool(build_system_prompt()))
    print("User prompt ready:", bool(build_user_prompt(sample_message)))


if __name__ == "__main__":
    main()
