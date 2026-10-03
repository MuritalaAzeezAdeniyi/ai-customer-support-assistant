# AGENTS.md

## Project purpose
This repository contains a small CLI-based Python project for AI-powered fintech customer support triage. The goal is to classify incoming customer messages, estimate urgency and sentiment, and draft a professional support reply.

## Technology stack
- Python 3.12+
- OpenAI Python SDK
- Pydantic
- pydantic-settings
- pytest
- python-dotenv (only if needed for local configuration loading)

## Architecture principles
- Keep the OpenAI SDK behind a reusable LLM client abstraction.
- Prefer small, focused modules and clear naming.
- Keep business logic separate from configuration and prompting.
- Add only the minimum abstractions necessary for clarity.
- Preserve a simple CLI-oriented structure.

## Security requirements
- Never hard-code API keys.
- Never commit `.env` files or any secret material.
- Never log API keys or token values.
- Do not expose secrets in error messages.
- Do not add unnecessary external services.
- Do not make real API calls from unit tests.

## Working rules for coding agents
- Inspect the existing code before modifying it.
- Keep changes small and easy to review.
- Run tests after meaningful edits.
- Do not add unnecessary dependencies.
- Keep OpenAI interactions isolated in the LLM client layer.
- Follow the project's structure and avoid broad refactors without justification.
