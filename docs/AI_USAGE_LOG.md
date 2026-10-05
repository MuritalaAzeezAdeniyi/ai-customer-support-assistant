# AI Usage Log

## 2026-10-03 - Initial project foundation and Stage 2 OpenAI integration
This file records the use of AI coding assistance during development for the AI Customer Support Assistant project.

- Purpose: create the initial repository foundation, define the structured response contract, and implement the first production-stage OpenAI SDK integration.
- Tooling used: AI-assisted code generation, project scaffolding, CLI design, structured-output wiring, and test authoring.
- Scope: repository setup, configuration, Pydantic validation, prompt design, interactive CLI flow, LLM client integration, and unit test coverage.
- Notes: no secret values, API keys, or `.env` contents were committed. No real OpenAI requests were made as part of unit tests. A real API call requires a valid API key configured locally in `.env`.

## 2026-10-05 - Gemini provider support and provider selection
This entry records the AI-assisted addition of Gemini as a second LLM provider while preserving the existing OpenAI provider.

- Purpose: extend configuration, support provider selection through `LLM_PROVIDER`, and keep the shared customer-support logic reusable across providers.
- Tooling used: AI-assisted refactoring, provider-selection design, SDK integration planning, and Gemini test authoring.
- Scope: Gemini environment variables, dependency update, provider-aware client design, CLI output improvements, README updates, and unit tests for Gemini success and failure handling.
- Notes: no real Gemini API key or `.env` secret material was committed. The tests mock the Gemini SDK interaction and do not require a live API key.

Future entries should follow this format:

## [YYYY-MM-DD] - [Short topic]
- Summary:
- Files touched:
- Outcome:
- Follow-up:
