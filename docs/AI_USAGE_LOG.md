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

## 2026-10-05 - Retry behavior and exponential backoff
This entry records the AI-assisted implementation of Stage 3 retry handling and exponential backoff for the LLM client.

- Purpose: add a reusable retry wrapper for transient provider failures while preserving the OpenAI and Gemini integrations and their structured response contract.
- Tooling used: AI-assisted retry design, transient error classification, backoff calculation, and test validation.
- Scope: configuration for `LLM_MAX_RETRIES`, `LLM_INITIAL_BACKOFF_SECONDS`, and `LLM_MAX_BACKOFF_SECONDS`, provider-agnostic retry behavior, CLI-safe error messaging, and retry-focused unit tests.
- Notes: no real API calls were made during tests. The retry suite uses mocked provider SDK behavior and does not require live API credentials.

## 2026-10-05 - Rate-limit handling and fallback model support
This entry records the AI-assisted implementation of Stage 4: rate-limit handling and provider/model fallback behavior.

- Purpose: extend the existing LLM client so rate-limited or temporarily unavailable calls reuse the Stage 3 retry process and then fall back to a configured secondary provider/model without recursive loops.
- Tooling used: AI-assisted fallback design, retry-path extension, configuration updates, and regression test coverage.
- Scope: `LLM_FALLBACK_PROVIDER`, `LLM_FALLBACK_MODEL`, and related fallback wiring, rate-limit detection, sanitized error messages, and test coverage for fallback success and failure paths.
- Notes: all tests use mocked provider SDK behavior only. No live Gemini or OpenAI API calls or real secrets were introduced into the project.

## 2026-10-05 - Streaming support for text output
This entry records the AI-assisted implementation of Stage 5: incremental streaming support for the existing LLM client.

- Purpose: add a simple `stream_response()` interface that yields text chunks without altering the existing structured `generate()` contract or the retry/fallback flow.
- Tooling used: AI-assisted stream design, provider-specific SDK adaptation, CLI demo wiring, and mocked regression tests for streaming and fallback behavior.
- Scope: OpenAI streaming through `responses.stream`, Gemini streaming through `generate_content_stream`, provider-agnostic chunk normalization, CLI `--stream` support, and documentation updates.
- Notes: the stream is treated as a plain-text presentation layer and intentionally does not restart a partially consumed stream. All testing remains mocked and secret-safe.

Future entries should follow this format:

## [YYYY-MM-DD] - [Short topic]
- Summary:
- Files touched:
- Outcome:
- Follow-up:
