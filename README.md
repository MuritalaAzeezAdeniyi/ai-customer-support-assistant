# AI Customer Support Assistant

## Project purpose
This project is a CLI-based Python application for triaging fintech customer support requests with an LLM. The current stage connects the application to the OpenAI SDK using a reusable client wrapper and validates the model output against a Pydantic contract.

## Current implementation status
The project now includes:
- environment-based configuration
- Pydantic validation models
- reusable prompt builders
- an LLM abstraction layer backed by the OpenAI SDK
- structured output validation using the `SupportResponse` model
- interactive CLI input for customer support messages
- unit tests covering validation and client behavior

The project does not yet include retries, fallback logic, streaming, or advanced API error handling.

## Technology stack
- Python 3.12+
- OpenAI Python SDK
- Pydantic
- pydantic-settings
- pytest
- python-dotenv

## Project structure
```text
ai-customer-support-assistant/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── llm_client.py
│   ├── main.py
│   ├── models.py
│   └── prompts.py
├── docs/
│   └── AI_USAGE_LOG.md
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_llm_client.py
│   └── test_models.py
├── .env.example
├── .gitignore
├── AGENTS.md
├── README.md
├── requirements.txt
├── run.py
└── .env
```

## Setup instructions
1. Create a virtual environment.
2. Install dependencies:
   ```bash
   python -m venv .venv
   . .venv/bin/activate   # Linux/macOS
   # or .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and add your real OpenAI API key.
4. Run the CLI:
   ```bash
   python run.py
   ```
5. Enter a customer support message when prompted.

## Environment variable configuration
The project expects the following values in `.env`:

```env
LLM_PROVIDER=openai

OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_FALLBACK_MODEL=gpt-4o-mini
OPENAI_MAX_RETRIES=3
OPENAI_MAX_OUTPUT_TOKENS=512

GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash

LLM_MAX_RETRIES=3
LLM_INITIAL_BACKOFF_SECONDS=1.0
LLM_MAX_BACKOFF_SECONDS=8.0
```

For local testing with Gemini, set:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash
LLM_MAX_RETRIES=3
LLM_INITIAL_BACKOFF_SECONDS=1.0
LLM_MAX_BACKOFF_SECONDS=8.0
```

Never commit `.env` or hard-code secrets. Real API calls require a valid API key configured in `.env` only.

## Retry and backoff behavior
The LLM client retries only transient failures. These include rate-limit responses (HTTP 429), temporary outage statuses (HTTP 500, 502, 503, 504), timeout conditions, and network-level connection problems.

The client does not retry persistent or client-side failures such as 400 bad requests, 401 authentication errors, 403 permission errors, or 404 model/resource errors. Structured validation errors caused by the project schema are also not retried.

When a transient error is encountered, the client waits using exponential backoff with a capped maximum delay:

- attempt 1: 1s
- attempt 2: 2s
- attempt 3: 4s
- capped at `LLM_MAX_BACKOFF_SECONDS`

This behavior is controlled by environment settings rather than hardcoded values.

## Running the application
```bash
python run.py
```

The app will prompt for a customer message and print the structured response in a readable format. If `LLM_PROVIDER=gemini`, the request is sent through the Google Gemini SDK instead of OpenAI.

## Testing instructions
Run the unit tests with:

```bash
pytest -q
```

This project intentionally avoids real OpenAI or Gemini calls in normal unit tests. Any live API requests should be triggered only with a valid local environment configuration.
