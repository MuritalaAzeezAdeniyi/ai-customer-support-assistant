# AI Customer Support Assistant

## Project purpose
This project is a CLI-based Python application for triaging fintech customer support requests with an LLM. The initial scope focuses on project setup, validation, and modular design rather than production API integration.

## Current implementation status
The project is in its initial setup phase. The repository now includes:
- environment-based configuration
- Pydantic validation models
- reusable prompt builders
- an LLM abstraction layer
- unit tests for configuration and model validation

No live OpenAI API calls, retry logic, or fallback logic are implemented yet.

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
3. Copy `.env.example` to `.env` and add your environment values.
4. Run the CLI:
   ```bash
   python run.py
   ```

## Environment variable configuration
The project expects the following values in `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_FALLBACK_MODEL=gpt-4o-mini
OPENAI_MAX_RETRIES=2
OPENAI_MAX_OUTPUT_TOKENS=512
```

Never commit `.env` or hard-code secrets.

## Testing instructions
Run the unit tests with:

```bash
pytest -q
```

This project intentionally avoids real OpenAI calls in unit tests.
