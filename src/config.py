"""Central configuration: API key and model name, loaded from the environment / .env."""

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "gemini-2.5-flash"


def get_api_key() -> str:
    """Return the Gemini API key, or an empty string if it is not set."""
    return os.getenv("GEMINI_API_KEY", "").strip()


def get_model_name() -> str:
    """Return the Gemini model name, falling back to the default."""
    return os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_MODEL


def llm_available() -> bool:
    """True when an API key is configured, so LLM features can be attempted."""
    return bool(get_api_key())
