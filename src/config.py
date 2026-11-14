"""Central configuration: API key and model name, loaded from the environment / .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CATALOG_PATH = DATA_DIR / "songs.csv"
RAW_SOURCE_PATH = DATA_DIR / "spotify_songs.csv"
LATIN_EXCLUSIONS_PATH = DATA_DIR / "latin_exclusions.txt"
KNOWLEDGE_PATH = DATA_DIR / "knowledge" / "notes.json"

load_dotenv(ROOT_DIR / ".env")

DEFAULT_MODEL = "gemini-3.5-flash-lite"


def get_api_key() -> str:
    """Return the Gemini API key, or an empty string if it is not set."""
    return os.getenv("GEMINI_API_KEY", "").strip()


def get_model_name() -> str:
    """Return the Gemini model name, falling back to the default."""
    return os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_MODEL


def llm_available() -> bool:
    """True when an API key is configured, so LLM features can be attempted."""
    return bool(get_api_key())
