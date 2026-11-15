import pytest

from src import config, llm_parser


@pytest.fixture(autouse=True)
def no_real_gemini_calls(monkeypatch):
    """Every test starts with no API key and a Gemini client that fails loudly.

    The developer's real key is loaded from .env at import time, so without this an unmocked
    test would make live API calls (slow, flaky, and it spends quota). Tests that need the
    LLM path opt in by patching llm_parser.call_gemini and setting GEMINI_API_KEY themselves.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "")

    def blocked(*args, **kwargs):
        raise AssertionError("test attempted a real Gemini call; mock llm_parser.call_gemini")

    monkeypatch.setattr(llm_parser, "call_gemini", blocked)


@pytest.fixture(autouse=True)
def isolated_run_log(monkeypatch, tmp_path):
    """Send run logs to a temp file so tests never write to the developer's real logs/ folder."""
    monkeypatch.setattr(config, "LOG_PATH", tmp_path / "run.jsonl")
