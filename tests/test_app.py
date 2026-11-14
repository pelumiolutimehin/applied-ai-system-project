import json
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import llm_parser

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def run_app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=30).run()


def test_app_renders_default_recommendations():
    at = run_app()
    assert not at.exception
    assert at.title[0].value.endswith("Music Recommender")
    # default k is 5: one bordered result block per song, each with a subheader
    assert len(at.subheader) == 5


def test_app_respects_result_count_slider():
    at = run_app()
    at.sidebar.slider(key="k").set_value(8).run()
    assert not at.exception
    assert len(at.subheader) == 8


def test_app_updates_when_preferences_change():
    at = run_app()
    first = [s.value for s in at.subheader]
    at.sidebar.selectbox(key="genre").select("rock")
    at.sidebar.selectbox(key="mood").select("intense")
    at.sidebar.slider(key="energy").set_value(0.95)
    at.run()
    assert not at.exception
    assert [s.value for s in at.subheader] != first


# ---------- free-text requests (Gemini mocked) ----------

ROCK_REPLY = json.dumps({
    "favorite_genre": "rock", "favorite_mood": "intense",
    "target_energy": 0.95, "target_valence": 0.3, "target_acousticness": 0.05,
})


@pytest.fixture
def fake_llm(monkeypatch):
    """Enable the LLM path with a counting fake in place of the real Gemini call."""
    st.cache_data.clear()
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls = []
    reply = {"value": ROCK_REPLY}

    def fake(system_prompt, user_text):
        calls.append(user_text)
        return reply["value"]

    monkeypatch.setattr(llm_parser, "call_gemini", fake)
    fake.calls, fake.reply = calls, reply
    yield fake
    st.cache_data.clear()


def test_free_text_request_drives_recommendations(fake_llm):
    at = run_app()
    baseline = [s.value for s in at.subheader]
    at.text_input(key="request").input("loud rock").run()
    assert not at.exception
    assert any("Interpreted as" in c.value and "rock" in c.value for c in at.caption)
    assert [s.value for s in at.subheader] != baseline
    assert len(at.subheader) == 5
    assert fake_llm.calls == ["loud rock"]


def test_same_request_is_not_sent_again_when_sliders_move(fake_llm):
    at = run_app()
    at.text_input(key="request").input("loud rock").run()
    at.sidebar.slider(key="k").set_value(7).run()
    assert len(at.subheader) == 7
    assert len(fake_llm.calls) == 1


def test_unparseable_request_warns_and_falls_back_to_sliders(fake_llm):
    fake_llm.reply["value"] = "{}"
    at = run_app()
    at.text_input(key="request").input("what is the capital of France").run()
    assert not at.exception
    assert any("couldn't find" in w.value for w in at.warning)
    assert len(at.subheader) == 5


def test_failed_request_is_not_retried_on_every_rerun(fake_llm):
    fake_llm.reply["value"] = "{}"
    at = run_app()
    at.text_input(key="request").input("nonsense").run()
    at.sidebar.slider(key="k").set_value(6).run()
    assert len(fake_llm.calls) == 1


def test_request_without_api_key_shows_note_and_uses_sliders(monkeypatch):
    st.cache_data.clear()
    monkeypatch.setenv("GEMINI_API_KEY", "")
    at = run_app()
    at.text_input(key="request").input("loud rock").run()
    assert not at.exception
    assert any("GEMINI_API_KEY" in i.value for i in at.info)
    assert len(at.subheader) == 5
