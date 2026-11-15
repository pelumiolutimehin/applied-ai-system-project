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

WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}

ROCK_REPLY = json.dumps({
    "favorite_genre": "rock", "favorite_mood": "intense",
    "target_energy": 0.95, "target_valence": 0.3, "target_acousticness": 0.05,
})


@pytest.fixture
def fake_llm(monkeypatch):
    """Enable the LLM path with a counting fake in place of the real Gemini call.

    Parser requests (system prompt mentions "taste profile") get `reply`; explanation requests
    get a valid explanation for every "SONG n" in the message. Calls are recorded separately.
    """
    st.cache_data.clear()
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    parser_calls, explain_calls = [], []
    reply = {"value": ROCK_REPLY}

    def fake(system_prompt, user_text):
        if "taste profile" in system_prompt:
            parser_calls.append(user_text)
            return reply["value"]
        explain_calls.append(user_text)
        ranks = [int(line.split()[1].rstrip(":")) for line in user_text.splitlines() if line.startswith("SONG ")]
        return json.dumps({"explanations": [{"rank": r, "text": f"Because of reason {WORDS[r]}."} for r in ranks]})

    monkeypatch.setattr(llm_parser, "call_gemini", fake)
    fake.calls, fake.explain_calls, fake.reply = parser_calls, explain_calls, reply
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


# ---------- AI explanations (Gemini mocked) ----------

def expander_text(at):
    return " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)


def test_explanations_are_shown_with_their_source_notes(fake_llm):
    at = run_app()
    assert not at.exception
    assert len(fake_llm.explain_calls) == 1
    text = expander_text(at)
    assert "Because of reason one." in text and "Because of reason five." in text
    assert "AI-written from these notes: genre:" in text


def test_toggle_off_skips_explanation_call(fake_llm):
    at = run_app()
    fake_llm.explain_calls.clear()
    at.sidebar.checkbox(key="explain").uncheck().run()
    at.sidebar.slider(key="k").set_value(4).run()
    assert fake_llm.explain_calls == []
    assert "Because of reason" not in expander_text(at)
    assert len(at.subheader) == 4


def test_explanations_are_cached_across_reruns(fake_llm):
    at = run_app()
    at.text_input(key="request").input("loud rock").run()
    calls_after_first = len(fake_llm.explain_calls)
    at.run()
    assert len(fake_llm.explain_calls) == calls_after_first


def test_failed_explanations_fall_back_to_breakdown_only(fake_llm, monkeypatch):
    def broken(system_prompt, user_text):
        raise RuntimeError("down")

    monkeypatch.setattr(llm_parser, "call_gemini", broken)
    at = run_app()
    assert not at.exception
    assert len(at.subheader) == 5
    assert "Score breakdown" in expander_text(at)
    assert "AI-written" not in expander_text(at)


def test_checkbox_is_disabled_without_api_key():
    at = run_app()
    assert at.sidebar.checkbox(key="explain").disabled
    assert not at.sidebar.checkbox(key="explain").value


# ---------- confidence and logging ----------

def all_text(at):
    return " ".join(m.value for m in at.markdown) + " " + " ".join(c.value for c in at.caption)


def test_confidence_is_shown_with_its_explanation():
    at = run_app()
    assert not at.exception
    assert "Confidence:" in all_text(at)
    assert any("How is confidence calculated?" in e.label for e in at.expander)


def test_weak_results_show_a_low_confidence_warning():
    at = run_app()
    # an energetic edm fan who also wants a sad, acoustic, quiet sound: no song fits well
    at.sidebar.selectbox(key="genre").select("edm")
    at.sidebar.selectbox(key="mood").select("sad")
    at.sidebar.slider(key="energy").set_value(1.0)
    at.sidebar.slider(key="valence").set_value(0.0)
    at.sidebar.slider(key="acoustic").set_value(1.0)
    at.run()
    assert not at.exception
    assert "Confidence: low" in all_text(at)
    assert any("Low confidence" in w.value for w in at.warning)


def test_each_distinct_run_is_logged_once_not_once_per_rerun(tmp_path):
    from src import runlog
    at = run_app()
    at.run()
    at.run()
    assert len(runlog.read_log()) == 1            # reruns with identical inputs are not re-logged
    at.sidebar.slider(key="k").set_value(4).run()
    assert len(runlog.read_log()) == 2            # a changed input is a new run


def test_log_records_source_and_confidence(fake_llm):
    from src import runlog
    at = run_app()
    at.text_input(key="request").input("loud rock").run()
    record = runlog.read_log()[-1]
    assert record["source"] == "text" and record["request"] == "loud rock"
    assert record["profile"]["favorite_genre"] == "rock"
    assert record["confidence"]["level"] in {"high", "medium", "low"}
    assert len(record["explanations"]) == 5


def test_failed_text_request_is_logged_as_a_fallback(fake_llm):
    from src import runlog
    fake_llm.reply["value"] = "{}"
    at = run_app()
    at.text_input(key="request").input("what is the capital of France").run()
    record = runlog.read_log()[-1]
    assert record["source"] == "sliders_fallback"
    assert "couldn't find" in record["parse_error"]
