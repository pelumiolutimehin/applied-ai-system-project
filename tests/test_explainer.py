import json

import pytest

from src.config import CATALOG_PATH, KNOWLEDGE_PATH
from src.explainer import (
    MAX_EXPLANATION_CHARS,
    build_system_prompt,
    build_user_message,
    explain_recommendations,
    fallback_explanation,
    matched_features,
    parse_explanations,
)
from src.profile import validate_profile
from src.recommender import load_songs, recommend_songs
from src.retriever import Retriever, load_notes

PROFILE = validate_profile({
    "favorite_genre": "rock", "favorite_mood": "intense",
    "target_energy": 0.9, "target_valence": 0.3, "target_acousticness": 0.1,
})


@pytest.fixture(scope="module")
def retriever():
    return Retriever(load_notes(KNOWLEDGE_PATH))


@pytest.fixture(scope="module")
def recs():
    return recommend_songs(PROFILE, load_songs(str(CATALOG_PATH)), k=3)


class FakeLLM:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, system_prompt, user_text):
        self.calls.append((system_prompt, user_text))
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply


RANK_WORDS = {1: "first", 2: "second", 3: "third"}


def reply_for(*ranks, text="A good match."):
    """Digit-free texts: the faithfulness check rejects numbers that are not in the notes."""
    return json.dumps({"explanations": [{"rank": r, "text": f"{text} Pick {RANK_WORDS[r]}."} for r in ranks]})


def explain(recs, retriever, llm, **kw):
    return explain_recommendations(PROFILE, recs, retriever, generate=llm, sleep=lambda s: None, **kw)


# ---------- happy path ----------

def test_llm_explanations_are_returned_in_order_with_their_notes(recs, retriever):
    out = explain(recs, retriever, FakeLLM(reply_for(1, 2, 3)))
    assert [e.source for e in out] == ["llm"] * 3
    assert [e.text for e in out] == [f"A good match. Pick {w}." for w in ("first", "second", "third")]
    first_song = recs[0][0]
    assert out[0].notes_used == [
        f"genre:{first_song['genre']}", f"subgenre:{first_song['subgenre']}", f"mood:{first_song['mood']}",
    ]


def test_whole_list_is_explained_with_a_single_call(recs, retriever):
    llm = FakeLLM(reply_for(1, 2, 3))
    explain(recs, retriever, llm)
    assert len(llm.calls) == 1


# ---------- grounding: what the model is shown ----------

def test_user_message_contains_breakdown_notes_and_definitions(recs, retriever):
    notes = [retriever.notes_for_song(r[0]) for r in recs]
    message = build_user_message(PROFILE, recs, notes, retriever.get("feature"))
    song, _, reasons = recs[0]
    assert song["title"] in message and song["artist"] in message
    assert f"BREAKDOWN: {reasons}" in message
    assert "SONG 3" in message
    assert "Energy runs from 0 to 1" in message  # feature definition
    assert retriever.notes_for_song(song)[0].text in message  # retrieved genre note


def test_system_prompt_forbids_invention_and_false_matches():
    prompt = build_system_prompt()
    assert "ONLY facts" in prompt
    assert "mismatch" in prompt
    assert "Do not mention scores" in prompt


# ---------- partial and total failure ----------

def test_missing_rank_falls_back_for_that_song_only(recs, retriever):
    out = explain(recs, retriever, FakeLLM(reply_for(1, 3)))
    assert [e.source for e in out] == ["llm", "fallback", "llm"]
    assert out[1].text == fallback_explanation(recs[1][2].split(", "))


@pytest.mark.parametrize(
    "bad_item",
    [
        {"rank": 0, "text": "x"},
        {"rank": 4, "text": "x"},
        {"rank": True, "text": "x"},
        {"rank": "1", "text": "x"},
        {"rank": 1, "text": ""},
        {"rank": 1, "text": "   "},
        {"rank": 1, "text": "y" * (MAX_EXPLANATION_CHARS + 1)},
        {"rank": 1},
        "not a dict",
    ],
)
def test_invalid_items_are_ignored(recs, retriever, bad_item):
    reply = json.dumps({"explanations": [bad_item, {"rank": 2, "text": "ok"}, {"rank": 3, "text": "ok"}]})
    out = explain(recs, retriever, FakeLLM(reply))
    assert out[0].source == "fallback"
    assert [e.source for e in out[1:]] == ["llm", "llm"]


def test_unreadable_replies_retry_once_then_fall_back(recs, retriever):
    llm = FakeLLM("garbage")
    out = explain(recs, retriever, llm)
    assert len(llm.calls) == 2
    assert [e.source for e in out] == ["fallback"] * 3


def test_api_error_retries_then_succeeds(recs, retriever):
    llm = FakeLLM(RuntimeError("boom"), reply_for(1, 2, 3))
    out = explain(recs, retriever, llm)
    assert len(llm.calls) == 2
    assert [e.source for e in out] == ["llm"] * 3


def test_never_raises_even_if_generate_always_fails(recs, retriever):
    out = explain(recs, retriever, FakeLLM(RuntimeError("down")))
    assert [e.source for e in out] == ["fallback"] * 3


def test_wrong_json_shape_falls_back(recs, retriever):
    out = explain(recs, retriever, FakeLLM(json.dumps({"explanations": "nope"})))
    assert [e.source for e in out] == ["fallback"] * 3


def test_no_api_key_means_fallback_without_any_call(recs, retriever, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    out = explain_recommendations(PROFILE, recs, retriever)
    assert [e.source for e in out] == ["fallback"] * 3


def test_empty_recommendations_returns_empty_list(retriever):
    llm = FakeLLM(reply_for())
    assert explain([], retriever, llm) == []
    assert llm.calls == []


# ---------- deterministic fallback ----------

def test_matched_features_reads_match_and_close_only():
    reasons = [
        "energy close (+0.28)", "mood match (+0.25)", "valence far (+0.10)",
        "genre mismatch (+0.00)", "acousticness close (+0.09)",
    ]
    assert matched_features(reasons) == ["energy", "mood", "acousticness"]


def test_fallback_explanation_text():
    assert fallback_explanation(["mood match (+0.25)", "energy close (+0.28)"]) == \
        "Matched your settings on: mood, energy."
    assert fallback_explanation(["mood mismatch (+0.00)", "energy far (+0.10)"]) == \
        "This was the closest available overall match to your settings."


def test_parse_explanations_keeps_first_duplicate():
    raw = json.dumps({"explanations": [{"rank": 1, "text": "first"}, {"rank": 1, "text": "second"}]})
    assert parse_explanations(raw, 3) == {1: "first"}


# ---------- failure reasons are recorded ----------

class CodedError(Exception):
    code = 503


def test_fallback_records_the_exception_type_and_code_but_not_the_message(recs, retriever):
    out = explain(recs, retriever, FakeLLM(CodedError("secret=abc123")))
    assert {e.error for e in out} == {"CodedError 503"}
    assert "abc123" not in " ".join(e.error for e in out)


def test_plain_exception_records_just_its_type(recs, retriever):
    out = explain(recs, retriever, FakeLLM(RuntimeError("boom")))
    assert {e.error for e in out} == {"RuntimeError"}


def test_missing_rank_records_a_reason_even_though_the_call_worked(recs, retriever):
    out = explain(recs, retriever, FakeLLM(reply_for(1, 3)))
    assert out[1].error == "no usable text for this song"
    assert out[0].error is None and out[2].error is None


def test_rejected_text_is_kept_for_audit_and_is_not_an_error(recs, retriever):
    bad = json.dumps({"explanations": [{"rank": r, "text": "It scores 0.97."} for r in (1, 2, 3)]})
    out = explain(recs, retriever, FakeLLM(bad))
    assert all(e.faithful is False and e.rejected_text == "It scores 0.97." and e.error is None for e in out)
    assert all(any("0.97" in p for p in e.problems) for e in out)
