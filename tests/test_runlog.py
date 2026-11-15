import json
from types import SimpleNamespace

from src import config, runlog
from src.profile import UserProfile
from src.reliability import ConfidenceReport

PROFILE = UserProfile("rock", "intense", 0.9, 0.3, 0.1)
SONG = {"id": 7, "title": "Loud", "artist": "Band"}
CONF = ConfidenceReport(0.82, "high", {"strength": 0.9, "genre_support": None}, ["a note"])


def record(**overrides):
    base = dict(request="loud rock", source="text", profile=PROFILE, recommendations=[(SONG, 0.9, "why")],
                confidence=CONF, explanations=[SimpleNamespace(source="llm", faithful=True, problems=[], rejected_text=None, error=None)])
    base.update(overrides)
    return runlog.build_record(**base)


def test_build_record_is_json_serializable_and_complete():
    r = record()
    json.dumps(r)
    assert r["profile"]["favorite_genre"] == "rock"
    assert r["results"] == [{"id": 7, "title": "Loud", "score": 0.9}]
    assert r["confidence"]["level"] == "high" and r["confidence"]["factors"]["genre_support"] is None
    assert r["explanations"] == [{"source": "llm", "faithful": True, "problems": [], "rejected_text": None, "error": None}]


def test_log_run_appends_one_json_line_per_run(tmp_path):
    path = tmp_path / "nested" / "run.jsonl"
    assert runlog.log_run(record(), path) and runlog.log_run(record(request="again"), path)
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["request"] == "loud rock" and "ts" in first


def test_default_path_comes_from_config_at_call_time(tmp_path, monkeypatch):
    target = tmp_path / "elsewhere.jsonl"
    monkeypatch.setattr(config, "LOG_PATH", target)
    runlog.log_run(record())
    assert len(runlog.read_log()) == 1 and target.exists()


def test_log_run_never_raises_when_the_path_is_unwritable(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert runlog.log_run(record(), blocker / "run.jsonl") is False   # parent is a file


def test_read_log_handles_missing_file_and_skips_bad_lines(tmp_path):
    assert runlog.read_log(tmp_path / "missing.jsonl") == []
    path = tmp_path / "run.jsonl"
    path.write_text('{"source": "text"}\nnot json\n\n{"source": "sliders"}\n')
    assert [r["source"] for r in runlog.read_log(path)] == ["text", "sliders"]


def test_summarize_log_counts_sources_levels_and_rejections():
    low = ConfidenceReport(0.3, "low", {}, [])
    rejected = SimpleNamespace(source="fallback", faithful=False, rejected_text="bad text", error=None,
                               problems=["mentions 'rap', which is not this song's genre or in its notes",
                                         "quotes the number 0.9, which is not in the notes"])
    records = [
        record(),
        record(source="sliders_fallback", confidence=low, explanations=[rejected]),
        record(source="sliders", request=None),
    ]
    s = runlog.summarize_log(records)
    assert s["runs"] == 3
    assert s["by_source"] == {"text": 1, "sliders_fallback": 1, "sliders": 1}
    assert s["by_confidence_level"] == {"high": 2, "low": 1}
    assert s["text_requests_that_fell_back"] == 1
    assert s["explanations_total"] == 3 and s["explanations_rejected"] == 1
    assert s["rejection_reasons"] == {"mentions a genre the song does not have": 1,
                                      "quotes a number not in the notes": 1}


def test_summarize_empty_log():
    s = runlog.summarize_log([])
    assert s["runs"] == 0 and s["explanations_rejected"] == 0


def test_categorize_problem_covers_every_message_the_checker_produces():
    messages = {
        "claims the listener's genre 'rock' but the song is 'pop'": "claims a genre match that is not true",
        "mentions mood 'sad' but the song's mood is 'happy'": "states the wrong mood",
        "mentions 'rap', which is not this song's genre or in its notes": "mentions a genre the song does not have",
        "quotes the number 5, which is not in the notes": "quotes a number not in the notes",
        "calls the song acoustic but its acousticness is low": "describes the sound against its measured values",
        "calls the song high-energy but its energy is low": "describes the sound against its measured values",
        "calls the song calm or slow but its energy is high": "describes the sound against its measured values",
    }
    for message, category in messages.items():
        assert runlog.categorize_problem(message) == category
    assert runlog.categorize_problem("something new") == "other"
