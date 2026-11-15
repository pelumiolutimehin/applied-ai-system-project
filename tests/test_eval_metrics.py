import json

import pytest

from eval import metrics, profiles
from src.config import CATALOG_PATH
from src.profile import validate_profile
from src.recommender import load_songs


def prof(genre="pop", mood="happy", e=0.9, v=0.8, a=0.1):
    return validate_profile({"favorite_genre": genre, "favorite_mood": mood,
                             "target_energy": e, "target_valence": v, "target_acousticness": a})


NAMED = profiles._case("pop run", "pop", {"happy"}, energy=(0.6, 1), valence=(0.6, 1))
UNNAMED = profiles._case("focus", None, {"chill"}, energy=(0, 0.5), acceptable_genres={"r&b", "pop"})


def outcome(p=None, error=None, text="x"):
    return {"text": text, "profile": p, "error": error}


# ---------- score_parser_case ----------

def test_named_case_passes_when_everything_matches():
    s = metrics.score_parser_case(NAMED, prof())
    assert s["passed"] and s["genre_ok"] and s["mood_ok"] and s["numeric_ok"] and s["coherent"]


@pytest.mark.parametrize(
    "bad, field",
    [(prof(genre="rock"), "genre_ok"), (prof(mood="sad"), "mood_ok"), (prof(e=0.2), "numeric_ok")],
)
def test_named_case_fails_on_each_wrong_field(bad, field):
    s = metrics.score_parser_case(NAMED, bad)
    assert not s[field] and not s["passed"]


def test_unnamed_case_fails_if_a_genre_is_invented():
    invented = metrics.score_parser_case(UNNAMED, prof(genre="rock", mood="chill", e=0.3, v=0.6))
    assert invented["genre_invented"] and not invented["passed"]


def test_unnamed_case_passes_without_a_genre():
    no_genre = prof(genre="rock", mood="chill", e=0.3, v=0.6)
    object.__setattr__(no_genre, "favorite_genre", None)  # the post-fix parser returns None
    s = metrics.score_parser_case(UNNAMED, no_genre)
    assert not s["genre_invented"] and s["passed"]


def focus_profile(genre):
    p = prof(genre="rock", mood="chill", e=0.3, v=0.6)
    object.__setattr__(p, "favorite_genre", genre)  # None is what the post-fix parser returns
    return p


@pytest.mark.parametrize(
    "genre, tier, passed, passed_strict",
    [
        (None, "ideal", True, True),
        ("pop", "acceptable", True, False),     # defensible pick: passes tiered, fails strict
        ("r&b", "acceptable", True, False),
        ("rock", "wrong", False, False),
    ],
)
def test_no_genre_request_is_graded_in_three_tiers(genre, tier, passed, passed_strict):
    s = metrics.score_parser_case(UNNAMED, focus_profile(genre))
    assert s["genre_tier"] == tier
    assert s["passed"] is passed
    assert s["passed_strict"] is passed_strict
    assert s["genre_invented"] is (genre is not None)


def test_genre_tier_for_with_no_acceptable_set_makes_any_genre_wrong():
    assert metrics.genre_tier_for("pop", []) == "wrong"
    assert metrics.genre_tier_for(None, []) == "ideal"


def test_named_case_has_no_tier():
    assert metrics.score_parser_case(NAMED, prof())["genre_tier"] is None


def test_wrong_mood_still_fails_even_with_an_acceptable_genre():
    s = metrics.score_parser_case(UNNAMED, focus_profile("pop"))
    assert s["passed"]
    bad = focus_profile("pop")
    object.__setattr__(bad, "favorite_mood", "intense")
    assert not metrics.score_parser_case(UNNAMED, bad)["passed"]


def test_refused_request_scores_as_total_miss():
    s = metrics.score_parser_case(NAMED, None)
    assert not s["parsed"] and not s["passed"] and s["coherent"] is None


def test_coherence_flags_mood_that_disagrees_with_its_numbers():
    # intense means high energy AND valence below 0.5; v=0.8 contradicts it
    assert metrics.score_parser_case(NAMED, prof(mood="intense", e=0.9, v=0.8))["coherent"] is False
    assert metrics.score_parser_case(NAMED, prof(mood="intense", e=0.9, v=0.3))["coherent"] is True


# ---------- summaries ----------

def test_summarize_parser_rates():
    cases = [NAMED, NAMED, UNNAMED, UNNAMED]
    outcomes = [
        outcome(prof()),                                              # pass
        outcome(None, "refused"),                                     # miss
        outcome(prof(genre="rock", mood="chill", e=0.3, v=0.6)),      # genre invented
        outcome(prof(genre="rock", mood="chill", e=0.3, v=0.6)),      # genre invented
    ]
    s = metrics.summarize_parser(cases, outcomes)
    assert s["cases"] == 4
    assert s["parse_success_rate"] == 0.75
    assert s["named_genre_accuracy"] == 0.5
    assert s["genre_invented_rate"] == 1.0
    assert s["genre_indefensible_rate"] == 1.0   # rock is not in UNNAMED's acceptable set
    assert s["no_genre_tiers"] == {"ideal": 0, "acceptable": 0, "wrong": 2}
    assert s["overall_case_pass_rate"] == 0.25
    assert s["overall_case_pass_rate_strict"] == 0.25
    assert len(s["per_case"]) == 4 and s["per_case"][1]["error"] == "refused"
    json.dumps(s)  # the report must be JSON-serializable


def test_off_topic_rejection_rate_and_accepted_list():
    out = metrics.off_topic_rejection([outcome(None, "no"), outcome(prof(), text="oops")])
    assert out["rejection_rate"] == 0.5 and out["accepted"] == ["oops"]


# ---------- consistency ----------

def runs_of(*profile_lists):
    return [[outcome(p) for p in plist] for plist in profile_lists]


def test_consistency_exact_and_tolerance():
    texts = ["a", "b"]
    runs = runs_of(
        [prof(), prof(genre="rock")],
        [prof(a=0.2), prof(genre="rock")],      # a: 0.1 jitter within tolerance; b identical
        [prof(), prof(genre="rock")],
    )
    r = metrics.consistency(texts, runs)
    assert r["exact_identical_rate"] == 0.5
    assert r["within_tolerance_rate"] == 1.0
    assert r["max_numeric_deviation_seen"] == pytest.approx(0.1)
    assert r["unstable"] == []


def test_consistency_flags_label_flip_and_large_drift():
    texts = ["flip", "drift"]
    runs = runs_of(
        [prof(genre="rock"), prof(e=0.9)],
        [prof(genre="pop"), prof(e=0.5)],
    )
    r = metrics.consistency(texts, runs)
    assert r["within_tolerance_rate"] == 0.0
    assert r["unstable"] == ["flip", "drift"]


def test_consistency_error_vs_profile_is_unstable_but_two_errors_match():
    runs = [[outcome(None, "boom")], [outcome(prof())]]
    assert metrics.consistency(["t"], runs)["unstable"] == ["t"]
    runs = [[outcome(None, "boom")], [outcome(None, "boom")]]
    assert metrics.consistency(["t"], runs)["exact_identical_rate"] == 1.0


# ---------- run_parser ----------

def test_run_parser_collects_profiles_errors_and_waits_between_calls():
    good = json.dumps({"favorite_genre": "pop", "favorite_mood": "happy",
                       "target_energy": 0.8, "target_valence": 0.8, "target_acousticness": 0.1})
    replies = iter([good, "{}"])
    naps = []
    out = metrics.run_parser(["one", "two"], ["pop"], ["happy"], generate=lambda s, u: next(replies),
                             delay=2.5, sleep=naps.append)
    assert out[0]["profile"].favorite_genre == "pop" and out[0]["error"] is None
    assert out[1]["profile"] is None and "couldn't find" in out[1]["error"]
    assert naps == [2.5]


# ---------- guardrail and ranking ----------

def test_guardrail_suite_catches_every_malformed_profile_and_accepts_every_valid_one():
    r = metrics.guardrail_report(profiles.ALL_VALID_PROFILES, profiles.MALFORMED_PROFILES)
    assert r["wrongly_accepted"] == [] and r["wrongly_rejected"] == []
    assert r["malformed_rejected_rate"] == 1.0 and r["valid_accepted_rate"] == 1.0


def test_guardrail_report_names_the_misclassified_profiles():
    r = metrics.guardrail_report({"bad_valid": {"favorite_genre": "pop"}}, {"ok_malformed": profiles.NORMAL_PROFILES["starter"]})
    assert r["wrongly_rejected"] == ["bad_valid"] and r["wrongly_accepted"] == ["ok_malformed"]


def test_sweep_profiles_covers_the_full_grid_with_valid_profiles():
    grid = metrics.sweep_profiles(["pop", "rock"], ["happy", "sad", "chill"])
    assert len(grid) == 2 * 3 * len(metrics.SWEEP_PRESETS)
    for raw in grid:
        validate_profile(raw)


def test_ranking_metrics_on_the_real_catalog():
    songs = load_songs(str(CATALOG_PATH))
    genres = sorted({s["genre"] for s in songs})
    moods = sorted({s["mood"] for s in songs})
    r = metrics.ranking_metrics(songs, metrics.sweep_profiles(genres, moods), k=5)
    assert r["profiles"] == len(genres) * len(moods) * 3
    assert 0 < r["coverage"] <= 1
    # coverage is rounded to 3 decimals, so allow less than one song of rounding error
    assert abs(r["never_recommended"] - r["catalog_size"] * (1 - r["coverage"])) < 1
    assert sum(r["recommended_genre_share"].values()) == pytest.approx(1.0, abs=0.01)


def test_ranking_metrics_tiny_catalog_by_hand():
    songs = [
        {"id": 1, "title": "A", "artist": "X", "genre": "pop", "mood": "happy", "energy": 0.9, "valence": 0.9, "acousticness": 0.1},
        {"id": 2, "title": "B", "artist": "Y", "genre": "rock", "mood": "sad", "energy": 0.1, "valence": 0.1, "acousticness": 0.9},
        {"id": 3, "title": "C", "artist": "Z", "genre": "rock", "mood": "sad", "energy": 0.1, "valence": 0.1, "acousticness": 0.9},
    ]
    profile = {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.9, "target_valence": 0.9, "target_acousticness": 0.1}
    r = metrics.ranking_metrics(songs, [profile], k=1)
    assert r["coverage"] == pytest.approx(0.333, abs=0.001)
    assert r["genre_match_rate"] == 1.0 and r["mood_match_rate"] == 1.0
    assert r["never_recommended"] == 2


# ---------- the labeled data itself ----------

def test_labeled_cases_only_use_labels_that_exist_in_the_catalog():
    songs = load_songs(str(CATALOG_PATH))
    genres = {s["genre"] for s in songs}
    moods = {s["mood"] for s in songs}
    for case in profiles.NAMED_GENRE_CASES + profiles.NO_GENRE_CASES:
        assert case["genre"] is None or case["genre"] in genres, case["text"]
        assert case["moods"] <= moods, case["text"]
    assert all(c["genre"] is not None for c in profiles.NAMED_GENRE_CASES)
    assert all(c["genre"] is None for c in profiles.NO_GENRE_CASES)


def test_labeled_texts_are_unique_and_within_the_length_limit():
    from src.llm_parser import MAX_REQUEST_CHARS
    texts = [c["text"] for c in profiles.NAMED_GENRE_CASES + profiles.NO_GENRE_CASES] + profiles.OFF_TOPIC_TEXTS
    assert len(texts) == len(set(texts))
    assert all(len(t) <= MAX_REQUEST_CHARS for t in texts)


def test_summary_reports_strict_and_tiered_pass_rates_side_by_side():
    cases = [UNNAMED, UNNAMED, UNNAMED]
    outcomes = [outcome(focus_profile(None)), outcome(focus_profile("pop")), outcome(focus_profile("rock"))]
    s = metrics.summarize_parser(cases, outcomes)
    assert s["no_genre_tiers"] == {"ideal": 1, "acceptable": 1, "wrong": 1}
    assert s["genre_invented_rate"] == pytest.approx(0.667, abs=0.001)
    assert s["genre_indefensible_rate"] == pytest.approx(0.333, abs=0.001)
    assert s["overall_case_pass_rate"] == pytest.approx(0.667, abs=0.001)
    assert s["overall_case_pass_rate_strict"] == pytest.approx(0.333, abs=0.001)


def test_every_no_genre_case_defines_acceptable_genres_from_the_catalog():
    genres = {s["genre"] for s in load_songs(str(CATALOG_PATH))}
    for case in profiles.NO_GENRE_CASES:
        assert case["acceptable_genres"], case["text"]
        assert case["acceptable_genres"] <= genres, case["text"]
    for case in profiles.NAMED_GENRE_CASES:
        assert not case["acceptable_genres"]


# ---------- regrade ----------

def test_regrade_applies_current_answer_key_to_saved_answers():
    from eval.regrade import regrade_parser
    cases = profiles.NAMED_GENRE_CASES + profiles.NO_GENRE_CASES
    saved = []
    for case in cases:
        genre = case["genre"] or sorted(case["acceptable_genres"])[0]    # an acceptable genre
        mood = sorted(case["moods"])[0]
        lo = lambda b, d: (b[0] + b[1]) / 2 if b else d
        saved.append({"text": case["text"], "error": None, "profile": {
            "favorite_genre": genre, "favorite_mood": mood,
            "target_energy": lo(case["energy"], 0.5), "target_valence": lo(case["valence"], 0.5),
            "target_acousticness": lo(case["acoustic"], 0.5)}})
    out = regrade_parser({"per_case": saved})
    assert out["overall_case_pass_rate"] == 1.0
    assert out["overall_case_pass_rate_strict"] == pytest.approx(8 / 13, abs=0.001)
    assert out["no_genre_tiers"] == {"ideal": 0, "acceptable": 5, "wrong": 0}


def test_regrade_refuses_a_report_missing_a_case():
    from eval.regrade import regrade_parser
    with pytest.raises(SystemExit):
        regrade_parser({"per_case": []})
