"""Pure evaluation functions. No printing and no file I/O, so they can be unit-tested offline."""

import time
from collections import Counter
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from src.catalog import derive_mood
from src.llm_parser import Generate, ParseError, parse_request
from src.profile import ProfileError, UserProfile, validate_profile
from src.explainer import explain_recommendations
from src.recommender import recommend_songs
from src.reliability import Vocabulary, assess_confidence, check_faithfulness

API_FAILURE_PREFIX = "The language model request failed"  # from llm_parser.parse_request

SWEEP_PRESETS = {  # (energy, valence, acousticness)
    "calm": (0.2, 0.4, 0.8),
    "mid": (0.5, 0.5, 0.5),
    "intense": (0.9, 0.4, 0.1),
}


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return round(numerator / denominator, 3) if denominator else None


def in_range(value: float, bounds) -> bool:
    return bounds is None or bounds[0] <= value <= bounds[1]


# ---------- parser ----------

def genre_tier_for(genre: Optional[str], acceptable: Iterable[str]) -> str:
    """Grade the genre a parser returned for a request that named none."""
    if genre is None:
        return "ideal"
    return "acceptable" if genre in set(acceptable) else "wrong"


def score_parser_case(case: Dict, profile: Optional[UserProfile]) -> Dict:
    """Compare one parsed profile with the labeled expectations. profile=None means it was refused."""
    if profile is None:
        return {"parsed": False, "genre_ok": False, "genre_invented": None, "genre_tier": None,
                "mood_ok": False, "numeric_ok": False, "coherent": None,
                "passed": False, "passed_strict": False}

    named = case["genre"] is not None
    genre_ok = profile.favorite_genre == case["genre"] if named else None
    genre_invented = (profile.favorite_genre is not None) if not named else None
    genre_tier = None if named else genre_tier_for(profile.favorite_genre, case["acceptable_genres"])
    mood_ok = profile.favorite_mood in case["moods"]
    numeric_ok = (
        in_range(profile.target_energy, case["energy"])
        and in_range(profile.target_valence, case["valence"])
        and in_range(profile.target_acousticness, case["acoustic"])
    )
    coherent = profile.favorite_mood == derive_mood(profile.target_energy, profile.target_valence)
    rest_ok = mood_ok and numeric_ok
    # tiered: no genre or an acceptable genre passes. strict: any invented genre fails.
    passed = (genre_ok if named else genre_tier in ("ideal", "acceptable")) and rest_ok
    passed_strict = (genre_ok if named else not genre_invented) and rest_ok
    return {"parsed": True, "genre_ok": genre_ok, "genre_invented": genre_invented, "genre_tier": genre_tier,
            "mood_ok": mood_ok, "numeric_ok": numeric_ok, "coherent": coherent,
            "passed": bool(passed), "passed_strict": bool(passed_strict)}


def run_parser(
    texts: Iterable[str],
    genres: Sequence[str],
    moods: Sequence[str],
    generate: Optional[Generate] = None,
    delay: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
) -> List[Dict]:
    """Parse each text; returns [{'text', 'profile' (UserProfile or None), 'error'}]."""
    outcomes = []
    for i, text in enumerate(texts):
        if i and delay:
            sleep(delay)
        try:
            outcomes.append({"text": text, "profile": parse_request(text, genres, moods, generate=generate),
                             "error": None, "api_error": False})
        except ParseError as err:
            outcomes.append({"text": text, "profile": None, "error": str(err),
                             "api_error": str(err).startswith(API_FAILURE_PREFIX)})
    return outcomes


def count_api_errors(*outcome_lists: Sequence[Dict]) -> int:
    return sum(1 for outcomes in outcome_lists for o in outcomes if o.get("api_error"))


def summarize_parser(cases: Sequence[Dict], outcomes: Sequence[Dict]) -> Dict:
    """Aggregate per-case scores. Refused requests count as misses on every labeled expectation."""
    scored = [score_parser_case(c, o["profile"]) for c, o in zip(cases, outcomes)]
    n = len(scored)
    parsed = [s for s in scored if s["parsed"]]
    named = [s for c, s in zip(cases, scored) if c["genre"] is not None]
    unnamed_parsed = [s for c, s in zip(cases, scored) if c["genre"] is None and s["parsed"]]
    tiers = Counter(s["genre_tier"] for s in unnamed_parsed)
    return {
        "cases": n,
        "parse_success_rate": _rate(len(parsed), n),
        "named_genre_accuracy": _rate(sum(1 for s in named if s["genre_ok"]), len(named)),
        "genre_invented_rate": _rate(sum(1 for s in unnamed_parsed if s["genre_invented"]), len(unnamed_parsed)),
        "genre_indefensible_rate": _rate(tiers["wrong"], len(unnamed_parsed)),
        "no_genre_tiers": {t: tiers[t] for t in ("ideal", "acceptable", "wrong")},
        "mood_accuracy": _rate(sum(1 for s in scored if s["mood_ok"]), n),
        "numeric_in_range_rate": _rate(sum(1 for s in scored if s["numeric_ok"]), n),
        "mood_numbers_coherence": _rate(sum(1 for s in parsed if s["coherent"]), len(parsed)),
        "overall_case_pass_rate": _rate(sum(1 for s in scored if s["passed"]), n),
        "overall_case_pass_rate_strict": _rate(sum(1 for s in scored if s["passed_strict"]), n),
        "per_case": [
            {"text": c["text"], **s,
             "profile": o["profile"].to_dict() if o["profile"] else None, "error": o["error"]}
            for c, s, o in zip(cases, scored, outcomes)
        ],
    }


def off_topic_rejection(outcomes: Sequence[Dict]) -> Dict:
    rejected = sum(1 for o in outcomes if o["profile"] is None)
    return {"cases": len(outcomes), "rejection_rate": _rate(rejected, len(outcomes)),
            "accepted": [o["text"] for o in outcomes if o["profile"] is not None]}


CONSISTENCY_TOLERANCE = 0.1


def _profile_key(outcome: Dict):
    return outcome["profile"].to_dict() if outcome["profile"] else {"error": outcome["error"]}


def _max_deviation(a: Dict, b: Dict) -> Optional[float]:
    """Largest numeric difference between two profile dicts, or None if the labels differ."""
    if set(a) != set(b):
        return None
    deviation = 0.0
    for key in a:
        if isinstance(a[key], (int, float)) and isinstance(b[key], (int, float)):
            deviation = max(deviation, abs(a[key] - b[key]))
        elif a[key] != b[key]:
            return None
    return deviation


def consistency(texts: Sequence[str], runs: Sequence[Sequence[Dict]], tolerance: float = CONSISTENCY_TOLERANCE) -> Dict:
    """runs[r] is the outcomes list for run r over the same texts.

    exact: identical profile every run. within_tolerance: same labels every run and every
    number within `tolerance` of the first run (LLM output jitters in the last digit).
    """
    exact = close = 0
    unstable = []
    skipped = []
    worst = 0.0
    for i, text in enumerate(texts):
        if any(run[i].get("api_error") for run in runs):
            skipped.append(text)  # an API failure says nothing about whether the model is consistent
            continue
        keys = [_profile_key(run[i]) for run in runs]
        if all(k == keys[0] for k in keys):
            exact += 1
            close += 1
            continue
        deviations = [_max_deviation(keys[0], k) for k in keys[1:]]
        if all(d is not None and d <= tolerance + 1e-9 for d in deviations):
            close += 1
            worst = max(worst, *[d for d in deviations if d is not None])
        else:
            unstable.append(text)
    judged = len(texts) - len(skipped)
    return {
        "texts": len(texts), "judged": judged, "skipped_api_errors": skipped,
        "runs": len(runs), "tolerance": tolerance,
        "exact_identical_rate": _rate(exact, judged),
        "within_tolerance_rate": _rate(close, judged),
        "max_numeric_deviation_seen": round(worst, 3),
        "unstable": unstable,
    }


# ---------- guardrail ----------

def guardrail_report(valid: Dict[str, object], malformed: Dict[str, object]) -> Dict:
    def accepted(raw) -> bool:
        try:
            validate_profile(raw)
            return True
        except ProfileError:
            return False

    wrongly_rejected = [name for name, raw in valid.items() if not accepted(raw)]
    wrongly_accepted = [name for name, raw in malformed.items() if accepted(raw)]
    return {
        "malformed_rejected_rate": _rate(len(malformed) - len(wrongly_accepted), len(malformed)),
        "valid_accepted_rate": _rate(len(valid) - len(wrongly_rejected), len(valid)),
        "wrongly_accepted": wrongly_accepted,
        "wrongly_rejected": wrongly_rejected,
    }


# ---------- ranking and bias ----------

def sweep_profiles(genres: Sequence[str], moods: Sequence[str]) -> List[Dict]:
    """Every genre x mood x numeric preset: a systematic grid of profiles to probe the ranker."""
    profiles = []
    for genre in sorted(genres):
        for mood in sorted(moods):
            for energy, valence, acoustic in SWEEP_PRESETS.values():
                profiles.append({"favorite_genre": genre, "favorite_mood": mood, "target_energy": energy,
                                 "target_valence": valence, "target_acousticness": acoustic})
    return profiles


def ranking_metrics(songs: Sequence[Dict], profiles: Sequence[Dict], k: int = 5) -> Dict:
    """Coverage, concentration and match rates over many profiles."""
    slots = Counter()
    artist_slots = Counter()
    genre_of_recs = Counter()
    genre_hits = mood_hits = total = 0
    for raw in profiles:
        profile = validate_profile(raw)
        for song, _score, _why in recommend_songs(profile, songs, k=k):
            slots[song["id"]] += 1
            artist_slots[song["artist"]] += 1
            genre_of_recs[song["genre"]] += 1
            genre_hits += song["genre"] == profile.favorite_genre
            mood_hits += song["mood"] == profile.favorite_mood
            total += 1
    top10 = sum(count for _, count in slots.most_common(10))
    return {
        "profiles": len(profiles),
        "k": k,
        "catalog_size": len(songs),
        "coverage": _rate(len(slots), len(songs)),
        "never_recommended": len(songs) - len(slots),
        "top10_songs_share_of_slots": _rate(top10, total),
        "max_artist_share_of_slots": _rate(artist_slots.most_common(1)[0][1], total),
        "genre_match_rate": _rate(genre_hits, total),
        "mood_match_rate": _rate(mood_hits, total),
        "recommended_genre_share": {g: _rate(c, total) for g, c in sorted(genre_of_recs.items())},
        "catalog_genre_share": {g: _rate(c, len(songs)) for g, c in sorted(Counter(s["genre"] for s in songs).items())},
        "catalog_mood_share": {m: _rate(c, len(songs)) for m, c in sorted(Counter(s["mood"] for s in songs).items())},
    }


# ---------- confidence and explanation faithfulness ----------

def confidence_check(songs: Sequence[Dict], profiles: Dict[str, Dict], expected: Dict[str, set], k: int = 5) -> Dict:
    """Score each profile and compare the level with what a sensible score should give."""
    genres = {s["genre"] for s in songs}
    moods = {s["mood"] for s in songs}
    rows = {}
    for name, raw in profiles.items():
        profile = validate_profile(raw)
        report = assess_confidence(profile, recommend_songs(profile, songs, k=k), genres, moods)
        want = expected.get(name)
        rows[name] = {"score": report.score, "level": report.level,
                      "expected": sorted(want) if want else None,
                      "as_expected": None if want is None else report.level in want}
    judged = [r for r in rows.values() if r["as_expected"] is not None]
    return {
        "profiles": len(rows),
        "as_expected_rate": _rate(sum(1 for r in judged if r["as_expected"]), len(judged)),
        "per_profile": rows,
    }


def explanation_faithfulness(
    songs: Sequence[Dict],
    profiles: Dict[str, Dict],
    retriever,
    generate: Optional[Generate] = None,
    k: int = 5,
    delay: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Dict:
    """Generate real explanations for each profile and count how many the checker rejects.

    rejected_examples keeps the model's raw text and the reasons so a person can judge whether
    each rejection was right. The checker is a heuristic, so that review matters.
    """
    llm = rejected = unusable = 0
    failures: Counter = Counter()
    reasons: Counter = Counter()
    examples = []
    samples = []
    for i, (name, raw) in enumerate(profiles.items()):
        if i and delay:
            sleep(delay)
        profile = validate_profile(raw)
        recs = recommend_songs(profile, songs, k=k)
        for rec, exp in zip(recs, explain_recommendations(profile, recs, retriever, generate=generate)):
            if exp.source == "llm":
                llm += 1
                samples.append({"profile": name, "song_id": rec[0]["id"], "text": exp.text})
            elif exp.faithful is False:
                rejected += 1
                for problem in exp.problems:
                    reasons[problem.split(",")[0][:60]] += 1
                examples.append({"profile": name, "song": rec[0]["title"], "text": exp.rejected_text,
                                 "problems": exp.problems})
            else:
                unusable += 1
                failures[exp.error or "unknown"] += 1
    checked = llm + rejected
    return {
        "explanations": llm + rejected + unusable,
        "passed_check": llm,
        "rejected": rejected,
        "no_usable_text": unusable,
        "no_usable_text_reasons": dict(failures),
        "rejection_rate": _rate(rejected, checked),
        "rejected_examples": examples,
        "samples": samples,  # every accepted text, kept for the mutation test and for human review
    }


# ---------- validating the checker itself ----------

def _unused(candidates: Sequence[str], forbidden_text: str, exclude: Iterable[str]) -> Optional[str]:
    """First candidate that is neither excluded nor mentioned in forbidden_text."""
    skip = {x.lower() for x in exclude if x}
    for word in candidates:
        if word.lower() not in skip and word.lower() not in forbidden_text.lower():
            return word
    return None


def _mutations(song: Dict, profile, notes_text: str, genres: Sequence[str], moods: Sequence[str]) -> Dict[str, Optional[str]]:
    """Sentences that would make a faithful explanation false. None = does not apply to this song."""
    other_genre = _unused(genres, notes_text, [song["genre"], song.get("subgenre"), profile.favorite_genre])
    other_mood = _unused(moods, notes_text, [song["mood"]])
    listener_genre = profile.favorite_genre
    return {
        "invented_genre": f" This is a typical {other_genre} track." if other_genre else None,
        "wrong_mood": f" Its {other_mood} feel stands out." if other_mood else None,
        "made_up_number": " It scores 0.97 overall.",
        "false_acoustic": " Its acoustic warmth shines." if song["acousticness"] < 0.5 else None,
        "false_energy": " A calm, quiet listen." if song["energy"] >= 0.6 else " A loud, high-energy blast.",
        "false_genre_match": (f" Perfect for your {listener_genre} taste."
                              if listener_genre and listener_genre != song["genre"] else None),
    }


def checker_mutation_test(samples: Sequence[Dict], songs: Sequence[Dict], profiles: Dict[str, Dict], retriever) -> Dict:
    """Corrupt real accepted explanations on purpose and measure how many corruptions are caught.

    Also reports how many of the untouched originals the checker flags (false alarms). The
    corruptions are blunt, one false sentence each, so recall here shows the mechanism works and
    says nothing about subtle errors.
    """
    by_id = {s["id"]: s for s in songs}
    vocab = Vocabulary.from_notes(retriever.notes)
    genres, moods = sorted(vocab.genres), sorted(vocab.moods)
    applied: Counter = Counter()
    caught: Counter = Counter()
    false_alarms = 0
    for sample in samples:
        song = by_id[sample["song_id"]]
        profile = validate_profile(profiles[sample["profile"]])
        notes = retriever.notes_for_song(song)
        notes_text = " ".join(f"{n.name} {n.text}" for n in notes)
        if not check_faithfulness(sample["text"], song, profile, notes, vocab).passed:
            false_alarms += 1
        for name, addition in _mutations(song, profile, notes_text, genres, moods).items():
            if addition is None:
                continue
            applied[name] += 1
            if not check_faithfulness(sample["text"] + addition, song, profile, notes, vocab).passed:
                caught[name] += 1
    return {
        "originals": len(samples),
        "false_alarm_rate": _rate(false_alarms, len(samples)),
        "overall_recall": _rate(sum(caught.values()), sum(applied.values())),
        "per_mutation": {name: {"applied": applied[name], "caught": caught[name],
                                "recall": _rate(caught[name], applied[name])} for name in sorted(applied)},
    }
