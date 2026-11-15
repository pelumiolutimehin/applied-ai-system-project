from types import SimpleNamespace

import pytest

from src.config import KNOWLEDGE_PATH
from src.profile import UserProfile
from src.reliability import (
    HIGH_CONFIDENCE,
    UNKNOWN_LABEL_CAP,
    Vocabulary,
    _find_mentions,
    _is_negated,
    assess_confidence,
    check_faithfulness,
)
from src.retriever import Retriever, load_notes


@pytest.fixture(scope="module")
def retriever():
    return Retriever(load_notes(KNOWLEDGE_PATH))


@pytest.fixture(scope="module")
def vocab(retriever):
    return Vocabulary.from_notes(retriever.notes)


def song(genre="rock", subgenre="hard rock", mood="intense", energy=0.9, valence=0.3, acousticness=0.1):
    return {"title": "T", "artist": "A", "genre": genre, "subgenre": subgenre, "mood": mood,
            "energy": energy, "valence": valence, "acousticness": acousticness}


def profile(genre="rock", mood="intense", e=0.9, v=0.3, a=0.1):
    return UserProfile(genre, mood, e, v, a)


def check(text, retriever, vocab, s=None, p=None):
    s = s or song()
    return check_faithfulness(text, s, p or profile(), retriever.notes_for_song(s), vocab)


# ---------- faithful text passes ----------

@pytest.mark.parametrize(
    "text",
    [
        "This hard rock track brings heavy riffs and distorted guitars with an intense, driving feel.",
        "Unlike rap, this one is built on loud guitars.",                      # negated genre
        "It is not acoustic at all; it is loud and powerful.",                 # negated feature
        "THIS HARD ROCK SONG IS INTENSE.",                                     # case-insensitive
        "A guitar-driven rock song with a live-band feel.",                   # grounded in the rock note
        "",                                                                    # nothing claimed, nothing wrong
    ],
)
def test_faithful_explanations_pass(text, retriever, vocab):
    result = check(text, retriever, vocab)
    assert result.passed, result.problems


def test_numbers_that_appear_in_the_notes_are_allowed(retriever, vocab):
    trap = song(genre="rap", subgenre="trap", mood="intense")
    ok = check("Trap is known for deep 808 bass.", retriever, vocab, s=trap, p=profile("rap"))
    assert ok.passed, ok.problems
    classic = song(subgenre="classic rock")
    ok = check("Classic rock from the 1960s to the 1980s.", retriever, vocab, s=classic)
    assert ok.passed, ok.problems


def test_short_genre_inside_a_longer_one_is_not_counted_separately(retriever, vocab):
    # song is r&b / hip pop, listener wants pop. "hip pop" must not count as a mention of "pop".
    s = song(genre="r&b", subgenre="hip pop", mood="chill", energy=0.4, acousticness=0.8)
    result = check("This hip pop track blends R&B vocals with hip hop beats.", retriever, vocab,
                   s=s, p=profile("pop", "chill", 0.4, 0.6, 0.8))
    assert result.passed, result.problems


# ---------- unfaithful text is caught ----------

@pytest.mark.parametrize(
    "text, fragment",
    [
        ("This rap song is full of attitude.", "rap"),                          # invented genre
        ("A great edm banger.", "edm"),                                         # invented genre
        ("This progressive electro house anthem hits hard.", "progressive electro house"),  # wrong subgenre
        ("It scores 0.95 on energy.", "0.95"),                                  # ungrounded number
        ("The song has a relaxed, chill mood.", "chill"),                       # wrong mood
        ("A sad and slow ballad.", "sad"),
        ("A beautiful acoustic sound.", "acoustic"),                            # acousticness is 0.1
        ("A calm and quiet track.", "calm"),                                    # energy is 0.9
    ],
)
def test_unfaithful_explanations_are_rejected(text, fragment, retriever, vocab):
    result = check(text, retriever, vocab)
    assert not result.passed
    assert any(fragment in problem for problem in result.problems), result.problems


def test_high_energy_claim_is_rejected_for_a_low_energy_song(retriever, vocab):
    result = check("A high-energy workout track.", retriever, vocab, s=song(energy=0.2))
    assert not result.passed and any("high-energy" in p for p in result.problems)


def test_claiming_the_listeners_genre_for_a_different_genre_is_rejected(retriever, vocab):
    s = song(genre="pop", subgenre="dance pop", mood="happy", energy=0.8, valence=0.8)
    result = check("This fits your rock taste perfectly.", retriever, vocab, s=s, p=profile("rock", "happy"))
    assert not result.passed
    assert any("listener's genre 'rock'" in p for p in result.problems)


def test_naming_the_listeners_genre_is_fine_when_negated_or_matching(retriever, vocab):
    s = song(genre="pop", subgenre="dance pop", mood="happy", energy=0.8, valence=0.8)
    negated = check("Although it is not rock, it is upbeat dance pop.", retriever, vocab, s=s, p=profile("rock", "happy"))
    assert negated.passed, negated.problems
    matching = check("A rock song for your rock taste.", retriever, vocab, p=profile("rock"))
    assert matching.passed, matching.problems


def test_multiple_problems_are_all_reported(retriever, vocab):
    result = check("A calm acoustic rap song that scores 0.9.", retriever, vocab)
    assert len(result.problems) >= 4


# ---------- helpers ----------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("this is not rock", True),
        ("it isn't rock", True),
        ("unlike rock", True),
        ("this is rock", False),
        ("not a chance this is truly a great rock", False),   # negation is too far from the term
    ],
)
def test_negation_window(text, expected):
    assert _is_negated(text, text.rindex("rock")) is expected


def test_find_mentions_masks_terms_inside_longer_terms():
    found = {term for term, _ in _find_mentions("a hip pop and a pop song", ["pop", "hip pop"])}
    assert found == {"pop", "hip pop"}
    only_long = {term for term, _ in _find_mentions("a hip pop song", ["pop", "hip pop"])}
    assert only_long == {"hip pop"}


def test_r_and_b_is_matched_as_a_whole_word():
    assert _find_mentions("smooth r&b vocals", ["r&b"])
    assert not _find_mentions("a herb&bread shop", ["r&b"])


def test_vocabulary_is_built_from_the_notes(vocab):
    assert {"pop", "rock", "r&b", "edm", "latin", "rap"} == set(vocab.genres)
    assert "hard rock" in vocab.subgenres
    assert {"happy", "chill", "intense", "sad", "melancholic"} == set(vocab.moods)


# ---------- confidence ----------

CATALOG_GENRES = {"pop", "rock", "r&b", "edm", "latin", "rap"}
CATALOG_MOODS = {"happy", "chill", "intense", "sad", "melancholic"}


def recs(*songs_and_scores):
    return [(s, score, "reasons") for s, score in songs_and_scores]


def assess(p, r, explanations=None):
    return assess_confidence(p, r, CATALOG_GENRES, CATALOG_MOODS, explanations)


def test_strong_consistent_results_get_high_confidence():
    report = assess(profile(), recs(*[(song(), 0.99)] * 5))
    assert report.level == "high" and report.score >= HIGH_CONFIDENCE
    assert report.notes == []


def test_unknown_labels_cap_the_score_and_explain_why():
    report = assess(profile("k-pop", "nostalgia"), recs(*[(song(), 0.99)] * 5))
    assert report.score <= UNKNOWN_LABEL_CAP and report.level == "low"
    assert any("not in the catalog" in n for n in report.notes)


def test_weak_matches_with_no_support_are_low_confidence():
    other = song(genre="pop", subgenre="dance pop", mood="happy", energy=0.1)
    report = assess(profile(), recs(*[(other, 0.3)] * 5))
    assert report.level == "low"
    assert any("requested genre" in n for n in report.notes)
    assert any("requested mood" in n for n in report.notes)


def test_incoherent_profile_is_flagged():
    # "intense" means high energy and low positivity; 0.9 positivity contradicts it
    report = assess(profile(v=0.9), recs(*[(song(), 0.99)] * 5))
    assert report.factors["coherence"] == 0.0
    assert any("disagrees" in n for n in report.notes)
    assert report.score < assess(profile(), recs(*[(song(), 0.99)] * 5)).score


def test_no_genre_requested_drops_that_factor_and_rescales():
    report = assess(profile(genre=None), recs(*[(song(), 0.99)] * 5))
    assert report.factors["genre_support"] is None
    assert report.level == "high"  # not penalised for having no genre


def test_rejected_explanations_lower_confidence():
    good = [SimpleNamespace(faithful=True)] * 5
    mixed = [SimpleNamespace(faithful=True)] * 3 + [SimpleNamespace(faithful=False)] * 2
    r = recs(*[(song(), 0.9)] * 5)
    high, lower = assess(profile(), r, good), assess(profile(), r, mixed)
    assert lower.score < high.score
    assert lower.factors["faithful"] == pytest.approx(0.6)
    assert any("replaced" in n for n in lower.notes)


def test_unchecked_explanations_do_not_count():
    report = assess(profile(), recs(*[(song(), 0.99)] * 3), [SimpleNamespace(faithful=None)] * 3)
    assert report.factors["faithful"] is None


def test_no_results_is_low_confidence():
    report = assess(profile(), [])
    assert report.level == "low" and report.score == 0.0


def test_score_is_always_between_zero_and_one():
    for strength in (0.0, 0.5, 1.0):
        for p in (profile(), profile("k-pop", "x"), profile(genre=None)):
            report = assess(p, recs(*[(song(), strength)] * 3))
            assert 0.0 <= report.score <= 1.0


def test_a_genre_the_catalog_lacks_cannot_be_claimed_as_the_listeners_taste(retriever, vocab):
    # listener asked for "k-pop" (not a catalog genre); the song is pop
    s = song(genre="pop", subgenre="dance pop", mood="happy", energy=0.8, valence=0.8)
    p = profile("k-pop", "happy")
    bad = check("Perfect for your k-pop taste.", retriever, vocab, s=s, p=p)
    assert not bad.passed and any("k-pop" in x for x in bad.problems)
    fine = check("Not k-pop, but an upbeat dance pop track.", retriever, vocab, s=s, p=p)
    assert fine.passed, fine.problems


def test_negation_does_not_leak_across_sentences(retriever, vocab):
    s = song(genre="pop", subgenre="dance pop", mood="happy", energy=0.8, valence=0.8)
    p = profile("rock", "happy")
    leaked = check("This one is not quiet at all. Perfect for your rock taste.", retriever, vocab, s=s, p=p)
    assert not leaked.passed, "a negation in the previous sentence must not excuse the claim"
    same_sentence = check("Although it is not rock, it is upbeat dance pop.", retriever, vocab, s=s, p=p)
    assert same_sentence.passed, same_sentence.problems


def test_negation_window_stops_at_sentence_boundary():
    text = "It is not loud. Rock."
    assert _is_negated(text, text.index("Rock")) is False
    assert _is_negated("It is not rock", len("It is not ")) is True
