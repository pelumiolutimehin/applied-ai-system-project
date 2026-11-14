import pytest

from src.config import CATALOG_PATH, LATIN_EXCLUSIONS_PATH
from src.catalog import GENRES, derive_mood, is_latin_script, load_exclusions
from src.recommender import load_songs


@pytest.mark.parametrize(
    "energy, valence, expected",
    [
        (0.9, 0.9, "happy"),
        (0.9, 0.2, "intense"),
        (0.3, 0.8, "chill"),
        (0.3, 0.1, "sad"),
        (0.3, 0.4, "melancholic"),
        # boundaries: thresholds are inclusive on the "high" side
        (0.60, 0.50, "happy"),
        (0.60, 0.49, "intense"),
        (0.59, 0.50, "chill"),
        (0.59, 0.30, "melancholic"),
        (0.59, 0.29, "sad"),
    ],
)
def test_derive_mood(energy, valence, expected):
    assert derive_mood(energy, valence) == expected


def test_catalog_schema_and_ranges():
    songs = load_songs(str(CATALOG_PATH))
    assert len(songs) >= 150
    ids = [s["id"] for s in songs]
    assert len(ids) == len(set(ids))
    for s in songs:
        assert s["title"].strip() and s["artist"].strip()
        for key in ("energy", "valence", "danceability", "acousticness"):
            assert 0.0 <= s[key] <= 1.0


def test_catalog_has_expected_genres_and_no_duplicate_songs():
    songs = load_songs(str(CATALOG_PATH))
    assert {s["genre"] for s in songs} == set(GENRES.values())
    keys = [(s["title"].lower(), s["artist"].lower()) for s in songs]
    assert len(keys) == len(set(keys))


def test_latin_exclusions_are_applied_only_to_latin():
    excluded = load_exclusions(str(LATIN_EXCLUSIONS_PATH))
    assert excluded  # the list is not empty
    for s in load_songs(str(CATALOG_PATH)):
        if s["genre"] == "latin":
            assert s["artist"].lower() not in excluded


def test_load_exclusions_ignores_comments_and_blanks(tmp_path):
    f = tmp_path / "ex.txt"
    f.write_text("# comment\n\nSome Artist\n  Other One  \n", encoding="utf-8")
    assert load_exclusions(str(f)) == {"some artist", "other one"}
    assert load_exclusions(str(tmp_path / "missing.txt")) == set()
    assert load_exclusions(None) == set()


def test_catalog_moods_match_derivation():
    for s in load_songs(str(CATALOG_PATH)):
        assert s["mood"] == derive_mood(s["energy"], s["valence"])


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Beyoncé", True),
        ("Tego Calderón", True),
        ("Sigur Rós", True),
        ("Don't Stop (feat. Artist) - Radio Edit 2019", True),
        ("21 Savage", True),
        ("ONE OK ROCK", True),
        ("Дима Билан", False),
        ("방탄소년단", False),
        ("宇多田ヒカル", False),
        ("أم كلثوم", False),
        ("Mixed Дима", False),
        ("", True),
    ],
)
def test_is_latin_script(text, expected):
    assert is_latin_script(text) is expected
