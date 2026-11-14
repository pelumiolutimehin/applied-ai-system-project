import pytest

from src.profile import ProfileError, UserProfile, normalize_label, validate_profile
from src.recommender import load_songs, recommend_songs, score_song


def make_song(**overrides):
    song = {
        "id": 1, "title": "Test Track", "artist": "Test Artist",
        "genre": "pop", "mood": "happy",
        "energy": 0.8, "tempo_bpm": 120, "valence": 0.9,
        "danceability": 0.8, "acousticness": 0.2,
    }
    song.update(overrides)
    return song


POP_HAPPY = {
    "favorite_genre": "pop", "favorite_mood": "happy",
    "target_energy": 0.8, "target_valence": 0.9, "target_acousticness": 0.2,
}


# ---------- scorer ----------

def test_perfect_match_scores_one():
    score, reasons = score_song(POP_HAPPY, make_song())
    assert score == pytest.approx(1.0)
    assert len(reasons) == 5
    assert "mood match (+0.25)" in reasons
    assert "genre match (+0.15)" in reasons


def test_mismatch_lowers_score_and_is_labeled():
    score, reasons = score_song(POP_HAPPY, make_song(genre="metal", mood="angry"))
    assert score == pytest.approx(0.60)
    assert "mood mismatch (+0.00)" in reasons
    assert "genre mismatch (+0.00)" in reasons


def test_numeric_closeness_is_symmetric():
    above = score_song(POP_HAPPY, make_song(energy=0.9))[0]
    below = score_song(POP_HAPPY, make_song(energy=0.7))[0]
    assert above == pytest.approx(below)


# ---------- ranking ----------

def test_recommend_sorted_best_first_and_respects_k():
    songs = [
        make_song(id=1, genre="metal", mood="angry", energy=0.1),
        make_song(id=2),
        make_song(id=3, energy=0.7),
    ]
    results = recommend_songs(POP_HAPPY, songs, k=2)
    assert [r[0]["id"] for r in results] == [2, 3]
    assert results[0][1] >= results[1][1]


def test_k_larger_than_catalog_returns_all():
    assert len(recommend_songs(POP_HAPPY, [make_song()], k=5)) == 1


def test_mood_match_outranks_genre_match_when_numbers_equal():
    # Mood weight (0.25) is larger than genre weight (0.15).
    prefs = dict(POP_HAPPY, target_energy=0.5, target_valence=0.5, target_acousticness=0.5)
    mood_only = make_song(id=1, genre="rock", mood="happy", energy=0.5, valence=0.5, acousticness=0.5)
    genre_only = make_song(id=2, genre="pop", mood="sad", energy=0.5, valence=0.5, acousticness=0.5)
    results = recommend_songs(prefs, [genre_only, mood_only], k=2)
    assert results[0][0]["id"] == 1


def test_equal_scores_keep_catalog_order():
    a = make_song(id=1)
    b = make_song(id=2)
    results = recommend_songs(POP_HAPPY, [a, b], k=2)
    assert [r[0]["id"] for r in results] == [1, 2]


def test_accepts_userprofile_object():
    profile = validate_profile(POP_HAPPY)
    assert isinstance(profile, UserProfile)
    assert recommend_songs(profile, [make_song()], k=1)[0][1] == pytest.approx(1.0)


def test_unnormalized_labels_still_match():
    prefs = dict(POP_HAPPY, favorite_genre="  POP ", favorite_mood="Happy")
    assert recommend_songs(prefs, [make_song()], k=1)[0][1] == pytest.approx(1.0)


# ---------- normalization and validation ----------

def test_normalize_label_variants():
    assert normalize_label("Lo-Fi ") == "lofi"
    assert normalize_label("Hip-Hop") == "hip hop"
    assert normalize_label("  Indie   Pop ") == "indie pop"


def test_normalize_does_not_merge_real_moods():
    assert normalize_label("relaxed") == "relaxed"
    assert normalize_label("chill") == "chill"


def test_validate_rejects_out_of_range():
    with pytest.raises(ProfileError):
        validate_profile(dict(POP_HAPPY, target_energy=3.0))
    with pytest.raises(ProfileError):
        validate_profile(dict(POP_HAPPY, target_valence=-1.0))


def test_validate_rejects_missing_key():
    bad = {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8}
    with pytest.raises(ProfileError, match="missing"):
        validate_profile(bad)


def test_validate_rejects_string_number_bool_nan():
    with pytest.raises(ProfileError):
        validate_profile(dict(POP_HAPPY, target_energy="0.8"))
    with pytest.raises(ProfileError):
        validate_profile(dict(POP_HAPPY, target_energy=True))
    with pytest.raises(ProfileError):
        validate_profile(dict(POP_HAPPY, target_energy=float("nan")))


def test_validate_rejects_bad_labels_and_non_dict():
    with pytest.raises(ProfileError):
        validate_profile(dict(POP_HAPPY, favorite_genre=""))
    with pytest.raises(ProfileError):
        validate_profile(dict(POP_HAPPY, favorite_mood=None))
    with pytest.raises(ProfileError):
        validate_profile(None)


def test_validate_accepts_boundaries_and_ints():
    profile = validate_profile(dict(POP_HAPPY, target_energy=0, target_valence=1))
    assert profile.target_energy == 0.0 and profile.target_valence == 1.0


def test_recommend_songs_rejects_bad_dict():
    with pytest.raises(ProfileError):
        recommend_songs({"favorite_genre": "pop"}, [make_song()])


# ---------- data loading ----------

def test_load_songs_from_catalog():
    songs = load_songs("data/songs.csv")
    assert len(songs) > 0
    for s in songs:
        assert 0.0 <= s["energy"] <= 1.0
        assert 0.0 <= s["valence"] <= 1.0
        assert 0.0 <= s["acousticness"] <= 1.0
        assert s["genre"] == s["genre"].lower()
