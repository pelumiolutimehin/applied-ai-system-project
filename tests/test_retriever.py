import pytest

from src.catalog import ENERGY_HIGH, VALENCE_HIGH, VALENCE_SAD
from src.config import CATALOG_PATH, KNOWLEDGE_PATH
from src.recommender import load_songs
from src.retriever import Note, Retriever, load_notes


@pytest.fixture(scope="module")
def notes():
    return load_notes(KNOWLEDGE_PATH)


@pytest.fixture(scope="module")
def retriever(notes):
    return Retriever(notes)


@pytest.fixture(scope="module")
def songs():
    return load_songs(str(CATALOG_PATH))


# ---------- the notes file itself ----------

def test_notes_are_well_formed(notes):
    ids = [n.id for n in notes]
    assert len(ids) == len(set(ids))
    for n in notes:
        assert n.kind in {"genre", "subgenre", "mood", "feature"}
        assert n.id == f"{n.kind}:{n.name}"
        assert n.text.strip()


def test_every_catalog_label_has_a_note(notes, songs):
    ids = {n.id for n in notes}
    for s in songs:
        assert f"genre:{s['genre']}" in ids
        assert f"subgenre:{s['subgenre']}" in ids
        assert f"mood:{s['mood']}" in ids


def test_mood_notes_quote_the_catalog_thresholds(notes):
    """If derive_mood's thresholds change, these notes would silently become wrong."""
    mood_text = " ".join(n.text for n in notes if n.kind == "mood")
    for threshold in (ENERGY_HIGH, VALENCE_HIGH, VALENCE_SAD):
        assert f"{threshold:.2f}" in mood_text


# ---------- retrieval quality ----------

def test_notes_for_song_returns_exactly_its_genre_subgenre_mood(retriever, songs):
    for s in songs:
        got = [n.id for n in retriever.notes_for_song(s)]
        assert got == [f"genre:{s['genre']}", f"subgenre:{s['subgenre']}", f"mood:{s['mood']}"], s


def test_ampersand_label_is_searchable(retriever):
    hits = retriever.retrieve("r&b", n=1, kinds=("genre",))
    assert hits[0][0].id == "genre:r&b"


@pytest.mark.parametrize(
    "label, expected",
    [
        ("electro house", "subgenre:electro house"),  # nested inside "progressive electro house"
        ("hip hop", "subgenre:hip hop"),              # nested inside "latin hip hop"
        ("pop", "genre:pop"),
        ("edm", "genre:edm"),                          # also a word inside "pop edm"
    ],
)
def test_nested_labels_resolve_to_the_exact_note(retriever, label, expected):
    kind = expected.split(":")[0]
    assert retriever.retrieve(label, n=1, kinds=(kind,))[0][0].id == expected


@pytest.mark.parametrize(
    "query, expected_top",
    [
        ("sad slow songs", "mood:sad"),
        ("loud distorted guitars", "subgenre:hard rock"),
        ("club festival drops", "subgenre:big room"),
    ],
)
def test_free_text_queries_find_relevant_notes(retriever, query, expected_top):
    assert retriever.retrieve(query, n=1)[0][0].id == expected_top


# ---------- behavior ----------

def test_empty_and_unmatched_queries_return_nothing(retriever):
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []
    assert retriever.retrieve("zzzzqqqq") == []


def test_kinds_filter_and_n_limit(retriever):
    hits = retriever.retrieve("rock", n=2, kinds=("genre", "subgenre"))
    assert len(hits) <= 2
    assert all(n.kind in ("genre", "subgenre") for n, _ in hits)


def test_scores_are_sorted_descending(retriever):
    scores = [s for _, s in retriever.retrieve("energetic dance music", n=5)]
    assert scores == sorted(scores, reverse=True)


def test_get_returns_feature_definitions(retriever):
    assert {n.name for n in retriever.get("feature")} == {"energy", "valence", "acousticness"}


def test_empty_notes_rejected():
    with pytest.raises(ValueError):
        Retriever([])


def test_song_without_subgenre_still_gets_other_notes(retriever):
    got = [n.id for n in retriever.notes_for_song({"genre": "rock", "mood": "happy"})]
    assert got == ["genre:rock", "mood:happy"]


def test_note_is_immutable():
    note = Note("genre:x", "genre", "x", "text")
    with pytest.raises(Exception):
        note.text = "changed"
