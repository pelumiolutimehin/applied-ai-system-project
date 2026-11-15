"""Reliability layer: is an AI explanation faithful to the evidence, and how confident are we?

Faithfulness is a rule-based check, not another LLM call. An explanation is rejected if it says
something the evidence does not support:
  - a genre, subgenre or mood the song does not have (and the notes do not mention),
  - that the song matches the listener's genre or mood when it does not,
  - that the song is acoustic, loud or calm when its measured values say otherwise,
  - a number that is not in the retrieved notes (the model was told not to quote scores).
Phrases like "not rock" or "unlike rap" are allowed. The check is a heuristic: it catches clear
contradictions and invented labels, not every possible mistake.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set

from src.catalog import derive_mood
from src.profile import UserProfile
from src.retriever import Note

ACOUSTIC_MIN = 0.5
HIGH_ENERGY_MIN = 0.6

HIGH_ENERGY_WORDS = ("high-energy", "high energy", "energetic", "loud")
LOW_ENERGY_WORDS = ("calm", "quiet", "slow", "low-energy", "low energy", "gentle")
NEGATION_WORDS = {
    "not", "no", "unlike", "without", "rather", "instead", "less", "non", "nor", "neither",
    "though", "although", "despite", "but", "however",
}
NEGATION_WINDOW = 5  # how many words before a term count as its context


@dataclass(frozen=True)
class Vocabulary:
    """The catalog's own labels, so the checker knows which words are genres or moods."""
    genres: frozenset
    subgenres: frozenset
    moods: frozenset

    @classmethod
    def from_notes(cls, notes: Iterable[Note]) -> "Vocabulary":
        def names(kind):
            return frozenset(n.name.lower() for n in notes if n.kind == kind)
        notes = list(notes)
        return cls(names("genre"), names("subgenre"), names("mood"))


@dataclass
class FaithfulnessResult:
    passed: bool
    problems: List[str] = field(default_factory=list)


def _word_pattern(term: str) -> "re.Pattern":
    # "&" counts as part of a word so "r&b" is one term and "b" alone never matches inside it.
    return re.compile(rf"(?<![\w&]){re.escape(term)}(?![\w&])", re.IGNORECASE)


def _is_negated(text: str, start: int) -> bool:
    """True if one of the few words before position `start`, in the same sentence, is a negation cue."""
    sentence = re.split(r"[.!?;]", text[:start])[-1]
    words = re.findall(r"[\w'&-]+", sentence.lower())[-NEGATION_WINDOW:]
    return any(w in NEGATION_WORDS or w.endswith("n't") for w in words)


def _find_mentions(text: str, terms: Iterable[str]) -> List[tuple]:
    """(term, start) for each term found, longest terms first, masking matches so that a short
    term inside a longer one ("pop" in "hip pop", "house" in "electro house") is not counted."""
    working = text
    found = []
    for term in sorted(set(terms), key=len, reverse=True):
        for m in _word_pattern(term).finditer(working):
            found.append((term, m.start()))
            working = working[: m.start()] + " " * (m.end() - m.start()) + working[m.end():]
    return found


def check_faithfulness(
    text: str,
    song: Dict,
    profile: UserProfile,
    notes: Sequence[Note],
    vocab: Vocabulary,
) -> FaithfulnessResult:
    """Check one explanation against the song, the listener profile and the retrieved notes."""
    problems: List[str] = []
    notes_text = " ".join(f"{n.name} {n.text}" for n in notes).lower()
    song_genre = (song.get("genre") or "").lower()
    song_subgenre = (song.get("subgenre") or "").lower()
    song_mood = (song.get("mood") or "").lower()
    listener_genre = (profile.favorite_genre or "").lower()
    evidence_labels = {x for x in (song_genre, song_subgenre) if x}

    def in_notes(term: str) -> bool:
        return bool(_word_pattern(term).search(notes_text))

    def inside_evidence_label(term: str) -> bool:
        return any(_word_pattern(term).search(ev) for ev in evidence_labels)

    # 1. Genre and subgenre mentions.
    # The listener's own genre is always checked, even if it is not a catalog label (e.g. "k-pop").
    genre_terms = vocab.genres | vocab.subgenres | ({listener_genre} if listener_genre else set())
    for term, start in _find_mentions(text, genre_terms):
        negated = _is_negated(text, start)
        if term == listener_genre and term != song_genre:
            if not negated:
                problems.append(f"claims the listener's genre '{term}' but the song is '{song_genre}'")
            continue
        supported = term in evidence_labels or inside_evidence_label(term) or in_notes(term)
        if not supported and not negated:
            problems.append(f"mentions '{term}', which is not this song's genre or in its notes")

    # 2. Mood mentions.
    for term, start in _find_mentions(text, vocab.moods):
        if term != song_mood and not in_notes(term) and not _is_negated(text, start):
            problems.append(f"mentions mood '{term}' but the song's mood is '{song_mood}'")

    # 3. Feature claims against the song's measured values.
    def claimed(words):
        return [(w, s) for w, s in _find_mentions(text, words) if not _is_negated(text, s)]

    if claimed(("acoustic",)) and song.get("acousticness", 0) < ACOUSTIC_MIN:
        problems.append("calls the song acoustic but its acousticness is low")
    if claimed(HIGH_ENERGY_WORDS) and song.get("energy", 1) < HIGH_ENERGY_MIN:
        problems.append("calls the song high-energy but its energy is low")
    if claimed(LOW_ENERGY_WORDS) and song.get("energy", 0) >= HIGH_ENERGY_MIN:
        problems.append("calls the song calm or slow but its energy is high")

    # 4. Numbers must come from the notes (the model was told not to quote scores).
    for number in re.findall(r"\d+(?:\.\d+)?", text):
        if number not in notes_text:
            problems.append(f"quotes the number {number}, which is not in the notes")

    return FaithfulnessResult(passed=not problems, problems=problems)


# ---------- confidence ----------

CONFIDENCE_WEIGHTS = {
    "strength": 0.35,       # how good the best match is
    "mood_support": 0.15,   # share of results that match the requested mood
    "genre_support": 0.15,  # share of results that match the requested genre (if one was given)
    "coherence": 0.15,      # the requested mood agrees with the requested numbers
    "known_labels": 0.10,   # the requested genre and mood exist in the catalog
    "faithful": 0.10,       # share of AI explanations that passed the faithfulness check
}
HIGH_CONFIDENCE = 0.80
MEDIUM_CONFIDENCE = 0.60
UNKNOWN_LABEL_CAP = 0.40


@dataclass
class ConfidenceReport:
    score: float
    level: str  # "high", "medium" or "low"
    factors: Dict[str, Optional[float]]  # None means not applicable for this request
    notes: List[str] = field(default_factory=list)


def assess_confidence(
    profile: UserProfile,
    recommendations: Sequence[tuple],
    catalog_genres: Set[str],
    catalog_moods: Set[str],
    explanations: Optional[Sequence] = None,
) -> ConfidenceReport:
    """Combine simple, inspectable signals into one 0..1 score. Not a calibrated probability.

    Factors that do not apply (no genre requested, no AI explanations) are left out and the
    remaining weights are rescaled. Unknown genre or mood labels cap the score, because the
    recommender cannot honor a preference it has no songs for.
    """
    if not recommendations:
        return ConfidenceReport(0.0, "low", {}, ["no results"])

    songs = [rec[0] for rec in recommendations]
    k = len(songs)
    genre = profile.favorite_genre

    factors: Dict[str, Optional[float]] = {
        "strength": recommendations[0][1],
        "mood_support": sum(s["mood"] == profile.favorite_mood for s in songs) / k,
        "genre_support": None if genre is None else sum(s["genre"] == genre for s in songs) / k,
        "coherence": 1.0 if profile.favorite_mood == derive_mood(profile.target_energy, profile.target_valence) else 0.0,
        "known_labels": 1.0 if (genre is None or genre in catalog_genres) and profile.favorite_mood in catalog_moods else 0.0,
        "faithful": None,
    }
    checked = [e for e in (explanations or []) if getattr(e, "faithful", None) is not None]
    if checked:
        factors["faithful"] = sum(1 for e in checked if e.faithful) / len(checked)

    applicable = {name: v for name, v in factors.items() if v is not None}
    total_weight = sum(CONFIDENCE_WEIGHTS[name] for name in applicable)
    score = sum(CONFIDENCE_WEIGHTS[name] * v for name, v in applicable.items()) / total_weight

    notes = []
    if factors["known_labels"] == 0.0:
        score = min(score, UNKNOWN_LABEL_CAP)
        notes.append("the requested genre or mood is not in the catalog")
    if factors["coherence"] == 0.0:
        notes.append("the requested mood disagrees with the requested energy and positivity")
    if factors["genre_support"] is not None and factors["genre_support"] == 0.0:
        notes.append("none of the results match the requested genre")
    if factors["mood_support"] == 0.0:
        notes.append("none of the results match the requested mood")
    if factors["faithful"] is not None and factors["faithful"] < 1.0:
        notes.append("some AI explanations were replaced because they were not supported by the evidence")

    score = round(score, 3)
    level = "high" if score >= HIGH_CONFIDENCE else "medium" if score >= MEDIUM_CONFIDENCE else "low"
    return ConfidenceReport(score, level, factors, notes)
