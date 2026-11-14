"""TF-IDF retrieval over the knowledge notes (the "R" in RAG)."""

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Keep "&" inside tokens so "r&b" is one term; the default pattern would drop it.
TOKEN_PATTERN = r"[a-z0-9&]+"
# A note's name is repeated in its indexed text so a label match outweighs incidental mentions.
NAME_BOOST = 3
# A query that is exactly a note's name always ranks that note first. Needed because labels
# nest ("electro house" inside "progressive electro house") and TF-IDF cannot separate them.
EXACT_NAME_BONUS = 1.0


@dataclass(frozen=True)
class Note:
    id: str
    kind: str
    name: str
    text: str


def load_notes(path) -> List[Note]:
    with open(path, encoding="utf-8") as f:
        return [Note(**item) for item in json.load(f)]


class Retriever:
    def __init__(self, notes: Sequence[Note]):
        if not notes:
            raise ValueError("Retriever needs at least one note")
        self.notes = list(notes)
        docs = [f"{(n.name + ' ') * NAME_BOOST}{n.text}" for n in self.notes]
        self._vectorizer = TfidfVectorizer(
            lowercase=True, token_pattern=TOKEN_PATTERN, ngram_range=(1, 2), sublinear_tf=True
        )
        self._matrix = self._vectorizer.fit_transform(docs)

    def retrieve(
        self, query: str, n: int = 3, kinds: Optional[Iterable[str]] = None
    ) -> List[Tuple[Note, float]]:
        """Return up to n (note, score) pairs, best first. Notes with score 0 are never returned.

        Score is TF-IDF cosine similarity (0 to 1), plus EXACT_NAME_BONUS when the query equals
        a note's name.
        """
        if not query.strip():
            return []
        allowed = set(kinds) if kinds is not None else None
        scores = linear_kernel(self._vectorizer.transform([query]), self._matrix).ravel()
        wanted = " ".join(query.lower().split())
        for i, note in enumerate(self.notes):
            if note.name.lower() == wanted:
                scores[i] += EXACT_NAME_BONUS
        # Stable order: higher score first, ties keep the notes-file order.
        order = sorted(range(len(self.notes)), key=lambda i: (-scores[i], i))
        results = []
        for i in order:
            if scores[i] <= 0:
                break
            if allowed is not None and self.notes[i].kind not in allowed:
                continue
            results.append((self.notes[i], float(scores[i])))
            if len(results) == n:
                break
        return results

    def get(self, kind: str) -> List[Note]:
        """All notes of one kind, in file order (used for the fixed feature definitions)."""
        return [n for n in self.notes if n.kind == kind]

    def notes_for_song(self, song: Dict) -> List[Note]:
        """Retrieve the genre, subgenre and mood notes that describe a song.

        One TF-IDF lookup per note kind, so a short label like "edm" cannot be outranked by
        a longer note that merely mentions it (e.g. the "pop edm" subgenre). Kinds with no
        label on the song, or no match, are skipped.
        """
        notes = []
        for kind in ("genre", "subgenre", "mood"):
            label = song.get(kind)
            if not label:
                continue
            hits = self.retrieve(label, n=1, kinds=(kind,))
            if hits:
                notes.append(hits[0][0])
        return notes
