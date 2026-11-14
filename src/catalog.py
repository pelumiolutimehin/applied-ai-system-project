"""Build the app's song catalog from the "30000 Spotify Songs" dataset (TidyTuesday 2020-01-21).

Source: https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-01-21
Songs come from curated Spotify playlists grouped into 6 genres (pop, rap, rock, r&b,
edm, latin). Latin is kept because it is a major genre in the US market.
"""

import os
from typing import Optional, Set

import pandas as pd

# Source playlist_genre -> catalog genre label.
GENRES = {
    "pop": "pop",
    "rap": "rap",
    "rock": "rock",
    "r&b": "r&b",
    "edm": "edm",
    "latin": "latin",
}

OUTPUT_COLUMNS = [
    "id", "title", "artist", "genre", "subgenre", "mood",
    "energy", "tempo_bpm", "valence", "danceability", "acousticness",
]

ENERGY_HIGH = 0.60
VALENCE_HIGH = 0.50
VALENCE_SAD = 0.30


def derive_mood(energy: float, valence: float) -> str:
    """Heuristic mood from energy and valence (documented in the model card).

    high energy + positive  -> happy
    high energy + negative  -> intense
    low energy  + positive  -> chill
    low energy  + very low valence -> sad
    low energy  + low valence      -> melancholic
    """
    if energy >= ENERGY_HIGH:
        return "happy" if valence >= VALENCE_HIGH else "intense"
    if valence >= VALENCE_HIGH:
        return "chill"
    return "sad" if valence < VALENCE_SAD else "melancholic"


def load_exclusions(path: Optional[str]) -> Set[str]:
    """Read one artist name per line (lowercased); blank lines and '#' comments are ignored."""
    if not path or not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        lines = (line.strip() for line in f)
        return {line.lower() for line in lines if line and not line.startswith("#")}


def build_catalog(
    source_csv: str,
    per_subgenre: int = 10,
    min_popularity: int = 30,
    max_per_artist: int = 2,
    seed: int = 42,
    latin_exclusions_path: Optional[str] = "data/latin_exclusions.txt",
) -> pd.DataFrame:
    """Sample a balanced, de-duplicated catalog and map it to the app schema.

    Samples the same number of songs from every subgenre so each genre is balanced and
    varied. Excludes non-ASCII titles/artists, low-popularity tracks and remixes of the
    same song (one copy per track id and per title+artist).
    """
    df = pd.read_csv(source_csv)

    df = df.dropna(subset=["track_name", "track_artist"])
    df = df[df["playlist_genre"].isin(GENRES)]
    df = df[df["track_popularity"] >= min_popularity]
    df = df[df["track_name"].map(str.isascii) & df["track_artist"].map(str.isascii)]

    df["title"] = df["track_name"].str.strip()
    df["artist"] = df["track_artist"].str.strip()

    # Some latin-playlist entries are not latin music; drop listed artists from that genre only.
    excluded = load_exclusions(latin_exclusions_path)
    is_excluded = (df["playlist_genre"] == "latin") & df["artist"].str.lower().isin(excluded)
    df = df[~is_excluded]

    # A track can be in several playlists; keep one copy per track and per title+artist.
    df = df.drop_duplicates(subset="track_id")
    df = df.drop_duplicates(subset=["title", "artist"])

    parts = []
    for (genre, subgenre), pool in df.groupby(["playlist_genre", "playlist_subgenre"]):
        pool = pool.sample(frac=1, random_state=seed)
        pool = pool.groupby("artist").head(max_per_artist)
        if len(pool) < per_subgenre:
            raise ValueError(f"only {len(pool)} usable songs for {genre}/{subgenre}")
        chosen = pool.head(per_subgenre).copy()
        chosen["genre"] = GENRES[genre]
        chosen["subgenre"] = subgenre
        parts.append(chosen)

    out = pd.concat(parts, ignore_index=True)
    out["mood"] = [derive_mood(e, v) for e, v in zip(out["energy"], out["valence"])]
    out["tempo_bpm"] = out["tempo"].round().astype(int)
    out["id"] = range(1, len(out) + 1)

    for col in ("energy", "valence", "danceability", "acousticness"):
        out[col] = out[col].round(3)

    return out[OUTPUT_COLUMNS]
