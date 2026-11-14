import csv
from typing import Dict, List, Tuple, Union

from src.profile import UserProfile, normalize_label, validate_profile

WEIGHTS = {
    "energy": 0.30,
    "mood": 0.25,
    "valence": 0.20,
    "genre": 0.15,
    "acousticness": 0.10,
}


def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from a CSV file into a list of dicts with numeric fields converted."""
    songs = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["id"] = int(row["id"])
            row["tempo_bpm"] = int(float(row["tempo_bpm"]))
            row["genre"] = normalize_label(row["genre"])
            row["mood"] = normalize_label(row["mood"])

            for key in ("energy", "valence", "danceability", "acousticness"):
                row[key] = float(row[key])

            songs.append(row)

    return songs


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Return a song's weighted match score against validated preferences and the reasons behind it."""
    scores = {
        "energy": 1 - abs(song["energy"] - user_prefs["target_energy"]),
        "mood": 1.0 if song["mood"] == user_prefs["favorite_mood"] else 0.0,
        "valence": 1 - abs(song["valence"] - user_prefs["target_valence"]),
        "genre": 1.0 if song["genre"] == user_prefs["favorite_genre"] else 0.0,
        "acousticness": 1 - abs(song["acousticness"] - user_prefs["target_acousticness"]),
    }

    total = 0.0
    reasons = []

    for feature, weight in WEIGHTS.items():
        contribution = weight * scores[feature]
        total += contribution

        if feature in ("mood", "genre"):
            label = f"{feature} match" if scores[feature] == 1.0 else f"{feature} mismatch"
        else:
            label = f"{feature} {'close' if scores[feature] >= 0.8 else 'far'}"
        reasons.append(f"{label} (+{contribution:.2f})")

    return total, reasons


def recommend_songs(
    user: Union[UserProfile, Dict], songs: List[Dict], k: int = 5
) -> List[Tuple[Dict, float, str]]:
    """Return the top-k songs as (song, score, explanation) tuples, ranked by score.

    Accepts a UserProfile or a raw dict. Dicts are validated and normalized first,
    so bad input raises ProfileError instead of a KeyError or a silent wrong answer.
    """
    prefs = user.to_dict() if isinstance(user, UserProfile) else validate_profile(user).to_dict()

    scored = []
    for song in songs:
        score, reasons = score_song(prefs, song)
        scored.append((song, score, ", ".join(reasons)))

    # Highest score first; ties go to a mood match, then stay in CSV order (stable sort).
    ranked = sorted(
        scored,
        key=lambda item: (item[1], item[0]["mood"] == prefs["favorite_mood"]),
        reverse=True,
    )

    return ranked[:k]
