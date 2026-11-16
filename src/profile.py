"""User taste profile, label normalization, and input validation (the guardrail)."""

import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional

NUMERIC_FIELDS = ("target_energy", "target_valence", "target_acousticness")
LABEL_FIELDS = ("favorite_genre", "favorite_mood")

# Formatting variants only. Do not map one real catalog label onto another
# (e.g. "relaxed" and "chill" are both distinct moods in the data).
LABEL_ALIASES = {
    "lo-fi": "lofi",
    "lo fi": "lofi",
    "hip-hop": "hip hop",
    "hiphop": "hip hop",
    "rnb": "r&b",
    "r and b": "r&b",
    "indie-pop": "indie pop",
}


class ProfileError(ValueError):
    """Raised when a user profile is missing fields or has invalid values."""


def normalize_label(label: str) -> str:
    """Lowercase, trim, collapse whitespace, and apply formatting aliases."""
    cleaned = " ".join(label.strip().lower().split())
    return LABEL_ALIASES.get(cleaned, cleaned)


@dataclass
class UserProfile:
    favorite_genre: Optional[str]  # None means no genre preference
    favorite_mood: str
    target_energy: float
    target_valence: float
    target_acousticness: float

    def to_dict(self) -> Dict:
        return asdict(self)


def validate_profile(raw: Dict) -> UserProfile:
    """Turn a raw dict into a UserProfile, or raise ProfileError.

    A genre of None means no preference (the key must still be present). Rejects: non-dict
    input, missing keys, non-string or empty labels (the mood is always required),
    non-numeric targets (including numbers given as strings, booleans, NaN),
    and targets outside 0..1. Labels are normalized.
    """
    if not isinstance(raw, dict):
        raise ProfileError("profile must be a dict")

    missing = [k for k in LABEL_FIELDS + NUMERIC_FIELDS if k not in raw]
    if missing:
        raise ProfileError(f"missing fields: {', '.join(missing)}")

    labels = {}
    for field in LABEL_FIELDS:
        value = raw[field]
        if field == "favorite_genre" and value is None:
            labels[field] = None  # explicit "no genre preference"
            continue
        if not isinstance(value, str) or not value.strip():
            raise ProfileError(f"{field} must be a non-empty string")
        labels[field] = normalize_label(value)

    numbers = {}
    for field in NUMERIC_FIELDS:
        value = raw[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProfileError(f"{field} must be a number, got {type(value).__name__}")
        if math.isnan(value) or not 0.0 <= value <= 1.0:
            raise ProfileError(f"{field} must be between 0 and 1, got {value}")
        numbers[field] = float(value)

    return UserProfile(**labels, **numbers)
