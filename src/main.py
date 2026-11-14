"""Command line runner: prints top recommendations for the fixed evaluation profiles.

Run from anywhere:  python -m src.main   (from the repo root)
The web app is the main interface (streamlit run app.py); this is a quick developer check.
"""

from eval.profiles import ALL_VALID_PROFILES, MALFORMED_PROFILES
from src.config import CATALOG_PATH
from src.profile import ProfileError
from src.recommender import load_songs, recommend_songs


def print_recommendations(recommendations) -> None:
    """Prints each (song, score, explanation) as a numbered block with one reason per line."""
    width = 50
    print()
    print("=" * width)
    print("TOP RECOMMENDATIONS".center(width))
    print("=" * width)

    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print()
        print(f"#{rank}  {song['title']} by {song['artist']}")
        print(f"    Score: {score:.2f} / 1.00")
        print("    Reasons:")
        for reason in explanation.split(", "):
            print(f"      - {reason}")

    print()
    print("=" * width)


def main() -> None:
    songs = load_songs(str(CATALOG_PATH))
    print(f"Loaded songs: {len(songs)}")

    for name, profile in {**ALL_VALID_PROFILES, **MALFORMED_PROFILES}.items():
        print(f"\n>>> Profile: {name}")
        try:
            print_recommendations(recommend_songs(profile, songs, k=5))
        except ProfileError as err:
            print(f"    Rejected by guardrail: {err}")


if __name__ == "__main__":
    main()
