"""Streamlit web UI for the music recommender.

Run from the repo root:
    streamlit run app.py
"""

from pathlib import Path

import streamlit as st

from src.profile import ProfileError, UserProfile, validate_profile
from src.recommender import load_songs, recommend_songs

CATALOG_PATH = Path(__file__).parent / "data" / "songs.csv"


@st.cache_data
def get_songs():
    return load_songs(str(CATALOG_PATH))


def sidebar_profile(songs) -> UserProfile:
    """Collect the taste profile from sidebar controls (always valid by construction)."""
    genres = sorted({s["genre"] for s in songs})
    moods = sorted({s["mood"] for s in songs})

    st.sidebar.header("Your taste")
    genre = st.sidebar.selectbox("Favorite genre", genres, key="genre")
    mood = st.sidebar.selectbox("Favorite mood", moods, key="mood")
    energy = st.sidebar.slider("Energy (calm to intense)", 0.0, 1.0, 0.5, 0.05, key="energy")
    valence = st.sidebar.slider("Positivity (dark to upbeat)", 0.0, 1.0, 0.5, 0.05, key="valence")
    acoustic = st.sidebar.slider("Acousticness (electronic to acoustic)", 0.0, 1.0, 0.5, 0.05, key="acoustic")

    # Same guardrail as every other entry point.
    return validate_profile({
        "favorite_genre": genre,
        "favorite_mood": mood,
        "target_energy": energy,
        "target_valence": valence,
        "target_acousticness": acoustic,
    })


def show_recommendations(recommendations) -> None:
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        with st.container(border=True):
            st.subheader(f"#{rank}  {song['title']}")
            st.write(f"by **{song['artist']}**")
            meta = [song["genre"], song["mood"]]
            if song.get("subgenre"):
                meta.insert(1, song["subgenre"])
            st.caption(" · ".join(meta))
            st.progress(min(max(score, 0.0), 1.0), text=f"Match score: {score:.2f} / 1.00")
            with st.expander("Why this song?"):
                for reason in explanation.split(", "):
                    st.write(f"- {reason}")


def main() -> None:
    st.set_page_config(page_title="Music Recommender", page_icon="🎵", layout="centered")
    st.title("🎵 Music Recommender")

    songs = get_songs()
    st.caption(f"Content-based matching over {len(songs)} songs.")

    profile = sidebar_profile(songs)
    k = st.sidebar.slider("Number of results", 3, 10, 5, key="k")

    try:
        recommendations = recommend_songs(profile, songs, k=k)
    except ProfileError as err:
        st.error(f"Invalid profile: {err}")
        return

    show_recommendations(recommendations)


main()
