"""Streamlit web UI for the music recommender.

Run from the repo root:
    streamlit run app.py
"""

import streamlit as st

from src import config
from src.config import CATALOG_PATH, KNOWLEDGE_PATH
from src.explainer import explain_recommendations
from src.llm_parser import MAX_REQUEST_CHARS, ParseError, parse_request
from src.profile import ProfileError, UserProfile, validate_profile
from src.recommender import load_songs, recommend_songs
from src.retriever import Retriever, load_notes


@st.cache_data
def get_songs():
    return load_songs(str(CATALOG_PATH))


@st.cache_resource
def get_retriever() -> Retriever:
    return Retriever(load_notes(KNOWLEDGE_PATH))


@st.cache_data(ttl=120, show_spinner="Writing explanations...")
def cached_explain(profile: tuple, k: int):
    """Explain the top-k results for a profile. Cached so reruns do not repeat the API call.

    The profile arrives as a tuple of its field values because cache keys must be hashable.
    """
    user = UserProfile(*profile)
    recommendations = recommend_songs(user, get_songs(), k=k)
    return explain_recommendations(user, recommendations, get_retriever())


@st.cache_data(ttl=120, show_spinner="Reading your request...")
def cached_parse(text: str, genres: tuple, moods: tuple):
    """Parse a free-text request. Returns (profile, error_message); exactly one is None.

    Failures are returned, not raised, so they are cached too. Streamlit reruns the script
    on every widget change, and without this a failing request would hit the API each time.
    """
    try:
        return parse_request(text, genres, moods), None
    except ParseError as err:
        return None, str(err)


def describe(profile: UserProfile) -> str:
    return (
        f"genre **{profile.favorite_genre}**, mood **{profile.favorite_mood}**, "
        f"energy {profile.target_energy:.2f}, positivity {profile.target_valence:.2f}, "
        f"acousticness {profile.target_acousticness:.2f}"
    )


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


def show_recommendations(recommendations, explanations=None) -> None:
    for rank, (song, score, breakdown) in enumerate(recommendations, start=1):
        with st.container(border=True):
            st.subheader(f"#{rank}  {song['title']}")
            st.write(f"by **{song['artist']}**")
            meta = [song["genre"], song["mood"]]
            if song.get("subgenre"):
                meta.insert(1, song["subgenre"])
            st.caption(" · ".join(meta))
            st.progress(min(max(score, 0.0), 1.0), text=f"Match score: {score:.2f} / 1.00")
            with st.expander("Why this song?"):
                ai = explanations[rank - 1] if explanations else None
                if ai is not None and ai.source == "llm":
                    st.write(ai.text)
                    st.caption("AI-written from these notes: " + ", ".join(ai.notes_used))
                st.markdown("**Score breakdown**")
                for reason in breakdown.split(", "):
                    st.write(f"- {reason}")


def main() -> None:
    st.set_page_config(page_title="Music Recommender", page_icon="🎵", layout="centered")
    st.title("🎵 Music Recommender")

    songs = get_songs()
    st.caption(f"Content-based matching over {len(songs)} songs.")

    slider_profile = sidebar_profile(songs)
    k = st.sidebar.slider("Number of results", 3, 10, 5, key="k")
    ai_available = config.llm_available()
    want_explanations = st.sidebar.checkbox(
        "AI explanations",
        value=ai_available,
        disabled=not ai_available,
        key="explain",
        help="Writes a short explanation per song from the knowledge notes. "
        "Makes one API call per change of results." if ai_available else "Needs a GEMINI_API_KEY.",
    )

    request = st.text_input(
        "Describe what you want to hear",
        placeholder="e.g. upbeat pop for a morning run",
        max_chars=MAX_REQUEST_CHARS,
        key="request",
        help="Optional. If left empty, or if it can't be understood, the sidebar settings are used.",
    )

    profile = slider_profile
    if request.strip():
        if not config.llm_available():
            st.info("Free-text requests need a GEMINI_API_KEY (see .env.example). Using the sidebar settings instead.")
        else:
            genres = tuple(sorted({s["genre"] for s in songs}))
            moods = tuple(sorted({s["mood"] for s in songs}))
            parsed, error = cached_parse(request.strip(), genres, moods)
            if parsed is not None:
                profile = parsed
                st.caption(f"Interpreted as: {describe(parsed)}")
            else:
                st.warning(f"{error} Using the sidebar settings instead.")

    try:
        recommendations = recommend_songs(profile, songs, k=k)
    except ProfileError as err:
        st.error(f"Invalid profile: {err}")
        return

    explanations = None
    if want_explanations:
        explanations = cached_explain(tuple(profile.to_dict().values()), k)

    show_recommendations(recommendations, explanations)


main()
