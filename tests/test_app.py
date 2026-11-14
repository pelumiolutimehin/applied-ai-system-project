from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def run_app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=30).run()


def test_app_renders_default_recommendations():
    at = run_app()
    assert not at.exception
    assert at.title[0].value.endswith("Music Recommender")
    # default k is 5: one bordered result block per song, each with a subheader
    assert len(at.subheader) == 5


def test_app_respects_result_count_slider():
    at = run_app()
    at.sidebar.slider(key="k").set_value(8).run()
    assert not at.exception
    assert len(at.subheader) == 8


def test_app_updates_when_preferences_change():
    at = run_app()
    first = [s.value for s in at.subheader]
    at.sidebar.selectbox(key="genre").select("rock")
    at.sidebar.selectbox(key="mood").select("intense")
    at.sidebar.slider(key="energy").set_value(0.95)
    at.run()
    assert not at.exception
    assert [s.value for s in at.subheader] != first
