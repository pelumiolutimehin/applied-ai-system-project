"""Grounded explanations for recommendations: retrieve notes, then ask Gemini to write from them.

One LLM call explains the whole result list (the free tier is rate-limited). The model sees only
the retrieved notes, fixed feature definitions and the scorer's own breakdown, and is told not to
invent anything else. If anything goes wrong, each song falls back to a deterministic sentence
built from the breakdown, so a failure here never breaks the results page.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from src import config, llm_parser
from src.llm_parser import Generate, extract_json
from src.profile import UserProfile
from src.reliability import Vocabulary, check_faithfulness
from src.retriever import Note, Retriever

MAX_EXPLANATION_CHARS = 400


@dataclass
class Explanation:
    text: str
    source: str  # "llm" or "fallback"
    notes_used: List[str] = field(default_factory=list)  # ids of the notes the model was shown
    # Result of the faithfulness check on the model's text: True = passed, False = rejected and
    # replaced by the fallback sentence, None = the model wrote nothing usable (check not run).
    faithful: Optional[bool] = None
    problems: List[str] = field(default_factory=list)  # why a rejected text was rejected
    rejected_text: Optional[str] = None  # what the model wrote, kept so a human can audit the rejection
    error: Optional[str] = None  # why no model text was used, e.g. "ServerError 503" (no message: it may hold secrets)


def matched_features(reasons: Sequence[str]) -> List[str]:
    """Names of features the scorer counted as a match or as close (e.g. ['mood', 'energy'])."""
    matched = []
    for reason in reasons:
        label = reason.split(" (")[0]
        feature, _, verdict = label.partition(" ")
        if verdict in ("match", "close"):
            matched.append(feature)
    return matched


def fallback_explanation(reasons: Sequence[str]) -> str:
    """Deterministic explanation from the score breakdown alone."""
    matched = matched_features(reasons)
    if not matched:
        return "This was the closest available overall match to your settings."
    return "Matched your settings on: " + ", ".join(matched) + "."


def build_system_prompt() -> str:
    return (
        "You write short explanations for a music recommender. For each song, write 1 or 2 "
        "sentences (at most 45 words) telling the listener why it was recommended.\n\n"
        "Rules:\n"
        "- Use ONLY facts from that song's NOTES, the DEFINITIONS, and its BREAKDOWN. Do not "
        "invent anything about the artist, lyrics, release date or popularity.\n"
        "- Credit a feature as a reason only if the BREAKDOWN says it is a match or close. If "
        "mood or genre says mismatch, do not claim it matched.\n"
        "- Prefer concrete details from the song's subgenre and genre notes (its sound or feel) "
        "over generic praise like \"great match\".\n"
        "- Do not mention scores or numbers.\n\n"
        'Reply with ONLY a JSON object: {"explanations": [{"rank": 1, "text": "..."}]} '
        "with one entry per song, using the rank numbers you were given."
    )


def build_user_message(
    profile: UserProfile,
    recommendations: Sequence[Tuple[Dict, float, str]],
    notes_per_song: Sequence[Sequence[Note]],
    definitions: Sequence[Note],
) -> str:
    lines = [
        "LISTENER PROFILE: "
        f"genre={profile.favorite_genre}, mood={profile.favorite_mood}, "
        f"energy={profile.target_energy:.2f}, positivity={profile.target_valence:.2f}, "
        f"acousticness={profile.target_acousticness:.2f}",
        "",
        "DEFINITIONS:",
    ]
    lines += [f"- {d.name}: {d.text}" for d in definitions]

    for rank, ((song, _score, reasons), notes) in enumerate(zip(recommendations, notes_per_song), 1):
        lines += [
            "",
            f"SONG {rank}: \"{song['title']}\" by {song['artist']}",
            f"  genre={song['genre']}, subgenre={song.get('subgenre', '')}, mood={song['mood']}",
            f"  BREAKDOWN: {reasons}",
            "  NOTES:",
        ]
        lines += [f"  - ({n.kind}) {n.name}: {n.text}" for n in notes]
    return "\n".join(lines)


def parse_explanations(raw: str, count: int) -> Dict[int, str]:
    """Return {rank: text} for the entries that are valid; invalid entries are left out."""
    data = extract_json(raw)
    items = data.get("explanations")
    if not isinstance(items, list):
        raise ValueError("missing 'explanations' list")
    texts: Dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        rank, text = item.get("rank"), item.get("text")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= count:
            continue
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_EXPLANATION_CHARS:
            continue
        texts.setdefault(rank, text.strip())
    return texts


def explain_recommendations(
    profile: UserProfile,
    recommendations: Sequence[Tuple[Dict, float, str]],
    retriever: Retriever,
    generate: Optional[Generate] = None,
    retries: int = 1,
    sleep: Callable[[float], None] = time.sleep,
) -> List[Explanation]:
    """Return one Explanation per recommendation, in order. Never raises.

    Each model-written text is checked for faithfulness; a text that fails is replaced by the
    deterministic fallback sentence and the reasons are kept in Explanation.problems.
    """
    reasons_per_song = [rec[2].split(", ") for rec in recommendations]
    notes_per_song = [retriever.notes_for_song(rec[0]) for rec in recommendations]
    ids_per_song = [[n.id for n in notes] for notes in notes_per_song]

    def fallback(i: int, faithful: Optional[bool] = None, problems: Optional[List[str]] = None,
                 rejected_text: Optional[str] = None, error: Optional[str] = None) -> Explanation:
        return Explanation(fallback_explanation(reasons_per_song[i]), "fallback", [], faithful,
                           problems or [], rejected_text, error)

    if not recommendations:
        return []

    if generate is None:
        if not config.llm_available():
            return [fallback(i) for i in range(len(recommendations))]
        generate = llm_parser.call_gemini  # looked up at call time so tests can patch it

    system_prompt = build_system_prompt()
    user_message = build_user_message(profile, recommendations, notes_per_song, retriever.get("feature"))

    texts: Dict[int, str] = {}
    failure: Optional[str] = None
    for attempt in range(retries + 1):
        try:
            texts = parse_explanations(generate(system_prompt, user_message), len(recommendations))
            failure = None
            break
        except Exception as err:  # API, network, quota or unreadable reply: retry once, then fall back
            code = getattr(err, "code", None)
            failure = f"{type(err).__name__} {code}" if code else type(err).__name__
            if attempt < retries:
                sleep(1.0)

    vocab = Vocabulary.from_notes(retriever.notes)
    results = []
    for i, (song, _score, _reasons) in enumerate(recommendations):
        if (i + 1) not in texts:
            results.append(fallback(i, error=failure or "no usable text for this song"))
            continue
        verdict = check_faithfulness(texts[i + 1], song, profile, notes_per_song[i], vocab)
        if verdict.passed:
            results.append(Explanation(texts[i + 1], "llm", ids_per_song[i], True))
        else:
            results.append(fallback(i, faithful=False, problems=verdict.problems, rejected_text=texts[i + 1]))
    return results
