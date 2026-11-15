"""Run the evaluation suite and save a JSON report.

    python -m eval.run_eval --name baseline            # offline checks + live Gemini parser checks
    python -m eval.run_eval --name baseline --no-llm   # offline checks only (no API calls)

Live parser checks make about 30 Gemini calls, spaced by --delay seconds. Reports are written to
eval/results/<name>.json so runs can be compared before and after a change.
"""

import argparse
import json
from datetime import date
from pathlib import Path

from eval import metrics, profiles
from src import config
from src.recommender import load_songs

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CONSISTENCY_RUNS = 3


def run(use_llm: bool, delay: float) -> dict:
    songs = load_songs(str(config.CATALOG_PATH))
    genres = sorted({s["genre"] for s in songs})
    moods = sorted({s["mood"] for s in songs})

    report = {
        "date": date.today().isoformat(),
        "model": config.get_model_name() if use_llm else None,
        "catalog_size": len(songs),
        "guardrail": metrics.guardrail_report(profiles.ALL_VALID_PROFILES, profiles.MALFORMED_PROFILES),
        "ranking": metrics.ranking_metrics(songs, metrics.sweep_profiles(genres, moods)),
    }

    if use_llm:
        if not config.llm_available():
            raise SystemExit("GEMINI_API_KEY is not set; use --no-llm for offline checks only.")
        cases = profiles.NAMED_GENRE_CASES + profiles.NO_GENRE_CASES
        outcomes = metrics.run_parser([c["text"] for c in cases], genres, moods, delay=delay)
        report["parser"] = metrics.summarize_parser(cases, outcomes)

        off = metrics.run_parser(profiles.OFF_TOPIC_TEXTS, genres, moods, delay=delay)
        report["off_topic"] = metrics.off_topic_rejection(off)

        runs = [metrics.run_parser(profiles.CONSISTENCY_TEXTS, genres, moods, delay=delay)
                for _ in range(CONSISTENCY_RUNS)]
        report["consistency"] = metrics.consistency(profiles.CONSISTENCY_TEXTS, runs)
    return report


def print_report(report: dict) -> None:
    g, r = report["guardrail"], report["ranking"]
    print(f"\n=== Evaluation ({report['date']}, catalog {report['catalog_size']} songs) ===")
    print("\nGuardrail")
    print(f"  malformed profiles rejected : {g['malformed_rejected_rate']}  (wrongly accepted: {g['wrongly_accepted']})")
    print(f"  valid profiles accepted     : {g['valid_accepted_rate']}  (wrongly rejected: {g['wrongly_rejected']})")
    print(f"\nRanking over {r['profiles']} profiles, top-{r['k']}")
    print(f"  catalog coverage            : {r['coverage']}  ({r['never_recommended']} songs never recommended)")
    print(f"  top-10 songs' share of slots: {r['top10_songs_share_of_slots']}")
    print(f"  most-recommended artist     : {r['max_artist_share_of_slots']} of slots")
    print(f"  genre match / mood match    : {r['genre_match_rate']} / {r['mood_match_rate']}")
    if "parser" in report:
        p, o, c = report["parser"], report["off_topic"], report["consistency"]
        print(f"\nParser ({report['model']}, {p['cases']} labeled requests)")
        for key in ("parse_success_rate", "named_genre_accuracy", "genre_invented_rate",
                    "genre_indefensible_rate", "mood_accuracy", "numeric_in_range_rate",
                    "mood_numbers_coherence", "overall_case_pass_rate", "overall_case_pass_rate_strict"):
            print(f"  {key:<30}: {p[key]}")
        print(f"  {'no-genre tiers':<30}: {p['no_genre_tiers']}")
        print(f"  off-topic rejected         : {o['rejection_rate']}  (accepted: {o['accepted']})")
        print(f"  identical across {c['runs']} runs   : {c['exact_identical_rate']} exact, "
              f"{c['within_tolerance_rate']} within {c['tolerance']}  (unstable: {c['unstable']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="latest", help="report name, saved as eval/results/<name>.json")
    parser.add_argument("--no-llm", action="store_true", help="skip live Gemini checks")
    parser.add_argument("--delay", type=float, default=2.0, help="seconds between live API calls")
    args = parser.parse_args()

    report = run(use_llm=not args.no_llm, delay=args.delay)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{args.name}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_report(report)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
