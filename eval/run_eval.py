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
from src.retriever import Retriever, load_notes

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
        "confidence": metrics.confidence_check(
            songs, {**profiles.ALL_VALID_PROFILES, **profiles.CONFLICTING_EXTRA}, profiles.EXPECTED_CONFIDENCE),
    }

    if use_llm:
        if not config.llm_available():
            raise SystemExit("GEMINI_API_KEY is not set; use --no-llm for offline checks only.")
        cases = profiles.NAMED_GENRE_CASES + profiles.NO_GENRE_CASES
        outcomes = metrics.run_parser([c["text"] for c in cases], genres, moods, delay=delay)
        report["parser"] = metrics.summarize_parser(cases, outcomes)
        parser_api_errors = metrics.count_api_errors(outcomes)

        off = metrics.run_parser(profiles.OFF_TOPIC_TEXTS, genres, moods, delay=delay)
        report["off_topic"] = metrics.off_topic_rejection(off)

        runs = [metrics.run_parser(profiles.CONSISTENCY_TEXTS, genres, moods, delay=delay)
                for _ in range(CONSISTENCY_RUNS)]
        report["consistency"] = metrics.consistency(profiles.CONSISTENCY_TEXTS, runs)
        report["api_errors"] = {
            "parser_requests": parser_api_errors,
            "off_topic_requests": metrics.count_api_errors(off),
            "consistency_requests": metrics.count_api_errors(*runs),
        }

        report["explanations"] = metrics.explanation_faithfulness(
            songs, {**profiles.ALL_VALID_PROFILES, **profiles.CONFLICTING_EXTRA},
            Retriever(load_notes(config.KNOWLEDGE_PATH)), delay=delay)
        report["checker_validation"] = metrics.checker_mutation_test(
            report["explanations"]["samples"], songs,
            {**profiles.ALL_VALID_PROFILES, **profiles.CONFLICTING_EXTRA},
            Retriever(load_notes(config.KNOWLEDGE_PATH)))
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
    conf = report["confidence"]
    print(f"\nConfidence over {conf['profiles']} profiles: level as expected {conf['as_expected_rate']}")
    for name, row in conf["per_profile"].items():
        flag = "" if row["as_expected"] in (None, True) else "   <-- unexpected"
        print(f"  {name:<18} {row['score']:.2f} {row['level']:<7} expected {row['expected']}{flag}")

    if "parser" not in report:
        return

    parsed, off, cons = report["parser"], report["off_topic"], report["consistency"]
    print(f"\nParser ({report['model']}, {parsed['cases']} labeled requests)")
    for key in ("parse_success_rate", "named_genre_accuracy", "genre_invented_rate",
                "genre_indefensible_rate", "mood_accuracy", "numeric_in_range_rate",
                "mood_numbers_coherence", "overall_case_pass_rate", "overall_case_pass_rate_strict"):
        print(f"  {key:<30}: {parsed[key]}")
    print(f"  {'no-genre tiers':<30}: {parsed['no_genre_tiers']}")
    print(f"  {'off-topic rejected':<30}: {off['rejection_rate']}  (accepted: {off['accepted']})")
    errs = report["api_errors"]
    if any(errs.values()) or report["explanations"]["no_usable_text_reasons"]:
        print(f"\n  !! API errors during this run: {errs}, explanation failures: "
              f"{report['explanations']['no_usable_text_reasons']}")
        print("  !! Affected numbers reflect the API, not the model. Re-run with a larger --delay.")
    print(f"  {'identical across runs':<30}: {cons['exact_identical_rate']} exact, "
          f"{cons['within_tolerance_rate']} within {cons['tolerance']}  (unstable: {cons['unstable']})")

    exp = report["explanations"]
    print(f"\nAI explanations ({exp['explanations']} written)")
    print(f"  passed faithfulness check  : {exp['passed_check']}")
    print(f"  rejected and replaced      : {exp['rejected']}  (rate {exp['rejection_rate']})")
    print(f"  no usable text from model  : {exp['no_usable_text']}  {exp['no_usable_text_reasons'] or ''}")
    for ex in exp["rejected_examples"][:6]:
        print(f"    - [{ex['profile']}] {ex['song']}: {ex['text']!r}\n        {ex['problems']}")

    val = report["checker_validation"]
    print(f"\nChecker validation (real explanations, deliberately corrupted)")
    print(f"  false alarms on {val['originals']} untouched originals: {val['false_alarm_rate']}")
    print(f"  corruptions caught overall : {val['overall_recall']}")
    for name, row in val["per_mutation"].items():
        print(f"    {name:<20} {row['caught']}/{row['applied']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="latest", help="report name, saved as eval/results/<name>.json")
    parser.add_argument("--no-llm", action="store_true", help="skip live Gemini checks")
    parser.add_argument("--delay", type=float, default=6.0, help="seconds between live API calls")
    args = parser.parse_args()

    report = run(use_llm=not args.no_llm, delay=args.delay)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{args.name}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_report(report)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
