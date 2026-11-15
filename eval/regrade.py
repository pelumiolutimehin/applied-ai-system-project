"""Regrade the parser section of a saved report under the current answer key, with no API calls.

    python -m eval.regrade --report baseline --name baseline_regraded

Uses the profiles the model actually returned (stored in the report), so the same answers are
graded with the newer rules. The original report is left untouched.
"""

import argparse
import json
from pathlib import Path

from eval import metrics, profiles
from eval.run_eval import RESULTS_DIR
from src.profile import UserProfile


def regrade_parser(saved_parser: dict) -> dict:
    cases = profiles.NAMED_GENRE_CASES + profiles.NO_GENRE_CASES
    by_text = {item["text"]: item for item in saved_parser["per_case"]}
    missing = [c["text"] for c in cases if c["text"] not in by_text]
    if missing:
        raise SystemExit(f"report has no saved answer for: {missing}")
    outcomes = []
    for case in cases:
        saved = by_text[case["text"]]
        profile = UserProfile(**saved["profile"]) if saved["profile"] else None
        outcomes.append({"text": case["text"], "profile": profile, "error": saved["error"]})
    return metrics.summarize_parser(cases, outcomes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", default="baseline", help="saved report name in eval/results")
    parser.add_argument("--name", default=None, help="output name (default: <report>_regraded)")
    args = parser.parse_args()

    report = json.loads((RESULTS_DIR / f"{args.report}.json").read_text(encoding="utf-8"))
    report["parser"] = regrade_parser(report["parser"])
    report["regraded_from"] = args.report
    out = RESULTS_DIR / f"{args.name or args.report + '_regraded'}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    p = report["parser"]
    print(f"Regraded {args.report} (same answers, three-tier genre grading)")
    for key in ("genre_invented_rate", "genre_indefensible_rate", "overall_case_pass_rate_strict", "overall_case_pass_rate"):
        print(f"  {key:<30}: {p[key]}")
    print(f"  {'no-genre tiers':<30}: {p['no_genre_tiers']}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
