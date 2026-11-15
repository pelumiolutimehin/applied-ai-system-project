"""Append-only run log (JSON lines) so every recommendation run can be inspected afterwards.

Each line records what was asked, how it was interpreted, what came back, how confident the app
was, and whether any AI explanation was rejected. Logging must never break the app, so every
failure here is swallowed. Logs stay local (logs/ is gitignored) because they contain the text
users typed.
"""

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src import config


def build_record(
    request: Optional[str],
    source: str,  # "sliders", "text" or "sliders_fallback" (text could not be used)
    profile,
    recommendations,
    confidence,
    explanations=None,
    parse_error: Optional[str] = None,
) -> Dict:
    return {
        "request": request or None,
        "source": source,
        "parse_error": parse_error,
        "profile": profile.to_dict(),
        "results": [{"id": song["id"], "title": song["title"], "score": round(score, 3)}
                    for song, score, _ in recommendations],
        "confidence": {"score": confidence.score, "level": confidence.level,
                       "factors": confidence.factors, "notes": confidence.notes},
        "explanations": [{"source": e.source, "faithful": e.faithful, "problems": e.problems,
                          "rejected_text": e.rejected_text, "error": e.error}
                         for e in (explanations or [])],
    }


def log_run(record: Dict, path: Optional[Path] = None) -> bool:
    """Append one record. Returns False (and does nothing else) if the log cannot be written."""
    try:
        target = Path(path) if path else Path(config.LOG_PATH)
        target.parent.mkdir(parents=True, exist_ok=True)
        entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
        with open(target, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        return True
    except Exception:
        return False


def read_log(path: Optional[Path] = None) -> List[Dict]:
    """All readable records, oldest first. Missing file gives []; malformed lines are skipped."""
    target = Path(path) if path else Path(config.LOG_PATH)
    if not target.exists():
        return []
    records = []
    with open(target, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def categorize_problem(problem: str) -> str:
    """Group a faithfulness problem message into a stable category for counting."""
    if "listener's genre" in problem:
        return "claims a genre match that is not true"
    if problem.startswith("mentions mood"):
        return "states the wrong mood"
    if problem.startswith("mentions '"):
        return "mentions a genre the song does not have"
    if problem.startswith("quotes the number"):
        return "quotes a number not in the notes"
    if problem.startswith("calls the song"):
        return "describes the sound against its measured values"
    return "other"


def summarize_log(records: Iterable[Dict]) -> Dict:
    """Counts that answer 'how is the app behaving in practice?'"""
    records = list(records)
    explained = [e for r in records for e in r.get("explanations", [])]
    rejected = [e for e in explained if e.get("faithful") is False]
    reasons = Counter(categorize_problem(p) for e in rejected for p in e.get("problems", []))
    return {
        "runs": len(records),
        "by_source": dict(Counter(r.get("source") for r in records)),
        "by_confidence_level": dict(Counter(r.get("confidence", {}).get("level") for r in records)),
        "text_requests_that_fell_back": sum(1 for r in records if r.get("source") == "sliders_fallback"),
        "explanations_total": len(explained),
        "explanations_rejected": len(rejected),
        "rejection_reasons": dict(reasons.most_common()),
    }
