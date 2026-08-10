"""Command line for the eval suite.

    uv run python -m atlas_eval review      # mark items reviewed, one at a time
    uv run python -m atlas_eval record      # hit live Tavily, write the corpus
    uv run python -m atlas_eval run         # run + score (replay only)
    uv run python -m atlas_eval label       # human calibration labels
    uv run python -m atlas_eval calibrate   # judge-human agreement
    uv run python -m atlas_eval gate        # compare a run against baseline
    uv run python -m atlas_eval claim       # what the dataset supports claiming

``review`` and ``label`` are interactive and **write after every keystroke**, so
stopping halfway loses nothing and resuming picks up where you left off. That is
not a nicety: a labeling tool that loses work at item 40 gets abandoned at item 12,
and then the calibration never happens at all.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from atlas_eval.dataset import (
    DATA_DIR,
    DATASET_PATH,
    Dataset,
    EvalItem,
    load_dataset,
    provenance_summary,
    write_jsonl,
)
from atlas_eval.gate import Baseline, compare, load_baseline
from atlas_eval.runner import RunRecord, run_items
from atlas_eval.score import SuiteScore, score_all

RUNS_DIR = DATA_DIR / "runs"
LABELS_PATH = DATA_DIR / "labels" / "human.v1.jsonl"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _select(dataset: Dataset, smoke: bool, only: str | None) -> list[EvalItem]:
    if only:
        wanted = {i.strip() for i in only.split(",") if i.strip()}
        return [item for item in dataset.items if item.id in wanted]
    return dataset.smoke() if smoke else dataset.items


# --- review --------------------------------------------------------------------


def cmd_review(args: argparse.Namespace) -> int:
    """Walk the dataset one item at a time, marking each reviewed.

    Nothing else sets ``reviewed_by``. Skipping an item leaves it unreviewed, and
    that is what ``claim`` reports — the point is that the description of the
    dataset tracks the work actually done on it.
    """
    dataset = load_dataset()
    pending = [i for i in dataset.items if i.provenance.reviewed_by is None]
    if not pending:
        print("every item is already reviewed")
        return 0

    print(f"{len(pending)} item(s) to review. [enter]=approve  s=skip  q=save and quit\n")
    for position, item in enumerate(pending, start=1):
        print("=" * 78)
        print(
            f"[{position}/{len(pending)}]  {item.id}   width {item.expected_subtasks.min}-"
            f"{item.expected_subtasks.max}   {item.recency}   {', '.join(item.tags)}"
        )
        print(f"\nQUERY\n  {item.query}")
        print("\nRUBRIC (an answer must contain all of these)")
        for entry in item.rubric:
            marker = "[det]" if entry.check == "deterministic" else "[jdg]"
            print(f"  {marker} {entry.claim}")
            if entry.patterns:
                print(f"        patterns: {entry.patterns}")
            if entry.number is not None:
                print(f"        number: {entry.number} +/- {entry.tolerance}")
        print(f"\nWRONG BUT PLAUSIBLE\n  {item.wrong_but_plausible}")

        choice = input("\napprove? ").strip().lower()
        if choice == "q":
            break
        if choice == "s":
            continue
        item.provenance.reviewed_by = "human"
        item.provenance.reviewed_at = _now()
        # Written immediately: an interrupted session keeps everything approved
        # so far.
        write_jsonl(DATASET_PATH, dataset.items)

    summary = provenance_summary(load_dataset())
    print(f"\n{summary.reviewed}/{summary.total} reviewed")
    print(f'the dataset currently supports: "{summary.claim}"')
    return 0


def cmd_claim(args: argparse.Namespace) -> int:
    """Print the strongest description the dataset's provenance supports."""
    summary = provenance_summary(load_dataset())
    print(summary.claim)
    if not summary.fully_reviewed:
        print(
            f"  ({summary.total - summary.reviewed} item(s) unreviewed — "
            '"hand-labeled" would overstate this)'
        )
    return 0


# --- record --------------------------------------------------------------------


def cmd_record(args: argparse.Namespace) -> int:
    """Hit live Tavily and write the frozen corpus. The only online command."""
    from atlas_eval.record import record_corpus

    dataset = load_dataset()
    items = _select(dataset, args.smoke, args.only)
    asyncio.run(record_corpus(items, passes=args.passes))
    return 0


# --- run -----------------------------------------------------------------------


def _write_run(name: str, records: list[RunRecord], scores: SuiteScore) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{name}.json"
    payload = {
        "name": name,
        "created": _now(),
        "commit": _git_commit(),
        "means": scores.means(),
        "per_item": scores.per_item(),
        "check_split": {k: list(v) for k, v in scores.check_split().items()},
        "records": [json.loads(r.model_dump_json()) for r in records],
        "scores": [json.loads(s.model_dump_json()) for s in scores.scores],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def cmd_run(args: argparse.Namespace) -> int:
    dataset = load_dataset()
    items = _select(dataset, args.smoke, args.only)
    print(f"running {len(items)} item(s) against the frozen corpus")

    records = asyncio.run(run_items(items, concurrency=args.concurrency))
    failed = [r for r in records if r.error]
    if failed:
        print(f"  {len(failed)} run(s) errored: {[r.item_id for r in failed]}")

    by_id = {item.id: item for item in items}
    scores = asyncio.run(score_all([(by_id[r.item_id], r) for r in records]))

    print("\nmeans:")
    for category, mean in scores.means().items():
        print(f"  {category:<11} {mean:.3f}")
    print("\ndeterministic:judge checks actually run:")
    for category, (det, jdg) in scores.check_split().items():
        total = det + jdg
        share = f"{det / total:.0%}" if total else "n/a"
        print(f"  {category:<11} {det} deterministic / {jdg} judge  ({share} deterministic)")

    path = _write_run(args.name, records, scores)
    print(f"\nwrote {path}")

    if args.write_baseline:
        baseline = Baseline(
            commit=_git_commit(),
            dataset_version=dataset.meta.dataset_version,
            corpus_recorded_at=args.corpus_date,
            run_date=_now()[:10],
            n_items=len(records),
            means=scores.means(),
            notes=args.notes,
        )
        (DATA_DIR / "baseline.json").write_text(
            baseline.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote baseline from {len(records)} items")
    return 0


# --- gate ----------------------------------------------------------------------


def cmd_gate(args: argparse.Namespace) -> int:
    baseline = load_baseline()
    current = json.loads((RUNS_DIR / f"{args.run}.json").read_text(encoding="utf-8"))
    baseline_run_path = RUNS_DIR / f"{args.baseline_run}.json"
    baseline_per_item = None
    if baseline_run_path.exists():
        baseline_per_item = json.loads(baseline_run_path.read_text(encoding="utf-8"))["per_item"]

    result = compare(baseline, current["per_item"], baseline_per_item, threshold_pp=args.threshold)
    print(result.report())
    return 0 if result.passed else 1


# --- label / calibrate ---------------------------------------------------------


def cmd_label(args: argparse.Namespace) -> int:
    """Collect human binary labels for calibration, resumably."""
    from atlas_eval.labeling import run_labeling

    return run_labeling(args.run, args.limit)


def cmd_calibrate(args: argparse.Namespace) -> int:
    from atlas_eval.labeling import report_agreement

    return report_agreement(args.run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlas_eval", description="Atlas evaluation suite")
    sub = parser.add_subparsers(dest="command", required=True)

    review = sub.add_parser("review", help="mark dataset items reviewed, one at a time")
    review.set_defaults(func=cmd_review)

    claim = sub.add_parser("claim", help="what the dataset's provenance supports claiming")
    claim.set_defaults(func=cmd_claim)

    record = sub.add_parser("record", help="hit live Tavily and write the frozen corpus")
    record.add_argument("--passes", type=int, default=3)
    record.add_argument("--smoke", action="store_true")
    record.add_argument("--only", default=None, help="comma-separated item ids")
    record.set_defaults(func=cmd_record)

    run = sub.add_parser("run", help="run and score against the frozen corpus")
    run.add_argument("--name", default="latest")
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--only", default=None)
    run.add_argument("--concurrency", type=int, default=5)
    run.add_argument("--write-baseline", action="store_true")
    run.add_argument("--corpus-date", default="")
    run.add_argument("--notes", default="")
    run.set_defaults(func=cmd_run)

    gate = sub.add_parser("gate", help="compare a run against the baseline")
    gate.add_argument("--run", default="latest")
    gate.add_argument("--baseline-run", default="baseline")
    gate.add_argument("--threshold", type=float, default=2.0)
    gate.set_defaults(func=cmd_gate)

    label = sub.add_parser("label", help="collect human calibration labels")
    label.add_argument("--run", default="baseline")
    label.add_argument("--limit", type=int, default=15, help="items to label")
    label.set_defaults(func=cmd_label)

    calibrate = sub.add_parser("calibrate", help="judge-human agreement")
    calibrate.add_argument("--run", default="baseline")
    calibrate.set_defaults(func=cmd_calibrate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result
