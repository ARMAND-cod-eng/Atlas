"""Command line for the eval suite.

    uv run python -m atlas_eval review      # mark items reviewed, one at a time
    uv run python -m atlas_eval record      # hit live Tavily, write the corpus
    uv run python -m atlas_eval run         # run + score (replay only)
    uv run python -m atlas_eval label       # human calibration labels
    uv run python -m atlas_eval calibrate   # judge-human agreement
    uv run python -m atlas_eval gate        # compare a run against baseline
    uv run python -m atlas_eval noise       # run-to-run noise from reruns
    uv run python -m atlas_eval report      # results table and chart from run files
    uv run python -m atlas_eval claim       # what the dataset supports claiming

``review`` and ``label`` are interactive and **write after every keystroke**, so
stopping halfway loses nothing and resuming picks up where you left off. That is
not a nicety: a labeling tool that loses work at item 40 gets abandoned at item 12,
and then the calibration never happens at all.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import subprocess
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks import get_usage_metadata_callback

from atlas.models import primary_provider
from atlas.runtime import configure_logging
from atlas_eval.baselines import run_single_call
from atlas_eval.dataset import (
    DATA_DIR,
    DATASET_PATH,
    Dataset,
    EvalItem,
    load_dataset,
    provenance_summary,
    write_jsonl,
)
from atlas_eval.gate import (
    CATEGORIES,
    THRESHOLDS_PP,
    Baseline,
    compare,
    load_baseline,
    noise_floor,
)
from atlas_eval.runner import RunRecord, Variant, run_item, run_items
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
    with measured_spend() as spend:
        asyncio.run(record_corpus(items, passes=args.passes))
    print(_format_spend({"recording": spend}))
    return 0


# --- run -----------------------------------------------------------------------


def runner_for(variant: Variant) -> Callable[[EvalItem], Awaitable[RunRecord]]:
    """How one item is run under each variant."""
    if variant == "single-call":
        return run_single_call
    return functools.partial(run_item, critic_enabled=variant == "atlas")


# {model: {"input_tokens": n, "output_tokens": n}}
Spend = dict[str, dict[str, int]]

# Past this share of agent calls answered by a provider other than the primary,
# a run measures the fallback, not the system. Anthropic's credit ran out partway
# through the second rerun: the fallback answered everything after that, three
# runs exited 0 — two entirely on GPT-4o — and nothing said so. A timeout or two
# handed to the fallback is real production behaviour and stays in the numbers;
# most of a run answered by it is a broken environment, and is refused.
FALLBACK_LIMIT = 0.05


def _off_primary(providers: Sequence[str], primary: str) -> tuple[int, int]:
    """(calls answered by anything but ``primary``, all calls)."""
    return sum(1 for provider in providers if provider != primary), len(providers)


def _refused_for_fallback(name: str, run: Mapping[str, Any]) -> bool:
    """Print why and return True if a run file was mostly answered by the fallback."""
    primary = str(run.get("primary_provider") or primary_provider())
    records = run.get("records", [])
    providers = [c["provider"] for r in records for c in r["metrics"]["llm_calls"]]
    off, total = _off_primary(providers, primary)
    if total and off / total > FALLBACK_LIMIT:
        print(
            f"refusing run {name!r}: {off} of {total} agent calls ({off / total:.0%}) were "
            f"answered by the fallback, not the primary ({primary})"
        )
        return True
    return False


@contextmanager
def measured_spend() -> Iterator[Spend]:
    """Tokens per model for every call made inside, from the providers' usage.

    Filled in when the block exits. Measured rather than summed from run
    metrics because the judge's calls appear in no run metrics: before this,
    the cost of scoring was a guess.
    """
    spend: Spend = {}
    with get_usage_metadata_callback() as handler:
        yield spend
    for model, usage in handler.usage_metadata.items():
        spend[model] = {
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
        }


def _format_spend(spend: Mapping[str, Spend]) -> str:
    lines = ["\ntokens spent:"]
    for phase, models in spend.items():
        if not models:
            lines.append(f"  {phase:<10} none")
        for model, usage in sorted(models.items()):
            lines.append(
                f"  {phase:<10} {model:<28} {usage['input_tokens']:>11,} in "
                f"{usage['output_tokens']:>10,} out"
            )
    return "\n".join(lines)


def _write_run(
    name: str,
    records: list[RunRecord],
    scores: SuiteScore,
    variant: Variant = "atlas",
    spend: Mapping[str, Spend] | None = None,
    agent_commit: str | None = None,
    primary: str | None = None,
) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{name}.json"
    scored_commit = _git_commit()
    payload = {
        "name": name,
        "variant": variant,
        "spend": dict(spend or {}),
        "created": _now(),
        # The commit the agent ran on, and the one whose scorers judged it. The
        # same for a fresh run; different after `rescore`.
        "commit": agent_commit or scored_commit,
        "scored_commit": scored_commit,
        # Which provider the run was meant to measure. A run file alone could not
        # say, so a fallback-answered run looked like any other.
        "primary_provider": primary or primary_provider(),
        "means": scores.means(),
        "per_item": scores.per_item(),
        "check_split": {k: list(v) for k, v in scores.check_split().items()},
        "records": [json.loads(r.model_dump_json()) for r in records],
        "scores": [json.loads(s.model_dump_json()) for s in scores.scores],
    }
    # newline="\n": the same bytes on every platform. On Windows, text mode
    # wrote CRLF and every run file arrived in git needing conversion.
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
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
    if args.write_baseline and args.variant != "atlas":
        # Checked before anything runs: the gate compares production against the
        # baseline, so a baseline from another system would gate the wrong thing —
        # and finding that out should not cost a fifty-item run.
        print(f"refusing to write a baseline from the {args.variant!r} variant; use 'atlas'")
        return 2

    dataset = load_dataset()
    items = _select(dataset, args.smoke, args.only)
    print(f"running {len(items)} item(s) as {args.variant!r} against the frozen corpus")

    with measured_spend() as agent_spend:
        records = asyncio.run(
            run_items(items, concurrency=args.concurrency, run_one=runner_for(args.variant))
        )
    failed = [r for r in records if r.error]
    if failed:
        print(f"  {len(failed)} run(s) errored: {[r.item_id for r in failed]}")

    # Saved before the judge is asked anything. The first rerun lost a whole
    # agent run — 50 items, about 2M tokens — when the judge's provider ran out
    # of credit mid-scoring, because nothing was written until scoring finished.
    _write_run(args.name, records, SuiteScore(), args.variant, {"agent": agent_spend})

    by_id = {item.id: item for item in items}
    try:
        with measured_spend() as judge_spend:
            scores = asyncio.run(score_all([(by_id[r.item_id], r) for r in records]))
    except Exception as exc:
        print(
            f"\nscoring failed: {type(exc).__name__}: {str(exc)[:300]}\n"
            f"the agent's records are saved; once the judge is reachable, run\n"
            f"  uv run python -m atlas_eval rescore --run {args.name}"
        )
        return 1
    spend = {"agent": agent_spend, "judge": judge_spend}
    print(_format_spend(spend))
    primary = primary_provider()
    on_fallback, agent_calls = _off_primary(
        [c.provider for r in records for c in r.metrics.llm_calls], primary
    )
    fallback_share = on_fallback / agent_calls if agent_calls else 0.0
    print(
        f"\nagent calls answered by the fallback: {on_fallback} of {agent_calls}"
        f" ({fallback_share:.0%})"
    )

    print("\nmeans:")
    for category, mean in scores.means().items():
        print(f"  {category:<11} {mean:.3f}")
    print("\ndeterministic:judge checks actually run:")
    for category, (det, jdg) in scores.check_split().items():
        total = det + jdg
        share = f"{det / total:.0%}" if total else "n/a"
        print(f"  {category:<11} {det} deterministic / {jdg} judge  ({share} deterministic)")

    # Written even when something crashed: it is what the crash is debugged from.
    path = _write_run(args.name, records, scores, args.variant, spend, primary=primary)
    print(f"\nwrote {path}")

    if failed:
        # A crash is never a pass. This used to exit 0 however many items died,
        # so the CI smoke — which exists to catch crashes — passed with all of
        # them dead.
        print(f"\nFAIL: {len(failed)} of {len(records)} run(s) crashed")
        if args.write_baseline:
            print("not writing a baseline: it would record these crashes as normal")
        return 1

    if fallback_share > FALLBACK_LIMIT:
        print(
            f"\nFAIL: {fallback_share:.0%} of agent calls were answered by the fallback, not "
            f"{primary}. This run measures the fallback, not the system — check the "
            "primary's credit, quota and status, then run it again."
        )
        if args.write_baseline:
            print("not writing a baseline from it")
        return 1

    if args.write_baseline:
        _write_baseline(
            _git_commit(), dataset, records, scores, args.corpus_date, _now()[:10], args.notes
        )
    return 0


def _write_baseline(
    commit: str,
    dataset: Dataset,
    records: list[RunRecord],
    scores: SuiteScore,
    corpus_date: str,
    run_date: str,
    notes: str,
) -> None:
    baseline = Baseline(
        commit=commit,
        dataset_version=dataset.meta.dataset_version,
        corpus_recorded_at=corpus_date,
        run_date=run_date,
        n_items=len(records),
        means=scores.means(),
        notes=notes,
    )
    (DATA_DIR / "baseline.json").write_text(
        baseline.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote baseline from {len(records)} items")


# --- rescore -------------------------------------------------------------------


def _variant_of(run: Mapping[str, Any]) -> Variant:
    """The variant a run file records, checked rather than cast."""
    value = run.get("variant", "atlas")
    if value == "no-critic":
        return "no-critic"
    if value == "single-call":
        return "single-call"
    if value == "atlas":
        return "atlas"
    raise ValueError(f"run file names an unknown variant: {value!r}")


def cmd_rescore(args: argparse.Namespace) -> int:
    """Re-judge a run's recorded answers with the current scorers.

    For when the scoring was wrong and the answers were not — a judge prompt
    fix, a scorer bug. The agent is not re-run: its records, its spend and the
    commit it ran on are kept, and the run file says which commit scored it.
    Costs only the judge.
    """
    payload = _load_run(args.run)
    records = [RunRecord.model_validate(r) for r in payload["records"]]
    if any(r.error for r in records) and args.write_baseline:
        print("not writing a baseline: the run has crashed items")
        return 1
    variant = _variant_of(payload)
    if args.write_baseline and variant != "atlas":
        print(f"refusing to write a baseline from the {variant!r} variant; use 'atlas'")
        return 2

    dataset = load_dataset()
    by_id = {item.id: item for item in dataset.items}
    with measured_spend() as judge_spend:
        scores = asyncio.run(score_all([(by_id[r.item_id], r) for r in records]))
    spend = {**payload.get("spend", {}), "judge": judge_spend}
    print(_format_spend({"judge": judge_spend}))
    print("\nmeans:")
    for category, mean in scores.means().items():
        print(f"  {category:<11} {mean:.3f}")

    agent_commit = str(payload.get("commit") or "unknown")
    path = _write_run(
        args.run,
        records,
        scores,
        variant,
        spend,
        agent_commit=agent_commit,
        primary=str(payload.get("primary_provider") or primary_provider()),
    )
    print(f"\nrewrote {path}")
    if args.write_baseline:
        _write_baseline(
            agent_commit,
            dataset,
            records,
            scores,
            args.corpus_date,
            str(payload.get("created", ""))[:10],
            args.notes,
        )
    return 0


# --- gate ----------------------------------------------------------------------


def _thresholds(forced: float | None) -> dict[str, float]:
    """Each category's measured threshold, or one value for all if forced."""
    return dict(THRESHOLDS_PP) if forced is None else dict.fromkeys(CATEGORIES, forced)


def cmd_gate(args: argparse.Namespace) -> int:
    baseline = load_baseline()
    current = json.loads((RUNS_DIR / f"{args.run}.json").read_text(encoding="utf-8"))
    baseline_run_path = RUNS_DIR / f"{args.baseline_run}.json"
    baseline_per_item = None
    if baseline_run_path.exists():
        baseline_per_item = json.loads(baseline_run_path.read_text(encoding="utf-8"))["per_item"]

    result = compare(
        baseline, current["per_item"], baseline_per_item, thresholds=_thresholds(args.threshold)
    )
    print(result.report())
    return 0 if result.passed else 1


# --- noise ---------------------------------------------------------------------


def _load_run(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((RUNS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return loaded


def cmd_noise(args: argparse.Namespace) -> int:
    """Run-to-run noise from reruns of identical code, and what the gate can catch."""
    runs = [_load_run(name) for name in args.runs]
    if [
        name for name, run in zip(args.runs, runs, strict=True) if _refused_for_fallback(name, run)
    ]:
        return 2
    variants = sorted({str(run.get("variant", "atlas")) for run in runs})
    if len(variants) > 1:
        print(f"noise is measured between reruns of one variant; got {', '.join(variants)}")
        return 2
    try:
        estimates = noise_floor([run["per_item"] for run in runs])
    except ValueError as exc:
        print(exc)
        return 2

    thresholds = _thresholds(args.threshold)
    print(f"noise across {len(runs)} runs of the same code ({variants[0]})\n")
    for e in estimates:
        threshold = thresholds[e.category]
        print(
            f"  {e.category:<11} n={e.items}  paired SD {e.paired_sd_pp:.2f}pp  "
            f"drift {e.drift_pp:.2f}pp  detectable >= {e.detectable_pp:.2f}pp  "
            f"threshold {threshold:g}pp  -> {e.verdict(threshold)}"
        )
    print(
        "\nA regression smaller than 'detectable' cannot be told apart from run-to-run"
        " noise at this n, whatever the threshold."
    )
    return 0


# --- report --------------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> int:
    """The results table (and, with --chart, the chart) from committed run files."""
    from atlas_eval.report import (
        critic_effect,
        deltas_against,
        render_chart,
        render_markdown,
        summarize,
    )

    names = [args.reference, *(name for name in args.runs if name != args.reference)]
    runs = {name: _load_run(name) for name in names}
    if [name for name, run in runs.items() if _refused_for_fallback(name, run)]:
        return 2
    reference = runs[args.reference]
    variant = str(reference.get("variant", "atlas"))
    if variant != "atlas":
        print(
            "the reference must be a run of the production system ('atlas'); "
            f"{args.reference!r} is {variant!r}"
        )
        return 2

    no_critic = next((r for r in runs.values() if r.get("variant") == "no-critic"), None)
    markdown = render_markdown(
        [summarize(name, run) for name, run in runs.items()],
        args.reference,
        {
            name: deltas_against(reference, run)
            for name, run in runs.items()
            if name != args.reference
        },
        critic_effect(reference, no_critic) if no_critic is not None else None,
    )
    print(markdown)
    if args.out:
        Path(args.out).write_text(markdown, encoding="utf-8")
        print(f"wrote {args.out}")
    if args.chart:
        for path in render_chart([summarize(n, r) for n, r in runs.items()], Path(args.chart)):
            print(f"wrote {path}")
    return 0


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
    run.add_argument(
        "--variant",
        choices=["atlas", "no-critic", "single-call"],
        default="atlas",
        help="atlas is production; the others exist to measure it against",
    )
    run.add_argument("--write-baseline", action="store_true")
    run.add_argument("--corpus-date", default="")
    run.add_argument("--notes", default="")
    run.set_defaults(func=cmd_run)

    gate = sub.add_parser("gate", help="compare a run against the baseline")
    gate.add_argument("--run", default="latest")
    gate.add_argument("--baseline-run", default="baseline")
    gate.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="one threshold in pp for every category (default: each category's measured one)",
    )
    gate.set_defaults(func=cmd_gate)

    rescore = sub.add_parser("rescore", help="re-judge a run's answers without re-running it")
    rescore.add_argument("--run", required=True)
    rescore.add_argument("--write-baseline", action="store_true")
    rescore.add_argument("--corpus-date", default="")
    rescore.add_argument("--notes", default="")
    rescore.set_defaults(func=cmd_rescore)

    noise = sub.add_parser("noise", help="run-to-run noise from reruns of identical code")
    noise.add_argument("--runs", nargs="+", required=True, help="run names, two or more")
    noise.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="compare against one threshold in pp instead of each category's own",
    )
    noise.set_defaults(func=cmd_noise)

    report = sub.add_parser("report", help="results table and chart from run files")
    report.add_argument("--runs", nargs="+", required=True, help="run names to compare")
    report.add_argument("--reference", default="baseline", help="the atlas run to compare against")
    report.add_argument("--out", default=None, help="also write the markdown here")
    report.add_argument("--chart", default=None, help="path stem for <stem>.light/.dark.png")
    report.set_defaults(func=cmd_report)

    label = sub.add_parser("label", help="collect human calibration labels")
    label.add_argument("--run", default="baseline")
    label.add_argument("--limit", type=int, default=15, help="items to label")
    label.set_defaults(func=cmd_label)

    calibrate = sub.add_parser("calibrate", help="judge-human agreement")
    calibrate.add_argument("--run", default="baseline")
    calibrate.set_defaults(func=cmd_calibrate)

    return parser


def main(argv: list[str] | None = None) -> int:
    # The same JSON lines as the API and worker, so a warning's ``extra`` fields —
    # which node, which schema, what failed to parse — reach the output. Without
    # this, Python's last-resort handler printed bare messages, and the first live
    # recording's three structured-output failures said nothing about why.
    # WARNING, not LOG_LEVEL: `claim` and `review` run without API keys, and the
    # settings that hold LOG_LEVEL refuse to load without them.
    configure_logging("WARNING")
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result
