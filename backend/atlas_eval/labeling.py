"""Human calibration labelling, and the agreement report it feeds.

**Why the labels are binary.** A human scoring "0.7" against a rubric is making a
soft judgement, and agreement between two such numbers needs an interval statistic
whose assumptions nobody checks. A human ticking "is this fact present? y/n" is
making a binary decision, so agreement is a 2x2 table and Cohen's kappa applies
directly. So this asks about one rubric entry, or one atomic claim, at a time.

**Why it writes after every keystroke.** A labelling tool that loses work at item
40 gets abandoned at item 12, and then the calibration never happens — which
leaves an uncalibrated judge, which is the thing that makes the whole suite
decoration.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from atlas_eval.calibration import (
    Agreement,
    CalibrationReport,
    HumanLabel,
    LabelSet,
    agreement,
)
from atlas_eval.dataset import DATA_DIR

LABELS_PATH = DATA_DIR / "labels" / "human.v1.jsonl"
RUNS_DIR = DATA_DIR / "runs"

# 15 items: 10 the judge prompt may be iterated against, 5 held back. The holdout
# is what distinguishes a calibrated judge from a prompt tuned to its own test.
CALIBRATION_ITEMS = 15
HOLDOUT_ITEMS = 5


def load_labels(path: Path = LABELS_PATH) -> LabelSet:
    if not path.exists():
        return LabelSet()
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            labels.append(HumanLabel.model_validate_json(line))
    return LabelSet(labels=labels)


def append_label(label: HumanLabel, path: Path = LABELS_PATH) -> None:
    """One line, flushed immediately. Resumability depends on this."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(label.model_dump_json() + "\n")
        handle.flush()


def _load_run(name: str) -> dict[str, object]:
    path = RUNS_DIR / f"{name}.json"
    if not path.exists():
        raise SystemExit(
            f"no run at {path}. Run `atlas_eval run --name {name}` first — "
            "labels are collected against a specific run's answers."
        )
    result: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    return result


def run_labeling(run_name: str, limit: int) -> int:
    """Walk one decision at a time, recording each immediately."""
    run = _load_run(run_name)
    scores = run["scores"]
    records = run["records"]
    assert isinstance(scores, list) and isinstance(records, list)

    answers = {r["item_id"]: r["answer"] for r in records}
    done = {(lbl.item_id, lbl.dimension, lbl.decision_id) for lbl in load_labels().labels}

    # A fixed slice, so the split does not shift between sessions.
    chosen = scores[:limit]
    pending: list[tuple[str, str, str, str, str, str]] = []
    for index, score in enumerate(chosen):
        item_id = score["item_id"]
        split = "holdout" if index >= limit - HOLDOUT_ITEMS else "iteration"
        for entry in score["completion"]["entries"]:
            if entry["check"] != "judge":
                continue  # deterministic entries need no human opinion
            key = (item_id, "completion", entry["entry_id"])
            if key not in done:
                pending.append(
                    (
                        item_id,
                        "completion",
                        entry["entry_id"],
                        entry["claim"],
                        split,
                        answers.get(item_id, ""),
                    )
                )
        for verdict in score["grounding"]["verdicts"]:
            key = (item_id, "grounding", verdict["claim"][:120])
            if key not in done:
                pending.append(
                    (item_id, "grounding", verdict["claim"][:120], verdict["claim"], split, "")
                )

    if not pending:
        print("nothing left to label")
        return 0

    print(f"{len(pending)} decision(s). y=yes  n=no  q=save and quit")
    print("Everything is written as you go; stopping is safe.\n")

    for position, (item_id, dimension, decision_id, subject, split, answer) in enumerate(
        pending, start=1
    ):
        print("=" * 78)
        print(f"[{position}/{len(pending)}]  {item_id}  {dimension}  ({split})")
        if dimension == "completion":
            print(f"\nANSWER\n{answer[:1500]}")
            print(f"\nIs this fact present in the answer?\n  {subject}")
        else:
            print(f"\nIs this claim supported by the retrieved sources?\n  {subject}")

        while True:
            choice = input("\n[y/n/q] ").strip().lower()
            if choice in {"y", "n", "q"}:
                break
        if choice == "q":
            break

        append_label(
            HumanLabel(
                item_id=item_id,
                dimension=dimension,
                decision_id=decision_id,
                subject=subject,
                human=choice == "y",
                labeled_at=datetime.now(UTC).isoformat(timespec="seconds"),
                split=split,
            )
        )

    total = len(load_labels().labels)
    print(f"\n{total} label(s) recorded")
    return 0


def report_agreement(run_name: str) -> int:
    """Judge vs human, per dimension, on the iteration set and the holdout."""
    run = _load_run(run_name)
    scores = run["scores"]
    assert isinstance(scores, list)
    labels = load_labels()
    if not labels.labels:
        raise SystemExit("no human labels yet — run `atlas_eval label` first")

    judge_by_key: dict[tuple[str, str, str], bool] = {}
    for score in scores:
        item_id = score["item_id"]
        for entry in score["completion"]["entries"]:
            judge_by_key[(item_id, "completion", entry["entry_id"])] = bool(entry["satisfied"])
        for verdict in score["grounding"]["verdicts"]:
            judge_by_key[(item_id, "grounding", verdict["claim"][:120])] = bool(
                verdict["supported"]
            )

    exit_code = 0
    for dimension in ("completion", "grounding"):
        tables: dict[str, Agreement] = {}
        for split in ("iteration", "holdout"):
            pairs = [
                (judge_by_key[(lbl.item_id, lbl.dimension, lbl.decision_id)], lbl.human)
                for lbl in labels.for_dimension(dimension, split)
                if (lbl.item_id, lbl.dimension, lbl.decision_id) in judge_by_key
            ]
            tables[split] = agreement([j for j, _ in pairs], [h for _, h in pairs])

        report = CalibrationReport(
            dimension=dimension, iteration=tables["iteration"], holdout=tables["holdout"]
        )
        print(report.summary())
        print()
        if not report.passes:
            exit_code = 1

    return exit_code
