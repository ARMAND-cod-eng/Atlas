"""The domain eval from the command line: which system runs for which variant,
what a compliance run file records, and the free citation comparison of runs.

No model is called: runs are built from answers written here, because what is
under test is the bookkeeping, and a mistake there misreports every number.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import pytest

from atlas_eval import cli
from atlas_eval.baselines import run_plain, run_single_call
from atlas_eval.citation_metrics import citation_table, score_run
from atlas_eval.dataset import DATASETS, load_dataset
from atlas_eval.runner import RunRecord, Variant, run_item

COMPLIANCE = load_dataset(*DATASETS["compliance"])


def test_each_variant_runs_its_own_system_in_the_dataset_domain() -> None:
    def described(runner: Any) -> tuple[Any, dict[str, Any]]:
        if isinstance(runner, functools.partial):
            return runner.func, runner.keywords
        return runner, {}

    variants: tuple[Variant, ...] = ("atlas", "no-critic", "single-call", "plain")
    assert [described(cli.runner_for(variant, "compliance")) for variant in variants] == [
        (run_item, {"critic_enabled": True, "domain": "compliance"}),
        (run_item, {"critic_enabled": False, "domain": "compliance"}),
        (run_single_call, {"domain": "compliance"}),
        (run_plain, {}),
    ]
    assert described(cli.runner_for("single-call", "general")) == (
        run_single_call,
        {"domain": "general"},
    )


@pytest.mark.parametrize(
    ("argv", "says"),
    [
        (["run", "--variant", "plain"], "plain answers compliance questions only"),
        (
            ["run", "--dataset", "compliance", "--write-baseline"],
            "the gate's baseline is the general dataset's",
        ),
    ],
)
def test_runs_that_would_measure_the_wrong_thing_are_refused_before_they_cost_anything(
    argv: list[str], says: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def must_not_run(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("refused runs must not start")

    monkeypatch.setattr(cli, "run_items", must_not_run)

    assert cli.main(argv) == 2
    assert says in capsys.readouterr().out


def _run(variant: Variant, answers: dict[str, str]) -> dict[str, Any]:
    records = [
        RunRecord(item_id=item_id, query="q", answer=answer, variant=variant)
        for item_id, answer in answers.items()
    ]
    return {
        "dataset": "compliance",
        "variant": variant,
        "records": [r.model_dump(mode="json") for r in records],
    }


def test_a_run_is_scored_from_its_answers_against_the_dataset() -> None:
    first = COMPLIANCE.items[0]
    run = _run("plain", {first.id: "It is high-risk [AI Act Art. 114]."})

    [score] = score_run(run, COMPLIANCE)

    assert (score.item_id, score.nonexistent, score.recall) == (first.id, ["AI Act Art. 114"], 0.0)


def test_a_compliance_run_file_records_its_dataset_its_web_and_its_citations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from atlas_eval.score import SuiteScore

    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path)
    first = COMPLIANCE.items[0]
    record = RunRecord(item_id=first.id, query=first.query, answer="[AI Act Art. 6(2)]")

    path = cli._write_run(
        "x", [record], SuiteScore(), "atlas", dataset_name="compliance", dataset=COMPLIANCE
    )
    written = json.loads(path.read_text(encoding="utf-8"))
    must = first.citations.must if first.citations is not None else []

    assert (written["dataset"], written["web"]) == ("compliance", "off")
    assert written["citations"]["per_item"] == [
        {
            "item_id": first.id,
            "cited": ["AI Act Art. 6(2)"],
            "nonexistent": [],
            "unsupported": [],
            "must_covered": ["AI Act Art. 6(2)"],
            "must_missed": [m for m in must if m != "AI Act Art. 6(2)"],
        }
    ]


def test_the_table_compares_each_run_with_the_reference_item_by_item() -> None:
    ids = [item.id for item in COMPLIANCE.items[:2]]
    atlas = _run("atlas", {ids[0]: "[AI Act Art. 6(2)]", ids[1]: "[AI Act Art. 6(2)]"})
    plain = _run("plain", {ids[0]: "[AI Act Art. 114]", ids[1]: "No citations."})

    table = citation_table({"atlas-v1": atlas, "plain-v1": plain}, COMPLIANCE, reference="atlas-v1")

    lines = table.splitlines()
    assert lines[0] == (
        "| run | variant | answers | fabricated | uncited | validity | precision | recall |"
    )
    assert lines[2].startswith("| atlas-v1 | atlas | 2 | 0.0% ")
    assert lines[3].startswith("| plain-v1 | plain | 2 | 50.0% ")
    # Against the reference, paired by item: plain fabricates in one of two answers more.
    assert "fabricated vs atlas-v1: +50.0 pp" in table


def test_a_compliance_run_end_to_end_saves_and_prints_its_citations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from atlas_eval.score import SuiteScore

    smoke = COMPLIANCE.smoke()

    async def agent(items: list[Any], concurrency: int, run_one: Any) -> list[RunRecord]:
        return [
            RunRecord(item_id=i.id, query=i.query, answer="[AI Act Art. 114]", variant="plain")
            for i in items
        ]

    async def judge(pairs: list[Any]) -> SuiteScore:
        return SuiteScore()

    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(cli, "run_items", agent)
    monkeypatch.setattr(cli, "score_all", judge)

    code = cli.main(
        ["run", "--dataset", "compliance", "--variant", "plain", "--smoke", "--name", "p"]
    )

    written = json.loads((tmp_path / "p.json").read_text(encoding="utf-8"))
    out = capsys.readouterr().out
    assert code == 0
    assert f"running {len(smoke)} compliance item(s) as 'plain' against the law library" in out
    assert f"| p | plain | {len(smoke)} | 100.0% " in out
    assert written["citations"]["summary"]["fabricated"]["value_pp"] == 100.0
