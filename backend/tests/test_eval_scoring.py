"""Scoring a run, and what a broken run is allowed to score.

Three defects, one shape: the eval reported something better than what happened.

* A run that **crashed** scored 1.0 on grounding and 0.67 on tools. Its empty
  answer decomposed into no claims, which the grounding scorer reads as an honest
  abstention, and four of the six tool checks pass vacuously on an empty plan. A
  regression that crashed more items pushed the grounding mean *up*.
* ``retry_count_within_cap`` read its counts from ``research_results``, which the
  upsert reducer keeps at one entry per subtask. It could not fail.
* ``atlas_eval run`` exited 0 however many items crashed, so the CI smoke — whose
  whole job is to catch crashes — passed with every item dead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from atlas.graph.state import ResearchResult, Subtask
from atlas.metrics import NodeTiming, RunMetrics
from atlas_eval import cli
from atlas_eval.dataset import load_dataset
from atlas_eval.runner import RunRecord
from atlas_eval.score import ItemScore, SuiteScore, researcher_runs, score_run
from atlas_eval.scorers import completion, grounding, tools
from atlas_eval.scorers.completion import CompletionScore, FactVerdict
from atlas_eval.scorers.grounding import ClaimList, GroundingScore, VerdictList, score_grounding
from atlas_eval.scorers.tools import PlanVerdict, ToolScore, score_tools


@pytest.fixture
def judge_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """A judge that approves everything, and records which schemas it was asked for."""
    calls: list[str] = []

    async def approving_judge(schema: type[BaseModel], system: str, user: str) -> BaseModel:
        calls.append(schema.__name__)
        if schema is FactVerdict:
            return FactVerdict(present=True, evidence="e")
        if schema is ClaimList:
            return ClaimList(claims=[])
        if schema is VerdictList:
            return VerdictList(verdicts=[])
        if schema is PlanVerdict:
            return PlanVerdict(independent=True, relevant=True)
        raise AssertionError(f"unexpected judge schema {schema.__name__}")

    for module in (completion, grounding, tools):
        monkeypatch.setattr(module, "judge", approving_judge)
    return calls


def _item_and_plan(subtask_count: int = 2) -> tuple[Any, list[Subtask]]:
    item = load_dataset().items[0]
    plan = [Subtask(id=f"s{i}", question=f"q{i}", rationale="r") for i in range(subtask_count)]
    return item, plan


# --- a crashed run --------------------------------------------------------------


async def test_a_crashed_run_scores_zero_in_every_category(judge_calls: list[str]) -> None:
    item, _ = _item_and_plan()
    crashed = RunRecord(item_id=item.id, query=item.query, answer="", error="ProviderError: x")

    score = await score_run(item, crashed)

    assert score.as_dict == {"completion": 0.0, "grounding": 0.0, "tools": 0.0}
    assert score.error == "ProviderError: x"
    # Nothing to judge, so nothing is spent judging it.
    assert judge_calls == []


async def test_a_blank_answer_is_not_an_abstention(judge_calls: list[str]) -> None:
    """No answer at all is not the same as an answer that asserts nothing."""
    blank = await score_grounding("", "some retrieved context")

    assert (blank.score, blank.not_applicable) == (0.0, False)


async def test_an_answer_with_no_factual_claims_is_still_an_abstention(
    judge_calls: list[str],
) -> None:
    """The control: the abstention path the failure-expected items rely on survives."""
    honest = await score_grounding("The sources do not cover this.", "context")

    assert (honest.score, honest.not_applicable) == (1.0, True)
    assert judge_calls == ["ClaimList"]


# --- retries are counted from executions ------------------------------------------


def _record_with_researcher_runs(runs: dict[str, int]) -> RunRecord:
    item, plan = _item_and_plan(len(runs))
    timings = [
        NodeTiming(node="researcher", subtask_id=subtask_id, duration_s=1.0)
        for subtask_id, count in runs.items()
        for _ in range(count)
    ]
    return RunRecord(
        item_id=item.id,
        query=item.query,
        answer="an answer",
        subtasks=plan,
        # One entry per subtask, however often it ran — that is the reducer's job.
        research_results=[ResearchResult(subtask_id=sid, findings="f") for sid in runs],
        metrics=RunMetrics(node_timings=timings),
    )


def test_researcher_runs_counts_executions_not_surviving_results() -> None:
    record = _record_with_researcher_runs({"s0": 3, "s1": 1})

    assert researcher_runs(record) == {"s0": 3, "s1": 1}
    # The control: what the check used to read. A retry is invisible in it.
    assert {r.subtask_id: 1 for r in record.research_results} == {"s0": 1, "s1": 1}


@pytest.mark.parametrize(("runs", "passed"), [(2, True), (3, False)])
async def test_the_retry_cap_check_can_fail(
    judge_calls: list[str], runs: int, passed: bool
) -> None:
    """MAX_CRITIC_ITERATIONS=2 allows one first pass and one retry: 2 runs, not 3."""
    record = _record_with_researcher_runs({"s0": runs, "s1": 1})
    item = load_dataset().by_id(record.item_id)

    score = await score_run(item, record)
    check = next(c for c in score.tools.checks if c.name == "retry_count_within_cap")

    assert check.passed is passed


@pytest.mark.parametrize(("searches", "passed"), [(12, True), (13, False)])
async def test_the_search_ceiling_is_the_real_bound(
    judge_calls: list[str], searches: int, passed: bool
) -> None:
    """2 subtasks x 3 attempts x 2 runs each = 12. The old ceiling allowed 18."""
    item, plan = _item_and_plan(2)
    calls = [f"distinct search number {i} {'x' * i}" for i in range(searches)]

    score = await score_tools(item, plan, ["s0", "s1"], calls, {"s0": 2, "s1": 2})
    check = next(c for c in score.checks if c.name == "search_count_within_bounds")

    assert check.passed is passed


# --- the command's exit status --------------------------------------------------


def _patch_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, crash_first: bool
) -> argparse.Namespace:
    async def fake_run_items(
        items: list[Any], concurrency: int = 5, run_one: Any = None
    ) -> list[RunRecord]:
        return [
            RunRecord(
                item_id=item.id,
                query=item.query,
                answer="" if crash_first and index == 0 else "an answer",
                error="ProviderError: x" if crash_first and index == 0 else None,
            )
            for index, item in enumerate(items)
        ]

    monkeypatch.setattr(cli, "run_items", fake_run_items)
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path)
    return cli.build_parser().parse_args(["run", "--smoke", "--name", "t", "--write-baseline"])


def test_a_crashed_item_fails_the_run_command(
    judge_calls: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = _patch_run(monkeypatch, tmp_path, crash_first=True)

    assert cli.cmd_run(args) == 1
    # The artefact is still written: it is what you debug the crash from.
    assert (tmp_path / "runs" / "t.json").exists() is True
    # A baseline with a crash baked in would make every later crash look normal.
    assert (tmp_path / "baseline.json").exists() is False


def test_a_clean_run_passes_and_writes_its_baseline(
    judge_calls: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The control: the same command, nothing crashed."""
    args = _patch_run(monkeypatch, tmp_path, crash_first=False)

    assert cli.cmd_run(args) == 0
    assert (tmp_path / "baseline.json").exists() is True


# --- comparison variants --------------------------------------------------------------


async def test_the_single_call_baseline_is_not_scored_on_tool_use(
    judge_calls: list[str],
) -> None:
    """It has no plan and no researchers, so tool use does not apply.

    Scoring it would charge the baseline for checks it cannot pass by design
    ("no subtasks were produced") and flatter Atlas by comparison.
    """
    item, _ = _item_and_plan()
    record = RunRecord(item_id=item.id, query=item.query, answer="an answer", variant="single-call")

    score = await score_run(item, record)

    assert (score.tools.score, score.tools.not_applicable) == (0.0, True)
    # Three judged rubric entries and one decomposition; no plan verdict.
    assert sorted(judge_calls) == ["ClaimList", "FactVerdict", "FactVerdict", "FactVerdict"]


def test_tool_use_means_skip_the_runs_it_does_not_apply_to() -> None:
    def item(item_id: str, tools: ToolScore) -> ItemScore:
        return ItemScore(
            item_id=item_id,
            completion=CompletionScore(score=1.0),
            grounding=GroundingScore(score=1.0),
            tools=tools,
        )

    baseline_only = SuiteScore(scores=[item("a", ToolScore(score=0.0, not_applicable=True))])
    assert baseline_only.means() == {"completion": 1.0, "grounding": 1.0}
    assert baseline_only.per_item() == {"completion": {"a": 1.0}, "grounding": {"a": 1.0}}

    # The control: an applicable tool score of 0.5 is averaged in as usual.
    mixed = SuiteScore(
        scores=[
            item("a", ToolScore(score=0.0, not_applicable=True)),
            item("b", ToolScore(score=0.5)),
        ]
    )
    assert mixed.means() == {"completion": 1.0, "grounding": 1.0, "tools": 0.5}
    assert mixed.per_item()["tools"] == {"b": 0.5}


async def test_each_variant_runs_its_own_system(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, bool | None]] = []

    async def fake_run_item(item: Any, critic_enabled: bool = True) -> RunRecord:
        seen.append(("graph", critic_enabled))
        return RunRecord(item_id=item.id, query=item.query, answer="a")

    async def fake_single(item: Any) -> RunRecord:
        seen.append(("single-call", None))
        return RunRecord(item_id=item.id, query=item.query, answer="a")

    monkeypatch.setattr(cli, "run_item", fake_run_item)
    monkeypatch.setattr(cli, "run_single_call", fake_single)
    item = load_dataset().items[0]

    for variant in ("atlas", "no-critic", "single-call"):
        await cli.runner_for(variant)(item)

    assert seen == [("graph", True), ("graph", False), ("single-call", None)]


def test_a_baseline_is_only_written_from_the_full_system(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Refused before anything runs: a no-critic baseline would gate production
    against a system it is not, and the refusal should not cost a 50-item run."""
    ran: list[int] = []

    async def must_not_run(
        items: list[Any], concurrency: int = 5, run_one: Any = None
    ) -> list[RunRecord]:
        ran.append(len(items))
        return []

    monkeypatch.setattr(cli, "run_items", must_not_run)
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path)
    args = cli.build_parser().parse_args(
        ["run", "--variant", "no-critic", "--name", "t", "--write-baseline"]
    )

    assert cli.cmd_run(args) == 2
    assert ran == []
    assert (tmp_path / "baseline.json").exists() is False


# --- noise ------------------------------------------------------------------------------


def _write_run(directory: Path, name: str, variant: str, scores: dict[str, float]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"name": name, "variant": variant, "per_item": {"completion": scores}}
    (directory / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_noise_reports_each_category_and_what_binds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    items = [f"q{i}" for i in range(50)]
    runs = tmp_path / "runs"
    _write_run(runs, "baseline", "atlas", dict.fromkeys(items, 0.8))
    _write_run(
        runs, "rerun1", "atlas", {q: 0.9 if i % 2 == 0 else 0.7 for i, q in enumerate(items)}
    )
    monkeypatch.setattr(cli, "RUNS_DIR", runs)

    code = cli.cmd_noise(cli.build_parser().parse_args(["noise", "--runs", "baseline", "rerun1"]))

    assert code == 0
    assert (
        "  completion  n=50  paired SD 10.10pp  drift 0.00pp  detectable >= 4.00pp"
        "  -> raise the threshold to at least 5pp"
    ) in capsys.readouterr().out.splitlines()


def test_noise_refuses_to_mix_variants(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The gap between Atlas and the baseline is the result, not noise."""
    runs = tmp_path / "runs"
    _write_run(runs, "baseline", "atlas", {"q0": 0.8, "q1": 0.8})
    _write_run(runs, "single", "single-call", {"q0": 0.5, "q1": 0.5})
    monkeypatch.setattr(cli, "RUNS_DIR", runs)

    code = cli.cmd_noise(cli.build_parser().parse_args(["noise", "--runs", "baseline", "single"]))

    assert code == 2
    assert capsys.readouterr().out.splitlines() == [
        "noise is measured between reruns of one variant; got atlas, single-call"
    ]


# --- what a run cost --------------------------------------------------------------------


def _fake_llm(model: str, input_tokens: int, output_tokens: int) -> Any:
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    def reply() -> AIMessage:
        return AIMessage(
            content="x",
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            response_metadata={"model_name": model},
        )

    return GenericFakeChatModel(messages=iter([reply() for _ in range(50)]))


def test_a_run_records_what_the_agent_and_the_judge_spent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Measured from provider usage, agent and judge apart.

    The eval's own judge calls appear in no run metrics, so before this the
    cost of scoring was a guess — and the full suite's budget was extrapolated
    from half of what it actually spends.
    """
    agent = _fake_llm("claude-x", 100, 20)
    judge_model = _fake_llm("gpt-x", 7, 3)

    async def fake_run_items(
        items: list[Any], concurrency: int = 5, run_one: Any = None
    ) -> list[RunRecord]:
        await agent.ainvoke("q")
        return [RunRecord(item_id=i.id, query=i.query, answer="an answer") for i in items]

    async def measured_judge(schema: type[BaseModel], system: str, user: str) -> BaseModel:
        await judge_model.ainvoke(user)
        if schema is FactVerdict:
            return FactVerdict(present=True)
        if schema is ClaimList:
            return ClaimList(claims=[])
        raise AssertionError(schema.__name__)

    for module in (completion, grounding, tools):
        monkeypatch.setattr(module, "judge", measured_judge)
    monkeypatch.setattr(cli, "run_items", fake_run_items)
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path / "runs")

    args = cli.build_parser().parse_args(["run", "--only", "q01", "--name", "t"])
    assert cli.cmd_run(args) == 0

    written = json.loads((tmp_path / "runs" / "t.json").read_text(encoding="utf-8"))
    # q01: three judged rubric entries and one claim decomposition — four calls.
    assert written["spend"] == {
        "agent": {"claude-x": {"input_tokens": 100, "output_tokens": 20}},
        "judge": {"gpt-x": {"input_tokens": 28, "output_tokens": 12}},
    }


# --- logging ----------------------------------------------------------------------------


def test_the_eval_logs_warnings_with_their_detail(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The first live recording logged "structured output failed validation"
    three times and dropped the node, schema and parsing error that say why —
    the CLI never configured logging, so Python's last-resort handler printed
    the bare message. Every field passed as ``extra`` now reaches the output."""
    import logging

    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    cli.main(["claim"])
    logging.getLogger("atlas.models").warning(
        "structured output failed validation; discarding the response",
        extra={"node": "critic", "schema": "CriticVerdict"},
    )

    [line] = [ln for ln in capsys.readouterr().err.splitlines() if "structured output" in ln]
    record = json.loads(line)
    assert (record["level"], record["node"], record["schema"]) == (
        "WARNING",
        "critic",
        "CriticVerdict",
    )


# --- rescoring ---------------------------------------------------------------------------


def test_rescoring_rejudges_the_answers_and_keeps_everything_else(
    judge_calls: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A judge fix must not cost a re-run of the agent.

    The baseline's tool-use judge misread independence and failed five correct
    single-subtask plans. The answers were fine; only their scoring was wrong.
    Rescoring recomputes the scores from the recorded answers and keeps what
    the agent did — its records, its spend, the commit it ran on — while saying
    which commit scored it.
    """
    item = load_dataset().items[0]
    record = RunRecord(item_id=item.id, query=item.query, answer="an answer")
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "old.json").write_text(
        json.dumps(
            {
                "name": "old",
                "variant": "atlas",
                "commit": "agent123",
                "spend": {"agent": {"claude-x": {"input_tokens": 9, "output_tokens": 1}}},
                "per_item": {"completion": {item.id: 0.0}},
                "records": [json.loads(record.model_dump_json())],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "RUNS_DIR", runs)
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cli, "_git_commit", lambda: "scorer456")

    code = cli.cmd_rescore(cli.build_parser().parse_args(["rescore", "--run", "old"]))

    rewritten = json.loads((runs / "old.json").read_text(encoding="utf-8"))
    assert code == 0
    assert rewritten["per_item"]["completion"] == {item.id: 1.0}
    assert rewritten["records"] == [json.loads(record.model_dump_json())]
    assert (rewritten["commit"], rewritten["scored_commit"]) == ("agent123", "scorer456")
    assert rewritten["spend"]["agent"] == {"claude-x": {"input_tokens": 9, "output_tokens": 1}}
    # The approving fake judge spends nothing measurable.
    assert rewritten["spend"]["judge"] == {}
    assert (tmp_path / "baseline.json").exists() is False


def test_a_judge_failure_loses_no_agent_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The records are saved before scoring starts.

    The first rerun lost a whole agent run — 50 items, about 2M Claude tokens —
    when the judge's provider ran out of credit during scoring: nothing was
    written until scoring finished, so there was nothing to rescore. Now the
    run file holds the records before the judge is asked anything, and a
    scoring failure says how to recover.
    """

    async def fake_run_items(
        items: list[Any], concurrency: int = 5, run_one: Any = None
    ) -> list[RunRecord]:
        return [RunRecord(item_id=i.id, query=i.query, answer="an answer") for i in items]

    async def broken_judge(schema: type[BaseModel], system: str, user: str) -> BaseModel:
        raise RuntimeError("Error code: 429 - insufficient_quota")

    for module in (completion, grounding, tools):
        monkeypatch.setattr(module, "judge", broken_judge)
    monkeypatch.setattr(cli, "run_items", fake_run_items)
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path / "runs")

    code = cli.cmd_run(cli.build_parser().parse_args(["run", "--only", "q01", "--name", "t"]))

    saved = json.loads((tmp_path / "runs" / "t.json").read_text(encoding="utf-8"))
    assert code == 1
    assert [r["item_id"] for r in saved["records"]] == ["q01"]
    assert (saved["per_item"], saved["means"]) == ({}, {})


def test_run_files_are_written_with_unix_line_endings(
    judge_calls: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same bytes on every platform: written on Windows they came out CRLF."""
    args = _patch_run(monkeypatch, tmp_path, crash_first=False)

    cli.cmd_run(args)

    assert b"\r\n" not in (tmp_path / "runs" / "t.json").read_bytes()
    assert b"\r\n" not in (tmp_path / "baseline.json").read_bytes()
