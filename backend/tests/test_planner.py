"""Planner: decomposition, truncation, and the degraded path."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Overwrite

from atlas.graph import nodes
from atlas.graph.builder import build_state_graph
from atlas.graph.state import PlannerOutput, Source, Subtask
from atlas.metrics import DEGRADED_PLANNER_FALLBACK, RunMetrics
from tests.conftest import initial_state, make_call_model, make_call_structured, run_config


def _metrics(result: dict[str, object]) -> RunMetrics:
    """The planner opens a run, so its metrics replace the last run's rather than merge."""
    update = result["run_metrics"]
    assert isinstance(update, Overwrite)
    assert isinstance(update.value, RunMetrics)
    return update.value


def _subtasks(count: int) -> list[Subtask]:
    return [Subtask(id=f"s{i}", question=f"q{i}", rationale="r") for i in range(count)]


def _patch_planner(monkeypatch: pytest.MonkeyPatch, output: PlannerOutput) -> None:
    async def fake_structured(
        node: str,
        messages: object,
        schema: type[PlannerOutput],
        metrics: RunMetrics,
        temperature: float | None = None,
    ) -> PlannerOutput:
        return output

    monkeypatch.setattr(nodes, "call_structured", fake_structured)


async def test_returns_the_planned_subtasks(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_planner(monkeypatch, PlannerOutput(subtasks=_subtasks(3)))
    result = await nodes.planner(initial_state())

    assert len(result["subtasks"]) == 3
    assert all(isinstance(s, Subtask) for s in result["subtasks"])


async def test_truncates_to_max_subtasks(monkeypatch: pytest.MonkeyPatch) -> None:
    """A generous planner must not blow past the configured fan-out width."""
    _patch_planner(monkeypatch, PlannerOutput(subtasks=_subtasks(12)))
    result = await nodes.planner(initial_state())

    # MAX_SUBTASKS defaults to 5.
    assert len(result["subtasks"]) == 5


async def test_records_its_own_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_planner(monkeypatch, PlannerOutput(subtasks=_subtasks(2)))
    result = await nodes.planner(initial_state())

    metrics = _metrics(result)
    assert [t.node for t in metrics.node_timings] == [nodes.PLANNER]


async def test_failure_degrades_to_a_single_subtask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One bad structured-output parse must not take down the whole run.

    The planner sits ahead of the entire fan-out, so failing the request over it
    would turn a transient parse error into a total outage. Researching the
    original question as one unit is a far better outcome.
    """

    async def boom(
        node: str,
        messages: object,
        schema: type[PlannerOutput],
        metrics: RunMetrics,
        temperature: float | None = None,
    ) -> PlannerOutput:
        raise RuntimeError("structured output parse failed")

    monkeypatch.setattr(nodes, "call_structured", boom)
    state = initial_state("what is RAG?")
    result = await nodes.planner(state)

    subtasks = result["subtasks"]
    assert len(subtasks) == 1
    assert subtasks[0].question == "what is RAG?"


async def test_degradation_is_visible_in_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It must reach the client, not just the server log."""

    async def boom(
        node: str,
        messages: object,
        schema: type[PlannerOutput],
        metrics: RunMetrics,
        temperature: float | None = None,
    ) -> PlannerOutput:
        raise RuntimeError("nope")

    monkeypatch.setattr(nodes, "call_structured", boom)
    result = await nodes.planner(initial_state())

    assert DEGRADED_PLANNER_FALLBACK in _metrics(result).degraded


def test_json_encoded_subtask_list_is_accepted() -> None:
    """Observed live: the provider returns the nested list as a JSON string.

    Rejecting it would collapse a perfectly good multi-subtask plan into the
    single-subtask degraded path — losing the entire fan-out — while the correct
    content sat in the payload.
    """
    encoded = '[{"id": "s0", "question": "q", "rationale": "r"}]'
    output = PlannerOutput.model_validate({"subtasks": encoded})

    assert len(output.subtasks) == 1
    assert output.subtasks[0].id == "s0"


def test_a_real_list_still_works() -> None:
    output = PlannerOutput.model_validate(
        {"subtasks": [{"id": "s0", "question": "q", "rationale": "r"}]}
    )
    assert len(output.subtasks) == 1


def test_unparseable_string_still_fails_loudly() -> None:
    """A genuinely broken payload must not be silently swallowed."""
    with pytest.raises(ValueError):
        PlannerOutput.model_validate({"subtasks": "not json at all"})


async def test_empty_plan_is_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero subtasks is a legitimate outcome; fan_out routes it to synthesis."""
    _patch_planner(monkeypatch, PlannerOutput(subtasks=[]))
    result = await nodes.planner(initial_state())

    assert result["subtasks"] == []
    # Not a degradation — the planner succeeded, it just found nothing to split.
    assert DEGRADED_PLANNER_FALLBACK not in _metrics(result).degraded


# --- subtask identity -----------------------------------------------------------


async def test_subtask_ids_are_assigned_by_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model writes the questions; it does not get to choose the keys.

    ``subtask_id`` keys the upsert reducer, the ``Send`` payloads and the critic's
    verdicts. A model that repeats an id, or leaves one blank, would otherwise
    decide which branches survive the merge.
    """
    _patch_planner(
        monkeypatch,
        PlannerOutput(
            subtasks=[
                Subtask(id="a", question="q0", rationale="r"),
                Subtask(id="a", question="q1", rationale="r"),
                Subtask(id="", question="q2", rationale="r"),
            ]
        ),
    )
    result = await nodes.planner(initial_state())

    assert [(s.id, s.question) for s in result["subtasks"]] == [
        ("s0", "q0"),
        ("s1", "q1"),
        ("s2", "q2"),
    ]


async def test_the_degraded_plan_uses_the_same_id_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(
        node: str,
        messages: object,
        schema: type[PlannerOutput],
        metrics: RunMetrics,
        temperature: float | None = None,
    ) -> PlannerOutput:
        raise RuntimeError("nope")

    monkeypatch.setattr(nodes, "call_structured", boom)
    result = await nodes.planner(initial_state())

    assert [s.id for s in result["subtasks"]] == ["s0"]


async def test_a_repeated_model_id_does_not_lose_a_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two subtasks the model labelled "1" are still two branches with two results.

    Before ids were assigned here, both results arrived under ``subtask_id="1"``
    and the upsert reducer kept only the second: a planned branch vanished from
    the answer with nothing in ``degraded`` to say so.
    """

    async def fake_search(query: str, max_results: int | None = None) -> list[Source]:
        return [Source(url=f"https://ex.example/{query}", title="T", snippet="s")]

    plan = PlannerOutput(
        subtasks=[
            Subtask(id="1", question="q0", rationale="r"),
            Subtask(id="1", question="q1", rationale="r"),
        ]
    )
    monkeypatch.setattr(nodes, "search", fake_search)
    monkeypatch.setattr(nodes, "call_model", make_call_model("findings"))
    monkeypatch.setattr(nodes, "call_structured", make_call_structured(plan))

    graph = build_state_graph().compile(checkpointer=MemorySaver())
    result = await graph.ainvoke(initial_state(), config=run_config("repeated-ids"))

    assert sorted((r.subtask_id, r.sources[0].url) for r in result["research_results"]) == [
        ("s0", "https://ex.example/q0"),
        ("s1", "https://ex.example/q1"),
    ]
