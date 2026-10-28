"""Offline mode: the real product with the models and the web scripted.

Run here through the real graph, runner, library, citation checks and store,
as the worker runs it with ATLAS_OFFLINE=true. What is tested is that the
product works end to end without a key, including a revision, and that every
answer and every call says it was scripted.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver

from atlas import eventlog, models, worker
from atlas.config import Settings, get_settings
from atlas.domains import using_domain
from atlas.events import CITATION_SUPPORT, RUN_COMPLETED
from atlas.graph import nodes
from atlas.graph.builder import build_state_graph, checkpoint_serde
from atlas.library import load_library
from atlas.library.citations import check_citations
from atlas.offline import (
    NOTICE,
    no_fallback,
    no_web,
    offline_model,
    offline_structured,
    plan_of,
    scripted,
)
from atlas.result import QueryResponse
from atlas.runner import execute_revision, execute_run
from atlas.store import InMemoryRunStore

QUESTION = "Is a CV-screening tool high-risk under the AI Act, and what must its provider do?"


def test_the_plan_is_the_questions_own_clauses() -> None:
    assert [s.question for s in plan_of(QUESTION).subtasks] == [
        "Is a CV-screening tool high-risk under the AI Act?",
        "What must its provider do?",
    ]
    assert [s.question for s in plan_of("When must a breach be notified?").subtasks] == [
        "When must a breach be notified?"
    ]


async def _ask(store: InMemoryRunStore) -> tuple[object, list[object]]:
    graph = build_state_graph().compile(checkpointer=MemorySaver(serde=checkpoint_serde()))
    with using_domain("compliance"), scripted(offline_model, offline_structured):
        await execute_run("run-1", QUESTION, "t1", graph, store=store)
    return graph, await eventlog.replay("run-1", 0)


async def test_a_question_is_answered_end_to_end_with_no_key() -> None:
    store = InMemoryRunStore()

    _, events = await _ask(store)

    [done] = [e for e in events if e.type == RUN_COMPLETED]
    answer = done.payload["answer"]
    assert answer.startswith(NOTICE)
    assert done.payload["degraded"] == []
    record = await store.get("run-1")
    assert record is not None and record.result is not None
    # Real citations of retrieved law: the check against the law passes them.
    research = QueryResponse.model_validate(record.result).research_results
    report = check_citations(answer, research, load_library())
    assert (report.ok, len(report.cited) > 0) == (True, True)
    # The support check judges nothing offline: a scripted "yes" would be a
    # judgement nobody made. Every citation is left unchecked.
    [support, *_] = [e for e in events if e.type == CITATION_SUPPORT]
    assert (support.payload["supported"], support.payload["unchecked"]) == ([], len(report.cited))


async def test_every_call_says_it_was_scripted() -> None:
    store = InMemoryRunStore()

    await _ask(store)

    calls = await store.calls("run-1")
    assert {c.name for c in calls if c.kind == "model"} == {"scripted"}
    assert {c.name for c in calls if c.kind == "tool"} == {"library", "web"}
    assert all(c.response == [] for c in calls if c.name == "web")


async def test_a_rewritten_branch_reads_different_law_and_changes_the_answer() -> None:
    store = InMemoryRunStore()
    graph, events = await _ask(store)
    [first] = [e for e in events if e.type == RUN_COMPLETED]

    with using_domain("compliance"), scripted(offline_model, offline_structured):
        await execute_revision(
            "run-2",
            "t1",
            graph,
            parent_run_id="run-1",
            subtask_id="s1",
            question="What must a deployer of a high-risk AI system do?",
            store=store,
        )

    [second] = [e for e in await eventlog.replay("run-2", 0) if e.type == RUN_COMPLETED]
    searched = [(c.subtask_id, c.name) for c in await store.calls("run-2") if c.kind == "tool"]
    assert searched == [("s1", "library"), ("s1", "web")]
    assert second.payload["answer"] != first.payload["answer"]
    assert "AI Act Art. 26" in second.payload["answer"]


def test_keys_are_required_unless_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TAVILY_API_KEY", "ATLAS_OFFLINE"):
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="ATLAS_OFFLINE=true"):
            get_settings()

        monkeypatch.setenv("ATLAS_OFFLINE", "true")
        get_settings.cache_clear()
        assert (get_settings().atlas_offline, get_settings().anthropic_api_key) == (True, "")
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize("offline", [True, False])
async def test_the_worker_scripts_every_model_only_in_offline_mode(
    offline: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Store:
        def __init__(self, engine: object) -> None: ...

        async def setup(self) -> None: ...

    async def graph() -> object:
        return object()

    monkeypatch.setattr(worker, "build_graph", graph)
    monkeypatch.setattr(worker, "PostgresRunStore", Store)
    monkeypatch.setattr(worker, "get_engine", lambda: None)
    monkeypatch.setenv("ATLAS_OFFLINE", "true" if offline else "false")
    # Whatever startup installs, the test puts back.
    for module, name in (
        (models, "get_llm"),
        (models, "get_structured_llm"),
        (models, "get_structured_fallback"),
        (nodes, "search"),
    ):
        monkeypatch.setattr(module, name, getattr(module, name))
    real_search = nodes.search
    get_settings.cache_clear()
    try:
        await worker.startup({})
    finally:
        get_settings.cache_clear()

    assert (nodes.search is no_web, models.get_structured_fallback is no_fallback) == (
        offline,
        offline,
    )
    assert (nodes.search is real_search) == (not offline)
