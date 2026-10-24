"""In the compliance domain, an answer with a bad citation does not get through.

The critic is a model, and a model can approve an answer that cites an article
that does not exist. So after its review, the compliance critic also checks the
answer's citations against the law (atlas.library.citations). A failed check
sends the answer back for a rewrite with the problems spelled out, whatever the
model said. Out of rounds, the answer goes out marked: unverified_citation, and
an "Unverified" note naming what is wrong.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver

from atlas.events import CITATION_CHECK
from atlas.graph import nodes
from atlas.graph.builder import build_state_graph, checkpoint_serde
from atlas.graph.state import PlannerOutput, Source, Subtask
from atlas.metrics import (
    DEGRADED_CRITIC_MAX_ITERATIONS,
    DEGRADED_UNVERIFIED_CITATION,
    RunMetrics,
)
from tests.conftest import initial_state, make_call_structured

QUESTION = "What is the deadline to notify a personal data breach under the GDPR?"
GOOD = "Notify within 72 hours [GDPR Art. 33(1)]."
INVENTED = "Notify within 72 hours [GDPR Art. 33(1)], and register it [AI Act Art. 114]."


class Synthesis:
    """A synthesizer that writes the given drafts in turn; records its prompts."""

    def __init__(self, drafts: list[str]) -> None:
        self.drafts = drafts
        self.prompts: list[str] = []


def _install(monkeypatch: pytest.MonkeyPatch, drafts: list[str]) -> Synthesis:
    synthesis = Synthesis(drafts)

    async def no_web(query: str, max_results: int | None = None) -> list[Source]:
        return []

    async def fake_model(
        node: str,
        messages: list[BaseMessage],
        metrics: RunMetrics,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AIMessage:
        if node != nodes.SYNTHESIZER:
            return AIMessage(content="findings [GDPR Art. 33(1)]")
        synthesis.prompts.append(str(messages[-1].content))
        index = min(len(synthesis.prompts), len(synthesis.drafts)) - 1
        return AIMessage(content=synthesis.drafts[index])

    plan = PlannerOutput(subtasks=[Subtask(id="s0", question=QUESTION, rationale="r")])
    monkeypatch.setattr(nodes, "search", no_web)
    monkeypatch.setattr(nodes, "call_model", fake_model)
    # The model critic approves everything it sees.
    monkeypatch.setattr(nodes, "call_structured", make_call_structured(plan))
    return synthesis


async def _run(domain: str = "compliance") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    graph = build_state_graph().compile(checkpointer=MemorySaver(serde=checkpoint_serde()))
    config = {"configurable": {"thread_id": "t", "run_id": "r", "domain": domain}}
    final: dict[str, Any] = {}
    async for mode, chunk in graph.astream(
        initial_state(QUESTION), config=config, stream_mode=["custom", "values"]
    ):
        if mode == "custom" and chunk["type"] == CITATION_CHECK:
            events.append(dict(chunk))
        elif mode == "values":
            final = dict(chunk)
    return final, events


async def test_an_invented_citation_is_sent_back_even_when_the_critic_approves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthesis = _install(monkeypatch, [INVENTED, GOOD])

    final, _ = await _run()

    rounds = [(r.iteration, r.overall) for r in final["run_metrics"].critic_rounds]
    assert rounds == [(1, "revise"), (2, "accept")]
    assert final["answer"] == GOOD
    assert final["run_metrics"].degraded == []
    # The rewrite was told exactly what was wrong.
    assert "[AI Act Art. 114] is not a provision of that act" in synthesis.prompts[1]


async def test_each_draft_reports_its_citation_check(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, [INVENTED, GOOD])

    _, events = await _run()

    assert [(e["cited"], e["unresolved"], e["ok"]) for e in events] == [
        (["GDPR Art. 33(1)", "AI Act Art. 114"], ["AI Act Art. 114"], False),
        (["GDPR Art. 33(1)"], [], True),
    ]


async def test_out_of_rounds_the_answer_goes_out_marked(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, [INVENTED])

    final, _ = await _run()

    assert final["run_metrics"].degraded == [
        DEGRADED_UNVERIFIED_CITATION,
        DEGRADED_CRITIC_MAX_ITERATIONS,
    ]
    assert "[AI Act Art. 114] is not a provision of that act" in final["answer"]
    assert "**Unverified:**" in final["answer"]


async def test_the_general_domain_is_not_held_to_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """The general baseline has no library, and its behaviour must not move."""
    _install(monkeypatch, [INVENTED])

    final, events = await _run("general")

    assert [(r.iteration, r.overall) for r in final["run_metrics"].critic_rounds] == [(1, "accept")]
    assert events == []
