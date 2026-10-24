"""The compliance domain: researchers read the law first, and cite it by label.

The general domain is Atlas as the first baseline measured it, and must stay
that way prompt for prompt. The compliance domain changes what each stage is
told and where researchers look: the regulation library before the web, whose
results stay beside the law as secondary sources, never in its place.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

from atlas import prompts
from atlas.domains import COMPLIANCE, GENERAL, current_domain
from atlas.graph import nodes
from atlas.graph.builder import build_state_graph, checkpoint_serde
from atlas.graph.state import PlannerOutput, Source, Subtask
from atlas.metrics import DEGRADED_WEB_SEARCH_FAILED, RunMetrics
from tests.conftest import initial_state, make_call_structured

# sha256 of the four general prompts, recorded before they moved out of
# atlas.graph.nodes: the baseline was scored with exactly these.
GENERAL_PROMPTS_DIGEST = "1bb485526b585e8ba16042cdbe15255cbebeca1dd0a54b1559f1ab3f535ba8ce"

BREACH = "What is the deadline to notify a personal data breach under the GDPR?"


def test_the_general_prompts_are_the_baselines_byte_for_byte() -> None:
    joined = "\x00".join(
        [
            prompts.GENERAL_PLANNER,
            prompts.GENERAL_RESEARCHER,
            prompts.GENERAL_SYNTHESIZER,
            prompts.GENERAL_CRITIC,
        ]
    )

    assert hashlib.sha256(joined.encode()).hexdigest() == GENERAL_PROMPTS_DIGEST


def test_outside_a_graph_the_configured_domain_applies() -> None:
    """The test environment sets ATLAS_DOMAIN=general (tests/conftest.py)."""
    assert current_domain() is GENERAL


class Seen:
    def __init__(self) -> None:
        self.web: list[str] = []
        self.prompts: dict[str, list[list[BaseMessage]]] = {}


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch) -> Seen:
    record = Seen()

    async def fake_search(query: str, max_results: int | None = None) -> list[Source]:
        record.web.append(query)
        return [
            Source(url="https://www.edpb.europa.eu/guidelines/breach", title="EDPB", snippet="g"),
            Source(url="https://blog.example.com/gdpr-breach", title="Blog", snippet="b"),
        ]

    async def fake_model(
        node: str,
        messages: list[BaseMessage],
        metrics: RunMetrics,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AIMessage:
        record.prompts.setdefault(node, []).append(list(messages))
        return AIMessage(content=f"{node} output")

    plan = PlannerOutput(subtasks=[Subtask(id="s0", question=BREACH, rationale="r")])
    monkeypatch.setattr(nodes, "search", fake_search)
    monkeypatch.setattr(nodes, "call_model", fake_model)
    monkeypatch.setattr(nodes, "call_structured", make_call_structured(plan))
    return record


async def _run(domain: str) -> dict[str, Any]:
    graph = build_state_graph().compile(checkpointer=MemorySaver(serde=checkpoint_serde()))
    config = {"configurable": {"thread_id": "t", "run_id": "r", "domain": domain}}
    return dict(await graph.ainvoke(initial_state(BREACH), config=config))


async def test_a_compliance_researcher_reads_the_law_before_the_web(seen: Seen) -> None:
    final = await _run("compliance")
    [result] = final["research_results"]

    law = [s for s in result.sources if s.authority == "primary_law"]
    web = [s for s in result.sources if s.authority != "primary_law"]
    assert law[: len(law)] == result.sources[: len(law)]
    assert "GDPR Art. 33(1)" in [s.label for s in law]
    assert all(s.url.startswith("http://data.europa.eu/eli/reg/2016/679/oj#art_") for s in law)
    assert [(s.url, s.authority) for s in web] == [
        ("https://www.edpb.europa.eu/guidelines/breach", "official_guidance"),
        ("https://blog.example.com/gdpr-breach", "secondary"),
    ]


async def test_the_researcher_is_shown_the_law_apart_from_the_guidance(seen: Seen) -> None:
    await _run("compliance")

    [messages] = seen.prompts[nodes.RESEARCHER]
    system, prompt = str(messages[0].content), str(messages[-1].content)
    assert system == COMPLIANCE.researcher
    assert prompt.index("PRIMARY LAW") < prompt.index("[GDPR Art. 33(1)]")
    assert prompt.index("[GDPR Art. 33(1)]") < prompt.index("SECONDARY SOURCES")
    assert "without undue delay and, where feasible, not later than 72 hours" in prompt


async def test_every_stage_gets_the_compliance_prompt(seen: Seen) -> None:
    await _run("compliance")

    systems = {
        node: str(calls[0][0].content)
        for node, calls in seen.prompts.items()
        if isinstance(calls[0][0], SystemMessage)
    }
    assert systems == {
        nodes.RESEARCHER: COMPLIANCE.researcher,
        nodes.SYNTHESIZER: COMPLIANCE.synthesizer,
    }


async def test_the_synthesizer_sees_provisions_by_label(seen: Seen) -> None:
    await _run("compliance")

    [messages] = seen.prompts[nodes.SYNTHESIZER]
    assert "- [GDPR Art. 33(1)]" in str(messages[-1].content)


async def test_the_general_domain_never_touches_the_library(seen: Seen) -> None:
    final = await _run("general")
    [result] = final["research_results"]

    assert [s.authority for s in result.sources] == ["secondary", "secondary"]
    [messages] = seen.prompts[nodes.RESEARCHER]
    assert str(messages[0].content) == GENERAL.researcher


async def test_a_failed_web_search_does_not_cost_a_branch_its_law(
    seen: Seen, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def down(query: str, max_results: int | None = None) -> list[Source]:
        raise ConnectionError("tavily is down")

    monkeypatch.setattr(nodes, "search", down)

    final = await _run("compliance")
    [result] = final["research_results"]

    assert result.error is None
    assert {s.authority for s in result.sources} == {"primary_law"}
    assert final["run_metrics"].degraded == [f"{DEGRADED_WEB_SEARCH_FAILED}:s0"]
