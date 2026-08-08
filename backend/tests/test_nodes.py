"""Node behaviour, with the model stubbed out."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from atlas.graph import nodes
from atlas.graph.state import AtlasState


class RecordingModel:
    """Captures what the node sends, and returns a canned reply."""

    def __init__(self) -> None:
        self.seen: list[BaseMessage] = []

    async def ainvoke(self, messages: Any, *args: Any, **kwargs: Any) -> AIMessage:
        self.seen = list(messages)
        return AIMessage(content="stub answer")


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> RecordingModel:
    model = RecordingModel()
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: model)
    return model


async def test_answer_is_written_to_state(recorder: RecordingModel) -> None:
    state: AtlasState = {"query": "hello", "answer": "", "messages": []}
    result = await nodes.answer_query(state)
    assert result["answer"] == "stub answer"


async def test_question_and_reply_are_appended(recorder: RecordingModel) -> None:
    state: AtlasState = {"query": "hello", "answer": "", "messages": []}
    result = await nodes.answer_query(state)
    assert len(result["messages"]) == 2


async def test_prior_history_is_sent_to_the_model(recorder: RecordingModel) -> None:
    """Regression: the checkpointer's history must actually reach the model.

    Sending only the new question makes persisted history write-only — stored
    every turn, read never — so a follow-up on the same thread_id answers as if
    the conversation had just begun, even though the checkpoint is correct.
    """
    state: AtlasState = {
        "query": "and what did I just ask?",
        "answer": "",
        "messages": [HumanMessage(content="what is RAG?"), AIMessage(content="it is...")],
    }
    await nodes.answer_query(state)

    assert len(recorder.seen) == 3, "expected 2 prior messages plus the new question"
    assert recorder.seen[0].text == "what is RAG?"
    assert recorder.seen[-1].text == "and what did I just ask?"
