"""The HTTP surface of a contract review: send a contract, get its findings.

``POST /reviews`` takes pasted text or a file, base64 in JSON, and refuses one it
cannot read before anything is queued or spent. ``GET /reviews/{id}`` serves the
finished review. The queue is the inline FakeArq running the real review graph,
with the reviewer's model scripted beneath the model layer.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from atlas import eventlog, models
from atlas.api.main import create_app
from atlas.domains import DomainName
from atlas.offline import no_fallback, scripted_structured
from atlas.review.graph import build_review_graph
from atlas.review.judge import RequirementJudgement
from atlas.runner import execute_review
from atlas.store import InMemoryRunStore, RunRecord
from atlas.worker import RUN_REVIEW
from tests.conftest import FakeArq, FakeRedis
from tests.test_review_contract import tiny_docx

CONTRACT = """1. Instructions
The Processor shall process personal data only on documented instructions from the Controller.
"""


@pytest.fixture(autouse=True)
def scripted_reviewer(monkeypatch: pytest.MonkeyPatch) -> None:
    async def judge(node: str, messages: list[Any], schema: type[Any], metrics: Any) -> Any:
        return RequirementJudgement(status="missing", reason="Not addressed.")

    monkeypatch.setattr(models, "get_structured_llm", scripted_structured(judge))
    monkeypatch.setattr(models, "get_structured_fallback", no_fallback)


class Harness:
    def __init__(self) -> None:
        graph = build_review_graph()
        self.store = InMemoryRunStore()

        async def research(*args: object, **kwargs: object) -> None:
            raise AssertionError("a review queued a research run")

        async def review(
            run_id: str,
            thread_id: str,
            filename: str,
            text: str,
            *,
            domain: DomainName | None = None,
        ) -> dict[str, Any]:
            return await execute_review(
                run_id, thread_id, graph, filename=filename, text=text, store=self.store
            )

        self.arq = FakeArq(research, {RUN_REVIEW: review})
        app = create_app()
        app.state.arq = self.arq
        app.state.store = self.store
        self.client = TestClient(app)


def test_a_pasted_contract_is_reviewed(fake_redis: FakeRedis) -> None:
    harness = Harness()

    accepted = harness.client.post("/reviews", json={"filename": "dpa.txt", "text": CONTRACT})

    assert accepted.status_code == 202, accepted.text
    run_id = accepted.json()["run_id"]
    assert accepted.json()["events_url"] == f"/runs/{run_id}/events"
    [(function, queued_run, _thread, filename, text)] = harness.arq.calls
    assert (function, queued_run, filename, text) == (
        RUN_REVIEW,
        run_id,
        "dpa.txt",
        CONTRACT.strip(),
    )

    review = harness.client.get(f"/reviews/{run_id}").json()
    assert review["status"] == "completed"
    assert [c["label"] for c in review["result"]["clauses"]] == ["Clause 1"]
    assert len(review["result"]["findings"]) == 9


def test_a_file_is_read_before_it_is_queued(fake_redis: FakeRedis) -> None:
    harness = Harness()
    content = base64.b64encode(tiny_docx(["1. Instructions", "Only on documented instructions."]))

    accepted = harness.client.post(
        "/reviews", json={"filename": "dpa.docx", "content_base64": content.decode()}
    )

    assert accepted.status_code == 202, accepted.text
    assert harness.arq.calls[0][4] == "1. Instructions\nOnly on documented instructions."


@pytest.mark.parametrize(
    ("body", "detail"),
    [
        (
            {"filename": "dpa.pages", "content_base64": "YQ=="},
            "Atlas can read .txt, .md, .docx and .pdf, not .pages.",
        ),
        ({"filename": "dpa.txt", "content_base64": "not base64!"}, "dpa.txt could not be decoded."),
        # Junk a lax decoder would drop, decoding to a different file.
        ({"filename": "dpa.txt", "content_base64": "YQ==!!"}, "dpa.txt could not be decoded."),
        ({"filename": "dpa.txt"}, "Send the contract's text or its file, one of the two."),
        (
            {"filename": "dpa.txt", "text": "x", "content_base64": "YQ=="},
            "Send the contract's text or its file, one of the two.",
        ),
        ({"filename": "scan.pdf", "text": "   "}, "scan.pdf has no text in it."),
    ],
    ids=["unsupported", "not-base64", "junk-in-base64", "neither", "both", "empty"],
)
def test_a_contract_that_cannot_be_read_is_refused_and_not_queued(
    fake_redis: FakeRedis, body: dict[str, str], detail: str
) -> None:
    harness = Harness()

    response = harness.client.post("/reviews", json=body)

    assert response.status_code == 422
    assert response.json() == {"detail": detail}
    assert harness.arq.calls == []


async def test_a_research_run_is_not_a_review(fake_redis: FakeRedis) -> None:
    harness = Harness()
    # A research run with a result, both where a live run keeps it and in the record.
    await eventlog.store_result("research-1", {"answer": "Yes.", "thread_id": "t"})
    await harness.store.started(
        RunRecord(
            run_id="research-2",
            thread_id="t",
            kind="question",
            query="q",
            status="completed",
            result={"answer": "Yes.", "thread_id": "t"},
            created_at=datetime.now(UTC),
        )
    )

    assert harness.client.get("/reviews/research-1").status_code == 404
    assert harness.client.get("/reviews/research-2").status_code == 404
    assert harness.client.get("/reviews/unknown-run").status_code == 404
