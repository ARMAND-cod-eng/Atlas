"""A reviewer signs off a review, finding by finding: agree or disagree, with a note.

Atlas's finding and the person's decision are kept side by side, and a changed
decision keeps the one it replaced, so the review says who decided what, and
when. A decision is for a finished review and one of its requirements; it goes
when its conversation is deleted.

The reviewer is named by the reviewer: Atlas has no sign-in, and the record
says so by keeping the name as given.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from atlas import models
from atlas.api.main import create_app
from atlas.domains import DomainName
from atlas.offline import no_fallback, scripted_structured
from atlas.retention import forget_thread
from atlas.review.graph import build_review_graph
from atlas.review.judge import RequirementJudgement
from atlas.runner import execute_review
from atlas.store import InMemoryRunStore, RunRecord
from atlas.worker import FORGET_THREAD, RUN_REVIEW
from tests.conftest import FakeArq, FakeRedis

CONTRACT = (
    "1. Instructions\nThe Processor acts only on documented instructions from the Controller."
)


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
        saver = MemorySaver()

        async def research(*args: object, **kwargs: object) -> None:
            raise AssertionError("no research here")

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

        async def forget(thread_id: str, **kwargs: object) -> list[str]:
            return await forget_thread(thread_id, self.store, saver)

        app = create_app()
        app.state.arq = FakeArq(research, {RUN_REVIEW: review, FORGET_THREAD: forget})
        app.state.store = self.store
        self.client = TestClient(app)

    def review(self) -> dict[str, Any]:
        accepted = self.client.post("/reviews", json={"filename": "dpa.txt", "text": CONTRACT})
        assert accepted.status_code == 202, accepted.text
        return dict(accepted.json())

    def decide(self, run_id: str, requirement: str, **body: object) -> Any:
        return self.client.put(f"/reviews/{run_id}/decisions/{requirement}", json=body)


def test_a_decision_is_kept_and_a_changed_one_keeps_its_history(fake_redis: FakeRedis) -> None:
    harness = Harness()
    run_id = harness.review()["run_id"]

    agreed = harness.decide(run_id, "audits", decision="agree", reviewer="A. Reviewer")
    changed = harness.decide(
        run_id, "audits", decision="disagree", note="Clause 9 covers audits.", reviewer="B. Counsel"
    )

    assert (agreed.status_code, changed.status_code) == (200, 200)
    body = harness.client.get(f"/reviews/{run_id}/decisions").json()
    assert [
        (d["requirement_id"], d["decision"], d["note"], d["reviewer"]) for d in body["current"]
    ] == [("audits", "disagree", "Clause 9 covers audits.", "B. Counsel")]
    assert [(d["decision"], d["reviewer"]) for d in body["history"]] == [
        ("agree", "A. Reviewer"),
        ("disagree", "B. Counsel"),
    ]


@pytest.mark.parametrize(
    ("requirement", "body", "status", "detail"),
    [
        (
            "not-a-requirement",
            {"decision": "agree", "reviewer": "A"},
            404,
            "this review has no requirement not-a-requirement",
        ),
        ("audits", {"decision": "maybe", "reviewer": "A"}, 422, None),
        ("audits", {"decision": "agree", "reviewer": ""}, 422, None),
    ],
    ids=["unknown-requirement", "not-a-decision", "no-reviewer"],
)
def test_a_decision_that_is_not_one_is_refused(
    fake_redis: FakeRedis, requirement: str, body: dict[str, str], status: int, detail: str | None
) -> None:
    harness = Harness()
    run_id = harness.review()["run_id"]

    response = harness.decide(run_id, requirement, **body)

    assert response.status_code == status
    if detail is not None:
        assert response.json() == {"detail": detail}
    assert harness.client.get(f"/reviews/{run_id}/decisions").json() == {
        "current": [],
        "history": [],
    }


def test_there_is_nothing_to_decide_on_a_review_that_does_not_exist(fake_redis: FakeRedis) -> None:
    harness = Harness()

    response = harness.decide("no-such-review", "audits", decision="agree", reviewer="A")

    assert response.status_code == 404
    assert response.json() == {"detail": "unknown review: no-such-review"}


async def test_a_review_still_running_cannot_be_signed_off(fake_redis: FakeRedis) -> None:
    harness = Harness()
    await harness.store.started(
        RunRecord(
            run_id="review-running",
            thread_id="t",
            kind="review",
            query="dpa.txt",
            created_at=datetime.now(UTC),
        )
    )

    response = harness.decide("review-running", "audits", decision="agree", reviewer="A")

    assert response.status_code == 409
    assert response.json() == {"detail": "review review-running is not finished"}


def test_deleting_the_conversation_deletes_its_decisions(fake_redis: FakeRedis) -> None:
    harness = Harness()
    review = harness.review()
    harness.decide(review["run_id"], "audits", decision="agree", reviewer="A")

    assert harness.client.delete(f"/threads/{review['thread_id']}").status_code == 202

    assert harness.client.get(f"/reviews/{review['run_id']}/decisions").status_code == 404
