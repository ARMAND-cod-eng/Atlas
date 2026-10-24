"""A provision's own words, by the label a citation uses.

What a citation chip in the answer opens: hover "[GDPR Art. 33(1)]" and read
what Article 33(1) actually says, with a link to the Official Journal.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from atlas.api.main import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def test_a_label_returns_the_provisions_text_and_where_it_is_published() -> None:
    response = _client().get("/library/provision", params={"label": "GDPR Art. 33(1)"})

    assert response.status_code == 200
    body = response.json()
    assert (body["label"], body["act"], body["title"]) == (
        "GDPR Art. 33(1)",
        "GDPR",
        "Notification of a personal data breach to the supervisory authority",
    )
    assert body["text"].startswith("In the case of a personal data breach, the controller shall")
    assert body["url"] == "http://data.europa.eu/eli/reg/2016/679/oj#art_33"


def test_a_citation_spelled_another_way_is_read_as_the_library_spells_it() -> None:
    response = _client().get("/library/provision", params={"label": "GDPR Article 33(1)"})

    assert response.json()["label"] == "GDPR Art. 33(1)"


def test_a_provision_that_does_not_exist_is_404() -> None:
    response = _client().get("/library/provision", params={"label": "AI Act Art. 114"})

    assert response.status_code == 404
    assert response.json() == {"detail": "no such provision: AI Act Art. 114"}


def test_something_that_is_not_a_citation_is_404() -> None:
    assert _client().get("/library/provision", params={"label": "hello"}).status_code == 404
