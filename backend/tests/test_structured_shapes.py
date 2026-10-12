"""The shapes Claude's structured output actually arrives in, not just the documented one.

The first live eval runs handed five of eight structured answers to the
fallback. None of them was wrong — each was a correct plan or verdict in an
extra wrapper that strict validation rejected:

* the planner's ``subtasks`` as a JSON string holding the whole object,
  ``{"subtasks": "{\\"subtasks\\": [...]}"}`` — the earlier tolerance only
  decoded a string holding the *list*;
* the critic's verdict nested under a single key,
  ``{"$PARAMETER_VALUE": {"overall": ..., ...}}``.

Each cost a GPT-4o call and a ``structured_output_fallback`` mark, and on the
critic it meant a different model did the reviewing. The schemas now unwrap
exactly these shapes. The controls matter as much: a payload that is genuinely
missing its content must still fail, or tolerance becomes acceptance of
anything.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from atlas.graph.state import CriticVerdict, PlannerOutput, Subtask, SubtaskVerdict

PLAN = [{"id": "1", "question": "What is retrieval-augmented generation?", "rationale": "r"}]
VERDICT = {
    "overall": "accept",
    "per_subtask": [{"subtask_id": "s0", "verdict": "accept", "reason": "r", "guidance": ""}],
    "synthesis_feedback": None,
}


# --- the planner -------------------------------------------------------------------------


def test_a_subtasks_string_holding_the_whole_object_is_unwrapped() -> None:
    """Observed live (q01): the string is the object, not the list."""
    payload = {"subtasks": json.dumps({"subtasks": PLAN})}

    plan = PlannerOutput.model_validate(payload)

    assert plan.subtasks == [Subtask(**PLAN[0])]


def test_the_plan_nested_under_a_single_key_is_unwrapped() -> None:
    plan = PlannerOutput.model_validate({"$PARAMETER_VALUE": {"subtasks": PLAN}})

    assert plan.subtasks == [Subtask(**PLAN[0])]


def test_a_string_holding_some_other_object_still_fails() -> None:
    with pytest.raises(ValidationError):
        PlannerOutput.model_validate({"subtasks": json.dumps({"tasks": PLAN})})


# --- the critic --------------------------------------------------------------------------


def test_a_verdict_nested_under_a_single_key_is_unwrapped() -> None:
    """Observed live (q05): the whole verdict under "$PARAMETER_VALUE"."""
    verdict = CriticVerdict.model_validate({"$PARAMETER_VALUE": VERDICT})

    assert verdict == CriticVerdict(
        overall="accept",
        per_subtask=[SubtaskVerdict(subtask_id="s0", verdict="accept", reason="r", guidance="")],
    )


def test_per_subtask_as_a_json_string_is_decoded() -> None:
    """The planner's original wobble, on the critic's list — the same failure,
    one schema over."""
    verdict = CriticVerdict.model_validate(
        {**VERDICT, "per_subtask": json.dumps(VERDICT["per_subtask"])}
    )

    assert [v.subtask_id for v in verdict.per_subtask] == ["s0"]


@pytest.mark.parametrize(
    "payload",
    [
        {"$PARAMETER_VALUE": "accept"},  # wrapped, but nothing inside to unwrap
        {"verdict": VERDICT, "notes": "two keys is not a wrapper"},
        {"per_subtask": []},  # overall genuinely missing
    ],
)
def test_a_verdict_missing_its_content_still_fails(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        CriticVerdict.model_validate(payload)
