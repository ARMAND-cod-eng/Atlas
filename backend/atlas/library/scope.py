"""Whether a question is one for the regulation library: the AI Act or the GDPR.

The judgment is the planner's, made in the plan it writes anyway (see
``atlas.graph.nodes.planner``), because word overlap cannot make it. Measured
before this was built, a core GDPR question ("we had a data breach, who do we
have to tell and how fast?") shared less of its words with the law than "How do
I file my income taxes in the United States?" did, and the best ranking score of
an off-topic question overlapped the in-scope ones by half their range.

What can be decided without a model is here:

* ``names_the_law``: a question that names an act, cites a provision or asks
  what a term the acts define means is about the law, whatever the planner
  judged. It holds the planner to account in the one direction that matters
  most: turning away a question the product exists to answer.
* ``closest_match``: offline mode's stand-in for the planner's judgment, which
  has no model to make it (see ``OFFLINE_FLOOR``).

``atlas_eval scope`` measures both, and the planner's judgment, on labelled
in-scope and off-topic questions.
"""

from __future__ import annotations

from atlas.library.search import names_an_act
from atlas.library.sources import library_search

# Below this best ranking score, offline mode says a question is about neither
# act. The lowest any in-scope question scored in the measured sets was 8.2
# ("what are the obligations of a deployer?"); the floor leaves room under it, so
# offline mode turns away only questions the law has nothing near: "history of
# cameroon" scores 4.4. Offline mode only. With models, the planner judges.
OFFLINE_FLOOR = 6.0

OFFLINE_REASON = "Offline mode: nothing in the AI Act or the GDPR comes close to this question."


def names_the_law(query: str) -> bool:
    """Whether the question names an act, cites a provision, or asks what a defined term means."""
    if names_an_act(query):
        return True
    return any(hit.reason != "match" for hit in library_search().search(query))


def declined_reason(out_of_scope: str | None, query: str) -> str | None:
    """The planner's judgment as a run acts on it: its reason, unless the question names the law."""
    reason = (out_of_scope or "").strip()
    if not reason or names_the_law(query):
        return None
    return reason


def closest_match(query: str) -> float:
    """The best ranking score any passage of the law gets for the question; 0 for none."""
    return max(
        (hit.score for hit in library_search().search(query) if hit.reason == "match"), default=0.0
    )


def declined_answer(reason: str) -> str:
    """What a declined run answers, for any client that reads only the answer."""
    return (
        "Not researched. Atlas answers questions about the EU AI Act and the GDPR, and this "
        f"one was judged to be about neither. {reason.strip()} To research it anyway, ask it "
        "again in this conversation."
    )
