"""Factual grounding: is every claim supported by the retrieved sources?

**This is not the same check as ``atlas.graph.nodes.ungrounded_urls``, and
conflating them would overstate what Atlas measures.**

``ungrounded_urls`` (Phase 1) compares the URLs cited in an answer against the
URLs actually retrieved. It is a *syntactic citation* check: it catches a
fabricated link. It cannot catch an answer that cites a genuine retrieved URL and
then asserts something that URL does not say — which is the more common and more
dangerous failure, because the citation looks checkable and is.

This scorer reads the snippet *text*. It decomposes the answer into atomic claims
and asks, for each, whether the retrieved context supports it. Two judge calls per
run: one to decompose, one to verify.

Both checks are worth having and they measure different things. The README says so
in those words, because "factual grounding" is the phrase a reader is most likely
to over-read.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from atlas_eval.scorers.judge import judge

DECOMPOSE_SYSTEM = """You split an answer into atomic factual claims.

An atomic claim is one assertion that could independently be true or false. Split \
compound sentences. Keep each claim self-contained — resolve pronouns and \
references so the claim stands alone.

EXCLUDE, because they are not factual assertions about the world:
- hedges and meta-commentary ("the sources disagree", "this is contested")
- statements about what the research did or failed to find
- pure restatements of the question
- recommendations and opinions ("you should prefer X")

Return at most 12 claims, the most substantive ones. If the answer makes no \
factual assertions at all, return an empty list.
"""

VERIFY_SYSTEM = """You check whether retrieved source material supports a claim.

You are given CONTEXT (excerpts actually retrieved by a research system) and a \
list of CLAIMS taken from an answer built on that context.

For each claim, decide:
- `supported`  : the context states this, or states something that plainly entails it
- `unsupported`: the context does not contain this, or contradicts it

Judge ONLY against the context given. Do not use your own knowledge — a claim that \
is true in the world but absent from the context is `unsupported`, because that is \
precisely the failure being measured: an answer asserting more than its sources do.

Be strict about specifics. If the claim gives a number, a date or a name that the \
context does not, it is unsupported even when the surrounding gist matches.
"""


class ClaimList(BaseModel):
    claims: list[str] = Field(default_factory=list)


class ClaimVerdict(BaseModel):
    claim: str
    supported: bool
    reason: str = ""


class VerdictList(BaseModel):
    verdicts: list[ClaimVerdict] = Field(default_factory=list)


class GroundingScore(BaseModel):
    score: float
    claims_total: int = 0
    claims_supported: int = 0
    verdicts: list[ClaimVerdict] = Field(default_factory=list)
    # Set when there was nothing to score. Distinguished from 0.0, which means
    # "claims were made and none held up" — a very different result.
    not_applicable: bool = False


async def decompose(answer: str) -> list[str]:
    if not answer.strip():
        return []
    result = await judge(ClaimList, DECOMPOSE_SYSTEM, f"ANSWER:\n{answer}")
    return [c.strip() for c in result.claims if c.strip()]


async def verify(claims: list[str], context: str) -> list[ClaimVerdict]:
    if not claims:
        return []
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    result = await judge(
        VerdictList,
        VERIFY_SYSTEM,
        f"CONTEXT:\n{context or '(no sources were retrieved)'}\n\nCLAIMS:\n{numbered}",
    )
    # The judge may return fewer verdicts than claims. Missing ones count as
    # unsupported: silently shrinking the denominator would inflate the score.
    by_claim = {v.claim.strip().casefold(): v for v in result.verdicts}
    verdicts: list[ClaimVerdict] = []
    for claim in claims:
        found = by_claim.get(claim.strip().casefold())
        if found is not None:
            verdicts.append(found)
        else:
            verdicts.append(
                ClaimVerdict(claim=claim, supported=False, reason="no verdict returned")
            )
    return verdicts


async def score_grounding(answer: str, context: str) -> GroundingScore:
    """Fraction of the answer's atomic claims the retrieved context supports."""
    claims = await decompose(answer)
    if not claims:
        # An answer that asserts nothing — an honest "the sources do not cover
        # this" — is not ungrounded. Scoring it 0 would punish exactly the
        # behaviour the failure-expected items are there to reward.
        return GroundingScore(score=1.0, not_applicable=True)

    verdicts = await verify(claims, context)
    supported = sum(1 for v in verdicts if v.supported)
    return GroundingScore(
        score=supported / len(verdicts) if verdicts else 0.0,
        claims_total=len(verdicts),
        claims_supported=supported,
        verdicts=verdicts,
    )
