"""The LLM judge: GPT-4o at temperature 0.

**The model choice is load-bearing, for two independent reasons.**

*Temperature 0 is impossible on the primary.* Current Claude models reject an
explicit ``temperature`` — the Phase 0 defect where ``temperature=0.0`` returned
400 and every call silently fell through to the fallback. A Claude judge therefore
cannot satisfy the "judge at temperature 0" requirement at all.

*A model should not grade its own homework.* The system under test answers on
Claude. Models show a measurable preference for their own output, so scoring
Claude's answers with Claude would inflate every number in a way no amount of
prompt tuning would reveal.

Temperature 0 is not a determinism guarantee — batching and hardware can still
move a token — so ``atlas_eval calibrate`` measures the judge's run-to-run
disagreement on identical inputs rather than assuming it away.
"""

from __future__ import annotations

import logging

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from atlas.config import get_settings

logger = logging.getLogger(__name__)

# Pinned rather than following FALLBACK_MODEL: the judge is measurement
# apparatus, and it must not change because someone retuned the agent's fallback.
JUDGE_MODEL = "gpt-4o"
JUDGE_TEMPERATURE = 0.0


def build_judge[SchemaT: BaseModel](
    schema: type[SchemaT],
) -> Runnable[LanguageModelInput, dict[str, object] | BaseModel]:
    """A structured-output judge. No fallback chain, deliberately.

    The agent has a fallback because a dead provider must not fail a user's query.
    The judge is the opposite: if the judge provider is down, the correct outcome
    is a failed eval run, not a run silently scored by a different model.
    """
    model = ChatOpenAI(
        model=JUDGE_MODEL,
        api_key=SecretStr(get_settings().openai_api_key),
        temperature=JUDGE_TEMPERATURE,
        max_retries=2,
    )
    return model.with_structured_output(schema, include_raw=False)


async def judge[SchemaT: BaseModel](
    schema: type[SchemaT],
    system: str,
    user: str,
) -> SchemaT:
    """Run one judge call and return the parsed verdict."""
    result = await build_judge(schema).ainvoke(
        [SystemMessage(system), HumanMessage(user)],
    )
    if isinstance(result, schema):
        return result
    # with_structured_output can hand back a dict depending on the method used;
    # validating rather than casting keeps the type honest at runtime.
    return schema.model_validate(result)
