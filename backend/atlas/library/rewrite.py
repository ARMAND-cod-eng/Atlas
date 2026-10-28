"""A question, rewritten into the acts' own terms before the library is searched.

The glossary (atlas.library.glossary) bridges the everyday words people use for
things the acts name otherwise. What it leaves is the misses that take
reasoning: that screening job applicants is an "evaluation of personal aspects",
that moving data abroad needs "appropriate safeguards". One cheap model call
names the legal concepts a question turns on, in the acts' words, and they are
added to the search.

Held to vocabulary. Which act and which provision a question is about come from
the question: a rewrite naming "Article 52" would steer the lookup to Article 52
(the proposal's numbering, not the act's), and one naming the GDPR would widen
an AI Act question to both acts. Both are dropped from what the model writes.

Off unless QUERY_REWRITE is set: it is adopted only if it measures better on the
held-out half (atlas_eval retrieval --rewrite --split test).
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from atlas.metrics import RunMetrics
from atlas.prompts import COMPLIANCE_QUERY_REWRITE

# Enough for a line of terms, which is all the search needs.
MAX_TERMS_CHARS = 300
_MAX_TOKENS = 120

ModelCall = Callable[..., Awaitable[BaseMessage]]

_REFERENCES = re.compile(
    r"\b(?:art(?:icle|\.)?\s*\d{1,3}(?:\s*\(\s*[0-9a-z]{1,4}\s*\))*"
    r"|annex\s+[IVXLC]+\b(?:\s*\(\s*[0-9a-z]{1,4}\s*\))*"
    r"|recitals?\s+\d{1,3}\b)",
    re.IGNORECASE,
)
_ACT_NAMES = re.compile(
    r"\b(?:the\s+)?(?:ai act|artificial intelligence act|gdpr|general data protection regulation"
    r"|regulation\s*\(eu\)\s*\d{4}/\d+)\b",
    re.IGNORECASE,
)


def _words_only(text: str) -> str:
    """The reply without provision references or act names, as one search line."""
    cleaned = _ACT_NAMES.sub("", _REFERENCES.sub("", text))
    parts = [" ".join(part.split()) for part in re.split(r"[,;\n]", cleaned)]
    line = ", ".join(part for part in parts if part.strip(" .:-"))
    return line[:MAX_TERMS_CHARS].rsplit(" ", 1)[0] if len(line) > MAX_TERMS_CHARS else line


async def statutory_terms(question: str, metrics: RunMetrics, call: ModelCall) -> str:
    """The acts' terms for what the question turns on, as one search line."""
    # The researcher's node: the rewrite is part of its reading, and is traced
    # in its span like its searches.
    from atlas.graph.nodes import RESEARCHER

    reply = await call(
        RESEARCHER,
        [SystemMessage(COMPLIANCE_QUERY_REWRITE), HumanMessage(question)],
        metrics,
        max_tokens=_MAX_TOKENS,
    )
    return _words_only(str(reply.content))
