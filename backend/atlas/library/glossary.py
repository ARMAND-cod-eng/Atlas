"""Everyday words for things the acts name otherwise.

A question says "chatbot"; the AI Act says "AI systems intended to interact
directly with natural persons". A question asks for a "lawful basis"; the GDPR
says "lawfulness of processing". Lexical search cannot bridge that on its own,
and the misses were mostly this: the question and the provision share no word.

Each entry maps a phrase people use to the act's own words, which are added to
the query's search terms, never put in place of them. An entry has to be
vocabulary that holds for any question using the phrase, not a route to one
provision for one question: "high-risk" questions are pointed at the words of
the classification rules, not at Article 6.

Built on the dev half of the compliance questions only (atlas_eval retrieval
--split dev), and measured once on the test half.
"""

from __future__ import annotations

import re

GLOSSARY: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(phrase, re.IGNORECASE), words)
    for phrase, words in (
        # AI Act Art. 50(1) never says chatbot.
        (r"\bchat ?bots?\b", "AI systems intended to interact directly with natural persons"),
        # The Official Journal writes "deep fake", two words.
        (r"\bdeep-?fakes?\b", "deep fake"),
        # Recruitment tools are described by what they do, not by "CV".
        (
            r"\b(cvs?|r[eé]sum[eé]s?)\b",
            "recruitment or selection of natural persons job applications candidates",
        ),
        # Whether something is high-risk is what the classification rules decide.
        (
            r"\b(is|are|counts? as|qualif\w* as|classed as)\b[^.?]{0,60}\bhigh-risk\b",
            "classification rules considered to be high-risk",
        ),
        # When rules apply: the acts' "entry into force" and "shall apply from".
        (
            r"\b(start|starts|begin|begins) to apply\b|\bcomes? into force\b|"
            r"\btakes? effect\b|\bdeadlines?\b",
            "entry into force shall apply from",
        ),
        # GDPR practice says "lawful basis"; Article 6 says "lawfulness".
        (r"\b(lawful|legal) (basis|bases|grounds?)\b", "lawfulness of processing"),
        # A country outside the EU is a "third country" in both acts.
        (
            r"\b(countr(y|ies)|states?) outside\b|\bnon-eu countr(y|ies)\b",
            "third country",
        ),
        # The "right to explanation", in the GDPR's words and the AI Act's.
        (
            r"\bright to (an )?explanation\b",
            "meaningful information about the logic involved explanation of individual "
            "decision-making",
        ),
    )
)


def expand(query: str) -> str:
    """The query with the acts' words for any everyday phrase in it appended."""
    extra = [words for pattern, words in GLOSSARY if pattern.search(query)]
    return " ".join([query, *extra])
