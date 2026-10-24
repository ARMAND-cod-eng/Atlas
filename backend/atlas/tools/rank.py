"""BM25 ranking, shared by the regulation library and the eval's frozen search.

One implementation for both, so what the eval measures about retrieval is what
production does. Okapi BM25 with the usual k1 and b; the scoring function
matters less than the property every caller relies on — a **total** order.
Ties are broken by a caller-supplied key, because equal scores left in arbitrary
order make two runs of identical code differ.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric runs. The eval's corpus is ranked with exactly this."""
    return _TOKEN.findall(text.lower())


class BM25:
    """Okapi BM25 over a fixed list of already-tokenized documents."""

    K1 = 1.5
    B = 0.75

    def __init__(self, documents: Sequence[Sequence[str]]) -> None:
        self._counts = [Counter(tokens) for tokens in documents]
        self._lengths = [len(tokens) for tokens in documents]
        self._avg_length = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        self._doc_freq: Counter[str] = Counter()
        for tokens in documents:
            self._doc_freq.update(set(tokens))

    def __len__(self) -> int:
        return len(self._counts)

    def score(self, index: int, terms: Iterable[str]) -> float:
        total = len(self._counts)
        counts = self._counts[index]
        length = self._lengths[index] or 1
        score = 0.0
        for term in terms:
            frequency = counts.get(term, 0)
            if frequency == 0:
                continue
            df = self._doc_freq.get(term, 0)
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            denominator = frequency + self.K1 * (
                1 - self.B + self.B * length / (self._avg_length or 1)
            )
            score += idf * (frequency * (self.K1 + 1)) / denominator
        return score

    def rank(self, terms: Sequence[str], keys: Sequence[str]) -> list[int]:
        """Document indices, best first; ties broken by ``keys``."""
        return sorted(range(len(self)), key=lambda i: (-self.score(i, terms), keys[i]))
