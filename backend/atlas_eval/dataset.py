"""The evaluation dataset: schema, loading, and validation.

The dataset is an **artefact with a version**, not a fixture. It is versioned
JSONL so a diff shows which item changed, and it carries per-item provenance so
the claim made about it in the README can be checked rather than trusted.

That last part is deliberate. "Hand-labeled" is a claim about who made the
judgements, not about who typed the queries. :func:`provenance_summary` computes
what the data actually supports, and a test asserts the README does not overstate
it — because a dataset description is exactly the kind of thing that stays true
at the moment it is written and quietly stops being true afterwards.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

DATA_DIR = Path(__file__).resolve().parent / "data"
DATASET_PATH = DATA_DIR / "dataset.v1.jsonl"
META_PATH = DATA_DIR / "dataset.meta.json"

# How a rubric entry is checked. Recorded per entry so the deterministic:judge
# split is data we can print, rather than a ratio anyone has to take on trust.
CheckKind = Literal["deterministic", "judge"]

Recency = Literal["current", "evergreen"]


class RubricEntry(BaseModel):
    """One required claim for task completion.

    A checklist entry, not a sentence of a reference answer. Scoring against a
    reference answer measures how closely the prose matches someone's phrasing;
    scoring against a checklist measures whether the required content is there.
    """

    id: str
    claim: str
    check: CheckKind
    # Deterministic entries only: alternative surface forms, any one of which
    # counts as present. Numbers get a tolerance instead of a literal match.
    patterns: list[str] = Field(default_factory=list)
    number: float | None = None
    tolerance: float = 0.0

    @model_validator(mode="after")
    def _deterministic_entries_need_something_to_match(self) -> RubricEntry:
        if self.check == "deterministic" and not self.patterns and self.number is None:
            raise ValueError(
                f"rubric entry {self.id!r} is deterministic but has neither patterns "
                "nor a number — it could never be satisfied"
            )
        return self


class ExpectedSubtasks(BaseModel):
    """The reasonable decomposition width for this query, hand-set.

    A range rather than a number: "how many subtasks is sensible" genuinely has
    a band, and pretending otherwise would make tool-call validity punish
    reasonable plans.
    """

    min: int = Field(ge=1)
    max: int = Field(ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> ExpectedSubtasks:
        if self.min > self.max:
            raise ValueError(f"expected_subtasks min ({self.min}) exceeds max ({self.max})")
        return self


class Provenance(BaseModel):
    """Who drafted this item and who reviewed it.

    ``reviewed_by`` is what the word "hand-labeled" rests on. It is null until a
    human has actually looked at the item in the review tool, and nothing sets it
    automatically.
    """

    drafted_by: Literal["model", "human"] = "model"
    reviewed_by: Literal["human"] | None = None
    reviewed_at: str | None = None


class EvalItem(BaseModel):
    """One labelled query trace."""

    id: str
    query: str
    expected_subtasks: ExpectedSubtasks
    rubric: list[RubricEntry] = Field(min_length=1)
    # Prose, not scored: what a confident-and-wrong answer looks like here. It is
    # what makes a rubric reviewable by a human six months later.
    wrong_but_plausible: str
    recency: Recency
    tags: list[str] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)

    @model_validator(mode="after")
    def _rubric_ids_unique(self) -> EvalItem:
        ids = [entry.id for entry in self.rubric]
        if len(set(ids)) != len(ids):
            raise ValueError(f"item {self.id!r} has duplicate rubric entry ids")
        return self


class DatasetMeta(BaseModel):
    """Dataset-level metadata, versioned alongside the items."""

    schema_version: int
    dataset_version: str
    created: str
    # The smoke subset is a committed list of ids, never resampled — resampling
    # per run makes consecutive runs incomparable, which is precisely how a
    # subset becomes worse than no subset.
    smoke_ids: list[str] = Field(default_factory=list)
    notes: str = ""


class Dataset(BaseModel):
    """The full dataset: metadata plus items."""

    meta: DatasetMeta
    items: list[EvalItem]

    @model_validator(mode="after")
    def _ids_unique_and_smoke_resolves(self) -> Dataset:
        ids = [item.id for item in self.items]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"duplicate item ids: {sorted(duplicates)}")
        missing = set(self.meta.smoke_ids) - set(ids)
        if missing:
            raise ValueError(f"smoke_ids reference unknown items: {sorted(missing)}")
        return self

    def by_id(self, item_id: str) -> EvalItem:
        for item in self.items:
            if item.id == item_id:
                return item
        raise KeyError(f"no dataset item {item_id!r}")

    def smoke(self) -> list[EvalItem]:
        """The committed smoke subset, in dataset order."""
        wanted = set(self.meta.smoke_ids)
        return [item for item in self.items if item.id in wanted]


def iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    """Yield one parsed object per non-blank line."""
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name} line {number}: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"{path.name} line {number}: expected an object")
            yield parsed


def write_jsonl(path: Path, rows: Sequence[BaseModel]) -> None:
    """Write models one per line, sorted keys, trailing newline.

    Stable serialisation matters: the dataset lives in git, and a diff should
    show what a human changed rather than how the serialiser felt that day.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(json.loads(row.model_dump_json()), sort_keys=True) + "\n")


def load_dataset(path: Path = DATASET_PATH, meta_path: Path = META_PATH) -> Dataset:
    """Load and validate the dataset. Raises on anything malformed.

    Loud rather than lenient: an eval that silently drops a malformed item
    reports a score over a different denominator than the one in the README.
    """
    if not path.exists():
        raise FileNotFoundError(f"dataset not found at {path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"dataset metadata not found at {meta_path}")

    meta = DatasetMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
    items = [EvalItem.model_validate(row) for row in iter_jsonl(path)]
    return Dataset(meta=meta, items=items)


class ProvenanceSummary(BaseModel):
    """What the dataset's provenance actually supports being claimed."""

    total: int
    reviewed: int

    @property
    def fully_reviewed(self) -> bool:
        return self.total > 0 and self.reviewed == self.total

    @property
    def claim(self) -> str:
        """The strongest description the data supports. Verbatim, for the README.

        "Hand-labeled" is a claim about the judgements being a human's. If any
        item went unreviewed, the honest phrasing is the weaker one — and this
        returns it, so the README cannot drift ahead of the work.
        """
        if self.fully_reviewed:
            return f"{self.total} hand-labeled query traces"
        return (
            f"{self.total} model-drafted query traces, "
            f"{self.reviewed} of {self.total} human-reviewed"
        )


def provenance_summary(dataset: Dataset) -> ProvenanceSummary:
    reviewed = sum(1 for item in dataset.items if item.provenance.reviewed_by == "human")
    return ProvenanceSummary(total=len(dataset.items), reviewed=reviewed)
