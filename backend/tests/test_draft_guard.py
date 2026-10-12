"""Redrafting must never erase a human review.

scripts/draft_dataset.py writes the dataset from its own drafts. Before anyone
reviewed it, that was the only way the file came to exist. After a review it is
a way to lose the review: rerun it — on purpose, or by accident from shell
history — and every approval, and every correction made during review, is
silently replaced by the original drafts. The claim would drop back to
"model-drafted" with nothing to say why.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
from pathlib import Path
from types import ModuleType

import pytest

from atlas_eval.dataset import DATASET_PATH, META_PATH, load_dataset, write_jsonl

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "draft_dataset.py"


def _script(monkeypatch: pytest.MonkeyPatch, data: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("draft_dataset", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "DATASET_PATH", data / "dataset.v1.jsonl")
    monkeypatch.setattr(module, "META_PATH", data / "dataset.meta.json")
    return module


def _copy_shipped(data: Path) -> None:
    shutil.copy(DATASET_PATH, data / "dataset.v1.jsonl")
    shutil.copy(META_PATH, data / "dataset.meta.json")


def test_a_reviewed_dataset_is_never_overwritten(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _copy_shipped(tmp_path)
    before = (tmp_path / "dataset.v1.jsonl").read_bytes()
    script = _script(monkeypatch, tmp_path)

    with pytest.raises(
        SystemExit, match=re.escape("refusing to overwrite dataset.v1.jsonl: 50 item")
    ):
        script.main()

    assert (tmp_path / "dataset.v1.jsonl").read_bytes() == before


def test_an_unreviewed_dataset_can_still_be_redrafted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The control: before review, redrafting is the script's whole job."""
    _copy_shipped(tmp_path)
    shipped = load_dataset(tmp_path / "dataset.v1.jsonl", tmp_path / "dataset.meta.json")
    for item in shipped.items:
        item.provenance.reviewed_by = None
        item.provenance.reviewed_at = None
    write_jsonl(tmp_path / "dataset.v1.jsonl", shipped.items)
    script = _script(monkeypatch, tmp_path)

    script.main()

    redrafted = load_dataset(tmp_path / "dataset.v1.jsonl", tmp_path / "dataset.meta.json")
    assert len(redrafted.items) == 50
    assert sum(1 for i in redrafted.items if i.provenance.reviewed_by is not None) == 0
