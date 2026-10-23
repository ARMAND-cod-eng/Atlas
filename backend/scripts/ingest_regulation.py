"""Fetch the official text of each act in the library, once, and commit it.

    uv run python scripts/ingest_regulation.py            # every act
    uv run python scripts/ingest_regulation.py aia        # one

The only online step of the library. EUR-Lex's website puts scripted downloads
behind a bot check; the EU Publications Office's content API serves the same
Official Journal text by CELEX number, without one. The raw XHTML is stored
gzipped (mtime 0, so the bytes depend only on the text) next to a manifest of
checksums and retrieval dates, and everything after this parses that file —
offline, deterministically, and from text anyone can verify against the
Official Journal.

The texts are the acts as published, not consolidated versions: amendments
adopted later are not in them, and the manifest says when they were fetched.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import urllib.request
from datetime import UTC, datetime

from atlas.library import ACTS, SOURCES_DIR

CELLAR = "http://publications.europa.eu/resource/celex/{celex}"


def fetch(celex: str) -> bytes:
    request = urllib.request.Request(
        CELLAR.format(celex=celex),
        headers={"Accept": "application/xhtml+xml", "Accept-Language": "eng"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body: bytes = response.read()
    if b'class="oj-ti-art"' not in body:
        raise RuntimeError(f"{celex}: the response is not the act's XHTML ({len(body)} bytes)")
    return body


def main(selected: list[str]) -> None:
    manifest_path = SOURCES_DIR / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    for act in ACTS.values():
        if selected and act.id not in selected:
            continue
        raw = fetch(act.celex)
        (SOURCES_DIR / f"{act.id}.xhtml.gz").write_bytes(gzip.compress(raw, mtime=0))
        manifest[act.id] = {
            "celex": act.celex,
            "eli": act.eli,
            "source": CELLAR.format(celex=act.celex),
            "retrieved_at": datetime.now(UTC).date().isoformat(),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        print(f"{act.id}: {len(raw):,} bytes from {act.celex}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
