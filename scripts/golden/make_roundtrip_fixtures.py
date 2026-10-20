# SPDX-License-Identifier: GPL-3.0-or-later
"""What the Python app writes when it opens a document and saves it again
(docs/rust-rewrite-plan.md, M2.9).

Every .igz of the repository (examples, shipped components, golden start
scenes, the fixtures under rust/crates/*/tests/data) is opened with
``formats.igz.load_into`` and saved with ``formats.igz.save_scene``. The
result is recorded as hashes: the whole file, the inner ``document.json``
and every container member (name, size, sha256); the document text itself
is kept for the small ones. The Rust test
(rust/crates/ingetrazo-model/tests/roundtrip_corpus.rs) opens the same
files and must write the same bytes.

Each file is saved twice, the "Group N" counter reset before each open
(Rust numbers from 1 as a fresh process does). The one thing that may
differ between the two saves is a group uid: ``core.group.new_uid`` is
random, and a document without uids (the shipped examples) is written with
new ones every time. Such a file is marked ``random_uids`` and its uids
are recorded as ``~`` (the Rust test blanks them the same way, and does
not compare the whole-file hash); any other difference fails the script.

Run from the repository root (a Windows build; the ZIP central directory
says create_system 0 there):

    QT_QPA_PLATFORM=offscreen PYTHONUTF8=1 PYTHONPATH=. \
        .venv/Scripts/python.exe scripts/golden/make_roundtrip_fixtures.py

Deleted with the Python app at M10.
"""
from __future__ import annotations

import hashlib
import io
import itertools
import json
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QGuiApplication  # noqa: E402

import core.group  # noqa: E402
from core.scene import Scene  # noqa: E402
from core.version import __version__  # noqa: E402
from formats import igz  # noqa: E402

OUT = (ROOT / "rust" / "crates" / "ingetrazo-model" / "tests" / "data"
       / "roundtrip_corpus.json")
#: Documents up to this size keep their text in the fixture.
KEEP_TEXT = 1500
SKIP_DIRS = {".venv", "target", ".git", ".claude", "node_modules", "vendor"}
UID = re.compile(rb'"uid":\s*"[0-9a-f]{16}"')


def corpus() -> list[Path]:
    found = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        found += [Path(dirpath) / f for f in filenames if f.endswith(".igz")]
    return sorted(found)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def blank_uids(doc: bytes) -> bytes:
    return UID.sub(b'"uid":"~"', doc)


def open_save(src: Path, dst: Path) -> None:
    core.group._counter = itertools.count(1)
    scene = Scene()
    igz.load_into(scene, src)
    igz.save_scene(scene, dst)


def members(raw: bytes, blank: bool) -> list:
    """(name, bytes) of every container member; a plain file is one."""
    def fix(name, data):
        return blank_uids(data) if blank and name == "document.json" else data
    if raw[:4] != b"PK\x03\x04":
        return [("document.json", fix("document.json", raw))]
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        return [(i.filename, fix(i.filename, zf.read(i)))
                for i in zf.infolist()]


def describe(raw: bytes, random_uids: bool) -> dict:
    out = {"size": len(raw), "random_uids": random_uids}
    if not random_uids:
        out["sha256"] = sha(raw)
    parts = members(raw, random_uids)
    container = raw[:4] == b"PK\x03\x04"
    out["entries"] = ([[n, len(d), sha(d)] for n, d in parts]
                      if container else None)
    doc = dict(parts)["document.json"]
    out["document_sha256"] = sha(doc)
    out["document_size"] = len(doc)
    if len(doc) <= KEEP_TEXT:
        out["document"] = doc.decode("utf-8")
    return out


def main() -> None:
    app = QGuiApplication.instance() or QGuiApplication(sys.argv)  # noqa: F841
    files = corpus()
    entries = {}
    unstable = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for n, src in enumerate(files):
            rel = src.relative_to(ROOT).as_posix()
            a, b = tmp / f"{n}a.igz", tmp / f"{n}b.igz"
            open_save(src, a)
            open_save(src, b)
            ra, rb = a.read_bytes(), b.read_bytes()
            random_uids = ra != rb
            if random_uids and members(ra, True) != members(rb, True):
                unstable.append(rel)
            entries[rel] = describe(ra, random_uids)
    doc = {"app_version": __version__, "files": entries}
    OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False, sort_keys=True)
                   + "\n", encoding="utf-8")
    print(f"{len(entries)} files -> {OUT}")
    if unstable:
        print("NONDETERMINISTIC beyond uids:", *unstable, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
