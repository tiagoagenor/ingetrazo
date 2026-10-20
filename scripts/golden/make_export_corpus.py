# SPDX-License-Identifier: GPL-3.0-or-later
"""What the real exporters write for every document of the repository
(docs/rust-rewrite-plan.md, M5.8).

Each .igz of the corpus (examples, shipped components, the golden start
scenes, the fixtures under rust/crates/*/tests/data) is opened with
``formats.igz.load_into`` and exported the way the File menu does:

    obj    formats.obj.save_obj          every file it writes (obj, mtl, textures)
    stl    formats.stl.save_stl          each unit the dialog offers
    dae    formats.dae.save_dae          every file it writes (dae, textures);
                                         the two time stamps are masked
    glb    formats.gltf.save_glb         whole model, and "only the selection"
                                         (the even-numbered loose faces and
                                         top-level groups)
    dxf    formats.dxf_out.save_dxf_layers
                                         the writer behind "Export view as
                                         DXF", fed with the top-view outline
                                         of every face of the document (one
                                         layer per face layer) and, on a
                                         DASHED layer, of its holes. The
                                         real feed is the view's hidden-line
                                         drawing, tested on its own

Only hashes are kept (sha256 per file written, and the size). An exporter
that raises is recorded with its message. The Rust test
(rust/crates/ingetrazo-formats-mesh/tests/export_corpus.rs) opens the same
documents and must write the same bytes. Text files are the Windows text
mode ones (CRLF), as every other fixture of this family.

Run from the repository root with the native engine off (a Windows build):

    QT_QPA_PLATFORM=offscreen PYTHONUTF8=1 PYTHONPATH=. \
        .venv/Scripts/python.exe scripts/golden/make_export_corpus.py

Deleted with the Python app at M10.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.pop("INGETRAZO_RUST", None)

from PySide6.QtGui import QGuiApplication  # noqa: E402

import core.group  # noqa: E402
from core.scene import Scene  # noqa: E402
from formats import dae, dxf_out, gltf, igz, obj, stl  # noqa: E402
import make_roundtrip_fixtures as base  # noqa: E402

OUT = (ROOT / "rust" / "crates" / "ingetrazo-formats-mesh" / "tests" / "data"
       / "export_corpus.json")
UNITS = ("mm", "cm", "m", "in", "ft")
STAMP = re.compile(rb"<(created|modified)>[^<]*</\1>")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def files_of(directory: Path, mask_stamps: bool = False) -> dict:
    out = {}
    for p in sorted(directory.iterdir()):
        data = p.read_bytes()
        if mask_stamps and p.suffix == ".dae":
            data = STAMP.sub(rb"<\1>T</\1>", data)
        out[p.name] = [len(data), sha(data)]
    return out


def dxf_groups(scene) -> list:
    """The outline segments (x0, y0, x1, y1) of every face, per layer, and
    the hole rings on a dashed layer, in the order the faces come."""
    layers: dict = {}
    holes: list = []

    def ring(pts):
        n = len(pts)
        return [(pts[i].x(), pts[i].y(), pts[(i + 1) % n].x(),
                 pts[(i + 1) % n].y()) for i in range(n)]

    for face in obj._faces(scene):
        name = face.attrs.get("layer") or "VISTA"
        layers.setdefault(name, []).extend(ring(face.vertices))
        for h in face.holes:
            holes.extend(ring(h))
    groups: list = [(name, segs) for name, segs in layers.items()]
    groups.append(("agujeros", holes, "DASHED"))
    return groups


def attempt(fn):
    try:
        fn()
        return None
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def export_all(scene, tmp: Path) -> dict:
    out: dict = {}

    def run(name, write, mask=False):
        d = Path(tempfile.mkdtemp(dir=tmp))
        res = attempt(lambda: write(d))
        out[name] = res if isinstance(res, dict) else files_of(d, mask)

    run("obj", lambda d: obj.save_obj(scene, d / "model.obj"))
    for unit in UNITS:
        run(f"stl.{unit}", lambda d, u=unit: stl.save_stl(scene, d / "model.stl", u))
    run("dae", lambda d: dae.save_dae(scene, d / "model.dae"), mask=True)
    run("glb", lambda d: gltf.save_glb(scene, d / "model.glb"))
    run("dxf", lambda d: dxf_out.save_dxf_layers(d / "model.dxf", dxf_groups(scene)))
    # "Only the selection": the even-numbered loose faces and top-level groups.
    loose = list(scene.loose_mesh.faces)
    only = set(loose[::2]) | set(scene.groups[::2])
    if only:
        run("glb.only", lambda d: gltf.save_glb(scene, d / "model.glb", only=only))
    return out


def main() -> None:
    app = QGuiApplication.instance() or QGuiApplication(sys.argv)  # noqa: F841
    files = base.corpus()
    result = {}
    with tempfile.TemporaryDirectory() as tmp:
        for src in files:
            rel = src.relative_to(ROOT).as_posix()
            core.group._counter = itertools.count(1)
            scene = Scene()
            igz.load_into(scene, src)
            result[rel] = export_all(scene, Path(tmp))
            print(rel, flush=True)
    OUT.write_text(json.dumps({"files": result}, indent=1, sort_keys=True)
                   + "\n", encoding="utf-8")
    print(f"{len(result)} documents -> {OUT}")


if __name__ == "__main__":
    main()
