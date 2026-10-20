# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""Fixtures that hold ``ingetrazo-ops``' Intersect Faces to the Python app
(docs/rust-rewrite-plan.md, M3.7): boxes crossing boxes, slanted planes
through boxes, groups and component instances (rotated, nested) crossing the
loose mesh, coplanar and touching faces, faces with holes, a crossing shorter
than the weld, nearly parallel planes, a lattice of 24 bars and random
scatters, in the three modes (With Model / With Selection / With Context),
loose or inside an open group.

Each scene is saved as a bare-JSON .igz (``<scene>.igz.json``) and reopened in
a fresh scene, so Python and Rust start from the very same bits. Every case
runs the REAL ``core.intersect.segments_for`` (the engine off: the pure
NumPy/QVector3D path) and then what ``MainWindow._on_intersect_faces`` does
with it: drop the crossings shorter than the weld and run
``build_add_edges(scene, segs, detect_faces=True)`` through ``History``.
``cases.json`` holds, per case, the selection, the segments Python found (f32
bits, in order), the ones it kept, and the active mesh's whole ordered state
before the apply, after it and after the undo (vertices, edges with their
flags and faces, faces with their loops, holes, interior flag and attrs).

Python's own order is not repeatable in one place, pinned here: the selection
is a ``set`` of objects, so ``segments_for`` would walk it in address order.
The selection iterates in creation order (loose faces in mesh order, then
groups in scene order), and the Rust side takes the same order.

Committed; rerun only to widen (PYTHONPATH at the repository root):

    python scripts/golden/make_intersect_fixtures.py

Deleted with the Python app at M10.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.pop("INGETRAZO_RUST", None)

from PySide6.QtGui import QGuiApplication, QMatrix4x4, QVector3D as V  # noqa: E402

from core import intersect as X  # noqa: E402
from core.edits import build_add_edges  # noqa: E402
from core.group import Group  # noqa: E402
from core.history import History  # noqa: E402
from core.mesh import Mesh  # noqa: E402
from core.scene import Scene  # noqa: E402
from formats import igz  # noqa: E402
from tests.golden.signature import _dumps, canonical, sha  # noqa: E402

OUT = ROOT / "rust" / "crates" / "ingetrazo-ops" / "tests" / "data" / "intersect"

_APP = QGuiApplication.instance() or QGuiApplication([])

#: The window's own constant (views/main_window.py).
WELD_LENGTH = 1e-4


def h32(x: float) -> str:
    return struct.pack(">f", x).hex()


def pbits(p) -> list:
    return [h32(p.x()), h32(p.y()), h32(p.z())]


# ---- geometry -----------------------------------------------------------------------

def box_faces(m: Mesh, x, y, z, sx, sy, sz, color=None):
    x1, y1, z1 = x + sx, y + sy, z + sz
    sides = [
        [V(x, y, z), V(x, y1, z), V(x1, y1, z), V(x1, y, z)],
        [V(x, y, z1), V(x1, y, z1), V(x1, y1, z1), V(x, y1, z1)],
        [V(x, y, z), V(x1, y, z), V(x1, y, z1), V(x, y, z1)],
        [V(x, y1, z), V(x, y1, z1), V(x1, y1, z1), V(x1, y1, z)],
        [V(x, y, z), V(x, y, z1), V(x, y1, z1), V(x, y1, z)],
        [V(x1, y, z), V(x1, y1, z), V(x1, y1, z1), V(x1, y, z1)],
    ]
    for pts in sides:
        f = m.add_face(pts)
        if color:
            f.attrs["color"] = list(color)


def box(x, y, z, sx, sy, sz, color=None) -> Mesh:
    m = Mesh()
    box_faces(m, x, y, z, sx, sy, sz, color)
    return m


def xform(dx=0.0, dy=0.0, dz=0.0, angle=0.0, axis=(0, 0, 1)):
    m = QMatrix4x4()
    m.translate(dx, dy, dz)
    if angle:
        m.rotate(angle, *axis)
    return m


def group(mesh, name, m=None, children=None) -> Group:
    g = Group(mesh, name=name)
    if m is not None:
        g.xform = m
    if children:
        g.adopt(children)
    return g


def slab_with_hole(z=0.0) -> Mesh:
    m = Mesh()
    m.add_face([V(0, 0, z), V(4, 0, z), V(4, 4, z), V(0, 4, z)],
               [[V(1, 1, z), V(1, 3, z), V(3, 3, z), V(3, 1, z)]])
    return m


def wall(x) -> Mesh:
    m = Mesh()
    m.add_face([V(x, -1, -1), V(x, 5, -1), V(x, 5, 1), V(x, -1, 1)])
    return m


def slanted_quad(tilt_deg, n=0) -> Mesh:
    m = Mesh()
    m.add_face([V(-1, -1, 0), V(3, -1, 0), V(3, 3, 0), V(-1, 3, 0)])
    return m


RED, BLUE, GREEN = (0.9, 0.1, 0.1), (0.2, 0.3, 0.9), (0.1, 0.8, 0.2)


def scenes() -> dict:
    out = {}
    out["two_boxes"] = ([], [
        group(box(0, 0, 0, 2, 2, 2, RED), "A"),
        group(box(1, 1, 1, 2, 2, 2, BLUE), "B"),
    ])
    out["loose_box_bar"] = (["box:0,0,0,2,2,2"], [
        group(box(1, -1, 1, 0.5, 4, 2, BLUE), "bar"),
    ])
    out["slanted"] = (["box:0,0,0,2,2,2"], [
        group(slanted_quad(0), "slant", m=xform(0, 0, 1, 25.0, (1, 0.3, 0))),
        group(box(0, 4, 0, 1, 1, 1), "far"),
    ])
    out["slanted_loose"] = (["box:0,0,0,2,2,2", "slant"], [])
    out["components"] = (["box:0.5,-0.5,0.2,1,3,0.6"], [
        group(box(0, 0, 0, 2, 1, 1, RED), "comp1", m=xform(0.5, 0.2, 0.0, 30.0)),
        group(box(0.5, -0.5, 0.2, 1, 3, 0.6, BLUE), "B"),
        group(box(0, 0, 0, 2, 1, 1, RED), "comp2", m=xform(1.0, -0.2, 0.1, 90.0, (0, 1, 0))),
    ])
    out["nested"] = (["box:0,0,0,2,2,2"], [
        group(Mesh(), "outer", m=xform(0.2, 0.1, 0.0, 10.0), children=[
            group(box(0, 0, 0, 1, 3, 1, GREEN), "c1", m=xform(0.5, -0.5, 0.5, 20.0, (0, 0, 1))),
            group(box(0, 0, 0, 3, 1, 1, BLUE), "c2", m=xform(0.0, 0.5, 1.2, 45.0, (1, 0, 0))),
        ]),
    ])
    out["coplanar"] = (["quad:0,0,0,2,2", "quad:1,1,0,2,2", "box:2,0,0,1,1,1", "box:0,2,0,1,1,1"], [
        group(box(0, 0, 0, 1, 1, 1, RED), "g1"),
        group(box(1, 0, 0, 1, 1, 1, BLUE), "g2"),   # shares a wall with g1
    ])
    out["holes"] = ([], [
        group(slab_with_hole(0.0), "slab"),
        group(wall(2.0), "wall1"),
        group(wall(0.5), "wall2"),
        group(wall(3.5), "wall3"),
    ])
    out["grazing"] = ([], [
        group(box(0, 0, 0, 2, 2, 2, RED), "A"),
        group(box(2 - 5e-5, 2 - 5e-5, 2 - 5e-5, 1, 1, 1, BLUE), "B"),
    ])
    tilt = 2e-5
    m = Mesh()
    m.add_face([V(-1, -1, 0), V(2, -1, 0), V(2, 2, 2 * tilt), V(-1, 2, 2 * tilt)])
    out["nearly_parallel"] = (["box:0,0,0,1,1,1"], [group(m, "tilted")])
    bars = []
    for i in range(6):
        bars.append(group(box(i * 1.0, -0.5, 0.0, 0.4, 6.0, 0.3, RED), f"v{i}"))
    for j in range(6):
        bars.append(group(box(-0.5, j * 1.0, 0.1, 6.0, 0.4, 0.3, BLUE), f"h{j}"))
    out["lattice"] = ([], bars)
    for seed in (1, 2, 3):
        rnd = random.Random(7000 + seed)
        gs = []
        for k in range(14):
            sz = [rnd.uniform(0.4, 1.6) for _ in range(3)]
            p = [rnd.uniform(0, 3) for _ in range(3)]
            m = None
            if rnd.random() < 0.7:
                m = xform(rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd.uniform(-1, 1),
                          rnd.uniform(0, 90), (rnd.uniform(0, 1), rnd.uniform(0, 1), rnd.uniform(0.1, 1)))
            gs.append(group(box(*p, *sz), f"r{k}", m=m))
        out[f"scatter{seed}"] = ([f"box:{rnd.uniform(0,3):.3f},{rnd.uniform(0,3):.3f},0,1.5,1.5,1.5"], gs)
    # a group edited from inside: its mesh holds a box and a wall through it
    inner = box(0, 0, 0, 2, 2, 2, GREEN)
    inner_extra = wall(1.0)
    for f in inner_extra.faces:
        inner.add_face(f.vertices)
    out["inside_group"] = ([], [group(inner, "G"), group(box(1, 1, 1, 2, 2, 2, BLUE), "other")])
    return out


def build_scene(name: str, loose_spec: list, groups: list) -> Scene:
    s = Scene()
    for spec in loose_spec:
        kind, args = spec.split(":") if ":" in spec else (spec, "")
        if spec == "slant":
            s.mesh.add_face([V(-1, -1, 0.7), V(3, -1, 0.3), V(3, 3, 1.4), V(-1, 3, 1.1)])
            continue
        a = [float(x) for x in args.split(",")]
        if kind == "box":
            box_faces(s.mesh, *a)
        elif kind == "quad":
            x, y, z, sx, sy = a
            s.mesh.add_face([V(x, y, z), V(x + sx, y, z), V(x + sx, y + sy, z), V(x, y + sy, z)])
    s.groups.extend(groups)
    return s


# ---- recording ----------------------------------------------------------------------

def mesh_state(mesh) -> dict:
    vi = {id(v): i for i, v in enumerate(mesh.vertices)}
    fi = {id(f): i for i, f in enumerate(mesh.faces)}
    return {
        "v": [pbits(v.position) for v in mesh.vertices],
        "e": [[vi[id(e.v0)], vi[id(e.v1)], 1 if e.soft else 0, 1 if e.hidden else 0,
               e.layer, e.curve, [fi[id(f)] for f in e.faces]] for e in mesh.edges],
        "f": [[[vi[id(v)] for v in f.loop],
               [[vi[id(v)] for v in h] for h in f.hole_loops],
               1 if getattr(f, "interior", False) else 0,
               sha(_dumps(canonical(f.attrs)), 16) if f.attrs else ""]
              for f in mesh.faces],
    }


def save_scene_json(scene, dest: Path) -> None:
    saved = (igz._LARGE_DOCUMENT, igz._pack_textures)
    igz._LARGE_DOCUMENT = 10 ** 9
    igz._pack_textures = lambda payload: ({}, 0)
    try:
        igz.save_scene(scene, dest)
    finally:
        igz._LARGE_DOCUMENT, igz._pack_textures = saved
    raw = dest.read_bytes()
    assert not raw.startswith(b"PK"), dest
    doc = json.loads(raw.decode("utf-8"))
    dest.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")


def reopen(path: Path) -> Scene:
    s = Scene()
    igz.load_into(s, path)
    return s


class Pinned(set):
    """The selection, iterated in the order given (a plain set walks the
    objects in address order)."""

    def __init__(self, ordered) -> None:
        super().__init__(ordered)
        self._order = list(ordered)

    def __iter__(self):
        return iter(self._order)


def run_case(path: Path, name: str, mode: str, sel: list, edit):
    scene = reopen(path)
    if edit is not None:
        scene.begin_group_edit(scene.groups[edit[0]])
    ctx = scene.edit_group.children if scene.edit_group is not None else scene.groups
    ents = []
    for kind, i in sel:
        ents.append(scene.mesh.faces[i] if kind == "f" else ctx[i])
    scene.selection = Pinned(ents)
    pre = mesh_state(scene.mesh)
    raw = X.segments_for(scene, mode)
    kept = [(a, b) for a, b in raw if (b - a).length() >= WELD_LENGTH]
    rec = {"scene": name, "mode": mode, "edit": edit, "sel": sel,
           "raw": [[pbits(a), pbits(b)] for a, b in raw],
           "kept": [[pbits(a), pbits(b)] for a, b in kept],
           "pre": pre}
    if kept:
        hist = History(scene)
        hist.execute(build_add_edges(scene, kept, detect_faces=True))
        rec["error"] = hist.last_error
        rec["post"] = mesh_state(scene.mesh)
        hist.undo()
        rec["undone"] = mesh_state(scene.mesh)
        assert rec["undone"] == pre, name
    return rec


def case_list(opened: dict) -> list:
    cases = []

    def faces(name, lo=0, hi=None):
        n = len(opened[name].mesh.faces)
        return [["f", i] for i in range(lo, n if hi is None else hi)]

    M, S, C = X.WITH_MODEL, X.WITH_SELECTION, X.WITH_CONTEXT
    cases += [("two_boxes", M, [["g", 0]], None), ("two_boxes", M, [["g", 1]], None),
              ("two_boxes", S, [["g", 0], ["g", 1]], None), ("two_boxes", C, [["g", 0]], None),
              ("two_boxes", M, [["g", 0], ["g", 1]], None), ("two_boxes", S, [["g", 0]], None)]
    cases += [("loose_box_bar", M, faces("loose_box_bar"), None),
              ("loose_box_bar", M, [["g", 0]], None),
              ("loose_box_bar", C, [["f", 1]], None),
              ("loose_box_bar", C, faces("loose_box_bar"), None),
              ("loose_box_bar", S, faces("loose_box_bar") + [["g", 0]], None),
              ("loose_box_bar", M, [["f", 0], ["f", 1], ["g", 0]], None)]
    cases += [("slanted", M, [["g", 0]], None), ("slanted", M, faces("slanted"), None),
              ("slanted", S, faces("slanted") + [["g", 0]], None),
              ("slanted", M, [["f", 1], ["f", 2]], None)]
    cases += [("slanted_loose", C, [["f", 6]], None), ("slanted_loose", M, [["f", 6]], None),
              ("slanted_loose", S, faces("slanted_loose"), None),
              ("slanted_loose", C, faces("slanted_loose", 0, 6), None)]
    cases += [("components", M, [["g", 0]], None), ("components", M, [["g", 2]], None),
              ("components", S, [["g", 0], ["g", 1], ["g", 2]], None),
              ("components", M, [["g", 0], ["g", 2]], None),
              ("components", M, faces("components"), None),
              ("components", S, faces("components") + [["g", 0], ["g", 2]], None)]
    cases += [("nested", M, [["g", 0]], None), ("nested", M, faces("nested"), None),
              ("nested", S, faces("nested") + [["g", 0]], None)]
    cases += [("coplanar", M, [["f", 0]], None), ("coplanar", C, [["f", 0]], None),
              ("coplanar", S, faces("coplanar"), None), ("coplanar", M, [["g", 0]], None),
              ("coplanar", S, [["g", 0], ["g", 1]], None), ("coplanar", M, faces("coplanar", 2, 8), None),
              ("coplanar", C, faces("coplanar", 6, 12), None)]
    cases += [("holes", M, [["g", 0]], None), ("holes", S, [["g", 0], ["g", 1], ["g", 2]], None),
              ("holes", S, [["g", 0], ["g", 1], ["g", 2], ["g", 3]], None), ("holes", M, [["g", 1]], None)]
    cases += [("grazing", M, [["g", 0]], None), ("grazing", S, [["g", 0], ["g", 1]], None)]
    cases += [("nearly_parallel", M, faces("nearly_parallel"), None),
              ("nearly_parallel", M, [["g", 0]], None)]
    cases += [("lattice", S, [["g", i] for i in range(12)], None),
              ("lattice", M, [["g", i] for i in range(6)], None),
              ("lattice", M, [["g", 0], ["g", 7], ["g", 9]], None)]
    for seed in (1, 2, 3):
        n = f"scatter{seed}"
        cases += [(n, S, [["g", i] for i in range(14)], None),
                  (n, M, [["g", i] for i in range(0, 14, 3)], None),
                  (n, M, faces(n), None),
                  (n, S, faces(n) + [["g", 2], ["g", 5]], None)]
    cases += [("inside_group", M, faces("inside_group"), None),
              ("inside_group", M, [["g", 0]], None)]
    # inside the open group G: its own faces, and the sibling group is not in
    # its context
    cases += [("inside_group@0", M, [["f", 0], ["f", 1], ["f", 2]], [0]),
              ("inside_group@0", C, [["f", 6]], [0]),
              ("inside_group@0", S, [["f", i] for i in range(7)], [0]),
              ("inside_group@0", M, [["f", 6]], [0])]
    return cases


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    opened: dict = {}
    for name, (loose, groups) in scenes().items():
        s = build_scene(name, loose, groups)
        path = OUT / f"{name}.igz.json"
        save_scene_json(s, path)
        opened[name] = reopen(path)
    out_cases = []
    for name, mode, sel, edit in case_list(opened):
        base = name.split("@")[0]
        rec = run_case(OUT / f"{base}.igz.json", base, mode, sel, edit)
        out_cases.append(rec)
    n_seg = sum(len(c["raw"]) for c in out_cases)
    n_apply = sum(1 for c in out_cases if "post" in c)
    (OUT / "cases.json").write_text(
        json.dumps(out_cases, sort_keys=True, separators=(",", ":")),
        encoding="utf-8", newline="\n")
    print(len(out_cases), "cases,", n_seg, "segments,", n_apply, "applied")


if __name__ == "__main__":
    main()
