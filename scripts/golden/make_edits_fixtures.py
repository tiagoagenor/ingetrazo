# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""Fixtures that hold ``ingetrazo-ops``'s edits planner and drawing commands
to ``core.edits`` / ``core.history`` (docs/rust-rewrite-plan.md, M3.1).

Each line is one scenario: a fresh scene, then a random sequence of
drawing steps, the way the tools do them — a Line (``build_add_edge``), a
Rectangle (``build_add_edges`` with its own face and the plane rebuild), a
Circle (its face, ``TagCurve`` and the arrangement), a polyline, a divide
into n, an erase, a Reverse Faces — all on grid coordinates, so crossings,
T-junctions, overlaps and shared edges happen all the time, some of them on
a tilted plane or a wall so the scene is 3D. Every step records:

* the command list the PLANNER built, encoded (class, positions as float32
  bits, flags): the Rust planner must emit exactly that list;
* the L2 signature of the scene after running it through ``History``, after
  undoing it and after redoing it (the Rust commands must land on the same).

Python's own order is not repeatable in a few places, so this script pins
it as ``make_stitch_fixtures.py`` does (``core.mesh.set`` iterates mesh
objects in creation order; ``_coplanar_component`` and ``_trace_loops`` are
deterministic). Committed; rerun only to widen:

    python scripts/golden/make_edits_fixtures.py

Deleted with the Python app at M10.
"""
from __future__ import annotations

import json
import os
import random
import struct
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import core.history as ch  # noqa: E402
import core.mesh as cm  # noqa: E402
from core import edits  # noqa: E402
from core.mesh import Mesh  # noqa: E402
from core.scene import Scene  # noqa: E402
from core.vecmath import QVector3D as V  # noqa: E402
from tests.golden.signature import scene_signature, signature_hash  # noqa: E402

OUT = ROOT / "rust" / "crates" / "ingetrazo-golden" / "tests" / "data" / "edits_plans.jsonl"

# ---- pinning Python's order ---------------------------------------------------------

ORD: dict = {}
KEEP: list = []


def _reg(o) -> None:
    if id(o) not in ORD:
        ORD[id(o)] = len(KEEP)
        KEEP.append(o)


class DetSet(set):
    def __iter__(self):
        return iter(sorted(set.__iter__(self), key=lambda o: ORD.get(id(o), 1 << 60)))


cm.set = DetSet


def _wrap(name: str) -> None:
    orig = getattr(Mesh, name)

    def wrapper(self, *a, **k):
        o = orig(self, *a, **k)
        _reg(o)
        return o
    setattr(Mesh, name, wrapper)


_wrap("vertex")
_wrap("_link_edge")
_wrap("add_face")

_orig_component = ch._coplanar_component
ch._coplanar_component = lambda *a, **k: DetSet(_orig_component(*a, **k))


def _det_trace_loops(outline):
    adj: dict = {}
    edges: dict = {}
    for u, w in outline:
        if u is w:
            return None
        adj.setdefault(u, []).append(w)
        adj.setdefault(w, []).append(u)
        edges.setdefault(frozenset((u, w)), (u, w))
    for nbrs in adj.values():
        if len(nbrs) != 2:
            return None
    loops = []
    while edges:
        a, b = next(iter(edges.values()))
        edges.pop(frozenset((a, b)), None)
        loop = [a, b]
        prev, cur = a, b
        while True:
            nbrs = adj[cur]
            nxt = nbrs[0] if nbrs[0] is not prev else nbrs[1]
            edges.pop(frozenset((cur, nxt)), None)
            if nxt is loop[0]:
                break
            loop.append(nxt)
            prev, cur = cur, nxt
        loops.append(loop)
    return loops


cm._trace_loops = _det_trace_loops

# ---- encoding -------------------------------------------------------------------------


def h32(x: float) -> str:
    return struct.pack(">f", x).hex()


def bits(p) -> list:
    return [h32(p.x()), h32(p.y()), h32(p.z())]


def unbits(b) -> V:
    return V(*[struct.unpack(">f", bytes.fromhex(x))[0] for x in b])


def enc(c) -> list:
    """A command as the Rust side will compare it."""
    n = type(c).__name__
    if n == "AddEdgeCommand":
        return ["AddEdge", bits(c.a), bits(c.b), c.soft, c.curve]
    if n == "AddFaceCommand":
        holes = None if not c.preset_holes else [[bits(p) for p in h] for h in c.preset_holes]
        return ["AddFace", [bits(p) for p in c.vertices], c.auto, holes, c.attrs]
    if n == "DeleteEdgesCommand":
        return ["DeleteEdges", [[bits(a), bits(b)] for a, b in c._endpoints], c.cascade_faces]
    if n == "DeleteFaceCommand":
        return ["DeleteFace", [bits(p) for p in c._loop]]
    if n == "RebuildPlanarFacesCommand":
        return ["RebuildPlanar"]
    if n == "RebuildPlaneFacesCommand":
        return ["RebuildPlane", bits(c.origin), bits(c.normal)]
    if n == "TagCurveCommand":
        return ["TagCurve", [bits(p) for p in c.pts], c.closed]
    if n == "AutoTagDrawnFacesCommand":
        return ["AutoTag", c._tag, len(c._face_cmds)]
    if n == "EraseSelectionCommand":
        return ["Erase", [[bits(a), bits(b)] for a, b in c._edge_endpoints],
                [[bits(p) for p in lp] for lp in c._face_loops]]
    if n == "FlipFacesCommand":
        return ["Flip", len(c._faces)]
    if n == "SnapshotCompound":
        return ["Snapshot", [enc(i) for i in c.inner]]
    raise ValueError(n)


def mesh_hash(scene) -> str:
    return signature_hash(scene_signature(scene))


# ---- one scenario ------------------------------------------------------------------------

GRID = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


class Abort(Exception):
    pass


def pick_pt(rnd, z=0.0, plane="xy"):
    a, b = rnd.choice(GRID), rnd.choice(GRID)
    if plane == "xy":
        return V(a, b, z)
    if plane == "xz":
        return V(a, z, b)
    return V(z, a, b)


def rect_corners(rnd, plane, z):
    a, b = pick_pt(rnd, z, plane), pick_pt(rnd, z, plane)
    if plane == "xy":
        return [a, V(b.x(), a.y(), z), b, V(a.x(), b.y(), z)]
    if plane == "xz":
        return [a, V(b.x(), z, a.z()), b, V(a.x(), z, b.z())]
    return [a, V(z, b.y(), a.z()), b, V(z, a.y(), b.z())]


def circle_pts(rnd, plane, z, sides):
    import math
    c = pick_pt(rnd, z, plane)
    r = rnd.choice([0.5, 1.0, 1.5, 2.0, 2.5])
    pts = []
    for i in range(sides):
        t = 2 * math.pi * i / sides
        x, y = r * math.cos(t), r * math.sin(t)
        if plane == "xy":
            pts.append(V(c.x() + x, c.y() + y, z))
        elif plane == "xz":
            pts.append(V(c.x() + x, z, c.z() + y))
        else:
            pts.append(V(z, c.y() + x, c.z() + y))
    return pts


def run_scenario(seed: int, n_steps: int) -> dict:
    ORD.clear()
    KEEP.clear()
    rnd = random.Random(seed)
    cm._CURVE_COUNTER = 1
    scene = Scene()
    hist = ch.History(scene)
    ifc = None
    if rnd.random() < 0.2:
        ifc = {"class": "IfcWall", "name": "Wall"}
        scene.active_ifc = dict(ifc)
    planes = ["xy"] if rnd.random() < 0.55 else ["xy", "xz", "yz"]
    steps = []
    sigs = [mesh_hash(scene)]
    def emit_rect(corners) -> bool:
        """The Rectangle tool: its own face, the plane rebuild."""
        if (corners[0] - corners[2]).length() < 1e-3 or any(
                (corners[i] - corners[(i + 1) % 4]).length() < 1e-3 for i in range(4)):
            return False
        step = {"kind": "rect", "ctr": cm._CURVE_COUNTER}
        segs = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
        extra = [ch.AddFaceCommand(list(corners))]
        step.update(segs=[[bits(a), bits(b)] for a, b in segs], detect=False, plane_rebuild=True,
                    extra=[["AddFace", [bits(p) for p in corners]]])
        cmd = edits.build_add_edges(scene, segs, detect_faces=False, extra=extra, plane_rebuild=True)
        step["plan"] = enc(cmd)
        hist.execute(cmd)
        if hist.last_error is not None or hist.last_rejected is not None:
            raise Abort(hist.last_error or hist.last_rejected)
        step["sig"] = mesh_hash(scene)
        steps.append(step)
        return True

    for _ in range(n_steps):
        kind = rnd.choice(["line", "line", "rect", "rect", "poly", "circle", "erase", "erase_face",
                           "flip", "undo", "box", "divide", "join"])
        plane = rnd.choice(planes)
        z = rnd.choice([0.0, 0.0, 0.0, 1.0])
        ctr = cm._CURVE_COUNTER
        step: dict = {"kind": kind, "ctr": ctr}
        m = scene.mesh
        if kind == "line":
            a, b = pick_pt(rnd, z, plane), pick_pt(rnd, z, plane)
            if rnd.random() < 0.3:
                b = V(b.x() + 0.5, b.y(), b.z())
            if (a - b).length() < 1e-3:
                continue
            eye = V(8, -9, 7) if rnd.random() < 0.5 else None
            step.update(a=bits(a), b=bits(b), eye=None if eye is None else bits(eye))
            cmd = edits.build_add_edge(scene, a, b, detect_faces=True, eye=eye)
        elif kind == "poly":
            pts = [pick_pt(rnd, z, plane) for _ in range(rnd.randint(3, 5))]
            closed = rnd.random() < 0.5
            segs = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
            if closed:
                segs.append((pts[-1], pts[0]))
            segs = [s for s in segs if (s[0] - s[1]).length() > 1e-3]
            if not segs:
                continue
            detect = rnd.random() < 0.7
            step.update(segs=[[bits(a), bits(b)] for a, b in segs], detect=detect)
            cmd = edits.build_add_edges(scene, segs, detect_faces=detect)
        elif kind == "rect":
            corners = rect_corners(rnd, plane, z)
            if not emit_rect(corners):
                continue
            continue
        elif kind == "box":
            x0, y0, z0 = rnd.choice(GRID[:3]), rnd.choice(GRID[:3]), rnd.choice(GRID[:2])
            dx, dy, dz = rnd.choice([1.0, 2.0, 3.0]), rnd.choice([1.0, 2.0, 3.0]), rnd.choice([1.0, 2.0])
            x1, y1, z1 = x0 + dx, y0 + dy, z0 + dz
            faces = [
                [V(x0, y0, z0), V(x1, y0, z0), V(x1, y1, z0), V(x0, y1, z0)],
                [V(x0, y0, z1), V(x1, y0, z1), V(x1, y1, z1), V(x0, y1, z1)],
                [V(x0, y0, z0), V(x1, y0, z0), V(x1, y0, z1), V(x0, y0, z1)],
                [V(x0, y1, z0), V(x1, y1, z0), V(x1, y1, z1), V(x0, y1, z1)],
                [V(x0, y0, z0), V(x0, y1, z0), V(x0, y1, z1), V(x0, y0, z1)],
                [V(x1, y0, z0), V(x1, y1, z0), V(x1, y1, z1), V(x1, y0, z1)],
            ]
            rnd.shuffle(faces)
            for corners in faces:
                if rnd.random() < 0.4:
                    corners = corners[::-1]
                emit_rect(corners)
            continue
        elif kind == "divide":
            if not m.edges:
                continue
            e = rnd.choice(list(m.edges))
            n = rnd.randint(2, 5)
            step.update(edge=[bits(e.a), bits(e.b)], n=n)
            step["ret"] = edits.divide_edges(m, [e], n)
            hist.undo_stack.clear()
            hist.redo_stack.clear()
            step["sig"] = mesh_hash(scene)
            steps.append(step)
            continue
        elif kind == "join":
            if not m.edges:
                continue
            es = rnd.sample(list(m.edges), max(1, len(m.edges) // rnd.choice([1, 2, 3])))
            step.update(edges=[[bits(e.a), bits(e.b)] for e in es])
            step["ret"] = len(edits.join_edges(m, es))
            hist.undo_stack.clear()
            hist.redo_stack.clear()
            step["sig"] = mesh_hash(scene)
            steps.append(step)
            continue
        elif kind == "circle":
            sides = rnd.choice([6, 8, 12])
            pts = circle_pts(rnd, plane, z, sides)
            segs = [(pts[i], pts[(i + 1) % sides]) for i in range(sides)]
            from core.arrangement import coplanar_plane
            verts = [v.position for v in m.vertices] + list(pts)
            if coplanar_plane(verts) is not None:
                extra = [ch.TagCurveCommand(list(pts), closed=True), ch.RebuildPlanarFacesCommand()]
                pre = [["TagCurve", [bits(p) for p in pts], True], ["RebuildPlanar"]]
            else:
                extra = [ch.AddFaceCommand(list(pts)), ch.TagCurveCommand(list(pts), closed=True)]
                pre = [["AddFace", [bits(p) for p in pts]], ["TagCurve", [bits(p) for p in pts], True]]
            step.update(segs=[[bits(a), bits(b)] for a, b in segs], detect=False, plane_rebuild=False, extra=pre)
            cmd = edits.build_add_edges(scene, segs, detect_faces=False, extra=extra)
        elif kind == "erase":
            if not m.edges:
                continue
            es = rnd.sample(list(m.edges), min(len(m.edges), rnd.randint(1, 2)))
            ends = [(V(e.a), V(e.b)) for e in es]
            step.update(edges=[[bits(a), bits(b)] for a, b in ends])
            cmd = ch.EraseSelectionCommand(es)
        elif kind == "erase_face":
            if not m.faces:
                continue
            fs = rnd.sample(list(m.faces), min(len(m.faces), rnd.randint(1, 2)))
            step.update(faces=[[bits(p) for p in f.vertices] for f in fs])
            cmd = ch.EraseSelectionCommand([], fs)
        elif kind == "flip":
            if not m.faces:
                continue
            fs = rnd.sample(list(m.faces), min(len(m.faces), 2))
            step.update(faces=[[bits(p) for p in f.vertices] for f in fs])
            cmd = ch.FlipFacesCommand(fs)
        else:  # undo / redo of what is on the stack
            if hist.undo_stack and rnd.random() < 0.7:
                hist.undo()
                step["kind"] = "undo"
            elif hist.redo_stack:
                hist.redo()
                step["kind"] = "redo"
            else:
                continue
            step["sig"] = mesh_hash(scene)
            steps.append(step)
            continue
        step["plan"] = enc(cmd)
        hist.execute(cmd)
        if hist.last_error is not None or hist.last_rejected is not None:
            raise Abort(hist.last_error or hist.last_rejected)
        step["sig"] = mesh_hash(scene)
        step["ctr_after"] = cm._CURVE_COUNTER
        steps.append(step)
    return {"seed": seed, "ifc": ifc, "steps": steps}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    seed = 0
    stats = {"steps": 0, "snap": 0, "single": 0}
    while len(lines) < 120 and seed < 2000:
        seed += 1
        try:
            sc = run_scenario(seed, random.Random(seed).randint(4, 14))
        except (Abort, ValueError, RuntimeError, AssertionError) as exc:
            print("skip", seed, repr(exc)[:100])
            continue
        for s in sc["steps"]:
            stats["steps"] += 1
            if "plan" in s:
                stats["snap" if s["plan"][0] == "Snapshot" else "single"] += 1
        lines.append(json.dumps(sc, separators=(",", ":")))
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(len(lines), "scenarios", stats)


if __name__ == "__main__":
    main()
