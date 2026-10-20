# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""Fixtures that hold ``ingetrazo-model``'s stitch primitives and stitch pass
to ``core.mesh`` / ``core.history`` (docs/rust-rewrite-plan.md, M3.3):
random meshes (boxes pushed flush, coincident and near-coincident vertices
across weld-cell boundaries, stacked duplicate faces, T-junctions, collinear
chains, coplanar faces and regions, curves crossed by other geometry) run
through the REAL Python methods — tag_curve, resplit_curves, weld_coincident,
dedupe_faces, _weld_into, interior_vertex_on, split_edge_at, collapse_vertex,
dissolve_*, revert_stitch, run_stitch and the three commands — with the whole
state after every step: the lists in order, each object's links and flags,
face attrs, the weld registry, the displaced queue and the curve counter.
Objects are named by CREATION order, which is their arena index in Rust (the
wrappers below number every vertex/edge/face the moment it is made, so ones
made and dropped inside a step keep their slot).

Python's own order is not repeatable in a few places, so this script pins it
(the Rust side takes the same order):

* ``core.mesh.set`` is swapped for a set that iterates its mesh objects in
  creation order: ``Vertex.edges``, ``Vertex.faces()``, ``set(e1.faces)`` in
  collapse_vertex, the curve walk's ``next(n for n in v.edges ...)``.
* ``core.history._coplanar_component`` returns such a set, so
  ``dissolve_coplanar_region`` gets its faces in creation order (the first
  face's normal decides the union's winding).
* ``core.mesh._trace_loops`` starts a loop at the first outline edge not yet
  traced, in the direction it was listed (Python picks ``next(iter(set))`` of
  frozensets of objects, address order, either direction).

Query results Python returns as sets (surface_of, edges_at, faces_at) are
recorded sorted. Committed; rerun only to widen:

    python scripts/golden/make_stitch_fixtures.py

Deleted with the Python app at M10.
"""
from __future__ import annotations

import itertools
import json
import os
import random
import struct
import sys
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import core.history as ch  # noqa: E402
import core.mesh as cm  # noqa: E402
from core.mesh import Mesh  # noqa: E402
from core.vecmath import QVector3D as V  # noqa: E402

OUT = ROOT / "rust" / "crates" / "ingetrazo-model" / "tests" / "data" / "stitch"

# ---- pinning Python's order ----------------------------------------------------

ORD: dict = {}                       # id(object) -> creation ordinal
TAB = {"v": [], "e": [], "f": []}    # ordinal -> object (also keeps them alive)


def _reg(kind: str, o) -> None:
    if id(o) not in ORD:
        ORD[id(o)] = len(TAB[kind])
        TAB[kind].append(o)


class DetSet(set):
    """A set that iterates mesh objects in creation order."""

    def __iter__(self):
        return iter(sorted(set.__iter__(self), key=lambda o: ORD.get(id(o), 1 << 60)))


cm.set = DetSet


def _wrap(name: str, kind: str) -> None:
    orig = getattr(Mesh, name)

    def wrapper(self, *a, **k):
        o = orig(self, *a, **k)
        _reg(kind, o)
        return o
    setattr(Mesh, name, wrapper)


_wrap("vertex", "v")
_wrap("_link_edge", "e")
_wrap("add_face", "f")

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

# weld_coincident's cross-cell pass can cycle forever on a mesh a random weld_into
# or move has corrupted (Python hangs too): cap the welds one op may do.
_WELDS = [0]
_orig_weld_into = Mesh._weld_into


def _capped_weld_into(self, keep, dup):
    _WELDS[0] += 1
    if _WELDS[0] > 3000:
        raise RuntimeError("weld loop")
    return _orig_weld_into(self, keep, dup)


Mesh._weld_into = _capped_weld_into

# ---- encoding ---------------------------------------------------------------------


def h32(x: float) -> str:
    return struct.pack(">f", x).hex()


def bits(p) -> list:
    return [h32(p.x()), h32(p.y()), h32(p.z())]


def unbits(b) -> V:
    return V(*[struct.unpack(">f", bytes.fromhex(x))[0] for x in b])


def o(x) -> int:
    return ORD[id(x)]


def state(mesh) -> dict:
    return {
        "v": [[o(v), *bits(v.position)] for v in mesh.vertices],
        "e": [[o(e), o(e.v0), o(e.v1), [o(f) for f in e.faces],
               sorted(o(x) for x in e.v0.edges), sorted(o(x) for x in e.v1.edges),
               e.soft, e.hidden, e.layer, e.curve] for e in mesh.edges],
        "f": [[o(f), [o(v) for v in f.loop], [[o(v) for v in h] for h in f.hole_loops],
               f.interior, json.loads(json.dumps(f.attrs))] for f in mesh.faces],
        "reg": sorted([*k, o(v)] for k, v in mesh._registry.items()),
        "disp": sorted([*k, [o(v) for v in vs]] for k, vs in mesh._displaced.items()),
    }


#: Read-only ops: their step carries no state (the next one checks it still holds).
QUERIES = {"interior_vertex_on", "collapsible_vertex", "curve_edges", "surface_of", "edges_at",
           "faces_at", "counts"}


STATS: dict = {}


class Abort(Exception):
    pass


class Run:
    """One sequence: a mesh, the objects ops refer to by ordinal, the log."""

    def __init__(self, seed: int) -> None:
        ORD.clear()
        for t in TAB.values():
            t.clear()
        cm._CURVE_COUNTER = 1
        self.rnd = random.Random(seed)
        self.mesh = Mesh()
        self.recs: list = []
        self.cmds: list = []
        self.steps: list = []
        self.scene = types.SimpleNamespace(
            mesh=self.mesh, edges=self.mesh.edges, faces=self.mesh.faces,
            version=0, selection=set())

    # ---- running one op ----------------------------------------------------------
    def run(self, op: list, build: bool = False):
        before = state(self.mesh)
        _WELDS[0] = 0
        try:
            ret = self.apply(op)
        except Exception as exc:  # noqa: BLE001 - the op is invalid here: end the sequence
            raise Abort(f"{op[0]}: {exc!r}") from exc
        step = {"op": op, "ctr": cm._CURVE_COUNTER}
        if ret is not None:
            step["ret"] = ret
        after = state(self.mesh)
        if after != before:
            STATS[op[0] + ":changed"] = STATS.get(op[0] + ":changed", 0) + 1
        if ret not in (None, False, 0, [], "degenerate") or op[0] == "interior_vertex_on" and ret is not None:
            STATS[op[0] + ":hit"] = STATS.get(op[0] + ":hit", 0) + 1
        if not build and op[0] not in QUERIES:
            step["state"] = after
        self.steps.append(step)
        return ret

    def apply(self, op):
        m = self.mesh
        name = op[0]
        gv, ge, gf = (lambda i: TAB["v"][i]), (lambda i: TAB["e"][i]), (lambda i: TAB["f"][i])
        if name == "add_face":
            m.add_face([unbits(b) for b in op[1]], [[unbits(b) for b in h] for h in op[2]])
        elif name == "add_edge":
            try:
                m.add_edge(unbits(op[1]), unbits(op[2]))
            except ValueError:
                return "degenerate"
        elif name == "remove_face":
            m.remove_face(gf(op[1]))
        elif name == "remove_edge":
            m.remove_edge(ge(op[1]))
        elif name == "split_edge":
            m.split_edge(ge(op[1]), unbits(op[2]))
        elif name == "move_vertex":
            m.move_vertex(gv(op[1]), unbits(op[2]))
        elif name == "place_vertex":
            m.place_vertex(gv(op[1]), unbits(op[2]))
        elif name == "set_attrs":
            gf(op[1]).attrs = json.loads(json.dumps(op[2]))   # a copy: turn_face_over edits in place
        elif name == "set_edge":
            e = ge(op[1])
            e.soft, e.hidden, e.layer, e.curve = op[2], op[3], op[4], op[5]
        elif name == "turn_face_over":
            cm.turn_face_over(gf(op[1]))
        elif name == "tag_curve":
            return m.tag_curve([unbits(b) for b in op[1]], op[2])
        elif name == "resplit_curves":
            m.resplit_curves()
        elif name == "curve_edges":
            return [o(e) for e in m.curve_edges(ge(op[1]))]
        elif name == "surface_of":
            return sorted(o(f) for f in m.surface_of(gf(op[1])))
        elif name == "edges_at":
            return sorted(o(e) for e in m.edges_at(gv(op[1])))
        elif name == "faces_at":
            return sorted(o(f) for f in m.faces_at(gv(op[1])))
        elif name == "weld_coincident":
            return m.weld_coincident()
        elif name == "dedupe_faces":
            return m.dedupe_faces()
        elif name == "weld_into":
            m._weld_into(gv(op[1]), gv(op[2]))
        elif name == "interior_vertex_on":
            v = m.interior_vertex_on(ge(op[1]))
            return None if v is None else o(v)
        elif name == "split_edge_at":
            self.recs.append(m.split_edge_at(ge(op[1]), gv(op[2])))
        elif name == "collapsible_vertex":
            return m.collapsible_vertex(gv(op[1]))
        elif name == "collapse_vertex":
            self.recs.append(m.collapse_vertex(gv(op[1])))
        elif name == "dissolve_edge_recorded":
            rec = m.dissolve_edge_recorded(ge(op[1]))
            if rec is not None:
                self.recs.append(rec)
            return rec is not None
        elif name == "dissolve_coplanar_pair":
            rec = m.dissolve_coplanar_pair(gf(op[1]), gf(op[2]))
            if rec is not None:
                self.recs.append(rec)
            return rec is not None
        elif name == "dissolve_coplanar_region":
            f = m.dissolve_coplanar_region([gf(i) for i in op[1]])
            return None if f is None else o(f)
        elif name == "dissolve_edge":
            f = m.dissolve_edge(ge(op[1]))
            return None if f is None else o(f)
        elif name == "revert_stitch":
            m.revert_stitch(self.recs[op[1]])
        elif name == "run_stitch":
            seed = {ch._key(unbits(b)) for b in op[1]}
            nf = None if op[2] is None else {gf(i) for i in op[2]}
            ch.run_stitch(m, seed, nf, op[3], op[4])
            if nf is not None:
                return sorted(o(f) for f in nf)
        elif name == "cmd":
            seed = [unbits(b) for b in op[2]]
            cls = {"stitch_solid": ch.StitchSolidCommand, "coplanar_merge": ch.CoplanarMergeCommand,
                   "prune_orphan_edges": ch.PruneOrphanEdgesCommand}[op[1]]
            c = cls(seed)
            c.do(self.scene)
            self.cmds.append(c)
        elif name == "cmd_undo":
            self.cmds[op[1]].undo(self.scene)
        elif name == "counts":
            return {"fc": cm.face_count(m), "vc": cm.vertex_count(m),
                    "vp": [bits(V(*map(float, r))) for r in cm.vertex_positions(m)],
                    "fa": json.loads(json.dumps(list(cm.face_attrs(m))))}
        else:
            raise ValueError(name)
        return None


# ---- picking helpers -----------------------------------------------------------------

ATTRS = [
    {"color": [255, 0, 0]},
    {"mat": "brick", "opacity": 0.5},
    {"color": [1, 2, 3], "back": {"color": [9, 9, 9]}},
    {"mat": "wood", "back": {"mat": "paint", "opacity": 0.25}, "layer": "A"},
    {"back": True, "color": [7, 7, 7]},
    {"texture": {"file": "t.png"}, "ifc": {"cls": "IfcWall"}},
]


def frame(rnd):
    perm = rnd.choice(list(itertools.permutations(range(3))))
    sg = [rnd.choice([1, -1]) for _ in range(3)]
    off = [rnd.choice([0.0, 1.5, -2.0, 10.25]) for _ in range(3)]

    def pt(u, v, w=0.0):
        a = [0.0] * 3
        a[perm[0]], a[perm[1]], a[perm[2]] = sg[0] * u, sg[1] * v, sg[2] * w
        return V(a[0] + off[0], a[1] + off[1], a[2] + off[2])
    return pt


def add_face(r: Run, pts, holes=(), build=True) -> None:
    r.run(["add_face", [bits(p) for p in pts], [[bits(p) for p in h] for h in holes]], build=build)


def box(pt, x, y, z, sx, sy, sz) -> list:
    def c(i, j, k):
        return pt(x + i * sx, y + j * sy, z + k * sz)
    return [
        [c(0, 0, 0), c(0, 1, 0), c(1, 1, 0), c(1, 0, 0)],
        [c(0, 0, 1), c(1, 0, 1), c(1, 1, 1), c(0, 1, 1)],
        [c(0, 0, 0), c(1, 0, 0), c(1, 0, 1), c(0, 0, 1)],
        [c(0, 1, 0), c(0, 1, 1), c(1, 1, 1), c(1, 1, 0)],
        [c(0, 0, 0), c(0, 0, 1), c(0, 1, 1), c(0, 1, 0)],
        [c(1, 0, 0), c(1, 1, 0), c(1, 1, 1), c(1, 0, 1)],
    ]


# ---- scenario builders ------------------------------------------------------------------


def scn_grid(r: Run) -> None:
    rnd = r.rnd
    pt = frame(rnd)
    nx, ny = rnd.randint(2, 4), rnd.randint(1, 3)
    for i in range(nx):
        for j in range(ny):
            if rnd.random() < 0.1:
                continue
            x, y = float(i), float(j)
            q = [pt(x, y), pt(x + 1, y), pt(x + 1, y + 1), pt(x, y + 1)]
            kind = rnd.choices(["quad", "halves", "tris", "hole"], [4, 4, 1, 1])[0]
            if kind == "halves":
                polys = [[pt(x, y), pt(x + .5, y), pt(x + .5, y + 1), pt(x, y + 1)],
                         [pt(x + .5, y), pt(x + 1, y), pt(x + 1, y + 1), pt(x + .5, y + 1)]]
                holes = [[]] * 2
            elif kind == "tris":
                polys, holes = [[q[0], q[1], q[2]], [q[0], q[2], q[3]]], [[]] * 2
            elif kind == "hole":
                polys = [q]
                holes = [[[pt(x + .25, y + .25), pt(x + .75, y + .25), pt(x + .75, y + .75), pt(x + .25, y + .75)]]]
            else:
                polys, holes = [q], [[]]
            for poly, hl in zip(polys, holes):
                if rnd.random() < 0.15:
                    poly = poly[::-1]
                add_face(r, poly, hl)
    if rnd.random() < 0.4:                   # a wall standing on the first cell's bottom edge: a crease
        add_face(r, [pt(0, 0, 0), pt(1, 0, 0), pt(1, 0, 1), pt(0, 0, 1)])
    decorate(r)


def scn_tj(r: Run) -> None:
    """A big face beside a column of small ones: its edge runs past their corners."""
    rnd = r.rnd
    pt = frame(rnd)
    k = rnd.randint(2, 4)
    add_face(r, [pt(0, 0), pt(1, 0), pt(1, k), pt(0, k)])
    for j in range(k):
        if rnd.random() < 0.85:
            add_face(r, [pt(1, j), pt(2, j), pt(2, j + 1), pt(1, j + 1)])
    if rnd.random() < 0.5:                   # and one on the far side, split differently
        add_face(r, [pt(-1, 0), pt(0, 0), pt(0, k // 2 + 1), pt(-1, k // 2 + 1)])
        add_face(r, [pt(-1, k // 2 + 1), pt(0, k // 2 + 1), pt(0, k), pt(-1, k)])
    if rnd.random() < 0.5:
        add_face(r, [pt(0, 0), pt(0, 0, 1) if False else pt(1, 0), pt(1, 0, 1), pt(0, 0, 1)])
    decorate(r)


def decorate(r: Run, curves: bool = True) -> None:
    rnd, m = r.rnd, r.mesh
    for f in rnd.sample(m.faces, min(len(m.faces), rnd.randint(0, 3))):
        r.run(["set_attrs", o(f), rnd.choice(ATTRS)], build=True)
    for e in rnd.sample(m.edges, min(len(m.edges), rnd.randint(0, 3))):
        r.run(["set_edge", o(e), rnd.random() < 0.6, rnd.random() < 0.15, rnd.choice([None, None, "A", "B"]),
               rnd.choice([None, None, 1, 2]) if curves else e.curve], build=True)


def scn_boxes(r: Run) -> None:
    rnd = r.rnd
    pt = frame(rnd)
    for q in box(pt, 0, 0, 0, 1, 1, 1):
        add_face(r, q)
    variant = rnd.choice(["adjacent", "flush", "cells", "stack"])
    if variant == "adjacent":                # a neighbour box sharing a wall (same height, or lower)
        for q in box(pt, 1, 0, 0, 1, 1, rnd.choice([1, 0.5])):
            add_face(r, q)
    elif variant == "flush":                 # a cap raised above, then pushed flush onto the top ring
        cap = [pt(0, 0, 2), pt(1, 0, 2), pt(1, 1, 2), pt(0, 1, 2)]
        add_face(r, cap)
        m = r.mesh
        for v in list(m.vertices)[-4:]:
            r.run(["move_vertex", o(v), bits(pt(0, 0, -1) - pt(0, 0, 0))], build=True)
    elif variant == "cells":                 # near-coincident corners across a weld-cell boundary
        add_face(r, [pt(5, 5), pt(6, 5), pt(6, 6), pt(5, 6)])
        m = r.mesh
        top = [v for v in m.vertices][-4:]
        corners = [pt(1, 1, 1), pt(0, 1, 1), pt(0, 0, 1), pt(1, 0, 1)]
        for v, c in zip(top, corners):
            j = rnd.choice([0.0, 4.9e-5, -4.9e-5, 5.1e-5, -5.2e-5, 2e-5])
            r.run(["place_vertex", o(v), bits(V(c.x() + j, c.y(), c.z() - j))], build=True)
    else:                                    # an identical face stacked on another
        for q in rnd.sample(box(pt, 0, 0, 0, 1, 1, 1), 2):
            add_face(r, q[::-1] if rnd.random() < 0.5 else q)
    decorate(r)


def scn_collinear(r: Run) -> None:
    rnd, m = r.rnd, r.mesh
    pt = frame(rnd)
    add_face(r, [pt(0, 0), pt(1, 0), pt(1, 1), pt(0, 1)])
    if rnd.random() < 0.7:
        add_face(r, [pt(1, 0), pt(2, 0), pt(2, 1), pt(1, 1)])
    if rnd.random() < 0.4:
        add_face(r, [pt(0, 1), pt(1, 1), pt(1, 2), pt(0, 2)])
    for _ in range(rnd.randint(1, 4)):
        e = rnd.choice(m.edges)
        t = rnd.choice([0.5, 0.25, 0.75])
        p = e.v0.position * (1 - t) + e.v1.position * t
        r.run(["split_edge", o(e), bits(p)], build=True)
    decorate(r)


def circle(rnd, pt, n, radius, cx=0.0, cy=0.0):
    out = []
    import math
    for k in range(n):
        a = 2 * math.pi * k / n
        out.append(pt(cx + radius * math.cos(a), cy + radius * math.sin(a)))
    return out


def scn_curves(r: Run) -> None:
    rnd, m = r.rnd, r.mesh
    pt = frame(rnd)
    n = rnd.choice([4, 6, 8, 12])
    ring = circle(rnd, pt, n, 1.0)
    closed = rnd.random() < 0.75
    last = n if closed else n - 1
    for k in range(last):
        r.run(["add_edge", bits(ring[k]), bits(ring[(k + 1) % n])], build=True)
    r.run(["tag_curve", [bits(p) for p in ring], closed], build=True)
    kind = rnd.choice(["disc", "chords", "square", "cross", "plain"])
    if kind == "disc" and closed:
        add_face(r, ring)
    elif kind == "chords":
        for _ in range(rnd.randint(1, 3)):
            a, b = rnd.sample(range(n), 2)
            r.run(["add_edge", bits(ring[a]), bits(ring[b])], build=True)
    elif kind == "square" and n % 4 == 0:
        add_face(r, [ring[0], ring[n // 4], ring[n // 2], ring[3 * n // 4]])
    elif kind == "cross":                    # a line crossing an arc mid-edge
        e = rnd.choice(m.edges)
        p = e.v0.position * 0.5 + e.v1.position * 0.5
        r.run(["split_edge", o(e), bits(p)], build=True)
        r.run(["add_edge", bits(p), bits(pt(0, 0))], build=True)
    if rnd.random() < 0.5:                   # a second curve over part of the first
        r.run(["tag_curve", [bits(p) for p in ring[: rnd.randint(2, n)]], False], build=True)
    decorate(r, curves=False)


# ---- the work each scenario is built for -----------------------------------------------


def tjunctions(r: Run, p: float = 0.6) -> None:
    """Resolve T-junctions by hand: the interior vertex, then the split."""
    rnd, m = r.rnd, r.mesh
    for e in list(m.edges):
        if e in m.edges and rnd.random() < p:
            v = r.run(["interior_vertex_on", o(e)])
            if v is not None:
                r.run(["split_edge_at", o(e), v])


def collapses(r: Run, p: float = 0.8) -> None:
    rnd, m = r.rnd, r.mesh
    for v in list(m.vertices):
        if v in m.vertices and len(v.edges) == 2 and rnd.random() < p:
            if r.run(["collapsible_vertex", o(v)]):
                r.run(["collapse_vertex", o(v)])


def tail_grid(r: Run) -> None:
    rnd = r.rnd
    tjunctions(r)
    collapses(r)
    m = r.mesh
    for _ in range(rnd.randint(1, 3)):
        two = [e for e in m.edges if len(e.faces) == 2]
        if not two:
            break
        e = rnd.choice(two)
        kind = rnd.choice(["edge", "rec", "pair"])
        if kind == "edge":
            r.run(["dissolve_edge", o(e)])
        elif kind == "rec":
            r.run(["dissolve_edge_recorded", o(e)])
        else:
            r.run(["dissolve_coplanar_pair", o(e.faces[0]), o(e.faces[1])])
    for _ in range(rnd.randint(1, 3)):
        reg = grow_region(r)
        if reg and len(reg) >= 2:
            r.run(["dissolve_coplanar_region", reg])
    for f in rnd.sample(m.faces, min(2, len(m.faces))):
        r.run(["surface_of", o(f)])
    nf = [o(f) for f in rnd.sample(r.mesh.faces, min(2, len(r.mesh.faces)))] if rnd.random() < 0.5 else None
    r.run(["run_stitch", some_positions(r), nf, True, rnd.random() < 0.7])


def tail_boxes(r: Run) -> None:
    rnd = r.rnd
    r.run(["weld_coincident"])
    r.run(["dedupe_faces"])
    r.run(["run_stitch", some_positions(r), None, rnd.random() < 0.5, True])
    r.run(["cmd", "stitch_solid", some_positions(r)])
    if rnd.random() < 0.7:
        r.run(["cmd_undo", len(r.cmds) - 1])


def tail_collinear(r: Run) -> None:
    rnd = r.rnd
    collapses(r, 1.0)
    while r.recs and rnd.random() < 0.6:
        r.run(["revert_stitch", len(r.recs) - 1])
        r.recs.pop()
    r.run(["run_stitch", some_positions(r), None, rnd.random() < 0.5, True])


def tail_curves(r: Run) -> None:
    rnd, m = r.rnd, r.mesh
    r.run(["resplit_curves"])
    for e in rnd.sample(m.edges, min(3, len(m.edges))):
        r.run(["curve_edges", o(e)])
    tjunctions(r, 0.3)
    r.run(["run_stitch", some_positions(r), None, True, True])
    r.run(["resplit_curves"])


SCENARIOS = [(scn_grid, tail_grid), (scn_tj, tail_grid), (scn_tj, tail_grid), (scn_boxes, tail_boxes), (scn_collinear, tail_collinear), (scn_curves, tail_curves)]

# ---- random stitch ops --------------------------------------------------------------------


def some_positions(r: Run, k=None) -> list:
    m = r.mesh
    vs = m.vertices
    if not vs:
        return []
    k = k or r.rnd.randint(1, len(vs))
    return [bits(v.position) for v in r.rnd.sample(vs, min(k, len(vs)))]


def grow_region(r: Run):
    rnd, m = r.rnd, r.mesh
    if not m.faces:
        return None
    f0 = rnd.choice(m.faces)
    region = [f0]
    for _ in range(rnd.randint(1, 5)):
        f = rnd.choice(region)
        nbrs = []
        for lp in (f.loop, *f.hole_loops):
            for a, b in zip(lp, lp[1:] + lp[:1]):
                e = m.find_edge(a, b)
                if e is not None:
                    nbrs += [g for g in e.faces if g not in region]
        if nbrs:
            region.append(rnd.choice(nbrs))
    return [o(f) for f in region]


def random_ops(r: Run, n: int) -> None:
    rnd, m = r.rnd, r.mesh
    for _ in range(n):
        if not m.vertices:
            return
        roll = rnd.random()
        if roll < 0.14:
            nf = None
            if m.faces and rnd.random() < 0.6:
                nf = [o(f) for f in rnd.sample(m.faces, rnd.randint(1, min(3, len(m.faces))))]
            seed = some_positions(r, None if rnd.random() < 0.5 else rnd.randint(1, 4))
            r.run(["run_stitch", seed, nf, rnd.random() < 0.8, rnd.random() < 0.8])
        elif roll < 0.20:
            r.run(["weld_coincident"])
        elif roll < 0.25:
            r.run(["dedupe_faces"])
        elif roll < 0.35 and m.edges:
            e = rnd.choice(m.edges)
            v = r.run(["interior_vertex_on", o(e)])
            if v is not None and rnd.random() < 0.85:
                r.run(["split_edge_at", o(e), v])
            elif rnd.random() < 0.15:
                others = [w for w in m.vertices if w is not e.v0 and w is not e.v1]
                if others:
                    r.run(["split_edge_at", o(e), o(rnd.choice(others))])
        elif roll < 0.45:
            two = [v for v in m.vertices if len(v.edges) == 2]
            v = rnd.choice(two or m.vertices)
            ok = r.run(["collapsible_vertex", o(v)])
            if len(v.edges) == 2 and (ok or rnd.random() < 0.3):
                r.run(["collapse_vertex", o(v)])
        elif roll < 0.59:
            two = [e for e in m.edges if len(e.faces) == 2]
            kind = rnd.choice(["edge", "edge", "rec", "pair", "region"])
            if kind == "edge" and two:
                r.run(["dissolve_edge", o(rnd.choice(two))])
            elif kind == "rec" and two:
                r.run(["dissolve_edge_recorded", o(rnd.choice(two))])
            elif kind == "pair" and len(m.faces) >= 2:
                if two and rnd.random() < 0.8:
                    e = rnd.choice(two)
                    a, b = e.faces[0], e.faces[1]
                else:
                    a, b = rnd.sample(m.faces, 2)
                r.run(["dissolve_coplanar_pair", o(a), o(b)])
            elif kind == "region":
                reg = grow_region(r)
                if reg and len(reg) >= 2:
                    r.run(["dissolve_coplanar_region", reg])
        elif roll < 0.66:
            if r.recs:
                # LIFO only: a record is valid right after what it recorded
                r.run(["revert_stitch", len(r.recs) - 1])
                r.recs.pop()
        elif roll < 0.74 and len(m.vertices) >= 2:
            v, w = rnd.sample(m.vertices, 2)
            if rnd.random() < 0.5:
                r.run(["move_vertex", o(v), bits(w.position - v.position)])
            else:
                j = rnd.choice([0.0, 0.0, 1e-5, 4.9e-5, -4.9e-5, 5.1e-5, -5.2e-5])
                p = w.position
                r.run(["place_vertex", o(v), bits(V(p.x() + j, p.y(), p.z() - j))])
        elif roll < 0.79 and m.faces:
            f = rnd.choice(m.faces)
            ring = [v.position for v in f.loop]
            r.run(["add_face", [bits(p) for p in (ring[::-1] if rnd.random() < 0.5 else ring)], []])
        elif roll < 0.83 and m.faces:
            r.run(["turn_face_over", o(rnd.choice(m.faces))])
        elif roll < 0.87:
            if m.faces and rnd.random() < 0.6:
                r.run(["set_attrs", o(rnd.choice(m.faces)), rnd.choice(ATTRS)])
            elif m.edges:
                r.run(["set_edge", o(rnd.choice(m.edges)), rnd.random() < 0.6, rnd.random() < 0.1,
                       rnd.choice([None, "A"]), rnd.choice([None, 1, 2])])
        elif roll < 0.92:
            kind = rnd.choice(["stitch_solid", "coplanar_merge", "prune_orphan_edges"])
            r.run(["cmd", kind, some_positions(r, rnd.randint(1, 4))])
            if rnd.random() < 0.7:
                r.run(["cmd_undo", len(r.cmds) - 1])
        elif roll < 0.96:
            q = rnd.choice(["surface_of", "edges_at", "faces_at", "curve_edges", "counts"])
            if q == "surface_of" and m.faces:
                r.run([q, o(rnd.choice(m.faces))])
            elif q in ("edges_at", "faces_at"):
                r.run([q, o(rnd.choice(m.vertices))])
            elif q == "curve_edges" and m.edges:
                r.run([q, o(rnd.choice(m.edges))])
            else:
                r.run(["counts"])
        elif roll < 0.985:
            if rnd.random() < 0.5:
                r.run(["resplit_curves"])
            else:
                path = rnd.sample(m.vertices, min(len(m.vertices), rnd.randint(2, 6)))
                r.run(["tag_curve", [bits(v.position) for v in path], rnd.random() < 0.5])
        elif len(m.vertices) >= 2:
            keep, dup = rnd.sample(m.vertices, 2)
            r.run(["weld_into", o(keep), o(dup)])


def sequence(seed: int):
    r = Run(seed)
    try:
        rnd = r.rnd
        build, tail = rnd.choice(SCENARIOS)
        build(r)
        tail(r)                      # the work this scenario is for ...
        random_ops(r, rnd.randint(2, 7))   # ... then a random mix
    except Abort as exc:
        r.aborted = str(exc)
    return r


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    stats = STATS
    seed0 = 20261019
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 140
    with open(OUT / "stitch_ops.jsonl", "w", encoding="ascii", newline="\n") as fh:
        for i in range(count):
            r = sequence(seed0 + i)
            if getattr(r, "aborted", None):
                stats["ABORT " + r.aborted.split(":")[0]] = stats.get("ABORT " + r.aborted.split(":")[0], 0) + 1
            if not r.steps:
                continue
            for s in r.steps:
                k = s["op"][0]
                stats[k] = stats.get(k, 0) + 1
            fh.write(json.dumps(r.steps, separators=(",", ":")) + "\n")
            total += len(r.steps)
    (OUT / ".gitattributes").write_text("# Bits Python produced; compared exactly.\n* -text\n")
    print("steps", total, "bytes", (OUT / "stitch_ops.jsonl").stat().st_size)
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")


if __name__ == "__main__":
    main()
