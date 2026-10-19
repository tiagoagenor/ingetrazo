# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixtures that hold ``ingetrazo-model``'s Mesh to ``core.mesh.Mesh``
(docs/rust-rewrite-plan.md, M2.1): random sequences of the mesh primitives
— add_face, add_edge, remove/relink, split_edge, move/place_vertex, holes,
orphan pruning, capture/restore — and the whole state after every step:
the vertices/edges/faces lists IN ORDER, each object's links, the weld
registry and the displaced queue. Objects are named by creation order,
which is their arena index in Rust. Committed; rerun only to widen:

    python scripts/golden/make_mesh_fixtures.py

Deleted with the Python app at M10.
"""
from __future__ import annotations

import json
import random
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.mesh import Mesh, _key  # noqa: E402
from core.vecmath import QVector3D as V  # noqa: E402

OUT = ROOT / "rust" / "crates" / "ingetrazo-model" / "tests" / "data"


def h32(x: float) -> str:
    return struct.pack(">f", x).hex()


def pbits(p) -> list:
    return [h32(p.x()), h32(p.y()), h32(p.z())]


class Names:
    """Creation-order ordinals for vertices, edges and faces."""

    def __init__(self) -> None:
        self.v, self.e, self.f = {}, {}, {}
        self.keep = []                       # objects stay alive: ids stay unique

    def scan(self, mesh) -> None:
        for table, items in ((self.v, mesh.vertices), (self.e, mesh.edges), (self.f, mesh.faces)):
            for o in items:
                if id(o) not in table:
                    table[id(o)] = len(table)
                    self.keep.append(o)


def state(mesh, n: Names) -> dict:
    return {
        "v": [[n.v[id(v)], *pbits(v.position)] for v in mesh.vertices],
        "e": [[n.e[id(e)], n.v[id(e.v0)], n.v[id(e.v1)], [n.f[id(f)] for f in e.faces],
               sorted(n.e[id(x)] for x in e.v0.edges), sorted(n.e[id(x)] for x in e.v1.edges)]
              for e in mesh.edges],
        "f": [[n.f[id(f)], [n.v[id(v)] for v in f.loop], [[n.v[id(v)] for v in h] for h in f.hole_loops]]
              for f in mesh.faces],
        "reg": sorted([*k, n.v[id(v)]] for k, v in mesh._registry.items()),
        "disp": sorted([*k, [n.v[id(v)] for v in vs]] for k, vs in mesh._displaced.items()),
    }


class Gen:
    def __init__(self, seed: int) -> None:
        self.r = random.Random(seed)

    def coord(self) -> float:
        c = self.r.randint(0, 4) * 0.5
        roll = self.r.random()
        if roll < 0.08:
            c += self.r.choice([1e-5, -1e-5, 4e-5, -4.9e-5, 5e-5])   # welds, probes, boundaries
        elif roll < 0.12:
            c += self.r.uniform(-0.2, 0.2)
        return c

    def point(self, flat: bool) -> V:
        return V(self.coord(), self.coord(), 0.0 if flat else self.coord())

    def quad(self) -> list:
        x, y = self.r.randint(0, 3) * 0.5, self.r.randint(0, 3) * 0.5
        s = self.r.choice([0.5, 1.0])
        z = self.r.choice([0.0, 0.0, 1.0])
        pts = [V(x, y, z), V(x + s, y, z), V(x + s, y + s, z), V(x, y + s, z)]
        if self.r.random() < 0.5:
            pts.reverse()
        return pts


def sequence(g: Gen, steps: int) -> list:
    mesh, n = Mesh(), Names()
    removed_f, removed_e, snaps, out = [], [], [], []
    for _ in range(steps):
        r = g.r.random()
        op = None
        if r < 0.24 or not mesh.faces:
            q = g.quad()
            holes = []
            if g.r.random() < 0.2:
                c = (q[0] + q[2]) * 0.5
                d = 0.1
                holes = [[c + V(-d, -d, 0), c + V(d, -d, 0), c + V(d, d, 0), c + V(-d, d, 0)]]
            op = ["add_face", [pbits(p) for p in q], [[pbits(p) for p in h] for h in holes]]
            mesh.add_face(q, holes)
        elif r < 0.34:
            a, b = g.point(g.r.random() < 0.5), g.point(g.r.random() < 0.5)
            op = ["add_edge", pbits(a), pbits(b)]
            try:
                mesh.add_edge(a, b)
            except ValueError:
                op.append("degenerate")
        elif r < 0.42 and mesh.faces:
            f = g.r.choice(mesh.faces)
            op = ["remove_face", n.f[id(f)]]
            mesh.remove_face(f)
            removed_f.append(f)
        elif r < 0.47 and mesh.edges:
            e = g.r.choice(mesh.edges)
            op = ["remove_edge", n.e[id(e)]]
            mesh.remove_edge(e)
            removed_e.append(e)
        elif r < 0.52 and removed_f:
            f = g.r.choice(removed_f)
            op = ["relink_face", n.f[id(f)]]
            mesh.relink_face(f)
        elif r < 0.55 and removed_e:
            e = g.r.choice(removed_e)
            other = mesh.find_edge(e.v0, e.v1)
            if other is not None and other is not e:
                # Two edges on one vertex pair: find_edge then answers
                # from a Python set, in address order — no run repeats
                # it, so no fixture can hold it. Undo never does this.
                continue
            op = ["relink_edge", n.e[id(e)]]
            mesh.relink_edge(e)
        elif r < 0.65 and mesh.edges:
            e = g.r.choice(mesh.edges)
            t = g.r.choice([0.5, 0.25, 0.0])
            p = e.v0.position * (1 - t) + e.v1.position * t
            op = ["split_edge", n.e[id(e)], pbits(p)]
            mesh.split_edge(e, p)
        elif r < 0.75 and mesh.vertices:
            v = g.r.choice(mesh.vertices)
            if g.r.random() < 0.4 and len(mesh.vertices) > 1:
                w = g.r.choice(mesh.vertices)
                d = w.position - v.position               # onto another vertex
            else:
                d = V(g.r.choice([0.5, -0.5, 0.0, 3e-5]), g.r.choice([0.5, 0.0]), g.r.choice([0.0, 1.0]))
            op = ["move_vertex", n.v[id(v)], pbits(d)]
            mesh.move_vertex(v, d)
        elif r < 0.80 and mesh.vertices:
            v = g.r.choice(mesh.vertices)
            p = g.point(False)
            op = ["place_vertex", n.v[id(v)], pbits(p)]
            mesh.place_vertex(v, p)
        elif r < 0.84 and mesh.faces:
            f = g.r.choice(mesh.faces)
            c = f.loop[0].position * 0.5 + f.loop[2 % len(f.loop)].position * 0.5
            d = 0.05
            h = [c + V(-d, -d, 0), c + V(d, -d, 0), c + V(d, d, 0), c + V(-d, d, 0)]
            op = ["add_hole", n.f[id(f)], [pbits(p) for p in h]]
            mesh.add_hole(f, h)
        elif r < 0.87 and any(f.hole_loops for f in mesh.faces):
            f = g.r.choice([f for f in mesh.faces if f.hole_loops])
            i = g.r.randrange(len(f.hole_loops))
            op = ["remove_hole", n.f[id(f)], i]
            mesh.remove_hole(f, f.hole_loops[i])
        elif r < 0.90:
            op = ["prune_orphan_vertices", mesh.prune_orphan_vertices()]
        elif r < 0.92 and mesh.vertices:
            vs = g.r.sample(mesh.vertices, min(3, len(mesh.vertices)))
            op = ["drop_orphan_vertices", [n.v[id(v)] for v in vs], mesh.drop_orphan_vertices(vs)]
        elif r < 0.96:
            snaps.append(mesh.capture_state())
            op = ["capture", len(snaps) - 1]
        elif snaps:
            i = g.r.randrange(len(snaps))
            op = ["restore", i]
            mesh.restore_state(snaps[i])
        if op is None:
            continue
        n.scan(mesh)
        out.append({"op": op, "state": state(mesh, n)})
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    g = Gen(20261020)
    with open(OUT / "mesh_ops.jsonl", "w", encoding="ascii", newline="\n") as fh:
        for _ in range(160):
            fh.write(json.dumps(sequence(g, g.r.randint(5, 30)), separators=(",", ":")) + "\n")
    (OUT / ".gitattributes").write_text("# Bits Python produced; compared exactly.\n* -text\n")
    print("written", (OUT / "mesh_ops.jsonl").stat().st_size, "bytes")


if __name__ == "__main__":
    main()
