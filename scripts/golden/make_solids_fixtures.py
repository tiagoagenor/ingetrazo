# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixtures that hold ``ingetrazo-ops``' solid booleans to ``core.solids``
(docs/rust-rewrite-plan.md, M3.8).

Pairs (and a few triples) of closed solids — boxes that overlap, touch,
share a face, coincide or sit apart; a rotated component against a classic
group; a cylinder with soft seams; one solid nested in another (a void);
a tunnel block; a concave prism; painted containers and coloured faces;
open and single-pick inputs that must raise — go through the REAL
``core.solids.run`` with the native engine off. Each scene is

* saved as a bare-JSON .igz (``<scene>.igz.json``) and reopened in a fresh
  scene, so Python and Rust start from the very same bits, and
* run once per operation: ``cases.json`` lists the cases, and each case's
  result holds the input solids' report (``is_solid``, volume bits), the
  groups removed (as indexes into the picks), and for every created group
  its name, layer, the L2 mesh signature hash and the whole mesh state in
  order (vertices, edges with their soft flag and faces, faces with their
  loops and attrs hash).

Committed; rerun only to widen (PYTHONPATH at the repository root):

    python scripts/golden/make_solids_fixtures.py

Deleted with the Python app at M10.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.pop("INGETRAZO_RUST", None)

from PySide6.QtGui import QGuiApplication, QMatrix4x4, QVector3D as V  # noqa: E402

from core import solids  # noqa: E402
from core.group import Group  # noqa: E402
from core.mesh import Mesh  # noqa: E402
from core.scene import Scene  # noqa: E402
from formats import igz  # noqa: E402
from tests.golden.signature import (_dumps, canonical, mesh_signature,  # noqa: E402
                                    sha)

OUT = ROOT / "rust" / "crates" / "ingetrazo-ops" / "tests" / "data" / "solids"

_APP = QGuiApplication.instance() or QGuiApplication([])


def h32(x: float) -> str:
    return struct.pack(">f", x).hex()


def pbits(p) -> list:
    return [h32(p.x()), h32(p.y()), h32(p.z())]


def f64bits(x: float) -> str:
    return struct.pack(">d", x).hex()


# ---- solids ---------------------------------------------------------------------

def box_faces(m: Mesh, x, y, z, sx, sy, sz, colors=None, skip=()):
    """Six outward faces; ``colors`` maps a side name to a colour."""
    x1, y1, z1 = x + sx, y + sy, z + sz
    sides = {
        "bottom": [V(x, y, z), V(x, y1, z), V(x1, y1, z), V(x1, y, z)],
        "top": [V(x, y, z1), V(x1, y, z1), V(x1, y1, z1), V(x, y1, z1)],
        "front": [V(x, y, z), V(x1, y, z), V(x1, y, z1), V(x, y, z1)],
        "back": [V(x, y1, z), V(x, y1, z1), V(x1, y1, z1), V(x1, y1, z)],
        "left": [V(x, y, z), V(x, y, z1), V(x, y1, z1), V(x, y1, z)],
        "right": [V(x1, y, z), V(x1, y1, z), V(x1, y1, z1), V(x1, y, z1)],
    }
    out = {}
    for name, pts in sides.items():
        if name in skip:
            continue
        f = m.add_face(pts)
        if colors and name in colors:
            f.attrs["color"] = list(colors[name])
        out[name] = f
    return out


def box(x, y, z, sx, sy, sz, colors=None) -> Mesh:
    m = Mesh()
    box_faces(m, x, y, z, sx, sy, sz, colors)
    return m


def open_box(x, y, z, s) -> Mesh:
    m = Mesh()
    box_faces(m, x, y, z, s, s, s, skip=("top",))
    return m


def cylinder(cx, cy, z0, r, h, n=16, colors=None) -> Mesh:
    m = Mesh()
    ang = [2 * math.pi * i / n for i in range(n)]
    ring = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in ang]
    m.add_face([V(px, py, z0) for px, py in reversed(ring)])
    m.add_face([V(px, py, z0 + h) for px, py in ring])
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        f = m.add_face([V(a[0], a[1], z0), V(b[0], b[1], z0),
                        V(b[0], b[1], z0 + h), V(a[0], a[1], z0 + h)])
        if colors:
            f.attrs["color"] = list(colors)
    for e in m.edges:
        a, b = e.v0.position, e.v1.position
        if abs(a.x() - b.x()) + abs(a.y() - b.y()) < 1e-6 and a.z() != b.z():
            e.soft = True          # the vertical seams of the curved side
    return m


def tunnel(x, y, z, s, hole, colors=None) -> Mesh:
    """A cube with a square hole through it along z (faces with holes)."""
    m = Mesh()
    x1, y1, z1 = x + s, y + s, z + s
    hx0, hy0, hx1, hy1 = x + hole[0], y + hole[1], x + hole[2], y + hole[3]
    top = m.add_face([V(x, y, z1), V(x1, y, z1), V(x1, y1, z1), V(x, y1, z1)],
                     [[V(hx0, hy0, z1), V(hx0, hy1, z1), V(hx1, hy1, z1),
                       V(hx1, hy0, z1)]])
    bot = m.add_face([V(x, y, z), V(x, y1, z), V(x1, y1, z), V(x1, y, z)],
                     [[V(hx0, hy0, z), V(hx1, hy0, z), V(hx1, hy1, z),
                       V(hx0, hy1, z)]])
    if colors:
        top.attrs["color"] = list(colors)
    m.add_face([V(x, y, z), V(x1, y, z), V(x1, y, z1), V(x, y, z1)])
    m.add_face([V(x, y1, z), V(x, y1, z1), V(x1, y1, z1), V(x1, y1, z)])
    m.add_face([V(x, y, z), V(x, y, z1), V(x, y1, z1), V(x, y1, z)])
    m.add_face([V(x1, y, z), V(x1, y1, z), V(x1, y1, z1), V(x1, y, z1)])
    # the tunnel walls, facing into the hole
    m.add_face([V(hx0, hy0, z), V(hx0, hy0, z1), V(hx1, hy0, z1), V(hx1, hy0, z)])
    m.add_face([V(hx1, hy0, z), V(hx1, hy0, z1), V(hx1, hy1, z1), V(hx1, hy1, z)])
    m.add_face([V(hx1, hy1, z), V(hx1, hy1, z1), V(hx0, hy1, z1), V(hx0, hy1, z)])
    m.add_face([V(hx0, hy1, z), V(hx0, hy1, z1), V(hx0, hy0, z1), V(hx0, hy0, z)])
    return m


def l_prism(x, y, z, h) -> Mesh:
    """An L-shaped (concave) prism."""
    pts = [(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)]
    m = Mesh()
    m.add_face([V(x + px, y + py, z) for px, py in reversed(pts)])
    m.add_face([V(x + px, y + py, z + h) for px, py in pts])
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        m.add_face([V(x + a[0], y + a[1], z), V(x + b[0], y + b[1], z),
                    V(x + b[0], y + b[1], z + h), V(x + a[0], y + a[1], z + h)])
    return m


def wall_with_doors() -> Mesh:
    """A wall slab with two door openings reaching the floor, their tops at
    one height: the triangulation of such a face can leave a corner on a
    triangle edge (the T-junction case ``core.solids`` conforms)."""
    m = Mesh()
    L, T, H, top = 6.0, 0.4, 3.0, 2.1
    doors = ((1.0, 2.0), (3.5, 4.5))
    front = [V(0, 0, 0)]
    for a, b in doors:
        front += [V(a, 0, 0), V(a, 0, top), V(b, 0, top), V(b, 0, 0)]
    front += [V(L, 0, 0), V(L, 0, H), V(0, 0, H)]
    m.add_face(front)
    m.add_face([V(p.x(), T, p.z()) for p in reversed(front)])
    # the floor between the openings, the roof and the two ends
    xs = [0.0, 1.0, 2.0, 3.5, 4.5, L]
    for x0, x1 in ((xs[0], xs[1]), (xs[2], xs[3]), (xs[4], xs[5])):
        m.add_face([V(x0, 0, 0), V(x0, T, 0), V(x1, T, 0), V(x1, 0, 0)])
    m.add_face([V(0, 0, H), V(L, 0, H), V(L, T, H), V(0, T, H)])
    m.add_face([V(0, 0, 0), V(0, 0, H), V(0, T, H), V(0, T, 0)])
    m.add_face([V(L, 0, 0), V(L, T, 0), V(L, T, H), V(L, 0, H)])
    # the reveals of the openings, facing into them
    for a, b in doors:
        m.add_face([V(a, 0, 0), V(a, T, 0), V(a, T, top), V(a, 0, top)])
        m.add_face([V(b, 0, 0), V(b, 0, top), V(b, T, top), V(b, T, 0)])
        m.add_face([V(a, 0, top), V(a, T, top), V(b, T, top), V(b, 0, top)])
    return m


def wall_with_windows(holes, L=8.0, T=0.4, H=3.0) -> Mesh:
    """A slab whose front and back faces have rectangular windows as holes
    (``(x0, x1, z0, z1)``), tops at one height among them: the earcut of
    such a face leaves corners on triangle edges, so Manifold refuses the
    triangles until ``core.solids`` conforms them (the T-junction case)."""
    m = Mesh()
    front = [V(0, 0, 0), V(L, 0, 0), V(L, 0, H), V(0, 0, H)]
    back = [V(0, T, 0), V(0, T, H), V(L, T, H), V(L, T, 0)]
    m.add_face(front, [[V(a, 0, z0), V(a, 0, z1), V(b, 0, z1), V(b, 0, z0)]
                       for (a, b, z0, z1) in holes])
    m.add_face(back, [[V(a, T, z0), V(b, T, z0), V(b, T, z1), V(a, T, z1)]
                      for (a, b, z0, z1) in holes])
    m.add_face([V(0, 0, 0), V(0, T, 0), V(L, T, 0), V(L, 0, 0)])
    m.add_face([V(0, 0, H), V(L, 0, H), V(L, T, H), V(0, T, H)])
    m.add_face([V(0, 0, 0), V(0, 0, H), V(0, T, H), V(0, T, 0)])
    m.add_face([V(L, 0, 0), V(L, T, 0), V(L, T, H), V(L, 0, H)])
    for (a, b, z0, z1) in holes:
        m.add_face([V(a, 0, z0), V(a, T, z0), V(a, T, z1), V(a, 0, z1)])
        m.add_face([V(b, 0, z0), V(b, 0, z1), V(b, T, z1), V(b, T, z0)])
        m.add_face([V(a, 0, z0), V(b, 0, z0), V(b, T, z0), V(a, T, z0)])
        m.add_face([V(a, 0, z1), V(a, T, z1), V(b, T, z1), V(b, 0, z1)])
    return m


def hollow(x, y, z, s, wall) -> Mesh:
    """A cube with a closed void inside (the inner shell faces inward)."""
    m = Mesh()
    box_faces(m, x, y, z, s, s, s)
    inner = Mesh()
    box_faces(inner, x + wall, y + wall, z + wall, s - 2 * wall, s - 2 * wall, s - 2 * wall)
    for f in inner.faces:
        m.add_face(list(reversed(f.vertices)))
    return m


def textured_box(x, y, z, s) -> Mesh:
    """A box whose faces carry textures with their fitted world to UV maps
    (one repeated, so coplanar triangles of a face may fuse, one mapped
    differently), plus a colour and a named material."""
    m = Mesh()
    faces = box_faces(m, x, y, z, s, s, s)
    faces["top"].attrs["texture"] = {
        "path": "tex/wood.png", "sw": 1.0, "sh": 1.0, "rot": 0,
        "uvw": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]}
    faces["front"].attrs["texture"] = {
        "path": "tex/brick.png", "sw": 0.5, "sh": 0.5,
        "uvw": [0.5, 0.0, 0.0, 0.25, 0.0, 0.0, 0.5, 0.125]}
    faces["left"].attrs["color"] = [0.2, 0.4, 0.6]
    faces["left"].attrs["mat"] = "Concreto"
    return m


def xform(dx=0.0, dy=0.0, dz=0.0, angle=0.0, axis=(0, 0, 1)):
    m = QMatrix4x4()
    m.translate(dx, dy, dz)
    if angle:
        m.rotate(angle, *axis)
    return m


def group(mesh, name, layer=None, material=None, m=None) -> Group:
    g = Group(mesh, name=name)
    if layer:
        g.layer = layer
    if material:
        g.material = dict(material)
    if m is not None:
        g.xform = m
    return g


RED, GREEN, BLUE = (0.9, 0.1, 0.1), (0.1, 0.8, 0.2), (0.2, 0.3, 0.9)


# ---- scenes ----------------------------------------------------------------------

def scenes() -> dict:
    out = {}

    out["overlap"] = [
        group(box(0, 0, 0, 2, 2, 2, {"top": RED, "front": GREEN}), "A", "Layer 0"),
        group(box(1, 1, 1, 2, 2, 2, {"bottom": BLUE, "left": RED}), "B", "Layer 1"),
    ]
    out["touch_face"] = [
        group(box(0, 0, 0, 1, 1, 1, {"right": RED}), "A"),
        group(box(1, 0, 0, 1, 1, 1, {"left": BLUE}), "B"),
    ]
    out["touch_partial"] = [
        group(box(0, 0, 0, 2, 2, 1, {"top": RED}), "A"),
        group(box(0.5, 0.5, 1, 1, 1, 1, {"bottom": BLUE}), "B"),
    ]
    out["disjoint"] = [
        group(box(0, 0, 0, 1, 1, 1), "A"),
        group(box(5, 5, 5, 1, 1, 1), "B"),
    ]
    out["coincident"] = [
        group(box(0, 0, 0, 1, 1, 1, {"top": RED}), "A"),
        group(box(0, 0, 0, 1, 1, 1, {"top": BLUE}), "B"),
    ]
    proto = box(0, 0, 0, 2, 1, 1, {"front": RED, "top": GREEN})
    out["rotated"] = [
        group(proto, "comp1", "Layer 0", m=xform(0.5, 0.2, 0.0, 30.0)),
        group(box(0.5, -0.5, 0.2, 1, 3, 0.6, {"left": BLUE}), "B"),
        group(proto, "comp2", None, m=xform(4, 0, 0, 90.0, (0, 1, 0))),
    ]
    out["rotated_tilt"] = [
        group(box(0, 0, 0, 2, 2, 2, {"top": RED}), "A", m=xform(0, 0, 0, 45.0, (1, 1, 0))),
        group(box(0.2, 0.2, 0.2, 1.5, 1.5, 1.5, {"front": BLUE}), "B",
              m=xform(0.3, 0.1, 0.2, 20.0, (0, 0, 1))),
    ]
    out["cylinder"] = [
        group(cylinder(1, 1, -1, 0.5, 4, 16, GREEN), "cyl", "Layer 2"),
        group(box(0, 0, 0, 2, 2, 2, {"top": RED}), "B", "Layer 3"),
    ]
    out["cylinder_cylinder"] = [
        group(cylinder(0, 0, 0, 1, 2, 12), "c1"),
        group(cylinder(0.8, 0.3, 0.5, 1, 2, 12, BLUE), "c2"),
    ]
    out["nested"] = [
        group(box(0, 0, 0, 3, 3, 3, {"top": RED}), "outer"),
        group(box(1, 1, 1, 1, 1, 1, {"top": BLUE}), "inner"),
    ]
    out["tunnel"] = [
        group(tunnel(0, 0, 0, 3, (1, 1, 2, 2), RED), "tun"),
        group(box(-1, 1.2, 1, 5, 0.6, 1, {"top": GREEN}), "bar"),
    ]
    out["tunnel_plug"] = [
        group(tunnel(0, 0, 0, 3, (1, 1, 2, 2)), "tun"),
        group(box(1, 1, -1, 1, 1, 5, {"front": BLUE}), "plug"),
    ]
    out["concave"] = [
        group(l_prism(0, 0, 0, 1.5), "L", "Layer 4"),
        group(box(0.5, 0.5, 0.5, 2, 2, 1.5, {"top": BLUE}), "B"),
    ]
    out["painted"] = [
        group(box(0, 0, 0, 2, 2, 2, {"top": RED}), "A", material={"color": [0.2, 0.6, 0.2]}),
        group(box(1, 1, 1, 2, 2, 2), "B", material={"color": [0.9, 0.9, 0.1], "opacity": 0.5}),
    ]
    out["three"] = [
        group(box(0, 0, 0, 2, 2, 2, {"top": RED}), "A"),
        group(box(1, 1, 0, 2, 2, 2, {"top": GREEN}), "B"),
        group(box(0.5, 0.5, 0.5, 2, 2, 2, {"bottom": BLUE}), "C"),
    ]
    out["doors"] = [
        group(wall_with_doors(), "wall"),
        group(box(0.5, -0.5, 1.0, 5, 1.0, 0.5, {"top": BLUE}), "beam"),
    ]
    out["windows"] = [
        group(wall_with_windows([(1, 2, 0.9, 2.1), (3.5, 4.5, 0.9, 2.1)]), "wall", "Layer 6"),
        group(box(0.5, -0.5, 1.0, 5, 1.0, 0.5, {"top": BLUE}), "beam"),
    ]
    out["windows_grid"] = [
        group(wall_with_windows([(1, 2, 0.5, 1.2), (1, 2, 1.6, 2.4), (3.5, 4.5, 0.5, 1.2),
                                 (3.5, 4.5, 1.6, 2.4)]), "wall",
              m=xform(1.0, 2.0, 0.0, 15.0)),
        group(box(1.2, 2.2, 0.8, 4, 1.5, 1.2), "block"),
    ]
    out["cylinder_big"] = [
        group(cylinder(0, 0, 0, 1, 2, 128, GREEN), "cyl", m=xform(0.2, 0.1, 0.0, 25.0, (1, 0, 0))),
        group(box(-0.5, -0.5, 0.5, 2, 2, 1, {"top": RED}), "B"),
    ]
    out["textured"] = [
        group(textured_box(0, 0, 0, 2), "tex", m=xform(0.3, 0.2, 0.0, 33.0, (0, 1, 1))),
        group(box(0.5, 0.5, 0.5, 2, 2, 2, {"left": BLUE}), "B"),
    ]
    out["hollow"] = [
        group(hollow(0, 0, 0, 3, 0.5), "hollow", "Layer 5"),
        group(box(2, 2, 2, 2, 2, 2, {"top": BLUE}), "B"),
    ]
    out["open"] = [
        group(open_box(0, 0, 0, 2), "open"),
        group(box(1, 1, 1, 2, 2, 2), "B"),
    ]
    return out


# ---- recording ---------------------------------------------------------------------

def mesh_state(mesh) -> dict:
    vi = {id(v): i for i, v in enumerate(mesh.vertices)}
    ei = {id(e): i for i, e in enumerate(mesh.edges)}
    fi = {id(f): i for i, f in enumerate(mesh.faces)}
    return {
        "v": [pbits(v.position) for v in mesh.vertices],
        "e": [[vi[id(e.v0)], vi[id(e.v1)], 1 if e.soft else 0,
               [fi[id(f)] for f in e.faces]] for e in mesh.edges],
        "f": [[[vi[id(v)] for v in f.loop],
               [[vi[id(v)] for v in h] for h in f.hole_loops],
               1 if getattr(f, "interior", False) else 0,
               sha(_dumps(canonical(f.attrs)), 16) if f.attrs else ""]
              for f in mesh.faces],
    }
    del ei


def group_state(g) -> dict:
    return {
        "name": g.name,
        "layer": g.layer,
        "sig": sha(_dumps(mesh_signature(g.mesh)), 32),
        "mesh": mesh_state(g.mesh),
    }


def report(g) -> dict:
    vol = solids.solid_volume(g)
    return {"solid": vol is not None, "volume": None if vol is None else f64bits(vol)}


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


#: (scene, op, picks) — subtract/trim: the first pick cuts the second.
def case_list(names) -> list:
    cases = []
    for name in names:
        n = len(scenes_cache[name])
        if name == "open":
            cases.append((name, solids.UNION, [0, 1]))
            cases.append((name, solids.UNION, [1, 0]))
            continue
        if name == "three":
            for op in (solids.UNION, solids.OUTER_SHELL, solids.INTERSECT):
                cases.append((name, op, [0, 1, 2]))
            for op in solids.OPS:
                cases.append((name, op, [0, 1]))
            cases.append((name, solids.SUBTRACT, [2, 0, 1]))
            continue
        for op in solids.OPS:
            cases.append((name, op, [0, 1]))
        for op in (solids.SUBTRACT, solids.TRIM, solids.SPLIT):
            cases.append((name, op, [1, 0]))
        if name == "rotated":
            cases.append((name, solids.UNION, [0, 2]))
            cases.append((name, solids.UNION, [0, 1, 2]))
            cases.append((name, solids.SUBTRACT, [2, 1]))
        del n
    cases.append(("overlap", solids.UNION, [0]))
    cases.append(("overlap", solids.UNION, []))
    return cases


scenes_cache: dict = {}
CONFORMED: list = []
_conform = solids._conform_t_junctions


def _spy(*args, **kwargs):
    CONFORMED.append(1)
    return _conform(*args, **kwargs)


solids._conform_t_junctions = _spy


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    scenes_cache.update(scenes())
    opened: dict = {}
    for name, groups in scenes_cache.items():
        s = Scene()
        s.groups.extend(groups)
        path = OUT / f"{name}.igz.json"
        save_scene_json(s, path)
        opened[name] = reopen(path)
    out_cases = []
    for name, op, picks in case_list(list(scenes_cache)):
        scene = opened[name]
        groups = [scene.groups[i] for i in picks]
        rec = {"scene": name, "op": op, "picks": picks,
               "inputs": [report(g) for g in groups]}
        CONFORMED.clear()
        try:
            removed, created = solids.run(op, groups)
        except solids.SolidError as e:
            rec["error"] = str(e)
        else:
            rec["removed"] = [groups.index(g) for g in removed]
            rec["created"] = [group_state(g) for g in created]
        if CONFORMED:
            rec["conformed"] = True
        out_cases.append(rec)
    (OUT / "cases.json").write_text(
        json.dumps(out_cases, sort_keys=True, separators=(",", ":")),
        encoding="utf-8", newline="\n")
    print(len(out_cases), "cases")


if __name__ == "__main__":
    main()
