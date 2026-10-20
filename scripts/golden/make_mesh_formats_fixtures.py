# SPDX-License-Identifier: GPL-3.0-or-later
"""Write the fixtures that hold the Rust `ingetrazo-formats-mesh` crate to
Python's OBJ and STL (docs/rust-rewrite-plan.md, M5.1 and M5.2).

Scenes are built with the real app code (loose faces with holes, groups,
components with rotated and scaled transforms, nested placements, materials,
colours, textures, hidden things). Each one is

* saved as a bare-JSON .igz (format 1, textures kept as paths relative to the
  fixture folder) for the Rust test to open, and
* exported with the REAL exporters: ``<scene>.obj`` + ``.mtl`` and
  ``<scene>.<unit>.stl`` for several units.

Import fixtures are the exported files plus hand-written odd ones (quads,
n-gons, negative indices, missing normals, solid/endsolid variants, bad
input). Next to each goes the L2 signature Python's scene has after the
import, computed with the REAL ``run_stitch`` and ``orient_outward`` (Python's
unrepeatable set order is pinned as ``_pin_order.py`` says); the Rust import
runs the same engine and must give the same signature.
``import/cases.json`` lists them with the options and, for bad input, the
message Python raised.

The files are committed, so `cargo test` needs no Python; rerun this only to
widen them (PYTHONPATH at the repository root, native engine OFF):

    python scripts/golden/make_mesh_formats_fixtures.py

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
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.pop("INGETRAZO_RUST", None)

import _pin_order  # noqa: E402
from PySide6.QtGui import QGuiApplication, QImage, QMatrix4x4, QVector3D as V  # noqa: E402

from core.group import Group  # noqa: E402
from core.layers import Layer  # noqa: E402
from core.mesh import Mesh  # noqa: E402
from core.scene import Scene  # noqa: E402
from formats import igz, obj, stl  # noqa: E402
from tests.golden.signature import scene_signature  # noqa: E402

OUT = ROOT / "rust" / "crates" / "ingetrazo-formats-mesh" / "tests" / "data"


def xform(dx=0.0, dy=0.0, dz=0.0, angle=0.0, axis=(0, 0, 1), scale=None):
    m = QMatrix4x4()
    m.translate(dx, dy, dz)
    if angle:
        m.rotate(angle, *axis)
    if scale is not None:
        m.scale(*scale) if isinstance(scale, tuple) else m.scale(scale)
    return m


def png(path: Path, color: int, seed: int) -> Path:
    img = QImage(4, 4, QImage.Format_RGB32)
    img.fill(color)
    img.setPixel(seed % 4, (seed // 4) % 4, 0x336699)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert img.save(str(path))
    return path


def grid(n: int, size: float = 1.0, z=lambda i, j: 0.0) -> Mesh:
    """n x n quads, a few with a coloured face: a mesh heavy enough that a
    transformed copy takes the bulk (vectorised) path."""
    m = Mesh()
    for i in range(n):
        for j in range(n):
            x0, y0 = i * size, j * size
            f = m.add_face([V(x0, y0, z(i, j)), V(x0 + size, y0, z(i + 1, j)),
                            V(x0 + size, y0 + size, z(i + 1, j + 1)),
                            V(x0, y0 + size, z(i, j + 1))])
            if (i + j) % 7 == 0:
                f.attrs["color"] = [0.8, 0.2 + 0.01 * i, 0.1]
    return m


# ---- scenes -------------------------------------------------------------------

def scene_faces(tex: Path) -> Scene:
    """Loose faces: holes, concave, colours, a named material, hidden bits."""
    s = Scene()
    s.layers.append(Layer("Oculta", visible=False))
    m = s.mesh
    m.add_face([V(0, 0, 0), V(3, 0, 0), V(3, 2.5, 0), V(0, 2.5, 0)],
               [[V(1, 1, 0), V(1.4, 1, 0), V(1.4, 1.3, 0), V(1, 1.3, 0)],
                [V(2, 1, 0), V(2.5, 1, 0), V(2.5, 1.5, 0)]])
    m.add_face([V(0, 0, 0), V(3, 0, 0), V(3, 0, 1.2), V(0, 0, 1.2)])
    # L-shaped (concave): earcut.
    m.add_face([V(5, 0, 0), V(8, 0, 0), V(8, 1, 0), V(6, 1, 0), V(6, 3, 0),
                V(5, 3, 0)])
    red = m.add_face([V(10, 0, 0), V(11, 0, 0), V(11, 1, 0), V(10, 1, 0)])
    red.attrs["color"] = [0.9, 0.1, 0.1]
    ints = m.add_face([V(12, 0, 0), V(13, 0, 0), V(13, 1, 0)])
    ints.attrs["color"] = [0, 1, 0]
    rgba = m.add_face([V(14, 0, 0), V(15, 0, 0), V(15, 1, 0)])
    rgba.attrs["color"] = [0.25, 0.5, 0.75, 1.0]
    named = m.add_face([V(16, 0, 0), V(17, 0, 0), V(17, 1, 0)])
    named.attrs.update({"color": [0.5, 0.4, 0.3], "mat": "Concreto visto 1"})
    named2 = m.add_face([V(18, 0, 0), V(19, 0, 0), V(19, 1, 0)])
    named2.attrs.update({"color": [0.1, 0.4, 0.3], "mat": "Concreto/visto 1"})
    first = m.add_face([V(20, 0, 0), V(21, 0, 0), V(21, 1, 0)])
    first.attrs.update({"color": [0.5, 0.4, 0.3]})        # same colour as named
    # Collinear corners: zero-area triangles are left out.
    m.add_face([V(0, 10, 0), V(1, 10, 0), V(2, 10, 0), V(2, 11, 0)])
    m.add_face([V(0, 12, 0), V(1, 12, 0), V(2, 12, 0)])        # all slivers
    m.add_face([V(5, 12, 0), V(5.00002, 12, 0), V(5, 12.00002, 0)])  # tiny
    # Slanted, tilted and mirrored normals.
    m.add_face([V(0, 20, 0), V(2, 20, 0), V(2, 21, 1), V(0, 21, 1)])
    m.add_face([V(0, 30, 0), V(0, 30, 1), V(1, 31, 1), V(1, 31, 0)])
    m.add_face([V(0.1, 0.2, 0.3), V(1 / 3, 2 / 3, 0.7), V(-0.001, 1e-6, 5.5)])
    # Hidden face, a face on a hidden layer: never exported.
    h = m.add_face([V(30, 0, 0), V(31, 0, 0), V(31, 1, 0)])
    h.attrs["hidden"] = True
    lay = m.add_face([V(32, 0, 0), V(33, 0, 0), V(33, 1, 0)])
    lay.attrs["layer"] = "Oculta"
    m.add_edge(V(40, 0, 0), V(41, 0, 0))
    return s


def scene_textured(tex: Path) -> Scene:
    """Planar and fitted textures, a rotated one, a named textured material."""
    s = Scene()
    m = s.mesh
    wood = tex / "wood.png"
    brick = tex / "ladrillo rústico.png"
    a = m.add_face([V(0, 0, 0), V(2, 0, 0), V(2, 2, 0), V(0, 2, 0)])
    a.attrs["texture"] = {"path": str(wood), "sw": 1.5, "sh": 0.75}
    b = m.add_face([V(3, 0, 0), V(5, 0, 0), V(5, 2, 0)])
    b.attrs["texture"] = {"path": str(wood), "sw": 1.5, "sh": 0.75, "rot": 30.0}
    c = m.add_face([V(6, 0, 0), V(8, 0, 0), V(8, 0, 2), V(6, 0, 2)])
    c.attrs["texture"] = {"path": str(wood), "sw": 0.0, "sh": 2.0}
    d = m.add_face([V(9, 0, 0), V(11, 0, 0), V(11, 2, 0), V(9, 2, 0)])
    d.attrs["texture"] = {"path": str(brick), "sw": 1.0, "sh": 1.0,
                          "uvw": [0.5, 0.0, 0.0, 0.25, 0.0, 0.5, 0.0, 0.125]}
    d.attrs["mat"] = "Ladrillo"
    e = m.add_face([V(12, 0, 0), V(14, 0, 0), V(14, 1, 1), V(12, 1, 1)])
    e.attrs["texture"] = {"path": str(brick), "sw": 2.0, "sh": 1.0,
                          "rot": 12.5}
    e.attrs["mat"] = "Pared"
    f = m.add_face([V(15, 0, 0), V(16, 0, 0), V(16, 1, 0)])
    f.attrs["color"] = [0.2, 0.2, 0.9]
    g = m.add_face([V(17, 0, 0), V(18, 0, 0), V(18, 1, 0)],
                   [])
    g.attrs["texture"] = {"path": str(wood), "sw": 1.0, "sh": 1.0}
    g.attrs["mat"] = "Madera"
    return s


def scene_components(tex: Path) -> Scene:
    """Components sharing a prototype with rotated and scaled transforms, a
    classic group, a group owning placements, hidden and face-me objects."""
    s = Scene()
    s.layers.append(Layer("Oculta", visible=False))
    s.mesh.add_face([V(-5, -5, 0), V(-4, -5, 0), V(-4, -4, 0)])
    wood = tex / "wood.png"
    proto = Mesh()
    proto.add_face([V(0, 0, 0), V(1, 0, 0), V(1, 1, 0), V(0, 1, 0)],
                   [[V(0.25, 0.25, 0), V(0.5, 0.25, 0), V(0.5, 0.5, 0)]])
    proto.add_face([V(0, 0, 0), V(1, 0, 0), V(1, 0, 2), V(0, 0, 2)])
    proto.add_face([V(2, 0, 0), V(3, 0, 0), V(3, 1, 0.5)])
    proto.add_edge(V(0, 0, 0), V(0, 0, 1)).curve = 3
    proto.faces[0].attrs["color"] = [0.2, 0.6, 0.2]
    proto.faces[1].attrs["texture"] = {
        "path": str(wood), "sw": 1.0, "sh": 1.0,
        "uvw": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0]}
    proto.faces[2].attrs.update({"color": [0.3, 0.3, 0.8], "mat": "Azul"})
    for k, (dx, ang, ax, sc) in enumerate((
            (0, 0.0, (0, 0, 1), None),
            (5, 37.0, (0, 0, 1), None),
            (10, 61.0, (1, 0.3, 0.2), None),
            (15, 15.0, (0, 1, 0), (1.0, 2.0, 0.5)),
            (20, 0.0, (0, 0, 1), (-1.0, 1.0, 1.0)))):
        g = Group(proto, name=f"Silla {k}")
        g.xform = xform(dx, 20, 0.5, ang, ax, sc)
        s.groups.append(g)
    hidden = Group(proto, name="Oculta")
    hidden.xform = xform(30, 20, 0)
    hidden.hidden = True
    s.groups.append(hidden)
    layered = Group(proto, name="En capa oculta")
    layered.xform = xform(35, 20, 0)
    layered.layer = "Oculta"
    s.groups.append(layered)
    classic = Group(name="Mesa")
    classic.mesh.add_face([V(0, 30, 0), V(2, 30, 0), V(2, 31, 0)])
    classic.mesh.faces[0].attrs["hidden"] = True       # groups ignore it
    classic.mesh.add_face([V(0, 31, 0), V(2, 31, 0), V(2, 32, 0)])
    s.groups.append(classic)
    quad = Group(name="Figura")
    quad.billboard = True
    q = quad.mesh.add_face([V(0, 0, 0), V(1, 0, 0), V(1, 0, 2), V(0, 0, 2)])
    q.attrs["texture"] = {"path": str(wood), "sw": 1.0, "sh": 2.0}
    s.groups.append(quad)
    # A group that owns placements of a second prototype.
    leaf = Mesh()
    leaf.add_face([V(0, 0, 0), V(0.5, 0, 0), V(0.5, 0.5, 0.2)])
    leaf.faces[0].attrs["texture"] = {
        "path": str(wood), "sw": 1.0, "sh": 1.0,
        "uvw": [2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0]}
    parent = Group(name="Conjunto")
    parent.mesh.add_face([V(0, 40, 0), V(3, 40, 0), V(3, 43, 0), V(0, 43, 0)])
    kids = []
    for k in range(2):
        child = Group(leaf, name=f"Hoja {k}")
        child.xform = xform(k, 41, 0, 25.0 * k)
        kids.append(child)
    inner = Group(name="Interno")
    inner.mesh.add_face([V(0, 50, 0), V(1, 50, 0), V(1, 51, 0)])
    kids.append(inner)
    parent.adopt(kids)
    parent.xform = xform(100, 0, 0, 20.0)
    parent.component = False
    s.groups.append(parent)
    return s


def scene_bulk(tex: Path) -> Scene:
    """A prototype heavy enough for the vectorised copy, placed rotated; a
    loose edge in it lands next to a face corner after the scale."""
    s = Scene()
    rng = random.Random(7)
    wood = tex / "wood.png"
    proto = grid(18, 0.37, z=lambda i, j: 0.01 * ((i * 3 + j) % 5))
    # Uneven coordinates so the matrix product rounds.
    for f in proto.faces[::5]:
        f.attrs["texture"] = {"path": str(wood), "sw": 1.0, "sh": 1.0,
                              "uvw": [rng.uniform(-1, 1) for _ in range(8)]}
    for a in range(4):
        g = Group(proto, name=f"Rejilla {a}")
        g.xform = xform(a * 9.1, 0.3 * a, 0.77, 19.0 + 33.0 * a,
                        (0.2, 1.0, 0.1 * a))
        s.groups.append(g)
    return s


def scene_weld(tex: Path) -> Scene:
    """A prototype in millimetres placed at 1:1000: corners 0.04 apart land
    5e-5 apart and weld in the copy (the free edges come first when the copy
    is bulk). One small prototype takes the per-face walk, one big the bulk."""
    s = Scene()
    for name, pad in (("walk", 0), ("bulk", 20)):
        proto = Mesh()
        proto.add_edge(V(0.03, 0.02, 0), V(300, 300, 0))   # edges weld first
        proto.add_face([V(0, 0, 0), V(40, 0, 0), V(40, 40, 0), V(0, 40, 0)])
        proto.add_face([V(0.04, 0.0, 0), V(40, 0.03, 0), V(20, 40, 0.02)])
        proto.add_face([V(100, 0, 0), V(100.04, 0, 0), V(100.04, 0.04, 0),
                        V(100, 0.04, 0)])
        if pad:
            for i in range(pad):
                for j in range(pad):
                    proto.add_face([V(500 + i, 500 + j, 0),
                                    V(501 + i, 500 + j, 0),
                                    V(501 + i, 501 + j, 0)])
        g = Group(proto, name=name)
        g.xform = xform(0 if pad == 0 else 50, 0, 0, 10.0, scale=0.001)
        s.groups.append(g)
    return s


def scene_classic(tex: Path) -> Scene:
    """Groups with no transform at all: the scene reads each group's own
    mesh (the render_faces path)."""
    s = Scene()
    s.layers.append(Layer("Oculta", visible=False))
    s.mesh.add_face([V(0, 0, 0), V(1, 0, 0), V(1, 1, 0), V(0, 1, 0)])
    hid = s.mesh.add_face([V(5, 0, 0), V(6, 0, 0), V(6, 1, 0)])
    hid.attrs["hidden"] = True
    for k in range(3):
        g = Group(name=f"Grupo {k}")
        g.mesh.add_face([V(10 * k, 5, 0), V(10 * k + 2, 5, 0),
                         V(10 * k + 2, 7, 1), V(10 * k, 7, 1)])
        g.mesh.faces[0].attrs["color"] = [0.1 * k, 0.5, 0.9]
        if k == 1:
            g.mesh.faces[0].attrs["hidden"] = True       # ignored in groups
            g.mesh.add_face([V(10, 9, 0), V(12, 9, 0), V(12, 10, 0)])
        if k == 2:
            g.layer = "Oculta"
        s.groups.append(g)
    return s


def scene_empty(tex: Path) -> Scene:
    return Scene()


SCENES = {
    "faces": scene_faces,
    "textured": scene_textured,
    "components": scene_components,
    "bulk": scene_bulk,
    "weld": scene_weld,
    "classic": scene_classic,
    "empty": scene_empty,
}


def relative_paths(node, root: str):
    """The scene JSON with every texture path made relative to the fixtures."""
    if isinstance(node, dict):
        return {k: (os.path.relpath(v, root).replace("\\", "/")
                    if k == "path" and isinstance(v, str) else
                    relative_paths(v, root)) for k, v in node.items()}
    if isinstance(node, list):
        return [relative_paths(x, root) for x in node]
    return node


def save_scene_json(scene, dest: Path) -> None:
    saved = (igz._LARGE_DOCUMENT, igz._pack_textures)
    igz._LARGE_DOCUMENT = 10 ** 9              # bare JSON, no binary blocks
    igz._pack_textures = lambda payload: ({}, 0)  # keep texture paths
    try:
        igz.save_scene(scene, dest)
    finally:
        igz._LARGE_DOCUMENT, igz._pack_textures = saved
    raw = dest.read_bytes()
    assert not raw.startswith(b"PK"), dest
    doc = json.loads(raw.decode("utf-8"))
    doc = relative_paths(doc, str(OUT / "export"))
    dest.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")


def export_all() -> None:
    exp = OUT / "export"
    shutil.rmtree(exp, ignore_errors=True)
    exp.mkdir(parents=True)
    tex = exp / "tex"
    png(tex / "wood.png", 0x8B5A2B, 6)
    png(tex / "ladrillo rústico.png", 0xB04030, 9)
    for name, make in SCENES.items():
        scene = make(tex)
        save_scene_json(scene, exp / f"{name}.igz.json")
        obj.save_obj(scene, exp / f"{name}.obj")
        for unit in ("m", "mm", "in"):
            stl.save_stl(scene, exp / f"{name}.{unit}.stl", unit)
        print(name, "exported")
    # The images the OBJ export copied beside the .obj stay (the test checks
    # the copies); their number is part of the contract.


# ---- number formatting ---------------------------------------------------------

def write_numbers() -> None:
    rng = random.Random(20261019)
    xs = []
    for _ in range(3000):
        x = rng.uniform(-1e3, 1e3) * 10 ** rng.randint(-8, 2)
        xs.append(struct.unpack("<f", struct.pack("<f", x))[0])   # float32 coords
    for _ in range(2000):
        xs.append(rng.uniform(-5, 5))
    for k in range(-3000, 3000):                                   # decimal ties
        xs.append(k / 128.0)
        xs.append(k / 1024.0)
        xs.append(k * 5e-7)
    xs += [0.0, -0.0, 1e-7, -1e-7, 5e-7, -5e-7, 4.9999999e-7, 1e15, -3e17,
           2.5e-6, 0.5, 1.5, 2.5, 123456789.123456789, 1e-320]
    lines = []
    for x in xs:
        lines.append(f"{struct.pack('>d', x).hex()} {x:.6f} {x:.4f} "
                     f"{struct.pack('>d', round(x, 6)).hex()} "
                     f"{struct.pack('>d', round(x, 4)).hex()}")
    (OUT / "numbers.txt").write_text("\n".join(lines) + "\n", encoding="ascii",
                                     newline="\n")


# ---- imports ---------------------------------------------------------------------

def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, sort_keys=True, separators=(",", ":")),
                    encoding="utf-8", newline="\n")


def cube_triangles(size=1.0, origin=(0.0, 0.0, 0.0)):
    ox, oy, oz = origin
    c = [(ox + x * size, oy + y * size, oz + z * size)
         for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6),
             (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = []
    for a, b, cc, d in quads:
        tris.append((c[a], c[b], c[cc]))
        tris.append((c[a], c[cc], c[d]))
    return tris


def sphere_triangles(n_lon=24, n_lat=16, r=1.5):
    pts = []
    for i in range(n_lat + 1):
        th = math.pi * i / n_lat
        row = []
        for j in range(n_lon):
            ph = 2 * math.pi * j / n_lon
            row.append((r * math.sin(th) * math.cos(ph),
                        r * math.sin(th) * math.sin(ph), r * math.cos(th)))
        pts.append(row)
    tris = []
    for i in range(n_lat):
        for j in range(n_lon):
            a, b = pts[i][j], pts[i][(j + 1) % n_lon]
            c, d = pts[i + 1][j], pts[i + 1][(j + 1) % n_lon]
            if i > 0:
                tris.append((a, c, b))
            if i < n_lat - 1:
                tris.append((b, c, d))
    return tris


def binary_stl(tris, header=b"binary export", extra=0, attr=0) -> bytes:
    out = bytearray(header.ljust(80, b"\x00"))
    out += struct.pack("<I", len(tris))
    for a, b, c in tris:
        n = (0.0, 0.0, 1.0)
        out += struct.pack("<12fH", *n, *a, *b, *c, attr)
    out += b"\x00" * extra
    return bytes(out)


def ascii_stl(tris, name="part", eol="\n", upper=False, indent="  ",
              comment=False) -> str:
    kw = (lambda s: s.upper()) if upper else (lambda s: s)
    lines = [f"{kw('solid')} {name}"]
    if comment:
        lines.append("# written by hand")
    for a, b, c in tris:
        lines.append(f"{indent}{kw('facet normal')} 0 0 1")
        lines.append(f"{indent}{indent}{kw('outer loop')}")
        for p in (a, b, c):
            lines.append(f"{indent}{indent}{indent}{kw('vertex')} "
                         + " ".join(repr(float(x)) for x in p))
        lines.append(f"{indent}{indent}{kw('endloop')}")
        lines.append(f"{indent}{kw('endfacet')}")
    lines.append(f"{kw('endsolid')} {name}")
    return eol.join(lines) + eol


def grid_triangles(origin, du, dv, n, m, flip=False):
    """An n x m grid of cells on the plane origin + i*du + j*dv, two
    triangles each."""
    def p(i, j):
        return tuple(origin[k] + i * du[k] + j * dv[k] for k in range(3))
    tris = []
    for i in range(n):
        for j in range(m):
            a, b, c, d = p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1)
            t = [(a, b, c), (a, c, d)]
            if (i + j) % 3 == 0:                   # the other diagonal
                t = [(a, b, d), (b, c, d)]
            if flip:
                t = [(x, z, y) for x, y, z in t]
            tris.extend(t)
    return tris


def cube_grid_triangles(n=4, size=2.0, origin=(0.0, 0.0, 0.0), flipped=()):
    """A cube whose faces are n x n grids; ``flipped`` lists face indices
    wound inwards."""
    ox, oy, oz = origin
    s = size / n
    faces = [
        ((ox, oy, oz), (s, 0, 0), (0, s, 0)),            # bottom (z=0), normal down when flipped
        ((ox, oy, oz + size), (0, s, 0), (s, 0, 0)),
        ((ox, oy, oz), (0, 0, s), (s, 0, 0)),
        ((ox, oy + size, oz), (s, 0, 0), (0, 0, s)),
        ((ox, oy, oz), (0, s, 0), (0, 0, s)),
        ((ox + size, oy, oz), (0, 0, s), (0, s, 0)),
    ]
    tris = []
    for k, (o, du, dv) in enumerate(faces):
        # bottom, y-min and x-min are wound so their normal points inwards
        # in the table above; reverse those to face outwards.
        inward = k in (0, 2, 4)
        tris.extend(grid_triangles(o, du, dv, n, n, flip=inward ^ (k in flipped)))
    return tris


def plate_with_hole_triangles(n=6, hole=(2, 4), z=0.0):
    """A square plate of n x n cells with the cells inside ``hole`` left out."""
    tris = []
    for i in range(n):
        for j in range(n):
            if hole[0] <= i < hole[1] and hole[0] <= j < hole[1]:
                continue
            a, b, c, d = ((i, j, z), (i + 1, j, z), (i + 1, j + 1, z), (i, j + 1, z))
            tris.extend([(a, b, c), (a, c, d)])
    return tris


def rotated(tris, axis, deg, shift=(0.0, 0.0, 0.0)):
    import math
    ax, ay, az = axis
    ln = math.sqrt(ax * ax + ay * ay + az * az)
    ax, ay, az = ax / ln, ay / ln, az / ln
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))

    def rot(p):
        x, y, z = p
        d = ax * x + ay * y + az * z
        return (x * c + (ay * z - az * y) * s + ax * d * (1 - c) + shift[0],
                y * c + (az * x - ax * z) * s + ay * d * (1 - c) + shift[1],
                z * c + (ax * y - ay * x) * s + az * d * (1 - c) + shift[2])
    return [tuple(rot(p) for p in t) for t in tris]


def doubled(tris):
    """Every triangle twice, the second reversed (a two-sided export)."""
    return list(tris) + [(a, c, b) for a, b, c in tris]


def obj_from_triangles(groups, header="", uv_of=None, smooth=None, mtl=None,
                       usemtl=None):
    """OBJ text for ``groups``: a list of (statement lines, triangles) pairs.
    Corners are shared by position; ``uv_of(p)`` adds a vt per corner."""
    verts: dict = {}
    uvs: dict = {}
    out = []
    body = []
    for lines, tris in groups:
        body.extend(lines)
        for t in tris:
            idx = []
            for p in t:
                i = verts.setdefault(p, len(verts) + 1)
                if uv_of is not None:
                    uv = uv_of(p)
                    j = uvs.setdefault(uv, len(uvs) + 1)
                    idx.append(f"{i}/{j}")
                else:
                    idx.append(str(i))
            body.append("f " + " ".join(idx))
    if mtl:
        out.append(f"mtllib {mtl}")
    out.append(header.rstrip())
    for p in verts:
        out.append("v " + " ".join(repr(float(x)) for x in p))
    for uv in uvs:
        out.append("vt " + " ".join(repr(float(x)) for x in uv))
    out.extend(body)
    return "\n".join(x for x in out if x != "") + "\n"


def write_imports() -> None:
    imp = OUT / "import"
    shutil.rmtree(imp, ignore_errors=True)
    imp.mkdir(parents=True)
    exp = OUT / "export"
    png(imp / "wood.png", 0x8B5A2B, 6)
    cases = []

    def add(kind, fname, **opts):
        cases.append(dict(kind=kind, file=fname, **opts))

    # -- STL -----------------------------------------------------------------
    cube = cube_triangles(2.0, (1.0, 2.0, 3.0))
    (imp / "cube.bin.stl").write_bytes(binary_stl(cube))
    (imp / "cube.solidheader.stl").write_bytes(
        binary_stl(cube, header=b"solid binary that lies"))
    (imp / "cube.trailing.stl").write_bytes(binary_stl(cube, extra=7))
    (imp / "cube.ascii.stl").write_text(ascii_stl(cube), newline="\n")
    (imp / "cube.crlf.stl").write_bytes(
        ascii_stl(cube, eol="\r\n", upper=True).encode())
    (imp / "cube.cr.stl").write_bytes(ascii_stl(cube, eol="\r").encode())
    (imp / "cube.comment.stl").write_bytes(
        b"\xef\xbb\xbf" + ascii_stl(cube, comment=True, indent="\t").encode())
    degenerate = cube + [((0, 0, 0), (1, 0, 0), (2, 0, 0)),
                         ((5, 5, 5), (5, 5, 5), (6, 5, 5)),
                         ((0, 0, 0), (1e-7, 0, 0), (0, 1e-7, 0))]
    (imp / "degenerate.bin.stl").write_bytes(binary_stl(degenerate))
    (imp / "degenerate.ascii.stl").write_text(ascii_stl(degenerate),
                                              newline="\n")
    # Corners 0.04 apart at scale 0.001 weld into one vertex.
    weld = [((0, 0, 0), (40, 0, 0), (0, 40, 0)),
            ((0.04, 0.0, 0), (40, 0, 0), (40, 40, 0))]
    (imp / "weld.ascii.stl").write_text(ascii_stl(weld), newline="\n")
    (imp / "weld.bin.stl").write_bytes(binary_stl(weld))
    sphere = sphere_triangles()
    assert len(sphere) > 400
    (imp / "sphere.bin.stl").write_bytes(binary_stl(sphere))
    (imp / "sphere.ascii.stl").write_text(ascii_stl(sphere, name="sphere"),
                                          newline="\n")
    odd_floats = ("solid odd\n facet normal 0 0 1\n  outer loop\n"
                  "   vertex 1_0 +.5 1e1\n   vertex 5. -0 0x\n"
                  "   vertex 0 0 0\n  endloop\n endfacet\nendsolid\n")
    (imp / "underscore.stl").write_text(odd_floats.replace(" 0x", " 3"),
                                        newline="\n")
    for name, scale in (("cube.bin.stl", 0.001), ("cube.ascii.stl", 0.0254),
                        ("weld.ascii.stl", 0.001), ("weld.bin.stl", 0.001),
                        ("sphere.bin.stl", 0.01), ("degenerate.bin.stl", 1.0),
                        ("cube.bin.stl", 1.0), ("cube.ascii.stl", 1.0),
                        ("cube.solidheader.stl", 1.0),
                        ("cube.trailing.stl", 1.0), ("cube.crlf.stl", 1.0),
                        ("cube.cr.stl", 1.0), ("cube.comment.stl", 1.0),
                        ("degenerate.ascii.stl", 1.0),
                        ("sphere.ascii.stl", 1.0), ("underscore.stl", 1.0),
                        ("weld.ascii.stl", 1.0), ("weld.bin.stl", 1.0)):
        add("stl", name, scale=scale)
    # Exported scenes read back (the loose faces of the first two).
    for name in ("faces", "textured", "classic"):
        shutil.copy(exp / f"{name}.mm.stl", imp / f"export-{name}.mm.stl")
        add("stl", f"export-{name}.mm.stl", scale=0.001)
    # Bad input.
    bad = {
        "empty.stl": b"",
        "short.stl": b"solid",
        "notstl.stl": b"hello world\nthis is not a mesh\n",
        "missing-endloop.stl": b"solid x\nfacet normal 0 0 1\nouter loop\n"
                               b"vertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\n"
                               b"endfacet\nendsolid\n",
        "bad-normal.stl": b"solid x\nfacet normal 0 a 1\n",
        "short-normal.stl": b"solid x\nfacet normal 0 1\n",
        "nan-normal.stl": b"solid x\nfacet normal nan 0 1\n",
        "bad-vertex.stl": b"solid x\nfacet normal 0 0 1\nouter loop\n"
                          b"vertex 0 oops 0\n",
        "huge-vertex.stl": b"solid x\nfacet normal 0 0 1\nouter loop\n"
                           b"vertex 0 1e999 0\n",
        "huge-vertex2.stl": b"solid x\nfacet normal 0 0 1\nouter loop\n"
                            b"vertex 0 1e39 0\n",
        "two-vertex.stl": b"solid x\nfacet normal 0 0 1\nouter loop\n"
                          b"vertex 0 0\n",
        "no-outer.stl": b"solid x\nfacet normal 0 0 1\nvertex 0 0 0\n",
        "no-endfacet.stl": b"solid x\nfacet normal 0 0 1\nouter loop\n"
                           b"vertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\n"
                           b"endloop\nendloop\n",
        "incomplete.stl": b"solid x\nfacet normal 0 0 1\nouter loop\n",
        "only-degenerate.stl": ascii_stl([((0, 0, 0), (1, 0, 0),
                                           (2, 0, 0))]).encode(),
        "far.bin.stl": binary_stl([((0, 0, 0), (3.0e12, 0, 0),
                                     (0, 1, 0))]),
        "nan.bin.stl": binary_stl([((0, 0, 0), (float("nan"), 0, 0),
                                    (0, 1, 0))]),
        "degenerate.bin2.stl": binary_stl([((0, 0, 0), (1, 0, 0),
                                            (2, 0, 0))]),
    }
    for name, data in bad.items():
        (imp / name).write_bytes(data)
        add("stl", name, scale=1.0)
    add("stl", "cube.bin.stl", scale=0.0, name="zero-scale")
    add("stl", "cube.bin.stl", scale=float("nan"), name="nan-scale")
    add("stl", "cube.bin.stl", scale=1.0, simplify="principal",
        name="simplify")
    add("stl", "cube.bin.stl", scale=1.0, simplify="all", name="simplify-all")
    add("stl", "cube.ascii.stl", scale=0.0254, simplify="all",
        name="simplify-all")
    # Flat and curved soups that give the fuse pass something to do.
    plane = grid_triangles((0.0, 0.0, 0.0), (0.5, 0, 0), (0, 0.5, 0), 20, 20)
    (imp / "plane.bin.stl").write_bytes(binary_stl(plane))
    tilt = rotated(grid_triangles((0.0, 0.0, 0.0), (0.25, 0, 0), (0, 0.25, 0),
                                  12, 12), (1.0, 0.7, 0.2), 33.0,
                   (1.0, 2.0, 3.0))
    (imp / "tilt.bin.stl").write_bytes(binary_stl(tilt))
    cubegrid = cube_grid_triangles(4, 2.0, (1.0, 2.0, 3.0))
    (imp / "cubegrid.bin.stl").write_bytes(binary_stl(cubegrid))
    (imp / "cubegrid.ascii.stl").write_text(ascii_stl(cubegrid, name="cg"),
                                            newline="\n")
    mixed = cube_grid_triangles(3, 1.0, (0.0, 0.0, 0.0), flipped=(1, 4))
    (imp / "cubemixed.bin.stl").write_bytes(binary_stl(mixed))
    two = (cube_grid_triangles(2, 1.0, (0.0, 0.0, 0.0))
           + cube_grid_triangles(2, 1.0, (5.0, 0.0, 0.0), flipped=(0, 1, 2, 3, 4, 5)))
    (imp / "twocubes.bin.stl").write_bytes(binary_stl(two))
    (imp / "cubetwosided.bin.stl").write_bytes(
        binary_stl(doubled(cube_grid_triangles(3, 1.0))))
    plate = plate_with_hole_triangles(6, (2, 4))
    (imp / "plate.ascii.stl").write_text(ascii_stl(plate, name="plate"),
                                         newline="\n")
    (imp / "plate.bin.stl").write_bytes(binary_stl(plate))
    ball = sphere_triangles(12, 8, 1.5)
    (imp / "ball.bin.stl").write_bytes(binary_stl(ball))
    bigplane = grid_triangles((0.0, 0.0, 0.0), (1, 0, 0), (0, 1, 0), 16, 16)
    (imp / "bigplane.ascii.stl").write_text(ascii_stl(bigplane, name="bp"),
                                            newline="\n")
    for name in ("plane.bin.stl", "tilt.bin.stl", "cubegrid.bin.stl",
                 "cubegrid.ascii.stl", "cubemixed.bin.stl", "twocubes.bin.stl",
                 "cubetwosided.bin.stl", "plate.ascii.stl", "plate.bin.stl",
                 "ball.bin.stl", "bigplane.ascii.stl"):
        for mode in ("none", "principal", "all"):
            add("stl", name, scale=0.001, simplify=mode, name=f"s-{mode}")
    add("stl", "cubegrid.bin.stl", scale=1.0, name="plain")

    # -- OBJ -----------------------------------------------------------------
    def put(name, text, **kw):
        data = text.encode(kw.get("encoding", "utf-8"))
        (imp / name).write_bytes(data)

    put("quad.obj", "# a quad\nmtllib quad.mtl\nv 0 0 0\nv 2 0 0\nv 2 1 0\n"
        "v 0 1 0\nvn 0 0 1\nusemtl red\nf 1//1 2//1 3//1 4//1\n")
    put("quad.mtl", "newmtl red\nKd 0.8 0.1 0.1\nnewmtl cream\n"
        "Kd 0.96 0.95 0.925\n")
    put("ngon.obj", "v 0 0 0\nv 3 0 0\nv 3 2 0\nv 1.5 3 0\nv 0 2 0\n"
        "f 1 2 3 4 5\n")
    put("negative.obj", "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
        "f -4 -3 -2\nf -4 -2 -1\n")
    put("cream.obj", "mtllib quad.mtl\nv 0 0 0\nv 1 0 0\nv 1 1 0\n"
        "usemtl cream\nf 1 2 3\nusemtl missing\nf 1 3 2\n")
    put("smooth.obj", "mtllib quad.mtl\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
        "v 2 0 0.2\nv 2 1 0.2\nusemtl red\ns 1\nf 1 2 3\nf 1 3 4\n"
        "f 2 5 6 3\ns off\nf 1 4 3\ns 2\nf 1 2 5\ns 2\nf 1 5 4\n")
    put("texture.obj", "mtllib texture.mtl\nv 0 0 0\nv 1 0 0\nv 1 1 0\n"
        "v 0 1 0\nv 0 0 1\nv 1 0 1\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
        "usemtl wood\nf 1/1 2/2 3/3 4/4\nf 1/1 2/2 6/3 5/4\n"
        "f 1 2 3\nf 1/1 2/9 3/3\nf 1/1/1 2/2/1 3/3/1\n")
    put("texture.mtl", "newmtl wood\nKd 1 1 1\nmap_Kd -s 1 1 1 wood.png\n"
        "newmtl nope\nKd 0.2 0.3 0.4\nmap_Kd missing.png\n"
        "newmtl spaced\nmap_Kd \"C:/Documents and Settings/x/wood.png\"\n")
    put("texture2.obj", "mtllib texture.mtl\nv 0 0 0\nv 1 0 0\nv 1 1 0\n"
        "usemtl nope\nf 1 2 3\nusemtl spaced\nf 1 3 2\n")
    put("odd.obj", "# odd input\r\nmtllib nothere.mtl\r\nv 0 0 0 1\r\n"
        "v 1 0 0\r\nv 1 1 0\r\nv 0 1 0\r\nvt 0.5\r\nvt x y\r\nvt\r\n"
        "usemtl nothing\r\nf 1 2 3 4\r\nf 1 x 3\r\nf 1 2\r\nf 1 2 9\r\n"
        "f 0 1 2\r\nf 1/ 2/ 3/\r\nf 1//1 2//1 3//1\r\nf +1 -1 -2\r\n"
        "f 1_0 2 3\r\nf 1 2 3 4 5 6 7\r\nl 1 2\r\np 1\r\n"
        "g\r\no\r\ns\r\nf 1 2 3\r\n")
    put("bom.obj", "\ufeffv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    put("cp1252.obj", "# Escultura p\xfablica\nv 0 0 0\nv 1 0 0\nv 0 1 0\n"
        "f 1 2 3\n", encoding="cp1252")
    put("yup.obj", "v 0 0 0\nv 100 0 0\nv 100 50 0\nv 0 50 20\n"
        "f 1 2 3\nf 1 3 4\n")
    put("parts.obj", "o Seat_Cube.001\nv 0 0 0\nv 1 0 0\nv 1 1 0\nf 1 2 3\n"
        "o Leg\nv 5 0 0\nv 6 0 0\nv 6 1 0\nf 4 5 6\n")
    put("bigobj.obj", "".join(f"v {i} 0 0\nv {i} 1 0\nv {i + 1} 0 0\n"
                              for i in range(0, 450)) +
        "".join(f"f {3 * i + 1} {3 * i + 2} {3 * i + 3}\n"
                for i in range(450)))
    put("badv.obj", "v 0 0\nf 1 2 3\n")
    put("badnum.obj", "v 0 0 abc\nf 1 2 3\n")
    put("badmtl.obj", "mtllib bad.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    put("bad.mtl", "newmtl\nKd 1 1 1\n")
    put("badkd.obj", "mtllib badkd.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    put("badkd.mtl", "newmtl m\nKd 1 x 1\n")
    put("slash.obj", "mtllib sub/quad.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\n"
        "f 1 2 3\n")
    put("cm.obj", "v 0 0 0\nv 500 0 0\nv 500 100 0\nf 1 2 3\n")
    put("mm.obj", "v 0 0 0\nv 30000 0 0\nv 0 20000 5\nf 1 2 3\n")
    put("nanunit.obj", "v 0 0 0\nv nan 1 1\nv 4 5 6\nf 1 2 3\n")
    put("badunit.obj", "v 0 0 0\nv 1 x 1\nv 4 5 6\nf 1 2 3\n")
    keys = [("Seat_Cube.001", ""), ("", "Door.001_Cube.007"),
            ("Sphere.021_Sphere.005", ""), ("", "1"), ("", ""), ("", " "),
            ("  spaced name ", ""), ("Base-unit-Box-Right_Cube.002", ""),
            ("12", ""), ("Leg", "Pata"), ("A_B.1", ""), ("x.001.002", ""),
            ("_Cube", ""), ("Mesa_Pata", ""), ("o", "g_Cube.12"), (".001", ""),
            ("ñandú_Cube", ""), ("3.5", ""), ("a_Cube.", "")]
    write_json(imp / "pieces.json", [
        [o, g, n, obj._piece_name((o, g), n), obj._tidy_blender_name(o or g)]
        for n, (o, g) in enumerate(keys, start=1)])
    for name, kw in (("cm.obj", {}), ("mm.obj", {}), ("nanunit.obj", {}),
                     ("badunit.obj", {}),
                     ("quad.obj", {}), ("ngon.obj", {}), ("negative.obj", {}),
                     ("cream.obj", {}), ("smooth.obj", {}),
                     ("texture.obj", {}), ("texture2.obj", {}),
                     ("odd.obj", {}), ("bom.obj", {}), ("cp1252.obj", {}),
                     ("yup.obj", {"scale": 0.01, "matrix": "y_up"}),
                     ("yup.obj", {"scale": 1.0, "matrix": [
                         0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0]}),
                     ("parts.obj", {}), ("bigobj.obj", {}),
                     ("badv.obj", {}), ("badnum.obj", {}),
                     ("badmtl.obj", {}), ("badkd.obj", {}),
                     ("slash.obj", {})):
        add("obj", name, **kw)
    # Big and multi-part models: the fused paths.
    cg = cube_grid_triangles(4, 2.0, (1.0, 2.0, 3.0))
    big_plane = grid_triangles((0.0, 0.0, 0.0), (1, 0, 0), (0, 1, 0), 30, 30)
    left = [t for t in big_plane if t[0][0] < 15]
    right = [t for t in big_plane if t[0][0] >= 15]
    put("quad.mtl", "newmtl red\nKd 0.8 0.1 0.1\nnewmtl cream\n"
        "Kd 0.96 0.95 0.925\nnewmtl blue\nKd 0.1 0.2 0.9\n")
    put("gridbig.obj", obj_from_triangles(
        [(["usemtl red", "s 1"], left), (["usemtl cream", "s 1"], right)],
        mtl="quad.mtl"))
    put("gridsmooth.obj", obj_from_triangles(
        [(["usemtl red", "s 1"], left), (["usemtl red", "s 2"], right)],
        mtl="quad.mtl"))
    put("ballbig.obj", obj_from_triangles(
        [(["usemtl blue", "s 1"], sphere_triangles(24, 16, 1.5))],
        mtl="quad.mtl"))
    put("cubegridbig.obj", obj_from_triangles(
        [(["usemtl red"], cube_grid_triangles(9, 2.0))], mtl="quad.mtl"))
    put("texgridbig.obj", obj_from_triangles(
        [(["usemtl wood"], big_plane)], mtl="texture.mtl",
        uv_of=lambda p: (p[0] / 30.0, p[1] / 30.0)))
    put("texcubebig.obj", obj_from_triangles(
        [(["usemtl wood"], cube_grid_triangles(9, 2.0))], mtl="texture.mtl",
        uv_of=lambda p: (p[0] / 2.0 + p[2] / 4.0, p[1] / 2.0)))
    quads = []
    for i in range(25):
        for j in range(25):
            quads.append((i, j))
    qverts = "".join(f"v {i} {j} 0\nv {i + 1} {j} 0\nv {i + 1} {j + 1} 0\n"
                     f"v {i} {j + 1} 0\n" for i, j in quads)
    put("quadsbig.obj", qverts + "".join(
        f"f {4 * k + 1} {4 * k + 2} {4 * k + 3} {4 * k + 4}\n"
        for k in range(len(quads))))
    put("parts2.obj", obj_from_triangles([
        (["o Seat_Cube.001", "usemtl red", "s 1"], cg),
        (["o Leg", "usemtl blue"], rotated(cg, (0, 0, 1), 0.0, (10.0, 0, 0))),
        (["o Sliver", "usemtl blue"],
         [((20.0, 0.0, 0.0), (21.0, 0.0, 0.0), (22.0, 0.0, 0.0))]),
        (["o Top", "g Tapa"], grid_triangles((30.0, 0, 0), (1, 0, 0),
                                              (0, 1, 0), 3, 3)),
        (["g 2"], grid_triangles((40.0, 0, 0), (1, 0, 0), (0, 1, 0), 2, 2)),
        (["g"], grid_triangles((50.0, 0, 0), (1, 0, 0), (0, 1, 0), 2, 2)),
    ], mtl="quad.mtl"))
    put("partstex.obj", obj_from_triangles([
        (["o Wall", "usemtl wood"], big_plane[:60]),
        (["o Floor", "usemtl red"], big_plane[60:120]),
    ], mtl="texture.mtl", uv_of=lambda p: (p[0] / 30.0, p[1] / 30.0)))
    put("bigparts.obj", obj_from_triangles([
        (["o A"], big_plane[:900]), (["o B"], rotated(
            big_plane[900:], (0, 1, 0), 90.0, (0, 0, 40.0)))]))
    put("sliveronly.obj", obj_from_triangles([
        (["o A"], [((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))]),
        (["o B"], [((0.0, 5.0, 0.0), (1.0, 5.0, 0.0), (2.0, 5.0, 0.0))]),
    ]))
    put("edge400.obj", obj_from_triangles([([], big_plane[:400])]))
    put("edge401.obj", obj_from_triangles([([], big_plane[:401])]))
    put("twosidedbig.obj", obj_from_triangles(
        [(["usemtl red"], doubled(big_plane[:300]))], mtl="quad.mtl"))
    for name in ("gridbig.obj", "gridsmooth.obj", "ballbig.obj",
                 "cubegridbig.obj", "texgridbig.obj", "texcubebig.obj",
                 "quadsbig.obj", "parts2.obj", "partstex.obj", "bigparts.obj",
                 "sliveronly.obj", "edge400.obj", "edge401.obj",
                 "twosidedbig.obj"):
        add("obj", name)
    add("obj", "cubegridbig.obj", scale=0.1)
    # Exported scenes read back: only those that are one small part.
    for name in ("faces", "textured", "classic", "empty"):
        for ext in (".obj", ".mtl"):
            shutil.copy(exp / f"{name}{ext}", imp / f"export-{name}{ext}")
        (imp / f"export-{name}.obj").write_text(
            (exp / f"{name}.obj").read_text(encoding="utf-8").replace(
                f"mtllib {name}.mtl", f"mtllib export-{name}.mtl"),
            encoding="utf-8", newline="")
        add("obj", f"export-{name}.obj")
    for src in (exp / "tex").iterdir():
        shutil.copy(src, imp / src.name)
    (imp / "sub").mkdir()

    # -- run them with the real importers ---------------------------------------
    _pin_order.install()
    import formats.fuse
    real_fuse = formats.fuse.fuse_coplanar_loops
    fused = []

    def spy(*a, **k):
        fused.append(True)
        return real_fuse(*a, **k)
    formats.fuse.fuse_coplanar_loops = spy
    try:
        for case in cases:
            del fused[:]
            _pin_order.fresh_group_names()
            _pin_order.ORD.clear()
            _pin_order.KEEP.clear()
            path = imp / case["file"]
            scene = Scene()
            if case["kind"] == "obj":
                case["unit"] = obj.suggest_unit(path)
            scale = case.get("scale", 1.0)
            case["scale"] = scale if math.isfinite(scale) else str(scale)
            try:
                if case["kind"] == "stl":
                    stl.load_stl(scene, path, scale=scale,
                                 simplify_mode=case.get("simplify", "none"))
                else:
                    matrix = case.get("matrix")
                    if matrix == "y_up":
                        matrix = obj.Y_UP_TO_Z_UP
                    obj.load_obj(scene, path, scale=scale, matrix=matrix)
            except Exception as exc:  # noqa: BLE001 — the message is the datum
                case["error"] = f"{type(exc).__name__}: {exc}"
                continue
            if fused:
                case["fused"] = True
            tag = case["file"] + (f".{case['name']}" if "name" in case else "")
            sig_name = f"{tag}.{len(case.get('matrix') or '')}.{scale}.sig.json"
            case["sig"] = sig_name
            write_json(imp / sig_name, scene_signature(scene))
    finally:
        formats.fuse.fuse_coplanar_loops = real_fuse
    # Same input, same options twice would clash on the signature file name.
    names = [c.get("sig") for c in cases if "sig" in c]
    assert len(names) == len(set(names)), "signature names clash"
    write_json(imp / "cases.json", cases)
    print(len(cases), "import cases")


def main() -> None:
    app = QGuiApplication.instance() or QGuiApplication(sys.argv)  # noqa: F841
    OUT.mkdir(parents=True, exist_ok=True)
    if "--imports-only" not in sys.argv:        # the exports are committed
        export_all()
        write_numbers()
    write_imports()


if __name__ == "__main__":
    main()
