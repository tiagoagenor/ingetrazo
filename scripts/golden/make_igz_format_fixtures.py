# SPDX-License-Identifier: GPL-3.0-or-later
"""Write the .igz fixtures that hold the Rust reader to the three formats
(docs/rust-rewrite-plan.md, M2.7).

Each scene is built with the real app code and saved with
``formats.igz.save_scene`` as format 1 (bare JSON), 2 (ZIP, textures only)
and 3 (binary geometry blocks, forced by lowering the size thresholds), as
far as the scene allows: format 1 cannot carry a texture and format 2 is
only written when there is one. Beside every file goes ``<file>.sig.json``,
the L2 signature Python computes after REOPENING it. The files are
committed, so `cargo test` needs no Python; rerun this only to widen them
(PYTHONPATH at the repository root):

    python scripts/golden/make_igz_format_fixtures.py

Deleted with the Python app at M10.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QGuiApplication, QImage, QMatrix4x4, QVector3D as V  # noqa: E402

from core.group import Group  # noqa: E402
from core.mesh import Mesh  # noqa: E402
from core.scene import Scene  # noqa: E402
from formats import igz  # noqa: E402
from tests.golden.signature import scene_signature  # noqa: E402

OUT = ROOT / "rust" / "crates" / "ingetrazo-formats" / "tests" / "data" / "igz"


def xform(dx, dy, dz, angle=0.0):
    m = QMatrix4x4()
    m.translate(dx, dy, dz)
    if angle:
        m.rotate(angle, 0, 0, 1)
    return m


def texture_png(tmp: Path) -> Path:
    png = tmp / "wood.png"
    img = QImage(4, 4, QImage.Format_RGB32)
    img.fill(0x8B5A2B)
    img.setPixel(1, 2, 0x336699)
    assert img.save(str(png))
    return png


def scene_curves(tmp) -> Scene:
    """Curves, soft and hidden edges, edge layers."""
    s = Scene()
    m = s.mesh
    pts = [V(i, (i % 3) * 0.5, 0) for i in range(6)]
    for i, (a, b) in enumerate(zip(pts, pts[1:])):
        e = m.add_edge(a, b)
        e.curve = 4        # one curve of five segments
        if i == 2:
            e.soft = True
    ring = [V(10 + 2 * 0.7071 * k, 0, 0) for k in range(4)]
    for a, b in zip(ring, ring[1:]):
        e = m.add_edge(a, b)
        e.curve = 9
        e.layer = "Guias"
    h = m.add_edge(V(0, 5, 0), V(3, 5, 1.5))
    h.hidden = True
    h.layer = "Ocultas"
    s1 = m.add_edge(V(0, 6, 0), V(3, 6, 0))
    s1.soft = True
    s1.hidden = True
    m.add_edge(V(0, 7, 0), V(3, 7.25, 0.1))      # plain
    return s


def scene_faces(tmp) -> Scene:
    """Faces with holes and every kind of attribute, no texture."""
    s = Scene()
    m = s.mesh
    slab = m.add_face(
        [V(0, 0, 0), V(10, 0, 0), V(10, 10, 0), V(0, 10, 0)],
        [[V(2, 2, 0), V(4, 2, 0), V(4, 4, 0), V(2, 4, 0)],
         [V(6, 6, 0), V(8, 6, 0), V(8, 8, 0), V(6, 8, 0)],
         # nested in the second hole: dropped on load
         [V(6.5, 6.5, 0), V(7, 6.5, 0), V(7, 7, 0), V(6.5, 7, 0)]])
    slab.attrs.update({"color": [0.5, 0.4, 0.3], "layer": "Pisos",
                       "mat": "Concreto", "back": True})
    wall = m.add_face([V(0, 0, 0), V(10, 0, 0), V(10, 0, 3), V(0, 0, 3)])
    wall.attrs.update({"opacity": 0.5, "ifc": {"class": "IfcWall"},
                       "color": [0.9, 0.9, 0.8]})
    hid = m.add_face([V(0, 10, 0), V(10, 10, 0), V(10, 10, 3)])
    hid.attrs["hidden"] = True
    m.add_face([V(20, 0, 0), V(21, 0, 0), V(21, 1, 0.3), V(20, 1, 0.3)])
    twin = m.add_face([V(30, 0, 0), V(31, 0, 0), V(31, 1, 0)])
    twin.attrs.update({"color": [0.5, 0.4, 0.3], "layer": "Pisos",
                       "mat": "Concreto", "back": True})   # same table row
    m.add_edge(V(40, 0, 0), V(41, 0, 0))
    return s


def scene_textured(tmp) -> Scene:
    """A texture that embeds, shared by two faces, plus a back texture."""
    s = Scene()
    m = s.mesh
    png = texture_png(tmp)
    a = m.add_face([V(0, 0, 0), V(2, 0, 0), V(2, 2, 0), V(0, 2, 0)])
    a.attrs["texture"] = {"path": str(png), "sw": 1.5, "sh": 0.75}
    b = m.add_face([V(3, 0, 0), V(5, 0, 0), V(5, 2, 0)])
    b.attrs["texture"] = {"path": str(png), "sw": 1.5, "sh": 0.75}
    b.attrs["opacity"] = 0.25
    c = m.add_face([V(6, 0, 0), V(8, 0, 0), V(8, 2, 0)])
    c.attrs["back"] = {"texture": {"path": str(png), "sw": 2.0, "sh": 2.0}}
    m.add_edge(V(0, 5, 0), V(1, 5, 0)).curve = 2
    return s


def scene_groups(tmp) -> Scene:
    """Groups, components sharing a prototype, nested children."""
    s = Scene()
    s.mesh.add_face([V(-5, -5, 0), V(-4, -5, 0), V(-4, -4, 0)])
    proto = Mesh()
    proto.add_face([V(0, 0, 0), V(1, 0, 0), V(1, 1, 0), V(0, 1, 0)],
                   [[V(0.25, 0.25, 0), V(0.5, 0.25, 0), V(0.5, 0.5, 0)]])
    proto.add_edge(V(0, 0, 0), V(0, 0, 1)).curve = 3
    proto.faces[0].attrs["color"] = [0.2, 0.6, 0.2]
    for k, dx in enumerate((0, 5, 10)):
        g = Group(proto, name=f"Silla {k}")
        g.xform = xform(dx, 20, 0, 15.0 * k)
        if k == 2:
            g.layer = "Mobiliario"
            g.hidden = True
        s.groups.append(g)
    classic = Group(name="Mesa")
    classic.mesh.add_face([V(0, 30, 0), V(2, 30, 0), V(2, 31, 0)])
    classic.mesh.add_edge(V(0, 30, 0), V(0, 30, 1)).layer = "Patas"
    classic.locked = True
    # A group that owns placements of a second prototype.
    leaf = Mesh()
    leaf.add_face([V(0, 0, 0), V(0.5, 0, 0), V(0.5, 0.5, 0.2)])
    parent = Group(name="Conjunto")
    parent.mesh.add_face([V(0, 40, 0), V(3, 40, 0), V(3, 43, 0), V(0, 43, 0)])
    kids = []
    for k in range(2):
        child = Group(leaf, name=f"Hoja {k}")
        child.xform = xform(k, 41, 0)
        kids.append(child)
    inner = Group(name="Interno")
    inner.mesh.add_face([V(0, 50, 0), V(1, 50, 0), V(1, 51, 0)])
    kids.append(inner)
    parent.adopt(kids)          # a group that owns placements is an instance
    parent.component = False
    s.groups.append(classic)
    s.groups.append(parent)
    return s


SCENES = {
    "curves": scene_curves,
    "faces": scene_faces,
    "textured": scene_textured,
    "groups": scene_groups,
}


def write(name: str, make, fmt: int, tmp: Path) -> None:
    scene = make(tmp)
    path = OUT / f"{name}.f{fmt}.igz"
    saved = (igz._LARGE_DOCUMENT, igz._BULK_MIN)
    if fmt == 3:
        igz._LARGE_DOCUMENT, igz._BULK_MIN = 0, 1
    try:
        igz.save_scene(scene, path)
    finally:
        igz._LARGE_DOCUMENT, igz._BULK_MIN = saved
    raw = path.read_bytes()
    assert raw.startswith(b"PK") == (fmt != 1), name
    doc, _archive = igz._read_document(path)
    assert doc["igz_format"] == fmt, (name, fmt, doc["igz_format"])
    back = Scene()
    igz.load_into(back, path)
    sig = scene_signature(back)
    (OUT / f"{name}.f{fmt}.sig.json").write_text(
        json.dumps(sig, sort_keys=True, separators=(",", ":")),
        encoding="utf-8", newline="\n")
    print(f"{path.name}: {len(raw)} bytes")


def main() -> None:
    app = QGuiApplication.instance() or QGuiApplication(sys.argv)
    OUT.mkdir(parents=True, exist_ok=True)
    for old in list(OUT.glob("*.igz")) + list(OUT.glob("*.sig.json")):
        old.unlink()
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        for name, make in SCENES.items():
            for fmt in ((2, 3) if name == "textured" else (1, 3)):
                write(name, make, fmt, tmp)
    del app


if __name__ == "__main__":
    main()
