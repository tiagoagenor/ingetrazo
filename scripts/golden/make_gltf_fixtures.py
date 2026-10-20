# SPDX-License-Identifier: GPL-3.0-or-later
"""Write the fixtures that hold the Rust `ingetrazo-formats-mesh` glTF/GLB
code to Python's (docs/rust-rewrite-plan.md, M5.3).

Export. Scenes are built with the real app code (the seven of
make_mesh_formats_fixtures.py, which the OBJ/STL tests already open, and
three of their own: face-me figures, a deep node hierarchy, finishes and
transparency with every kind of texture image). Each is saved as a
bare-JSON .igz and exported with the REAL exporter, `formats.gltf.save_glb`:
the Rust test opens the .igz and must write the same bytes.

Import. The exported files read back, plus hand-made glTF/GLB files for what
real exporters write that ours does not: TRS and matrix nodes, interleaved
and normalised accessors, external and data-URI buffers and images, the
specular-glossiness and texture-transform extensions, blend/mask materials,
foliage, smooth spheres, random UV maps (their `uvw` is hashed into the
signature, so NumPy's LAPACK solve must be matched to the last bit), unit
guessing and bad input. Next to each goes the L2 signature of the scene
Python had after `formats.glb.load_glb` WITHOUT the Rust engine (the lazy
bulk path of a big model is not the one ported), or the message it raised.

The files are committed, so `cargo test` needs no Python; rerun this only to
widen them (PYTHONPATH at the repository root, native engine OFF):

    python scripts/golden/make_gltf_fixtures.py

Deleted with the Python app at M10.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import math
import os
import random
import shutil
import struct
import sys
import tempfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.pop("INGETRAZO_RUST", None)
CACHE = tempfile.mkdtemp(prefix="ingetrazo-glb-cache-")
os.environ["INGETRAZO_TEXTURE_CACHE"] = CACHE

from PySide6.QtGui import QGuiApplication, QImage  # noqa: E402

from core.group import Group  # noqa: E402
from core.layers import Layer  # noqa: E402
from core.materials import Material  # noqa: E402
from core.mesh import Mesh  # noqa: E402
from core.scene import Scene  # noqa: E402
from core.vecmath import QVector3D as V  # noqa: E402
from formats import glb, gltf  # noqa: E402
from georef.datum import SceneDatum  # noqa: E402
from tests.golden.signature import scene_signature  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "mesh_formats_fixtures", ROOT / "scripts" / "golden"
    / "make_mesh_formats_fixtures.py")
mf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mf)

DATA = ROOT / "rust" / "crates" / "ingetrazo-formats-mesh" / "tests" / "data"
OUT = DATA / "gltf"
EXPORT = DATA / "export"


# ---- PNGs of every colour type ---------------------------------------------------

def chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(
        ">I", zlib.crc32(body) & 0xFFFFFFFF)


def write_png(path: Path, w: int, h: int, color_type: int, depth: int,
              pixel, plte=None, trns=None) -> Path:
    """``pixel(x, y)`` -> the bytes of one pixel at this colour type."""
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        row = b"".join(pixel(x, y) for x in range(w))
        raw += row
    out = b"\x89PNG\r\n\x1a\n" + chunk(
        b"IHDR", struct.pack(">IIBBBBB", w, h, depth, color_type, 0, 0, 0))
    if plte is not None:
        out += chunk(b"PLTE", plte)
    if trns is not None:
        out += chunk(b"tRNS", trns)
    out += chunk(b"IDAT", zlib.compress(bytes(raw))) + chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(out)
    return path


def rgba_png(path, w, h, clear=(), alpha=0, base=(200, 120, 40)) -> Path:
    """RGBA 8-bit, opaque except the pixels in ``clear`` (given ``alpha``)."""
    clear = set(clear)

    def px(x, y):
        a = alpha if (x, y) in clear else 255
        return bytes((base[0], base[1] + (x % 3), base[2] + (y % 2), a))
    return write_png(path, w, h, 6, 8, px)


def rgb_png(path, w, h) -> Path:
    return write_png(path, w, h, 2, 8, lambda x, y: bytes(
        ((x * 7) % 256, (y * 5) % 256, 90)))


def png_variants(tex: Path) -> None:
    """The images the cut-out rule has to tell apart."""
    # 64 wide: Qt's fast scaling takes columns 0, 2, 4...: (1,1) is missed.
    rgba_png(tex / "cut64_missed.png", 64, 64, clear=[(1, 1)])
    rgba_png(tex / "cut64_hit.png", 64, 64, clear=[(2, 2)])
    # 100 wide: columns 1, 4, 6...: (0, 0) is missed.
    rgba_png(tex / "cut100_missed.png", 100, 100, clear=[(0, 0)])
    rgba_png(tex / "cut100_hit.png", 100, 100, clear=[(1, 1)])
    rgba_png(tex / "cut_a31.png", 40, 40, clear=[(5, 5), (6, 5)], alpha=31)
    rgba_png(tex / "cut_a32.png", 40, 40, clear=[(5, 5), (6, 5)], alpha=32)
    rgba_png(tex / "cut_odd.png", 37, 91, clear=[(20, 45), (36, 90)])
    rgba_png(tex / "cut_odd_hit.png", 37, 91, clear=[(20, 44)])
    rgba_png(tex / "cut_big.png", 640, 320, clear=[(319, 159)])
    rgba_png(tex / "cut_big_hit.png", 640, 320, clear=[(10, 5)])
    # Widths that are multiples of 64: Qt's boundary tie goes to the lower.
    rgba_png(tex / "cut128_hit.png", 128, 128, clear=[(1, 1)])
    rgba_png(tex / "cut128_missed.png", 128, 128, clear=[(0, 0)])
    rgba_png(tex / "opaque_rgba.png", 16, 16)
    rgb_png(tex / "rgb.png", 33, 17)
    # Gray + alpha, palette with tRNS, RGB with a colour key, 16-bit RGBA.
    write_png(tex / "ga.png", 20, 20, 4, 8, lambda x, y: bytes(
        (x * 12 % 256, 0 if (x, y) == (10, 10) else 255)))
    write_png(tex / "pal_trns.png", 8, 8, 3, 8, lambda x, y: bytes((x % 2,)),
              plte=b"\x10\x20\x30\xa0\xb0\xc0", trns=b"\xff\x00")
    write_png(tex / "pal_opaque_trns.png", 8, 8, 3, 8,
              lambda x, y: bytes((x % 2,)), plte=b"\x10\x20\x30\xa0\xb0\xc0",
              trns=b"\xff\xff")
    write_png(tex / "key.png", 8, 8, 2, 8, lambda x, y: bytes(
        (255, 0, 255) if (x, y) == (0, 0) else (10, 20, 30)),
        trns=struct.pack(">HHH", 255, 0, 255))
    write_png(tex / "rgba16.png", 12, 12, 6, 16, lambda x, y: struct.pack(
        ">HHHH", 1000 * x, 2000, 3000, 0x0800 if (x, y) == (6, 6) else 0xFFFF))
    write_png(tex / "rgba16_high.png", 12, 12, 6, 16, lambda x, y:
              struct.pack(">HHHH", 1000 * x, 2000, 3000,
                          0x1FFF if (x, y) == (6, 6) else 0xFFFF))
    img = QImage(10, 10, QImage.Format_RGB32)
    img.fill(0x4080C0)
    assert img.save(str(tex / "photo.jpg"), "JPG", 90)
    assert img.save(str(tex / "photo.jpeg"), "JPG", 70)
    (tex / "notimage.png").write_bytes(b"this is not an image at all")


# ---- export scenes -----------------------------------------------------------------

def scene_figures(tex: Path) -> Scene:
    """Face-me figures: both kinds, loose and placed, nested, and the
    degenerate ones."""
    s = Scene()
    s.layers.append(Layer("Oculta", visible=False))
    wood = tex / "wood.png"
    cut = tex / "cut64_hit.png"
    s.mesh.add_face([V(-3, -3, 0), V(-2, -3, 0), V(-2, -2, 0)])
    quad = Group(name="Figura A")
    quad.billboard = True
    q = quad.mesh.add_face([V(0, 0, 0), V(1, 0, 0), V(1, 0, 2), V(0, 0, 2)])
    q.attrs["texture"] = {"path": str(wood), "sw": 1.0, "sh": 2.0}
    q.attrs["mat"] = "Figura madera"
    s.groups.append(quad)
    quad2 = Group(name="")                      # nameless: "Figure"
    quad2.billboard = True
    q2 = quad2.mesh.add_face([V(5, 5, 1), V(6.5, 4, 1), V(6.5, 4, 3.2),
                              V(5, 5, 3.2)])
    q2.attrs["texture"] = {"path": str(cut), "sw": 1.0, "sh": 1.0,
                           "uvw": [0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.3, 0.5]}
    s.groups.append(quad2)
    outline = Group(name="Contorno")            # kind "mesh": turned about
    outline.billboard = "mesh"
    big = outline.mesh.add_face([V(10, 0, 0), V(12, 1, 0), V(12, 1, 3),
                                 V(10, 0, 3)])
    big.attrs["texture"] = {"path": str(wood), "sw": 1.0, "sh": 1.0}
    big.attrs["color"] = [0.2, 0.8, 0.2]
    outline.mesh.add_face([V(12, 1, 0), V(12.4, 1.2, 0), V(12, 1, 0.5)])
    outline.mesh.add_edge(V(10, 0, 0), V(10, 0, -0.5))   # lowers the feet
    s.groups.append(outline)
    proto = Mesh()
    pf = proto.add_face([V(0, 0, 0), V(2, 0, 0), V(2, 0, 3), V(0, 0, 3)])
    pf.attrs["texture"] = {"path": str(wood), "sw": 2.0, "sh": 3.0,
                           "uvw": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3333, 0.1]}
    placed = Group(proto, name="Persona colocada")
    placed.billboard = True
    m = mf.xform(20, 3, 0.5, 33.0)
    placed.xform = m
    s.groups.append(placed)
    placed_mesh = Group(proto, name="Persona 2")
    placed_mesh.billboard = "mesh"
    placed_mesh.xform = mf.xform(25, 3, 0.0, 130.0, scale=(1.0, 1.0, 1.5))
    s.groups.append(placed_mesh)
    parent = Group(name="Plaza")
    parent.mesh.add_face([V(40, 0, 0), V(44, 0, 0), V(44, 4, 0), V(40, 4, 0)])
    kids = []
    for k in range(2):
        kid = Group(proto, name=f"Visitante {k}")
        kid.billboard = True
        kid.xform = mf.xform(41 + k, 1, 0, 90.0 * k)
        kids.append(kid)
    solid = Group(proto, name="Banco")
    solid.xform = mf.xform(43, 2, 0)
    kids.append(solid)
    parent.adopt(kids)
    parent.xform = mf.xform(0, 0, 0, 15.0)
    parent.component = False
    s.groups.append(parent)
    hidden = Group(name="Figura oculta")
    hidden.billboard = True
    hidden.mesh.add_face([V(60, 0, 0), V(61, 0, 0), V(61, 0, 2), V(60, 0, 2)])
    hidden.hidden = True
    s.groups.append(hidden)
    empty = Group(name="Figura vacia")
    empty.billboard = True
    s.groups.append(empty)
    flat = Group(name="Figura plana")           # zero height: left as it is
    flat.billboard = True
    fl = flat.mesh.add_face([V(70, 0, 1), V(71, 0, 1), V(71, 1, 1), V(70, 1, 1)])
    fl.attrs["texture"] = {"path": str(wood), "sw": 1.0, "sh": 1.0}
    s.groups.append(flat)
    # A heavy figure takes the bulk copy.
    heavy = Group(mf.grid(18, 0.3), name="Hierba")
    heavy.billboard = "mesh"
    heavy.xform = mf.xform(80, 0, 0, 20.0, (0.1, 0.2, 1.0))
    s.groups.append(heavy)
    return s


def scene_hierarchy(tex: Path) -> Scene:
    """Names, nesting, shared meshes, identity and mirrored transforms."""
    s = Scene()
    s.layers.append(Layer("Oculta", visible=False))
    wood = tex / "wood.png"
    proto = Mesh()
    proto.add_face([V(0, 0, 0), V(1, 0, 0), V(1, 1, 0), V(0, 1, 0)])
    proto.add_face([V(0, 0, 0), V(1, 0, 0), V(1, 0, 1), V(0, 0, 1)])
    proto.faces[1].attrs["texture"] = {"path": str(wood), "sw": 0.5, "sh": 2.0}
    proto.faces[0].attrs["color"] = [0.3, 0.5, 0.7]
    ident = Group(proto, name="Identidad")
    ident.xform = mf.xform()
    s.groups.append(ident)
    for k, (m, name) in enumerate((
            (mf.xform(3, 0, 0, 90.0), "Cañería 90°"),
            (mf.xform(6, 0, 0, 0.0, scale=(-1.0, 1.0, 1.0)), "Espejo"),
            (mf.xform(9, 0, 0, 33.0, (1.0, 0.5, 0.2), (2.0, 2.0, 0.5)),
             "Ventana ✓ \"quoted\" \\ back"),
            (mf.xform(12, 0, 0), ""))):
        g = Group(proto, name=name)
        g.xform = m
        s.groups.append(g)
    root = Group(name="Edificio")
    root.mesh.add_face([V(0, 10, 0), V(10, 10, 0), V(10, 12, 0), V(0, 12, 0)])
    floor = Group(name="Piso 1")
    floor.mesh.add_face([V(0, 20, 0), V(5, 20, 0), V(5, 25, 0)])
    rooms = []
    for k in range(3):
        room = Group(proto, name=f"Cuarto {k}")
        room.xform = mf.xform(2 * k, 21, 0, 10.0 * k)
        sub = Group(proto, name=f"Mueble {k}")
        sub.xform = mf.xform(0.5, 0.5, 0, 45.0)
        room.adopt([sub])
        rooms.append(room)
    floor.adopt(rooms)
    floor.xform = mf.xform(0, 0, 3.0)
    floor.component = False
    root.adopt([floor, Group(Mesh(), name="Vacio")])
    root.xform = mf.xform(50, 0, 0, 5.0)
    root.component = False
    s.groups.append(root)
    shared = Group(proto, name="Oculto")
    shared.xform = mf.xform(70, 0, 0)
    shared.layer = "Oculta"
    s.groups.append(shared)
    emptyg = Group(name="Sin geometria")
    s.groups.append(emptyg)
    only_edge = Group(name="Solo arista")
    only_edge.mesh.add_edge(V(0, 0, 0), V(1, 1, 1))
    s.groups.append(only_edge)
    s.mesh.add_face([V(100, 0, 0), V(101, 0, 0), V(101, 1, 0)])
    return s


def scene_finishes(tex: Path) -> Scene:
    """Every material the exporter can write."""
    s = Scene()
    s.georef = SceneDatum(-12.0464, -77.0428, 154.25)
    s.layers.append(Layer("Oculta", visible=False))
    s.materials["Agua"] = Material("Agua", color=(0.1, 0.3, 0.6),
                                   finish="glass")          # chosen
    s.materials["Acero inox"] = Material("Acero inox", color=(0.7, 0.7, 0.7))
    s.materials["Piso_marmol"] = Material("Piso_marmol", color=(0.9, 0.9, 0.9))
    s.materials["Madera"] = Material("Madera", color=(0.5, 0.3, 0.1),
                                     finish="auto")         # unknown: guessed
    s.materials["Metal pintado"] = Material("Metal pintado",
                                            color=(0.2, 0.2, 0.2),
                                            finish="gloss")
    m = s.mesh
    x = [0.0]

    def tri(attrs, dy=0.0):
        a = x[0]
        x[0] += 2.0
        f = m.add_face([V(a, dy, 0), V(a + 1, dy, 0), V(a + 1, dy + 1, 0.2)])
        f.attrs.update(attrs)
        return f
    tri({"mat": "Agua", "color": [0.1, 0.3, 0.6]})
    tri({"mat": "Acero inox", "color": [0.7, 0.7, 0.7]})
    tri({"mat": "Piso_marmol", "color": [0.9, 0.9, 0.9]})
    tri({"mat": "Madera", "color": [0.5, 0.3, 0.1]})
    tri({"mat": "Metal pintado", "color": [0.2, 0.2, 0.2]})
    tri({"mat": "[Water Pool Light]", "color": [0.0, 0.5, 1.0]})
    tri({"mat": "Sin registro mar", "color": [0.0, 0.6, 1.0]})
    tri({"mat": "Vidrio", "color": [0.8, 0.9, 1.0], "opacity": 0.35})
    tri({"color": [0.8, 0.9, 1.0], "opacity": 0.5})              # glass guess
    tri({"color": [0.8, 0.9, 1.0], "opacity": 0.5004})           # same key
    tri({"color": [0.8, 0.9, 1.0], "opacity": 0.9995})           # opaque
    tri({"color": [0.8, 0.9, 1.0], "opacity": 0.998})
    tri({"color": [0, 1, 0]})                                    # ints
    tri({"color": [0.25, 0.5, 0.75, 1.0]})
    tri({"color": [-0.0, 0.0, 1]})
    tri({"color": [0.0, 0.0, 1]})                                # same key
    tri({"color": [True, 0.5, 0.5]})
    tri({})                                                       # cream
    tri({"mat": "  ", "color": [0.1, 0.1, 0.1]})
    tri({"mat": "12 años", "color": [0.1, 0.2, 0.1]})
    tri({"mat": "ñandú/ñu", "color": [0.1, 0.2, 0.3]})
    tri({"hidden": True, "color": [1.0, 0.0, 0.0]})
    tri({"layer": "Oculta", "color": [1.0, 0.0, 1.0]})
    for k, name in enumerate(("wood.png", "ladrillo rústico.png",
                              "cut64_hit.png", "cut64_missed.png",
                              "cut100_hit.png", "cut100_missed.png",
                              "cut_a31.png", "cut_a32.png", "cut_odd.png",
                              "cut_big.png", "cut_odd_hit.png",
                              "cut_big_hit.png", "cut128_hit.png",
                              "cut128_missed.png", "opaque_rgba.png", "rgb.png",
                              "ga.png", "pal_trns.png", "pal_opaque_trns.png",
                              "key.png", "rgba16.png", "rgba16_high.png",
                              "photo.jpg", "photo.jpeg", "notimage.png",
                              "missing.png", "nodir/also missing.JPG")):
        path = tex / name if "missing" not in name else tex / name
        tri({"texture": {"path": str(path), "sw": 1.0 + k * 0.1, "sh": 1.0}},
            dy=3.0)
    # The same image translucent and not, with and without a finish name.
    tri({"texture": {"path": str(tex / "cut64_hit.png"), "sw": 1.0, "sh": 1.0},
         "opacity": 0.6}, dy=3.0)
    tri({"texture": {"path": str(tex / "wood.png"), "sw": 1.0, "sh": 1.0},
         "opacity": 0.4, "mat": "Madera"}, dy=3.0)
    tri({"texture": {"path": str(tex / "wood.png"), "sw": 1.0, "sh": 1.0},
         "mat": "Metal pintado"}, dy=3.0)
    tri({"texture": {"path": str(tex / "photo.jpg"), "sw": 0.0, "sh": 0.0,
                     "rot": 20.0}}, dy=3.0)
    # A texture whose path is relative to nowhere, and an empty path.
    tri({"texture": {"path": "", "sw": 1.0, "sh": 1.0}, "color": [0.4, 0.4, 0.4]},
        dy=3.0)
    return s


NEW_SCENES = {
    "figures": scene_figures,
    "hierarchy": scene_hierarchy,
    "finishes": scene_finishes,
}


def relative_paths(node, root: str):
    if isinstance(node, dict):
        return {k: (os.path.relpath(v, root).replace("\\", "/")
                    if k == "path" and isinstance(v, str) and v else
                    relative_paths(v, root)) for k, v in node.items()}
    if isinstance(node, list):
        return [relative_paths(x, root) for x in node]
    return node


def save_scene_json(scene, dest: Path, root: Path) -> None:
    from formats import igz
    saved = (igz._LARGE_DOCUMENT, igz._pack_textures)
    igz._LARGE_DOCUMENT = 10 ** 9
    igz._pack_textures = lambda payload: ({}, 0)
    try:
        igz.save_scene(scene, dest)
    finally:
        igz._LARGE_DOCUMENT, igz._pack_textures = saved
    doc = json.loads(dest.read_bytes().decode("utf-8"))
    doc = relative_paths(doc, str(root))
    dest.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")


def export_all() -> list:
    """GLBs of every scene; the .igz of the new ones. Returns the names."""
    from core.i18n import tr
    assert tr("Loose geometry") == "Loose geometry" and tr("Figure") == "Figure"
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    tex = OUT / "tex"
    png_variants(tex)
    shutil.copy(EXPORT / "tex" / "wood.png", tex / "wood.png")
    shutil.copy(EXPORT / "tex" / "ladrillo rústico.png",
                tex / "ladrillo rústico.png")
    names = []
    for name, make in mf.SCENES.items():
        # Their .igz and images are in export/ (the OBJ and STL tests).
        scene = make(EXPORT / "tex")
        gltf.save_glb(scene, OUT / f"{name}.glb")
        names.append(name)
    for name, make in NEW_SCENES.items():
        scene = make(tex)
        save_scene_json(scene, OUT / f"{name}.igz.json", OUT)
        gltf.save_glb(scene, OUT / f"{name}.glb")
        names.append(name)
        print(name, "exported", (OUT / f"{name}.glb").stat().st_size, "bytes")
    return names


# ---- hand-made glTF -----------------------------------------------------------------

F32, U32, U16, U8, I16, I8 = 5126, 5125, 5123, 5121, 5122, 5120


class Gltf:
    """A glTF under construction: one buffer, views and accessors on demand."""

    def __init__(self, generator="Hand"):
        self.buf = bytearray()
        self.js = {"asset": {"version": "2.0", "generator": generator}}

    def view(self, data: bytes, stride=None) -> int:
        while len(self.buf) % 4:
            self.buf.append(0)
        v = {"buffer": 0, "byteOffset": len(self.buf), "byteLength": len(data)}
        if stride:
            v["byteStride"] = stride
        self.buf += data
        self.js.setdefault("bufferViews", []).append(v)
        return len(self.js["bufferViews"]) - 1

    def accessor(self, view, comp, count, typ, offset=0, **extra) -> int:
        a = {"bufferView": view, "componentType": comp, "count": count,
             "type": typ}
        if offset:
            a["byteOffset"] = offset
        a.update(extra)
        self.js.setdefault("accessors", []).append(a)
        return len(self.js["accessors"]) - 1

    def vec3(self, pts) -> int:
        data = b"".join(struct.pack("<3f", *p) for p in pts)
        return self.accessor(self.view(data), F32, len(pts), "VEC3")

    def vec2(self, uvs) -> int:
        data = b"".join(struct.pack("<2f", *p) for p in uvs)
        return self.accessor(self.view(data), F32, len(uvs), "VEC2")

    def indices(self, idx, comp=U16) -> int:
        fmt = {U16: "H", U32: "I", U8: "B"}[comp]
        data = struct.pack(f"<{len(idx)}{fmt}", *idx)
        return self.accessor(self.view(data), comp, len(idx), "SCALAR")

    def mesh(self, primitives, name=None) -> int:
        m = {"primitives": primitives}
        if name:
            m["name"] = name
        self.js.setdefault("meshes", []).append(m)
        return len(self.js["meshes"]) - 1

    def node(self, **kw) -> int:
        self.js.setdefault("nodes", []).append(kw)
        return len(self.js["nodes"]) - 1

    def material(self, **kw) -> int:
        self.js.setdefault("materials", []).append(kw)
        return len(self.js["materials"]) - 1

    def image_view(self, blob: bytes, mime="image/png") -> int:
        v = self.view(blob)
        self.js.setdefault("images", []).append(
            {"bufferView": v, "mimeType": mime})
        return len(self.js["images"]) - 1

    def texture(self, image) -> int:
        self.js.setdefault("textures", []).append({"source": image,
                                                   "sampler": 0})
        self.js.setdefault("samplers", [{}])
        return len(self.js["textures"]) - 1

    def scene(self, *roots) -> None:
        self.js.setdefault("scenes", []).append({"nodes": list(roots)})
        self.js["scene"] = 0

    def glb(self) -> bytes:
        js = dict(self.js)
        if self.buf:
            js["buffers"] = [{"byteLength": len(self.buf)}]
        return pack_glb(js, bytes(self.buf))

    def gltf_with_bin(self, bin_name: str):
        js = dict(self.js)
        js["buffers"] = [{"byteLength": len(self.buf), "uri": bin_name}]
        return json.dumps(js).encode(), bytes(self.buf)

    def gltf_data_uri(self) -> bytes:
        js = dict(self.js)
        js["buffers"] = [{"byteLength": len(self.buf),
                          "uri": "data:application/octet-stream;base64,"
                          + base64.b64encode(bytes(self.buf)).decode()}]
        return json.dumps(js).encode()


def pack_glb(js: dict, binary: bytes = b"", json_pad=b" ") -> bytes:
    j = json.dumps(js, separators=(",", ":")).encode()
    j += json_pad * ((-len(j)) % 4)
    b = binary + b"\x00" * ((-len(binary)) % 4)
    total = 12 + 8 + len(j) + (8 + len(b) if b else 0)
    out = struct.pack("<III", 0x46546C67, 2, total)
    out += struct.pack("<II", len(j), 0x4E4F534A) + j
    if b:
        out += struct.pack("<II", len(b), 0x004E4942) + b
    return out


def cube(size=1.0, origin=(0, 0, 0)):
    """Y-up cube: 8 corners and 12 triangles."""
    ox, oy, oz = origin
    pts = [(ox + x * size, oy + y * size, oz + z * size)
           for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6),
             (0, 2, 6, 4), (1, 5, 7, 3)]
    idx = []
    for a, b, c, d in quads:
        idx += [a, b, c, a, c, d]
    return pts, idx


def sphere(n_lon=24, n_lat=16, r=1.5):
    pts = []
    for i in range(n_lat + 1):
        th = math.pi * i / n_lat
        for j in range(n_lon):
            ph = 2 * math.pi * j / n_lon
            pts.append((r * math.sin(th) * math.cos(ph), r * math.cos(th),
                        r * math.sin(th) * math.sin(ph)))
    idx = []
    for i in range(n_lat):
        for j in range(n_lon):
            a = i * n_lon + j
            b = i * n_lon + (j + 1) % n_lon
            c = (i + 1) * n_lon + j
            d = (i + 1) * n_lon + (j + 1) % n_lon
            if i > 0:
                idx += [a, c, b]
            if i < n_lat - 1:
                idx += [b, c, d]
    return pts, idx


def one_mesh(g: Gltf, pts, idx, material=None, uvs=None, comp=U16,
             uv_name="TEXCOORD_0", node=True, **node_kw):
    attrs = {"POSITION": g.vec3(pts)}
    if uvs is not None:
        attrs[uv_name] = g.vec2(uvs)
    prim = {"attributes": attrs}
    if idx is not None:
        prim["indices"] = g.indices(idx, comp)
    if material is not None:
        prim["material"] = material
    mi = g.mesh([prim])
    if node:
        n = g.node(mesh=mi, **node_kw)
        g.scene(n)
        return n
    return mi


def tri_uvs(pts, idx, rng, spread=2.0):
    return [(rng.uniform(-spread, spread), rng.uniform(-spread, spread))
            for _ in pts]


def write_imports() -> None:
    imp = OUT / "import"
    imp.mkdir(parents=True)
    tex = OUT / "tex"
    wood = (tex / "wood.png").read_bytes()
    cut = (tex / "cut64_hit.png").read_bytes()
    opaque_big = (tex / "rgb.png").read_bytes()
    jpg = (tex / "photo.jpg").read_bytes()
    cases = []
    rng = random.Random(20261019)

    def add(fname, data=None, **opts):
        if data is not None:
            (imp / fname).write_bytes(data)
        cases.append(dict(file=fname, **opts))

    # -- the exported files read back -------------------------------------------
    for name in list(mf.SCENES) + list(NEW_SCENES):
        cases.append(dict(file=f"../{name}.glb"))

    # -- plain geometry --------------------------------------------------------------
    pts, idx = cube()
    g = Gltf()
    one_mesh(g, pts, idx)
    add("cube_u16.glb", g.glb())
    g = Gltf()
    one_mesh(g, pts, idx, comp=U8)
    add("cube_u8.glb", g.glb())
    g = Gltf()
    one_mesh(g, pts, idx, comp=U32)
    add("cube_u32.glb", g.glb())
    g = Gltf()
    soup = [pts[i] for i in idx]
    one_mesh(g, soup, None)
    add("cube_soup.glb", g.glb())
    g = Gltf("IngeTrazo")
    one_mesh(g, [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
             [0, 1, 2, 0, 2, 3, 0])                        # a stray index
    add("stray_index.glb", g.glb())
    for dim, tag in ((0.05, "tiny"), (15.0, "m"), (100.0, "inch"),
                     (1200.0, "inch2"), (5000.0, "mm"), (3e6, "huge"),
                     (0.0, "flat")):
        g = Gltf()
        p, i = cube(dim)
        if dim == 0.0:
            p = [(0, 0, 0)] * 8
        one_mesh(g, p, i)
        add(f"scale_{tag}.glb", g.glb())
    g = Gltf("IngeTrazo")
    p, i = cube(300.0)
    one_mesh(g, p, i)
    add("scale_ours_300.glb", g.glb())

    # -- materials -----------------------------------------------------------------------
    g = Gltf()
    m1 = g.material(pbrMetallicRoughness={"baseColorFactor":
                                          [0.8, 0.2, 0.1, 1.0]})
    m2 = g.material(pbrMetallicRoughness={"baseColorFactor":
                                          [1.0, 1.0, 1.0, 1.0]})
    m3 = g.material(pbrMetallicRoughness={"baseColorFactor":
                                          [0.1, 0.5, 0.9, 0.4]},
                    alphaMode="BLEND")
    m4 = g.material(pbrMetallicRoughness={"baseColorFactor": [0.1, 0.9, 0.9, 1]},
                    alphaMode="BLEND")
    m5 = g.material(pbrMetallicRoughness={"baseColorFactor":
                                          [0.3, 0.3, 0.3, 0.2]},
                    alphaMode="MASK")
    m6 = g.material()                                         # nothing at all
    m7 = g.material(pbrMetallicRoughness={"baseColorFactor": [0.5, 0.5]})
    m8 = g.material(pbrMetallicRoughness={"baseColorFactor": []})
    prims = []
    for k, mat in enumerate((m1, m2, m3, m4, m5, m6, m7, m8, None, 99)):
        p, i = cube(1.0, (3 * k, 0, 0))
        prims.append({"attributes": {"POSITION": g.vec3(p)},
                      "indices": g.indices(i),
                      **({"material": mat} if mat is not None else {})})
    g.scene(g.node(mesh=g.mesh(prims)))
    add("materials.glb", g.glb())

    # Specular-glossiness legacy.
    g = Gltf()
    sg = g.material(extensions={"KHR_materials_pbrSpecularGlossiness": {
        "diffuseFactor": [0.9, 0.4, 0.1, 0.5]}}, alphaMode="BLEND",
        pbrMetallicRoughness={"baseColorFactor": [0, 0, 1, 1]})
    p, i = cube()
    one_mesh(g, p, i, material=sg)
    add("specgloss.glb", g.glb())

    # Textured, embedded, with a texture transform.
    uv4 = [(0, 0), (1, 0), (1, 1), (0, 1)]
    quad = [(0, 0, 0), (2, 0, 0), (2, 1.5, 0), (0, 1.5, 0)]
    qi = [0, 1, 2, 0, 2, 3]
    g = Gltf()
    im = g.image_view(wood)
    tx = g.texture(im)
    mat = g.material(pbrMetallicRoughness={
        "baseColorTexture": {"index": tx, "extensions": {
            "KHR_texture_transform": {"offset": [0.25, 0.5],
                                      "scale": [2, 3]}}},
        "baseColorFactor": [0.5, 0.5, 0.5, 1]})
    one_mesh(g, quad, qi, material=mat, uvs=uv4)
    add("tex_embedded.glb", g.glb())

    g = Gltf("Sketchfab")
    g.js["asset"]["extras"] = {"title": "Una silla ñandú",
                               "author": "someone"}
    im = g.image_view(cut)
    tx = g.texture(im)
    mat = g.material(pbrMetallicRoughness={"baseColorTexture": {
        "index": tx, "texCoord": 1}}, alphaMode="BLEND")
    one_mesh(g, quad, qi, material=mat, uvs=uv4, uv_name="TEXCOORD_1")
    add("tex_texcoord1.gltf", g.gltf_data_uri())

    g = Gltf()
    im = g.image_view(wood)
    tx = g.texture(im)
    mat = g.material(pbrMetallicRoughness={"baseColorTexture": {"index": tx}},
                     alphaMode="BLEND")
    one_mesh(g, quad, qi, material=mat)                       # no UVs at all
    add("tex_nouv.glb", g.glb())

    g = Gltf()
    im = g.image_view(wood)
    tx = g.texture(im)
    mat = g.material(pbrMetallicRoughness={"baseColorTexture": {
        "index": tx, "texCoord": 3}, "baseColorFactor": [0.2, 0.4, 0.6, 0.7]},
        alphaMode="BLEND")
    one_mesh(g, quad, qi, material=mat, uvs=uv4)              # TEXCOORD_3 absent
    add("tex_wrong_set.glb", g.glb())

    # External buffer and images, and an image that does not exist.
    g = Gltf()
    g.js["images"] = [{"uri": "tex/wood.png"}, {"uri": "tex/missing.png"},
                      {"uri": "data:image/jpeg;base64,"
                       + base64.b64encode(jpg).decode()},
                      {"uri": ""}, {}, {"bufferView": 999}]
    g.js["samplers"] = [{}]
    g.js["textures"] = [{"source": k} for k in range(6)] + [{}, {"source": 77}]
    mats = [g.material(pbrMetallicRoughness={"baseColorTexture": {"index": k}})
            for k in range(8)]
    prims = []
    for k, m in enumerate(mats):
        p = [(3 * k, 0, 0), (3 * k + 2, 0, 0), (3 * k + 2, 1.5, 0),
             (3 * k, 1.5, 0)]
        prims.append({"attributes": {"POSITION": g.vec3(p),
                                     "TEXCOORD_0": g.vec2(uv4)},
                      "indices": g.indices(qi), "material": m})
    g.scene(g.node(mesh=g.mesh(prims)))
    j, b = g.gltf_with_bin("scene.bin")
    shutil.copytree(tex, imp / "tex")
    add("external.gltf", j)
    (imp / "scene.bin").write_bytes(b)

    # -- nodes ------------------------------------------------------------------------
    pts, idx = cube(1.0)
    g = Gltf()
    mi = one_mesh(g, pts, idx, node=False)
    c = g.node(mesh=mi, translation=[0, 1, 0], rotation=[0, 0.3826834, 0, 0.9238795])
    b = g.node(mesh=mi, scale=[2, 1, 0.5], translation=[5, 0, 0], children=[c])
    a = g.node(translation=[0, 0, -5], rotation=[0.5, 0.5, 0.5, 0.5],
               scale=[1, 2, 3], children=[b], mesh=mi)
    d = g.node(mesh=mi, matrix=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                                10, 2, 3, 1])
    e = g.node(mesh=mi, matrix=[0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 2, 0,
                                0, 0, 9, 1], translation=[100, 0, 0])
    f = g.node(mesh=mi, rotation=[1, 2, 3, 4])                 # not unit
    h = g.node(mesh=mi, scale=[0, 1, 1])
    g.js["scenes"] = [{"nodes": [a]}, {"nodes": [d, e]}, {"nodes": [f, h]}]
    add("nodes.glb", g.glb())

    g = Gltf()
    mi = one_mesh(g, pts, idx, node=False)
    n0 = g.node(mesh=mi, translation=[1, 0, 0])
    n1 = g.node(mesh=mi, translation=[0, 3, 0], children=[n0])
    n2 = g.node(mesh=99)                                       # no such mesh
    n3 = g.node(name="empty")
    add("noscene.glb", Gltf.glb(g))                           # roots: all nodes
    g.scene(n1)
    g.js["scenes"][0]["nodes"] = [n1, n0]                      # n0 twice
    add("nodes_twice.glb", g.glb())

    # Random TRS nodes: matmul order, to the last bit.
    g = Gltf()
    pts, idx = sphere(6, 4, 2.0)
    mi = one_mesh(g, pts, idx, node=False)
    roots = []
    for k in range(24):
        def rq():
            q = [rng.uniform(-1, 1) for _ in range(4)]
            n = math.sqrt(sum(v * v for v in q))
            return [v / n for v in q]
        kw = dict(mesh=mi, translation=[rng.uniform(-50, 50) for _ in range(3)],
                  rotation=rq(),
                  scale=[rng.uniform(0.2, 3) * rng.choice((1, -1))
                         for _ in range(3)])
        kids = []
        for _ in range(rng.randint(0, 2)):
            kids.append(g.node(mesh=mi, translation=[rng.uniform(-9, 9)
                                                     for _ in range(3)],
                               rotation=rq(),
                               scale=[rng.uniform(0.5, 2) for _ in range(3)]))
        if kids:
            kw["children"] = kids
        roots.append(g.node(**kw))
    g.scene(*roots)
    add("trs_random.glb", g.glb())

    # The same nodes over textured triangles: the f64 world points feed the
    # texture solve, so the matmul order shows in the hashed uvw.
    g = Gltf()
    im = g.image_view(wood)
    mat = g.material(pbrMetallicRoughness={"baseColorTexture":
                                           {"index": g.texture(im)}})
    tp, tu, ti = [], [], []
    for t in range(6):
        b = len(tp)
        o = [rng.uniform(-2, 2) for _ in range(3)]
        for _ in range(3):
            tp.append(tuple(o[k] + rng.uniform(-1, 1) for k in range(3)))
            tu.append((rng.uniform(-2, 2), rng.uniform(-2, 2)))
        ti += [b, b + 1, b + 2]
    mi = one_mesh(g, tp, ti, material=mat, uvs=tu, node=False)
    roots = []
    for k in range(14):
        q = [rng.uniform(-1, 1) for _ in range(4)]
        n = math.sqrt(sum(v * v for v in q))
        kw = dict(mesh=mi, translation=[rng.uniform(-40, 40) for _ in range(3)],
                  rotation=[v / n for v in q],
                  scale=[rng.uniform(0.3, 2.5) for _ in range(3)])
        roots.append(g.node(**kw))
    g.scene(*roots)
    add("trs_textured.glb", g.glb())

    # -- accessors ------------------------------------------------------------------------
    pts, idx = cube(2.0)
    g = Gltf()
    inter = b"".join(struct.pack("<3f3f", *p, 0.0, 1.0, 0.0) for p in pts)
    v = g.view(b"\xff" * 4 + inter, stride=24)
    pos = g.accessor(v, F32, 8, "VEC3", offset=4)
    prim = {"attributes": {"POSITION": pos}, "indices": g.indices(idx)}
    g.scene(g.node(mesh=g.mesh([prim])))
    add("interleaved.glb", g.glb())

    g = Gltf()
    shorts = b"".join(struct.pack("<3h", *[int(c * 10000) for c in p])
                      for p in pts)
    pos = g.accessor(g.view(shorts), I16, 8, "VEC3", normalized=True)
    prim = {"attributes": {"POSITION": pos}, "indices": g.indices(idx)}
    g.scene(g.node(mesh=g.mesh([prim])))
    add("normalized_short.glb", g.glb())

    g = Gltf()
    bytes3 = b"".join(struct.pack("<3b", *[int(c * 60) - 60 for c in p])
                      + b"\x00" for p in pts)
    pos = g.accessor(g.view(bytes3, stride=4), I8, 8, "VEC3", normalized=False)
    prim = {"attributes": {"POSITION": pos}, "indices": g.indices(idx)}
    g.scene(g.node(mesh=g.mesh([prim])))
    add("bytes_position.glb", g.glb())

    g = Gltf()                                    # float indices, signed index
    sidx = struct.pack(f"<{len(idx)}h", *idx)
    prim = {"attributes": {"POSITION": g.vec3(pts)},
            "indices": g.accessor(g.view(sidx), I16, len(idx), "SCALAR")}
    g.scene(g.node(mesh=g.mesh([prim])))
    add("short_indices.glb", g.glb())

    g = Gltf()                                    # primitive modes
    prims = [{"attributes": {"POSITION": g.vec3(pts)}, "indices":
              g.indices(idx), "mode": 4},
             {"attributes": {"POSITION": g.vec3(pts)}, "mode": 1},
             {"attributes": {"POSITION": g.vec3(pts)}, "mode": 5},
             {"attributes": {"NORMAL": g.vec3(pts)}}]
    g.scene(g.node(mesh=g.mesh(prims)))
    add("modes.glb", g.glb())

    # -- UV maps: the solve ------------------------------------------------------------------
    for tag, n_tris, mag, spread in (("small", 30, 1.0, 1.0),
                                     ("mixed", 150, 30.0, 4.0),
                                     ("tiny", 200, 0.003, 1.0)):
        g = Gltf("IngeTrazo")
        pts, uvs, idx = [], [], []
        for t in range(n_tris):
            base = len(pts)
            o = [rng.uniform(-20, 20) for _ in range(3)]
            scale = mag * 10 ** rng.uniform(-1.5, 1.0) if tag == "mixed" else mag
            for _ in range(3):
                pts.append(tuple(o[k] + rng.uniform(-1, 1) * scale
                                 for k in range(3)))
                uvs.append((rng.uniform(-spread, spread),
                            rng.uniform(-spread, spread)))
            idx += [base, base + 1, base + 2]
        im = g.image_view(wood)
        tx = g.texture(im)
        mat = g.material(pbrMetallicRoughness={"baseColorTexture":
                                               {"index": tx}})
        one_mesh(g, pts, idx, material=mat, uvs=uvs, comp=U32)
        add(f"uv_random_{tag}.glb", g.glb())

    # Degenerate triangles, collapsed corners, repeats.
    g = Gltf("IngeTrazo")
    pts = [(0, 0, 0), (1, 0, 0), (2, 0, 0),              # collinear
           (0, 1, 0), (1, 1, 0), (0, 2, 0),
           (5, 5, 5), (5, 5, 5), (5, 5, 5),               # a point
           (0, 0, 0), (1, 0, 0), (0, 1, 0),               # repeated corners
           (3, 0, 0), (3.00001, 0, 0), (3, 0.00001, 0)]   # welds
    uvs = [(0, 0), (1, 0), (2, 0), (0, 0), (1, 0), (0, 1), (0, 0), (0, 0),
           (0, 0), (0, 0), (1, 0), (0, 1), (0, 0), (1, 0), (0, 1)]
    im = g.image_view(wood)
    mat = g.material(pbrMetallicRoughness={"baseColorTexture":
                                           {"index": g.texture(im)}},
                     alphaMode="BLEND")
    one_mesh(g, pts, list(range(15)), material=mat, uvs=uvs)
    add("degenerate.glb", g.glb())

    # -- smoothing, foliage ------------------------------------------------------------------
    pts, idx = sphere(16, 10)
    g = Gltf("Blender")
    mat = g.material(pbrMetallicRoughness={"baseColorFactor": [0.9, 0.5, 0.2, 1]})
    one_mesh(g, pts, idx, material=mat)
    add("sphere.glb", g.glb())

    g = Gltf("Blender")                        # two materials on one sphere
    prims = []
    half = len(idx) // 6 * 3
    for part, color in ((idx[:half], [0.9, 0.5, 0.2, 1]),
                        (idx[half:], [0.1, 0.5, 0.2, 1])):
        prims.append({"attributes": {"POSITION": g.vec3(pts)},
                      "indices": g.indices(part),
                      "material": g.material(pbrMetallicRoughness={
                          "baseColorFactor": color})})
    g.scene(g.node(mesh=g.mesh(prims)))
    add("sphere2.glb", g.glb())

    for tag, png_bytes, n, size, tex_alpha in (
            ("foliage_alpha", cut, 40, 0.5, True),
            ("foliage_confetti", opaque_big, 210, 0.1, False),
            ("foliage_big_opaque", opaque_big, 210, 1.5, False),
            ("foliage_few", opaque_big, 30, 0.1, False)):
        g = Gltf("Sketchfab")
        im = g.image_view(png_bytes)
        mat = g.material(pbrMetallicRoughness={"baseColorTexture":
                                               {"index": g.texture(im)}})
        pts, uvs, idx = [], [], []
        for k in range(n):
            o = (rng.uniform(-3, 3), rng.uniform(0, 3), rng.uniform(-3, 3))
            b = len(pts)
            pts += [o, (o[0] + size, o[1], o[2]),
                    (o[0] + size, o[1] + size, o[2]), (o[0], o[1] + size, o[2])]
            uvs += uv4
            idx += [b, b + 1, b + 2, b, b + 2, b + 3]
        one_mesh(g, pts, idx, material=mat, uvs=uvs, comp=U32)
        add(f"{tag}.glb", g.glb())

    # A textured sphere: smoothing needs the same image and colour.
    pts, idx = sphere(12, 8, 1.0)
    g = Gltf("Sketchfab")
    im = g.image_view(wood)
    mat = g.material(pbrMetallicRoughness={"baseColorTexture":
                                           {"index": g.texture(im)}})
    uvs = [(j / 12, i / 8) for i in range(9) for j in range(12)]
    one_mesh(g, pts, idx, material=mat, uvs=uvs)
    add("sphere_textured.glb", g.glb())

    # -- titles and names -----------------------------------------------------------------------
    pts, idx = cube()
    for tag, extras in (("title", {"title": "Mi modelo"}),
                        ("title_long", {"title": "x" * 80}),
                        ("title_empty", {"title": ""}),
                        ("title_num", {"title": 5}),
                        ("extras_empty", {}),
                        ("title_unicode", {"title": "ñandú ✓ ✓" * 10})):
        g = Gltf()
        g.js["asset"]["extras"] = extras
        one_mesh(g, pts, idx)
        add(f"{tag}.glb", g.glb())
    g = Gltf()
    one_mesh(g, pts, idx)
    add("a-very-long-file-name-" + "z" * 70 + ".glb", g.glb())
    add("dots.in.name.glb", g.glb())
    add("dotted.name.v2.glb", g.glb())

    # -- errors ----------------------------------------------------------------------------------
    g = Gltf()
    one_mesh(g, pts, idx)
    good = dict(g.js, buffers=[{"byteLength": len(g.buf)}])
    d = dict(good, extensionsRequired=["KHR_draco_mesh_compression"])
    add("err_draco.glb", pack_glb(d, bytes(g.buf)))
    d = dict(good, extensionsRequired=["KHR_texture_basisu"])
    add("ok_other_required.glb", pack_glb(d, bytes(g.buf)))
    acc = [dict(a) for a in good["accessors"]]
    acc[0]["sparse"] = {"count": 1}
    add("err_sparse.glb", pack_glb(dict(good, accessors=acc), bytes(g.buf)))
    acc = [dict(a) for a in good["accessors"]]
    acc[0]["componentType"] = 5124
    add("err_comptype.glb", pack_glb(dict(good, accessors=acc), bytes(g.buf)))
    acc = [dict(a) for a in good["accessors"]]
    acc[0]["type"] = "VEC9"
    add("err_type.glb", pack_glb(dict(good, accessors=acc), bytes(g.buf)))
    acc = [dict(a) for a in good["accessors"]]
    acc[0]["count"] = 10000
    add("err_count.glb", pack_glb(dict(good, accessors=acc), bytes(g.buf)))
    acc = [dict(a) for a in good["accessors"]]
    acc[1]["count"] = 9                                    # indices past the end
    add("err_index_oob.glb", pack_glb(dict(good, accessors=acc), bytes(g.buf)))
    g2 = Gltf()
    one_mesh(g2, pts, [0, 1, 2, 3, 4, 99])
    add("err_index_big.glb", g2.glb())
    g2 = Gltf()
    one_mesh(g2, pts, [0, 1, 2], comp=U16)
    add("ok_three.glb", g2.glb())
    add("err_nojson.glb", struct.pack("<III", 0x46546C67, 2, 20)
        + struct.pack("<II", 4, 0x004E4942) + b"\0\0\0\0")
    add("err_short.glb", b"glTF\x02\x00")
    add("err_badjson.glb", pack_glb({}, b"")[:20] + b"{notjson}    ")
    add("err_text.gltf", b"this is not json")
    add("err_empty.gltf", b"")
    add("err_notobj.gltf", b"[1, 2, 3]")
    add("err_nomesh.gltf", json.dumps({"asset": {"version": "2.0"}}).encode())
    add("err_nobuffer.gltf", json.dumps({
        "asset": {"version": "2.0"},
        "buffers": [{"uri": "nowhere.bin", "byteLength": 4}]}).encode())
    add("err_b64.gltf", json.dumps({
        "asset": {"version": "2.0"},
        "buffers": [{"uri": "data:application/octet-stream;base64,QQ",
                     "byteLength": 1}]}).encode())
    add("err_b64comma.gltf", json.dumps({
        "asset": {"version": "2.0"},
        "buffers": [{"uri": "data:nonsense", "byteLength": 1}]}).encode())
    g2 = Gltf()
    one_mesh(g2, pts, idx[:2])                             # < 3 indices
    add("err_two_indices.glb", g2.glb())
    g2 = Gltf("IngeTrazo")
    one_mesh(g2, pts, idx[:2])
    add("ok_two_indices_ours.glb", g2.glb())
    g2 = Gltf()
    one_mesh(g2, pts, idx[:2])
    one_mesh(g2, pts, idx)                                 # one empty batch
    add("err_empty_batch.glb", g2.glb())
    g = Gltf()
    g.scene(g.node(mesh=g.mesh([{"attributes": {}}])))
    add("err_no_position.glb", g.glb())
    g = Gltf()
    n0 = g.node(children=[1])
    n1 = g.node(children=[0])
    g.scene(n0)
    add("err_cycle.glb", g.glb())
    g = Gltf()
    g.js["nodes"] = [{"mesh": 0, "scale": [1, 2]}]
    g.js["meshes"] = [{"primitives": []}]
    g.scene(0)
    add("err_scale.glb", g.glb())
    g = Gltf()
    g.js["nodes"] = [{"mesh": 0, "matrix": [1, 2, 3]}]
    g.js["meshes"] = [{"primitives": []}]
    g.scene(0)
    add("err_matrix.glb", g.glb())

    # -- run them with the real importer ------------------------------------------------------
    kept = []
    for case in cases:
        path = (imp / case["file"]).resolve()
        scene = Scene()
        try:
            glb.load_glb(scene, path)
        except Exception as exc:  # noqa: BLE001 — the message is the datum
            case["error"] = f"{type(exc).__name__}: {exc}"
            kept.append(case)
            continue
        tag = case["file"].replace("../", "export-")
        case["sig"] = f"{tag}.sig.json"
        mf.write_json(imp / case["sig"], scene_signature(scene))
        kept.append(case)
    names = [c.get("sig") for c in kept if "sig" in c]
    assert len(names) == len(set(names)), "signature names clash"
    mf.write_json(imp / "cases.json", kept)
    print(len(kept), "import cases,",
          sum(1 for c in kept if "error" in c), "errors")


def write_solve_vectors() -> None:
    """Random 3 x 3 systems and what `np.linalg.solve` answered: the Rust
    LU is held to the last bit even where no file exercises it."""
    import numpy as np
    rng = np.random.default_rng(11)
    lines = []
    for k in range(1500):
        scale = 10.0 ** rng.uniform(-3, 3)
        a = rng.normal(size=(3, 3)) * scale
        if k % 7 == 0:
            a[1] = a[0] * rng.uniform(0.5, 2) + rng.normal(size=3) * 1e-9
        if k % 11 == 0:
            a[:, 2] = 0.0
            a[2, 2] = 1.0
        if k % 13 == 0:
            a = np.round(a, 1)
        b = np.array([rng.normal(), rng.normal(), 0.0]) * 10.0 ** rng.uniform(-2, 2)
        try:
            x = np.linalg.solve(a[None], b[None, :, None])[0, :, 0]
        except np.linalg.LinAlgError:
            continue
        det = float(np.linalg.det(a[None])[0])
        row = [struct.pack(">d", v).hex() for v in
               list(a.reshape(-1)) + list(b) + list(x) + [det]]
        lines.append(" ".join(row))
    (OUT / "solve.txt").write_text("\n".join(lines) + "\n", encoding="ascii",
                                   newline="\n")
    print(len(lines), "solve vectors")


def main() -> None:
    app = QGuiApplication.instance() or QGuiApplication(sys.argv)  # noqa: F841
    export_all()
    write_imports()
    write_solve_vectors()
    shutil.rmtree(CACHE, ignore_errors=True)


if __name__ == "__main__":
    main()
