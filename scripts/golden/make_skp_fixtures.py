# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""Write the fixtures that hold the Rust SketchUp reader and importer
(``ingetrazo-skp`` and ``ingetrazo-formats-mesh``, module ``skp_in``) to the
Python ones (docs/rust-rewrite-plan.md, M5.7).

The ``.skp`` inputs are written here, by ``skp_synth.py`` (the classic MFC
archive) and ``skp_synth_vff.py`` (the 2021+ zip): they contain no byte of
any SketchUp-written template. For each file the REAL Python code runs:

* ``openskp.SkpFile.parse()`` plus the two fixes IngeTrazo applies
  (``formats/openskp_compat``): dumped as ``<name>.model.json``
  (``skp_dump.py``), what the Rust parse must reproduce bit for bit;
* ``formats.skp.parse_skp`` + ``apply_payload`` into a fresh ``Scene``: its
  L2 signature as ``<name>.sig.json`` and, in ``cases.json``, the things the
  signature only counts (saved views, dimensions, leader texts, the style's
  back colour). A file Python cannot read is recorded with ``unreadable``. A re-tinted texture is written by
  Qt's PNG encoder here and by another one in Rust, so every PNG of the
  texture cache is first replaced by the digest of its pixels (the Rust test
  does the same); a texture re-tinted from a JPEG is flagged
  ``tint_from_jpeg`` (Qt and the Rust decoder differ by a level or two).

``real_cases.json`` is written when ``--real LIST`` names a text file with one
path per line: the same two dumps, as hashes, for models on this machine. They
are NOT copied into the repository, and the file (paths of private models) is
git-ignored; the test skips each model that is not there.

The files are committed, so ``cargo test`` needs no Python. Rerun only to
widen them (PYTHONPATH at the repository root, native engine OFF, the main
venv's python):

    python scripts/golden/make_skp_fixtures.py [--real LIST]

Deleted with the Python app at M10.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import sys
import tempfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.pop("INGETRAZO_RUST", None)
CACHE = Path(tempfile.mkdtemp(prefix="ingetrazo-skp-fixtures-"))
os.environ["INGETRAZO_TEXTURE_CACHE"] = str(CACHE)

from PySide6.QtGui import QGuiApplication  # noqa: E402

import skp_dump  # noqa: E402
import skp_synth as S  # noqa: E402
import skp_synth_vff as V  # noqa: E402
from core.scene import Scene  # noqa: E402
from formats import openskp_compat  # noqa: E402
from formats import skp as skp_format  # noqa: E402
from formats import skp_openskp  # noqa: E402
from tests.golden.signature import scene_signature  # noqa: E402

OUT = ROOT / "rust" / "crates" / "ingetrazo-skp" / "tests" / "data"


# ---- small helpers ------------------------------------------------------------------

def dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path: Path, obj) -> None:
    path.write_text(dumps(obj) + "\n", encoding="utf-8", newline="\n")


def png(w=8, h=8, color=(180, 120, 60, 255), pattern=False) -> bytes:
    """A deterministic RGBA PNG (no Qt encoder involved)."""
    rows = b""
    for y in range(h):
        row = b"\x00"
        for x in range(w):
            c = color
            if pattern and (x + y) % 3 == 0:
                # a second colour when given, else the first one darkened
                c = tuple(pattern) if isinstance(pattern, (tuple, list)) else (color[0] // 2, color[1] // 2, color[2] // 2, color[3])
            row += bytes(c)
        rows += row

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


FAKE_JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00" + bytes(range(64)) + b"\xff\xd9"


def xf(tx=0.0, ty=0.0, tz=0.0, rot=0.0, scale=1.0, sx=None):
    """13 numbers of an instance: row-major 3x3, translation, scale scalar."""
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    k = [scale if sx is None else sx, scale, scale]
    return [c * k[0], -s * k[1], 0.0, s * k[0], c * k[1], 0.0, 0.0, 0.0, k[2], tx, ty, tz, 1.0]


def grid(add_face, n=8, size=10.0, z=0.0, mats=(None, None), m=None):
    """An n x m grid of quads (m = n by default); ``add_face(pts, mat)``."""
    count = 0
    for i in range(n):
        for j in range(n if m is None else m):
            pts = [(i * size, j * size, z), ((i + 1) * size, j * size, z),
                   ((i + 1) * size, (j + 1) * size, z), (i * size, (j + 1) * size, z)]
            add_face(pts, mats[(i + j) % 2])
            count += 1
    return count


# ---- classic (MFC) scenarios ---------------------------------------------------------

def legacy_basic(path, ver=17):
    d = S.Doc(ver)
    red = d.material("Red", rgba=(255, 0, 0, 255))
    blue = d.material("Blue", rgba=(0, 0, 255, 255), opacity=0.5)
    green = d.material("Green", rgba=(0, 200, 0, 255))
    d.layer("Layer0")
    walls = d.layer("Walls", hidden=1, rgba=(200, 30, 30, 255))
    roof = d.layer("Roof", rgba=(10, 20, 30, 255))
    w = d.w
    sq = [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0)]
    shift = lambda pts, dx, dy, dz=0: [(x + dx, y + dy, z + dz) for x, y, z in pts]  # noqa: E731
    items = [
        lambda: w.face(sq, mat=red),
        lambda: w.face([(20, 0, 0), (30, 0, 0), (25, 8, 0)], mat=blue, back=green),
        lambda: w.face(shift(sq, 40, 0), holes=[[(43, 3, 0), (47, 3, 0), (47, 7, 0), (43, 7, 0)]], mat=red),
        lambda: w.face(shift(sq, 60, 0), soft=1, mat=green),
        lambda: w.face(shift(sq, 80, 0), hidden_edges=1),
        lambda: w.face(shift(sq, 100, 0), smooth=1, soft=1),
        lambda: w.face(shift(sq, 120, 0), hidden=1, mat=red),
        lambda: w.face(shift(sq, 140, 0), layer=walls, mat=red),
        lambda: w.face(shift(sq, 160, 0), layer=roof, mat=blue),
        lambda: w.face(shift(sq, 0, 20), mat=red),
        lambda: w.face(shift(sq, 10, 20), mat=green),  # shares an edge with the one before
        lambda: w.edge_item((200, 0, 0), (210, 5, 5), soft=1),
        lambda: w.construction_line((0, 0, 0), (1, 0, 0)),
        lambda: w.construction_point((5, 5, 5)),
        lambda: w.section_plane((0.0, 0.0, 1.0, -3.0), name="Cut", label="C1"),
        lambda: w.text("Hello", (1, 2, 3), label=(10, 10, 10)),
        lambda: w.text("Screen", (4, 5, 6)),
        lambda: w.dimension((0, 0, 0), (100, 0, 0), offset=12.0, text=""),
    ]
    for it in items:
        d.root_item(it)
    d.finish(path)


def legacy_components(path, ver=17):
    d = S.Doc(ver)
    red = d.material("Red", rgba=(255, 0, 0, 255))
    blue = d.material("Blue", rgba=(0, 0, 255, 255))
    green = d.material("Green", rgba=(0, 200, 0, 255))
    d.layer("Layer0")
    walls = d.layer("Walls")
    roof = d.layer("Roof", hidden=1)
    w = d.w
    sq = [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0)]

    def small():
        w.face(sq, mat=red)
        w.face([(0, 0, 10), (10, 0, 10), (10, 10, 10), (0, 10, 10)], mat=blue)
        return 2

    def bench():
        return grid(lambda pts, m: w.face(pts, mat=m or 0), 8, 10.0, 0.0, (red, 0))

    small_d = d.definition("Small", small)
    bench_d = d.definition("Bench", bench)

    def outer():
        w.face(sq, mat=green)
        w.instance(small_d, xf(20, 0, 0), name="s")
        w.instance(small_d, xf(40, 0, 0, rot=90), name="s2")
        return 3

    def outer2():
        w.face(sq, mat=red)
        w.instance(bench_d, xf(0, 100, 0), name="bench inside")
        return 2

    def wrapper():
        w.face(sq, mat=blue)
        w.instance(small_d, xf(0, 0, 30), name="tagged", layer=walls)
        return 2

    def gdef():
        w.face(sq, mat=green)
        w.face(sq, mat=0)
        return 2

    def sixty():
        return grid(lambda pts, m: w.face(pts, mat=m or 0), 6, 10.0, 0.0, (red, 0), m=10)

    def hundred():
        return grid(lambda pts, m: w.face(pts, mat=m or 0), 10, 10.0, 0.0, (blue, 0))

    sixty_d = d.definition("Sixty", sixty)          # 60 polygons, 8 copies: 60 * 7 = 420 >= 400
    hundred_d = d.definition("Hundred", hundred)    # 100 polygons, 5 copies: 100 * 4 = 400 >= 400
    outer_d = d.definition("Outer", outer)
    outer2_d = d.definition("Outer2", outer2)
    wrapper_d = d.definition("Wrapper", wrapper)
    g_d = d.definition("Gdef", gdef)
    for i in range(9):
        kw = {}
        if i in (2, 5):
            kw["layer"] = roof
        if i in (3, 7):
            kw["mat"] = green
        d.root_item(lambda i=i, kw=kw: w.instance(bench_d, xf(i * 100, 0, 0, rot=15 * i, scale=1 + 0.1 * i), name=f"bench{i}", **kw))
    for i in range(8):
        d.root_item(lambda i=i: w.instance(sixty_d, xf(i * 100, 900, 0), name=f"sixty{i}"))
    for i in range(5):
        d.root_item(lambda i=i: w.instance(hundred_d, xf(i * 150, 1000, 0, rot=5 * i), name=f"hundred{i}"))
    d.root_item(lambda: w.instance(small_d, xf(0, 300, 0), name="a"))
    d.root_item(lambda: w.instance(small_d, xf(30, 300, 0, rot=45), name="b"))
    d.root_item(lambda: w.instance(outer_d, xf(0, 400, 0), name="once"))
    d.root_item(lambda: w.instance(outer2_d, xf(0, 500, 0), name="once-with-shared"))
    d.root_item(lambda: w.instance(wrapper_d, xf(0, 600, 0), name="wrap", layer=walls))
    d.root_item(lambda: w.instance(g_d, xf(0, 700, 0, rot=30), name="grp", group=True))
    d.root_item(lambda: w.instance(g_d, xf(50, 700, 0), name="hidden grp", group=True, hidden=1))
    d.root_item(lambda: w.face(sq, mat=red))
    d.finish(path)


def legacy_textures(path, ver=17):
    d = S.Doc(ver)
    # the stored average colour equals the image's: Qt would not re-tint it
    brick = d.material("Brick", texture=(png(16, 16, (160, 70, 50, 255)), "brick.png", 20.0, 10.0, False),
                       rgba=(160, 70, 50, 255))
    photo = d.material("Photo", texture=(FAKE_JPEG, "photo.jpg", 30.0, 30.0, False), rgba=(90, 90, 90, 255),
                       opacity=0.8)
    paint = d.material("Paint", rgba=(250, 250, 250, 255))
    d.layer("Layer0")
    w = d.w
    wall = [(0, 0, 0), (40, 0, 0), (40, 0, 30), (0, 0, 30)]
    floor = [(0, 20, 0), (40, 20, 0), (40, 60, 0), (0, 60, 0)]
    ident = (1, 0, 0, 0, 1, 0, 0, 0, 1)
    items = [
        lambda: w.face(wall, mat=brick),
        lambda: w.face(floor, mat=brick, ftc=((0.5, 0.0, 0.0, 0.0, 0.25, 0.0, 12.0, 6.0, 1.0), ident, False, False)),
        lambda: w.face([(60, 0, 0), (100, 0, 0), (100, 0, 30), (60, 0, 30)], back=brick),
        lambda: w.face([(120, 0, 0), (160, 0, 0), (160, 0, 30), (120, 0, 30)], mat=brick, back=photo),
        lambda: w.face([(0, 100, 0), (30, 100, 0), (40, 110, 0), (30, 125, 0), (0, 120, 0)], mat=photo),
        lambda: w.face([(60, 100, 0), (100, 100, 0), (100, 140, 0), (60, 140, 0)], mat=brick,
                       ftc=((0.02, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 1.0), ident, True, False)),
        lambda: w.face([(120, 100, 0), (160, 100, 0), (160, 140, 0), (120, 140, 0)], mat=paint, back=brick),
        lambda: w.face([(0, 200, 0), (40, 200, 20), (40, 240, 20), (0, 240, 0)], mat=brick,
                       ftc=(ident, (0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 3.0, 4.0, 1.0), False, False)),
    ]
    for it in items:
        d.root_item(it)
    d.finish(path)


def legacy_attrs(path, ver=17):
    d = S.Doc(ver)
    red = d.material("Red", rgba=(255, 0, 0, 255))
    blue = d.material("Blue", rgba=(0, 0, 255, 255))
    d.layer("Layer0")
    w = d.w
    sq = [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0)]
    comp = d.definition("Part", lambda: (w.face(sq, mat=red), 1)[1])
    attrs = [("dynamic_attributes", {"name": "Door", "w": 90.0, "n": 3, "flag": True, "pt": (1.0, 2.0, 3.0),
                                     "list": [1, "two", 3.5], "nothing": None}),
             ("other", {"k": "v"})]
    d.root_item(lambda: w.instance(comp, xf(0, 0, 0), name="with attrs", attrs=attrs))
    d.root_item(lambda: w.instance(comp, xf(20, 0, 0), name="plain"))
    d.finish(path)


def legacy_faceme(path, ver=17):
    d = S.Doc(ver)
    red = d.material("Red", rgba=(255, 0, 0, 255))
    blue = d.material("Blue", rgba=(0, 0, 255, 255))
    d.layer("Layer0")
    w = d.w
    sq = [(0, 0, 0), (10, 0, 0), (10, 0, 20), (0, 0, 20)]
    person = d.definition("Person", lambda: (w.face(sq, mat=red), 1)[1], behavior=1)
    d.root_item(lambda: w.instance(person, xf(0, 0, 0), name="susan"))
    d.finish(path)


def legacy_image(path, ver=17):
    d = S.Doc(ver)
    photo = d.material("Photo", texture=(png(8, 8, (10, 200, 10, 128)), "p.png", 10.0, 10.0, False),
                       rgba=(10, 200, 10, 128))
    d.material("Other", rgba=(1, 2, 3, 255))
    d.layer("Layer0")
    w = d.w
    quad = [(0, 0, 0), (48, 0, 0), (48, 36, 0), (0, 36, 0)]
    idef = d.definition("Image1", lambda: (w.face(quad, mat=photo), 1)[1])
    d.root_item(lambda: w.image(idef, [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 40]))
    d.finish(path)


def legacy_scene(path, ver=17):
    d = S.Doc(ver)
    red = d.material("Red", rgba=(255, 0, 0, 255))
    d.material("Other", rgba=(1, 2, 3, 255))
    d.layer("Layer0")
    walls = d.layer("Walls")
    w = d.w
    sq = [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0)]
    d.root_item(lambda: w.face(sq, mat=red, layer=walls))
    d.finish(path, scene=("Scene A", [walls]))


def legacy_one_material(path):
    d = S.Doc(17)
    red = d.material("Red", rgba=(255, 0, 0, 255))
    d.layer("Layer0")
    w = d.w
    d.root_item(lambda: w.face([(0, 0, 0), (10, 0, 0), (10, 10, 0)], mat=red))
    d.finish(path)


def legacy_no_material(path):
    d = S.Doc(17)
    d.layer("Layer0")
    w = d.w
    d.root_item(lambda: w.face([(0, 0, 0), (10, 0, 0), (10, 10, 0)]))
    d.finish(path)


def legacy_conflict(path):
    """Materials and a tag that share a name with ``legacy-basic`` but not its
    recipe: imported after it, the registry numbers them ``Red (2)``."""
    d = S.Doc(17)
    red = d.material("Red", rgba=(1, 2, 3, 255))
    blue = d.material("Blue", rgba=(0, 0, 255, 255), opacity=0.5)   # the same recipe as legacy-basic's
    d.layer("Layer0")
    walls = d.layer("Walls")
    roof = d.layer("Roof2")
    w = d.w
    d.root_item(lambda: w.face([(0, 0, 0), (10, 0, 0), (10, 10, 0)], mat=red, layer=walls))
    d.root_item(lambda: w.face([(20, 0, 0), (30, 0, 0), (30, 10, 0)], mat=blue, layer=roof))
    d.finish(path)


def legacy_dup_material(path):
    d = S.Doc(17)
    d.material("Same", rgba=(255, 0, 0, 255))
    b = d.material("Same", rgba=(0, 0, 255, 255))
    c = d.material("", rgba=(9, 9, 9, 255))
    d.layer("Layer0")
    w = d.w
    d.root_item(lambda: w.face([(0, 0, 0), (10, 0, 0), (10, 10, 0)], mat=b))
    d.root_item(lambda: w.face([(0, 20, 0), (10, 20, 0), (10, 30, 0)], mat=c))
    d.finish(path)


# ---- 2021+ (VFF) scenarios ----------------------------------------------------------------

def base_vff(version="21.0.0") -> V.Vff:
    d = V.Vff(version)
    d.thumbnail = png(8, 8, (255, 255, 255, 255))
    d.styles.append(("Simple Style", -16777216, -9498257))
    return d


def vff_basic(path, wrap="none"):
    d = base_vff()
    d.wrap = wrap
    red = d.material("Red", (255, 0, 0))
    glass = d.material("Glass", (10, 20, 30), trans=0.7)
    wood = d.material("Wood", (120, 80, 40), texture=("wood.png", png(16, 16, (120, 80, 40, 255), True), 24.0, 12.0))
    d.layer("Layer0")
    walls = d.layer("Walls")
    roof = d.layer("Roof", hidden=True)
    for n, rgb in (("Walls", (200, 30, 30)), ("Roof", (10, 20, 30))):
        d.material("Layer_" + n, rgb, listed=False)
    sq = lambda x, y, z=0.0, s=10.0: [(x, y, z), (x + s, y, z), (x + s, y + s, z), (x, y + s, z)]  # noqa: E731
    g = d.geometry()
    g.face(sq(0, 0), mat=red)
    g.face(sq(10, 0), mat=glass, back=red, edge_flags=0x1E)
    g.face(sq(20, 0), hidden=True, edge_flags=0x07)
    g.face(sq(30, 0), layer=walls, edge_layer=walls, curve=4242)
    g.face(sq(40, 0), holes=[[(43, 3, 0), (47, 3, 0), (47, 7, 0), (43, 7, 0)]], mat=wood)
    g.face(sq(50, 0), mat=wood, uv_front=(0.5, 0, 0, 0, 0.25, 0, 3, 4, 1), uv_back=(1, 0, 0, 0, 1, 0, 0, 0, 1))
    g.face(sq(60, 0), back=wood)
    g.face([(70, 0, 0), (80, 0, 0), (75, 8, 0)], mat=wood)
    g.face([(90, 0, 0), (100, 0, 0), (104, 6, 0), (95, 12, 0), (88, 6, 0)], mat=wood)
    g.loose_edge((0, 50, 0), (10, 50, 5), flags=0x18)
    box = d.definition("Box", g)

    t = d.geometry()
    grid(lambda pts, m: t.face(pts, mat=m), 8, 10.0, 0.0, (red, None))
    tile = d.definition("Tile", t)

    f = d.geometry()
    f.face(sq(0, 0, 5), mat=glass)
    f.instance(box, xf(0, 20, 0, rot=90), name="inner box", layer=walls)
    f.instance(box, xf(0, 40, 0, scale=0.5), name="small box", mat=red)
    frame = d.definition("Frame", f)

    r = d.geometry()
    r.face(sq(0, 0, -5), mat=red)
    r.face(sq(20, 0, -5), layer=walls)
    r.face(sq(40, 0, -5), layer=walls, hidden=True)
    r.face([(0, 60, -5), (100, 60, -5), (100, 160, -5), (0, 160, -5)],
           holes=[[(20, 80, -5), (80, 80, -5), (80, 140, -5), (20, 140, -5)],
                  [(40, 100, -5), (60, 100, -5), (60, 120, -5), (40, 120, -5)]], mat=red)
    for i in range(9):
        kw = {}
        if i in (1, 4, 6):
            kw["layer"] = roof
        if i in (2, 5):
            kw["mat"] = red
        r.instance(tile, xf(i * 120, 200, 0, rot=10 * i), name=f"tile{i}", **kw)
    r.instance(frame, xf(0, 400, 0, rot=33, scale=2.0), name="frame once")
    r.instance(box, xf(200, 400, 0), name="box once", mat=glass)
    r.instance(box, xf(300, 400, 0), name="hidden box", hidden=True)
    r.instance(box, xf(400, 400, 0, sx=-1.0), name="mirrored box", layer=walls)
    d.root(r)
    d.write(path)


def vff_textures(path):
    d = base_vff("22.0.1")
    flat = png(8, 8, (200, 100, 50, 255))
    img = png(16, 16, (30, 60, 90, 255), True)
    tex = d.material("Tex", (30, 60, 90), texture=("tex.png", img, 36.0, 18.0))
    odd = d.material("Odd", (1, 2, 3), texture=("odd.png", png(4, 4), 0.0, 0.0))
    huge = d.material("Huge", (1, 2, 3), texture=("huge.png", png(4, 4), 1e231, 1e-231))
    # a colourized copy: its image lives in another material's folder; the
    # declared colour is the image's own, so Qt would not re-tint it
    d.material("Tex1", (200, 100, 50), texture=("missing.png", None, 36.0, 18.0), colorized=True, ctype=0,
               xml_extra='<mat:images><mat:image path="./materials/Flat/flat.png"/></mat:images>')
    d.material("Flat", (200, 100, 50), texture=("flat.png", flat, 10.0, 10.0))
    d.material("NoImage", (5, 5, 5), texture=("gone.png", None, 5.0, 5.0))
    tex1 = d.material("Same", (9, 9, 9))
    d.material("Same", (8, 8, 8), folder="Same_2")
    d.layer("Layer0")
    g = d.geometry()
    w = lambda x, y, z=0.0: [(x, y, z), (x + 40, y, z), (x + 40, y, z + 30), (x, y, z + 30)]  # noqa: E731
    g.face(w(0, 0), mat=tex)
    g.face([(0, 20, 0), (40, 20, 0), (40, 60, 0), (0, 60, 0)], mat=tex, uv_front=(0.5, 0, 0, 0, 0.25, 0, 12, 6, 1))
    g.face(w(60, 0), mat=odd)
    g.face(w(120, 0), mat=huge)
    g.face(w(180, 0), mat=tex1)
    # normals whose length C pow(x, 0.5) does not give as sqrt(x) to the last
    # bit (one in two thousand): Python's `** 0.5` and Rust's sqrt part here
    g.face(w(240, 0), mat=tex, normal=(0.172, -0.653, -0.479))
    g.face(w(300, 0), mat=tex, normal=(-0.829, 0.76, 0.996))
    g.face(w(360, 0), mat=tex, normal=(0.616, -0.669, 0.992))
    g.face([(0, 100, 0), (40, 100, 0), (40, 140, 10), (0, 140, 10)], mat=tex,
           uv_front=(1, 0, 0, 0, 1, 0, 0, 0, 1), uv_back=(0.1, 0.02, 0, -0.02, 0.1, 0, 5, 5, 1))
    g.face([(60, 100, 0), (100, 100, 0), (100, 140, 0), (60, 140, 0)], back=tex,
           uv_back=(0.3, 0, 0, 0, 0.3, 0, 1, 2, 1))
    d.root(g)
    d.write(path)


def vff_tint(path, ctype=1, color=(20, 20, 220, 255), declared=(255, 0, 0), pattern=False, name="Tinted"):
    """A colourized copy whose colour is not the image's: Qt re-tints it."""
    d = base_vff()
    d.material(name, declared, texture=("t.png", png(8, 8, color, pattern), 10.0, 10.0),
               colorized=True, ctype=ctype)
    d.layer("Layer0")
    g = d.geometry()
    g.face([(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0)], mat=d.materials[0]["id"])
    d.root(g)
    d.write(path)


def vff_tint_two(path):
    """Two materials re-tinted from one shared image, different modes, one of
    them with the image's alpha varying, and a fully transparent image."""
    d = base_vff()
    img = png(8, 8, (200, 90, 40, 180), (40, 200, 90, 180))
    d.material("TintA", (30, 160, 60), texture=("shared.png", img, 10.0, 10.0), colorized=True, ctype=0, folder="TintA")
    d.material("TintB", (30, 160, 60), texture=("shared.png", img, 10.0, 10.0), colorized=True, ctype=1, folder="TintB")
    d.material("Clear", (10, 200, 10), texture=("clear.png", png(4, 4, (50, 60, 70, 0)), 5.0, 5.0), colorized=True, ctype=0)
    d.layer("Layer0")
    g = d.geometry()
    for i, m in enumerate(d.materials):
        g.face([(i * 20, 0, 0), (i * 20 + 10, 0, 0), (i * 20 + 10, 10, 0), (i * 20, 10, 0)], mat=m["id"])
    d.root(g)
    d.write(path)


def legacy_tint(path, ver=17):
    """A classic textured material is read as colourized when its stored
    average colour is opaque: here the stored colour is not the image's."""
    d = S.Doc(ver)
    tinted = d.material("Tinted", texture=(png(16, 16, (20, 20, 220, 255), (220, 180, 20, 255)), "t.png", 20.0, 10.0, False),
                        rgba=(255, 0, 0, 255))
    d.material("Other", rgba=(1, 2, 3, 255))
    d.layer("Layer0")
    w = d.w
    d.root_item(lambda: w.face([(0, 0, 0), (40, 0, 0), (40, 0, 30), (0, 0, 30)], mat=tinted))
    d.finish(path)


def terrain_geom(d, mat, tilt=True, matrix=(1, 0, 0, 0, 1, 0, 0, 0, 1)):
    g = d.geometry()
    n = 4
    heights = [[((i * 7 + j * 13) % 5) * 3.0 for j in range(n + 1)] for i in range(n + 1)]
    for i in range(n):
        for j in range(n):
            pts = [(i * 20, j * 20, heights[i][j]), ((i + 1) * 20, j * 20, heights[i + 1][j]),
                   ((i + 1) * 20, (j + 1) * 20, heights[i + 1][j + 1]), (i * 20, (j + 1) * 20, heights[i][j + 1])]
            if tilt:
                pts = [pts[0], pts[1], pts[2]]
            g.face(pts, mat=mat, uv_front=matrix)
    return g


def vff_drape(path):
    d = base_vff("24.0.1")
    tex = d.material("Aerial", (90, 120, 60), texture=("aerial.png", png(16, 16, (90, 120, 60, 255), True), 100.0, 100.0))
    d.layer("Layer0")
    d.root(terrain_geom(d, tex, tilt=True, matrix=(0.02, 0, 0, 0, 0.02, 0, 0.1, 0.2, 1)))
    d.write(path)


def vff_bulk(path):
    """Enough corners (>= 1024) for the vectorised build, plus a face whose
    second hole lies inside its first."""
    d = base_vff("23.0.0")
    red = d.material("Red", (255, 0, 0))
    wood = d.material("Wood", (120, 80, 40), texture=("wood.png", png(16, 16, (120, 80, 40, 255), True), 24.0, 12.0))
    d.layer("Layer0")
    g = d.geometry()
    n = 20
    for i in range(n):
        for j in range(n):
            pts = [(i * 10, j * 10, 0), (i * 10 + 10, j * 10, 0), (i * 10 + 10, j * 10 + 10, 0), (i * 10, j * 10 + 10, 0)]
            g.face(pts, mat=red if (i + j) % 3 else wood, edge_flags=0x1E if i % 5 == 0 else 0x06,
                   hidden=(i == j == 7))
    big = [(0, 300, 0), (100, 300, 0), (100, 400, 0), (0, 400, 0)]
    g.face(big, holes=[[(20, 320, 0), (80, 320, 0), (80, 380, 0), (20, 380, 0)],
                       [(40, 340, 0), (60, 340, 0), (60, 360, 0), (40, 360, 0)]], mat=red)
    d.root(g)
    d.write(path)


def vff_flat_textured(path):
    d = base_vff("24.0.1")
    tex = d.material("Floor", (90, 120, 60), texture=("floor.png", png(16, 16, (90, 120, 60, 255), True), 100.0, 100.0))
    d.layer("Layer0")
    g = d.geometry()
    for i in range(10):
        g.face([(i * 20, 0, 0), (i * 20 + 20, 0, 0), (i * 20 + 20, 20, 0), (i * 20, 20, 0)], mat=tex,
               uv_front=(0.01, 0, 0, 0, 0.01, 0, 0.5, 0.5, 1))
    d.root(g)
    d.write(path)


def vff_pages(path):
    d = base_vff("23.1.0")
    red = d.material("Red", (255, 0, 0))
    d.layer("Layer0")
    walls = d.layer("Walls")
    roof = d.layer("Roof", hidden=True)
    d.material("Layer_Walls", (255, 0, 0), listed=False)
    d.material("Layer_Roof", (0, 255, 0), listed=False)
    sq = [(0, 0, 0), (100, 0, 0), (100, 100, 0), (0, 100, 0)]
    g = d.geometry()
    g.face(sq, mat=red)
    box = d.definition("Box", g)
    r = d.geometry()
    inst = r.instance(box, xf(50, 50, 0, rot=90), name="b")
    root_vertex = r.vertex((300, 0, 0))
    r.vertex((310, 0, 0))
    r.loose_edge((300, 0, 0), (310, 0, 0))
    d.root(r)
    # the first vertex of the definition, seen through the instance
    first_vertex = g.verts[(0, 0, 0)]
    d.tail.append(V.page_record([
        dict(name="Plan", eye=(100, 100, 500), target=(100, 100, 0), up=(0, 1, 0), fov=35.0, perspective=False, height=240.0),
        dict(name="", eye=(300, -400, 200), target=(50, 50, 0), up=(0, 0, 1), fov=45.0, hidden=[walls, roof]),
        dict(name="Persp", eye=(200, 200, 200), target=(0, 0, 0), up=(0, 0, 1)),
    ]))
    d.tail.append(V.dimension_record((0, 0, 0), (100, 0, 0), offset=12.0))
    d.tail.append(V.dimension_record((0, 0, 0), (0, 80, 0), offset=-5.0, normal=(0, 0, 1)))
    d.tail.append(V.dimension_record((0, 0, 0), (10, 0, 0), anchored=((first_vertex, inst), (root_vertex, None))))
    d.tail.append(V.dimension_record((0, 0, 0), (10, 0, 0), anchored=((999999, None), None)))
    d.write(path)


def vff_attrs(path):
    d = base_vff("25.0.1")
    red = d.material("Red", (255, 0, 0))
    d.layer("Layer0")
    sq = [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0)]
    g = d.geometry()
    g.face(sq, mat=red)
    part = d.definition("Part", g)
    r = d.geometry()
    r.instance(part, xf(0, 0, 0), name="with attrs",
               attrs=[("dynamic_attributes", [("name", "Door"), ("w", 90.0), ("n", 3), ("pt", (1.0, 2.0, 3.0)),
                                              ("list", [1, "two", 3.5]), ("nothing", None)]),
                      ("other", [("k", "v")])])
    r.instance(part, xf(20, 0, 0), name="plain")
    d.root(r)
    d.write(path)


def vff_empty(path):
    d = base_vff()
    d.thumbnail = png(8, 8, (255, 255, 255, 255))
    d.layer("Layer0")
    d.write(path)


def vff_empty_busy_thumbnail(path):
    d = base_vff()
    d.thumbnail = png(8, 8, (255, 255, 255, 255), pattern=True)
    d.layer("Layer0")
    d.write(path)


def vff_faceme(path):
    d = base_vff()
    red = d.material("Red", (255, 0, 0))
    d.layer("Layer0")
    g = d.geometry()
    g.face([(0, 0, 0), (10, 0, 0), (10, 0, 20), (0, 0, 20)], mat=red)
    person = d.definition("Person", g, faces_camera=True, shadows=True)
    r = d.geometry()
    r.instance(person, xf(0, 0, 0), name="susan")
    d.root(r)
    d.write(path)


def vff_image(path):
    d = base_vff()
    red = d.material("Red", (255, 0, 0))
    d.layer("Layer0")
    g = d.geometry()
    g.face([(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0)], mat=red)
    img = d.definition("Image#1", g, kind=2)
    r = d.geometry()
    r.instance(img, xf(0, 0, 0), name="photo")
    d.root(r)
    d.write(path)


def vff_odd(path):
    """Duplicate ids, dangling references, a definition without a name."""
    d = base_vff()
    red = d.material("Red", (255, 0, 0))
    d.material("Orphan", (1, 1, 1), folder="OrphanFolder")
    d.layer("Layer0")
    sq = [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0)]
    g1 = d.geometry()
    g1.face(sq, mat=red)
    g2 = d.geometry()
    g2.face([(0, 0, 5), (10, 0, 5), (10, 10, 5)], mat=red)
    d1 = d.definition("First", g1, def_id=5000)
    d.definition("Second", g2, def_id=5000)  # the same id: the dict keeps the last
    g3 = d.geometry()
    g3.face(sq, mat=7777)  # a material that does not exist
    d.definition("", g3)
    r = d.geometry()
    r.instance(5000, xf(0, 0, 0), name="to-dup")
    r.instance(123456, xf(0, 0, 0), name="dangling")
    r.instance(d1, xf(30, 0, 0), name="again")
    d.root(r)
    d.write(path)


def vff_no_matrix(path):
    """An instance without a transformation: the adapter cannot place it."""
    d = base_vff()
    d.layer("Layer0")
    sq = [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0)]
    g = d.geometry()
    g.face(sq)
    part = d.definition("Part", g)
    r = d.geometry()
    r.instance(part, xf(0, 0, 0), name="x")
    r.insts[0] = r.insts[0].replace(V.rec("6619", S.f64s(*xf())), b"")  # strip the matrix record
    # the record header's length is now wrong: rebuild the whole 6419 instead
    r.insts = [V.rec("6419", V.rec("D007", V.entity_dc05(r.doc.new_id()) + V.rec("D307", b"\x06"))
                     + V.rec("6519", b"x") + V.rec("6719", V.vid(part)))]
    d.root(r)
    d.write(path)


# ---- broken files -----------------------------------------------------------------------

def write_errors(out: Path) -> list:
    cases = []

    def put(name, data):
        p = out / name
        p.write_bytes(data)
        cases.append({"file": name, "kind": "error"})

    put("err-not-skp.skp", b"this is not a sketchup file at all")
    put("err-empty.skp", b"")
    put("err-magic-only.skp", b"\xff\xfe\xff\x0e")
    put("err-header-no-zip.skp", b"\xff\xfe\xff\x0e" + "SketchUp Model".encode("utf-16-le") + S.STR + bytes([8]) + "{21.0.0}".encode("utf-16-le") + b"VFF" + bytes(64))
    d = base_vff()
    d.layer("Layer0")
    d.write(out / "err-no-model-dat.skp", skip_model=True)
    cases.append({"file": "err-no-model-dat.skp", "kind": "error"})
    d = base_vff()
    d.layer("Layer0")
    g = d.geometry()
    g.face([(0, 0, 0), (10, 0, 0), (10, 10, 0)])
    d.root(g)
    d.write(out / "ok-for-truncation.skp")
    data = (out / "ok-for-truncation.skp").read_bytes()
    (out / "ok-for-truncation.skp").unlink()
    put("err-truncated-zip.skp", data[:len(data) // 2])
    flipped = bytearray(data)
    # damage the middle of the compressed model.dat: the CRC (or inflate) fails
    for i in range(len(flipped) // 2, len(flipped) // 2 + 6):
        flipped[i] ^= 0xFF
    put("err-corrupt-zip.skp", bytes(flipped))
    # a legacy header whose archive does not start anywhere
    put("err-legacy-garbage.skp", b"\xff\xfe\xff\x0e" + "SketchUp Model".encode("utf-16-le") + S.STR + bytes([8])
        + "{17.0.0}".encode("utf-16-le") + b"CVersionMap" + bytes(300))
    # the page record without a camera item is a TypeError in openskp
    d = base_vff()
    d.layer("Layer0")
    g = d.geometry()
    g.face([(0, 0, 0), (10, 0, 0), (10, 10, 0)])
    d.root(g)
    d.tail.append(V.rec("0702", V.rec_le(0x6D60, V.rec_le(0x6D61, V.rec_le(0x7148, V.rec_le(0x7150, b"\x01\x05"))))))
    d.write(out / "err-page-no-camera.skp")
    cases.append({"file": "err-page-no-camera.skp", "kind": "error"})
    return cases


# ---- the harness ------------------------------------------------------------------------

LEGACY = [
    ("legacy-basic", legacy_basic), ("legacy-components", legacy_components),
    ("legacy-textures", legacy_textures), ("legacy-attrs", legacy_attrs),
    ("legacy-faceme", legacy_faceme), ("legacy-image", legacy_image), ("legacy-scene", legacy_scene),
    ("legacy-one-material", legacy_one_material), ("legacy-no-material", legacy_no_material),
    ("legacy-dup-material", legacy_dup_material), ("legacy-conflict", legacy_conflict),
    ("legacy-v16", lambda p: legacy_basic(p, ver=16)), ("legacy-v13", lambda p: legacy_components(p, ver=13)),
    ("legacy-v18", lambda p: legacy_textures(p, ver=18)), ("legacy-tint", legacy_tint),
]
VFF = [
    ("vff-basic", vff_basic), ("vff-textures", vff_textures), ("vff-tint", vff_tint), ("vff-drape", vff_drape),
    ("vff-tint-shift", lambda p: vff_tint(p, ctype=0, pattern=(40, 200, 90, 255), name="Shifted")), ("vff-tint-two", vff_tint_two),
    ("vff-flat-textured", vff_flat_textured), ("vff-bulk", vff_bulk), ("vff-pages", vff_pages), ("vff-attrs", vff_attrs),
    ("vff-empty", vff_empty), ("vff-empty-busy", vff_empty_busy_thumbnail), ("vff-faceme", vff_faceme),
    ("vff-image", vff_image), ("vff-odd", vff_odd), ("vff-no-matrix", vff_no_matrix),
    ("vff-wrapped-f901", lambda p: vff_basic(p, wrap="f901")), ("vff-wrapped-f401", lambda p: vff_basic(p, wrap="f401")),
]


def load_model(path):
    openskp_compat.apply()
    import openskp
    model = openskp.SkpFile.open(str(path)).parse()
    openskp_compat.resolve_face_layers(model)
    return model


def view_dict(v):
    return v.to_dict()


def f32hex(x):
    return struct.pack(">f", x).hex()


def vec(v):
    return [f32hex(v.x()), f32hex(v.y()), f32hex(v.z())]


def extras_of(scene):
    return {
        "saved_views": [view_dict(v) for v in scene.saved_views],
        "dimensions": [[vec(d.a), vec(d.b), vec(d.offset)] for d in scene.dimensions],
        "text_labels": [[vec(t.anchor), vec(t.offset), t.text] for t in scene.text_labels],
        "back_face_color": None if getattr(scene, "back_face_color", None) is None
        else [skp_dump.fb(c) for c in scene.back_face_color],
        "groups": len(scene.groups),
    }


def run_import(path):
    """Python's import of ``path``: the outcome and, when it worked, the
    signature and extras."""
    tint_calls = []
    real_colorize = skp_openskp._colorize_image

    def spy(data, *a, **k):
        tint_calls.append(bytes(data[:3]))
        return real_colorize(data, *a, **k)

    skp_openskp._colorize_image = spy
    try:
        try:
            payload = skp_format.parse_skp(path)
        except skp_format.NeedsConverter:
            return {"outcome": "unreadable"}, None
    finally:
        skp_openskp._colorize_image = real_colorize
    outcome = "ok"
    scene = Scene()
    skp_format.apply_payload(scene, payload)
    pixelize_textures()
    info = {"outcome": outcome, "empty": bool(payload.get("empty")),
            "tinted": len(tint_calls),
            # Qt decodes a JPEG differently from the Rust decoder: a texture
            # tinted from one is not compared pixel by pixel
            "tint_from_jpeg": any(c == b"\xff\xd8\xff" for c in tint_calls)}
    return info, scene


def pixelize_textures() -> None:
    """Every PNG the import wrote to the texture cache becomes the text
    ``px:<w>x<h>:<sha256 of its RGBA pixels>``: a re-tinted texture is written
    by Qt's PNG encoder here and by another in Rust, so the signature (which
    hashes the file) can only be compared through the pixels."""
    from PySide6.QtGui import QImage
    for p in CACHE.rglob("*"):
        if not p.is_file():
            continue
        data = p.read_bytes()
        if not data.startswith(b"\x89PNG"):
            continue
        img = QImage.fromData(data)
        if img.isNull():
            continue
        img = img.convertToFormat(QImage.Format.Format_RGBA8888)
        raw = bytes(img.constBits())[: img.width() * img.height() * 4]
        p.write_text(f"px:{img.width()}x{img.height()}:{hashlib.sha256(raw).hexdigest()}", encoding="ascii")


def process(name, path, out: Path, kind, note=""):
    entry = {"file": path.name, "name": name, "kind": kind, "note": note}
    try:
        model = load_model(path)
    except Exception as exc:  # noqa: BLE001 — the message is the datum
        entry["parse_error"] = f"{type(exc).__name__}: {exc}"
        entry["import"] = {"outcome": "unreadable"}
        return entry
    write_json(out / f"{name}.model.json", skp_dump.dump_model(model))
    info, scene = run_import(path)
    entry["import"] = info
    if scene is not None and info["outcome"] == "ok":
        write_json(out / f"{name}.sig.json", scene_signature(scene))
        entry["extras"] = extras_of(scene)
    return entry


def main() -> None:
    app = QGuiApplication.instance() or QGuiApplication(sys.argv)  # noqa: F841
    args = sys.argv[1:]
    real_list = None
    if "--real" in args:
        real_list = Path(args[args.index("--real") + 1])
    if real_list is None:
        if OUT.exists():
            for p in OUT.iterdir():
                if p.is_file() and p.name != "real_cases.json":
                    p.unlink()
        OUT.mkdir(parents=True, exist_ok=True)
        cases = []
        for name, fn in LEGACY:
            path = OUT / f"{name}.skp"
            fn(path)
            cases.append(process(name, path, OUT, "legacy"))
        for name, fn in VFF:
            path = OUT / f"{name}.skp"
            fn(path)
            cases.append(process(name, path, OUT, "vff"))
        for c in write_errors(OUT):
            path = OUT / c["file"]
            entry = process(path.stem, path, OUT, "error")
            cases.append(entry)
        # Files imported one after the other into one scene: the registry, the
        # tags and the saved views merge.
        for name, files in (("seq-conflict", ["legacy-basic", "legacy-conflict"]),
                            ("seq-twice", ["vff-basic", "vff-basic"]),
                            ("seq-mixed", ["vff-pages", "legacy-scene", "vff-textures"])):
            shutil.rmtree(CACHE / "skp", ignore_errors=True)
            scene = Scene()
            for f in files:
                skp_format.load_skp(scene, OUT / f"{f}.skp")
            pixelize_textures()
            write_json(OUT / f"{name}.sig.json", scene_signature(scene))
            cases.append({"file": None, "name": name, "kind": "sequence", "files": files,
                          "import": {"outcome": "ok"}, "extras": extras_of(scene)})
        write_json(OUT / "cases.json", cases)
        outcomes = {}
        for c in cases:
            outcomes[c["import"]["outcome"]] = outcomes.get(c["import"]["outcome"], 0) + 1
        print(len(cases), "files", outcomes)
        for c in cases:
            print(f"  {c['name']:<24} {c['kind']:<7} {c['import']['outcome']:<11} {c.get('parse_error', '')[:70]}")
    else:
        cases = []
        for line in real_list.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            tag = hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:10]
            entry = {"path": str(p), "name": f"real-{tag}", "size": p.stat().st_size}
            try:
                model = load_model(p)
                entry["model_sha"] = hashlib.sha256(dumps(skp_dump.dump_model(model)).encode()).hexdigest()
                info, scene = run_import(p)
                entry["import"] = info
                if scene is not None and info["outcome"] == "ok":
                    sig = scene_signature(scene)
                    entry["sig_sha"] = hashlib.sha256(dumps(sig).encode()).hexdigest()
                    entry["extras_sha"] = hashlib.sha256(dumps(extras_of(scene)).encode()).hexdigest()
            except Exception as exc:  # noqa: BLE001
                entry["parse_error"] = f"{type(exc).__name__}: {exc}"
                entry["import"] = {"outcome": "unreadable"}
            cases.append(entry)
            print(entry["name"], entry.get("import", {}).get("outcome"), entry.get("parse_error", "")[:60])
        write_json(OUT / "real_cases.json", cases)
    shutil.rmtree(CACHE, ignore_errors=True)


if __name__ == "__main__":
    main()
