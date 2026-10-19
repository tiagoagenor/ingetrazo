# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixtures that hold ``ingetrazo-model``'s annotations to the Python
classes, byte for byte (docs/rust-rewrite-plan.md, M2.5): dimensions,
leader texts, guides, section planes, saved views and image planes.

The REAL classes (core/dimension.py, textlabel.py, guide.py, section.py,
saved_views.py, image_plane.py, core/units.py) run on random inputs, edge
cases and the legacy / partial dicts their ``from_dict`` accepts, and what
they produce is written down as exact bits (float32 vectors as 8 hex digits,
float64 as 16) and as the very text ``json.dumps`` gave. The .igz side is
the real thing too: dimensions come out of ``formats.igz.save_scene`` and
whole annotation payloads go through ``load_into`` and ``save_scene`` again.
Committed, so cargo test needs no Python; rerun only to widen them:

    python scripts/golden/make_annotation_fixtures.py

Deleted with the Python app at M10.
"""
from __future__ import annotations

import json
import math
import random
import struct
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from PySide6 import __version__ as pyside_version
from PySide6.QtCore import qVersion
from PySide6.QtGui import QVector3D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "rust" / "crates" / "ingetrazo-model" / "tests" / "data" / "annotations"

from core import section as section_mod                      # noqa: E402
from core import units                                        # noqa: E402
from core.dimension import (Dimension, _ALIGNED_CONE_DEG,   # noqa: E402
                            _LINEAR_MIN_DEG)
from core.guide import Guide                                  # noqa: E402
from core.image_plane import IMAGE_LAYER, ImagePlane          # noqa: E402
from core.saved_views import SavedView, from_lookat           # noqa: E402
from core.scene import Scene                                  # noqa: E402
from core.section import (SectionPlane, effective_fill,       # noqa: E402
                          next_symbol)
from core.textlabel import TextLabel                          # noqa: E402
from formats import igz                                       # noqa: E402


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def h32(x: float) -> str:
    return struct.pack(">f", x).hex()


def h64(x: float) -> str:
    return struct.pack(">d", x).hex()


def vb(v) -> list:
    return [h32(v.x()), h32(v.y()), h32(v.z())]


def vf(bits) -> QVector3D:
    return QVector3D(*[struct.unpack(">f", bytes.fromhex(b))[0] for b in bits])


def dumps(x) -> str:
    return json.dumps(x)


STRINGS = ["", "A", "L = <>", "<>", "2x<> (cafe\u0301)", "emoji \U0001F600 <>",
           'quote"back\\slash', "line\nbreak", "tab\t<>", "\u00e9\u4e2d <> <>",
           "Muro 3.5", "<> + <>", "ASCII only"]
LAYERS = [None, "", "Anotaciones", "Cotas", "Capa \u00f1", "Images", "Other"]


class Gen:
    def __init__(self, seed: int) -> None:
        self.r = random.Random(seed)

    def num(self, lo=-50.0, hi=50.0) -> float:
        r = self.r.random()
        if r < 0.12:
            return f32(self.r.choice([0.0, 1.0, -1.0, 2.0, 0.5, -0.0, 3.0]))
        if r < 0.35:
            return f32(round(self.r.uniform(lo, hi), self.r.randint(0, 3)))
        return f32(self.r.uniform(lo, hi))

    def vec(self, lo=-50.0, hi=50.0) -> QVector3D:
        return QVector3D(self.num(lo, hi), self.num(lo, hi), self.num(lo, hi))

    def axisvec(self) -> QVector3D:
        v = [0.0, 0.0, 0.0]
        v[self.r.randrange(3)] = self.r.choice([1.0, -1.0, 2.5, -4.0])
        return QVector3D(*v)

    def pick(self, seq):
        return self.r.choice(seq)

    def f64(self, lo=-10.0, hi=10.0) -> float:
        r = self.r.random()
        if r < 0.15:
            return self.r.choice([0.0, 1.0, -1.0, 0.5, 20.0, 45.0, 0.25])
        if r < 0.4:
            return round(self.r.uniform(lo, hi), self.r.randint(0, 4))
        return self.r.uniform(lo, hi)


def write(name: str, meta: dict, rows: list) -> None:
    with open(OUT / name, "w", encoding="ascii", newline="\n") as fh:
        fh.write(json.dumps({"meta": meta}) + "\n")
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    print(f"  {name}: {len(rows)} rows, {(OUT / name).stat().st_size} bytes")


# ---- the .igz round trip, with the real save/load ---------------------------
_TMP = Path(tempfile.mkdtemp(prefix="annot_fixtures_"))
SECTION_KEYS = ("saved_views", "dimensions", "section_planes", "section_view",
                "text_labels", "guides", "image_planes")


def saved_sections(scene) -> list:
    """The annotation sections of what ``save_scene`` writes for ``scene``,
    in file order, as ``[key, json.dumps(value)]`` pairs."""
    p = _TMP / "doc.igz"
    igz.save_scene(scene, p)
    payload = json.loads(p.read_text(encoding="utf-8"))["scene"]
    return [[k, dumps(payload[k])] for k in payload if k in SECTION_KEYS]


def roundtrip(sections: dict) -> dict:
    """``sections`` (raw payload blocks) put in a document, loaded with
    ``load_into`` and saved again: what the Python app makes of them."""
    scene = Scene()
    p = _TMP / "base.igz"
    igz.save_scene(scene, p)
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["scene"].update(sections)
    src = _TMP / "raw.igz"
    src.write_text(json.dumps(doc), encoding="utf-8")
    section_mod._last_uid = 0
    try:
        igz.load_into(scene, src)
    except Exception as e:  # noqa: BLE001
        return {"err": type(e).__name__}
    try:
        out = saved_sections(scene)
    except Exception as e:  # noqa: BLE001
        return {"err": "save:" + type(e).__name__}
    return {"out": out, "repairs": int(scene.load_repairs),
            "planes_shown": bool(scene.show_section_planes),
            "cuts_shown": bool(scene.show_section_cuts),
            "next_uid": section_mod._last_uid}


def try_(fn):
    try:
        return {"out": fn()}
    except Exception as e:  # noqa: BLE001
        return {"err": type(e).__name__}


# ---- units -------------------------------------------------------------------
UNIT_CODES = ["m", "cm", "mm", "in", "ft", "ft-in", "in-frac", "ft-in-frac",
              "km", "yd", "mi", "us-ft", "mm2", "", "xyz", None]


def length_values(g: Gen) -> float:
    r = g.r.random()
    if r < 0.03:
        return g.pick([float("nan"), float("inf"), float("-inf")])
    if r < 0.10:
        return g.pick([0.0, -0.0, 1.0, -1.0, 0.0254, 0.3048, 0.0127, 12 * 0.0254,
                       1.2345, 0.005, -0.005, 0.995, 1.005, 2.675, 0.0005])
    if r < 0.25:
        return round(g.r.uniform(-30, 30), g.r.randint(0, 5))
    if r < 0.40:
        return g.r.randint(-400, 400) * 0.0254 / g.pick([1, 2, 4, 8, 16, 32, 64, 3])
    if r < 0.50:
        return g.r.randint(-60, 60) * 0.3048 + g.r.choice([0, 1e-9, -1e-9, 0.0127])
    if r < 0.60:
        return g.r.uniform(-1, 1) * 10 ** g.r.randint(-9, 12)
    return g.r.uniform(-200, 200)


def units_fixtures() -> list:
    g = Gen(7001)
    rows = []
    for _ in range(2200):
        m = length_values(g)
        u = g.pick(UNIT_CODES)
        d = g.pick([0, 1, 2, 3, 4, 5, 6, 2, 2, 3, -1, 7, 9])
        rows.append({"m": h64(m), "u": u, "d": d,
                     "out": units.format_length(m, u, d)})
    # fmt_len: the settings are raw (validated by model_units_of).
    settings = [None, {"length": "m", "precision": 2}, {"length": "cm", "precision": 1},
                {"length": "mm", "precision": 0, "decimal": ","},
                {"length": "ft-in-frac", "precision": 2, "decimal": ","},
                {"length": "in-frac", "precision": 4}, {"length": "ft", "precision": 3},
                {"length": "km", "precision": 3}, {"length": "m", "precision": 99},
                {"length": "m", "precision": -4}, {"length": "m", "precision": "3"},
                {"length": "ft-in", "precision": 2.9}, {"length": "m", "precision": None},
                {"length": "m", "precision": 2, "decimal": "."},
                {"length": "m", "precision": 2, "decimal": ";"},
                {"length": "in", "precision": True}, {"precision": 3}]
    for _ in range(300):
        m = length_values(g)
        s = g.pick(settings)
        units.bind_scene(SimpleNamespace(units=s) if s is not None else None)
        rows.append({"fmt": h64(m), "settings": s, "out": units.fmt_len(m)})
    units.bind_scene(None)
    return rows


# ---- dimensions ----------------------------------------------------------------
UNIT_SETTINGS = [None, {"length": "m", "precision": 2}, {"length": "cm", "precision": 1},
                 {"length": "mm", "precision": 0}, {"length": "ft-in", "precision": 2},
                 {"length": "ft-in-frac", "precision": 3, "decimal": ","},
                 {"length": "in-frac", "precision": 2}, {"length": "m", "precision": 4,
                                                          "decimal": ","}]
AXES = [None, None, "x", "y", "z", "x", "y", "z", "w", "", "X"]


def dimension_geometry(g: Gen):
    kind = g.r.random()
    a = g.vec()
    if kind < 0.08:
        b = QVector3D(a)
    elif kind < 0.30:
        b = a + g.axisvec() * g.num(0.2, 12.0)
    elif kind < 0.40:
        b = a + g.axisvec() * g.num(0.2, 12.0) + QVector3D(
            f32(g.r.uniform(-1e-3, 1e-3)), f32(g.r.uniform(-1e-3, 1e-3)),
            f32(g.r.uniform(-1e-3, 1e-3)))
    elif kind < 0.55:
        b = a + QVector3D(g.num(-6, 6), g.num(-6, 6), 0.0)
    else:
        b = g.vec()
    r = g.r.random()
    if r < 0.15:
        off = QVector3D(0, 0, 0)
    elif r < 0.30:
        off = (b - a) * g.num(-2, 2)
    else:
        off = g.vec(-5, 5)
    return a, b, off


def cursor_for(g: Gen, a, b) -> QVector3D:
    r = g.r.random()
    d = b - a
    if r < 0.3:
        return g.vec()
    if r < 0.8:
        mid = (a + b) * 0.5
        # beyond both endpoints along an axis direction, or off to the side
        ax = g.axisvec()
        return mid + ax * g.num(0.0, 12.0) + d * g.num(-0.4, 0.4)
    if r < 0.9:
        return mid_perp(g, a, b)
    return a + d * g.num(-1, 2) + g.axisvec() * g.num(-3, 3)


def mid_perp(g, a, b):
    d = b - a
    t = QVector3D.crossProduct(d, g.axisvec())
    return (a + b) * 0.5 + t * g.num(-2, 2)


def dim_to_igz_input(d) -> dict:
    return {"a": vb(d.a), "b": vb(d.b), "off": vb(d.offset), "axis": d.axis,
            "layer": d.layer, "text": d.text}


def dimension_fixtures():
    g = Gen(7002)
    consts = {"tan_cone": h64(math.tan(math.radians(_ALIGNED_CONE_DEG))),
              "sin_min": h64(math.sin(math.radians(_LINEAR_MIN_DEG))),
              "cos_min": h64(math.cos(math.radians(_LINEAR_MIN_DEG)))}
    rows, dims = [], []
    placed_axes: dict = {}
    for _ in range(400):
        a, b, off = dimension_geometry(g)
        axis = g.pick(AXES)
        layer = g.pick(LAYERS)
        text = g.pick([None, None] + STRINGS)
        d = Dimension(a, b, off, layer=layer, text=text, axis=axis)
        cur = cursor_for(g, a, b)
        st = g.pick(UNIT_SETTINGS)
        units.bind_scene(SimpleNamespace(units=st) if st is not None else None)
        lab = d.label()
        pl_off, pl_axis = Dimension.placement_for_cursor(a, b, cur)
        placed_axes[pl_axis] = placed_axes.get(pl_axis, 0) + 1
        lp = d.line_points()
        row = {
            "a": vb(a), "b": vb(b), "off": vb(off), "axis": axis, "layer": layer,
            "text": text, "settings": st, "cursor": vb(cur),
            "o_axis": d.axis, "value": h64(d.value()), "label": lab,
            "perp": vb(d.perpendicular_offset()), "line": [vb(lp[0]), vb(lp[1])],
            "mid": vb(d.midpoint()),
            "disp": [d.display_text(lab), d.display_text("12.5 m"), d.display_text("")],
            "ofc": vb(Dimension.offset_for_cursor(a, b, cur)),
            "pfc": [vb(pl_off), pl_axis],
        }
        rows.append(row)
        dims.append(d)
    units.bind_scene(None)
    print("  placement_for_cursor axes:", placed_axes)
    # The real writer: every dimension in one document.
    scene = Scene()
    scene.dimensions.extend(dims)
    p = _TMP / "dims.igz"
    igz.save_scene(scene, p)
    entries = json.loads(p.read_text(encoding="utf-8"))["scene"]["dimensions"]
    assert len(entries) == len(rows)
    for row, e in zip(rows, entries):
        row["igz"] = dumps(e)
    # from_dict side: the reader's expression, on valid, legacy and broken dicts.
    raws = [dict(e) for e in entries[:100]]
    variants = []
    for e in entries[100:150]:
        v = dict(e)
        for k in g.r.sample(["axis", "layer", "text"], g.r.randint(0, 3)):
            v.pop(k, None)
        variants.append(v)
    for e in entries[150:190]:
        v = dict(e)
        v["axis"] = g.pick(["", "w", None, "x", "Z", 5, 0, True])
        v["layer"] = g.pick(["", None, "Capa", 0, False])
        v["text"] = g.pick(["", None, "t <>", 0, False])
        variants.append(v)
    for e in entries[190:220]:
        v = dict(e)
        for k in ("a", "b", "offset"):
            v[k] = [int(x) if g.r.random() < 0.5 and float(x).is_integer() else x
                    for x in v[k]]
        variants.append(v)
    variants += [{"a": [1, 2, 3], "b": [4, 5, 6], "offset": [0, 0, 1]},
                 {"a": [1, 2, 3], "b": [4, 5, 6], "offset": [0, 0, 1], "extra": 5},
                 {"a": [1, 2, 3], "b": [4, 5, 6]},                      # KeyError
                 {"a": [1, 2], "b": [4, 5, 6], "offset": [0, 0, 1]},     # TypeError
                 {"a": [1, 2, 3, 4], "b": [4, 5, 6], "offset": [0, 0, 1]},
                 {"a": [], "b": [4, 5, 6], "offset": [0, 0, 1]},         # (0,0,0)
                 {"a": [1e30, -1e30, 2 ** 70], "b": [1e-300, 0, 3], "offset": [True, 0, 1]},
                 {"a": "xyz", "b": [4, 5, 6], "offset": [0, 0, 1]},
                 {"a": None, "b": [4, 5, 6], "offset": [0, 0, 1]},
                 {"a": [1, 2, 3], "b": [4, 5, 6], "offset": [0, 0, 1], "axis": [1]},
                 {"a": [0.1, 0.2, 0.30000000000000004], "b": [1 / 3, 2 / 3, 1.0],
                  "offset": [-0.0, 1e-7, 123456789.123456789]}]
    raws += variants
    loads = []
    ok_dims = []
    for raw in raws:
        def build(raw=raw):
            return Dimension(
                QVector3D(*raw["a"]), QVector3D(*raw["b"]),
                QVector3D(*raw["offset"]), layer=raw.get("layer") or None,
                text=raw.get("text") or None, axis=raw.get("axis") or None)
        r = try_(build)
        if "out" in r:
            ok_dims.append(r["out"])
            r = {"idx": len(ok_dims) - 1}
        loads.append({"raw": raw, **r})
    scene = Scene()
    scene.dimensions.extend(ok_dims)
    igz.save_scene(scene, p)
    entries = json.loads(p.read_text(encoding="utf-8"))["scene"]["dimensions"]
    for row in loads:
        if "idx" in row:
            row["out"] = dumps(entries[row.pop("idx")])
    return consts, rows, loads


# ---- text labels -----------------------------------------------------------------
def textlabel_fixtures() -> list:
    g = Gen(7003)
    rows = []
    for _ in range(220):
        an, off = g.vec(), g.vec(-5, 5)
        text, layer = g.pick(STRINGS), g.pick(LAYERS)
        t = TextLabel(an, off, text, layer=layer)
        rows.append({"anchor": vb(an), "offset": vb(off), "text": text, "layer": layer,
                     "pos": vb(t.position()), "dict": dumps(t.to_dict())})
    base = [TextLabel(g.vec(), g.vec(), g.pick(STRINGS), g.pick(LAYERS)).to_dict()
            for _ in range(40)]
    raws = [dict(b) for b in base]
    for b in base[:25]:
        v = dict(b)
        for k in g.r.sample(["anchor", "offset", "text", "layer"], g.r.randint(1, 3)):
            v.pop(k, None)
        raws.append(v)
    raws += [{}, {"anchor": [1, 2, 3]}, {"anchor": [1, 2], "text": "x"},
             {"anchor": None}, {"anchor": [], "offset": [1, 1, 1], "text": "e"},
             {"anchor": [1, 2, 3], "offset": [4, 5, 6], "text": "t", "layer": 0},
             {"anchor": [1, 2, 3], "offset": [4, 5, 6], "text": "t", "layer": ""},
             {"anchor": [1e300, 2, 3], "offset": [True, 5, 6], "text": "t", "layer": None},
             {"anchor": [0.1, 0.7, 1 / 3], "offset": [1e-9, 123456.789, -5], "text": "u\u00e9",
              "layer": "L", "extra": [1]}]
    for raw in raws:
        def build(raw=raw):
            t = TextLabel.from_dict(raw)
            return {"dict": dumps(t.to_dict()), "pos": vb(t.position())}
        r = try_(build)
        rows.append({"raw": raw, **r})
    return rows


# ---- guides -------------------------------------------------------------------------
def guide_fixtures() -> list:
    g = Gen(7004)
    rows = []
    specials = [float("nan"), float("inf"), float("-inf")]
    for _ in range(300):
        p = g.vec()
        r = g.r.random()
        if r < 0.3:
            d = None
        elif r < 0.4:
            d = QVector3D(0, 0, 0)
        elif r < 0.5:
            d = QVector3D(f32(1e-13), 0, 0)
        elif r < 0.55:
            d = QVector3D(f32(1e-11), f32(-2e-11), 0)
        elif r < 0.65:
            d = g.axisvec()
        elif r < 0.7:
            d = QVector3D(f32(1.0 + g.r.uniform(-2e-6, 2e-6)), 0, 0)
        else:
            d = g.vec()
        o = g.vec() if g.r.random() < 0.5 else None
        if g.r.random() < 0.06:
            p = QVector3D(g.pick(specials), p.y(), p.z())
        if d is not None and g.r.random() < 0.03:
            d = QVector3D(d.x(), g.pick(specials), d.z())
        if o is not None and g.r.random() < 0.03:
            o = QVector3D(o.x(), o.y(), g.pick(specials))
        gd = Guide(p, d, o)
        s = gd.segment()
        rows.append({"point": vb(p), "dir": None if d is None else vb(d),
                     "origin": None if o is None else vb(o),
                     "o_dir": None if gd.direction is None else vb(gd.direction),
                     "o_origin": None if gd.origin is None else vb(gd.origin),
                     "is_line": gd.is_line, "finite": gd.is_finite(),
                     "seg": [vb(s[0]), vb(s[1])], "a": vb(gd.a), "b": vb(gd.b),
                     "dict": dumps(gd.to_dict())})
    base = []
    for _ in range(40):
        p = g.vec()
        d = g.vec() if g.r.random() < 0.6 else None
        o = g.vec() if (d is None and g.r.random() < 0.6) else None
        base.append(Guide(p, d, o).to_dict())
    raws = [dict(b) for b in base]
    for b in base[:25]:
        v = dict(b)
        for k in g.r.sample(["direction", "origin"], g.r.randint(1, 2)):
            v.pop(k, None)
        raws.append(v)
    raws += [{}, {"point": [1, 2]}, {"point": [1, 2, 3]},
             {"point": [1, 2, 3], "direction": []},
             {"point": [1, 2, 3], "direction": [], "origin": [4, 5, 6]},
             {"point": [1, 2, 3], "direction": None, "origin": [4, 5, 6]},
             {"point": [1, 2, 3], "direction": [0, 0, 0], "origin": [4, 5, 6]},
             {"point": [1, 2, 3], "direction": [0, 0, 7]},
             {"point": [1, 2, 3], "direction": [1, 2], "origin": None},
             {"point": [1, 2, 3], "direction": [3, 4, 0], "origin": [1, 1, 1]},
             {"point": [1e300, 2, 3], "direction": [float("nan"), 0, 1]},
             {"point": [float("inf"), 0, 0]},
             {"point": [0.1, 0.2, 0.3], "direction": [0.3, 0.2, 0.1], "extra": 1}]
    for raw in raws:
        def build(raw=raw):
            gd = Guide.from_dict(raw)
            return {"dict": dumps(gd.to_dict()), "finite": gd.is_finite()}
        rows.append({"raw": raw, **try_(build)})
    return rows


# ---- section planes --------------------------------------------------------------------
FILLS = [None, {}, {"pattern": "solid"}, {"pattern": "hatch", "spacing": 0.1},
         {"pattern": "cross", "color": [0.1, 0.2, 0.3], "spacing": 0.02},
         {"pattern": "none"}, {"pattern": "bogus", "color": [1, 0, 0]},
         {"color": [1, 2]}, {"color": [1, 2, 3, 4]}, {"color": (0.5, 0.5, 0.5)},
         {"spacing": -1}, {"spacing": 0}, {"spacing": 0.5}, {"spacing": True},
         {"spacing": "x"}, {"color": "abc"}, {"pattern": ["x"]}, "str", [1, 2],
         {"color": [0, 0, 1], "pattern": "hatch", "spacing": 3, "junk": 5},
         {"spacing": 1e-9, "color": [True, 0, 2 ** 70]}]


def section_fixtures():
    g = Gen(7005)
    section_mod._last_uid = 0
    ops, planes = [], []
    for i in range(300):
        r = g.r.random()
        if r < 0.5:
            pt, nm = g.vec(), g.vec()
            q = g.r.random()
            if q < 0.1:
                nm = QVector3D(0, 0, 0)
            elif q < 0.15:
                nm = QVector3D(f32(1e-13), 0, 0)
            elif q < 0.35:
                nm = g.axisvec()
            uid = g.pick([None, None, None, None, 7, 3, 12, 100, 5, 0, -4])
            name, symbol = g.pick(["", "Corte A", "n\u00e9"]), g.pick(["", "A", "B-B"])
            active, hidden = g.r.random() < 0.3, g.r.random() < 0.2
            sp = SectionPlane(pt, nm, name=name, symbol=symbol, active=active,
                              uid=uid, hidden=hidden)
            fill = g.pick(FILLS[:12]) if g.r.random() < 0.3 else None
            sp.fill = fill if fill else None
            ops.append({"op": "new", "point": vb(pt), "normal": vb(nm), "name": name,
                        "symbol": symbol, "active": active, "uid": uid,
                        "hidden": hidden, "fill": fill})
        else:
            pt, nm = g.vec(), g.vec()
            raw = {"point": [pt.x(), pt.y(), pt.z()], "normal": [nm.x(), nm.y(), nm.z()]}
            for k, vals in (("name", ["", "Plano", "\u00e9"]), ("symbol", ["", "C"]),
                            ("active", [True, False, 1, 0, "yes", None, []]),
                            ("hidden", [True, False, 1, 0, None]),
                            ("uid", [1, 5, 9, 9, 2.7, "11", True, None, 40, 3]),
                            ("fill", FILLS)):
                if g.r.random() < 0.5:
                    raw[k] = g.pick(vals)
            q = g.r.random()
            if q < 0.04:
                raw.pop("point")
            elif q < 0.08:
                raw["normal"] = [0, 0, 0]
            elif q < 0.1:
                raw["normal"] = [1, 2]
            ops.append({"op": "from", "raw": raw})
        # What the class makes of that op (re-run in order below).
    # replay for real, in order, recording; the uid counter is module state
    section_mod._last_uid = 0
    rows = []
    for op in ops:
        if op["op"] == "new":
            sp = SectionPlane(vf(op["point"]), vf(op["normal"]), name=op["name"],
                              symbol=op["symbol"], active=op["active"], uid=op["uid"],
                              hidden=op["hidden"])
            sp.fill = op["fill"] if op["fill"] else None
            res = {"out": sp}
        else:
            res = try_(lambda: SectionPlane.from_dict(op["raw"]))
        row = dict(op)
        if "out" in res:
            sp = res["out"]
            pts = [g.vec(), g.vec(), QVector3D(sp.point)]
            row["res"] = {
                "normal": vb(sp.normal), "uid": sp.uid, "dict": dumps(sp.to_dict()),
                "side": [h32(sp.side(p)) for p in pts], "pts": [vb(p) for p in pts],
                "fill": None if sp.fill is None else dumps(sp.fill),
            }
            ef = []
            for sf, col in ((True, (0.35, 0.37, 0.41)), (False, (1.0, 0.5, 0.25)),
                            (True, (0.0, 0.0, 0.0))):
                style = SimpleNamespace(section_fill=sf, section_fill_color=col)
                pat, c, sp_ = effective_fill(sp, style)
                ef.append([sf, list(col), pat, [h64(x) for x in c], h64(sp_)])
            row["res"]["eff"] = ef
            sp.flip()
            row["res"]["flipped"] = dumps(sp.to_dict())
            row["res"]["flip_side"] = h32(sp.side(pts[0]))
        else:
            row["err"] = res["err"]
        row["next_uid"] = section_mod._last_uid
        rows.append(row)
    # next_symbol
    syms = []
    pool = ["", "A", "B", "C", "Z", "AA", "AZ", " a ", "b", "ZZ", "AB", "x", "?", "\u00e9"]
    for _ in range(150):
        lst = [g.pick(pool + [None]) for _ in range(g.r.randint(0, 8))]
        syms.append({"syms": lst, "out": next_symbol(
            [SimpleNamespace(symbol=s) for s in lst])})
    alph = [chr(ord("A") + i) for i in range(26)]
    full = alph + [a + b for a in alph for b in alph[:3]]
    for n in (0, 1, 25, 26, 27, 28, 29, 52, 78, 100):
        lst = full[:n]
        syms.append({"syms": lst, "out": next_symbol(
            [SimpleNamespace(symbol=s) for s in lst])})
    # effective_fill on raw dicts that bypass from_dict's cleaning (UI-set fills)
    eff = []
    for fill in FILLS:
        if isinstance(fill, dict) and isinstance(fill.get("color"), str):
            continue
        for sf in (True, False):
            for col in ((0.35, 0.37, 0.41), (0.2, 0.4, 0.6)):
                style = SimpleNamespace(section_fill=sf, section_fill_color=col)
                plane = SimpleNamespace(fill=fill)
                r = try_(lambda: effective_fill(plane, style))
                eff.append({"fill": fill, "sf": sf, "col": list(col),
                            **({"out": [r["out"][0], [h64(x) for x in r["out"][1]],
                                        h64(r["out"][2])]} if "out" in r else r)})
    return rows, syms, eff


# ---- saved views -------------------------------------------------------------------------
STYLE_DICTS = [None, {}, {"section_fill": True, "edges": "x"}, {"a": [1, 2.5], "b": None},
               {"section_fill_color": [0.35, 0.37, 0.41], "name": "caf\u00e9"}]
SECTION_DICTS = [None, {}, {"active": None, "planes_shown": True, "cuts_shown": True},
                 {"active": 3, "planes_shown": False, "cuts_shown": True}, {"active": 7}]
GEOREF_DICTS = [None, {}, {"map": True, "terrain": False, "survey": False},
                {"map": False, "terrain": True, "survey": True}]
SHADOW_DICTS = [None, {}, {"on": True, "date": "2026-06-21", "time": 12.5, "dark": 0.4}]
HSHOWN = [None, {}, {"objects": False, "geometry": True}, {"objects": True, "geometry": True}]


def view_args(g: Gen) -> dict:
    a = {"name": g.pick(["Planta", "Vista 1", "Escena \u00f1", "", "A B"]),
         "target": [g.f64(-50, 50) for _ in range(3)],
         "distance": g.f64(0.5, 200), "yaw": g.f64(-3.2, 3.2), "pitch": g.f64(-1.5, 1.5),
         "fov_deg": g.pick([45.0, 45.0, 30.0, 60.0, g.f64(10, 100)]),
         "perspective": g.r.random() < 0.7, "two_point": g.r.random() < 0.2,
         "hidden_layers": g.pick([None, [], ["Muros"], ["A", "B\u00e9", "C"]]),
         "style": g.pick(STYLE_DICTS), "section": g.pick(SECTION_DICTS),
         "georef": g.pick(GEOREF_DICTS), "shadows": g.pick(SHADOW_DICTS),
         "hidden_objects": g.pick([None, [], [1, 2, 3], [7]]),
         "hidden_shown": g.pick(HSHOWN)}
    return a


def view_fixtures():
    g = Gen(7006)
    rows = []
    dicts = []
    for _ in range(200):
        a = view_args(g)
        if g.r.random() < 0.1:
            a["target"] = g.pick([(1, 2, 3), (0, 0, 0.5)])
        v = SavedView(a["name"], target=a["target"], distance=a["distance"],
                      yaw=a["yaw"], pitch=a["pitch"], fov_deg=a["fov_deg"],
                      perspective=a["perspective"], two_point=a["two_point"],
                      hidden_layers=a["hidden_layers"], style=a["style"],
                      section=a["section"], georef=a["georef"], shadows=a["shadows"],
                      hidden_objects=a["hidden_objects"], hidden_shown=a["hidden_shown"])
        d = v.to_dict()
        dicts.append(d)
        rows.append({"args": a, "dict": dumps(d)})
    raws = [dict(d) for d in dicts[:60]]
    for d in dicts[60:120]:
        v = dict(d)
        for k in g.r.sample(sorted(v), g.r.randint(1, 4)):
            v.pop(k, None)
        raws.append(v)
    for d in dicts[120:150]:
        v = dict(d)
        v["fov"] = g.pick([45, 45.0, 30, "x", None])
        v["distance"] = g.pick([20, 5, 0, 12.5])
        v["yaw"] = g.pick([0, 1, -1])
        v["parallel"] = g.pick([True, False, 1, 0, None])
        v["two_point"] = g.pick([True, 1, 0, "", None, "x"])
        v["target"] = g.pick([[0, 0, 0], [1, 2, 3, 4], None, [], [1, 2]])
        raws.append(v)
    raws += [{}, {"name": "only"}, {"target": [1, 2, 3]}, {"target": [1.5, 2.5, 3.5, 99]},
             {"hidden_objects": []}, {"hidden_objects": None},
             {"section": {}, "georef": {}, "shadows": {}, "style": {}, "hidden_shown": {}},
             {"section": {"active": None}}, {"hidden_layers": []},
             {"hidden_layers": None, "fov": 45.0, "parallel": False},
             {"name": "n", "distance": 1e-320, "yaw": -0.0, "pitch": 1e22,
              "fov": 1e16, "target": [-0.0, 1e-5, 123456789012345678]}]
    loads = []
    for raw in raws:
        def build(raw=raw):
            return dumps(SavedView.from_dict(raw).to_dict())
        loads.append({"raw": raw, **try_(build)})
    look = []
    for _ in range(250):
        eye = [g.f64(-30, 30) for _ in range(3)]
        tgt = [g.f64(-30, 30) for _ in range(3)]
        r = g.r.random()
        if r < 0.12:
            eye = [tgt[0], tgt[1], tgt[2] + g.pick([10.0, -10.0, 3.5])]    # straight up/down
        elif r < 0.17:
            eye = list(tgt)                                                # zero distance
        elif r < 0.25:
            eye = [tgt[0] + 1e-9, tgt[1], tgt[2] + 5.0]
        up = [g.f64(-1, 1), g.f64(-1, 1), g.pick([1.0, 0.0, g.f64(-1, 1)])]
        if g.r.random() < 0.15:
            up = [0.0, 0.0, 1.0]
        fov = g.pick([45.0, 30.0, 60.0, g.f64(5, 120), 0.0])
        persp = g.r.random() < 0.5
        oh = g.pick([None, 0.0, 12.0, g.f64(0.1, 50), -3.0])
        hl = g.pick([None, ["A"], []])
        r = try_(lambda: dumps(from_lookat(
            "Importada", eye, tgt, up, fov_deg=fov, perspective=persp,
            ortho_height=oh, hidden_layers=hl).to_dict()))
        look.append({"eye": eye, "target": tgt, "up": up, "fov": fov, "persp": persp,
                     "oh": oh, "hl": hl, **r})
    return rows, loads, look


# ---- image planes -------------------------------------------------------------------------
def image_fixtures() -> list:
    g = Gen(7007)
    rows, dicts = [], []
    for _ in range(220):
        o, u, v = g.vec(), g.vec(-10, 10), g.vec(-10, 10)
        r = g.r.random()
        if r < 0.1:
            v = QVector3D(0, 0, 0)
        elif r < 0.15:
            u = QVector3D(0, 0, 0)
        elif r < 0.3:
            u, v = QVector3D(g.num(0.5, 20), 0, 0), QVector3D(0, g.num(0.5, 20), 0)
        elif r < 0.35:
            v = u * 2.0
        aspect = g.pick([1.0, 0.75, 1.5, 0.0, -1.0, g.f64(0.2, 3), float("nan")])
        name = g.pick(["", "Plano", "Foto \u00f1"])
        opacity = g.pick([1.0, 0.5, 0.0, g.f64(0, 1), 1.0])
        layer = g.pick([IMAGE_LAYER, IMAGE_LAYER, None, "Other", "", "Images "])
        locked = g.r.random() < 0.2
        path = g.pick(["a.png", "C:\\plans\\p 1.png", "caf\u00e9/\u4e2d.jpg", "", "x/y.tif"])
        ip = ImagePlane(path, o, u, v, aspect=aspect, name=name, opacity=opacity,
                        layer=layer, locked=locked)
        pt = g.vec()
        prj = ip.project(pt)
        sc = []
        for w in (g.f64(-1.0, 30.0), 10.0):
            for keep in (True, False):
                a_, b_ = ip.scaled(w, keep)
                sc.append([h64(w), keep, vb(a_), vb(b_)])
        frac = [(0.0, 0.0), (1.0, 1.0), (-1e-9, 0.5), (1.0 + 1e-9, 0.5), (-2e-9, 0.2),
                (g.f64(-0.5, 1.5), g.f64(-0.5, 1.5))]
        cp, cn = ip.plane()
        rows.append({
            "path": path, "origin": vb(o), "u": vb(u), "v": vb(v), "aspect": h64(aspect),
            "name": name, "opacity": h64(opacity), "layer": layer, "locked": locked,
            "o_aspect": h64(ip.aspect),
            "corners": [vb(c) for c in ip.corners()], "center": vb(ip.center()),
            "width": h64(ip.width()), "height": h64(ip.height()),
            "normal": vb(ip.normal()), "plane": [vb(cp), vb(cn)],
            "border": [[vb(p), vb(q)] for p, q in ip.border_edges()],
            "uv_in": [[h64(a), h64(b), ip.contains_uv(a, b)] for a, b in frac],
            "pt": vb(pt), "project": [h64(prj[0]), h64(prj[1])], "scaled": sc,
            "dict": dumps(ip.to_dict())})
        dicts.append(ip.to_dict())
    raws = [dict(d) for d in dicts[:50]]
    for d in dicts[50:110]:
        v = dict(d)
        for k in g.r.sample(sorted(v), g.r.randint(1, 3)):
            if k not in ("origin", "u", "v"):
                v.pop(k, None)
        raws.append(v)
    for d in dicts[110:140]:
        v = dict(d)
        v.pop("texture", None)
        v["path"] = g.pick(["legacy.png", "", "dir/\u00e9.png"])
        raws.append(v)
    for d in dicts[140:180]:
        v = dict(d)
        v["texture"] = g.pick([None, "str", {}, {"path": "t.png", "extra": 1}, {"nopath": 1}])
        v["layer"] = g.pick(["Images", None, "", "Z"])
        v["aspect"] = g.pick([2, 0, -1, 0.5, "1.5"])
        v["opacity"] = g.pick([1, 0, 0.25, True])
        v["locked"] = g.pick([True, False, 1, 0, None, "x"])
        raws.append(v)
    raws += [{}, {"origin": [0, 0, 0], "u": [1, 0, 0]}, {"origin": [0, 0, 0], "u": [1, 0, 0],
                                                          "v": [0, 1, 0]},
             {"origin": [0, 0], "u": [1, 0, 0], "v": [0, 1, 0]},
             {"origin": [0, 0, 0], "u": [1, 0, 0], "v": [0, 1, 0], "aspect": "x"},
             {"origin": [0, 0, 0], "u": [1, 0, 0], "v": [0, 1, 0], "layer": "Images",
              "name": None}]
    for raw in raws:
        def build(raw=raw):
            ip = ImagePlane.from_dict(raw)
            return {"dict": dumps(ip.to_dict()), "layer": ip.layer, "aspect": h64(ip.aspect)}
        rows.append({"raw": raw, **try_(build)})
    return rows


# ---- whole payloads through the real loader and writer -------------------------------------
def scene_fixtures(dim_loads, tl_rows, gd_rows, sec_rows, view_loads, img_rows):
    g = Gen(7008)

    def finite(raws):
        return [r for r in raws if "NaN" not in dumps(r) and "Infinity" not in dumps(r)]

    dims = finite([r["raw"] for r in dim_loads if "out" in r])
    labels = finite([r["raw"] for r in tl_rows if "raw" in r and "out" in r])
    guides = [r["raw"] for r in gd_rows if "raw" in r and "out" in r]
    planes = finite([r["raw"] for r in sec_rows if r["op"] == "from" and "res" in r])
    views = finite([r["raw"] for r in view_loads if "out" in r])
    images = finite([r["raw"] for r in img_rows if "raw" in r and "out" in r])
    rows = []

    def some(pool, hi):
        return [g.pick(pool) for _ in range(g.r.randint(0, hi))]

    for i in range(50):
        s: dict = {}
        if g.r.random() < 0.8:
            s["saved_views"] = some(views, 3)
        if g.r.random() < 0.8:
            s["dimensions"] = some(dims, 4)
        if g.r.random() < 0.8:
            s["section_planes"] = some(planes, 4)
        if g.r.random() < 0.5:
            s["section_view"] = g.pick([{}, {"planes_hidden": True}, {"cuts_hidden": True},
                                        {"planes_hidden": True, "cuts_hidden": True},
                                        {"planes_hidden": 0, "cuts_hidden": 1},
                                        {"planes_hidden": None}, [], "x", None])
        if g.r.random() < 0.8:
            s["text_labels"] = some(labels, 3)
        if g.r.random() < 0.8:
            s["guides"] = some(guides, 4)
        if g.r.random() < 0.6:
            s["image_planes"] = some(images, 3)
        if g.r.random() < 0.07:
            s["section_planes"] = g.pick([None, {}, "x", 5])
        rows.append({"sections": s, **roundtrip(s)})
    # Hand-made structural cases.
    bad_guides = [
        {"guides": [{"point": [float("nan"), 0, 0]}, {"point": [1, 2, 3]}]},
        {"guides": [{"point": [0, 0, 0], "direction": [0, float("inf"), 0]}]},
        {"guides": [{"point": [1, 2, 3], "origin": [0, 0, float("-inf")]}, {"point": [4, 5, 6]}]},
        {"guides": [{"point": [1e300, 0, 0]}]},
        {"guides": [{"point": ["a", 0, 0]}]},
        {"guides": [{"point": [1, 2, 3]}, {"direction": [1, 0, 0]}]},
        {"guides": [{"point": [1, 2]}]},
        {"guides": [{"point": [1, 2, 3], "direction": [], "origin": []}]},
        {"guides": []},
        {"guides": [{"point": [1, 2, 3]}, {"point": [1, 2, 3]}]},
        {"section_planes": [{"point": [0, 0, 0], "normal": [0, 0, 1], "active": True},
                            {"point": [0, 0, 1], "normal": [0, 0, 1], "active": True},
                            {"point": [0, 0, 2], "normal": [0, 0, 1]}],
         "section_view": {"planes_hidden": True}},
        {"section_planes": [{"point": [0, 0, 0], "normal": [0, 0, 1], "uid": 9},
                            {"point": [0, 0, 1], "normal": [0, 0, 1], "uid": 4},
                            {"point": [0, 0, 2], "normal": [0, 0, 1]}]},
        {"section_planes": []},
        {"section_planes": [{"normal": [0, 0, 1]}]},
        {"section_view": {"cuts_hidden": True}},
        {"dimensions": [{"a": [1, 2, 3], "b": [4, 5, 6]}]},
        {"dimensions": [{"a": [1, 2, 3], "b": [4, 5, 6], "offset": [0, 0, 1], "axis": "y"}]},
        {"text_labels": [{}]},
        {"saved_views": [{}]},
        {"image_planes": [{"origin": [0, 0, 0], "u": [1, 0, 0], "v": [0, 1, 0]}]},
        {},
    ]
    for s in bad_guides:
        rows.append({"sections": s, **roundtrip(s)})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"qt": qVersion(), "pyside": pyside_version, "platform": sys.platform,
            "python": sys.version.split()[0]}
    print("writing", OUT.relative_to(ROOT), meta)
    write("units.jsonl", meta, units_fixtures())
    consts, dim_rows, dim_loads = dimension_fixtures()
    write("dimension.jsonl", {**meta, **consts}, dim_rows)
    write("dimension_load.jsonl", meta, dim_loads)
    tl = textlabel_fixtures()
    write("textlabel.jsonl", meta, tl)
    gd = guide_fixtures()
    write("guide.jsonl", meta, gd)
    sec_rows, syms, eff = section_fixtures()
    write("section.jsonl", meta, sec_rows)
    write("section_symbols.jsonl", meta, syms)
    write("section_fill.jsonl", meta, eff)
    v_rows, v_loads, v_look = view_fixtures()
    write("saved_view.jsonl", meta, v_rows)
    write("saved_view_load.jsonl", meta, v_loads)
    write("saved_view_lookat.jsonl", meta, v_look)
    im = image_fixtures()
    write("image_plane.jsonl", meta, im)
    write("scene.jsonl", meta, scene_fixtures(dim_loads, tl, gd, sec_rows, v_loads, im))


if __name__ == "__main__":
    main()
