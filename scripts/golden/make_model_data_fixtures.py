# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixtures that hold ``ingetrazo-model``'s document data classes to the
Python ones, byte for byte (docs/rust-rewrite-plan.md, M2.4).

The REAL classes run on many inputs: ``core.materials`` (Material, the
registry, face stamping, container paint), ``core.layers`` (Layer, the igz
layer block, layer_of / assign_layer), the BIM tag data in ``core.bim`` and
the igz tag readers, the pure-data half of ``core.texture`` (the texture
dict, ``texture_file_name``, the image sniffing) and the face-attrs reader
and writer of ``formats.igz``. Every input is recorded as the JSON text it
was built from, and each result as the ``json.dumps`` text the Python code
produced (or the exception class it raised), so Rust must reproduce text,
not just values: key order, int versus float, float repr, escapes.

Odd inputs are the point: legacy and partial dicts, wrong types where Python
duck-types them (a string for a colour is a tuple of characters), unicode
names, out-of-range numbers, floats spelled as strings.

Also writes ``rust/crates/ingetrazo-model/src/pyalnum.rs``, the table of code
points for which Python's ``str.isalnum()`` is true (``texture_file_name``
filters with it, and Rust's ``char::is_alphanumeric`` differs on combining
marks).

Committed, so cargo test needs no Python; rerun only to widen them:

    python scripts/golden/make_model_data_fixtures.py

Deleted with the Python app at M10.
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
import unicodedata
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "rust" / "crates" / "ingetrazo-model" / "tests" / "data" / "model_data"
ALNUM_RS = ROOT / "rust" / "crates" / "ingetrazo-model" / "src" / "pyalnum.rs"

from core import bim, layers, materials, texture                 # noqa: E402
from core.layers import DEFAULT_LAYER, Layer                      # noqa: E402
from core.materials import Material                               # noqa: E402
from formats import igz                                           # noqa: E402


def dumps(obj) -> str:
    return json.dumps(obj)


def attempt(fn):
    """{"ok": json text} or {"err": exception class}."""
    try:
        return {"ok": dumps(fn())}
    except Exception as exc:                                       # noqa: BLE001
        return {"err": type(exc).__name__}


# ---- random JSON-ish values ------------------------------------------------

NAMES = ["Concreto visto", "Wood_Floor", "Tarrajeo", "", "Material", "Vidrio",
         "agua", "Ladrillo \u00e1\u00e9\u00ed\u00f3\u00fa \u00f1", "\u7816\u5757",
         "emoji \U0001f9f1", "a\"b\\c", "tab\tnew\nline", "Layer 0", "Survey",
         "x" * 70, "  spaced  ", "0", "Material (2)"]


def wild(r: random.Random, depth: int = 0):
    k = r.random()
    if depth > 2 or k < 0.5:
        return r.choice([None, True, False, 0, 1, -1, 7, 2 ** 70, 0.0, -0.0,
                         0.5, 1.0, 1e-7, 1e22, 123456.789, "", "abc", "0.5",
                         "x", " 1_0 ", "nan", "inf", "1e3", "ab", "\u00e9\u00e8",
                         [], {}, [1, 2], ["a", "b"], {"a": 1}])
    if k < 0.75:
        return [wild(r, depth + 1) for _ in range(r.randint(0, 3))]
    return {r.choice(["a", "b", "path", "k"]): wild(r, depth + 1)
            for _ in range(r.randint(0, 3))}


def rgb(r: random.Random):
    k = r.random()
    if k < 0.55:
        return [round(r.random(), r.choice([1, 2, 3, 6])) if r.random() < 0.7
                else r.random() for _ in range(3)]
    if k < 0.7:
        return [r.choice([0, 1]) for _ in range(3)]
    if k < 0.8:
        return [r.random() for _ in range(r.choice([0, 1, 2, 4]))]
    return wild(r)


def tex_dict(r: random.Random):
    k = r.random()
    if k < 0.6:
        d = {"path": r.choice(["C:\\tex\\brick.png", "/tmp/a b.jpg", "tex/\u00e9.png",
                               "textures/0123456789abcdef-wall.png", ""]),
             "sw": r.choice([1.0, 0.6, 2, 0.25, 1e-3]),
             "sh": r.choice([1.0, 0.6, 2, 0.3])}
        if r.random() < 0.3:
            d["rot"] = r.choice([0.0, 90.0, 12.5])
        if r.random() < 0.2:
            d["uvw"] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        if r.random() < 0.2:
            d["base"] = "orig.png"
            d["tint"] = [0.5, 0.5, 0.5]
            d["tint_mode"] = 1
        items = list(d.items())
        r.shuffle(items)
        return dict(items)
    if k < 0.7:
        return [["path", "x.png"], ["sw", 2]]
    if k < 0.8:
        return r.choice(["ab", "abc", "", [], {}, 0, 5, True, None, [[1, 2]],
                         [["a", "b", "c"]], [[[1], 2]], ["ab", "cd"], [5]])
    return wild(r)


def material_dict(r: random.Random, sane: bool = False):
    d: dict = {}
    keys = ["name", "color", "texture", "opacity", "finish", "extra"]
    r.shuffle(keys)
    for key in keys:
        if r.random() < (0.9 if key == "name" else 0.5):
            if key == "name":
                d[key] = r.choice(NAMES) if (sane or r.random() < 0.85) else wild(r)
            elif key == "color":
                d[key] = rgb(r) if not sane else [r.random() for _ in range(3)]
            elif key == "texture":
                d[key] = tex_dict(r) if not sane else {"path": "t.png", "sw": 1.0, "sh": 1.0}
            elif key == "opacity":
                d[key] = (r.choice([0.5, 1, 0, 0.25, 0.999, 1.0, 0.0])
                          if sane or r.random() < 0.7
                          else r.choice([True, "0.7", " .5 ", "1_0", None, [], "abc", "nan",
                                         -2.5, 3, 1e30, 2 ** 70, {}]))
            elif key == "finish":
                d[key] = r.choice(["gloss", "matte", "metal", "", None, "auto"]) if (
                    sane or r.random() < 0.8) else wild(r)
            else:
                d[key] = "ignored"
    return d


# ---- materials ----------------------------------------------------------------

def mat_case(raw_text: str) -> dict:
    raw = json.loads(raw_text)
    case: dict = {"op": "material", "in": raw_text}
    try:
        m = Material.from_dict(raw)
    except Exception as exc:                                       # noqa: BLE001
        case["from"] = {"err": type(exc).__name__}
        return case
    case["from"] = {"ok": None}
    case["to_dict"] = attempt(m.to_dict)
    case["face_attrs"] = attempt(m.face_attrs)
    if "ok" in case["to_dict"]:
        again = Material.from_dict(json.loads(case["to_dict"]["ok"]))
        case["round"] = attempt(again.to_dict)
    return case


def gen_materials(r: random.Random, out: list) -> None:
    raws = [
        "{}", "[]", '"text"', "5", "null", "true", '{"name": "m"}',
        '{"name": "m", "color": [1, 0.5, 0.25]}',
        '{"name": "m", "color": "abc"}', '{"name": "m", "color": 7}',
        '{"name": "m", "color": {"r": 1, "g": 2}}', '{"name": "m", "color": null}',
        '{"name": "m", "texture": {}}', '{"name": "m", "texture": null}',
        '{"name": "m", "texture": [["path", "a.png"], ["sw", 2]]}',
        '{"name": "m", "texture": [[1, "a"], [true, "b"], [null, 3], [1.5, 2]]}',
        '{"name": "m", "texture": [[1, "a"], ["1", "b"], [1.0, "c"], [true, "d"]]}',
        '{"name": "m", "texture": ["ab", "cd", "ab"]}',
        '{"name": "m", "texture": [{"k": 1, "v": 2}, ["x", "y"]]}',
        '{"name": "m", "texture": ["ab"]}', '{"name": "m", "texture": [{"k": 1}]}',
        '{"name": "m", "texture": "ab"}','{"name": "m", "texture": "abc"}',
        '{"name": "m", "texture": 5}', '{"name": "m", "texture": [[[1], 2]]}',
        '{"name": "m", "opacity": 1e400}', '{"name": "m", "opacity": -1e400}',
        '{"name": "m", "opacity": NaN}', '{"name": "m", "opacity": "inf"}',
        '{"name": "m", "opacity": "  0.25  "}', '{"name": "m", "opacity": "1_000.5"}',
        '{"name": "m", "opacity": "_1"}', '{"name": "m", "opacity": "1__0"}',
        '{"name": "m", "opacity": 1' + "0" * 400 + '}', '{"name": "m", "opacity": 12345678901234567890}',
        '{"name": "m", "opacity": true}', '{"name": "m", "opacity": false}',
        '{"name": "m", "opacity": 0}', '{"name": "m", "opacity": 0.0}',
        '{"name": "m", "opacity": -0.0}', '{"name": "m", "opacity": 1E2}',
        '{"name": "m", "opacity": 0.1}', '{"name": "m", "opacity": 1e-7}',
        '{"name": "m", "opacity": 1e22}', '{"name": "m", "opacity": 5e-324}',
        '{"name": "m", "finish": 0}', '{"name": "m", "finish": []}',
        '{"name": "m", "finish": {"a": 1}}', '{"name": "m", "finish": "x"}',
        '{"name": 5}', '{"name": null}', '{"name": [1]}', '{"name": {"a": [1, 2]}}',
        '{"name": "dup", "name": "last"}', '{"color": [1, 0, 0]}',
        '{"name": "\\u00f1\\ud83e\\uddf1", "color": [0.1, 0.2, 0.30000000000000004]}',
        '{"name": "Concreto", "color": [0.5, 0.5, 0.5], "texture": {"path": "c.png", '
        '"sw": 1.0, "sh": 1.0}, "opacity": 0.5, "finish": "gloss"}',
        '{"name": "old", "color": [128, 128, 128], "diffuse": [1, 2, 3]}',
    ]
    for t in raws:
        out.append(mat_case(t))
    for _ in range(700):
        out.append(mat_case(dumps(material_dict(r))))


def gen_registry(r: random.Random, out: list) -> None:
    """The igz "materials" block: load, then write, plus register()."""

    def load_write(payload):
        mats = {}
        for raw in payload.get("materials", []):
            mat = Material.from_dict(raw)
            if mat.name:
                mats[mat.name] = mat
        return [m.to_dict() for m in mats.values()] if mats else None

    payloads = [
        {}, {"materials": []}, {"materials": None}, {"materials": "ab"},
        {"materials": {}}, {"materials": {"a": 1}}, {"materials": 5},
        {"materials": [{"name": ""}, {"name": "x"}]},
        {"materials": [{"name": "x", "opacity": 1}, {"name": "x", "opacity": 0.5}]},
        {"materials": [{"name": 1}, {"name": 1.0}, {"name": True}]},
        {"materials": [{"name": [1]}]}, {"materials": [{"name": {"a": 1}}]},
        {"materials": [{"name": "a"}, {"name": "b"}, {"name": "a", "finish": "metal"}]},
        {"materials": [[], {}]}, {"materials": ["x"]},
    ]
    for _ in range(80):
        payloads.append({"materials": [material_dict(r, sane=r.random() < 0.6)
                                       for _ in range(r.randint(0, 5))]})
    for p in payloads:
        out.append({"op": "materials_block", "in": dumps(p), "out": attempt(lambda: load_write(p))})

    def reg(items):
        mats: dict = {}
        names = []
        for raw in items:
            names.append(materials.register(mats, Material.from_dict(raw)))
        return {"names": names, "registry": [m.to_dict() for m in mats.values()]}

    base = [{"name": "A", "color": [1, 0, 0]}, {"name": "A", "color": [1, 0, 0]},
            {"name": "A", "color": [1, 0, 0.0]}, {"name": "A", "color": [0, 0, 1]},
            {"name": "A", "color": [0, 0, 1], "finish": "gloss"},
            {"name": ""}, {"name": ""}, {"name": "", "opacity": 0.5},
            {"name": "Material"}, {"name": "A (2)", "color": [9, 9, 9]},
            {"name": "T", "texture": {"path": "a.png", "sw": 1, "sh": 2}},
            {"name": "T", "texture": {"sh": 2.0, "sw": 1.0, "path": "a.png"}},
            {"name": "T", "texture": {"path": "b.png", "sw": 1, "sh": 2}},
            {"name": "O", "opacity": 1}, {"name": "O", "opacity": 1.0}, {"name": "O", "opacity": True},
            {"name": "O", "opacity": "1"}, {"name": 5, "opacity": 1}, {"name": 5, "opacity": 2},
            {"name": 5.0}, {"name": [1]}, {"name": {"a": 1}}, {"name": None}]
    seqs = [base[i:j] for i in range(len(base)) for j in (i + 1, i + 2, i + 3)]
    seqs.append(base)
    for _ in range(80):
        seqs.append([material_dict(r, sane=True) for _ in range(r.randint(1, 6))])
    for s in seqs:
        out.append({"op": "register", "in": dumps(s), "out": attempt(lambda: reg(s))})


def stampable(r: random.Random):
    """attrs-like dicts for the stamping helpers."""
    pick = r.random()
    if pick < 0.08:
        return r.choice([None, {}, [], "", 0, "str", [1], 5, True])
    d: dict = {}
    for key in ["color", "texture", "opacity", "mat", "back", "layer", "hidden"]:
        if r.random() < 0.4:
            if key == "color":
                d[key] = r.choice([[0.5, 0.5, 0.5], None, (), "abc", 5, [1, 2, 3]])
                if isinstance(d[key], tuple):
                    d[key] = list(d[key])
            elif key == "texture":
                d[key] = r.choice([{"path": "t.png", "sw": 1, "sh": 1}, None, {}, {"path": ""},
                                   {"path": "nofile.png"}, {"path": 5}, "x", [], {"base": "b"}])
            elif key == "opacity":
                d[key] = r.choice([0.5, 1.0, 1, 0, 0.9989, 0.999, 0.9999, "0.5", None, True, "x", []])
            elif key == "mat":
                d[key] = r.choice(["m", "", None, 5])
            elif key == "back":
                d[key] = r.choice([True, False, None, {"color": [1, 0, 0]}, {}, "x", 1])
            else:
                d[key] = r.choice(["L", True, False, None, 1])
    return d


def container_material(r: random.Random):
    if r.random() < 0.1:
        return r.choice([None, {}, [], "", 0, "color", "xcolorx", ["color"], ["a"], 5, "plain"])
    d: dict = {}
    for key in ["color", "texture", "opacity", "mat"]:
        if r.random() < 0.6:
            d[key] = r.choice({"color": [[0.2, 0.3, 0.4], None, "abc", 5],
                               "texture": [{"path": "p.png", "sw": 1, "sh": 2, "rot": 5}, None, "x", {}, []],
                               "opacity": [0.5, None, 1, "x"],
                               "mat": ["Brick", None, "", 5]}[key])
    return d


def gen_stamping(r: random.Random, out: list) -> None:
    for _ in range(500):
        a = stampable(r)
        m = container_material(r)
        out.append({"op": "stamp", "attrs": dumps(a), "material": dumps(m),
                    "has_own": attempt(lambda: materials.has_own_material(a)),
                    "effective": attempt(lambda: materials.effective_attrs(a, m)),
                    "sig": attempt(lambda: materials.material_sig(m)),
                    "translucent": attempt(lambda: materials.is_translucent(a)),
                    "back_default": attempt(lambda: materials.back_is_default(a))})


# ---- layers --------------------------------------------------------------------

def gen_layers(r: random.Random, out: list) -> None:
    def layer_case(text: str) -> dict:
        raw = json.loads(text)
        case: dict = {"op": "layer", "in": text}
        try:
            ly = Layer.from_dict(raw)
        except Exception as exc:                                   # noqa: BLE001
            case["from"] = {"err": type(exc).__name__}
            return case
        case["from"] = {"ok": None}
        case["to_dict"] = attempt(ly.to_dict)
        case["round"] = attempt(lambda: Layer.from_dict(json.loads(dumps(ly.to_dict()))).to_dict())
        return case

    texts = ["{}", "[]", "null", '"x"', "5", '{"name": "Walls"}', '{"name": "W", "visible": false}',
             '{"name": "W", "locked": true}', '{"name": "W", "visible": false, "locked": true}',
             '{"name": "W", "visible": 0}', '{"name": "W", "visible": ""}',
             '{"name": "W", "visible": null}', '{"name": "W", "visible": "no"}',
             '{"name": "W", "locked": 1}', '{"name": "W", "locked": "yes"}',
             '{"name": "W", "locked": 0}', '{"name": "W", "locked": []}',
             '{"name": "W", "locked": [0]}', '{"name": "W", "visible": 0.0}',
             '{"name": "W", "visible": {"a": 1}}', '{"name": null}', '{"name": 7}',
             '{"visible": false}', '{"name": "\\u00c1rea \\u2603 \\ud83c\\udfd7", "locked": true}',
             '{"name": "W", "color": [1, 2, 3]}']
    for t in texts:
        out.append(layer_case(t))
    for _ in range(150):
        d: dict = {}
        for key in ["name", "visible", "locked", "color"]:
            if r.random() < 0.7:
                d[key] = (r.choice(NAMES) if key == "name" and r.random() < 0.8
                          else r.choice([True, False, 0, 1, None, "", "x", [], 0.0, 2]) if key != "color"
                          else [1, 2, 3])
        out.append(layer_case(dumps(d)))

    def block(payload):
        lays = [Layer(DEFAULT_LAYER)]
        raw_layers = payload.get("layers")
        if raw_layers:
            lays = [Layer.from_dict(x) for x in raw_layers]
            if not any(ly.name == DEFAULT_LAYER for ly in lays):
                lays.insert(0, Layer(DEFAULT_LAYER))
        written = None
        if lays is not None and (len(lays) > 1 or any(not ly.visible or ly.locked for ly in lays)):
            written = [ly.to_dict() for ly in lays]
        return {"loaded": [ly.to_dict() for ly in lays], "written": written}

    payloads = [{}, {"layers": []}, {"layers": None}, {"layers": [{"name": "Layer 0"}]},
                {"layers": [{"name": "Layer 0", "visible": False}]},
                {"layers": [{"name": "Walls"}]}, {"layers": [{"name": "Walls"}, {"name": "Layer 0"}]},
                {"layers": [{"name": "Walls", "locked": True}, {"name": "Layer 0", "locked": 1}]},
                {"layers": "ab"}, {"layers": 5}, {"layers": {"a": 1}}, {"layers": [{}]},
                {"layers": [{}, {}]}, {"layers": [[]]}, {"layers": [{"name": 1}, {"name": True}]},
                {"layers": [{"name": "layer 0"}]}, {"layers": [{"name": ["Layer 0"]}]}]
    for _ in range(60):
        payloads.append({"layers": [{k: v for k, v in [
            ("name", r.choice(NAMES)), ("visible", r.choice([True, False, 0, 1])),
            ("locked", r.choice([True, False, 0, 1]))] if r.random() < 0.7}
            for _ in range(r.randint(0, 5))]})
    for p in payloads:
        out.append({"op": "layers_block", "in": dumps(p), "out": attempt(lambda: block(p))})

    for _ in range(120):
        attrs = {k: v for k, v in [("layer", r.choice(NAMES + [None, 5, "", 0, True])), ("color", [1, 2, 3])]
                 if r.random() < 0.6}
        name = r.choice(NAMES + [DEFAULT_LAYER, DEFAULT_LAYER, 5, None, "", True])
        face = SimpleNamespace(attrs=dict(attrs))
        grp = SimpleNamespace(layer=attrs.get("layer"))
        nolayer = SimpleNamespace()

        def assign_face():
            f = SimpleNamespace(attrs=dict(attrs))
            layers.assign_layer(f, name)
            return f.attrs

        def assign_slot():
            g = SimpleNamespace(layer=attrs.get("layer", "orig"))
            layers.assign_layer(g, name)
            return g.layer

        out.append({"op": "layer_ops", "attrs": dumps(attrs), "name": dumps(name),
                    "of_face": attempt(lambda: layers.layer_of(face)),
                    "of_slot": attempt(lambda: layers.layer_of(grp)),
                    "of_none": attempt(lambda: layers.layer_of(nolayer)),
                    "assign_face": attempt(assign_face),
                    "assign_slot": attempt(assign_slot)})


# ---- BIM tags ------------------------------------------------------------------

def gen_bim(r: random.Random, out: list) -> None:
    out.append({"op": "bim_consts",
                "classes": dumps(bim.IFC_CLASSES), "qsets": dumps(bim.QSET_NAMES),
                "units": dumps(bim.METRADO_UNIT), "steel": dumps(bim.STEEL_DENSITY)})

    def next_id(faces):
        scene = SimpleNamespace(loose_mesh=SimpleNamespace(
            faces=[SimpleNamespace(attrs=a) for a in faces]))
        return bim.next_object_id(scene)

    def ids():
        return r.choice([1, 2, 5, 40, 0, -3, True, False, 2.5, 7.0, "9", None, [3], 2 ** 70])

    cases = [[], [{}], [{"ifc": None}], [{"ifc": {}}], [{"ifc": {"id": 4}}],
             [{"ifc": {"id": 4}}, {"ifc": {"id": 9}}, {"ifc": {"id": 2}}],
             [{"ifc": {"id": True}}], [{"ifc": {"id": -5}}], [{"ifc": {"id": 1.5}}],
             [{"ifc": [1]}], [{"ifc": "x"}], [{"ifc": 5}], [{"ifc": 0}], [{"ifc": {"class": "IfcWall"}}]]
    for _ in range(80):
        cases.append([{"ifc": {"id": ids(), "class": "IfcWall", "name": "w"}} if r.random() < 0.8 else {}
                      for _ in range(r.randint(0, 6))])
    for c in cases:
        out.append({"op": "next_object_id", "in": dumps(c), "out": attempt(lambda: next_id(c))})

    def tag(cls, name, oid):
        f = SimpleNamespace(attrs={"color": [1, 2, 3], "ifc": "old"})
        g = SimpleNamespace(attrs={})
        bim.tag_faces([f, g], cls, name, oid)
        a = dumps([f.attrs, g.attrs])
        bim.untag_faces([f])
        grp = SimpleNamespace(ifc=None)
        bim.tag_group(grp, cls, name)
        return {"tagged": a, "untagged": f.attrs, "group": grp.ifc}

    for _ in range(80):
        cls, name, oid = r.choice(bim.IFC_CLASSES + ["", None, 5]), r.choice(NAMES + [None, 3]), ids()
        out.append({"op": "tag", "class": dumps(cls), "name": dumps(name), "id": dumps(oid),
                    "out": attempt(lambda: tag(cls, name, oid))})

    def entry_ifc(raw):
        out_ifc = None
        if raw.get("ifc"):
            out_ifc = dict(raw["ifc"])
        return out_ifc

    raws = [{}, {"ifc": None}, {"ifc": {}}, {"ifc": {"class": "IfcWall", "name": "w", "id": 3}},
            {"ifc": {"name": "n", "class": "IfcSlab"}}, {"ifc": [["id", 1], ["class", "IfcDoor"]]},
            {"ifc": "ab"}, {"ifc": "abc"}, {"ifc": 5}, {"ifc": [1]}, {"ifc": [[1, 2, 3]]},
            {"ifc": [["a", 1]]}, {"ifc": True}, {"ifc": 0}, {"ifc": 1.5}, {"ifc": [[None, 1], [True, 2]]},
            {"ifc": {"k": {"nested": [1, 2]}}}, [], "str", None]
    for _ in range(40):
        raws.append({"ifc": wild(r)})
    for raw in raws:
        out.append({"op": "ifc_entry", "in": dumps(raw), "out": attempt(lambda: entry_ifc(raw))})


# ---- texture -------------------------------------------------------------------

def gen_texture(r: random.Random, out: list) -> None:
    out.append({"op": "tex_meta", "limit": texture.NAME_LIMIT,
                "python": sys.version.split()[0], "unidata": unicodedata.unidata_version,
                "platform": sys.platform})
    names = ["brick.png", "C:\\Users\\x\\tex\\wall.jpg", "/tmp/a/b/c.png", "dir/", "dir\\", "",
             ".", "..", "C:", "C:foo.png", "a:b", "//host/share/x.png", "//host/share", "//host", "///x",
             "\\\\srv\\sh\\dir\\f.png", "a/./b.png", "./b.png", "b.png/.", "  ..a b.png.. ", "...", " . ",
             "0123456789abcdef-wall.png", "0123456789abcdef-0123456789abcdef-wall.png",
             "0123456789abcdef-", "0123456789ABCDEF-wall.png", "0123456789abcdeg-wall.png",
             "0123456789abcdef-0123456789abcdef-", "0123456789abcdef0123456789abcdef-x.png",
             "n\u00e4me \u00fc.png", "\u7816\u5757.png", "emoji\U0001f9f1.png", "a\u0301b.png",
             "\u0905\u093e.png", "x\u2460.png", "\u00b2.png", "\u0660\u0661.png", "\u2167.png",
             "a b_c-d.e.png", "a&b#c%d.png", "tab\there.png", "new\nline.png", "nul\x00.png",
             "x" * 64 + ".png", "x" * 70 + ".png", "x" * 59 + ".png", "x" * 60 + ".png",
             "x" * 100, "x" * 70 + ".jpeg", "x" * 70 + ".tiff1", "x" * 70 + ".png.png", "x" * 70 + "..",
             "x" * 62 + " .png", "x" * 63 + ".ab", "x" * 61 + ". .png",
             ("ab " * 30) + ".png", "\u00e9" * 70 + ".png", "\u00e9" * 64, "\u00e9" * 65,
             "a" * 66 + ".\u00e9\u00e8", "a" * 70 + ".\u0301ab", "a" * 70 + ".abcdef", "a" * 70 + ".abcde",
             "a" * 70 + ".", "." + "a" * 70, "." + "a" * 70 + ".png"]
    pool = ("abcxyz XYZ019._-/\\: \u00e9\u00f1\u7816\u0301\u0905\u093e\u2460\U0001f9f1&#%'\"()[]~!\t0123456789abcdef-")
    for _ in range(1200):
        n = r.choice([1, 3, 8, 20, 40, 70, 90])
        s = "".join(r.choice(pool) for _ in range(n))
        if r.random() < 0.3:
            s = "0123456789abcdef-" * r.randint(1, 3) + s
        if r.random() < 0.3:
            s = s + r.choice([".png", ".jpg", ".tiff", ".abcdefg", ".a b", ".", " "])
        names.append(s)
    for n in names:
        out.append({"op": "tex_file_name", "in": dumps(n), "out": attempt(lambda: texture.texture_file_name(n))})

    heads = [b"", b"\x89PNG\r\n\x1a\n", b"\x89PNG\r\n\x1a", b"\xff\xd8\xff\xe0", b"\xff\xd8", b"GIF87a", b"GIF89a",
             b"GIF88a", b"BM", b"B", b"II*\x00", b"MM\x00*", b"II*", b"RIFF\x00\x00\x00\x00WEBP", b"RIFF\x00\x00\x00\x00WEBQ",
             b"RIFF\x00\x00\x00\x00WEB", b"RIFF1234WEBPVP8 ", b"RIFFxx", b"<svg", b"PK\x03\x04", b"-----BEGIN OPENSSH",
             b"\x00" * 16, b"BM" + b"\x00" * 30, b"\x89PNG\r\n\x1a\n" + b"x" * 40]
    for _ in range(60):
        heads.append(bytes(r.randrange(256) for _ in range(r.randint(0, 20))))
    for h in heads:
        out.append({"op": "tex_is_image_bytes", "in": h.hex(), "out": attempt(lambda: texture.is_image_bytes(h))})

    tmp = Path(tempfile.mkdtemp(prefix="ingetrazo-modeldata-"))
    for i, h in enumerate(heads):
        for suffix in (".png", ".txt", ".TGA", ""):
            f = tmp / f"f{i}{suffix}"
            f.write_bytes(h)
            res = attempt(lambda: texture.is_image_file(f))
            out.append({"op": "tex_is_image_file", "head": h.hex(), "suffix": suffix, "out": res})
    out.append({"op": "tex_is_image_file", "head": None, "suffix": ".png",
                "out": attempt(lambda: texture.is_image_file(tmp / "missing.png"))})

    def tex_from(d):
        t = texture.Texture.from_dict(d)
        return {"as_dict": t.as_dict(), "path": t.path}

    ds = [{"path": "a.png"}, {"path": "a.png", "sw": 2, "sh": "3"}, {"path": "a.png", "sw": None}, {},
          {"sw": 1}, [], "str", None, 5, {"path": None, "sw": True, "sh": False},
          {"path": ["x"], "sw": " 2.5 "}, {"path": "a", "sw": "abc"}, {"path": "a", "sw": [1]},
          {"path": "a", "sw": 1e400}, {"path": "a", "sw": 2 ** 70}, {"path": "a", "sh": 10 ** 400},
          {"path": "\u00e9\u7816", "extra": 1}]
    for _ in range(60):
        ds.append(tex_dict(r))
    for d in ds:
        out.append({"op": "tex_dict", "in": dumps(d), "from": attempt(lambda: tex_from(d)),
                    "flattened": attempt(lambda: texture.flattened_texture(d)),
                    "untinted": attempt(lambda: texture.untinted_texture(d))})


# ---- face attrs ----------------------------------------------------------------

def gen_face_attrs(r: random.Random, out: list) -> None:
    def write(attrs):
        return igz._face_attrs_json(SimpleNamespace(attrs=attrs))

    def read(raw):
        return igz._face_attrs_from_json(raw)

    def full(r):
        d: dict = {}
        keys = ["color", "texture", "layer", "ifc", "opacity", "hidden", "mat", "back"]
        r.shuffle(keys)
        for k in keys:
            if r.random() < 0.5:
                if k == "color":
                    d[k] = r.choice([[0.5, 0.25, 0.125], [1, 0, 0], None, "abc", 5, [], rgb(r)])
                elif k == "texture":
                    d[k] = tex_dict(r) if r.random() < 0.7 else None
                elif k == "layer":
                    d[k] = r.choice(["Walls", "", None, 5, "Layer 0", "\u00c1rea"])
                elif k == "ifc":
                    d[k] = r.choice([{"id": 1, "class": "IfcWall", "name": "w"}, None, {}, "ab", [["id", 2]], 5])
                elif k == "opacity":
                    d[k] = r.choice([0.5, 1, 0, None, True, "0.2", "x", [], 1e400 if False else 2.0])
                elif k == "hidden":
                    d[k] = r.choice([True, False, 1, 0, None, "x"])
                elif k == "mat":
                    d[k] = r.choice(["Brick", "", None, 5])
                else:
                    d[k] = r.choice([True, False, None, 1, "x", {}, {"color": [1, 0, 0]},
                                     {"texture": {"path": "a.png", "sw": 1}, "color": [0, 1, 0]},
                                     {"texture": [["path", "x"]]}, {"a": {"b": 1}}])
        return d

    fixed = [{}, [], None, "x", 5, {"color": [1, 2, 3]}, {"back": True}, {"back": {"texture": {"path": "z.png"}}},
             {"hidden": True, "mat": "m"}, {"opacity": 0.5}, {"opacity": "0.5"}, {"ifc": {"id": 3, "class": "IfcSlab"}}]
    cases = fixed + [full(r) for _ in range(500)]
    for a in cases:
        res_w = attempt(lambda: write(a))
        res_r = attempt(lambda: read(a))
        case = {"op": "face_attrs", "in": dumps(a), "write": res_w, "read": res_r}
        if "ok" in res_w:
            case["write_read"] = attempt(lambda: read(json.loads(res_w["ok"])))
        if "ok" in res_r:
            case["read_write"] = attempt(lambda: write(read(a) or {}))
        out.append(case)


# ---- pyalnum.rs ----------------------------------------------------------------

def write_alnum_table() -> int:
    ranges = []
    start = None
    for cp in range(0x110000):
        ok = chr(cp).isalnum()
        if ok and start is None:
            start = cp
        elif not ok and start is not None:
            ranges.append((start, cp - 1))
            start = None
    if start is not None:
        ranges.append((start, 0x10FFFF))
    lines = [
        "// SPDX-License-Identifier: GPL-3.0-or-later",
        "// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.",
        "//! GENERATED by scripts/golden/make_model_data_fixtures.py: do not edit.",
        f"//! The code points for which Python's `str.isalnum()` is true (Python "
        f"{sys.version.split()[0]}, Unicode {unicodedata.unidata_version}).",
        "//! `char::is_alphanumeric` differs on combining marks and a few number forms,",
        "//! and `texture_file_name` must keep exactly the characters Python keeps.",
        "",
        "const RANGES: &[(u32, u32)] = &[",
    ]
    row = []
    for a, b in ranges:
        row.append(f"({a:#x}, {b:#x}),")
        if len(row) == 6:
            lines.append("    " + " ".join(row))
            row = []
    if row:
        lines.append("    " + " ".join(row))
    lines += [
        "];",
        "",
        "/// `c.isalnum()` in Python.",
        "pub fn is_alnum(c: char) -> bool {",
        "    let cp = c as u32;",
        "    RANGES",
        "        .binary_search_by(|&(lo, hi)| {",
        "            if cp < lo {",
        "                std::cmp::Ordering::Greater",
        "            } else if cp > hi {",
        "                std::cmp::Ordering::Less",
        "            } else {",
        "                std::cmp::Ordering::Equal",
        "            }",
        "        })",
        "        .is_ok()",
        "}",
        "",
    ]
    ALNUM_RS.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return len(ranges)


def write_jsonl(name: str, cases: list) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(c, ensure_ascii=True) + "\n" for c in cases)
    (OUT / name).write_text(text, encoding="utf-8", newline="\n")
    print(f"{name}: {len(cases)} cases, {len(text) // 1024} KiB")


def main() -> None:
    n = write_alnum_table()
    print(f"pyalnum.rs: {n} ranges")
    r = random.Random(20261019)
    cases: list = []
    gen_materials(r, cases)
    write_jsonl("materials.jsonl", cases)
    cases = []
    gen_registry(r, cases)
    gen_stamping(r, cases)
    write_jsonl("registry.jsonl", cases)
    cases = []
    gen_layers(r, cases)
    write_jsonl("layers.jsonl", cases)
    cases = []
    gen_bim(r, cases)
    write_jsonl("bim.jsonl", cases)
    cases = []
    gen_texture(r, cases)
    write_jsonl("texture.jsonl", cases)
    cases = []
    gen_face_attrs(r, cases)
    write_jsonl("face_attrs.jsonl", cases)
    (OUT / ".gitattributes").write_text("# Text Python produced; compared byte for byte.\n* -text\n",
                                        encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
