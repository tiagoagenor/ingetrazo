# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""The harness behind ``make_modify_tool_fixtures.py`` (docs/rust-rewrite-plan.md,
M3.13): drives a REAL tool of the Python app (Move, Rotate, Scale, Offset,
Eraser, Paint, Tape, Protractor, Flip) the way a user does, headless.

The REAL ``Viewport`` (offscreen Qt, no window, as ``make_pick_fixtures.py``
does) receives real ``QMouseEvent``s / ``QKeyEvent``s through a real camera, so
the snap engine, the axis locks, the VCB parser, the Esc cascade, the arrow
keys... all run for real. What the tool is *handed* is recorded, call by call,
and that is what Rust replays:

* every outermost call into the tool (``on_activate``, ``on_hover``,
  ``on_click``, ``on_double_click``, ``on_release``, ``on_key``,
  ``on_key_release``, ``on_value``, ``on_array_value``, ``on_cancel``,
  ``on_deactivate``...) with its arguments: a ``ToolContext`` is recorded as
  the snapped world point (float32 bits), the screen position (float64), the
  modifier flags and the snap (kind, point, axis, context);
* the viewport's own effects on the selection (``clear_selection``, ``select``
  outside a tool call: the Esc cascade clears it);
* what the tool asks the viewport to draw/flash: ``flash_status`` texts,
  ``set_hover``, the group-preview hooks (``begin_groups_preview`` ...).

After every event the whole state is recorded: the ordered active mesh, every
group (mesh, transform, axes), the guides, the selection, the undo depth, the
tool's probe (rubber band, guide preview, label, status clause...) - so Rust is
held to the same state after every event, not only at the end.

Python's own order is not repeatable in a few places, pinned here:

* the selection is a ``set`` of objects (address order): replaced by a set that
  iterates in insertion order (Rust keeps a ``Vec``);
* ``_pin_order.install()``: mesh sets iterate in creation order, loops trace
  from the first outline edge;
* group uids come from a counter, ``Group N`` names count from 1, and curve ids
  from 5000, per program.

Deleted with the Python app at M10.
"""
from __future__ import annotations

import json
import math
import os
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.pop("INGETRAZO_RUST", None)

from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_APP = QApplication.instance() or QApplication([])

import functools  # noqa: E402
import itertools  # noqa: E402

import _pin_order  # noqa: E402
import core.group as core_group  # noqa: E402
import core.mesh as cm  # noqa: E402
from core.vecmath import QMatrix4x4, QVector3D  # noqa: E402

_pin_order.install()

from core.group import Group  # noqa: E402
from core.guide import Guide  # noqa: E402
from core.mesh import Mesh  # noqa: E402
from core.scene import Scene  # noqa: E402
from formats import igz  # noqa: E402
from tests.golden.signature import _dumps, canonical, sha  # noqa: E402
from views.viewport import Viewport  # noqa: E402

ORD = _pin_order.ORD
KEEP = _pin_order.KEEP


class _OrderedPieces(dict):
    """An insertion-ordered set for ``fold_nonplanar_faces``'s ``pieces``
    (a set of faces, in address order in Python, which no run repeats - the
    pin of ``make_topology_fixtures.py``, which the Rust topology takes)."""

    def __init__(self, it=()):
        super().__init__((x, None) for x in it)

    def add(self, x):
        self.setdefault(x, None)

    def discard(self, x):
        self.pop(x, None)


def _pin_fold() -> None:
    import inspect
    import textwrap

    import core.history as ch
    import core.topology as ct
    src = textwrap.dedent(inspect.getsource(ct.fold_nonplanar_faces))
    old = """        pieces = {
            mesh.add_face([QVector3D(a), QVector3D(b), QVector3D(c)])
            for a, b, c in tris
        }"""
    new = """        pieces = _OrderedPieces(
            mesh.add_face([QVector3D(a), QVector3D(b), QVector3D(c)])
            for a, b, c in tris
        )"""
    assert old in src, "fold_nonplanar_faces changed"
    ns = dict(ct.__dict__)
    ns["_OrderedPieces"] = _OrderedPieces
    exec(src.replace(old, new), ns)
    ct.fold_nonplanar_faces = ns["fold_nonplanar_faces"]
    ch.fold_nonplanar_faces = ns["fold_nonplanar_faces"]


_pin_fold()

OUT_ROOT = ROOT / "rust" / "crates" / "ingetrazo-tools" / "tests" / "data" / "modify"

W_DEFAULT, H_DEFAULT = 1000, 600


def V(x, y, z=0.0):
    return QVector3D(float(x), float(y), float(z))


# ---- encodings -------------------------------------------------------------------------

def h32(x: float) -> str:
    return struct.pack(">f", x).hex()


def pbits(p) -> list:
    return [h32(p.x()), h32(p.y()), h32(p.z())]


def mbits(m) -> list:
    return [h32(x) for x in m.data()]


def mods_int(m) -> int:
    try:
        return int(m.value)
    except AttributeError:
        return int(m)


SHIFT, CTRL, ALT = 0x02000000, 0x04000000, 0x08000000

KEYS = {
    "ctrl": Qt.Key_Control, "shift": Qt.Key_Shift, "alt": Qt.Key_Alt,
    "right": Qt.Key_Right, "left": Qt.Key_Left, "up": Qt.Key_Up, "down": Qt.Key_Down,
    "esc": Qt.Key_Escape, "enter": Qt.Key_Return, "del": Qt.Key_Delete,
}
#: Qt's integer key codes, as Rust names them.
KEY_CODES = {name: int(getattr(k, "value", k)) for name, k in KEYS.items()}


def qt_mods(shift=False, ctrl=False, alt=False):
    m = Qt.NoModifier
    if shift:
        m |= Qt.ShiftModifier
    if ctrl:
        m |= Qt.ControlModifier
    if alt:
        m |= Qt.AltModifier
    return m


# ---- building scenes ----------------------------------------------------------------------

def box_faces(m: Mesh, x, y, z, sx, sy, sz, color=None):
    x1, y1, z1 = x + sx, y + sy, z + sz
    sides = [
        [V(x, y, z), V(x, y1, z), V(x1, y1, z), V(x1, y, z)],
        [V(x, y, z1), V(x1, y, z1), V(x1, y1, z1), V(x, y1, z1)],
        [V(x, y, z), V(x1, y, z), V(x1, y, z1), V(x, y, z1)],
        [V(x, y1, z), V(x, y1, z1), V(x1, y1, z1), V(x1, y1, z)],
        [V(x, y, z), V(x, y, z1), V(x, y1, z1), V(x, y1, z)],
        [V(x1, y, z), V(x1, y1, z), V(x1, y1, z1), V(x1, y, z1)],
    ]
    out = []
    for pts in sides:
        f = m.add_face(pts)
        if color:
            f.attrs["color"] = list(color)
        out.append(f)
    return out


def box(x, y, z, sx, sy, sz, color=None) -> Mesh:
    m = Mesh()
    box_faces(m, x, y, z, sx, sy, sz, color)
    return m


def xf(dx=0.0, dy=0.0, dz=0.0, angle=0.0, axis=(0, 0, 1), scale=None):
    m = QMatrix4x4()
    m.translate(dx, dy, dz)
    if angle:
        m.rotate(angle, *axis)
    if scale:
        m.scale(*scale)
    return m


def group(mesh, name, m=None, children=None, layer=None, hidden=False, locked=False) -> Group:
    g = Group(mesh, name=name)
    if m is not None:
        g.xform = m
    if children:
        g.adopt(children)
    g.layer = layer
    g.hidden = hidden
    g.locked = locked
    return g


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


# ---- the ordered selection -------------------------------------------------------------------

class OrderedSelection(set):
    """``Scene.selection`` is a ``set`` of objects (address order, not
    repeatable): this one iterates in insertion order."""

    def __init__(self, items=()):
        super().__init__()
        self._order: list = []
        self.update(items)

    def __iter__(self):
        return iter(list(self._order))

    def add(self, x):
        if x not in self:
            self._order.append(x)
        super().add(x)

    def update(self, *others):
        for o in others:
            for x in o:
                self.add(x)

    def discard(self, x):
        if x in self:
            self._order = [y for y in self._order if y is not x]
        super().discard(x)

    def remove(self, x):
        self.discard(x)

    def difference_update(self, *others):
        for o in others:
            for x in list(o):
                self.discard(x)

    def clear(self):
        self._order = []
        super().clear()


# ---- identities ----------------------------------------------------------------------------------

class Ids:
    """Stable identities of everything a tool can hold or a pick can return:
    a loose face or edge by its index in the active mesh's list, a group by its
    path through ``children``, an annotation by its index in its scene list."""

    def __init__(self, vp: Viewport) -> None:
        sc = vp.scene
        self.sc = sc
        self.face = {id(f): i for i, f in enumerate(sc.mesh.faces)}
        self.edge = {id(e): i for i, e in enumerate(sc.mesh.edges)}
        self.vertex = {id(v): i for i, v in enumerate(sc.mesh.vertices)}
        self.gpath: dict = {}

        def walk(g, path):
            self.gpath[id(g)] = path
            for i, c in enumerate(g.children):
                walk(c, path + [i])
        for i, g in enumerate(sc.groups):
            walk(g, [i])
        vp._placements()
        cache = getattr(vp, "_placement_proxies", None) or {}
        for k, p in cache.items():
            if k in self.gpath:
                self.gpath[id(p)] = self.gpath[k]
        self.tables = {
            "dim": {id(d): i for i, d in enumerate(sc.dimensions)},
            "label": {id(d): i for i, d in enumerate(sc.text_labels)},
            "guide": {id(d): i for i, d in enumerate(sc.guides)},
            "section": {id(d): i for i, d in enumerate(sc.section_planes)},
            "image": {id(d): i for i, d in enumerate(sc.image_planes)},
            "geo": {id(d): i for i, d in enumerate(sc.geo_paths)},
        }

    def ref(self, ent):
        if ent is None:
            return None
        if id(ent) in self.face:
            return ["face", self.face[id(ent)]]
        if id(ent) in self.edge:
            return ["edge", self.edge[id(ent)]]
        if id(ent) in self.gpath:
            return ["group", self.gpath[id(ent)]]
        for kind, table in self.tables.items():
            if id(ent) in table:
                return [kind, table[id(ent)]]
        return ["unknown", type(ent).__name__]

    def resolve(self, spec):
        kind, idx = spec
        sc = self.sc
        if kind == "face":
            return sc.mesh.faces[idx]
        if kind == "edge":
            return sc.mesh.edges[idx]
        if kind == "group":
            g = sc.groups[idx[0]]
            for i in idx[1:]:
                g = g.children[i]
            return g
        lists = {"dim": sc.dimensions, "label": sc.text_labels, "guide": sc.guides,
                 "section": sc.section_planes, "image": sc.image_planes, "geo": sc.geo_paths}
        return lists[kind][idx]


# ---- the state recorded after every event -------------------------------------------------------

def mesh_state(mesh) -> dict:
    vi = {id(v): i for i, v in enumerate(mesh.vertices)}
    fi = {id(f): i for i, f in enumerate(mesh.faces)}
    return {
        "v": [pbits(v.position) for v in mesh.vertices],
        "e": [[vi[id(e.v0)], vi[id(e.v1)], 1 if e.soft else 0, 1 if e.hidden else 0,
               e.layer, e.curve, [fi[id(f)] for f in e.faces]] for e in mesh.edges],
        "f": [[[vi[id(v)] for v in f.loop],
               [[vi[id(v)] for v in h] for h in f.hole_loops],
               1 if getattr(f, "interior", False) else 0,
               sha(_dumps(canonical(f.attrs)), 16) if f.attrs else ""]
              for f in mesh.faces],
    }


def group_state(g) -> dict:
    return {
        "name": g.name, "uid": g.uid, "hidden": 1 if g.hidden else 0, "locked": 1 if g.locked else 0,
        "xform": None if g.xform is None else mbits(g.xform),
        "axes": None if getattr(g, "axes", None) is None else mbits(g.axes),
        "mesh": mesh_state(g.mesh),
        "kids": [group_state(c) for c in g.children],
    }


def guide_state(g) -> list:
    return [pbits(g.point), None if g.direction is None else pbits(g.direction),
            None if g.origin is None else pbits(g.origin)]


def scene_state(vp: Viewport, ids: Ids) -> dict:
    sc = vp.scene
    return {
        "m": mesh_state(sc.mesh),
        "g": [group_state(g) for g in sc.groups],
        "guides": [guide_state(g) for g in sc.guides],
        "sel": [ids.ref(e) for e in sc.selection],
        "edit": sc.edit_group is not None,
    }


def enc(x, ids: Ids | None = None):
    """JSON-able form of whatever a probe returns."""
    if x is None or isinstance(x, (bool, int, str)):
        return x
    if isinstance(x, float):
        return x
    if isinstance(x, QVector3D):
        return pbits(x)
    if isinstance(x, QMatrix4x4):
        return mbits(x)
    if isinstance(x, (set, frozenset)):
        return sorted(json.dumps(enc(i, ids), sort_keys=True) for i in x)
    if isinstance(x, (list, tuple)):
        return [enc(i, ids) for i in x]
    if isinstance(x, dict):
        return {str(k): enc(v, ids) for k, v in x.items()}
    if ids is not None:
        return ids.ref(x)
    raise TypeError(type(x))


# ---- recording what the tool is handed ----------------------------------------------------------------

TOOL_ENTRY = ("on_activate", "on_deactivate", "on_click", "on_hover", "on_double_click",
              "on_triple_click", "on_cancel", "on_release", "on_key", "on_key_release",
              "on_value", "on_array_value", "on_box_select", "close_array_window")


def rec_ctx(ctx) -> dict:
    snap = ctx.snap
    out = {"world": pbits(ctx.world), "screen": [ctx.screen.x(), ctx.screen.y()],
           "mods": mods_int(ctx.modifiers), "snap": None}
    if snap is not None:
        out["snap"] = {"kind": snap.kind, "point": pbits(snap.point),
                       "axis": getattr(snap, "axis", None), "context": getattr(snap, "context", None)}
    return out


def rec_value(v):
    """A typed value: a number, a tuple of numbers, ``("abs_len", v)``."""
    if isinstance(v, tuple):
        if v and v[0] == "abs_len":
            return {"abs_len": rec_value(v[1])}
        return {"tuple": [rec_value(i) for i in v]}
    if isinstance(v, bool):
        return {"bool": v}
    return {"num": float(v)}


class Recorder:
    """Wraps the tool's entry points and the viewport's hooks; the outermost
    call only is logged (a tool calling its own ``on_value`` is the tool's
    business)."""

    def __init__(self, vp: Viewport, tool, ids_fn) -> None:
        self.vp, self.tool, self.ids_fn = vp, tool, ids_fn
        self.calls: list = []
        self.flashes: list = []
        self.gfx: list = []
        self.depth = 0
        for name in TOOL_ENTRY:
            if hasattr(tool, name):
                setattr(tool, name, self._wrap(name, getattr(tool, name)))
        # the viewport's hooks that a tool drives
        vp.flash_status = self._flash
        vp.set_hover = self._set_hover
        vp.begin_groups_preview = self._begin_preview
        vp.set_groups_preview_offset = self._preview_offset
        vp.set_groups_preview_matrix = self._preview_matrix
        vp.end_groups_preview = self._end_preview
        sc = vp.scene
        orig_clear, orig_select = sc.clear_selection, sc.select

        def clear_selection():
            if self.depth == 0:
                self.calls.append({"m": "vp_clear_selection"})
            return orig_clear()

        def select(edges, additive=False, mode=None):
            edges = list(edges)
            if self.depth == 0:
                self.calls.append({"m": "vp_select", "ents": [self.ids_fn().ref(e) for e in edges],
                                   "mode": mode or ("add" if additive else "replace")})
            return orig_select(edges, additive, mode)
        sc.clear_selection = clear_selection
        sc.select = select

    def _wrap(self, name, orig):
        @functools.wraps(orig)
        def wrapper(*a, **k):
            top = self.depth == 0
            args = None
            if top:
                args = self._args(name, a, k)
            self.depth += 1
            try:
                r = orig(*a, **k)
            finally:
                self.depth -= 1
            if top:
                rec = {"m": name, **args}
                if isinstance(r, bool):
                    rec["ret"] = r
                self.calls.append(rec)
            return r
        return wrapper

    def _args(self, name, a, k) -> dict:
        vp = self.vp
        if name in ("on_click", "on_hover", "on_double_click", "on_triple_click"):
            return {"ctx": rec_ctx(a[0])}
        if name == "on_key":
            return {"key": int(getattr(a[1], "value", a[1])), "mods": mods_int(a[2])}
        if name == "on_key_release":
            return {"key": int(getattr(a[1], "value", a[1])), "tapped": bool(vp.ctrl_tapped())}
        if name == "on_value":
            return {"value": rec_value(a[1])}
        if name == "on_array_value":
            step = k.get("step", a[3] if len(a) > 3 else None)
            return {"count": int(a[1]), "mode": a[2], "step": step}
        if name == "on_box_select":
            return {"rect": list(a[1]), "crossing": bool(a[2])}
        return {}

    # -- viewport hooks
    def _flash(self, text, msec=2500):
        self.flashes.append([str(text), int(msec)])
        if self.depth == 0:
            # the viewport itself speaking (e.g. a VCB form the tool does not take)
            self.calls.append({"m": "vp_flash", "text": str(text), "msec": int(msec)})

    def _set_hover(self, entity, placement=None):
        ids = self.ids_fn()
        self.gfx.append(["hover", ids.ref(entity), None if placement is None else ids.ref(placement)])

    def _begin_preview(self, groups=(), external=False):
        ids = self.ids_fn()
        self.gfx.append(["begin", [ids.ref(g) for g in groups], bool(external)])

    def _preview_offset(self, delta):
        self.gfx.append(["offset", pbits(delta)])

    def _preview_matrix(self, m):
        self.gfx.append(["matrix", mbits(m)])

    def _end_preview(self):
        self.gfx.append(["end"])

    def take(self):
        calls, flashes, gfx = self.calls, self.flashes, self.gfx
        self.calls, self.flashes, self.gfx = [], [], []
        return calls, flashes, gfx


# ---- the generic probe ----------------------------------------------------------------------------------

def _try(fn, default=None):
    try:
        return fn()
    except Exception:  # noqa: BLE001 - the probe only reports what a tool offers
        return default


def generic_probe(tool, vp, ids: Ids) -> dict:
    """What every tool offers the viewport to draw and to word."""
    out: dict = {}
    if hasattr(tool, "rubber_band_lines"):
        out["rb"] = enc([[a, b] for a, b in tool.rubber_band_lines()])
    if hasattr(tool, "guide_preview_lines"):
        out["gl"] = enc([[a, b] for a, b in tool.guide_preview_lines()])
    if hasattr(tool, "value_label"):
        lab = tool.value_label()
        out["vl"] = None if lab is None else [lab[0], enc(lab[1])]
    if hasattr(tool, "status_clause"):
        out["sc"] = tool.status_clause()
    if hasattr(tool, "cursor_plus"):
        out["plus"] = bool(tool.cursor_plus)
    wf = getattr(tool, "wireframe_color", None)
    out["wf"] = None if wf is None else [float(c) for c in wf]
    if hasattr(tool, "vcb_caption"):
        out["cap"] = tool.vcb_caption()
    if hasattr(tool, "value_is_unitless"):
        out["unitless"] = bool(tool.value_is_unitless())
    out["uses_snap"] = bool(tool.uses_snap)
    if hasattr(tool, "snap_excluded"):
        ex = tool.snap_excluded()
        if ex is None:
            out["excl"] = None
        else:
            edges, groups = ex
            ev = {i: k for k, i in ids.edge.items()}
            out["excl"] = [sorted(ids.edge[i] for i in edges if i in ids.edge),
                           sorted(json.dumps(ids.gpath.get(i)) for i in groups)]
    if hasattr(tool, "array_pending"):
        out["ap"] = bool(tool.array_pending(vp))
    sp = getattr(tool, "start_point", None)
    out["start"] = None if sp is None else pbits(sp)
    hp = getattr(tool, "hover_point", None)
    out["hover"] = None if hp is None else pbits(hp)
    return out


# ---- the runner --------------------------------------------------------------------------------------------

def hover(where, **mods):
    return {"e": "hover", "where": where, **mods}


def press(where, **mods):
    return {"e": "press", "where": where, **mods}


def release(where, **mods):
    return {"e": "release", "where": where, **mods}


def click(where, **mods):
    """A press and its release (the usual click)."""
    return {"e": "click", "where": where, **mods}


def dbl(where, **mods):
    return {"e": "dbl", "where": where, **mods}


def key(name, **mods):
    return {"e": "key", "key": name, **mods}


def keyup(name, **mods):
    return {"e": "keyup", "key": name, **mods}


def tap(name):
    """A key pressed and released alone (a Ctrl tap)."""
    return {"e": "tap", "key": name}


def typed(text):
    """Digits/letters through the keyboard, then Enter."""
    return {"e": "type", "text": text}


def esc():
    return {"e": "esc"}


def select(*refs, mode="replace"):
    return {"e": "select", "refs": [list(r) for r in refs], "mode": mode}


def undo():
    return {"e": "undo"}


def redo():
    return {"e": "redo"}


def activate():
    return {"e": "activate"}


def deactivate():
    return {"e": "deactivate"}


def camera(spec: dict):
    return {"e": "camera", "cam": spec}


def at(p, dx=0.0, dy=0.0):
    """A world point plus a pixel offset."""
    return ("at", p, dx, dy)


def px(x, y):
    return ("px", x, y)


def above(p, dz):
    return (p[0], p[1], p[2] + dz)


class UidSource:
    """``core.group.new_uid`` with a counter that can be put back."""

    def __init__(self) -> None:
        self.n = 1

    def __call__(self) -> str:
        v = self.n
        self.n += 1
        return "%016x" % v


class Session:
    """One program: a viewport on a reopened scene, a tool, a camera."""

    def __init__(self, scene_path: Path, cam: dict, make_tool, extra_probe=None) -> None:
        ORD.clear()
        KEEP.clear()
        self.vp = vp = Viewport(None)
        self.W, self.H = cam.get("w", W_DEFAULT), cam.get("h", H_DEFAULT)
        vp.resize(self.W, self.H)
        igz.load_into(vp.scene, scene_path)
        # Counters start where the Rust side's do, AFTER the document is
        # loaded (building a Group draws a uid, the file then overwrites it).
        core_group._counter = itertools.count(1)
        self.uids = UidSource()
        core_group.new_uid = self.uids
        # The viewport's placement proxies are Groups too, made when a pick
        # first needs them: they do not draw from the document's uids.
        expand = vp._expand_placements

        def quiet(*a, **k):
            n = self.uids.n
            try:
                return expand(*a, **k)
            finally:
                self.uids.n = n
        vp._expand_placements = quiet
        cm._CURVE_COUNTER = 5000
        vp.scene.selection = OrderedSelection()
        vp.scene.version += 1
        self.set_camera(cam)
        self.tool = make_tool()
        self.rec = Recorder(vp, self.tool, lambda: Ids(vp))
        self.extra_probe = extra_probe
        self.cam_spec = cam
        self.log: list = []
        self.depth_before = 0
        # the tool picked up
        self.ids = Ids(vp)
        self.rec.calls.clear()
        vp.set_active_tool(self.tool)

    def set_camera(self, cam: dict) -> None:
        vp = self.vp
        c = vp.camera
        c.yaw = math.radians(cam["yaw"])
        c.pitch = math.radians(cam["pitch"])
        c.distance = cam["distance"]
        c.target = V(*cam["target"])
        c.perspective = cam.get("perspective", True)
        if "fov" in cam:
            c.fov_deg = cam["fov"]
        if "w" in cam:
            self.W, self.H = cam["w"], cam["h"]
            vp.resize(self.W, self.H)
        c.set_aspect(self.W, self.H)
        vp.scene.version += 1

    def pixel(self, where):
        dx = dy = 0.0
        if where[0] == "px":
            return float(where[1]), float(where[2])
        if where[0] == "at":
            _, where, dx, dy = where
        p = self.vp._world_to_pixel(V(*where))
        assert p is not None, f"{where} behind the camera"
        return p[0] + dx, p[1] + dy

    def mouse(self, kind, x, y, mods, drag=False):
        t = {"move": QEvent.MouseMove, "press": QEvent.MouseButtonPress,
             "release": QEvent.MouseButtonRelease, "dbl": QEvent.MouseButtonDblClick}[kind]
        button = Qt.NoButton if kind == "move" else Qt.LeftButton
        buttons = Qt.LeftButton if (kind in ("press", "dbl") or drag) else Qt.NoButton
        return QMouseEvent(t, QPointF(x, y), QPointF(x, y), button, buttons, mods)

    def key_event(self, kind, k, mods, text=""):
        t = QEvent.KeyPress if kind == "press" else QEvent.KeyRelease
        return QKeyEvent(t, k, mods, text)

    def run_event(self, ev: dict) -> dict:
        vp = self.vp
        kind = ev["e"]
        out: dict = {"e": kind}
        mods = qt_mods(ev.get("shift", False), ev.get("ctrl", False), ev.get("alt", False))
        for k in ("shift", "ctrl", "alt", "drag"):
            if ev.get(k):
                out[k] = True
        self.ids = Ids(vp)
        before = len(vp.history.undo_stack)
        if "where" in ev:
            x, y = self.pixel(ev["where"])
            out["x"], out["y"] = x, y
        drag = ev.get("drag", False)
        if kind == "hover":
            # The viewport coalesces hovers by the measured cost of the last
            # one (a deferred hover waits for a timer that never fires here):
            # a gate of zero makes every hover run, whatever the machine does.
            vp._hover_cost = 0.0
            vp.mouseMoveEvent(self.mouse("move", x, y, mods, drag))
        elif kind == "press":
            vp.mousePressEvent(self.mouse("press", x, y, mods))
        elif kind == "release":
            vp.mouseReleaseEvent(self.mouse("release", x, y, mods))
        elif kind == "click":
            vp.mousePressEvent(self.mouse("press", x, y, mods))
            vp.mouseReleaseEvent(self.mouse("release", x, y, mods))
        elif kind == "dbl":
            vp.mouseDoubleClickEvent(self.mouse("dbl", x, y, mods))
        elif kind == "key":
            out["key"] = ev["key"]
            vp.keyPressEvent(self.key_event("press", KEYS[ev["key"]], mods))
        elif kind == "keyup":
            out["key"] = ev["key"]
            vp.keyReleaseEvent(self.key_event("release", KEYS[ev["key"]], mods))
        elif kind == "tap":
            out["key"] = ev["key"]
            m = {"ctrl": Qt.ControlModifier, "shift": Qt.ShiftModifier, "alt": Qt.AltModifier}.get(ev["key"], Qt.NoModifier)
            vp.keyPressEvent(self.key_event("press", KEYS[ev["key"]], m))
            vp.keyReleaseEvent(self.key_event("release", KEYS[ev["key"]], Qt.NoModifier))
        elif kind == "type":
            out["text"] = ev["text"]
            for ch in ev["text"]:
                code = ord(ch.upper()) if ch.isalnum() else {" ": Qt.Key_Space, ";": Qt.Key_Semicolon, ",": Qt.Key_Comma,
                                                             ".": Qt.Key_Period, "-": Qt.Key_Minus, ":": Qt.Key_Colon,
                                                             "/": Qt.Key_Slash, "*": Qt.Key_Asterisk,
                                                             "'": Qt.Key_Apostrophe, '"': Qt.Key_QuoteDbl}[ch]
                vp.keyPressEvent(self.key_event("press", code, Qt.NoModifier, ch))
            vp.keyPressEvent(self.key_event("press", Qt.Key_Return, Qt.NoModifier, "\r"))
        elif kind == "esc":
            vp.keyPressEvent(self.key_event("press", Qt.Key_Escape, Qt.NoModifier))
        elif kind == "select":
            ents = [self.ids.resolve(r) for r in ev["refs"]]
            out["refs"] = ev["refs"]
            out["mode"] = ev["mode"]
            vp.scene.select(ents, mode=ev["mode"])
        elif kind == "undo":
            vp.history.undo()
        elif kind == "redo":
            vp.history.redo()
        elif kind == "activate":
            vp.set_active_tool(self.tool)
        elif kind == "deactivate":
            vp.set_active_tool(None)
        elif kind == "camera":
            self.set_camera(ev["cam"])
            out["cam"] = ev["cam"]
        else:
            raise ValueError(kind)
        calls, flashes, gfx = self.rec.take()
        # the harness's own selection edits are not tool calls
        self.ids = Ids(vp)
        out["calls"] = [c for c in calls if not (kind == "select" and c["m"] == "vp_select")]
        out["flash"] = flashes
        out["gfx"] = gfx
        out["lv"] = None if vp.history.last_error is None else str(vp.history.last_error)
        out["rejected"] = getattr(vp.history, "last_rejected", None)
        out["depth"] = len(vp.history.undo_stack)
        out["redo"] = len(vp.history.redo_stack)
        out["state"] = scene_state(vp, self.ids)
        probe = generic_probe(self.tool, vp, self.ids)
        if self.extra_probe is not None:
            probe["x"] = enc(self.extra_probe(self.tool, vp, self.ids), self.ids)
        out["probe"] = probe
        out["axis_lock"] = vp.axis_lock
        return out


def run_program(scene_path: Path, cam: dict, make_tool, events: list, extra_probe=None, setup=None, n=None) -> dict:
    s = Session(scene_path, cam, make_tool, extra_probe)
    rec = {"cam": cam, "pre": scene_state(s.vp, s.ids), "events": []}
    if setup is not None:
        # a module's per-program extras for the Rust side (text metrics, the
        # initial state of class-level tool state...)
        rec["setup"] = setup(s, n)
    # a tool's own on_activate during set_active_tool
    calls, flashes, gfx = s.rec.take()
    rec["activate"] = {"calls": calls, "flash": flashes, "gfx": gfx,
                       "probe": generic_probe(s.tool, s.vp, s.ids),
                       "state": scene_state(s.vp, s.ids)}
    if extra_probe is not None:
        rec["activate"]["probe"]["x"] = enc(extra_probe(s.tool, s.vp, s.ids), s.ids)
    for ev in events:
        out = s.run_event(ev)
        # Python keeps an entity in the selection after a command made it
        # vanish without saying so (a split edge...); the Rust selection drops
        # what no longer exists. A program is cut where the two would differ.
        # The same for a tool holding an object that is gone (its probe `x`
        # says `stale`): Rust names groups by path and could not follow it.
        if (any(r[0] == "unknown" for r in out["state"]["sel"])
                or (out["probe"].get("x") or {}).get("stale")):
            rec["truncated"] = True
            break
        rec["events"].append(out)
    return rec
