# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""Write the fixtures that hold the Rust ``ingetrazo-snap`` engine to the
Python one (docs/rust-rewrite-plan.md, M3.11).

The REAL ``Viewport`` runs headless (offscreen Qt, a real ``MainWindow``, as
``scripts/probe_snap_matrix.py`` does) over varied scenes — loose faces and
edges, walls, a box, construction guides, near-coincident corners, a circle
and an arc, classic groups, a turned component, a nested placement, and a
turned group open for editing — seen from presets and oblique cameras
(perspective and parallel), with every tool state the engine reads: start
point, chain's first point, arrow-key and Shift locks, the Down-arrow
reference edge, acquired point / edge / face normal, two encouraged points,
the Alt mode, a captured work plane, a hovered circle centre.

For each sample ``Viewport.inference.snap`` answers; while it does, the
script records

* what ``compute_snap`` was called with (every argument, the snap scene's
  edges in a pool), what it returned, and what the engine finally returned
  after the axis-source cue;
* the cursor's world point (``_world_from_pixel``), the work plane's normal
  and whether a face is under the cursor;
* every occlusion question the engine asked and the answer (so the rules can
  be replayed without the occluder, and the occluder checked against the
  answers on its own);
* the ScreenTip's text, English and Spanish.

Alongside go the camera's matrices and pixel functions (``_pixel_to_ray``,
``_world_to_pixel``, ``_project_to_lock_line``, ``_clip_segment_front``,
``_guide_snap_segment``) and each scene as a bare-JSON .igz.

Numbers: a QVector3D is stored as three float32 bit patterns (integers), a
Python float as a JSON number (repr round-trips).

Python's own order is not repeatable in two places, and neither reaches the
fixtures: ``Viewport._camera_planar_work_plane`` keeps its candidate planes in
a ``set`` of one-letter strings and takes ``max`` over them, so an exact tie
would depend on the string-hash seed - the rule's tie margin (0.05) decides
every tie that can occur with both candidates present, and one candidate
leaves nothing to tie; and ``np.argsort`` (not stable) orders the group edges
near the cursor only past 48 of them, which no scene here reaches.

Committed; rerun only to widen (PYTHONPATH at the repository root, the main
checkout's venv, ``INGETRAZO_RUST`` unset):

    python scripts/golden/make_snap_fixtures.py

Deleted with the Python app at M10.
"""
from __future__ import annotations

import json
import math
import os
import random
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.pop("INGETRAZO_RUST", None)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from core.vecmath import QMatrix4x4, QVector3D as V  # noqa: E402

# Group identities are random; the committed scenes must not change on a rerun.
import core.group as _core_group  # noqa: E402
import itertools as _itertools  # noqa: E402

_uid_counter = _itertools.count(1)
_core_group.new_uid = lambda: "%016x" % next(_uid_counter)

OUT = ROOT / "rust" / "crates" / "ingetrazo-snap" / "tests" / "data"


# ---- number encodings ---------------------------------------------------------------

def fb(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def b3(v):
    return None if v is None else [fb(v.x()), fb(v.y()), fb(v.z())]


def b3t(t):
    return [fb(t[0]), fb(t[1]), fb(t[2])]


def fl(x):
    return None if x is None else float(x)


def mat_bits(m):
    return [fb(x) for x in m.data()]


# ---- scenes -------------------------------------------------------------------------

def box(mesh, lo, hi):
    """Six faces of an axis-aligned box."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    P = V
    for quad in (
        [(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
        [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)],
        [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)],
    ):
        mesh.add_face([P(*p) for p in quad])


def circle_pts(c, r, n, a0=0.0, a1=2 * math.pi, closed=True):
    pts = []
    steps = n if closed else n + 1
    for i in range(steps):
        t = a0 + (a1 - a0) * i / n
        pts.append(V(c[0] + r * math.cos(t), c[1] + r * math.sin(t), c[2]))
    return pts


def turn(rz=30.0, rx=20.0, t=(0.0, 0.0, 0.0)):
    m = QMatrix4x4()
    m.translate(*t)
    m.rotate(rz, 0.0, 0.0, 1.0)
    m.rotate(rx, 1.0, 0.0, 0.0)
    return m


_SCENE_SERIAL = [0]


def build_scene(vp, name):
    """Replace the viewport's scene with ``name``'s; returns the edit target
    (a group to open for editing after the document is saved) or None."""
    from core.group import Group
    from core.mesh import Mesh
    from core.scene import Scene
    if getattr(vp.scene, "edit_group", None) is not None:
        while vp.scene.edit_group is not None:
            vp.end_group_edit()
    vp.scene = Scene()
    sc = vp.scene
    m = sc.mesh
    edit = None
    if name == "corner":
        m.add_face([V(1.5, 0.0, 0.0), V(2.5, 0.0, 0.0), V(2.5, 1.0, 0.0), V(1.5, 1.0, 0.0)])
    elif name == "midline":
        m.add_edge(V(2.0, 1.0, 0.0), V(6.0, 1.0, 0.0))
        m.add_edge(V(2.0, 1.0, 0.0), V(2.0, 1.0, 2.5))
    elif name == "walls":
        P = V
        m.add_face([P(-1.5, -1.5, 0.0), P(1.5, -1.5, 0.0), P(1.5, 1.5, 0.0), P(-1.5, 1.5, 0.0)])
        m.add_face([P(-1.5, -1.5, 0.0), P(1.5, -1.5, 0.0), P(1.5, -1.5, 3.0), P(-1.5, -1.5, 3.0)])
        m.add_face([P(1.5, -1.5, 0.0), P(1.5, 1.5, 0.0), P(1.5, 1.5, 3.0), P(1.5, -1.5, 3.0)])
        m.add_face([P(1.5, 1.5, 0.0), P(-1.5, 1.5, 0.0), P(-1.5, 1.5, 3.0), P(1.5, 1.5, 3.0)])
    elif name == "box":
        box(m, (2.0, -1.0, 0.0), (4.0, 1.0, 2.0))
        m.add_edge(V(-3.0, -2.0, 0.0), V(-1.0, 2.0, 0.0))
        m.add_edge(V(-3.0, 2.0, 0.0), V(-1.0, -2.0, 0.0))
        m.add_edge(V(-2.0, 0.0, 0.0), V(-2.0, 0.0, 3.0))
    elif name == "guides":
        from core.guide import Guide
        sc.guides.append(Guide(V(0.0, 2.0, 0.0), V(1.0, 0.0, 0.0)))
        sc.guides.append(Guide(V(1.0, 0.0, 0.0), V(1.0, 1.0, 0.0).normalized()))
        sc.guides.append(Guide(V(-2.0, -1.0, 0.0), V(0.0, 0.0, 1.0)))
        sc.guides.append(Guide(V(3.0, 3.0, 0.0), None))
        m.add_edge(V(-1.0, -1.0, 0.0), V(2.0, 3.0, 0.0))
    elif name == "tight":
        # Near-coincident and tie-prone: two corners a hair apart (past the
        # weld's reach), a T-junction, an X crossing, collinear edges.
        m.add_edge(V(0.0, 0.0, 0.0), V(2.0, 0.0, 0.0))
        m.add_edge(V(2.0003, 0.0, 0.0), V(4.0, 0.0, 0.0))
        m.add_edge(V(1.0, -1.0, 0.0), V(1.0, 0.0, 0.0))
        m.add_edge(V(3.0, -1.0, 0.0), V(3.0, 1.0, 0.0))
        m.add_edge(V(0.0, 1.0, 0.0), V(2.0, 3.0, 0.0))
        m.add_edge(V(0.0, 3.0, 0.0), V(2.0, 1.0, 0.0))
        m.add_edge(V(-2.0, 2.0, 0.0), V(-1.0, 2.0, 0.0))
        m.add_edge(V(-1.0, 2.0005, 0.0), V(0.0, 2.0005, 0.0))
        m.add_face([V(5.0, 0.0, 0.0), V(6.0, 0.0, 0.0), V(6.0, 1.0, 0.0), V(5.0, 1.0, 0.0)])
        m.add_face([V(6.0, 0.0, 0.0), V(7.0, 0.0, 0.0), V(7.0, 1.0, 0.0), V(6.0, 1.0, 0.0)])
    elif name == "arcs":
        pts = circle_pts((3.0, 0.0, 0.0), 1.5, 24)
        for a, b in zip(pts, pts[1:] + pts[:1]):
            m.add_edge(a, b)
        m.tag_curve(pts, closed=True)
        pts = circle_pts((-3.0, 1.0, 0.0), 2.0, 12, 0.0, math.pi, closed=False)
        for a, b in zip(pts, pts[1:]):
            m.add_edge(a, b)
        m.tag_curve(pts, closed=False)
        m.add_edge(V(-1.0, -1.0, 0.0), V(-1.0, -1.0, 2.0))
    elif name == "group":
        g = Group(Mesh(), name="Group A")
        box(g.mesh, (2.0, -1.0, 0.0), (3.5, 0.5, 1.5))
        g.mesh.add_edge(V(-3.0, -1.0, 0.0), V(-3.0, 2.0, 0.0))
        sc.groups.append(g)
        m.add_edge(V(0.0, -2.0, 0.0), V(0.0, 2.0, 0.0))
        m.add_face([V(-2.0, -3.0, 0.0), V(-1.0, -3.0, 0.0), V(-1.0, -2.0, 0.0), V(-2.0, -2.0, 0.0)])
    elif name == "comp":
        proto = Mesh()
        box(proto, (-0.5, -0.5, 0.0), (0.5, 0.5, 1.0))
        proto.add_edge(V(0.5, 0.0, 0.5), V(1.5, 0.0, 0.5))
        for i, (rz, t) in enumerate(((30.0, (3.0, 0.0, 0.0)), (0.0, (-3.0, 1.0, 0.0)))):
            g = Group(proto, name=f"Component {i}")
            g.xform = turn(rz, 0.0, t)
            sc.groups.append(g)
        m.add_edge(V(0.0, -2.0, 0.0), V(0.0, 2.0, 0.0))
    elif name == "nested":
        proto = Mesh()
        box(proto, (-0.4, -0.4, 0.0), (0.4, 0.4, 0.8))
        outer = Group(Mesh(), name="Container")
        outer.mesh.add_edge(V(-1.0, -1.0, 0.0), V(2.0, -1.0, 0.0))
        kids = []
        for i, t in enumerate(((0.0, 0.0, 0.0), (1.5, 0.0, 0.0))):
            k = Group(proto, name=f"Kid {i}")
            k.xform = turn(15.0 * i, 0.0, t)
            kids.append(k)
        outer.adopt(kids)
        sc.groups.append(outer)
        m.add_edge(V(2.0, 2.0, 0.0), V(2.0, 2.0, 2.0))
    elif name == "edit-group":
        edit = _turned_edit(sc, "group")
    elif name == "edit-comp":
        edit = _turned_edit(sc, "comp")
    elif name != "empty":
        raise ValueError(name)
    sc.version += 1
    # A new document in the app starts the viewport's caches over; here the
    # version is the only thing that tells scenes apart, so it never repeats.
    _SCENE_SERIAL[0] += 1
    sc.version = 1000 * _SCENE_SERIAL[0] + sc.version
    for attr in _VIEWPORT_CACHES:
        if attr in vars(vp):
            setattr(vp, attr, None)
    return edit


#: The viewport's scene-bound caches, emptied when the scene is swapped.
_VIEWPORT_CACHES = (
    "_arc_mid_cache", "_arc_mid_by_mesh", "_corigin_cache", "_group_edge_tables",
    "_gedge_px_cache", "_gedge_dist_cache", "_gedge_grid_cache", "_ledge_px_cache",
    "_pick_index_cache", "_pick_block", "_pick_fixed_part", "_pick_static_memo",
    "_group_chunks", "_inst_chunks", "_container_obb", "_hover_hits_cache",
    "_face_any_memo", "_placements_memo", "_placement_proxies", "_tri_memo",
    "_newell_memo", "_bounds_cache", "_pick_last_q", "_vertex_px_cache",
)


def _turned_edit(sc, kind):
    from core.group import Group
    from core.mesh import Mesh
    local = Mesh()
    local.add_face([V(-1.0, -1.0, 0.0), V(1.0, -1.0, 0.0), V(1.0, 1.0, 0.0), V(-1.0, 1.0, 0.0)])
    local.add_edge(V(1.5, 0.0, 0.0), V(3.0, 0.0, 0.0))
    turn_m = turn(30.0, 20.0)
    if kind == "comp":
        g = Group(local, name="turned component")
        g.xform = turn_m
        sc.groups.append(g)
        return g
    from core.history import RotateGroupCommand
    g = Group(local, name="turned group")
    sc.groups.append(g)
    return ("rotate", g)


SCENES = ["empty", "corner", "midline", "walls", "box", "guides", "tight", "arcs",
          "group", "comp", "nested", "edit-group", "edit-comp"]

#: Notable world points per scene: where states put their start / acquired
#: points and where the pixel samples cluster.
POIS = {
    "empty": [(0, 0, 0), (2, 1, 0), (0, 0, 1.5), (-2, -2, 0)],
    "corner": [(0, 0, 0), (1.5, 0, 0), (2.0, 0, 0), (2.5, 1, 0), (2.0, 1.0, 0)],
    "midline": [(0, 0, 0), (2, 1, 0), (4, 1, 0), (6, 1, 0), (2, 1, 1.2)],
    "walls": [(0, 0, 0), (1.5, -1.5, 3), (0.5, -1.5, 1.2), (1.5, 0, 1.0), (0, 1.5, 1.5)],
    "box": [(0, 0, 0), (2, -1, 0), (4, 1, 2), (3, 0, 2), (-2, 0, 0), (-2, 0, 1.5)],
    "guides": [(0, 0, 0), (0, 2, 0), (1, 1, 0), (3, 3, 0), (-2, -1, 1), (2, 3, 0)],
    "tight": [(0, 0, 0), (2, 0, 0), (2.0003, 0, 0), (1, 0, 0), (3, 0, 0), (1, 2, 0), (-1, 2, 0)],
    "arcs": [(0, 0, 0), (3, 0, 0), (4.5, 0, 0), (-3, 1, 0), (-3, 3, 0), (-1, -1, 1)],
    "group": [(0, 0, 0), (2, -1, 0), (3.5, 0.5, 1.5), (0, 0, 0), (-3, -1, 0)],
    "comp": [(0, 0, 0), (3, 0, 0), (3, 0, 1), (-3, 1, 0), (-2.5, 1, 0.5)],
    "nested": [(0, 0, 0), (0, 0, 0.8), (1.5, 0, 0), (-1, -1, 0), (2, 2, 2)],
    "edit-group": [(0, 0, 0), (1.5, 0, 0), (0.5, 0.5, 0)],
    "edit-comp": [(0, 0, 0), (1.5, 0, 0), (0.5, 0.5, 0)],
}

#: (name, yaw degrees, pitch degrees, distance, perspective, target)
CAMERAS = [
    ("top", None, None, 14.0, True, (0, 0, 0)),
    ("front", None, None, 14.0, True, (0, 0, 0)),
    ("right", None, None, 14.0, False, (0, 0, 0)),
    ("iso", None, None, 16.0, True, (0, 0, 0)),
    ("obl-a", -30.0, 12.0, 14.0, True, (0.5, 0, 0.5)),
    ("obl-b", -75.0, 55.0, 12.0, True, (1, 0, 0)),
    ("obl-c", 20.0, -20.0, 14.0, True, (0, 0, 0)),
    ("obl-d-ortho", 160.0, 40.0, 14.0, False, (0, 0, 0)),
    ("close", -40.0, 30.0, 4.0, True, (1.0, 0.5, 0.5)),
]

TOOLS = ["line", "tape", "rectangle", "circle", "move", "rotate", "select"]
#: The tools whose hover is acquired (PushPull hovers a group's edges though it does not snap).
ACQ_TOOLS = ["line", "circle", "rectangle", "tape", "move", "pushpull", "select"]
ALT_MODES = ["all", "all", "all", "off", "parallel_perp"]


# ---- recording ---------------------------------------------------------------------

EDGE_FLAGS = (("soft", 1), ("hidden", 2), ("figure", 4), ("center", 8),
              ("component_origin", 16), ("arc_midpoint", 32), ("guide", 64))


def edge_row(e):
    flags = 0
    for attr, bit in EDGE_FLAGS:
        if getattr(e, attr, False):
            flags |= bit
    ctx = getattr(e, "context", None)
    if ctx == "group":
        flags |= 128
    elif ctx == "component":
        flags |= 256
    if getattr(e, "curve", None) is not None:
        flags |= 512
    return [*b3(e.a), *b3(e.b), flags]


def result_row(r):
    if r is None:
        return None
    out = {"k": r.kind, "p": b3(r.point), "c": [float(c) for c in r.color],
           "ax": r.axis}
    if r.guide is not None:
        out["g"] = [b3(r.guide[0]), b3(r.guide[1])]
    if r.guide_color is not None:
        out["gc"] = [float(c) for c in r.guide_color]
    if r.guides:
        out["gs"] = [[b3(a), b3(b), [float(c) for c in col]] for a, b, col in r.guides]
    if r.context is not None:
        out["cx"] = r.context
    if getattr(r, "label", None):
        out["lb"] = r.label
    return out


class Rec:
    """What one ``engine.snap`` call did, collected by spies."""

    def __init__(self):
        self.pool: list = []
        self.pool_ix: dict = {}
        self.reset()

    def reset(self):
        self.occ: dict = {}
        self.args = None
        self.cs_result = None
        self.world_raw = "unset"
        self.wpn = None
        self.fu = None

    def pool_edges(self, edges):
        rows = [edge_row(e) for e in edges]
        key = json.dumps(rows, separators=(",", ":"))
        ix = self.pool_ix.get(key)
        if ix is None:
            ix = self.pool_ix[key] = len(self.pool)
            self.pool.append(rows)
        return ix


def label_of(vp, snap, lang):
    """The ScreenTip text (``_draw_snap_indicator``), in ``lang``."""
    from core.i18n import set_language, tr
    set_language(lang)
    label = vp._SNAP_LABELS.get(snap.kind)
    if snap.kind == "aligned" and snap.axis == "z":
        label = "Level with point"
    if label:
        label = tr(label)
    if getattr(snap, "label", None):
        label = snap.label
    if label:
        ctx_ = getattr(snap, "context", None)
        if ctx_ == "component":
            label += " " + tr("in component")
        elif ctx_ == "group":
            label += " " + tr("in group")
    return label or None


def install_spies(vp, rec):
    import core.inference as ci
    orig_cs = ci.compute_snap

    def cs(**kw):
        a = {}
        a["cw"] = b3(kw["candidate_world"])
        a["cp"] = [float(kw["candidate_pixel"][0]), float(kw["candidate_pixel"][1])]
        a["ed"] = rec.pool_edges(kw["scene"].edges)
        a["th"] = float(kw["threshold_px"])
        cf, sp = kw["chain_first_point"], kw["start_point"]
        a["cf"] = b3(cf)
        a["cfs"] = cf is sp
        a["sp"] = b3(sp)
        a["al"] = kw["axis_lock"]
        a["sh"] = bool(kw["shift_held"])
        re = kw["reference_edge"]
        a["re"] = None if re is None else [b3(re.a), b3(re.b)]
        a["rm"] = kw["reference_mode"]
        a["ia"] = float(kw["inference_angle_deg"])
        a["sa"] = fl(kw["screen_axis_px"])
        a["fu"] = bool(kw["face_under_cursor"])
        a["et"] = fl(kw["edge_threshold_px"])
        a["ma"] = fl(kw["magnetic_axis_deg"])
        ae = kw["acquired_edge"]
        a["ae"] = None if ae is None else [b3(ae.a), b3(ae.b)]
        a["ap"] = b3(kw["acquired_point"])
        a["afn"] = b3(kw["acquired_face_normal"])
        a["aps"] = [b3(p) for p in (kw["acquired_points"] or [])]
        a["sld"] = b3(kw["shift_lock_dir"])
        a["slc"] = None if kw["shift_lock_color"] is None else [float(c) for c in kw["shift_lock_color"]]
        a["lm"] = kw["linear_mode"]
        a["wpn"] = b3(kw["work_plane_normal"])
        a["ra"] = bool(kw["radial_arm"])
        a["aip"] = bool(kw["axis_in_plane"])
        rec.args = a
        r = orig_cs(**kw)
        rec.cs_result = result_row(r)
        return r

    ci.compute_snap = cs

    orig_occ, orig_many = vp._is_occluded, vp._occluded_many

    def occ(w):
        r = orig_occ(w)
        rec.occ[tuple(b3(w))] = 1 if r else 0
        return r

    def many(ws):
        ws = list(ws)
        rs = orig_many(ws)
        for w, r in zip(ws, rs):
            rec.occ[tuple(b3(w))] = 1 if r else 0
        return rs

    vp._is_occluded = occ
    vp._occluded_many = many

    orig_wfp = vp._world_from_pixel

    def wfp(x, y):
        r = orig_wfp(x, y)
        rec.world_raw = b3(r)
        rec.wpn = b3(vp._work_plane_normal())
        return r

    vp._world_from_pixel = wfp
    orig_sfu = vp._still_face_under

    def sfu(x, y):
        r = orig_sfu(x, y)
        rec.fu = bool(r)
        return r

    vp._still_face_under = sfu


# ---- tool state -----------------------------------------------------------------------

def reset_vp(vp):
    tool = vp.active_tool
    for attr in ("start_point", "work_plane", "chain_first_point", "plane_lock", "plane_ref"):
        if hasattr(tool, attr):
            try:
                setattr(tool, attr, None)
            except AttributeError:
                pass
    vp._acquired_edge = None
    vp._acquired_point = None
    vp._acquired_face_normal = None
    vp._encouraged = []
    vp._dwell_point = None
    vp._shift_lock = None
    vp.axis_lock = None
    vp.reference_edge = None
    vp.reference_mode = None
    vp._center_ref = None
    vp.inference._hover_edge = None
    vp.linear_inference_mode = "all"


def q(p):
    return None if p is None else V(*p)


def apply_state(vp, win, spec, scene_edges):
    """Put the tool and the engine in ``spec``'s state; returns whether it
    was applicable."""
    reset_vp(vp)
    tool = vp.active_tool
    if spec["start"] is not None:
        if not hasattr(tool, "start_point"):
            return False
        tool.start_point = q(spec["start"])
    if spec["chain"] is not None and hasattr(tool, "chain_first_point"):
        tool.chain_first_point = q(spec["chain"]) if spec["chain"] != "same" else tool.start_point
    if spec["work_plane"] is not None:
        if not hasattr(tool, "work_plane"):
            return False
        tool.work_plane = (q(spec["work_plane"][0]), q(spec["work_plane"][1]))
    if spec["plane_lock"] is not None:
        if not hasattr(tool, "plane_lock"):
            return False
        tool.plane_lock = spec["plane_lock"]
    vp.axis_lock = spec["lock"]
    if spec["shift_lock"] is not None:
        from core.snap import COLOR_AXIS_X, COLOR_AXIS_Z
        d, c = ((V(1, 0, 0), COLOR_AXIS_X) if spec["shift_lock"] == "x"
                else (V(0, 0, 1), COLOR_AXIS_Z))
        vp._shift_lock = (d, c)
    if spec["ref"] is not None:
        mode, a, b = spec["ref"]
        vp.reference_edge = SimpleNamespace(a=q(a), b=q(b))
        vp.reference_mode = mode
    if spec["acq"] is not None:
        vp._acquired_point = q(spec["acq"])
    vp._encouraged = [q(p) for p in spec["enc"]]
    if spec["acq_edge"] is not None:
        a, b = spec["acq_edge"]
        vp._acquired_edge = SimpleNamespace(a=q(a), b=q(b))
    if spec["acq_fn"] is not None:
        vp._acquired_face_normal = q(spec["acq_fn"])
    vp.linear_inference_mode = spec["alt"]
    return True


def state_row(vp, win, tool_key, spec):
    from core.guide import GUIDE_HALF_LEN  # noqa: F401
    from tools.base import PlaneLock
    t = vp.active_tool
    wp = getattr(t, "work_plane", None)
    own = t.drag_plane(vp) if callable(getattr(t, "drag_plane", None)) else None
    sh = vp._shift_lock
    ref = vp.reference_edge
    row = {
        "tool": tool_key,
        "us": bool(t.uses_snap),
        "plc": isinstance(t, PlaneLock),
        "pv": bool(getattr(t, "prefers_vertical_drag", False)),
        "ra": bool(getattr(t, "radial_arm", False)),
        "mag": fl(getattr(t, "magnetic_axis_deg", None)),
        "scr": fl(getattr(t, "screen_axis_px", None)),
        "mop": bool(t.magnet_off_plane()) if hasattr(t, "magnet_off_plane") else True,
        "axs": bool(getattr(t, "axis_source", False)),
        "axe": bool(getattr(t, "axis_source_empty_only", False)),
        "hge": bool(getattr(t, "hover_group_edges", False)),
        "sp": b3(getattr(t, "start_point", None)),
        "cf": b3(getattr(t, "chain_first_point", None)),
        "cfs": getattr(t, "chain_first_point", None) is not None
               and getattr(t, "chain_first_point", None) is getattr(t, "start_point", None),
        "wp": None if wp is None else [b3(wp[0]), b3(wp[1])],
        "pl": getattr(t, "plane_lock", None),
        "pr": None if getattr(t, "plane_ref", None) is None else b3(t.plane_ref[0]),
        "own": None if own is None else [b3(own[0]), b3(own[1])],
        "lock": vp.axis_lock,
        "sl": None if sh is None else [b3(sh[0]), [float(c) for c in sh[1]]],
        "re": None if ref is None else [b3(ref.a), b3(ref.b)],
        "rm": vp.reference_mode,
        "ae": None if vp._acquired_edge is None else [b3(vp._acquired_edge.a), b3(vp._acquired_edge.b)],
        "ap": b3(vp._acquired_point),
        "afn": b3(vp._acquired_face_normal),
        "enc": [b3(p) for p in vp._encouraged],
        "lm": vp.linear_inference_mode,
        "ctr": None,
    }
    return row


def axes_row():
    from core import axes
    return {"x": b3(axes.axis("x")), "y": b3(axes.axis("y")), "z": b3(axes.axis("z")),
            "o": b3(axes.origin()), "w": bool(axes.is_world())}


def camera_row(vp):
    c = vp.camera
    return {"target": b3(c.target), "dist": c.distance, "yaw": c.yaw, "pitch": c.pitch,
            "up": b3(c.up), "fov": c.fov_deg, "aspect": c.aspect, "znear": c.znear,
            "zfar": c.zfar, "persp": bool(c.perspective), "two_point": bool(c.two_point),
            "w": vp.width(), "h": vp.height(),
            "view": mat_bits(c.view_matrix()), "proj": mat_bits(c.projection_matrix()),
            "eye": b3(c.eye())}


def set_camera(vp, app, cam):
    name, yaw, pitch, dist, persp, target = cam
    vp.camera.perspective = persp
    if yaw is None:
        vp.camera.set_view(name)
    else:
        vp.camera.yaw = math.radians(yaw)
        vp.camera.pitch = math.radians(pitch)
    vp.camera.distance = dist
    vp.camera.target = V(*target)
    # ``pick_vertex`` keys its concatenated projections on ``id()`` of arrays
    # it does not keep alive, so a new camera can be handed the last one's
    # answer when Python reuses an address; the app moves the camera far
    # less often than this script does, and the cache is dropped here.
    vp._vertex_px_cache = None
    app.processEvents()


# ---- state generation ------------------------------------------------------------------

def make_specs(rng, scene, tool_key, pois, ref_edges, n):
    """``n`` tool states for a scene: the canonical ones first, then random."""
    base = dict(start=None, chain=None, work_plane=None, plane_lock=None, lock=None,
                shift_lock=None, ref=None, acq=None, enc=[], acq_edge=None, acq_fn=None,
                alt="all", shift=False)
    specs = []

    def add(**kw):
        d = dict(base)
        d.update(kw)
        specs.append(d)

    p0 = pois[0]
    add()
    add(acq=pois[1] if len(pois) > 1 else None)
    if tool_key == "select":
        return specs[:2]
    add(start=p0)
    add(start=p0, alt="off")
    add(start=p0, alt="parallel_perp")
    add(start=p0, acq=pois[-1])
    add(start=p0, lock="x")
    add(start=p0, lock="z")
    add(start=p0, shift_lock="x")
    add(start=p0, shift=True)
    if ref_edges:
        a, b = ref_edges[0]
        add(start=p0, ref=("parallel", a, b))
        add(start=p0, ref=("perpendicular", a, b))
        add(start=p0, acq_edge=(a, b))
    add(start=p0, acq=pois[1] if len(pois) > 1 else None, enc=[pois[1], pois[-1]] if len(pois) > 2 else [])
    add(start=p0, acq_fn=(0, 0, 1))
    for k in (2, 3):
        if len(pois) > k:
            add(acq=pois[k])
            add(start=p0, acq=pois[k], alt="parallel_perp")
    add(start=p0, acq_fn=(0, 0, 1), alt="parallel_perp")
    if ref_edges:
        add(start=p0, acq_edge=ref_edges[-1], alt="parallel_perp")
    add(start=pois[1] if len(pois) > 1 else p0, acq=p0)
    add(start=p0, acq=pois[1] if len(pois) > 1 else p0, alt="off")
    if tool_key == "line":
        add(start=pois[-1], chain=p0)
        add(start=pois[1] if len(pois) > 1 else p0, chain=pois[-1])
        add(start=pois[2] if len(pois) > 2 else p0, chain=pois[0], lock="x")
        add(start=p0, chain="same")
    if tool_key in ("rectangle", "circle"):
        add(start=p0, work_plane=((0, 0, 0), (0, 0, 1)))
        add(start=p0, work_plane=((0, -1.5, 0), (0, 1, 0)))
        add(plane_lock="x")
    while len(specs) < n:
        s = rng.choice(pois)
        add(start=rng.choice([None, s, p0]), acq=rng.choice([None, rng.choice(pois)]),
            lock=rng.choice([None, None, "x", "y", "z"]),
            alt=rng.choice(ALT_MODES),
            shift=rng.random() < 0.15,
            enc=rng.choice([[], [], [rng.choice(pois), rng.choice(pois)]]),
            acq_fn=rng.choice([None, None, (0, 0, 1), (0, 1, 0)]))
    return specs[:n]


def pixels_for(vp, rng, pois, extra, n_rand, whole=False):
    """Cursor pixels clustered around notable points + a few random ones."""
    w, h = vp.width(), vp.height()
    out = []
    offs = [(0, 0), (2, 1), (-4, 3), (7, -6), (12, 9), (-15, -3), (0.37, 0.61),
            (0, -35), (0, -80), (-55, 0), (48, 48), (0, 25)]
    for p in list(pois) + list(extra):
        px = vp._world_to_pixel(V(*p))
        if px is None:
            continue
        for ox, oy in offs:
            x, y = px[0] + ox, px[1] + oy
            if 0 <= x < w and 0 <= y < h:
                out.append((x, y))
    for _ in range(n_rand):
        out.append((rng.uniform(0, w), rng.uniform(0, h)))
    # whole pixels mostly (the app passes ints), some fractional
    res = []
    for x, y in out:
        if whole or rng.random() < 0.8:
            res.append((float(round(x)), float(round(y))))
        else:
            res.append((x, y))
    return res


# ---- camera / projection samples ---------------------------------------------------------

def projection_cases(vp, rng, scene_guides):
    """Pixel functions of the current camera: rays, projections, lock-line
    feet, plane hits, clipped segments, guide spans."""
    out = {"ray": [], "w2p": [], "lock": [], "plane": [], "clip": [], "guide": []}
    w, h = vp.width(), vp.height()
    for _ in range(8):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        o, d = vp._pixel_to_ray(x, y)
        out["ray"].append({"px": [x, y], "o": b3(o), "d": b3(d)})
        for s, dd in (((0, 0, 0), (1, 0, 0)), ((1, 2, 0), (0, 0, 1)), ((-1, 1, 0.5), (1, 1, 0))):
            f = vp._project_to_lock_line(V(*s), V(*dd), x, y)
            out["lock"].append({"s": b3t(s), "d": b3t(dd), "px": [x, y], "r": b3(f)})
        n = V(0, 0, 1)
        for allow in (False, True):
            hit = vp._ray_plane(o, d, V(0, 0, 0.5), n, allow_grazing=allow)
            out["plane"].append({"o": b3(o), "d": b3(d), "p": b3t((0, 0, 0.5)), "n": b3(n),
                                 "ag": allow, "r": b3(hit)})
    for _ in range(12):
        p = V(rng.uniform(-6, 6), rng.uniform(-6, 6), rng.uniform(-1, 4))
        r = vp._world_to_pixel(p)
        out["w2p"].append({"p": b3(p), "r": None if r is None else [r[0], r[1]]})
    for _ in range(6):
        a = V(rng.uniform(-8, 8), rng.uniform(-8, 8), rng.uniform(-2, 3))
        b = V(rng.uniform(-30, 30), rng.uniform(-30, 30), rng.uniform(-5, 5))
        seg = vp._clip_segment_front(a, b)
        out["clip"].append({"a": b3(a), "b": b3(b),
                            "r": None if seg is None else [b3(seg[0]), b3(seg[1])]})
    from core.guide import Guide
    for g in (Guide(V(0, 2, 0), V(1, 0, 0)), Guide(V(1, 0, 0), V(1, 1, 0).normalized()),
              Guide(V(-2, -1, 0), V(0, 0, 1)), Guide(V(5, 5, 5), V(0, 1, 0))):
        seg = vp._guide_snap_segment(g)
        out["guide"].append({"p": b3(g.point), "d": b3(g.direction),
                             "r": None if seg is None else [b3(seg[0]), b3(seg[1])]})
    return out


# ---- main -------------------------------------------------------------------------------

def main(quick=False):
    app = QApplication.instance() or QApplication([])
    from views.main_window import MainWindow
    win = MainWindow()
    win.show()
    win.resize(1200, 800)
    win.viewport.setFixedSize(800, 600)
    app.processEvents()
    vp = win.viewport
    rng = random.Random(20261019)
    rec = Rec()
    install_spies(vp, rec)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scenes").mkdir(exist_ok=True)

    import formats.igz as igz
    from core.i18n import set_language

    cams_out: list = []
    states_out: list = []
    state_ix: dict = {}
    samples = []
    acq_out = []
    proj_cases = []
    scene_meta = {}

    def save_scene_json(scene, dest):
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

    scenes = SCENES if not quick else ["empty", "box", "comp", "edit-group"]
    for sname in scenes:
        edit = build_scene(vp, sname)
        save_scene_json(vp.scene, OUT / "scenes" / f"{sname}.igz.json")
        # Turned group: rotated twice about the origin like the user does it.
        edit_path = None
        if isinstance(edit, tuple):
            from core.history import RotateGroupCommand
            g = edit[1]
            vp.history.execute(RotateGroupCommand(g, V(0, 0, 0), V(1, 0, 0), 20.0))
            vp.history.execute(RotateGroupCommand(g, V(0, 0, 0), V(0, 0, 1), 30.0))
            edit = g
            save_scene_json(vp.scene, OUT / "scenes" / f"{sname}.igz.json")
        if edit is not None:
            vp.begin_group_edit(edit)
            edit_path = [vp.scene.groups.index(edit)]
        vp._sync_axes()
        app.processEvents()
        scene_meta[sname] = {"edit": edit_path, "axes": axes_row()}
        pois = POIS[sname]
        ref_edges = []
        for e in vp.scene.edges[:2]:
            ref_edges.append(((e.a.x(), e.a.y(), e.a.z()), (e.b.x(), e.b.y(), e.b.z())))
        extra = [(0, 0, 0)] + [(e.a.x(), e.a.y(), e.a.z()) for e in vp.scene.edges[:6]]
        extra += [((e.a.x() + e.b.x()) / 2, (e.a.y() + e.b.y()) / 2, (e.a.z() + e.b.z()) / 2)
                  for e in vp.scene.edges[:6]]
        for ci, cam in enumerate(CAMERAS if not quick else CAMERAS[:3]):
            set_camera(vp, app, cam)
            crow = camera_row(vp)
            cams_out.append(crow)
            cam_ix = len(cams_out) - 1
            if sname == "empty":
                proj_cases.append({"cam": cam_ix, **projection_cases(vp, rng, None)})
            for tool_key in TOOLS:
                win._activate_tool(tool_key)
                specs = make_specs(rng, sname, tool_key, pois, ref_edges,
                                   40 if tool_key != "select" else 2)
                n_here = 18 if tool_key != "select" else 3
                for si in rng.sample(range(len(specs)), min(n_here, len(specs))):
                    spec = specs[si]
                    if not apply_state(vp, win, spec, vp.scene.edges):
                        continue
                    own_pois = list(pois) + [p for p in (spec["start"], spec["acq"], *spec["enc"])
                                             if isinstance(p, tuple)]
                    pix = pixels_for(vp, rng, own_pois, extra, 2,
                                     whole=sname in ("group", "comp", "nested", "edit-group", "edit-comp"))
                    x, y = rng.choice(pix)
                    reset_vp(vp)
                    apply_state(vp, win, spec, vp.scene.edges)
                    # The hovered circle centre: through the real hover.
                    if sname == "arcs" and tool_key != "select":
                        curve_edge = next((e for e in vp.scene.edges if e.curve is not None), None)
                        vp.inference._hover_edge = curve_edge
                        try:
                            vp._update_center_ref(x, y)
                        except Exception:
                            pass
                    srow = state_row(vp, win, tool_key, spec)
                    ctr = vp._valid_center_ref()
                    if ctr is not None:
                        srow["ctr"] = [b3(ctr[0]), fb(ctr[1])]
                    key = json.dumps(srow, sort_keys=True)
                    six = state_ix.get(key)
                    if six is None:
                        six = state_ix[key] = len(states_out)
                        states_out.append(srow)
                    vp._last_mouse_pos = QPointF(x, y)
                    rec.reset()
                    mods = Qt.KeyboardModifier.ShiftModifier if spec["shift"] else Qt.KeyboardModifier(0)
                    size_before = (vp.width(), vp.height())
                    try:
                        snap = vp.inference.snap(x, y, mods)
                    except Exception as exc:       # noqa: BLE001
                        print("ERROR", sname, cam[0], tool_key, x, y, repr(exc))
                        continue
                    now = camera_row(vp)
                    moved = [k for k in crow if now[k] != crow[k] and k not in ("w", "h")]
                    if moved or (now["w"], now["h"]) != (size_before):
                        print("CAMERA MOVED", sname, cam[0], tool_key, x, y, moved)
                    row = {"s": sname, "c": cam_ix, "t": six, "px": [x, y]}
                    # The offscreen window is resized now and then by Qt's layout.
                    if size_before != (crow["w"], crow["h"]):
                        row["wh"] = list(size_before)
                    if spec["shift"]:
                        row["sh"] = 1
                    if rec.world_raw != "unset":
                        row["wr"] = rec.world_raw
                    if rec.wpn is not None:
                        row["wpn"] = rec.wpn
                    if rec.fu:
                        row["fu"] = 1
                    row["r"] = result_row(snap)
                    if rec.args is not None:
                        row["ed"] = rec.args["ed"]
                        row["rc"] = rec.cs_result
                        if rec.occ:
                            row["occ"] = [[*k, v] for k, v in rec.occ.items()]
                        if len(samples) % 20 == 0:
                            row["a"] = rec.args
                    if snap is not None:
                        en, es = label_of(vp, snap, "en"), label_of(vp, snap, "es")
                        if en is not None:
                            row["lb"] = [en, es]
                    samples.append(row)
            # What the hover acquires (InferenceEngine.acquire), over short
            # sequences of cursor positions: the edge, the corner, the face
            # normal and the circle centre under the pointer, and what is
            # kept when the pointer moves off them.
            for tool_key in ACQ_TOOLS:
                win._activate_tool(tool_key)
                specs = make_specs(rng, sname, tool_key, pois, ref_edges, 40)
                for _ in range(6 if tool_key != "select" else 2):
                    spec = rng.choice(specs)
                    if not apply_state(vp, win, spec, vp.scene.edges):
                        continue
                    own_pois = list(pois) + [p for p in (spec["start"], spec["acq"]) if isinstance(p, tuple)]
                    pix = pixels_for(vp, rng, own_pois, extra, 2, whole=True)
                    pts = [rng.choice(pix) for _ in range(rng.choice([1, 2, 2, 3]))]
                    reset_vp(vp)
                    apply_state(vp, win, spec, vp.scene.edges)
                    srow = state_row(vp, win, tool_key, spec)
                    key = json.dumps(srow, sort_keys=True)
                    six = state_ix.get(key)
                    if six is None:
                        six = state_ix[key] = len(states_out)
                        states_out.append(srow)
                    for (x, y) in pts:
                        vp._last_mouse_pos = QPointF(x, y)
                        vp.inference.acquire(x, y)
                    he = vp.inference._hover_edge
                    ae = vp._acquired_edge
                    ctr = vp._valid_center_ref()
                    hc = vp.inference._hover_center
                    acq_out.append({
                        "s": sname, "c": cam_ix, "t": six, "px": [list(p) for p in pts],
                        "he": None if he is None else [b3(he.a), b3(he.b)],
                        "ae": None if ae is None else [b3(ae.a), b3(ae.b)],
                        "ap": b3(vp._acquired_point), "afn": b3(vp._acquired_face_normal),
                        "hc": None if hc is None else [b3(hc[0]), fb(hc[1])],
                        "ctr": None if ctr is None else [b3(ctr[0]), fb(ctr[1])],
                    })
        print(sname, len(samples), flush=True)
    set_language("en")

    with open(OUT / "acquire.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for row in acq_out:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    with open(OUT / "samples.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for row in samples:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    header = {"cams": cams_out, "states": states_out, "pool": rec.pool,
              "scenes": scene_meta, "proj": proj_cases,
              "thresholds": {"snap": vp.snap_threshold_px, "edge": vp.edge_snap_threshold_px,
                             "angle": vp.inference_angle_deg, "pick": vp.pick_threshold_px,
                             "center_min": vp.CENTER_MIN_RADIUS_PX, "center_hint": vp.CENTER_HINT_PX}}
    (OUT / "header.json").write_text(json.dumps(header, separators=(",", ":")),
                                     encoding="utf-8", newline="\n")
    print("samples", len(samples), "states", len(states_out), "pool", len(rec.pool))
    win._saved_version = vp.scene.version
    win.close()


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
