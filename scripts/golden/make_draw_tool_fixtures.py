# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""Fixtures that hold ``ingetrazo-tools``' drawing tools to ``tools/line.py``,
``rectangle.py``, ``rotated_rectangle.py``, ``circle.py`` (Circle and
Polygon), ``arc.py`` (Arc, 3-Point Arc, Center Arc, Pie) and ``freehand.py``
(docs/rust-rewrite-plan.md, M3.13).

Each *program* is a sequence of pointer, key and value-box events played on a
scene through a real camera. The REAL ``views.viewport.Viewport`` runs
headless (offscreen Qt, no window): the pointer goes through
``Viewport._process_hover`` (the snap engine, the work plane, ``acquire``),
``mousePressEvent`` / ``mouseDoubleClickEvent`` / ``mouseReleaseEvent``, the
keys through ``QWidget.event`` -> ``keyPressEvent`` / ``keyReleaseEvent``
(the arrows' locks, Shift's inference lock and plane mark, Esc's cascade, Alt,
Down, the value box buffer and its Enter) and Ctrl+Z / Ctrl+Y through the
main window's ``_on_undo`` / ``_on_redo`` (copied: the window is not built).
The tool, the snap engine and the history are the app's own.

After every event the script records

* the whole ordered state of the active mesh and of every group's (vertices
  as float32 bits, edges with flags / curve id / faces, faces with loops,
  holes, interior flag and attrs hash; distinct states are stored once and
  referenced by index);
* the tool's own state (the attributes named in ``FIELDS``), the rubber band
  (the preview lines), the floating label, the text of the measurements box
  (``Viewport._measurement_text``), the value box's buffer and caption;
* the cursor's snap (kind, point, axis, colour), the arrow axis lock, the
  Down reference, the Alt mode, the Shift lock, the encouraged points;
* the status texts flashed during the event and the undo / redo depths.

Programs are hand-written for each tool (every gesture the tool has: the
typed values, ``24s``, the arrow plane locks before and after the first
point, Shift's holds, Ctrl's modes, Esc, double-click, undo in the middle of
a chain, the fillet and the tangent inferences...) plus seeded random ones
over the same scenes and cameras.

Pixels are computed here from world points and recorded, so Rust is fed the
very same doubles. Scenes are saved as bare-JSON .igz and reopened by both
sides. ``QGuiApplication.keyboardModifiers`` (the held keys the app reports,
read by ``_refresh_snap``) answers the modifiers this script holds. The
dwell timer (a pause on a point encourages it) never fires headless; the
``dwell`` event calls what it would.

Python's own order is not repeatable in a few places, pinned by
``_pin_order.py`` (mesh objects iterate in creation order).

Committed; rerun only to widen (PYTHONPATH at the repository root, the main
checkout's venv, ``INGETRAZO_RUST`` unset):

    python scripts/golden/make_draw_tool_fixtures.py [tool ...]

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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "golden"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.pop("INGETRAZO_RUST", None)

import _pin_order  # noqa: E402

_pin_order.install()

from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication, QKeyEvent, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import core.mesh as cm  # noqa: E402
from core.history import History  # noqa: E402
from core.scene import Scene  # noqa: E402
from core.vecmath import QVector3D  # noqa: E402
from formats import igz  # noqa: E402
from tests.golden.signature import _dumps, canonical, sha  # noqa: E402

OUT = Path(os.environ.get("DRAW_FIXTURES_OUT", ROOT / "rust" / "crates" / "ingetrazo-tools" / "tests" / "data" / "draw"))

_APP = QApplication.instance() or QApplication([])

# The modifiers the application reports as held (``_refresh_snap`` asks).
_HELD = [Qt.NoModifier]
QGuiApplication.keyboardModifiers = staticmethod(lambda: _HELD[0])

from views.viewport import Viewport  # noqa: E402
import tools.arc as tarc  # noqa: E402
from tools.arc import ArcTool, CenterArcTool, PieTool, ThreePointArcTool  # noqa: E402
from tools.circle import CircleTool, PolygonTool  # noqa: E402
from tools.freehand import FreehandTool  # noqa: E402
from tools.line import LineTool  # noqa: E402
from tools.rectangle import RectangleTool  # noqa: E402
from tools.rotated_rectangle import RotatedRectangleTool  # noqa: E402


def V(x, y, z=0.0):
    return QVector3D(float(x), float(y), float(z))


# ---- the circle fit ----------------------------------------------------------------------------
# ``core.snap.fit_circle`` is LAPACK's least squares (numpy); the Rust one
# (``ingetrazo-snap``) is not (see its docs): on a short arc the fit is
# ill-conditioned, the few ulps of float64 grow past a float32 and the centre
# it finds can differ by one float32 ulp. A program whose outcome depends on
# the last bit of such a fit cannot be held bit for bit by a different solver.
# Each program is therefore played twice: once as it is, once with every fit
# of an arc sweeping less than FIT_MIN_SWEEP degrees nudged by one float32 ulp
# (``_FIT["nudge"]``). If anything recorded differs between the two, the
# program reads those bits and is left out (the count is printed).

FIT_MIN_SWEEP = 170.0
_FIT = {"nudge": False, "hits": 0}


def _install_fit_spy() -> None:
    import numpy as np
    import core.snap as csnap
    orig = csnap.fit_circle

    def spy(points, tol_rel=0.01):
        result = orig(points, tol_rel)
        if result is None or not _FIT["nudge"]:
            return result
        pts = np.array([[p.x(), p.y(), p.z()] for p in points], dtype=np.float64)
        c = np.array([result[0].x(), result[0].y(), result[0].z()], dtype=np.float64)
        q = pts - c
        _u, _s, vt = np.linalg.svd(q, full_matrices=False)
        ang = np.sort(np.arctan2(q @ vt[1], q @ vt[0]))
        gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]]))
        if 360.0 - math.degrees(float(gaps.max())) >= FIT_MIN_SWEEP:
            return result
        _FIT["hits"] += 1
        up = np.float32(np.inf)
        centre = result[0]
        nudged = QVector3D(float(np.nextafter(np.float32(centre.x()), up)),
                           float(np.nextafter(np.float32(centre.y()), up)),
                           float(np.nextafter(np.float32(centre.z()), up)))
        return nudged, result[1]

    csnap.fit_circle = spy


_install_fit_spy()


# ---- encodings -------------------------------------------------------------------------

def h32(x: float) -> str:
    return struct.pack(">f", x).hex()


def h64(x: float) -> str:
    return struct.pack(">d", x).hex()


def pb(p) -> str:
    """A point as 24 hex digits: three float32."""
    return h32(p.x()) + h32(p.y()) + h32(p.z())


def enc(v):
    """A tool attribute as JSON (the same shapes ``Field`` has in Rust)."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return "d" + h64(v)
    if isinstance(v, str):
        return v
    if isinstance(v, QVector3D):
        return pb(v)
    if isinstance(v, (tuple, list)):
        return [enc(x) for x in v]
    raise TypeError(type(v))


def mesh_state(mesh) -> dict:
    vi = {id(v): i for i, v in enumerate(mesh.vertices)}
    fi = {id(f): i for i, f in enumerate(mesh.faces)}
    return {
        "v": [h32(v.position.x()) + h32(v.position.y()) + h32(v.position.z()) for v in mesh.vertices],
        "e": [[vi.get(id(e.v0), -1), vi.get(id(e.v1), -1), 1 if e.soft else 0, 1 if e.hidden else 0,
               e.layer, e.curve, [fi[id(f)] for f in e.faces]] for e in mesh.edges],
        "f": [[[vi.get(id(v), -1) for v in f.loop],
               [[vi.get(id(v), -1) for v in h] for h in f.hole_loops],
               1 if getattr(f, "interior", False) else 0,
               sha(_dumps(canonical(f.attrs)), 16) if f.attrs else ""]
              for f in mesh.faces],
    }


def scene_state(scene) -> dict:
    return {"m": mesh_state(scene.mesh), "g": [mesh_state(g.mesh) for g in scene.groups],
            "edit": scene.edit_group is not None}


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


# ---- scenes ----------------------------------------------------------------------------

def box(mesh, lo, hi):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    for quad in (
        [(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
        [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)],
        [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)],
    ):
        mesh.add_face([V(*p) for p in quad])


def circle_pts(c, r, n, a0=0.0, a1=2 * math.pi, closed=True):
    pts = []
    steps = n if closed else n + 1
    for i in range(steps):
        t = a0 + (a1 - a0) * i / n
        pts.append(V(c[0] + r * math.cos(t), c[1] + r * math.sin(t), c[2]))
    return pts


def scn_empty(sc):
    pass


def scn_ground(sc):
    sc.mesh.add_face([V(0, 0, 0), V(8, 0, 0), V(8, 5, 0), V(0, 5, 0)])


def scn_cube(sc):
    box(sc.mesh, (0, 0, 0), (4, 4, 3))


def scn_cube_far(sc):
    box(sc.mesh, (0, 0, 0), (4, 4, 3))
    box(sc.mesh, (6, 1, 0), (8, 3, 1.5))


def scn_wires(sc):
    m = sc.mesh
    m.add_edge(V(0, 0, 0), V(5, 0, 0))
    m.add_edge(V(5, 0, 0), V(5, 4, 0))
    m.add_edge(V(5, 4, 0), V(5, 4, 3))
    m.add_edge(V(0, 0, 0), V(3, 3, 0))
    m.add_edge(V(1, 4, 0), V(4, 1, 0))
    m.add_edge(V(7, 0, 0), V(7, 3, 0))
    m.add_edge(V(7, 3, 0), V(9, 1.5, 0))
    m.add_edge(V(9, 1.5, 0), V(7, 0, 0))


def scn_circle(sc):
    m = sc.mesh
    pts = circle_pts((4, 3, 0), 2, 24)
    m.add_face(pts)
    m.tag_curve(pts, closed=True)
    m.add_edge(V(7, 3, 0), V(7, 6, 0))


def scn_arcs(sc):
    m = sc.mesh
    pts = circle_pts((3, 0, 0), 2, 12, 0.0, math.pi, closed=False)
    for a, b in zip(pts, pts[1:]):
        m.add_edge(a, b)
    m.tag_curve(pts, closed=False)
    m.add_edge(V(5, 0, 0), V(5, -3, 0))
    m.add_edge(V(1, 0, 0), V(-2, 2, 0))


def scn_corner(sc):
    sc.mesh.add_face([V(0, 0, 0), V(6, 0, 0), V(6, 4, 0), V(0, 4, 0)])


def scn_corners(sc):
    sc.mesh.add_face([V(0, 0, 0), V(6, 0, 0), V(6, 4, 0), V(0, 4, 0)])
    sc.mesh.add_edge(V(6, 4, 0), V(9, 4, 0))
    sc.mesh.add_face([V(8, -3, 0), V(11, -3, 0), V(11, 0, 0), V(8, 0, 0)])


def scn_wall(sc):
    sc.mesh.add_face([V(0, 0, 0), V(8, 0, 0), V(8, 6, 0), V(0, 6, 0)])
    sc.mesh.add_face([V(0, 0, 0), V(5, 0, 0), V(5, 0, 3), V(0, 0, 3)])
    sc.mesh.add_face([V(5, 0, 0), V(5, 4, 0), V(5, 4, 3), V(5, 0, 3)])


def scn_shim(sc):
    """Corners a few millimetres off the ground plane: past the 1 mm the
    planar shapes tolerate, within 1 cm."""
    sc.mesh.add_edge(V(1, 1, 0), V(5, 1, 0.004))
    sc.mesh.add_edge(V(5, 1, 0.004), V(5, 4, 0.004))


SCENES = {
    "shim": scn_shim, "empty": scn_empty, "ground": scn_ground, "cube": scn_cube, "cube_far": scn_cube_far,
    "wires": scn_wires, "circle": scn_circle, "arcs": scn_arcs, "corner": scn_corner,
    "corners": scn_corners, "wall": scn_wall,
}


def scene_pois(scene) -> list:
    """Notable world points of a scene: vertices, edge midpoints, and a grid
    of ground points."""
    pts = []
    for v in scene.mesh.vertices:
        p = v.position
        pts.append((p.x(), p.y(), p.z()))
    for e in scene.mesh.edges:
        a, b = e.a, e.b
        pts.append(((a.x() + b.x()) / 2, (a.y() + b.y()) / 2, (a.z() + b.z()) / 2))
    out = []
    seen = set()
    for p in pts:
        k = tuple(round(c, 4) for c in p)
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


# ---- cameras -----------------------------------------------------------------------------

CAMS = {
    "iso": dict(preset="iso", target=(4, 3, 0), distance=16.0),
    "obl": dict(yaw=-50.0, pitch=32.0, target=(4, 3, 0), distance=16.0),
    "obl2": dict(yaw=20.0, pitch=40.0, target=(3, 2, 1), distance=14.0),
    "low": dict(yaw=-60.0, pitch=12.0, target=(3, 2, 1), distance=14.0),
    "top": dict(preset="top", target=(4, 3, 0), distance=16.0),
    "front": dict(preset="front", target=(3, 0, 1.5), distance=14.0),
    "right": dict(preset="right", target=(0, 2, 1.5), distance=14.0),
    "ortho": dict(yaw=-45.0, pitch=35.0, target=(4, 3, 0), distance=16.0, perspective=False),
    "near": dict(yaw=-35.0, pitch=45.0, target=(3, 2, 0), distance=7.0),
}
W_PX, H_PX = 800, 600


def camera_row(cam) -> dict:
    return {"target": pb(cam.target), "distance": "d" + h64(cam.distance), "yaw": "d" + h64(cam.yaw),
            "pitch": "d" + h64(cam.pitch), "up": pb(cam.up), "fov": "d" + h64(cam.fov_deg),
            "aspect": "d" + h64(cam.aspect), "znear": "d" + h64(cam.znear), "zfar": "d" + h64(cam.zfar),
            "persp": bool(cam.perspective), "w": W_PX, "h": H_PX}


# ---- the program language ------------------------------------------------------------------
# An event is a dict; pointer events carry ``at`` = ("w", (x, y, z), dx, dy):
# a world point plus a pixel offset.

def W(p, dx=0.0, dy=0.0):
    p = tuple(p) + (0.0,) * (3 - len(p))
    return ("w", p, dx, dy)


def WP(p, along, px, perp=0.0):
    """A world point plus ``px`` pixels in the screen direction of the world
    vector ``along`` (and ``perp`` pixels square to it) - to stand an exact
    number of pixels from a point or from a line through it."""
    p = tuple(p) + (0.0,) * (3 - len(p))
    along = tuple(along) + (0.0,) * (3 - len(along))
    return ("wp", p, along, px, perp)


def hv(w):
    return {"e": "hover", "at": w}


def ck(w):
    return {"e": "press", "at": w}


def dbl(w):
    return {"e": "double", "at": w}


def rel():
    return {"e": "release"}


def go(w):
    """Hover, then click: what a user does."""
    return [hv(w), ck(w)]


KEYS = {
    "shift": (Qt.Key_Shift, Qt.ShiftModifier), "ctrl": (Qt.Key_Control, Qt.ControlModifier),
    "alt": (Qt.Key_Alt, Qt.AltModifier), "up": (Qt.Key_Up, None), "down": (Qt.Key_Down, None),
    "left": (Qt.Key_Left, None), "right": (Qt.Key_Right, None), "esc": (Qt.Key_Escape, None),
    "enter": (Qt.Key_Return, None), "back": (Qt.Key_Backspace, None), "p": (Qt.Key_P, None),
    "z": (Qt.Key_Z, None), "r": (Qt.Key_R, None),
}


def kd(name):
    return {"e": "keydown", "k": name}


def ku(name):
    return {"e": "keyup", "k": name}


def tap(name):
    return [kd(name), ku(name)]


def typ(text):
    return {"e": "type", "t": text}


def val(text):
    """Type a value and press Enter."""
    return [typ(text), kd("enter"), ku("enter")]


def undo():
    return {"e": "undo"}


def redo():
    return {"e": "redo"}


def dwell():
    return {"e": "dwell"}


def reselect():
    return {"e": "reselect"}


def flat(*items):
    out = []
    for it in items:
        if isinstance(it, list):
            out.extend(flat(*it))
        else:
            out.append(it)
    return out


# ---- the tools' recorded state ------------------------------------------------------------

PLANE = ["work_plane", "hover_plane", "plane_lock", "plane_ref", "snap_kind", "wireframe_color"]
FIELDS = {
    "LineTool": ["start_point", "hover_point", "chain_first_point", "chain_vertices", "work_plane"],
    "RectangleTool": ["start_point", "hover_point", "_from_center", "_shift_square_lock",
                      "_arrow_square_lock", "_live_side", "_ctrl_toggled"] + PLANE,
    "RotatedRectangleTool": ["start_point", "base_point", "hover_point", "angle", "_locked", "_disc_r",
                             "_near_disc", "_held_angle", "_held_dir"] + PLANE,
    "CircleTool": ["start_point", "hover_point", "sides"] + PLANE,
    "PolygonTool": ["start_point", "hover_point", "sides"] + PLANE,
    "ArcTool": ["start_point", "end_point", "hover_point", "segments", "_tangent_dir", "_snap_bulge",
                "_bulge_kind", "_fillet", "_equidistant"] + PLANE,
    "ThreePointArcTool": ["start_point", "mid_point", "hover_point", "segments"] + PLANE,
    "CenterArcTool": ["start_point", "arm_point", "hover_point", "segments"] + PLANE,
    "PieTool": ["start_point", "arm_point", "hover_point", "segments"] + PLANE,
    "FreehandTool": ["start_point", "_world", "_screen", "_stroke"],
}
DEFAULTS = {"_ctrl_toggled": False}


def tool_fields(tool) -> dict:
    out = {}
    for name in FIELDS[type(tool).__name__]:
        v = getattr(tool, name, DEFAULTS.get(name))
        if name == "_fillet" and v is not None:
            v = list(v)
        if name == "_equidistant" and v is not None:
            v = [v[0], v[1]]                      # (V, P2, edge B): the edge is the mesh's
        if name == "_screen" and v is not None:
            v = [["d" + h64(a), "d" + h64(b)] for a, b in v]
            out[name] = v
            continue
        if name == "_world" and v is not None:
            v = [pb(p) for p in v]
            out[name] = v
            continue
        if name == "snap_kind":
            out[name] = v
            continue
        out[name] = enc(v)
    return out


# ---- the runner ------------------------------------------------------------------------------

class Stop(Exception):
    """The program cannot go on (the Python tool raised)."""


class Runner:
    def __init__(self, scene_path: Path, cam_name: str, tool_cls):
        self.vp = Viewport(None)
        vp = self.vp
        vp.resize(W_PX, H_PX)
        vp.camera.set_aspect(W_PX, H_PX)
        spec = CAMS[cam_name]
        cam = vp.camera
        cam.perspective = spec.get("perspective", True)
        if "preset" in spec:
            cam.set_view(spec["preset"])
        else:
            cam.yaw = math.radians(spec["yaw"])
            cam.pitch = math.radians(spec["pitch"])
        cam.distance = spec["distance"]
        cam.target = V(*spec["target"])
        self.cam_row = camera_row(cam)
        self.scene = reopen(scene_path)
        self.flashes = []
        vp.flash_status = lambda text, msec=2500: self.flashes.append(text)
        cm._CURVE_COUNTER = 5000
        _pin_order.ORD.clear()
        _pin_order.KEEP.clear()
        for m in [self.scene.mesh] + [g.mesh for g in self.scene.groups]:
            for o in list(m.vertices) + list(m.edges) + list(m.faces):
                _pin_order.ORD[id(o)] = len(_pin_order.KEEP)
                _pin_order.KEEP.append(o)
        vp.set_document(self.scene, History(self.scene))
        _HELD[0] = Qt.NoModifier
        ArcTool.last_fillet_d = None
        self.tool = tool_cls()
        vp.set_active_tool(self.tool)
        vp._sync_axes()
        self.states = []
        self.state_ix = {}

    # -- pixels --
    def pixel(self, at):
        if at[0] == "wp":
            _, p, along, dist, perp = at
            a = self.vp._world_to_pixel(V(*p))
            b = self.vp._world_to_pixel(V(p[0] + 0.1 * along[0], p[1] + 0.1 * along[1], p[2] + 0.1 * along[2]))
            if a is None or b is None:
                raise Stop("behind the camera")
            n = math.hypot(b[0] - a[0], b[1] - a[1])
            ux, uy = (b[0] - a[0]) / n, (b[1] - a[1]) / n
            return a[0] + ux * dist - uy * perp, a[1] + uy * dist + ux * perp
        _, p, dx, dy = at
        px = self.vp._world_to_pixel(V(*p))
        if px is None:
            raise Stop("behind the camera")
        return px[0] + dx, px[1] + dy

    def state_id(self) -> int:
        st = scene_state(self.scene)
        key = json.dumps(st, sort_keys=True, separators=(",", ":"))
        ix = self.state_ix.get(key)
        if ix is None:
            ix = self.state_ix[key] = len(self.states)
            self.states.append(st)
        return ix

    def _held_int(self) -> int:
        return int(_HELD[0].value) if hasattr(_HELD[0], "value") else int(_HELD[0])

    # -- events --
    def play(self, ev: dict) -> dict:
        vp, tool = self.vp, self.tool
        self.flashes = []
        kind = ev["e"]
        out = {"e": kind}
        mods = _HELD[0]
        if kind in ("hover", "press", "double"):
            x, y = self.pixel(ev["at"])
            out["x"], out["y"] = x, y
            if not (0 <= x < W_PX and 0 <= y < H_PX):
                raise Stop("off screen")
            out["m"] = self._held_int()
            if kind == "hover":
                vp._process_hover(QPointF(x, y), mods)
            elif kind == "press":
                e = QMouseEvent(QEvent.MouseButtonPress, QPointF(x, y), QPointF(x, y),
                                Qt.LeftButton, Qt.LeftButton, mods)
                vp.mousePressEvent(e)
            else:
                e = QMouseEvent(QEvent.MouseButtonDblClick, QPointF(x, y), QPointF(x, y),
                                Qt.LeftButton, Qt.LeftButton, mods)
                vp.mouseDoubleClickEvent(e)
        elif kind == "release":
            e = QMouseEvent(QEvent.MouseButtonRelease, QPointF(0, 0), QPointF(0, 0),
                            Qt.LeftButton, Qt.NoButton, mods)
            vp.mouseReleaseEvent(e)
        elif kind in ("keydown", "keyup"):
            code, flag = KEYS[ev["k"]]
            if kind == "keydown":
                if flag is not None:
                    _HELD[0] = _HELD[0] | flag
                kev = QKeyEvent(QEvent.KeyPress, code, _HELD[0], "")
            else:
                if flag is not None:
                    _HELD[0] = _HELD[0] & ~flag
                kev = QKeyEvent(QEvent.KeyRelease, code, _HELD[0], "")
            out["k"] = int(code)
            out["m"] = self._held_int()
            vp.event(kev)
        elif kind == "type":
            out["t"] = ev["t"]
            out["m"] = self._held_int()
            for ch in ev["t"]:
                code = ord(ch.upper())
                vp.event(QKeyEvent(QEvent.KeyPress, code, _HELD[0], ch))
        elif kind == "undo":
            out["ret"] = self.undo()
        elif kind == "redo":
            out["ret"] = self.redo()
        elif kind == "dwell":
            vp.inference._encourage_dwelt()
        elif kind == "reselect":
            vp.set_active_tool(tool)
        else:
            raise ValueError(kind)
        out.update(self.record())
        return out

    def undo(self) -> bool:
        """``MainWindow._on_undo``."""
        vp = self.vp
        handler = getattr(vp.active_tool, "on_undo", None)
        if callable(handler) and handler(vp):
            vp.inference.clear_references()
            vp.notify_scene_changed()
            return True
        if vp.history.undo():
            vp.inference.clear_references()
            vp.notify_scene_changed()
            return True
        return False

    def redo(self) -> bool:
        vp = self.vp
        handler = getattr(vp.active_tool, "on_redo", None)
        if callable(handler) and handler(vp):
            vp.inference.clear_references()
            vp.notify_scene_changed()
            return True
        if vp.history.redo():
            vp.inference.clear_references()
            vp.notify_scene_changed()
            return True
        return False

    def record(self) -> dict:
        vp, tool = self.vp, self.tool
        r = {"st": self.state_id(), "tf": tool_fields(tool)}
        r["lwp"] = enc(getattr(vp, "_last_work_plane", None))
        snap = vp.last_snap
        r["snap"] = None if snap is None else [snap.kind, pb(snap.point), snap.axis,
                                               ["d" + h64(float(c)) for c in snap.color]]
        r["band"] = [[pb(a), pb(b)] for a, b in tool.rubber_band_lines()]
        lab = tool.value_label() if callable(getattr(tool, "value_label", None)) else None
        r["label"] = None if lab is None else [lab[0], pb(lab[1])]
        r["meas"] = vp._measurement_text()
        r["vcb"] = vp._value_buffer
        cap = tool.vcb_caption() if callable(getattr(tool, "vcb_caption", None)) else tool.vcb_label
        r["cap"] = cap
        r["flash"] = list(self.flashes)
        r["lock"] = vp.axis_lock
        sl = vp._shift_lock
        r["inf"] = {"rm": vp.reference_mode, "lm": vp.linear_inference_mode,
                    "sl": None if sl is None else [pb(sl[0]), ["d" + h64(float(c)) for c in sl[1]]],
                    "ap": None if vp._acquired_point is None else pb(vp._acquired_point),
                    "enc": [pb(p) for p in vp._encouraged]}
        r["depth"] = len(vp.history.undo_stack)
        r["redo"] = len(vp.history.redo_stack)
        r["err"] = vp.history.last_error
        return r


def run_program(scene_name: str, cam_name: str, tool_cls, events: list, paths: dict):
    """Play ``events`` (twice: see the circle fit above); returns the case."""
    _FIT["nudge"], _FIT["hits"] = False, 0
    case, runner = _play_program(scene_name, cam_name, tool_cls, events, paths)
    _FIT["nudge"] = True
    again, _ = _play_program(scene_name, cam_name, tool_cls, events, paths)
    _FIT["nudge"] = False
    case["fit_sensitive"] = (_FIT["hits"] > 0
                             and json.dumps(case["events"], sort_keys=True) != json.dumps(again["events"], sort_keys=True))
    return case, runner


def _play_program(scene_name: str, cam_name: str, tool_cls, events: list, paths: dict):
    runner = Runner(paths[scene_name], cam_name, tool_cls)
    case = {"scene": scene_name, "cam": runner.cam_row, "pre": runner.state_id(), "events": []}
    stopped = None
    for ev in events:
        try:
            case["events"].append(runner.play(ev))
        except Stop as why:
            stopped = str(why)
            break
        except Exception as exc:  # noqa: BLE001 - a bug of the Python tool: the program ends there
            stopped = "EXC " + type(exc).__name__ + ": " + str(exc)
            break
    if stopped is not None and stopped.startswith("EXC"):
        print("   python tool raised:", tool_cls.__name__, scene_name, len(case["events"]), stopped)
        case["raised"] = True
    case["stopped"] = stopped
    # Undo everything: the scene is back to where it started.
    vp = runner.vp
    while vp.history.undo_stack:
        vp.history.undo()
    case["undone"] = [runner.state_id()]
    case["states"] = runner.states
    return case, runner


# ---- the programs -------------------------------------------------------------------------------

PROGRAMS: dict = {}


def prog(tool_name, scene, cam, *events):
    PROGRAMS.setdefault(tool_name, []).append((scene, cam, flat(*events), True))


TOOLS = {"line": LineTool, "rectangle": RectangleTool, "rotated_rectangle": RotatedRectangleTool,
         "circle": CircleTool, "polygon": PolygonTool, "arc": ArcTool, "arc3": ThreePointArcTool,
         "center_arc": CenterArcTool, "pie": PieTool, "freehand": FreehandTool}


POIS: dict = {}


def build_scenes() -> dict:
    """Write every scene as bare-JSON .igz; returns name -> path."""
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, build in SCENES.items():
        s = Scene()
        build(s)
        path = OUT / f"{name}.igz.json"
        save_scene_json(s, path)
        paths[name] = path
        POIS[name] = scene_pois(reopen(path))
    return paths


# ---- random programs -----------------------------------------------------------------------------

#: Typed values by tool (metric forms the value box reads).
VALUES = {
    "line": ["1", "2.5", "0.75", "3", "-1.5", "0", "10", "1,5", "250cm", "1200mm", "0.3m", "3;2;1", "2;-1;0", "4;3"],
    "rectangle": ["3;2", "4,2.5", "1;1", "2", "0.5", "-2;3", "200cm;1m", "5,3", "0;2", "3;3"],
    "rotated_rectangle": ["3", "2;45", "1.5;90", "4;0", "-2", "2;-45", "3;180", "1;30", "0", "2,5;60", "5;90"],
    "circle": ["1", "2.5", "0.5", "12s", "3s", "6s", "40", "0", "-2", "24s", "5"],
    "polygon": ["1", "2.5", "0.5", "5s", "3s", "8s", "40", "0", "-2", "6s", "4"],
    "arc": ["0.5", "1", "-0.5", "2r", "3r", "0.2r", "8s", "4s", "1s", "16s", "0", "-1", "10", "30"],
    "arc3": ["8s", "4s", "1s", "16s", "12", "5", "2"],
    "center_arc": ["90", "45", "-45", "180", "8s", "6s", "1s", "0", "30", "270", "360"],
    "pie": ["90", "45", "-45", "180", "8s", "6s", "0", "30", "120"],
    "freehand": [],
}
ARITY = {"line": 4, "rectangle": 2, "rotated_rectangle": 3, "circle": 2, "polygon": 2, "arc": 3,
         "arc3": 3, "center_arc": 3, "pie": 3, "freehand": 0}
OFFS = [(0, 0), (0, 0), (2, 1), (-3, 2), (5, -4), (9, 3), (0, 14), (-12, 0), (0.37, 0.61), (20, 20)]


def rand_point(rng, scene_name):
    pois = POIS[scene_name]
    r = rng.random()
    if r < 0.5 and pois:
        p = rng.choice(pois)
        dx, dy = rng.choice(OFFS)
        return W(p, dx, dy)
    if r < 0.85:
        return W((round(rng.uniform(-1, 9), 2), round(rng.uniform(-1, 7), 2), 0.0), *rng.choice(OFFS[:5]))
    return W((round(rng.uniform(0, 8), 2), round(rng.uniform(0, 6), 2), round(rng.uniform(0, 3), 2)),
             *rng.choice(OFFS[:5]))


def rand_program(rng, tool_key, scene_name):
    ev = []
    arity = ARITY[tool_key]
    values = VALUES[tool_key]
    for _ in range(rng.randint(5, 10)):
        g = rng.random()
        if g < 0.40:
            if tool_key == "freehand":
                p = rand_point(rng, scene_name)
                ev += [hv(p), ck(p)]
                for _k in range(rng.randint(2, 8)):
                    ev.append(hv(rand_point(rng, scene_name)))
                ev.append(rel())
                continue
            n = rng.randint(1, arity)
            for _k in range(n):
                p = rand_point(rng, scene_name)
                if rng.random() < 0.6:
                    ev.append(hv(rand_point(rng, scene_name)))
                ev += [hv(p), ck(p)]
        elif g < 0.55 and values:
            ev += [hv(rand_point(rng, scene_name)), val(rng.choice(values))]
        elif g < 0.62:
            ev += tap(rng.choice(["left", "right", "up", "down", "down"]))
            if rng.random() < 0.7:
                ev.append(hv(rand_point(rng, scene_name)))
        elif g < 0.68:
            ev += [kd("shift"), hv(rand_point(rng, scene_name)), hv(rand_point(rng, scene_name)), ku("shift")]
        elif g < 0.72:
            ev += tap("ctrl")
        elif g < 0.75:
            ev += tap("alt")
        elif g < 0.80:
            ev += tap("esc")
        elif g < 0.86:
            ev += [undo()]
        elif g < 0.89:
            p = rand_point(rng, scene_name)
            ev += [hv(p), dbl(p)]
        elif g < 0.92:
            ev += [dwell()]
        elif g < 0.94:
            ev += [typ(rng.choice(["1", "2", "3;", "12"])), kd("back"), ku("back"), typ("5")]
        elif g < 0.96:
            ev += [rel()]
        elif g < 0.98:
            ev += [redo()]
        else:
            ev += [reselect()]
    return ev


RAND_SCENES = ["empty", "ground", "cube", "wires", "circle", "arcs", "corner", "wall", "cube_far", "corners"]
RAND_CAMS = ["iso", "obl", "obl2", "low", "top", "front", "ortho", "near", "right"]


def add_random(tool_key, count, seed):
    name = TOOL_NAMES[tool_key]
    rng = random.Random(seed)
    for i in range(count):
        scene_name = rng.choice(RAND_SCENES)
        cam = rng.choice(RAND_CAMS)
        r2 = random.Random(seed * 1000 + i)
        PROGRAMS.setdefault(name, []).append((scene_name, cam, flat(rand_program(r2, tool_key, scene_name)), False))


TOOL_NAMES = {k: v.__name__ for k, v in TOOLS.items()}


def write_tool(tool_key, paths) -> None:
    cls = TOOLS[tool_key]
    cases = []
    broken = raised = skipped_fit = 0
    events_n = 0
    states: list = []
    state_ix: dict = {}
    for n_prog, (scene_name, cam, events, hand) in enumerate(PROGRAMS.get(cls.__name__, [])):
        case, runner = run_program(scene_name, cam, cls, events, paths)
        if not case["events"]:
            broken += 1
            continue
        if case.get("raised"):
            raised += 1
        if case.pop("fit_sensitive"):
            skipped_fit += 1
            if hand:
                print(f"   hand-written program {n_prog} on {scene_name} left out: it reads the fit of a short arc")
            continue
        if hand and len(case["events"]) < len(events) and not case.get("raised"):
            print(f"   hand-written program {n_prog} on {scene_name} stops at event {len(case['events'])} of "
                  f"{len(events)} ({case.get('stopped')})")
        # Re-number the states into the file's table.
        remap = {}
        for i, st in enumerate(case["states"]):
            key = json.dumps(st, sort_keys=True, separators=(",", ":"))
            j = state_ix.get(key)
            if j is None:
                j = state_ix[key] = len(states)
                states.append(st)
            remap[i] = j
        case["pre"] = remap[case["pre"]]
        for e in case["events"]:
            e["st"] = remap[e["st"]]
        case["undone"] = [remap[i] for i in case["undone"]]
        del case["states"]
        cases.append(case)
        events_n += len(case["events"])
    data = {"tool": cls.__name__, "states": states, "cases": cases}
    path = OUT / f"cases_{tool_key}.json"
    path.write_text(json.dumps(data, sort_keys=True, separators=(",", ":")), encoding="utf-8", newline="\n")
    print(f"{tool_key}: {len(cases)} programs, {events_n} events, {len(states)} states, "
          f"{broken} empty, {raised} ended by a Python exception, {skipped_fit} left out (short-arc fit), "
          f"{path.stat().st_size // 1024} KiB")


def main() -> None:
    wanted = [a for a in sys.argv[1:] if a in TOOLS] or list(TOOLS)
    paths = build_scenes()
    (OUT / ".gitattributes").write_text("* -text\n", encoding="utf-8", newline="\n")
    register_programs()
    for key in wanted:
        write_tool(key, paths)


# The programs are below (one function per tool). A hand-written program
# that cannot be played to its end (a point behind the camera or off the
# window) is reported when the fixtures are written.

def register_programs() -> None:
    programs_line()
    programs_rectangle()
    programs_rotated_rectangle()
    programs_circle()
    programs_polygon()
    programs_arc()
    programs_arc3()
    programs_center_arc()
    programs_pie()
    programs_freehand()
    for key, count, seed in (("line", 20, 101), ("rectangle", 20, 102), ("rotated_rectangle", 20, 103),
                             ("circle", 16, 104), ("polygon", 14, 105), ("arc", 24, 106), ("arc3", 14, 107),
                             ("center_arc", 14, 108), ("pie", 14, 109), ("freehand", 14, 110)):
        add_random(key, count, seed)


def circle_path(c, r, n, a0=0.0, a1=360.0, z=0.0):
    out = []
    for i in range(n + 1):
        a = math.radians(a0 + (a1 - a0) * i / n)
        out.append((c[0] + r * math.cos(a), c[1] + r * math.sin(a), z))
    return out


O, PA, PB, PC = (0, 0, 0), (4, 0, 0), (4, 3, 0), (0, 3, 0)


def programs_line() -> None:
    t = "LineTool"
    prog(t, "empty", "iso", go(W(O)), go(W(PA)), undo(), undo())
    # a closed loop makes a face; undo and redo it
    prog(t, "empty", "iso", go(W(O)), go(W(PA)), go(W(PB)), go(W(PC)), go(W(O)), undo(), redo(), undo(), undo())
    # near an axis the point lands on it (the magnet)
    prog(t, "empty", "top", go(W((1, 1))), hv(W((4, 1.02))), ck(W((4, 1.02))), hv(W((4.1, 4))), ck(W((4.1, 4))),
         hv(W((7, 4.05))), ck(W((7, 4.05))))
    # Esc cancels the chain; the buffer first
    prog(t, "empty", "iso", go(W(O)), hv(W(PA)), typ("2"), tap("esc"), tap("esc"), hv(W(PB)), go(W(PB)),
         go(W((6, 3))), tap("esc"), tap("esc"), hv(W(PC)))
    # typed lengths along the rubber band, negative, zero, comma, units, delta
    prog(t, "empty", "iso", go(W(O)), hv(W(PA)), val("2.5"), val("-1.5"), hv(W(PB)), val("1,5"), val("0"),
         val("3;2;1"), val("2;-1;0"), val("4;3"), val("250cm"), val("1200mm"), val("0.3m"), hv(W(PB)), val("0.005"), val("-0.004"), val("abc"),
         typ("12"), kd("back"), ku("back"), kd("enter"), ku("enter"))
    # arrow axis locks: held through the chain, let go with the click
    prog(t, "empty", "iso", go(W(O)), tap("right"), hv(W((3, 1.5))), hv(W((4, 2))), ck(W((4, 2))), hv(W((5, 3))),
         tap("up"), hv(W((5, 3, 2))), hv(W((5, 3, 2.5), 4, 0)), ck(W((5, 3, 2.5), 4, 0)), tap("left"), hv(W((2, 3))),
         tap("left"), tap("up"), tap("up"))
    # a lock before the first click, then typed length along it
    prog(t, "empty", "iso", tap("up"), go(W(O)), hv(W((3, 2, 1))), val("2"), tap("esc"))
    # Shift holds an inference
    prog(t, "empty", "top", go(W(O)), hv(W((3, 0.05))), kd("shift"), hv(W((5, 1.5))), hv(W((6, 3))), ck(W((6, 3))),
         ku("shift"), hv(W((6, 5))))
    prog(t, "empty", "obl", go(W(O)), hv(W((0.05, 3))), kd("shift"), hv(W((2, 4))), ck(W((2, 4))), ku("shift"),
         hv(W((3, 6))))
    # Ctrl+Z steps the chain back one vertex at a time
    prog(t, "empty", "iso", go(W(O)), go(W(PA)), go(W(PB)), go(W(PC)), undo(), undo(), hv(W((2, 5))),
         go(W((2, 5))), undo(), undo(), undo(), undo(), undo())
    # existing geometry: endpoints, midpoints, on-edge, intersections
    prog(t, "wires", "iso", go(W((0, 0, 0))), hv(W((2.5, 0, 0))), hv(W((5, 0, 0), 3, 2)), ck(W((5, 0, 0), 3, 2)),
         hv(W((5, 2, 0))), ck(W((5, 2, 0))), hv(W((2.5, 2.5, 0))), ck(W((2.5, 2.5, 0))))
    prog(t, "wires", "top", go(W((1, 4))), hv(W((4, 1), 2, 2)), ck(W((4, 1), 2, 2)), hv(W((2.5, 2.5))),
         ck(W((2.5, 2.5))), hv(W((1, 4))), ck(W((1, 4))))
    prog(t, "wires", "low", go(W((5, 4, 3))), hv(W((5, 4, 0))), ck(W((5, 4, 0))), hv(W((3, 3, 0))), ck(W((3, 3, 0))))
    # the screen detector finds the blue axis within 9 px of it, not 10
    prog(t, "empty", "iso", go(W(O)), hv(WP(O, (0, 0, 1), 60, 8.5)), hv(WP(O, (0, 0, 1), 60, 9.5)), ck(WP(O, (0, 0, 1), 60, 9.5)),
         hv(WP(PA, (0, 0, 1), 90, -9.5)), hv(WP(PA, (0, 0, 1), 90, -8.5)), hv(WP(PA, (0, 0, 1), 90, 8.8)))
    prog(t, "empty", "iso", go(W(O)), hv(WP(O, (0, 0, 1), 40, 9.2)), tap("esc"), go(W((2, 2))), hv(WP((2, 2), (0, 0, 1), 70, -9.7)),
         ck(WP((2, 2), (0, 0, 1), 70, -9.7)))
    # the triangle on the right: closing it makes a face
    prog(t, "wires", "top", go(W((7, 3))), go(W((9, 1.5))), go(W((7, 0))), go(W((7, 3))), undo())
    # on faces: the chain keeps the plane of the face it started on
    prog(t, "cube", "obl", go(W((2, 2, 3))), hv(W((3, 2, 3))), ck(W((3, 2, 3))), hv(W((3, 3, 3))), ck(W((3, 3, 3))),
         hv(W((2, 3, 3))), ck(W((2, 3, 3))), hv(W((2, 2, 3))), ck(W((2, 2, 3))))
    prog(t, "cube", "obl2", go(W((4, 1, 1))), hv(W((4, 3, 1))), ck(W((4, 3, 1))), hv(W((4, 3, 2.5))), ck(W((4, 3, 2.5))))
    prog(t, "cube", "front", go(W((1, 0, 1))), hv(W((3, 0, 1))), ck(W((3, 0, 1))), hv(W((3, 0, 2.5))),
         ck(W((3, 0, 2.5))))
    prog(t, "wall", "obl2", go(W((1, 0, 1))), hv(W((4, 0, 1))), ck(W((4, 0, 1))), hv(W((4, 0, 2.5))),
         ck(W((4, 0, 2.5))), hv(W((1, 0, 2.5))), ck(W((1, 0, 2.5))))
    # Down: parallel / perpendicular to the edge under the cursor
    prog(t, "wires", "top", hv(W((2.5, 0))), go(W((1, 2))), hv(W((2.5, 0))), tap("down"), hv(W((3, 2))),
         hv(W((4, 2.2))), tap("down"), hv(W((3, 2))), hv(W((4, 2.2))), ck(W((4, 2.2))), tap("down"))
    # Alt: all -> off -> parallel/perp, one operation
    prog(t, "wires", "top", go(W((1, 4))), tap("alt"), hv(W((3, 4.1))), tap("alt"), hv(W((3, 4.1))), tap("alt"),
         hv(W((3.1, 4.1))), ck(W((3, 4.1))), hv(W((5, 5))))
    # double-click is two clicks in a row
    prog(t, "empty", "iso", hv(W(O)), dbl(W(O)), hv(W(PA)), dbl(W(PA)), hv(W(PB)), ck(W(PB)), rel(), tap("esc"))
    # a pause on a corner encourages it: the from-point inference
    prog(t, "wires", "top", hv(W((5, 0))), dwell(), hv(W((5, 4))), dwell(), go(W((2, 2))), hv(W((5.02, 2))),
         hv(W((3.4, 4))), ck(W((3.4, 4))))
    # a press right after a double-click in place is a triple click: a plain click without the follow-ups
    prog(t, "empty", "iso", hv(W(O)), dbl(W(O)), ck(W(O)), ck(W(O, 3, 0)), hv(W(PA)), ck(W(PA)), dbl(W(PA)), ck(W(PA, 2, 2)),
         ck(W(PB)), tap("esc"))
    # a pause on a circle's rim encourages its centre
    prog(t, "circle", "top", hv(W((6, 3))), dwell(), go(W((9, 5))), hv(W((4, 5.02))), hv(W((9, 3.03))), ck(W((9, 3.03))),
         hv(W((4, 3))), dwell(), hv(W((7, 6))))
    # starting over by picking the tool again
    prog(t, "empty", "iso", go(W(O)), go(W(PA)), reselect(), hv(W(PB)), go(W(PB)), undo())
    # circle and arcs as references: the centre, the arc midpoint
    prog(t, "circle", "top", hv(W((4, 3))), go(W((4, 3))), hv(W((6, 3))), ck(W((6, 3))), hv(W((7, 5))), ck(W((7, 5))))
    prog(t, "arcs", "top", hv(W((3, 2))), go(W((3, 2))), hv(W((1, 0))), ck(W((1, 0))), hv(W((5, 0))), ck(W((5, 0))))


def programs_rectangle() -> None:
    t = "RectangleTool"
    prog(t, "empty", "iso", go(W(O)), go(W((4, 3))), undo(), redo(), undo())
    # the square snaps; Shift holds it; Down holds it
    prog(t, "empty", "iso", go(W(O)), hv(W((3, 2.9))), hv(W((3.02, 3))), ck(W((3.02, 3))))
    prog(t, "empty", "top", go(W(O)), hv(W((3, 3.1))), kd("shift"), hv(W((3, 5))), hv(W((4, 3))), ck(W((4, 3))), ku("shift"))
    prog(t, "empty", "top", go(W(O)), hv(W((3, 3.05))), tap("down"), hv(W((4, 2))), hv(W((2, 6))), tap("down"), ck(W((2, 6))))
    # Ctrl: from the centre; a Ctrl+Z chord takes it back
    prog(t, "empty", "top", go(W((0, 0))), hv(W((8, 8.36))), ck(W((8, 8.36))), go(W((0, 0))), hv(W((8, 7.66))), ck(W((8, 7.66))))
    prog(t, "empty", "top", go(W((0, 0))), hv(W((8, 8.55))), ck(W((8, 8.55))), go(W((0, 0))), hv(W((8, 8.22))), ck(W((8, 8.22))),
         tap("ctrl"), go(W((3, 3))), hv(W((7, 7.3))), ck(W((7, 7.3))))
    prog(t, "empty", "iso", tap("ctrl"), go(W((3, 2))), hv(W((5, 3))), ck(W((5, 3))), tap("ctrl"), go(W(O)), hv(W((2, 1))),
         ck(W((2, 1))))
    prog(t, "empty", "iso", kd("ctrl"), kd("z"), ku("z"), undo(), ku("ctrl"), go(W(O)), hv(W((2, 1))), val("2;1"))
    # typed sizes: width;height, a list comma, a lone number (the side being stretched)
    prog(t, "empty", "iso", go(W(O)), hv(W((3, 2))), val("3;2"), go(W((5, 0))), hv(W((7, 1))), val("4,2.5"), go(W((0, 4))),
         hv(W((3, 4.2))), val("2"), hv(W((0.4, 6))), val("1.5"), val("0"), val("-2;3"), tap("esc"))
    prog(t, "empty", "iso", go(W(O)), hv(W((2, 3))), val("200cm;1m"), go(W((6, 0))), hv(W((5, 2))), val("1"),
         val("3;3"), typ("1,2"), tap("esc"), tap("esc"))
    # arrow plane locks before the first click
    prog(t, "empty", "iso", tap("right"), hv(W((3, 2))), go(W((3, 2))), hv(W((3, 3, 2))), hv(W((3, 4, 2))), ck(W((3, 4, 2))))
    prog(t, "empty", "iso", tap("left"), hv(W((1, 0))), go(W((1, 0))), hv(W((3, 0, 2))), ck(W((3, 0, 2))))
    prog(t, "empty", "obl2", tap("up"), tap("up"), tap("up"), hv(W((1, 1))), go(W((1, 1))), hv(W((3, 2))), ck(W((3, 2))))
    # ... and after it (#543)
    prog(t, "empty", "iso", go(W((1, 1))), tap("right"), hv(W((1, 3, 2))), hv(W((1, 4, 2.5))), tap("right"), hv(W((3, 3))),
         tap("up"), hv(W((3, 3.5))), val("2;2"))
    # on a cube: the plane of the face under the first click
    prog(t, "cube", "obl", go(W((1, 1, 3))), hv(W((3, 3, 3))), ck(W((3, 3, 3))), go(W((4, 1, 2))), hv(W((4, 3, 1))),
         ck(W((4, 3, 1))))
    prog(t, "cube", "obl2", go(W((4, 1, 0.5))), hv(W((4, 2.5, 2))), ck(W((4, 2.5, 2))), go(W((2, 2, 3))), hv(W((1, 1, 3))),
         val("1;1"))
    prog(t, "wall", "obl2", go(W((1, 0, 0.5))), hv(W((3, 0, 2))), ck(W((3, 0, 2))), go(W((5, 1, 0.5))), hv(W((5, 3, 2))),
         ck(W((5, 3, 2))))
    # a corner snapped off the plane re-plans the rectangle (a window opening)
    prog(t, "cube", "obl", go(W((0, 0, 3))), hv(W((4, 0, 3))), ck(W((4, 0, 3))), go(W((4, 0, 0))), hv(W((0, 0, 0), 2, 2)),
         ck(W((0, 0, 0), 2, 2)))
    # a side of zero: the reason is flashed (and the plane arrow it names)
    prog(t, "empty", "iso", go(W(O)), hv(W((4, 0))), ck(W((4, 0))), hv(W((0, 3))), ck(W((0, 3))))
    prog(t, "cube", "obl", go(W((0, 0, 3))), hv(W((4, 4, 3))), ck(W((4, 4, 3))), go(W((0, 0, 3))), hv(W((0, 0, 0))),
         ck(W((0, 0, 0))))
    # existing geometry: snaps to corners and edges, a rectangle over a face
    prog(t, "wires", "iso", go(W((0, 0, 0))), hv(W((5, 4, 0))), ck(W((5, 4, 0))), go(W((5, 0, 0))), hv(W((3, 3, 0))),
         ck(W((3, 3, 0))))
    prog(t, "ground", "iso", go(W((1, 1))), hv(W((3, 2))), ck(W((3, 2))), go(W((2, 1.5))), hv(W((5, 4))), ck(W((5, 4))),
         undo(), undo())
    prog(t, "corner", "top", go(W((1, 1))), hv(W((2, 2))), ck(W((2, 2))), go(W((0, 0))), hv(W((6, 4))), ck(W((6, 4))))
    prog(t, "empty", "ortho", go(W(O)), hv(W((2, 3))), ck(W((2, 3))), tap("esc"), go(W((3, 3))), tap("esc"), tap("esc"))
    prog(t, "empty", "iso", hv(W(O)), dbl(W(O)), hv(W((3, 2))), dbl(W((3, 2))), hv(W((1, 1))), ck(W((1, 1))))
    prog(t, "shim", "iso", go(W((1, 1, 0))), hv(W((5, 1, 0.004))), ck(W((5, 1, 0.004))), go(W((1, 1, 0))), hv(W((5, 4, 0.004))),
         ck(W((5, 4, 0.004))))
    prog(t, "empty", "front", go(W((0, 0, 0))), hv(W((3, 0, 2))), ck(W((3, 0, 2))), go(W((4, 0, 0))), hv(W((6, 0, 1))),
         val("2;1"))
    prog(t, "empty", "iso", go(W(O)), reselect(), hv(W((3, 2))), ck(W((3, 2))), hv(W((5, 5))), ck(W((5, 5))), kd("alt"),
         ku("alt"))
    prog(t, "wires", "iso", hv(W((2.5, 0, 0))), tap("down"), go(W((2.5, 0, 0))), hv(W((2.5, 1, 1))), hv(W((2.5, 2, 2))),
         ck(W((2.5, 2, 2))), tap("down"))
    prog(t, "cube", "obl", hv(W((2, 2, 3))), kd("shift"), hv(W((2.5, 2, 3))), go(W((2.5, 2, 3))), hv(W((3, 3, 3))),
         ck(W((3, 3, 3))), ku("shift"))
    prog(t, "cube", "obl2", hv(W((4, 2, 1.5))), tap("down"), hv(W((4, 3, 2))), go(W((4, 3, 2))), hv(W((4, 3.5, 2.5))),
         ck(W((4, 3.5, 2.5))), tap("down"), hv(W((3, 3, 3))))


def programs_rotated_rectangle() -> None:
    t = "RotatedRectangleTool"
    # basic: base edge, then the width
    prog(t, "empty", "iso", go(W(O)), hv(W((4, 2))), ck(W((4, 2))), hv(W((3, 4))), ck(W((3, 4))))
    # near the first protractor's rim the direction snaps to 15 degrees
    prog(t, "empty", "top", go(W(O)), hv(W(O, 52, -30)), hv(W(O, 48, -33)), ck(W(O, 48, -33)), hv(W(O, 80, -90)),
         hv(W(O, 30, -120)), ck(W(O, 30, -120)))
    prog(t, "empty", "top", go(W(O)), hv(W(O, 10, 60)), hv(W(O, 40, 40)), ck(W(O, 40, 40)), hv(W(O, 100, 60)),
         hv(W(O, 120, 30)), ck(W(O, 120, 30)))
    # Shift holds the direction of the base edge
    prog(t, "empty", "top", go(W(O)), hv(W((3, 1.5))), kd("shift"), hv(W((5, 1.5))), hv(W((5, 4))), ck(W((5, 4))), ku("shift"),
         hv(W((6, 5))), hv(W((5, 7))), ck(W((5, 7))))
    # Shift holds the width's angle
    prog(t, "empty", "iso", go(W(O)), hv(W((4, 0))), ck(W((4, 0))), hv(W((2, 3))), kd("shift"), hv(W((2, 4))), hv(W((3, 6))),
         ck(W((3, 6))), ku("shift"))
    # typed: length; angle, then width; angle
    prog(t, "empty", "iso", go(W(O)), hv(W((4, 2))), val("3;30"), val("2;90"), hv(W((3, 3))), val("1.5"))
    prog(t, "empty", "top", go(W((1, 1))), hv(W((3, 1))), val("-2"), hv(W((2, 3))), val("1;45"), val("0"), val("-1"),
         val("4;0"), val("1,5;60"), val("2;3;4"))
    prog(t, "empty", "iso", go(W(O)), val("3"), hv(W((3, 2))), val("2;-45"), val("0"), tap("esc"), go(W((1, 1))),
         hv(W((3, 3))), val("5"), val("2"))
    # arrow plane locks pick the plane of the base edge: stand it up
    prog(t, "empty", "iso", tap("left"), go(W(O)), hv(W((3, 0, 1))), ck(W((3, 0, 1))), hv(W((3, -2, 1))),
         hv(W((3, 2, 2))), ck(W((3, 2, 2))))
    prog(t, "empty", "iso", tap("right"), go(W(O)), hv(W((0, 3, 0.5))), ck(W((0, 3, 0.5))), hv(W((2, 3, 2))), ck(W((2, 3, 2))))
    # after the base is down the arrows are the viewport's axis lock, read by the width
    prog(t, "empty", "iso", go(W(O)), hv(W((4, 0))), ck(W((4, 0))), tap("up"), hv(W((4, 3, 1))), hv(W((4, 3, 2))),
         ck(W((4, 3, 2))), tap("right"), tap("up"))
    prog(t, "empty", "iso", go(W(O)), hv(W((4, 0))), ck(W((4, 0))), tap("left"), hv(W((2, 2))), hv(W((2, 3))),
         ck(W((2, 3))), tap("left"))
    # the plane of a face under the first click; a snapped corner for the width
    prog(t, "cube", "obl", go(W((1, 1, 3))), hv(W((3, 1, 3))), ck(W((3, 1, 3))), hv(W((3, 3, 3))), ck(W((3, 3, 3))))
    prog(t, "cube", "obl2", go(W((4, 0.5, 0.5))), hv(W((4, 3, 0.5))), ck(W((4, 3, 0.5))), hv(W((4, 3, 2))),
         hv(W((4, 3, 3), 3, 0)), ck(W((4, 3, 3), 3, 0)))
    prog(t, "cube", "obl", go(W((0, 0, 3))), hv(W((4, 0, 3))), ck(W((4, 0, 3))), hv(W((4, 4, 3))), ck(W((4, 4, 3))), undo())
    prog(t, "wall", "obl2", go(W((1, 0, 0.5))), hv(W((4, 0, 0.5))), ck(W((4, 0, 0.5))), hv(W((4, 2, 0.5))), ck(W((4, 2, 0.5))),
         hv(W((4, 0, 2))), ck(W((4, 0, 2))))
    # the width sticks to flat / upright within 3 degrees, not 4
    prog(t, "empty", "obl2", go(W(O)), hv(W((4, 0))), ck(W((4, 0))), hv(W((4, 6, 0.3665))), hv(W((4, 6, 0.4))), ck(W((4, 6, 0.3665))))
    prog(t, "empty", "obl2", go(W(O)), hv(W((4, 0))), ck(W((4, 0))), hv(W((4, 0.2445, 3.99))), ck(W((4, 0.2445, 3.99))))
    prog(t, "empty", "obl2", go(W(O)), hv(W((4, 0))), ck(W((4, 0))), hv(W((4, 4, 0.3))), hv(W((4, 4, -0.3))), ck(W((4, 4, -0.3))))
    # the base edge perpendicular to the plane: nothing to draw the width in
    prog(t, "cube", "obl", tap("up"), hv(W((0, 0, 0))), go(W((0, 0, 0))), hv(W((0, 0, 3))), ck(W((0, 0, 3))), hv(W((2, 2, 0))),
         ck(W((2, 2, 0))))
    # no width: flashed, and the click goes on waiting
    prog(t, "empty", "iso", go(W(O)), hv(W((4, 0))), ck(W((4, 0))), hv(W((2, 0))), ck(W((2, 0))), hv(W((2, 2))), ck(W((2, 2))))
    # a tiny base edge is nothing
    prog(t, "empty", "iso", go(W(O)), ck(W(O)), hv(W((0.001, 0))), ck(W((0.001, 0))), tap("esc"), hv(W((1, 1))))
    prog(t, "wires", "iso", go(W((0, 0, 0))), hv(W((5, 0, 0))), ck(W((5, 0, 0))), hv(W((5, 4, 0))), ck(W((5, 4, 0))),
         undo(), undo())
    prog(t, "empty", "ortho", go(W(O)), hv(W((4, 1))), ck(W((4, 1))), hv(W((3, 3))), ck(W((3, 3))), redo(), undo())
    prog(t, "empty", "front", go(W((0, 0, 0))), hv(W((3, 0, 0))), ck(W((3, 0, 0))), hv(W((3, 0, 2))), ck(W((3, 0, 2))))
    prog(t, "empty", "iso", go(W(O)), hv(W((4, 2))), dbl(W((4, 2))), hv(W((3, 4))), dbl(W((3, 4))))


def programs_circle() -> None:
    t = "CircleTool"
    prog(t, "empty", "iso", go(W((3, 2))), go(W((5, 2))), undo(), redo(), undo())
    # the radius typed; the side count typed before the centre and as "Ns"
    prog(t, "empty", "iso", go(W((3, 2))), hv(W((4, 2.5))), val("1.5"), val("12"), go(W((6, 2))), hv(W((6.8, 2))), val("0.7"),
         val("0"), val("-1"))
    prog(t, "empty", "iso", val("8"), val("2"), go(W((3, 2))), hv(W((4.5, 3))), ck(W((4.5, 3))), val("16s"), val("5s"),
         val("2s"), val("7s"), undo())
    prog(t, "empty", "top", typ("12s"), kd("enter"), ku("enter"), go(W((3, 2))), val("2s"), val("2.5"), val("30s"), val("4s"),
         val("3s"), val("24s"))
    # arrow plane locks: circles on the three planes, before and after the centre
    prog(t, "empty", "iso", tap("right"), go(W((3, 2))), hv(W((3, 3, 1))), hv(W((3, 3.5, 1.5))), ck(W((3, 3.5, 1.5))))
    prog(t, "empty", "iso", tap("left"), go(W((3, 2))), hv(W((4, 2, 1))), ck(W((4, 2, 1))), tap("up"), go(W((6, 2))),
         hv(W((7, 2.5))), ck(W((7, 2.5))))
    prog(t, "empty", "obl2", go(W((3, 2))), tap("right"), hv(W((3, 3, 1))), ck(W((3, 3, 1))))
    # the arrows also move a preview ring over the cursor before the first click
    prog(t, "empty", "iso", hv(W((3, 2))), tap("up"), hv(W((3, 2))), tap("up"), tap("right"), hv(W((3.1, 2))), tap("left"),
         hv(W((3.2, 2))), tap("down"), tap("esc"))
    # on faces
    prog(t, "cube", "obl", go(W((2, 2, 3))), hv(W((3, 2, 3))), ck(W((3, 2, 3))), go(W((4, 2, 1.5))), hv(W((4, 3, 1.5))),
         ck(W((4, 3, 1.5))))
    prog(t, "wall", "obl2", go(W((2, 0, 1.5))), hv(W((3, 0, 1.5))), val("0.6"), go(W((6, 2, 0))), hv(W((7, 2, 0))), ck(W((7, 2, 0))))
    prog(t, "ground", "iso", go(W((3, 2))), hv(W((4.5, 2))), ck(W((4.5, 2))), go(W((5, 2))), hv(W((6.5, 2))), ck(W((6.5, 2))),
         undo(), undo())
    # over another circle: the centre inference and the scoped arrangement
    prog(t, "circle", "top", hv(W((4, 3))), go(W((4, 3))), hv(W((5, 3))), ck(W((5, 3))), go(W((6, 3))), hv(W((8, 3))), ck(W((8, 3))))
    prog(t, "circle", "top", go(W((4, 5))), hv(W((4, 3))), ck(W((4, 3))), tap("esc"), hv(W((6, 3))), go(W((6, 3))), val("1"))
    prog(t, "wires", "iso", go(W((2.5, 0))), hv(W((2.5, 2))), ck(W((2.5, 2))), go(W((1.5, 1.5))), hv(W((3, 1.5))), val("1"))
    # a circle in the air next to the wall; Esc; the second centre
    prog(t, "wall", "obl2", tap("up"), hv(W((3, 1))), go(W((3, 1))), hv(W((3, 2.5))), tap("esc"), tap("esc"), hv(W((2, 2, 1))))
    prog(t, "empty", "iso", go(W((3, 2))), hv(W((3.0005, 2))), ck(W((3.0005, 2))), tap("esc"), hv(W((4, 2))), ck(W((4, 2))),
         hv(W((6, 2))), ck(W((6, 2))))
    # Down: the plane perpendicular to the edge under the cursor; Shift marks the plane
    prog(t, "wires", "iso", hv(W((5, 2, 0))), tap("down"), go(W((5, 2, 0))), hv(W((5, 3, 1))), ck(W((5, 3, 1))), tap("down"))
    prog(t, "cube", "obl", hv(W((2, 2, 3))), kd("shift"), go(W((2, 2, 3))), hv(W((3, 2, 3))), ku("shift"), ck(W((3, 2, 3))))
    prog(t, "cube", "obl", hv(W((2, 2, 3))), tap("down"), hv(W((3, 2, 3))), go(W((3, 2, 3))), hv(W((3.5, 2, 3))), ck(W((3.5, 2, 3))))
    prog(t, "empty", "ortho", go(W((3, 2))), hv(W((5, 3))), ck(W((5, 3))), hv(W((1, 1))), dbl(W((1, 1))), hv(W((2, 2))), ck(W((2, 2))))
    prog(t, "empty", "front", go(W((3, 0, 1.5))), hv(W((4, 0, 1.5))), ck(W((4, 0, 1.5))), hv(W((1, 0, 1))), ck(W((1, 0, 1))))
    prog(t, "empty", "right", go(W((0, 2, 1.5))), hv(W((0, 3, 1.5))), ck(W((0, 3, 1.5))), tap("alt"), hv(W((0, 1, 1))))
    # (the centre of a circle in a tilted plane is the fit's float64 noise in a coordinate that should be 0: not hovered)
    prog(t, "shim", "iso", go(W((1, 1, 0))), hv(W((5, 1, 0.004))), ck(W((5, 1, 0.004))), go(W((5, 4, 0.004))), hv(W((3, 3, 0))),
         ck(W((3, 3, 0))))


def programs_polygon() -> None:
    t = "PolygonTool"
    prog(t, "empty", "iso", go(W((3, 2))), go(W((5, 2.5))), undo(), redo())
    prog(t, "empty", "iso", go(W((3, 2))), hv(W((4, 3))), val("1.5"), val("3"), go(W((6, 2))), hv(W((7, 2))), val("1"), val("8"),
         go(W((2, 5))), hv(W((3, 5))), ck(W((3, 5))))
    prog(t, "empty", "top", go(W((3, 2))), hv(W((5, 2))), ck(W((5, 2))), val("8s"), val("3s"), val("12s"), val("2s"), val("5s"))
    prog(t, "empty", "top", val("5s"), go(W((3, 2))), hv(W((4, 3.3))), ck(W((4, 3.3))), val("11s"), undo())
    prog(t, "empty", "iso", tap("right"), go(W((3, 2))), hv(W((3, 3, 1))), ck(W((3, 3, 1))), tap("left"), go(W((3, 2))),
         hv(W((4, 2, 1))), ck(W((4, 2, 1))))
    prog(t, "empty", "iso", go(W((3, 2))), tap("up"), hv(W((5, 2))), ck(W((5, 2))), tap("up"))
    prog(t, "cube", "obl", go(W((2, 2, 3))), hv(W((3, 2.5, 3))), ck(W((3, 2.5, 3))), go(W((4, 2, 1.5))), hv(W((4, 3, 1.5))),
         ck(W((4, 3, 1.5))), undo())
    prog(t, "wall", "obl2", go(W((2, 0, 1.5))), hv(W((3, 0, 1.5))), val("0.7"), go(W((6, 2, 0))), hv(W((7, 2, 0))), ck(W((7, 2, 0))))
    prog(t, "circle", "top", go(W((4, 3))), hv(W((5, 3))), ck(W((5, 3))), go(W((4, 3))), hv(W((3, 4))), ck(W((3, 4))))
    prog(t, "ground", "iso", go(W((2, 2))), hv(W((4, 2))), ck(W((4, 2))), go(W((5, 2))), hv(W((7, 2))), ck(W((7, 2))), undo())
    prog(t, "wires", "iso", go(W((2.5, 0))), hv(W((2.5, 2))), ck(W((2.5, 2))), go(W((1, 1))), hv(W((3, 1))), val("1"))
    prog(t, "empty", "iso", hv(W((3, 2))), go(W((3, 2))), tap("esc"), hv(W((4, 2))), go(W((4, 2))), hv(W((4, 2))), ck(W((4, 2))))
    prog(t, "empty", "ortho", go(W((3, 2))), hv(W((4, 4))), ck(W((4, 4))), reselect(), go(W((5, 5))))
    prog(t, "empty", "iso", val("0"), val("2"), val("-3"), val("4"), go(W((3, 2))), val("2s"), val("0"), tap("esc"))
    prog(t, "wires", "iso", hv(W((2.5, 0, 0))), tap("down"), go(W((2.5, 0, 0))), hv(W((2.5, 1, 0.7))), ck(W((2.5, 1, 0.7))),
         tap("down"), hv(W((3, 3, 0))))


def programs_arc() -> None:
    t = "ArcTool"
    # the three clicks and the bulge
    prog(t, "empty", "iso", go(W(O)), go(W(PA)), hv(W((2, 1))), ck(W((2, 1))), undo())
    prog(t, "empty", "top", go(W(O)), go(W(PA)), hv(W((2, -1.2))), hv(W((2, -2))), ck(W((2, -2))), redo())
    # a flat bulge is the chord; half circle snaps; the number is the bulge or "2r" the radius
    prog(t, "empty", "top", go(W(O)), go(W(PA)), hv(W((2, 0.00003))), ck(W((2, 0.00003))), go(W((0, 3))), go(W((4, 3))),
         hv(W((2, 4.97))), hv(W((2, 5))), ck(W((2, 5))))
    prog(t, "empty", "iso", go(W(O)), go(W(PA)), hv(W((2, 1))), val("0.5"), go(W((0, 3))), go(W((4, 3))), hv(W((2, 2))),
         val("-0.8"), go(W((0, 5))), go(W((4, 5))), hv(W((2, 6))), val("2r"), go(W((6, 0))), go(W((8, 0))), hv(W((7, 1))),
         val("0.5r"), val("1r"))
    # the segment count: before the first click as a number, "Ns" any time, rebuild right after
    prog(t, "empty", "iso", val("8"), go(W(O)), go(W(PA)), hv(W((2, 1))), ck(W((2, 1))), val("4s"), val("16s"), val("1s"),
         val("2s"), val("0"), undo())
    prog(t, "empty", "iso", go(W(O)), go(W(PA)), hv(W((2, 1.5))), val("6s"), ck(W((2, 1.5))), val("10s"), val("3s"), undo(), redo())
    # arrow plane locks
    prog(t, "empty", "iso", tap("right"), go(W((0, 1, 0))), hv(W((0, 3, 0.5))), ck(W((0, 3, 0.5))), hv(W((0, 2, 2))),
         ck(W((0, 2, 2))))
    prog(t, "empty", "iso", go(W(O)), tap("left"), hv(W((4, 0, 0.5))), ck(W((4, 0, 0.5))), hv(W((2, 0, 2))), ck(W((2, 0, 2))))
    # on a face and over geometry
    prog(t, "cube", "obl", go(W((1, 1, 3))), hv(W((3, 1, 3))), ck(W((3, 1, 3))), hv(W((2, 3, 3))), ck(W((2, 3, 3))))
    prog(t, "wall", "obl2", go(W((1, 0, 0.5))), hv(W((4, 0, 0.5))), ck(W((4, 0, 0.5))), hv(W((2.5, 0, 2.5))), ck(W((2.5, 0, 2.5))))
    prog(t, "wires", "iso", go(W((0, 0, 0))), hv(W((5, 0, 0))), ck(W((5, 0, 0))), hv(W((2.5, 1.5, 0))), ck(W((2.5, 1.5, 0))))
    # tangent to an edge (cyan) from a point inside it; the fillet at a corner (magenta)
    prog(t, "corner", "top", go(W((2, 0))), hv(W((1, 1.5))), hv(W((0, 2))), ck(W((0, 2))), hv(W((0.586, 0.586))),
         hv(W((0.62, 0.6))), ck(W((0.62, 0.6))), undo())
    prog(t, "corner", "top", go(W((2, 0))), hv(W((0, 2), 3, 2)), ck(W((0, 2), 3, 2)), dbl(W((0.586, 0.586))), undo(), redo())
    prog(t, "corner", "top", go(W((2, 0))), hv(W((0, 2))), ck(W((0, 2))), val("1.5"), go(W((3, 0))), hv(W((0, 3))), ck(W((0, 3))),
         val("1r"), go(W((1, 0))), hv(W((0, 1))), ck(W((0, 1))), hv(W((0.3, 0.3))), tap("alt"), ck(W((0.3, 0.3))))
    prog(t, "corner", "top", go(W((2, 0))), hv(W((0, 2))), ck(W((0, 2))), val("9"), val("3"), tap("esc"), go(W((4, 0))),
         hv(W((6, 2))), ck(W((6, 2))), hv(W((5.4, 0.6))), ck(W((5.4, 0.6))))
    prog(t, "corner", "top", go(W((2, 0))), hv(W((0, 2))), ck(W((0, 2))), dbl(W((0.586, 0.586))), reselect(), hv(W((0, 4), 1, 0)),
         dbl(W((0, 4), 1, 0)), hv(W((6, 4), 1, 0)), dbl(W((6, 4), 1, 0)))
    prog(t, "corners", "top", go(W((6, 3))), hv(W((7.5, 4))), hv(W((9, 4))), ck(W((9, 4))), hv(W((7.9, 3.8))), ck(W((7.9, 3.8))))
    # the fillet point snaps within 10 px of the equidistant point, not 12; the tangent bulge within 8 px, not 9
    prog(t, "corner", "top", go(W((2, 0))), hv(WP((0, 2), (0, 1, 0), 11)), ck(WP((0, 2), (0, 1, 0), 11)), hv(W((0.586, 0.586))),
         ck(W((0.586, 0.586))))
    prog(t, "corner", "top", go(W((2, 0))), hv(WP((0, 2), (0, 1, 0), 9.5)), ck(WP((0, 2), (0, 1, 0), 9.5)), hv(W((0.586, 0.586))),
         ck(W((0.586, 0.586))))
    prog(t, "corner", "top", go(W((2, 0))), hv(W((0, 2))), ck(W((0, 2))), hv(WP((0.5858, 0.5858), (-1, -1, 0), 8.5)),
         hv(WP((0.5858, 0.5858), (1, 1, 0), 8.5)), ck(WP((0.5858, 0.5858), (1, 1, 0), 8.5)))
    prog(t, "corner", "top", go(W((2, 0))), hv(W((0, 2))), ck(W((0, 2))), hv(WP((0.5858, 0.5858), (-1, -1, 0), 7.5)),
         ck(WP((0.5858, 0.5858), (-1, -1, 0), 7.5)))
    # a double-click within 20 px of another two-edge corner repeats the last rounding
    prog(t, "corners", "top", go(W((2, 0))), hv(W((0, 2))), ck(W((0, 2))), dbl(W((0.586, 0.586))), go(W((4, 0))),
         dbl(WP((6, 0), (1, 0, 0), 25)), go(W((4, 4))), dbl(WP((0, 4), (0, 1, 0), 15)))
    prog(t, "corners", "top", go(W((2, 0))), hv(W((0, 2))), ck(W((0, 2))), dbl(W((0.586, 0.586))), go(W((4, 0))),
         dbl(WP((6, 0), (-1, 0, 0), 18)), go(W((1, 0))), dbl(WP((6, 0), (1, 1, 0), 28)))
    prog(t, "corners", "top", go(W((5, 0))), hv(W((6, 1))), ck(W((6, 1))), val("0.5"), go(W((6, 3))), hv(W((5, 4))), ck(W((5, 4))), val("2"))
    # tangent at the vertex an arc ended on
    prog(t, "arcs", "top", hv(W((5, 0))), go(W((5, 0))), hv(W((6, -1))), hv(W((7, -2))), ck(W((7, -2))), hv(W((6.4, -1.3))),
         hv(W((6.0, -2.5))), ck(W((6.0, -2.5))), undo())
    prog(t, "arcs", "top", go(W((5, -1.5))), hv(W((6, -3))), ck(W((6, -3))), hv(W((7, -2))), ck(W((7, -2))))
    prog(t, "arcs", "top", go(W((1, 0))), hv(W((-1, 1))), ck(W((-1, 1))), hv(W((0, 0.7))), ck(W((0, 0.7))))
    prog(t, "circle", "top", go(W((6, 3))), hv(W((4, 5))), ck(W((4, 5))), hv(W((4.5, 5.8))), ck(W((4.5, 5.8))))
    prog(t, "empty", "ortho", go(W(O)), go(W(PA)), dbl(W((2, 1))), hv(W((2, 1))), ck(W((2, 1))), reselect())
    prog(t, "cube", "obl", hv(W((2, 2, 3))), tap("down"), go(W((1, 1, 3))), hv(W((3, 1, 3))), ck(W((3, 1, 3))), hv(W((2, 0, 3))),
         ck(W((2, 0, 3))), tap("down"))
    prog(t, "empty", "iso", go(W(O)), go(W(PA)), tap("up"), hv(W((2, 1))), tap("left"), hv(W((2, 1, 1))), ck(W((2, 1, 1))),
         tap("right"))


def programs_arc3() -> None:
    t = "ThreePointArcTool"
    prog(t, "empty", "iso", go(W(O)), go(W((2, 1.5))), hv(W((4, 0))), ck(W((4, 0))), undo(), redo())
    prog(t, "empty", "top", go(W(O)), hv(W((2, 2))), ck(W((2, 2))), hv(W((4, 0))), hv(W((4, 3))), ck(W((4, 3))))
    # collinear: a straight chord; a mid point on the start is nothing
    prog(t, "empty", "top", go(W(O)), ck(W(O)), go(W((2, 0))), hv(W((4, 0))), ck(W((4, 0))), hv(W((1, 1))))
    prog(t, "empty", "iso", val("6"), go(W(O)), go(W((2, 1))), hv(W((4, 0))), val("3"), ck(W((4, 0))), val("12s"), val("1s"))
    prog(t, "empty", "iso", go(W(O)), go(W((2, 1))), val("5s"), hv(W((4, 0))), ck(W((4, 0))), val("9s"), val("2s"))
    prog(t, "empty", "iso", tap("right"), go(W((0, 1))), go(W((0, 2, 1))), hv(W((0, 3, 0))), ck(W((0, 3, 0))))
    prog(t, "empty", "iso", go(W(O)), tap("up"), go(W((2, 1))), hv(W((4, 0))), ck(W((4, 0))))
    prog(t, "cube", "obl", go(W((1, 1, 3))), go(W((2, 3, 3))), hv(W((3, 1, 3))), ck(W((3, 1, 3))))
    prog(t, "cube", "obl2", go(W((4, 1, 0.5))), go(W((4, 2, 2))), hv(W((4, 3, 0.5))), ck(W((4, 3, 0.5))))
    prog(t, "wires", "iso", go(W((0, 0, 0))), go(W((5, 4, 3))), hv(W((5, 0, 0))), ck(W((5, 0, 0))))
    prog(t, "wires", "top", go(W((0, 0))), go(W((3, 3))), hv(W((5, 0))), ck(W((5, 0))), go(W((1, 4))), go(W((2.5, 2.5))), hv(W((4, 1))),
         ck(W((4, 1))))
    prog(t, "wall", "obl2", go(W((1, 0, 0.5))), go(W((2.5, 0, 2.5))), hv(W((4, 0, 0.5))), ck(W((4, 0, 0.5))))
    prog(t, "circle", "top", go(W((2, 3))), hv(W((4, 5))), ck(W((4, 5))), hv(W((6, 3))), ck(W((6, 3))))
    prog(t, "empty", "ortho", go(W(O)), go(W((2, 1))), hv(W((4, 0))), dbl(W((4, 0))), reselect(), tap("esc"))
    prog(t, "empty", "front", go(W((0, 0, 0))), go(W((1.5, 0, 1.5))), hv(W((3, 0, 0))), ck(W((3, 0, 0))))
    prog(t, "empty", "low", go(W(O)), go(W((2, 1))), hv(W((4, 0))), tap("alt"), ck(W((4, 0))))


def programs_center_arc() -> None:
    t = "CenterArcTool"
    prog(t, "empty", "iso", go(W((2, 2))), go(W((4, 2))), hv(W((2, 4))), ck(W((2, 4))), undo(), redo())
    prog(t, "empty", "top", go(W((2, 2))), hv(W((4, 2.5))), ck(W((4, 2.5))), hv(W((2, 5))), hv(W((0, 2))), hv(W((2, -1))), ck(W((2, -1))))
    # the sweep typed (sign from the cursor), 0 and 360, the segment count
    prog(t, "empty", "top", go(W((2, 2))), go(W((4, 2))), hv(W((2, 4))), val("90"), go(W((2, 2))), go(W((4, 2))), hv(W((2, 0))),
         val("45"), go(W((2, 2))), go(W((4, 2))), val("30"), go(W((2, 2))), go(W((4, 2))), hv(W((2, 4))), val("0"), val("360"),
         val("180"), val("-90"))
    prog(t, "empty", "iso", val("6"), go(W((2, 2))), go(W((4, 2))), hv(W((2, 4))), ck(W((2, 4))), val("10s"), val("3s"), go(W((6, 2))),
         go(W((8, 2))), hv(W((6, 4))), val("1s"), val("2s"), val("60"))
    prog(t, "empty", "iso", go(W((2, 2))), val("8s"), go(W((4, 2))), hv(W((2, 4))), ck(W((2, 4))), undo())
    prog(t, "empty", "iso", tap("right"), go(W((0, 2, 0))), go(W((0, 4, 0))), hv(W((0, 2, 2))), ck(W((0, 2, 2))))
    prog(t, "empty", "iso", go(W((2, 2))), tap("left"), go(W((4, 2, 0))), hv(W((2, 2, 2))), ck(W((2, 2, 2))))
    prog(t, "cube", "obl", go(W((2, 2, 3))), go(W((3.5, 2, 3))), hv(W((2, 3.5, 3))), ck(W((2, 3.5, 3))))
    prog(t, "cube", "obl2", go(W((4, 2, 1.5))), go(W((4, 3.5, 1.5))), hv(W((4, 2, 3))), ck(W((4, 2, 3))))
    prog(t, "circle", "top", hv(W((4, 3))), go(W((4, 3))), hv(W((6, 3))), ck(W((6, 3))), hv(W((4, 5))), ck(W((4, 5))))
    prog(t, "circle", "top", go(W((4, 3))), go(W((5, 3))), hv(W((4, 4))), val("90"), val("15s"), undo())
    prog(t, "wires", "iso", go(W((0, 0, 0))), go(W((3, 0, 0))), hv(W((0, 3, 0))), ck(W((0, 3, 0))))
    prog(t, "wall", "obl2", go(W((2, 0, 1))), go(W((3.5, 0, 1))), hv(W((2, 0, 2.5))), ck(W((2, 0, 2.5))))
    prog(t, "empty", "ortho", go(W((2, 2))), go(W((2, 2))), hv(W((2, 4))), ck(W((2, 4))), dbl(W((4, 2))), dbl(W((2, 4))))
    prog(t, "empty", "front", go(W((1, 0, 1))), go(W((2, 0, 1))), hv(W((1, 0, 2))), ck(W((1, 0, 2))), reselect())
    prog(t, "empty", "right", go(W((0, 1, 1))), go(W((0, 2, 1))), hv(W((0, 1, 2))), ck(W((0, 1, 2))))
    # angle snapped by the pitch: 100 degrees is 7 steps, 10 is 1
    prog(t, "empty", "top", go(W((2, 2))), go(W((4, 2))), hv(W((1.6, 3.9))), ck(W((1.6, 3.9))), go(W((2, 2))), go(W((4, 2))),
         hv(W((3.9, 2.4))), ck(W((3.9, 2.4))))


def programs_pie() -> None:
    t = "PieTool"
    prog(t, "empty", "iso", go(W((2, 2))), go(W((4, 2))), hv(W((2, 4))), ck(W((2, 4))), undo(), redo())
    prog(t, "empty", "top", go(W((2, 2))), hv(W((4, 2.5))), ck(W((4, 2.5))), hv(W((2, 5))), hv(W((0, 2))), ck(W((0, 2))))
    prog(t, "empty", "top", go(W((2, 2))), go(W((4, 2))), hv(W((2, 4))), val("90"), go(W((2, 2))), go(W((4, 2))), hv(W((2, 0))),
         val("120"), go(W((6, 2))), go(W((8, 2))), val("45"), val("0"), val("360"), val("-30"))
    prog(t, "empty", "iso", val("8"), go(W((2, 2))), go(W((4, 2))), hv(W((2, 4))), ck(W((2, 4))), val("6s"), val("2s"), undo())
    prog(t, "empty", "iso", tap("up"), go(W((2, 2))), go(W((4, 2))), hv(W((2, 4))), ck(W((2, 4))))
    prog(t, "empty", "iso", tap("right"), go(W((0, 2, 0))), go(W((0, 4, 0))), hv(W((0, 2, 2))), ck(W((0, 2, 2))))
    prog(t, "ground", "iso", go(W((2, 2))), go(W((4, 2))), hv(W((2, 4))), ck(W((2, 4))), go(W((6, 2))), go(W((7, 2))), hv(W((6, 3))),
         ck(W((6, 3))), undo(), undo())
    prog(t, "cube", "obl", go(W((2, 2, 3))), go(W((3.5, 2, 3))), hv(W((2, 3.5, 3))), ck(W((2, 3.5, 3))))
    prog(t, "circle", "top", go(W((4, 3))), go(W((5, 3))), hv(W((4, 4))), ck(W((4, 4))))
    prog(t, "wires", "iso", go(W((0, 0, 0))), go(W((3, 0, 0))), hv(W((0, 3, 0))), ck(W((0, 3, 0))))
    prog(t, "wall", "obl2", go(W((2, 0, 1))), go(W((3.5, 0, 1))), hv(W((2, 0, 2.5))), ck(W((2, 0, 2.5))))
    prog(t, "empty", "ortho", go(W((2, 2))), go(W((4, 3))), hv(W((1, 3))), ck(W((1, 3))), reselect())
    prog(t, "corner", "top", go(W((1, 1))), go(W((3, 1))), hv(W((1, 3))), ck(W((1, 3))), go(W((0, 0))), go(W((2, 0))), hv(W((0, 2))),
         ck(W((0, 2))))


def programs_freehand() -> None:
    t = "FreehandTool"
    c = circle_path((4, 3), 2, 24)
    # a closed stroke (a circle)
    prog(t, "empty", "top", [ck(W(c[0])) if False else hv(W(c[0]))], ck(W(c[0])), [hv(W(p)) for p in c[1:-1]], hv(W(c[-1], 4, 3)),
         rel())
    # an open S curve
    s = [(t / 4.0, 2 + 1.5 * math.sin(t / 1.2), 0) for t in range(0, 25)]
    prog(t, "empty", "top", hv(W(s[0])), ck(W(s[0])), [hv(W(p)) for p in s[1:]], rel(), undo(), redo())
    # a straight drag collapses to two points; a click and a tiny drag are nothing
    prog(t, "empty", "iso", hv(W((0, 0))), ck(W((0, 0))), [hv(W((i * 0.5, i * 0.25))) for i in range(1, 12)], rel())
    prog(t, "empty", "iso", hv(W((0, 0))), ck(W((0, 0))), hv(W((0, 0), 1, 1)), hv(W((0, 0), 2, 1)), rel(), hv(W((2, 2))), ck(W((2, 2))),
         hv(W((2, 2), 5, 0)), hv(W((2, 2), 10, 0)), rel())
    prog(t, "empty", "iso", hv(W((0, 0))), ck(W((0, 0))), hv(W((1, 1))), hv(W((2, 0))), tap("esc"), hv(W((3, 3))), rel())
    # the stroke closes within 10 px of its start (not 12); samples are 3 px apart (not 4)
    ring = circle_path((4, 3), 2, 20)
    prog(t, "empty", "top", hv(W(ring[0])), ck(W(ring[0])), [hv(W(p)) for p in ring[1:-1]], hv(WP(ring[0], (1, 0, 0), 11)), rel())
    prog(t, "empty", "top", hv(W(ring[0])), ck(W(ring[0])), [hv(W(p)) for p in ring[1:-1]], hv(WP(ring[0], (0, 1, 0), 9)), rel())
    prog(t, "empty", "top", hv(W(ring[0])), ck(W(ring[0])), [hv(W(p)) for p in ring[1:-1]], hv(WP(ring[0], (0, 1, 0), 10.6)), rel())
    prog(t, "empty", "iso", hv(W((1, 1))), ck(W((1, 1))), [hv(WP((1, 1), (1, 0.3, 0), 3.5 * k)) for k in range(1, 30)], rel())
    prog(t, "empty", "iso", hv(W((1, 1))), ck(W((1, 1))), [hv(WP((1, 1), (1, 0.3, 0), 3.2 * k, 0.4 * (k % 3))) for k in range(1, 40)],
         rel())
    # a stroke on a face; a closed one makes a face of its own
    sq = [(1, 1, 3), (2, 1, 3), (3, 1, 3), (3, 2, 3), (3, 3, 3), (2, 3, 3), (1, 3, 3), (1, 2, 3), (1, 1, 3)]
    prog(t, "cube", "obl", hv(W(sq[0])), ck(W(sq[0])), [hv(W(p)) for p in sq[1:-1]], hv(W(sq[-1], 2, 2)), rel())
    prog(t, "cube", "obl2", hv(W((4, 1, 0.5))), ck(W((4, 1, 0.5))), hv(W((4, 2, 1.5))), hv(W((4, 3, 0.5))), hv(W((4, 3.5, 2))), rel())
    prog(t, "wall", "obl2", hv(W((1, 0, 0.5))), ck(W((1, 0, 0.5))), hv(W((2, 0, 1.5))), hv(W((3, 0, 0.8))), hv(W((4, 0, 2))), rel())
    prog(t, "wires", "iso", hv(W((0, 0, 0))), ck(W((0, 0, 0))), [hv(W((i * 0.5, 3 - abs(i - 5) * 0.5))) for i in range(0, 12)], rel())
    prog(t, "empty", "ortho", hv(W((1, 1))), ck(W((1, 1))), hv(W((3, 3))), hv(W((5, 1))), hv(W((3, -1))), hv(W((1, 1), 2, 2)), rel())
    # arrows are the viewport's axis lock; undo and the lock released with the release
    prog(t, "empty", "iso", tap("up"), hv(W((1, 1))), ck(W((1, 1))), hv(W((2, 3))), hv(W((3, 2))), hv(W((4, 4))), rel(), tap("up"))
    prog(t, "empty", "iso", hv(W((1, 1))), dbl(W((1, 1))), hv(W((2, 3))), hv(W((3, 2))), hv(W((4, 4))), rel(), reselect())


if __name__ == "__main__":
    main()
