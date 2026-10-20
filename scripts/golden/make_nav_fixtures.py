# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""Fixtures that hold ``ingetrazo-app``'s navigation to the Python viewport
(docs/rust-rewrite-plan.md, M4.4): the camera after each orbit, pan, zoom,
zoom-extents and projection event.

The REAL ``Viewport`` (offscreen Qt, no GL context) receives synthetic mouse
and wheel events through its own ``mousePressEvent`` / ``mouseMoveEvent`` /
``mouseReleaseEvent`` / ``wheelEvent`` -- the gesture state machine, the
pivot and focus resolution, the zoom-focus cache -- and ``camera_after`` is
the camera it ends up with (every float as its bit pattern: f64 as 16 hex
digits, f32 as 8). Each event also records the arguments the viewport handed
to the camera (``calls``: the pivot of an orbit, the focus of a zoom, the
depth of a pan), so the Rust replay can be held to the camera maths alone
(recorded inputs) and to the whole path (its own pivots and foci).

The models are ``examples/*.igz``, opened by ``formats.igz.load_into``.

Committed; rerun only to widen (from the repository root, PYTHONPATH at it):

    QT_QPA_PLATFORM=offscreen PYTHONUTF8=1 PYTHONPATH=. \\
        .venv/Scripts/python.exe scripts/golden/make_nav_fixtures.py

Deleted with the Python app at M10.
"""
from __future__ import annotations

import json
import os
import random
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.pop("INGETRAZO_RUST", None)

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent, QVector3D, QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_APP = QApplication.instance() or QApplication([])

from core.scene import Scene  # noqa: E402
from formats import igz  # noqa: E402
from views.viewport import Viewport  # noqa: E402

OUT = ROOT / "rust" / "crates" / "ingetrazo-app" / "tests" / "data" / "nav"
EXAMPLES = ["arco-yanque", "banca-pergola-yanque", "luminaria-solar-yanque",
            "pileta-fuente-yanque"]
W, H = 1000, 600


def h32(x: float) -> str:
    return struct.pack(">f", x).hex()


def h64(x: float) -> str:
    return struct.pack(">d", x).hex()


def vbits(p) -> list:
    return [h32(p.x()), h32(p.y()), h32(p.z())]


def cam_rec(c) -> dict:
    return {
        "target": vbits(c.target), "distance": h64(c.distance),
        "yaw": h64(c.yaw), "pitch": h64(c.pitch), "fov": h64(c.fov_deg),
        "aspect": h64(c.aspect), "perspective": c.perspective,
        "two_point": c.two_point,
    }


class Rig:
    """A viewport with its camera calls recorded."""

    def __init__(self, name: str, invert_y: bool, invert_wheel: bool):
        self.vp = Viewport(None)
        vp = self.vp
        vp.resize(W, H)
        vp.camera.set_aspect(W, H)
        vp.flash_status = lambda *a, **k: None
        igz.load_into(vp.scene, ROOT / "examples" / f"{name}.igz")
        vp.scene.version += 1
        # QSettings would bring the developer's preferences in.
        vp._invert_orbit_y = invert_y
        vp._invert_wheel = invert_wheel
        self.calls: list = []
        self._depth = 0
        cam = vp.camera
        for method in ("orbit", "orbit_about", "pan", "zoom", "zoom_to"):
            self._wrap(cam, method)

    def _wrap(self, cam, method: str) -> None:
        real = getattr(cam, method)

        def spy(*args, **kw):
            # Only what the viewport calls: orbit_about calls orbit, and
            # zoom_to calls zoom, themselves.
            if self._depth == 0:
                self.calls.append(self._call_rec(method, args, kw))
            self._depth += 1
            try:
                return real(*args, **kw)
            finally:
                self._depth -= 1
        setattr(cam, method, spy)

    @staticmethod
    def _arg(a):
        if isinstance(a, QVector3D):
            return {"v3": vbits(a)}
        if a is None:
            return {"none": True}
        if isinstance(a, bool):
            return {"bool": a}
        if isinstance(a, int):
            return {"int": a}
        return {"f64": h64(a)}

    def _call_rec(self, method, args, kw) -> dict:
        names = {
            "orbit": ["dx", "dy", "h"],
            "orbit_about": ["pivot", "dx", "dy", "h"],
            "pan": ["dx", "dy", "h", "depth"],
            "zoom": ["steps"],
            "zoom_to": ["steps", "focus", "min_step"],
        }[method]
        rec = {n: self._arg(a) for n, a in zip(names, args)}
        rec.update({k: self._arg(v) for k, v in kw.items()})
        return {"method": method, "args": rec}

    # ---- events ------------------------------------------------------------

    def _mouse(self, kind, x, y, button, buttons, shift):
        mods = Qt.ShiftModifier if shift else Qt.NoModifier
        pos = QPointF(x, y)
        return QMouseEvent(kind, pos, button, buttons, mods)

    def press(self, button, x, y, shift=False):
        b = {"left": Qt.LeftButton, "middle": Qt.MiddleButton}[button]
        self.vp.mousePressEvent(
            self._mouse(QEvent.MouseButtonPress, x, y, b, b, shift))

    def move(self, button, x, y, shift=False):
        b = {"left": Qt.LeftButton, "middle": Qt.MiddleButton,
             "none": Qt.NoButton}[button]
        self.vp.mouseMoveEvent(
            self._mouse(QEvent.MouseMove, x, y, Qt.NoButton, b, shift))

    def release(self, button, x, y, shift=False):
        b = {"left": Qt.LeftButton, "middle": Qt.MiddleButton}[button]
        self.vp.mouseReleaseEvent(
            self._mouse(QEvent.MouseButtonRelease, x, y, b, Qt.NoButton, shift))

    def wheel(self, x, y, delta):
        pos = QPointF(x, y)
        ev = QWheelEvent(pos, pos, QPoint(0, 0), QPoint(0, delta),
                         Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
        self.vp.wheelEvent(ev)

    def zoom_extents(self):
        # MainWindow._on_zoom_extents without the face-me figures and the
        # reference images (they are no part of the scene bounds).
        lo, hi = self.vp.scene.bounds()
        if lo is not None:
            self.vp.camera.fit_box(lo, hi)


def run_case(name: str, seed: int, invert_y=False, invert_wheel=False) -> dict:
    rig = Rig(name, invert_y, invert_wheel)
    rng = random.Random(seed)
    events: list = []

    def record(ev: dict) -> None:
        events.append({"ev": ev, "calls": list(rig.calls),
                       "camera_after": cam_rec(rig.vp.camera)})
        rig.calls.clear()

    def px() -> tuple[int, int]:
        return rng.randrange(0, W), rng.randrange(0, H)

    def centre_px() -> tuple[int, int]:
        return rng.randrange(W // 4, 3 * W // 4), rng.randrange(H // 4, 3 * H // 4)

    def drag(button, shift, steps, start=None):
        x, y = start or centre_px()
        record_ev = {"type": "press", "button": button, "x": x, "y": y,
                     "shift": shift}
        rig.press(button, x, y, shift)
        record(record_ev)
        for _ in range(steps):
            x = max(0, min(W - 1, x + rng.randint(-40, 40)))
            y = max(0, min(H - 1, y + rng.randint(-30, 30)))
            sh = shift if rng.random() > 0.1 else (not shift)
            fx, fy = x + rng.random() * 0.98, y + rng.random() * 0.98
            rig.move(button, fx, fy, sh)
            record({"type": "move", "button": button, "x": fx, "y": fy,
                    "shift": sh})
        rig.release(button, x, y, shift)
        record({"type": "release", "button": button, "x": x, "y": y,
                "shift": shift})

    def ev_zoom_extents():
        rig.zoom_extents()
        record({"type": "zoom_extents"})

    def ev_toggle():
        rig.vp.toggle_projection()
        record({"type": "toggle_projection"})

    def ev_two_point():
        rig.vp.toggle_two_point()
        record({"type": "toggle_two_point"})

    def ev_view(name_):
        rig.vp.camera.set_view(name_)
        record({"type": "set_view", "name": name_})

    def ev_nav(mode):
        rig.vp.set_nav_mode(mode)
        record({"type": "nav_mode", "mode": mode})

    def ev_wheel(x, y, delta):
        rig.wheel(x, y, delta)
        record({"type": "wheel", "x": x, "y": y, "delta": delta})

    ev_zoom_extents()
    # orbits, pans, wheel zooms
    for _ in range(3):
        drag("middle", False, 12)
    for _ in range(2):
        drag("middle", True, 10)
    for _ in range(12):
        x, y = px()
        ev_wheel(x, y, rng.choice([120, -120, 240, -240, 360, 60, -60]))
    x, y = centre_px()
    for _ in range(8):                                    # a burst: cached focus
        ev_wheel(x + rng.randint(-5, 5), y + rng.randint(-5, 5), 120)
    drag("middle", False, 6)                              # pose changes: re-project
    ev_wheel(x, y, -120)
    ev_view("top")
    drag("middle", False, 5)
    ev_view("front")
    ev_wheel(*px(), 120)
    ev_view("iso")
    ev_zoom_extents()
    # parallel projection
    ev_toggle()
    drag("middle", False, 6)
    drag("middle", True, 6)
    for _ in range(6):
        ev_wheel(*px(), rng.choice([120, -120]))
    ev_zoom_extents()
    ev_toggle()
    # two-point perspective
    ev_two_point()
    drag("middle", False, 4)
    ev_wheel(*px(), 120)
    ev_zoom_extents()
    ev_two_point()
    # left-drag navigation modes
    ev_nav("orbit")
    drag("left", False, 8)
    drag("left", True, 6)
    ev_nav("pan")
    drag("left", False, 8)
    ev_nav("zoom")
    drag("left", False, 8)
    ev_nav(None)
    # the pitch clamp: orbit far past the poles
    x, y = W // 2, H // 2
    rig.press("middle", x, y)
    record({"type": "press", "button": "middle", "x": x, "y": y, "shift": False})
    for dy in (400, 400, -900, -900):
        y = max(-2000, min(2000, y + dy))
        rig.move("middle", x, y)
        record({"type": "move", "button": "middle", "x": x, "y": y,
                "shift": False})
    rig.release("middle", x, y)
    record({"type": "release", "button": "middle", "x": x, "y": y, "shift": False})
    ev_zoom_extents()
    # the limits: zoom out to MAX_DISTANCE, then in to MIN_DISTANCE
    for _ in range(90):
        ev_wheel(*px(), -120)
    for _ in range(140):
        ev_wheel(*centre_px(), 120)
    for _ in range(6):                                     # stuck close-up: min_step
        ev_wheel(*px(), -120)
    ev_zoom_extents()
    return {"scene": name, "seed": seed, "w": W, "h": H,
            "invert_orbit_y": invert_y, "invert_wheel": invert_wheel,
            "events": events}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for i, name in enumerate(EXAMPLES):
        cases.append(run_case(name, 100 + i))
    cases.append(run_case("arco-yanque", 200, invert_y=True))
    cases.append(run_case("banca-pergola-yanque", 201, invert_wheel=True))
    path = OUT / "cases.json"
    path.write_text(json.dumps(cases, separators=(",", ":")), encoding="utf-8")
    n = sum(len(c["events"]) for c in cases)
    print(f"{len(cases)} cases, {n} events -> {path}")


if __name__ == "__main__":
    main()
