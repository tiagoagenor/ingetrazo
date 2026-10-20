# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixtures that hold ``ingetrazo-geom``'s hidden-line pipeline to
``core/hlr.py``, bit for bit (docs/rust-rewrite-plan.md, M1.8).

The REAL pure-Python pass (no native engine: INGETRAZO_RUST must be unset)
runs on varied scenes — boxes, stacked and interpenetrating boxes, prisms
with soft side edges, an L-shape, coplanar overlaps, edges on silhouettes,
open and folded soft surfaces, a block of towers, hollow walls under a
section — seen from front/top/side/tilted/rolled cameras, parallel and in
perspective (two-point included). Each case records what the Rust side
takes (the geometry ``collect_geometry`` produced, the camera, the section
plane, the options, the perspective matrices) and what the Python made of
it, every number as the big-endian hex of its bits (f64: 16 digits, f32: 8).

Run it with the MAIN venv and this tree on the path:

    PYTHONPATH=<tree> QT_QPA_PLATFORM=offscreen PYTHONUTF8=1 \\
        <main>/.venv/Scripts/python.exe scripts/golden/make_hlr_fixtures.py

Needs NumPy's OpenBLAS to be the Haswell-class kernel the Rust side mirrors
(see the module docs of ``hlr_draw.rs``). Deleted with the Python app (M10).
"""
from __future__ import annotations

import json
import math
import os
import random
import struct
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.environ.get("INGETRAZO_RUST"):
    sys.exit("unset INGETRAZO_RUST: the fixtures come from the pure-Python path")

import numpy as np
from PySide6.QtGui import QVector3D as V
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rust" / "crates" / "ingetrazo-geom" / "tests" / "data"

_app = QApplication.instance() or QApplication([])

import core._native as _native            # noqa: E402
from core import hlr                      # noqa: E402
from core.camera import OrbitCamera       # noqa: E402
from core.scene import Scene              # noqa: E402
from core.section import SectionPlane     # noqa: E402

assert _native.native is None, "the native engine must be off"


def h64(x: float) -> str:
    return struct.pack(">d", float(x)).hex()


def h32(x: float) -> str:
    return struct.pack(">f", float(x)).hex()


def hx(arr) -> str:
    """A float64 array as one string of 16-digit hex numbers."""
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64)).ravel()
    return "".join(h64(x) for x in a.tolist())


# ── Scenes ──────────────────────────────────────────────────────────────────

def box(m, x0, y0, z0, x1, y1, z1):
    m.add_face([V(x0, y0, z0), V(x1, y0, z0), V(x1, y1, z0), V(x0, y1, z0)])
    m.add_face([V(x0, y0, z1), V(x1, y0, z1), V(x1, y1, z1), V(x0, y1, z1)])
    m.add_face([V(x0, y0, z0), V(x1, y0, z0), V(x1, y0, z1), V(x0, y0, z1)])
    m.add_face([V(x0, y1, z0), V(x1, y1, z0), V(x1, y1, z1), V(x0, y1, z1)])
    m.add_face([V(x0, y0, z0), V(x0, y1, z0), V(x0, y1, z1), V(x0, y0, z1)])
    m.add_face([V(x1, y0, z0), V(x1, y1, z0), V(x1, y1, z1), V(x1, y0, z1)])


def prism(m, cx, cy, z0, z1, r, n, soft_sides=True, phase=0.0):
    """An n-gon prism; the vertical edges are soft (a faceted cylinder)."""
    ring = [(cx + r * math.cos(phase + 2 * math.pi * i / n),
             cy + r * math.sin(phase + 2 * math.pi * i / n)) for i in range(n)]
    m.add_face([V(x, y, z0) for x, y in reversed(ring)])
    m.add_face([V(x, y, z1) for x, y in ring])
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        m.add_face([V(a[0], a[1], z0), V(b[0], b[1], z0),
                    V(b[0], b[1], z1), V(a[0], a[1], z1)])
    if soft_sides:
        for e in m.edges:
            p, q = e.v0.position, e.v1.position
            if abs(p.x() - q.x()) < 1e-6 and abs(p.y() - q.y()) < 1e-6:
                e.soft = True


def lshape(m, z0, z1):
    pts = [(0, 0), (4, 0), (4, 1.5), (1.5, 1.5), (1.5, 4), (0, 4)]
    m.add_face([V(x, y, z0) for x, y in reversed(pts)])
    m.add_face([V(x, y, z1) for x, y in pts])
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        m.add_face([V(a[0], a[1], z0), V(b[0], b[1], z0),
                    V(b[0], b[1], z1), V(a[0], a[1], z1)])


def scene_box():
    s = Scene()
    box(s.mesh, -1, -1.5, 0, 1.2, 1, 2.1)
    return s


def scene_stack():
    s = Scene()
    box(s.mesh, -2, -2, 0, 2, 2, 1)
    box(s.mesh, -1, -1, 1, 1, 1, 2.5)          # stands on the first one
    box(s.mesh, 2, -1, 0, 3.5, 0.5, 3)         # beside it, touching a face
    return s


def scene_overlap():
    s = Scene()
    box(s.mesh, 0, 0, 0, 3, 3, 3)
    box(s.mesh, 1.5, 1.5, 1, 4.5, 4.5, 4)      # interpenetrating
    box(s.mesh, -1, 0.5, 0.5, 1, 2, 2.5)
    return s


def scene_cyl():
    s = Scene()
    prism(s.mesh, 0, 0, 0, 3, 1.2, 16)
    prism(s.mesh, 3.2, 0.5, 0, 1.6, 0.8, 24, phase=0.13)
    box(s.mesh, -0.5, -3, 0, 0.6, -1.8, 2)     # a box in front of them
    return s


def scene_lshape():
    s = Scene()
    lshape(s.mesh, 0, 1.2)
    box(s.mesh, 2.5, 2.5, 0, 3.4, 3.4, 2)
    return s


def scene_coplanar():
    s = Scene()
    m = s.mesh
    m.add_face([V(-4, -4, 0), V(4, -4, 0), V(4, 4, 0), V(-4, 4, 0)])
    m.add_face([V(-1, -1, 0), V(2, -1, 0), V(2, 2, 0), V(-1, 2, 0)])   # coplanar, overlapping
    box(m, -1, -1, 0, 1, 1, 1.5)               # sits on it, shares the plane
    m.add_face([V(0, 0, 1.5), V(2, 0, 1.5), V(2, 2, 1.5), V(0, 2, 1.5)])
    return s


def scene_silhouette():
    s = Scene()
    m = s.mesh
    box(m, -0.2, -2, 0, 0.2, 2, 3)             # a thin wall: edge-on from the sides
    m.add_edge(V(0.2, -2, 3), V(0.2, 2, 3))    # edges lying on its silhouette
    m.add_edge(V(-3, 0, 1.5), V(3, 0, 1.5))    # a bar through it
    m.add_edge(V(-3, 3, 0.5), V(3, 3, 2.5))    # a bar behind it
    m.add_edge(V(0, -3, 0), V(0, -3, 4))       # a vertical in front
    box(m, 1.0, -0.5, 0, 2.0, 0.5, 1.0)
    return s


def scene_open():
    s = Scene()
    m = s.mesh
    m.add_face([V(0, 0, 0), V(3, 0, 0), V(3, 2, 0), V(0, 2, 0)])
    # a folded surface sharing ONE soft edge
    m.add_face([V(0, 4, 0), V(3, 4, 0), V(3, 5, 1), V(0, 5, 1)])
    m.add_face([V(0, 5, 1), V(3, 5, 1), V(3, 6, 0.2), V(0, 6, 0.2)])
    for e in m.edges:
        p, q = e.v0.position, e.v1.position
        if abs(p.y() - 5) < 1e-6 and abs(q.y() - 5) < 1e-6 and abs(p.z() - 1) < 1e-6:
            e.soft = True
    for e in m.edges:
        if len(e.faces) == 1 and abs(e.v0.position.z()) < 1e-6 \
                and abs(e.v0.position.y() - 0) < 1e-6 and abs(e.v1.position.y()) < 1e-6:
            e.soft = True                       # an open boundary, soft
    return s


def scene_city():
    r = random.Random(7)
    s = Scene()
    for i in range(4):
        for j in range(3):
            x, y = i * 2.6, j * 2.9
            box(s.mesh, x, y, 0, x + 2.0 + r.random() * 0.4, y + 2.0,
                1.0 + r.random() * 5.0)
    prism(s.mesh, 5.0, 8.6, 0, 6.5, 0.9, 12)
    return s


def scene_tube():
    s = Scene()
    box(s.mesh, 0, 0, 0, 6, 8, 4)               # outer skin
    box(s.mesh, 1, 0, 1, 5, 8, 3)               # inner skin: a tube
    box(s.mesh, 2, 3, 1.5, 4, 5, 2.5)           # a marker inside the hole
    box(s.mesh, 0.2, 3, 0.2, 0.8, 5, 0.8)       # a marker inside the wall
    return s


def scene_single_edge():
    s = Scene()
    s.mesh.add_edge(V(0, 0, 0), V(2, 1, 3))
    return s


def scene_cyl_section():
    s = Scene()
    prism(s.mesh, 0, 0, 0, 4, 1.5, 20)
    prism(s.mesh, 0, 0, 0.5, 3.5, 0.7, 20)
    box(s.mesh, 3, -1, 0, 4.5, 1, 2)
    return s


SCENES = {
    "box": scene_box, "stack": scene_stack, "overlap": scene_overlap,
    "cyl": scene_cyl, "lshape": scene_lshape, "coplanar": scene_coplanar,
    "silhouette": scene_silhouette, "open": scene_open, "city": scene_city,
    "tube": scene_tube, "single_edge": scene_single_edge,
    "cyl_section": scene_cyl_section,
}

# ── Cameras ─────────────────────────────────────────────────────────────────

VIEWS = {
    "front": (-90.0, 0.0), "right": (0.0, 0.0), "back": (90.0, 0.0),
    "top": (-45.0, 90.0), "top0": (-90.0, 90.0), "bottom": (-90.0, -90.0),
    "iso": (-45.0, 30.0), "tilt1": (-62.3, 17.1), "tilt2": (123.4, -41.7),
    "tilt3": (10.0, 5.0), "tilt4": (-170.5, 61.2),
}


def make_cam(view, target=(0.0, 0.0, 1.0), distance=20.0, rolled=False,
             fov=45.0, aspect=1.6, two_point=False, perspective=False):
    yaw, pitch = VIEWS[view]
    c = OrbitCamera()
    c.target = V(*target)
    c.distance = distance
    c.yaw, c.pitch = math.radians(yaw), math.radians(pitch)
    if rolled:
        c.up = V(0.3, 0.2, 0.93)
    c.fov_deg = fov
    c.aspect = aspect
    c.perspective = perspective
    c.two_point = two_point
    return c


def cam_rec(c):
    return {"target": [h32(c.target.x()), h32(c.target.y()), h32(c.target.z())],
            "distance": h64(c.distance), "yaw": h64(c.yaw), "pitch": h64(c.pitch),
            "up": [h32(c.up.x()), h32(c.up.y()), h32(c.up.z())]}


def basis_rec(c):
    e, r, u, f = hlr.camera_basis(c)
    return hx(np.concatenate([e, r, u, f]))


def geom_rec(tris, hard, soft):
    """The collected geometry as flat hex strings. ``soft`` is the list
    ``collect_geometry`` returns: (p0, p1, na, nb | None)."""
    t = hx(np.asarray(tris, dtype=np.float64).reshape(-1, 9)) if len(tris) else ""
    h = hx(np.asarray(hard, dtype=np.float64).reshape(-1, 6)) if len(hard) else ""
    s = ""
    nb_none = []
    for p0, p1, na, nb in soft:
        s += hx(p0) + hx(p1) + hx(na) + hx(nb if nb is not None else (0.0, 0.0, 0.0))
        nb_none.append(0 if nb is not None else 1)
    return {"tris": t, "hard": h, "soft": s, "soft_open": nb_none}


def arrays_of(tris, hard, soft):
    """What Viewport.hlr_geometry() hands over: (tris, hard, soft, soft_n)."""
    T = np.asarray(tris, dtype=np.float64).reshape(-1, 3, 3)
    H = np.asarray(hard, dtype=np.float64).reshape(-1, 2, 3)
    S = np.array([[p0, p1] for p0, p1, _a, _b in soft],
                 dtype=np.float64).reshape(-1, 2, 3)
    N = np.array([[na, nb if nb is not None else (np.nan,) * 3]
                  for _p0, _p1, na, nb in soft],
                 dtype=np.float64).reshape(-1, 2, 3)
    return T, H, S, N


def section_rec(sp):
    if sp is None:
        return None
    return {"normal": [h32(sp.normal.x()), h32(sp.normal.y()), h32(sp.normal.z())],
            "point": [h32(sp.point.x()), h32(sp.point.y()), h32(sp.point.z())]}


def drawing_rec(d):
    return {"segs": hx(d.segs), "world": hx(d.world),
            "kinds": [int(k) for k in d.kinds],
            "loops": [hx(l) for l in d.loops], "n": len(d.segs)}


def activate(scene, point, normal):
    sp = SectionPlane(V(*point), V(*normal))
    scene.section_planes.append(sp)
    scene.set_active_section(sp)
    return sp


# ── Cases ───────────────────────────────────────────────────────────────────

cases: list = []
#: Geometry shared by many cases is written once: key -> record.
geoms: dict = {}


def put_geom(rec) -> str:
    key = json.dumps(rec, separators=(",", ":"))
    return geoms.setdefault(key, f"g{len(geoms)}")


def drawing_case(name, scene, cam, mode, **opts):
    """One hlr_drawing call. mode: 'collect' (the scene walk) or 'arrays'
    (geometry= as the viewport hands it)."""
    tris, hard, soft = hlr.collect_geometry(scene)
    sp = scene.active_section() if getattr(scene, "show_section_cuts", True) else None
    o = {"profiles": True, "fills": True, "caps": True, "hidden": False}
    o.update(opts)
    if mode == "collect":
        d = hlr.hlr_drawing(scene, cam, **o)
    else:
        d = hlr.hlr_drawing(scene, cam, geometry=arrays_of(tris, hard, soft), **o)
    cases.append({"kind": "drawing", "name": name, "mode": mode,
                  "cam": cam_rec(cam), "basis": basis_rec(cam), "opts": o,
                  "section": section_rec(sp), "geom": put_geom(geom_rec(tris, hard, soft)),
                  "out": drawing_rec(d)})


def view_case(name, scene, cam, mode):
    tris, hard, soft = hlr.collect_geometry(scene)
    sp = scene.active_section()
    if mode == "collect":
        segs, world = hlr.hlr_view(scene, cam, return_world=True)
    else:
        segs, world = hlr.hlr_view(scene, cam, return_world=True,
                                   geometry=arrays_of(tris, hard, soft))
    cases.append({"kind": "view", "name": name, "mode": mode,
                  "cam": cam_rec(cam), "section": section_rec(sp),
                  "geom": put_geom(geom_rec(tris, hard, soft)),
                  "out": {"segs": hx(segs), "world": hx(world), "n": len(segs)}})


def persp_case(name, scene, cam, mode):
    tris, hard, soft = hlr.collect_geometry(scene)
    sp = scene.active_section()
    if mode == "collect":
        segs = hlr.hlr_perspective(scene, cam)
    else:
        segs = hlr.hlr_perspective(scene, cam, geometry=arrays_of(tris, hard, soft))
    view = cam.view_matrix()
    proj = cam.projection_matrix()
    cases.append({"kind": "persp", "name": name, "mode": mode,
                  "cam": cam_rec(cam), "section": section_rec(sp),
                  "view": "".join(h32(x) for x in view.data()),
                  "proj": "".join(h32(x) for x in proj.data()),
                  "fov": h64(cam.fov_deg), "aspect": h64(cam.aspect),
                  "geom": put_geom(geom_rec(tris, hard, soft)),
                  "out": {"segs": hx(segs), "n": len(segs)}})


def clip_case(name, scene, point, normal, split):
    tris, hard, soft = hlr.collect_geometry(scene)
    sp = SectionPlane(V(*point), V(*normal))
    res = hlr.clip_to_section(tris, hard, soft, sp, split_cuts=split)
    ot, oh, os_ = res[0], res[1], res[2]
    cuts = res[3] if split else []
    out = {"tris": hx(np.asarray(ot, dtype=np.float64).reshape(-1, 9)) if ot else "",
           "hard": hx(np.asarray(oh, dtype=np.float64).reshape(-1, 6)) if oh else "",
           "cuts": hx(np.asarray(cuts, dtype=np.float64).reshape(-1, 6)) if len(cuts) else "",
           "soft": geom_rec([], [], os_)["soft"], "soft_open": geom_rec([], [], os_)["soft_open"]}
    cases.append({"kind": "clip", "name": name, "split": split,
                  "section": section_rec(sp), "geom": put_geom(geom_rec(tris, hard, soft)),
                  "out": out})


def main() -> None:
    # 1. every scene from the classic views, both geometry paths.
    for sname, mk in SCENES.items():
        if sname in ("tube", "cyl_section"):
            continue
        s = mk()
        tgt = {"city": (5.0, 4.5, 3.0), "overlap": (2.0, 2.0, 1.5),
               "lshape": (2.0, 2.0, 0.6)}.get(sname, (0.0, 0.0, 1.0))
        for view in ("front", "top", "right", "iso", "tilt1", "tilt2", "bottom"):
            cam = make_cam(view, target=tgt)
            mode = "collect" if view in ("front", "iso", "tilt2", "top") else "arrays"
            drawing_case(f"{sname}/{view}", s, cam, mode)
        # profiles off, hidden lines on, the rolled and the remaining views
        drawing_case(f"{sname}/iso/noprof", s, make_cam("iso", target=tgt), "collect", profiles=False)
        drawing_case(f"{sname}/iso/hidden", s, make_cam("iso", target=tgt), "collect", hidden=True)
        drawing_case(f"{sname}/tilt1/hidden-arrays", s, make_cam("tilt1", target=tgt), "arrays", hidden=True)
        drawing_case(f"{sname}/front/hidden", s, make_cam("front", target=tgt), "collect", hidden=True)
        drawing_case(f"{sname}/top0/hidden", s, make_cam("top0", target=tgt), "arrays", hidden=True)
        drawing_case(f"{sname}/rolled", s, make_cam("tilt3", target=tgt, rolled=True), "collect")
        drawing_case(f"{sname}/tilt4", s, make_cam("tilt4", target=tgt), "arrays")
        view_case(f"{sname}/view-iso", s, make_cam("iso", target=tgt), "collect")
        view_case(f"{sname}/view-tilt3-arrays", s, make_cam("tilt3", target=tgt), "arrays")

    # 2. sections: plans and cross-sections, with and without the cap.
    for sname, mk, plane, views in (
            ("tube", scene_tube, ((0, 2.0, 0), (0, -1, 0)), ("front", "iso", "tilt1")),
            ("tube", scene_tube, ((0, 0, 2.0), (0, 0, 1)), ("top", "iso", "tilt2")),
            ("box", scene_box, ((0, 0, 1.0), (0, 0, 1)), ("top", "front", "iso")),
            ("cyl_section", scene_cyl_section, ((0, 0, 2.0), (0, 0, 1)), ("top", "iso", "tilt1")),
            ("cyl_section", scene_cyl_section, ((0.3, 0, 0), (1, 0, 0.0)), ("right", "iso", "tilt4")),
            ("city", scene_city, ((0, 4.0, 0), (0.2, -1, 0.1)), ("front", "tilt1", "iso")),
            ("overlap", scene_overlap, ((1.5, 1.5, 1.5), (0.6, 0.5, 0.62)), ("iso", "tilt2", "top")),
            ("stack", scene_stack, ((0, 0, 1.7), (0, 0, -1)), ("top", "iso"))):
        for view in views:
            s = mk()
            activate(s, *plane)
            tgt = {"city": (5.0, 4.5, 3.0), "overlap": (2.0, 2.0, 1.5)}.get(sname, (0.0, 3.0, 2.0) if sname == "tube" else (0.0, 0.0, 1.0))
            nm = f"sect/{sname}/{plane[1]}/{view}"
            cam = make_cam(view, target=tgt)
            drawing_case(nm, s, cam, "collect")
            drawing_case(nm + "/nocap", s, cam, "collect", caps=False)
            drawing_case(nm + "/hidden-arrays", s, cam, "arrays", hidden=True)
            if view == "iso":
                drawing_case(nm + "/nofills-noprof", s, cam, "arrays", fills=False, profiles=False)
                view_case(nm + "/view", s, cam, "collect")
        clip_case(f"clip/{sname}/{plane[1]}/split", mk(), *plane, True)
        clip_case(f"clip/{sname}/{plane[1]}/join", mk(), *plane, False)

    # 3. perspective, two-point included, eye inside and beside the model.
    for sname in ("box", "stack", "cyl", "city", "silhouette", "open", "coplanar", "single_edge", "lshape"):
        s = SCENES[sname]()
        tgt = {"city": (5.0, 4.5, 3.0)}.get(sname, (0.0, 0.0, 1.0))
        for view in ("iso", "tilt1", "tilt2", "front"):
            cam = make_cam(view, target=tgt, distance=14.0 if sname != "city" else 28.0,
                           perspective=True)
            persp_case(f"persp/{sname}/{view}", s, cam, "collect" if view == "iso" else "arrays")
        cam = make_cam("tilt1", target=tgt, distance=14.0, perspective=True, two_point=True)
        persp_case(f"persp/{sname}/two-point", s, cam, "arrays")
        cam = make_cam("tilt3", target=tgt, distance=3.0, perspective=True)     # eye inside / near
        persp_case(f"persp/{sname}/near", s, cam, "collect")
        cam = make_cam("tilt2", target=tgt, distance=7.0, perspective=True, fov=70.0, aspect=0.8)
        persp_case(f"persp/{sname}/wide", s, cam, "arrays")
    # perspective cut by a section plane
    s = scene_tube()
    activate(s, (0, 2.0, 0), (0, -1, 0))
    persp_case("persp/tube/section", s, make_cam("tilt1", target=(0, 3, 2), distance=18.0, perspective=True), "collect")
    persp_case("persp/tube/section-arrays", s, make_cam("front", target=(0, 3, 2), distance=18.0, perspective=True), "arrays")
    s = scene_cyl_section()
    activate(s, (0, 0, 2.0), (0, 0, 1))
    persp_case("persp/cyl/section", s, make_cam("iso", target=(0, 0, 1), distance=16.0, perspective=True), "collect")
    # a camera that looks away from everything (w_t <= 0 branch)
    c = make_cam("front", target=(0, 0, 1), distance=0.05, perspective=True)
    persp_case("persp/box/tiny-distance", scene_box(), c, "arrays")

    # random scenes and cameras for the long tail
    r = random.Random(2026)
    for k in range(10):
        s = Scene()
        for _ in range(r.randint(2, 6)):
            x, y = r.uniform(-4, 4), r.uniform(-4, 4)
            box(s.mesh, x, y, r.uniform(0, 1), x + r.uniform(0.4, 3), y + r.uniform(0.4, 3), r.uniform(1.2, 4))
        for _ in range(r.randint(0, 3)):
            s.mesh.add_edge(V(r.uniform(-5, 5), r.uniform(-5, 5), r.uniform(0, 4)),
                            V(r.uniform(-5, 5), r.uniform(-5, 5), r.uniform(0, 4)))
        if r.random() < 0.4:
            prism(s.mesh, r.uniform(-3, 3), r.uniform(-3, 3), 0, r.uniform(1, 3), r.uniform(0.5, 1.5), r.choice([8, 12, 20]))
        cam = OrbitCamera()
        cam.target = V(r.uniform(-1, 1), r.uniform(-1, 1), r.uniform(0, 2))
        cam.distance = r.uniform(8, 30)
        cam.yaw, cam.pitch = r.uniform(-math.pi, math.pi), r.uniform(-1.5, 1.5)
        cam.aspect = r.uniform(0.6, 2.0)
        sec = r.random() < 0.4
        if sec:
            n = V(r.uniform(-1, 1), r.uniform(-1, 1), r.uniform(-1, 1))
            activate(s, (r.uniform(-1, 1), r.uniform(-1, 1), r.uniform(0.5, 2.5)), (n.x(), n.y(), n.z()))
        drawing_case(f"rand{k}", s, cam, r.choice(["collect", "arrays"]),
                     hidden=r.random() < 0.5, profiles=r.random() < 0.8)
        cam.perspective = True
        persp_case(f"rand{k}/persp", s, cam, r.choice(["collect", "arrays"]))

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "hlr_pipeline.jsonl"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for key, gid in geoms.items():
            f.write(json.dumps({"kind": "geom", "id": gid, "geom": json.loads(key)},
                               separators=(",", ":")) + "\n")
        for c in cases:
            f.write(json.dumps(c, separators=(",", ":")) + "\n")
    print(f"{len(cases)} cases, {path.stat().st_size / 1e6:.2f} MB -> {path}")
    segs = sum(c["out"].get("n", 0) for c in cases)
    print(f"{segs} output segments")


if __name__ == "__main__":
    main()
