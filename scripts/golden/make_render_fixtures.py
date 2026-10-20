# SPDX-License-Identifier: GPL-3.0-or-later
"""Reference images for ``ingetrazo-render`` (docs/rust-rewrite-plan.md, M4.1).

Renders a handful of scenes with the REAL Python viewport (its OpenGL
pipeline, MSAA 4x) at 400x300 and writes, per case, the scene as an .igz, the
camera and style as JSON and the picture as a PNG. The Rust renderer draws
the same scene from the same camera in ``tests/parity.rs`` and the two are
compared with the corpus' ``render_pixels`` rule (golden/tolerances.toml).

OpenGL needs a real window system: on Windows the script uses the native
"windows" platform with a fully transparent window parked off screen (the
"offscreen" platform has no GL there); elsewhere the usual offscreen one.

Run it with the MAIN venv and this tree on the path, INGETRAZO_RUST unset:

    PYTHONPATH=<tree> PYTHONUTF8=1 <main>/.venv/Scripts/python.exe \\
        scripts/golden/make_render_fixtures.py

Deleted with the Python app (M10); the fixtures stay as files.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    os.environ["QT_QPA_PLATFORM"] = "windows"
else:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.environ.get("INGETRAZO_RUST"):
    sys.exit("unset INGETRAZO_RUST: the fixtures come from the pure-Python path")

from PySide6.QtGui import QColor, QImage, QMatrix4x4, QVector3D as V
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rust" / "crates" / "ingetrazo-render" / "tests" / "data"
OUT.mkdir(parents=True, exist_ok=True)

_app = QApplication.instance() or QApplication([])

from core.group import Group                      # noqa: E402
from core.mesh import Mesh                        # noqa: E402
from core.section import SectionPlane             # noqa: E402
from core.style import style_by_name              # noqa: E402
from formats.igz import save_scene                # noqa: E402
from views.main_window import MainWindow          # noqa: E402

W, H = 400, 300


# ---- scene helpers ------------------------------------------------------------------

def box(mesh, x, y, z, sx, sy, sz, color=None, **attrs):
    """An axis-aligned box, faces wound outward."""
    p = [V(x, y, z), V(x + sx, y, z), V(x + sx, y + sy, z), V(x, y + sy, z),
         V(x, y, z + sz), V(x + sx, y, z + sz), V(x + sx, y + sy, z + sz),
         V(x, y + sy, z + sz)]
    for ring in ([0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5],
                 [2, 3, 7, 6], [3, 0, 4, 7]):
        f = mesh.add_face([p[i] for i in ring])
        if color is not None:
            f.attrs["color"] = tuple(color)
        f.attrs.update(attrs)


def prism(mesh, sides, radius, height, center=(0, 0, 0), color=None):
    """A prism whose side edges are soft (a curved surface)."""
    cx, cy, cz = center
    ring = [V(cx + radius * math.cos(2 * math.pi * i / sides),
              cy + radius * math.sin(2 * math.pi * i / sides), cz)
            for i in range(sides)]
    top = [V(p.x(), p.y(), cz + height) for p in ring]
    faces = [mesh.add_face(list(reversed(ring))), mesh.add_face(top)]
    for i in range(sides):
        j = (i + 1) % sides
        faces.append(mesh.add_face([ring[i], ring[j], top[j], top[i]]))
    for e in mesh.edges:
        if abs(e.v0.position.z() - e.v1.position.z()) > 1e-6:
            e.soft = True
    if color is not None:
        for f in faces:
            f.attrs["color"] = tuple(color)


def camera(target, distance, yaw_deg, pitch_deg, perspective=True,
           two_point=False):
    return {"target": list(target), "distance": distance,
            "yaw": math.radians(yaw_deg), "pitch": math.radians(pitch_deg),
            "perspective": perspective, "two_point": two_point}


def checker(path: Path) -> None:
    img = QImage(64, 64, QImage.Format_RGB32)
    for y in range(64):
        for x in range(64):
            on = ((x // 8) + (y // 8)) % 2 == 0
            img.setPixelColor(x, y, QColor(200, 60, 40) if on
                              else QColor(240, 230, 200))
    img.save(str(path))


# ---- the cases ---------------------------------------------------------------------------

def scene_blocks(sc):
    box(sc.mesh, 0, 0, 0, 2, 2, 2, (0.8, 0.45, 0.3))
    box(sc.mesh, 3, 0.5, 0, 1.5, 3, 1)
    box(sc.mesh, -2.5, -1, 0, 1, 1, 3.5, (0.35, 0.55, 0.8))


def scene_curved(sc):
    prism(sc.mesh, 24, 1.5, 2.5, (0, 0, 0), (0.7, 0.75, 0.8))
    box(sc.mesh, 2.5, -0.5, 0, 1, 1, 1)


def scene_components(sc):
    proto = Mesh()
    box(proto, -0.5, -0.5, 0, 1, 1, 1.4, (0.75, 0.6, 0.4))
    box(proto, -0.2, -0.2, 1.4, 0.4, 0.4, 0.6, (0.4, 0.5, 0.7))
    mats = []
    m = QMatrix4x4(); m.translate(-3, 0, 0); mats.append(m)
    m = QMatrix4x4(); m.translate(0, 0, 0); m.rotate(35, 0, 0, 1); mats.append(m)
    m = QMatrix4x4(); m.translate(3, 0, 0); m.scale(1.0, 1.0, 1.4); mats.append(m)
    m = QMatrix4x4(); m.translate(0, 3, 0); m.scale(-1.0, 1.0, 1.0); mats.append(m)
    for i, mt in enumerate(mats):
        g = Group(proto, f"inst{i}")
        g.xform = mt
        sc.groups.append(g)
    # a classic group too
    gm = Mesh()
    box(gm, 4.5, 3, 0, 1, 1, 1, (0.5, 0.7, 0.5))
    sc.groups.append(Group(gm, "classic"))


def scene_translucent(sc):
    box(sc.mesh, 0, 0, 0, 1.2, 1.2, 1.2, (0.85, 0.3, 0.3))
    box(sc.mesh, -1, -1, 0, 3.2, 3.2, 2, (0.6, 0.8, 0.95), opacity=0.4)


def scene_back_faces(sc):
    f = sc.mesh.add_face([V(-2, 0, 0), V(2, 0, 0), V(2, 0, 2), V(-2, 0, 2)])
    f.attrs["color"] = (0.9, 0.8, 0.3)
    g = sc.mesh.add_face([V(-2, 1, 0), V(2, 1, 0), V(2, 1, 2), V(-2, 1, 2)])
    g.attrs["color"] = (0.4, 0.8, 0.5)
    g.attrs["back"] = True


def scene_textured(sc):
    tex = OUT / "checker.png"
    checker(tex)
    f = sc.mesh.add_face([V(-2, 0, 0), V(2, 0, 0), V(2, 0, 3), V(-2, 0, 3)])
    f.attrs["texture"] = {"path": str(tex), "sw": 1.0, "sh": 1.0}
    g = sc.mesh.add_face([V(-2, 0, 0), V(2, 0, 0), V(2, 3, 0), V(-2, 3, 0)])
    g.attrs["texture"] = {"path": str(tex), "sw": 2.0, "sh": 2.0, "rot": 30.0}
    box(sc.mesh, 3, 1, 0, 1, 1, 1, (0.6, 0.6, 0.9))


def scene_section(sc):
    box(sc.mesh, 0, 0, 0, 3, 3, 2, (0.8, 0.7, 0.55))
    box(sc.mesh, 4, 0, 0, 1.5, 1.5, 1.5, (0.55, 0.7, 0.8))
    sp = SectionPlane(V(0, 1.5, 0), V(0, 1, 0), "A", "A", active=True)
    sc.section_planes.append(sp)


def scene_arco(sc):
    from formats.igz import load_into
    load_into(sc, ROOT / "examples" / "arco-yanque.igz")


def fit_camera(sc, yaw_deg, pitch_deg):
    """Frame the scene the way Zoom Extents does."""
    from core.camera import OrbitCamera
    lo, hi = sc.bounds()
    c = OrbitCamera()
    c.aspect = W / H
    c.fit_to(lo, hi)
    return camera((c.target.x(), c.target.y(), c.target.z()), c.distance,
                  yaw_deg, pitch_deg)


def scene_guides(sc):
    from core.guide import Guide
    box(sc.mesh, 0, 0, 0, 2, 2, 2, (0.8, 0.45, 0.3))
    sc.guides.append(Guide(V(0, 4, 0), V(1, 0, 0)))
    sc.guides.append(Guide(V(-3, 0, 0), V(0, 1, 0)))


STYLES = ["Architectural", "Shaded", "Hidden line", "Monochrome", "Wireframe",
          "X-ray"]
ISO = camera((0.5, 0.5, 1.0), 14.0, -45, 30)

CASES = []   # (id, builder, style name, camera dict)
for st in STYLES:
    CASES.append((f"blocks_{st.lower().replace(' ', '_').replace('-', '')}",
                  scene_blocks, st, ISO))
CASES += [
    ("blocks_default_sky", scene_blocks, "Default",
     camera((0.5, 0.5, 1.0), 16.0, -35, 12)),
    ("blocks_parallel_top", scene_blocks, "Architectural",
     camera((0.5, 0.5, 1.0), 14.0, -90, 90, perspective=False)),
    ("blocks_two_point", scene_blocks, "Architectural",
     camera((0.5, 0.5, 1.0), 14.0, -45, 20, two_point=True)),
    ("curved_profiles", scene_curved, "Architectural",
     camera((0.8, 0, 1.2), 11.0, -50, 25)),
    ("curved_wireframe", scene_curved, "Wireframe",
     camera((0.8, 0, 1.2), 11.0, -50, 25)),
    ("components", scene_components, "Architectural",
     camera((1.5, 1.5, 0.7), 16.0, -40, 30)),
    ("components_xray", scene_components, "X-ray",
     camera((1.5, 1.5, 0.7), 16.0, -40, 30)),
    ("translucent", scene_translucent, "Architectural",
     camera((0.6, 0.6, 1.0), 8.0, -40, 28)),
    ("back_faces", scene_back_faces, "Architectural",
     camera((0, 0.5, 1.0), 8.0, 125, 15)),
    ("back_faces_monochrome", scene_back_faces, "Monochrome",
     camera((0, 0.5, 1.0), 8.0, 125, 15)),
    ("textured", scene_textured, "Architectural",
     camera((0.5, 1.0, 1.0), 10.0, -50, 30)),
    ("textured_shaded", scene_textured, "Shaded",
     camera((0.5, 1.0, 1.0), 10.0, -50, 30)),
    ("section", scene_section, "Architectural",
     camera((2.5, 1.5, 0.8), 11.0, -60, 35)),
    ("arco_yanque", scene_arco, "Default", lambda sc: fit_camera(sc, -40, 25),
     "examples/arco-yanque.igz"),
    ("arco_yanque_shaded", scene_arco, "Shaded", lambda sc: fit_camera(sc, 40, 25),
     "examples/arco-yanque.igz"),
    ("guides_axes", scene_guides, "Default",
     camera((0.5, 0.5, 1.0), 12.0, -45, 30)),
]


def shot(vp, path: Path) -> None:
    img = vp.render_image(W, H, overlays=False)
    # The FBO comes back as premultiplied ARGB and translucent faces blend
    # its alpha too: reading the same bytes as RGB32 keeps the colour GL
    # wrote and ignores the alpha, where converting or saving the ARGB would
    # divide the colour by it (and clip it).
    img = QImage(img.constBits(), img.width(), img.height(),
                 img.bytesPerLine(), QImage.Format_RGB32).copy()
    img.save(str(path))


def main() -> None:
    win = MainWindow()
    win.setWindowOpacity(0.0)
    win.move(-4000, -4000)
    win.show()
    _app.processEvents()
    vp = win.viewport
    if getattr(vp, "_gl", None) is None or not vp.isValid():
        sys.exit("no OpenGL context on this platform")
    info = dict(vp.gl_info)
    meta = {"size": [W, H], "view_height": vp.height(), "gl": info,
            "cases": []}
    for cid, build, style_name, cam, *src in CASES:
        sc = vp.scene
        sc.clear()
        build(sc)
        if callable(cam):
            cam = cam(sc)
        sc.display_style = style_by_name(style_name)
        sc.version += 1
        c = vp.camera
        c.target = V(*cam["target"])
        c.distance = cam["distance"]
        c.yaw, c.pitch = cam["yaw"], cam["pitch"]
        c.perspective = cam["perspective"]
        c.two_point = cam["two_point"]
        c.aspect = W / H
        _app.processEvents()
        shot(vp, OUT / f"{cid}.png")
        # The same picture with no line work: no edges, no profiles, and a
        # style override, which also leaves out the axes and the guides.
        # Faces, backdrop, translucency, section fill and cut are then drawn
        # by triangles only, which every GPU rasterises the same way.
        faces_only = sc.display_style.face_mode != "wireframe"
        if faces_only:
            fs = sc.display_style.copy()
            fs.edges = False
            fs.profiles = False
            fs.back_edges = False
            vp.style_override = fs
            try:
                shot(vp, OUT / f"{cid}.faces.png")
            finally:
                vp.style_override = None
        if src:
            igz_rel = src[0]          # a real document of the repository
        else:
            save_scene(sc, OUT / f"{cid}.igz")
            igz_rel = None
        # The matrices the viewport hands to GL (QMatrix4x4.data(), column
        # major): the Rust camera must reproduce them to the last bit.
        mvp = (c.projection_matrix() * c.view_matrix()).data()
        view = c.view_matrix().data()
        proj = c.projection_matrix().data()
        meta["cases"].append({"id": cid, "style": style_name, "camera": cam,
                              "igz": igz_rel,
                              "faces_only": faces_only,
                              "mvp": [float(x) for x in mvp],
                              "view": [float(x) for x in view],
                              "projection": [float(x) for x in proj]})
        print(cid, W, H)
    (OUT / "render_cases.json").write_text(
        json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    win._saved_version = vp.scene.version
    win.close()


if __name__ == "__main__":
    main()
