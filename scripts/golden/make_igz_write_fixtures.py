# SPDX-License-Identifier: GPL-3.0-or-later
"""Write the .igz fixtures that hold the Rust WRITER to Python's bytes
(docs/rust-rewrite-plan.md, M2.8).

Each scene is built with the real app code and saved with
``formats.igz.save_scene`` in every format it allows, under the size
thresholds that force each way of writing it:

    f1    format 1, bare indented JSON (default thresholds)
    f2    format 2, ZIP with textures (default thresholds)
    f3    format 3, every mesh a binary block (_LARGE_DOCUMENT=0, _BULK_MIN=1)
    fast  a "large" document with no binary block: compact JSON, level 1
          (_LARGE_DOCUMENT=0, _BULK_MIN huge)
    mix   format 3 where only the bigger meshes go binary (_BULK_MIN=60)

The suffix of a fixture's name tells the Rust test which thresholds to
write it with. Group ids are fixed (a random uid would change every run).
The files are committed, so `cargo test` needs no Python; rerun this only
to widen them (PYTHONPATH at the repository root):

    python scripts/golden/make_igz_write_fixtures.py

Deleted with the Python app at M10.
"""
from __future__ import annotations

import os
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QGuiApplication, QImage, QMatrix4x4, QVector3D as V  # noqa: E402

from core.dimension import Dimension  # noqa: E402
from core.group import Group  # noqa: E402
from core.guide import Guide  # noqa: E402
from core.image_plane import ImagePlane  # noqa: E402
from core.materials import Material  # noqa: E402
from core.mesh import Mesh  # noqa: E402
from core.saved_views import SavedView  # noqa: E402
from core.scene import Scene  # noqa: E402
from core.section import SectionPlane  # noqa: E402
from core.style import Style  # noqa: E402
from core.sun import ShadowSettings  # noqa: E402
from core.textlabel import TextLabel  # noqa: E402
from formats import igz  # noqa: E402
from georef.datum import SceneDatum  # noqa: E402
from georef.geopath import GeoPath  # noqa: E402
from georef.points import GeoPoint  # noqa: E402
from georef.tiles import PRESETS, TileLayer  # noqa: E402
import make_igz_format_fixtures as base  # noqa: E402

OUT = (ROOT / "rust" / "crates" / "ingetrazo-model" / "tests" / "data"
       / "igz_write")

#: suffix -> (_LARGE_DOCUMENT, _BULK_MIN); None = the module's defaults
THRESHOLDS = {"f1": None, "f2": None, "f3": (0, 1), "fast": (0, 10 ** 9),
              "mix": (0, 60)}


def png(tmp: Path, name: str, color: int, size: int = 4) -> Path:
    path = tmp / name
    img = QImage(size, size, QImage.Format_RGB32)
    img.fill(color)
    img.setPixel(1, 2, 0x336699)
    assert img.save(str(path))
    return path


def fix_uids(scene: Scene) -> None:
    n = [0]

    def walk(g):
        n[0] += 1
        g.uid = f"{n[0]:016x}"
        for c in g.children or ():
            walk(c)
    for g in list(scene.groups) + list(scene.kept_components):
        walk(g)


def scene_empty(tmp) -> Scene:
    return Scene()


def scene_everything(tmp) -> Scene:
    """Every optional block, every group field."""
    s = Scene()
    m = s.mesh
    wall = m.add_face([V(0, 0, 0), V(4, 0, 0), V(4, 0, 2.5), V(0, 0, 2.5)])
    wall.attrs.update({"mat": "Ladrillo", "layer": "Muros",
                       "ifc": {"class": "IfcWall", "name": "M1"}})
    m.add_face([V(0, 0, 0), V(4, 0, 0), V(4, 3, 0), V(0, 3, 0)]).attrs[
        "opacity"] = 1
    m.add_edge(V(9, 9, 0), V(9, 9, 2)).layer = "Muros"
    # layers (one hidden, one locked) -> the block is written
    from core.layers import Layer
    s.layers = [Layer("Layer 0"), Layer("Muros"),
                Layer("Cotas", visible=False), Layer("Fijos", locked=True)]
    s.saved_views = [
        SavedView("Planta", target=(1.5, 2.5, 0.0), distance=30.0, yaw=0.0,
                  pitch=1.5707963267948966, perspective=False),
        SavedView("Perspectiva", target=(0, 0, 1), hidden_layers=["Cotas"],
                  two_point=True, fov_deg=60.0)]
    from core.composition import Composicion
    s.compositions = [Composicion(name="Lámina A", paper="A3"),
                      Composicion(name="Lámina B", landscape=False,
                                  border=True, guides_v=[10.0, 20.5])]
    d1 = Dimension(V(0, 0, 0), V(4, 0, 0), V(0, -1, 0), layer="Cotas",
                   text="4.00 <>", axis="x")
    d1.bind(s)
    d2 = Dimension(V(0, 0, 0), V(0, 3, 0), V(-1, 0, 0))
    d2.bind(s)
    s.dimensions = [d1, d2]
    s.materials["Ladrillo"] = Material("Ladrillo", color=(0.6, 0.25, 0.2),
                                       opacity=1.0)
    s.materials["Vidrio"] = Material("Vidrio", color=(0.7, 0.85, 0.95),
                                     opacity=0.35, finish="glass")
    s.dimension_style.update({"decimals": 3, "ends": "tick", "norma": "din"})
    s.camera_home = {"target": [1.5, 1.5, 1.0], "distance": 12.5,
                     "yaw": 0.6, "pitch": 0.4}
    s.units = {"length": "ft", "precision": 3}
    s.plugin_data = {"myext": {"n": 1, "xs": [1, 2.5, "ñ"]}, "other": "✓"}
    s.custom_scales = [0.5, 25.0, 100.0]
    s.back_face_color = (0.8, 0.75, 0.7)
    s.section_planes = [
        SectionPlane(V(0, 1, 0), V(0, 1, 0), name="A-A", symbol="A",
                     active=True, uid=1),
        SectionPlane(V(2, 0, 0), V(1, 0, 0), name="B-B", symbol="B",
                     uid=2, hidden=True)]
    s.show_section_planes = False
    s.show_section_cuts = False
    s.show_hidden_objects = True
    s.show_hidden_geometry = True
    st = Style()
    st.face_mode = "monochrome"
    st.edge_color = (0.0, 0.5, 1.0)
    st.back_color = (0.2, 0.3, 0.4)
    s.display_style = st
    s.shadows = ShadowSettings(enabled=True, month=12, day=21, hour=15,
                               minute=30, darkness=0.4, utc_offset=-5)
    s.georef = SceneDatum(-13.5, -71.97, 3400.0, north=12.5)
    tl = TileLayer(PRESETS[sorted(PRESETS)[0]], zoom=17, radius_m=500.0)
    tl.opacity = 0.6
    s.tile_layer = tl
    s.geo_paths = [GeoPath([V(0, 0, 0), V(10, 5, 0.5), V(20, 0, 1)],
                           closed=True, name="Camino", surface="flat")]
    s.geo_points = [GeoPoint(V(1, 2, 3), name="Hito", desc="Esquina ñ"),
                    GeoPoint(V(4, 5, 6))]
    s.text_labels = [TextLabel(V(1, 1, 0), V(0.5, 0.5, 1), "Muro ñandú"),
                     TextLabel(V(2, 2, 0), V(0, 0, 1), "Capa", layer="Muros")]
    s.guides = [Guide(V(0, 0, 0), V(1, 0, 0)), Guide(V(2, 2, 0)),
                Guide(V(1, 1, 1), V(0, 1, 1))]
    # groups with every field
    proto = Mesh()
    proto.add_face([V(0, 0, 0), V(1, 0, 0), V(1, 1, 0), V(0, 1, 0)])
    a = Group(proto, name="Silla")
    a.xform = base.xform(3, 4, 0, 30.0)
    a.layer = "Muros"
    a.ifc = {"class": "IfcFurniture", "name": "S1"}
    a.billboard = "mesh"
    a.material = {"color": [0.1, 0.2, 0.3]}
    a.text3d = {"text": "Hola", "height": 0.5}
    a.ext = {"plug": {"k": [1, 2]}}
    a.hidden = True
    a.locked = True
    a.exploded = {"axis": "z", "amount": 2.0}
    a.explode_offset = (0.0, 0.0, 1.5)
    a.axes = base.xform(1, 1, 1)
    b = Group(proto, name="Silla 2")
    b.xform = base.xform(6, 4, 0)
    b.component = False
    classic = Group(name="Mesa ñ")
    classic.mesh.add_face([V(0, 30, 0), V(2, 30, 0), V(2, 31, 0)])
    classic.axes = base.xform(0, 30, 0)
    s.groups = [a, b, classic]
    kept = Group(proto, name="Guardado")
    kept.xform = QMatrix4x4()
    s.kept_components = [kept]
    return s


def scene_textures(tmp) -> Scene:
    """Textures everywhere they can hang, shared and colourised, with a
    non-ASCII file name; an image plane and a textured material."""
    s = Scene()
    m = s.mesh
    wood = png(tmp, "wood.png", 0x8B5A2B)
    tinted = png(tmp, "wood tinted.png", 0xAA7744, 6)
    nandu = png(tmp, "ñandú 日本.png", 0x44AA77, 5)
    f = m.add_face([V(0, 0, 0), V(2, 0, 0), V(2, 2, 0), V(0, 2, 0)])
    f.attrs["texture"] = {"path": str(tinted), "base": str(wood),
                          "sw": 1.0, "sh": 1.0}
    f.attrs["back"] = {"texture": {"path": str(nandu), "sw": 2.0, "sh": 2.0},
                       "color": [0.5, 0.5, 0.5]}
    g = m.add_face([V(3, 0, 0), V(5, 0, 0), V(5, 2, 0)])
    g.attrs["texture"] = {"path": str(wood), "sw": 0.5, "sh": 0.5}
    g.attrs["mat"] = "Madera"
    # a texture whose file is gone keeps its path, counted as missing
    h = m.add_face([V(6, 0, 0), V(8, 0, 0), V(8, 2, 0)])
    h.attrs["texture"] = {"path": str(tmp / "gone.png"), "sw": 1.0, "sh": 1.0}
    s.materials["Madera"] = Material("Madera", texture={
        "path": str(wood), "sw": 0.5, "sh": 0.5}, opacity=0.9)
    s.image_planes = [ImagePlane(str(nandu), V(0, 5, 0), V(4, 0, 0),
                                 V(0, 3, 0), aspect=0.75, name="Plano ñ",
                                 opacity=0.8)]
    proto = Mesh()
    pf = proto.add_face([V(0, 0, 0), V(1, 0, 0), V(1, 1, 0)])
    pf.attrs["texture"] = {"path": str(wood), "sw": 1.0, "sh": 1.0}
    grp = Group(proto, name="Con textura")
    grp.xform = base.xform(10, 10, 0)
    s.groups = [grp]
    return s


def scene_grid(tmp) -> Scene:
    """Enough geometry to make deflate work: a jittered quad grid, a
    component placed many times, edges with curves and layers."""
    rng = random.Random(8)
    s = Scene()
    m = s.mesh
    n = 16
    z = [[round(rng.uniform(0, 2), 3) for _ in range(n + 1)]
         for _ in range(n + 1)]
    for i in range(n):
        for j in range(n):
            f = m.add_face([V(i, j, z[i][j]), V(i + 1, j, z[i + 1][j]),
                            V(i + 1, j + 1, z[i + 1][j + 1]),
                            V(i, j + 1, z[i][j + 1])])
            if (i + j) % 7 == 0:
                f.attrs["color"] = [0.2, 0.4 + 0.01 * i, 0.6]
            if (i * j) % 11 == 0:
                f.attrs["layer"] = "Zona %d" % (i % 3)
    for k in range(40):
        e = m.add_edge(V(-3, k * 0.37, 0.1), V(-2.5, k * 0.37 + 0.2, 0.3))
        e.curve = k // 8
        if k % 5 == 0:
            e.soft = True
        if k % 9 == 0:
            e.layer = "Curvas"
    proto = Mesh()
    for i in range(6):
        proto.add_face([V(i, 0, 0), V(i + 1, 0, 0), V(i + 1, 1, 0.5),
                        V(i, 1, 0.5)],
                       [[V(i + .2, .2, .1), V(i + .5, .2, .1),
                         V(i + .5, .5, .2)]] if i % 2 else None)
    for k in range(30):
        g = Group(proto, name=f"Pieza {k}")
        g.xform = base.xform(k * 2.5, 30, rng.uniform(0, 1), 7.0 * k)
        s.groups.append(g)
    return s


# name -> (builder, formats)
SCENES = {
    "empty": (scene_empty, ("f1", "fast")),
    "curves": (base.scene_curves, ("f1", "f3", "fast", "mix")),
    "faces": (base.scene_faces, ("f1", "f3", "fast", "mix")),
    "groups": (base.scene_groups, ("f1", "f3", "fast", "mix")),
    "everything": (scene_everything, ("f1", "f3", "fast", "mix")),
    "textures": (scene_textures, ("f2", "f3", "fast", "mix")),
    "grid": (scene_grid, ("f1", "f3", "fast", "mix")),
}


def write(name: str, make, suffix: str, tmp: Path) -> None:
    scene = make(tmp)
    fix_uids(scene)
    path = OUT / f"{name}.{suffix}.igz"
    saved = (igz._LARGE_DOCUMENT, igz._BULK_MIN)
    limits = THRESHOLDS[suffix]
    if limits:
        igz._LARGE_DOCUMENT, igz._BULK_MIN = limits
    try:
        igz.save_scene(scene, path)
    finally:
        igz._LARGE_DOCUMENT, igz._BULK_MIN = saved
    raw = path.read_bytes()
    doc, _archive = igz._read_document(path)
    print(f"{path.name}: {len(raw)} bytes, format {doc['igz_format']}"
          f"{' (container)' if raw.startswith(b'PK') else ''}")


def resave_examples(tmp: Path) -> None:
    """Shipped documents re-saved by the current writer (the committed ones
    come from older builds): the loose-mesh heavy ones become format 3."""
    for stem in ("luminaria-solar-yanque", "banca-pergola-yanque"):
        scene = Scene()
        igz.load_into(scene, ROOT / "examples" / f"{stem}.igz")
        fix_uids(scene)
        path = OUT / f"example-{stem}.f2.igz"
        igz.save_scene(scene, path)
        print(f"{path.name}: {path.stat().st_size} bytes")


def main() -> None:
    app = QGuiApplication.instance() or QGuiApplication(sys.argv)
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.igz"):
        old.unlink()
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        for name, (make, suffixes) in SCENES.items():
            for suffix in suffixes:
                write(name, make, suffix, tmp)
        resave_examples(tmp)
    del app


if __name__ == "__main__":
    main()
