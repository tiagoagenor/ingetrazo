# SPDX-License-Identifier: GPL-3.0-or-later
"""Hand-written op scripts for the annotation commands (docs/rust-rewrite-plan.md,
M3.9), run through the real Python app.

The harvested corpus compares the L2 signature of the scene after each step,
and that signature holds only the COUNT of the dimensions, labels, guides,
section planes, image planes and geo paths. These scripts also record the
``document.json`` the Python app writes after every step (``"doc"``), so the
Rust replay compares everything the commands touch: where each dimension end
reads after the geometry moved, the offset of a leader, a guide's direction,
a plane's pose and fill, an image's corner, a path's nodes, the datum.

    <main>/.venv/Scripts/python.exe -X utf8 scripts/golden/make_annotation_scripts.py

(run from this tree's root; INGETRAZO_RUST must be unset). Writes
``golden/ops/hand-annotations-*.json`` and the start documents they open under
``golden/starts``. Deleted with the Python app (M10).
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import sys
import tempfile
import zipfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.environ.get("INGETRAZO_RUST"):
    sys.exit("unset INGETRAZO_RUST: the scripts come from the pure-Python path")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from PySide6.QtGui import QMatrix4x4, QVector3D   # noqa: E402
from PySide6.QtWidgets import QApplication        # noqa: E402

_app = QApplication.instance() or QApplication([])

from core.group import Group                       # noqa: E402
from core.history import History                   # noqa: E402
from core.mesh import Mesh                         # noqa: E402
from core.scene import Scene                       # noqa: E402
from formats import igz                            # noqa: E402
from tests.golden import recorder                  # noqa: E402
from tests.golden.signature import scene_signature, signature_hash   # noqa: E402

GOLDEN = ROOT / "golden"


def V(x, y, z=0.0):
    return QVector3D(float(x), float(y), float(z))


def v(x, y, z=0.0):
    return {"$v": [float(x), float(y), float(z)]}


def ref(name, i):
    return {"$ref": [name, i]}


def cube(mesh, x0, y0, z0, s):
    x1, y1, z1 = x0 + s, y0 + s, z0 + s
    for quad in (
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
    ):
        mesh.add_face([V(*p) for p in quad])


def instance(name, square, xform):
    g = Group(Mesh(), name)
    g.uid = hashlib.sha1(name.encode()).hexdigest()[:16]   # the start document must not change from run to run
    g.mesh.add_face([V(*p) for p in square])
    g.xform = xform
    return g


def placed(tx, ty, tz=0.0, deg=0.0):
    m = QMatrix4x4()
    m.translate(tx, ty, tz)
    if deg:
        m.rotate(deg, 0.0, 0.0, 1.0)
    return m


# ---- running a script in Python ---------------------------------------------------

def document_text(scene) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "d.igz"
        igz.save_scene(scene, path)
        raw = path.read_bytes()
    if zipfile.is_zipfile(io.BytesIO(raw)):
        return zipfile.ZipFile(io.BytesIO(raw)).read("document.json").decode("utf-8")
    return raw.decode("utf-8")


def save_start(scene) -> tuple[str, str]:
    """Store the start document (by content) and return its path and the
    signature Python reads back from it."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "start.igz"
        igz.save_scene(scene, path)
        raw = path.read_bytes()
        again = Scene()
        igz.load_into(again, path)
    h = hashlib.sha1(raw).hexdigest()[:16]
    starts = GOLDEN / "starts"
    starts.mkdir(parents=True, exist_ok=True)
    (starts / f"{h}.igz").write_bytes(raw)
    parts = scene_signature(again)
    with open(starts / f"{h}.sig.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(parts, fh, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"golden/starts/{h}.igz", signature_hash(parts)


def run(name: str, scene, steps: list) -> None:
    """Replay ``steps`` on the scene a start document opens to; write the
    script with each step's signature and document."""
    from core.version import __version__

    start, start_sig = save_start(scene)
    live = Scene()
    igz.load_into(live, GOLDEN.parent / start)
    history = History(live)
    reg = recorder.registry()
    out = []
    for step in steps:
        step = dict(step)
        if step["op"] in ("undo", "redo"):
            ok = getattr(history, step["op"])()
            step["ok"] = ok
        else:
            cmd = recorder.from_op(step, live, reg)
            step["counters"] = recorder.counters()
            recorder.set_counters(step["counters"])
            history.execute(cmd)
            if history.last_error is not None:
                raise SystemExit(f"{name}: {step['op']}: {history.last_error}")
            if history.last_rejected is not None:
                step["rejected"] = history.last_rejected
        step["sig"] = signature_hash(scene_signature(live))
        step["doc"] = document_text(live)
        out.append(step)
    # The runner undoes everything at the end; the group commands undo by the
    # inverse arithmetic, which float32 does not always round-trip. Where the
    # start does not come back, note the signature Python ends at.
    while history.undo():
        pass
    drift_path = GOLDEN / "undo_all_drift.json"
    drift = json.loads(drift_path.read_text(encoding="utf-8")) if drift_path.exists() else {}
    gid = f"hand-annotations-{name}"
    end_sig = signature_hash(scene_signature(live))
    if end_sig != start_sig:
        drift[gid] = end_sig
    else:
        drift.pop(gid, None)
    with open(drift_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(x)}" for k, x in sorted(drift.items())) + "\n}\n")
    doc = {"schema": 1, "id": gid,
           "source": {"hand": f"scripts/golden/make_annotation_scripts.py::{name}",
                      "app_version": __version__, "harvested": "2026-10-19"},
           "start": {"igz": start}, "start_sig": start_sig}
    lines = ["{"]
    lines += [f" {json.dumps(k)}: {json.dumps(x, ensure_ascii=False)}," for k, x in doc.items()]
    steps_json = [json.dumps(s, ensure_ascii=False, separators=(",", ":")) for s in out]
    lines.append(' "steps": [')
    lines += [f"  {s}," for s in steps_json[:-1]] + [f"  {steps_json[-1]}"]
    lines += [" ]", "}"]
    (GOLDEN / "ops").mkdir(parents=True, exist_ok=True)
    path = GOLDEN / "ops" / f"hand-annotations-{name}.json"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"{path.name}: {len(out)} steps")


def undo(n=1):
    return [{"op": "undo"}] * n


def redo(n=1):
    return [{"op": "redo"}] * n


def dim(a, b, offset, layer=None, text=None, axis=None):
    return {"op": "add_dimension", "args": {"dimension": {"$obj": "Dimension", "data": {
        "a": list(a), "b": list(b), "offset": list(offset), "layer": layer, "text": text, "axis": axis}}}}


CORNERS = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0), (0, 0, 2), (2, 0, 2), (2, 2, 2), (0, 2, 2)]


# ---- dimensions -----------------------------------------------------------------------

def dimensions():
    scene = Scene()
    cube(scene.mesh, 0, 0, 0, 2.0)
    g = instance("Comp", [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], placed(5.0, 2.0, 0.0, 30.0))
    scene.groups.append(g)
    gm = g.xform
    ga, gb = gm.map(V(0, 0, 0)), gm.map(V(1, 0, 0))
    allc = [list(map(float, c)) for c in CORNERS]
    steps = [
        dim((0, 0, 0), (2, 0, 0), (0, -1, 0)),
        dim((0, 0, 0), (2, 2, 0), (0, 3, 0), axis="x"),
        dim((0.5, 0, 0), (1.5, 0, 0), (0, 0, 1), text="<> long", layer="Cotas"),
        {"op": "move_vertices", "args": {"positions": [v(2, 0, 0), v(2, 2, 0), v(2, 0, 2), v(2, 2, 2)], "delta": v(1, 0, 0)}},
        {"op": "scale_vertices", "args": {"positions": [v(0, 0, 0), v(3, 0, 0), v(3, 2, 0), v(0, 2, 0), v(0, 0, 2), v(3, 0, 2), v(3, 2, 2), v(0, 2, 2)],
                                           "center": v(0, 0, 0), "factor": 1.5}},
        {"op": "move_dimensions", "args": {"modes": {"$dict": [[ref("dimensions", 0), "line"], [ref("dimensions", 2), "rigid"]]}, "delta": v(0.3, 0.2, 0.1)}},
        {"op": "edit_dimension_text", "args": {"dim": ref("dimensions", 1), "text": "L=<>"}},
        dim([ga.x(), ga.y(), ga.z()], [gb.x(), gb.y(), gb.z()], (0, 0, 1)),
        {"op": "move_group", "args": {"group": {"$group": [0]}, "delta": v(1, 1, 0)}},
        {"op": "scale_group", "args": {"group": {"$group": [0]}, "center": v(0, 0, 0), "factor": 2.0}},
        {"op": "delete_dimensions", "args": {"dimensions": [ref("dimensions", 0), ref("dimensions", 2)]}},
    ] + undo(1) + redo(1) + undo(2) + [
        {"op": "edit_dimension_text", "args": {"dim": ref("dimensions", 1), "text": None}},
        {"op": "move_dimensions", "args": {"modes": {"$dict": [[ref("dimensions", 1), "rigid"]]}, "delta": v(-0.5, 0.0, 0.25)}},
    ] + undo(12) + redo(5)
    run("dimensions", scene, steps)


# ---- leader texts and guides ---------------------------------------------------------

def texts_and_guides():
    scene = Scene()
    cube(scene.mesh, 0, 0, 0, 2.0)
    label = lambda anchor, offset, text, layer=None: {"op": "add_text_label", "args": {"label": {  # noqa: E731
        "$obj": "core.textlabel.TextLabel", "data": dict(anchor=list(anchor), offset=list(offset), text=text, **({"layer": layer} if layer else {}))}}}
    guide = lambda **d: {"op": "add_guide", "args": {"guide": {"$obj": "core.guide.Guide", "data": d}}}   # noqa: E731
    steps = [
        label((1, 2, 0), (0.5, 0, 1), "Muro eje A"),
        label((0, 0, 2), (-1, -1, 1), "Pilar", layer="Notas"),
        {"op": "move_text_labels", "args": {"labels": [ref("text_labels", 0), ref("text_labels", 1)], "delta": v(1, 0.5, 2)}},
        {"op": "edit_text_label", "args": {"label": ref("text_labels", 1), "text": "Pilar P1"}},
        {"op": "delete_text_labels", "args": {"labels": [ref("text_labels", 0)]}},
    ] + undo(2) + redo(1) + [
        guide(point=[0.0, 2.5, 0.0]),
        guide(point=[1.0, 1.0, 0.0], direction=[3.0, 4.0, 0.0]),
        guide(point=[2.0, 2.0, 2.0], origin=[0.0, 0.0, 0.0]),
        {"op": "change_guide", "args": {"guide": ref("guides", 1), "direction": v(0.8660254, 0.5, 0.0)}},
        {"op": "change_guide", "args": {"guide": ref("guides", 1), "direction": v(0.0, 0.0, 5.0)}},
        {"op": "delete_guides", "args": {"guides": [ref("guides", 0), ref("guides", 2)]}},
    ] + undo(4) + redo(3) + undo(9)
    run("texts-guides", scene, steps)


# ---- section planes -------------------------------------------------------------------

def sections():
    scene = Scene()
    cube(scene.mesh, 0, 0, 0, 2.0)
    plane = lambda point, normal, uid, name, symbol: {"op": "place_section_plane", "args": {"plane": {  # noqa: E731
        "$obj": "core.section.SectionPlane", "data": {"point": list(point), "normal": list(normal), "uid": uid, "name": name, "symbol": symbol}}}}
    steps = [
        plane((0, 0, 1), (0, 0, 1), 101, "Section 1", "A"),
        plane((1, 0, 0), (1, 1, 0), 102, "Section 2", "B"),
        {"op": "reverse_section_plane", "args": {"plane": ref("section_planes", 1)}},
        {"op": "set_active_section", "args": {"plane": None}},
        {"op": "set_active_section", "args": {"plane": ref("section_planes", 0)}},
        {"op": "set_section_mode", "args": {"plane": ref("section_planes", 1), "cut": True, "frame": False}},
        {"op": "set_section_mode", "args": {"plane": ref("section_planes", 0), "cut": False, "frame": True}},
        {"op": "set_section_fill", "args": {"plane": ref("section_planes", 0), "fill": {"pattern": "hatch", "color": {"$t": [0.2, 0.4, 0.6]}, "spacing": 0.1}}},
        {"op": "set_section_fill", "args": {"plane": ref("section_planes", 0), "fill": None}},
        {"op": "rename_section_plane", "args": {"plane": ref("section_planes", 0), "name": "Planta baja", "symbol": "PB"}},
        {"op": "move_section_planes", "args": {"planes": [ref("section_planes", 0), ref("section_planes", 1)], "delta": v(0.25, -0.5, 1.0)}},
        {"op": "rotate_section_planes", "args": {"planes": [ref("section_planes", 0), ref("section_planes", 1)], "centre": v(1, 1, 0), "axis": v(0, 0, 1), "deg": 33.0}},
        {"op": "rotate_section_planes", "args": {"planes": [ref("section_planes", 1)], "centre": v(0, 0, 0), "axis": v(1, 0.5, 0.25), "deg": -71.5}},
        {"op": "delete_section_planes", "args": {"planes": [ref("section_planes", 0), ref("section_planes", 1)]}},
    ] + undo(5) + redo(2) + undo(14) + redo(14)
    run("sections", scene, steps)


# ---- images ---------------------------------------------------------------------------

def images():
    scene = Scene()
    cube(scene.mesh, 0, 0, 0, 2.0)
    image = lambda path, **d: {"op": "add_image_plane", "args": {"image": {  # noqa: E731
        "$obj": "core.image_plane.ImagePlane", "data": dict({"texture": {"path": path}, "origin": [0.0, 0.0, 0.0], "u": [4.0, 0.0, 0.0],
                                                             "v": [0.0, 3.0, 0.0], "aspect": 0.75}, **d)}}}
    steps = [
        image("scan.png"),
        image("facade.png", layer=None, name="Fachada", opacity=0.5, locked=True),
        {"op": "transform_image_plane", "args": {"image": ref("image_planes", 0), "origin": None, "u": v(8, 0, 0), "v": v(0, 6, 0)}},
        {"op": "set_image_plane_opacity", "args": {"image": ref("image_planes", 0), "opacity": 0.4}},
        {"op": "set_image_plane_opacity", "args": {"image": ref("image_planes", 1), "opacity": 3.0}},
        {"op": "move_image_planes", "args": {"images": [ref("image_planes", 0), ref("image_planes", 1)], "delta": v(1, 2, 3)}},
        {"op": "rotate_image_planes", "args": {"images": [ref("image_planes", 0)], "centre": v(2, 1.5, 0), "axis": v(0, 0, 1), "deg": 33.0}},
        {"op": "rotate_image_planes", "args": {"images": [ref("image_planes", 1)], "centre": v(0, 0, 0), "axis": v(1, 0, 0), "deg": 90.0}},
        {"op": "scale_image_planes", "args": {"images": [ref("image_planes", 0)], "anchor": v(0, 0, 0), "factor": 2.5}},
        {"op": "scale_image_planes", "args": {"images": [ref("image_planes", 1)], "anchor": v(1, 1, 0), "factor": {"$t": [2.0, 0.5, 1.5]}}},
        {"op": "scale_image_planes", "args": {"images": [ref("image_planes", 0)], "anchor": v(1, 1, 0), "factor": {"$t": [2.0, 3.0, 1.0]},
                                              "axes": {"$t": [v(0.6, 0.8, 0), v(-0.8, 0.6, 0), v(0, 0, 1)]}}},
        {"op": "delete_image_planes", "args": {"images": [ref("image_planes", 1)]}},
    ] + undo(4) + redo(2) + undo(14) + redo(14)
    run("images", scene, steps)


# ---- georef ---------------------------------------------------------------------------

def georef():
    scene = Scene()
    cube(scene.mesh, 0, 0, 0, 2.0)
    path = lambda pts, **d: {"op": "add_geo_path", "args": {"path": {  # noqa: E731
        "$obj": "georef.geopath.GeoPath", "data": dict({"points": pts}, **d)}}}
    gp = lambda x, y, z, name, desc: {"$obj": "georef.points.GeoPoint", "data": {"position": [x, y, z], "name": name, "desc": desc}}   # noqa: E731
    steps = [
        path([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 8.0, 1.0]]),
        path([[1.0, 1.0, 0.0], [5.0, 1.0, 0.0]], name="Via", closed=True),
        {"op": "toggle_geo_path_closed", "args": {"paths": [ref("geo_paths", 0)]}},
        {"op": "set_geo_path_surface", "args": {"paths": [ref("geo_paths", 0), ref("geo_paths", 1)], "mode": "flat"}},
        {"op": "set_geo_path_surface", "args": {"paths": [ref("geo_paths", 0)], "mode": None}},
        {"op": "move_geo_path_node", "args": {"path": ref("geo_paths", 0), "index": 1, "new_point": v(12, 1, 0)}},
        {"op": "move_geo_path_node", "args": {"path": ref("geo_paths", 1), "index": -1, "new_point": v(6, 2, 0.5)}},
        {"op": "add_geo_points", "args": {"points": [gp(0.5, 0.25, 0.0, "E1", "ESTACION"), gp(8.25, 12.5, 0.9, "V1", "VERTICE LOTE")],
                                          "datum": {"$obj": "georef.datum.SceneDatum", "data": {"lat": -13.5235, "lon": -71.9837, "alt": 3335.2}}}},
        {"op": "delete_geo_points", "args": {"points": [ref("geo_points", 0)]}},
        {"op": "delete_geo_paths", "args": {"paths": [ref("geo_paths", 0)]}},
    ] + undo(6) + redo(3) + undo(12) + redo(12)
    run("georef", scene, steps)


# ---- saved scenes ---------------------------------------------------------------------

def saved_scenes():
    from core.saved_views import SavedView
    scene = Scene()
    cube(scene.mesh, 0, 0, 0, 2.0)
    for i, name in enumerate(["Planta", "Fachada", "Detalle"]):
        scene.saved_views.append(SavedView(name, target=(float(i), 1.0, 2.0), distance=10.0 + i, yaw=0.5 * i, pitch=0.3))
    steps = [
        {"op": "delete_scenes", "args": {"views": [ref("saved_views", 0), ref("saved_views", 2)]}},
    ] + undo(1) + redo(1) + undo(1) + [
        {"op": "delete_scenes", "args": {"views": [ref("saved_views", 1)]}},
    ] + undo(1) + redo(1)
    run("saved-scenes", scene, steps)


# ---- Straighten Model ------------------------------------------------------------------

def straighten():
    from core.composition import Composicion, CotaItem, MarcoVista
    from core.saved_views import SavedView
    from georef.datum import SceneDatum
    from georef.geopath import GeoPath
    from georef.points import GeoPoint
    scene = Scene()
    cube(scene.mesh, 0, 0, 0, 2.0)
    scene.georef = SceneDatum(-13.5235, -71.9837, 3335.2, 12.5)
    model = instance("Casa", [(0, 0, 0), (4, 0, 0), (4, 3, 0), (0, 3, 0)], placed(100.0, 50.0, 0.0, 30.0))
    model.mesh.add_face([V(0, 0, 0), V(0, 3, 0), V(0, 3, 2), V(0, 0, 2)])
    scene.groups.append(model)
    other = Group(Mesh(), "Muro")
    other.uid = "00000000000000aa"
    other.mesh.add_face([V(10, 10, 0), V(12, 10, 0), V(12, 10, 3), V(10, 10, 3)])
    scene.groups.append(other)
    other_inst = instance("Banco", [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], placed(-5.0, 7.0, 1.0, -20.0))
    scene.groups.append(other_inst)
    from core.dimension import Dimension
    d = Dimension(V(0, 0, 0), V(2, 0, 0), V(0, -1, 0))
    d.bind(scene)
    scene.dimensions.append(d)
    scene.dimensions.append(Dimension(V(0.5, 0, 0), V(1.5, 0, 0), V(0, 0, 1), text="<> m"))
    from core.guide import Guide
    from core.section import SectionPlane
    from core.textlabel import TextLabel
    from core.image_plane import ImagePlane
    scene.text_labels.append(TextLabel(V(1, 1, 1), V(0.5, 0, 1), "Eje"))
    scene.guides.append(Guide(V(3, 3, 0), V(1, 2, 0)))
    scene.guides.append(Guide(V(1, 5, 0), origin=V(0, 0, 0)))
    scene.section_planes.append(SectionPlane(V(0, 0, 1), V(0, 1, 0), name="S1", symbol="A", active=True))
    scene.image_planes.append(ImagePlane("scan.png", V(0, 0, 0), V(4, 0, 0), V(0, 3, 0), 0.75))
    scene.geo_paths.append(GeoPath([V(0, 0, 0), V(10, 0, 0), V(10, 8, 1)]))
    scene.geo_points.append(GeoPoint(V(1, 2, 3), "P1", "ESTACION"))
    scene.saved_views.append(SavedView("Vista", target=(1.0, 2.0, 3.0), distance=12.0, yaw=0.7, pitch=0.2))
    c = Composicion(name="Lamina")
    c.frames.append(MarcoVista(cam_target=[1.0, 2.0, 3.0], cam_yaw=0.4, uid="f1"))
    c.cotas.append(CotaItem(a_world=[0.0, 1.0, 0.0], b_world=[4.0, 1.0, 0.0], anchor_uid="f1"))
    scene.compositions.append(c)
    steps = [
        {"op": "straighten_model", "args": {"group": {"$group": [0]}}},
    ] + undo(1) + redo(1) + [
        {"op": "move_group", "args": {"group": {"$group": [2]}, "delta": v(1, 0, 0)}},
    ] + undo(2) + redo(2)
    run("straighten", scene, steps)


def main():
    dimensions()
    texts_and_guides()
    sections()
    images()
    georef()
    saved_scenes()
    straighten()


if __name__ == "__main__":
    main()
