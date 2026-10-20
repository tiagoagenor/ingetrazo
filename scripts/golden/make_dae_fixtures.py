# SPDX-License-Identifier: GPL-3.0-or-later
"""Write the fixtures that hold the Rust COLLADA code (``ingetrazo-formats-mesh``,
module ``dae``) to ``formats/dae.py`` (docs/rust-rewrite-plan.md, M5.4).

Export: the scenes of ``make_mesh_formats_fixtures.py`` (their .igz.json are
already in ``tests/data/export``) plus a few of our own (a georeferenced
scene with opacity, material names that collide, a material with no
triangles) are exported with the REAL ``save_dae``. The ``<created>`` and
``<modified>`` stamps are the time of the run: the test masks them.

Import: the exported files read back, plus hand-written documents in the
styles other modellers write (SketchUp: library_nodes + instance_node +
matrices, inches, textures with sampler chains; Blender: triangles with
interleaved inputs, nested nodes with translate/rotate/scale; polygons;
bare geometry; Y_UP and X_UP; geographic locations) and a pile of odd or
broken ones. Next to each goes the L2 signature of the scene Python has
after the import, with ``run_stitch`` and ``orient_outward`` (the edit
engine, M3) run for real, with Python's set order pinned as ``_pin_order.py``
says. ``import/cases.json`` lists them with the
georeference Python adopted and, for bad input, the message it raised.

Run it with ``PYTHONHASHSEED=0``: a material whose image is missing gets a
tint derived from ``hash(path)``, which Python salts per process (the Rust
side mirrors the unsalted SipHash).

    PYTHONHASHSEED=0 python scripts/golden/make_dae_fixtures.py

(PYTHONPATH at the repository root, native engine OFF.) Deleted with the
Python app at M10.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.pop("INGETRAZO_RUST", None)

from PySide6.QtGui import QGuiApplication, QImage, QVector3D as V  # noqa: E402

import make_mesh_formats_fixtures as base  # noqa: E402
from core.layers import Layer  # noqa: E402
from core.scene import Scene  # noqa: E402
from formats import dae, igz  # noqa: E402
from georef.datum import SceneDatum  # noqa: E402
from tests.golden.signature import scene_signature  # noqa: E402

OUT = base.OUT
DOUT = OUT / "dae"
NSDECL = 'xmlns="http://www.collada.org/2005/11/COLLADASchema"'


# ---- export ---------------------------------------------------------------------

def scene_geo(tex: Path) -> Scene:
    """A georeferenced scene; opacity on colours and textures."""
    s = Scene()
    s.georef = SceneDatum(-12.0464, -77.0428, 154.3)
    m = s.mesh
    wood = tex / "wood.png"
    a = m.add_face([V(0, 0, 0), V(2, 0, 0), V(2, 2, 0), V(0, 2, 0)])
    a.attrs["color"] = [0.2, 0.4, 0.6]
    a.attrs["opacity"] = 0.5
    b = m.add_face([V(3, 0, 0), V(5, 0, 0), V(5, 2, 0)])
    b.attrs["color"] = [0.2, 0.4, 0.6]
    b.attrs["opacity"] = 0.5004                 # rounds to the same 0.5
    c = m.add_face([V(6, 0, 0), V(8, 0, 0), V(8, 2, 0)])
    c.attrs["color"] = [0.2, 0.4, 0.6]
    c.attrs["opacity"] = "0.25"
    d = m.add_face([V(9, 0, 0), V(11, 0, 0), V(11, 2, 0)])
    d.attrs["color"] = [0.2, 0.4, 0.6]
    d.attrs["opacity"] = 0.9995                 # opaque: same key as none
    e = m.add_face([V(12, 0, 0), V(14, 0, 0), V(14, 2, 0)])
    e.attrs["color"] = [0.2, 0.4, 0.6]
    f = m.add_face([V(15, 0, 0), V(17, 0, 0), V(17, 2, 0), V(15, 2, 0)])
    f.attrs["texture"] = {"path": str(wood), "sw": 1.0, "sh": 1.0}
    f.attrs["opacity"] = 0.3
    g = m.add_face([V(18, 0, 0), V(20, 0, 0), V(20, 2, 0), V(18, 2, 0)])
    g.attrs["texture"] = {"path": str(wood), "sw": 1.0, "sh": 1.0}
    return s


def scene_names(tex: Path) -> Scene:
    """Named materials that collide or need cleaning; a material with no
    triangle that still takes an index; int colours equal to float ones."""
    s = Scene()
    m = s.mesh
    # all slivers: its material exists but owns no triangle
    dead = m.add_face([V(0, 12, 0), V(1, 12, 0), V(2, 12, 0)])
    dead.attrs.update({"color": [0.9, 0.9, 0.0], "mat": "Fantasma"})
    names = ["Concreto visto 1", "Concreto/visto 1", "1abc", "  Ñandú  ",
             "..", "", "a" * 3, "Concreto visto 1"]
    for k, nm in enumerate(names):
        f = m.add_face([V(3 * k, 0, 0), V(3 * k + 2, 0, 0), V(3 * k + 2, 2, 0)])
        f.attrs["color"] = [0.1 * k, 0.2, 0.3 + 0.01 * k]
        if nm:
            f.attrs["mat"] = nm
    i1 = m.add_face([V(0, 5, 0), V(1, 5, 0), V(1, 6, 0)])
    i1.attrs["color"] = [0, 1, 0]
    i2 = m.add_face([V(3, 5, 0), V(4, 5, 0), V(4, 6, 0)])
    i2.attrs["color"] = [0.0, 1.0, 0.0]
    i3 = m.add_face([V(6, 5, 0), V(7, 5, 0), V(7, 6, 0)])
    i3.attrs["color"] = [True, 0, 0]
    return s


def scene_badcolor(tex: Path) -> Scene:
    s = Scene()
    f = s.mesh.add_face([V(0, 0, 0), V(1, 0, 0), V(1, 1, 0)])
    f.attrs["color"] = [0.5, 0.5]                # two channels: unpack fails
    return s


EXTRA_SCENES = {"geo": scene_geo, "names": scene_names}


def settle(scene):
    """Put every placement matrix in the state a scene opened from a file has.

    A QMatrix4x4 built with translate/rotate/scale carries Qt's special-case
    flags, and ``inverted()`` takes a different (slightly different in the
    last float32 bit) path while they last; reading an element through
    ``m(row, col)`` resets them to General. ``_remap_uvws`` inverts first and
    reads elements after, so the first export of a freshly built scene remaps
    texture maps with the special-case inverse and every later one with the
    general inverse. A document read from an .igz always has General
    matrices (that is what the Rust side has), so the fixtures are made from
    that state."""
    def walk(g):
        if getattr(g, "xform", None) is not None:
            g.xform(0, 0)
        for c in getattr(g, "children", None) or ():
            walk(c)
    for g in getattr(scene, "groups", []):
        walk(g)
    return scene


def export_all() -> None:
    exp = DOUT / "export"
    shutil.rmtree(DOUT, ignore_errors=True)
    exp.mkdir(parents=True)
    tex = OUT / "export" / "tex"
    assert tex.is_dir(), "run make_mesh_formats_fixtures.py first"
    index = []
    for name, make in base.SCENES.items():
        dae.save_dae(settle(make(tex)), exp / f"{name}.dae")
        index.append({"scene": name, "file": f"{name}.dae"})
    # a project name that needs escaping
    dae.save_dae(settle(base.SCENES["faces"](tex)), exp / "faces.special.dae",
                 project_name='Casa & <Jardín> "x"')
    index.append({"scene": "faces", "file": "faces.special.dae",
                  "project": 'Casa & <Jardín> "x"'})
    for name, make in EXTRA_SCENES.items():
        scene = make(tex)
        saved = (igz._LARGE_DOCUMENT, igz._pack_textures)
        igz._LARGE_DOCUMENT = 10 ** 9
        igz._pack_textures = lambda payload: ({}, 0)
        try:
            igz.save_scene(scene, exp / f"{name}.igz.json")
        finally:
            igz._LARGE_DOCUMENT, igz._pack_textures = saved
        doc = json.loads((exp / f"{name}.igz.json").read_text(encoding="utf-8"))
        doc = base.relative_paths(doc, str(OUT / "export"))
        (exp / f"{name}.igz.json").write_text(
            json.dumps(doc, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8", newline="\n")
        dae.save_dae(settle(scene), exp / f"{name}.dae")
        index.append({"scene": name, "file": f"{name}.dae", "igz": True})
    # the export that fails
    bad = scene_badcolor(tex)
    saved = (igz._LARGE_DOCUMENT, igz._pack_textures)
    igz._LARGE_DOCUMENT = 10 ** 9
    igz._pack_textures = lambda payload: ({}, 0)
    try:
        igz.save_scene(bad, exp / "badcolor.igz.json")
    finally:
        igz._LARGE_DOCUMENT, igz._pack_textures = saved
    try:
        dae.save_dae(bad, exp / "badcolor.dae")
        raise AssertionError("badcolor must fail")
    except ValueError as exc:
        index.append({"scene": "badcolor", "igz": True,
                      "error": f"{type(exc).__name__}: {exc}"})
    (exp / "badcolor.dae").unlink(missing_ok=True)
    write_json(exp / "index.json", index)
    print(len(index), "exports")


# ---- hand-written documents -------------------------------------------------------

def doc(body, up="Z_UP", unit='<unit name="meter" meter="1"/>',
        asset_extra="", head='<?xml version="1.0" encoding="utf-8"?>',
        root_attrs=f'{NSDECL} version="1.4.1"'):
    return (f"{head}\n<COLLADA {root_attrs}>\n"
            f"  <asset>{asset_extra}{unit}<up_axis>{up}</up_axis></asset>\n"
            f"{body}\n</COLLADA>\n")


def floats_el(gid, tag, values, stride, params):
    flat = " ".join(repr(float(v)) if not isinstance(v, str) else v
                    for v in values)
    n = len(values)
    ps = "".join(f'<param name="{p}" type="float"/>' for p in params)
    return (f'<source id="{gid}-{tag}"><float_array id="{gid}-{tag}-arr" '
            f'count="{n}">{flat}</float_array><technique_common>'
            f'<accessor source="#{gid}-{tag}-arr" count="{n // stride}" '
            f'stride="{stride}">{ps}</accessor></technique_common></source>')


def mesh(gid, pos, prims, uvs=None, stride=3, name=None):
    extra = ""
    if uvs is not None:
        extra = floats_el(gid, "uv", uvs, 2, "ST")
    return (f'<geometry id="{gid}" name="{name or gid}"><mesh>'
            + floats_el(gid, "pos", pos, stride, "XYZ"[:stride])
            + extra
            + f'<vertices id="{gid}-v"><input semantic="POSITION" '
              f'source="#{gid}-pos"/></vertices>'
            + prims + "</mesh></geometry>")


def effect_color(eid, rgba, shader="lambert"):
    c = " ".join(str(x) for x in rgba)
    return (f'<effect id="{eid}"><profile_COMMON><technique sid="common">'
            f'<{shader}><diffuse><color sid="diffuse">{c}</color></diffuse>'
            f'</{shader}></technique></profile_COMMON></effect>')


def effect_tex(eid, image_id, shader="phong", chain=True):
    if chain:
        return (f'<effect id="{eid}"><profile_COMMON>'
                f'<newparam sid="{image_id}-surface"><surface type="2D">'
                f'<init_from>{image_id}</init_from></surface></newparam>'
                f'<newparam sid="{image_id}-sampler"><sampler2D>'
                f'<source>{image_id}-surface</source></sampler2D></newparam>'
                f'<technique sid="common"><{shader}><diffuse>'
                f'<texture texture="{image_id}-sampler" texcoord="UVSET0"/>'
                f'</diffuse></{shader}></technique></profile_COMMON></effect>')
    return (f'<effect id="{eid}"><profile_COMMON><technique sid="common">'
            f'<{shader}><diffuse><texture texture="{image_id}" '
            f'texcoord="UVSET0"/></diffuse></{shader}></technique>'
            f'</profile_COMMON></effect>')


def material(mid, eid):
    return (f'<material id="{mid}" name="{mid}"><instance_effect '
            f'url="#{eid}"/></material>')


def images(*pairs):
    inner = "".join(f'<image id="{i}" name="{i}"><init_from>{p}</init_from>'
                    f"</image>" for i, p in pairs)
    return f"<library_images>{inner}</library_images>"


def inst_geom(gid, binds=None):
    if not binds:
        return f'<instance_geometry url="#{gid}"/>'
    inner = "".join(f'<instance_material symbol="{s}" target="#{m}"/>'
                    for s, m in binds.items())
    return (f'<instance_geometry url="#{gid}"><bind_material>'
            f"<technique_common>{inner}</technique_common></bind_material>"
            f"</instance_geometry>")


def node(nid, content, xf="", name=None):
    return f'<node id="{nid}" name="{name or nid}">{xf}{content}</node>'


def vscene(*nodes):
    return ('<library_visual_scenes><visual_scene id="scene">'
            + "".join(nodes) + "</visual_scene></library_visual_scenes>"
            '<scene><instance_visual_scene url="#scene"/></scene>')


def lib(tag, *items):
    return f"<library_{tag}>" + "".join(items) + f"</library_{tag}>"


def cube_prim(material=None):
    tris = ("0 3 2 0 2 1  4 5 6 4 6 7  0 1 5 0 5 4  "
            "1 2 6 1 6 5  2 3 7 2 7 6  3 0 4 3 4 7")
    m = f' material="{material}"' if material else ""
    return (f'<triangles{m} count="12"><input semantic="VERTEX" '
            f'source="#CUBE-v" offset="0"/><p>{tris}</p></triangles>')


CUBE_POS = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
            0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1]


def cube_mesh(gid="CUBE", prim_material=None, pos=CUBE_POS):
    prim = cube_prim(prim_material).replace("#CUBE-v", f"#{gid}-v")
    return mesh(gid, pos, prim)


def quad_tex_prims(gid, material, mode="triangles"):
    """A quad (0,0)-(2,1) in XZ, with UVs, as two triangles or a polylist."""
    inputs = (f'<input semantic="VERTEX" source="#{gid}-v" offset="0"/>'
              f'<input semantic="NORMAL" source="#{gid}-n" offset="1"/>'
              f'<input semantic="TEXCOORD" source="#{gid}-uv" offset="2" '
              f'set="0"/>')
    if mode == "triangles":
        p = "0 0 0  1 0 1  2 0 2   0 0 0  2 0 2  3 0 3"
        return (f'<triangles material="{material}" count="2">{inputs}'
                f"<p>{p}</p></triangles>")
    p = "0 0 0  1 0 1  2 0 2  3 0 3"
    return (f'<polylist material="{material}" count="1">{inputs}'
            f"<vcount>4</vcount><p>{p}</p></polylist>")


def normals(gid, n=1):
    return floats_el(gid, "n", [0, -1, 0] * n, 3, "XYZ")


def textured_quad_geom(gid, material, mode="triangles", pos=None, uvs=None):
    pos = pos or [0, 0, 0, 2, 0, 0, 2, 0, 1, 0, 0, 1]
    uvs = uvs or [0, 0, 1, 0, 1, 1, 0, 1]
    g = mesh(gid, pos, quad_tex_prims(gid, material, mode), uvs=uvs)
    return g.replace("<vertices", normals(gid) + "<vertices", 1)


# ---- the import cases -------------------------------------------------------------

def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, sort_keys=True, separators=(",", ":")),
                    encoding="utf-8", newline="\n")


def write_documents(imp: Path) -> dict:
    """Write the hand-made inputs; returns ``{file: options}``."""
    cases: dict = {}

    def put(name, text, **opts):
        data = text if isinstance(text, bytes) else text.encode("utf-8")
        (imp / name).write_bytes(data)
        cases[name] = opts

    # -- SketchUp style: inches, library_nodes, instance_node + matrices ----
    leg = mesh("leg-g", [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
                         0, 0, 3, 1, 0, 3, 1, 1, 3, 0, 1, 3],
               '<polylist material="MatLeg" count="5">'
               '<input semantic="VERTEX" source="#leg-g-v" offset="0"/>'
               "<vcount>4 4 4 4 4</vcount>"
               "<p>0 3 2 1  4 5 6 7  0 1 5 4  1 2 6 5  3 0 4 7</p>"
               "</polylist>")
    top = textured_quad_geom("top-g", "MatWood", "polylist",
                             pos=[0, 0, 0, 10, 0, 0, 10, 0, 2, 0, 0, 2],
                             uvs=[0, 0, 2.5, 0, 2.5, 0.5, 0, 0.5])
    sk = doc(
        images(("wood-img", "materials/wood.png"))
        + lib("effects", effect_color("eff-leg", [0.2, 0.3, 0.4, 1], "phong"),
              effect_tex("eff-wood", "wood-img"))
        + lib("materials", material("MatLeg", "eff-leg"),
              material("MatWood", "eff-wood"))
        + lib("geometries", leg, top)
        + lib("nodes",
              node("ID1", inst_geom("leg-g", {"MatLeg": "MatLeg"}),
                   name="Leg"),
              node("ID2", inst_geom("top-g", {"MatWood": "MatWood"})
                   + '<instance_node url="#ID1"/>', name="Table"))
        + vscene(
            node("n1", '<instance_node url="#ID2" name="Table"/>',
                 xf='<matrix sid="transform">1 0 0 0  0 1 0 0  0 0 1 0  '
                    "0 0 0 1</matrix>"),
            node("n2", '<instance_node url="#ID2" name="Table"/>',
                 xf='<matrix>0 -1 0 25.5  1 0 0 3.25  0 0 1 0  0 0 0 1'
                    "</matrix>")),
        unit='<unit name="inch" meter="0.0254"/>',
        asset_extra="<contributor><authoring_tool>SketchUp 8.0.0000"
                    "</authoring_tool></contributor>")
    put("sketchup.dae", sk)

    # -- Blender style: interleaved inputs, nested nodes, transforms ---------
    quad = textured_quad_geom("Cube-mesh", "Mat-material")
    red = mesh("Plane-mesh", [-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0],
               '<triangles material="Red-material" count="2">'
               '<input semantic="VERTEX" source="#Plane-mesh-v" offset="0"/>'
               '<input semantic="NORMAL" source="#Plane-mesh-n" offset="1"/>'
               "<p>0 0 1 0 2 0  0 0 2 0 3 0</p></triangles>")
    red = red.replace("<vertices", normals("Plane-mesh") + "<vertices", 1)
    bl = doc(
        images(("wood-png", "wood.png"))
        + "<library_cameras><camera id=\"Camera-camera\" name=\"Camera\">"
          "<optics><technique_common><perspective><xfov sid=\"xfov\">39.6"
          "</xfov></perspective></technique_common></optics></camera>"
          "</library_cameras>"
        + "<library_lights><light id=\"Lamp-light\" name=\"Lamp\">"
          "<technique_common><point><color sid=\"color\">1 1 1</color>"
          "</point></technique_common></light></library_lights>"
        + lib("effects", effect_tex("Mat-effect", "wood-png"),
              effect_color("Red-effect", [0.8, 0.1, 0.1, 1], "phong"))
        + lib("materials", material("Mat-material", "Mat-effect"),
              material("Red-material", "Red-effect"))
        + lib("geometries", quad, red)
        + vscene(
            node("Camera", "", xf="<translate sid=\"location\">7.4 -6.5 4.9"
                 "</translate>"),
            node("Cube", inst_geom("Cube-mesh", {"Mat-material":
                                                  "Mat-material"})
                 + node("Child", inst_geom("Plane-mesh",
                                           {"Red-material": "Red-material"}),
                        xf='<translate sid="location">1.5 -0.25 0.1'
                           "</translate>"
                           '<rotate sid="rotationZ">0 0 1 45</rotate>'
                           '<rotate sid="rotationY">0 1 0 -12.5</rotate>'
                           '<scale sid="scale">2 1 0.5</scale>'),
                 xf='<matrix sid="transform">1 0 0 3.333  0 1 0 -1.1  '
                    "0 0 1 0.7  0 0 0 1</matrix>")),
        root_attrs=f'{NSDECL} xmlns:xsi="http://www.w3.org/2001/XMLSchema-'
                   f'instance" version="1.4.1"')
    put("blender.dae", bl)

    # -- Y_UP centimetres and X_UP -------------------------------------------
    for up, meter, nm in (("Y_UP", "0.01", "yup"), ("X_UP", "1", "xup"),
                          ("Y_UP", "0.3048", "yup-feet"),
                          ("Z_UP", "0.001", "zup-mm")):
        put(f"{nm}.dae", doc(
            lib("geometries", cube_mesh("CUBE", "RED"))
            + lib("effects", effect_color("fx", [1, 0, 0, 1]))
            + lib("materials", material("mat", "fx"))
            + vscene(node("n", inst_geom("CUBE", {"RED": "mat"}),
                          xf="<translate>1 2 3</translate><scale>10 10 10"
                             "</scale>")),
            up=up, unit=f'<unit name="u" meter="{meter}"/>'))

    # -- polygons, with 3, 4 and 5 corners ----------------------------------
    put("polygons.dae", doc(
        lib("geometries", mesh(
            "P", [0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 2, 0, 3, 1, 0,
                  5, 0, 0, 5, 2, 0, 4, 3, 0, 4, 1, 1],
            '<polygons material="m" count="3">'
            '<input semantic="VERTEX" source="#P-v" offset="0"/>'
            "<p>0 1 2 3</p><p>1 4 2</p><p>5 6 7 8 4</p><p>0 1</p><p></p>"
            "</polygons>"))
        + vscene(node("n", inst_geom("P")))))

    # -- bare geometry (no visual scene) and an empty one -------------------
    put("bare.dae", doc(lib("geometries", mesh(
        "B", [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
        '<triangles count="2"><input semantic="VERTEX" source="#B-v" '
        'offset="0"/><p>0 1 2 1 3 2</p></triangles>')),
        up="Y_UP", unit='<unit name="cm" meter="0.01"/>'))
    put("bare-scene.dae", doc(lib("geometries", mesh(
        "B", [0, 0, 0, 1, 0, 0, 0, 1, 0],
        '<triangles count="1"><input semantic="VERTEX" source="#B-v" '
        'offset="0"/><p>0 1 2</p></triangles>'))
        + vscene(node("empty", ""))))
    put("nogeom.dae", doc(lib("effects", effect_color("e", [1, 1, 1, 1]))),
        )

    # -- geographic location --------------------------------------------------
    tri = (lib("geometries", mesh("T", [0, 0, 0, 1, 0, 0, 0, 1, 0],
                                  '<triangles count="1"><input semantic='
                                  '"VERTEX" source="#T-v" offset="0"/>'
                                  "<p>0 1 2</p></triangles>"))
           + vscene(node("n", inst_geom("T"))))
    put("geo.dae", doc(
        tri, asset_extra="<coverage><geographic_location><longitude>"
        "-77.0428</longitude><latitude>-12.0464</latitude><altitude "
        'mode="absolute">154.3</altitude></geographic_location></coverage>'),
        )
    put("geo-noalt.dae", doc(
        tri, asset_extra="<coverage><geographic_location><longitude> 10.5 "
        "</longitude><latitude>-1e1</latitude></geographic_location>"
        "</coverage>"))
    put("geo-badalt.dae", doc(
        tri, asset_extra="<coverage><geographic_location><longitude>10"
        "</longitude><latitude>20</latitude><altitude>high</altitude>"
        "</geographic_location></coverage>"))
    put("geo-empty.dae", doc(
        tri, asset_extra="<coverage><geographic_location><longitude/>"
        "<latitude>20</latitude></geographic_location></coverage>"))
    put("geo-nolat.dae", doc(
        tri, asset_extra="<coverage><geographic_location><longitude>3"
        "</longitude></geographic_location></coverage>"))

    # -- a node that instances itself: the depth guard -------------------------
    put("cyclic.dae", doc(
        lib("geometries", mesh("T", [0, 0, 0, 1, 0, 0, 0, 1, 0],
                               '<triangles count="1"><input semantic='
                               '"VERTEX" source="#T-v" offset="0"/>'
                               "<p>0 1 2</p></triangles>"))
        + lib("nodes", node("loop", inst_geom("T")
                            + '<instance_node url="#loop"/>',
                            xf="<translate>0.5 0 0.25</translate>"))
        + vscene(node("n", '<instance_node url="#loop"/>'))))

    # -- odd primitives -------------------------------------------------------
    pos4 = [0, 0, 0, 7, 1, 0, 0, 7, 1, 1, 0, 7, 0, 1, 0, 7]
    put("stride4.dae", doc(lib("geometries", mesh(
        "S", pos4, '<triangles count="2"><input semantic="VERTEX" '
        'source="#S-v" offset="0"/><p>0 1 2 0 2 3</p></triangles>',
        stride=4)) + vscene(node("n", inst_geom("S")))))
    put("offsets.dae", doc(lib("geometries", mesh(
        "O", [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 2, 2, 2],
        '<triangles count="2"><input semantic="NORMAL" source="#O-n" '
        'offset="0"/><input semantic="VERTEX" source="#O-v" offset="3"/>'
        '<input semantic="TEXCOORD" source="#O-uv" offset="1"/>'
        '<input semantic="TEXCOORD" source="#O-uv2" offset="2"/>'
        "<p>0 0 0 0  0 0 0 1  0 0 0 2   0 0 0 0  0 0 0 2  0 0 0 3</p>"
        "</triangles>")) + vscene(node("n", inst_geom("O")))))
    put("badindex.dae", doc(lib("geometries", mesh(
        "I", [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
        '<triangles count="4"><input semantic="VERTEX" source="#I-v" '
        'offset="0"/><p>0 1 2  0 2 9  0 -1 2  0 2 3  0 1</p></triangles>'
        '<polylist count="3"><input semantic="VERTEX" source="#I-v" '
        'offset="0"/><vcount>3 -3 4 9</vcount><p>0 1 2  0 1 2 3  1 2 3</p>'
        "</polylist>")) + vscene(node("n", inst_geom("I")))))
    put("negoffset.dae", doc(lib("geometries", mesh(
        "N", [0, 0, 0, 1, 0, 0, 0, 1, 0],
        '<triangles count="1"><input semantic="VERTEX" source="#N-v" '
        'offset="-1"/><p>0 1 2</p></triangles>')) + vscene(
        node("n", inst_geom("N")))))
    put("emptyp.dae", doc(lib("geometries", mesh(
        "E", [0, 0, 0, 1, 0, 0, 0, 1, 0],
        '<triangles count="0"><input semantic="VERTEX" source="#E-v" '
        'offset="0"/></triangles>')) + vscene(node("n", inst_geom("E")))))
    put("cdata.dae", doc(lib("geometries", mesh(
        "C", ["<![CDATA[0 0 0 ]]>", "1 0 0 <!-- c --> 0 1", "0",
              "&#x31; 1 1"],
        '<triangles count="2"><input semantic="VERTEX" source="#C-v" '
        'offset="0"/><p>0 1 2 <![CDATA[1 3 2]]></p></triangles>'))
        + vscene(node("n", inst_geom("C"), name="a &amp; &lt;b&gt;"))))

    # -- materials ------------------------------------------------------------
    two = (lib("geometries", textured_quad_geom("Q", "SYM"))
           + vscene(node("n", inst_geom("Q", {"SYM": "mat"}))))
    put("missing-img.dae", doc(
        images(("img", "textures/no existe.png"))
        + lib("effects", effect_tex("fx", "img"))
        + lib("materials", material("mat", "fx")) + two))
    for k, ref in enumerate(("Ñandú/ñ.png", "日本/絵.png", "tex/😀.png", "")):
        put(f"missing-img{k}.dae", doc(
            images(("img", ref))
            + lib("effects", effect_tex("fx", "img"))
            + lib("materials", material("mat", "fx")) + two))
    flat = (lib("geometries", textured_quad_geom(
        "Q", "SYM", pos=[0, 0, 0, 2, 0, 0, 2, 1, 0, 0, 1, 0]))
        + vscene(node("n", inst_geom("Q", {"SYM": "mat"}))))
    put("texture-direct.dae", doc(
        images(("img", "wood.png"))
        + lib("effects", effect_tex("fx", "img", chain=False))
        + lib("materials", material("mat", "fx")) + flat))
    put("texture-percent.dae", doc(
        images(("img", "my%20wood.png"))
        + lib("effects", effect_tex("fx", "img"))
        + lib("materials", material("mat", "fx")) + flat))
    put("texture-dot.dae", doc(
        images(("img", "./sub/../wood.png"))
        + lib("effects", effect_tex("fx", "img"))
        + lib("materials", material("mat", "fx")) + flat))
    put("texture-nouv.dae", doc(
        images(("img", "wood.png"))
        + lib("effects", effect_tex("fx", "img"))
        + lib("materials", material("mat", "fx"))
        + lib("geometries", mesh(
            "Q", [0, 0, 0, 2, 0, 0, 2, 0, 1, 0, 0, 1],
            '<triangles material="SYM" count="2"><input semantic="VERTEX" '
            'source="#Q-v" offset="0"/><p>0 1 2 0 2 3</p></triangles>'))
        + vscene(node("n", inst_geom("Q", {"SYM": "mat"})))),
        qimage=True)
    put("texture-badindex.dae", doc(
        images(("img", "wood.png"))
        + lib("effects", effect_tex("fx", "img"))
        + lib("materials", material("mat", "fx"))
        + lib("geometries", textured_quad_geom(
            "Q", "SYM", uvs=[0, 0, 1, 0, 1, 1]))   # 3 UVs: index 3 invalid
        + vscene(node("n", inst_geom("Q", {"SYM": "mat"})))),
        qimage=True)
    put("texture-degenerate.dae", doc(
        images(("img", "wood.png"))
        + lib("effects", effect_tex("fx", "img"))
        + lib("materials", material("mat", "fx"))
        + lib("geometries", textured_quad_geom(
            "Q", "SYM", uvs=[0, 0, 0, 0, 0, 0, 0, 0]))
        + vscene(node("n", inst_geom("Q", {"SYM": "mat"})))),
        qimage=True)
    put("colors.dae", doc(
        lib("effects",
            effect_color("a", [0.1, 0.2, 0.3, 1], "lambert"),
            effect_color("b", [0.4, 0.5, 0.6], "phong"),
            effect_color("c", [0.7, 0.8], "blinn"),
            effect_color("d", [1, 0, 1, 1], "constant"),
            '<effect id="e"><profile_COMMON><technique sid="t"><lambert>'
            "<emission><color>1 1 1 1</color></emission></lambert>"
            "</technique></profile_COMMON></effect>")
        + lib("materials", material("ma", "a"), material("mb", "b"),
              material("mc", "c"), material("md", "d"), material("me", "e"),
              '<material id="mf"><instance_effect url="#nowhere"/>'
              "</material>")
        + lib("geometries", *[
            mesh(f"G{k}", [k, 0, 0, k + 1, 0, 0, k, 1, 0],
                 f'<triangles material="s" count="1"><input semantic='
                 f'"VERTEX" source="#G{k}-v" offset="0"/><p>0 1 2</p>'
                 "</triangles>") for k in range(6)])
        + vscene(*[node(f"n{k}", inst_geom(f"G{k}", {"s": m}))
                   for k, m in enumerate(("ma", "mb", "mc", "md", "me",
                                          "mf"))])))
    put("symbols.dae", doc(
        lib("effects", effect_color("a", [0.9, 0, 0, 1]),
            effect_color("b", [0, 0.9, 0, 1]))
        + lib("materials", material("ma", "a"), material("mb", "b"))
        + lib("geometries", mesh(
            "G", [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
            '<triangles count="1"><input semantic="VERTEX" source="#G-v" '
            'offset="0"/><p>0 1 2</p></triangles>'
            '<triangles material="x" count="1"><input semantic="VERTEX" '
            'source="#G-v" offset="0"/><p>1 3 2</p></triangles>'))
        + vscene(node("n", '<instance_geometry url="#G"><bind_material>'
                      "<technique_common>"
                      '<instance_material symbol="x" target="#ma"/>'
                      '<instance_material symbol="x" target="#mb"/>'
                      '<instance_material target="#ma"/>'
                      "</technique_common></bind_material>"
                      "</instance_geometry>"))))

    # -- face-me candidates and big models (fuse pass: not ported) -----------
    cut = [0, 0, 0, 1, 0, 0, 1, 0, 1, 0.5, 0, 1.5, 0, 0, 1]   # 5 corners
    sprite = (images(("img", "wood.png"))
              + lib("effects", effect_tex("fx", "img"))
              + lib("materials", material("mat", "fx"))
              + lib("geometries", mesh(
                  "Q", cut, '<polylist material="SYM" count="1">'
                  '<input semantic="VERTEX" source="#Q-v" offset="0"/>'
                  '<input semantic="TEXCOORD" source="#Q-uv" offset="1"/>'
                  "<vcount>5</vcount><p>0 0 1 1 2 2 3 3 4 4</p></polylist>",
                  uvs=[0, 0, 1, 0, 1, 0.7, 0.5, 1, 0, 0.7]))
              + vscene(node("n", inst_geom("Q", {"SYM": "mat"}))))
    put("sprite.dae", doc(sprite), fused=True)
    put("sprite-child.dae", doc(sprite.replace(
        vscene(node("n", inst_geom("Q", {"SYM": "mat"}))),
        lib("nodes", node("comp", inst_geom("Q", {"SYM": "mat"})))
        + vscene(node("top", '<instance_node url="#comp"/>',
                      xf="<translate>3 0 0</translate>")))), fused=True)
    put("panel.dae", doc(images(("img", "wood.png"))
                         + lib("effects", effect_tex("fx", "img"))
                         + lib("materials", material("mat", "fx"))
                         + lib("geometries", textured_quad_geom("Q", "SYM"))
                         + vscene(node("n", inst_geom("Q", {"SYM": "mat"})))),
        qimage=True)
    put("flat-sign.dae", doc(images(("img", "wood.png"))
                             + lib("effects", effect_tex("fx", "img"))
                             + lib("materials", material("mat", "fx"))
                             + lib("geometries", textured_quad_geom(
                                 "Q", "SYM", pos=[0, 0, 0, 2, 0, 0, 2, 1, 0,
                                                  0, 1, 0]))
                             + vscene(node("n", inst_geom("Q",
                                                          {"SYM": "mat"})))))
    n = 18
    grid = []
    for i in range(n):
        for j in range(n):
            grid += [i, j, 0]
    idx = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, \
                i * n + j + 1
            idx += [a, b, c, a, c, d]
    big = doc(lib("geometries", mesh(
        "BIG", grid, f'<triangles count="{len(idx) // 3}"><input semantic='
        f'"VERTEX" source="#BIG-v" offset="0"/><p>'
        + " ".join(map(str, idx)) + "</p></triangles>"))
        + vscene(node("n", inst_geom("BIG"))))
    put("big.dae", big, fused=True)
    # exactly at the limit: 400 polygons stay in the loose mesh
    k = 400
    flat = []
    idx = []
    for t in range(k):
        flat += [t * 2, 0, 0, t * 2 + 1, 0, 0, t * 2, 1, 0]
        idx += [t * 3, t * 3 + 1, t * 3 + 2]
    put("limit400.dae", doc(lib("geometries", mesh(
        "L", flat, f'<triangles count="{k}"><input semantic="VERTEX" '
        'source="#L-v" offset="0"/><p>' + " ".join(map(str, idx))
        + "</p></triangles>")) + vscene(node("n", inst_geom("L")))))
    put("limit401.dae", doc(lib("geometries", mesh(
        "L", flat + [900, 0, 0, 901, 0, 0, 900, 1, 0],
        f'<triangles count="{k + 1}"><input semantic="VERTEX" '
        'source="#L-v" offset="0"/><p>' + " ".join(map(str, idx))
        + f" {k * 3} {k * 3 + 1} {k * 3 + 2}"
        "</p></triangles>")) + vscene(node("n", inst_geom("L")))),
        fused=True)

    # -- big models: groups, shared components, billboards ------------------------
    def grid_geom(gid, n, material=None, z=0.0, step=1.0):
        pos = []
        for i in range(n):
            for j in range(n):
                pos += [i * step, j * step, z]
        idx = []
        for i in range(n - 1):
            for j in range(n - 1):
                a, b, c, d = (i * n + j, (i + 1) * n + j,
                              (i + 1) * n + j + 1, i * n + j + 1)
                idx += [a, b, c, a, c, d]
        m = f' material="{material}"' if material else ""
        return mesh(gid, pos, f'<triangles{m} count="{len(idx) // 3}">'
                    f'<input semantic="VERTEX" source="#{gid}-v" offset="0"/>'
                    "<p>" + " ".join(map(str, idx)) + "</p></triangles>")

    def cutout_png(path, alpha_at=None):
        img = QImage(8, 8, QImage.Format_ARGB32)
        img.fill(0xFF336699)
        for (x, y) in (alpha_at or []):
            img.setPixel(x, y, 0x00336699)
        img.save(str(path))

    cutout_png(imp / "cutout.png", [(x, y) for x in range(0, 8) for y in (0, 1)])
    cutout_png(imp / "solid-alpha.png")                  # alpha plane, all 255
    img = QImage(64, 64, QImage.Format_ARGB32)
    img.fill(0xFF336699)
    img.setPixel(33, 33, 0x00336699)                     # one see-through pixel
    img.save(str(imp / "tiny-hole.png"))
    colors_fx = lib("effects", effect_color("fa", [0.8, 0.1, 0.1, 1]),
                    effect_color("fb", [0.1, 0.2, 0.9, 1]))
    colors_mat = lib("materials", material("ma", "fa"), material("mb", "fb"))
    tree = doc(
        colors_fx + colors_mat
        + lib("geometries", grid_geom("G1", 31, "s"), grid_geom("G2", 31, "s"),
              grid_geom("G3", 20, "s"))
        + vscene(node("site", inst_geom("G3", {"s": "ma"})
                      + node("wing1", inst_geom("G1", {"s": "ma"}),
                             xf="<translate>0 40 0</translate>")
                      + node("wing2", inst_geom("G2", {"s": "mb"}),
                             xf="<translate>40 0 0</translate>"))))
    put("tree.dae", tree, fused=True)
    put("tree-yup.dae", doc(
        colors_fx + colors_mat
        + lib("geometries", grid_geom("G1", 31, "s"), grid_geom("G2", 31, "s"),
              grid_geom("G3", 20, "s"))
        + vscene(node("site", inst_geom("G3", {"s": "ma"})
                      + node("wing1", inst_geom("G1", {"s": "ma"}),
                             xf="<translate>0 40 0</translate>")
                      + node("wing2", inst_geom("G2", {"s": "mb"}),
                             xf="<translate>40 0 0</translate>"))),
        up="Y_UP", unit='<unit name="x" meter="0.0254"/>'), fused=True)
    chair = (lib("geometries", grid_geom("CH", 8, "s"),
                 grid_geom("FLOOR", 12, "s"))
             + lib("nodes", node("chair", inst_geom("CH", {"s": "ma"}),
                                 name="Silla"))
             + vscene(node("floor", inst_geom("FLOOR", {"s": "mb"}))
                      + "".join(node(
                          f"c{k}", '<instance_node url="#chair"/>',
                          xf=f"<translate>{3 * k} 20 0</translate>"
                             f"<rotate>0 0 1 {17 * k}</rotate>"
                             + ("<scale>1 2 0.5</scale>" if k == 3 else ""))
                          for k in range(6))))
    put("instances.dae", doc(colors_fx + colors_mat + chair), fused=True)
    put("instances-yup.dae", doc(colors_fx + colors_mat + chair, up="Y_UP",
                                 unit='<unit name="x" meter="0.0254"/>'),
        fused=True)
    put("instances-matrix.dae", doc(colors_fx + colors_mat + chair.replace(
        "<translate>3 20 0</translate>", "<matrix>0.5 -0.5 0 7  0.5 0.5 0 "
        "8  0 0 2 0  0 0 0 1</matrix>")), fused=True)
    # a reused component that holds a sprite falls back to a copy per use
    sprite_geom = mesh(
        "SP", cut, '<polylist material="SYM" count="1">'
        '<input semantic="VERTEX" source="#SP-v" offset="0"/>'
        '<input semantic="TEXCOORD" source="#SP-uv" offset="1"/>'
        "<vcount>5</vcount><p>0 0 1 1 2 2 3 3 4 4</p></polylist>",
        uvs=[0, 0, 1, 0, 1, 0.7, 0.5, 1, 0, 0.7])
    sprite_fx = (images(("img", "wood.png"))
                 + lib("effects", effect_tex("fx", "img"),
                       effect_color("fa", [0.8, 0.1, 0.1, 1]))
                 + lib("materials", material("mat", "fx"),
                       material("ma", "fa")))
    put("instances-sprite.dae", doc(
        sprite_fx + lib("geometries", grid_geom("CH", 8, "s"), sprite_geom)
        + lib("nodes", node("comp", inst_geom("CH", {"s": "ma"}) + node(
            "leaf", inst_geom("SP", {"SYM": "mat"}), name="Hoja"),
            name="Arbol"))
        + vscene(*[node(f"c{k}", '<instance_node url="#comp"/>',
                        xf=f"<translate>{4 * k} 0 0</translate>")
                   for k in range(6)])), fused=True, qimage=True)
    put("faceme-big.dae", doc(
        sprite_fx + lib("geometries", grid_geom("BIG", 18, "s"), sprite_geom)
        + vscene(node("floor", inst_geom("BIG", {"s": "ma"}))
                 + node("tree", inst_geom("SP", {"SYM": "mat"}),
                        xf="<translate>5 0 0</translate>"))),
        fused=True)
    put("faceme-small-two.dae", doc(
        sprite_fx + lib("geometries", sprite_geom, cube_mesh("CUBE"))
        + vscene(node("tree", inst_geom("SP", {"SYM": "mat"}))
                 + node("box", inst_geom("CUBE"),
                        xf="<translate>5 0 0</translate>"))), fused=True)
    for nm, png in (("cutout", "cutout.png"), ("solid-alpha", "solid-alpha.png"),
                    ("tiny-hole", "tiny-hole.png"), ("opaque", "wood.png")):
        put(f"rect-{nm}.dae", doc(
            images(("img", png)) + lib("effects", effect_tex("fx", "img"))
            + lib("materials", material("mat", "fx"))
            + lib("geometries", textured_quad_geom("Q", "SYM"))
            + vscene(node("n", inst_geom("Q", {"SYM": "mat"})))),
            fused=(nm == "cutout"), qimage=True)
    put("rect-cutout-big.dae", doc(
        images(("img", "cutout.png")) + lib("effects", effect_tex("fx", "img"))
        + lib("materials", material("mat", "fx"), material("ma", "fx"))
        + lib("geometries", textured_quad_geom("Q", "SYM"),
              grid_geom("BIG", 18, "s"))
        + vscene(node("n", inst_geom("Q", {"SYM": "mat"}))
                 + node("m", inst_geom("BIG", {"s": "ma"})))),
        fused=True, qimage=True)
    bare = (lib("geometries", grid_geom("BARE", 18),
                grid_geom("BARE2", 5, z=2.0)))
    put("bare-big.dae", doc(bare), fused=True)
    # the colour of an image, when the polygon has no texture coordinates
    sizes = {"7x5": (7, 5), "64x48": (64, 48), "1x1": (1, 1), "120x90": (120, 90),
             "3x300": (3, 300), "33x2": (33, 2), "64x64": (64, 64)}
    import random as _random
    rng = _random.Random(11)
    for nm, (w, hgt) in sizes.items():
        im = QImage(w, hgt, QImage.Format_RGB32)
        for y in range(hgt):
            for x in range(w):
                im.setPixel(x, y, 0xFF000000 | rng.randrange(1 << 24))
        im.save(str(imp / f"avg-{nm}.png"))
    for nm, fmt in (("alpha", QImage.Format_ARGB32), ("gray", QImage.Format_Grayscale8),
                    ("indexed", QImage.Format_Indexed8)):
        w, hgt = 21, 13
        im = QImage(w, hgt, fmt)
        if fmt == QImage.Format_Indexed8:
            im.setColorTable([0xFF000000 | rng.randrange(1 << 24) for _ in range(16)])
        for y in range(hgt):
            for x in range(w):
                if fmt == QImage.Format_ARGB32:
                    im.setPixel(x, y, rng.randrange(1 << 32))
                elif fmt == QImage.Format_Grayscale8:
                    im.setPixel(x, y, 0xFF000000 | (rng.randrange(256) * 0x010101))
                else:
                    im.setPixel(x, y, rng.randrange(16))
        im.save(str(imp / f"avg-{nm}.png"))
    for nm in list(sizes) + ["alpha", "gray", "indexed"]:
        put(f"avg-{nm}.dae", doc(
            images(("img", f"avg-{nm}.png"))
            + lib("effects", effect_tex("fx", "img"))
            + lib("materials", material("mat", "fx"))
            + lib("geometries", mesh(
                "Q", [0, 0, 0, 2, 0, 0, 2, 0, 1, 0, 0, 1],
                '<triangles material="SYM" count="2"><input semantic='
                '"VERTEX" source="#Q-v" offset="0"/><p>0 1 2 0 2 3</p>'
                "</triangles>"))
            + vscene(node("n", inst_geom("Q", {"SYM": "mat"})))), qimage=True)

    # -- prefixes, no namespace, encodings --------------------------------------
    body = (lib("geometries", mesh(
        "T", [0, 0, 0, 1, 0, 0, 0, 1, 0],
        '<triangles count="1"><input semantic="VERTEX" source="#T-v" '
        'offset="0"/><p>0 1 2</p></triangles>')) + vscene(
        node("n", inst_geom("T"))))
    put("nons.dae", doc(body, root_attrs='version="1.4.1"'))
    put("bom.dae", b"\xef\xbb\xbf" + doc(body).encode("utf-8"))
    put("latin1.dae", doc(body, head='<?xml version="1.0" encoding="ISO-8859-1"'
                         '?>').replace('name="n"', 'name="caf\xe9"')
        .encode("latin-1"))
    put("utf16.dae", b"\xff\xfe" + doc(body, head='<?xml version="1.0" '
                                                  'encoding="UTF-16"?>')
        .encode("utf-16-le"))
    put("prefixed.dae", doc(body, root_attrs='xmlns:c="http://www.collada.org/'
                                             '2005/11/COLLADASchema" '
                                             'version="1.4.1"')
        .replace("<COLLADA", "<c:COLLADA").replace("</COLLADA", "</c:COLLADA")
        .replace("<asset>", "<c:asset>").replace("</asset>", "</c:asset>")
        .replace("<unit ", "<c:unit ").replace("<up_axis>", "<c:up_axis>")
        .replace("</up_axis>", "</c:up_axis>")
        .replace("<library_", "<c:library_").replace("</library_",
                                                      "</c:library_")
        .replace("<geometry", "<c:geometry").replace("</geometry",
                                                     "</c:geometry")
        .replace("<mesh>", "<c:mesh>").replace("</mesh>", "</c:mesh>")
        .replace("<source", "<c:source").replace("</source", "</c:source")
        .replace("<float_array", "<c:float_array")
        .replace("</float_array", "</c:float_array")
        .replace("<technique_common", "<c:technique_common")
        .replace("</technique_common", "</c:technique_common")
        .replace("<accessor", "<c:accessor").replace("</accessor",
                                                     "</c:accessor")
        .replace("<param", "<c:param")
        .replace("<vertices", "<c:vertices").replace("</vertices",
                                                     "</c:vertices")
        .replace("<input", "<c:input")
        .replace("<triangles", "<c:triangles").replace("</triangles",
                                                       "</c:triangles")
        .replace("<p>", "<c:p>").replace("</p>", "</c:p>")
        .replace("<visual_scene", "<c:visual_scene")
        .replace("</visual_scene", "</c:visual_scene")
        .replace("<node", "<c:node").replace("</node", "</c:node")
        .replace("<instance_geometry", "<c:instance_geometry")
        .replace("<scene>", "<c:scene>").replace("</scene>", "</c:scene>")
        .replace("<instance_visual_scene", "<c:instance_visual_scene"))

    # -- broken input -----------------------------------------------------------
    put("malformed.dae", doc(body).replace("</asset>", "</asse>"),
        xml_error=True)
    put("empty.dae", b"", xml_error=True)
    put("junk.dae", b"not xml at all", xml_error=True)
    put("ent.dae", doc(body).replace("<COLLADA", "<!DOCTYPE c [<!ENTITY x "
                                                 "'y &amp; &#65;'>]><COLLADA",
                                     1).replace('name="n"', 'name="&x;"'))
    put("badfloat.dae", doc(lib("geometries", mesh(
        "T", [0, 0, 0, 1, 0, 0, "0,5", 1, 0],
        '<triangles count="1"><input semantic="VERTEX" source="#T-v" '
        'offset="0"/><p>0 1 2</p></triangles>')) + vscene(
        node("n", inst_geom("T")))))
    put("badint.dae", doc(lib("geometries", mesh(
        "T", [0, 0, 0, 1, 0, 0, 0, 1, 0],
        '<triangles count="1"><input semantic="VERTEX" source="#T-v" '
        'offset="0"/><p>0 1 x</p></triangles>')) + vscene(
        node("n", inst_geom("T")))))
    put("badoffset.dae", doc(lib("geometries", mesh(
        "T", [0, 0, 0, 1, 0, 0, 0, 1, 0],
        '<triangles count="1"><input semantic="VERTEX" source="#T-v" '
        'offset=""/><p>0 1 2</p></triangles>')) + vscene(
        node("n", inst_geom("T")))))
    put("badmeter.dae", doc(body, unit='<unit name="x" meter="abc"/>'))
    put("emptymeter.dae", doc(body, unit='<unit name="x" meter=""/>'))
    put("badstride.dae", doc(body.replace('stride="3"', 'stride="0"')))
    put("badstride2.dae", doc(body.replace('stride="3"', 'stride="x"')))
    put("negstride.dae", doc(body.replace('stride="3"', 'stride="-3"')))
    put("badcount.dae", doc(body.replace('count="1"', 'count="x"')))
    put("badmatrix.dae", doc(lib("geometries", mesh(
        "T", [0, 0, 0, 1, 0, 0, 0, 1, 0],
        '<triangles count="1"><input semantic="VERTEX" source="#T-v" '
        'offset="0"/><p>0 1 2</p></triangles>')) + vscene(
        node("n", inst_geom("T"), xf="<matrix>1 0 0 oops</matrix>"))))
    put("shortmatrix.dae", doc(lib("geometries", mesh(
        "T", [0, 0, 0, 1, 0, 0, 0, 1, 0],
        '<triangles count="1"><input semantic="VERTEX" source="#T-v" '
        'offset="0"/><p>0 1 2</p></triangles>')) + vscene(
        node("n", inst_geom("T"), xf="<matrix>1 0 0 5</matrix>"
                                     "<translate>1 2</translate>"
                                     "<rotate>0 0 1</rotate>"
                                     "<scale>2 2</scale>"))))
    put("transforms.dae", doc(
        lib("geometries", mesh(
            "T", [0.1, 0.2, 0.3, 1.7, 0.2, 0.9, 0.4, 2.3, -0.7],
            '<triangles count="1"><input semantic="VERTEX" source="#T-v" '
            'offset="0"/><p>0 1 2</p></triangles>'))
        + lib("nodes", node("c", inst_geom("T"),
                            xf="<rotate>1 2 3 33.3</rotate>"
                               "<scale>1.1 0.9 1.3</scale>"))
        + vscene(
            node("a", node("b", '<instance_node url="#c"/>',
                           xf="<translate>0.123 -4.5 6.789</translate>"
                              "<rotate>0.3 0.4 0.5 -71.7</rotate>")
                 + inst_geom("T"),
                 xf="<translate>2.5 -1.25 0.75</translate><rotate>0 1 0 20</rotate><matrix>0.9 0.1 0.2 3.3  -0.1 1.2 0.05 -2.2  0.3 0 1.1 "
                    "7.7  0 0 0 1</matrix><scale>0.5 0.5 0.5</scale>"),
            node("c2", '<instance_node url="#c"/>',
                 xf="<rotate>0 0 1 90</rotate><translate>1 0 0</translate>")),
        up="Y_UP", unit='<unit name="x" meter="0.0254"/>'))
    return cases


def write_imports() -> None:
    imp = DOUT / "import"
    shutil.rmtree(imp, ignore_errors=True)
    imp.mkdir(parents=True)
    base.png(imp / "wood.png", 0x8B5A2B, 6)
    base.png(imp / "materials" / "wood.png", 0x8B5A2B, 7)
    base.png(imp / "my wood.png", 0x6B4A1B, 3)
    (imp / "sub").mkdir()
    cases_by_file = write_documents(imp)
    cases = []
    # The exported files read back (they bring their images along).
    exp = DOUT / "export"
    for entry in json.loads((exp / "index.json").read_text(encoding="utf-8")):
        if "file" in entry:
            cases.append({"file": f"../export/{entry['file']}",
                          "export": True})
    for name, opts in sorted(cases_by_file.items()):
        cases.append({"file": name, **opts})

    assert sys.flags.hash_randomization == 0, "run with PYTHONHASHSEED=0"
    base._pin_order.install()
    import formats.fuse
    real_fuse = formats.fuse.fuse_coplanar_loops
    fused = []

    def spy(*a, **k):
        fused.append(True)
        return real_fuse(*a, **k)
    formats.fuse.fuse_coplanar_loops = spy
    try:
        for case in cases:
            del fused[:]
            base._pin_order.fresh_group_names()
            base._pin_order.ORD.clear()
            base._pin_order.KEEP.clear()
            path = imp / case["file"]
            scene = Scene()
            try:
                dae.load_dae(scene, path)
            except ET.ParseError as exc:
                case["error"] = f"ParseError: {exc}"
                case["xml_error"] = True
                continue
            except Exception as exc:  # noqa: BLE001 — the message is the datum
                case["error"] = f"{type(exc).__name__}: {exc}"
                continue
            if fused:
                case["fused"] = True
            g = getattr(scene, "georef", None)
            if g is not None:
                case["georef"] = {"lat": g.lat, "lon": g.lon, "alt": g.alt}
            tag = case["file"].replace("../export/", "export-")
            sig_name = f"{tag}.sig.json"
            case["sig"] = sig_name
            write_json(imp / sig_name, scene_signature(scene))
    finally:
        formats.fuse.fuse_coplanar_loops = real_fuse
    # Entity declarations are expanded by expat; the Rust reader refuses them.
    for case in cases:
        if case.get("entity"):
            assert "error" not in case, case
    names = [c.get("sig") for c in cases if "sig" in c]
    assert len(names) == len(set(names)), "signature names clash"
    write_json(imp / "cases.json", cases)
    print(len(cases), "import cases")
    # hash(str) without the salt: the Rust side re-implements SipHash-1-3.
    samples = ["a", "ab", "hello", "textures/no existe.png", "wood.png",
               "Ñandú/ñ.png", "日本/絵.png", "tex/😀.png", "x" * 33,
               "mixed ñ 日 😀 text of some length", "./sub/../wood.png"]
    (imp / "hashes.txt").write_text(
        "".join(f"{hash(s)} {s}\n" for s in samples), encoding="utf-8",
        newline="\n")


def main() -> None:
    app = QGuiApplication.instance() or QGuiApplication(sys.argv)  # noqa: F841
    DOUT.parent.mkdir(parents=True, exist_ok=True)
    if "--imports-only" not in sys.argv:         # the exports are committed
        export_all()
    write_imports()


if __name__ == "__main__":
    main()
