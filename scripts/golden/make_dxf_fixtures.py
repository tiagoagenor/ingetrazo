# SPDX-License-Identifier: GPL-3.0-or-later
"""Write the fixtures that hold the Rust DXF importer and exporter in
`ingetrazo-formats-mesh` to Python's (docs/rust-rewrite-plan.md, M5.5).

Export (`formats/dxf_out.py`, a plain text writer, no library behind it):
the view lines of real scenes through the app's own HLR (parallel with
profile and cut layers, perspective, hidden lines dashed) plus synthetic
groups that stress the `.9g` number formatting, layer-name rules and empty
layers. Every case is written by the REAL `save_dxf_layers`; the Rust side
must reproduce the bytes (`export/<case>.dxf`, CRLF as Python's text mode
writes them on Windows). `export/g9.txt` holds 6000 doubles with Python's
`format(x, ".9g")`.

Import (`formats/dxf_in.py`, ezdxf 1.4.5): DXF files made with ezdxf (LINE,
LWPOLYLINE with bulges and extrusions, POLYLINE 2D / 3D / meshes, CIRCLE,
ARC, ELLIPSE, SPLINE, 3DFACE, SOLID, TRACE, blocks with transforms, nested
and multi-inserts, layers, skipped entity kinds, units, survey coordinates, a
big layer of 3DFACEs that is fused) and hand-written R12 / odd files. Next to
each goes the L2 signature of the scene Python builds, with ``run_stitch``
(the edit engine, ported in M3) replaced by a no-op: the Rust import is exact
up to that point. `import/cases.json` lists the files with the scale, the
stats Python returned and, for refused input, the message of the error.

The files are committed, so `cargo test` needs no Python; rerun this only to
widen them (PYTHONPATH at the repository root, native engine OFF):

    python scripts/golden/make_dxf_fixtures.py

Deleted with the Python app at M10.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.pop("INGETRAZO_RUST", None)

import ezdxf  # noqa: E402
from PySide6.QtGui import QGuiApplication, QVector3D as V  # noqa: E402

from core.scene import Scene  # noqa: E402
from formats import dxf_in, dxf_out  # noqa: E402
from tests.golden.signature import scene_signature  # noqa: E402

OUT = (ROOT / "rust" / "crates" / "ingetrazo-formats-mesh" / "tests" / "data"
       / "dxf")


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, sort_keys=True, separators=(",", ":")),
                    encoding="utf-8", newline="\n")


def hexbits(x: float) -> str:
    return f"{struct.unpack('>Q', struct.pack('>d', float(x)))[0]:016x}"


# ---- export ----------------------------------------------------------------------

def hlr_groups():
    import numpy as np  # noqa: F401
    from core.ai import _recipe_helpers
    from core.camera import OrbitCamera
    from core.hlr import (KIND_CUT, KIND_EDGE, KIND_HIDDEN, KIND_PROFILE,
                          hlr_drawing, hlr_perspective)
    s = Scene()
    _recipe_helpers(s)["revolve"](
        [(0.5, 0.0), (0.6, 0.5), (0.4, 1.0)], segments=24, name="Vaso")
    s.mesh.add_face([V(-2, -2, 0), V(2, -2, 0), V(2, 2, 0), V(-2, 2, 0)])
    s.mesh.add_face([V(3, 0, 0), V(5, 0, 0), V(5, 0, 2), V(3, 0, 2)])
    cam = OrbitCamera()
    cam.set_aspect(800, 600)
    lo, hi = s.bounds()
    cam.fit_to(lo, hi, 1.2)
    out = {}
    d = hlr_drawing(s, cam)
    k = d.kinds
    out["parallel"] = [
        ("VISTA", d.segs[k == KIND_EDGE]),
        ("VISTA-PERFIL", d.segs[k == KIND_PROFILE]),
        ("VISTA-CORTE", d.segs[k == KIND_CUT])]
    d2 = hlr_drawing(s, cam, hidden=True)
    k2 = d2.kinds
    out["hidden"] = [
        ("Planta baja", d2.segs[k2 == KIND_EDGE]),
        ("Planta baja-PERFIL", d2.segs[k2 == KIND_PROFILE]),
        ("Planta baja-CORTE", d2.segs[k2 == KIND_CUT]),
        ("Planta baja-OCULTAS", d2.segs[k2 == KIND_HIDDEN], "DASHED")]
    cam.perspective = True
    out["perspective"] = [("VISTA", hlr_perspective(s, cam))]
    # The same drawing as float32, what a GPU-side array would hand over.
    out["float32"] = [("VISTA", d.segs.astype("float32"))]
    return out


def synthetic_groups():
    rnd = random.Random(20260901)
    wild = [0.0, -0.0, 1e-5, 1.23456789e-5, 9.99999999e-5, 1e-4, 0.0001234567891,
            0.1 + 0.2, 1 / 3, 2 / 3, 100.0, 123456789.0, 123456789.5,
            999999999.5, 1234567890.0, 1e9, 1e15, 1.5e20, 1e100, -1e-300,
            5e-324, 1.7976931348623157e308, 12.5, 0.5, 2.5e-7, -3.75, 1e8,
            99999999.95, 0.000099999999995, 7.0, float("inf"), float("-inf"),
            float("nan")]
    segs = []
    for i in range(0, len(wild) - 3):
        segs.append((wild[i], wild[i + 1], wild[i + 2], wild[i + 3]))
    for _ in range(300):
        e = rnd.choice([-12, -6, -3, -1, 0, 1, 3, 6, 8, 9, 12])
        segs.append(tuple(rnd.uniform(-1, 1) * 10.0 ** e for _ in range(4)))
    return {
        "numbers": [("VISTA", segs)],
        "names": [("vista 1", [(0, 0, 1, 1)]), ("  ", [(1, 1, 2, 2)]),
                  ("", [(2, 2, 3, 3)]), ("Planta  alta", []),
                  ("ñandú ábc ß", [(3, 3, 4, 4)]), ("vista 1", [(4, 4, 5, 5)]),
                  ("VISTA_1", [(5, 5, 6, 6)])],
        "empty": [],
        "empty_layers": [("A", []), ("B", [], "DASHED")],
        "dashed_second": [("A", [(0, 0, 1, 0)]), ("B", [(0, 1, 1, 1)], "DASHED"),
                          ("C", [(0, 2, 1, 2)], "")],
    }


def write_export() -> None:
    exp = OUT / "export"
    exp.mkdir(parents=True, exist_ok=True)
    groups = {}
    groups.update(hlr_groups())
    groups.update(synthetic_groups())
    cases = []
    for name, gs in groups.items():
        path = exp / f"{name}.dxf"
        n = dxf_out.save_dxf_layers(path, [tuple(g) for g in gs])
        row = {"name": name, "lines": n, "groups": []}
        for g in gs:
            segs = [[hexbits(v) for v in s] for s in g[1]]
            row["groups"].append({"layer": g[0], "segs": segs,
                                  "ltype": g[2] if len(g) > 2 else None})
        cases.append(row)
    # save_dxf_lines is save_dxf_layers with one group.
    n = dxf_out.save_dxf_lines(exp / "lines.dxf", [(0, 0, 3, 4), (1.5, 2, 3, 4)],
                               layer="muros")
    cases.append({"name": "lines", "lines": n, "single": "muros", "groups": [
        {"layer": "muros", "ltype": None,
         "segs": [[hexbits(v) for v in s] for s in [(0, 0, 3, 4), (1.5, 2, 3, 4)]]}]})
    write_json(exp / "cases.json", cases)
    # Python's .9g over many doubles, for the number formatter alone.
    rnd = random.Random(7)
    lines = []
    for i in range(4000):
        e = rnd.choice(range(-30, 30)) if i % 3 else rnd.choice([-5, -4, 8, 9, 10])
        x = rnd.uniform(-10, 10) * 10.0 ** e
        if i % 7 == 0:
            x = round(x, rnd.randint(0, 9))
        if i % 11 == 0:
            x = float(f"{x:.9g}")
        lines.append(f"{hexbits(x)} {x:.9g}")
    (exp / "g9.txt").write_text("\n".join(lines) + "\n", encoding="ascii",
                                newline="\n")
    print(len(cases), "export cases")


# ---- import ----------------------------------------------------------------------

def new_doc(version="R2018", units=6, setup=False):
    doc = ezdxf.new(version, setup=setup)
    if version != "R12":
        doc.header["$INSUNITS"] = units if units is not None else 0
    return doc


def faces_grid(msp, n, z0=0.0, layer=None):
    attribs = {"layer": layer} if layer else {}
    for i in range(n):
        for j in range(n):
            if (i, j) == (n // 2, n // 2):
                continue
            msp.add_3dface([(i, j, z0), (i + 1, j, z0), (i + 1, j + 1, z0),
                            (i, j + 1, z0)], dxfattribs=attribs)
            msp.add_3dface([(i, 0, j + 10), (i + 1, 0, j + 10),
                            (i + 1, 0, j + 11), (i, 0, j + 11)],
                           dxfattribs=attribs)
    msp.add_3dface([(0, 0, 40), (3, 0, 40), (3, 2, 41), (0, 2, 41)],
                   dxfattribs=attribs)


def slim(path: Path) -> None:
    """Keep the HEADER variables the importer reads, BLOCKS and ENTITIES of a
    file ezdxf wrote: the tables and objects are 90 % of its bytes."""
    text = path.read_text(encoding="latin-1").replace("\r\n", "\n")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    pairs = [(lines[i].strip(), lines[i + 1]) for i in range(0, len(lines) - 1, 2)]
    secs = []
    i = 0
    while i < len(pairs):
        if pairs[i] == ("0", "SECTION"):
            j = i + 2
            while pairs[j] != ("0", "ENDSEC"):
                j += 1
            secs.append((pairs[i + 1][1], pairs[i:j + 1]))
            i = j + 1
        else:
            i += 1
    res = []
    for name, ps in secs:
        if name == "HEADER":
            body = ps[2:-1]
            res += [("0", "SECTION"), ("2", "HEADER")]
            k = 0
            while k < len(body):
                if body[k][0] == "9":
                    n = k + 1
                    while n < len(body) and body[n][0] != "9":
                        n += 1
                    if body[k][1] in ("$ACADVER", "$INSUNITS", "$DWGCODEPAGE"):
                        res += body[k:n]
                    k = n
                else:
                    k += 1
            res += [("0", "ENDSEC")]
        elif name in ("BLOCKS", "ENTITIES"):
            res += ps
    res += [("0", "EOF")]
    path.write_text("".join(f"{c}\n{v}\n" for c, v in res), encoding="latin-1",
                    newline="")


def build_docs(dst: Path) -> list:
    """(file, scale, extra) for every ezdxf-made file written to ``dst``."""
    made = []

    def save(doc, fname, scale=1.0, full=False, **extra):
        doc.saveas(str(dst / fname))
        if not full:
            slim(dst / fname)
        made.append((fname, scale, extra))

    # -- linework --------------------------------------------------------------------
    doc = new_doc(setup=True)
    msp = doc.modelspace()
    msp.add_line((0, 0), (10, 0))
    msp.add_line((0, 0, 0), (3, 4, 5))
    msp.add_line((1, 1, 1), (1, 1, 1))                       # zero length
    msp.add_lwpolyline([(0, 0), (0, 5), (10, 5), (10, 0)])
    msp.add_lwpolyline([(0, 8), (4, 8), (4, 12)], close=True)
    msp.add_lwpolyline([(20, 0), (20, 0), (21, 0)])           # repeated vertex
    msp.add_lwpolyline([(5, 5)])                              # one point
    save(doc, "lines.dxf", full=True)

    doc = new_doc()
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0, 0, 0, 1.0), (4, 0, 0, 0, 0)])
    msp.add_lwpolyline([(0, 5), (4, 5), (4, 9)])
    msp.add_lwpolyline([(10, 0, 0, 0, -0.5), (14, 0, 0, 0, 0.3),
                        (14, 4, 0, 0, 2.0), (10, 4, 0, 0, 0)], close=True)
    msp.add_lwpolyline([(20, 0, 0, 0, 0.4), (23, 2, 0, 0, 0),
                        (26, 0, 0, 0, 1e-7)], close=True)
    msp.add_lwpolyline([(30, 0, 0, 0, 0.7), (30, 0, 0, 0, 0)])  # coincident
    msp.add_lwpolyline([(0, 20, 0, 0, 3.0), (6, 20, 0, 0, -3.0),
                        (6, 26, 0, 0, 0.1)])
    save(doc, "bulge.dxf")

    doc = new_doc()
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0, 0, 0, 0.5), (4, 0), (4, 3)], close=True,
                       dxfattribs={"elevation": 2.5})
    msp.add_lwpolyline([(0, 0, 0, 0, 0.5), (4, 0), (4, 3)], close=True,
                       dxfattribs={"extrusion": (0, 0, -1), "elevation": 1.0})
    msp.add_lwpolyline([(1, 1), (5, 1), (5, 4)],
                       dxfattribs={"extrusion": (1, 1, 1), "elevation": 0.5})
    msp.add_lwpolyline([(1, 1, 0, 0, -0.8), (5, 1), (5, 4)],
                       dxfattribs={"extrusion": (0.3, -0.2, 0.9)})
    msp.add_lwpolyline([(1, 1), (5, 1)], dxfattribs={"extrusion": (1, 0, 0)})
    msp.add_lwpolyline([(1, 1), (5, 1)], dxfattribs={"extrusion": (0, 0, 0)})
    save(doc, "lw_ocs.dxf")

    doc = new_doc()
    msp = doc.modelspace()
    msp.add_polyline2d([(0, 0), (5, 0), (5, 5)])
    msp.add_polyline2d([(0, 10, 0, 0, 0.5), (5, 10), (5, 15)], format="xyseb",
                       close=True)
    msp.add_polyline2d([(20, 0), (25, 0), (25, 5), (20, 5)], close=True,
                       dxfattribs={"elevation": (0, 0, 3.0)})
    msp.add_polyline3d([(0, 30, 0), (5, 30, 1), (5, 35, 2)])
    msp.add_polyline3d([(10, 30, 0), (15, 30, 1), (15, 35, 2), (10, 35, 3)],
                       close=True)
    msp.add_polyline3d([(40, 0, 0), (40, 0, 0), (41, 0, 0), (41, 0, 0)],
                       close=True)
    msp.add_polyline3d([(50, 0, 0)])
    pm = msp.add_polymesh((3, 3))
    for i in range(3):
        for j in range(3):
            pm.set_mesh_vertex((i, j), (60 + i, j, (i + j) % 2))
    pf = msp.add_polyface()
    pf.append_face([(70, 0, 0), (71, 0, 0), (71, 1, 0), (70, 1, 0)])
    pf.append_face([(71, 0, 0), (72, 0, 0), (72, 1, 0), (71, 1, 0)])
    save(doc, "polylines.dxf")

    # -- curves ---------------------------------------------------------------------
    doc = new_doc()
    msp = doc.modelspace()
    msp.add_circle((0, 0), 2.0)
    msp.add_circle((10, 0, 3), 0.5)
    msp.add_circle((30, 0), 1e-13)                           # radius ~ 0
    msp.add_circle((40, 0), 250.0)
    msp.add_circle((60, 0), 5.0, dxfattribs={"extrusion": (0, 0, -1)})
    msp.add_circle((70, 0, 1), 5.0, dxfattribs={"extrusion": (0, 1, 0)})
    msp.add_circle((80, 0, 1), 5.0,
                   dxfattribs={"extrusion": (0.3, 0.4, 0.8)})
    save(doc, "circles.dxf")

    doc = new_doc()
    msp = doc.modelspace()
    msp.add_arc((0, 0), 3.0, 0, 90)
    msp.add_arc((10, 0), 3.0, 270, 90)
    msp.add_arc((20, 0), 3.0, 350, 10)
    msp.add_arc((30, 0), 3.0, 0, 360)
    msp.add_arc((40, 0), 3.0, 45.5, 45.5)
    msp.add_arc((50, 0), 3.0, 10, 190.123456)
    msp.add_arc((60, 0), 3.0, -30, 200)
    msp.add_arc((70, 0), 3.0, 0, 0.000001)
    msp.add_arc((80, 0, 2), 3.0, 30, 120, dxfattribs={"extrusion": (0, 0, -1)})
    msp.add_arc((90, 0, 2), 3.0, 30, 300,
                dxfattribs={"extrusion": (1, 2, 3)})
    msp.add_arc((100, 0), 12.0, 10, 350)
    save(doc, "arcs.dxf")

    doc = new_doc()
    msp = doc.modelspace()
    msp.add_ellipse((0, 0), (4, 0), 0.5)
    msp.add_ellipse((20, 0), (3, 3), 0.25, 0, math.pi)
    msp.add_ellipse((40, 0), (0, 5, 0), 0.6, 1.0, 4.0)
    msp.add_ellipse((60, 0, 2), (5, 0, 0), 1.0, 0.0, math.tau,
                    dxfattribs={"extrusion": (0, 0, -1)})
    msp.add_ellipse((80, 0), (6, 1, 2), 0.3, 0.5, 5.5,
                    dxfattribs={"extrusion": (0, 0.5, 1)})
    msp.add_ellipse((100, 0), (4, 0), 0.5, 0.0, 1e-12)
    save(doc, "ellipses.dxf")

    doc = new_doc()
    msp = doc.modelspace()
    msp.add_open_spline([(0, 0), (1, 2), (3, 3), (5, 1), (7, 4), (9, 0)],
                        degree=3)
    msp.add_open_spline([(0, 10, 0), (2, 12, 1), (4, 9, 2), (6, 11, 0),
                         (8, 10, 3)], degree=3)
    msp.add_open_spline([(0, 20), (1, 22), (3, 23), (5, 21), (7, 24)],
                        degree=3, knots=[0, 0, 0, 0, 0.4, 1, 1, 1, 1][:9])
    msp.add_open_spline([(0, 30), (2, 33), (4, 30), (6, 33)], degree=3)
    save(doc, "splines.dxf")

    # Splines ezdxf approximates by cubic Beziers: periodic, other degrees,
    # rational.
    doc = new_doc()
    msp = doc.modelspace()
    msp.add_line((0, 0), (1, 0))
    from ezdxf.math import closed_uniform_bspline
    msp.add_spline().apply_construction_tool(
        closed_uniform_bspline([(0, 20), (4, 24), (8, 20), (4, 16)], order=4))
    msp.add_open_spline([(0, 40), (2, 43), (5, 40), (7, 44), (9, 40)],
                        degree=2)
    msp.add_rational_spline([(0, 50), (2, 53), (4, 50), (6, 53)],
                            weights=[1, 2, 1, 2], degree=3)
    msp.add_open_spline([(0, 60), (2, 63), (5, 60), (7, 64), (9, 60),
                         (12, 66), (14, 61)], degree=5)
    save(doc, "splines_approx.dxf")

    # A spline that only has fit points (not ported: ezdxf solves a system
    # through NumPy for its control points).
    doc = new_doc()
    msp = doc.modelspace()
    msp.add_line((0, 0), (1, 0))
    msp.add_spline(fit_points=[(0, 5), (2, 7), (4, 5), (6, 8)])
    save(doc, "splines_fit.dxf", pending="SPLINE from fit points")

    # -- faces ----------------------------------------------------------------------
    doc = new_doc()
    msp = doc.modelspace()
    msp.add_3dface([(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)])
    msp.add_3dface([(10, 0, 0), (12, 0, 0), (12, 2, 1)])
    msp.add_3dface([(20, 0, 0), (22, 0, 0), (22, 0, 0), (20, 2, 0)])
    msp.add_3dface([(30, 0, 0), (30, 0, 0), (30, 0, 0), (30, 0, 0)])
    msp.add_3dface([(40, 0, 0), (42, 0, 0), (42, 2, 0), (40, 2, 0)],
                   dxfattribs={"color": 1})
    msp.add_solid([(50, 0), (52, 0), (50, 2), (52, 2)])
    msp.add_solid([(60, 0), (62, 0), (60, 2)])
    msp.add_solid([(70, 0, 5), (72, 0, 5), (70, 2, 5), (72, 2, 5)])
    msp.add_trace([(80, 0), (82, 0), (80, 2), (82, 2)])
    msp.add_solid([(90, 0), (92, 0), (92, 0), (92, 0)])
    msp.add_solid([(100, 0), (102, 0), (100, 2), (102, 2)],
                  dxfattribs={"extrusion": (0, 0, -1)})
    save(doc, "faces.dxf")

    doc = new_doc()
    msp = doc.modelspace()
    msp.add_3dface([(0, 0, 0), (4, 0, 0), (4, 3, 0)])
    msp.add_3dface([(0, 0, 0), (4, 3, 0), (0, 3, 0)])
    msp.add_3dface([(10, 0, 0), (14, 0, 0), (14, 0, 3)])
    msp.add_3dface([(10, 0, 0), (14, 0, 3), (10, 0, 3)])
    msp.add_3dface([(20, 0, 0), (24, 0, 0), (22, 3, 0)])        # lone triangle
    msp.add_3dface([(0, 10, 0), (4, 10, 0), (4, 13, 0), (0, 13, 0)],
                   dxfattribs={"layer": "LOSA"})
    msp.add_3dface([(4, 10, 0), (8, 10, 0), (8, 13, 0), (4, 13, 0)],
                   dxfattribs={"layer": "LOSA"})
    save(doc, "faces_merge.dxf")

    for n, tag in ((15, "bigfaces"), (8, "midfaces")):
        doc = new_doc()
        msp = doc.modelspace()
        faces_grid(msp, n)
        save(doc, f"{tag}.dxf")
    doc = new_doc()                                   # two layers, one big
    msp = doc.modelspace()
    faces_grid(msp, 13, layer="PISO")
    msp.add_3dface([(0, 0, 0), (1, 0, 0), (1, 1, 0)],
                   dxfattribs={"layer": "OTRO"})
    msp.add_line((0, 0), (5, 5), dxfattribs={"layer": "PISO"})
    save(doc, "bigfaces_layers.dxf")

    # -- blocks ---------------------------------------------------------------------
    doc = new_doc(setup=True)
    doc.layers.add("MOBILIARIO")
    doc.layers.add("ARBOLES")
    blk = doc.blocks.new("ARBOL")
    blk.add_line((0, 0), (0, 2))
    blk.add_circle((0, 2.5), 0.5)
    blk.add_lwpolyline([(0, 0, 0, 0, 0.5), (1, 0), (1, 1)])
    banca = doc.blocks.new("BANCA")
    banca.add_line((0, 0), (1, 0))
    banca.add_3dface([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    based = doc.blocks.new("BASED", base_point=(5, 5, 0))
    based.add_line((5, 5), (7, 5))
    based.add_line((7, 5, 0), (7, 8, 2))
    rotulo = doc.blocks.new("ROTULO")
    rotulo.add_text("N.P.T. +0.15")
    msp = doc.modelspace()
    msp.add_blockref("ARBOL", (10, 10))
    msp.add_blockref("ARBOL", (20, 10), dxfattribs={"layer": "ARBOLES"})
    msp.add_blockref("ARBOL", (30, 10), dxfattribs={
        "rotation": 90.0, "xscale": 2.0, "yscale": 2.0})
    msp.add_blockref("ARBOL", (40, 10), dxfattribs={
        "rotation": 33.3, "xscale": 1.5, "yscale": 0.5, "zscale": 3.0})
    msp.add_blockref("ARBOL", (50, 10), dxfattribs={
        "xscale": -1.0, "layer": "ARBOLES"})
    msp.add_blockref("ARBOL", (60, 10, 4), dxfattribs={
        "extrusion": (0, 0, -1), "rotation": 15})
    msp.add_blockref("ARBOL", (70, 10, 2), dxfattribs={
        "extrusion": (1, 1, 1), "rotation": 45, "xscale": 1.2})
    msp.add_blockref("BANCA", (0, 0), dxfattribs={"layer": "MOBILIARIO"})
    msp.add_blockref("BANCA", (3, 0), dxfattribs={"rotation": 180})
    msp.add_blockref("BASED", (100, 100), dxfattribs={
        "rotation": 30, "xscale": 2.0, "yscale": 2.0, "zscale": 2.0})
    msp.add_blockref("ROTULO", (5, 5))
    msp.add_blockref("NOEXISTE", (6, 6))
    msp.add_blockref("arbol", (80, 10))
    msp.add_line((0, 0), (2, 0))
    save(doc, "blocks.dxf", full=True)

    doc = new_doc()
    doc.layers.add("MUROS")
    leaf = doc.blocks.new("HOJA")
    leaf.add_line((0, 0), (0, 1))
    leaf.add_line((0, 1), (1, 1))
    leaf.add_3dface([(0, 0, 0), (1, 0, 0), (1, 1, 0)])
    leaf.add_lwpolyline([(0, 0), (1, 0), (1, 1)])
    leaf.add_circle((0.5, 0.5), 0.25)
    mid = doc.blocks.new("RAMA", base_point=(1, 0, 0))
    mid.add_blockref("HOJA", (1, 0))
    mid.add_blockref("HOJA", (2, 0), dxfattribs={"rotation": 90,
                                                  "layer": "MUROS"})
    mid.add_line((1, 0), (3, 0), dxfattribs={"layer": "MUROS"})
    top = doc.blocks.new("ARBOLITO")
    top.add_blockref("RAMA", (0, 0), dxfattribs={"xscale": 2, "yscale": 2})
    top.add_blockref("RAMA", (0, 3), dxfattribs={"rotation": 45})
    msp = doc.modelspace()
    msp.add_blockref("ARBOLITO", (5, 5), dxfattribs={"rotation": 20})
    msp.add_blockref("ARBOLITO", (15, 5), dxfattribs={"layer": "MUROS"})
    msp.add_blockref("RAMA", (0, 20))
    save(doc, "nested.dxf")

    doc = new_doc()                                  # a block that includes itself
    loop = doc.blocks.new("BUCLE")
    loop.add_line((0, 0), (1, 0))
    loop.add_blockref("BUCLE", (1, 0))
    doc.modelspace().add_blockref("BUCLE", (0, 0))
    save(doc, "recursive.dxf")

    doc = new_doc()
    blk = doc.blocks.new("POSTE")
    blk.add_line((0, 0), (0, 3))
    blk.add_circle((0, 0), 0.2)
    msp = doc.modelspace()
    ref = msp.add_blockref("POSTE", (0, 0))
    ref.dxf.row_count = 3
    ref.dxf.column_count = 4
    ref.dxf.row_spacing = 5.0
    ref.dxf.column_spacing = 6.0
    ref.dxf.rotation = 30
    save(doc, "minsert.dxf")

    # -- layers, colours, skipped kinds ------------------------------------------------
    doc = new_doc()
    doc.layers.add("MUROS", color=1)
    doc.layers.add("EJES", color=3)
    doc.layers.add("VACIA")
    msp = doc.modelspace()
    msp.add_line((0, 0), (1, 0), dxfattribs={"layer": "MUROS"})
    msp.add_line((0, 1), (1, 1), dxfattribs={"layer": "EJES", "color": 5})
    msp.add_line((0, 2), (1, 2))
    msp.add_circle((5, 5), 1, dxfattribs={"layer": "MUROS"})
    msp.add_3dface([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
                   dxfattribs={"layer": "EJES"})
    save(doc, "layers.dxf", name="Mi plano")

    doc = new_doc()
    msp = doc.modelspace()
    msp.add_line((0, 0), (1, 0))
    msp.add_text("COTA 3.50")
    msp.add_text("N.P.T.")
    msp.add_mtext("varias\nlineas")
    msp.add_point((3, 3))
    hatch = msp.add_hatch()
    hatch.paths.add_polyline_path([(0, 0), (2, 0), (2, 2), (0, 2)])
    msp.add_ray((0, 0), (1, 1))
    msp.add_xline((0, 0), (1, 0))
    msp.add_mesh()
    msp.add_leader([(0, 0), (1, 1)])
    dim = msp.add_linear_dim(base=(0, -2), p1=(0, 0), p2=(4, 0))
    dim.render()
    blk = doc.blocks.new("CONATT")
    blk.add_line((0, 0), (1, 0))
    blk.add_attdef("TAG", (0, 0), text="x")
    blk.add_text("dentro")
    blk.add_point((0, 0))
    msp.add_blockref("CONATT", (9, 9))
    save(doc, "skipped.dxf")

    doc = new_doc()
    doc.modelspace().add_text("solo texto")
    save(doc, "empty.dxf")

    doc = new_doc()                                  # paper space has content
    doc.modelspace().add_line((0, 0), (1, 1))
    layout = doc.layout("Layout1")
    layout.add_line((0, 0), (100, 100))
    layout.add_circle((50, 50), 10)
    save(doc, "paperspace.dxf")

    # -- units and survey coordinates -----------------------------------------------------
    for code, tag in ((4, "mm"), (5, "cm"), (1, "in"), (2, "ft"), (0, "unitless"),
                      (8, "other")):
        doc = new_doc(units=code)
        msp = doc.modelspace()
        for i in range(6):
            msp.add_line((i * 100, 0), (i * 100, 3500))
        msp.add_circle((50, 50), 300)
        save(doc, f"units_{tag}.dxf", scale={4: 0.001, 5: 0.01, 1: 0.0254,
                                             2: 0.3048}.get(code, 1.0))
    doc = new_doc(units=4)                           # the lying header
    msp = doc.modelspace()
    for i in range(30):
        msp.add_line((i, 0), (i, 3.0 + (i % 4)))
    save(doc, "units_lying.dxf", scale=1.0)
    doc = new_doc(units=6)                           # lying the other way
    msp = doc.modelspace()
    for i in range(30):
        msp.add_line((i * 1000, 0), (i * 1000, 3500))
    save(doc, "units_lying2.dxf", scale=0.001)
    doc = new_doc(units=6)
    msp = doc.modelspace()
    for i in range(30):
        msp.add_line((0, i * 0.002), (0.004, i * 0.002))
    msp.add_lwpolyline([(0, 0), (0.002, 0), (0.002, 0.003)])
    msp.add_arc((0, 0), 0.001, 0, 90)
    save(doc, "units_tiny.dxf")
    doc = new_doc(units=6)          # more sizes than the 4000 it looks at
    msp = doc.modelspace()
    msp.add_lwpolyline([(i * 0.01 * (1 + i % 3), (i * 7) % 5) for i in range(4100)])
    for i in range(200):
        msp.add_line((0, i), (1e5, i))
    save(doc, "units_longpoly.dxf", nosig=True)

    doc = new_doc()
    east, north = 276_000.0, 8_214_000.0
    doc.modelspace().add_line((east, north), (east + 12, north + 8))
    doc.modelspace().add_circle((east + 5.25, north + 3.1), 2.0)
    doc.modelspace().add_3dface([(east, north, 3800.5), (east + 3, north, 3800.5),
                                 (east + 3, north + 3, 3801.5)])
    save(doc, "utm.dxf")
    doc = new_doc()
    doc.modelspace().add_line((100, 200), (300, 400))
    save(doc, "local.dxf")
    doc = new_doc()                                  # vertical line: lo[0] == hi[0]
    doc.modelspace().add_line((200_000, 0), (200_000, 5))
    save(doc, "utm_vertical.dxf")
    doc = new_doc()
    blk = doc.blocks.new("P")
    blk.add_line((0, 0), (1, 0))
    doc.modelspace().add_blockref("P", (500_000, 7_000_000))
    doc.modelspace().add_line((500_000, 7_000_000), (500_010, 7_000_020))
    save(doc, "utm_blocks.dxf")

    # -- versions ------------------------------------------------------------------------
    for version in ("R12", "R2000", "R2007"):
        doc = new_doc(version)
        msp = doc.modelspace()
        doc.layers.add("CAPA-ÑÁ" if version != "R12" else "CAPA1")
        msp.add_line((0, 0), (4, 0), dxfattribs={
            "layer": "CAPA-ÑÁ" if version != "R12" else "CAPA1"})
        msp.add_lwpolyline([(0, 1), (4, 1), (4, 3)]) if version != "R12" else \
            msp.add_polyline2d([(0, 1), (4, 1), (4, 3)])
        msp.add_circle((8, 2), 1.5)
        msp.add_3dface([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
        msp.add_solid([(10, 0), (12, 0), (10, 2), (12, 2)])
        blk = doc.blocks.new("CAJA")
        blk.add_line((0, 0), (1, 1))
        msp.add_blockref("CAJA", (20, 20), dxfattribs={"rotation": 10})
        save(doc, f"version_{version}.dxf", full=version == "R2018")
    return made




# ---- random documents -----------------------------------------------------------

NESTED_ENTS = {
    "solid": lambda b: b.add_solid([(0, 0), (2, 0), (0, 2), (2, 2)]),
    "solid_ext": lambda b: b.add_solid([(0, 0), (2, 0), (0, 2), (2, 2)], dxfattribs={"extrusion": (0.3, 0.2, 0.9)}),
    "poly2": lambda b: b.add_polyline2d([(0, 0), (3, 0), (3, 2), (1, 4)]),
    "poly2_elev": lambda b: b.add_polyline2d([(0, 0), (3, 0), (3, 2)], dxfattribs={"elevation": (0, 0, 1.5)}),
    "poly2_bulge": lambda b: b.add_polyline2d([(0, 0, 0, 0, 0.5), (3, 0, 0, 0, 0), (3, 2, 0, 0, 0)], format="xyseb"),
    "poly2_ext": lambda b: b.add_polyline2d([(0, 0), (3, 0), (3, 2)], dxfattribs={"extrusion": (0, 1, 1)}),
    "poly3": lambda b: b.add_polyline3d([(0, 0, 0), (3, 0, 1), (3, 2, 2)], close=True),
    "lw": lambda b: b.add_lwpolyline([(0, 0), (3, 0), (3, 2)]),
    "lw_bulge": lambda b: b.add_lwpolyline([(0, 0, 0, 0, 0.5), (3, 0), (3, 2)], close=True),
    "lw_ext": lambda b: b.add_lwpolyline([(0, 0, 0, 0, 0.5), (3, 0), (3, 2)], dxfattribs={"extrusion": (0.1, 0.9, 0.3), "elevation": 0.4}),
    "circle": lambda b: b.add_circle((1, 1, 0.5), 2.0),
    "circle_ext": lambda b: b.add_circle((1, 1, 0.5), 2.0, dxfattribs={"extrusion": (0.3, 0.2, 0.9)}),
    "arc": lambda b: b.add_arc((1, 1, 0.5), 2.0, 20, 200),
    "arc_ext": lambda b: b.add_arc((1, 1, 0.5), 2.0, 20, 300, dxfattribs={"extrusion": (0.3, 0.2, 0.9)}),
    "arc_semi": lambda b: b.add_arc((1, 1, 0.5), 2.0, 20, 200),
    "ellipse": lambda b: b.add_ellipse((1, 1), (3, 0), 0.5, 0.2, 5.0),
    "ellipse_full": lambda b: b.add_ellipse((1, 1, 1), (3, 1, 0), 0.4),
    "spline": lambda b: b.add_open_spline([(0, 0), (1, 2), (3, 3), (5, 1), (7, 4)], degree=3),
    "line": lambda b: b.add_line((0, 0, 0), (2, 3, 1)),
    "face": lambda b: b.add_3dface([(0, 0, 0), (2, 0, 0), (2, 2, 1), (0, 2, 1)]),
}
NESTED_STYLES = {
    "id": {},
    "rot": {"rotation": 37.0},
    "uni": {"xscale": 2, "yscale": 2, "zscale": 2, "rotation": 15},
    "neg": {"xscale": -1.5, "yscale": 1.5, "zscale": 1.5},
    "nonuni": {"xscale": 2, "yscale": 0.5, "zscale": 1.2},
    "nonuni_rot": {"xscale": 2, "yscale": 0.5, "zscale": 1.2, "rotation": 50},
    "flip": {"xscale": 0.3, "yscale": 3.0, "zscale": 1.0, "rotation": 25},
    "ext": {"extrusion": (0.2, -0.3, 0.9), "rotation": 20},
    "extneg": {"extrusion": (0, 0, -1), "xscale": 1.3, "yscale": 1.3, "zscale": 1.3},
}



def nested_docs(dst: Path) -> list:
    """One entity kind in a block placed through two levels of references,
    each with a different insert style (rotation, uniform and non-uniform
    scale, mirror, tilted extrusion): ezdxf moves the entity by the block's
    matrix and flattens it afterwards."""
    made = []
    for en, ef in NESTED_ENTS.items():
        for sn in ("id", "uni", "neg", "nonuni_rot", "ext", "flip"):
            style = NESTED_STYLES[sn]
            doc = ezdxf.new("R2018", setup=False)
            leaf = doc.blocks.new("LEAF", base_point=(0.4, -0.2, 0.1))
            ef(leaf)
            mid = doc.blocks.new("MID")
            mid.add_blockref("LEAF", (3, 4, 1), dxfattribs=style)
            top = doc.blocks.new("TOP")
            top.add_blockref("MID", (1, 1, 1), dxfattribs=style)
            doc.modelspace().add_blockref("TOP", (5, 5, 5), dxfattribs={"rotation": 10})
            name = f"m_{en}_{sn}.dxf"
            doc.saveas(str(dst / name))
            slim(dst / name)
            made.append((name, 1.0, {}))
    return made


def random_docs(dst: Path, seeds) -> list:
    made = []
    for seed in seeds:
        doc, scale = random_doc(seed)
        name = f"f{seed}.dxf"
        doc.saveas(str(dst / name))
        slim(dst / name)
        made.append((name, scale, {}))
    return made


def _rnd_ext(r):
    c = r.random()
    if c < 0.6:
        return None
    if c < 0.7:
        return (0, 0, -1)
    if c < 0.8:
        return (1, 0, 0)
    return tuple(r.uniform(-1, 1) for _ in range(3))


def _pt(r, mag, z=True):
    return (r.uniform(-mag, mag), r.uniform(-mag, mag), r.uniform(-mag, mag) if z else 0.0)


def _add_entities(r, layout, mag, n, blocknames, depth_ok=True):
    for _ in range(n):
        k = r.choice(["line", "lw", "lw", "poly2", "poly3", "circle", "arc", "ellipse",
                      "face", "face", "solid", "spline", "text", "point", "insert", "insert"])
        attribs = {}
        if r.random() < 0.3:
            attribs["layer"] = r.choice(["A", "B", "MUROS", "cañería"])
        try:
            if k == "line":
                layout.add_line(_pt(r, mag), _pt(r, mag), dxfattribs=attribs)
            elif k == "lw":
                m = r.randint(1, 7)
                pts = [(r.uniform(-mag, mag), r.uniform(-mag, mag), 0, 0,
                        r.choice([0, 0, 0, r.uniform(-2, 2), 1.0, -1.0, 1e-7])) for _ in range(m)]
                a = dict(attribs)
                e = _rnd_ext(r)
                if e:
                    a["extrusion"] = e
                if r.random() < 0.3:
                    a["elevation"] = r.uniform(-mag, mag)
                layout.add_lwpolyline(pts, close=r.random() < 0.4, dxfattribs=a)
            elif k == "poly2":
                m = r.randint(2, 6)
                pts = [(r.uniform(-mag, mag), r.uniform(-mag, mag), 0, 0,
                        r.choice([0, 0, r.uniform(-2, 2)])) for _ in range(m)]
                a = dict(attribs)
                e = _rnd_ext(r)
                if e:
                    a["extrusion"] = e
                if r.random() < 0.5:
                    a["elevation"] = (0, 0, r.uniform(-mag, mag))
                layout.add_polyline2d(pts, format="xyseb", close=r.random() < 0.4, dxfattribs=a)
            elif k == "poly3":
                m = r.randint(2, 6)
                layout.add_polyline3d([_pt(r, mag) for _ in range(m)], close=r.random() < 0.4, dxfattribs=attribs)
            elif k == "circle":
                a = dict(attribs)
                e = _rnd_ext(r)
                if e:
                    a["extrusion"] = e
                layout.add_circle(_pt(r, mag), r.choice([r.uniform(0.01, mag), 1e-13, mag * 3]), dxfattribs=a)
            elif k == "arc":
                a = dict(attribs)
                e = _rnd_ext(r)
                if e:
                    a["extrusion"] = e
                s = r.choice([0, r.uniform(-360, 720), 90, 270])
                en = r.choice([360, r.uniform(-360, 720), 90, s, s + 1e-7])
                layout.add_arc(_pt(r, mag), r.uniform(0.01, mag), s, en, dxfattribs=a)
            elif k == "ellipse":
                a = dict(attribs)
                e = _rnd_ext(r)
                if e:
                    a["extrusion"] = e
                maj = _pt(r, mag)
                layout.add_ellipse(_pt(r, mag), maj, r.uniform(0.05, 1), r.uniform(0, 3), r.choice([math.tau, r.uniform(3, 7)]), dxfattribs=a)
            elif k == "face":
                p = [_pt(r, mag) for _ in range(4)]
                if r.random() < 0.4:
                    p = p[:3]
                if r.random() < 0.15:
                    p[1] = p[0]
                layout.add_3dface(p, dxfattribs=attribs)
            elif k == "solid":
                a = dict(attribs)
                e = _rnd_ext(r)
                if e:
                    a["extrusion"] = e
                layout.add_solid([_pt(r, mag, False) for _ in range(r.choice([3, 4]))], dxfattribs=a)
            elif k == "spline":
                m = r.randint(4, 8)
                layout.add_open_spline([_pt(r, mag) for _ in range(m)], degree=3, dxfattribs=attribs)
            elif k == "text":
                layout.add_text("T", dxfattribs=attribs)
            elif k == "point":
                layout.add_point(_pt(r, mag), dxfattribs=attribs)
            elif k == "insert" and blocknames:
                a = dict(attribs)
                if r.random() < 0.5:
                    a["rotation"] = r.choice([0, 90, 180, r.uniform(0, 360)])
                c = r.random()
                if c < 0.4:
                    s = r.choice([1, 2, 0.5, -1, 1.5])
                    a.update(xscale=s, yscale=s, zscale=s)
                elif c < 0.55:
                    a.update(xscale=r.uniform(0.3, 2), yscale=r.uniform(0.3, 2), zscale=r.uniform(0.3, 2))
                e = _rnd_ext(r) if r.random() < 0.4 else None
                if e:
                    a["extrusion"] = e
                layout.add_blockref(r.choice(blocknames), _pt(r, mag), dxfattribs=a)
        except Exception as exc:  # ezdxf refusing a random parameter
            continue


def random_doc(seed):
    r = random.Random(seed)
    version = r.choice(["R2000", "R2018", "R2010"])
    doc = ezdxf.new(version)
    doc.header["$INSUNITS"] = r.choice([0, 4, 5, 6, 1])
    mag = r.choice([1.0, 5.0, 30.0, 300.0, 5000.0])
    nblocks = r.randint(0, 4)
    names = [f"B{i}" for i in range(nblocks)]
    for i, name in enumerate(names):
        blk = doc.blocks.new(name, base_point=r.choice([(0, 0, 0), _pt(r, 2)]))
        _add_entities(r, blk, mag / 5, r.randint(1, 6), names[:i] if r.random() < 0.6 else [])
    _add_entities(r, doc.modelspace(), mag, r.randint(2, 18), names)
    if r.random() < 0.15:
        # a layer of coplanar triangles
        for i in range(r.randint(2, 6)):
            doc.modelspace().add_3dface([(i, 0, 0), (i + 1, 0, 0), (i + 1, 1, 0)])
            doc.modelspace().add_3dface([(i, 0, 0), (i + 1, 1, 0), (i, 1, 0)])
    scales = [s for s in (1.0, 1.0, 0.001, 0.0254, 0.01) if mag * s <= 40.0]
    return doc, r.choice(scales)




def quirk_files() -> dict:
    """Hand-written DXF text that leans on ezdxf's reader: elevation in R12 and
    marker-less records, missing corners, layer names, attributes, structure
    errors, unterminated sections, odd numbers and line ends."""
    def T(*pairs):
        out = []
        for c, v in pairs:
            out.append(f"{c}\n{v}")
        return "\n".join(out) + "\n"


    def header(ver="AC1009", extra=""):
        return T((0, "SECTION"), (2, "HEADER"), (9, "$ACADVER"), (1, ver)) + extra + T((0, "ENDSEC"))


    def ents(body):
        return T((0, "SECTION"), (2, "ENTITIES")) + body + T((0, "ENDSEC"))


    def blocks(body):
        return T((0, "SECTION"), (2, "BLOCKS")) + body + T((0, "ENDSEC"))


    def line(x0, y0, x1, y1, layer="0", z0=None, z1=None):
        t = [(0, "LINE"), (8, layer), (10, x0), (20, y0)]
        if z0 is not None:
            t.append((30, z0))
        t += [(11, x1), (21, y1)]
        if z1 is not None:
            t.append((31, z1))
        return T(*t)


    def block(name, body, base=(0, 0, 0)):
        return T((0, "BLOCK"), (8, "0"), (2, name), (70, 0), (10, base[0]), (20, base[1]), (30, base[2]), (3, name), (1, "")) + body + T((0, "ENDBLK"), (8, "0"))


    EOF = T((0, "EOF"))
    files = {}

    # elevation (group 38) in R12 and in marker-less records
    circ = T((0, "CIRCLE"), (8, "0"), (10, 5), (20, 5), (38, 2.5), (40, 3))
    arc = T((0, "ARC"), (8, "0"), (10, 5), (20, 5), (38, -1.5), (40, 3), (50, 10), (51, 100))
    ins = T((0, "INSERT"), (8, "0"), (2, "B"), (10, 9), (20, 9), (38, 4.0))
    b = blocks(block("B", line(0, 0, 1, 1)))
    files["q_elev_r12.dxf"] = header("AC1009") + b + ents(circ + arc + ins) + EOF
    files["q_elev_r2000_nomarkers.dxf"] = header("AC1015") + b + ents(circ + arc + ins) + EOF
    circ100 = T((0, "CIRCLE"), (5, "AA"), (100, "AcDbEntity"), (8, "0"), (100, "AcDbCircle"), (10, 5), (20, 5), (38, 2.5), (40, 3))
    files["q_elev_r2000_markers.dxf"] = header("AC1015") + b + ents(circ100) + EOF
    # SOLID elevation on a marker-less record in AC1015
    solid = T((0, "SOLID"), (8, "0"), (10, 0), (20, 0), (11, 2), (21, 0), (12, 0), (22, 2), (13, 2), (23, 2), (38, 7.0))
    files["q_solid_elev.dxf"] = header("AC1015") + ents(solid) + EOF
    # 3DFACE with three corners only; SOLID with three
    f3 = T((0, "3DFACE"), (8, "0"), (10, 0), (20, 0), (30, 1), (11, 2), (21, 0), (31, 1), (12, 2), (22, 2), (32, 1))
    s3 = T((0, "SOLID"), (8, "0"), (10, 0), (20, 0), (11, 2), (21, 0), (12, 0), (22, 2))
    files["q_three_corners.dxf"] = header() + ents(f3 + s3 + line(0, 0, 3, 3)) + EOF
    files["q_three_corners_block.dxf"] = header() + blocks(block("C", f3 + s3)) + ents(
        T((0, "INSERT"), (8, "0"), (2, "C"), (10, 5), (20, 5), (50, 30)) +
        T((0, "INSERT"), (8, "0"), (2, "CC"), (10, 5), (20, 5))) + EOF
    files["q_nested_missing_vtx3.dxf"] = header() + blocks(
        block("C", s3) +
        block("D", T((0, "INSERT"), (8, "0"), (2, "C"), (10, 1), (20, 1), (50, 45), (41, 2)))) + ents(
        T((0, "INSERT"), (8, "0"), (2, "D"), (10, 5), (20, 5))) + EOF
    # no layer, spaces in layer names, unicode escapes
    files["q_layers.dxf"] = header() + ents(
        T((0, "LINE"), (10, 0), (20, 0), (11, 1), (21, 1)) +
        line(0, 1, 1, 2, " MUROS ") + line(0, 2, 1, 3, "CAPA\\U+00E9") + line(0, 3, 1, 4, "capa") +
        line(0, 4, 1, 5, "CAPA") + line(0, 5, 1, 6, "")) + EOF
    # cp1252 bytes / utf-8
    files["q_layers_cp1252.dxf"] = header("AC1009") + ents(line(0, 0, 1, 1, "CAÑERÍA")) + EOF
    files["q_layers_utf8.dxf"] = header("AC1021") + ents(line(0, 0, 1, 1, "CAÑERÍA") + line(0, 1, 1, 2, "日本")) + EOF
    # INSERT with attributes
    att = T((0, "ATTRIB"), (8, "0"), (10, 5), (20, 5), (40, 1), (1, "x"), (2, "TAG"))
    files["q_attribs.dxf"] = header() + blocks(block("B", line(0, 0, 1, 0))) + ents(
        T((0, "INSERT"), (8, "0"), (2, "B"), (66, 1), (10, 5), (20, 5)) + att + T((0, "SEQEND"), (8, "0")) + line(0, 0, 2, 2)) + EOF
    # polyline without SEQEND; polyline then other entity
    pl = T((0, "POLYLINE"), (8, "0"), (66, 1), (70, 0)) + "".join(T((0, "VERTEX"), (8, "0"), (10, x), (20, y)) for x, y in ((0, 0), (1, 0), (1, 1)))
    files["q_poly_noseqend.dxf"] = header() + ents(pl + line(5, 5, 6, 6)) + EOF
    # padded group codes and trailing spaces
    files["q_padding.dxf"] = "  0\nSECTION\n  2\nENTITIES\n  0\nLINE\n  8\nA \n 10\n 1.5 \n 20\n2\n 11\n3\n 21\n4\n  0\nENDSEC\n  0\nEOF\n"
    # no BLOCKS but an INSERT; comments; ENTITIES before BLOCKS
    files["q_noblocks.dxf"] = header() + ents(T((0, "INSERT"), (8, "0"), (2, "NOPE"), (10, 1), (20, 1)) + line(0, 0, 1, 1)) + EOF
    files["q_comments.dxf"] = header() + ents(T((999, "hi"), (0, "LINE"), (999, "mid"), (8, "0"), (10, 0), (20, 0), (11, 1), (21, 1))) + EOF
    files["q_order.dxf"] = header() + ents(T((0, "INSERT"), (8, "0"), (2, "B"), (10, 1), (20, 1))) + blocks(block("B", line(0, 0, 3, 3))) + EOF
    # paper space entity mix
    files["q_paper.dxf"] = header("AC1015") + ents(line(0, 0, 1, 1) + T((0, "LINE"), (8, "0"), (67, 1), (10, 5), (20, 5), (11, 6), (21, 6))) + EOF
    # defaults: circle without radius, arc without angles, ellipse without ratio
    files["q_defaults.dxf"] = header("AC1015") + ents(
        T((0, "CIRCLE"), (8, "0"), (10, 0), (20, 0)) +
        T((0, "ARC"), (8, "0"), (10, 5), (20, 0), (40, 2)) +
        T((0, "ELLIPSE"), (8, "0"), (10, 10), (20, 0), (11, 3), (21, 0)) +
        T((0, "ELLIPSE"), (8, "0"), (10, 20), (20, 0), (11, 0), (21, 0)) +
        T((0, "CIRCLE"), (8, "0"), (10, 30), (20, 0), (40, 2), (210, 0), (220, 0), (230, 0))) + EOF
    # spline variants
    def spline(cps, knots, degree=3, weights=None, flags=8, tol=None):
        t = [(0, "SPLINE"), (100, "AcDbEntity"), (8, "0"), (100, "AcDbSpline"), (210, 0), (220, 0), (230, 1), (70, flags), (71, degree), (72, len(knots)), (73, len(cps))]
        if tol is not None:
            t.append((42, tol))
        for k in knots:
            t.append((40, k))
        if weights:
            for w in weights:
                t.append((41, w))
        for c in cps:
            t += [(10, c[0]), (20, c[1]), (30, c[2])]
        return T(*t)

    cps = [(0, 0, 0), (1, 2, 0), (3, 3, 1), (5, 1, 0), (7, 4, 2)]
    files["q_spline_a.dxf"] = header("AC1015") + ents(
        spline(cps, [0, 0, 0, 0, 0.5, 1, 1, 1, 1]) +
        spline(cps, [10, 10, 10, 10, 15.123456789012, 20, 20, 20, 20]) +
        spline(cps, [0, 0, 0, 0, 0.5000000001, 1, 1, 1, 1], tol=1e-4) +
        spline(cps, [0, 0, 0, 0, 0.5, 0.5, 1, 1, 1][:9])) + EOF
    files["q_spline_bad.dxf"] = header("AC1015") + ents(
        spline(cps[:3], [0, 0, 0, 1, 1, 1], degree=2) +
        spline(cps, [0, 0, 0, 0, 1, 1, 1, 1]) +
        spline(cps[:3], [0, 0, 0, 0, 1, 1, 1, 1], degree=3) +
        spline(cps, [0, 0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1][:9]) +
        spline([], [])) + EOF
    files["q_spline_unported.dxf"] = header("AC1015") + ents(
        spline(cps, [0, 0, 0, 0, 0.5, 1, 1, 1, 1], weights=[1, 1, 2, 1, 1], flags=12) +
        spline(cps, [0, 1, 2, 3, 4, 5, 6, 7, 8])) + EOF
    # 2D line coordinates only; partial coordinates
    files["q_partial.dxf"] = header() + ents(
        T((0, "LINE"), (8, "0"), (10, 1), (20, 2), (11, 3), (21, 4)) +
        T((0, "LINE"), (8, "0"), (10, 1), (20, 2)) +
        T((0, "3DFACE"), (8, "0"), (10, 1), (20, 2), (11, 3), (21, 4), (12, 5), (22, 6))) + EOF
    # polyline 2d with elevation header and varying z, closed, bulges
    pl2 = T((0, "POLYLINE"), (8, "0"), (66, 1), (10, 0), (20, 0), (30, 3.5), (70, 1)) + "".join(
        T((0, "VERTEX"), (8, "0"), (10, x), (20, y), (30, z), (42, b)) for x, y, z, b in ((0, 0, 1, 0.5), (4, 0, 2, 0), (4, 3, 3, 0))) + T((0, "SEQEND"), (8, "0"))
    pl2n = T((0, "POLYLINE"), (8, "0"), (66, 1), (70, 0)) + "".join(
        T((0, "VERTEX"), (8, "0"), (10, x), (20, y), (30, z)) for x, y, z in ((0, 0, 1), (4, 0, 2), (4, 3, 3))) + T((0, "SEQEND"), (8, "0"))
    files["q_poly2d.dxf"] = header() + ents(pl2 + pl2n) + EOF
    # lwpolyline variants
    lw = T((0, "LWPOLYLINE"), (100, "AcDbEntity"), (8, "0"), (100, "AcDbPolyline"), (90, 3), (70, 1), (43, 0.5), (10, 0), (20, 0), (40, 1), (41, 2), (42, 0.3), (10, 4), (20, 0), (10, 4), (20, 3), (42, -1))
    files["q_lw.dxf"] = header("AC1015") + ents(lw) + EOF
    # invisible-edge 3DFACE, MINSERT
    files["q_minsert.dxf"] = header() + blocks(block("B", line(0, 0, 1, 0))) + ents(
        T((0, "INSERT"), (8, "0"), (2, "B"), (10, 0), (20, 0), (70, 3), (71, 2), (44, 5), (45, 4), (50, 20)) +
        T((0, "3DFACE"), (8, "0"), (10, 0), (20, 0), (30, 0), (11, 1), (21, 0), (31, 0), (12, 1), (22, 1), (32, 0), (13, 0), (23, 1), (33, 0), (70, 15))) + EOF
    # a block that references a missing block and itself; two nesting levels with scales
    files["q_nest_missing.dxf"] = header() + blocks(
        block("A", T((0, "INSERT"), (8, "0"), (2, "NOPE"), (10, 1), (20, 1)) + line(0, 0, 1, 1)) +
        block("S", T((0, "INSERT"), (8, "0"), (2, "S"), (10, 1), (20, 0)) + line(0, 0, 1, 0)))+ ents(
        T((0, "INSERT"), (8, "0"), (2, "A"), (10, 5), (20, 5)) + T((0, "INSERT"), (8, "0"), (2, "S"), (10, 0), (20, 0))) + EOF
    # header oddities
    files["q_units_pad.dxf"] = T((0, "SECTION"), (2, "HEADER"), (9, "$ACADVER"), (1, "AC1015"), (9, "$INSUNITS"), (70, "     4"), (0, "ENDSEC")) + ents(line(0, 0, 3000, 0)) + EOF
    files["q_nohdr_insunits.dxf"] = ents(line(0, 0, 3000, 0)) + EOF
    files["q_empty_header.dxf"] = T((0, "SECTION"), (2, "HEADER"), (0, "ENDSEC")) + ents(line(0, 0, 3000, 0)) + EOF
    # no EOF / no ENDSEC
    files["q_noeof.dxf"] = header() + T((0, "SECTION"), (2, "ENTITIES")) + line(0, 0, 1, 1)
    # CR-only line ends
    files["q_cr.dxf"] = (header() + ents(line(0, 0, 1, 1)) + EOF).replace("\n", "\r")
    files["q_crlf.dxf"] = (header() + ents(line(0, 0, 1, 1)) + EOF).replace("\n", "\r\n")
    # duplicate tags
    files["q_dups.dxf"] = header() + ents(T((0, "LINE"), (8, "0"), (10, 0), (20, 0), (10, 5), (20, 5), (11, 1), (21, 1))) + EOF
    # huge numbers, exponent formats, nan
    files["q_numbers.dxf"] = header() + ents(
        T((0, "LINE"), (8, "0"), (10, "1e3"), (20, "-2.5E-2"), (11, "+7"), (21, ".5")) +
        T((0, 'LINE'), (8, '0'), (10, 5), (20, 0), (11, 7), (21, 1))) + EOF

    hdr9 = T((0, "SECTION"), (2, "HEADER"), (9, "$ACADVER"), (1, "AC1009"), (0, "ENDSEC"))
    line9 = T((0, "LINE"), (8, "0"), (10, 0), (20, 0), (11, 1), (21, 1))
    vtx9 = lambda x, y: T((0, "VERTEX"), (8, "0"), (10, x), (20, y))
    poly9 = T((0, "POLYLINE"), (8, "0"), (66, 1), (70, 0))
    att9 = T((0, "ATTRIB"), (8, "0"), (10, 5), (20, 5), (40, 1), (1, "x"), (2, "TAG"))
    ins1 = T((0, "INSERT"), (8, "0"), (2, "B"), (66, 1), (10, 5), (20, 5))
    ins0 = T((0, "INSERT"), (8, "0"), (2, "B"), (10, 5), (20, 5))
    seq9 = T((0, "SEQEND"))
    S9 = lambda body: hdr9 + T((0, "SECTION"), (2, "ENTITIES")) + body + T((0, "ENDSEC"), (0, "EOF"))
    sec = lambda body: hdr9 + T((0, "SECTION"), (2, "ENTITIES")) + body
    structure = {
        "s_a_noendsec_eof": sec(line9 + T((0, "EOF"))),
        "s_b_noendsec_noeof": sec(line9),
        "s_c_endsec_noeof": sec(line9 + T((0, "ENDSEC"))),
        "s_d_poly_noseqend_endsec": sec(poly9 + vtx9(0, 0) + vtx9(1, 1) + T((0, "ENDSEC"), (0, "EOF"))),
        "s_e_poly_noseqend_line": S9(poly9 + vtx9(0, 0) + vtx9(1, 1) + line9),
        "s_f_poly_novertices_seqend": S9(poly9 + seq9 + line9),
        "s_g_stray_vertex": S9(vtx9(0, 0) + line9),
        "s_h_stray_seqend": S9(seq9 + line9),
        "s_i_no_sections": line9 + T((0, "EOF")),
        "s_j_two_entities_sections": hdr9 + T((0, "SECTION"), (2, "ENTITIES")) + line9 + T((0, "ENDSEC")) + T((0, "SECTION"), (2, "ENTITIES")) + line9 + line9 + T((0, "ENDSEC"), (0, "EOF")),
        "s_k_block_noendblk": hdr9 + T((0, "SECTION"), (2, "BLOCKS"), (0, "BLOCK"), (8, "0"), (2, "B"), (70, 0), (10, 0), (20, 0), (30, 0)) + line9 + T((0, "ENDSEC")) + T((0, "SECTION"), (2, "ENTITIES")) + ins0 + T((0, "ENDSEC"), (0, "EOF")),
        "s_l_header_noendsec": T((0, "SECTION"), (2, "HEADER"), (9, "$INSUNITS"), (70, 4)) + T((0, "SECTION"), (2, "ENTITIES")) + line9 + T((0, "ENDSEC"), (0, "EOF")),
        "s_m_text_before_section": "junk\n0\n" + hdr9 + T((0, "SECTION"), (2, "ENTITIES")) + line9 + T((0, "ENDSEC"), (0, "EOF")),
        "s_n_unknown_section": hdr9 + T((0, "SECTION"), (2, "FOO"), (0, "LINE")) + T((0, "ENDSEC")) + T((0, "SECTION"), (2, "ENTITIES")) + line9 + T((0, "ENDSEC"), (0, "EOF")),
        "s_o_ins66_line": S9(ins1 + line9),
        "s_p_ins66_att_noseq_line": S9(ins1 + att9 + line9),
        "s_q_ins66_seq": S9(ins1 + seq9 + line9),
        "s_r_ins0_att_seq": S9(ins0 + att9 + seq9 + line9),
        "s_s_ins0_seq": S9(ins0 + seq9 + line9),
        "s_t_poly_poly": S9(poly9 + vtx9(0, 0) + poly9 + vtx9(1, 1) + seq9),
        "s_u_poly_seq_noverts_then_vertex": S9(poly9 + seq9 + vtx9(0, 0)),
        "s_v_ins66_att_att_seq": S9(ins1 + att9 + att9 + seq9),
        "s_w_poly_vertex_attrib": S9(poly9 + vtx9(0, 0) + att9 + seq9),
        "s_x_ins66_end": S9(ins1),
        "s_y_poly_end": S9(poly9 + vtx9(0, 0)),
        "s_z_poly_attrib_noverts": S9(poly9 + att9 + seq9),
    }
    for k, v in structure.items():
        files[k + ".dxf"] = v


    # strict reading fails on a number, ezdxf's recover reader takes over
    files["r_float.dxf"] = header() + ents(
        T((0, "LINE"), (8, "0"), (10, "1d3"), (20, "47.11xyz"), (11, "1.#INF"), (21, " 2 5 ")) +
        T((0, "LINE"), (8, "0"), (10, "0"), (20, "0"), (11, "3"), (21, "4"))) + EOF
    files["r_int.dxf"] = header("AC1015") + ents(
        T((0, "LINE"), (8, "0"), (10, "0"), (20, "0"), (11, "3"), (21, "4"), (62, "5.0")) +
        T((0, "LWPOLYLINE"), (100, "AcDbEntity"), (8, "0"), (100, "AcDbPolyline"), (90, "3x"), (70, "1"), (10, 0), (20, 0), (10, 4), (20, 0), (10, 4), (20, 3))) + EOF
    files["r_upper.dxf"] = header() + ents(
        T((0, " line "), (8, "A\\U+00E9"), (10, 0), (20, 0), (11, 3), (21, 4), (62, "x1")) +
        T((0, "Circle"), (8, "0"), (10, 0), (20, 0), (40, 2))) + EOF
    files["r_unicode_strict.dxf"] = header() + ents(
        T((0, "LINE"), (8, "A\\U+00E9"), (10, 0), (20, 0), (11, 3), (21, 4))) + EOF
    files["r_unicode_recover.dxf"] = header() + ents(
        T((0, "LINE"), (8, "A\\U+00E9"), (10, 0), (20, 0), (11, 3), (21, "4z"))) + EOF
    files["r_nonsense.dxf"] = header() + ents(
        T((0, "LINE"), (8, "0"), (10, "abc"), (20, 0), (11, 3), (21, 4))) + EOF
    files["r_blocks.dxf"] = header() + blocks(block("b", T((0, "LINE"), (8, "0"), (10, 0), (20, 0), (11, "1x"), (21, 1)))) + ents(
        T((0, "INSERT"), (8, "0"), (2, "B"), (10, 5), (20, 5), (41, "2.5x"))) + EOF
    return files


#: Quirk files the port does not follow yet.
PENDING = set()

HAND = {
    # R12 without a HEADER: ezdxf assumes AC1009 and reads $INSUNITS as unset.
    "hand_r12.dxf": """0
SECTION
2
ENTITIES
0
LINE
8
MUROS
10
0.0
20
0.0
30
0.0
11
4.0
21
3.0
31
0.0
0
SOLID
8
0
10
10
20
0
38
2.5
11
12
21
0
12
10
22
2
13
12
23
2
0
POLYLINE
8
CAPA
66
1
70
1
0
VERTEX
8
CAPA
10
20
20
0
0
VERTEX
8
CAPA
10
24
20
0
42
0.5
0
VERTEX
8
CAPA
10
24
20
4
0
SEQEND
8
CAPA
0
ENDSEC
0
EOF
""",
    # Comments, odd spacing and a block whose name differs in case.
    "hand_odd.dxf": """999
made by hand
0
SECTION
2
HEADER
9
$ACADVER
1
AC1015
9
$INSUNITS
70
     4
0
ENDSEC
0
SECTION
2
BLOCKS
0
BLOCK
8
0
2
Caja
70
0
10
1.0
20
1.0
30
0.0
3
Caja
1

0
LINE
8
0
10
1.0
20
1.0
30
0.0
11
2.0
21
1.0
31
0.0
0
ENDBLK
8
0
0
ENDSEC
0
SECTION
2
ENTITIES
0
INSERT
8
0
2
CAJA
10
3.0
20
3.0
30
0.0
0
3DFACE
8
0
10
0
20
0
30
0
11
1
21
0
31
0
12
1
22
1
32
0
0
LINE
8
0
10
0
20
0
11
1000
21
1000
0
ENDSEC
0
EOF
""",
}


def write_imports() -> None:
    imp = OUT / "import"
    if imp.exists():
        shutil.rmtree(imp)
    imp.mkdir(parents=True)
    made = build_docs(imp)
    for name, text in HAND.items():
        (imp / name).write_bytes(text.replace("\n", "\r\n").encode("cp1252"))
        made.append((name, 1.0, {}))
    (imp / "notdxf.dxf").write_text("this is not a DXF file\n", encoding="ascii")
    made.append(("notdxf.dxf", 1.0, {}))
    for name, text in quirk_files().items():
        data = text.encode("cp1252" if name == "q_layers_cp1252.dxf" else "utf-8")
        (imp / name).write_bytes(data)
        # Files that stand for what Python cannot import yet say so.
        extra = {"pending": "not ported"} if name in PENDING else {}
        made.append((name, 1.0, extra))
    made += nested_docs(imp)
    made += random_docs(imp, range(1, 61))

    # Every file loads at its own scale, some also at others.
    plan = []
    seen = set()
    for name, scale, extra in made:
        if name in seen and not extra:
            continue
        seen.add(name)
        plan.append({"file": name, "scale": scale, **extra})
    for name, scale in (("layers.dxf", 1.0), ("circles.dxf", 0.001),
                        ("arcs.dxf", 0.0254),
                        ("blocks.dxf", 0.01), ("nested.dxf", 0.3048),
                        ("bulge.dxf", 10.0), ("utm.dxf", 0.001),
                        ("polylines.dxf", 0.5), ("bigfaces.dxf", 0.5)):
        plan.append({"file": name, "scale": scale})
    saved_stitch = None
    import core.history
    saved_stitch = core.history.run_stitch
    core.history.run_stitch = lambda *a, **k: None
    cases = []
    try:
        for case in plan:
            path = imp / case["file"]
            row = dict(case)
            try:
                doc = dxf_in.open_document(path)
                row["suggest"] = list(dxf_in.suggest_unit_scale(doc))
                row["header"] = list(dxf_in.read_unit_scale(doc))
            except Exception as exc:  # noqa: BLE001
                row["open_error"] = f"{type(exc).__name__}: {exc}"
                cases.append(row)
                continue
            if row.pop("nosig", False):
                cases.append(row)
                continue
            scene = Scene()
            try:
                stats = dxf_in.load_dxf(scene, path, scale=row["scale"],
                                        doc=doc, name=row.get("name"))
            except Exception as exc:  # noqa: BLE001 — the message is the datum
                row["error"] = f"{type(exc).__name__}: {exc}"
                cases.append(row)
                continue
            stats = dict(stats)
            stats["offset"] = (list(stats["offset"]) if stats["offset"]
                               else None)
            row["stats"] = stats
            tag = f"{row['file']}.{row['scale']}"
            if "name" in row:
                tag += f".{row['name']}"
            row["sig"] = f"{tag}.sig.json"
            write_json(imp / row["sig"], scene_signature(scene))
            cases.append(row)
    finally:
        core.history.run_stitch = saved_stitch
    names = [c["sig"] for c in cases if "sig" in c]
    assert len(names) == len(set(names)), "signature names clash"
    write_json(imp / "cases.json", cases)
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(len(cases), "import cases,", total // 1024, "KiB of fixtures")


def main() -> None:
    app = QGuiApplication.instance() or QGuiApplication(sys.argv)  # noqa: F841
    OUT.mkdir(parents=True, exist_ok=True)
    write_export()
    write_imports()


if __name__ == "__main__":
    main()
