# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixtures that hold ``ingetrazo-geom``'s section cut to ``core/section.py``,
bit for bit (docs/rust-rewrite-plan.md, M3.9): ``cut_chords``,
``fill_triangles`` and ``hatch_segments``.

Triangle soups of boxes, hollow boxes, rings of walls, crossing walls, prisms
of several sides (rings of 7, 16 and 40 points exercise NumPy's strided
``ddot``), nested hollows and an open surface are cut by horizontal, vertical
and tilted planes; each case records the inputs and what the Python made of
them, every number as the big-endian hex of its bits (f32: 8 digits, f64: 16).

Run it with the MAIN venv from this tree's root:

    QT_QPA_PLATFORM=offscreen <main>/.venv/Scripts/python.exe -X utf8 \\
        scripts/golden/make_section_fixtures.py

Needs NumPy's OpenBLAS to be the Haswell-class kernel the Rust side mirrors
(see the module docs of ``section_cut.rs``). Deleted with the Python app (M10).
"""
from __future__ import annotations

import json
import math
import os
import random
import struct
import zlib
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.environ.get("INGETRAZO_RUST"):
    sys.exit("unset INGETRAZO_RUST: the fixtures come from the pure-Python path")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np                       # noqa: E402
from PySide6.QtWidgets import QApplication   # noqa: E402

_app = QApplication.instance() or QApplication([])

from core.section import cut_chords, fill_triangles, hatch_segments   # noqa: E402
from core.vecmath import QVector3D as V   # noqa: E402

OUT = ROOT / "rust" / "crates" / "ingetrazo-geom" / "tests" / "data"


def h32(x: float) -> str:
    return struct.pack(">f", float(x)).hex()


def hx32(arr) -> str:
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32)).ravel()
    return "".join(h32(x) for x in a.tolist())


def h64(x: float) -> str:
    return struct.pack(">d", float(x)).hex()


# ── Triangle soups ──────────────────────────────────────────────────────────

def tri_list(quads):
    """Triangles (3 corner points each) of quads given as 4 corner tuples."""
    out = []
    for a, b, c, d in quads:
        out.append((a, b, c))
        out.append((a, c, d))
    return out


def box_tris(x0, y0, z0, x1, y1, z1):
    p = lambda x, y, z: (x, y, z)           # noqa: E731
    return tri_list([
        (p(x0, y0, z0), p(x1, y0, z0), p(x1, y1, z0), p(x0, y1, z0)),
        (p(x0, y0, z1), p(x1, y0, z1), p(x1, y1, z1), p(x0, y1, z1)),
        (p(x0, y0, z0), p(x1, y0, z0), p(x1, y0, z1), p(x0, y0, z1)),
        (p(x0, y1, z0), p(x1, y1, z0), p(x1, y1, z1), p(x0, y1, z1)),
        (p(x0, y0, z0), p(x0, y1, z0), p(x0, y1, z1), p(x0, y0, z1)),
        (p(x1, y0, z0), p(x1, y1, z0), p(x1, y1, z1), p(x1, y0, z1)),
    ])


def prism_tris(cx, cy, z0, z1, r, n, phase=0.0):
    ring = [(cx + r * math.cos(phase + 2 * math.pi * i / n),
             cy + r * math.sin(phase + 2 * math.pi * i / n)) for i in range(n)]
    tris = []
    for i in range(1, n - 1):
        tris.append(((ring[0][0], ring[0][1], z0), (ring[i][0], ring[i][1], z0), (ring[i + 1][0], ring[i + 1][1], z0)))
        tris.append(((ring[0][0], ring[0][1], z1), (ring[i][0], ring[i][1], z1), (ring[i + 1][0], ring[i + 1][1], z1)))
    quads = []
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        quads.append(((a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1)))
    return tris + tri_list(quads)


def arrays(tris):
    """(v0, e1, e2) float32 arrays as the pick index keeps them."""
    t = np.asarray(tris, dtype=np.float64).reshape(-1, 3, 3).astype(np.float32)
    v0 = np.ascontiguousarray(t[:, 0])
    e1 = np.ascontiguousarray(t[:, 1] - t[:, 0])
    e2 = np.ascontiguousarray(t[:, 2] - t[:, 0])
    return v0, e1, e2


SOUPS = {}


def soup(name, tris):
    SOUPS[name] = tris


soup("box", box_tris(-1, -1.5, 0, 1.2, 1, 2.1))
soup("hollow", box_tris(0, 0, 0, 6, 8, 4) + box_tris(1, 1, 0.5, 5, 7, 3.5))
soup("room", (box_tris(0, 0, 0, 5, 0.3, 3) + box_tris(0, 4.7, 0, 5, 5, 3)
              + box_tris(0, 0.3, 0, 0.3, 4.7, 3) + box_tris(4.7, 0.3, 0, 5, 4.7, 3)))
soup("crossing", box_tris(0, 0, 0, 6, 0.4, 3) + box_tris(2, -2, 0, 2.4, 3, 3))
soup("lwall", box_tris(0, 0, 0, 4, 1.5, 3) + box_tris(0, 1.5, 0, 1.5, 4, 3))
soup("prism7", prism_tris(0, 0, 0, 3, 1.3, 7, 0.21))
soup("prism16", prism_tris(0, 0, 0, 3, 1.3, 16))
soup("prism40", prism_tris(0.3, -0.2, 0, 3, 2.1, 40, 0.07))
soup("pipe", prism_tris(0, 0, 0, 3, 2.0, 24) + prism_tris(0, 0, 0, 3, 1.4, 24, 0.05))
soup("nested", (box_tris(0, 0, 0, 10, 10, 4) + box_tris(1, 1, 0, 9, 9, 4)
                + box_tris(2, 2, 0, 8, 8, 4) + box_tris(3, 3, 0, 7, 7, 4)))
soup("pipe_in_hollow", (box_tris(0, 0, 0, 8, 8, 3) + box_tris(0.5, 0.5, 0.2, 7.5, 7.5, 2.8)
                        + prism_tris(4, 4, 0, 3, 1.0, 16)))
soup("open", tri_list([((0, 0, 0), (3, 0, 0), (3, 0, 3), (0, 0, 3))]))
soup("two_apart", box_tris(0, 0, 0, 1, 1, 2) + box_tris(3, 3, 0, 4.5, 4.2, 2))
r = random.Random(11)
city = []
for i in range(4):
    for j in range(3):
        x, y = i * 2.6, j * 2.9
        city += box_tris(x, y, 0, x + 2.0 + r.random() * 0.4, y + 2.0, 1.0 + r.random() * 5.0)
city += prism_tris(5.0, 8.6, 0, 6.5, 0.9, 12)
soup("city", city)
soup("random", [tuple(tuple(r.uniform(-2, 2) for _ in range(3)) for _ in range(3)) for _ in range(40)])

PLANES = {
    "z_mid": ((0.0, 0.0, 1.0), (0.0, 0.0, 1.2)),
    "z_low": ((0.0, 0.0, 1.0), (0.1, 0.2, 0.37)),
    "x_cut": ((1.0, 0.0, 0.0), (0.6, 0.0, 0.0)),
    "y_cut": ((0.0, -1.0, 0.0), (0.0, 0.9, 0.0)),
    "diag": ((0.7, 0.7, 0.0), (1.0, 1.0, 1.0)),
    "tilt": ((0.31, -0.42, 0.85), (0.3, 0.2, 1.1)),
    "tilt2": ((-0.9, 0.2, 0.3), (2.0, 1.0, 1.0)),
    "flat_down": ((0.0, 0.0, -1.0), (0.2, 0.1, 1.7)),
}

cases = []


def vec(t):
    return V(*t)


def one_cut(sname, pname, with_visible=False, rows=None):
    tris = SOUPS[sname]
    v0, e1, e2 = arrays(tris)
    if rows is not None:
        v0, e1, e2 = v0[:rows], e1[:rows], e2[:rows]
    normal, point = PLANES[pname]
    n, o = vec(normal), vec(point)
    vis = None
    if with_visible:
        rr = random.Random(zlib.crc32((sname + pname).encode()))
        vis = np.array([rr.random() < 0.7 for _ in range(len(v0))])
    chords = cut_chords(v0, e1, e2, n, o, vis)
    cases.append({
        "kind": "cut", "name": f"{sname}/{pname}/{rows}/{int(with_visible)}", "n": len(v0),
        "v0": hx32(v0), "e1": hx32(e1), "e2": hx32(e2),
        "normal": [h32(n.x()), h32(n.y()), h32(n.z())],
        "point": [h32(o.x()), h32(o.y()), h32(o.z())],
        "visible": None if vis is None else "".join("1" if b else "0" for b in vis.tolist()),
        "out": None if chords is None else hx32(chords),
    })
    return chords, n


def fill_case(sname, pname, chords, n):
    if chords is None:
        return
    tris = fill_triangles(chords, n)
    hatches = []
    small = sname in ("box", "prism7", "open", "two_apart")
    for spacing in ((0.05, 0.37) if small else (0.37,) if pname.startswith("rand") else (0.1, 0.37)):
        for cross in (False, True):
            h = hatch_segments(chords, n, spacing, cross)
            hatches.append({"spacing": h64(spacing), "cross": cross, "out": hx32(h), "n": len(h)})
    cases.append({
        "kind": "fill", "name": f"{sname}/{pname}", "chords": hx32(chords), "n": len(chords),
        "normal": [h32(n.x()), h32(n.y()), h32(n.z())],
        "tris": hx32(tris), "ntris": len(tris), "hatches": hatches,
    })


def main():
    for sname in SOUPS:
        for pname in PLANES:
            chords, n = one_cut(sname, pname)
            fill_case(sname, pname, chords, n)
        one_cut(sname, "z_mid", with_visible=True)
    for rows in (1, 2, 3, 5, 6, 7):
        for pname in ("z_mid", "tilt"):
            one_cut("random", pname, rows=rows)
    # No triangles at all: a plane that misses everything.
    v0, e1, e2 = arrays(SOUPS["box"])
    n, o = V(0, 0, 1), V(0, 0, 50.0)
    out = cut_chords(v0, e1, e2, n, o, None)
    cases.append({"kind": "cut", "name": "miss", "n": len(v0), "v0": hx32(v0), "e1": hx32(e1), "e2": hx32(e2),
                  "normal": [h32(0), h32(0), h32(1)], "point": [h32(0), h32(0), h32(50)],
                  "visible": None, "out": None if out is None else hx32(out)})
    # Random planes through random city slabs.
    rr = random.Random(2027)
    for k in range(40):
        sname = rr.choice(["city", "pipe", "pipe_in_hollow", "nested", "crossing", "prism40"])
        nrm = (rr.uniform(-1, 1), rr.uniform(-1, 1), rr.uniform(-1, 1))
        if sum(c * c for c in nrm) < 0.05:
            continue
        PLANES[f"rand{k}"] = (nrm, (rr.uniform(0, 4), rr.uniform(0, 4), rr.uniform(0.3, 3.5)))
        chords, n = one_cut(sname, f"rand{k}")
        fill_case(sname, f"rand{k}", chords, n)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "section_cut.jsonl"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for c in cases:
            f.write(json.dumps(c, separators=(",", ":")) + "\n")
    nf = sum(1 for c in cases if c["kind"] == "fill")
    print(f"{len(cases)} cases ({nf} fills), {path.stat().st_size / 1e6:.2f} MB -> {path}")


if __name__ == "__main__":
    main()
