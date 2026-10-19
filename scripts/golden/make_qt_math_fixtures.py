# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixtures that hold ``ingetrazo-math`` to Qt's QMatrix4x4 and QVector3D,
bit for bit and flag for flag (docs/rust-rewrite-plan.md, M1.1).

Random chains of the matrix operations the app uses (translate, scale,
rotate — the special angles and axis-aligned axes included —, products,
inverses, transposes, lookAt, ortho, perspective, optimize), each written
with its float32 arguments as bits, and what Qt 6 made of them: the 16
floats, the type flags, points and vectors mapped through it, the inverse
and the determinant. Plus vector arithmetic. Committed, so cargo test needs
no Python; rerun only to widen them:

    python scripts/golden/make_qt_math_fixtures.py

Deleted with the Python app at M10.
"""
from __future__ import annotations

import json
import math
import random
import struct
import sys
from pathlib import Path

from PySide6 import __version__ as pyside_version
from PySide6.QtCore import qVersion
from PySide6.QtGui import QMatrix4x4, QVector3D

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rust" / "crates" / "ingetrazo-math" / "tests" / "data"


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def h32(x: float) -> str:
    return struct.pack(">f", x).hex()


def h64(x: float) -> str:
    return struct.pack(">d", x).hex()


def vbits(v: QVector3D) -> list:
    return [h32(v.x()), h32(v.y()), h32(v.z())]


def mbits(m: QMatrix4x4) -> list:
    return [h32(x) for x in m.data()]          # column major


class Gen:
    def __init__(self, seed: int) -> None:
        self.r = random.Random(seed)

    def num(self, lo=-50.0, hi=50.0) -> float:
        r = self.r.random()
        if r < 0.15:
            return f32(self.r.choice([0.0, 1.0, -1.0, 2.0, 0.5, -0.0]))
        if r < 0.3:
            return f32(round(self.r.uniform(lo, hi), self.r.randint(0, 3)))
        return f32(self.r.uniform(lo, hi))

    def angle(self) -> float:
        r = self.r.random()
        if r < 0.35:
            return f32(self.r.choice([0.0, 90.0, -90.0, 180.0, -180.0, 270.0, -270.0, 45.0, 30.0]))
        return f32(self.r.uniform(-360.0, 360.0))

    def axis(self):
        r = self.r.random()
        if r < 0.45:
            return [f32(x) for x in self.r.choice([
                (0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0),
                (0, 0, 2.5), (3, 0, 0)])]
        if r < 0.6:
            a = [self.num(-1, 1) for _ in range(3)]
            n = math.sqrt(sum(x * x for x in a)) or 1.0
            return [f32(x / n) for x in a]
        return [self.num(-5, 5) for _ in range(3)]

    def vec(self, lo=-100.0, hi=100.0) -> QVector3D:
        return QVector3D(self.num(lo, hi), self.num(lo, hi), self.num(lo, hi))

    def op(self, depth: int):
        r = self.r.random()
        if r < 0.22:
            return ["translate", *[self.num() for _ in range(3)]]
        if r < 0.32:
            return ["translate2", self.num(), self.num()]
        if r < 0.42:
            return ["scale", *[self.num(-4, 4) for _ in range(3)]]
        if r < 0.47:
            return ["scale1", self.num(-4, 4)]
        if r < 0.67:
            return ["rotate", self.angle(), *self.axis()]
        if r < 0.75 and depth < 2:
            return ["mul", self.chain(depth + 1)]
        if r < 0.80 and depth < 2:
            return ["mul_assign", self.chain(depth + 1)]
        if r < 0.84:
            return ["inverted"]
        if r < 0.86:
            return ["transposed"]
        if r < 0.89:
            return ["optimize"]
        if r < 0.92:
            eye, center, up = self.vec(-20, 20), self.vec(-20, 20), QVector3D(*self.axis())
            return ["look_at", *[float(c) for v in (eye, center, up) for c in (v.x(), v.y(), v.z())]]
        if r < 0.95:
            vals = [self.num(-10, 10) for _ in range(6)]
            return ["ortho", *vals]
        if r < 0.97:
            return ["perspective", f32(self.r.uniform(10, 120)), f32(self.r.uniform(0.3, 3)),
                    f32(self.r.uniform(0.01, 1)), f32(self.r.uniform(10, 1000))]
        return ["general16", *[self.num(-3, 3) for _ in range(16)]]

    def chain(self, depth: int = 0) -> dict:
        start = "identity" if self.r.random() < 0.85 else "general16"
        c = {"start": start, "ops": [self.op(depth) for _ in range(self.r.randint(1, 6))]}
        if start == "general16":
            c["values"] = [self.num(-3, 3) for _ in range(16)]
        return c


def build(c: dict) -> QMatrix4x4:
    m = QMatrix4x4(*c["values"]) if c["start"] == "general16" else QMatrix4x4()
    for op in c["ops"]:
        name, args = op[0], op[1:]
        if name == "translate":
            m.translate(*args)
        elif name == "translate2":
            m.translate(*args)
        elif name == "scale":
            m.scale(*args)
        elif name == "scale1":
            m.scale(args[0])
        elif name == "rotate":
            m.rotate(*args)
        elif name == "mul":
            m = m * build(args[0])
        elif name == "mul_assign":
            m *= build(args[0])
        elif name == "inverted":
            m = m.inverted()[0]
        elif name == "transposed":
            m = m.transposed()
        elif name == "optimize":
            m.optimize()
        elif name == "look_at":
            m.lookAt(QVector3D(*args[0:3]), QVector3D(*args[3:6]), QVector3D(*args[6:9]))
        elif name == "ortho":
            m.ortho(*args)
        elif name == "perspective":
            m.perspective(*args)
        elif name == "general16":
            m = QMatrix4x4(*args)
        else:
            raise ValueError(name)
    return m


def as_bits(c: dict) -> dict:
    """The chain with every float argument as float32 bits (exact)."""
    out = {"start": c["start"], "ops": []}
    if "values" in c:
        out["values"] = [h32(x) for x in c["values"]]
    for op in c["ops"]:
        if op[0] in ("mul", "mul_assign"):
            out["ops"].append([op[0], as_bits(op[1])])
        else:
            out["ops"].append([op[0], *[h32(x) for x in op[1:]]])
    return out


def matrices(n: int, seed: int) -> list:
    g = Gen(seed)
    rows = []
    for _ in range(n):
        c = g.chain()
        m = build(c)
        pts = [g.vec(), g.vec(-2, 2), QVector3D(0, 0, 0)]
        inv, ok = m.inverted()
        rows.append({
            "chain": as_bits(c),
            "m": mbits(m), "flags": m.flags().value,
            "identity": m.isIdentity(),
            "map": [[vbits(p), vbits(m.map(p))] for p in pts],
            "map_vector": [[vbits(p), vbits(m.mapVector(p))] for p in pts[:2]],
            "inverted": {"m": mbits(inv), "flags": inv.flags().value, "ok": ok},
            "det": h64(m.determinant()),
        })
    return rows


def vectors(n: int, seed: int) -> list:
    g = Gen(seed)
    rows = []
    for _ in range(n):
        a, b, c = g.vec(), g.vec(), g.vec()
        if g.r.random() < 0.1:
            a = QVector3D(f32(1.0 + g.r.uniform(-2e-6, 2e-6)), 0.0, 0.0)   # normalized's fuzz
        s = g.num(-10, 10)
        rows.append({
            "a": vbits(a), "b": vbits(b), "c": vbits(c), "s": h32(s),
            "add": vbits(a + b), "sub": vbits(a - b), "neg": vbits(-a),
            "mul_s": vbits(a * s), "s_mul": vbits(s * a), "mul_v": vbits(a * b),
            "div_s": vbits(a / s) if s != 0 else None,
            "dot": h32(QVector3D.dotProduct(a, b)),
            "cross": vbits(QVector3D.crossProduct(a, b)),
            "len": h32(a.length()), "len2": h32(a.lengthSquared()),
            "normalized": vbits(a.normalized()),
            "normal2": vbits(QVector3D.normal(a, b)),
            "normal3": vbits(QVector3D.normal(a, b, c)),
            "dist_point": h32(a.distanceToPoint(b)),
            "dist_plane": h32(a.distanceToPlane(b, c)),
            "dist_plane3": h32(a.distanceToPlane(a + b, b, c)),
            "dist_line": h32(a.distanceToLine(b, c)),
        })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"qt": qVersion(), "pyside": pyside_version, "platform": sys.platform}
    for name, rows in (("qmatrix4x4.jsonl", matrices(3000, 20261019)),
                       ("qvector3d.jsonl", vectors(2000, 1019))):
        with open(OUT / name, "w", encoding="ascii", newline="\n") as fh:
            fh.write(json.dumps({"meta": meta}) + "\n")
            for r in rows:
                fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    print("written to", OUT.relative_to(ROOT), meta)


if __name__ == "__main__":
    main()
