# SPDX-License-Identifier: GPL-3.0-or-later
"""Write the fixtures that hold the Rust `ingetrazo-formats` crate to
Python's json and .igz bytes (docs/rust-rewrite-plan.md, M0.7).

The files are committed, so `cargo test` needs no Python; rerun this only
to widen them:

    python scripts/golden/make_pyjson_fixtures.py

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
OUT = ROOT / "rust" / "crates" / "ingetrazo-formats" / "tests" / "data"


def floats() -> list[float]:
    rng = random.Random(20261019)
    xs: list[float] = []
    # Any bit pattern: every exponent, subnormals included.
    while len(xs) < 2000:
        x = struct.unpack("<d", rng.getrandbits(64).to_bytes(8, "little"))[0]
        if x == x and abs(x) != float("inf"):
            xs.append(x)
    # What a document holds: float32 coordinates widened to double, and
    # decimals typed by a user.
    for _ in range(2000):
        x = rng.uniform(-1e4, 1e4) * 10 ** rng.randint(-6, 3)
        xs.append(struct.unpack("<f", struct.pack("<f", x))[0])
    for _ in range(1000):
        xs.append(round(rng.uniform(-1e5, 1e5), rng.randint(0, 6)))
    # Where repr switches notation, and the extremes.
    for e in range(-8, 20):
        for m in (1.0, 9.999999999999998, 1.5, 9.5, 1.0000000000000002):
            xs += [m * 10.0 ** e, -m * 10.0 ** e]
    xs += [0.0, -0.0, 5e-324, 2.2250738585072014e-308, 1.7976931348623157e308,
           0.1, 0.2, 0.30000000000000004, 1 / 3, 2 / 3, 100.0, 1e-4, 1e-5,
           1e15, 1e16, 9007199254740993.0, 123456789012345678.0]
    return xs


def cases() -> dict:
    return {
        "empty": {"list": [], "dict": {}, "str": ""},
        "nested": [[[]], [{}], {"a": [{"b": {"c": []}}]}],
        "ints": [0, -1, 2**53 + 1, -(2**70), 10**30],
        "floats": [0.0, -0.0, 1.5, -2.25e-7, 6.02214076e23, 1e16, 1e-5],
        "literals": [True, False, None],
        "strings": ["plain", "quote \" backslash \\ slash /",
                    "\b\f\n\r\t\x00\x1f\x7f", "ñandú café — 30°",
                    "日本語", "emoji 😀🏗️", "  ", "﻿"],
        "keys": {"z": 1, "a": 2, "é": 3, "": 4, "A": 5, "10": 6, "9": 7},
    }


def write_scene_igz() -> None:
    """A real scene saved by the app, in format 1, indented and compact."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, str(ROOT))
    from PySide6.QtGui import QVector3D as V
    from PySide6.QtWidgets import QApplication

    from core.scene import Scene
    from formats import igz

    app = QApplication.instance() or QApplication([])  # noqa: F841
    scene = Scene()
    m = scene.mesh
    m.add_face([V(0, 0, 0), V(3, 0, 0), V(3, 2.5, 0), V(0, 2.5, 0)],
               [[V(1, 1, 0), V(1.4, 1, 0), V(1.4, 1.3, 0), V(1, 1.3, 0)]])
    m.add_face([V(0, 0, 0), V(3, 0, 0), V(3, 0, 1.2), V(0, 0, 1.2)])
    m.add_face([V(0.1, 0.2, 0.3), V(1 / 3, 2 / 3, 0.7), V(-0.001, 1e-6, 5.5)])
    m.add_edge(V(5, 5, 0), V(7.25, 5.125, 0.0625))
    igz.save_scene(scene, OUT / "scene.igz")
    saved = igz._LARGE_DOCUMENT
    try:
        igz._LARGE_DOCUMENT = 0          # compact, as a large model is
        igz.save_scene(scene, OUT / "scene-compact.igz")
    finally:
        igz._LARGE_DOCUMENT = saved
    for name in ("scene.igz", "scene-compact.igz"):
        assert not (OUT / name).read_bytes().startswith(b"PK"), name


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "floats.txt", "w", encoding="ascii", newline="\n") as f:
        for x in floats():
            f.write(f"{struct.pack('>d', x).hex()} {x!r}\n")
    c = cases()
    variants = {
        "cases.default.json": json.dumps(c),
        "cases.indent.json": json.dumps(c, indent=2),
        "cases.compact.json": json.dumps(c, separators=(",", ":")),
        "cases.sorted.json": json.dumps(c, indent=2, sort_keys=True),
        "cases.utf8.json": json.dumps(c, ensure_ascii=False),
    }
    for name, text in variants.items():
        (OUT / name).write_bytes(text.encode("utf-8"))
    write_scene_igz()
    print("fixtures written to", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
