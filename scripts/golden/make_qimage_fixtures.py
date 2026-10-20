# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixtures that hold the Rust image helpers of ``ingetrazo-formats-mesh``
(``img.rs``) to Qt (docs/rust-rewrite-plan.md, M5.4): PNG files of the kinds
a COLLADA or OBJ import meets, each with what PySide6 answers for them,

* ``smooth``: ``QImage(path).scaled(1, 1, IgnoreAspectRatio,
  SmoothTransformation).pixelColor(0, 0)`` as 8-bit r, g, b (what
  ``formats.dae._image_color`` averages a texture with), and
* ``cutout``: ``formats.dae._image_has_cutout`` (some pixel of the image
  scaled to 32 x 32 with ``FastTransformation`` has alpha under 32).

Images are written by Qt itself (RGB, RGBA, gray, palette with and without
alpha, 1 bit), random content, sizes from 1 x 1 up to 100 x 70. Committed;
rerun only to widen (PYTHONPATH at the repository root):

    python scripts/golden/make_qimage_fixtures.py

Deleted with the Python app at M10.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication, QImage, qRgba  # noqa: E402

OUT = ROOT / "rust" / "crates" / "ingetrazo-formats-mesh" / "tests" / "data" / "qimage"


def cutout(img: QImage) -> bool:
    if img.isNull() or not img.hasAlphaChannel():
        return False
    small = img.scaled(32, 32, Qt.IgnoreAspectRatio, Qt.FastTransformation)
    for y in range(small.height()):
        for x in range(small.width()):
            if small.pixelColor(x, y).alpha() < 32:
                return True
    return False


def smooth(img: QImage) -> list:
    px = img.scaled(1, 1, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
    c = px.pixelColor(0, 0)
    return [c.red(), c.green(), c.blue()]


def make(kind: str, w: int, h: int, rng: random.Random) -> QImage:
    if kind == "rgb":
        img = QImage(w, h, QImage.Format_RGB32)
        for y in range(h):
            for x in range(w):
                img.setPixel(x, y, 0xFF000000 | rng.randrange(1 << 24))
    elif kind in ("rgba", "rgba-low", "rgba-binary", "rgba-opaque"):
        img = QImage(w, h, QImage.Format_ARGB32)
        for y in range(h):
            for x in range(w):
                if kind == "rgba":
                    a = rng.choice((0, 255, rng.randrange(256), rng.randrange(256)))
                elif kind == "rgba-low":
                    a = rng.randrange(1, 40)
                elif kind == "rgba-binary":
                    a = rng.choice((0, 255))
                else:
                    a = 255
                img.setPixel(x, y, a << 24 | rng.randrange(1 << 24))
    elif kind == "gray":
        img = QImage(w, h, QImage.Format_Grayscale8)
        for y in range(h):
            for x in range(w):
                img.setPixel(x, y, 0xFF000000 | rng.randrange(256) * 0x010101)
    elif kind in ("palette", "palette-alpha"):
        img = QImage(w, h, QImage.Format_Indexed8)
        img.setColorTable([qRgba(rng.randrange(256), rng.randrange(256), rng.randrange(256),
                                 255 if kind == "palette" else rng.choice((255, 0, rng.randrange(256))))
                           for _ in range(16)])
        for y in range(h):
            for x in range(w):
                img.setPixel(x, y, rng.randrange(16))
    elif kind == "mono":
        img = QImage(w, h, QImage.Format_Mono)
        img.setColorTable([0xFF000000, 0xFFFFFFFF])
        for y in range(h):
            for x in range(w):
                img.setPixel(x, y, rng.randrange(2))
    else:
        raise ValueError(kind)
    return img


def main() -> None:
    app = QGuiApplication.instance() or QGuiApplication(sys.argv)  # noqa: F841
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    rng = random.Random(20261020)
    cases = []
    sizes = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 3), (1, 17), (17, 1), (4, 4), (8, 8), (16, 16),
             (32, 32), (64, 8), (8, 64), (5, 7), (13, 10), (24, 24), (33, 2), (2, 33), (100, 70),
             (31, 29), (7, 5), (64, 48)]
    kinds = ["rgb", "rgba", "rgba-low", "rgba-binary", "rgba-opaque", "gray", "palette",
             "palette-alpha", "mono"]
    n = 0
    for kind in kinds:
        for (w, h) in sizes:
            if kind in ("rgba-low", "rgba-binary", "rgba-opaque", "mono") and (w * h > 700):
                continue
            if rng.random() < 0.45 and kind not in ("rgb", "rgba"):
                continue
            img = make(kind, w, h, rng)
            name = f"{kind}-{w}x{h}.png"
            assert img.save(str(OUT / name)), name
            loaded = QImage(str(OUT / name))
            assert not loaded.isNull()
            cases.append({"file": name, "kind": kind, "w": w, "h": h,
                          "format": str(loaded.format()), "smooth": smooth(loaded),
                          "cutout": cutout(loaded)})
            n += 1
    # a see-through hole too small for the 32 x 32 sample, and one big enough
    for name, hole in (("hole-1px-64.png", [(33, 33)]), ("hole-row-64.png", [(x, 10) for x in range(64)]),
                       ("hole-col-0.png", [(0, y) for y in range(64)])):
        img = QImage(64, 64, QImage.Format_ARGB32)
        img.fill(0xFF336699)
        for (x, y) in hole:
            img.setPixel(x, y, 0x00336699)
        img.save(str(OUT / name))
        loaded = QImage(str(OUT / name))
        cases.append({"file": name, "kind": "hole", "w": 64, "h": 64, "format": str(loaded.format()),
                      "smooth": smooth(loaded), "cutout": cutout(loaded)})
    (OUT / "index.json").write_text(json.dumps(cases, indent=1), encoding="utf-8", newline="\n")
    print(len(cases), "images")


if __name__ == "__main__":
    main()
