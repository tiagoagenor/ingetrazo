# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""The L2 signature of the scene Python has after importing a .skp, as
canonical JSON on stdout, and the time of ``parse_skp`` and ``apply_payload``
on stderr (M5.7). The Rust twin is the example ``skp_sig`` of
``ingetrazo-formats-mesh``; ``--diff A B`` prints where two signatures first
differ.

    python scripts/golden/skp_sig.py FILE.skp [CACHE_DIR]
    python scripts/golden/skp_sig.py --diff A.json B.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.pop("INGETRAZO_RUST", None)


def diff(a, b, path="$"):
    if type(a) is not type(b):
        return f"{path}: {type(a).__name__} vs {type(b).__name__}: {str(a)[:120]} | {str(b)[:120]}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}.{k}: only in {'A' if k in a else 'B'}"
            d = diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{path}: {str(a)[:160]} | {str(b)[:160]}"


def main() -> None:
    if sys.argv[1] == "--diff":
        a = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        b = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
        print(diff(a, b) or "equal")
        return
    if len(sys.argv) > 2:
        os.environ["INGETRAZO_TEXTURE_CACHE"] = sys.argv[2]
    from PySide6.QtGui import QGuiApplication
    app = QGuiApplication.instance() or QGuiApplication(sys.argv)  # noqa: F841
    from core.scene import Scene
    from formats import skp as skp_format
    from tests.golden.signature import scene_signature

    t = time.time()
    payload = skp_format.parse_skp(sys.argv[1])
    t1 = time.time()
    scene = Scene()
    skp_format.apply_payload(scene, payload)
    t2 = time.time()
    print("parse %.2fs apply %.2fs" % (t1 - t, t2 - t1), file=sys.stderr)
    if os.environ.get("SKP_SIG_FACES"):
        # one line per face of every mesh: its corners and its attrs
        from tests.golden.signature import canonical, vkey
        meshes = [scene.mesh] + [g.mesh for g in scene.groups]
        for mi, mesh in enumerate(meshes):
            for f in mesh.faces:
                key = ",".join(vkey(v.position) for v in f.loop)
                print(mi, key, json.dumps(canonical(f.attrs), sort_keys=True, separators=(",", ":")))
        return
    if os.environ.get("SKP_SIG_HASH"):
        import hashlib
        text = json.dumps(scene_signature(scene), sort_keys=True, separators=(",", ":"), allow_nan=False)
        print(hashlib.sha256(text.encode()).hexdigest(), len(text))
        return
    if not os.environ.get("SKP_SIG_QUIET"):
        print(json.dumps(scene_signature(scene), sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
