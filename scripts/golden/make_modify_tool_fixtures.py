# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""Fixtures that hold ``ingetrazo-tools``' MODIFY and MEASURE tools to the Python
app (docs/rust-rewrite-plan.md, M3.13): Move, Rotate, Scale, Offset, Eraser,
Paint, Tape Measure, Protractor and Flip.

Each tool has a module here (``modify_<tool>.py``) with its scenes and its
programs; ``_modify_harness.py`` drives the REAL tool through the REAL
``Viewport`` (real mouse and key events through a real camera, the VCB typed
for real) and records, call by call, what the tool was handed and, after every
event, the whole state: the ordered meshes, every group's transform, the
guides, the selection, the undo depth and the tool's probe (rubber band, guide
preview, label, status clause...). The Rust tools replay the calls and are held
to the same state after every event (``tests/modify_*_parity.rs``).

Scenes are saved as bare-JSON .igz and reopened in a fresh viewport by every
program, so Python and Rust start from the very same bits. Programs include
seeded random ones; a program that makes Python raise, or that reaches
something not ported (see each module's ``is_valid``), is dropped and counted.

Committed; rerun only to widen (PYTHONPATH at the repository root, the main
checkout's venv, ``INGETRAZO_RUST`` unset):

    python scripts/golden/make_modify_tool_fixtures.py [tool ...]

Deleted with the Python app at M10.
"""
from __future__ import annotations

import importlib
import json
import shutil
import sys
import traceback
from pathlib import Path

import _modify_harness as H

TOOLS = ["move", "rotate", "protractor", "scale", "flip", "offset", "eraser", "paint", "tape"]


def build_tool(name: str) -> None:
    try:
        mod = importlib.import_module(f"modify_{name}")
    except ModuleNotFoundError as e:
        if e.name == f"modify_{name}":
            print(f"{name}: no module yet")
            return
        raise
    out = H.OUT_ROOT / name
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / ".gitattributes").write_text("* -text\n", encoding="utf-8", newline="\n")
    paths = {}
    for sname, build in mod.SCENES.items():
        # Group identities are random; the committed scenes must not change on
        # a rerun.
        import core.group as core_group
        core_group.new_uid = H.UidSource()
        scene = build()
        path = out / f"{sname}.igz.json"
        H.save_scene_json(scene, path)
        paths[sname] = path
    cases, dropped, events = [], 0, 0
    extra = getattr(mod, "extra_probe", None)
    valid = getattr(mod, "is_valid", lambda rec: True)
    for n, (sname, cam, evs) in enumerate(mod.PROGRAMS):
        try:
            rec = H.run_program(paths[sname], cam, mod.make_tool, evs, extra)
        except Exception:  # noqa: BLE001
            dropped += 1
            print(f"{name}: program {n} on {sname} raised, dropped", file=sys.stderr)
            traceback.print_exc()
            continue
        if not valid(rec):
            dropped += 1
            print(f"{name}: program {n} on {sname} not representable, dropped", file=sys.stderr)
            continue
        rec["scene"] = sname
        rec["n"] = n
        cases.append(rec)
        events += len(rec["events"])
    (out / "cases.json").write_text(json.dumps(cases, sort_keys=True, separators=(",", ":")),
                                    encoding="utf-8", newline="\n")
    print(f"{name}: {len(cases)} programs, {events} events, {dropped} dropped")


def main() -> None:
    only = [a for a in sys.argv[1:] if not a.startswith("-")] or TOOLS
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    for name in only:
        build_tool(name)


if __name__ == "__main__":
    main()
