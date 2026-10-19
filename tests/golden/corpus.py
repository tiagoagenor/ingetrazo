# SPDX-License-Identifier: GPL-3.0-or-later
"""The golden corpus on disk (golden/ops, golden/out): list and load."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "golden"


def ids() -> list[str]:
    return sorted(p.stem for p in (ROOT / "ops").glob("*.json"))


def load(gid: str) -> dict:
    """A script as ``recorder.replay`` takes it, read from the files only."""
    doc = json.loads((ROOT / "ops" / f"{gid}.json").read_text(encoding="utf-8"))
    if doc.get("schema") != 1:
        raise ValueError(f"{gid}: schema {doc.get('schema')}")
    start = doc["start"]
    if "igz" not in start:
        raise ValueError(f"{gid}: start {start}")
    return {"start": (ROOT.parent / start["igz"]).read_bytes(),
            "start_sig": doc["start_sig"], "steps": doc["steps"],
            "source": doc["source"]}
