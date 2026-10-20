# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""Fixtures for ``ingetrazo-ops``'s ``PySet``: CPython's set of integer
pairs, to the slot. Random add/discard histories over keys of the size the
planar arrangement uses (``round(x / 1e-4)``, negative too); each case
records the hash of every key, the history, the iteration order and the
iteration order of ``set(s)`` (the copy ``set_merge`` makes). Committed;
rerun only to widen:

    python scripts/golden/make_pyset_fixtures.py

Deleted with the Python app at M10.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rust" / "crates" / "ingetrazo-ops" / "tests" / "data" / "pyset.json"


def main() -> None:
    rnd = random.Random(20261019)
    cases = []
    for _ in range(160):
        span = rnd.choice([3, 10, 100, 100000])
        keys = list({(rnd.randint(-span, span * 3), rnd.randint(-span, span * 3)) for _ in range(rnd.randint(1, 45))})
        rnd.shuffle(keys)
        hist = []
        s = set()
        for _ in range(rnd.randint(1, 140)):
            k = rnd.randrange(len(keys))
            if rnd.random() < 0.8:
                s.add(keys[k])
                hist.append(["a", k])
            else:
                s.discard(keys[k])
                hist.append(["d", k])
        cases.append({
            "keys": keys,
            "hash": [hash(k) for k in keys],
            "hist": hist,
            "order": [keys.index(k) for k in s],
            "copy": [keys.index(k) for k in set(s)],
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases, separators=(",", ":")), encoding="utf-8", newline="\n")
    print(len(cases), "cases", sys.version.split()[0])
    # random.Random(seed): the draws core.orient makes (seed 12345), and a few
    # other seeds (0, a word boundary, a two-word one).
    draws = []
    for seed in (12345, 0, 1, 2**32 - 1, 2**32, 2**40 + 5, 987654321):
        r = random.Random(seed)
        draws.append({"seed": seed,
                      "random": [r.random().hex() for _ in range(700)],
                      "uniform": [r.uniform(-3e-4, 3e-4).hex() for _ in range(14)]})
    (OUT.parent / "pyrandom.json").write_text(json.dumps(draws, separators=(",", ":")), encoding="utf-8",
                                              newline="\n")


if __name__ == "__main__":
    main()
