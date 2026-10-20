# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""Counts of an openskp parse of one .skp, in the format of the Rust example
``cargo run --release -p ingetrazo-skp --example summary -- FILE`` (M5.7), so
the two parsers can be compared by eye or ``diff`` on real files.

    python scripts/golden/skp_summary.py FILE.skp
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import openskp  # noqa: E402

from formats import openskp_compat  # noqa: E402


def main() -> None:
    openskp_compat.apply()
    t = time.time()
    m = openskp.SkpFile.open(sys.argv[1]).parse()
    openskp_compat.resolve_face_layers(m)
    dt = time.time() - t
    tot = [0, 0, 0, 0]
    for d in m.definitions.values():
        tot[0] += len(d.vertices)
        tot[1] += len(d.edges)
        tot[2] += len(d.faces)
        tot[3] += len(d.instances)
    r = m.root
    print(f"version={m.version} units={m.units!r} defs={len(m.definitions)} "
          f"root={(len(r.vertices), len(r.edges), len(r.faces), len(r.instances))} "
          f"tot={tuple(tot)} layers={len(m.layers)} materials={len(m.materials)} "
          f"by_id={len(m.materials_by_id)} pages={len(m.pages)} "
          f"dims={len(m.dimensions)} styles={len(m.styles)}")
    print("parse %.2fs" % dt, file=sys.stderr)


if __name__ == "__main__":
    main()
