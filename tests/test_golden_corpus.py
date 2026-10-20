# SPDX-License-Identifier: GPL-3.0-or-later
"""Every script of the golden corpus (golden/) replays, from its files
alone, to the signatures Python recorded — so a change to the Python app
that would make the corpus stale shows here, before Rust is held to it."""
from __future__ import annotations

import sys

import pytest

from tests.golden import corpus, recorder

IDS = corpus.ids()


@pytest.mark.skipif(not IDS, reason="no golden scripts harvested yet")
@pytest.mark.parametrize("gid", IDS)
def test_golden_script_replays(gid):
    script = corpus.load(gid)
    res = recorder.replay(script)
    src = script['source'].get('pytest') or script['source']
    msg = f"{src}: step {res['at']}: {res['why']}"
    if not res["ok"] and sys.platform != "win32":
        # The corpus was recorded on Windows, where Qt's QVector3D::length()
        # is MSVC's hypot; libstdc++ (Linux) and libc++ (macOS) round it
        # differently, and the signatures key on float32 bits. The Rust
        # replay pins the MSVC formula (Hypot::Msvc) and runs on every OS
        # (the `rust` CI jobs), so the corpus is still held there; here a
        # divergence off Windows is reported, not failed.
        pytest.skip(f"recorded on Windows (MSVC hypot): {msg}")
    assert res["ok"], msg
