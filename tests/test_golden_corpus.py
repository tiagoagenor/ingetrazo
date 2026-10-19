# SPDX-License-Identifier: GPL-3.0-or-later
"""Every script of the golden corpus (golden/) replays, from its files
alone, to the signatures Python recorded — so a change to the Python app
that would make the corpus stale shows here, before Rust is held to it."""
from __future__ import annotations

import pytest

from tests.golden import corpus, recorder

IDS = corpus.ids()


@pytest.mark.skipif(not IDS, reason="no golden scripts harvested yet")
@pytest.mark.parametrize("gid", IDS)
def test_golden_script_replays(gid):
    script = corpus.load(gid)
    res = recorder.replay(script)
    assert res["ok"], f"{script['source']['pytest']}: step {res['at']}: {res['why']}"
