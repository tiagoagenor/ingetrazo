# SPDX-License-Identifier: GPL-3.0-or-later
"""``pytest --golden-check``: record every History a test drives, replay
each script on a fresh scene, and report per command class what records
and replays exactly (docs/rust-rewrite-plan.md, M0.4).
``--golden-harvest`` also writes each script that replays exactly into
``golden/`` (M0.5): ``ops/<id>.json``, ``starts/<hash>.igz``, ``out/<id>/`` and a row of
``ledger.csv``.

    pytest --golden-check --golden-report=golden-check.json
    pytest --golden-harvest
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import pytest

_report = {"tests": 0, "scripts": 0, "replayed_ok": 0,
           "ops": Counter(), "ops_ok": Counter(),
           "errors": defaultdict(Counter), "diverged": defaultdict(Counter),
           "script_errors": Counter(), "examples": {}}


def pytest_addoption(parser):
    g = parser.getgroup("golden")
    g.addoption("--golden-check", action="store_true",
                help="record and replay every History (golden corpus check)")
    g.addoption("--golden-report", default="golden-check.json",
                help="where --golden-check writes its JSON report")
    g.addoption("--golden-harvest", action="store_true",
                help="--golden-check, and write every exact script to golden/")


def _ops(steps):
    for s in steps:
        yield s
        for v in (s.get("args") or {}).values():
            yield from _inner(v)


def _inner(v):
    if isinstance(v, dict):
        if "$cmd" in v and isinstance(v["$cmd"], dict):
            yield from _ops([v["$cmd"]])
        else:
            for x in v.values():
                yield from _inner(x)
    elif isinstance(v, list):
        for x in v:
            yield from _inner(x)


@pytest.fixture(autouse=True)
def _golden_check(request):
    harvest = request.config.getoption("--golden-harvest")
    if not (request.config.getoption("--golden-check") or harvest):
        yield
        return
    from tests.golden import recorder

    recorder.take_histories()
    recorder.start()
    try:
        yield
    finally:
        recorder.stop()
    histories = recorder.take_histories()
    _report["tests"] += 1
    nodeid = request.node.nodeid
    written = 0
    for h in histories:
        script = recorder.script_of(h)
        if script is None:
            continue
        _report["scripts"] += 1
        steps = script["steps"]
        bad = list(script["errors"])
        for op in _ops(steps):
            if op["op"] in ("undo", "redo"):
                continue
            _report["ops"][op["op"]] += 1
            if "error" in op:
                _report["errors"][op["op"]][op["error"][:120]] += 1
                bad.append(op["error"])
        if bad:
            for e in script["errors"]:
                _report["script_errors"][e.split(":")[-1].strip()[:120]] += 1
            continue
        if script["start"] is None:
            continue
        res = recorder.replay(script)
        if res["ok"]:
            _report["replayed_ok"] += 1
            if harvest:
                written += 1
                _write_golden(nodeid, written, script, h.scene)
            for op in _ops(steps):
                if op["op"] not in ("undo", "redo"):
                    _report["ops_ok"][op["op"]] += 1
        else:
            at = res["at"]
            name = steps[at]["op"] if at is not None and 0 <= at < len(steps) else "start"
            _report["diverged"][name][res["why"][:160]] += 1
            _report["examples"].setdefault(f"{name}: {res['why'][:80]}", nodeid)


def golden_id(nodeid: str, n: int = 1) -> str:
    """golden/SCHEMA.md, section 1."""
    import hashlib
    import re

    path, _, rest = nodeid.partition("::")
    stem = path.replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".py")
    name, _, param = rest.partition("[")
    parts = [stem, name.replace("::", "__")] + ([param.rstrip("]")] if param else [])
    slug = re.sub(r"[^a-z0-9_]+", "-", "__".join(parts).lower())[:80]
    key = nodeid if n == 1 else f"{nodeid}#{n}"
    return f"{slug}-{hashlib.sha1(key.encode('utf-8')).hexdigest()[:8]}"


def _write_golden(nodeid: str, n: int, script: dict, scene) -> None:
    import datetime
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "golden"
    gid = golden_id(nodeid, n)
    import hashlib

    from tests.golden.signature import scene_signature, signature_hash
    # Start scenes are stored once by content (many tests start empty).
    h = hashlib.sha1(script["start"]).hexdigest()[:16]
    starts = root / "starts"
    starts.mkdir(parents=True, exist_ok=True)
    if not (starts / f"{h}.igz").exists():
        (starts / f"{h}.igz").write_bytes(script["start"])
        with open(starts / f"{h}.sig.json", "w", encoding="utf-8", newline="\n") as fh:
            json.dump(script["start_parts"], fh, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    final = scene_signature(scene)
    if signature_hash(final) == script["steps"][-1]["sig"]:   # the test may go on after
        out = root / "out" / gid
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "final.sig.json", "w", encoding="utf-8", newline="\n") as fh:
            json.dump(final, fh, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    from core.version import __version__ as app_version
    doc = {"schema": 1, "id": gid,
           "source": {"pytest": nodeid if n == 1 else f"{nodeid}#{n}",
                      "app_version": app_version,
                      "harvested": datetime.date.today().isoformat()},
           "start": {"igz": f"golden/starts/{h}.igz"},
           "start_sig": script["start_sig"],
           "steps": script["steps"]}
    (root / "ops").mkdir(parents=True, exist_ok=True)
    # One key per line, one step per line: small diffs, readable files.
    head = {k: v for k, v in doc.items() if k != "steps"}
    lines = ["{"]
    lines += [f" {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}," for k, v in head.items()]
    steps = [json.dumps(s, ensure_ascii=False, separators=(",", ":")) for s in doc["steps"]]
    lines.append(' "steps": [')
    lines += [f"  {s}," for s in steps[:-1]] + [f"  {steps[-1]}"]
    lines += [" ]", "}"]
    with open(root / "ops" / f"{gid}.json", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    _ledger_rows.append((nodeid, gid))


_ledger_rows: list = []


def _write_ledger() -> None:
    """Add the harvested rows to golden/ledger.csv: one row per pytest id
    (a test with several scripts names the first), existing rows kept."""
    import csv
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "golden" / "ledger.csv"
    rows = {}
    if path.exists():
        with open(path, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                rows[r["pytest_id"]] = r
    seen = set()
    for nodeid, gid in _ledger_rows:
        if nodeid not in seen:                     # a test's first script
            seen.add(nodeid)
            rows[nodeid] = {"pytest_id": nodeid, "status": "golden",
                            "target": gid, "reason": ""}
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        w = csv.DictWriter(fh, ["pytest_id", "status", "target", "reason"],
                           lineterminator="\n")
        w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])


def pytest_sessionfinish(session, exitstatus):
    harvest = session.config.getoption("--golden-harvest", default=False)
    if harvest and _ledger_rows:
        _write_ledger()
    if not (session.config.getoption("--golden-check", default=False) or harvest):
        return
    out = {k: (dict(v) if isinstance(v, Counter) else
               {a: dict(b) for a, b in v.items()} if isinstance(v, defaultdict) else v)
           for k, v in _report.items()}
    with open(session.config.getoption("--golden-report"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, sort_keys=True)
