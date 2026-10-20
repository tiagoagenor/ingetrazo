# SPDX-License-Identifier: GPL-3.0-or-later
"""Write the fixture that holds the Rust DWG bridge (`ingetrazo-formats-mesh`,
`dwg`, M5.6) to `formats/dwg_bridge.py`.

Everything here comes out of the REAL module: its constants and messages, the
native-executable magic table, the two DXF repairs run over synthetic files
(`_strip_null_handles`, `_dedupe_handles`) and the messages `_run` raises for
converter outcomes (a real child process writes the given stdout / stderr and
exits with the given code). Bytes are hex so the JSON stays ASCII.

    PYTHONPATH=<repo root> python scripts/golden/make_dwg_fixtures.py

Deleted with the Python app at M10.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from formats import dwg_bridge  # noqa: E402

OUT = ROOT / "rust/crates/ingetrazo-formats-mesh/tests/data/dwg/cases.json"


def pairs(*items) -> bytes:
    return "\n".join(str(x) for x in items).encode("latin-1")


REPAIR_INPUTS = {
    "null_handle": pairs(0, "SECTION", 2, "BLOCKS", 0, "ENDBLK", 5, 0,
                         0, "ENDSEC", 0, "EOF"),
    "null_handle_spaced": b"  0\nSECTION\n2\nBLOCKS\n0\nENDBLK\n 5 \n 0 \n0\nENDSEC\n0\nEOF\n",
    "reused_handle": pairs(
        0, "SECTION", 2, "TABLES", 0, "BLOCK_RECORD", 5, 2, 0, "ENDSEC",
        0, "SECTION", 2, "OBJECTS", 0, "LAYOUT", 5, 2, 0, "ENDSEC", 0, "EOF"),
    "handseed": pairs(
        0, "SECTION", 2, "HEADER", 9, "$HANDSEED", 5, 2, 0, "ENDSEC",
        0, "SECTION", 2, "TABLES", 0, "BLOCK_RECORD", 5, 2, 0, "ENDSEC",
        0, "EOF"),
    "handseed_rewritten": pairs(
        0, "SECTION", 2, "HEADER", 9, "$HANDSEED", 5, "FF", 0, "ENDSEC",
        0, "SECTION", 2, "TABLES",
        0, "BLOCK_RECORD", 5, "A", 0, "LAYER", 5, "A", 0, "LTYPE", 5, "a",
        0, "STYLE", 5, "B", 0, "VPORT", 5, "B", 0, "ENDSEC",
        0, "SECTION", 2, "OBJECTS", 0, "LAYOUT", 5, "A", 0, "GROUP", 5, "C",
        0, "ENDSEC", 0, "EOF"),
    "fresh_handle_skips_used": pairs(
        0, "SECTION", 2, "ENTITIES",
        0, "LINE", 5, 1, 0, "LINE", 5, 1, 0, "LINE", 5, 2,
        0, "LINE", 5, 1, 0, "ENDSEC", 0, "EOF"),
    "odd_handles": pairs(
        0, "SECTION", 2, "ENTITIES",
        0, "LINE", 5, "0x10", 0, "LINE", 5, "0X10", 0, "LINE", 5, "1_0",
        0, "LINE", 5, "ZZ", 0, "LINE", 5, "ZZ", 0, "LINE", 5, "-3",
        0, "LINE", 5, "-3", 0, "LINE", 5, "100000000000",
        0, "LINE", 5, "100000000000",
        0, "ENDSEC", 0, "EOF"),
    "weird_separators": (b"0\nSECTION\n2\nENTITIES\n0\nTEXT\n1\nA\x0bB\x0cC\x85D\n"
                         b"5\n7\n0\nTEXT\n5\n7\n0\nENDSEC\n0\nEOF\n"),
    "latin1_names": (b"0\nSECTION\n2\nTABLES\n0\nLAYER\n5\n3\n2\nCA\xd1ER\xcdAS\n"
                     b"0\nLAYER\n5\n3\n0\nENDSEC\n0\nEOF"),
    "odd_trailing_line": b"0\nSECTION\n2\nENTITIES\n0\nEOF\nTAIL",
    "crlf": b"0\r\nSECTION\r\n2\r\nENTITIES\r\n0\r\nLINE\r\n5\r\n4\r\n0\r\nLINE\r\n5\r\n4\r\n0\r\nENDSEC\r\n0\r\nEOF\r\n",
    "no_collision": pairs(0, "SECTION", 2, "ENTITIES", 0, "LINE", 5, "1",
                          0, "LINE", 5, "2", 0, "ENDSEC", 0, "EOF"),
    "empty": b"",
}


def repair_cases(tmp: Path):
    out = []
    for name, data in REPAIR_INPUTS.items():
        p = tmp / "r.dxf"
        p.write_bytes(data)
        dwg_bridge._strip_null_handles(p)
        stripped = p.read_bytes()
        p.write_bytes(data)
        n = dwg_bridge._dedupe_handles(p)
        deduped = p.read_bytes()
        out.append({"name": name, "input": data.hex(), "stripped": stripped.hex(),
                    "dedupe_count": n, "deduped": deduped.hex()})
    return out


#: (name, stdout bytes, stderr bytes, exit code, writes output: None | bytes)
RUN_CASES = [
    ("no_output_stderr_tail", b"", b"one\ntwo\nthree\nfour\n", 1, None),
    ("no_output_stderr_two", b"", b"first\nsecond\n", 0, None),
    ("no_output_stdout_only", b"out1\nout2\n", b"", 2, None),
    ("no_output_stderr_wins", b"out\n", b"err\n", 2, None),
    ("no_output_nothing", b"", b"", 7, None),
    ("no_output_blank", b"", b"  \n\n  \n", 5, None),
    ("no_output_codepage", b"", b"layer CA\xd1ER\xcdAS bad\nsecond\n", 1, None),
    ("no_output_crlf", b"", b"a\r\nb\r\nc\r\nd\r\n", 1, None),
    ("empty_output", b"", b"empty\n", 0, b""),
    ("warnings_but_output", b"", b"warning: recovered\n", 1, b"0\nEOF\n"),
    ("clean_output", b"", b"", 0, b"0\nEOF\n"),
]

HELPER = r"""
import sys
from pathlib import Path
out, stdout_hex, stderr_hex, rc, write_hex = sys.argv[1:6]
sys.stdout.buffer.write(bytes.fromhex(stdout_hex))
sys.stderr.buffer.write(bytes.fromhex(stderr_hex))
sys.stdout.flush(); sys.stderr.flush()
if write_hex != "none":
    Path(out).write_bytes(bytes.fromhex(write_hex))
sys.exit(int(rc))
"""


def run_cases(tmp: Path):
    out = []
    for name, so, se, rc, wr in RUN_CASES:
        target = tmp / "o.dxf"
        if target.exists():
            target.unlink()
        cmd = [sys.executable, "-c", HELPER, str(target), so.hex(), se.hex(),
               str(rc), "none" if wr is None else wr.hex()]
        try:
            stderr = dwg_bridge._run(cmd, target)
            outcome = {"ok": True, "stderr": stderr.encode("utf-8").hex()}
        except dwg_bridge.DwgBridgeError as exc:
            outcome = {"ok": False, "message": str(exc)}
        out.append({"name": name, "stdout": so.hex(), "stderr": se.hex(),
                    "rc": rc, "write": None if wr is None else wr.hex(),
                    **outcome})
    return out


def main() -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        cases = {
            "timeout_seconds": dwg_bridge._TIMEOUT,
            "temp_prefix": "ingetrazo-dwg-",
            "temp_name": "converted.dxf",
            "vendor_rel": ["vendor", "libredwg", "bin"],
            "sibling_rel": ["ingecad", "vendor", "libredwg", "bin"],
            "native_magic": {k: [m.hex() for m in v]
                             for k, v in dwg_bridge._NATIVE_MAGIC.items()},
            "handle_sections": sorted(dwg_bridge._HANDLE_SECTIONS),
            "max_sane_handle": dwg_bridge._MAX_SANE_HANDLE,
            "not_available": None,
            "timed_out_template": "converter timed out: {cmd}",
            "repairs": repair_cases(tmp),
            "runs": run_cases(tmp),
        }
        # The "not available" message, from the real dwg_to_dxf.
        saved = dwg_bridge.find_dwg2dxf
        dwg_bridge.find_dwg2dxf = lambda: None
        try:
            dwg_bridge.dwg_to_dxf(Path("x.dwg"))
        except dwg_bridge.DwgBridgeError as exc:
            cases["not_available"] = str(exc)
        finally:
            dwg_bridge.find_dwg2dxf = saved
        # The timeout message, with a one-second timeout and a sleeper.
        saved_t = dwg_bridge._TIMEOUT
        dwg_bridge._TIMEOUT = 1
        try:
            dwg_bridge._run([sys.executable, "-c", "import time; time.sleep(30)"],
                            tmp / "never.dxf")
        except dwg_bridge.DwgBridgeError as exc:
            cases["timed_out_sample"] = str(exc).replace(sys.executable, "<exe>")
        finally:
            dwg_bridge._TIMEOUT = saved_t
    # The user-facing strings main_window shows (tr() sources) must still be
    # what the Rust constants carry.
    src = (ROOT / "views/main_window.py").read_text(encoding="utf-8")
    for needle in ("Import DWG failed", "Converting DWG…",
                   "The LibreDWG converter (dwg2dxf) is not available in ",
                   "To work on the drawing anyway: save it as DXF from your ",
                   "AutoCAD DWG (*.dwg);;All files (*)"):
        assert needle in src, needle
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases, indent=1, ensure_ascii=True) + "\n",
                   encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
