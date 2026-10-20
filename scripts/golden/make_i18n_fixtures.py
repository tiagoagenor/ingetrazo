# SPDX-License-Identifier: GPL-3.0-or-later
"""Write the fixtures that hold the Rust `ingetrazo-i18n` crate to the Python
it ports (docs/rust-rewrite-plan.md, M6.12): core/i18n.py.

The REAL core.i18n runs, once per language, over every key of every
i18n/*.json (their union, so a key a language lacks exercises the English
fallback), with several sets of field values built from the field specs the
key itself uses ({name}, {n:02d}, {x:+.2f}, ...). A key that contains no brace
anywhere (in itself or any translation) is plain: only a 32-bit FNV-1a hash of
its output is kept, with and without unrelated keyword arguments. Python
errors are recorded by exception class. source_of() is recorded for every
translated value. The files are committed, so `cargo test` needs no Python;
rerun only to widen them:

    PYTHONPATH=<repo> QT_QPA_PLATFORM=offscreen PYTHONUTF8=1 \
        python scripts/golden/make_i18n_fixtures.py

Deleted with the Python app at M10.
"""
from __future__ import annotations

import json
import os
import string
import struct
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "rust" / "crates" / "ingetrazo-i18n" / "tests" / "data"

from core import i18n  # noqa: E402

INTS = [7, -12345, 0, 1000000, 42]
FLOATS = [3.14159, -0.5, 2.5, 1234567.891, 0.125, 1e22, -0.0004, 100.0, 0.0,
          1e-7, 123456789012345678.0, 0.1]
STRS = ["plano", "Ñandú ✓ 日本", "a{b}", "", "x" * 20]
LANGS = ["en", "es", "pt-BR", "zh-CN", "it", "id", "fr", "de", "pl", "ru", "xx"]
VARIANTS = 9


def fnv(s: str) -> str:
    h = 0x811C9DC5
    for b in s.encode("utf-8"):
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def enc(v):
    if isinstance(v, int):
        return {"i": v}
    if isinstance(v, float):
        return {"f": struct.pack(">d", v).hex()}
    return {"s": v}


def fields_of(text: str):
    """(name, kind) in order of first appearance; kind d, f or s."""
    out = {}
    try:
        for _lit, name, spec, _conv in string.Formatter().parse(text):
            if name is None or name in out:
                continue
            out[name] = ("d" if spec.endswith("d") else
                         "f" if spec.endswith("f") else "s")
    except ValueError:
        pass
    return list(out.items())


def variant_args(fields, v):
    args = {}
    for j, (name, kind) in enumerate(fields):
        if v == 5 and j == 0:
            continue
        if v == 7:
            continue
        if v == 6:
            args[name] = STRS[j % len(STRS)]
        elif v == 8:
            args[name] = (FLOATS[j % len(FLOATS)] if kind == "d"
                          else INTS[j % len(INTS)] if kind == "f"
                          else STRS[j % len(STRS)])
        elif kind == "s" and v == 3:
            args[name] = FLOATS[(j + 3) % len(FLOATS)]
        elif kind == "s" and v == 4:
            args[name] = INTS[(j + 1) % len(INTS)]
        elif kind == "d":
            args[name] = INTS[(j + 2 * (v % 3)) % len(INTS)]
        elif kind == "f":
            args[name] = FLOATS[(j + 4 * (v % 3) + (v == 3) * 5) % len(FLOATS)]
        else:
            args[name] = STRS[(j + 2 * (v % 3)) % len(STRS)]
    if v == 7:
        args["unrelated_zz"] = 1
    return args


def run(text, args):
    try:
        return {"o": i18n.tr(text, **args)}
    except Exception as e:  # noqa: BLE001 - the class is the datum
        return {"e": type(e).__name__}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cats = {}
    for lang in LANGS:
        i18n.set_language(lang)
        cats[lang] = dict(i18n._catalog)
    keys = sorted({k for c in cats.values() for k in c})
    kidx = {k: i for i, k in enumerate(keys)}

    def braced(k):
        return "{" in k or "}" in k or any(
            "{" in c.get(k, "") or "}" in c.get(k, "") for c in cats.values())

    lines = [{"langs": LANGS, "keys": keys}]
    nfield = 0
    for k in keys:
        if not braced(k):
            continue
        fields = fields_of(k)
        argsets = [variant_args(fields, v) for v in range(VARIANTS)]
        out = {}
        for lang in LANGS:
            i18n.set_language(lang)
            out[lang] = [run(k, a) for a in argsets]
        lines.append({"k": kidx[k],
                      "a": [{n: enc(x) for n, x in a.items()} for a in argsets],
                      "o": out})
        nfield += 1
    for lang in LANGS:
        i18n.set_language(lang)
        rows = []
        for k in keys:
            if not braced(k):
                rows.append([kidx[k], fnv(i18n.tr(k)),
                             fnv(i18n.tr(k, unrelated_zz=1))])
        cat = i18n._catalog
        lines.append({"lang": lang, "plain": rows,
                      "cat_keys": [kidx[k] for k in cat],
                      "source_of": [kidx.get(i18n.source_of(v), -1)
                                    for v in cat.values()],
                      "current": i18n.current_language()})
    i18n.set_language("en")
    with open(OUT / "tr.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for ln in lines:
            f.write(json.dumps(ln, ensure_ascii=False,
                               separators=(",", ":")) + "\n")
    (OUT / ".gitattributes").write_text("* -text\n", encoding="utf-8",
                                        newline="\n")
    print("keys", len(keys), "field-bearing", nfield)


if __name__ == "__main__":
    main()
