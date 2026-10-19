# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""The Rust TIFF decompressors (georef.geotiff) give the Python's bytes:
LZW streams from a reference encoder (code widths 9 to 12 and the early
change, KwKwK codes, clear codes, a missing end code, a cut tail) and
random garbage, which walks every branch of a decoder; PackBits literals,
runs, the no-op 128 and cut streams."""
from __future__ import annotations

import random

import pytest

import core._native as _native
from georef import geotiff as G

nat = _native.available()
pytestmark = pytest.mark.skipif(
    nat is None or not hasattr(nat, "lzw_decode"),
    reason="native module not built")


def _encode(data: bytes, clear_every: int = 0, end: bool = True) -> bytes:
    """TIFF LZW, MSB-first, early change — the writer's side."""
    out_bits: list = []
    bits = 9

    def put(code):
        for k in range(bits - 1, -1, -1):
            out_bits.append((code >> k) & 1)

    table = {bytes([i]): i for i in range(256)}
    nxt = 258
    put(256)
    w = b""
    count = 0
    for byte in data:
        wc = w + bytes([byte])
        if wc in table:
            w = wc
            continue
        put(table[w])
        table[wc] = nxt
        nxt += 1
        count += 1
        # The decoder adds its entry one code later, so its early change
        # (len + 1 >= 2**bits) is this side's nxt >= 2**bits.
        if nxt >= (1 << bits) and bits < 12:
            bits += 1
        if nxt >= 4094 or (clear_every and count % clear_every == 0):
            put(256)
            table = {bytes([i]): i for i in range(256)}
            nxt = 258
            bits = 9
        w = bytes([byte])
    if w:
        put(table[w])
    if end:
        put(257)
    while len(out_bits) % 8:
        out_bits.append(0)
    return bytes(int("".join(map(str, out_bits[i:i + 8])), 2)
                 for i in range(0, len(out_bits), 8))


def _payloads(rng):
    yield b""
    yield b"A"
    yield b"ABABABABABABABABABABAB"                 # KwKwK
    yield bytes([7]) * 5000                         # long runs: KwKwK chains
    yield bytes(rng.randrange(4) for _ in range(20000))   # 9→12 bits
    yield bytes(rng.randrange(256) for _ in range(9000))  # table overflow
    img = bytes((x * y) % 251 for y in range(64) for x in range(64))
    yield img


@pytest.mark.parametrize("clear_every", [0, 37, 600])
def test_lzw_streams(clear_every):
    rng = random.Random(clear_every)
    for payload in _payloads(rng):
        for end in (True, False):
            stream = _encode(payload, clear_every, end)
            py = G._lzw_decode(stream)
            assert nat.lzw_decode(stream) == py
            if end:
                assert py == payload
            cut = stream[:len(stream) * 2 // 3]
            assert nat.lzw_decode(cut) == G._lzw_decode(cut)


@pytest.mark.parametrize("seed", range(30))
def test_lzw_garbage(seed):
    rng = random.Random(seed)
    data = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 3000)))
    assert nat.lzw_decode(data) == G._lzw_decode(data)


@pytest.mark.parametrize("seed", range(30))
def test_packbits(seed):
    rng = random.Random(seed)
    data = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 600)))
    assert nat.unpackbits(data) == G._unpackbits(data)
    for crafted in (bytes([2, 1, 2, 3, 254, 7, 128, 255]), bytes([5, 1]),
                    bytes([129]), b""):
        assert nat.unpackbits(crafted) == G._unpackbits(crafted)
