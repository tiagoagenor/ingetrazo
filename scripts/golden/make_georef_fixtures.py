# SPDX-License-Identifier: GPL-3.0-or-later
"""Write the fixtures that hold the Rust `ingetrazo-georef` crate to the
Python functions it ports (docs/rust-rewrite-plan.md, M7.1 and M7.4):
georef/datum.py, georef/dem.py (and the slippy-map math it uses from
georef/tiles.py) and georef/geotiff.py.

Every number is recorded as exact bits (f64 as big-endian hex of
struct.pack('>d'), f32 as '>f'), so the Rust tests compare bit for bit.
The files are committed, so `cargo test` needs no Python; rerun this only to
widen them:

    PYTHONPATH=<repo> QT_QPA_PLATFORM=offscreen PYTHONUTF8=1 \
        python scripts/golden/make_georef_fixtures.py

Deleted with the Python app at M10.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import struct
import sys
import zlib
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "rust" / "crates" / "ingetrazo-georef" / "tests" / "data"
TIF = OUT / "tif"

import numpy as np  # noqa: E402
from PySide6.QtGui import QGuiApplication, QImage  # noqa: E402

from core.vecmath import QVector3D  # noqa: E402
from georef import datum as D  # noqa: E402
from georef import dem as M  # noqa: E402
from georef import geotiff as G  # noqa: E402
from georef import tiles as T  # noqa: E402
from tests.geotiff_fixture import write_geotiff  # noqa: E402

_app = QGuiApplication.instance() or QGuiApplication([])
RNG = random.Random(20261019)


def d(x: float) -> str:
    return struct.pack(">d", x).hex()


def f(x: float) -> str:
    return struct.pack(">f", x).hex()


def f32(x: float) -> float:
    return struct.unpack(">f", struct.pack(">f", x))[0]


def dl(xs) -> list:
    return [d(float(x)) for x in xs]


def fl(xs) -> str:
    return "".join(f(float(x)) for x in xs)


def vec(v) -> list:
    return [f(v.x()), f(v.y()), f(v.z())]


class Out:
    def __init__(self, name):
        self.path = OUT / name
        self.lines = []

    def add(self, **rec):
        self.lines.append(json.dumps(rec, separators=(",", ":"), ensure_ascii=True))

    def close(self):
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8", newline="\n")
        print(f"{self.path.name}: {len(self.lines)} records")


# ---------------------------------------------------------------------------
# datum
# ---------------------------------------------------------------------------
LATS = [0.0, -0.0, 1e-9, -1e-9, 1e-4, 12.5, -12.0464, -0.1807, 23.4, 33.9, -33.9, 45.0, -45.0,
        59.9139, 60.0, -60.0, 64.1, 80.0, -80.0, 84.0, -84.0, 89.9, -89.9, 89.999999999,
        90.0, -90.0, 1e-300, -1e-300]
LONS = [0.0, -0.0, -77.0428, 10.7522, -78.4678, 179.99999999999997, -179.99999999999997,
        180.0, -180.0, 540.0, -540.0, 1e-9, -1e-9]
for z in (1, 2, 3, 17, 18, 19, 30, 31, 32, 33, 45, 58, 59, 60):
    b = -180.0 + 6.0 * (z - 1)
    LONS += [b, math.nextafter(b, 1e9), math.nextafter(b, -1e9), b + 3.0, b + 6.0 - 1e-12]


def rlat():
    return RNG.uniform(-84.0, 84.0)


def rlon():
    return RNG.uniform(-180.0, 180.0)


def make_datum() -> None:
    o = Out("datum.jsonl")
    # zone_for_lon / _lon0
    lons = list(LONS) + [rlon() for _ in range(1500)] + [RNG.uniform(-720, 720) for _ in range(200)]
    for lon in lons:
        o.add(fn="zone_for_lon", lon=d(lon), out=D.zone_for_lon(lon))
    for z in list(range(-3, 70)) + [RNG.randint(-1000, 1000) for _ in range(50)]:
        o.add(fn="lon0", zone=z, out=d(D._lon0(z)))
    # utm_forward: edge grid x its own zone and a forced neighbour zone
    for lat in LATS:
        for lon in LONS:
            for zone in {D.zone_for_lon(lon), D.zone_for_lon(lon) + 1}:
                e, n = D.utm_forward(lat, lon, zone)
                o.add(fn="utm_forward", lat=d(lat), lon=d(lon), zone=zone, e=d(e), n=d(n))
    for _ in range(1500):
        lat, lon = rlat(), rlon()
        zone = D.zone_for_lon(lon) if RNG.random() < 0.7 else RNG.randint(1, 60)
        e, n = D.utm_forward(lat, lon, zone)
        o.add(fn="utm_forward", lat=d(lat), lon=d(lon), zone=zone, e=d(e), n=d(n))
    # utm_inverse
    for northern in (True, False):
        for e in (100000.0, 166021.0, 500000.0, 833978.0, 900000.0, 499999.99999999994):
            for n in (0.0, 1.0, 1000000.0, 5000000.0, 9999999.0, 10000000.0, 10000001.0, 8300000.0):
                for zone in (1, 17, 18, 32, 60):
                    la, lo = D.utm_inverse(e, n, zone, northern)
                    o.add(fn="utm_inverse", e=d(e), n=d(n), zone=zone, northern=northern,
                          lat=d(la), lon=d(lo))
    for _ in range(1500):
        e = RNG.uniform(100000.0, 900000.0)
        n = RNG.uniform(0.0, 9400000.0)
        zone = RNG.randint(1, 60)
        northern = RNG.random() < 0.5
        la, lo = D.utm_inverse(e, n, zone, northern)
        o.add(fn="utm_inverse", e=d(e), n=d(n), zone=zone, northern=northern,
              lat=d(la), lon=d(lo))
    # _wrap_deg
    wraps = [0.0, -0.0, 180.0, -180.0, 179.99999999999997, -179.99999999999997, 180.00000000000003,
             -180.00000000000003, 360.0, -360.0, 540.0, -540.0, 350.0, -10.0, 1e10, -1e10, 1e300,
             -1e300, 5e-324, -5e-324, 1e-18, -1e-18, 359.99999999999994, 720.0, 90.0, -90.0]
    wraps += [RNG.uniform(-1000, 1000) for _ in range(1500)]
    wraps += [float(RNG.randint(-3600, 3600)) for _ in range(300)]
    for x in wraps:
        o.add(fn="wrap_deg", x=d(x), out=d(D._wrap_deg(x)))
    # math.hypot (the GeoTIFF ground size uses it)
    pairs = [(0.0, 0.0), (3.0, 4.0), (1e-320, 1e-320), (5e-324, 0.0), (1e308, 1e308), (0.1, 0.1),
             (-0.5, 0.5), (1.0, 1e-9), (0.0, -7.25), (2.2250738585072014e-308, 1e-310)]
    for _ in range(1800):
        e1, e2 = RNG.uniform(-8, 6), RNG.uniform(-8, 6)
        pairs.append((RNG.choice((-1, 1)) * RNG.random() * 10.0 ** e1,
                      RNG.choice((-1, 1)) * RNG.random() * 10.0 ** e2))
    for _ in range(500):
        pairs.append((RNG.choice((-1, 1)) * RNG.uniform(0.01, 2.0), RNG.choice((-1, 1)) * RNG.uniform(0.01, 2.0)))
    for x, y in pairs:
        o.add(fn="hypot", x=d(x), y=d(y), out=d(math.hypot(x, y)))
    # SceneDatum
    anchors = [(-12.0464, -77.0428, 154.0, 0.0), (-0.1807, -78.4678, 2850.0, 10.0),
               (59.9139, 10.7522, 0.0, 350.0), (0.0, 0.0, 0.0, 0.0), (-0.0, -0.0, -3.5, -0.0),
               (84.0, 179.9, 12.0, 180.0), (-80.0, -179.9, 0.0, -180.0), (45.0, 6.0, 1.0, 540.0),
               (-33.9, 18.4, 7.0, 90.0), (89.9, 3.0, 0.0, 45.5), (-15.4604, -72.0, 3200.0, -120.25)]
    for _ in range(120):
        anchors.append((rlat(), rlon(), RNG.uniform(-100, 5000), RNG.uniform(-720, 720)))
    for lat, lon, alt, north in anchors:
        s = D.SceneDatum(lat, lon, alt, north=north)
        pts = []
        for _ in range(4):
            la = s.lat + RNG.uniform(-0.02, 0.02)
            lo = s.lon + RNG.uniform(-0.02, 0.02)
            al = None if RNG.random() < 0.3 else RNG.uniform(-50, 4000)
            v = s.geodetic_to_local(la, lo, al)
            pts.append({"op": "geodetic_to_local", "lat": d(la), "lon": d(lo),
                        "alt": None if al is None else d(al), "out": vec(v)})
        # a point beyond the zone (frozen zone keeps going)
        la, lo = s.lat + 0.5, s.lon + 7.0 if s.lon < 170 else s.lon - 7.0
        pts.append({"op": "geodetic_to_local", "lat": d(la), "lon": d(lo), "alt": None,
                    "out": vec(s.geodetic_to_local(la, lo))})
        for _ in range(4):
            p = [f32(RNG.uniform(-3000, 3000)), f32(RNG.uniform(-3000, 3000)), f32(RNG.uniform(-100, 500))]
            q = QVector3D(*p)
            pts.append({"op": "local_to_geodetic", "p": vec(q),
                        "out": dl(s.local_to_geodetic(q))})
            pts.append({"op": "local_to_utm", "p": vec(q), "out": dl(s.local_to_utm(q))})
            e = s._east0 + RNG.uniform(-2000, 2000)
            n = s._north0 + RNG.uniform(-2000, 2000)
            al = RNG.uniform(-20, 3000)
            pts.append({"op": "utm_to_local", "e": d(e), "n": d(n), "alt": d(al),
                        "out": vec(s.utm_to_local(e, n, al))})
            pts.append({"op": "utm_to_local", "e": d(e), "n": d(n), "alt": None,
                        "out": vec(s.utm_to_local(e, n))})
            x, y = RNG.uniform(-5000, 5000), RNG.uniform(-5000, 5000)
            pts.append({"op": "grid_to_local_xy", "a": d(x), "b": d(y),
                        "out": dl(s.grid_to_local_xy(x, y))})
            pts.append({"op": "local_to_grid_xy", "a": d(x), "b": d(y),
                        "out": dl(s.local_to_grid_xy(x, y))})
        o.add(fn="datum", lat=d(lat), lon=d(lon), alt=d(alt), north=d(north),
              out={"lat": d(s.lat), "lon": d(s.lon), "alt": d(s.alt), "north": d(s.north),
                   "zone": s.zone, "northern": s.northern, "hemisphere": s.hemisphere,
                   "east0": d(s._east0), "north0": d(s._north0), "cos_n": d(s._cos_n),
                   "sin_n": d(s._sin_n), "north_vector": vec(s.north_vector()),
                   "to_dict": json.dumps(s.to_dict())},
              points=pts)
    # from_dict
    dicts = [
        {"lat": -12.0464, "lon": -77.0428},
        {"lat": -12.0464, "lon": -77.0428, "alt": 154.0, "north": 15.5},
        {"lat": 1, "lon": 2, "alt": 3, "north": 4},
        {"lat": 1, "lon": 2, "north": 0},
        {"lat": 1.5, "lon": 2.5, "alt": 0.0, "north": 360.0},
        {"lat": 1.5, "lon": 2.5, "alt": 0.0, "north": -190.0},
        {"lat": True, "lon": False},
        {"lat": "10.5", "lon": " 20 "},
        {"lat": 12345678901234567890, "lon": 10},
        {"lat": 1.0},
        {"lon": 1.0},
        {"lat": None, "lon": 1.0},
        {"lat": 1.0, "lon": 1.0, "alt": None},
        {"lat": [1], "lon": 1.0},
    ]
    for dd in dicts:
        try:
            s = D.SceneDatum.from_dict(dd)
            o.add(fn="from_dict", data=json.dumps(dd),
                  out={"lat": d(s.lat), "lon": d(s.lon), "alt": d(s.alt), "north": d(s.north),
                       "zone": s.zone, "to_dict": json.dumps(s.to_dict())})
        except Exception as ex:  # noqa: BLE001
            o.add(fn="from_dict", data=json.dumps(dd), error=type(ex).__name__)
    o.close()


# ---------------------------------------------------------------------------
# dem (+ tiles)
# ---------------------------------------------------------------------------
def make_dem() -> None:
    o = Out("dem.jsonl")
    # bilinear
    shapes = [(1, 1), (1, 5), (5, 1), (2, 2), (3, 4), (5, 7), (16, 16), (9, 3)]
    for _ in range(500):
        h, w = RNG.choice(shapes)
        kind = RNG.random()
        if kind < 0.4:
            vals = [RNG.uniform(-100, 5000) for _ in range(h * w)]
        elif kind < 0.7:
            vals = [float(RNG.randint(-30, 6000)) + RNG.choice((0, 0.5, 0.25, 1 / 256)) for _ in range(h * w)]
        else:
            vals = [RNG.uniform(-1e5, 1e5) * 10 ** RNG.randint(-4, 2) for _ in range(h * w)]
        a = np.array(vals, dtype=np.float32).reshape(h, w)
        for _ in range(3):
            c = RNG.random()
            if c < 0.5:
                px, py = RNG.uniform(-2, w + 2), RNG.uniform(-2, h + 2)
            elif c < 0.7:
                px, py = float(RNG.randint(-2, w + 1)), float(RNG.randint(-2, h + 1))
            elif c < 0.85:
                px, py = RNG.randint(0, max(w, 1)) + RNG.choice((0.5, 0.25, 0.75, 0.999999)), RNG.randint(0, h) + 0.5
            else:
                px, py = RNG.uniform(-1e3, 1e3), RNG.uniform(-1e3, 1e3)
            o.add(fn="bilinear", h=h, w=w, field=fl(a.ravel()), px=d(px), py=d(py),
                  out=d(M.bilinear(a, px, py)))
    # decode_elevation (real QImage)
    for _ in range(150):
        w, h = RNG.randint(1, 7), RNG.randint(1, 7)
        raw = bytes(RNG.getrandbits(8) for _ in range(w * h * 4))
        if RNG.random() < 0.3:        # sea-level-ish terrarium values
            raw = bytes((128 + RNG.randint(-3, 5) if i % 4 == 0 else RNG.getrandbits(8)) for i in range(w * h * 4))
        for enc in ("terrarium", "mapbox"):
            img = QImage(raw, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
            fld = M.decode_elevation(img, enc)
            assert fld.dtype == np.float32 and fld.shape == (h, w)
            o.add(fn="decode", w=w, h=h, encoding=enc, rgba=raw.hex(), out=fl(fld.ravel()))
    # tiles math
    for _ in range(800):
        lat = RNG.uniform(-89.9, 89.9) if RNG.random() < 0.9 else RNG.choice((90.0, -90.0, 85.05112878, -85.05112878, 85.06, 0.0, -0.0))
        lon = rlon()
        zoom = RNG.randint(0, 20)
        xf, yf = T.deg2num(lat, lon, zoom)
        o.add(fn="deg2num", lat=d(lat), lon=d(lon), zoom=zoom, x=d(xf), y=d(yf))
        tx, ty, px, py = M.pixel_in_tile(lat, lon, zoom)
        o.add(fn="pixel_in_tile", lat=d(lat), lon=d(lon), zoom=zoom, tx=tx, ty=ty,
              px=d(px), py=d(py))
    for _ in range(200):
        zoom = RNG.randint(0, 14)
        la = rlat()
        lo = rlon()
        dla, dlo = RNG.uniform(0, 0.1), RNG.uniform(0, 0.1)
        a = (la, lo, la + dla, lo + dlo)
        o.add(fn="tiles_covering", s=d(a[0]), w=d(a[1]), n=d(a[2]), e=d(a[3]), zoom=zoom,
              out=[list(t) for t in T.tiles_covering(*a, zoom)])
    # DEMSampler over injected fields
    class _Stub:
        def request(self, *a):
            return None

    cases = [(-12.0464, -77.0428, 154.0, 0.0, 13),
             (59.9139, 10.7522, 3.0, -45.0, 14)]
    for lat, lon, alt, north, zoom in cases:
        dat = D.SceneDatum(lat, lon, alt, north=north)
        s = M.DEMSampler(dat, zoom=zoom)
        s._fetcher = _Stub()
        tx, ty, _, _ = M.pixel_in_tile(lat, lon, zoom)
        tiles = []
        for ddx, ddy in ((0, 0), (1, 0)):
            vals = [RNG.uniform(-50, 4500) for _ in range(256 * 256)]
            a = np.array(vals, dtype=np.float32).reshape(256, 256)
            s._fields[(tx + ddx, ty + ddy, zoom)] = a
            tiles.append({"tx": tx + ddx, "ty": ty + ddy, "field": fl(a.ravel())})
        qs = []
        for _ in range(60):
            span = 0.2 / (2 ** (zoom - 12))
            la, lo = lat + RNG.uniform(-span, span), lon + RNG.uniform(-span, span)
            qs.append({"op": "elevation_at", "lat": d(la), "lon": d(lo),
                       "out": None if (r := s.elevation_at(la, lo)) is None else d(r)})
            qs.append({"op": "has_tile", "lat": d(la), "lon": d(lo), "out": s.has_tile(la, lo)})
            p = QVector3D(f32(RNG.uniform(-1500, 1500)), f32(RNG.uniform(-1500, 1500)), 0.0)
            qs.append({"op": "elevation_at_local", "p": vec(p),
                       "out": None if (r := s.elevation_at_local(p)) is None else d(r)})
        o.add(fn="sampler", lat=d(lat), lon=d(lon), alt=d(alt), north=d(north), zoom=zoom,
              tiles=tiles, queries=qs)
    o.close()


# ---------------------------------------------------------------------------
# geotiff
# ---------------------------------------------------------------------------
_FMT = {1: "B", 3: "H", 4: "I", 6: "b", 8: "h", 9: "i", 11: "f", 12: "d", 16: "Q", 17: "q", 18: "Q"}


def build_tiff(path, entries, blocks, *, bo="<", big=False, off_tag=273, align=2):
    """A TIFF with one IFD; `entries` are (tag, type, values) and the tag
    `off_tag` is patched to the blocks' offsets."""
    E = bo
    cntfmt = "Q" if big else "I"

    def pack(typ, values):
        if typ == 2:
            return values if isinstance(values, bytes) else values.encode("latin-1")
        if typ == 7:
            return bytes(values)
        if typ in (5, 10):
            flat = [x for pair in values for x in pair]
            return struct.pack(E + ("I" if typ == 5 else "i") * len(flat), *flat)
        return struct.pack(E + _FMT[typ] * len(values), *values)

    def count(typ, values):
        if typ in (2, 7):
            return len(values if isinstance(values, bytes) else values.encode("latin-1"))
        return len(values)

    entries = sorted(entries, key=lambda e: e[0])
    esize = 20 if big else 12
    head = 16 if big else 8
    ifd_size = (8 if big else 2) + len(entries) * esize + (8 if big else 4)
    inline = 8 if big else 4
    cursor = head + ifd_size
    pos = {}
    for tag, typ, values in entries:
        if tag == off_tag:
            values = [0] * len(blocks)
        data = pack(typ, values)
        if len(data) > inline:
            pos[tag] = cursor
            cursor += len(data) + (len(data) & 1)
    bpos = []
    for b in blocks:
        bpos.append(cursor)
        cursor += len(b) + (len(b) & 1)
    out = bytearray()
    out += (b"II" if bo == "<" else b"MM")
    if big:
        out += struct.pack(E + "HHHQ", 43, 8, 0, 16)
        out += struct.pack(E + "Q", len(entries))
    else:
        out += struct.pack(E + "HI", 42, 8)
        out += struct.pack(E + "H", len(entries))
    blobs = []
    for tag, typ, values in entries:
        if tag == off_tag:
            values = bpos
        data = pack(typ, values)
        out += struct.pack(E + "HH" + cntfmt, tag, typ, count(typ, values))
        if len(data) > inline:
            out += struct.pack(E + ("Q" if big else "I"), pos[tag])
            blobs.append(data)
        else:
            out += data.ljust(inline, b"\x00")
    out += b"\x00" * (8 if big else 4)
    for data in blobs:
        out += data + (b"\x00" if len(data) & 1 else b"")
    for b in blocks:
        out += b + (b"\x00" if len(b) & 1 else b"")
    Path(path).write_bytes(bytes(out))


def picture(h=24, w=30, seed=1, alpha_hole=True):
    rng = np.random.default_rng(seed)
    rgba = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    if alpha_hole:
        rgba[:3, :3, 3] = 0
    return rgba


def ukeys(entries):
    return entries


GEOKEYS_UTM = lambda epsg, raster=1: [1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, raster, 3072, 0, 1, epsg]  # noqa: E731


def base_entries(w, h, samples, bits, comp, photo, planar=1, extra=None):
    e = [(256, 4, [w]), (257, 4, [h]), (258, 3, [bits] * samples), (259, 3, [comp]),
         (262, 3, [photo]), (277, 3, [samples]), (284, 3, [planar])]
    if extra:
        e.append((338, 3, extra))
    return e


def strips(arr, rps):
    return [arr[y:y + rps] for y in range(0, arr.shape[0], rps)]


def info_record(info) -> dict:
    return {
        "width": info.width, "height": info.height, "samples": info.samples,
        "bits": list(info.bits), "compression": info.compression,
        "photometric": info.photometric, "planar": info.planar, "tiled": info.tiled,
        "tile_w": info.tile_w, "tile_h": info.tile_h, "offsets": list(info.offsets),
        "bytecounts": list(info.bytecounts), "predictor": info.predictor,
        "jpeg_tables": info.jpeg_tables.hex(), "extra_samples": list(info.extra_samples),
        "sample_format": list(info.sample_format), "big": info.big,
        "byte_order": info.byte_order,
        "transform": None if info.transform is None else dl(info.transform),
        "epsg": info.epsg, "zone": info.zone, "northern": info.northern,
        "geographic": info.geographic, "citation": info.citation, "nodata": info.nodata,
        "georeferenced": info.georeferenced,
    }


def record_file(o: Out, rel: str, rgba_runs=(8192,)) -> None:
    path = TIF / rel
    rec = {"file": f"tif/{rel}"}
    try:
        info = G.read_info(path)
    except Exception as ex:  # noqa: BLE001
        rec["info_error"] = {"cls": type(ex).__name__, "msg": str(ex)}
        o.add(fn="file", **rec)
        return
    rec["info"] = info_record(info)
    rec["describe"] = G.describe(info)
    rec["unsupported"] = G.unsupported_reason(info)
    if info.transform is not None:
        rec["ground_size"] = dl(info.ground_size())
        rec["pixel_to_map"] = [
            {"col": d(c), "row": d(r), "out": dl(info.pixel_to_map(c, r))}
            for c, r in ((0.0, 0.0), (1.0, 1.0), (info.width / 2.0, info.height / 3.0),
                         (float(info.width), float(info.height)), (-2.5, 7.25))]
        if info.georeferenced:
            rec["corners"] = [dl(c) for c in info.corners_geodetic()]
    runs = []
    if info.compression != 7:
        for max_px in rgba_runs:
            run = {"max_px": max_px}
            seen = []
            try:
                arr, factor = G.read_rgba(info, max_px=max_px,
                                          progress=lambda fr: seen.append(fr) or True)
                run.update(factor=factor, h=arr.shape[0], w=arr.shape[1],
                           hex=arr.tobytes().hex(), progress=dl(seen))
            except Exception as ex:  # noqa: BLE001
                run.update(error={"cls": type(ex).__name__, "msg": str(ex)})
            runs.append(run)
        # cancelled read
        try:
            G.read_rgba(info, progress=lambda fr: False)
            rec["cancel"] = None
        except Exception as ex:  # noqa: BLE001
            rec["cancel"] = {"cls": type(ex).__name__, "msg": str(ex)}
    rec["rgba"] = runs
    o.add(fn="file", **rec)


def make_geotiff() -> None:
    if TIF.exists():
        shutil.rmtree(TIF)
    TIF.mkdir(parents=True)
    names = []          # (rel, rgba_runs)

    def reg(rel, runs=(8192,)):
        names.append((rel, runs))

    # --- the project's own writer: compression x predictor x layout
    pic = picture()
    for comp, pred, tile in [(1, 1, 0), (8, 1, 0), (8, 2, 0), (32773, 1, 0), (5, 1, 0), (5, 2, 0),
                             (1, 1, 16), (8, 2, 16), (5, 2, 16), (32773, 1, 16), (32946, 1, 16)]:
        rel = f"std_c{comp}_p{pred}_t{tile}.tif"
        write_geotiff(TIF / rel, pic, tile=tile, compression=comp if comp != 32946 else 8,
                      predictor=pred, origin=(720106.4, 8289766.2), scale=(0.1, 0.1), epsg=32718)
        reg(rel, (8192, 12))
    write_geotiff(TIF / "std_rgb.tif", pic, samples=3, compression=8, origin=(1000.0, 2000.0),
                  scale=(0.5, 0.25), epsg=32633)
    reg("std_rgb.tif")
    write_geotiff(TIF / "std_point.tif", pic, origin=(1000.0, 2000.0), scale=(2.0, 2.0),
                  pixel_is_point=True, epsg=32760)
    reg("std_point.tif")
    write_geotiff(TIF / "std_epsg32601.tif", pic, epsg=32601, origin=(166021.4, 0.0), scale=(1 / 3, 1 / 3))
    reg("std_epsg32601.tif")
    write_geotiff(TIF / "std_epsg32667.tif", pic, epsg=32667)         # not a UTM code: no zone
    reg("std_epsg32667.tif")
    write_geotiff(TIF / "std_epsg32767.tif", pic, epsg=32767)         # user-defined
    reg("std_epsg32767.tif")
    # big mosaic, reductions (factor 2/4/8, carry across strips, partial tiles)
    big = picture(70, 100, seed=3)
    write_geotiff(TIF / "big_tiled.tif", big, compression=8, tile=32)
    reg("big_tiled.tif", (8192, 40, 17, 9))
    write_geotiff(TIF / "big_strips.tif", big, compression=1)
    reg("big_strips.tif", (8192, 40, 17, 9))
    write_geotiff(TIF / "big_lzw_tiled.tif", big, compression=5, tile=16, predictor=2)
    reg("big_lzw_tiled.tif", (8192, 30, 20))
    # a tile that does not divide the reduction (tile 24, factor 4..8)
    write_geotiff(TIF / "odd_tile24.tif", picture(50, 61, seed=4), compression=8, tile=24)
    reg("odd_tile24.tif", (8192, 30))

    # --- hand-built TIFFs
    h, w = 13, 21
    px4 = picture(h, w, seed=7)
    px3 = px4[..., :3].copy()

    def geo(scale=(0.5, 0.5), tie=(0.0, 0.0, 0.0, 500000.0, 4000000.0, 0.0), keys=None, ascii_=None):
        e = [(33550, 12, [scale[0], scale[1], 0.0]), (33922, 12, list(tie))]
        if keys is not None:
            e.append((34735, 3, keys))
        if ascii_ is not None:
            e.append((34737, 2, ascii_))
        return e

    # big-endian classic, RGB raw strips, UTM N
    for bo in ("<", ">"):
        tag = "le" if bo == "<" else "be"
        e = base_entries(w, h, 3, 8, 1, 2) + [(278, 4, [5])] + geo(keys=GEOKEYS_UTM(32633))
        e.append((273, 4, [0]))
        e.append((279, 4, [5 * w * 3, 5 * w * 3, 3 * w * 3]))
        build_tiff(TIF / f"hand_rgb_{tag}.tif", e, [s.tobytes() for s in strips(px3, 5)], bo=bo)
        reg(f"hand_rgb_{tag}.tif", (8192, 8))
        # BigTIFF
        e = base_entries(w, h, 4, 8, 8, 2, extra=[2]) + [(278, 4, [h])] + geo(keys=GEOKEYS_UTM(32718))
        blk = [zlib.compress(px4.tobytes())]
        e.append((273, 16, [0]))
        e.append((279, 16, [len(blk[0])]))
        build_tiff(TIF / f"hand_big_{tag}.tif", e, blk, bo=bo, big=True)
        reg(f"hand_big_{tag}.tif")
        # 16-bit RGB, with and without predictor, strips
        for pred in (1, 2):
            rng = np.random.default_rng(11 + pred)
            v16 = rng.integers(0, 65536, (h, w, 3), dtype=np.uint16)
            arr = v16
            if pred == 2:
                arr = np.diff(v16.astype(np.int32), axis=1, prepend=0).astype(np.uint16)
            data = arr.astype(("<" if bo == "<" else ">") + "u2").tobytes()
            e = base_entries(w, h, 3, 16, 1, 2) + [(278, 4, [h]), (273, 4, [0]), (279, 4, [len(data)])]
            if pred == 2:
                e.append((317, 3, [2]))
            e += geo(keys=GEOKEYS_UTM(32632))
            build_tiff(TIF / f"hand_16bit_{tag}_p{pred}.tif", e, [data], bo=bo)
            reg(f"hand_16bit_{tag}_p{pred}.tif")

    # greyscale, WhiteIsZero, grey + alpha, 1 sample tiled lzw
    g1 = px4[..., 0:1].copy()
    ga = px4[..., 0:2].copy()
    for rel, arr, photo, extra in (("grey.tif", g1, 1, None), ("grey_wiz.tif", g1, 0, None),
                                   ("grey_alpha.tif", ga, 1, [2]), ("grey_alpha_wiz.tif", ga, 0, [2])):
        data = arr.tobytes()
        e = base_entries(w, h, arr.shape[2], 8, 1, photo, extra=extra) + [
            (278, 4, [h]), (273, 4, [0]), (279, 4, [len(data)])] + geo(keys=GEOKEYS_UTM(32717))
        build_tiff(TIF / rel, e, [data])
        reg(rel)
    # planar configuration 2
    planes = [px3[..., i].tobytes() for i in range(3)]
    rps = 6
    blocks, counts = [], []
    for p in range(3):
        arr = px3[..., p]
        for y in range(0, h, rps):
            blocks.append(arr[y:y + rps].tobytes())
            counts.append(len(blocks[-1]))
    e = base_entries(w, h, 3, 8, 1, 2, planar=2) + [(278, 4, [rps]), (273, 4, [0]), (279, 4, counts)]
    e += geo(keys=GEOKEYS_UTM(32718))
    build_tiff(TIF / "planar2.tif", e, blocks)
    reg("planar2.tif", (8192, 5))
    # RGB with 5 samples (photometric 2): extra samples ignored
    px5 = np.concatenate([px4, px4[..., :1]], axis=2)
    data = px5.tobytes()
    e = base_entries(w, h, 5, 8, 1, 2, extra=[2, 0]) + [(278, 4, [h]), (273, 4, [0]),
                                                        (279, 4, [len(data)])] + geo(keys=GEOKEYS_UTM(32718))
    build_tiff(TIF / "samples5.tif", e, [data])
    reg("samples5.tif")

    # --- placement variants (tiny raster)
    tiny = px3[:6, :8].copy()
    td = tiny.tobytes()

    def placed(rel, extra_entries, runs=(8192,)):
        e = base_entries(8, 6, 3, 8, 1, 2) + [(278, 4, [6]), (273, 4, [0]), (279, 4, [len(td)])] + extra_entries
        build_tiff(TIF / rel, e, [td])
        reg(rel, runs)

    placed("geo_none.tif", [])
    placed("geo_scale_only.tif", [(33550, 12, [1.0, 1.0, 0.0])])
    placed("geo_geographic_model2.tif", geo(scale=(1e-5, 1e-5), tie=(0, 0, 0, -77.04, -12.04, 0),
                                            keys=[1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326]))
    placed("geo_geographic_type_only.tif", geo(scale=(1e-5, 1e-5), tie=(0, 0, 0, 120.5, -45.25, 0),
                                               keys=[1, 1, 0, 1, 2048, 0, 1, 4326]))
    placed("geo_geographic_out_of_range.tif", geo(scale=(1.0, 1.0), tie=(0, 0, 0, 500.0, 40.0, 0),
                                                  keys=[1, 1, 0, 1, 2048, 0, 1, 4326]))
    placed("geo_geographic_edge.tif", geo(scale=(1.0, 1.0), tie=(0, 0, 0, 180.0, -90.0, 0),
                                          keys=[1, 1, 0, 1, 2048, 0, 1, 4326]))
    placed("geo_geographic_point.tif", geo(scale=(2e-5, 3e-5), tie=(0, 0, 0, -77.04, -12.04, 0),
                                           keys=[1, 1, 0, 4, 1024, 0, 1, 2, 1025, 0, 1, 2, 2048, 0, 1, 4326]))
    placed("geo_matrix.tif", [(34264, 12, [0.25, 0.01, 0.0, 300000.5, -0.02, -0.25, 0.0, 6000000.25,
                                           0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                              (34735, 3, GEOKEYS_UTM(32718))])
    placed("geo_matrix_point.tif", [(34264, 12, [0.25, 0.01, 0.0, 300000.5, -0.02, -0.25, 0.0, 6000000.25,
                                                 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                                    (34735, 3, GEOKEYS_UTM(32718, raster=2))])
    placed("geo_matrix_short.tif", geo(keys=GEOKEYS_UTM(32718)) + [(34264, 12, [1.0] * 12)])
    placed("geo_tie_short.tif", [(33550, 12, [1.0, 1.0, 0.0]), (33922, 12, [0.0, 0.0, 0.0]),
                                 (34735, 3, GEOKEYS_UTM(32718))])
    placed("geo_tie_offset.tif", geo(scale=(0.3, 0.7), tie=(2.0, 3.0, 0.0, 600000.0, 5000000.0, 0.0),
                                     keys=GEOKEYS_UTM(32615)))
    # citation-only UTM zone (no EPSG)
    cit = "WGS 84 / UTM zone 18S|"
    placed("geo_citation_utm.tif", geo(keys=[1, 1, 0, 3, 1024, 0, 1, 1, 1026, 34737, len(cit), 0, 3073, 34737, 0, 0],
                                       ascii_=cit))
    placed("geo_citation_pcs.tif", geo(keys=[1, 1, 0, 3, 1024, 0, 1, 1, 3073, 34737, 24, 0, 1026, 34737, 8, 24],
                                       ascii_="WGS 84 / UTM zone 5n|GCS name|x"))
    ascii_multi = "Some CRS|junk  UTM  ZONE  7 s||tail|"
    placed("geo_citation_loose.tif", geo(keys=[1, 1, 0, 2, 1024, 0, 1, 1, 1026, 34737, 8, 0], ascii_=ascii_multi))
    placed("geo_citation_zone_digits.tif", geo(keys=[1, 1, 0, 1, 1024, 0, 1, 1], ascii_="utm zone 123N and UTM zone 12 N"))
    placed("geo_citation_nomatch.tif", geo(keys=[1, 1, 0, 1, 1024, 0, 1, 1], ascii_="UTM zone N|zone 18S"))
    placed("geo_epsg0.tif", geo(keys=GEOKEYS_UTM(0), ascii_="UTM zone 30N"))
    placed("geo_keys_trunc.tif", geo(keys=[1, 1, 0, 9, 1024, 0, 1, 1, 3072, 0, 1, 32718]))
    placed("geo_keys_short.tif", geo(keys=[1, 1, 0]))
    placed("geo_model_projected_no_epsg.tif", geo(keys=[1, 1, 0, 1, 1024, 0, 1, 1]))
    # nodata variants and assorted tag types
    placed("nodata_ascii.tif", geo(keys=GEOKEYS_UTM(32718)) + [(42113, 2, "-9999")])
    placed("nodata_double.tif", geo(keys=GEOKEYS_UTM(32718)) + [(42113, 12, [-9999.0])])
    placed("nodata_double2.tif", geo(keys=GEOKEYS_UTM(32718)) + [(42113, 12, [0.1, -1e300])])
    placed("nodata_short.tif", geo(keys=GEOKEYS_UTM(32718)) + [(42113, 3, [255])])
    placed("nodata_float.tif", geo(keys=GEOKEYS_UTM(32718)) + [(42113, 11, [-3.4028234663852886e38])])
    placed("nodata_bytes.tif", geo(keys=GEOKEYS_UTM(32718)) + [(42113, 7, b"a\x00'\\\xffz\n")])
    placed("nodata_empty.tif", geo(keys=GEOKEYS_UTM(32718)) + [(42113, 12, [])])
    placed("tag_zoo.tif", geo(keys=GEOKEYS_UTM(32718)) + [
        (282, 5, [(300, 1), (1, 3), (5, 0)]), (283, 10, [(-300, 7), (4, -2)]),
        (700, 6, [-1, 127, -128]), (701, 8, [-2, 300 - 600]), (702, 9, [-70000, 2]),
        (703, 11, [1.5, -2.25, 0.1]), (704, 16, [2 ** 40, 7]) if False else (704, 4, [2 ** 31, 7]),
        (705, 1, [1, 2, 3, 4, 5, 6, 7]), (706, 7, b"undefined-bytes!"), (707, 2, "ascii text\x00\x00"),
        (708, 3, list(range(40))), (709, 12, [1.0] * 6)])
    # the unknown type 99 entry is skipped: patch type bytes of the last tag
    # (done by a separate file below)
    # --- BigTIFF with 64-bit types
    e = base_entries(8, 6, 3, 8, 1, 2) + [(278, 16, [6]), (273, 16, [0]), (279, 16, [len(td)])]
    e += geo(keys=GEOKEYS_UTM(32718)) + [(704, 16, [2 ** 40, 7]), (705, 17, [-5, 2 ** 40]), (706, 18, [1])]
    build_tiff(TIF / "big_types.tif", e, [td], bo="<", big=True)
    reg("big_types.tif")
    build_tiff(TIF / "big_types_be.tif", e, [td], bo=">", big=True)
    reg("big_types_be.tif")
    # a stripped file whose RowsPerStrip is zero / larger than the height / missing
    for name, rps in (("rps0", 0), ("rps_big", 1000), ("rps_none", None)):
        e = base_entries(8, 6, 3, 8, 1, 2) + [(273, 4, [0]), (279, 4, [len(td)])]
        if rps is not None:
            e.append((278, 4, [rps]))
        e += geo(keys=GEOKEYS_UTM(32718))
        build_tiff(TIF / f"{name}.tif", e, [td])
        reg(f"{name}.tif")
    # defaults: no BitsPerSample, no photometric, no samples, 3-sample photometric default
    e = [(256, 4, [8]), (257, 4, [6]), (273, 4, [0]), (279, 4, [8 * 6])]
    build_tiff(TIF / "defaults_1bit.tif", e, [bytes(8 * 6)])
    reg("defaults_1bit.tif")
    e = [(256, 4, [8]), (257, 4, [6]), (258, 3, [8]), (273, 4, [0]), (279, 4, [8 * 6]), (277, 3, [3])]
    build_tiff(TIF / "bits_shorter.tif", e, [tiny.tobytes()[:48]])
    reg("bits_shorter.tif")
    # tiled with fewer tiles than needed is undefined in Python (np.empty): skipped.
    # data shorter than needed pads with zeros; extra data is cut
    for name, cut in (("short_data", 40), ("long_data", 300)):
        data = (td + bytes(300))[:cut]
        e = base_entries(8, 6, 3, 8, 1, 2) + [(278, 4, [6]), (273, 4, [0]), (279, 4, [len(data)])]
        build_tiff(TIF / f"{name}.tif", e, [data])
        reg(f"{name}.tif")
    # truncated deflate / corrupt lzw / packbits garbage
    good = zlib.compress(td)
    for name, comp, data in (("deflate_corrupt", 8, good[:len(good) // 2]),
                             ("deflate_garbage", 8, b"not zlib at all"),
                             ("lzw_corrupt", 5, bytes(range(1, 40))),
                             ("packbits_garbage", 32773, bytes(range(1, 60)))):
        e = base_entries(8, 6, 3, 8, comp, 2) + [(278, 4, [6]), (273, 4, [0]), (279, 4, [len(data)])]
        e += geo(keys=GEOKEYS_UTM(32718))
        build_tiff(TIF / f"{name}.tif", e, [data])
        reg(f"{name}.tif")
    # unsupported layouts
    for name, ent in (
            ("unsup_float", base_entries(8, 6, 1, 32, 1, 1) + [(339, 3, [3])]),
            ("unsup_float_bits8", base_entries(8, 6, 1, 8, 1, 1) + [(339, 3, [3])]),
            ("unsup_int16_signed", base_entries(8, 6, 1, 16, 1, 1) + [(339, 3, [2])]),
            ("unsup_bits12", base_entries(8, 6, 3, 12, 1, 2)),
            ("unsup_palette", base_entries(8, 6, 1, 8, 1, 3)),
            ("unsup_jp2k", base_entries(8, 6, 3, 8, 34712, 2)),
            ("unsup_samples6_grey", base_entries(8, 6, 6, 8, 1, 1)),
            ("unsup_jpeg16", base_entries(8, 6, 3, 16, 7, 2)),
            ("unsup_jpeg_planar", base_entries(8, 6, 3, 8, 7, 2, planar=2))):
        data = bytes(8 * 6 * 6)
        e = ent + [(278, 4, [6]), (273, 4, [0]), (279, 4, [len(data)])]
        build_tiff(TIF / f"{name}.tif", e, [data])
        reg(f"{name}.tif")
    # error cases
    (TIF / "not_tiff.png").write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(40))
    reg("not_tiff.png")
    (TIF / "tiny.tif").write_bytes(b"II*\x00")
    reg("tiny.tif")
    (TIF / "bad_magic.tif").write_bytes(b"II" + struct.pack("<H", 44) + bytes(20))
    reg("bad_magic.tif")
    (TIF / "empty.tif").write_bytes(b"")
    reg("empty.tif")
    e = [(257, 4, [6]), (273, 4, [0]), (279, 4, [4])]
    build_tiff(TIF / "no_width.tif", e, [bytes(4)])
    reg("no_width.tif")
    e = [(256, 4, [8]), (257, 4, [6])]
    build_tiff(TIF / "no_data.tif", e, [])
    reg("no_data.tif")
    e = [(256, 4, [8]), (257, 4, [6]), (273, 4, [0])]
    build_tiff(TIF / "no_counts.tif", e, [bytes(48)])
    reg("no_counts.tif")
    e = [(256, 3, []), (257, 4, [6]), (273, 4, [0]), (279, 4, [4])]
    build_tiff(TIF / "empty_width.tif", e, [bytes(4)])
    reg("empty_width.tif")
    raw = (TIF / "hand_rgb_le.tif").read_bytes()
    (TIF / "trunc_ifd.tif").write_bytes(raw[:60])
    reg("trunc_ifd.tif")
    (TIF / "trunc_data.tif").write_bytes(raw[:len(raw) - 400])
    reg("trunc_data.tif")
    # unknown field type 99 (entry skipped): patch a copy of tag_zoo
    z = bytearray((TIF / "tag_zoo.tif").read_bytes())
    nent = struct.unpack("<H", z[8:10])[0]
    for i in range(nent):
        o_ = 10 + i * 12
        if struct.unpack("<H", z[o_:o_ + 2])[0] == 700:
            z[o_ + 2:o_ + 4] = struct.pack("<H", 99)
    (TIF / "tag_unknown_type.tif").write_bytes(bytes(z))
    reg("tag_unknown_type.tif")
    # out-of-line value that points past the end of the file (skipped by Python)
    z = bytearray((TIF / "tag_zoo.tif").read_bytes())
    for i in range(nent):
        o_ = 10 + i * 12
        if struct.unpack("<H", z[o_:o_ + 2])[0] == 708:
            z[o_ + 8:o_ + 12] = struct.pack("<I", len(z) + 1000)
    (TIF / "tag_past_eof.tif").write_bytes(bytes(z))
    reg("tag_past_eof.tif")
    # the real GDAL JPEG-in-TIFF tile of the Python tests (header only)
    shutil.copy(ROOT / "tests" / "data" / "orto_jpeg_tile.tif", TIF / "orto_jpeg_tile.tif")
    reg("orto_jpeg_tile.tif")

    o = Out("geotiff.jsonl")
    for rel, runs in names:
        record_file(o, rel, runs)
    # unsupported_reason over mutated headers
    base = G.read_info(TIF / "std_c8_p1_t0.tif")
    muts = [
        {}, {"compression": 34712}, {"compression": 7}, {"compression": 7, "planar": 2},
        {"compression": 7, "bits": (16, 16, 16, 16)}, {"bits": (12, 12, 12, 12)},
        {"bits": (8, 8, 16, 8)}, {"bits": (8, 8, 8, 12)}, {"bits": (4,)},
        {"sample_format": (3,)}, {"sample_format": (1, 1, 1, 1)}, {"sample_format": (4, 2)},
        {"sample_format": (1, 1, 1, 1, 3)}, {"photometric": 3}, {"photometric": 5},
        {"photometric": 6}, {"samples": 5, "photometric": 1}, {"samples": 5, "photometric": 2},
        {"samples": 0, "photometric": 1}, {"samples": 1, "bits": (8,)},
        {"compression": 5}, {"compression": 32773}, {"compression": 32946}, {"compression": 2},
    ]
    for m in muts:
        import dataclasses
        mi = dataclasses.replace(base, **m)
        o.add(fn="unsupported_reason",
              set={k: (list(v) if isinstance(v, tuple) else v) for k, v in m.items()},
              out=G.unsupported_reason(mi))
    o.close()
    total = sum(p.stat().st_size for p in TIF.iterdir())
    print(f"tif/: {len(list(TIF.iterdir()))} files, {total} bytes")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ".gitattributes").write_text("* -text\n", newline="\n")
    make_datum()
    make_dem()
    make_geotiff()


if __name__ == "__main__":
    main()
