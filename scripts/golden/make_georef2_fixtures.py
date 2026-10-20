# SPDX-License-Identifier: GPL-3.0-or-later
"""Write the fixtures that hold the second half of the Rust `ingetrazo-georef`
crate to the Python it ports (docs/rust-rewrite-plan.md, M7.2, M7.3, M7.5):
georef/tiles.py, georef/tile_fetcher.py, georef/terrain.py, georef/surface.py,
georef/geocode.py, georef/geopath.py, georef/points.py, georef/profile.py and
georef/geoimport.py. (datum, dem, geotiff: make_georef_fixtures.py.)

Every number is recorded as exact bits (f64 as big-endian hex of
struct.pack('>d'), f32 as '>f'). The REAL Python functions run on the inputs;
the network is never touched (the fetcher's downloads are replaced by a
recorder, samplers read injected heightfields). The files are committed, so
`cargo test` needs no Python; rerun only to widen them:

    PYTHONPATH=<repo> QT_QPA_PLATFORM=offscreen PYTHONUTF8=1 \
        python scripts/golden/make_georef2_fixtures.py

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
import tempfile
import zipfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "rust" / "crates" / "ingetrazo-georef" / "tests" / "data"

import numpy as np  # noqa: E402
from PySide6.QtCore import QBuffer, QByteArray, QIODevice  # noqa: E402
from PySide6.QtGui import QGuiApplication, QImage  # noqa: E402
from PySide6.QtNetwork import QNetworkReply  # noqa: E402

from core.vecmath import QVector3D  # noqa: E402
from georef import datum as D  # noqa: E402
from georef import dem as M  # noqa: E402
from georef import geocode as GC  # noqa: E402
from georef import geoimport as GI  # noqa: E402
from georef import geopath as GP  # noqa: E402
from georef import points as PT  # noqa: E402
from georef import profile as PF  # noqa: E402
from georef import surface as S  # noqa: E402
from georef import terrain as TR  # noqa: E402
from georef import tile_fetcher as TF  # noqa: E402
from georef import tiles as T  # noqa: E402

_app = QGuiApplication.instance() or QGuiApplication([])
RNG = random.Random(20261020)


def d(x: float) -> str:
    return struct.pack(">d", x).hex()


def f(x: float) -> str:
    return struct.pack(">f", x).hex()


def f32(x: float) -> float:
    return struct.unpack(">f", struct.pack(">f", x))[0]


def vec(v) -> list:
    return [f(v.x()), f(v.y()), f(v.z())]


def fnv64(data: bytes) -> str:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def run(fn, *a, **k):
    """(True, result) or (False, exception class name)."""
    try:
        return True, fn(*a, **k)
    except Exception as e:  # noqa: BLE001 - the class name is the record
        return False, type(e).__name__


class Out:
    def __init__(self, name):
        self.path = OUT / name
        self.lines = []

    def add(self, **rec):
        self.lines.append(json.dumps(rec, separators=(",", ":"), ensure_ascii=True))

    def close(self):
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8", newline="\n")
        print(f"{self.path.name}: {len(self.lines)} records, {self.path.stat().st_size} bytes")


def dd(x):
    """A float or None as a record value."""
    return None if x is None else d(x)


def tl(lst) -> dict:
    """A list of (x, y) tiles: the hash always, the list itself when short."""
    b = b"".join(struct.pack(">qq", x, y) for x, y in lst)
    rec = {"n": len(lst), "hash": fnv64(b)}
    if len(lst) <= 40:
        rec["list"] = [list(t) for t in lst]
    return rec


def pj(v) -> str:
    return json.dumps(v)


DATUMS = [
    (-12.0464, -77.0428, 154.0, 0.0),
    (-13.532, -71.9675, 3400.0, 20.0),
    (59.9139, 10.7522, 3.0, -45.0),
    (-17.7134, 179.9990, 5.0, 0.0),
    (-17.7134, -179.9990, 5.0, 10.0),
    (70.2, 23.4, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (-33.9, 18.4, 10.0, 90.0),
    (35.0, 139.0, -4.5, 180.0),
]


def mkdatum(t):
    return D.SceneDatum(t[0], t[1], t[2], north=t[3])


def drec(t):
    return {"lat": d(t[0]), "lon": d(t[1]), "alt": d(t[2]), "north": d(t[3])}


class _Stub:
    """A fetcher that never fetches (the sampler's tile requests go nowhere)."""

    def request(self, *a):
        return None


def field_of(seed: int, h: int = 256, w: int = 256):
    """A deterministic heightfield made of dyadic rationals, so Rust rebuilds
    it exactly from the seed alone."""
    y, x = np.mgrid[0:h, 0:w].astype(np.int64)
    v = (((x * 7 + y * 13 + seed * 101) % 509).astype(np.float64) * 0.0625
         + ((x * y + seed) % 31).astype(np.float64) * 0.125
         + (x + y).astype(np.float64) * 0.25 + seed * 10.0)
    return v.astype(np.float32)


def make_sampler(dat, zoom, ll_bbox, drop=(), margin=0.002):
    """A real DEMSampler with injected fields over the tiles covering the
    geodetic bbox (minus the `drop` indices). Returns (sampler, tile specs)."""
    s = M.DEMSampler(dat, zoom=zoom)
    s._fetcher = _Stub()
    la_s, lo_w, la_n, lo_e = ll_bbox
    tiles = T.tiles_covering(la_s - margin, lo_w - margin, la_n + margin, lo_e + margin, zoom)
    specs = []
    for i, (tx, ty) in enumerate(tiles):
        if i in drop:
            continue
        seed = (tx * 31 + ty * 17) % 97
        s._fields[(tx, ty, zoom)] = field_of(seed)
        specs.append({"tx": tx, "ty": ty, "seed": seed})
    return s, specs


# ---------------------------------------------------------------------------
# tiles
# ---------------------------------------------------------------------------
def png_bytes(color=0xFFFF0000, w=4, h=4) -> bytes:
    img = QImage(w, h, QImage.Format.Format_RGB32)
    img.fill(color)
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    return bytes(buf.data())


def src_rec(s) -> dict:
    return {"id": s.id, "name": s.name, "url_template": s.url_template, "max_zoom": s.max_zoom,
            "attribution": s.attribution, "subdomains": list(s.subdomains), "tile_size": s.tile_size}


def layer_rec(layer) -> dict:
    return {"source": src_rec(layer.source), "zoom": layer.zoom, "radius_m": d(layer.radius_m),
            "visible": layer.visible, "opacity": d(layer.opacity),
            "patches": [[d(v) for v in p] for p in layer.patches]}


FROM_DICT_TEXTS = [
    '{}', '{"source":"osm"}', '{"source":"s2cloudless","zoom":12}',
    '{"source":"esri_imagery","zoom":12,"radius_m":500,"visible":false,"opacity":0.5,'
    '"patches":[[0,0,100,100],[5,5,1,2]]}',
    '{"source":"custom-x","name":"X","url_template":"http://a/{z}/{x}/{y}","attribution":"AT","zoom":"13"}',
    '{"source":"custom"}', '{"source":"nope"}', '{"source":""}', '{"source":"nope","name":"N"}',
    '{"zoom":"abc"}', '{"zoom":null}', '{"zoom":2.9}', '{"zoom":-2.9}', '{"zoom":" 7 "}',
    '{"zoom":"1_0"}', '{"zoom":true}', '{"zoom":1e400}', '{"zoom":[1]}',
    '{"opacity":5}', '{"opacity":-3}', '{"opacity":"0.25"}', '{"opacity":NaN}', '{"opacity":null}',
    '{"opacity":Infinity}', '{"opacity":-Infinity}', '{"opacity":true}',
    '{"radius_m":"1e3"}', '{"radius_m":"1_000.5"}', '{"radius_m":"  nan "}', '{"radius_m":"inf"}',
    '{"radius_m":"0x10"}', '{"radius_m":""}', '{"radius_m":[]}', '{"radius_m":100000000000000000000}',
    '{"patches":[]}', '{"patches":[[1,2,3]]}', '{"patches":[[1,2,3,"x"]]}', '{"patches":5}',
    '{"patches":[5]}', '{"patches":["abc"]}', '{"patches":null}', '{"patches":[["1","2","3","4"]]}',
    '{"patches":[[1,2,3,4,5]]}',
    '[]', '5', 'null', '"x"',
    '{"source":[1]}', '{"source":{"a":1}}', '{"source":5}', '{"source":null}',
    '{"visible":0}', '{"visible":"no"}', '{"visible":""}', '{"visible":[]}',
    '{"name":5,"source":"zz"}', '{"url_template":5,"source":"zz"}',
    '{"source":"zz","name":"","url_template":"","attribution":""}',
]
FROM_DICT_NM = {'{"patches":[[1,2,3]]}', '{"patches":[[1,2,3,4,5]]}', '{"source":5}', '{"source":null}',
                '{"name":5,"source":"zz"}', '{"url_template":5,"source":"zz"}'}


def make_tiles() -> None:
    o = Out("tiles.jsonl")
    # num2deg
    for _ in range(700):
        zoom = RNG.randint(0, 24)
        n = 2 ** zoom
        kind = RNG.random()
        if kind < 0.5:
            x, y = RNG.uniform(0, n), RNG.uniform(0, n)
        elif kind < 0.8:
            x, y = float(RNG.randint(0, n)), float(RNG.randint(0, n))
        else:
            x, y = RNG.uniform(-2, n + 2), RNG.uniform(-2, n + 2)
        lat, lon = T.num2deg(x, y, zoom)
        o.add(fn="num2deg", x=d(x), y=d(y), zoom=zoom, lat=d(lat), lon=d(lon))
    for zoom in (0, 1, 2, 30):
        for x, y in ((0.0, 0.0), (0.5, 0.5), (float(2 ** zoom), float(2 ** zoom)), (0.0, float(2 ** zoom))):
            lat, lon = T.num2deg(x, y, zoom)
            o.add(fn="num2deg", x=d(x), y=d(y), zoom=zoom, lat=d(lat), lon=d(lon))
    # tile_bbox
    for _ in range(500):
        zoom = RNG.randint(0, 22)
        n = 2 ** zoom
        x, y = RNG.randint(0, n - 1), RNG.randint(0, n - 1)
        o.add(fn="tile_bbox", x=x, y=y, zoom=zoom, out=[d(v) for v in T.tile_bbox(x, y, zoom)])
    for zoom in (0, 1, 2, 5, 22):
        n = 2 ** zoom
        for x, y in ((0, 0), (n - 1, 0), (0, n - 1), (n - 1, n - 1)):
            o.add(fn="tile_bbox", x=x, y=y, zoom=zoom, out=[d(v) for v in T.tile_bbox(x, y, zoom)])
    # tiles_covering: antimeridian, poles, swapped corners, zero area, world
    boxes = [(-80, -179, 80, 179), (-90, -180, 90, 180), (-85.06, -180, 85.06, 180), (10, 170, 11, 190),
             (10, -190, 11, -170), (-60, 179.9, -59.9, 180.0), (84.9, -10, 90, 10), (-90, 0, -84.9, 1),
             (5, 5, 5, 5), (5, 5, -5, -5), (0, 0, 0, 0), (-0.0, -0.0, 0.0, 0.0), (45.0, 10.0, 45.001, 10.001)]
    for box in boxes:
        for zoom in (0, 1, 2, 3, 5, 8, 12):
            if (box[2] - box[0]) * (box[3] - box[1]) > 1000 and zoom > 5:
                continue
            o.add(fn="tiles_covering", s=d(box[0]), w=d(box[1]), n=d(box[2]), e=d(box[3]), zoom=zoom,
                  out=tl(T.tiles_covering(box[0], box[1], box[2], box[3], zoom)))
    for _ in range(250):
        zoom = RNG.randint(0, 16)
        la, lo = RNG.uniform(-89, 89), RNG.uniform(-185, 185)
        span = 10 ** RNG.uniform(-4, 0.5) / (2 ** max(zoom - 8, 0))
        a = (la, lo, la + span * RNG.uniform(0, 1), lo + span * RNG.uniform(0, 1))
        o.add(fn="tiles_covering", s=d(a[0]), w=d(a[1]), n=d(a[2]), e=d(a[3]), zoom=zoom,
              out=tl(T.tiles_covering(*a, zoom)))
    # sources
    for p in T.PRESETS.values():
        o.add(fn="preset", src=src_rec(p))
    srcs = list(T.PRESETS.values()) + [
        T.TileSource("t", "T", "https://{s}.example.com/{z}/{x}/{y}.png", subdomains=("a", "b", "c")),
        T.TileSource("u", "U", "{s}{s}/{x}/{x}/{y}/{z}{z}", subdomains=("{z}", "q")),
        T.TileSource("v", "V", "no placeholders at all"),
        T.custom_source("https://mytiles.example/{z}/{x}/{y}.jpg", 20),
    ]
    for s in srcs:
        for _ in range(12):
            z = RNG.randint(0, 20)
            x, y = RNG.randint(0, 2 ** z), RNG.randint(0, 2 ** z)
            o.add(fn="url", src=src_rec(s), x=x, y=y, z=z, out=s.url(x, y, z))
        o.add(fn="url", src=src_rec(s), x=-3, y=1, z=4, out=s.url(-3, 1, 4))
        o.add(fn="url", src=src_rec(s), x=0, y=0, z=0, out=s.url(0, 0, 0))
    names = ["Mi Fuente", "  ", "", "Ñandú 2020!", "A--B  C", "---", "ÁÉÍ", "İstanbul", "K", "x/y\\z", "tile.v1",
             "UPPER lower 123", "日本語", "a", "-a-", "ß"]
    for nm in names:
        o.add(fn="slug", name=nm, out=T.source_slug(nm))
    for nm in (None, "", "Mi Fuente", "Ñandú", "   ", "x"):
        for mz in (19, 12):
            c = T.custom_source("https://e/{z}/{x}/{y}", mz, nm)
            o.add(fn="custom_source", name=nm, max_zoom=mz, out=src_rec(c))
    # layers
    cfgs = []
    for dt in DATUMS:
        for sid, zoom, radius in (("esri_imagery", 16, 1200.0), ("osm", 17, 300.0), ("s2cloudless", 14, 4000.0)):
            cfgs.append((dt, sid, zoom, radius, []))
    for dt in DATUMS[:4]:
        cfgs.append((dt, "osm", 17, 200.0, [["rect", 800.0, 200.0, 50.0, -30.0]]))
        cfgs.append((dt, "esri_imagery", 18, 100.0, [["patch", 500.0, 500.0, 50.0, 50.0], ["patch", -300.0, 100.0, 80.0, 20.0],
                                                     ["patch", 0.0, 0.0, 10.0, 10.0]]))
        cfgs.append((dt, "esri_imagery", 18, 3000.0, []))
        cfgs.append((dt, "osm", 19, 3000.0, [["rect", 6000.0, 300.0, 0.0, 0.0]]))
    for dt, sid, zoom, radius, ops in cfgs:
        dat = mkdatum(dt)
        layer = T.TileLayer(T.PRESETS[sid], zoom=zoom, radius_m=radius)
        for op in ops:
            if op[0] == "rect":
                layer.set_rectangle(op[1], op[2], op[3], op[4])
            else:
                layer.add_patch(*op[1:])
        layer.visible = RNG.random() < 0.8
        layer.opacity = RNG.choice((1.0, 0.5, 0.0, 0.123456789))
        ft = layer.flat_tiles(dat)
        rec = {"fn": "layer", "datum": drec(dt), "layer": layer_rec(layer),
               "to_dict": json.dumps(layer.to_dict()), "flat_tiles": tl(ft),
               "visible_tiles": tl(layer.visible_tiles(dat))}
        quads = []
        for (tx, ty) in ft[:2]:
            quads.append({"x": tx, "y": ty, "out": [[vec(p), [d(u), d(v)]] for p, (u, v) in layer.quad_local(dat, tx, ty)]})
        rec["quads"] = quads
        caps = []
        for max_tiles, min_zoom in ((500, 8), (10, 8), (10, 16), (1, 0), (100000, 8)):
            lc = T.TileLayer.from_dict(layer.to_dict())
            reduced = lc.cap_detail(dat, max_tiles, min_zoom)
            caps.append({"max_tiles": max_tiles, "min_zoom": min_zoom, "reduced": reduced, "zoom": lc.zoom})
        rec["caps"] = caps
        o.add(**rec)
    # from_dict
    for text in FROM_DICT_TEXTS:
        ok, v = run(lambda: T.TileLayer.from_dict(json.loads(text)))
        rec = {"fn": "layer_from_dict", "data": text}
        if text in FROM_DICT_NM:
            rec["nm"] = True
        if ok:
            rec["out"] = layer_rec(v)
            rec["to_dict"] = json.dumps(v.to_dict()) if not rec.get("nm") else None
        else:
            rec["error"] = v
        o.add(**rec)
    # cache: path layout, put/get/evict sequences on real files
    for sid, x, y, z in (("osm", 1, 2, 3), ("esri_imagery", 4, 5, 6), ("custom-x", 0, 0, 0), ("a", 123, 456, 19)):
        c = T.TileCache(Path(tempfile.mkdtemp()))
        rel = c.path_for(sid, x, y, z).relative_to(c.root).as_posix()
        o.add(fn="cache_path", sid=sid, x=x, y=y, z=z, out=rel)
    for _ in range(40):
        tmp = Path(tempfile.mkdtemp())
        budget = RNG.choice((50, 120, 300, 1000))
        cache = T.TileCache(tmp, budget)
        ops, results = [], []
        for _ in range(RNG.randint(4, 14)):
            sid = RNG.choice(("osm", "esri"))
            x, y, z = RNG.randint(0, 3), RNG.randint(0, 3), RNG.randint(0, 2)
            if RNG.random() < 0.7:
                size = RNG.choice((0, 10, 20, 40, 60, 100))
                ops.append(["put", sid, x, y, z, size])
                cache.put(sid, x, y, z, b"x" * size)
                results.append(None)
            else:
                ops.append(["get", sid, x, y, z, 0])
                got = cache.get(sid, x, y, z)
                results.append(None if got is None else len(got))
        surv = sorted(p.relative_to(tmp).as_posix() for p in tmp.rglob("*") if p.is_file())
        o.add(fn="cache_run", budget=budget, ops=ops, results=results, survivors=surv)
        shutil.rmtree(tmp, ignore_errors=True)
    for _ in range(40):
        tmp = Path(tempfile.mkdtemp())
        n = RNG.randint(1, 12)
        mtimes = RNG.sample(range(1_000_000, 1_000_100), n)
        files = []
        for i in range(n):
            size = RNG.choice((1, 5, 10, 50))
            p = tmp / f"s{i % 3}" / "1" / str(i)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"y" * size)
            os.utime(p, (mtimes[i], mtimes[i]))
            files.append([f"s{i % 3}/1/{i}", mtimes[i], size])
        budget = RNG.choice((0, 10, 30, 80, 200, 10000))
        c = T.TileCache(tmp, budget)
        c._evict_if_needed()
        surv = sorted(p.relative_to(tmp).as_posix() for p in tmp.rglob("*") if p.is_file())
        o.add(fn="evict", budget=budget, files=files, survivors=surv)
        shutil.rmtree(tmp, ignore_errors=True)
    o.close()


# ---------------------------------------------------------------------------
# tile_fetcher
# ---------------------------------------------------------------------------
class _Abortable:
    def abort(self):
        pass


class _Reply:
    def __init__(self, err=None, data=b""):
        self._err, self._data = err, data

    def error(self):
        return (QNetworkReply.NetworkError.NoError if self._err is None
                else QNetworkReply.NetworkError.UnknownNetworkError)

    def errorString(self):
        return self._err or ""

    def readAll(self):
        return QByteArray(self._data)

    def deleteLater(self):
        pass


def make_fetcher_scenarios() -> None:
    o = Out("tile_fetcher.jsonl")
    sources = [T.PRESETS["osm"], T.PRESETS["s2cloudless"], T.PRESETS["esri_imagery"]]
    good, junk = png_bytes(), b"not a png at all"
    for scen in range(36):
        tmp = Path(tempfile.mkdtemp())
        cache = T.TileCache(tmp)
        fetcher = TF.TileFetcher(cache)
        started, ready, failed = [], [], []

        def fake_start(source, x, y, z, fetcher=fetcher, started=started):
            started.append([source.id, x, y, z, source.url(x, y, z)])
            fetcher._inflight[(source.id, x, y, z)] = _Abortable()

        fetcher._start_download = fake_start
        fetcher.tileReady.connect(lambda sid, x, y, z, img: ready.append([sid, x, y, z]))
        fetcher.tileFailed.connect(lambda sid, x, y, z, reason: failed.append(reason))
        init = []
        for _ in range(RNG.randint(0, 6)):
            s = RNG.choice(sources)
            x, y, z = RNG.randint(0, 5), RNG.randint(0, 5), RNG.choice((5, 17, 18))
            kind = RNG.choice(("good", "corrupt"))
            cache.put(s.id, x, y, z, good if kind == "good" else junk)
            init.append([s.id, x, y, z, kind])
        ops = []
        n_ops = RNG.randint(8, 40) if scen % 6 else 60
        for _ in range(n_ops):
            r = RNG.random()
            inflight = list(fetcher._inflight)
            if r < 0.62 or (not inflight and r < 0.95):
                s = RNG.choice(sources)
                x, y = RNG.randint(0, 5), RNG.randint(0, 5)
                z = RNG.choice((5, 5, 17, 18, 19, 20, -1, 0))
                n0, q0, i0 = len(started), len(fetcher._queue), len(fetcher._inflight)
                img = fetcher.request(s, x, y, z)
                ops.append({"op": "request", "src": s.id, "x": x, "y": y, "z": z, "image": img is not None,
                            "started": started[n0:], "queued": len(fetcher._queue) - q0})
            elif r < 0.95:
                key = RNG.choice(inflight)
                kind = RNG.choice(("ok", "ok", "garbage", "empty", "error"))
                reply = {"ok": _Reply(None, good), "garbage": _Reply(None, junk),
                         "empty": _Reply(None, b""), "error": _Reply("Host not found")}[kind]
                n0, r0, f0 = len(started), len(ready), len(failed)
                fetcher._on_finished(key[0], key[1], key[2], key[3], reply)
                ops.append({"op": "finish", "key": list(key), "kind": kind, "started": started[n0:],
                            "ready": len(ready) - r0, "failed": failed[f0:]})
            else:
                fetcher.cancel_all()
                ops.append({"op": "cancel"})
            ops[-1]["inflight"] = [list(k) for k in fetcher._inflight]
            ops[-1]["queue"] = [[s.id, x, y, z] for s, x, y, z in fetcher._queue]
        o.add(fn="scenario", init=init, ops=ops)
        shutil.rmtree(tmp, ignore_errors=True)
    for scen_src in sources:
        o.add(fn="max_zoom", src=scen_src.id, max_zoom=scen_src.max_zoom)
    o.add(fn="user_agent", out=TF._USER_AGENT.decode())
    o.close()


# ---------------------------------------------------------------------------
# terrain
# ---------------------------------------------------------------------------
def terrain_rec(t) -> dict:
    if t is None:
        return None
    vb = b"".join(struct.pack(">fff", v.x(), v.y(), v.z()) for v in t.vertices)
    ub = b"".join(struct.pack(">dd", u, v) for u, v in t.uvs)
    tb = b"".join(struct.pack(">III", *tri) for tri in t.triangles)
    rec = {"nx": t.nx, "ny": t.ny, "tile_range": list(t.tile_range), "nverts": len(t.vertices),
           "ntris": len(t.triangles), "hash": fnv64(vb + ub + tb)}
    if len(t.vertices) <= 700:
        rec["vertices"] = [vec(v) for v in t.vertices]
        rec["uvs"] = [[d(u), d(v)] for u, v in t.uvs]
        rec["triangles"] = [list(tri) for tri in t.triangles]
    return rec


def make_terrain() -> None:
    o = Out("terrain.jsonl")
    cases = []
    for dt in DATUMS:
        cases.append((dt, (-500.0, -500.0, 500.0, 500.0), 100.0, 180, 15, 13, ()))
        cases.append((dt, (-300.0, -300.0, 300.0, 300.0), 40.0, 180, 15, 13, ()))
    for dt in DATUMS[:4]:
        cases.append((dt, (-100.0, -5000.0, 100.0, 5000.0), 100.0, 180, 15, 12, ()))
        cases.append((dt, (-100.0, -5000.0, 100.0, 5000.0), 100.0, 20, 14, 12, ()))
        cases.append((dt, (200.0, 300.0, 460.0, 380.0), 7.5, 180, 16, 14, ()))
        cases.append((dt, (-500.0, -500.0, 500.0, 500.0), 100.0, 180, 15, 13, (0,)))   # partial coverage
        cases.append((dt, (-500.0, -500.0, 500.0, 500.0), 100.0, 180, 15, 13, tuple(range(50))))  # none
        cases.append((dt, (10.0, 10.0, 10.0, 90.0), 10.0, 180, 15, 13, ()))            # zero width
        cases.append((dt, (10.0, 10.0, 5.0, 90.0), 10.0, 180, 15, 13, ()))             # inverted
        cases.append((dt, (-1.0, -1.0, 1.0, 1.0), 40.0, 180, 15, 13, ()))              # tiny: 2x2
        cases.append((dt, (-3000.0, -3000.0, 3000.0, 3000.0), 25.0, 60, 15, 12, ()))
    cases.append((DATUMS[0], (-2000.0, -2000.0, 2000.0, 2000.0), 11.1, 180, 15, 12, ()))   # 180x180 (hash)
    for k, (dt, bbox, cell, mg, zoom, szoom, drop) in enumerate(cases):
        dat = mkdatum(dt)
        ll = TR._bbox_ll(dat, bbox)
        sampler, specs = make_sampler(dat, szoom, ll, drop)
        ground = sampler.elevation_at(dat.lat, dat.lon)
        ground = ground if ground is not None else 1234.5
        t = TR.build_terrain(dat, sampler, ground, bbox, cell, mg, zoom)
        rec = {"fn": "terrain", "datum": drec(dt), "bbox": [d(v) for v in bbox], "cell": d(cell),
               "max_grid": mg, "zoom": zoom, "sampler_zoom": szoom, "tiles": specs, "ground": d(ground),
               "ll": [d(v) for v in ll], "out": terrain_rec(t)}
        if t is not None:
            hq = []
            minx, miny, maxx, maxy = bbox
            for i in range(40):
                if i < 28:
                    x, y = RNG.uniform(minx, maxx), RNG.uniform(miny, maxy)
                elif i < 34:
                    x, y = RNG.choice((minx, maxx)), RNG.choice((miny, maxy))
                else:
                    x, y = RNG.uniform(minx - 50, maxx + 50), RNG.uniform(miny - 50, maxy + 50)
                hq.append([d(x), d(y), dd(t.height_at(x, y))])
            rec["height_at"] = hq
            lo, hi = t.bounds()
            rec["bounds"] = [vec(lo), vec(hi)]
            # mosaic
            tx0, ty0, ntx, nty, z = t.tile_range
            keys = []
            for dx in range(-1, ntx + 1):
                for dy in range(-1, nty + 1):
                    if RNG.random() < 0.6:
                        keys.append((tx0 + dx, ty0 + dy, z if RNG.random() < 0.85 else z + 1))
            RNG.shuffle(keys)
            if k % 5 == 0:
                keys = []
            images = {}
            for i, key in enumerate(keys):
                img = QImage(256, 256, QImage.Format.Format_RGB32)
                img.fill(0xFF000000 | (10 + i * 3) << 16 | 0x2000 | (i & 0xFF))
                images[key] = img
            m = TR.build_mosaic(t, images)
            placed = []
            if m is not None:
                for i, key in enumerate(keys):
                    x, y, zz = key
                    if zz == z and tx0 <= x < tx0 + ntx and ty0 <= y < ty0 + nty:
                        px = m.pixel((x - tx0) * 256 + 5, (y - ty0) * 256 + 5) & 0xFFFFFFFF
                        if px == (0xFF000000 | (10 + i * 3) << 16 | 0x2000 | (i & 0xFF)):
                            placed.append(list(key))
            rec["mosaic"] = {"keys": [list(k) for k in keys],
                             "out": None if m is None else {"w": m.width(), "h": m.height(),
                                                            "corner": m.pixel(m.width() - 1, m.height() - 1) & 0xFFFFFFFF,
                                                            "placed": placed}}
        o.add(**rec)
    o.close()


# ---------------------------------------------------------------------------
# surface + geopath
# ---------------------------------------------------------------------------
def V(x, y, z=0.0):
    return QVector3D(f32(x), f32(y), f32(z))


def tris_rec(tris) -> dict:
    if tris is None:
        return None
    b = b"".join(struct.pack(">fff", v.x(), v.y(), v.z()) for t in tris for v in t)
    rec = {"n": len(tris), "hash": fnv64(b)}
    if len(tris) <= 60:
        rec["tris"] = [[vec(v) for v in t] for t in tris]
    return rec


def poly_cases():
    cases = []
    cases.append([(0, 0), (100, 0), (100, 100), (0, 100)])
    cases.append([(0, 0), (80, 0), (80, 30), (0, 30)])
    cases.append([(0, 0), (60, 0), (60, 20), (20, 20), (20, 60), (0, 60)])          # L
    cases.append([(0, 0), (40, 0), (20, 35)])
    cases.append([(-30, -30), (30, -30), (45, 0), (30, 30), (-30, 30)])
    cases.append([(0, 0), (6, 0), (6, 5), (0, 5)])                                   # depth 0
    cases.append([(0, 0), (600, 0), (600, 600), (0, 600)])                           # depth cap
    cases.append([(0, 0), (30, 0), (60, 0)])                                         # collinear
    cases.append([(0, 0), (100, 0)])                                                 # two points
    cases.append([(0, 0), (100, 0), (100, 0), (100, 100), (0, 100)])                 # duplicate node
    cases.append([(-0.0, -0.0), (50, -0.0), (50, 50), (-0.0, 50)])
    cases.append([(0, 0), (100, 0), (100, 100), (50, 20), (0, 100)])                 # concave notch
    cases.append([(0.0004, 0.0004), (10.0003, 0.0004), (10.0003, 10.0004)])          # rounding keys
    # Nodes a few millimetres apart: their lifted points share a rounded key
    # at 2 digits but not always at 3, so the cache's `round(., 3)` shows.
    for dx, dy in ((0.0004, 0.0004), (0.004, 0.004), (0.0006, 0.0006), (0.009, 0.0049), (0.0015, 0.0025)):
        cases.append([(0, 0), (50, 0), (50 + dx, dy), (50, 50), (0, 50)])
    for _ in range(14):
        n = RNG.randint(3, 9)
        cx, cy = RNG.uniform(-200, 200), RNG.uniform(-200, 200)
        r = 10 ** RNG.uniform(0.7, 2.5)
        angs = sorted(RNG.uniform(0, 2 * math.pi) for _ in range(n))
        cases.append([(cx + r * RNG.uniform(0.4, 1) * math.cos(a), cy + r * RNG.uniform(0.4, 1) * math.sin(a)) for a in angs])
    return cases


def make_surface() -> None:
    o = Out("surface.jsonl")
    # fit_plane
    for _ in range(300):
        n = RNG.randint(0, 12)
        pts = [V(RNG.uniform(-500, 500), RNG.uniform(-500, 500)) for _ in range(n)]
        mode = RNG.random()
        if mode < 0.15 and n:
            pts = [V(1.5 * i, 3.0 * i) for i in range(n)]              # collinear
        elif mode < 0.25 and n:
            pts = [V(7.0, 7.0)] * n
        zs = [RNG.uniform(-50, 3000) for _ in range(n)]
        a, b, c = S._fit_plane(pts, zs)
        o.add(fn="fit_plane", pts=[vec(p) for p in pts], zs=[d(z) for z in zs], out=[d(a), d(b), d(c)])
    # subdivide
    for _ in range(40):
        tri = tuple((RNG.uniform(-100, 100), RNG.uniform(-100, 100)) for _ in range(3))
        depth = RNG.randint(0, 3)
        out = []
        S._subdivide(tri, depth, out)
        o.add(fn="subdivide", tri=[[d(p[0]), d(p[1])] for p in tri], depth=depth,
              out=[[[d(p[0]), d(p[1])] for p in t] for t in out])
    cases = poly_cases()
    for dt in DATUMS[:3]:
        dat = mkdatum(dt)
        for pi, poly in enumerate(cases):
            if pi >= 13 and dt is not DATUMS[0] and pi % 3:
                continue
            if dt is not DATUMS[0] and pi % 2:
                continue
            for mode in ("flat", "draped", None, "bogus"):
                path = GP.GeoPath([V(x, y) for x, y in poly], closed=True, surface=mode)
                lp = [(float(p.x()), float(p.y())) for p in path.points]
                xs = [p[0] for p in lp] or [0.0]
                ys = [p[1] for p in lp] or [0.0]
                bb = (min(xs) - 5, min(ys) - 5, max(xs) + 5, max(ys) + 5)
                ll = TR._bbox_ll(dat, bb)
                for drop in ((), (0,)):
                    if drop and (mode not in ("draped",) or pi % 4):
                        continue
                    sampler, specs = make_sampler(dat, 15, ll, drop)
                    ground = S.ground_reference(sampler, dat)
                    rec = {"fn": "surface", "datum": drec(dt), "poly": [[f(x), f(y)] for x, y in
                                                                        [(p.x(), p.y()) for p in path.points]],
                           "mode": mode, "tiles": specs, "sampler_zoom": 15,
                           "ground": dd(ground)}
                    rec["build_surface"] = tris_rec(S.build_surface(path, sampler, dat))
                    if ground is not None:
                        rec["flat"] = tris_rec(S.build_flat_surface(path, sampler, ground))
                        rec["draped"] = tris_rec(S.build_draped_surface(path, sampler, ground))
                    o.add(**rec)
    o.close()

    # geopath
    o = Out("geopath.jsonl")
    for _ in range(90):
        n = RNG.randint(0, 9)
        pts = [V(RNG.uniform(-300, 300), RNG.uniform(-300, 300), RNG.uniform(-5, 50)) for _ in range(n)]
        if RNG.random() < 0.1 and n:
            pts = [pts[0]] * n
        closed = RNG.random() < 0.5
        name = RNG.choice(("", "Road A", "Ñandú — lote", "x"))
        surface = RNG.choice((None, "flat", "draped", ""))
        gp = GP.GeoPath(pts, closed=closed, name=name, surface=surface)
        segs = list(gp.segments())
        rec = {"fn": "path", "pts": [vec(p) for p in pts], "closed": closed, "name": name, "surface": surface,
               "segments": [[vec(a), vec(b)] for a, b in segs], "length": d(gp.length()),
               "perimeter": d(gp.perimeter()), "area": d(gp.area()),
               "profile_points": [vec(p) for p in gp.profile_points()], "to_dict": json.dumps(gp.to_dict())}
        # surface area over a random triangle set
        ntri = RNG.randint(0, 6)
        tris = [(V(RNG.uniform(-50, 50), RNG.uniform(-50, 50), RNG.uniform(-5, 5)),
                 V(RNG.uniform(-50, 50), RNG.uniform(-50, 50), RNG.uniform(-5, 5)),
                 V(RNG.uniform(-50, 50), RNG.uniform(-50, 50), RNG.uniform(-5, 5))) for _ in range(ntri)]
        gp._surface_tris = tris if RNG.random() < 0.85 else None
        rec["tris"] = None if gp._surface_tris is None else [[vec(v) for v in t] for t in tris]
        sa = gp.surface_area()
        rec["surface_area"] = dd(sa)
        rt = GP.GeoPath.from_dict(json.loads(rec["to_dict"]))
        rec["roundtrip"] = json.dumps(rt.to_dict())
        o.add(**rec)
    from_dicts = [
        '{}', '{"points":[[1,2,3],[4,5,6]]}', '{"points":[[1,2,3],[4,5,6.5],[7,8,9]],"closed":true,"name":"N","surface":"flat"}',
        '{"points":[[1,2]]}', '{"points":[[1,2,3,4]]}', '{"points":[[]]}', '{"points":[["a",2,3]]}',
        '{"points":[[1e40,2,3]]}', '{"points":[[true,false,3]]}', '{"points":[[0.1,0.2,0.3]]}', '{"points":5}',
        '{"points":null}', '{"points":"abc"}', '{"points":{}}', '{"points":[5]}', '{"points":[null]}',
        '{"points":[[1,2,3]],"closed":0}', '{"points":[[1,2,3]],"closed":"yes"}', '{"points":[[1,2,3]],"closed":[]}',
        '{"name":""}', '{"name":null}', '{"name":0}', '{"name":5}', '{"surface":""}', '{"surface":"draped"}',
        '{"surface":null}', '{"surface":5}', '{"surface":[]}', '[]', 'null', '5', '"s"',
        '{"points":[[1e-50,2,3]]}', '{"points":[[16777217,2,3]]}', '{"points":[[1,2,3]],"junk":1}',
    ]
    nm = {'{"name":5}', '{"surface":5}'}
    for text in from_dicts:
        ok, v = run(lambda: GP.GeoPath.from_dict(json.loads(text)))
        rec = {"fn": "path_from_dict", "data": text}
        if text in nm:
            rec["nm"] = True
        if ok:
            rec["to_dict"] = json.dumps(v.to_dict())
            rec["points"] = [vec(p) for p in v.points]
            rec["closed"] = v.closed
        else:
            rec["error"] = v
        o.add(**rec)
    o.close()


# ---------------------------------------------------------------------------
# points
# ---------------------------------------------------------------------------
CSV_TEXTS = [
    "P1,100.5,200.25,3.0,desc\nP2,101,201,4",
    "P,N,E,Z,D\n1,8600000,300000,150,corner\n2,8600010,300010,151,corner",
    "1;8600000,5;300000,25;150,5;stake, with comma",
    "1\t8600000\t300000\t150\tfoo bar",
    "1 8600000 300000 150 desc words here",
    "# comment\n\n  \nA,1,2,3\n#B,4,5,6\nC,7,8,9",
    "A,1,2,3\r\nB,4,5,6\rC,7,8,9\n",
    "\"A,1\",100,200,3,\"say \"\"hi\"\"\"",
    "\"A\",100,200,3,\"unterminated",
    "A,100,200,3,",
    "A,100,200,3,x,y,z",
    "A,100,200,3, spaced , out ",
    "a;b;c;d",
    "x,y",
    "A,1_000,2_0.5,3e2",
    "A,nan,inf,-inf",
    "A,+5,-5,.5",
    "A,5.,1e,2",
    "A,1__0,2,3",
    "A,_1,2,3",
    "A,0x10,2,3",
    "A, 100 ,200,3",
    "A,100,200,3,ñandú",
    "A\x0bB,1,2,3",
    "A,1,2,3\x0cB,4,5,6",
    "A,1,2,3 B,4,5,6 C,7,8,9\x85D,1,1,1",
    "A,1,2,3\x1cB,4,5,6",
    "",
    "   ",
    "# only a comment",
    "header,only,no,numbers",
    "A,1,2",
    ",1,2,3",
    "A,,2,3",
    "A,1,2,3,\"q,r\"",
    "A;1,5;2,5;3;\"x;y\"",
    "\"A\"x,1,2,3",
    "A\"B,1,2,3",
    "﻿A,1,2,3",
    "A,1e400,2,3",
    "A,1,2,3\n\n\nB,4,5,6\n",
    "  A , 1 , 2 , 3 , d  ",
    "A,1,2,3,\"\"",
    "A,1,2,3,\"\"\"\"",
    "A;1;2;3;\"a\"b",
    "A,1 2,3,4",
    "A,١٢٣,2,3",
]


def make_points() -> None:
    o = Out("points.jsonl")
    texts = list(CSV_TEXTS)
    for _ in range(80):
        lines = []
        for k in range(RNG.randint(1, 6)):
            delim = RNG.choice((",", ";", "\t", " "))
            dec = "," if delim in (";", "\t") and RNG.random() < 0.5 else "."
            nums = [f"{RNG.uniform(7e6, 9e6):.{RNG.randint(0, 3)}f}", f"{RNG.uniform(1e5, 9e5):.{RNG.randint(0, 3)}f}",
                    f"{RNG.uniform(-50, 4000):.{RNG.randint(0, 3)}f}"]
            nums = [x.replace(".", dec) for x in nums]
            cells = [RNG.choice(("P1", "p-2", "Ñ", f"S{k}"))] + nums
            if RNG.random() < 0.4:
                cells.append(RNG.choice(("tree", "pole", "x y", "a;b", "q")))
            if RNG.random() < 0.1:
                cells = ["P", "N", "E", "Z", "D"]
            lines.append(delim.join(cells))
        texts.append(RNG.choice(("\n", "\r\n")).join(lines))
    for text in texts:
        ok, v = run(PT.parse_points_csv, text)
        rec = {"fn": "csv", "text": text}
        if ok:
            rec["rows"] = [{"name": r["name"], "north": d(r["north"]), "east": d(r["east"]), "z": d(r["z"]),
                            "desc": r["desc"]} for r in v]
        else:
            rec["error"] = v
        o.add(**rec)
    # points_from_rows + datum_for_rows
    rows = [{"name": "A", "north": 8667000.5, "east": 288000.25, "z": 3400.0, "desc": "x"},
            {"name": "B", "north": 8667100.0, "east": 288100.75, "z": 3410.5, "desc": ""},
            {"name": "C", "north": 8666900.125, "east": 287900.0, "z": 3390.0, "desc": "end,of"}]
    for dt in DATUMS:
        dat = mkdatum(dt)
        pts = PT.points_from_rows(rows, dat)
        o.add(fn="points_from_rows", datum=drec(dt),
              rows=[{"name": r["name"], "north": d(r["north"]), "east": d(r["east"]), "z": d(r["z"]), "desc": r["desc"]}
                    for r in rows],
              out=[{"p": vec(p.position), "name": p.name, "desc": p.desc, "to_dict": json.dumps(p.to_dict())} for p in pts])
    for zone, northern, e, n, z in ((18, False, 288000.0, 8667000.0, 3400.0), (19, False, 288000.0, 8667000.0, 3400.0),
                                    (17, False, 833000.0, 8667000.0, 10.0), (31, True, 500000.0, 5000000.0, 0.0),
                                    (18, True, 288000.0, 100.0, -3.0), (1, True, 166000.0, 6000000.0, 5.0),
                                    (60, True, 833000.0, 6000000.0, 5.0), (2, True, 166000.0, 6000000.0, 5.0),
                                    (59, False, 833000.0, 2000000.0, 5.0), (33, True, 100000.0, 4000000.0, 1.0)):
        r0 = [{"name": "A", "north": n, "east": e, "z": z, "desc": ""}]
        dat = PT.datum_for_rows(r0, zone, northern)
        o.add(fn="datum_for_rows", zone=zone, northern=northern, row={"north": d(n), "east": d(e), "z": d(z)},
              out={"lat": d(dat.lat), "lon": d(dat.lon), "alt": d(dat.alt), "north": d(dat.north), "zone": dat.zone,
                   "northern": dat.northern, "east0": d(dat._east0), "north0": d(dat._north0)})
    # GeoPoint dicts
    for text in ('{}', '{"position":[1,2,3]}', '{"position":[1.5,2.5,3.5],"name":"A","desc":"d"}', '{"position":[1,2]}',
                 '{"position":null}', '{"position":5}', '{"position":[]}', '{"name":""}', '{"name":null}', '{"desc":0}',
                 '{"name":5}', '[]', 'null', '{"position":["x",2,3]}', '{"position":[1e40,0,0]}'):
        ok, v = run(lambda: PT.GeoPoint.from_dict(json.loads(text)))
        rec = {"fn": "point_from_dict", "data": text}
        if text == '{"name":5}':
            rec["nm"] = True
        if ok:
            rec["to_dict"] = json.dumps(v.to_dict())
            rec["p"] = vec(v.position)
        else:
            rec["error"] = v
        o.add(**rec)
    for _ in range(40):
        p = PT.GeoPoint(V(RNG.uniform(-500, 500), RNG.uniform(-500, 500), RNG.uniform(-5, 50)),
                        RNG.choice(("", "P1", "Ñ")), RNG.choice(("", "a desc", "—")))
        o.add(fn="point", p=vec(p.position), name=p.name, desc=p.desc, to_dict=json.dumps(p.to_dict()))
    o.close()


# ---------------------------------------------------------------------------
# geocode
# ---------------------------------------------------------------------------
PLACES = [
    '[{"display_name":"Cusco, Peru","lat":"-13.5319","lon":"-71.9675"}]',
    '[{"display_name":"A","lat":1,"lon":2},{"display_name":"B","lat":"x","lon":2},{"lat":3,"lon":4},'
    '{"display_name":"D","lon":5},{"display_name":"E","lat":null,"lon":1},{"display_name":"F","lat":[1],"lon":1}]',
    '[]', '{}', '"x"', 'null', '5', '', 'not json', '[{"display_name":"A","lat":1e400,"lon":1}]',
    '[{"display_name":"A","lat":100000000000000000000000000000000000000000000000000000000000000000000000000000000000000'
    '0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'
    '00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'
    '000000000000000000000000000000000000000000000000000000000000000000000000000000000,"lon":1}]',
    '[1,2]', '["a"]', '[null]', '[[1,2]]', '[{"display_name":"A","lat":1,"lon":2}, 5]',
    '[{"display_name":5,"lat":1,"lon":2}]', '[{"display_name":null,"lat":1,"lon":2}]',
    '[{"display_name":["x"],"lat":1,"lon":2}]', '[{"display_name":{"a":"b"},"lat":1,"lon":2}]',
    '[{"display_name":true,"lat":1,"lon":2}]', '[{"display_name":1.5,"lat":1,"lon":2}]',
    '[{"display_name":"\\u00d1and\\u00fa \\ud83d\\ude00","lat":" 1.5 ","lon":"2_0"}]',
    '[{"display_name":"A","lat":true,"lon":false}]', '[{"display_name":"A","lat":"nan","lon":"inf"}]',
    '[{"display_name":"A","lat":NaN,"lon":Infinity}]', '[{"display_name":"A","lat":"","lon":"1"}]',
    '  [ {"display_name" : "A" , "lat":1,"lon":2} ]  ', '﻿[]',
]
IPS = [
    '{"latitude":1.5,"longitude":2.5,"city":"Lima","region":"Lima","country_name":"Peru"}',
    '{"latitude":1.5,"longitude":2.5}', '{"latitude":"1.5","longitude":"2.5","city":"","region":null,"country_name":"Peru"}',
    '{"latitude":null,"longitude":2}', '{"latitude":1}', '{"longitude":1}', '{}', '[]', 'null', '5', '"x"', 'bad',
    '{"latitude":"x","longitude":2}', '{"latitude":[1],"longitude":2}', '{"latitude":1,"longitude":{}}',
    '{"latitude":0,"longitude":0,"city":0,"region":1,"country_name":[1,"a"]}',
    '{"latitude":0,"longitude":0,"city":{"a":1},"region":true,"country_name":2.5}',
    '{"latitude":0,"longitude":0,"city":"it\'s","region":"say \\"hi\\"","country_name":"a\\nb"}',
    '{"latitude":1e400,"longitude":0}',
    '{"latitude":100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'
    '00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'
    '0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'
    '00000000000000000000000000000000000000000000000000,"longitude":0}',
    '{"latitude":true,"longitude":false}',
]


def utf_variants(text: str) -> list:
    out = [text.encode("utf-8", "surrogatepass") if False else text.encode("utf-8")]
    out.append(b"\xef\xbb\xbf" + text.encode("utf-8"))
    out.append(text.encode("utf-16"))
    out.append(text.encode("utf-16-le"))
    out.append(text.encode("utf-16-be"))
    out.append(text.encode("utf-32"))
    out.append(text.encode("utf-32-le"))
    out.append(text.encode("utf-32-be"))
    return out


def make_geocode() -> None:
    o = Out("geocode.jsonl")
    queries = ["Cusco", "Santiago, Chile", "a b/c", "Ñandú", "café — 日本", "100% sure?&q=x#y",
               "x~y_z-w.v", "   ", "", "a+b", "\U0001f600", "'\"<>[]{}|\\^`", "a\tb\nc"]
    for q in queries:
        for lim in (8, 1, 0, 50, -3):
            o.add(fn="url", q=q, limit=lim, out=GC.geocode_url(q, lim))
        o.add(fn="url_default", q=q, out=GC.geocode_url(q))
    # Geocoder.search planning, through the real class with its GET replaced
    for q in queries + ["  padded  ", " nb ", "\x1cweird\x1c", "\t\n "]:
        g = GC.Geocoder()
        calls, results = [], []
        g._get = lambda url, ok, er, calls=calls: calls.append(url)
        g.resultsReady.connect(lambda lst, results=results: results.append(list(lst)))
        g.search(q)
        o.add(fn="plan_search", q=q, calls=calls, empty=len(results))
    for text in PLACES:
        for data in (utf_variants(text) if len(text) < 900 and RNG.random() < 0.5 else [text.encode("utf-8")]):
            ok, v = run(GC.parse_places, data)
            rec = {"fn": "places", "data": data.hex()}
            if ok:
                rec["out"] = [{"name": str(p.name), "lat": d(p.lat), "lon": d(p.lon)} for p in v]
            else:
                rec["error"] = v
            o.add(**rec)
    for raw in (b"\xff\xfe\xfd", b"[", b"\x00", b"\x00\x00", b"[\xc3(]", b"\xef\xbb\xbf", b"\xef\xbb\xbf[]", b"", b"  ",
                b"\x00\x00\x00", b"\xff\xfe", b"\xfe\xff\x00"):
        ok, v = run(GC.parse_places, raw)
        o.add(fn="places", data=raw.hex(), **({"out": [{"name": str(p.name), "lat": d(p.lat), "lon": d(p.lon)} for p in v]}
                                                if ok else {"error": v}))
        ok, v = run(GC.parse_ip, raw)
        o.add(fn="ip", data=raw.hex(), **({"out": None if v is None else [d(v[0]), d(v[1]), v[2]]} if ok else {"error": v}))
    for text in IPS:
        for data in (utf_variants(text) if len(text) < 600 and RNG.random() < 0.4 else [text.encode("utf-8")]):
            ok, v = run(GC.parse_ip, data)
            rec = {"fn": "ip", "data": data.hex()}
            if ok:
                rec["out"] = None if v is None else [d(v[0]), d(v[1]), v[2]]
            else:
                rec["error"] = v
            o.add(**rec)
    o.add(fn="consts", nominatim=GC.NOMINATIM, ipapi=GC.IPAPI, ua=GC._USER_AGENT.decode())
    o.close()


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------
def prof_rec(p) -> dict:
    return {"length": d(p.length), "complete": p.complete,
            "samples": [[d(s.station), d(s.x), d(s.y), dd(s.elevation)] for s in p.samples],
            "min": dd(p.min_elevation()), "max": dd(p.max_elevation()), "gain": d(p.total_gain()),
            "slopes": [[d(a), d(b)] for a, b in p.slopes()],
            "elevations": [d(e) for e in p.elevations()]}


def make_profile() -> None:
    o = Out("profile.jsonl")
    # pure helpers
    for _ in range(200):
        n = RNG.randint(0, 7)
        pts = [V(RNG.uniform(-500, 500), RNG.uniform(-500, 500)) for _ in range(n)]
        if RNG.random() < 0.1 and n > 1:
            pts[1] = pts[0]
        total = PF.polyline_length(pts)
        qs = []
        for _ in range(8):
            s = RNG.choice((-1.0, 0.0, total, total + 5, RNG.uniform(0, max(total, 1.0)), RNG.uniform(0, max(total, 1.0))))
            x, y = PF.point_at_station(pts, s)
            qs.append([d(s), d(x), d(y)])
        o.add(fn="helpers", pts=[vec(p) for p in pts], length=d(total), at=qs)
    # profiles over real samplers
    for k, dt in enumerate(DATUMS[:6]):
        dat = mkdatum(dt)
        for j in range(4):
            n = RNG.randint(2, 6)
            span = 10 ** RNG.uniform(1.5, 3.8)
            pts = [V(RNG.uniform(-span, span), RNG.uniform(-span, span)) for _ in range(n)]
            if RNG.random() < 0.1:
                pts = [pts[0], pts[0]]
            lats, lons = [], []
            for p in pts:
                la, lo, _ = dat.local_to_geodetic(p)
                lats.append(la)
                lons.append(lo)
            ll = (min(lats), min(lons), max(lats), max(lons))
            drop = (0,) if RNG.random() < 0.3 else ()
            sampler, specs = make_sampler(dat, 13, ll, drop, margin=0.0)
            spacing = RNG.choice((None, None, PF.polyline_length(pts) / 37.0, 1e9, -2.0, 100.0 if span < 300 else None))
            ok, prof = run(PF.sample_profile, pts, sampler, spacing)
            rec = {"fn": "sample", "datum": drec(dt), "pts": [vec(p) for p in pts], "spacing": dd(spacing),
                   "tiles": specs, "sampler_zoom": 13}
            if ok:
                rec["out"] = prof_rec(prof)
                if j == 0:
                    for vert, alt in (("odm", None), ("dem", dat.alt), ("bogus", None)):
                        rec.setdefault("csv", []).append(
                            {"vertical": vert, "alt": dd(alt), "out": PF.profile_to_csv(prof, vert, alt)})
            else:
                rec["error"] = prof
            o.add(**rec)
    # spacing 0, single point
    dat = mkdatum(DATUMS[0])
    sampler, specs = make_sampler(dat, 13, (-12.05, -77.05, -12.04, -77.04))
    for pts, spacing in (([V(0, 0), V(100, 0)], 0.0), ([V(0, 0)], None), ([], None), ([V(1, 1), V(1, 1)], None)):
        ok, prof = run(PF.sample_profile, pts, sampler, spacing)
        rec = {"fn": "sample", "datum": drec(DATUMS[0]), "pts": [vec(p) for p in pts], "spacing": dd(spacing),
               "tiles": specs, "sampler_zoom": 13}
        if ok:
            rec["out"] = prof_rec(prof)
        else:
            rec["error"] = prof
        o.add(**rec)
    # hand-made profiles: gaps, ties, NaN-free float noise
    for _ in range(60):
        n = RNG.randint(0, 10)
        samples, st = [], 0.0
        for i in range(n):
            st += RNG.choice((0.0, 1e-10, 0.5, 3.25, 10.0))
            e = None if RNG.random() < 0.25 else RNG.choice((RNG.uniform(-10, 3000), float(RNG.randint(0, 5)), 0.0625 * RNG.randint(-40, 40)))
            samples.append(PF.ProfileSample(st, RNG.uniform(-9, 9), RNG.uniform(-9, 9), e))
        prof = PF.Profile(samples, st, all(s.elevation is not None for s in samples))
        csv_variants = [{"vertical": v, "alt": dd(a), "out": PF.profile_to_csv(prof, v, a)}
                        for v in ("local", "dem") for a in (None, 154.3, -0.0004, 0.0625)]
        rec = prof_rec(prof)
        rec.update(fn="manual", csv=csv_variants)
        o.add(**rec)
    # number formatting: ties, signs, specials
    vals = [0.0, -0.0, 0.0625, 0.1875, 0.3125, 0.4375, 0.0005, 0.0015, 0.0025, 1.0005, 2.5, -0.0004, -0.0006,
            1e15, 1.2345678e-5, 123456789.0625, float("nan"), float("inf"), float("-inf"), 5e-324, 1e300]
    vals += [k / 16 for k in range(-40, 40)] + [k / 2048 for k in range(0, 200, 7)]
    vals += [RNG.uniform(-1e4, 1e4) for _ in range(150)]
    samples = [PF.ProfileSample(v, v, v, v) for v in vals]
    prof = PF.Profile(samples, 0.0, True)
    o.add(fn="format", vals=[d(v) for v in vals], out=PF.profile_to_csv(prof, "local", None))
    for vert in ("odm", "dem", "local", "x"):
        o.add(fn="note", vertical=vert, out=PF.VERTICAL_NOTES.get(vert, PF.VERTICAL_NOTES["local"]))
    o.close()


# ---------------------------------------------------------------------------
# geoimport
# ---------------------------------------------------------------------------
def feats_rec(fs) -> list:
    return [{"points": [[d(float(a)), d(float(b))] for a, b in x.points], "closed": x.closed, "name": x.name} for x in fs]


def kml(body: str, ns: str = ' xmlns="http://www.opengis.net/kml/2.2"') -> str:
    return f'<?xml version="1.0" encoding="UTF-8"?><kml{ns}><Document>{body}</Document></kml>'


def pm(name, geom, extra="") -> str:
    nm = f"<name>{name}</name>" if name is not None else ""
    return f"<Placemark>{nm}{extra}{geom}</Placemark>"


def ls(coords) -> str:
    return f"<LineString><coordinates>{coords}</coordinates></LineString>"


def poly(coords, inner=None) -> str:
    s = f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{coords}</coordinates></LinearRing></outerBoundaryIs>"
    if inner:
        s += f"<innerBoundaryIs><LinearRing><coordinates>{inner}</coordinates></LinearRing></innerBoundaryIs>"
    return s + "</Polygon>"


KML_TEXTS = [
    kml(pm("Road", ls("-77.0,-12.0,5 -77.1,-12.1,6 -77.2,-12.2"))),
    kml(pm("  Lot Ñ  ", poly("-77,-12,0 -77.1,-12,0 -77.1,-12.1,0 -77,-12,0"))),
    kml(pm("P2", poly("-77,-12 -77.1,-12 -77.1,-12.1 -77,-12.1"))),
    kml(pm("two", poly("1,2 3,4"))),
    kml(pm("one", ls("1,2"))),
    kml(pm(None, ls("1,2 3,4"))),
    kml(pm("", ls("1,2 3,4"))),
    kml(pm("cdata", ls("<![CDATA[1,2 3,4]]>")).replace("<name>cdata</name>", "<name><![CDATA[ A&B ]]></name>")),
    kml(pm("A&amp;B &lt;x&gt; &#65;&#x42; &quot;q&quot; &apos;a&apos;", ls("1,2 3,4"))),
    kml('<Folder><name>f</name>' + pm("in", ls("1,2 3,4")) + '</Folder>'),
    kml(pm("multi", "<MultiGeometry>" + ls("1,2 3,4") + ls("5,6 7,8") + poly("0,0 1,0 1,1") + "</MultiGeometry>")),
    kml(pm("nested", "<Polygon><outerBoundaryIs><LinearRing><coordinates>0,0 1,0 1,1 0,0</coordinates></LinearRing></outerBoundaryIs>"
                     "<innerBoundaryIs><LinearRing><coordinates>0.1,0.1 0.2,0.1 0.2,0.2</coordinates></LinearRing></innerBoundaryIs></Polygon>")),
    kml(pm("bad numbers", ls("a,b 1,2 3,x 4,5 6 7,8,9,10 ,3 3,"))),
    kml(pm("ws", ls("\n  1,2 \t 3,4\r\n 5,6   7,8  "))),
    kml(pm("nan", ls("nan,1 1,inf 2,3 1_0,2"))),
    kml(pm("two names", ls("1,2 3,4"), extra="<name>second</name>")),
    kml(pm("empty second", ls("1,2 3,4"), extra="<name></name>")),
    kml(pm("sub", "<Style><name>style name</name></Style>" + ls("1,2 3,4"))),
    kml(pm("textcoords", "<LineString><coordinates>x<b>y</b>1,2 3,4</coordinates></LineString>")),
    kml(pm("nocoords", "<LineString></LineString>")),
    kml(pm("emptycoords", "<LineString><coordinates></coordinates></LineString>")),
    kml(pm("closed-dup", poly("0,0 5,0 5,5 0,5 0,0"))),
    kml(pm("near-dup", poly("0,0 5,0 5,5 0,5 0.0000000001,0"))),
    kml(pm("not-near-dup", poly("0,0 5,0 5,5 0,5 0.000001,0"))),
    kml(pm("tri-closed", poly("0,0 5,0 5,5 0,0"))),
    kml(pm("a", ls("1,2 3,4")) + pm("b", poly("0,0 1,0 1,1")) + pm("c", ls("9,9 8,8"))),
    kml(pm("PM-in-PM", pm("inner", ls("1,2 3,4")) + ls("5,6 7,8"))),
    '<kml><Placemark><name>nons</name><LineString><coordinates>1,2 3,4</coordinates></LineString></Placemark></kml>',
    '<k:kml xmlns:k="http://www.opengis.net/kml/2.2"><k:Placemark><k:name>pfx</k:name><k:LineString><k:coordinates>1,2 3,4</k:coordinates></k:LineString></k:Placemark></k:kml>',
    '<kml xmlns="a" xmlns:g="b"><Placemark g:id="1" id="2"><name>attrs</name><LineString g:x="1"><coordinates>1,2 3,4</coordinates></LineString></Placemark></kml>',
    "<?xml version='1.0'?>\n<!-- c --><kml><!-- inner --><Placemark><name>a<!--x-->b</name><LineString><coordinates>1,2 3,4</coordinates></LineString></Placemark><?pi data?></kml>\n<!-- tail -->",
    '<!DOCTYPE kml [<!ENTITY e "v1,2"><!ELEMENT kml ANY>]><kml><Placemark><name>&e;</name><LineString><coordinates>&e; 3,4</coordinates></LineString></Placemark></kml>',
    '﻿<kml><Placemark><LineString><coordinates>1,2 3,4</coordinates></LineString></Placemark></kml>',
    '<kml><Placemark><name>crlf\r\nx</name><LineString><coordinates>1,2\r\n3,4</coordinates></LineString></Placemark></kml>',
    # malformed -> []
    '', '   ', 'not xml', '<kml>', '<kml></kml', '<kml><Placemark></kml>', '<a/><b/>', '<a/>x', ' <?xml version="1.0"?><a/>',
    '<a b="1" b="2"/>', '<x:a/>', '<a:b:c xmlns:a="u"/>', '<a>&nbsp;</a>', '<a>&#0;</a>', '<a b=c/>', '<a b="<"/>', '<1a/>',
    '<a>\x01</a>', '<a>]]></a>', '<a xmlns:xml="u"/>', '<a xmlns:y=""/>', '<a xmlns="u" xmlns="v"/>', '<a><!-- -- --></a>',
    '<a><?xml x?></a>', '<a b="1"c="2"/>', '<a>￿</a>', '<?xml?><a/>', '<?xml version="1.0" bogus="1"?><a/>',
    '<a b="&foo;"/>', '<a>&#xD800;</a>', '<a xmlns:x="u" x:b="1" x:b="2"/>', '<x:a xmlns:x="u" x:q="1" y:r="2"/>',
    '<a></b>', '<a><b></a></b>', '<kml><Placemark><name>x</name><LineString><coordinates>1,2 3,4</coordinates></LineString></Placemark></kml>junk',
    '<!DOCTYPE a><a/>', '<?xml version="2.0"?><kml><Placemark><LineString><coordinates>1,2 3,4</coordinates></LineString></Placemark></kml>',
    '<?xml version="1.0" encoding="ISO-8859-1" standalone="yes"?><kml><Placemark><name>é</name><LineString><coordinates>1,2 3,4</coordinates></LineString></Placemark></kml>',
    '<kml><Placemark><LineString><coordinates>1,2 3,4</coordinates></LineString></Placemark></kml><!--c-->  <?p q?> ',
    '<a><![CDATA[x</a>', '<a><![CDATA[x]]></a><b/>',
]


GEOJSON_TEXTS = [
    '{"type":"LineString","coordinates":[[10,20],[30,40]]}',
    '{"type":"LineString","coordinates":[[10,20,5],[30,40,6],[1],[7,8]]}',
    '{"type":"LineString","coordinates":[[10,20]]}', '{"type":"LineString"}', '{"type":"LineString","coordinates":null}',
    '{"type":"LineString","coordinates":[]}', '{"type":"LineString","coordinates":5}',
    '{"type":"LineString","coordinates":[5,6]}', '{"type":"LineString","coordinates":[["a","b"],["c","d"]]}',
    '{"type":"LineString","coordinates":[[true,false],[1,2]]}', '{"type":"LineString","coordinates":"ab"}',
    '{"type":"LineString","coordinates":{"a":1}}',
    '{"type":"MultiLineString","coordinates":[[[1,2],[3,4]],[[5,6],[7,8],[9,10]],[[1,1]]]}',
    '{"type":"MultiLineString","coordinates":[5]}', '{"type":"MultiLineString"}', '{"type":"MultiLineString","coordinates":0}',
    '{"type":"Polygon","coordinates":[[[0,0],[10,0],[10,10],[0,10],[0,0]]]}',
    '{"type":"Polygon","coordinates":[[[0,0],[10,0],[10,10],[0,10]],[[1,1],[2,1],[2,2]]]}',
    '{"type":"Polygon","coordinates":[[[0,0],[10,0]]]}', '{"type":"Polygon","coordinates":[]}', '{"type":"Polygon","coordinates":[[]]}',
    '{"type":"Polygon","coordinates":5}', '{"type":"Polygon","coordinates":"abc"}', '{"type":"Polygon","coordinates":{"a":1}}',
    '{"type":"Polygon","coordinates":[5]}', '{"type":"Polygon"}',
    '{"type":"MultiPolygon","coordinates":[[[[0,0],[10,0],[10,10],[0,0]]],[[[5,5],[6,5],[6,6]]],[],[[[1,1]]]]}',
    '{"type":"MultiPolygon","coordinates":[5]}', '{"type":"MultiPolygon","coordinates":[[5]]}',
    '{"type":"GeometryCollection","geometries":[{"type":"LineString","coordinates":[[1,2],[3,4]]},'
    '{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1]]]},5,{"type":"Point","coordinates":[1,2]},'
    '{"type":"GeometryCollection","geometries":[{"type":"LineString","coordinates":[[9,9],[8,8]]}]}]}',
    '{"type":"GeometryCollection"}', '{"type":"GeometryCollection","geometries":5}', '{"type":"GeometryCollection","geometries":null}',
    '{"type":"Feature","properties":{"name":"Road"},"geometry":{"type":"LineString","coordinates":[[1,2],[3,4]]}}',
    '{"type":"Feature","properties":null,"geometry":{"type":"LineString","coordinates":[[1,2],[3,4]]}}',
    '{"type":"Feature","properties":{},"geometry":{"type":"LineString","coordinates":[[1,2],[3,4]]}}',
    '{"type":"Feature","geometry":{"type":"LineString","coordinates":[[1,2],[3,4]]}}',
    '{"type":"Feature","properties":{"name":5},"geometry":{"type":"LineString","coordinates":[[1,2],[3,4]]}}',
    '{"type":"Feature","properties":{"name":0},"geometry":{"type":"LineString","coordinates":[[1,2],[3,4]]}}',
    '{"type":"Feature","properties":{"name":1.5},"geometry":{"type":"LineString","coordinates":[[1,2],[3,4]]}}',
    '{"type":"Feature","properties":{"name":["a",1,null,true]},"geometry":{"type":"LineString","coordinates":[[1,2],[3,4]]}}',
    '{"type":"Feature","properties":{"name":{"k":"it\'s"}},"geometry":{"type":"LineString","coordinates":[[1,2],[3,4]]}}',
    '{"type":"Feature","properties":{"name":true},"geometry":{"type":"LineString","coordinates":[[1,2],[3,4]]}}',
    '{"type":"Feature","properties":[],"geometry":{"type":"LineString","coordinates":[[1,2],[3,4]]}}',
    '{"type":"Feature","properties":[1],"geometry":{"type":"LineString","coordinates":[[1,2],[3,4]]}}',
    '{"type":"Feature","properties":"x","geometry":{"type":"LineString","coordinates":[[1,2],[3,4]]}}',
    '{"type":"Feature","properties":{"name":"x"},"geometry":null}', '{"type":"Feature","properties":{"name":"x"}}',
    '{"type":"Feature","properties":{"name":"x"},"geometry":5}',
    '{"type":"FeatureCollection","features":[{"type":"Feature","properties":{"name":"A"},"geometry":{"type":"LineString","coordinates":[[1,2],[3,4]]}},'
    '{"type":"Feature","properties":{"name":"B"},"geometry":{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,0]]]}},5,"x",null]}',
    '{"type":"FeatureCollection","features":5}', '{"type":"FeatureCollection","features":null}', '{"type":"FeatureCollection"}',
    '{"type":"FeatureCollection","features":"ab"}', '{"type":"FeatureCollection","features":{"a":1}}',
    '{"type":"FeatureCollection","features":[{"type":"FeatureCollection","features":[{"type":"LineString","coordinates":[[1,2],[3,4]]}]}]}',
    '{"type":"Point","coordinates":[1,2]}', '{"type":5}', '{"type":null}', '{}', '[]', '5', 'null', '"x"', 'true', '',
    'not json', '{"type":"LineString","coordinates":[[1e400,2],[3,4]]}', '{"type":"LineString","coordinates":[[NaN,2],[3,4]]}',
    '{"type":"LineString","coordinates":[[1,2],[3,4]]} trailing', '﻿{"type":"LineString","coordinates":[[1,2],[3,4]]}',
    '{"type":"LineString","coordinates":[[100000000000000000000000000000000,2],[3,4]]}',
    '{"type":"LineString","coordinates":[[1.5e-7,2],[3,4.25]]}',
    '{"type":"Polygon","coordinates":[[[0,0],[10,0],[10,10],[0,10],[1e-10,0]]]}',
]


def make_zip(entries, compress=zipfile.ZIP_DEFLATED) -> bytes:
    import io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compress) as zf:
        for name, data in entries:
            zf.writestr(name, data)
    return buf.getvalue()


def make_geoimport() -> None:
    o = Out("geoimport.jsonl")
    for text in KML_TEXTS:
        ok, v = run(GI.parse_kml, text)
        o.add(fn="kml", text=text, **({"out": feats_rec(v)} if ok else {"error": v}))
    # random coordinate lists
    for _ in range(40):
        pts = " ".join(f"{RNG.uniform(-180, 180):.{RNG.randint(0, 9)}f},{RNG.uniform(-90, 90):.{RNG.randint(0, 9)}f}"
                       + (f",{RNG.uniform(0, 100):.2f}" if RNG.random() < 0.5 else "") for _ in range(RNG.randint(1, 8)))
        text = kml(pm("r", ls(pts)) + pm("p", poly(pts)))
        ok, v = run(GI.parse_kml, text)
        o.add(fn="kml", text=text, **({"out": feats_rec(v)} if ok else {"error": v}))
    for text in GEOJSON_TEXTS:
        ok, v = run(GI.parse_geojson, text)
        rec = {"fn": "geojson", "text": text}
        if text in ('{"type":"LineString","coordinates":[[true,false],[1,2]]}',
                    '{"type":"LineString","coordinates":"ab"}', '{"type":"LineString","coordinates":{"a":1}}',
                    '{"type":"LineString","coordinates":[[100000000000000000000000000000000,2],[3,4]]}'):
            rec["nm"] = True
        if ok:
            try:
                rec["out"] = feats_rec(v)
            except (ValueError, TypeError):
                rec["nm"] = True          # Python holds non-numbers as coordinates
        else:
            rec["error"] = v
        o.add(**rec)
    # load_features through real files: extension dispatch and sniffing
    tmp = Path(tempfile.mkdtemp())
    k0 = KML_TEXTS[0]
    g0 = '{"type":"LineString","coordinates":[[10,20],[30,40]]}'
    for ext, text in ((".kml", k0), (".KML", k0), (".geojson", g0), (".json", g0), (".GeoJSON", g0), (".txt", k0),
                      (".txt", g0), ("", k0), ("", g0), (".kml", g0), (".json", k0), (".dat", "  \n\t" + k0),
                      (".dat", "garbage"), (".kml", ""), (".xyz", "﻿" + k0)):
        p = tmp / ("f" + ext)
        p.write_bytes(text.encode("utf-8"))
        ok, v = run(GI.load_features, p)
        o.add(fn="load", ext=ext.lower(), text=text, **({"out": feats_rec(v)} if ok else {"error": v}))
    bad = tmp / "bad.kml"
    bad.write_bytes(b"<kml><Placemark><name>\xff\xfe</name><LineString><coordinates>1,2 3,4</coordinates></LineString></Placemark></kml>")
    ok, v = run(GI.load_features, bad)
    o.add(fn="load_bytes", ext=".kml", data=bad.read_bytes().hex(), **({"out": feats_rec(v)} if ok else {"error": v}))
    # KMZ
    kz = k0
    kz2 = KML_TEXTS[5]
    cases = [
        [("doc.kml", kz)], [("other.kml", kz2), ("doc.kml", kz)], [("a/b.KML", kz)], [("readme.txt", "x")], [],
        [("first.kml", kz), ("second.kml", kz2)], [("x/doc.kml", kz2), ("doc.kml", kz)], [("doc.kml", kz.encode("utf-8") + b"\xff")],
    ]
    for entries in cases:
        for comp in (zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED):
            data = make_zip([(n, t if isinstance(t, bytes) else t.encode("utf-8")) for n, t in entries], comp)
            p = tmp / "x.kmz"
            p.write_bytes(data)
            ok, v = run(GI.parse_kmz, p)
            o.add(fn="kmz", data=data.hex(), **({"out": feats_rec(v)} if ok else {"error": v}))
    for junk in (b"", b"PK", b"not a zip at all", make_zip([("doc.kml", kz.encode())])[:-10]):
        p = tmp / "j.kmz"
        p.write_bytes(junk)
        ok, v = run(GI.parse_kmz, p)
        o.add(fn="kmz", data=junk.hex(), **({"out": feats_rec(v)} if ok else {"error": v}))
    shutil.rmtree(tmp, ignore_errors=True)
    o.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    make_tiles()
    make_fetcher_scenarios()
    make_terrain()
    make_surface()
    make_points()
    make_geocode()
    make_profile()
    make_geoimport()


if __name__ == "__main__":
    main()
