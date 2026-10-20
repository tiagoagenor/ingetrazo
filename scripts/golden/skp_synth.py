# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""Two small writers of SketchUp files for the M5.7 fixtures: the classic MFC
archive (2013-2020) and the VFF zip (2021+).

They are written from what the READERS decode (``openskp/legacy.py`` and
``openskp/_core.py``), not from a template: IngeTrazo does not distribute
the SDK-written blank document that ``openskp.create`` splices its output
into (tests/test_skp_no_vendor_files.py), so the fixtures cannot come from
that writer. A file here is valid when openskp reads it back the way it was
meant, which the fixture script checks entity by entity.

Lengths are inches, as in the format. Nothing here is a general writer: it
emits what the fixtures need, with the layout the readers expect.
"""
from __future__ import annotations

import io
import struct
import zipfile

STR = b"\xff\xfe\xff"


def u16(v: int) -> bytes:
    return struct.pack("<H", v)


def u32(v: int) -> bytes:
    return struct.pack("<I", v)


def f64(v: float) -> bytes:
    return struct.pack("<d", v)


def f64s(*vs) -> bytes:
    return b"".join(f64(v) for v in vs)


def utf16(text: str) -> bytes:
    raw = text.encode("utf-16-le")
    n = len(raw) // 2
    assert n < 0xFF
    return STR + bytes([n]) + raw


# ---------------------------------------------------------------------------
# classic MFC archive
# ---------------------------------------------------------------------------

class Legacy:
    """One ``CArchive`` object stream, written the way the walker reads it.

    Every class is declared in line the first time it appears (``FFFF``,
    schema, name) and named by its store index afterwards (``0x8000|slot``);
    an object seen before is a plain 16-bit back-reference. ``slot`` numbers
    follow the reader's: a class declaration takes one, and so does every
    object, the object's own before the ones nested inside it.
    """

    def __init__(self, ver: int = 17, base: int = 1):
        self.ver = ver
        self.buf = bytearray()
        self.next_slot = base
        self.class_slot: dict = {}
        self.verts: dict = {}
        self.edges: dict = {}
        self.pid = 0

    # -- primitives
    def raw(self, b: bytes) -> None:
        self.buf += b

    def mark(self) -> int:
        return len(self.buf)

    def _alloc(self) -> int:
        s = self.next_slot
        self.next_slot += 1
        return s

    def ref(self, slot) -> None:
        """A back-reference (null for ``None``)."""
        if slot is None:
            self.raw(u16(0))
            return
        assert slot < 0x7FFF
        self.raw(u16(slot))

    def begin(self, cls: str, schema: int) -> int:
        """Start an object of class ``cls``; returns its slot. The body
        follows through the other methods."""
        if cls in self.class_slot:
            self.raw(u16(0x8000 | self.class_slot[cls]))
        else:
            self.raw(u16(0xFFFF) + u16(schema) + u16(len(cls)) + cls.encode("ascii"))
            self.class_slot[cls] = self._alloc()
        return self._alloc()

    def preamble(self, attrs=None) -> None:
        """The entity preamble: the attribute container pointer (null or the
        container written by ``attrs()``), then the persistent-id mask."""
        if attrs is None:
            self.raw(u16(0))
        else:
            attrs()
        if self.ver >= 17:
            self.raw(b"\x00")

    def drawbase(self, mat=0, hidden=0, soft=0, smooth=0, layer=0) -> None:
        self.raw(u16(mat) + bytes([hidden, 1, 1, soft, smooth, 0]) + u16(layer))

    # -- geometry
    def vertex(self, p) -> int:
        """The vertex at ``p``: written in line the first time."""
        if p in self.verts:
            return self.verts[p]
        slot = self.begin("CVertex", 3)
        self.verts[p] = slot
        self.preamble()
        self.raw(f64s(*p))
        return slot

    def edge(self, a, b, soft=0, smooth=0, hidden=0, layer=0) -> int:
        key = (a, b)
        rkey = (b, a)
        if key in self.edges:
            return self.edges[key][0]
        if rkey in self.edges:
            return self.edges[rkey][0]
        slot = self.begin("CEdge", 5)
        self.edges[key] = (slot, a, b)
        self.preamble()
        self.drawbase(hidden=hidden, soft=soft, smooth=smooth, layer=layer)
        self.vertex(a) if a not in self.verts else self.ref(self.verts[a])
        self.vertex(b) if b not in self.verts else self.ref(self.verts[b])
        self.raw(u16(0))  # no curve
        return slot

    def _edge_use(self, loop_slot: int, a, b, edge_kw) -> None:
        self.begin("CEdgeUse", 2)
        self.preamble()
        key, rkey = (a, b), (b, a)
        if key in self.edges:
            known = self.edges[key]
            self.ref(known[0])
            sense = 0
        elif rkey in self.edges:
            known = self.edges[rkey]
            self.ref(known[0])
            sense = 1
        else:
            self.edge(a, b, **edge_kw)
            sense = 0
        self.raw(bytes([sense]))
        self.ref(loop_slot)

    def loop(self, pts, edge_kw) -> None:
        slot = self.begin("CLoop", 2)
        self.preamble()
        self.raw(b"\x00\x00")
        n = len(pts)
        for i in range(n):
            self._edge_use(slot, pts[i], pts[(i + 1) % n], edge_kw)
        self.raw(u16(0))

    def face(self, outer, holes=(), mat=0, back=0, layer=0, hidden=0, normal=None,
             soft=0, smooth=0, hidden_edges=0, ftc=None) -> None:
        """A face: its loops and (first use) their edges, in line."""
        if normal is None:
            normal = newell(outer)
        ekw = dict(soft=soft, smooth=smooth, hidden=hidden_edges, layer=layer)
        self.begin("CFace", 4)
        self.preamble(None if ftc is None else (lambda: self.attr_container([("ftc", ftc)])))
        self.drawbase(mat=mat, hidden=hidden, layer=layer)
        self.raw(f64s(*normal, 0.0))
        rings = [outer] + list(holes)
        self.raw(u32(len(rings)))
        for r in rings:
            self.loop(r, ekw)
        self.raw(u16(back))

    # -- attributes
    def attr_container(self, children) -> None:
        """``children``: ``("ftc", (front9, back9, front_proj, back_proj))``
        or ``("dict", name, {key: value})``."""
        self.begin("CAttributeContainer", 3)
        self.preamble()
        for kind, *rest in children:
            if kind == "ftc":
                front, back, fp, bp = rest[0]
                self.begin("CFaceTextureCoords", 4)
                self.preamble()
                self.raw(u32(0))
                self.raw(f64s(*front) + f64s(*([0.0] * 3)) + f64s(*back) + f64s(*([0.0] * 3)))
                self.raw(u32(0) + u32(0))
                self.raw(u32(2 if fp else 1) + u32(2 if bp else 1))
            else:
                name, entries = rest
                self.begin("CAttributeNamed", 2)
                self.preamble()
                self.raw(b"\x00" * 4)
                self.raw(utf16(name))
                for k, v in entries.items():
                    self.raw(utf16(k))
                    self._typed(v)
                self.raw(utf16(""))
                self.raw(u32(0))
        self.raw(u16(0))

    def _typed(self, v) -> None:
        if v is None:
            self.raw(b"\x00")
        elif isinstance(v, bool):
            self.raw(b"\x07" + bytes([1 if v else 0]))
        elif isinstance(v, int):
            self.raw(b"\x04" + struct.pack("<i", v))
        elif isinstance(v, float):
            self.raw(b"\x06" + f64(v))
        elif isinstance(v, str):
            self.raw(b"\x0a" + utf16(v))
        elif isinstance(v, tuple):
            self.raw(b"\x11" + f64s(*v))
        elif isinstance(v, list):
            self.raw(b"\x0b" + u32(len(v)))
            for x in v:
                self._typed(x)
        else:
            raise TypeError(v)

    # -- materials and layers
    def material(self, name, rgba=(200, 200, 200, 255), opacity=None, texture=None) -> int:
        slot = self.begin("CMaterial", 12)
        self.preamble()
        self.raw(utf16(name))
        if texture is None:
            self.raw(u16(0))
            self.raw(bytes(rgba))
            self.raw(utf16(""))
            self.raw(b"\x00" * 8)
            self.raw(f64(0.0 if opacity is None else 1.0 - opacity))
            self.raw(b"\x00" if opacity is None else b"\x01")
        else:
            self.raw(u16(1))
            self._texture_block(texture, rgba, opacity)
        return slot

    def _texture_block(self, tex, rgba, opacity) -> None:
        data, fname, w, h, colorized = tex
        self.raw(b"\x00\x00" if self.ver >= 17 else b"\x00")
        self.begin("CDib", 3)
        sub = 4 if data[:4] == b"\x89PNG" else 1
        self.raw(u32(sub) + u32(len(data)) + data)
        self.raw(f64(w) + f64(h))
        self.raw(utf16(fname))
        self.raw(bytes(rgba) + b"\x00" + bytes(rgba))
        self.raw(utf16(""))
        self.raw(u32(0) + u32(1 if colorized else 0))
        self.raw(f64(0.0 if opacity is None else 1.0 - opacity))
        self.raw(b"\x00" if opacity is None else b"\x01")

    def layer(self, name, hidden=0, rgba=(136, 136, 136, 255)) -> int:
        slot = self.begin("CLayer", 3)
        self.preamble()
        self.raw(utf16(name))
        self.raw(bytes([hidden, 0, 0, 0]))
        self.raw(utf16("Layer_" + name))
        self.raw(u16(0))
        self.raw(bytes(rgba))
        self.raw(utf16(""))
        self.raw(b"\x00" * 21)
        return slot

    # -- containers
    def definition(self, name, body, guid=b"\x11" * 16, behavior=0, layers=()) -> int:
        """A component definition; ``body()`` writes its entities and returns
        their count."""
        self.verts, self.edges = {}, {}
        slot = self.begin("CComponentDefinition", 11)
        self.preamble()
        self.raw(bytes([0, 0, 0, 1, 1]) + b"\x00" * ((22 if self.ver >= 17 else 20) - 5))
        self.raw(u32(len(layers)))
        for lay in layers:
            self.ref(lay)
        self.raw(u16(0))
        self.raw(u32(0))
        patch = self.mark()
        self.raw(u32(0))
        count = body()
        self.buf[patch:patch + 4] = u32(count)
        self.raw(u32(0))  # relationships
        self.raw(u16(0))
        self.raw(guid)
        self.raw(utf16(name) + utf16("") + utf16(""))
        self.raw(u32(0))
        self.raw(b"\x00" * 10 + bytes([behavior]) + b"\x00" * 8)
        self.begin("CThumbnail", 1)
        self.preamble()
        self.raw(u16(0) + u16(0))
        return slot

    def instance(self, def_slot, xf13, name="", group=False, mat=0, layer=0, hidden=0,
                 guid=b"\x22" * 16, attrs=None) -> None:
        self.begin("CGroup" if group else "CComponentInstance", 1 if group else 6)
        self.preamble(None if attrs is None else (lambda: self.attr_container([("dict", *a) for a in attrs])))
        self.drawbase(mat=mat, hidden=hidden, layer=layer)
        self.ref(def_slot)
        self.raw(f64s(*xf13))
        self.raw(utf16(name))
        if self.ver >= 14:
            self.raw(guid)

    def image(self, def_slot, xf12, mat=0, layer=0) -> None:
        self.begin("CImage", 6)
        self.preamble()
        self.drawbase(mat=mat, layer=layer)
        self.ref(def_slot)
        self.raw(f64s(*xf12) + f64(1.0) + utf16("") + b"\x33" * 16)

    def edge_item(self, a, b, **kw) -> None:
        """A loose edge as an entity-list item."""
        self.edge(a, b, **kw)

    # -- annotations
    def dimension(self, a, b, offset=10.0, text="") -> None:
        self.begin("CDimensionLinear", 2)
        self.preamble()
        self.drawbase()
        self.raw(utf16(text))
        self.raw(u16(0))  # font
        b37 = bytearray(37)
        b37[5:9] = u32(1)
        b37[13:37] = f64s(*a)
        self.raw(bytes(b37))
        self.raw(u16(0) + u16(0) + u32(0) + u32(0))
        b32 = bytearray(32)
        b32[0:4] = u32(1)
        b32[8:32] = f64s(*b)
        self.raw(bytes(b32))
        self.raw(u16(0) + u16(0) + u32(0) + u32(0))
        b72 = bytearray(72)
        b72[52:60] = f64(offset)
        self.raw(bytes(b72))

    def text(self, text, point, label=None, hidden=0) -> None:
        self.begin("CText", 2)
        self.preamble()
        self.drawbase(hidden=hidden)
        self.raw(u16(0))  # font
        mid = u32(1) + u32(4) + f64s(*point) + b"\x00" * 12 + f64s(*(label or (0.0, 0.0, 0.0))) + b"\x00" * 16 + f64(1.0) + u32(0)
        self.raw(mid + bytes.fromhex("0100000001000300000001"))
        self.raw(utf16(text))
        self.raw(b"\x00" * 5)

    def construction_line(self, point, direction, start=0.0, end=10.0) -> None:
        self.begin("CConstructionLine", 3)
        self.preamble()
        self.drawbase()
        self.raw(f64s(*point) + f64s(*direction) + f64(start) + f64(end))
        self.raw(b"\x00" * (7 if self.ver == 17 else 4))

    def construction_point(self, pos) -> None:
        self.begin("CConstructionPoint", 2)
        self.preamble()
        self.drawbase()
        self.raw(f64s(*pos) + f64s(0, 0, 0) + b"\x00")

    def section_plane(self, plane, name="", label="") -> None:
        self.begin("CSectionPlane", 2)
        self.preamble()
        self.drawbase()
        self.raw(f64s(*plane))
        if name or label:
            self.raw(utf16(name) + utf16(label))

    # -- the whole file
    def header(self) -> bytes:
        ver_text = "{%d.0.0}" % self.ver
        head = bytearray(b"\xff\xfe\xff\x0e" + "SketchUp Model".encode("utf-16-le"))
        head += STR + bytes([len(ver_text)]) + ver_text.encode("utf-16-le")
        head += b"CVersionMap" + b"\x00" * 8
        head += b"\x00" * (0x80 - len(head) % 0x80)
        return bytes(head)


def newell(pts):
    nx = ny = nz = 0.0
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        nx += (a[1] - b[1]) * (a[2] + b[2])
        ny += (a[2] - b[2]) * (a[0] + b[0])
        nz += (a[0] - b[0]) * (a[1] + b[1])
    ln = (nx * nx + ny * ny + nz * nz) ** 0.5 or 1.0
    return (nx / ln, ny / ln, nz / ln)


class Doc:
    """A classic document in the order the walker reads it: materials, the
    layer list, the active layer anchor, the definitions, the root entities,
    then (optionally) a scene record."""

    def __init__(self, ver: int = 17):
        self.w = Legacy(ver=ver)
        self.ver = ver
        self.materials: list = []
        self.layers: dict = {}
        self.defs: dict = {}
        self.state = "materials"
        self._root_patch = None
        self._root_count = 0
        self._def_patch = None
        self._def_count = 0
        self._layer_patch = None

    # materials come first
    def material(self, name, **kw):
        assert self.state == "materials"
        if not self.materials:
            self._mat_start = self.w.mark()
        slot = self.w.material(name, **kw)
        self.materials.append(slot)
        return slot

    def _to_layers(self):
        if self.state != "materials":
            return
        w = self.w
        self.state = "layers"
        w.raw(u32(0))
        if self.ver >= 17:
            w.raw(b"\x00")
        self._layer_patch = w.mark()
        w.raw(u32(0))

    def layer(self, name, **kw):
        self._to_layers()
        assert self.state == "layers"
        slot = self.w.layer(name, **kw)
        self.layers[name] = slot
        return slot

    def _to_defs(self):
        if self.state in ("materials", "layers"):
            self._to_layers()
            if not self.layers:
                self.layer("Layer0")
            self.w.buf[self._layer_patch:self._layer_patch + 4] = u32(len(self.layers))
            # the active layer: a back-reference to the first one
            self.w.ref(next(iter(self.layers.values())))
            self._def_patch = self.w.mark()
            self.w.raw(u32(0))
            self.state = "defs"

    def definition(self, name, body, **kw):
        self._to_defs()
        assert self.state == "defs"
        slot = self.w.definition(name, body, **kw)
        self.defs[name] = slot
        self._def_count += 1
        return slot

    def begin_root(self):
        self._to_defs()
        if self.state == "defs":
            self.w.verts, self.w.edges = {}, {}
            self.w.buf[self._def_patch:self._def_patch + 4] = u32(self._def_count)
            self._root_patch = self.w.mark()
            self.w.raw(u32(0))
            self.state = "root"

    def root_item(self, fn):
        self.begin_root()
        fn()
        self._root_count += 1

    def finish(self, path, scene=None):
        """``scene``: ``(name, hidden_layer_slots)`` appends a scene record."""
        self.begin_root()
        w = self.w
        w.buf[self._root_patch:self._root_patch + 4] = u32(self._root_count)
        if scene is not None:
            name, hidden = scene
            w.raw(utf16(name) + utf16("") + u32(0xC20 if hidden else 0xC00))
            w.raw(u16(0) + u16(0))
            for h in hidden:
                w.raw(u16(h))
            if hidden:
                w.raw(u32(1))
        header = w.header()
        body = bytes(w.buf)
        out = header + u32(len(self.materials)) + body
        # the walker finds the anchor itself; nothing may look like a zip
        assert b"PK\x03\x04" not in out[:0x100]
        with open(path, "wb") as fh:
            fh.write(out)
