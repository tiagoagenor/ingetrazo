# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""A small writer of SketchUp 2021+ (VFF) files for the M5.7 fixtures: a zip
holding ``model.dat`` (tag-length-value records), material and style XML and
``meta/meta.dat``. See ``skp_synth.py``: written from what
``openskp/_core.py`` reads, not from a template."""
from __future__ import annotations

import io
import struct
import zipfile

from skp_synth import STR, f64, f64s, newell, u32


def rec(tag: str, payload: bytes = b"") -> bytes:
    """A model.dat record: two tag bytes (written as hex), a u32 length."""
    return bytes.fromhex(tag) + u32(len(payload)) + payload


def rec_le(tag: int, payload: bytes = b"") -> bytes:
    """A record whose tag is a little-endian u16 (scenes, dimensions)."""
    return struct.pack("<H", tag) + u32(len(payload)) + payload


def vid(n: int) -> bytes:
    return struct.pack("<H", n) if n < 0x10000 else struct.pack("<I", n)


def entity_dc05(n: int, extra: bytes = b"") -> bytes:
    return rec("DC05", rec("DE05", vid(n)) + extra)


def xml_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def attr_value(v) -> bytes:
    if v is None:
        return b""
    if isinstance(v, str):
        return rec("AD38", v.encode())
    if isinstance(v, bool):
        return rec("A738", struct.pack("<i", int(v)))
    if isinstance(v, int):
        return rec("A738", struct.pack("<i", v))
    if isinstance(v, float):
        return rec("A938", f64(v))
    if isinstance(v, tuple):
        return rec("B438", f64s(*v))
    if isinstance(v, list):
        return rec("AE38", b"".join(rec("A438", attr_value(x)) for x in v))
    raise TypeError(v)


class Vff:
    """A VFF document. Ids come from one counter; a definition's id is the
    one its first ``D007`` carries, as the reader looks for it."""

    def __init__(self, version: str = "21.0.0"):
        self.version = version
        self.next_id = 0x100
        self.materials: list = []
        self.layers: list = []
        self.defs: list = []
        self.def_ids: dict = {}
        self.root_parts: list = []
        self.files: dict = {}
        self.tail: list = []
        self.units = "Meter"
        self.styles: list = []
        self.thumbnail: bytes = b""
        self.wrap = "none"

    def new_id(self) -> int:
        self.next_id += 1
        return self.next_id

    def material(self, name, rgb=(200, 200, 200), trans=None, texture=None, colorized=False,
                 ctype=0, folder=None, xml_extra="", mat_id=None, listed=True):
        """``trans``: opacity 0-1 (``useTrans``); ``texture``: ``(file_name,
        bytes, xscale, yscale)``; ``listed=False`` leaves it out of the id
        table (a ``Layer_<name>`` colour material has no id)."""
        mid = mat_id if mat_id is not None else self.new_id()
        self.materials.append(dict(id=mid, name=name, rgb=rgb, trans=trans, texture=texture,
                                   colorized=colorized, ctype=ctype, folder=folder or name,
                                   xml_extra=xml_extra, listed=listed))
        return mid

    def layer(self, name, hidden=False):
        lid = self.new_id()
        self.layers.append((lid, name, hidden))
        return lid

    def geometry(self) -> "VffGeom":
        return VffGeom(self)

    def definition(self, name, geom: "VffGeom", kind=0, faces_camera=False, guid=None,
                   def_id=None, shadows=False) -> int:
        did = def_id if def_id is not None else self.new_id()
        guid = guid or bytes([(did + i) & 0xFF for i in range(16)])
        flags = rec("5D1B", u32(1 if faces_camera else 0)) + rec("5E1B", u32(1 if shadows else 0))
        body = (rec("8813", rec("D007", entity_dc05(did) + rec("D307", b"\x06")) + geom.containers())
                + rec("7D15", guid) + rec("7E15", name.encode()) + rec("7F15") + rec("8015")
                + rec("8215", b"\x01") + rec("8315", u32(kind)) + rec("581B", flags))
        self.defs.append(rec("7C15", body))
        self.def_ids[name] = did
        return did

    def root(self, geom: "VffGeom") -> None:
        self.root_parts.append(rec("F601", rec("8813", rec("D007", entity_dc05(self.new_id()) + rec("D307", b"\x06"))
                                                + geom.containers(root=True))))

    def model_dat(self) -> bytes:
        mats = b"".join(rec("C832", entity_dc05(m["id"]) + rec("CC32", m["name"].encode())
                            + rec("CA32", b"\x00") + rec("CB32")) for m in self.materials if m["listed"])
        layers = b"".join(rec("8C3C", entity_dc05(i) + rec("8D3C", n.encode()) + rec("8E3C", b"\x01" if h else b"\x00")
                               + rec("903C", u32(0))) for i, n, h in self.layers)
        recs = []
        if any(m["listed"] for m in self.materials):
            recs.append(rec("F701", rec("D430", rec("D530", mats))))
        if self.layers:
            recs.append(rec("F801", rec("983A", rec("993A", layers))))
        defs = list(self.defs)
        if self.wrap == "f901" and defs:
            recs.append(rec("F901", rec("7017", rec("7117", b"".join(defs)))))
        else:
            recs.extend(defs)
        recs.extend(self.root_parts)
        recs.extend(self.tail)
        body = b"".join(recs)
        if self.wrap == "f401":
            body = rec("F401", body)
        return body

    def material_xml(self, m) -> str:
        r, g, b = m["rgb"]
        attrs = (f'name="{xml_escape(m["name"])}" type="{2 if m["colorized"] else 0}" colorRed="{r}" '
                 f'colorGreen="{g}" colorBlue="{b}" colorizeType="{m["ctype"]}"')
        if m["trans"] is not None:
            attrs += f' trans="{1.0 - m["trans"]!r}" useTrans="1"'
        else:
            attrs += ' trans="0" useTrans="0"'
        inner = ""
        if m["texture"] is not None:
            fn, _data, xs, ys = m["texture"]
            attrs += ' hasTexture="1"'
            inner = (f'<mat:texture textureFilename="{xml_escape(fn)}" xScale="{xs}" yScale="{ys}">'
                     f'{m["xml_extra"]}</mat:texture>')
        else:
            attrs += ' hasTexture="0"'
        return ('<?xml version="1.0" encoding="UTF-8" standalone="no" ?>\n'
                '<materialDocument xmlns="http://sketchup.google.com/schemas/sketchup/1.0/material" '
                'xmlns:mat="http://sketchup.google.com/schemas/sketchup/1.0/material">\n'
                f'  <mat:material {attrs}>{inner}</mat:material>\n</materialDocument>\n')

    def style_xml(self, name, front, back) -> str:
        def item(i, v):
            return f'<sty:item id="{i}"><t:variant type="4">{v}</t:variant></sty:item>'
        items = ""
        if front is not None:
            items += item(4000, front)
        if back is not None:
            items += item(4001, back)
        return ('<?xml version="1.0" encoding="UTF-8" standalone="no" ?>\n'
                '<styleDocument xmlns="http://sketchup.google.com/schemas/sketchup/1.0/style" '
                'xmlns:sty="http://sketchup.google.com/schemas/sketchup/1.0/style">\n'
                f'  <sty:style xmlns:t="http://sketchup.google.com/schemas/1.0/types" name="{xml_escape(name)}">'
                f'{items}</sty:style>\n</styleDocument>\n')

    def write(self, path, model_dat: bytes | None = None, zip_members: dict | None = None,
              skip_model: bool = False) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            def put(name, data, deflate=True):
                zi = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                zi.compress_type = zipfile.ZIP_DEFLATED if deflate else zipfile.ZIP_STORED
                zf.writestr(zi, data)
            if self.thumbnail:
                put("meta/model_thumbnail.png", self.thumbnail, False)
            if self.units is not None:
                put("meta/meta.dat", rec("6400", rec("6D00", self.units.encode())))
            for m in self.materials:
                put(f"materials/{m['folder']}/material.xml", self.material_xml(m).encode())
                if m["texture"] is not None and m["texture"][1] is not None:
                    put(f"materials/{m['folder']}/{m['texture'][0]}", m["texture"][1], False)
            for name, front, back in self.styles:
                put(f"styles/{name}/style.xml", self.style_xml(name, front, back).encode())
            for name, data in self.files.items():
                put(name, data)
            if not skip_model:
                put("model.dat", self.model_dat() if model_dat is None else model_dat)
            for name, data in (zip_members or {}).items():
                put(name, data)
        ver = "{%s}" % self.version
        head = (b"\xff\xfe\xff\x0e" + "SketchUp Model".encode("utf-16-le") + STR + bytes([len(ver)])
                + ver.encode("utf-16-le") + b"VFF\x08\x00\x01\x00\x08\x00" + bytes([0x67, 0xBB, 0x7F, 0xE3]))
        with open(path, "wb") as fh:
            fh.write(head + buf.getvalue())


class VffGeom:
    """The entities of one definition (or the model root)."""

    def __init__(self, doc: Vff):
        self.doc = doc
        self.verts: dict = {}
        self.vert_recs: list = []
        self.edges: dict = {}
        self.edge_recs: list = []
        self.faces: list = []
        self.insts: list = []
        self.extra: list = []

    def vertex(self, p) -> int:
        if p not in self.verts:
            i = self.doc.new_id()
            self.verts[p] = i
            self.vert_recs.append(rec("C409", entity_dc05(i) + rec("C509", f64s(*p))))
        return self.verts[p]

    def edge(self, a, b, flags=0x06, mat=None, layer=None, curve=None) -> tuple:
        """``(id, forward)`` for the edge a-b; created on first use."""
        if (a, b) in self.edges:
            return self.edges[(a, b)], True
        if (b, a) in self.edges:
            return self.edges[(b, a)], False
        va, vb = self.vertex(a), self.vertex(b)
        i = self.doc.new_id()
        self.edges[(a, b)] = i
        d007 = entity_dc05(i)
        if mat is not None:
            d007 += rec("D107", vid(mat))
        if layer is not None:
            d007 += rec("D207", vid(layer))
        d007 += rec("D307", bytes([flags]))
        body = rec("D007", d007) + rec("B90B", vid(va)) + rec("BA0B", vid(vb))
        if curve is not None:
            body += rec("BB0B", vid(curve))
        self.edge_recs.append(rec("B80B", body))
        return i, True

    def loose_edge(self, a, b, **kw):
        self.edge(a, b, **kw)

    def face(self, outer, holes=(), mat=None, back=None, layer=None, hidden=False, normal=None,
             uv_front=None, uv_back=None, edge_flags=0x06, edge_layer=None, curve=None, face_id=None):
        fid = face_id if face_id is not None else self.doc.new_id()
        rings = []
        for ring in [outer] + list(holes):
            items = b""
            n = len(ring)
            for k in range(n):
                eid, fwd = self.edge(ring[k], ring[(k + 1) % n], flags=edge_flags, layer=edge_layer, curve=curve)
                items += rec("A00F", rec("A10F", vid(eid)) + rec("A20F", b"\x00" if fwd else b"\x01"))
            rings.append(rec("9411", rec("9511", items)))
        extra = b""
        if uv_front is not None or uv_back is not None:
            def side(tag, m):
                return b"" if m is None else rec(tag, rec("1327", rec("1527", f64s(*m))))
            extra = rec("DD05", rec("B136", rec("B236", rec("1027", side("1127", uv_front) + side("1227", uv_back)))))
        d007 = entity_dc05(fid, extra)
        if mat is not None:
            d007 += rec("D107", vid(mat))
        if layer is not None:
            d007 += rec("D207", vid(layer))
        d007 += rec("D307", bytes([0x07 if hidden else 0x06]))
        body = rec("D007", d007)
        if back is not None:
            body += rec("AF0D", vid(back))
        nrm = normal if normal is not None else newell(outer)
        body += rec("AD0D", f64s(*nrm)) + rec("AE0D", b"".join(rings))
        self.faces.append(rec("AC0D", body))
        return fid

    def instance(self, def_id, xf13, name="", mat=None, layer=None, hidden=False, attrs=None,
                 guid=None, inst_id=None):
        iid = inst_id if inst_id is not None else self.doc.new_id()
        extra = b""
        if attrs:
            dicts = b""
            for dname, entries in attrs:
                ents = b""
                for k, v in entries:
                    ents += rec("B636", k.encode()) + rec("A438", attr_value(v))
                dicts += rec("B436", dname.encode()) + rec("B536", ents)
            extra = rec("DD05", dicts)
        d007 = entity_dc05(iid, extra)
        if mat is not None:
            d007 += rec("D107", vid(mat))
        if layer is not None:
            d007 += rec("D207", vid(layer))
        d007 += rec("D307", bytes([0x07 if hidden else 0x06]))
        body = (rec("D007", d007) + rec("6519", name.encode()) + rec("6619", f64s(*xf13))
                + rec("6719", vid(def_id)) + rec("6819", guid or bytes([(iid + i) & 0xFF for i in range(16)])))
        self.insts.append(rec("6419", body))
        return iid

    def containers(self, root=False) -> bytes:
        out = b""
        if self.vert_recs:
            out += rec("8913", b"".join(self.vert_recs))
        if self.edge_recs:
            out += rec("8A13", b"".join(self.edge_recs))
        if self.faces:
            out += rec("8B13", b"".join(self.faces))
        if self.insts:
            out += rec("8D13", rec("4C1D", b"".join(self.insts))) if root else rec("8C13", b"".join(self.insts))
        out += b"".join(self.extra)
        return out


def page_record(pages) -> bytes:
    """The ``0702`` scenes record. ``pages``: dicts with name, eye, target, up,
    fov, perspective, height, hidden (layer ids)."""
    out = b""
    for p in pages:
        cam = (rec_le(0x34BD, f64s(*p["eye"])) + rec_le(0x34BE, f64s(*p["target"])) + rec_le(0x34BF, f64s(*p["up"]))
               + rec_le(0x34C4, f64(p.get("fov", 35.0))) + rec_le(0x34C2, b"\x01" if p.get("perspective", True) else b"\x00")
               + rec_le(0x34C3, f64(p.get("height", 0.0))))
        hidden = b"".join(bytes([len(vid(h))]) + vid(h) for h in p.get("hidden", ()))
        item = rec_le(0x6F54, rec_le(0x6F55, p["name"].encode())) + rec_le(0x714A, rec_le(0x34BC, cam))
        if hidden:
            item += rec_le(0x7150, hidden)
        out += rec_le(0x7148, item)
    return rec("0702", rec_le(0x6D60, rec_le(0x6D61, out)))


def dimension_record(a, b, offset=12.0, plane_x=(1, 0, 0), normal=(0, 0, 1), anchored=None) -> bytes:
    """A ``5BCC`` dimension, as a loose top-level record. ``anchored``:
    ``((vertex_id, instance_id|None) | None, (...) | None)`` references
    geometry instead of a free point at that end."""
    def free(p):
        return rec_le(0x5208, rec_le(0x5209, u32(1)) + rec_le(0x520A, f64s(*p)))

    def ref(vertex_id, inst_id):
        fi = rec_le(0x53FD, vid(vertex_id))
        if inst_id is not None:
            ib = vid(inst_id)
            fi += rec_le(0x53FE, bytes([len(ib)]) + ib)
        return rec_le(0x5208, rec_le(0x5209, u32(2)) + rec_le(0x520A, f64s(0, 0, 0))
                      + rec_le(0x520B, rec_le(0x53FC, fi)))

    pa = free(a) if anchored is None or anchored[0] is None else ref(*anchored[0])
    pb = free(b) if anchored is None or anchored[1] is None else ref(*anchored[1])
    body = (rec_le(0x5BCD, pa) + rec_le(0x5BCE, pb) + rec_le(0x5BCF, f64s(*plane_x))
            + rec_le(0x5BD0, f64s(*normal)) + rec_le(0x5BD2, f64(offset)))
    return rec("1102", rec_le(0x5BCC, body))
