# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""A canonical dump of an openskp ``SkpModel`` (M5.7).

Every float travels as the hex of its big-endian IEEE-754 double so that the
comparison with the Rust parse is bit for bit; ids, flags and names as
themselves. The Rust side (``ingetrazo-skp/tests/common``) builds the same
structure from its own ``Model`` and the two are compared as canonical JSON.

A face's ``layer`` is what the app sees after ``resolve_face_layers``: a name
for the faces of a definition, the raw tag id for the faces of the model
root (the resolution does not reach those).
"""
from __future__ import annotations

import hashlib
import struct


def fb(x) -> str:
    return struct.pack(">d", float(x)).hex()


def fbs(seq):
    return None if seq is None else [fb(x) for x in seq]


def val(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return {"i": int(v)}
    if isinstance(v, int):
        return {"i": v}
    if isinstance(v, float):
        return {"f": fb(v)}
    if isinstance(v, str):
        return {"s": v}
    if isinstance(v, tuple):
        return {"p": [fb(x) for x in v]}
    if isinstance(v, list):
        return {"l": [val(x) for x in v]}
    raise TypeError(v)


def layer_of(f, is_root):
    if f.layer is None:
        return None
    if isinstance(f.layer, str):
        return ["s", f.layer]
    assert is_root, "an unresolved tag id outside the root"
    return ["i", int(f.layer)]


def dump_def(d, is_root=False):
    return {
        "id": d.id,
        "guid": d.guid or "",
        "name": d.name or "",
        "afc": bool(d.always_faces_camera),
        "sfs": bool(d.shadows_face_sun),
        "img": bool(d.is_image),
        "v": [[k, fb(v.x), fb(v.y), fb(v.z)] for k, v in d.vertices.items()],
        "e": [[k, e.id, e.v1_id, e.v2_id, e.soft, e.smooth, e.hidden, e.layer, e.curve_id]
              for k, e in d.edges.items()],
        "f": [[k, f.id, [[[e, s] for e, s in lp] for lp in f.loops], fbs(f.normal), f.material_id,
               f.back_material_id, fbs(f.uv_transform), fbs(f.uv_transform_back), bool(f.uv_projected),
               bool(f.uv_projected_back), bool(f.hidden), layer_of(f, is_root)]
              for k, f in d.faces.items()],
        "i": [[i.name or "", i.ref_idx, i.guid or "", [fb(x) for x in i.matrix], i.layer or "",
               i.material_id, bool(i.hidden), [[k, v] for k, v in i.properties.items()],
               [[n, [[k, val(v)] for k, v in e.items()]] for n, e in i.attribute_dictionaries.items()]]
              for i in d.instances],
        "sp": [[fbs(s.plane), s.name, s.label, bool(s.hidden)] for s in d.section_planes],
        "t": [[t.text, bool(t.hidden), fbs(t.point), fbs(t.label_point)] for t in d.texts],
        "d": [[x.text, bool(x.hidden), fbs(x.a), fbs(x.b), fb(x.offset), fbs(x.plane_x), fbs(x.normal)]
              for x in d.dimensions],
        "cl": [[fbs(c.point), fbs(c.direction), fbs(c.start), fbs(c.end)] for c in d.construction_lines],
        "cp": [[fbs(c.position)] for c in d.construction_points],
    }


def dump_model(m):
    index = {id(x): i for i, x in enumerate(m.materials)}
    return {
        "version": m.version,
        "units": m.units,
        "defs": [dump_def(d) for d in m.definitions.values()],
        "root": dump_def(m.root, True),
        "layers": [[l.name, l.color_r, l.color_g, l.color_b, bool(l.hidden)] for l in m.layers],
        "pages": [[p.name, fbs(p.eye), fbs(p.target), fbs(p.up), fb(p.fov), bool(p.parallel), fb(p.ortho_height),
                   list(p.hidden_layers)] for p in m.pages],
        "dims": [[x.text, bool(x.hidden), fbs(x.a), fbs(x.b), fb(x.offset), fbs(x.plane_x), fbs(x.normal)]
                 for x in m.dimensions],
        "mats": [[x.name, list(x.color), fb(x.transparency), x.id, bool(x.colorized), x.colorize_type,
                  None if x.texture is None else [x.texture.filename, fb(x.texture.width), fb(x.texture.height),
                                                  None if x.texture.data is None else hashlib.sha256(x.texture.data).hexdigest()]]
                 for x in m.materials],
        "by_id": [[k, index[id(v)]] for k, v in m.materials_by_id.items()],
        "styles": [[s.name, None if s.front_color is None else list(s.front_color),
                    None if s.back_color is None else list(s.back_color)] for s in m.styles],
    }
