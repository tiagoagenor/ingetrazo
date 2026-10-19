# SPDX-License-Identifier: GPL-3.0-or-later
"""The L2 canonical signature of a scene (golden/SCHEMA.md, section 5).

Order-free: two models of the same shape built in a different order give
the same signature, anything else does not. Python writes it; the Rust
runner computes the same thing and compares.
"""
from __future__ import annotations

import hashlib
import json
import struct


def f32hex(x: float) -> str:
    """The 8 hex digits of a float32 (``1.0`` → ``3f800000``)."""
    return struct.pack(">f", x).hex()


def vkey(p) -> str:
    """A vertex key: the float32 bits of x, y and z."""
    return f"{f32hex(p.x())},{f32hex(p.y())},{f32hex(p.z())}"


def _dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, default=_fallback)


def _fallback(obj):
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(obj, (set, frozenset, tuple)):
        return sorted(obj, key=repr) if isinstance(obj, (set, frozenset)) else list(obj)
    for attr in ("x", "y", "z"):          # QVector3D and friends
        if not callable(getattr(obj, attr, None)):
            break
    else:
        return [obj.x(), obj.y(), obj.z()]
    return repr(obj)


_FILE_HASHES: dict = {}


def canonical(obj):
    """``obj`` with every file path (a ``"path"`` value naming a file, as a
    texture's) replaced by the SHA-256 of the file's bytes: a texture
    embedded in an .igz reopens from another folder, and is the same."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "path" and isinstance(v, str):
                out[k] = _file_hash(v)
            else:
                out[k] = canonical(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [canonical(x) for x in obj]
    return obj


def _file_hash(path: str) -> str:
    import os

    try:
        st = os.stat(path)
    except OSError:
        return path
    key = (path, st.st_mtime_ns, st.st_size)
    if key not in _FILE_HASHES:
        with open(path, "rb") as fh:
            _FILE_HASHES[key] = "sha256:" + hashlib.sha256(fh.read()).hexdigest()
    return _FILE_HASHES[key]


def sha(text: str, n: int = 64) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def _fallback_or(v):
    """Plain JSON for ``canonical`` to walk."""
    return json.loads(_dumps(v))


def _rotated(keys: list) -> list:
    """The loop starting at its smallest vertex, winding kept."""
    if not keys:
        return keys
    i = keys.index(min(keys))
    return keys[i:] + keys[:i]


def mesh_signature(mesh) -> dict:
    verts = sorted({vkey(v.position) for v in mesh.vertices})
    curve_number: dict = {}
    edges = []
    raw = []
    for e in mesh.edges:
        a, b = sorted((vkey(e.v0.position), vkey(e.v1.position)))
        raw.append((a, b, e))
    raw.sort(key=lambda t: (t[0], t[1]))
    for a, b, e in raw:
        flags = ("s" if e.soft else "") + ("h" if e.hidden else "")
        if e.layer:
            flags += f"@{e.layer}"
        curve = getattr(e, "curve", None)
        if curve is not None:
            k = curve_number.setdefault(curve, len(curve_number))
            flags += f"c{k}"
        edges.append([a, b, flags])
    edges.sort()
    faces = []
    for f in mesh.faces:
        loop = _rotated([vkey(v.position) for v in f.loop])
        holes = sorted(_rotated([vkey(v.position) for v in h])
                       for h in f.hole_loops)
        attrs = sha(_dumps(canonical(f.attrs)), 16) if f.attrs else ""
        entry = [loop, holes, attrs]
        if getattr(f, "interior", False):
            entry.append("i")
        faces.append(entry)
    faces.sort()
    return {"v": verts, "e": edges, "f": faces}


_GROUP_SKIP = {"mesh", "children", "uid", "owner", "context", "name",
               "component", "xform", "material", "layer", "locked", "hidden"}


def _group_signature(g, defs: dict, shares: dict) -> dict:
    proto = g.mesh
    is_comp = bool(getattr(g, "is_component", lambda: False)())
    out = {
        "name": g.name,
        "kind": "component" if is_comp else "group",
        "m": [f32hex(x) for x in g.xform.data()] if g.xform is not None else None,
        "material": canonical(g.material),
        "layer": g.layer,
        "locked": bool(getattr(g, "locked", False)),
        "hidden": bool(getattr(g, "hidden", False)),
        "rest": sha(_dumps(canonical({k: _fallback_or(getattr(g, k, None))
                                      for k in type(g).__slots__
                                      if k not in _GROUP_SKIP})), 16),
    }
    if is_comp:
        # A definition is named by its content; which instances share one
        # prototype is kept apart (``shares``), so Make Unique still shows.
        mesh_sig = mesh_signature(proto)
        key = sha(_dumps(mesh_sig), 16)
        defs[key] = mesh_sig
        out["def"] = key
    else:
        out["mesh"] = mesh_signature(proto)
    kids = [_group_signature(c, defs, shares) for c in (g.children or ())]
    out["children"] = sorted(kids, key=_dumps)
    if is_comp:
        shares.setdefault(id(proto), []).append(sha(_dumps(out), 16))
    return out


def scene_signature(scene) -> dict:
    loose = getattr(scene, "_loose_mesh", None) or scene.mesh
    defs: dict = {}
    shares: dict = {}
    groups = [_group_signature(g, defs, shares) for g in (scene.groups or ())]

    def listed(name):
        items = getattr(scene, name, None) or ()
        if isinstance(items, dict):
            items = items.values()
        return sorted(_dumps(canonical(json.loads(_dumps(x)))) for x in items)

    return {
        "mesh": mesh_signature(loose),
        "groups": sorted(groups, key=_dumps),
        "defs": defs,
        "shares": sorted(sorted(members) for members in shares.values()),
        "materials": listed("materials"),
        "layers": listed("layers"),
        "counts": {k: len(getattr(scene, k, None) or ()) for k in (
            "dimensions", "text_labels", "guides", "section_planes",
            "image_planes", "geo_paths")},
    }


def signature_hash(sig: dict) -> str:
    return sha(_dumps(sig))
