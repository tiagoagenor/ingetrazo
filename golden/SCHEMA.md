# The golden corpus — schema

The Python app's behaviour, written down as files that Python and Rust both
read. While Python is alive it writes them; the Rust engine must reproduce
them. Plan: `docs/rust-rewrite-plan.md`, section 5. Schema version: **1**.

```
golden/
  SCHEMA.md            this file
  tolerances.toml      the only places a comparison is not exact
  ledger.csv           every pytest test → where its behaviour now lives
  ops/<id>.json        an op script: a start state and a list of steps
  out/<id>/            what Python produced for it (expected outputs)
```

## 1. Ids

`<id>` comes from the pytest node id it was harvested from:
`<file stem>__<test name>[__<param>]`, lower case, every run of characters
outside `[a-z0-9_]` turned into `-`, cut at 80 characters, then `-` and the
first 8 hex digits of the SHA-1 of the full node id (so a long or
parametrised name stays unique). Example:
`tests/test_pushpull.py::test_push_cap[deep]` →
`test_pushpull__test_push_cap__deep-3f9a01c2`.

A hand-written script uses a name of its own starting with `hand-`.

## 2. Numbers

- Every number is written by Python's `json.dumps` (the Rust writer
  reproduces it byte for byte: `ingetrazo-formats::pyjson`).
- A **point** is `[x, y, z]`: the float32 coordinates of a `QVector3D`
  widened to double, so they read back exactly. A **vector** likewise.
- A **matrix** is the 16 numbers of `QMatrix4x4.data()` (column major).
- Lengths are in metres, angles in degrees, as in the Python commands.
- Where a signature needs a float as a key it is the 8 hex digits of its
  float32 bits, lower case (`1.0` → `3f800000`, `-0.0` → `80000000`).

## 3. Op scripts — `ops/<id>.json`

```json
{
  "schema": 1,
  "id": "test_pushpull__test_push_cap__deep-3f9a01c2",
  "source": {"pytest": "tests/test_pushpull.py::test_push_cap[deep]",
             "app_version": "0.5.8", "harvested": "2026-10-19"},
  "start": {"empty": true},
  "steps": [
    {"op": "add_face", "loop": [[0,0,0],[2,0,0],[2,1,0],[0,1,0]]},
    {"op": "pushpull", "face": {"loop": [[0,0,0],[2,0,0],[2,1,0],[0,1,0]]}, "d": 2.5},
    {"op": "undo"},
    {"op": "redo"},
    {"op": "save", "fmt": "igz"}
  ]
}
```

`start` is one of:

| Form | Meaning |
|---|---|
| `{"empty": true}` | a new scene |
| `{"igz": "<path>"}` | a file in the repository (`examples/…`, `tests/data/…`), opened |
| `{"document": {…}}` | an .igz document (format 1 JSON) inline, opened |

### Steps

A step is an object with an `op` and that op's fields. There are three kinds.

**Command steps** replay one `Command` subclass of the Python app. `op` is
the class name in snake case without `Command` (`AddEdgeCommand` →
`add_edge`, `SetFaceMaterialTagCommand` → `set_face_material_tag`); the other
fields are its constructor arguments by parameter name, with live objects
replaced by selectors (§4). Each class serialises itself with `to_op()` and
is rebuilt with `from_op()`; a class without them cannot be harvested yet.
`compound` holds `"steps": [...]` and runs them as one undo step.
Optional `"in": <group selector>` runs the step inside that group's mesh
(editing context).

**Tool steps** are the operations that reach the history as a
`SnapshotMutation` (a closure, so no constructor to record). The tool tags
the mutation with the op it performed, at the level a user means it:

| op | fields |
|---|---|
| `pushpull` | `face`, `d` (signed metres), optional `copy` (bool) |
| `followme` | `profile` (face), `path` (list of edges) |
| `offset` | `face` or `edges`, `d` |
| `move` | `targets` (selection), `delta` (vector), optional `copy`, `n` |
| `rotate` | `targets`, `origin`, `axis`, `deg`, optional `copy`, `n` |
| `scale` | `targets`, `origin`, `factors` ([sx, sy, sz]) |

A tool not in this table records nothing and its test is not harvested;
adding it means adding a row here.

**Control steps**:

| op | meaning |
|---|---|
| `undo`, `redo` | one history step |
| `undo_all` | undo until the stack is empty |
| `save` | `{"fmt": "igz"}`; optional `"large": true` forces the compact, level-1 path |
| `export` | `{"fmt": "stl" \| "obj" \| "dae" \| "glb" \| "gltf" \| "dxf", "options": {…}}` |
| `import` | `{"file": "<repo path>", "fmt": …, "options": {…}}` |
| `select` | `{"targets": …}` — sets the selection later steps may use as `"targets": "selection"` |

## 4. Selectors

Live objects are named by where they are, never by Python ids. A selector
must match **exactly one** object when the script is harvested; if it does
not, the harvest skips the test (it reports why) rather than write an
ambiguous script.

| Object | Selector |
|---|---|
| vertex | `{"v": [x,y,z]}` — the float32 position, exactly |
| edge | `{"edge": [[x,y,z],[x,y,z]]}` — its two ends, in either order |
| face | `{"face": {"loop": [[…],…]}}` — the outer loop's vertex SET, exactly; `{"face": {"near": [x,y,z], "normal": [x,y,z]}}` — the face whose centroid is nearest, for faces with a loop too long to spell (above 64 vertices) |
| group | `{"group": [i, j, …]}` — index path from `scene.groups` through `children` |
| component definition | `{"definition": <group selector of an instance>}` |
| dimension, label, guide, section plane, image plane, geo path, saved view, sheet | `{"<kind>": i}` — its index in the scene's list of that kind, kind in snake case (`text_label`, `section_plane`, `geo_path`, `scene_view`…) |
| several | a list of selectors, or `"selection"` |

Faces and edges are looked up in the mesh of the step's `in` group, or the
loose mesh.

## 5. Outputs — `out/<id>/`

| File | Contents |
|---|---|
| `steps.jsonl` | one line per step: `{"i": n, "sig": "<L2 hash>"}`, or `{"i": n, "rejected": "<message>"}` when the command raised `CommandRejected` (the model must then be unchanged: same hash as the step before) |
| `start.sig.json` | the full L2 signature before the first step |
| `final.sig.json` | the full L2 signature after the last step |
| `save-<i>.document.json` | L1: the exact `document.json` bytes a `save` step produced |
| `save-<i>.members.json` | L1: every other ZIP member: `{"name": "<sha256>"}`, in archive order |
| `save-<i>.igz` | L1: the whole file, only when the format is 1 or the test asks for byte identity of the container |
| `export-<i>.<fmt>` | L1: the exported file (COLLADA's `<created>`/`<modified>` replaced by `MASKED`) |

After the last step the runner also undoes everything and checks the result
against `start.sig.json`: undo-all must restore the start exactly.

### L1 — bytes

Exact. Deflated members are compared uncompressed (§6 of the plan: whole
containers match only against madler zlib, the Windows reference build).

### L2 — the canonical signature

Order-free, so two models that are the same shape built in a different order
match, and anything else does not. It is a JSON object, written compact with
sorted keys; its hash is the SHA-256 of those bytes, lower-case hex.

```json
{
  "mesh": <mesh signature of the loose mesh>,
  "groups": [<group signature>, …],      // sorted by their own hash
  "materials": [<material>, …],          // as in the document, sorted by name
  "layers": [<layer>, …],                // as in the document, sorted by name
  "counts": {"dimensions": n, "text_labels": n, "guides": n,
             "section_planes": n, "image_planes": n, "geo_paths": n}
}
```

A **mesh signature**:

- `v`: every vertex as `"xxxxxxxx,yyyyyyyy,zzzzzzzz"` (float32 hex), sorted.
  Other entries refer to a vertex by this string.
- `e`: every edge as `[a, b, flags]` with `a < b`, sorted; `flags` is a
  string of the set letters in order — `s` soft, `h` hidden — then
  `@<layer>` when on a layer. Curves: `c<k>` where `k` numbers the curve ids
  by the first edge (in this sorted order) that carries each.
- `f`: every face as `[loop, holes, attrs]`, sorted: `loop` the outer loop's
  vertices rotated to start at the smallest, winding kept; `holes` each hole
  rotated the same way, then sorted; `attrs` the first 16 hex digits of the
  SHA-256 of `json.dumps(face.attrs, sort_keys=True, separators=(",", ":"))`
  (`""` when empty). Interior faces carry `"i"` as a fourth entry.

A **group signature**: `{"name", "kind" ("group" | "component"), "def"
(component definitions numbered by first appearance in this order), "m" (the
transform, 16 float32 hex), "material", "layer", "locked", "hidden", "rest"
(the first 16 hex digits of the SHA-256 of the group's .igz entry without
`mesh`, `children` and `uid`, written with sorted keys — every other field,
such as `ifc`, `billboard`, `axes` or `text3d`, is in there), "mesh" (its mesh
signature, written once per definition: instances carry only "def"),
"children" (sorted group signatures)}`.

### Determinism

Names and ids the app numbers from process-wide counters (`Group N`, curve
ids, uids) would depend on which tests ran before. The harvest resets those
counters before each test, so a script starts from the state of a fresh app;
the Rust runner starts its counters the same way.

## 6. Tolerances — `tolerances.toml`

Comparison is exact everywhere except the entries of this file. Each entry
names what it covers and how far apart is still equal:

```toml
[text3d_outline]   # font outlines differ between FreeType and the Rust rasteriser
applies = "ops whose steps include edit_text3d or insert_group of a 3D text"
coords_m = 1e-4

[georef_network]   # tiles and geocoding come from servers
applies = "ops with a georef step that reaches the network"
skip = true

[render_pixels]
applies = "image outputs"
channel = 2
fraction_equal = 0.995
```

A new entry needs a reason in its comment and is reviewed like code.

## 7. The ledger — `ledger.csv`

```
pytest_id,status,target,reason
tests/test_pushpull.py::test_push_cap[deep],golden,test_pushpull__test_push_cap__deep-3f9a01c2,
tests/test_viewport.py::test_orbit_keys,kittest,ingetrazo-app::orbit::keys,
tests/test_qt_style.py::test_palette,dropped,,Qt-only styling; egui themes are tested in ingetrazo-app
```

`status` is `golden` (target: an op script id), `rust_test` (target: crate
and test path), `kittest` (target: an `egui_kittest` test), `dropped`
(reason required) or `pending`. Python may be deleted (M10) only when no row
is `pending`.

## 8. Compatibility of this schema

`schema` is 1. A reader refuses a higher number. Adding an optional field or
an op is not a new version; changing what an existing field means is.
