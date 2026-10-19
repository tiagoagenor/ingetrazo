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
  "id": "test_faces__test_draw_and_flip-3f9a01c2",
  "source": {"pytest": "tests/test_faces.py::test_draw_and_flip",
             "app_version": "0.5.8", "harvested": "2026-10-19"},
  "start": {"igz": "golden/starts/5c1d0e9a7b3f2e41.igz"},
  "steps": [
    {"op": "add_face",
     "args": {"vertices": [{"$v": [0,0,0]}, {"$v": [2,0,0]}, {"$v": [2,1,0]}, {"$v": [0,1,0]}]},
     "counters": {"group": 1, "curve": 1}, "sig": "9c41…"},
    {"op": "flip_faces",
     "args": {"faces": [{"$face": {"loop": [[0,0,0],[2,0,0],[2,1,0],[0,1,0]], "in": null}}]},
     "counters": {"group": 1, "curve": 1}, "sig": "07d2…"},
    {"op": "undo", "ok": true, "sig": "9c41…"},
    {"op": "redo", "ok": true, "sig": "07d2…"}
  ]
}
```

`start` is one of:

| Form | Meaning |
|---|---|
| `{"empty": true}` | a new scene |
| `{"igz": "<path>"}` | an .igz of the repository, opened: `examples/…`, or `golden/starts/<hash>.igz` — the test's own setup, saved by the app when its first command ran, stored once by content (first 16 hex digits of its SHA-1) beside `<hash>.sig.json`, its full L2 signature |

### Steps

A step is an object with an `op`. Every step carries `sig`, the L2 hash of
the scene after it (§5); `undo`/`redo` carry `ok`, what History returned.

**Command steps** replay one `Command` subclass of the Python app. `op` is
the class name in snake case without `Command` (`AddEdgeCommand` →
`add_edge`, `SetFaceMaterialTagCommand` → `set_face_material_tag`). `args`
holds the constructor arguments by parameter name — nested, because one of
them is itself called `op` (`SolidOperationCommand`) — encoded as in §4.
`counters` are the process-wide counters as they stood when the command ran
(see Determinism, §5); the runner sets them first. `rejected` holds the
`CommandRejected` message when the command declined (the model is then
unchanged). The recorder (`tests/golden/recorder.py`) captures every class
generically: the constructor's arguments are its `to_op()`, and `from_op()`
calls the constructor again with them decoded.

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

Until a tool tags its mutation (and can be driven without a window, M3),
a script that holds one is not harvested.

**Control steps**:

| op | meaning |
|---|---|
| `undo`, `redo` | one history step |
| `undo_all` | undo until the stack is empty |
| `save` | `{"fmt": "igz"}`; optional `"large": true` forces the compact, level-1 path |
| `export` | `{"fmt": "stl" \| "obj" \| "dae" \| "glb" \| "gltf" \| "dxf", "options": {…}}` |
| `import` | `{"file": "<repo path>", "fmt": …, "options": {…}}` |

## 4. Argument values and selectors

Plain JSON values (null, booleans, numbers, strings, lists, objects with
plain string keys) stand for themselves. Everything else is a one-key object
whose key starts with `$`:

| Form | Value |
|---|---|
| `{"$v": [x,y,z]}` | a `QVector3D` (float32 widened, so exact) |
| `{"$m": [16 numbers]}` | a `QMatrix4x4`, `data()` order (column major) |
| `{"$color": "#aarrggbb"}` | a `QColor` |
| `{"$t": [...]}` | a tuple |
| `{"$set": [...]}` | a set (items sorted by their JSON) |
| `{"$dict": [[k, v], ...]}` | a dict whose keys are not all plain strings |
| `{"$obj": "<class>", "data": {...}}` | a scene object passed by value (a new guide, section plane, label…): its `to_dict()`, rebuilt with `from_dict()`; `"Dimension"` holds `a`, `b`, `offset`, `layer`, `text`, `axis` |
| `{"$cmd": <op>}` | another command, run by this one (`CompoundCommand`, `SnapshotCompound`, `MeshSnapshotCommand`…); it is rebuilt when it runs, so its selectors see the scene as it is then |
| `{"$ran": n}` | a command that already ran in this step (the op holding `"rid": n`) — the same instance, as `AutoTagDrawnFacesCommand` reads the faces its siblings made |

Live objects are named by where they are, never by Python ids. A selector
must match **exactly one** object when the script is recorded; if it does
not, the recorder marks the step and the test is not harvested.

| Object | Selector |
|---|---|
| face | `{"$face": {"loop": [[x,y,z], …], "in": <group path or null>}}` — the outer loop's vertex SET, exactly (the list keeps the face's own order) |
| edge | `{"$edge": {"ends": [[x,y,z],[x,y,z]], "in": …}}` — its two ends, in either order |
| vertex | `{"$vertex": {"at": [x,y,z], "in": …}}` |
| group | `{"$group": [i, j, …]}` — index path from `scene.groups` through `children` |
| kept component | `{"$kept": i}` — index in `scene.kept_components` |
| mesh | `{"$mesh": <group path or null>}` — a group's mesh, or the loose one |
| annotation, guide, plane, path, view, layer | `{"$ref": ["<scene list>", i]}` — the list is one of `dimensions`, `text_labels`, `guides`, `section_planes`, `image_planes`, `geo_paths`, `geo_points`, `saved_views`, `compositions`, `layers` |

`in` is the group whose mesh holds the face, edge or vertex (the first group
of a shared component mesh); `null` is the loose mesh.

## 5. Outputs — `out/<id>/`

| File | Contents |
|---|---|
| `final.sig.json` | the full L2 signature after the last step (absent when the test changed the scene after its last command) |
| `save-<i>.document.json` | L1: the exact `document.json` bytes a `save` step produced |
| `save-<i>.members.json` | L1: every other ZIP member: `{"name": "<sha256>"}`, in archive order |
| `save-<i>.igz` | L1: the whole file, only when the format is 1 or the test asks for byte identity of the container |
| `export-<i>.<fmt>` | L1: the exported file (COLLADA's `<created>`/`<modified>` replaced by `MASKED`) |

After the last step the runner also undoes everything and checks the result
against the start's signature: undo-all must restore the start exactly.

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
  "groups": [<group signature>, …],      // sorted by their JSON
  "defs": {"<def hash>": <mesh signature>, …},
  "shares": [["<instance hash>", …], …],  // instances sharing one prototype
  "materials": [<material JSON>, …],     // sorted
  "layers": [<layer JSON>, …],           // sorted
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

A **group signature**: `{"name", "kind" ("group" | "component"), "m" (the
transform, 16 float32 hex, or null), "material", "layer", "locked",
"hidden", "rest" (the first 16 hex digits of the SHA-256 of the group's
other fields as the .igz writes them — `axes` (16 floats), `ifc`, `text3d`,
`exploded`, `billboard`, `ext`, `explode_offset`, each only when set —
with sorted keys), "mesh" (a plain group: its mesh signature) or "def"
(a component instance: the first 16 hex digits of the SHA-256 of its
prototype's mesh signature, which is written once under `defs`), "children"
(sorted group signatures)}`. `shares` groups the instances (by the first 16
hex digits of the SHA-256 of their signature) that share one prototype
object, each list sorted, the lists sorted: two copies of a component are
one list, a copy made unique is a list of its own.

### Determinism

Names and ids the app numbers from process-wide counters (`Group N`, curve
ids) depend on what ran before in the same process. Each command step
therefore records them as they stood when it ran (`counters`: `group`, the
next default group number; `curve`, the next curve id), and the runner sets
them before running the step. Random uids are left out of signatures.

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
