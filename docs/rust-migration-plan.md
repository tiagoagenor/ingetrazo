# Rust migration plan: from Python kernels to a Rust core

> Status: **in progress** (branch `rust-migration`): Phases 0, 1 and 2 done
> (numbers in `benchmarks/results/motor-rust.md`); Phase 3 next.
> Written in English per the repo convention (code/docs/comments in English).

## Goal

Move IngeTrazo's domain logic (`core/`, `tools/`, `formats/`, `georef/`) from
Python to Rust, one module at a time, without the application ever being
broken along the way. The Qt user interface (`views/`, ~51k lines) stays in
Python until the core is done; whether it is then rewritten in Rust is a
separate decision taken at the end of Phase 3, with real numbers.

## Strategy (decided)

**Gradual migration through the existing PyO3 crate `rust/ingetrazo_core`.**

- The mechanism already exists: `core/_native.py` turns the engine on/off
  (`INGETRAZO_RUST`, Preferences), every caller keeps its Python code as the
  fallback, and `tests/test_*_native.py` check the two engines agree.
- The ~80k lines of pytest are the behavioural specification. Keeping Python
  alive turns them into a free parity oracle; a from-scratch rewrite would
  throw that away.

Rejected:

- **Full rewrite in pure Rust now** (egui/iced/Slint + wgpu): months with
  nothing usable, no Rust equivalent of openskp, the parity oracle lost.
  `views/viewport.py` + `views/composer.py` alone are ~28k lines.
- **Replacing `QVector3D` with a pure-Python float32 `Vec3`:** a shiboken
  `QVector3D` op costs ~0.1–0.3 µs, a Python one that rounds three
  components ~1.5–2 µs — 5–10× slower in the hover/snap loop (10 ms budget,
  `docs/performance-plan.md` P3) and the Move drag.
- **A `Vec3` pyclass in Rust:** the per-call PyO3 crossing costs about what
  shiboken does, so no gain; it would make Rust mandatory and cannot match
  Qt's `length()` on Windows/macOS yet.

Instead, hot loops move into Rust as whole kernels over float32 NumPy arrays
(the way `core/lazy_mesh.py` and `face_geometry` already work), and Qt math
leaves a module when that module's Python fallback is finally deleted.

## Rust stays optional (for now)

Rust becomes mandatory only once all of these hold:

1. the pull-request CI builds the crate and runs the parity tests (today
   `ci.yml` never builds it, so every native test is skipped);
2. the Flatpak ships the engine (needs vendored offline cargo sources);
3. `qt_hypot()` matches Qt on Linux, Windows and macOS (Linux and Windows
   proven; macOS's libc++ formula awaits a macOS run);
4. one release has shipped with the engine **on by default** and no
   `rust-engine` regressions;
5. every Phase 2 kernel's parity test runs on Windows and Linux CI.

manifold3d stays a Python dependency either way.

Even then, fallbacks are removed one module at a time, eager-mesh mutation
paths last.

## Phases

### Phase 0: safety net and one door for Qt math ✅

- **0.A CI builds the crate.** `ci.yml` installs a Rust toolchain, builds the
  wheel with maturin, asserts `core._native.native is not None` with
  `INGETRAZO_RUST=1`, and runs `tests/test_*_native.py` +
  `tests/test_lazy_mesh_plan.py` with the engine on.
- **0.B `core/vecmath.py`.** Re-exports `QVector2D, QVector3D, QVector4D,
  QMatrix4x4, QQuaternion` from `PySide6.QtGui` (the same objects, so
  `isinstance` and the plugin API keep working), plus `points_to_array` /
  `array_to_points` (float32 (n,3) ⇄ `list[QVector3D]`). The import lines in
  `core/ tools/ formats/ georef/` go through it. `views/` and `plugins/`
  stay as they are: they are the Qt boundary and the third-party API.
- **0.C `tests/test_vecmath.py`.** An AST guard (no module under those four
  packages but `core/vecmath.py` imports those names from `PySide6.QtGui`)
  and bit round-trip tests for the helpers.

Done when: the fast suite gives the same result as before; CI shows the
native tests passing instead of skipped.

### Phase 1: lazy meshes read without being built

**`Plan` is the store** (decided after the first kernels): no `MeshStore`
pyclass. Contiguous NumPy arrays already cross into Rust without a copy,
`Plan` and the face geometry are memoised on the mesh, six readers index
`Plan` directly, and the mutable meshes are the eager ones — a resident
store would need syncing on every edit. New work is stateless kernels over
`Plan` plus Python readers that stop building lazy meshes.

- ✅ **`Hypot` for every platform.** MSVC STL and libc++ formulas beside
  libstdc++; the start-up probe picks `msvc` on Windows
  (`test_hypot_native`). Lazy meshes are drawn from their arrays on
  Windows: opening the synthetic city to its first frame went from 4.3 to
  0.7 s (city-M) and 16.3 to 1.6 s (city-L).
- ✅ **`lazy_mesh.plan`'s topology in Rust** (`plan_topology`: edge dedup,
  rings, `edge.faces`), 407 → 187 ms on a 90 000-face block.
- ✅ **C0** `core.mesh.DEFERRED_BUILDS`, reported per gesture by
  `bench_interaction.py` — the metric for the reads below.
- ✅ **C1** Select hover/click: the owner under the cursor without building
  (`PickEntities.owner_at`).
- ✅ **C2** Drawing tools' hover: the face plane from a stand-alone copy
  (`Viewport._face_under`, `PickEntities.peek`).
- ✅ **C3** Selecting a group: box and counts from the plan
  (`vertex_positions`, `vertex_count`, `face_count` in Entity Info).
- ✅ **C4** `Plan.pos32` cached; `curve`/`layer` as int arrays with lazy list
  views (drops the per-edge Python loop of `_plan_native`).
- ✅ **C5** `signed_volume` kernel for Entity Info's solid volume of a lazy
  group (`core/bim.py` `face_set_volume`'s sum, f32 Qt order, f64 total),
  closedness from `ef_start` alone.
- **Postponed:** read-only topology queries (they run on meshes being
  edited, which are eager) move to Phase 2; batch vertex moves and
  snapshots on lazy meshes wait for the gate (weld order, viewport cache,
  undo identity and textures would all break).
- **Gate review** against the criteria above.

`core/geometry.py` is left alone (`docs/halfedge-migration-plan.md`, M3).

### Phase 2: geometry operations

**Pure compute kernels only.** Python packs what an operation touches into
arrays, Rust computes, Python applies the result through the existing
`Mesh`/`History` API. No Rust-side mesh and no Rust mutation: it would have
to duplicate the weld, `edge.faces`, attributes, textures and undo. Packing
costs ~1–2 µs per point (`points_to_array` 0.89, `array_to_points` 1.2), so
a kernel pays only where Python spends far more per element: pair loops
(n²) or ≥ 20 µs of geometry each. Linear walks over objects do not qualify.

In order:

- ✅ **P0 (Python)** — #541 apply: one incremental cycle adjacency per plan
  instead of `_cycle_adjacency` rebuilt per segment (`topology.py:107`,
  `edits.py:286`); fix the "degenerate edge" at 8 000 crossings. 2 000
  crossings take 4.2 s and 8 000 take 112 s; the cause is the algorithm,
  which no kernel fixes. Target: 8 000 in < 10 s. Plus `Face.normal()`
  memoised per scan.
- ✅ **K1 `arrangement_splits`** — `arrangement.py` `_build_graph`'s all-pairs
  segment splits (1.2 µs a pair: 184 ms for 300 segments), behind push/pull
  cap rebuild, history and the Offset drag. f64 scalar, no hypot. Python
  keeps the graph, face tracing and classification (set order decides loop
  starts).
- ✅ **K2 `section_loops`** — `hlr.py` chord chaining into rings, so the
  section fill keeps up with a plane drag. The fill and hatch stay NumPy.
- ✅ **K3 `intersect_cross`** (`PolySet`) — `intersect.py` per-face polygons and per-pair
  crossing (36 µs a face, 20 µs a pair), the rest of plaza's 8.7 s
  `intersect_find`. Qt `normalized()` with both fuzz branches, through
  `Hypot`; world points still mapped in Python (`QMatrix4x4.map` changes
  arithmetic with the matrix type).
- ✅ **K4 `conform_t_junctions`** — `solids.py`'s O(T·V) near-vertex scan, a
  grid in Rust (the Python twin first rewritten to scalar arithmetic).
- ✅ **K5 `outer_shell_mask`** (`shell_roots`) — the union-find labelling only; last, if K4
  lands.

Not in Phase 2: **booleans** (manifold3d is already native C++, crossed
once per operation; the time is the object glue around it, which a Rust
binding would leave in place — manifold3d stays a Python dependency);
**read-only topology queries** (linear walks: packing costs as much);
**snap/inference** (~10 ms a hover, ~20 rules of callbacks, occlusion
already native); **sweep, fillet, vertex bevel, offset** outside K1 (tiny
compute, the cost is mesh mutation); **`QMatrix4x4.map`** (a parity trap).

Done when P0 is in (8 000 crossings < 10 s, no rollback), K1–K4 each have
an exact `tests/test_<kernel>_native.py` that catches its sabotages, the
numbers are in `benchmarks/results/motor-rust.md`, `bench_engine` gains
`section_fill` and `arrangement_300`, and the fast suite is green on
Windows and Linux CI with the engine on. **Phase 2 removes no fallback**:
that waits for the gate, kernel by kernel, eager-mutation callers last.

### Phase 3: formats and georef

**openskp** (.skp) and **ezdxf** (DXF) stay in Python. Phases 1–2 showed
where time goes: not building Python `Mesh` objects and removing quadratic
algorithms gave the big wins; kernels paid only on pair loops or ≥ 20 µs
per element. Phase 3 applies the same three tests.

In order:

- **A1 (Python)** `_maximal_holes` (`topology.py`): every pair of holes,
  O(H²) on each open and save of a many-holed face — an AABB prefilter
  (inside is tested by centroid), the old version kept as test oracle.
- **A2 (Python)** `simplify_mesh`'s collapse loop (`fuse.py`) rescans from
  the start after every collapse: resume from the lowest neighbour index,
  same "first collapsible in list order" rule.
- **L1 `lzw_decode`** (`georef/geotiff.py`): TIFF LZW decoded code by code
  in Python; byte-equal output.
- **A4 (NumPy)** STL binary read/write, one `struct` call per triangle now.
- **A5 (Python) then K6 `fuse_regions`** (`fuse.py`, ~35 µs a triangle,
  behind OBJ/DAE/SKP-legacy import and solids): first make `_trace_region`
  independent of set order, then port.
- **E1** exports from arrays (`meshexport`, STL/OBJ/glTF/DAE): world
  points mapped per vertex in Python, then the weld and `face_geometry`
  kernels, a small `tri_normals` kernel — byte-identical files.
- **I1** imports produce lazy meshes (glb/skp/stl/OBJ/DAE).
- **K7 `obj_scan`**: OBJ text into arrays (obj.py and photomesh.py).
- **A3** DXF 3DFACE merge cap, **S1** `_share_repeats`, **G1** terrain
  grid in NumPy: each landed or rejected with numbers.

Done when those have landed (or been rejected with numbers), each kernel
has a sabotage-checked parity test, motor-rust.md has city-M/L numbers for
open, save and each import/export, `formats/` and `georef/` reach Qt only
through `core.vecmath` and one image helper, and the non-UI tests run in CI
without a `QApplication`. Linear Python glue stays: packing it would cost
as much as it does.

**Then the gate decides:** Rust mandatory (the five criteria), the order
fallbacks go, and the UI. **Recommendation: no Rust UI rewrite.** Every big
gain came from data structures or algorithms, never from Qt; what is left
in interaction (a click 45 ms, a move's p95 222 ms, undo 99 ms) is the
eager Python `Mesh`/`History`, which a Rust UI would still call. The next
step after the gate is an editable array store with array undo snapshots
(`docs/halfedge-migration-plan.md` M3), measured with `bench_interaction`;
the UI is revisited only if more than half of a frame is then spent in
`views/` Python.

## Rules for every port

- **Fallback.** The Python code stays and is the reference. Callers check
  `native is not None and hasattr(native, "<fn>")` at call time, so an older
  wheel degrades quietly. A kernel that takes a length or normalizes also
  needs `qt_hypot()`.
- **Parity tests.** One `tests/test_<kernel>_native.py` per kernel:
  `INGETRAZO_RUST=1` via `setdefault`, a `skipif` for a missing module,
  `monkeypatch` to force the Python path. Inputs: random, lattice (0.5 grid),
  multi-scale (1e-6…1e5), degenerate. Comparison is exact (`tolist() ==`).
- **Floats.** Contiguous float32 arrays (or float64 holding float32 values)
  cross the boundary, never Qt objects. Rust computes in `f32` in Qt's
  operator order, no `mul_add`/FMA; sums Python does in double (`_newell`)
  are `f64`; length goes through `Hypot`.
- **Determinism.** Output order fixed by index; no `HashMap` iteration order
  reaches a result; rayon only for order-preserving maps.
- **Commits.** One kernel per commit, in English. A speed claim carries
  before/after numbers (`INGETRAZO_PERF`, `bench_interaction`). CHANGELOG
  entries in Spanish under `[Sin publicar]`, in their own «CHANGELOG: …»
  commit.
- **Branch.** All work on `rust-migration`; `main` merged in regularly; back
  to `main` by PR after the fast suite passes on Windows and in CI.

## Estimate

Core (Phases 0–3): roughly 20–40 working sessions; the bottleneck is
validation in the real application, not writing code. A Rust UI, if chosen,
is another 30–60.
