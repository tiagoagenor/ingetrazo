# Rust migration plan: from Python kernels to a Rust core

> Status: **in progress** (branch `rust-migration`): Phase 0 done, Phase 1
> under way.
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
   `rust-engine` regressions.

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
- **C4** `Plan.pos32` cached; `curve`/`layer` as int arrays with lazy list
  views (drops the per-edge Python loop of `_plan_native`).
- **C5** `signed_volume` kernel for Entity Info's solid volume of a lazy
  group (`core/bim.py` `face_set_volume`'s sum, f32 Qt order, f64 total),
  closedness from `ef_start` alone.
- **Postponed:** read-only topology queries (they run on meshes being
  edited, which are eager) move to Phase 2; batch vertex moves and
  snapshots on lazy meshes wait for the gate (weld order, viewport cache,
  undo identity and textures would all break).
- **Gate review** against the criteria above.

`core/geometry.py` is left alone (`docs/halfedge-migration-plan.md`, M3).

### Phase 2: geometry operations

`intersect`, `offset`, `sweep`, `fillet`, `section`, `snap`/`inference`, and
the solid booleans of `core/solids.py` (manifold through its C API with
bindgen, validated against the current solids suite). Done when every module
has a green native parity test, Rust is the default, and the corresponding
Python fallback is removed.

### Phase 3: formats and georef

`formats/` and `georef/` (DEM through `ndarray`). **openskp** (.skp) and
**ezdxf** (DXF) stay in Python: no Rust equivalent of the first, and the
`dxf` crate covers far less than the second. Done when `core/ tools/ georef/
formats/` (but SKP/DXF) hold no domain logic in Python and the non-UI tests
run without a `QApplication`. Then: decide on the UI.

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
