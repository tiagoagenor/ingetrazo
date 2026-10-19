# IngeTrazo in 100% Rust: rewrite plan

> Status: decided, 2026-10-19. This replaces the "no Rust UI rewrite"
> recommendation in `docs/rust-migration-plan.md`. The user's new requirement
> is that nothing from Python is used at runtime or build time: no CPython, no
> PySide6, no PyO3, no Python plugins. Written in English per the repo
> convention.

## 0. Starting point

The Python app has about 118k lines: `views/` 55k (viewport 14.9k, composer
13.2k, main_window 7.5k, tray 4.4k, outliner 4.1k), `core/` 35k, `tools/` 16k,
`formats/` 11k, `plugins/` 4.7k, `georef/` 3.5k. There are about 99k lines of
pytest across 552 files.

The Rust crate `rust/ingetrazo_core` already has about 16 exact kernels,
PyO3-bound with NumPy arrays at the boundary: weld, plan_topology, facegeom
(Qt hypot), arrangement, intersect PolySet, section, hlr, fuse, conform, parity
BVH, occlusion, volume, tiff/LZW, instances and boxes.

"Composer" is the sheet layout window, modelled on the QGIS composer: a
`QGraphicsScene` in paper millimetres holding model-view frames, title blocks,
dimensions, legends, north arrows and terrain profiles. PDF export goes through
`QPdfWriter`; compositions are saved inside the `.igz`.

## 1. Strategy: strangle the core, then replace the UI in one go

- **Core: strangler.** All domain logic moves into pure-Rust crates that run
  headless and are checked continuously against the Python app, the oracle.
- **UI: one replacement.** Qt widgets and an egui window cannot sensibly live
  in one process, and wrapping a Rust core in PySide again would add PyO3 code
  only to delete it later. The Rust app is a separate binary, `ingetrazo2`,
  shipped as a side-by-side preview from M4 and the only app at M10.

**Python feature freeze from M2.** The Python app then takes bug fixes only,
so the oracle stops moving; every new feature is built once, in Rust.

## 2. Target architecture

A Cargo workspace at `rust/`, one crate per concern:

| Crate | Contents |
|---|---|
| `ingetrazo-math` | Exact copies of QVector2D/3D/4D, QMatrix4x4 (including the matrix-type flag paths that change `map` arithmetic) and QQuaternion; `qt_hypot` (libstdc++, MSVC, libc++); f32 in Qt operator order, no FMA |
| `ingetrazo-geom` | The existing kernels without PyO3, plus triangulate (in-house earcut), offset, fillet, sweep, vertex bevel, the HLR rest and simplification |
| `ingetrazo-model` | Editable mesh, Scene, groups/components, layers/tags, materials, dimensions, text, guides, sections, saved views, BIM tags, the composition data model, Command/History |
| `ingetrazo-ops` | edits planner, topology, stitch, push/pull, move/rotate/scale, follow-me, intersect apply, explode, purge, solids |
| `manifold-sys` | bindgen over Manifold's C API (`manifoldc`), built statically with the `cmake` crate |
| `ingetrazo-snap` | Inference/snap engine and picking |
| `ingetrazo-tools` | Tool state machines with no UI code: pointer ray, keys and VCB text in; preview geometry and Commands out; tested headless |
| `ingetrazo-formats` | igz, obj, stl, dae, gltf/glb, dxf in/out, the dwg2dxf bridge |
| `ingetrazo-skp` | Rust port of the openskp reader |
| `ingetrazo-georef` | UTM/datum, DEM, GeoTIFF, slippy tiles, terrain, geocode, geopath |
| `ingetrazo-text` | Font outlines for 3D text and labels |
| `ingetrazo-render` | wgpu renderer: styles, edges, section fill/hatch, shadows, ID-buffer picking, offscreen rendering |
| `ingetrazo-sheets` | Composition items as one display list; screen, PDF, SVG and PNG backends |
| `ingetrazo-script` | Rhai engine and the model API it exposes |
| `ingetrazo-ai` | AI providers and the agent loop |
| `ingetrazo-i18n` | Reads the existing `i18n/*.json` with the same keys |
| `ingetrazo-app` | eframe binary |
| `ingetrazo-cli` | Headless convert, golden replay and benchmarks |
| `xtask` | Packaging and the no-Python audit |

During M1–M9 the old crate shrinks to `ingetrazo_py`, a thin PyO3 shim over
`ingetrazo-geom`, so the Python app keeps working. It is deleted at M10.

### Editable mesh

Halfedge-plan M3 done properly: a **shared-vertex, non-manifold B-rep in
arenas**.

- **Vertices:** `Vec<VertexSlot>` holding `[f32;3]`; ids `VertexId(u32)` with
  a generation counter.
- **Edges:** two `VertexId`s, a radial `SmallVec<[FaceId;2]>` (0..n faces),
  flags (soft, smooth, hidden, curve id, layer).
- **Faces:** outer loop and hole loops as `Vec<VertexId>`, plus an `AttrId`
  into an interned attribute table (the table igz format 3 already writes).
- **Weld:** a grid-hash index keeping the 4-decimal key semantics.
- **Order:** arenas keep insertion order and copy Python's container order
  exactly (list removal shifts, dict insertion order); every L1 golden depends
  on it, and each container's order rule is written down in the source.
- **Groups and components:** definitions shared as `Arc<MeshDef>` with
  copy-on-write; the lazy/eager split disappears, every mesh being arrays.

### Commands and undo

- A `Command` trait: `apply(&mut Scene, &mut Journal) -> Result<(), Rejected>`.
- The mesh journals every primitive mutation (add/remove vertex, edge or face,
  set position, set attribute, relink); undo replays it backwards and puts the
  exact ids back — Python's "undo re-links the very objects" identity.
- Journals replace `SnapshotMutation`: same exactness, far cheaper.
- Live preview is apply-then-revert on a journal, as today.
- Composer keeps its own history, also on the journal.

## 3. UI toolkit and rendering: egui/eframe + wgpu

**egui 0.3x with eframe on wgpu; docking with `egui_dock`, trays and tables
with `egui_extras`, accessibility through AccessKit, native file dialogs with
`rfd`, the macOS menu bar with `muda`, UI tests with `egui_kittest`.**

Why egui: proven at this scale (Rerun is a large docked 3D app on egui +
wgpu); immediate mode fits a viewport-centred app whose UI mirrors the model
every frame (VCB, ScreenTips, inference labels, overlays); all Rust, no DSL,
no C++, no second toolchain; high-DPI, IME, shortcuts and AccessKit covered;
wgpu paint callbacks put our renderer in the viewport rect, on DX12, Vulkan,
Metal or GL, with a software fallback (WARP, llvmpipe) replacing
`gl_fallback.py`.

Rejected: **cxx-qt** (QObject/QML; QWidgets, docks and QGraphicsScene would be
C++ or QML — against "100% Rust" in spirit, three toolchains, Qt's packaging
weight); **Slint** (no docking, weak for a 13k-line canvas editor); **iced**
(only `pane_grid`, less mature text editing and accessibility); **GTK** (poor
on Windows and macOS).

Risks: not a native look (accepted; themed from `styles/`); CJK needs a
bundled Noto Sans CJK subset (~10 MB); no right-to-left (none of the ten
locales needs it); big tables must be virtualised (`TableBuilder` rows lazily).
**Spike gate (week 2):** city-L orbits at 60 fps, the dock layout persists,
zh-CN IME works. If docking fails it moves to `egui_tiles`; the toolkit stays.

## 4. Python dependencies and their replacements

| Today | Rust |
|---|---|
| PySide6 math | `ingetrazo-math` (exact Qt arithmetic, kept for good: changing the numerics would invalidate every golden) |
| numpy | plain `Vec<[f32;3]>` with `bytemuck`; the kernels already use flat slices |
| openskp (.skp read) | **Port the reader** to `ingetrazo-skp` (VFF 2021+ and MFC 2013–2020) with its own test corpus. **.skp write is dropped** (already off). The SketchUp C SDK is rejected (proprietary, Windows/macOS only, GPL-incompatible). |
| ezdxf | The `dxf` crate (MIT) plus our entity-to-mesh logic ported from `dxf_in.py`/`dxf_out.py`/fuse, and a small "recover" pre-pass. DWG stays on the LibreDWG `dwg2dxf` satellite (C). |
| manifold3d | `manifold-sys`, pinned to the wheel's Manifold version |
| trimesh | gone with the openskp port |
| QFont/QPainterPath (3D text, labels) | `cosmic-text` + `ttf-parser` outlines flattened with `lyon_path`; within the §5 tolerance allowlist |
| QImage | `image` crate plus our TIFF/LZW |
| QPdfWriter/QPainter (sheets) | `krilla` (PDF), `tiny-skia` (PNG), our SVG writer, from one display list |
| network (tiles, geocode, AI, catalog) | `ureq` + `rustls` on worker threads, results over channels |
| json | `serde_json` plus a Python-compatible writer for igz (§6) |
| zipfile | `zip` with `flate2` on static madler zlib (`libz-sys`) |
| QSettings / layout state | TOML in the platform config dir (`directories`); old Qt layouts not migrated |

## 5. Behavioural parity: the golden corpus

The pytest suite is the spec; it becomes **language-neutral golden files**
while Python is alive.

**Op scripts** (`golden/ops/*.json`): lists of steps at the **Command layer**,
not mouse events — e.g. `{"op": "pushpull", "face": {"near": [x,y,z]}, "d":
2.5}`, `{"op": "undo"}`, `{"op": "export", "fmt": "stl"}`. Selectors are
positional (nearest centroid or vertex).

**Recorder:** a Python `History` recorder serialises each Command's
constructor parameters; the pytest plugin `--golden-harvest` writes an op
script and its outputs for every test whose commands all serialise. Tools get
synthetic ray/key event streams for `ingetrazo-tools`; snap tests record
(camera, pixel) → (kind, point, label).

**Outputs** per step and at the end: **L2 canonical signature** (order-free:
sorted f32 bit-hex vertices, sorted edges + flags, rotation-canonical loops +
holes + attribute hash, group tree, materials, layers; undo-all must restore
the initial signature) and **L1 exact bytes** (igz JSON, every ZIP member,
export files; COLLADA timestamps masked).

**Comparison** is exact; the only tolerances are listed in
`golden/tolerances.toml`: 3D text outlines (1e-4 m), network georef, render
pixels (±2 per channel, 99.5 % of pixels).

**Runner:** `cargo test -p ingetrazo-golden`; CI blocks any change that turns a
passing golden red.

**Hand-ported tests:** UI behaviour, messages and edge cases — domain tests as
Rust `#[test]`, UI tests as `egui_kittest` over AccessKit. A ledger
(`golden/ledger.csv`: pytest node id → golden id, Rust test, or "dropped" with
a reason) tracks progress; Python may be deleted only when every row is filled.

## 6. .igz compatibility

All three igz formats read and write byte-compatibly:

- **A JSON writer reproducing `json.dumps`:** key order as the Python dicts
  were built; `(",",":")` separators for large documents, `indent=2`
  otherwise; `ensure_ascii` with surrogate pairs; `allow_nan=False`; **Python
  float repr** (shortest round-trip digits via `ryu`, then Python's formatting).
- **ZIP:** same member order, fixed 1980-01-01 timestamps, DEFLATE level 6
  (level 1 for `fast`), STORED blobs as today.
- **Deflate:** uncompressed member bytes always identical; whole-container
  identity against the Windows reference build (madler zlib, the backend we
  link) — not required on zlib-ng Linux builds, where Python differs too.
- **Forward compatibility:** unknown keys, `group.ext`, plugin data and
  compositions round-trip verbatim as `serde_json::Value`.
- **Fixtures:** every `.igz` in `examples/` and `tests/` round-trips
  byte-identical from M2 on.

## 7. Roadmap

Effort in person-weeks (PW) for one AI-assisted developer, realistic ranges.

| M | Content | Verifiable outcome | PW |
|---|---|---|---|
| **M0** | Workspace split; golden recorder, harvester, schema; CLI replay skeleton; UI spike | Kernel parity green through the shim; first 500 goldens; spike numbers | 3–4 |
| **M1** | `ingetrazo-math`; `ingetrazo-geom` with every remaining compute module | Geometry goldens green | 3–5 |
| **M2** | `ingetrazo-model`: mesh, scene, groups, materials, journal undo; igz in all 3 formats. **Python feature freeze.** | Fixture `.igz` byte-identical; signatures equal Python's | 6–9 |
| **M3** | `ingetrazo-ops` + `-snap` + `-tools`: edits, topology, stitch, push/pull, transforms, follow-me, intersect apply, solids, section, explode, purge | ≥ 90 % of goldens green; headless gestures faster than Python | 12–18 |
| **M4** | `ingetrazo-render` + minimal app: viewport, orbit/pan/zoom, select, line/rect/circle/pushpull/move, undo, open/save | **Alpha preview** beside 1.x; city-L at 60 fps | 6–9 |
| **M5** | Formats: obj/stl/gltf/glb/dae/dxf in and out, dwg bridge, **skp reader port** | Byte-identical exports; equal import signatures | 6–10 |
| **M6** | Full UI: main window, docking, tray, outliner, entity info, materials/library, preferences, F3 search, shortcuts, hints, VCB, all tools, styles, i18n, themes | UI ledger ported; **beta** | 10–16 |
| **M7** | georef: location, tiles, DEM terrain, geotiff, geocode, geopath, profile | georef goldens green | 3–5 |
| **M8** | Rhai console and scripts; AI assistant on Rhai | Console tests ported | 3–5 |
| **M9** | Packaging: Inno/MSI, notarised .app, AppImage, Snap, Flatpak (`cargo vendor`); no-Python audit | Artifacts pass the audit | 2–4 |
| **M10** | **IngeTrazo 2.0**: Rust only; **Python deleted from `main`** (shim and harvester too; goldens stay as files) | Ledger complete, goldens green | 1–2 |
| M11 (2.1) | Sheet composer + PDF/SVG/PNG | Composer rows ported | 6–10 |
| M12 (2.2) | WASM extensions | Two bundled examples as WASM | 4–6 |

**Totals:** about 55–90 PW to 2.0; 65–105 with 2.1 and 2.2.

**Critical path:** M0 → M1 → M2 → M3 → M4 → M6 → M10. M3 (topology, edits,
push/pull parity) is the biggest risk and block; M5, M7 and M8 slot between
UI chunks.

**Python ends at M10.** 1.x gets security and data-loss fixes for six months
after 2.0, so composer users have somewhere to go until 2.1.

## 8. Cut, deferred or simplified

- **Composer/sheets → 2.1.** 13k lines of QGraphicsScene; compositions are
  kept verbatim in the `.igz` and shown read-only in 2.0.
- **Python extension API → dropped at 2.0, replaced at 2.2** by sandboxed WASM
  components on `wasmtime` with a WIT interface (tools, menu actions, document
  data, snap providers, pickables, outliner kinds, vector overlays,
  declarative panels). The catalog gains a `wasm` artifact field; Python
  extensions are refused with a "needs a 2.x build" message; a porting guide
  maps every `ExtensionApp` call; `model_info` and `solid_inspector` become
  built in.
- **Python Console → Rhai console in 2.0** (one undo step per run, rollback on
  error, the model API under today's names;
  `scripts/create_architectural_showcase.py` ported as the reference).
- **AI assistant → M8, on Rhai;** `ai_bridge` (MCP) deferred to 2.1.
- **render_blender → cut** (it ships a `.py` run inside Blender).
- **Also:** .skp export dropped; IFC unplanned; `photomesh` and 3D-mouse
  (ndof) deferred to 2.1; old Qt layouts not migrated.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Long time with nothing to ship | Alpha at M4, beta at M6; 1.x keeps bug fixes |
| Python container and float order hard to copy | L2 required for every golden; order rules per container; replay diff shows the first diverging step |
| `QMatrix4x4.map` flag-type arithmetic | Copy Qt's `flagBits` paths; property tests against the shim until M10 |
| egui gaps (tables, docking, IME) | Spike gate in week 2; virtualised tables; small upstream patches |
| skp port size and upkeep | Port against openskp's corpus plus our goldens; contribute upstream |
| GPU and drivers | wgpu order DX12 → Vulkan → GL → WARP/llvmpipe; keep `glinfo` |
| Manifold version drift | Pinned; a golden checks a boolean on the reference corpus |
| Oracle moves | Feature freeze from M2; goldens record the Python commit |
| Flatpak/Snap offline builds | `cargo vendor` tarball from xtask; manifold from vendored source |
| Single-developer bus factor | Docs per crate; goldens describe behaviour independently of either codebase |

### Definition of done for "no Python" (CI job `xtask audit-no-python`, required on release tags)

1. The release workflow runs on runners **with no Python interpreter**; any
   build dependency that needs Python is replaced.
2. `cargo tree --workspace` has no `pyo3`, `numpy` or `pyo3-*`.
3. Every artifact (Inno via `innoextract`, .app, AppImage, Snap, Flatpak) is
   unpacked and scanned; the audit fails on any `*.py`, `*.pyc`, `*.pyd`,
   `*.pyz`, `python3*.dll`, `libpython*`, `PySide6`, `shiboken*`, and any
   binary containing `PyInit_` or `Py_Initialize`.
4. `main` has no `*.py`, `requirements.txt` or `pytest.ini`; a CI grep enforces
   it.

## 10. First two weeks

1. Turn `rust/` into a workspace; move the kernels from `ingetrazo_core` into
   `ingetrazo-math` and `ingetrazo-geom` with no `pyo3`; rename the PyO3 crate
   to the `ingetrazo_py` shim. `tests/test_*_native.py` stay green.
2. Write `golden/SCHEMA.md`: op-script v1, signature v1 (L1/L2), tolerance
   allowlist, ledger format.
3. Python: a `History` recorder and `to_op()` on every Command; the
   `--golden-harvest` plugin in `tests/conftest.py`; harvest the first 500
   goldens from the mesh, history and pushpull tests.
4. `ingetrazo-cli golden replay` with comparison and first-divergence diff,
   in CI on Windows, Linux and macOS.
5. The Python-compatible JSON writer with property tests against Python's
   output, and a format-1 igz round-trip of every fixture.
6. UI spike (own branch): eframe + egui_dock + wgpu drawing a format-3 city-L
   igz, orbit, a persisting dock layout, zh-CN IME entry, an AccessKit dump;
   apply the §3 gate.
7. CI: `cargo test --workspace` and clippy on three operating systems.

### Critical files

- `core/history.py`, `core/mesh.py`, `formats/igz.py`, `tools/pushpull.py`
- `rust/crates/ingetrazo-geom` (the kernels) and `rust/crates/ingetrazo_py/src/lib.rs` (the shim)
- `views/viewport.py`, `views/extension_api.py`
