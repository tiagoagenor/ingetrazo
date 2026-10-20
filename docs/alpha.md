# The Rust alpha (M4.9)

The Rust application (`rust/crates/ingetrazo-app`, the `ingetrazo` binary)
is the alpha preview that ships beside the Python 1.x app. M4.9 is the
milestone "city-L at 60 frames per second": window, viewport, navigation,
the tools, undo/redo and open/save (M4.3 to M4.8) had to be fast on the
largest synthetic model, not only correct.

This page has the numbers, how to reproduce them, what was changed to get
them, where the application looks for its data, and what is still missing.

## How to run the benchmark

The models are generated (and git-ignored):

```bash
.venv/Scripts/python.exe scripts/bench_models.py M L      # -> benchmarks/models/city-M.igz, city-L.igz
```

| model | objects | faces (as the Python app counts them, copies included) | `.igz` |
|---|---|---|---|
| city-M | 314 | 73 102 | 0.55 MB (components are stored once) |
| city-L | 1 456 | 331 124 | 2.3 MB |

Then, from `rust/`:

```bash
cargo build --release -p ingetrazo-app
target/release/ingetrazo --bench-orbit 10 ../benchmarks/models/city-L.igz
```

`--bench-orbit <seconds> file.igz` runs the real application loop (window,
egui, the renderer, the GPU) with synthesized `session::Input` events, one
step per frame, and quits printing `key value` lines on stdout. It implies
`--novsync` (with vsync every frame waits for the monitor and hides the
cost). The script (`ingetrazo-app/src/bench.rs`):

1. waits 30 frames (the model is baked, the framing is done);
2. **orbit**: a middle-button drag that keeps turning for `<seconds>`
   (the press asks the model for its pivot, as a real press does);
3. **hover**: the pointer sweeps the screen for 2 s with Select, then for 2 s
   with Line (the inference engine: snapping to vertices, edges, midpoints);
4. **edits**: a line, a rectangle, a click on a building (Select), that
   building moved (Move: base point, target) and the undo of the move.

What it prints:

| line | meaning |
|---|---|
| `open_ms` | reading the `.igz` and building the scene (the call `App::open`) |
| `first_frame_ms` | the longest renderer `prepare` of the first 30 frames: baking every mesh |
| `orbit_press_ms` | the button press that starts the orbit |
| `<phase>_frame_ms p50 p95 p99 max` | interval between two frame starts (what the eye sees); `_fps_p50` is `1000 / p50` |
| `<phase>_ui_ms`, `<phase>_prepare_ms` | CPU time of the window's `ui` pass, and of the renderer's sync + plan + encode |
| `hover_handler_ms`, `hover_line_handler_ms` | the pointer-move handler alone (pick, snap, tool) |
| `edit_<name> handler_ms … edit_to_next_frame_ms …` | the committing event's handler, and the time from just before it until the next frame *starts* (edit to screen). `NOT_APPLIED` when the history did not grow |
| `memory_mb` | resident set of the process at the end of the run |

`INGETRAZO_BENCH_TRACE=1` also prints each injected input with its time and
every frame over 50 ms. The target is p95 of the orbit frame at or under
16.7 ms (60 fps); the other gestures are judged against the same budget.

## Numbers

Tiago's machine: Windows 11, Xeon E5-2680 v4 (28 threads), RTX 3060, Vulkan,
release build, 1280 x 800 window. Other programs were running (other agents
building the repository), so each Rust figure is the median of three runs
of the benchmark (the runs differ by less than 10 %, except the stalls noted).
"Before" is the app as M4.5 to M4.8 left it, measured with the same script
(the Line hover and the click/undo gestures were added later and have no
"before" except where noted).

### Rust app, before and after

| | city-M before | city-M after | city-L before | city-L after |
|---|---:|---:|---:|---:|
| open (`open_ms`) | 476 ms | 420 ms | 1 886 ms | 1 740 ms |
| first frame (bake) | 119 ms | 57 ms | 184 ms | 218 ms |
| **orbit** frame p50 / p95 | 1.43 / 1.55 ms | 1.62 / 1.73 ms | 2.45 / 2.63 ms | **2.75 / 2.95 ms** |
| orbit, fps at p50 | 697 | 616 | 408 | **365** |
| orbit, worst frame | 258 ms (first press) | 5 ms | 994 ms (first press) | 16 ms |
| hover, Select: frame p50 / p95 | 7.0 / 19.2 ms | 2.1 / 2.2 ms | 76.7 / 79.7 ms (13 fps) | **8.2 / 9.8 ms** |
| hover, Select: handler p50 | 5.3 ms | 0.5 ms | 73.7 ms | 5.3 ms |
| hover, Line: frame p50 / p95 | (not measured) | 2.2 / 3.4 ms | ≈ 400–560 ms per move (traced) | **9.2 / 12–17 ms** |
| edit to next frame: line | 149 ms | 8 ms | 772 ms | **28–35 ms** |
| edit to next frame: rectangle | 146 ms | 5 ms | 636 ms | **25 ms** |
| edit to next frame: move a building | not applied | 6 ms | not applied | 26–50 ms |
| click on a building (after the edits above) | | 62 ms | | 240 ms |
| undo, edit to next frame | | 2 ms | | 5 ms |
| memory (resident) | 205 MB | 234 MB | 370 MB | 530 MB |

The orbit is a hair slower than in the first run (the frame is now drawn
from the window's `ui` pass and submitted on its own); both are an order of
magnitude under the budget. `first_frame_ms` of city-L is the same bake as
before (one frame, 220 ms, once per document).

### The Python app on the same machine

`scripts/bench_interaction.py out.json benchmarks/models/city-M.igz
benchmarks/models/city-L.igz --repeat 1` (Python 3.12, PySide6 6.11.2, vsync
off, once). It times each Qt event until its frame is painted and the GPU
has finished (`glFinish`), so a figure is what the hand waits for; the Rust
figures above are the interval between frames, the nearest equivalent. The
recorded baseline of the release history
(`benchmarks/interaction-baseline.json`, a Ryzen with an RTX 3070 Ti, 1 October
2026) agrees within noise.

| gesture (median / p95, ms) | city-M Python | city-M Rust | city-L Python | city-L Rust |
|---|---:|---:|---:|---:|
| open | 4 160 | 420 | 14 920 | 1 740 |
| orbit | 8.6 / 13.3 | 1.6 / 1.7 | 5.5 / **20.9** | **2.8 / 2.9** |
| pan, zoom | 5.8 / 7.6, 6.1 / 8.6 | (same path as orbit) | 5.4 / 8.6, 5.6 / 9.0 | (same path as orbit) |
| hover, Select | 8.1 / 40.0 | 2.1 / 2.2 | 20.9 / 57.3 (max 957) | 8.2 / 9.8 |
| hover, Line | 43.9 / 213 | 2.2 / 3.4 | 74.4 / 298 | 9.2 / 12–17 |
| click on a building | 31.4 / 78.6 | 62 | 76.9 / 165 | 240 |
| move a building (per event) | 40.4 / 114 | 6 | 53.6 / 276 | 26–50 |
| undo | 154 | 2 | 299 | 5 |

The Rust app meets the target on city-L: the orbit is at 365 fps with p95 at
2.95 ms (the Python app's orbit already drops 14.5 % of the frames over
16 ms there), and the hover and the edits are inside one 60 Hz frame except
the first click after a series of edits (below).

## What was changed

* **The scene is no longer copied to the renderer.** `GpuView` used to hold an
  `Arc<Scene>` that the window re-cloned after every change (125 ms on
  city-L). The frame is now drawn from the window's `ui` pass
  (`GpuView::render_now`) with a `&Scene` borrowed from the session; the
  renderer rebakes only the meshes whose `serial` moved, as before. The egui
  paint callback only blits the picture.
* **The snap world and the pick index survive an edit.** Both were rebuilt
  from the whole model after every change: every mesh triangulated again, once
  per placement for the snap world. What does not depend on where a mesh is
  placed (triangles, areas, planes, hard edges, arc midpoints) is now kept by
  mesh, valid while the mesh's serial and arena sizes stand (`WorldCache`,
  `ChunkCache`), shared by all copies of a component and rebuilt only for the
  meshes an edit touched. A cold build and the per-placement flattening run in
  parallel (rayon), batch by batch, and are appended in the same order as
  before: the result is bit for bit the old one
  (`ingetrazo-app/tests/perf_equivalence.rs` compares the cached and the plain
  builds before and after an edit, on `examples/pileta-fuente-yanque.igz`).
  `World` holds shared `Arc<Mesh>` copies instead of a deep clone.
* **Rays are culled.** From 4096 triangles on, a ray tests the triangles of
  the runs (at most 96 consecutive triangles of one placement) whose box it
  meets (`pick::TriGrid`), instead of all 737 000 of city-L. The test on each
  triangle is the old one, so the answers are the same bits; below the
  threshold (all the parity fixtures) the old scan runs. A test compares both
  on thousands of rays. The pick index's own `hover_face_t` reads only the
  entities the ray hit.
* **Hard edges are looked up, not scanned.** A hover projected and measured
  every group hard edge several times (708 792 on city-L). The projection is
  kept per camera and the edges are binned by the screen cells they cover
  (`screen_grid::ScreenGrid`); the edge picks, the snap scene, the vertex
  pick and the Select tool's group pick look only at the edges near the
  cursor, in index order, with the old arithmetic. A test checks the grid
  reaches every edge within the threshold of a lattice of pixels at several
  cameras.
* **The first pointer move does not stall.** Building the world and the pick
  index (0.3 s on city-L) happened on the first hover; `Session::warm` now
  does it two frames after a document is opened.
* **Open-time work in the window is measured**: `App::with_bench`, `bench.rs`.

The GPU ID-buffer picker (`ingetrazo_render::pick`, M4.2) was **not** wired
into the hover: the CPU picker, with the changes above, is inside the budget
on city-L (Select hover 5 ms of handler), and keeping the CPU picker as the
only arbiter keeps the priority and threshold rules of the Python app
untouched. The picker's module documentation says how to combine it when a
model needs it (city-XL, a million faces, has not been measured).

## Where the application finds its data

`ingetrazo_app::paths::app_root()` (one function, used for every file of the
install; `paths::resource_path("resources/textures/x.png")`) answers with the
folder that *contains* `resources/` (the same contract as `core/paths.py`):

| layout | executable | root |
|---|---|---|
| Windows installer / portable | `ingetrazo\ingetrazo.exe` | its own folder |
| Linux `PREFIX/lib/ingetrazo/`, AppImage, Snap, Flatpak | `lib/ingetrazo/ingetrazo` | its own folder |
| the Linux `bin/ingetrazo` wrapper (or a symlink to the binary) | `bin/ingetrazo` | `../lib/ingetrazo` (the link is followed first) |
| macOS bundle | `IngeTrazo.app/Contents/MacOS/ingetrazo` | `Contents/Resources` |
| `cargo run` / `cargo build` | `rust/target/{debug,release}/ingetrazo` | the nearest folder above that holds `resources/`: the repository root |

`INGETRAZO_HOME` (a folder holding `resources/`) overrides all of it. If no
layout matches, the folder of the executable is used and a missing resource
is an ordinary missing file. The images a document names by a path on disk
(the texture library) are read through it (`paths::InstallTextures`, also
what `Save` consults); the DWG bridge of `ingetrazo-formats-mesh` is meant to take
`app_root()` as its `from_app_root` argument once the app imports DWG (`vendor/libredwg/bin` beside
`resources/`, as `packaging/windows/build.ps1` stages it). The window sets
its app id to `IngeTrazo`, which is the Wayland app id and the X11
`WM_CLASS` that the `.desktop` files match with `StartupWMClass=IngeTrazo`.
`paths.rs` tests every row of the table with made-up folders, and that this
checkout resolves to its own repository.

## Known gaps

* **The first pointer move after an edit rebuilds the world and Select's pick
  index** (they are rebuilt from the model, now mostly from kept pieces): on
  city-L that is a stall of about 0.2 s (the `click_select` row; 60 ms on
  city-M). The edit itself is visible in 25 to 50 ms. A true incremental
  update (keep the placed part of the arrays and splice the loose part, which
  comes first) would remove it.
* **Memory grew** from 370 to 530 MB on city-L: the kept prototypes, the
  world, two pick indexes and the projection caches. Scenes of a million faces
  (city-XL) have not been tried.
* **Push/pull, Move and the other tools inside a group** are not part of the
  script (a click on a building selects its group; entering the group for a
  push/pull of one of its faces is a double click). The tools themselves are
  the ones of M4.5 to M4.8 and are covered by `tests/session_flow.rs`.
* **Open takes 1.7 s** on city-L (reading, expanding the document, building
  the scene: 8 times faster than Python, not optimized further) and the first
  frame bakes for 0.2 s.
* The numbers are from one machine under load; macOS and Linux were not run
  (the benchmark reads the resident set from `/proc` on Linux; there is no
  figure on macOS).
* The Python figures are one `--repeat 1` run of `scripts/bench_interaction.py`
  and time events, the Rust ones are frame intervals; they are comparable in
  order of magnitude, not to the millisecond.
