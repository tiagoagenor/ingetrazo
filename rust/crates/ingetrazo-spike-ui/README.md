# ingetrazo-spike-ui (M0.8, throwaway)

UI feasibility spike for the Rust rewrite (plan sections 2, 3, 10.6): eframe on
wgpu + `egui_dock` + a custom `egui_wgpu` paint callback. Not production code.

## What it shows

- Docked layout (egui_dock): "Outliner" | "3D Viewport" | "Entity info"; tabs are
  draggable and re-dockable, Window menu re-opens closed panels.
- Loads `examples/arco-yanque.igz` (the largest .igz in the repo, 3.6 MB) via
  `ingetrazo-formats` (`igz::read_document`), draws it with wgpu: flat-shaded
  triangles (face normal, two-sided) + edges as depth-tested lines.
  Groups with `proto` + 16-float column-major `xform` are expanded.
- Menu Model > "Synthetic stress scene": 45x45 = 2025 extruded 128-gon prisms,
  1,028,700 triangles + 777,600 edges. (No city-L .igz/.skp exists in the repo.)
- Orbit (left drag), pan (middle drag or shift+left drag), zoom (wheel).
- Status bar: fps / frame ms, UI cpu ms, triangles, edges, groups, wgpu backend.
- Entity info has a text field for IME / typing tests.

**Update (M4.1):** the viewport no longer has the spike's throwaway renderer
(fan triangulation, no colours, 1 px lines on egui's depth buffer). It draws
through `ingetrazo-render`: the renderer owns its multisampled targets, renders
into its own texture in the paint callback's `prepare`, and `paint` blits that
texture into egui's pass (`ingetrazo_render::blit`). The renderer does not
depend on egui. The measurements below are the ones of the original spike;
with the real renderer (4x MSAA, textures, instancing, profiles) the same
scenes take 1.40 ms/frame (arco-yanque.igz, 26,880 triangles, 77 draw calls)
and 2.15 ms/frame (the stress scene, now 2,025 placements of one 128-gon
prism: 6 draw calls) on the same machine, no vsync, `--bench 4`. The stress
scene's old numbers (1.27 ms) were for static vertex buffers with no MSAA.

Remaining simplifications: no picking (M4.2); the dock layout is not
persisted (egui_dock has a `serde` feature; not wired in the spike).

## Run

```
cd rust
cargo build -p ingetrazo-spike-ui --release
# from the repo root (the default model path is relative):
rust/target/release/ingetrazo-spike-ui.exe [--model PATH] [--stress] [--novsync]
                                           [--bench SECONDS] [--accesskit-dump]
```

`--bench N` auto-orbits with continuous repaint, skips 30 warm-up frames, prints
one `BENCH ...` line and exits. `WGPU_BACKEND=dx12|vulkan` forces a backend.

## Measurements

Windows 11, NVIDIA GeForce RTX 3060, release build (thin LTO), 1280x800 window,
`--bench` (continuous repaint, camera orbiting). Frame = wall-clock interval
between frames.

| Scene | Tris | Backend | Present | Frame mean / p95 / max | FPS |
|---|---|---|---|---|---|
| arco-yanque.igz | 26,806 (+23,798 edges) | Vulkan | vsync | 16.68 / 16.98 / 19.67 ms | 59.9 |
| arco-yanque.igz | 26,806 | Vulkan | no vsync | 1.27 / 1.40 / 2.35 ms | 789 |
| arco-yanque.igz | 26,806 | DX12 | no vsync | 1.32 / 1.40 / 2.05 ms | 757 |
| stress | 1,028,700 (+777,600 edges) | Vulkan | no vsync | 1.27 / 1.37 / 5.93 ms | 787 |
| stress | 1,028,700 | DX12 | no vsync | 1.38 / 1.48 / 2.31 ms | 725 |

UI CPU time per frame (building the egui UI, excluding GPU submit) ~0.4 ms.
Caveats: the stress scene is ~1M small triangles seen from a distance (most are
sub-pixel, cheap to rasterise, no overdraw cost) and frame time is nearly
identical to the small model, so these numbers show the pipeline is far from
GPU-bound for static vertex buffers, not a worst case for city-L (large
on-screen faces, textures, shadows, per-style passes). The 1M-triangle scene
builds in ~130 ms; arco loads in ~120 ms. A screenshot check confirmed both
scenes actually render.

## Results

Worked:
- Docking with egui_dock 0.21 is straightforward; panels dock/undock/tab/split.
- Custom wgpu drawing inside a dock tab works with `NativeOptions::depth_buffer = 32`
  (egui's pass then owns a Depth32Float target our pipelines use). Callback gets
  correct viewport/scissor for the tab rect automatically.
- Windows backend: wgpu picked **Vulkan** by default on this machine (NVIDIA
  driver); DX12 works and is within ~5%. The plan's order "DX12 first" must be set
  explicitly via `WgpuSetup`/backend options, it is not the default.
- AccessKit: eframe `accesskit` feature is on (default feature) and the app calls
  `ctx.enable_accesskit()`. `--accesskit-dump` runs one headless pass and prints the tree
  that would go to the platform adapter: roles Window, Label, Button (with label),
  TextInput, CheckBox (with label), TextRun. So a screen-reader tree **is produced**.
  Not verified with a real screen reader (NVDA/Narrator) here; the custom 3D viewport
  exposes no semantics (it is a single painted rect).
- Text field: typing works; egui-winit has full `WindowEvent::Ime` handling
  (preedit/commit) and TextEdit consumes `Event::Ime`.

Not verified / didn't work:
- **IME composition (zh-CN) was not tested** interactively; only the code path
  (egui-winit Ime events) was confirmed present. Needs a manual check with a
  Chinese IME on Windows (candidate window position follows the text cursor in
  egui; known historical rough edges there).
- Layout persistence (plan gate) not implemented; zh-CN IME gate open.
- No picking, no MSAA (egui's pass is single-sample unless `multisampling` is set;
  edges will alias), line width is fixed 1 px (wgpu has no wide lines), so
  production needs quad-expanded lines in `ingetrazo-render`.

## Risks for M4 / M6

- **Version coupling**: egui_dock 0.21.1 depends on egui 0.36, while eframe 0.37 is
  already out; the spike had to pin eframe/egui-wgpu 0.36.2 (wgpu 30.0.1). The
  dock crate lags egui by a release; every egui upgrade is gated on egui_dock (or
  `egui_tiles` fallback). Pin exact versions.
- **wgpu API churn**: wgpu pipeline descriptors changed between recent majors
  (`immediate_size`, `multiview_mask`, `Option` bind group layouts); renderer code
  must track eframe's wgpu version.
- **Depth/MSAA are owned by eframe's pass**: our renderer draws into egui's
  render pass (shared depth, no own offscreen target). For ID-buffer picking,
  shadows, section fills, and MSAA, `ingetrazo-render` should render to its own
  textures and blit/register them as egui textures (or use callback `prepare` to
  run extra passes via the encoder). Decide this in M4.
- **AccessKit**: tree exists for egui widgets; the viewport/model needs custom
  accesskit nodes if it is to be accessible. Needs real screen-reader testing.
- **IME**: untested for CJK, see above.
- **Build time / size**: first compile of the spike (cold deps, release) about
  2 min 06 s on this machine plus ~1.5 min dependency download; incremental check
  ~50 s for the whole workspace clippy. Release exe: **14.5 MB** (thin LTO,
  opt-level 3, not stripped, includes default fonts and wayland/x11 code paths
  that are inert on Windows).
- Fan triangulation / holes: real face tessellation (ear clipping / earcut with
  holes) is needed in the kernel; this is not a UI risk.
