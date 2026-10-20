// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `ingetrazo --bench-orbit <seconds> file.igz`: the alpha benchmark (M4.9).
//!
//! It runs the real application loop (window, egui, the paint callback, the
//! renderer, the GPU) and drives the session with synthesized
//! [`Input`] events, frame by frame, as a hand would make them:
//!
//! 1. **orbit**: a middle-button drag that keeps turning for `seconds`;
//! 2. **hover**: the pointer sweeps the model, first with the Select tool and
//!    then with Line (the inference engine);
//! 3. **edits**: a line, a rectangle, a click on a building, its move and the
//!    undo of the move, and the time
//!    from the committing click until the next frame starts is measured
//!    (edit to screen).
//!
//! The report goes to stdout as `key value` lines (see `docs/alpha.md`).
//! Use it with `--novsync`, otherwise every frame waits for the monitor.

use std::collections::VecDeque;
use std::time::Instant;

use ingetrazo_tools::modify_base::Sel;

use crate::session::{Btn, Input, Session};

/// One frame's samples, in milliseconds.
#[derive(Clone, Copy, Default)]
pub struct Frame {
    /// Interval since the previous frame started.
    pub dt: f32,
    /// What the previous frame's `ui` pass took (input, layout, overlay).
    pub ui: f32,
    /// What the renderer's `prepare` took on the CPU (sync + plan + encode).
    pub prepare: f32,
}

enum Step {
    /// Turn the camera for this many seconds.
    Orbit(f64),
    /// Sweep the pointer for this many seconds (`true`: with the Line tool).
    Hover(f64, bool),
    /// Inject the inputs in one frame, measured as `Measure` says.
    Inputs(Vec<Input>, Measure),
    /// Ctrl+Z, measured as an event.
    Undo(&'static str),
    Wait(u32),
}

/// What an [`Step::Inputs`] reports.
#[derive(Clone, Copy)]
enum Measure {
    No,
    /// An edit: reported when the history grew ("NOT_APPLIED" if it did not).
    Edit(&'static str),
    /// Any event: the time until the next frame starts.
    Event(&'static str),
}

/// The scripted run.
pub struct Bench {
    steps: VecDeque<Step>,
    started: Option<Instant>,
    cursor: (f64, f64),
    /// A pixel over a building (a group), seen while hovering.
    group_px: Option<(f64, f64)>,
    dragging: bool,
    n: u32,
    /// The edit waiting for the next frame: `(name, t0, handler ms)`.
    pending: Option<(&'static str, Instant, f64)>,
    pub orbit: Vec<Frame>,
    pub hover: Vec<Frame>,
    pub hover_handler: Vec<f32>,
    pub hover_line: Vec<Frame>,
    pub hover_line_handler: Vec<f32>,
    /// `(name, handler ms, edit-to-next-frame ms, did the edit land)`.
    pub edits: Vec<(&'static str, f64, f64, bool)>,
    pub done: bool,
    pub open_ms: f64,
    pub first_frame_ms: f64,
    /// The press that starts the orbit (it asks the model for its pivot).
    pub orbit_press_ms: f64,
}

fn mv(x: f64, y: f64) -> Input {
    Input::Move { x, y, mods: 0 }
}

fn click(x: f64, y: f64) -> Vec<Input> {
    vec![mv(x, y), Input::Press { button: Btn::Left, x, y, mods: 0 }, Input::Release { button: Btn::Left, x, y, mods: 0 }]
}

impl Bench {
    pub fn new(orbit_secs: f64) -> Bench {
        let mut steps = VecDeque::new();
        steps.push_back(Step::Wait(30));
        steps.push_back(Step::Orbit(orbit_secs));
        steps.push_back(Step::Hover(2.0, false));
        steps.push_back(Step::Inputs(vec![Input::Text { text: "l".into(), mods: 0 }], Measure::No));
        steps.push_back(Step::Wait(5));
        steps.push_back(Step::Hover(2.0, true));
        Bench {
            steps,
            started: None,
            cursor: (0.0, 0.0),
            group_px: None,
            dragging: false,
            n: 0,
            pending: None,
            orbit: Vec::new(),
            hover: Vec::new(),
            hover_handler: Vec::new(),
            hover_line: Vec::new(),
            hover_line_handler: Vec::new(),
            edits: Vec::new(),
            done: false,
            open_ms: 0.0,
            first_frame_ms: 0.0,
            orbit_press_ms: 0.0,
        }
    }

    /// Queue the edits, once the viewport size is known (they point at
    /// pixels): a line, a rectangle, a click on a building, that building
    /// moved, and the undo of the move.
    fn queue_edits(&mut self, size: (i32, i32)) {
        let (w, h) = (size.0 as f64, size.1 as f64);
        let (cx, cy) = (w * 0.5, h * 0.5);
        let key = |c: &str| Input::Text { text: c.to_string(), mods: 0 };
        let s = &mut self.steps;
        s.push_back(Step::Inputs(click(cx - 120.0, cy + 90.0), Measure::No));
        s.push_back(Step::Wait(3));
        s.push_back(Step::Inputs(click(cx + 60.0, cy + 120.0), Measure::Edit("line")));
        s.push_back(Step::Wait(6));
        s.push_back(Step::Inputs(vec![key("r")], Measure::No));
        s.push_back(Step::Wait(3));
        s.push_back(Step::Inputs(click(cx - 200.0, cy + 150.0), Measure::No));
        s.push_back(Step::Wait(3));
        s.push_back(Step::Inputs(click(cx - 100.0, cy + 200.0), Measure::Edit("rectangle")));
        s.push_back(Step::Wait(6));
        // A building: click it with Select, take the Move tool, base point,
        // target, then undo the move.
        if let Some((gx, gy)) = self.group_px {
            s.push_back(Step::Inputs(vec![key(" ")], Measure::No));
            s.push_back(Step::Wait(3));
            s.push_back(Step::Inputs(click(gx, gy), Measure::Event("click_select")));
            s.push_back(Step::Wait(6));
            s.push_back(Step::Inputs(vec![key("m")], Measure::No));
            s.push_back(Step::Wait(3));
            s.push_back(Step::Inputs(click(gx, gy), Measure::No));
            s.push_back(Step::Wait(3));
            s.push_back(Step::Inputs(vec![mv(gx + 40.0, gy)], Measure::No));
            s.push_back(Step::Wait(3));
            s.push_back(Step::Inputs(click(gx + 40.0, gy), Measure::Edit("move_building")));
            s.push_back(Step::Wait(6));
            s.push_back(Step::Undo("undo"));
            s.push_back(Step::Wait(6));
        }
        s.push_back(Step::Wait(10));
    }

    /// A frame started: the edit made before it is on screen by now.
    pub fn on_frame_start(&mut self, now: Instant) {
        if let Some((name, t0, handler)) = self.pending.take() {
            self.edits.push((name, handler, now.duration_since(t0).as_secs_f64() * 1000.0, true));
        }
    }

    /// One frame of the script.
    pub fn step(&mut self, sess: &mut Session, now: Instant, frame: Frame) {
        self.n += 1;
        if frame.dt > 50.0 && std::env::var_os("INGETRAZO_BENCH_TRACE").is_some() {
            eprintln!("  slow frame #{}: {:.0} ms (ui {:.0}, prepare {:.0})", self.n, frame.dt, frame.ui, frame.prepare);
        }
        if self.n <= 30 {
            // The first frames bake the model: the longest is the first frame.
            self.first_frame_ms = self.first_frame_ms.max(frame.prepare as f64);
        }
        let size = sess.size;
        let (cx, cy) = (size.0 as f64 * 0.5, size.1 as f64 * 0.5);
        let Some(front) = self.steps.front_mut() else {
            self.done = true;
            return;
        };
        match front {
            Step::Wait(n) => {
                if *n <= 1 {
                    self.steps.pop_front();
                } else {
                    *n -= 1;
                }
            }
            Step::Orbit(secs) => {
                let secs = *secs;
                if !self.dragging {
                    self.started = Some(now);
                    sess.handle(&mv(cx, cy));
                    let tp = Instant::now();
                    sess.handle(&Input::Press { button: Btn::Middle, x: cx, y: cy, mods: 0 });
                    self.orbit_press_ms = tp.elapsed().as_secs_f64() * 1000.0;
                    self.dragging = true;
                    self.cursor = (cx, cy);
                    return;
                }
                let el = now.duration_since(self.started.unwrap_or(now)).as_secs_f64();
                // A steady ~5 px per frame, turning back every 120 frames.
                let dir = if (self.n / 120).is_multiple_of(2) { 1.0 } else { -1.0 };
                self.cursor.0 += 5.0 * dir;
                sess.handle(&mv(self.cursor.0, self.cursor.1));
                self.orbit.push(frame);
                if el >= secs {
                    sess.handle(&Input::Release { button: Btn::Middle, x: self.cursor.0, y: self.cursor.1, mods: 0 });
                    self.dragging = false;
                    self.started = None;
                    self.steps.pop_front();
                }
            }
            Step::Hover(secs, line) => {
                let (secs, line) = (*secs, *line);
                let t0 = *self.started.get_or_insert(now);
                let el = now.duration_since(t0).as_secs_f64();
                let a = el * 2.0;
                let x = cx + (size.0 as f64 * 0.35) * a.cos();
                let y = cy + (size.1 as f64 * 0.30) * (a * 1.3).sin();
                let t = Instant::now();
                sess.handle(&mv(x, y));
                let ms = t.elapsed().as_secs_f32() * 1000.0;
                if line {
                    self.hover_line_handler.push(ms);
                    self.hover_line.push(frame);
                } else {
                    self.hover_handler.push(ms);
                    self.hover.push(frame);
                    if self.group_px.is_none() && matches!(sess.host.tool().hover(), Some(Sel::Group(_))) {
                        self.group_px = Some((x, y));
                    }
                }
                if el >= secs {
                    self.started = None;
                    self.steps.pop_front();
                    if line {
                        self.queue_edits(size);
                    }
                }
            }
            Step::Undo(name) => {
                let name = *name;
                self.steps.pop_front();
                let t0 = Instant::now();
                sess.undo();
                self.pending = Some((name, t0, t0.elapsed().as_secs_f64() * 1000.0));
            }
            Step::Inputs(inputs, measure) => {
                let (inputs, measure) = (std::mem::take(inputs), *measure);
                self.steps.pop_front();
                let before = sess.host.core.history.undo_len();
                let t0 = Instant::now();
                for i in &inputs {
                    let ti = Instant::now();
                    sess.handle(i);
                    if std::env::var_os("INGETRAZO_BENCH_TRACE").is_some() {
                        eprintln!("  input {:?} {:.1} ms", i, ti.elapsed().as_secs_f64() * 1000.0);
                    }
                }
                let handler = t0.elapsed().as_secs_f64() * 1000.0;
                match measure {
                    Measure::No => {}
                    Measure::Event(name) => self.pending = Some((name, t0, handler)),
                    Measure::Edit(name) => {
                        if sess.host.core.history.undo_len() > before {
                            self.pending = Some((name, t0, handler));
                        } else {
                            self.edits.push((name, handler, f64::NAN, false));
                        }
                    }
                }
            }
        }
    }

    pub fn report(&self, scene_stats: &str) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        let pct = |v: &[f32], q: f64| -> f32 {
            if v.is_empty() {
                return f32::NAN;
            }
            let mut v = v.to_vec();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[(((v.len() - 1) as f64) * q).round() as usize]
        };
        let _ = writeln!(s, "scene {scene_stats}");
        let _ = writeln!(s, "open_ms {:.1}", self.open_ms);
        let _ = writeln!(s, "first_frame_ms {:.1}", self.first_frame_ms);
        let _ = writeln!(s, "orbit_press_ms {:.1}", self.orbit_press_ms);
        for (name, frames) in [("orbit", &self.orbit), ("hover", &self.hover), ("hover_line", &self.hover_line)] {
            let dt: Vec<f32> = frames.iter().map(|f| f.dt).collect();
            let ui: Vec<f32> = frames.iter().map(|f| f.ui).collect();
            let pr: Vec<f32> = frames.iter().map(|f| f.prepare).collect();
            let _ = writeln!(s, "{name}_frames {}", frames.len());
            let _ = writeln!(s, "{name}_frame_ms p50 {:.2} p95 {:.2} p99 {:.2} max {:.2}", pct(&dt, 0.5), pct(&dt, 0.95), pct(&dt, 0.99), pct(&dt, 1.0));
            let worst = dt.iter().cloned().enumerate().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).map(|(i, _)| i).unwrap_or(0);
            let _ = writeln!(s, "{name}_worst_frame_index {worst} of {}", dt.len());
            let _ = writeln!(s, "{name}_fps_p50 {:.0}", 1000.0 / pct(&dt, 0.5));
            let _ = writeln!(s, "{name}_ui_ms p50 {:.2} p95 {:.2}", pct(&ui, 0.5), pct(&ui, 0.95));
            let _ = writeln!(s, "{name}_prepare_ms p50 {:.2} p95 {:.2}", pct(&pr, 0.5), pct(&pr, 0.95));
        }
        let _ = writeln!(
            s,
            "hover_handler_ms p50 {:.2} p95 {:.2} max {:.2}",
            pct(&self.hover_handler, 0.5),
            pct(&self.hover_handler, 0.95),
            pct(&self.hover_handler, 1.0)
        );
        let _ = writeln!(
            s,
            "hover_line_handler_ms p50 {:.2} p95 {:.2} max {:.2}",
            pct(&self.hover_line_handler, 0.5),
            pct(&self.hover_line_handler, 0.95),
            pct(&self.hover_line_handler, 1.0)
        );
        for (name, handler, latency, landed) in &self.edits {
            if *landed {
                let _ = writeln!(s, "edit_{name} handler_ms {handler:.2} edit_to_next_frame_ms {latency:.2}");
            } else {
                let _ = writeln!(s, "edit_{name} NOT_APPLIED handler_ms {handler:.2}");
            }
        }
        if let Some(m) = process_memory_mb() {
            let _ = writeln!(s, "memory_mb {m:.0}");
        }
        s
    }
}

/// The resident memory of this process, MB (Windows and Linux).
pub fn process_memory_mb() -> Option<f64> {
    #[cfg(windows)]
    {
        #[repr(C)]
        struct Counters {
            cb: u32,
            page_faults: u32,
            peak_working_set: usize,
            working_set: usize,
            rest: [usize; 6],
        }
        #[link(name = "kernel32")]
        extern "system" {
            fn GetCurrentProcess() -> isize;
            fn K32GetProcessMemoryInfo(p: isize, c: *mut Counters, cb: u32) -> i32;
        }
        let mut c = Counters { cb: std::mem::size_of::<Counters>() as u32, page_faults: 0, peak_working_set: 0, working_set: 0, rest: [0; 6] };
        // SAFETY: `c` is a PROCESS_MEMORY_COUNTERS-sized struct (`cb` says so)
        // and the pseudo-handle of the current process needs no closing.
        let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
        (ok != 0).then(|| c.working_set as f64 / 1_048_576.0)
    }
    #[cfg(target_os = "linux")]
    {
        let s = std::fs::read_to_string("/proc/self/status").ok()?;
        let line = s.lines().find(|l| l.starts_with("VmRSS:"))?;
        let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kb / 1024.0)
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        None
    }
}
