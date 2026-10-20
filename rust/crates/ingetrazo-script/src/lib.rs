// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! IngeTrazo's scripting (docs/rust-rewrite-plan.md, M8): a sandboxed
//! [Rhai](https://rhai.rs) engine over the Rust model. It replaces the
//! Python Console and the code the AI assistant writes (`plugins/python_console.py`,
//! `core/ai.py`), which ran `exec` against live objects.
//!
//! # Contract
//!
//! * **Reads** go straight to the [`Scene`]; a script sees faces, edges,
//!   vertices, groups and components, layers, materials and annotations
//!   through the functions in [`api`].
//! * **Writes** are never direct: every mutating function builds an
//!   `ingetrazo-ops` [`Command`](ingetrazo_ops::command::Command) and executes
//!   it through the document's [`History`], so each is undoable and takes the
//!   same path as the tools do (the edits planner for drawing, the transform,
//!   grouping and paint commands for the rest).
//! * **One run is one undo step** (as the console's `SnapshotImport` was): the
//!   commands a run executes are folded into one compound step
//!   ([`History::commit_batch`]); a run that changes nothing leaves the undo and
//!   redo stacks as they were; a run that fails is rolled back whole, so a
//!   script never leaves half-built geometry.
//! * **Sandbox**: no `import`/modules (Rhai is built with `no_module`), no
//!   `eval`, no file, network or process API exists to be called, and the
//!   [`Limits`] bound operations, call depth, expression depth, string, array
//!   and map sizes, commands run and wall-clock time.
//!
//! Handles ([`Face`](convert::FaceH), [`Edge`](convert::EdgeH),
//! [`Group`](convert::GroupH)) are arena ids and group paths; they go stale
//! when their entity is erased or the run is undone, and the functions that
//! take one say so instead of touching something else.
//!
//! The scripts edit the mesh that is active when the run starts
//! (`Cx::mesh`): the loose model, or a group being edited.

pub mod api;
pub mod convert;

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::{Duration, Instant};

use ingetrazo_model::scene::Scene;
use ingetrazo_ops::command::{Command, Cx, History};
use ingetrazo_ops::edit_cmds::Entity;
use rhai::{Dynamic, Engine, EvalAltResult, Scope};

use convert::{fail, Res};

/// The resource limits of a run.
#[derive(Clone, Debug)]
pub struct Limits {
    /// Rhai operations (each expression, call and loop turn counts).
    pub max_operations: u64,
    /// Nested function calls.
    pub max_call_levels: usize,
    /// Nested expression depth, and inside a function body.
    pub max_expr_depth: usize,
    pub max_fn_expr_depth: usize,
    /// Characters in a string.
    pub max_string_size: usize,
    /// Items in an array / a map.
    pub max_array_size: usize,
    pub max_map_size: usize,
    /// Commands one run may execute.
    pub max_commands: usize,
    /// Wall-clock time of one run.
    pub timeout: Option<Duration>,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_operations: 5_000_000,
            max_call_levels: 64,
            max_expr_depth: 64,
            max_fn_expr_depth: 32,
            max_string_size: 1 << 20,
            max_array_size: 200_000,
            max_map_size: 20_000,
            max_commands: 100_000,
            timeout: Some(Duration::from_secs(30)),
        }
    }
}

/// What a run needs of the document, swapped in for its duration.
pub(crate) struct State {
    pub scene: Scene,
    pub cx: Cx,
    pub history: History,
    pub selection: Vec<Entity>,
    pub commands: usize,
    pub max_commands: usize,
}

pub(crate) type Shared = Rc<RefCell<State>>;

/// Execute a command through the history; a command that fails or declines
/// stops the script with its message (and the run rolls back).
pub(crate) fn exec(sh: &Shared, cmd: Box<dyn Command>) -> Res<()> {
    let mut st = sh.borrow_mut();
    st.commands += 1;
    if st.commands > st.max_commands {
        return fail("too many commands in one run");
    }
    let State { scene, cx, history, .. } = &mut *st;
    history.execute(scene, cx, cmd);
    if let Some(e) = history.last_error.clone() {
        return fail(e);
    }
    if let Some(why) = history.last_rejected.clone() {
        return fail(format!("rejected: {why}"));
    }
    Ok(())
}

/// Several commands as one (a compound), or the command itself.
pub(crate) fn exec_all(sh: &Shared, mut cmds: Vec<Box<dyn Command>>) -> Res<()> {
    match cmds.len() {
        0 => Ok(()),
        1 => exec(sh, cmds.pop().expect("one command")),
        _ => exec(sh, Box::new(ingetrazo_ops::draw::CompoundCommand { commands: cmds })),
    }
}

/// A script that failed: the message, the line where known, and what it had
/// printed before.
#[derive(Clone, Debug, PartialEq)]
pub struct ScriptError {
    pub message: String,
    pub line: Option<usize>,
    pub output: String,
}

impl std::fmt::Display for ScriptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.line {
            Some(l) => write!(f, "line {l}: {}", self.message),
            None => write!(f, "{}", self.message),
        }
    }
}

/// A script that ran.
#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    /// What `print` / `debug` wrote, a line each.
    pub output: String,
    /// The script's last expression, as text (`""` for none).
    pub value: String,
    /// Commands executed (0 = the document was only read).
    pub commands: usize,
}

/// The Rhai engine with the model API registered.
pub struct ScriptEngine {
    engine: Engine,
    shared: Shared,
    output: Rc<RefCell<String>>,
    deadline: Rc<Cell<Option<Instant>>>,
    limits: Limits,
}

/// Put the document back into the caller's variables when a run ends, even
/// by a panic.
struct Swap<'a> {
    shared: &'a Shared,
    scene: &'a mut Scene,
    cx: &'a mut Cx,
    history: &'a mut History,
}

impl Swap<'_> {
    fn take(&mut self) {
        let mut st = self.shared.borrow_mut();
        std::mem::swap(&mut st.scene, self.scene);
        std::mem::swap(&mut st.cx, self.cx);
        std::mem::swap(&mut st.history, self.history);
    }
}

impl Drop for Swap<'_> {
    fn drop(&mut self) {
        self.take();
    }
}

impl Default for ScriptEngine {
    fn default() -> Self {
        ScriptEngine::new(Limits::default())
    }
}

impl ScriptEngine {
    pub fn new(limits: Limits) -> ScriptEngine {
        let shared: Shared = Rc::new(RefCell::new(State {
            scene: Scene::new(),
            cx: Cx::default(),
            history: History::new(),
            selection: Vec::new(),
            commands: 0,
            max_commands: limits.max_commands,
        }));
        let output = Rc::new(RefCell::new(String::new()));
        let deadline: Rc<Cell<Option<Instant>>> = Rc::new(Cell::new(None));
        let mut engine = Engine::new();
        engine.set_max_operations(limits.max_operations);
        engine.set_max_call_levels(limits.max_call_levels);
        engine.set_max_expr_depths(limits.max_expr_depth, limits.max_fn_expr_depth);
        engine.set_max_string_size(limits.max_string_size);
        engine.set_max_array_size(limits.max_array_size);
        engine.set_max_map_size(limits.max_map_size);
        engine.set_strict_variables(true);
        engine.set_fail_on_invalid_map_property(true);
        engine.disable_symbol("eval");
        let out = output.clone();
        engine.on_print(move |s| {
            let mut o = out.borrow_mut();
            o.push_str(s);
            o.push('\n');
        });
        let out = output.clone();
        engine.on_debug(move |s, _src, pos| {
            let mut o = out.borrow_mut();
            if pos.is_none() {
                o.push_str(&format!("{s}\n"));
            } else {
                o.push_str(&format!("[{pos}] {s}\n"));
            }
        });
        let dl = deadline.clone();
        engine.on_progress(move |ops| {
            if ops % 512 == 0 {
                if let Some(t) = dl.get() {
                    if Instant::now() > t {
                        return Some(Dynamic::from("the script ran out of time"));
                    }
                }
            }
            None
        });
        api::register(&mut engine, &shared);
        ScriptEngine { engine, shared, output, deadline, limits }
    }

    pub fn limits(&self) -> &Limits {
        &self.limits
    }

    /// The entities `selection()` answers in the next runs (the UI's
    /// selection).
    pub fn set_selection(&mut self, selection: Vec<Entity>) {
        self.shared.borrow_mut().selection = selection;
    }

    /// The Rhai engine, to register an extension on top of the API.
    pub fn engine_mut(&mut self) -> &mut Engine {
        &mut self.engine
    }

    /// Run `source` against the document. On success the commands it ran are
    /// ONE undo step of `history`; on error everything is rolled back and the
    /// document is as it was.
    pub fn run(&mut self, scene: &mut Scene, cx: &mut Cx, history: &mut History, source: &str) -> Result<Report, ScriptError> {
        self.output.borrow_mut().clear();
        {
            let mut st = self.shared.borrow_mut();
            st.commands = 0;
            st.max_commands = self.limits.max_commands;
        }
        self.deadline.set(self.limits.timeout.map(|t| Instant::now() + t));
        let mut swap = Swap { shared: &self.shared, scene, cx, history };
        swap.take();
        let batch = self.shared.borrow_mut().history.begin_batch();

        let ast = match self.engine.compile(source) {
            Ok(a) => a,
            Err(e) => {
                let line = e.1.line();
                let mut st = self.shared.borrow_mut();
                let State { scene, cx, history, .. } = &mut *st;
                history.rollback_batch(batch, scene, cx);
                return Err(ScriptError { message: e.0.to_string(), line, output: self.output.borrow().clone() });
            }
        };
        let mut scope = Scope::new();
        let result: Result<Dynamic, Box<EvalAltResult>> = self.engine.eval_ast_with_scope(&mut scope, &ast);
        let mut st = self.shared.borrow_mut();
        let State { scene, cx, history, commands, .. } = &mut *st;
        let output = self.output.borrow().clone();
        match result {
            Ok(v) => {
                history.commit_batch(batch);
                let value = if v.is_unit() { String::new() } else { v.to_string() };
                Ok(Report { output, value, commands: *commands })
            }
            Err(e) => {
                history.rollback_batch(batch, scene, cx);
                Err(ScriptError { message: error_text(&e), line: e.position().line(), output })
            }
        }
    }
}

fn error_text(e: &EvalAltResult) -> String {
    match e {
        EvalAltResult::ErrorRuntime(v, _) => v.to_string(),
        EvalAltResult::ErrorTerminated(v, _) => v.to_string(),
        other => other.to_string(),
    }
}
