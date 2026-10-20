// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `Command` and `History` (`core/history.py`).
//!
//! A command is applied to a [`Scene`] and undone against it; `History`
//! keeps the two stacks and makes `execute` transactional, as Python's
//! does: the active mesh is captured first and put back when the command
//! declines (`CommandRejected`) or fails.
//!
//! Differences from the plan's sketch (`apply(&mut Scene, &mut Journal)`):
//! the journal is the mesh's own (`Mesh::begin/commit`); the commands that
//! port a Python snapshot command (`capture_state`/`restore_state`) keep the
//! snapshot, because `restore_state` clears the mesh's displaced queue and a
//! journal entry would put the earlier queue back. Process-wide counters
//! (`Group N`, curve ids, uids) travel in [`Cx`] instead of globals.

use ingetrazo_model::scene::{MeshId, Scene};

/// Why a command did not happen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fail {
    /// `CommandRejected`: on purpose, nothing logged, the reason is shown.
    Rejected(String),
    /// Any other exception: rolled back and logged.
    Failed(String),
}

/// The process-wide counters a command may draw from (SCHEMA §5,
/// Determinism): `core.group._counter`, `core.mesh._CURVE_COUNTER`, and a
/// uid source (random in Python, never part of a signature).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Counters {
    pub group: i64,
    pub curve: i64,
    pub uid: u64,
}

impl Default for Counters {
    fn default() -> Self {
        Counters { group: 1, curve: 1, uid: 1 }
    }
}

impl Counters {
    /// A fresh group identity (`core.group.new_uid`: 16 hex digits).
    pub fn new_uid(&mut self) -> String {
        let u = format!("{:016x}", self.uid);
        self.uid += 1;
        u
    }
}

/// What a command runs against besides the scene: the mesh `scene.mesh`
/// points at (the loose one, or a group being edited) and the counters.
#[derive(Clone, Debug)]
pub struct Cx {
    /// This Qt's `std::hypot` formula (the golden corpus is MSVC's).
    pub hypot: ingetrazo_math::Hypot,
    /// `scene.active_ifc`: the BIM class a drawn face is tagged with
    /// (`{"class", "name"}`), when one is active.
    pub active_ifc: Option<ingetrazo_formats::pyjson::Value>,
    pub mesh: MeshId,
    pub counters: Counters,
}

impl Default for Cx {
    fn default() -> Self {
        Cx { hypot: ingetrazo_math::Hypot::default(), active_ifc: None, mesh: MeshId(0), counters: Counters::default() }
    }
}

/// `Command`: a reversible operation against a scene. `apply` is Python's
/// `do` (redo runs it again); `undo` reverses it.
pub trait Command {
    /// The class name without `Command`, in snake case (the golden op).
    fn name(&self) -> &'static str;
    fn apply(&mut self, scene: &mut Scene, cx: &mut Cx) -> Result<(), Fail>;
    fn undo(&mut self, scene: &mut Scene, cx: &mut Cx);
    /// For the planner's `isinstance` tests and `describe`: the commands
    /// that are looked into say so; the rest answer `None`.
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }
}

impl Cx {
    /// `scene.mesh`.
    pub fn mesh<'a>(&self, scene: &'a Scene) -> &'a ingetrazo_model::mesh::Mesh {
        &scene.meshes[self.mesh.0 as usize]
    }
    pub fn mesh_mut<'a>(&self, scene: &'a mut Scene) -> &'a mut ingetrazo_model::mesh::Mesh {
        &mut scene.meshes[self.mesh.0 as usize]
    }
}

struct Done {
    /// Identity of the executed command, for tools that ask whether theirs
    /// is still on top (`undo_stack[-1] is cmd`).
    serial: u64,
    cmd: Box<dyn Command>,
    /// `_history_mesh`: the mesh that was active when it ran.
    mesh: MeshId,
}

/// Undo/redo stacks (`History`).
pub struct History {
    undo_stack: Vec<Done>,
    redo_stack: Vec<Done>,
    /// Message of the last rolled-back failure.
    pub last_error: Option<String>,
    /// Why the last command declined to happen.
    pub last_rejected: Option<String>,
    /// How many steps back are kept (0 = unlimited).
    pub max_steps: usize,
    /// Serial of the last command `execute` put on the undo stack.
    pub last_serial: Option<u64>,
    next_serial: u64,
}

impl Default for History {
    fn default() -> Self {
        History::new()
    }
}

impl History {
    pub fn new() -> History {
        History { undo_stack: Vec::new(), redo_stack: Vec::new(), last_error: None, last_rejected: None, max_steps: 200, last_serial: None, next_serial: 1 }
    }

    /// Serial of the command on top of the undo stack.
    pub fn top_serial(&self) -> Option<u64> {
        self.undo_stack.last().map(|d| d.serial)
    }

    /// The command on top of the undo stack (`undo_stack[-1]`).
    pub fn top(&self) -> Option<&dyn Command> {
        self.undo_stack.last().map(|d| d.cmd.as_ref())
    }

    /// The command `redo` would run next (`redo_stack[-1]`).
    pub fn top_redo(&self) -> Option<&dyn Command> {
        self.redo_stack.last().map(|d| d.cmd.as_ref())
    }

    pub fn undo_len(&self) -> usize {
        self.undo_stack.len()
    }

    pub fn redo_len(&self) -> usize {
        self.redo_stack.len()
    }

    pub fn execute(&mut self, scene: &mut Scene, cx: &mut Cx, mut cmd: Box<dyn Command>) {
        let active = cx.mesh;
        let snapshot = scene.meshes[active.0 as usize].capture_state();
        self.last_rejected = None;
        match cmd.apply(scene, cx) {
            Ok(()) => {}
            Err(Fail::Rejected(why)) => {
                scene.meshes[active.0 as usize].restore_state(&snapshot);
                self.last_error = None;
                self.last_rejected = Some(why);
                return;
            }
            Err(Fail::Failed(why)) => {
                scene.meshes[active.0 as usize].restore_state(&snapshot);
                self.last_error = Some(format!("{}: {why}", cmd.name()));
                return;
            }
        }
        self.last_error = None;
        let serial = self.next_serial;
        self.next_serial += 1;
        self.last_serial = Some(serial);
        self.undo_stack.push(Done { serial, cmd, mesh: active });
        if self.max_steps > 0 && self.undo_stack.len() > self.max_steps {
            let extra = self.undo_stack.len() - self.max_steps;
            self.undo_stack.drain(..extra);
        }
        self.redo_stack.clear();
        scene.rebind_dimensions(cx.mesh);
    }

    /// Run `f` with `cx.mesh` on the mesh the command was executed against.
    fn in_command_mesh(cx: &mut Cx, mesh: MeshId, f: impl FnOnce(&mut Cx)) {
        let current = cx.mesh;
        cx.mesh = mesh;
        f(cx);
        cx.mesh = current;
    }

    pub fn undo(&mut self, scene: &mut Scene, cx: &mut Cx) -> bool {
        let Some(mut d) = self.undo_stack.pop() else { return false };
        History::in_command_mesh(cx, d.mesh, |cx| d.cmd.undo(scene, cx));
        self.redo_stack.push(d);
        scene.rebind_dimensions(cx.mesh);
        true
    }

    pub fn redo(&mut self, scene: &mut Scene, cx: &mut Cx) -> bool {
        let Some(mut d) = self.redo_stack.pop() else { return false };
        History::in_command_mesh(cx, d.mesh, |cx| {
            // A command that ran once cannot be rejected on redo: it
            // restores or recomputes what it already did.
            let _ = d.cmd.apply(scene, cx);
        });
        self.undo_stack.push(d);
        scene.rebind_dimensions(cx.mesh);
        true
    }

    /// `history.redo_stack.clear()`.
    pub fn clear_redo(&mut self) {
        self.redo_stack.clear();
    }

    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
    }

    /// Open a batch: whatever is executed from now on can be folded into ONE
    /// undo step ([`History::commit_batch`]) or taken back whole
    /// ([`History::rollback_batch`]). The redo stack is set aside (it would
    /// be cleared by the first change anyway) and `max_steps` is lifted so
    /// the batch is never trimmed from the front. Batches do not nest.
    pub fn begin_batch(&mut self) -> Batch {
        let batch = Batch { len: self.undo_stack.len(), redo: std::mem::take(&mut self.redo_stack), max_steps: self.max_steps };
        self.max_steps = 0;
        batch
    }

    fn end_batch(&mut self, batch: &Batch) {
        self.max_steps = batch.max_steps;
    }

    /// Close a batch successfully. The commands it executed become one undo
    /// step: the command itself when there was exactly one, a
    /// [`CompoundCommand`](crate::draw::CompoundCommand) of them otherwise;
    /// none at all leaves the history, redo stack included, as it was.
    /// Returns how many commands the batch ran.
    pub fn commit_batch(&mut self, mut batch: Batch) -> usize {
        let ran = self.undo_stack.len() - batch.len;
        if ran == 0 {
            self.redo_stack = std::mem::take(&mut batch.redo);
        } else if ran > 1 {
            let mesh = self.undo_stack[batch.len].mesh;
            let commands: Vec<Box<dyn Command>> = self.undo_stack.drain(batch.len..).map(|d| d.cmd).collect();
            // The compound is a new step: a new identity.
            let serial = self.next_serial;
            self.next_serial += 1;
            self.last_serial = Some(serial);
            self.undo_stack.push(Done { serial, cmd: Box::new(crate::draw::CompoundCommand { commands }), mesh });
        }
        self.end_batch(&batch);
        if self.max_steps > 0 && self.undo_stack.len() > self.max_steps {
            let extra = self.undo_stack.len() - self.max_steps;
            self.undo_stack.drain(..extra);
        }
        ran
    }

    /// Close a batch by undoing everything it did (newest first). The undone
    /// commands are dropped, not queued for redo; the redo stack is put back.
    pub fn rollback_batch(&mut self, mut batch: Batch, scene: &mut Scene, cx: &mut Cx) {
        while self.undo_stack.len() > batch.len {
            let Some(mut d) = self.undo_stack.pop() else { break };
            History::in_command_mesh(cx, d.mesh, |cx| d.cmd.undo(scene, cx));
        }
        self.redo_stack = std::mem::take(&mut batch.redo);
        self.end_batch(&batch);
        scene.rebind_dimensions(cx.mesh);
    }
}

/// An open batch of commands (see [`History::begin_batch`]).
pub struct Batch {
    len: usize,
    redo: Vec<Done>,
    max_steps: usize,
}
