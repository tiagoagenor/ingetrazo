// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Commands and undo (`core/history.py`: `Command`, `CommandRejected`,
//! `History`).
//!
//! A command applies itself to a [`Scene`] (`do`) and reverses itself
//! (`undo`); `History` keeps the two stacks and makes `execute`
//! transactional. As in Python, the inverse is each command's OWN
//! (`AddEdgeCommand` removes its edge and leaves the vertices; a snapshot
//! command restores its `State`), not a generic replay of the mesh journal:
//! the recorded signatures include whatever an undo leaves behind, so the
//! journal (`Mesh::begin`/`commit`) is not what these commands undo with.
//!
//! The pieces of process or scene state a command reads are in [`Env`]:
//! the curve counter (`core.mesh._CURVE_COUNTER`, process-wide in Python and
//! not undone with the mesh), the active mesh (`scene.mesh`: the loose one
//! or a group being edited), and the Qt `hypot` of the build the recording
//! comes from. The selection (`scene.selection`) and the version counter are
//! view state and are not modelled: nothing in a signature depends on them.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::Hypot;
use ingetrazo_model::mesh::Mesh;
use ingetrazo_model::scene::{MeshId, Scene};

/// Why a command did not run.
#[derive(Clone, Debug, PartialEq)]
pub enum CommandError {
    /// `CommandRejected`: declined on purpose; the scene is as it was.
    Rejected(String),
    /// Any other exception: rolled back and logged by `History`.
    Failed(String),
    /// The Rust engine lacks something this command needs (reported by the
    /// golden replay as a blocker rather than as a divergence).
    Unsupported(String),
}

pub type CmdResult = Result<(), CommandError>;

/// Process and scene state outside the meshes (see the module docs).
#[derive(Clone, Debug)]
pub struct Env {
    pub hypot: Hypot,
    /// `core.mesh._CURVE_COUNTER`: the next curve id.
    pub curve: i64,
    /// The next default group number (`Group N`).
    pub group: i64,
    /// `scene.mesh`: the mesh edits go to.
    pub active: MeshId,
    /// `scene.active_ifc`: the BIM class a drawn face is tagged with
    /// (`{"class", "name"}`), when one is active.
    pub active_ifc: Option<Value>,
}

impl Default for Env {
    fn default() -> Self {
        Env { hypot: Hypot::default(), curve: 1, group: 1, active: MeshId(0), active_ifc: None }
    }
}

impl Env {
    pub fn mesh<'a>(&self, scene: &'a Scene) -> &'a Mesh {
        &scene.meshes[self.active.0 as usize]
    }
    pub fn mesh_mut<'a>(&self, scene: &'a mut Scene) -> &'a mut Mesh {
        &mut scene.meshes[self.active.0 as usize]
    }
}

/// `Command`: a reversible operation against the scene.
pub trait Command {
    /// `do`.
    fn apply(&mut self, scene: &mut Scene, env: &mut Env) -> CmdResult;
    /// `undo`.
    fn undo(&mut self, scene: &mut Scene, env: &mut Env);
    /// The class name, as Python's `type(cmd).__name__`.
    fn name(&self) -> &'static str;
    /// For the planner's `isinstance` tests on the commands it emitted.
    fn as_any(&self) -> &dyn std::any::Any;
}

struct Item {
    cmd: Box<dyn Command>,
    /// `cmd._history_mesh`: the mesh it ran against.
    mesh: MeshId,
}

/// `History`: the undo and redo stacks.
#[derive(Default)]
pub struct History {
    undo: Vec<Item>,
    redo: Vec<Item>,
    /// Message of the last rolled-back failure.
    pub last_error: Option<String>,
    /// Why the last command declined (`CommandRejected`).
    pub last_rejected: Option<String>,
    /// `History.max_steps`: how many steps back are kept (0 = unlimited).
    pub max_steps: usize,
}

impl History {
    pub fn new() -> History {
        History { max_steps: 200, ..Default::default() }
    }

    pub fn undo_len(&self) -> usize {
        self.undo.len()
    }

    pub fn redo_len(&self) -> usize {
        self.redo.len()
    }

    /// `execute`, transactional: a command that declines or throws leaves
    /// the mesh as it found it and puts nothing on the undo stack. Returns
    /// the error (also kept in `last_rejected` / `last_error`).
    pub fn execute(&mut self, mut cmd: Box<dyn Command>, scene: &mut Scene, env: &mut Env) -> CmdResult {
        let snapshot = env.mesh(scene).capture_state();
        self.last_rejected = None;
        match cmd.apply(scene, env) {
            Ok(()) => {}
            Err(CommandError::Rejected(why)) => {
                env.mesh_mut(scene).restore_state(&snapshot);
                self.last_error = None;
                self.last_rejected = Some(why.clone());
                return Err(CommandError::Rejected(why));
            }
            Err(e) => {
                env.mesh_mut(scene).restore_state(&snapshot);
                if let CommandError::Failed(why) = &e {
                    self.last_error = Some(format!("{}: {why}", cmd.name()));
                }
                return Err(e);
            }
        }
        self.last_error = None;
        self.undo.push(Item { cmd, mesh: env.active });
        if self.max_steps > 0 && self.undo.len() > self.max_steps {
            let extra = self.undo.len() - self.max_steps;
            self.undo.drain(..extra);
        }
        self.redo.clear();
        Ok(())
    }

    pub fn undo(&mut self, scene: &mut Scene, env: &mut Env) -> bool {
        let Some(mut item) = self.undo.pop() else { return false };
        let current = env.active;
        env.active = item.mesh;
        item.cmd.undo(scene, env);
        env.active = current;
        self.redo.push(item);
        true
    }

    pub fn redo(&mut self, scene: &mut Scene, env: &mut Env) -> bool {
        let Some(mut item) = self.redo.pop() else { return false };
        let current = env.active;
        env.active = item.mesh;
        // A redo that throws is a bug in Python too (no rollback there).
        let _ = item.cmd.apply(scene, env);
        env.active = current;
        self.undo.push(item);
        true
    }

    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
    }
}
