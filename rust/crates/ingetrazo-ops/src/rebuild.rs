// stub
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::scene::Scene;

use crate::command::{CmdResult, Command, CommandError, Env};

pub struct RebuildPlaneFacesCommand {}
impl RebuildPlaneFacesCommand {
    pub fn new(_o: V3, _n: V3, _h: Hypot) -> Self {
        RebuildPlaneFacesCommand {}
    }
    pub fn from_args(_o: V3, _n: V3, _h: Hypot) -> Self {
        RebuildPlaneFacesCommand {}
    }
}
impl Command for RebuildPlaneFacesCommand {
    fn apply(&mut self, _scene: &mut Scene, _env: &mut Env) -> CmdResult {
        Err(CommandError::Unsupported("rebuild_plane_faces".into()))
    }
    fn undo(&mut self, _scene: &mut Scene, _env: &mut Env) {}
    fn name(&self) -> &'static str {
        "RebuildPlaneFacesCommand"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
pub struct RebuildPlanarFacesCommand {}
impl RebuildPlanarFacesCommand {
    pub fn new() -> Self {
        RebuildPlanarFacesCommand {}
    }
}
impl Command for RebuildPlanarFacesCommand {
    fn apply(&mut self, _scene: &mut Scene, _env: &mut Env) -> CmdResult {
        Err(CommandError::Unsupported("rebuild_planar_faces".into()))
    }
    fn undo(&mut self, _scene: &mut Scene, _env: &mut Env) {}
    fn name(&self) -> &'static str {
        "RebuildPlanarFacesCommand"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
