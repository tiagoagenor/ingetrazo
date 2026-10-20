//! Documents and synthetic scenes for the spike.

use ingetrazo_math::{M4, V3};
use ingetrazo_model::mesh::Mesh;
use ingetrazo_model::scene::{Group, MeshId, Scene};
use ingetrazo_render::document::open_igz;
use ingetrazo_render::TextureSource;

pub fn load_igz(path: &str) -> Result<(Scene, Box<dyn TextureSource + Send + Sync>), String> {
    let raw = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let (scene, tex) = open_igz(&raw).map_err(|e| format!("{path}: {e}"))?;
    Ok((scene, Box::new(tex)))
}

/// `n` x `n` extruded `sides`-gon prisms, as `n * n` placements of ONE
/// component (the renderer bakes the prototype once). With n = 45 and 128
/// sides: 1,028,700 triangles and 777,600 edges, the spike's old stress scene.
pub fn stress_scene(n: usize, sides: usize) -> Scene {
    let mut sc = Scene::new();
    let mut proto = Mesh::new();
    let ring: Vec<V3> = (0..sides)
        .map(|i| {
            let a = std::f32::consts::TAU * i as f32 / sides as f32;
            V3(a.cos(), a.sin(), 0.0)
        })
        .collect();
    let top: Vec<V3> = ring.iter().map(|p| V3(p.0, p.1, 2.0)).collect();
    let bottom: Vec<V3> = ring.iter().rev().copied().collect();
    proto.add_face(&bottom, &[]);
    proto.add_face(&top, &[]);
    for i in 0..sides {
        let j = (i + 1) % sides;
        proto.add_face(&[ring[i], ring[j], top[j], top[i]], &[]);
    }
    sc.meshes.push(proto);
    for gx in 0..n {
        for gy in 0..n {
            let mut m = M4::identity();
            m.translate(gx as f32 * 3.0, gy as f32 * 3.0, 0.0);
            let mut g = Group::new(MeshId(1), format!("prism {gx}-{gy}"));
            g.xform = Some(m);
            sc.groups.push(g);
        }
    }
    sc
}
