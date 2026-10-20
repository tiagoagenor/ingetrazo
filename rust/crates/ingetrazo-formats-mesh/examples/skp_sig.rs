// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Imports a `.skp` and prints the L2 signature of the scene (canonical JSON)
//! with the import time on stderr: a development aid for comparing with
//! `scripts/golden/skp_sig.py` on real models.
//!
//!     cargo run --release -p ingetrazo-formats-mesh --example skp_sig -- FILE.skp [CACHE_DIR]

use std::time::Instant;

use ingetrazo_formats_mesh::skp_in;
use ingetrazo_golden::model_sig::scene_signature;
use ingetrazo_golden::sig::{canon_dumps, Files};
use ingetrazo_math::Hypot;
use ingetrazo_model::scene::Scene;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("path of a .skp");
    let cache = args.next().map(std::path::PathBuf::from).unwrap_or_else(|| std::env::temp_dir().join("ingetrazo-skp-sig"));
    std::fs::create_dir_all(&cache).unwrap();
    let t = Instant::now();
    let payload = match skp_in::parse_skp(std::path::Path::new(&path), &cache) {
        Ok(p) => p,
        Err(e) => {
            println!("ERROR {e}");
            return;
        }
    };
    let parsed = t.elapsed();
    let mut scene = Scene::new();
    let t2 = Instant::now();
    skp_in::apply_payload(&mut scene, payload, Hypot::Msvc).unwrap();
    let applied = t2.elapsed();
    eprintln!("parse+adapt {parsed:?}, apply {applied:?}");
    if std::env::var_os("SKP_SIG_FACES").is_some() {
        let files = Files { zip: None };
        let mut meshes: Vec<&ingetrazo_model::Mesh> = vec![scene.loose()];
        meshes.extend(scene.groups.iter().map(|g| scene.mesh(g.mesh)));
        for (mi, m) in meshes.iter().enumerate() {
            for &f in &m.faces {
                let face = m.face(f);
                let key: Vec<String> = face
                    .outer
                    .iter()
                    .map(|&v| {
                        let p = m.position(v);
                        format!("{},{},{}", ingetrazo_golden::sig::f32hex(p.0 as f64), ingetrazo_golden::sig::f32hex(p.1 as f64), ingetrazo_golden::sig::f32hex(p.2 as f64))
                    })
                    .collect();
                let attrs = canon_dumps(&files.canonical(&ingetrazo_formats::pyjson::Value::Object(face.attrs.clone())));
                println!("{mi} {} {attrs}", key.join(","));
            }
        }
        return;
    }
    if std::env::var_os("SKP_SIG_HASH").is_some() {
        let text = canon_dumps(&scene_signature(&scene, &Files { zip: None }));
        println!("{} {}", ingetrazo_skp::dump::sha256_hex(text.as_bytes()), text.len());
        return;
    }
    if std::env::var_os("SKP_SIG_QUIET").is_none() {
        println!("{}", canon_dumps(&scene_signature(&scene, &Files { zip: None })));
    }
}
