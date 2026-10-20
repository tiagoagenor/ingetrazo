// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Prints the counts of a parsed .skp and the parse time: a development aid
//! for comparing with openskp (`scripts/golden/skp_summary.py`).
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).expect("path of a .skp");
    let t = Instant::now();
    let m = ingetrazo_skp::parse_file(std::path::Path::new(&path)).unwrap_or_else(|e| panic!("{e}"));
    let dt = t.elapsed();
    let count = |d: &ingetrazo_skp::Definition| (d.vertices.len(), d.edges.len(), d.faces.len(), d.instances.len());
    let mut tot = (0, 0, 0, 0);
    for d in &m.definitions {
        let c = count(d);
        tot = (tot.0 + c.0, tot.1 + c.1, tot.2 + c.2, tot.3 + c.3);
    }
    println!(
        "version={} units={} defs={} root={:?} tot={:?} layers={} materials={} by_id={} pages={} dims={} styles={}",
        m.version,
        m.units.as_ref().map(|u| format!("'{u}'")).unwrap_or_else(|| "None".into()),
        m.definitions.len(),
        count(&m.root),
        tot,
        m.layers.len(),
        m.materials.len(),
        m.materials_by_id.len(),
        m.pages.len(),
        m.dimensions.len(),
        m.styles.len()
    );
    eprintln!("parse {dt:?}");
}
