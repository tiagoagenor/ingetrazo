// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The image helpers against Qt (scripts/golden/make_qimage_fixtures.py):
//! the smooth 1 x 1 average of a PNG and the cut-out test of an image.

use std::path::{Path, PathBuf};

use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_formats_mesh::img::{image_has_cutout, smooth_average_rgb};

/// The images of two to four pixels with partial alpha whose average is a
/// level or two off (see `img.rs`).
const TINY_ALPHA: [&str; 5] = ["rgba-1x2.png", "rgba-2x1.png", "rgba-2x2.png", "rgba-binary-2x1.png", "palette-alpha-1x2.png"];

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/qimage")
}

#[test]
fn smooth_average_and_cutout_match_qt() {
    let text = std::fs::read_to_string(data().join("index.json")).unwrap();
    let Value::Array(cases) = loads(&text).unwrap() else { panic!() };
    let mut wrong = Vec::new();
    let mut inexact = 0;
    for case in &cases {
        let Some(Value::Str(file)) = case.get("file") else { panic!() };
        let bytes = std::fs::read(data().join(file)).unwrap();
        let Some(Value::Array(want)) = case.get("smooth") else { panic!() };
        let want: Vec<u8> = want.iter().map(|v| v.as_i64().unwrap() as u8).collect();
        match smooth_average_rgb(&bytes) {
            Ok(got) if got[..] == want[..] => {}
            // Two or four pixels with partial alpha: Qt scales those with
            // a precision this port does not reproduce (a level or two).
            Ok(got) if TINY_ALPHA.contains(&file.as_str()) && got.iter().zip(&want).all(|(&g, &w)| g.abs_diff(w) <= 2) => inexact += 1,
            other => wrong.push(format!("{file}: smooth {other:?}, Qt {want:?}")),
        }
        let want_cut = matches!(case.get("cutout"), Some(Value::Bool(true)));
        if image_has_cutout(&bytes) != want_cut {
            wrong.push(format!("{file}: cutout, Qt {want_cut}"));
        }
    }
    assert!(wrong.is_empty(), "{} of {} images differ:\n{}", wrong.len(), cases.len(), wrong.join("\n"));
    assert_eq!(cases.len(), 123);
    assert_eq!(inexact, TINY_ALPHA.len(), "the known approximations");
}
