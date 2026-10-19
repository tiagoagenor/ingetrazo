// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `core.bim.face_set_volume`'s signed tetrahedron sum over a lazy mesh's
//! triangles: per triangle Qt's float32 `crossProduct(t1, t2)` then
//! `dotProduct(t0, ·)`, widened to f64, divided by 6 and added in order —
//! the Python's float arithmetic, one triangle after the other.

/// The signed sum (the caller takes `abs`). ``tris`` index ``pos``.
pub fn signed_volume(pos: &[[f32; 3]], tris: &[[u32; 3]]) -> f64 {
    let mut total = 0.0f64;
    for t in tris {
        let a = pos[t[0] as usize];
        let b = pos[t[1] as usize];
        let c = pos[t[2] as usize];
        // QVector3D::crossProduct(b, c), component by component.
        let cx = b[1] * c[2] - b[2] * c[1];
        let cy = b[2] * c[0] - b[0] * c[2];
        let cz = b[0] * c[1] - b[1] * c[0];
        // QVector3D::dotProduct(a, cross), in float32.
        let d = a[0] * cx + a[1] * cy + a[2] * cz;
        total += d as f64 / 6.0;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_cube() {
        let p = [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        ];
        // Outward triangles of the six faces.
        let t = [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ];
        // Twelve sixths in f64: within an ulp of 1 (the Python sums the
        // same terms the same way; test_solid_volume_native holds the two
        // equal).
        assert!((signed_volume(&p, &t) - 1.0).abs() < 1e-15);
        let flipped: Vec<[u32; 3]> = t.iter().map(|x| [x[0], x[2], x[1]]).collect();
        assert!((signed_volume(&p, &flipped) + 1.0).abs() < 1e-15);
    }
}
