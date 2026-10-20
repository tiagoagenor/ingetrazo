// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! QVector3D's arithmetic, bit for bit.
//!
//! QVector3D is float32: every vector operation here is the float32
//! operation Qt performs, in Qt's order, with no fused multiply-add.
//! `length()` is `std::hypot(x, y, z)`, which differs between C++ standard
//! libraries; `Hypot` names the formula, and `core._native` checks at
//! start-up that it is the one this Qt uses.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hypot {
    /// libstdc++ (`std::__hypot3`): scale by the largest component.
    LibStdCxx,
    /// MSVC STL (`std::_Hypot3`): the largest component swapped to the
    /// front, the other two divided by it, `1 + dy² + dz²` in that order.
    Msvc,
    /// libc++ (`std::__hypot`): the plain sum of squares, scaled by a power
    /// of two only near float32 over/underflow.
    LibCxx,
}

impl Default for Hypot {
    /// The formula the platform's Qt uses: MSVC's on Windows, libc++'s on
    /// macOS, libstdc++'s elsewhere (what `core._native` probes at start-up).
    fn default() -> Self {
        if cfg!(target_env = "msvc") {
            Hypot::Msvc
        } else if cfg!(target_os = "macos") {
            Hypot::LibCxx
        } else {
            Hypot::LibStdCxx
        }
    }
}

/// A `QVector3D`: three float32. Methods are Qt's operators and member
/// functions under their Qt names, each in Qt's operation order.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct V3(pub f32, pub f32, pub f32);

impl V3 {
    #[inline]
    pub fn add(self, o: V3) -> V3 {
        V3(self.0 + o.0, self.1 + o.1, self.2 + o.2)
    }
    #[inline]
    pub fn neg(self) -> V3 {
        V3(-self.0, -self.1, -self.2)
    }
    /// `v1 * v2`: component by component.
    #[inline]
    pub fn mul_v(self, o: V3) -> V3 {
        V3(self.0 * o.0, self.1 * o.1, self.2 * o.2)
    }
    /// `v1 / v2`: component by component.
    #[inline]
    pub fn div_v(self, o: V3) -> V3 {
        V3(self.0 / o.0, self.1 / o.1, self.2 / o.2)
    }
    #[inline]
    pub fn length_squared(self) -> f32 {
        self.0 * self.0 + self.1 * self.1 + self.2 * self.2
    }
    /// `isNull()`: every component exactly zero (either sign).
    #[inline]
    pub fn is_null(self) -> bool {
        self.0 == 0.0 && self.1 == 0.0 && self.2 == 0.0
    }
    #[inline]
    pub fn length(self, h: Hypot) -> f32 {
        length(self, h)
    }
    /// `normalized()`: unchanged within 1e-5 of unit length, zero when null.
    #[inline]
    pub fn normalized(self, h: Hypot) -> V3 {
        normalized(self, h)
    }
    /// `normalize()`: like `normalized()`, but a null vector stays as it is.
    #[inline]
    pub fn normalize(self, h: Hypot) -> V3 {
        let len = length(self, h);
        if fuzzy_null(len - 1.0) || fuzzy_null(len) {
            self
        } else {
            V3(self.0 / len, self.1 / len, self.2 / len)
        }
    }
    /// `QVector3D::normal(v1, v2)`.
    pub fn normal2(v1: V3, v2: V3, h: Hypot) -> V3 {
        v1.cross(v2).normalized(h)
    }
    /// `QVector3D::normal(v1, v2, v3)`.
    pub fn normal3(v1: V3, v2: V3, v3: V3, h: Hypot) -> V3 {
        v2.sub(v1).cross(v3.sub(v1)).normalized(h)
    }
    pub fn distance_to_point(self, p: V3, h: Hypot) -> f32 {
        self.sub(p).length(h)
    }
    pub fn distance_to_plane(self, plane: V3, normal: V3) -> f32 {
        self.sub(plane).dot(normal)
    }
    /// `distanceToPlane(plane1, plane2, plane3)`.
    pub fn distance_to_plane3(self, p1: V3, p2: V3, p3: V3, h: Hypot) -> f32 {
        let n = V3::normal2(p2.sub(p1), p3.sub(p1), h);
        self.sub(p1).dot(n)
    }
    pub fn distance_to_line(self, point: V3, direction: V3, h: Hypot) -> f32 {
        if direction.is_null() {
            return self.sub(point).length(h);
        }
        let p = point.add(direction.mul(self.sub(point).dot(direction)));
        self.sub(p).length(h)
    }
    #[inline]
    pub fn sub(self, o: V3) -> V3 {
        V3(self.0 - o.0, self.1 - o.1, self.2 - o.2)
    }
    #[inline]
    pub fn mul(self, s: f32) -> V3 {
        V3(self.0 * s, self.1 * s, self.2 * s)
    }
    #[inline]
    pub fn div(self, s: f32) -> V3 {
        V3(self.0 / s, self.1 / s, self.2 / s)
    }
    #[inline]
    pub fn dot(self, o: V3) -> f32 {
        self.0 * o.0 + self.1 * o.1 + self.2 * o.2
    }
    #[inline]
    pub fn cross(self, o: V3) -> V3 {
        V3(
            self.1 * o.2 - self.2 * o.1,
            self.2 * o.0 - self.0 * o.2,
            self.0 * o.1 - self.1 * o.0,
        )
    }
}

#[inline]
pub fn length(v: V3, h: Hypot) -> f32 {
    match h {
        Hypot::LibStdCxx => {
            let (x, y, z) = (v.0.abs(), v.1.abs(), v.2.abs());
            let a = if x < y {
                if y < z { z } else { y }
            } else if x < z {
                z
            } else {
                x
            };
            if a != 0.0 {
                let (xa, ya, za) = (x / a, y / a, z / a);
                a * (xa * xa + ya * ya + za * za).sqrt()
            } else {
                0.0
            }
        }
        Hypot::Msvc => {
            let (mut dx, mut dy, mut dz) = (v.0.abs(), v.1.abs(), v.2.abs());
            if dx == f32::INFINITY || dy == f32::INFINITY || dz == f32::INFINITY {
                return f32::INFINITY;
            }
            if dy > dx {
                std::mem::swap(&mut dx, &mut dy);
            }
            if dz > dx {
                std::mem::swap(&mut dx, &mut dz);
            }
            // MSVC returns dx when the others are below dx·ε; 1 + r² is then
            // 1 in float32 anyway — only the all-zero case needs the branch.
            if dx == 0.0 {
                return 0.0;
            }
            dy /= dx;
            dz /= dx;
            dx * (1.0 + dy * dy + dz * dz).sqrt()
        }
        Hypot::LibCxx => {
            // max_exponent / 2 = 64: threshold 2⁶⁴, scale 2⁻⁸⁴.
            let threshold = 2f32.powi(64);
            let over = 2f32.powi(-84);
            let m = v.0.abs().max(v.1.abs().max(v.2.abs()));
            let scale = if m > threshold {
                over
            } else if m < 1.0 / threshold {
                1.0 / over
            } else {
                1.0
            };
            let (x, y, z) = (v.0 * scale, v.1 * scale, v.2 * scale);
            (x * x + y * y + z * z).sqrt() / scale
        }
    }
}

#[inline]
fn fuzzy_null(f: f32) -> bool {
    f.abs() <= 0.00001f32
}

/// `QVector3D::normalized()`.
#[inline]
pub fn normalized(v: V3, h: Hypot) -> V3 {
    let len = length(v, h);
    if fuzzy_null(len - 1.0) {
        v
    } else if fuzzy_null(len) {
        V3(0.0, 0.0, 0.0)
    } else {
        V3(v.0 / len, v.1 / len, v.2 / len)
    }
}

#[cfg(test)]
mod hypot_tests {
    use super::*;

    fn close(a: f32, b: f64) -> bool {
        let b32 = b as f32;
        a == b32 || a == f32::from_bits(b32.to_bits() + 1) || a == f32::from_bits(b32.to_bits() - 1)
    }

    #[test]
    fn exact_on_pythagorean_triples() {
        // The plain sum of squares is exact on small integers; the scaled
        // formulas round (2,3,6) to 7.0000005, as this Qt does on MSVC.
        assert_eq!(length(V3(0.0, -12.0, 5.0), Hypot::LibCxx), 13.0);
        assert_eq!(length(V3(2.0, 3.0, 6.0), Hypot::LibCxx), 7.0);
        for h in [Hypot::LibStdCxx, Hypot::Msvc, Hypot::LibCxx] {
            assert_eq!(length(V3(3.0, 4.0, 0.0), h), 5.0);
            assert_eq!(length(V3(0.0, 0.0, 0.0), h), 0.0);
        }
    }

    #[test]
    fn libcxx_scales_near_overflow_and_underflow() {
        // Unscaled, 1e30² overflows and 1e-30² underflows in float32.
        for s in [1e30f32, 1e-30f32, 1e38f32, 1e-38f32] {
            let v = V3(s, s * 0.5, -s * 0.25);
            let want = ((s as f64).powi(2) * (1.0 + 0.25 + 0.0625)).sqrt();
            let got = length(v, Hypot::LibCxx);
            assert!(got.is_finite() && close(got, want), "{s}: {got} vs {want}");
        }
    }

    #[test]
    fn msvc_takes_the_largest_component_first() {
        // Same components in every order: the same length.
        let l = length(V3(0.3, 1.7, -0.9), Hypot::Msvc);
        for v in [V3(1.7, 0.3, -0.9), V3(-0.9, 0.3, 1.7), V3(0.3, -0.9, 1.7)] {
            let w = length(v, Hypot::Msvc);
            assert!(close(w, l as f64), "{w} vs {l}");
        }
        assert_eq!(length(V3(f32::INFINITY, 1.0, 0.0), Hypot::Msvc), f32::INFINITY);
    }
}
