// SPDX-License-Identifier: GPL-3.0-or-later
//! Bindings to Manifold's C API (`manifoldc`), for the functions
//! `core/solids.py` uses through Python's `manifold3d`: build a triangle
//! mesh with per-triangle face ids, run the booleans, read the result mesh.
//!
//! Manifold is built statically by `build.rs` from a pinned release (3.5.4,
//! the version of the Python wheel). The raw declarations are in [`ffi`]; the
//! safe wrappers own their C++ objects and free them on drop.
//!
//! # Building
//!
//! Needs CMake (3.18+) and a C++17 compiler (MSVC, gcc or clang) on the
//! build machine. `build.rs` downloads two tarballs (Manifold v3.5.4 and the
//! Clipper2 commit its CMake pins), verifies their SHA-256 and builds them
//! single-threaded (no TBB) as static libraries, about half a minute on a
//! 16-core machine in release mode. Set `MANIFOLD_SYS_CACHE` to keep the
//! downloads across `cargo clean` (a CI cache), or `MANIFOLD_SRC_DIR` and
//! `CLIPPER2_SRC_DIR` to unpacked copies for an offline build. On Windows,
//! `CMAKE_GENERATOR=Ninja` from a Developer Prompt works when CMake cannot
//! find the Visual Studio instance.
//!
//! A `Manifold` is an immutable, reference-counted value in C++; the wrappers
//! are `Send + Sync` accordingly (Manifold's own docs: const methods are
//! thread-safe).

use std::os::raw::{c_int, c_void};

/// Version of the Manifold source this crate links.
pub const MANIFOLD_VERSION: &str = "3.5.4";

#[allow(non_camel_case_types, non_snake_case, dead_code)]
pub mod ffi {
    use std::os::raw::{c_int, c_void};

    #[repr(C)]
    pub struct ManifoldManifold {
        _p: [u8; 0],
    }
    #[repr(C)]
    pub struct ManifoldManifoldVec {
        _p: [u8; 0],
    }
    #[repr(C)]
    pub struct ManifoldMeshGL64 {
        _p: [u8; 0],
    }

    pub const MANIFOLD_ADD: c_int = 0;
    pub const MANIFOLD_SUBTRACT: c_int = 1;
    pub const MANIFOLD_INTERSECT: c_int = 2;

    extern "C" {
        pub fn manifold_alloc_manifold() -> *mut ManifoldManifold;
        pub fn manifold_alloc_manifold_vec() -> *mut ManifoldManifoldVec;
        pub fn manifold_alloc_meshgl64() -> *mut ManifoldMeshGL64;
        pub fn manifold_delete_manifold(m: *mut ManifoldManifold);
        pub fn manifold_delete_manifold_vec(ms: *mut ManifoldManifoldVec);
        pub fn manifold_delete_meshgl64(m: *mut ManifoldMeshGL64);

        pub fn manifold_meshgl64(
            mem: *mut c_void,
            vert_props: *mut f64,
            n_verts: usize,
            n_props: usize,
            tri_verts: *mut u64,
            n_tris: usize,
        ) -> *mut ManifoldMeshGL64;
        pub fn manifold_get_meshgl64(mem: *mut c_void, m: *mut ManifoldManifold) -> *mut ManifoldMeshGL64;
        pub fn manifold_meshgl64_num_prop(m: *mut ManifoldMeshGL64) -> usize;
        pub fn manifold_meshgl64_vert_properties_length(m: *mut ManifoldMeshGL64) -> usize;
        pub fn manifold_meshgl64_tri_length(m: *mut ManifoldMeshGL64) -> usize;
        pub fn manifold_meshgl64_face_id_length(m: *mut ManifoldMeshGL64) -> usize;
        pub fn manifold_meshgl64_vert_properties(mem: *mut c_void, m: *mut ManifoldMeshGL64) -> *mut f64;
        pub fn manifold_meshgl64_tri_verts(mem: *mut c_void, m: *mut ManifoldMeshGL64) -> *mut u64;
        pub fn manifold_meshgl64_face_id(mem: *mut c_void, m: *mut ManifoldMeshGL64) -> *mut u64;

        pub fn manifold_of_meshgl64(mem: *mut c_void, mesh: *mut ManifoldMeshGL64) -> *mut ManifoldManifold;
        pub fn manifold_status(m: *mut ManifoldManifold) -> c_int;
        pub fn manifold_is_empty(m: *mut ManifoldManifold) -> c_int;

        pub fn manifold_manifold_empty_vec(mem: *mut c_void) -> *mut ManifoldManifoldVec;
        pub fn manifold_manifold_vec_push_back(ms: *mut ManifoldManifoldVec, m: *mut ManifoldManifold);
        pub fn manifold_batch_boolean(mem: *mut c_void, ms: *mut ManifoldManifoldVec, op: c_int) -> *mut ManifoldManifold;
        pub fn manifold_union(mem: *mut c_void, a: *mut ManifoldManifold, b: *mut ManifoldManifold) -> *mut ManifoldManifold;
        pub fn manifold_difference(mem: *mut c_void, a: *mut ManifoldManifold, b: *mut ManifoldManifold) -> *mut ManifoldManifold;
        pub fn manifold_intersection(mem: *mut c_void, a: *mut ManifoldManifold, b: *mut ManifoldManifold) -> *mut ManifoldManifold;

        /// Not in Manifold: `shim.cpp`. Sets a MeshGL64's `faceID`.
        pub fn ingetrazo_meshgl64_set_face_id(mesh: *mut c_void, ids: *const u64, n: usize);
    }
}

/// `ManifoldError` (`Manifold::Error`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    NoError,
    NonFiniteVertex,
    NotManifold,
    VertexIndexOutOfBounds,
    PropertiesWrongLength,
    MissingPositionProperties,
    MergeVectorsDifferentLengths,
    MergeIndexOutOfBounds,
    TransformWrongLength,
    RunIndexWrongLength,
    FaceIdWrongLength,
    InvalidConstruction,
    ResultTooLarge,
    InvalidTangents,
    Cancelled,
    Unknown(i32),
}

impl Status {
    fn from_c(code: c_int) -> Status {
        match code {
            0 => Status::NoError,
            1 => Status::NonFiniteVertex,
            2 => Status::NotManifold,
            3 => Status::VertexIndexOutOfBounds,
            4 => Status::PropertiesWrongLength,
            5 => Status::MissingPositionProperties,
            6 => Status::MergeVectorsDifferentLengths,
            7 => Status::MergeIndexOutOfBounds,
            8 => Status::TransformWrongLength,
            9 => Status::RunIndexWrongLength,
            10 => Status::FaceIdWrongLength,
            11 => Status::InvalidConstruction,
            12 => Status::ResultTooLarge,
            13 => Status::InvalidTangents,
            14 => Status::Cancelled,
            n => Status::Unknown(n),
        }
    }
}

/// `OpType` of `Manifold.batch_boolean`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpType {
    Add,
    Subtract,
    Intersect,
}

/// A `MeshGL64` the caller owns: interleaved vertex properties (the first
/// three are the position), triangle corner indices, and a face id per
/// triangle.
pub struct MeshGl64 {
    ptr: *mut ffi::ManifoldMeshGL64,
}

unsafe impl Send for MeshGl64 {}

impl MeshGl64 {
    /// `Mesh64(vert_properties, tri_verts, face_id=...)`. `verts` holds
    /// `num_prop` values per vertex; `tris` three indices per triangle;
    /// `face_id`, when given, one id per triangle.
    pub fn new(verts: &[f64], num_prop: usize, tris: &[u64], face_id: Option<&[u64]>) -> MeshGl64 {
        assert!(num_prop >= 3 && verts.len().is_multiple_of(num_prop) && tris.len().is_multiple_of(3));
        if let Some(f) = face_id {
            assert_eq!(f.len(), tris.len() / 3, "one face id per triangle");
        }
        // The C constructor copies the arrays; it takes them as `*mut` only
        // for want of a `const`.
        let mut v = verts.to_vec();
        let mut t = tris.to_vec();
        unsafe {
            let mem = ffi::manifold_alloc_meshgl64();
            let ptr = ffi::manifold_meshgl64(mem as *mut c_void, v.as_mut_ptr(), v.len() / num_prop, num_prop, t.as_mut_ptr(), t.len() / 3);
            if let Some(f) = face_id {
                ffi::ingetrazo_meshgl64_set_face_id(ptr as *mut c_void, f.as_ptr(), f.len());
            }
            MeshGl64 { ptr }
        }
    }

    pub fn num_prop(&self) -> usize {
        unsafe { ffi::manifold_meshgl64_num_prop(self.ptr) }
    }

    /// Interleaved vertex properties (`num_prop` per vertex).
    pub fn vert_properties(&self) -> Vec<f64> {
        unsafe {
            let n = ffi::manifold_meshgl64_vert_properties_length(self.ptr);
            let mut out = vec![0f64; n];
            if n > 0 {
                ffi::manifold_meshgl64_vert_properties(out.as_mut_ptr() as *mut c_void, self.ptr);
            }
            out
        }
    }

    /// Triangle corners, three vertex indices per triangle.
    pub fn tri_verts(&self) -> Vec<u64> {
        unsafe {
            let n = ffi::manifold_meshgl64_tri_length(self.ptr);
            let mut out = vec![0u64; n];
            if n > 0 {
                ffi::manifold_meshgl64_tri_verts(out.as_mut_ptr() as *mut c_void, self.ptr);
            }
            out
        }
    }

    /// The face id of every triangle (empty when the mesh has none).
    pub fn face_id(&self) -> Vec<u64> {
        unsafe {
            let n = ffi::manifold_meshgl64_face_id_length(self.ptr);
            let mut out = vec![0u64; n];
            if n > 0 {
                ffi::manifold_meshgl64_face_id(out.as_mut_ptr() as *mut c_void, self.ptr);
            }
            out
        }
    }
}

impl Drop for MeshGl64 {
    fn drop(&mut self) {
        unsafe { ffi::manifold_delete_meshgl64(self.ptr) }
    }
}

/// A Manifold solid (`manifold3d.Manifold`).
pub struct Manifold {
    ptr: *mut ffi::ManifoldManifold,
}

unsafe impl Send for Manifold {}
unsafe impl Sync for Manifold {}

impl Manifold {
    /// `Manifold(mesh)`; check [`Manifold::status`] before using it.
    pub fn from_mesh(mesh: &MeshGl64) -> Manifold {
        unsafe {
            let mem = ffi::manifold_alloc_manifold();
            Manifold { ptr: ffi::manifold_of_meshgl64(mem as *mut c_void, mesh.ptr) }
        }
    }

    /// `Manifold.status()`.
    pub fn status(&self) -> Status {
        Status::from_c(unsafe { ffi::manifold_status(self.ptr) })
    }

    /// `Manifold.is_empty()`.
    pub fn is_empty(&self) -> bool {
        unsafe { ffi::manifold_is_empty(self.ptr) != 0 }
    }

    /// `a + b`.
    pub fn union(&self, other: &Manifold) -> Manifold {
        unsafe {
            let mem = ffi::manifold_alloc_manifold();
            Manifold { ptr: ffi::manifold_union(mem as *mut c_void, self.ptr, other.ptr) }
        }
    }

    /// `a - b`.
    pub fn difference(&self, other: &Manifold) -> Manifold {
        unsafe {
            let mem = ffi::manifold_alloc_manifold();
            Manifold { ptr: ffi::manifold_difference(mem as *mut c_void, self.ptr, other.ptr) }
        }
    }

    /// `a ^ b`.
    pub fn intersection(&self, other: &Manifold) -> Manifold {
        unsafe {
            let mem = ffi::manifold_alloc_manifold();
            Manifold { ptr: ffi::manifold_intersection(mem as *mut c_void, self.ptr, other.ptr) }
        }
    }

    /// `Manifold.batch_boolean(list, op)`.
    pub fn batch_boolean(list: &[&Manifold], op: OpType) -> Manifold {
        unsafe {
            let vmem = ffi::manifold_alloc_manifold_vec();
            let vec = ffi::manifold_manifold_empty_vec(vmem as *mut c_void);
            for m in list {
                ffi::manifold_manifold_vec_push_back(vec, m.ptr);
            }
            let code = match op {
                OpType::Add => ffi::MANIFOLD_ADD,
                OpType::Subtract => ffi::MANIFOLD_SUBTRACT,
                OpType::Intersect => ffi::MANIFOLD_INTERSECT,
            };
            let mem = ffi::manifold_alloc_manifold();
            let out = ffi::manifold_batch_boolean(mem as *mut c_void, vec, code);
            ffi::manifold_delete_manifold_vec(vec);
            Manifold { ptr: out }
        }
    }

    /// `Manifold.to_mesh64()`.
    pub fn to_mesh64(&self) -> MeshGl64 {
        unsafe {
            let mem = ffi::manifold_alloc_meshgl64();
            MeshGl64 { ptr: ffi::manifold_get_meshgl64(mem as *mut c_void, self.ptr) }
        }
    }
}

impl Drop for Manifold {
    fn drop(&mut self) {
        unsafe { ffi::manifold_delete_manifold(self.ptr) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An axis-aligned box as 12 outward triangles; one face id per side.
    fn cube(x0: f64, y0: f64, z0: f64, s: f64) -> MeshGl64 {
        let mut v = Vec::new();
        for i in 0..8u32 {
            v.extend([x0 + s * (i & 1) as f64, y0 + s * ((i >> 1) & 1) as f64, z0 + s * ((i >> 2) & 1) as f64]);
        }
        let t: Vec<u64> = vec![
            0, 2, 1, 1, 2, 3, // z0
            4, 5, 6, 5, 7, 6, // z1
            0, 1, 4, 1, 5, 4, // y0
            2, 6, 3, 3, 6, 7, // y1
            0, 4, 2, 2, 4, 6, // x0
            1, 3, 5, 3, 7, 5, // x1
        ];
        let f: Vec<u64> = (0..12).map(|i| i / 2).collect();
        MeshGl64::new(&v, 3, &t, Some(&f))
    }

    #[test]
    fn booleans_run_and_keep_face_ids() {
        let a = Manifold::from_mesh(&cube(0.0, 0.0, 0.0, 2.0));
        let b = Manifold::from_mesh(&cube(1.0, 1.0, 1.0, 2.0));
        assert_eq!(a.status(), Status::NoError);
        let both = a.intersection(&b).to_mesh64();
        // The overlap is a unit cube.
        assert_eq!(both.vert_properties().len() / both.num_prop(), 8);
        assert_eq!(both.tri_verts().len() / 3, 12);
        assert_eq!(both.face_id().len(), 12);
        let all = Manifold::batch_boolean(&[&a, &b], OpType::Add);
        assert!(!all.is_empty());
        assert!(a.difference(&b).to_mesh64().tri_verts().len() > 36);
        assert!(a.union(&b).to_mesh64().tri_verts().len() > 36);
    }

    #[test]
    fn an_open_mesh_is_not_a_manifold() {
        let v = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let m = Manifold::from_mesh(&MeshGl64::new(&v, 3, &[0, 1, 2], None));
        assert_eq!(m.status(), Status::NotManifold);
    }
}
