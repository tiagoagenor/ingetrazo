// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The Python app's bridge to `ingetrazo-geom`. Optional: `core/_native.py`
//! imports this module when it is built and falls back to the Python code
//! otherwise. Only conversions live here: numpy arrays in, kernels called,
//! numpy arrays out. Deleted with the Python app (rust-rewrite-plan M10).

use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray1};
use pyo3::prelude::*;

use ingetrazo_geom::{
    arrange, boxes, conform, facegeom, fuse, gltape, hlr, instances, intersect, occlusion, parity, plan,
    section, tiff, volume, weld,
};

#[pymodule]
fn ingetrazo_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<HlrOccluders>()?;
    m.add_function(wrap_pyfunction!(touching_boxes, m)?)?;
    m.add_class::<ParityTris>()?;
    m.add_function(wrap_pyfunction!(occlusion_span, m)?)?;
    m.add_function(wrap_pyfunction!(face_geometry, m)?)?;
    m.add_function(wrap_pyfunction!(qt_vector_probe, m)?)?;
    m.add_function(wrap_pyfunction!(weld_points, m)?)?;
    m.add_function(wrap_pyfunction!(plan_topology, m)?)?;
    m.add_function(wrap_pyfunction!(signed_volume, m)?)?;
    m.add_function(wrap_pyfunction!(arrangement_splits, m)?)?;
    m.add_function(wrap_pyfunction!(section_loops, m)?)?;
    m.add_class::<PolySet>()?;
    m.add_function(wrap_pyfunction!(conform_t_junctions, m)?)?;
    m.add_function(wrap_pyfunction!(shell_roots, m)?)?;
    m.add_function(wrap_pyfunction!(lzw_decode, m)?)?;
    m.add_function(wrap_pyfunction!(unpackbits, m)?)?;
    m.add_function(wrap_pyfunction!(fuse_regions, m)?)?;
    m.add_function(wrap_pyfunction!(tri_normals, m)?)?;
    m.add_class::<GlReplay>()?;
    m.add_class::<InstanceProto>()?;
    m.add_class::<InstanceSet>()?;
    Ok(())
}

/// The triangles of one hidden-line pass, in camera space, with a 2D grid
/// over their bounding boxes. Built once per view by `core.hlr`.
#[pyclass(frozen)]
struct HlrOccluders {
    inner: hlr::Occluders,
}

#[pymethods]
impl HlrOccluders {
    /// ``tc``: (T,3,3) triangle vertices in camera space (x, y, depth).
    #[new]
    fn new(tc: PyReadonlyArray3<'_, f64>) -> PyResult<Self> {
        let tc = tc.as_array();
        let n = tc.shape()[0];
        let mut tris = Vec::with_capacity(n);
        for t in 0..n {
            let mut v = [[0.0f64; 3]; 3];
            for k in 0..3 {
                for c in 0..3 {
                    v[k][c] = tc[[t, k, c]];
                }
            }
            tris.push(v);
        }
        Ok(Self { inner: hlr::Occluders::new(tris) })
    }

    /// Visible t-intervals of every edge: core.hlr.visible_spans for each
    /// row, with the same candidate filter as hlr_drawing. ``a``/``b``:
    /// (E,3) endpoints in camera space. Returns ``(offsets (E+1,), spans
    /// (S,2))``: edge i owns spans[offsets[i]:offsets[i+1]].
    fn visible_spans<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<'_, f64>,
        b: PyReadonlyArray2<'_, f64>,
        eps: f64,
    ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray2<f64>>)> {
        let a = a.as_array();
        let b = b.as_array();
        let n = a.shape()[0];
        let edges: Vec<([f64; 3], [f64; 3])> = (0..n)
            .map(|i| {
                ([a[[i, 0]], a[[i, 1]], a[[i, 2]]], [b[[i, 0]], b[[i, 1]], b[[i, 2]]])
            })
            .collect();
        let inner = &self.inner;
        let (offsets, spans) = py.detach(|| inner.visible_spans_all(&edges, eps));
        let flat: Vec<Vec<f64>> = spans.iter().map(|s| vec![s.0, s.1]).collect();
        let off = PyArray1::from_vec(py, offsets);
        let sp = if flat.is_empty() {
            PyArray2::zeros(py, [0, 2], false)
        } else {
            PyArray2::from_vec2(py, &flat)?
        };
        Ok((off, sp))
    }

    /// The Profiles test of hlr_drawing for many visible spans at once:
    /// for each row, is the background on either side of the span's
    /// midpoint? ``mid`` (N,2), ``nrm`` (N,2) the sideways probe offset,
    /// ``zm`` (N,) the span's depth at the midpoint.
    fn profile_flags<'py>(
        &self,
        py: Python<'py>,
        mid: PyReadonlyArray2<'_, f64>,
        nrm: PyReadonlyArray2<'_, f64>,
        zm: PyReadonlyArray1<'_, f64>,
        delta: f64,
        ptol: f64,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let mid = mid.as_array();
        let nrm = nrm.as_array();
        let zm = zm.as_array();
        let n = mid.shape()[0];
        let probes: Vec<([f64; 2], [f64; 2], f64)> = (0..n)
            .map(|i| ([mid[[i, 0]], mid[[i, 1]]], [nrm[[i, 0]], nrm[[i, 1]]], zm[i]))
            .collect();
        let inner = &self.inner;
        let out = py.detach(|| inner.profile_flags(&probes, delta, ptol));
        Ok(PyArray1::from_vec(py, out))
    }
}

fn rows3(a: &numpy::ndarray::ArrayView2<'_, f64>) -> Vec<[f64; 3]> {
    (0..a.shape()[0]).map(|i| [a[[i, 0]], a[[i, 1]], a[[i, 2]]]).collect()
}

/// core.intersect._touching for all rows at once: the index pairs ``(i,
/// j)`` whose boxes touch (within ``eps``), sorted by i then j; ``same``:
/// j > i only. ``lo_*``/``hi_*``: (N,3) box corners.
#[pyfunction]
fn touching_boxes<'py>(
    py: Python<'py>,
    lo_a: PyReadonlyArray2<'_, f64>,
    hi_a: PyReadonlyArray2<'_, f64>,
    lo_b: PyReadonlyArray2<'_, f64>,
    hi_b: PyReadonlyArray2<'_, f64>,
    same: bool,
    eps: f64,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    let a: Vec<boxes::Box3> = rows3(&lo_a.as_array()).into_iter()
        .zip(rows3(&hi_a.as_array())).collect();
    let b: Vec<boxes::Box3> = rows3(&lo_b.as_array()).into_iter()
        .zip(rows3(&hi_b.as_array())).collect();
    let (ii, jj) = py.detach(|| boxes::touching(&a, &b, same, eps));
    Ok((PyArray1::from_vec(py, ii), PyArray1::from_vec(py, jj)))
}

/// A packed triangle set (core.orient._PackedTris) with a ray index, for
/// the parity queries of orientation and of the push/pull rebuild.
#[pyclass(frozen)]
struct ParityTris {
    inner: parity::Tris,
}

#[pymethods]
impl ParityTris {
    /// ``a``/``e1``/``e2``: (N,3) vertex A and the two edge vectors;
    /// ``face_idx``: (N,) the face of each triangle.
    #[new]
    fn new(
        a: PyReadonlyArray2<'_, f64>,
        e1: PyReadonlyArray2<'_, f64>,
        e2: PyReadonlyArray2<'_, f64>,
        face_idx: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<Self> {
        let face = face_idx.as_array().to_vec();
        Ok(Self { inner: parity::Tris::new(rows3(&a.as_array()), rows3(&e1.as_array()),
                                           rows3(&e2.as_array()), face) })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Crossings ahead of ``origin`` (3,) along each row of ``dirs`` (R,3),
    /// over the triangles with ``mask`` True (all when None) that are not of
    /// face ``skip``. Returns a list of R ints.
    #[pyo3(signature = (origin, dirs, mask=None, skip=None))]
    fn crossings(
        &self,
        py: Python<'_>,
        origin: (f64, f64, f64),
        dirs: PyReadonlyArray2<'_, f64>,
        mask: Option<PyReadonlyArray1<'_, bool>>,
        skip: Option<i64>,
    ) -> PyResult<Vec<i64>> {
        let d = rows3(&dirs.as_array());
        let m: Option<Vec<bool>> = mask.map(|m| m.as_array().to_vec());
        if let Some(ref mm) = m {
            if mm.len() != self.inner.len() {
                return Err(pyo3::exceptions::PyValueError::new_err("mask length"));
            }
        }
        let inner = &self.inner;
        let o = [origin.0, origin.1, origin.2];
        Ok(py.detach(|| inner.crossings(o, &d, m.as_deref(), skip)))
    }
}

fn hypot_of(name: &str) -> PyResult<facegeom::Hypot> {
    match name {
        "libstdc++" => Ok(facegeom::Hypot::LibStdCxx),
        "msvc" => Ok(facegeom::Hypot::Msvc),
        "libc++" => Ok(facegeom::Hypot::LibCxx),
        _ => Err(pyo3::exceptions::PyValueError::new_err("unknown hypot")),
    }
}

/// QVector3D's own arithmetic on each row of ``v`` (N,3) float32, for the
/// start-up check that ``hypot`` is this Qt's: ``(length (N,), normalized
/// (N,3), plane u (N,3), plane v (N,3))`` — core.triangulate.plane_axes.
#[pyfunction]
fn qt_vector_probe<'py>(
    py: Python<'py>,
    v: PyReadonlyArray2<'_, f32>,
    hypot: &str,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray2<f32>>,
               Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
    let h = hypot_of(hypot)?;
    let v = v.as_array();
    let n = v.shape()[0];
    let (mut ln, mut nz, mut pu, mut pv) = (Vec::with_capacity(n), Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let x = facegeom::V3(v[[i, 0]], v[[i, 1]], v[[i, 2]]);
        ln.push(facegeom::length(x, h));
        let m = facegeom::normalized(x, h);
        nz.push(vec![m.0, m.1, m.2]);
        let (a, b) = facegeom::plane_axes(x, h);
        pu.push(vec![a.0, a.1, a.2]);
        pv.push(vec![b.0, b.1, b.2]);
    }
    Ok((PyArray1::from_vec(py, ln), PyArray2::from_vec2(py, &nz)?,
        PyArray2::from_vec2(py, &pu)?, PyArray2::from_vec2(py, &pv)?))
}

/// views.viewport._face_geometry for every face of a mesh given as arrays:
/// ``pos`` (V,3) float32 vertex positions, ``corners`` every ring's vertex
/// indices (rings concatenated, each face's outer ring first),
/// ``ring_sizes`` and ``ring_counts`` (rings per face). Returns ``(normal
/// (F,3) f32, ln (F,) f32 — the Newell length, area = ln / 2 — centroid
/// (F,3) f32, NaN for a face of under three corners, tris (T,3) u32 vertex
/// indices, tri_face (T,) u32, unit (F,3) f32, tris_unit, tri_face_unit)``;
/// each face's triangles in order. ``unit`` and its triangles are the
/// viewport's ``_normal_of`` / ``_tris_of`` (Qt's normalized()).
/// ``compensated_sum``: this Python's sum() of floats is Neumaier's (3.12+).
#[pyfunction]
fn face_geometry<'py>(
    py: Python<'py>,
    pos: PyReadonlyArray2<'_, f32>,
    corners: PyReadonlyArray1<'_, i64>,
    ring_sizes: PyReadonlyArray1<'_, i64>,
    ring_counts: PyReadonlyArray1<'_, i64>,
    hypot: &str,
    compensated_sum: bool,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<f32>>,
               Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<u32>>,
               Bound<'py, PyArray1<u32>>, Bound<'py, PyArray2<f32>>,
               Bound<'py, PyArray2<u32>>, Bound<'py, PyArray1<u32>>)> {
    use rayon::prelude::*;
    let h = hypot_of(hypot)?;
    let p = pos.as_array();
    let pts: Vec<facegeom::V3> = (0..p.shape()[0])
        .map(|i| facegeom::V3(p[[i, 0]], p[[i, 1]], p[[i, 2]])).collect();
    let nv = pts.len() as i64;
    let corners: Vec<u32> = corners.as_array().iter().map(|&c| {
        if c < 0 || c >= nv { u32::MAX } else { c as u32 }
    }).collect();
    if corners.iter().any(|&c| c == u32::MAX) {
        return Err(pyo3::exceptions::PyValueError::new_err("corner out of range"));
    }
    let sizes = ring_sizes.as_array().to_vec();
    let counts = ring_counts.as_array().to_vec();
    // Each face's first ring, and each ring's first corner.
    let mut ring_start = Vec::with_capacity(sizes.len() + 1);
    let mut acc = 0usize;
    for &s in &sizes {
        ring_start.push(acc);
        acc += s as usize;
    }
    ring_start.push(acc);
    if acc != corners.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("ring sizes"));
    }
    let mut face_ring = Vec::with_capacity(counts.len() + 1);
    let mut r = 0usize;
    for &c in &counts {
        face_ring.push(r);
        r += c as usize;
    }
    face_ring.push(r);
    if r != sizes.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("ring counts"));
    }
    let geoms: Vec<facegeom::FaceGeom> = py.detach(|| {
        (0..counts.len()).into_par_iter().map(|f| {
            let rings: Vec<&[u32]> = (face_ring[f]..face_ring[f + 1])
                .map(|k| &corners[ring_start[k]..ring_start[k + 1]]).collect();
            facegeom::face_geometry(&pts, &rings, h, compensated_sum)
        }).collect()
    });
    let nf = geoms.len();
    let mut normal = Vec::with_capacity(nf);
    let mut ln = Vec::with_capacity(nf);
    let mut cen = Vec::with_capacity(nf);
    let mut tris = Vec::new();
    let mut tri_face = Vec::new();
    let mut unit = Vec::with_capacity(nf);
    let mut tris_u = Vec::new();
    let mut tri_face_u = Vec::new();
    for (f, g) in geoms.into_iter().enumerate() {
        unit.push(vec![g.unit.0, g.unit.1, g.unit.2]);
        for t in g.tris_unit.as_ref().unwrap_or(&g.tris) {
            tris_u.push(t.to_vec());
            tri_face_u.push(f as u32);
        }
        normal.push(vec![g.normal.0, g.normal.1, g.normal.2]);
        ln.push(g.ln);
        cen.push(match g.centroid {
            Some(c) => vec![c.0, c.1, c.2],
            None => vec![f32::NAN; 3],
        });
        for t in g.tris {
            tris.push(t.to_vec());
            tri_face.push(f as u32);
        }
    }
    let tris_arr = if tris.is_empty() {
        PyArray2::zeros(py, [0, 3], false)
    } else {
        PyArray2::from_vec2(py, &tris)?
    };
    let normal_arr = if nf == 0 { PyArray2::zeros(py, [0, 3], false) } else { PyArray2::from_vec2(py, &normal)? };
    let cen_arr = if nf == 0 { PyArray2::zeros(py, [0, 3], false) } else { PyArray2::from_vec2(py, &cen)? };
    let unit_arr = if nf == 0 { PyArray2::zeros(py, [0, 3], false) } else { PyArray2::from_vec2(py, &unit)? };
    let tris_u_arr = if tris_u.is_empty() {
        PyArray2::zeros(py, [0, 3], false)
    } else {
        PyArray2::from_vec2(py, &tris_u)?
    };
    Ok((normal_arr, PyArray1::from_vec(py, ln), cen_arr, tris_arr, PyArray1::from_vec(py, tri_face),
        unit_arr, tris_u_arr, PyArray1::from_vec(py, tri_face_u)))
}

/// core.lazy_mesh.weld (Mesh.bulk_weld on an empty mesh): ``pos`` (N,3)
/// float32 points, ``scale`` the weld cell (10**decimals), ``tol2`` the
/// probe's squared distance. Returns ``(vertex positions (V,3) float32,
/// point → vertex (N,), the point each vertex was made at (V,))``.
#[pyfunction]
fn weld_points<'py>(
    py: Python<'py>,
    pos: PyReadonlyArray2<'_, f32>,
    scale: f64,
    tol2: f64,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<i64>>,
               Bound<'py, PyArray1<i64>>)> {
    let p = pos.as_array();
    let pts: Vec<[f32; 3]> = (0..p.shape()[0]).map(|i| [p[[i, 0]], p[[i, 1]], p[[i, 2]]]).collect();
    let (vpos, ids, origin) = py.detach(|| weld::weld(&pts, scale, tol2));
    let rows: Vec<Vec<f32>> = vpos.iter().map(|v| v.to_vec()).collect();
    let arr = if rows.is_empty() { PyArray2::zeros(py, [0, 3], false) } else { PyArray2::from_vec2(py, &rows)? };
    Ok((arr, PyArray1::from_vec(py, ids), PyArray1::from_vec(py, origin)))
}

/// core.lazy_mesh.plan after the weld (add_edges_welded + add_faces_welded
/// on an empty mesh): ``inverse`` (N,) point → vertex, ``edges`` (E,2)
/// point pairs, ``flags``/``curves``/``layers`` (E,) or None, ``rings``
/// every ring corner (point index), ``sizes`` corners per ring, ``counts``
/// rings per face — every size and count > 0, ``sum(sizes) == len(rings)``.
/// Returns ``(edges (E',2), soft, hidden, curve (-1 = None), layer
/// (-1 = None), corners, ef_start (E'+1,), ef_face)``.
#[pyfunction]
#[pyo3(signature = (inverse, edges, flags, curves, layers, rings, sizes, counts))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn plan_topology<'py>(
    py: Python<'py>,
    inverse: PyReadonlyArray1<'_, i64>,
    edges: PyReadonlyArray2<'_, i64>,
    flags: Option<PyReadonlyArray1<'_, i64>>,
    curves: Option<PyReadonlyArray1<'_, i64>>,
    layers: Option<PyReadonlyArray1<'_, i64>>,
    rings: PyReadonlyArray1<'_, i64>,
    sizes: PyReadonlyArray1<'_, i64>,
    counts: PyReadonlyArray1<'_, i64>,
) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray1<bool>>, Bound<'py, PyArray1<bool>>,
               Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>,
               Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    let inverse = inverse.as_slice()?;
    let e = edges.as_array();
    let pairs: Vec<[i64; 2]> = (0..e.shape()[0]).map(|i| [e[[i, 0]], e[[i, 1]]]).collect();
    let flags = flags.as_ref().map(|a| a.as_slice()).transpose()?;
    let curves = curves.as_ref().map(|a| a.as_slice()).transpose()?;
    let layers = layers.as_ref().map(|a| a.as_slice()).transpose()?;
    let (rings, sizes, counts) = (rings.as_slice()?, sizes.as_slice()?, counts.as_slice()?);
    let n = inverse.len() as i64;
    if pairs.iter().flatten().chain(rings).any(|&p| p < 0 || p >= n)
        || sizes.iter().chain(counts).any(|&s| s <= 0)
        || sizes.iter().sum::<i64>() != rings.len() as i64
        || counts.iter().sum::<i64>() != sizes.len() as i64
        || [flags, curves, layers].iter().flatten().any(|a| a.len() != pairs.len())
    {
        return Err(pyo3::exceptions::PyValueError::new_err("plan_topology: inconsistent input"));
    }
    let t = py.detach(|| plan::topology(inverse, &pairs, flags, curves, layers, rings, sizes, counts));
    let rows: Vec<Vec<i64>> = t.edges.iter().map(|v| v.to_vec()).collect();
    let arr = if rows.is_empty() { PyArray2::zeros(py, [0, 2], false) } else { PyArray2::from_vec2(py, &rows)? };
    Ok((arr, PyArray1::from_vec(py, t.soft), PyArray1::from_vec(py, t.hidden),
        PyArray1::from_vec(py, t.curve), PyArray1::from_vec(py, t.layer),
        PyArray1::from_vec(py, t.corners), PyArray1::from_vec(py, t.ef_start),
        PyArray1::from_vec(py, t.ef_face)))
}
/// core.bim.face_set_volume's signed sum (before ``abs``) over a lazy
/// mesh's triangles: ``pos`` (V,3) float32, ``tris`` (T,3) uint32 vertex
/// indices in face order (face_geometry's unit triangles — Face.
/// triangulate()'s).
#[pyfunction]
fn signed_volume(py: Python<'_>, pos: PyReadonlyArray2<'_, f32>, tris: PyReadonlyArray2<'_, u32>) -> PyResult<f64> {
    let p = pos.as_array();
    let t = tris.as_array();
    if p.ncols() != 3 || t.ncols() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("signed_volume: (V,3) and (T,3)"));
    }
    let n = p.nrows() as u32;
    let tri: Vec<[u32; 3]> = (0..t.nrows()).map(|i| [t[[i, 0]], t[[i, 1]], t[[i, 2]]]).collect();
    if tri.iter().flatten().any(|&k| k >= n) {
        return Err(pyo3::exceptions::PyValueError::new_err("signed_volume: index out of range"));
    }
    let pts: Vec<[f32; 3]> = (0..p.nrows()).map(|i| [p[[i, 0]], p[[i, 1]], p[[i, 2]]]).collect();
    Ok(py.detach(|| volume::signed_volume(&pts, &tri)))
}
/// core.arrangement._build_graph's split points: ``segs`` (n,2,2) float64
/// 2D segments. Returns ``(start (n+1,), points (m,2))`` — segment i's
/// sorted split points (its endpoints and every hit, in Python's order)
/// are ``points[start[i]:start[i+1]]``.
#[pyfunction]
fn arrangement_splits<'py>(
    py: Python<'py>,
    segs: PyReadonlyArray3<'_, f64>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray2<f64>>)> {
    let s = segs.as_array();
    if s.shape()[1] != 2 || s.shape()[2] != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("arrangement_splits: (n,2,2)"));
    }
    let v: Vec<[[f64; 2]; 2]> = (0..s.shape()[0])
        .map(|i| [[s[[i, 0, 0]], s[[i, 0, 1]]], [s[[i, 1, 0]], s[[i, 1, 1]]]])
        .collect();
    let (start, pts) = py.detach(|| arrange::splits(&v));
    let rows: Vec<Vec<f64>> = pts.iter().map(|p| p.to_vec()).collect();
    let arr = if rows.is_empty() { PyArray2::zeros(py, [0, 2], false) } else { PyArray2::from_vec2(py, &rows)? };
    Ok((PyArray1::from_vec(py, start), arr))
}
/// core.hlr.section_loops: ``chords`` (K,2,3) float64 plane∩triangle
/// chords, ``tol`` the node-merge tolerance as the Python computed it.
/// Returns ``(ring_start (R+1,), points (P,3))``, the closed rings in the
/// Python's order.
#[pyfunction]
fn section_loops<'py>(
    py: Python<'py>,
    chords: PyReadonlyArray3<'_, f64>,
    tol: f64,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray2<f64>>)> {
    let c = chords.as_array();
    if c.shape()[1] != 2 || c.shape()[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("section_loops: (K,2,3)"));
    }
    let v: Vec<[[f64; 3]; 2]> = (0..c.shape()[0])
        .map(|k| [[c[[k, 0, 0]], c[[k, 0, 1]], c[[k, 0, 2]]], [c[[k, 1, 0]], c[[k, 1, 1]], c[[k, 1, 2]]]])
        .collect();
    if !(tol > 0.0) || v.iter().flatten().flatten().any(|x| !x.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err("section_loops: finite input, tol > 0"));
    }
    let (start, pts) = py.detach(|| section::loops(&v, tol));
    let rows: Vec<Vec<f64>> = pts.iter().map(|p| p.to_vec()).collect();
    let arr = if rows.is_empty() { PyArray2::zeros(py, [0, 3], false) } else { PyArray2::from_vec2(py, &rows)? };
    Ok((PyArray1::from_vec(py, start), arr))
}
/// core.intersect's world polygons as one set (Intersect Faces):
/// ``PolySet(pts (P,3) float32, ring_start (R+1,), face_start (F+1,),
/// hypot)`` — every face's world points, outer ring first. ``kept``: the
/// input faces world_polys keeps (in order), ``lo``/``hi`` (F',3) their
/// boxes; ``cross_pairs`` crosses kept faces ``ii`` of this set with
/// ``jj`` of ``other`` (this set when None), in that pair order, and
/// returns the deduped segments (S,2,3) float32.
#[pyclass(frozen)]
struct PolySet {
    polys: Vec<intersect::Poly>,
    kept: Vec<i64>,
    h: facegeom::Hypot,
}

#[pymethods]
impl PolySet {
    #[new]
    fn new(
        py: Python<'_>,
        pts: PyReadonlyArray2<'_, f32>,
        ring_start: PyReadonlyArray1<'_, i64>,
        face_start: PyReadonlyArray1<'_, i64>,
        hypot: &str,
    ) -> PyResult<Self> {
        let h = hypot_of(hypot)?;
        let p = pts.as_array();
        let rs = ring_start.as_slice()?;
        let fs = face_start.as_slice()?;
        let np_ = p.nrows() as i64;
        let ok = p.ncols() == 3
            && rs.first() == Some(&0)
            && fs.first() == Some(&0)
            && rs.windows(2).all(|w| w[0] <= w[1])
            && fs.windows(2).all(|w| w[0] < w[1])
            && *rs.last().unwrap_or(&0) == np_
            && *fs.last().unwrap_or(&0) == rs.len() as i64 - 1;
        if !ok {
            return Err(pyo3::exceptions::PyValueError::new_err("PolySet: inconsistent input"));
        }
        let v: Vec<facegeom::V3> = (0..p.nrows()).map(|i| facegeom::V3(p[[i, 0]], p[[i, 1]], p[[i, 2]])).collect();
        let (polys, kept) = py.detach(|| {
            let mut polys = Vec::new();
            let mut kept = Vec::new();
            for f in 0..fs.len() - 1 {
                let rings: Vec<&[facegeom::V3]> = (fs[f] as usize..fs[f + 1] as usize)
                    .map(|r| &v[rs[r] as usize..rs[r + 1] as usize])
                    .collect();
                if let Some(pl) = intersect::poly(&rings, h) {
                    polys.push(pl);
                    kept.push(f as i64);
                }
            }
            (polys, kept)
        });
        Ok(Self { polys, kept, h })
    }

    #[getter]
    fn kept<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_vec(py, self.kept.clone())
    }

    #[getter]
    fn lo<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.boxes(py, true)
    }

    #[getter]
    fn hi<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.boxes(py, false)
    }

    #[pyo3(signature = (other, ii, jj))]
    fn cross_pairs<'py>(
        &self,
        py: Python<'py>,
        other: Option<PyRef<'_, PolySet>>,
        ii: PyReadonlyArray1<'_, i64>,
        jj: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<Bound<'py, numpy::PyArray3<f32>>> {
        let b: &PolySet = match &other {
            Some(o) => o,
            None => self,
        };
        let (ii, jj) = (ii.as_slice()?, jj.as_slice()?);
        if ii.len() != jj.len()
            || ii.iter().any(|&i| i < 0 || i as usize >= self.polys.len())
            || jj.iter().any(|&j| j < 0 || j as usize >= b.polys.len())
        {
            return Err(pyo3::exceptions::PyValueError::new_err("cross_pairs: bad pairs"));
        }
        let h = self.h;
        let segs = py.detach(|| {
            let mut out = Vec::new();
            for (&i, &j) in ii.iter().zip(jj) {
                intersect::cross(&self.polys[i as usize], &b.polys[j as usize], h, &mut out);
            }
            intersect::dedupe(out)
        });
        let flat: Vec<f32> = segs.iter().flat_map(|(p, q)| [p.0, p.1, p.2, q.0, q.1, q.2]).collect();
        use numpy::PyArrayMethods;
        let arr = numpy::PyArray1::from_vec(py, flat);
        Ok(arr.reshape([segs.len(), 2, 3])?)
    }
}

impl PolySet {
    fn boxes<'py>(&self, py: Python<'py>, low: bool) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let rows: Vec<Vec<f64>> = self
            .polys
            .iter()
            .map(|p| {
                let b = if low { p.lo } else { p.hi };
                vec![b[0] as f64, b[1] as f64, b[2] as f64]
            })
            .collect();
        if rows.is_empty() {
            return Ok(PyArray2::zeros(py, [0, 3], false));
        }
        Ok(PyArray2::from_vec2(py, &rows)?)
    }
}
/// core.solids._conform_t_junctions: ``v`` (V,3) float64 finite
/// vertices, ``t`` (T,3) int64 triangles, ``f`` (T,) int64 face ids,
/// ``tol`` as the Python computed it. Returns ``(vertices (V',3) float64,
/// triangles (T',3) uint32, face_ids (T',) uint32)``.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn conform_t_junctions<'py>(
    py: Python<'py>,
    v: PyReadonlyArray2<'_, f64>,
    t: PyReadonlyArray2<'_, i64>,
    f: PyReadonlyArray1<'_, i64>,
    tol: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<u32>>, Bound<'py, PyArray1<u32>>)> {
    let va = v.as_array();
    let ta = t.as_array();
    let fs = f.as_slice()?;
    let n = va.nrows() as i64;
    if va.ncols() != 3 || ta.ncols() != 3 || fs.len() != ta.nrows() || ta.nrows() == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("conform_t_junctions: shapes"));
    }
    let verts: Vec<[f64; 3]> = (0..va.nrows()).map(|i| [va[[i, 0]], va[[i, 1]], va[[i, 2]]]).collect();
    let tris: Vec<[i64; 3]> = (0..ta.nrows()).map(|i| [ta[[i, 0]], ta[[i, 1]], ta[[i, 2]]]).collect();
    if verts.iter().flatten().any(|x| !x.is_finite())
        || tris.iter().flatten().any(|&k| k < 0 || k >= n)
        || fs.iter().any(|&x| x < 0 || x > u32::MAX as i64)
        || !(tol >= 0.0)
    {
        return Err(pyo3::exceptions::PyValueError::new_err("conform_t_junctions: bad input"));
    }
    let o = py.detach(|| conform::conform(&verts, &tris, fs, tol));
    let vr: Vec<Vec<f64>> = o.verts.iter().map(|p| p.to_vec()).collect();
    let tr: Vec<Vec<u32>> = o.tris.iter().map(|p| p.to_vec()).collect();
    Ok((PyArray2::from_vec2(py, &vr)?, PyArray2::from_vec2(py, &tr)?, PyArray1::from_vec(py, o.fids)))
}
/// core.solids._outer_shell_mask's union-find: ``t`` (T,3) int64 over
/// ``nv`` vertices, merged triangle by triangle as the Python does (path
/// halving, b's root then c's under a's). Returns each triangle's root
/// (of its first corner), (T,) int64.
#[pyfunction]
fn shell_roots<'py>(py: Python<'py>, t: PyReadonlyArray2<'_, i64>, nv: usize) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let ta = t.as_array();
    if ta.ncols() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("shell_roots: (T,3)"));
    }
    let tris: Vec<[usize; 3]> = (0..ta.nrows())
        .map(|i| [ta[[i, 0]], ta[[i, 1]], ta[[i, 2]]])
        .map(|r| r.map(|k| if k >= 0 { k as usize } else { usize::MAX }))
        .collect();
    if tris.iter().flatten().any(|&k| k >= nv) {
        return Err(pyo3::exceptions::PyValueError::new_err("shell_roots: index out of range"));
    }
    let roots = py.detach(|| {
        let mut parent: Vec<usize> = (0..nv).collect();
        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }
        for &[a, b, c] in &tris {
            let ra = find(&mut parent, a);
            let rb = find(&mut parent, b);
            let rc = find(&mut parent, c);
            parent[rb] = ra;
            let r = find(&mut parent, rc);
            parent[r] = ra;
        }
        tris.iter().map(|t| find(&mut parent, t[0]) as i64).collect::<Vec<i64>>()
    });
    Ok(PyArray1::from_vec(py, roots))
}
/// georef.geotiff._lzw_decode: a TIFF LZW strip or tile, decoded.
#[pyfunction]
fn lzw_decode<'py>(py: Python<'py>, data: &[u8]) -> Bound<'py, pyo3::types::PyBytes> {
    let out = py.detach(|| tiff::lzw_decode(data));
    pyo3::types::PyBytes::new(py, &out)
}

/// georef.geotiff._unpackbits: a PackBits strip or tile, decoded.
#[pyfunction]
fn unpackbits<'py>(py: Python<'py>, data: &[u8]) -> Bound<'py, pyo3::types::PyBytes> {
    let out = py.detach(|| tiff::unpackbits(data));
    pyo3::types::PyBytes::new(py, &out)
}
/// formats.fuse.fuse_coplanar_loops after the signatures: ``pts`` (P,3)
/// float32 every loop's points, ``loop_start`` (L+1,), per loop ``sig``
/// (equal ids: equal signatures), ``tex_group`` (equal non-negative ids:
/// same colour, extras and image, a fuzzy-UV candidate), ``rank`` (2
/// textured, 1 else), ``uvw`` (L,W) float64 with ``uvw_len`` (L,) used
/// values each. Returns ``(traced (R,) u8, region_start (R+1,),
/// region_loops, ring_start (R+1,) into ring_bounds, ring_bounds,
/// ring_pts)``: region r's loops in order; when traced, its rings
/// (outer first) as point indices.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn fuse_regions<'py>(
    py: Python<'py>,
    pts: PyReadonlyArray2<'_, f32>,
    loop_start: PyReadonlyArray1<'_, i64>,
    sig: PyReadonlyArray1<'_, i64>,
    tex_group: PyReadonlyArray1<'_, i64>,
    rank: PyReadonlyArray1<'_, u8>,
    uvw: PyReadonlyArray2<'_, f64>,
    uvw_len: PyReadonlyArray1<'_, i64>,
    cos_tol: f64,
    principal: bool,
    hypot: &str,
) -> PyResult<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>,
               Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    let h = hypot_of(hypot)?;
    let pa = pts.as_array();
    let ls = loop_start.as_slice()?;
    let (sig, tg, rk, ul) = (sig.as_slice()?, tex_group.as_slice()?, rank.as_slice()?, uvw_len.as_slice()?);
    let ua = uvw.as_array();
    let nl = ls.len().saturating_sub(1);
    let ok = pa.ncols() == 3
        && ls.first() == Some(&0)
        && ls.windows(2).all(|w| w[0] <= w[1])
        && *ls.last().unwrap_or(&0) == pa.nrows() as i64
        && [sig.len(), tg.len(), rk.len(), ul.len(), ua.nrows()].iter().all(|&x| x == nl)
        && ul.iter().all(|&x| x >= 0 && x as usize <= ua.ncols());
    if !ok {
        return Err(pyo3::exceptions::PyValueError::new_err("fuse_regions: inconsistent input"));
    }
    let v: Vec<facegeom::V3> = (0..pa.nrows()).map(|i| facegeom::V3(pa[[i, 0]], pa[[i, 1]], pa[[i, 2]])).collect();
    let starts: Vec<usize> = ls.iter().map(|&x| x as usize).collect();
    let uv: Vec<Vec<f64>> = (0..nl).map(|i| (0..ul[i] as usize).map(|k| ua[[i, k]]).collect()).collect();
    let regions = py.detach(|| {
        let inp = fuse::Input { pts: &v, loop_start: &starts, sig, tex_group: tg, rank: rk, uvw: &uv };
        fuse::fuse(&inp, cos_tol, principal, h)
    });
    let (mut traced, mut rstart, mut rloops) = (Vec::new(), vec![0i64], Vec::new());
    let (mut ring_start, mut bounds, mut rpts) = (vec![0i64], vec![0i64], Vec::new());
    for r in regions {
        match r {
            fuse::Region::Pass(l) => {
                traced.push(0u8);
                rloops.extend(l.iter().map(|&x| x as i64));
            }
            fuse::Region::Traced(rings, l) => {
                traced.push(1u8);
                rloops.extend(l.iter().map(|&x| x as i64));
                for ring in rings {
                    rpts.extend(ring.iter().map(|&x| x as i64));
                    bounds.push(rpts.len() as i64);
                }
            }
        }
        rstart.push(rloops.len() as i64);
        ring_start.push(bounds.len() as i64 - 1);
    }
    Ok((PyArray1::from_vec(py, traced), PyArray1::from_vec(py, rstart), PyArray1::from_vec(py, rloops),
        PyArray1::from_vec(py, ring_start), PyArray1::from_vec(py, bounds), PyArray1::from_vec(py, rpts)))
}
/// formats.stl._normal per triangle: ``pos`` (P,3) float32, ``tris``
/// (T,3) uint32. Returns (T,3) float32 unit normals (zero below 1e-12).
#[pyfunction]
fn tri_normals<'py>(py: Python<'py>, pos: PyReadonlyArray2<'_, f32>, tris: PyReadonlyArray2<'_, u32>,
                    hypot: &str) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let h = hypot_of(hypot)?;
    let p = pos.as_array();
    let t = tris.as_array();
    if p.ncols() != 3 || t.ncols() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("tri_normals: (P,3) and (T,3)"));
    }
    let n = p.nrows() as u32;
    let tr: Vec<[u32; 3]> = (0..t.nrows()).map(|i| [t[[i, 0]], t[[i, 1]], t[[i, 2]]]).collect();
    if tr.iter().flatten().any(|&k| k >= n) {
        return Err(pyo3::exceptions::PyValueError::new_err("tri_normals: index out of range"));
    }
    let v: Vec<facegeom::V3> = (0..p.nrows()).map(|i| facegeom::V3(p[[i, 0]], p[[i, 1]], p[[i, 2]])).collect();
    let out = py.detach(|| volume::tri_normals(&v, &tr, h));
    let rows: Vec<Vec<f32>> = out.iter().map(|r| r.to_vec()).collect();
    if rows.is_empty() {
        return Ok(PyArray2::zeros(py, [0, 3], false));
    }
    Ok(PyArray2::from_vec2(py, &rows)?)
}
/// A recorded run of OpenGL calls, replayed through the current context's
/// entry points (views.viewport._GlTape): ``GlReplay(addrs)`` with the
/// eleven entry points in opcode order (glDrawArraysInstanced,
/// glBindVertexArray, glUniform1i, glUniform1f, glUniform4f, glEnable,
/// glDisable, glCullFace, glFrontFace, glActiveTexture, glBindTexture).
#[pyclass(frozen)]
struct GlReplay {
    tape: gltape::Tape,
}

#[pymethods]
impl GlReplay {
    #[new]
    fn new(addrs: Vec<usize>) -> PyResult<Self> {
        gltape::Tape::new(&addrs)
            .map(|tape| Self { tape })
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing GL entry point"))
    }

    /// Run ``ops`` (R,5) int64 ``[opcode, a, b, c, d]`` with ``vals`` (R,4)
    /// float32. The GL context the entry points came from must be current.
    fn run(&self, ops: PyReadonlyArray2<'_, i64>, vals: PyReadonlyArray2<'_, f32>) -> PyResult<usize> {
        let o = ops.as_array();
        let v = vals.as_array();
        if o.shape()[1] != 5 || v.shape()[1] != 4 || o.shape()[0] != v.shape()[0] {
            return Err(pyo3::exceptions::PyValueError::new_err("tape shape"));
        }
        let rows: Vec<[i64; 5]> = (0..o.shape()[0])
            .map(|r| [o[[r, 0]], o[[r, 1]], o[[r, 2]], o[[r, 3]], o[[r, 4]]]).collect();
        let fl: Vec<[f32; 4]> = (0..v.shape()[0])
            .map(|r| [v[[r, 0]], v[[r, 1]], v[[r, 2]], v[[r, 3]]]).collect();
        // SAFETY: the viewport calls this inside paintGL, its context
        // current, with a tape it recorded from its own calls.
        unsafe { self.tape.run(&rows, &fl) }
            .map_err(|r| pyo3::exceptions::PyValueError::new_err(format!("bad opcode at {r}")))
    }
}

/// One prototype for the spatial index: its pick triangles (``v0``,
/// ``e1``, ``e2``: (T,3) local, as the chunk keeps them) under a BVH, and
/// its hard edges (``edges``: (E,6) local endpoints).
#[pyclass(frozen)]
struct InstanceProto {
    inner: std::sync::Arc<instances::Proto>,
}

#[pymethods]
impl InstanceProto {
    #[new]
    fn new(
        py: Python<'_>,
        v0: PyReadonlyArray2<'_, f64>,
        e1: PyReadonlyArray2<'_, f64>,
        e2: PyReadonlyArray2<'_, f64>,
        edges: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        let (a, b, c) = (rows3(&v0.as_array()), rows3(&e1.as_array()), rows3(&e2.as_array()));
        if a.len() != b.len() || a.len() != c.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("triangle arrays"));
        }
        let ev = edges.as_array();
        if ev.shape()[0] > 0 && ev.shape()[1] != 6 {
            return Err(pyo3::exceptions::PyValueError::new_err("edges (E,6)"));
        }
        let es: Vec<[[f64; 3]; 2]> = (0..ev.shape()[0]).map(|i| [
            [ev[[i, 0]], ev[[i, 1]], ev[[i, 2]]], [ev[[i, 3]], ev[[i, 4]], ev[[i, 5]]]]).collect();
        let inner = py.detach(|| instances::Proto::new(a, b, c, es));
        Ok(Self { inner: std::sync::Arc::new(inner) })
    }

    fn __len__(&self) -> usize {
        self.inner.n_tris()
    }
}

/// The placements of a scene state over their prototypes: ``protos`` (a
/// list of InstanceProto), ``proto_of`` (P,) which one each placement
/// shows, ``mats`` (P,16) their matrices column-major (QMatrix4x4.data()).
#[pyclass(frozen)]
struct InstanceSet {
    protos: Vec<std::sync::Arc<instances::Proto>>,
    places: Vec<instances::Placement>,
}

#[pymethods]
impl InstanceSet {
    #[new]
    fn new(
        protos: Vec<PyRef<'_, InstanceProto>>,
        proto_of: PyReadonlyArray1<'_, i64>,
        mats: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        let pr: Vec<std::sync::Arc<instances::Proto>> = protos.iter().map(|p| p.inner.clone()).collect();
        let po = proto_of.as_array();
        let m = mats.as_array();
        if m.shape()[0] != po.len() || (po.len() > 0 && m.shape()[1] != 16) {
            return Err(pyo3::exceptions::PyValueError::new_err("placements"));
        }
        let mut places = Vec::with_capacity(po.len());
        for i in 0..po.len() {
            let k = po[i];
            if k < 0 || k as usize >= pr.len() {
                return Err(pyo3::exceptions::PyValueError::new_err("proto index"));
            }
            let mut mm = [0.0f64; 16];
            for j in 0..16 {
                mm[j] = m[[i, j]];
            }
            places.push(instances::Placement { proto: k as usize, m: mm, inv: instances::affine_inverse(&mm) });
        }
        Ok(Self { protos: pr, places })
    }

    fn __len__(&self) -> usize {
        self.places.len()
    }

    /// Of placements ``cand``, those whose triangles the world ray ``o`` +
    /// t·``d`` may meet for t in [0, ``reach``] (None: unbounded) — a
    /// placement whose matrix cannot be inverted is always kept.
    #[pyo3(signature = (cand, o, d, reach=None))]
    fn ray_filter(&self, py: Python<'_>, cand: PyReadonlyArray1<'_, i64>,
                  o: (f64, f64, f64), d: (f64, f64, f64), reach: Option<f64>) -> PyResult<Vec<i64>> {
        use rayon::prelude::*;
        let c = cand.as_array().to_vec();
        self.check(&c)?;
        let (o, d) = ([o.0, o.1, o.2], [d.0, d.1, d.2]);
        let lim = reach.unwrap_or(f64::INFINITY);
        Ok(py.detach(|| c.par_iter().copied().filter(|&i| {
            let p = &self.places[i as usize];
            match p.local_ray(o, d) {
                None => true,
                Some((lo, ld)) => self.protos[p.proto].ray_may_hit(lo, ld, lim),
            }
        }).collect()))
    }

    /// Of placements ``cand``, those with something near ``(px, py)`` on a
    /// ``w``×``h`` view through ``mvp`` (16 floats, column-major): a hard
    /// edge within ``edge_reach`` px, a triangle within ``tri_reach``.
    #[allow(clippy::too_many_arguments)]
    fn screen_filter(&self, py: Python<'_>, cand: PyReadonlyArray1<'_, i64>, mvp: Vec<f64>,
                     w: f64, h: f64, px: f64, py_: f64, edge_reach: f64,
                     tri_reach: f64) -> PyResult<Vec<i64>> {
        use rayon::prelude::*;
        if mvp.len() != 16 {
            return Err(pyo3::exceptions::PyValueError::new_err("mvp"));
        }
        let c = cand.as_array().to_vec();
        self.check(&c)?;
        let mut a = [0.0f64; 16];
        a.copy_from_slice(&mvp);
        Ok(py.detach(|| c.par_iter().copied().filter(|&i| {
            let p = &self.places[i as usize];
            let m = instances::mat_mul(&a, &p.m);
            self.protos[p.proto].screen_near(&m, w, h, px, py_, edge_reach, tri_reach)
        }).collect()))
    }
}

impl InstanceSet {
    fn check(&self, c: &[i64]) -> PyResult<()> {
        if c.iter().any(|&i| i < 0 || i as usize >= self.places.len()) {
            return Err(pyo3::exceptions::PyValueError::new_err("placement index"));
        }
        Ok(())
    }
}

/// ``Viewport._occluded_many``'s block for one span: rays from ``origin``
/// along the rows of ``dirs`` (R,3), against the span's triangles ``v0``,
/// ``e1``, ``e2`` (N,3). Returns ``(best, doubt)``: per ray, the minimum of
/// ``best`` (R,) and every accepted hit, and whether that answer could hang
/// on the last bit (then the caller asks NumPy), with ``limit`` (R,) the
/// distance of the point the ray goes to.
#[pyfunction]
fn occlusion_span<'py>(
    py: Python<'py>,
    origin: (f64, f64, f64),
    dirs: PyReadonlyArray2<'py, f64>,
    best: PyReadonlyArray1<'py, f64>,
    limit: PyReadonlyArray1<'py, f64>,
    v0: PyReadonlyArray2<'py, f64>,
    e1: PyReadonlyArray2<'py, f64>,
    e2: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<bool>>)> {
    let o = [origin.0, origin.1, origin.2];
    let d = rows3(&dirs.as_array());
    let b = best.as_array().to_vec();
    let l = limit.as_array().to_vec();
    let (a, x, y) = (rows3(&v0.as_array()), rows3(&e1.as_array()), rows3(&e2.as_array()));
    let (nb, doubt) = py.detach(|| occlusion::span(o, &d, &b, &l, &a, &x, &y));
    Ok((PyArray1::from_vec(py, nb), PyArray1::from_vec(py, doubt)))
}
