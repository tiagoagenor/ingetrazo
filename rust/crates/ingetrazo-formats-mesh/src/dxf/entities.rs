// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The entities `formats/dxf_in.py` translates, typed, and their conversion
//! to point runs and face loops: `_entity_segments`, `_entity_face` and the
//! block-reference explosion (`Insert.virtual_entities`, with ezdxf's
//! `transform` of each entity kind).

use super::geom::*;
use super::read::{Block, Doc, RawEntity};

/// A SPLINE's data (`Spline.construction_tool()` inputs).
#[derive(Clone, Debug, Default)]
pub struct SplineData {
    pub degree: i64,
    pub flags: i64,
    pub knots: Vec<f64>,
    pub weights: Vec<f64>,
    pub control_points: Vec<P>,
    pub fit_points: Vec<P>,
    pub knot_tolerance: f64,
    pub start_tangent: Option<P>,
    pub end_tangent: Option<P>,
}

/// An INSERT's coordinate system (`InsertCoordinateSystem`): the point is in
/// the OCS of `extrusion`.
#[derive(Clone, Copy, Debug)]
pub struct InsertParams {
    pub insert: P,
    pub sx: f64,
    pub sy: f64,
    pub sz: f64,
    /// Degrees.
    pub rotation: f64,
    pub extrusion: P,
}

impl InsertParams {
    /// `Insert.matrix44()` (block base point included when the block exists).
    pub fn matrix(&self, base: Option<P>) -> R<Mat> {
        let ocs = Ocs::new(self.extrusion)?;
        let extrusion = ocs.uz();
        let ux = ocs.to_wcs(X_AXIS);
        let uy = ocs.to_wcs(Y_AXIS);
        let mut m = Mat::ucs(ux.mul(self.sx), uy.mul(self.sy), extrusion.mul(self.sz));
        let angle = radians(self.rotation);
        if angle != 0.0 {
            m = m.mul(&Mat::axis_rotate(extrusion, angle)?);
        }
        let mut insert = ocs.to_wcs(self.insert);
        if let Some(b) = base {
            insert = insert.sub(m.direction(b));
        }
        m.m[12] = insert.x;
        m.m[13] = insert.y;
        m.m[14] = insert.z;
        Ok(m)
    }

    /// `InsertCoordinateSystem.transform(m)`: `Ok(None)` where ezdxf raises
    /// `InsertTransformationError` (a non-orthogonal target system), `Err`
    /// for any other failure.
    pub fn transform(&self, m: &Mat) -> R<Option<InsertParams>> {
        const TOL: f64 = 1e-9;
        let ocs = Ocs::new(self.extrusion)?;
        let ux = m.direction(ocs.ux());
        let uy = m.direction(ocs.uy());
        let uz = m.direction(ocs.uz());
        let x_scale = ux.magnitude() * self.sx;
        let mut y_scale = uy.magnitude() * self.sy;
        let z_scale = uz.magnitude() * self.sz;
        let ux = ux.normalize(1.0)?;
        let uy = uy.normalize(1.0)?;
        let uz = uz.normalize(1.0)?;
        let dot = |a: P, b: P| a.x * b.x + a.y * b.y + a.z * b.z;
        if dot(ux, uz).abs() > TOL || dot(ux, uy).abs() > TOL || dot(uz, uy).abs() > TOL {
            return Ok(None);
        }
        let expected_uy = uz.cross(ux);
        if !expected_uy.isclose(uy, REL_TOL, TOL) {
            y_scale = -y_scale;
        }
        let t = OcsTransform::from_ocs(Ocs::new(self.extrusion)?, Ocs::new(uz)?, m);
        Ok(Some(InsertParams {
            insert: t.transform_vertex(self.insert),
            sx: x_scale,
            sy: y_scale,
            sz: z_scale,
            rotation: t.transform_deg_angle(self.rotation),
            extrusion: uz,
        }))
    }
}

#[derive(Clone, Debug)]
pub enum Geom {
    Line(P, P),
    Face3d([P; 4]),
    /// SOLID / TRACE: the corners as stored (OCS), z-ordered; `has` says
    /// which of them the file gave.
    Solid { v: [P; 4], has: [bool; 4], extrusion: P },
    LwPoly { pts: Vec<(f64, f64, f64)>, closed: bool, elevation: f64, extrusion: P, curved: bool },
    /// A 2D POLYLINE: `zs` are the vertices' own z, `elevation` the header's
    /// when it has one.
    Poly2d { pts: Vec<(f64, f64, f64)>, zs: Vec<f64>, closed: bool, elevation: Option<f64>, extrusion: P, curved: bool, has_arc: bool },
    Poly3d { pts: Vec<P>, closed: bool, curved: bool },
    /// A polygon or polyface mesh: `make_path` refuses it.
    PolyMesh,
    Circle { center: P, radius: f64, extrusion: P },
    Arc { center: P, radius: f64, extrusion: P, start: f64, end: f64 },
    Ellipse { center: P, major: P, extrusion: P, ratio: f64, start: f64, end: f64 },
    Spline(Box<SplineData>),
    /// A block reference.
    Insert { name: String, p: InsertParams },
    /// Entities `_SKIP_SILENT` drops without counting.
    Silent,
    /// An INSERT whose matrix failed: the import loses only itself.
    Dropped,
    /// Anything else: counted in `skipped` under its DXF name.
    Other,
}

#[derive(Clone, Debug)]
pub struct Ent {
    pub kind: String,
    pub layer: String,
    pub geom: Geom,
}

fn extrusion_of(r: &RawEntity) -> P {
    r.pt(210).unwrap_or(Z_AXIS)
}

fn lw_points(r: &RawEntity) -> Vec<(f64, f64, f64)> {
    let mut pts: Vec<(f64, f64, f64)> = Vec::new();
    for t in &r.tags {
        match t.code {
            10 => pts.push((crate::pyutil::float(&t.value).unwrap_or(0.0), 0.0, 0.0)),
            20 => {
                if let Some(l) = pts.last_mut() {
                    l.1 = crate::pyutil::float(&t.value).unwrap_or(0.0);
                }
            }
            42 => {
                if let Some(l) = pts.last_mut() {
                    l.2 = crate::pyutil::float(&t.value).unwrap_or(0.0);
                }
            }
            _ => {}
        }
    }
    pts
}

fn spline_data(r: &RawEntity) -> SplineData {
    let mut s = SplineData { degree: r.i_or(71, 3), flags: r.i_or(70, 0), knot_tolerance: r.f_or(42, 1e-10), ..Default::default() };
    let num = |v: &str| crate::pyutil::float(v).unwrap_or(0.0);
    let mut cps: Vec<P> = Vec::new();
    let mut fits: Vec<P> = Vec::new();
    for t in &r.tags {
        match t.code {
            40 => s.knots.push(num(&t.value)),
            41 => s.weights.push(num(&t.value)),
            10 => cps.push(P::new(num(&t.value), 0.0, 0.0)),
            20 => {
                if let Some(l) = cps.last_mut() {
                    l.y = num(&t.value);
                }
            }
            30 => {
                if let Some(l) = cps.last_mut() {
                    l.z = num(&t.value);
                }
            }
            11 => fits.push(P::new(num(&t.value), 0.0, 0.0)),
            21 => {
                if let Some(l) = fits.last_mut() {
                    l.y = num(&t.value);
                }
            }
            31 => {
                if let Some(l) = fits.last_mut() {
                    l.z = num(&t.value);
                }
            }
            _ => {}
        }
    }
    s.control_points = cps;
    s.fit_points = fits;
    s.start_tangent = r.pt(12);
    s.end_tangent = r.pt(13);
    s
}

impl Ent {
    /// The typed entity for a raw record; `doc` resolves block references.
    pub fn parse(r: &RawEntity, doc: &Doc) -> Ent {
        let layer = r.layer();
        // ezdxf's `processor.r12`: an R12 file, or a record with no subclass
        // markers; such records keep the legacy elevation (group 38) apart
        // from their z values.
        let r12 = doc.r12() || !r.has(100);
        let with_elevation = |p: P| -> P {
            match r.f(38) {
                Some(e) if r12 && e != 0.0 && p.z == 0.0 => P::new(p.x, p.y, e),
                _ => p,
            }
        };
        let geom = match r.kind.as_str() {
            "LINE" => Geom::Line(r.pt_or_null(10), r.pt_or_null(11)),
            "3DFACE" => Geom::Face3d([r.pt_or_null(10), r.pt_or_null(11), r.pt_or_null(12), r.pt_or_null(13)]),
            "SOLID" | "TRACE" => {
                let mut v = [r.pt(10), r.pt(11), r.pt(12), r.pt(13)];
                let has = [v[0].is_some(), v[1].is_some(), v[2].is_some(), v[3].is_some()];
                // R12 stores a legacy elevation instead of z.
                if r12 {
                    if let Some(elevation) = r.f(38) {
                        if elevation != 0.0 {
                            for p in v.iter_mut().flatten() {
                                if p.z == 0.0 {
                                    p.z = elevation;
                                }
                            }
                        }
                    }
                }
                Geom::Solid {
                    v: [v[0].unwrap_or(NULLVEC), v[1].unwrap_or(NULLVEC), v[2].unwrap_or(NULLVEC), v[3].unwrap_or(NULLVEC)],
                    has,
                    extrusion: extrusion_of(r),
                }
            }
            "LWPOLYLINE" => {
                let pts = lw_points(r);
                let curved = pts.iter().any(|p| p.2 != 0.0);
                Geom::LwPoly { pts, closed: r.i_or(70, 0) & 1 != 0, elevation: r.f_or(38, 0.0), extrusion: extrusion_of(r), curved }
            }
            "POLYLINE" => {
                let flags = r.i_or(70, 0);
                let bulge = |v: &RawEntity| v.f_or(42, 0.0);
                let curved = r.subs.iter().any(|v| bulge(v).abs() > 1e-12);
                if flags & (16 | 64) != 0 {
                    Geom::PolyMesh
                } else if flags & 8 != 0 {
                    Geom::Poly3d { pts: r.subs.iter().map(|v| v.pt_or_null(10)).collect(), closed: flags & 1 != 0, curved }
                } else {
                    let pts = r.subs.iter().map(|v| (v.pt_or_null(10).x, v.pt_or_null(10).y, bulge(v))).collect();
                    let zs = r.subs.iter().map(|v| v.pt_or_null(10).z).collect();
                    let has_arc = r.subs.iter().any(|v| v.has(42) && bulge(v) != 0.0);
                    Geom::Poly2d {
                        pts,
                        zs,
                        closed: flags & 1 != 0,
                        elevation: r.pt(10).map(|e| e.z),
                        extrusion: extrusion_of(r),
                        curved,
                        has_arc,
                    }
                }
            }
            "CIRCLE" => Geom::Circle { center: r.pt(10).map(with_elevation).unwrap_or(NULLVEC), radius: r.f_or(40, 1.0), extrusion: extrusion_of(r) },
            "ARC" => Geom::Arc {
                center: r.pt(10).map(with_elevation).unwrap_or(NULLVEC),
                radius: r.f_or(40, 1.0),
                extrusion: extrusion_of(r),
                start: r.f_or(50, 0.0),
                end: r.f_or(51, 360.0),
            },
            "ELLIPSE" => Geom::Ellipse {
                center: r.pt_or_null(10),
                major: r.pt(11).unwrap_or(X_AXIS),
                extrusion: extrusion_of(r),
                ratio: r.f_or(40, 1.0),
                start: r.f_or(41, 0.0),
                end: r.f_or(42, std::f64::consts::TAU),
            },
            "SPLINE" => Geom::Spline(Box::new(spline_data(r))),
            "INSERT" => Geom::Insert {
                name: r.tag(2).unwrap_or("").to_string(),
                p: InsertParams {
                    insert: r.pt(10).map(with_elevation).unwrap_or(NULLVEC),
                    sx: r.f_or(41, 1.0),
                    sy: r.f_or(42, 1.0),
                    sz: r.f_or(43, 1.0),
                    rotation: r.f_or(50, 0.0),
                    extrusion: extrusion_of(r),
                },
            },
            "ATTDEF" | "VIEWPORT" | "XLINE" | "RAY" => Geom::Silent,
            _ => Geom::Other,
        };
        Ent { kind: r.kind.clone(), layer, geom }
    }
}

/// `Insert.matrix44()` of a reference entity (the matrix of its contents).
pub fn contents_matrix(doc: &Doc, g: &Geom) -> R<(String, Mat)> {
    match g {
        Geom::Insert { name, p } => Ok((name.clone(), p.matrix(doc.block(name).map(|b| b.base_point))?)),
        _ => Err(Raised),
    }
}

/// `Path` for a linework entity (`ezdxf.path.make_path`).
pub fn make_path(g: &Geom) -> R<Path> {
    match g {
        Geom::LwPoly { pts, closed, elevation, extrusion, .. } => {
            let mut path = Path::new(NULLVEC);
            add_2d_polyline(&mut path, pts, *closed, &Ocs::new(*extrusion)?, *elevation)?;
            Ok(path)
        }
        Geom::Poly2d { pts, zs, closed, elevation, extrusion, .. } => {
            let mut path = Path::new(NULLVEC);
            if pts.is_empty() {
                return Ok(path);
            }
            let elevation = elevation.unwrap_or(zs[0]);
            add_2d_polyline(&mut path, pts, *closed, &Ocs::new(*extrusion)?, elevation)?;
            Ok(path)
        }
        Geom::Poly3d { pts, closed, .. } => {
            if pts.is_empty() {
                return Ok(Path::new(NULLVEC));
            }
            Ok(from_vertices(pts, *closed))
        }
        Geom::PolyMesh => Err(Raised),
        Geom::Circle { center, radius, extrusion } => {
            let mut path = Path::new(NULLVEC);
            let radius = radius.abs();
            if radius > 1e-12 {
                let e = Ellipse::from_arc(*center, radius, *extrusion, 0.0, 360.0)?;
                add_ellipse(&mut path, &e, 1)?;
            }
            Ok(path)
        }
        Geom::Arc { center, radius, extrusion, start, end } => {
            let mut path = Path::new(NULLVEC);
            let radius = radius.abs();
            if radius > 1e-12 {
                let e = Ellipse::from_arc(*center, radius, *extrusion, *start, *end)?;
                add_ellipse(&mut path, &e, 1)?;
            }
            Ok(path)
        }
        Geom::Ellipse { center, major, extrusion, ratio, start, end } => {
            let mut path = Path::new(NULLVEC);
            let e = Ellipse::new(*center, *major, *extrusion, *ratio, *start, *end)?;
            add_ellipse(&mut path, &e, 1)?;
            Ok(path)
        }
        Geom::Spline(s) => super::spline::make_path(s),
        _ => Err(Raised),
    }
}

/// `_entity_segments(e, sagitta)`: `None` for kinds this does not
/// translate, else the point runs and whether they form a curve.
pub fn entity_segments(g: &Geom, sagitta: f64) -> Option<(Vec<Vec<P>>, bool)> {
    match g {
        Geom::Line(a, b) => Some((vec![vec![*a, *b]], false)),
        Geom::LwPoly { curved, .. } | Geom::Poly2d { curved, .. } | Geom::Poly3d { curved, .. } => linework(g, sagitta, *curved),
        Geom::PolyMesh => Some((vec![], false)),
        Geom::Circle { .. } | Geom::Arc { .. } | Geom::Ellipse { .. } | Geom::Spline(_) => linework(g, sagitta, true),
        _ => None,
    }
}

fn linework(g: &Geom, sagitta: f64, curved: bool) -> Option<(Vec<Vec<P>>, bool)> {
    let pts = match make_path(g).and_then(|p| p.flattening(sagitta, 8)) {
        Ok(p) => p,
        Err(Raised) => return Some((vec![], false)),
    };
    if pts.len() < 2 {
        return Some((vec![], false));
    }
    Some((vec![pts], curved))
}

/// `_entity_face(e)`: the corner loop of a face entity.
pub fn entity_face(g: &Geom) -> Option<Vec<P>> {
    let pts: Vec<P> = match g {
        Geom::Face3d(v) => {
            let mut pts = vec![v[0], v[1], v[2]];
            if v[3].sub(v[2]).magnitude() > 1e-12 {
                pts.push(v[3]);
            }
            pts
        }
        Geom::Solid { v, .. } => {
            let mut pts = vec![v[0], v[1], v[3], v[2]];
            if pts[2].sub(pts[3]).magnitude() < 1e-12 {
                pts.truncate(3);
            }
            pts
        }
        _ => return None,
    };
    let mut seen: Vec<P> = Vec::new();
    for p in pts {
        if seen.is_empty() || p.sub(*seen.last().unwrap()).magnitude() > 1e-12 {
            seen.push(p);
        }
    }
    if seen.len() >= 3 {
        Some(seen)
    } else {
        None
    }
}

/// ezdxf's `ABS_TOL`.
const ABS_TOL: f64 = 1e-12;

/// `entity.transform(m)`. `Ok(None)` is where ezdxf raises one of the
/// errors `virtual_block_reference_entities` handles (non-uniform scaling of
/// a curve, a non-orthogonal block reference); `Err` is any other failure,
/// which makes the whole `virtual_entities()` fail.
pub fn transform_geom(g: &Geom, m: &Mat) -> R<Option<Geom>> {
    let moved = |p: &P| m.transform(*p);
    Ok(Some(match g {
        Geom::Line(a, b) => Geom::Line(moved(a), moved(b)),
        Geom::Face3d(v) => Geom::Face3d([moved(&v[0]), moved(&v[1]), moved(&v[2]), moved(&v[3])]),
        Geom::Poly3d { pts, closed, curved } => Geom::Poly3d { pts: pts.iter().map(moved).collect(), closed: *closed, curved: *curved },
        Geom::Solid { v, has, extrusion } => {
            let t = OcsTransform::new(*extrusion, m)?;
            let mut nv = *v;
            for i in 0..4 {
                if has[i] {
                    nv[i] = t.transform_vertex(v[i]);
                }
            }
            Geom::Solid { v: nv, has: *has, extrusion: t.new_extrusion() }
        }
        Geom::Circle { center, radius, extrusion } => {
            let t = OcsTransform::new(*extrusion, m)?;
            if !t.uniform {
                return Ok(None);
            }
            Geom::Circle { center: t.transform_vertex(*center), radius: t.transform_length(P::new(*radius, 0.0, 0.0)), extrusion: t.new_extrusion() }
        }
        Geom::Arc { center, radius, extrusion, start, end } => {
            let t = OcsTransform::new(*extrusion, m)?;
            if !t.uniform {
                return Ok(None);
            }
            let (s, e) = if !isclose(arc_angle_span_deg(*start, *end), 360.0, 1e-9, 0.0) { t.transform_ccw_arc_angles_deg(*start, *end) } else { (*start, *end) };
            Geom::Arc {
                center: t.transform_vertex(*center),
                radius: t.transform_length(P::new(*radius, 0.0, 0.0)),
                extrusion: t.new_extrusion(),
                start: s,
                end: e,
            }
        }
        Geom::LwPoly { pts, closed, elevation, extrusion, curved } => {
            let t = OcsTransform::new(*extrusion, m)?;
            if !t.uniform && *curved {
                return Ok(None);
            }
            let verts: Vec<P> = pts.iter().map(|p| t.transform_vertex(P::new(p.0, p.1, *elevation))).collect();
            let new_pts: Vec<(f64, f64, f64)> = verts.iter().zip(pts).map(|(v, p)| (v.x, v.y, p.2)).collect();
            let elevation = verts.first().map(|v| v.z).unwrap_or(*elevation);
            Geom::LwPoly { pts: new_pts, closed: *closed, elevation, extrusion: t.new_extrusion(), curved: *curved }
        }
        Geom::Poly2d { pts, zs, closed, elevation, extrusion, curved, has_arc } => {
            let t = OcsTransform::new(*extrusion, m)?;
            if !t.uniform && *has_arc {
                return Ok(None);
            }
            let verts: Vec<P> = pts
                .iter()
                .zip(zs)
                .map(|(p, z)| t.transform_vertex(P::new(p.0, p.1, elevation.unwrap_or(*z))))
                .collect();
            let new_pts: Vec<(f64, f64, f64)> = verts.iter().zip(pts).map(|(v, p)| (v.x, v.y, p.2)).collect();
            let new_elev = if verts.is_empty() { *elevation } else { Some(verts[0].z) };
            Geom::Poly2d {
                pts: new_pts,
                zs: verts.iter().map(|v| v.z).collect(),
                closed: *closed,
                elevation: new_elev,
                extrusion: t.new_extrusion(),
                curved: *curved,
                has_arc: *has_arc,
            }
        }
        Geom::Ellipse { center, major, extrusion, ratio, start, end } => {
            let moved = Ellipse::new(*center, *major, *extrusion, *ratio, *start, *end)?.transform(m)?;
            Geom::Ellipse {
                center: moved.center,
                major: moved.major_axis,
                extrusion: moved.extrusion,
                ratio: moved.ratio.max(1e-6),
                start: moved.start_param,
                end: moved.end_param,
            }
        }
        Geom::Spline(s) => {
            let mut n = (**s).clone();
            n.control_points = s.control_points.iter().map(moved).collect();
            n.fit_points = s.fit_points.iter().map(moved).collect();
            n.start_tangent = s.start_tangent.map(|t| m.direction(t));
            n.end_tangent = s.end_tangent.map(|t| m.direction(t));
            Geom::Spline(Box::new(n))
        }
        Geom::Insert { name, p } => match p.transform(m)? {
            Some(np) => Geom::Insert { name: name.clone(), p: np },
            None => return Ok(None),
        },
        Geom::Silent | Geom::Dropped | Geom::Other | Geom::PolyMesh => g.clone(),
    }))
}

/// `LWPolyline.virtual_entities()` / `Polyline.virtual_entities()` for a 2D
/// polyline: LINE and ARC primitives in the entity's frame.
fn virtual_polyline_geoms(g: &Geom) -> Vec<Geom> {
    let (mut points, elevation, extrusion, closed): (Vec<(f64, f64, f64)>, f64, P, bool) = match g {
        Geom::LwPoly { pts, closed, elevation, extrusion, .. } => (pts.clone(), *elevation, *extrusion, *closed),
        Geom::Poly2d { pts, zs, closed, elevation, extrusion, .. } => (pts.clone(), elevation.unwrap_or(zs.first().copied().unwrap_or(0.0)), *extrusion, *closed),
        _ => return Vec::new(),
    };
    if points.len() < 2 {
        return Vec::new();
    }
    if closed {
        points.push(points[0]);
    }
    let ocs = match Ocs::new(extrusion) {
        Ok(o) => o,
        Err(Raised) => return Vec::new(),
    };
    let mut out = Vec::new();
    let mut prev: Option<P> = None;
    let mut prev_bulge = 0.0;
    for (x, y, bulge) in points {
        let point = P::new(x, y, elevation);
        let Some(pp) = prev else {
            prev = Some(point);
            prev_bulge = bulge;
            continue;
        };
        if prev_bulge != 0.0 {
            let (center, start_angle, end_angle, radius) = bulge_to_arc(pp, point, prev_bulge);
            if radius > 0.0 {
                out.push(Geom::Arc {
                    center: P::new(center.0, center.1, elevation),
                    radius,
                    extrusion,
                    start: degrees(start_angle),
                    end: degrees(end_angle),
                });
            }
        } else {
            out.push(Geom::Line(ocs.to_wcs(pp), ocs.to_wcs(point)));
        }
        prev = Some(point);
        prev_bulge = bulge;
    }
    out
}

/// One step of `virtual_block_reference_entities.transform`: the entity
/// moved by `m`, or what ezdxf replaces it with when it cannot.
fn transform_into(doc: &Doc, e: Ent, m: &Mat, out: &mut Vec<Ent>) -> R<()> {
    if let Some(g) = transform_geom(&e.geom, m)? {
        out.push(Ent { kind: e.kind, layer: e.layer, geom: g });
        return Ok(());
    }
    match &e.geom {
        Geom::Circle { center, radius, extrusion } | Geom::Arc { center, radius, extrusion, .. } => {
            if radius.abs() > ABS_TOL {
                let (start, end) = match &e.geom {
                    Geom::Arc { start, end, .. } => (*start, *end),
                    _ => (0.0, 360.0),
                };
                let el = Ellipse::from_arc(*center, *radius, *extrusion, start, end)?;
                let el = Ellipse::new(el.center, el.major_axis, el.extrusion, el.ratio, el.start_param, el.end_param)?.transform(m)?;
                out.push(Ent {
                    kind: "ELLIPSE".to_string(),
                    layer: e.layer,
                    geom: Geom::Ellipse {
                        center: el.center,
                        major: el.major_axis,
                        extrusion: el.extrusion,
                        ratio: el.ratio.max(1e-6),
                        start: el.start_param,
                        end: el.end_param,
                    },
                });
            }
        }
        Geom::LwPoly { .. } | Geom::Poly2d { .. } => {
            for v in virtual_polyline_geoms(&e.geom) {
                let kind = if matches!(v, Geom::Line(..)) { "LINE" } else { "ARC" };
                transform_into(doc, Ent { kind: kind.to_string(), layer: e.layer.clone(), geom: v }, m, out)?;
            }
        }
        Geom::Insert { name, p } => {
            // Not a coordinate system: its contents move twice.
            let block = doc.block(name).ok_or(Raised)?;
            let inner = p.matrix(Some(block.base_point))?;
            for child in explode(doc, block, &inner)? {
                transform_into(doc, child, m, out)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// `Insert.virtual_entities()`: the block's entities moved by `m`.
/// `Err` where ezdxf raises (the caller drops the reference).
pub fn explode(doc: &Doc, block: &Block, m: &Mat) -> R<Vec<Ent>> {
    let mut out = Vec::new();
    for raw in &block.entities {
        if raw.kind == "ATTDEF" {
            continue;
        }
        transform_into(doc, Ent::parse(raw, doc), m, &mut out)?;
    }
    Ok(out)
}
