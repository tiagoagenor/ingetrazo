// IngeTrazo ID-buffer shaders (M4.2): what entity covers each pixel.
//
// Three integer layers, each a pair of u32 per pixel (x = placement + 1,
// y = entity id + 1, 0 = nothing):
//  * faces: triangles with both sides pickable, the section cut clipping
//    the fragments; coplanar faces are told apart by a depth nudge toward
//    the eye that grows as the face gets smaller (the CPU picker takes the
//    smallest of the faces at the same depth);
//  * edges: screen-space capsules of `tol` pixels around each segment, the
//    nearest to the pixel winning (the distance is the depth), and hidden
//    behind a face when the unbiased occluder depth says so;
//  * vertices: discs of `tol` pixels around each edge end.
//
// Pixel coordinates are the viewport's: origin top-left, y down, the same as
// @builtin(position) in the fragment stage.

struct PickParams {
    mvp: mat4x4<f32>,
    clip: vec4<f32>,   // world plane n.p + d >= 0 is kept
    eye: vec4<f32>,    // xyz eye, w = 1 perspective / 0 parallel
    fwd: vec4<f32>,    // xyz unit view direction
    zrow: vec4<f32>,   // view-to-clip depth terms: P22, P23, P32, P33
    a: vec4<f32>,      // x clip_enable, y viewport_w, z viewport_h, w coplanar bias (relative)
    b: vec4<f32>,      // x edge tol px, y vertex tol px, z allow soft edges, w occlude edges
};

@group(0) @binding(0) var<uniform> P: PickParams;
@group(0) @binding(1) var occluder: texture_depth_2d;

struct Edge {
    a: vec3<f32>,
    v0: u32,
    b: vec3<f32>,
    v1: u32,
    id: u32,
    flags: u32,
    p0: u32,
    p1: u32,
};

struct Inst {
    m: mat4x4<f32>,
    info: vec4<u32>,   // x placement + 1, y flags
};

struct DrawInfo {
    n_edges: u32,
    edge_base: u32,
    inst_first: u32,
    pad: u32,
};

@group(1) @binding(0) var<storage, read> edges: array<Edge>;
@group(1) @binding(1) var<storage, read> insts: array<Inst>;
@group(1) @binding(2) var<uniform> D: DrawInfo;

// Instance flags.
const IF_LOOSE: u32 = 1u;   // the loose part (or the open group's own mesh)
const IF_SEL: u32 = 2u;     // a placement the cursor can touch
const IF_VIS: u32 = 4u;     // a placement that is seen (occludes)
// Face flags.
const FF_LOOSE_SEL: u32 = 1u;
const FF_LOOSE_VIS: u32 = 2u;
const FF_PLACED: u32 = 4u;
// Edge flags.
const EF_LOOSE_SEL: u32 = 1u;
const EF_SOFT: u32 = 2u;
const EF_HARD: u32 = 4u;

fn clipped(world: vec3<f32>) -> bool {
    return P.a.x > 0.5 && dot(vec4<f32>(world, 1.0), P.clip) < -1e-6;
}

// Distance from the eye along the view, and the direction toward it.
fn eye_depth(world: vec3<f32>) -> f32 {
    if (P.eye.w > 0.5) {
        return length(world - P.eye.xyz);
    }
    return dot(world - P.eye.xyz, P.fwd.xyz);
}

fn toward_eye(world: vec3<f32>) -> vec3<f32> {
    if (P.eye.w > 0.5) {
        return normalize(P.eye.xyz - world);
    }
    return -P.fwd.xyz;
}

// View depth (positive) of a clip-space z/w in wgpu's 0..1 range.
fn linear_depth(z: f32) -> f32 {
    let zv = (P.zrow.y - z * P.zrow.w) / (z * P.zrow.z - P.zrow.x);
    return -zv;
}

fn to_px(c: vec4<f32>) -> vec2<f32> {
    let n = c.xy / c.w;
    return vec2<f32>((n.x * 0.5 + 0.5) * P.a.y, (0.5 - n.y * 0.5) * P.a.z);
}

fn px_to_ndc(p: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(p.x / P.a.y * 2.0 - 1.0, 1.0 - p.y / P.a.z * 2.0, 0.5, 1.0);
}

// ---- faces ------------------------------------------------------------------

struct FaceIn {
    @location(0) pos: vec3<f32>,
    @location(1) face: u32,
    @location(2) flags: u32,
    @location(3) area: f32,
    @location(4) m0: vec4<f32>,
    @location(5) m1: vec4<f32>,
    @location(6) m2: vec4<f32>,
    @location(7) m3: vec4<f32>,
    @location(8) info: vec4<u32>,
};

struct FaceOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) ids: vec2<u32>,
    @location(1) world: vec3<f32>,
    @location(2) @interpolate(flat) ok: u32,
};

// mode 0: smaller faces come nearer (log2 of the world area, 16 octaves
// wide); 1: true depth; 2: the placements only, true depth; 3: the loose part
// only, pushed back by the whole bias, so that it only shows where it is
// nearer than the nearest placement by more than that (the CPU's
// `loose_t < t - eps` of `pick_group`).
fn face_vertex(i: FaceIn, mode: u32) -> FaceOut {
    var o: FaceOut;
    let m = mat4x4<f32>(i.m0, i.m1, i.m2, i.m3);
    let world = (m * vec4<f32>(i.pos, 1.0)).xyz;
    var w = world;
    if (mode == 0u) {
        let det = dot(m[0].xyz, cross(m[1].xyz, m[2].xyz));
        let aw = max(i.area * pow(abs(det), 0.6666667), 1e-12);
        let f = clamp(0.5 - 0.0625 * log2(aw), 0.0, 1.0);
        w = world + toward_eye(world) * (eye_depth(world) * P.a.w * f);
    } else if (mode == 3u) {
        w = world - toward_eye(world) * (eye_depth(world) * P.a.w);
    }
    o.pos = P.mvp * vec4<f32>(w, 1.0);
    o.world = world;
    o.ids = vec2<u32>(i.info.x, i.face + 1u);
    let loose = (i.info.y & IF_LOOSE) != 0u;
    var sel = false;
    var vis = false;
    if (loose) {
        sel = (i.flags & FF_LOOSE_SEL) != 0u;
        vis = (i.flags & FF_LOOSE_VIS) != 0u;
    } else {
        let placed = (i.flags & FF_PLACED) != 0u;
        sel = placed && (i.info.y & IF_SEL) != 0u;
        vis = placed && (i.info.y & IF_VIS) != 0u;
    }
    // Modes 2 and 3 are the two halves of the group layer: the placements
    // only, and the loose part only (pushed back by the bias).
    if ((mode == 2u && loose) || (mode == 3u && !loose)) { sel = false; }
    o.ok = select(0u, 1u, sel) | select(0u, 2u, vis);
    return o;
}

@vertex
fn vs_face(i: FaceIn) -> FaceOut {
    return face_vertex(i, 0u);
}

@vertex
fn vs_occluder(i: FaceIn) -> FaceOut {
    return face_vertex(i, 1u);
}

@vertex
fn vs_group_placed(i: FaceIn) -> FaceOut {
    return face_vertex(i, 2u);
}

@vertex
fn vs_group_loose(i: FaceIn) -> FaceOut {
    return face_vertex(i, 3u);
}

@fragment
fn fs_face(i: FaceOut) -> @location(0) vec2<u32> {
    if ((i.ok & 1u) == 0u || clipped(i.world)) { discard; }
    return i.ids;
}

@fragment
fn fs_occluder(i: FaceOut) {
    if ((i.ok & 2u) == 0u || clipped(i.world)) { discard; }
}

// ---- edges ----------------------------------------------------------------------

struct Seg {
    ok: bool,
    ca: vec4<f32>,
    cb: vec4<f32>,
    wa: vec3<f32>,
    wb: vec3<f32>,
};

// The part of a segment in front of the camera (the CPU's `_clip_segment_front`).
fn make_seg(a: vec3<f32>, b: vec3<f32>) -> Seg {
    var s: Seg;
    s.ca = P.mvp * vec4<f32>(a, 1.0);
    s.cb = P.mvp * vec4<f32>(b, 1.0);
    s.wa = a;
    s.wb = b;
    let eps = 1e-3;
    s.ok = !(s.ca.w <= eps && s.cb.w <= eps);
    if (!s.ok) { return s; }
    if (s.ca.w <= eps) {
        let t = (eps - s.ca.w) / (s.cb.w - s.ca.w);
        s.ca = mix(s.ca, s.cb, t);
        s.wa = mix(a, b, t);
    } else if (s.cb.w <= eps) {
        let t = (eps - s.ca.w) / (s.cb.w - s.ca.w);
        s.cb = mix(s.ca, s.cb, t);
        s.wb = mix(a, b, t);
    }
    return s;
}

// 0: the edge layer, the loose part's edges (`pick_edge`); 1: the group-edge
// layer, the hard edges of the placements the cursor can touch (what
// `pick_group` falls back on).
override EDGE_MODE: u32 = 0u;

fn edge_pickable(e: Edge, ins: Inst) -> bool {
    let loose = (ins.info.y & IF_LOOSE) != 0u;
    if (EDGE_MODE == 0u) {
        return loose && (e.flags & EF_LOOSE_SEL) != 0u && ((e.flags & EF_SOFT) == 0u || P.b.z > 0.5);
    }
    return !loose && (e.flags & EF_HARD) != 0u && (ins.info.y & IF_SEL) != 0u;
}

fn vertex_pickable(e: Edge, ins: Inst) -> bool {
    if ((ins.info.y & IF_LOOSE) != 0u) {
        return true;
    }
    return (e.flags & EF_HARD) != 0u && (ins.info.y & IF_SEL) != 0u;
}

fn corner(vi: u32) -> vec2<f32> {
    var c = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0));
    return c[vi];
}

struct EdgeOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) wa: vec3<f32>,
    @location(1) @interpolate(flat) wb: vec3<f32>,
    @location(2) @interpolate(flat) ids: vec2<u32>,
};

@vertex
fn vs_edge(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> EdgeOut {
    var o: EdgeOut;
    o.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);   // outside the clip volume: dropped
    let e = ii % D.n_edges;
    let k = ii / D.n_edges;
    let ed = edges[D.edge_base + e];
    let ins = insts[D.inst_first + k];
    if (!edge_pickable(ed, ins)) { return o; }
    let a = (ins.m * vec4<f32>(ed.a, 1.0)).xyz;
    let b = (ins.m * vec4<f32>(ed.b, 1.0)).xyz;
    let s = make_seg(a, b);
    if (!s.ok) { return o; }
    let pa = to_px(s.ca);
    let pb = to_px(s.cb);
    var dir = pb - pa;
    let len = length(dir);
    dir = select(vec2<f32>(1.0, 0.0), dir / max(len, 1e-9), len > 1e-6);
    let nrm = vec2<f32>(-dir.y, dir.x);
    let tol = P.b.x + 1.0;
    let c = corner(vi);
    let p = mix(pa - dir * tol, pb + dir * tol, c.x) + nrm * tol * (c.y * 2.0 - 1.0);
    o.pos = px_to_ndc(p);
    o.wa = a;
    o.wb = b;
    o.ids = vec2<u32>(ins.info.x, ed.id + 1u);
    return o;
}

struct EdgeFragOut {
    @location(0) ids: vec2<u32>,
    @builtin(frag_depth) depth: f32,
};

// Whether a face lies between the eye and a point (clip z/w `z`) seen at
// pixel `q`: the unbiased occluder, 1 mm or a relative sliver deeper.
fn occluded_at(q: vec2<f32>, z: f32) -> bool {
    if (P.b.w < 0.5) { return false; }
    let dims = vec2<i32>(i32(P.a.y), i32(P.a.z));
    let px = clamp(vec2<i32>(floor(q)), vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
    let zf = textureLoad(occluder, px, 0);
    if (zf >= 1.0) { return false; }
    let de = linear_depth(z);
    let df = linear_depth(zf);
    // The sample is the centre of the pixel, the point is somewhere in it:
    // on a face seen at a grazing angle the depth moves a lot across half a
    // pixel. Allow the face's own slope, read from the neighbours that are
    // the same surface (no background, no jump to another face).
    var slope = 0.0;
    for (var k = 0; k < 4; k++) {
        let o = select(select(vec2<i32>(0, -1), vec2<i32>(0, 1), k == 1), select(vec2<i32>(-1, 0), vec2<i32>(1, 0), k == 3), k >= 2);
        let n = clamp(px + o, vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
        let zn = textureLoad(occluder, n, 0);
        if (zn < 1.0) {
            let d = abs(linear_depth(zn) - df);
            if (d < 0.25 * df) { slope = max(slope, d); }
        }
    }
    return df < de - max(1e-3, 3e-4 * de) - slope;
}

@fragment
fn fs_edge(i: EdgeOut) -> EdgeFragOut {
    var o: EdgeFragOut;
    let s = make_seg(i.wa, i.wb);
    let pa = to_px(s.ca);
    let pb = to_px(s.cb);
    let p = i.pos.xy;
    let d = pb - pa;
    let l2 = dot(d, d);
    let t = clamp(dot(p - pa, d) / max(l2, 1e-12), 0.0, 1.0);
    let q = pa + d * t;
    let dist = length(p - q);
    let tol = P.b.x;
    if (dist >= tol) { discard; }
    // The point of the edge nearest the pixel (perspective-correct).
    let u = t * s.ca.w / (s.cb.w * (1.0 - t) + t * s.ca.w);
    let world = mix(s.wa, s.wb, u);
    if (clipped(world)) { discard; }
    let z = mix(s.ca.z / s.ca.w, s.cb.z / s.cb.w, t);
    if (occluded_at(q, z)) { discard; }
    o.ids = i.ids;
    o.depth = dist / tol * 0.999;
    return o;
}

// ---- vertices -------------------------------------------------------------------------

struct VertOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) center: vec2<f32>,
    @location(1) @interpolate(flat) world: vec3<f32>,
    @location(2) @interpolate(flat) z: f32,
    @location(3) @interpolate(flat) ids: vec2<u32>,
};

@vertex
fn vs_vertex(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertOut {
    var o: VertOut;
    o.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    let per = 2u * D.n_edges;
    let r = ii % per;
    let k = ii / per;
    let e = r / 2u;
    let end = r % 2u;
    let ed = edges[D.edge_base + e];
    let ins = insts[D.inst_first + k];
    if (!vertex_pickable(ed, ins)) { return o; }
    let local = select(ed.a, ed.b, end == 1u);
    let id = select(ed.v0, ed.v1, end == 1u);
    let world = (ins.m * vec4<f32>(local, 1.0)).xyz;
    let c = P.mvp * vec4<f32>(world, 1.0);
    if (c.w <= 1e-3) { return o; }
    let center = to_px(c);
    let tol = P.b.y + 1.0;
    let cn = corner(vi);
    o.pos = px_to_ndc(center + (cn * 2.0 - vec2<f32>(1.0, 1.0)) * tol);
    o.center = center;
    o.world = world;
    o.z = c.z / c.w;
    o.ids = vec2<u32>(ins.info.x, id + 1u);
    return o;
}

@fragment
fn fs_vertex(i: VertOut) -> EdgeFragOut {
    var o: EdgeFragOut;
    let dist = length(i.pos.xy - i.center);
    let tol = P.b.y;
    if (dist >= tol) { discard; }
    if (clipped(i.world)) { discard; }
    if (occluded_at(i.center, i.z)) { discard; }
    o.ids = i.ids;
    o.depth = dist / tol * 0.999;
    return o;
}
