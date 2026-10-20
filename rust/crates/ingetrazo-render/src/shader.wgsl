// IngeTrazo viewport shaders: the port of resources/shaders/basic.vert and
// basic.frag (OpenGL 3.3) to WGSL.
//
// Conventions kept from the GL version:
//  * every draw takes its model matrix from per-instance attributes
//    (identity for the loose mesh and for overlays);
//  * the section plane clips fragments (GL used gl_ClipDistance);
//  * gl_FragCoord has its origin at the bottom-left, so the stipple and dash
//    patterns flip y with the viewport height;
//  * a face seen from behind is `front_facing == false`, flipped for the
//    mirrored placements (negative determinant) by the instance flag.

struct Params {
    mvp: mat4x4<f32>,
    clip: vec4<f32>,        // world plane n.p + d >= 0 is kept
    color: vec4<f32>,       // flat colour (front)
    back_color: vec4<f32>,  // back tint / flat back colour
    fade_color: vec4<f32>,  // colour the faded context mixes toward
    v0: vec4<f32>,          // x clip_enable, y col_mode, z stipple, w opacity
    v1: vec4<f32>,          // x viewport_w, y viewport_h, z dash_px, w back_tint_on
    v2: vec4<f32>,          // x hard_cutout
};

@group(0) @binding(0) var<uniform> P: Params;
@group(1) @binding(0) var tex: texture_2d<f32>;
@group(1) @binding(1) var samp: sampler;

fn inst_matrix(m0: vec4<f32>, m1: vec4<f32>, m2: vec4<f32>, m3: vec4<f32>) -> mat4x4<f32> {
    return mat4x4<f32>(m0, m1, m2, m3);
}

// ---- screen-space patterns ---------------------------------------------------

fn gl_frag(fc: vec4<f32>) -> vec2<f32> {
    return vec2<f32>(fc.x, P.v1.y - fc.y);
}

// Returns true when the fragment must be dropped by the stipple mode.
fn stipple_discards(fc: vec4<f32>) -> bool {
    let mode = u32(P.v0.z + 0.5);
    let g = gl_frag(fc);
    if (mode == 1u) {
        return (g.x % 4.0 >= 1.0) && (g.y % 4.0 >= 1.0);
    } else if (mode == 2u) {
        return (g.x + g.y) % 6.0 >= 3.0;
    } else if (mode == 3u) {
        return (g.x % 6.0 >= 2.0) || (g.y % 6.0 >= 2.0);
    }
    return false;
}

fn clipped(world: vec3<f32>) -> bool {
    return P.v0.x > 0.5 && dot(vec4<f32>(world, 1.0), P.clip) < 0.0;
}

// 4x4 Bayer thresholds, strictly inside (0, 1).
fn bayer(x: i32, y: i32) -> f32 {
    var b = array<f32, 16>(
        0.5, 8.5, 2.5, 10.5,
        12.5, 4.5, 14.5, 6.5,
        3.5, 11.5, 1.5, 9.5,
        15.5, 7.5, 13.5, 5.5);
    return b[y * 4 + x] / 16.0;
}

// ---- faces -------------------------------------------------------------------

struct FaceIn {
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) rgba: vec4<f32>,
    @location(3) misc: u32,
    @location(10) m0: vec4<f32>,
    @location(11) m1: vec4<f32>,
    @location(12) m2: vec4<f32>,
    @location(13) m3: vec4<f32>,
    @location(14) imisc: vec4<f32>,
};

struct FaceOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) rgba: vec4<f32>,
    @location(2) @interpolate(flat) misc: u32,
    @location(3) world: vec3<f32>,
    @location(4) @interpolate(flat) imisc: vec4<f32>,
};

@vertex
fn vs_face(i: FaceIn) -> FaceOut {
    var o: FaceOut;
    let world = inst_matrix(i.m0, i.m1, i.m2, i.m3) * vec4<f32>(i.pos, 1.0);
    o.pos = P.mvp * world;
    o.uv = i.uv;
    o.rgba = i.rgba;
    o.misc = i.misc;
    o.world = world.xyz;
    o.imisc = i.imisc;
    return o;
}

@fragment
fn fs_face(i: FaceOut, @builtin(front_facing) ff: bool) -> @location(0) vec4<f32> {
    let fc = i.pos;
    // Sampled first: derivatives need uniform control flow.
    let texel = textureSample(tex, samp, i.uv);
    if (clipped(i.world)) { discard; }
    if (stipple_discards(fc)) { discard; }
    let front = ff != (i.imisc.y > 0.5);
    let flags = i.misc & 255u;
    let op_v = f32((i.misc >> 8u) & 255u) / 255.0;
    let shade = f32((i.misc >> 16u) & 63u) / 32.0;
    if ((flags & 2u) != 0u && !front) { discard; }
    if ((flags & 4u) != 0u && front) { discard; }
    let mode = u32(P.v0.y + 0.5);
    let opacity = op_v * P.v0.w;
    var c = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if (!front && (flags & 1u) != 0u && P.v1.w > 0.5 && mode != 2u) {
        // The default back: the style's tint on the rear of an unpainted back.
        c = vec4<f32>(P.back_color.rgb, 1.0);
    } else if (mode == 1u) {
        if (opacity > 0.999 && texel.a < 0.5) {
            if (P.v2.x > 0.5) { discard; }
            let bx = i32(fc.x % 4.0);
            let by = i32(gl_frag(fc).y % 4.0);
            if (texel.a < bayer(bx, by)) { discard; }
            c = vec4<f32>(texel.rgb * shade, 1.0);
        } else {
            c = vec4<f32>(texel.rgb * shade, texel.a * opacity);
        }
    } else if (mode == 2u) {
        c = select(P.back_color, P.color, front);
    } else if (mode == 3u) {
        c = vec4<f32>(P.color.rgb, 1.0);
    } else {
        c = vec4<f32>(i.rgba.rgb, opacity);
    }
    let rgb = mix(c.rgb, P.fade_color.rgb, i.imisc.x);
    return vec4<f32>(rgb, c.a);
}

// ---- lines ---------------------------------------------------------------------

struct LineIn {
    @location(0) pos: vec3<f32>,
    @location(1) anchor: vec3<f32>,
    @location(10) m0: vec4<f32>,
    @location(11) m1: vec4<f32>,
    @location(12) m2: vec4<f32>,
    @location(13) m3: vec4<f32>,
    @location(14) imisc: vec4<f32>,
};

struct LineOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) @interpolate(flat) anchor: vec4<f32>,
    @location(2) @interpolate(flat) fade: f32,
};

@vertex
fn vs_line(i: LineIn) -> LineOut {
    var o: LineOut;
    let m = inst_matrix(i.m0, i.m1, i.m2, i.m3);
    let world = m * vec4<f32>(i.pos, 1.0);
    o.pos = P.mvp * world;
    o.world = world.xyz;
    o.anchor = P.mvp * (m * vec4<f32>(i.anchor, 1.0));
    o.fade = i.imisc.x;
    return o;
}

@fragment
fn fs_line(i: LineOut) -> @location(0) vec4<f32> {
    let fc = i.pos;
    if (clipped(i.world)) { discard; }
    if (stipple_discards(fc)) { discard; }
    if (u32(P.v0.z + 0.5) == 4u) {
        let o = (i.anchor.xy / i.anchor.w * 0.5 + vec2<f32>(0.5)) * P.v1.xy;
        let d = length(gl_frag(fc) - o);
        if (d % (2.0 * P.v1.z) >= P.v1.z) { discard; }
    }
    let rgb = mix(P.color.rgb, P.fade_color.rgb, i.fade);
    return vec4<f32>(rgb, P.color.a);
}

// ---- profile (silhouette) edges ----------------------------------------------------

struct SoftIn {
    @location(0) pos: vec3<f32>,
    @location(1) n0: vec3<f32>,
    @location(2) d0: f32,
    @location(3) n1: vec3<f32>,
    @location(4) d1: f32,
    @location(5) single: f32,
    @location(10) m0: vec4<f32>,
    @location(11) m1: vec4<f32>,
    @location(12) m2: vec4<f32>,
    @location(13) m3: vec4<f32>,
    @location(14) imisc: vec4<f32>,
    @location(15) eye: vec4<f32>,
};

// A soft edge is a profile where one of its faces turns toward the eye and
// the other away (or where it bounds an open surface). The test runs in the
// placement's local frame, where the eye is `eye`.
@vertex
fn vs_soft(i: SoftIn) -> LineOut {
    var o: LineOut;
    let s0 = i.d0 - dot(i.n0, i.eye.xyz);
    let s1 = i.d1 - dot(i.n1, i.eye.xyz);
    // Placements outside the group open for editing (fade > 0) get no outline.
    let visible = (i.single > 0.5 || ((s0 < 0.0) != (s1 < 0.0))) && i.imisc.x < 0.01;
    let world = inst_matrix(i.m0, i.m1, i.m2, i.m3) * vec4<f32>(i.pos, 1.0);
    o.pos = select(vec4<f32>(0.0, 0.0, 2.0, 1.0), P.mvp * world, visible);
    o.world = world.xyz;
    o.anchor = o.pos;
    o.fade = i.imisc.x;
    return o;
}

// ---- backdrop -------------------------------------------------------------------------

struct SkyIn {
    @location(0) pos: vec2<f32>,
    @location(1) color: vec3<f32>,
};
struct SkyOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_sky(i: SkyIn) -> SkyOut {
    var o: SkyOut;
    o.pos = vec4<f32>(i.pos, 0.0, 1.0);
    o.color = i.color;
    return o;
}

@fragment
fn fs_sky(i: SkyOut) -> @location(0) vec4<f32> {
    return vec4<f32>(i.color, 1.0);
}

// ---- blit (egui integration) ---------------------------------------------------------------

struct BlitOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_blit(@builtin(vertex_index) vi: u32) -> BlitOut {
    var o: BlitOut;
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    o.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    o.uv = vec2<f32>(x, y);
    return o;
}

@fragment
fn fs_blit(i: BlitOut) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, i.uv);
}
