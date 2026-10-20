// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
// Port of openskp (MIT, Copyright (c) 2024-2026 Ahsan Mehmood): see NOTICE-openskp.md.
//! The classic (SketchUp 2013-2020) container: not a zip, but one
//! uncompressed MFC `CArchive` object stream with a single 1-based store map
//! (`openskp/legacy.py`, with the dimension fix IngeTrazo carried in
//! `formats/openskp_compat.py`).
//!
//! The stream is read exactly as the Python walker does, quirks included: a
//! class is declared once and later objects name it by its store index, an
//! object seen before is a back-reference, and a few records depend on the
//! file's version. A `Val` is what one record decodes to; the store map keeps
//! them behind `Rc`, as Python's slots share one dict.

use std::collections::HashMap;
use std::rc::Rc;

use crate::error::{Result, SkpError};
use crate::model::*;
use crate::pystr::{utf16le_ignore, utf16le_replace};
use crate::raw::*;

const STR_MARKER: [u8; 3] = [0xFF, 0xFE, 0xFF];

/// The two kinds of failure the walker tells apart: a structural error the
/// format reader raises itself, and the low-level ones (a read past the end,
/// a bad decode).
#[derive(Debug, Clone)]
pub enum LErr {
    Parse(String),
    Other(String),
}

impl LErr {
    pub fn message(&self) -> &str {
        match self {
            LErr::Parse(s) | LErr::Other(s) => s,
        }
    }
}

type L<T> = std::result::Result<T, LErr>;

fn perr<T>(msg: impl Into<String>) -> L<T> {
    Err(LErr::Parse(msg.into()))
}

fn oerr<T>(msg: &str) -> L<T> {
    Err(LErr::Other(msg.to_string()))
}

/// `_R`: the byte cursor.
#[derive(Clone)]
pub struct Cur<'a> {
    pub data: &'a [u8],
    pub pos: usize,
}

impl<'a> Cur<'a> {
    pub fn new(data: &'a [u8], pos: usize) -> Cur<'a> {
        Cur { data, pos }
    }
    fn take<const N: usize>(&mut self) -> L<[u8; N]> {
        let s = self.data.get(self.pos..self.pos + N).ok_or_else(|| LErr::Other("unpack requires a buffer".into()))?;
        self.pos += N;
        Ok(s.try_into().unwrap())
    }
    fn u8(&mut self) -> L<u8> {
        let v = *self.data.get(self.pos).ok_or_else(|| LErr::Other("index out of range".into()))?;
        self.pos += 1;
        Ok(v)
    }
    fn u16(&mut self) -> L<u16> {
        Ok(u16::from_le_bytes(self.take::<2>()?))
    }
    fn u32(&mut self) -> L<u32> {
        Ok(u32::from_le_bytes(self.take::<4>()?))
    }
    fn f64(&mut self) -> L<f64> {
        Ok(f64::from_le_bytes(self.take::<8>()?))
    }
    fn f64s<const N: usize>(&mut self) -> L<[f64; N]> {
        let mut out = [0.0; N];
        for x in out.iter_mut() {
            *x = self.f64()?;
        }
        Ok(out)
    }
    /// `raw(n)`: what is there, up to `n` bytes; the cursor moves by `n`.
    fn raw(&mut self, n: usize) -> &'a [u8] {
        let start = self.pos.min(self.data.len());
        let end = (self.pos.saturating_add(n)).min(self.data.len());
        self.pos = self.pos.saturating_add(n);
        &self.data[start..end]
    }
    fn peek(&self, n: usize) -> &'a [u8] {
        let start = self.pos.min(self.data.len());
        let end = (self.pos.saturating_add(n)).min(self.data.len());
        &self.data[start..end]
    }
    fn peek_u16(&self) -> L<u16> {
        match self.data.get(self.pos..self.pos + 2) {
            Some(b) => Ok(u16::from_le_bytes([b[0], b[1]])),
            None => oerr("unpack requires a buffer of 2 bytes"),
        }
    }
    fn utf16(&mut self) -> L<String> {
        if self.peek(3) != STR_MARKER {
            return perr(format!("expected a string record {}", self.ctx()));
        }
        self.pos += 3;
        let mut n = self.u8()? as usize;
        if n == 0xFF {
            n = self.u16()? as usize;
            if n == 0xFFFF {
                n = self.u32()? as usize;
            }
        }
        Ok(utf16le_replace(self.raw(2 * n)))
    }
    fn ctx(&self) -> String {
        let p = self.pos;
        let hex = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<Vec<_>>().join(" ");
        let back = &self.data[p.saturating_sub(16).min(self.data.len())..p.min(self.data.len())];
        let fwd = &self.data[p.min(self.data.len())..(p + 32).min(self.data.len())];
        format!("@{p:#x}: …{} | {}…", hex(back), hex(fwd))
    }
}

// ---- decoded records --------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct Db {
    pub mat: u16,
    pub hidden: u8,
    pub soft: u8,
    pub smooth: u8,
    pub layer: u16,
}

type Ent = (Option<i64>, Option<Rc<str>>, Option<Rc<Val>>);

#[derive(Debug)]
pub enum Val {
    Vertex([f64; 3]),
    Edge { db: Db, curve: Option<i64>, curve_is_arc: bool, v1: Option<i64>, v2: Option<i64> },
    Curve,
    ArcCurve,
    EdgeUse { edge: Option<i64>, sense: u8 },
    Loop { uses: Vec<Option<Rc<Val>>> },
    Face { db: Db, plane: [f64; 4], loops: Vec<Option<Rc<Val>>>, back_mat: u16, attrs: Option<Rc<Val>> },
    Layer { name: String, hidden: u8, rgba: [u8; 4] },
    Material(MatRec),
    Dib { data: Rc<Vec<u8>> },
    Attrs { children: Vec<(Option<Rc<str>>, Option<Rc<Val>>)> },
    Dict { name: String, entries: Vec<(String, AttrValue)> },
    Camera,
    Thumbnail,
    Relationship,
    Definition { name: String, guid: String, ents: Vec<Ent>, faces_camera: bool, shadows_face_sun: bool },
    Image { db: Db, def: Option<i64>, xform: [f64; 12], guid: String },
    Instance { db: Db, def: Option<i64>, xf: [f64; 13], name: String, guid: String, attrs: Option<Rc<Val>> },
    Ftc { front: [f64; 9], back: [f64; 9], front_projected: bool, back_projected: bool },
    CLine { point: [f64; 3], direction: [f64; 3], start: Option<[f64; 3]>, end: Option<[f64; 3]> },
    CPoint { pos: [f64; 3] },
    SectionPlane { plane: [f64; 4], name: String, label: String, hidden: bool },
    Font,
    Dim { db: Db, text: String, connect: (Option<i64>, Option<i64>), offset: f64, a: Option<[f64; 3]>, b: Option<[f64; 3]> },
    Text { text: String, db: Db, p: Option<[f64; 3]>, lp: Option<[f64; 3]> },
}

#[derive(Debug, Clone, Default)]
pub struct MatRec {
    pub name: String,
    pub rgba: [u8; 4],
    pub opacity: f64,
    pub use_opacity: u8,
    pub textured: bool,
    pub tex_dib: Option<i64>,
    pub tex_w: f64,
    pub tex_h: f64,
    pub tex_file: String,
    pub colorized: bool,
}

enum Slot {
    Class { name: Rc<str> },
    Obj { name: Rc<str>, value: Option<Rc<Val>> },
}

/// What `read_object` returns: `(slot, class name, value)`.
#[derive(Clone)]
struct ObjRef {
    slot: Option<i64>,
    name: Option<Rc<str>>,
    value: Option<Rc<Val>>,
}

impl ObjRef {
    fn null() -> ObjRef {
        ObjRef { slot: None, name: None, value: None }
    }
    fn is(&self, class: &str) -> bool {
        self.name.as_deref() == Some(class)
    }
}

struct Ar<'a> {
    data: &'a [u8],
    ver: i32,
    has_pid: bool,
    slots: HashMap<i64, Slot>,
    alloc_order: Vec<i64>,
    class_slot: HashMap<Rc<str>, i64>,
    next_slot: i64,
    walk_base: i64,
    in_entity_list: bool,
    current_loop: Option<i64>,
    burns: Vec<(i64, i64)>,
    cum_delta: i64,
    annot_watermark: Option<i64>,
    burn_stack: Vec<i64>,
    cline_tail: Option<usize>,
    depth: usize,
}

/// How many objects may be read inside one another.
const MAX_OBJECT_DEPTH: usize = 300;

impl<'a> Ar<'a> {
    fn new(data: &'a [u8], ver: i32) -> Ar<'a> {
        Ar {
            data,
            ver,
            has_pid: ver >= 17,
            slots: HashMap::new(),
            alloc_order: Vec::new(),
            class_slot: HashMap::new(),
            next_slot: 0,
            walk_base: 0,
            in_entity_list: false,
            current_loop: None,
            burns: Vec::new(),
            cum_delta: 0,
            annot_watermark: None,
            burn_stack: Vec::new(),
            cline_tail: None,
            depth: 0,
        }
    }

    fn alloc(&mut self, entry: Slot) -> i64 {
        let s = self.next_slot;
        self.slots.insert(s, entry);
        self.alloc_order.push(s);
        self.next_slot += 1;
        s
    }

    fn read_object(&mut self, r: &mut Cur, expect: Option<&str>) -> L<ObjRef> {
        let tag = r.u16()?;
        if tag == 0 {
            return Ok(ObjRef::null());
        }
        if tag == 0x7FFF {
            let big = r.u32()?;
            if big & 0x80000000 != 0 {
                return self.new_of_class(r, (big & 0x7FFF_FFFF) as i64, expect);
            }
            return self.backref(big as i64, r);
        }
        if tag == 0xFFFF {
            let _schema = r.u16()?;
            let namelen = r.u16()? as usize;
            if namelen > 40 {
                return perr(format!("implausible class name length {}", r.ctx()));
            }
            let raw = r.raw(namelen);
            if !raw.is_ascii() {
                return oerr("'ascii' codec can't decode byte");
            }
            let name: Rc<str> = Rc::from(std::str::from_utf8(raw).unwrap());
            let s = self.alloc(Slot::Class { name: name.clone() });
            self.class_slot.insert(name.clone(), s);
            return self.new_obj(r, &name);
        }
        if tag & 0x8000 != 0 {
            return self.new_of_class(r, (tag & 0x7FFF) as i64, expect);
        }
        self.backref(tag as i64, r)
    }

    fn new_of_class(&mut self, r: &mut Cur, cslot: i64, expect: Option<&str>) -> L<ObjRef> {
        if let std::collections::hash_map::Entry::Vacant(v) = self.slots.entry(cslot) {
            let Some(expect) = expect else {
                return perr(format!("class-ref to unknown slot {cslot} {}", r.ctx()));
            };
            let name: Rc<str> = Rc::from(expect);
            v.insert(Slot::Class { name: name.clone() });
            self.class_slot.insert(name, cslot);
        }
        match self.slots.get(&cslot) {
            Some(Slot::Class { name }) => {
                let name = name.clone();
                self.new_obj(r, &name)
            }
            Some(Slot::Obj { name, .. }) => perr(format!("class-ref to non-class slot {cslot} ({name}) {}", r.ctx())),
            None => unreachable!(),
        }
    }

    fn new_obj(&mut self, r: &mut Cur, name: &Rc<str>) -> L<ObjRef> {
        self.in_entity_list = false;
        // Python's recursion limit stops an archive that nests objects
        // deeper than any real model does; here it would be the stack.
        if self.depth >= MAX_OBJECT_DEPTH {
            return oerr("RecursionError: maximum recursion depth exceeded");
        }
        let slot = self.alloc(Slot::Obj { name: name.clone(), value: None });
        self.depth += 1;
        let read = self.read_class(name, r);
        self.depth -= 1;
        let value = match read {
            Some(res) => res?,
            None => return perr(format!("no reader for class {name} {}", r.ctx())),
        };
        let value = Rc::new(value);
        self.slots.insert(slot, Slot::Obj { name: name.clone(), value: Some(value.clone()) });
        if &**name == "CDimensionLinear" || &**name == "CText" {
            self.annot_watermark = Some(self.next_slot);
        }
        Ok(ObjRef { slot: Some(slot), name: Some(name.clone()), value: Some(value) })
    }

    fn translate_ref(&self, slot: i64) -> Option<i64> {
        let mut offset = 0;
        for &(start, width) in &self.burns {
            if slot < start {
                break;
            }
            if slot < start + width {
                return None;
            }
            offset += width;
        }
        Some(slot - offset)
    }

    fn backref(&mut self, slot: i64, r: &Cur) -> L<ObjRef> {
        let mut slot = slot;
        if !self.burns.is_empty() && slot >= self.burns[0].0 {
            match self.translate_ref(slot) {
                None => return Ok(ObjRef { slot: Some(slot), name: Some(Rc::from("reserved")), value: None }),
                Some(w) => slot = w,
            }
        }
        match self.slots.get(&slot) {
            None => {
                if slot < self.walk_base {
                    return Ok(ObjRef { slot: Some(slot), name: Some(Rc::from("premodel")), value: None });
                }
                perr(format!("back-ref to unwalked slot {slot} {}", r.ctx()))
            }
            Some(Slot::Class { .. }) => perr(format!("back-ref to class slot {slot} {}", r.ctx())),
            Some(Slot::Obj { name, value }) => Ok(ObjRef { slot: Some(slot), name: Some(name.clone()), value: value.clone() }),
        }
    }

    fn read_class(&mut self, name: &str, r: &mut Cur) -> Option<L<Val>> {
        Some(match name {
            "CVertex" => read_vertex(self, r),
            "CEdge" => read_edge(self, r),
            "CCurve" => read_curve(self, r),
            "CArcCurve" => read_arccurve(self, r),
            "CEdgeUse" => read_edgeuse(self, r),
            "CLoop" => read_loop(self, r),
            "CFace" => read_face(self, r),
            "CLayer" => read_layer(self, r),
            "CMaterial" => read_material(self, r),
            "CDib" => read_dib(self, r),
            "CAttributeContainer" => read_attr_container(self, r),
            "CAttributeNamed" => read_attr_named(self, r),
            "CCamera" => read_camera(self, r),
            "CThumbnail" => read_thumbnail(self, r),
            "CRelationship" => read_relationship(self, r),
            "CComponentDefinition" => read_definition(self, r),
            "CImage" => read_image(self, r),
            "CComponentInstance" | "CGroup" => read_instance(self, r),
            "CFaceTextureCoords" => read_ftc(self, r),
            "CConstructionLine" => read_constructionline(self, r),
            "CConstructionPoint" => read_constructionpoint(self, r),
            "CSectionPlane" => read_sectionplane(self, r),
            "CSkFont" => read_skfont(self, r),
            "CDimensionLinear" => read_dimlinear(self, r),
            "CText" => read_text(self, r),
            _ => return None,
        })
    }

    fn plausible_list_tag(&self, at: usize) -> bool {
        let Some(b) = self.data.get(at..at + 2) else { return false };
        let t = u16::from_le_bytes([b[0], b[1]]);
        if matches!(t, 0 | 0x7FFF | 0xFFFF) {
            return true;
        }
        if t & 0x8000 != 0 {
            return matches!(self.slots.get(&((t & 0x7FFF) as i64)), Some(Slot::Class { .. }));
        }
        (t as i64) < self.next_slot
    }

    fn strict_next_tag(&self, at: usize, allow_null: bool) -> bool {
        let Some(b) = self.data.get(at..at + 2) else { return false };
        let t = u16::from_le_bytes([b[0], b[1]]);
        if t == 0 {
            return allow_null;
        }
        if matches!(t, 0x7FFF | 0xFFFF) {
            return true;
        }
        if t & 0x8000 != 0 {
            return matches!(self.slots.get(&((t & 0x7FFF) as i64)), Some(Slot::Class { .. }));
        }
        false
    }
}

// ---- counts the SketchUp 2020 filler can hide ----------------------------------

fn find_from(data: &[u8], needle: &[u8], from: usize, to: usize) -> Option<usize> {
    let to = to.min(data.len());
    if from >= to {
        return None;
    }
    data[from..to].windows(needle.len()).position(|w| w == needle).map(|i| i + from)
}

/// `_retry_count_after_v20_filler(r, count_pos, limit, ar)`.
fn retry_count_after_v20_filler(r: &mut Cur, count_pos: usize, limit: u32, ar: Option<&Ar>) -> Option<u32> {
    let data = r.data;
    let mut marker_at: Option<usize> = None;
    let upto = (count_pos + 12).min(data.len().saturating_sub(3));
    for i in count_pos..upto {
        if data.get(i..i + 3) == Some(&STR_MARKER[..]) {
            marker_at = Some(i);
            break;
        }
    }
    let marker_at = marker_at?;
    if *data.get(marker_at + 3)? != 0 {
        return None;
    }
    let mut at = marker_at + 4;
    while at < data.len() && data[at] == 0 {
        at += 1;
    }
    for back in 0..4usize {
        let Some(at2) = at.checked_sub(back) else { continue };
        if at2 < marker_at + 4 || at2 + 4 > data.len() {
            continue;
        }
        let count = u32::from_le_bytes(data[at2..at2 + 4].try_into().unwrap());
        if !(0 < count && count <= limit) {
            continue;
        }
        if let Some(ar) = ar {
            if !ar.plausible_list_tag(at2 + 4) {
                continue;
            }
        }
        r.pos = at2 + 4;
        return Some(count);
    }
    None
}

fn is_class_ref(data: &[u8], p: usize, slot: i64) -> bool {
    if slot < 0x7FFF {
        return data.get(p..p + 2).is_some_and(|b| u16::from_le_bytes([b[0], b[1]]) as i64 == (0x8000 | slot));
    }
    match (data.get(p..p + 2), data.get(p + 2..p + 6)) {
        (Some(a), Some(b)) => u16::from_le_bytes([a[0], a[1]]) == 0x7FFF && u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64 == (0x8000_0000i64 | slot),
        _ => false,
    }
}

// ---- shared record blocks ----------------------------------------------------

struct Pre {
    attrs: Option<Rc<Val>>,
}

fn preamble(ar: &mut Ar, r: &mut Cur) -> L<Pre> {
    let o = ar.read_object(r, Some("CAttributeContainer"))?;
    if ar.has_pid {
        let mask = r.u8()?;
        for bit in 0..8 {
            if mask & (1 << bit) != 0 {
                r.u8()?;
            }
        }
    }
    Ok(Pre { attrs: o.value })
}

fn drawbase(ar: &mut Ar, r: &mut Cur) -> L<Db> {
    let b = r.raw(8);
    let lay_cls = ar.class_slot.get("CLayer").copied();
    let tag = r.peek_u16()?;
    let layer = if lay_cls.is_some_and(|c| tag as i64 == (0x8000 | c)) {
        ar.read_object(r, Some("CLayer"))?;
        0
    } else if tag == 0x7FFF {
        r.u16()?;
        let big = r.u32()?;
        if big & 0x80000000 != 0 {
            return perr(format!("drawbase layer: unexpected class {}", r.ctx()));
        }
        0
    } else {
        r.u16()?
    };
    let at = |i: usize| b.get(i).copied().ok_or_else(|| LErr::Other("index out of range".into()));
    if b.len() < 2 {
        return oerr("unpack requires a buffer of 2 bytes");
    }
    Ok(Db { mat: u16::from_le_bytes([b[0], b[1]]), hidden: at(2)?, soft: at(5)?, smooth: at(6)?, layer })
}

// ---- entity readers --------------------------------------------------------------

fn read_vertex(ar: &mut Ar, r: &mut Cur) -> L<Val> {
    preamble(ar, r)?;
    Ok(Val::Vertex(r.f64s::<3>()?))
}

fn read_edge(ar: &mut Ar, r: &mut Cur) -> L<Val> {
    preamble(ar, r)?;
    let db = drawbase(ar, r)?;
    let s1 = ar.read_object(r, Some("CVertex"))?;
    let s2 = ar.read_object(r, Some("CVertex"))?;
    let c = ar.read_object(r, None)?;
    if !(c.name.is_none() || c.is("CCurve") || c.is("CArcCurve")) {
        return perr(format!("edge curve pointer resolved to {} {}", c.name.as_deref().unwrap_or(""), r.ctx()));
    }
    Ok(Val::Edge { db, curve: c.slot, curve_is_arc: c.is("CArcCurve"), v1: s1.slot, v2: s2.slot })
}

fn read_curve(ar: &mut Ar, r: &mut Cur) -> L<Val> {
    preamble(ar, r)?;
    r.u8()?;
    r.u32()?;
    Ok(Val::Curve)
}

fn read_arccurve(ar: &mut Ar, r: &mut Cur) -> L<Val> {
    preamble(ar, r)?;
    r.raw(5);
    r.f64s::<14>()?;
    Ok(Val::ArcCurve)
}

fn register_burn(ar: &mut Ar, delta: i64) {
    let wm = ar.annot_watermark.unwrap_or(0);
    ar.burns.push((wm + ar.cum_delta, delta));
    ar.cum_delta += delta;
    ar.annot_watermark = None;
    if let Some(last) = ar.burn_stack.last_mut() {
        *last += 1;
    }
}

fn read_edgeuse(ar: &mut Ar, r: &mut Cur) -> L<Val> {
    preamble(ar, r)?;
    let es = ar.read_object(r, Some("CEdge"))?;
    let sense = r.u8()?;
    let p0 = r.pos;
    let tag = r.u16()?;
    let ps: Option<i64> = if tag == 0x7FFF {
        let v = r.u32()?;
        if v & 0x80000000 != 0 {
            return perr(format!("edge-use parent is a new object {}", r.ctx()));
        }
        Some(v as i64)
    } else if tag == 0xFFFF || tag & 0x8000 != 0 {
        return perr(format!("edge-use parent is a new object {}", r.ctx()));
    } else if tag != 0 {
        Some(tag as i64)
    } else {
        None
    };
    let expected = ar.current_loop.map(|c| c + ar.cum_delta);
    if ps != expected {
        let delta = match (ps, expected) {
            (Some(p), Some(e)) => p - e,
            _ => 0,
        };
        if 0 < delta && delta <= 4096 && ar.annot_watermark.is_some() {
            register_burn(ar, delta);
        } else {
            r.pos = p0;
            return perr(format!("edge-use parent slot {} != current loop {} {}", ps.map(|x| x.to_string()).unwrap_or("None".into()), expected.map(|x| x.to_string()).unwrap_or("None".into()), r.ctx()));
        }
    }
    Ok(Val::EdgeUse { edge: es.slot, sense })
}

fn read_loop(ar: &mut Ar, r: &mut Cur) -> L<Val> {
    let my_slot = ar.next_slot - 1;
    let prev = ar.current_loop;
    ar.current_loop = Some(my_slot);
    preamble(ar, r)?;
    r.raw(2);
    let mut uses = Vec::new();
    loop {
        if r.peek_u16()? == 0 {
            r.pos += 2;
            break;
        }
        let o = ar.read_object(r, Some("CEdgeUse"))?;
        uses.push(o.value);
    }
    ar.current_loop = prev;
    Ok(Val::Loop { uses })
}

fn read_face(ar: &mut Ar, r: &mut Cur) -> L<Val> {
    let pre = preamble(ar, r)?;
    let db = drawbase(ar, r)?;
    let plane = r.f64s::<4>()?;
    let nloops = r.u32()?;
    if nloops > 10000 {
        return perr(format!("implausible loop count {nloops} {}", r.ctx()));
    }
    let mut loops = Vec::new();
    for _ in 0..nloops {
        loops.push(ar.read_object(r, Some("CLoop"))?.value);
    }
    let back_mat = r.u16()?;
    Ok(Val::Face { db, plane, loops, back_mat, attrs: pre.attrs })
}

fn read_attr_container(ar: &mut Ar, r: &mut Cur) -> L<Val> {
    preamble(ar, r)?;
    let mut children = Vec::new();
    loop {
        if r.peek_u16()? == 0 {
            r.pos += 2;
            break;
        }
        let o = ar.read_object(r, Some("CAttributeNamed"))?;
        children.push((o.name, o.value));
    }
    Ok(Val::Attrs { children })
}

fn read_typed(r: &mut Cur, t: u8) -> L<AttrValue> {
    read_typed_at(r, t, 0)
}

fn read_typed_at(r: &mut Cur, t: u8, depth: usize) -> L<AttrValue> {
    if depth > 200 {
        return oerr("RecursionError: maximum recursion depth exceeded");
    }
    Ok(match t {
        0x00 => AttrValue::Null,
        0x04 => AttrValue::Int(i32::from_le_bytes(r.take::<4>()?) as i64),
        0x06 => AttrValue::Float(r.f64()?),
        0x07 => AttrValue::Int(r.u8()? as i64),
        0x09 => AttrValue::Int(r.u32()? as i64),
        0x0A => AttrValue::Str(r.utf16()?),
        0x0C => AttrValue::Float(r.f64()?),
        0x0B => {
            let n = r.u32()?;
            if n > 100000 {
                return perr(format!("implausible attr array count {}", r.ctx()));
            }
            let mut v = Vec::new();
            for _ in 0..n {
                let tt = r.u8()?;
                v.push(read_typed_at(r, tt, depth + 1)?);
            }
            AttrValue::List(v)
        }
        0x11 | 0x12 => AttrValue::Vec3(r.f64s::<3>()?),
        _ => return perr(format!("unknown attribute value type {t:#x} {}", r.ctx())),
    })
}

fn read_attr_named(ar: &mut Ar, r: &mut Cur) -> L<Val> {
    preamble(ar, r)?;
    r.raw(4);
    let name = r.utf16()?;
    let mut entries: Vec<(String, AttrValue)> = Vec::new();
    loop {
        let key = r.utf16()?;
        if key.is_empty() {
            break;
        }
        let t = r.u8()?;
        let v = read_typed(r, t)?;
        match entries.iter_mut().find(|(k, _)| *k == key) {
            Some(slot) => slot.1 = v,
            None => entries.push((key, v)),
        }
    }
    r.u32()?;
    Ok(Val::Dict { name, entries })
}

struct TexBlock {
    rgba: [u8; 4],
    opacity: f64,
    use_opacity: u8,
    tex_dib: Option<i64>,
    tex_w: f64,
    tex_h: f64,
    tex_file: String,
    colorized: bool,
}

fn texture_block(ar: &mut Ar, r: &mut Cur) -> L<TexBlock> {
    r.raw(if ar.ver >= 17 { 2 } else { 1 });
    let o = ar.read_object(r, Some("CDib"))?;
    if !matches!(o.value.as_deref(), Some(Val::Dib { .. })) {
        return perr(format!("texture object is not a dib {}", r.ctx()));
    }
    let marker = find_from(r.data, &STR_MARKER, r.pos, r.pos + 28);
    let diff = marker.map(|m| m as i64 - r.pos as i64).unwrap_or(-1 - r.pos as i64);
    if diff == 20 {
        r.u32()?;
    } else if diff != 16 {
        return perr(format!("texture size block misaligned {}", r.ctx()));
    }
    let w = r.f64()?;
    let h = r.f64()?;
    let fname = r.utf16()?;
    let avg = r.raw(9);
    r.utf16()?;
    let blob = r.raw(8);
    let opacity = r.f64()?;
    let use_op = r.u8()?;
    if avg.len() < 4 || blob.len() < 5 {
        return oerr("index out of range");
    }
    let colorized = blob[4] != 0 || avg[3] == 0xFF;
    Ok(TexBlock { rgba: [avg[0], avg[1], avg[2], avg[3]], opacity, use_opacity: use_op, tex_dib: o.slot, tex_w: w, tex_h: h, tex_file: fname, colorized })
}

fn rgba4(b: &[u8]) -> L<[u8; 4]> {
    if b.len() < 4 {
        return oerr("tuple index out of range");
    }
    Ok([b[0], b[1], b[2], b[3]])
}

fn read_layer(ar: &mut Ar, r: &mut Cur) -> L<Val> {
    preamble(ar, r)?;
    let name = r.utf16()?;
    let mut mid: Vec<u8> = Vec::new();
    while mid.len() < 8 && r.peek(3) != STR_MARKER {
        mid.extend_from_slice(r.raw(1));
        if r.pos > r.data.len() + 8 {
            break;
        }
    }
    r.utf16()?;
    let flags = r.u16()?;
    let hidden = mid.first().copied().unwrap_or(0);
    if flags & 0x00FF != 0 {
        let tex = texture_block(ar, r)?;
        r.raw(4);
        return Ok(Val::Layer { name, hidden, rgba: tex.rgba });
    }
    let rgba = rgba4(r.raw(4))?;
    r.utf16()?;
    r.raw(21);
    Ok(Val::Layer { name, hidden, rgba })
}

fn read_material(ar: &mut Ar, r: &mut Cur) -> L<Val> {
    preamble(ar, r)?;
    let name = r.utf16()?;
    let texflag = r.u16()?;
    let mut m = MatRec { name, ..Default::default() };
    if texflag == 0 {
        m.rgba = rgba4(r.raw(4))?;
        r.utf16()?;
        r.raw(8);
        m.opacity = r.f64()?;
        m.use_opacity = r.u8()?;
    } else {
        let t = texture_block(ar, r)?;
        m.textured = true;
        m.rgba = t.rgba;
        m.opacity = t.opacity;
        m.use_opacity = t.use_opacity;
        m.tex_dib = t.tex_dib;
        m.tex_w = t.tex_w;
        m.tex_h = t.tex_h;
        m.tex_file = t.tex_file;
        m.colorized = t.colorized;
    }
    Ok(Val::Material(m))
}

fn read_dib(_ar: &mut Ar, r: &mut Cur) -> L<Val> {
    let _subtype = r.u32()?;
    let length = r.u32()? as usize;
    if length > r.data.len() {
        return perr(format!("implausible dib length {length} {}", r.ctx()));
    }
    Ok(Val::Dib { data: Rc::new(r.raw(length).to_vec()) })
}

fn read_ftc(ar: &mut Ar, r: &mut Cur) -> L<Val> {
    preamble(ar, r)?;
    r.u32()?;
    let ks = r.f64s::<24>()?;
    for _ in 0..r.u32()? {
        r.f64s::<4>()?;
    }
    for _ in 0..r.u32()? {
        r.f64s::<4>()?;
    }
    let fflags = r.u32()?;
    let bflags = r.u32()?;
    let mut front = [0.0; 9];
    let mut back = [0.0; 9];
    front.copy_from_slice(&ks[0..9]);
    back.copy_from_slice(&ks[12..21]);
    Ok(Val::Ftc { front, back, front_projected: fflags & 2 != 0, back_projected: bflags & 2 != 0 })
}

fn read_camera(_ar: &mut Ar, r: &mut Cur) -> L<Val> {
    r.raw(137);
    r.u16()?;
    r.utf16()?;
    r.raw(33);
    Ok(Val::Camera)
}

fn read_thumbnail(ar: &mut Ar, r: &mut Cur) -> L<Val> {
    preamble(ar, r)?;
    ar.read_object(r, Some("CCamera"))?;
    ar.read_object(r, Some("CDib"))?;
    Ok(Val::Thumbnail)
}

fn read_image(ar: &mut Ar, r: &mut Cur) -> L<Val> {
    preamble(ar, r)?;
    let db = drawbase(ar, r)?;
    let d = ar.read_object(r, None)?;
    let xform = r.f64s::<12>()?;
    r.f64()?;
    r.utf16()?;
    let guid = hex_upper(r.raw(16));
    Ok(Val::Image { db, def: d.slot, xform, guid })
}

fn hex_upper(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02X}")).collect()
}

/// `_entity_ref`: a reference that may point forward; the slot, or `None`.
fn entity_ref(r: &mut Cur) -> L<Option<i64>> {
    let tag = r.u16()?;
    if tag == 0 {
        return Ok(None);
    }
    if tag == 0x7FFF {
        let big = r.u32()?;
        if big & 0x80000000 != 0 {
            return perr(format!("entity ref is a new object {}", r.ctx()));
        }
        return Ok(Some(big as i64));
    }
    if tag == 0xFFFF || tag & 0x8000 != 0 {
        return perr(format!("entity ref is a new object {}", r.ctx()));
    }
    Ok(Some(tag as i64))
}

fn read_relationship(ar: &mut Ar, r: &mut Cur) -> L<Val> {
    preamble(ar, r)?;
    entity_ref(r)?;
    entity_ref(r)?;
    Ok(Val::Relationship)
}

fn read_constructionline(ar: &mut Ar, r: &mut Cur) -> L<Val> {
    preamble(ar, r)?;
    let _db = drawbase(ar, r)?;
    let point = r.f64s::<3>()?;
    let direction = r.f64s::<3>()?;
    let [start_param, end_param] = r.f64s::<2>()?;
    let k = match ar.cline_tail {
        Some(k) => k,
        None => {
            let default = if ar.ver == 17 { 7 } else { 4 };
            let mut order = vec![default];
            order.extend([0usize, 4, 7].iter().copied().filter(|&c| c != default));
            let mut found = None;
            'outer: for allow_null in [false, true] {
                for &cand in &order {
                    if ar.strict_next_tag(r.pos + cand, allow_null) {
                        found = Some(cand);
                        break 'outer;
                    }
                }
            }
            let k = found.unwrap_or(default);
            ar.cline_tail = Some(k);
            k
        }
    };
    r.raw(k);
    let huge = 1e20;
    let at = |param: f64| -> Option<[f64; 3]> {
        if param.abs() >= huge {
            None
        } else {
            Some([point[0] + direction[0] * param, point[1] + direction[1] * param, point[2] + direction[2] * param])
        }
    };
    Ok(Val::CLine { point, direction, start: at(start_param), end: at(end_param) })
}

fn read_constructionpoint(ar: &mut Ar, r: &mut Cur) -> L<Val> {
    preamble(ar, r)?;
    drawbase(ar, r)?;
    let pos = r.f64s::<3>()?;
    r.f64s::<3>()?;
    r.u8()?;
    Ok(Val::CPoint { pos })
}

fn read_sectionplane(ar: &mut Ar, r: &mut Cur) -> L<Val> {
    preamble(ar, r)?;
    let db = drawbase(ar, r)?;
    let first = f64::from_le_bytes(r.data.get(r.pos..r.pos + 8).ok_or_else(|| LErr::Other("unpack".into()))?.try_into().unwrap());
    if !(first.abs() <= 1.0001) {
        ar.read_object(r, None)?;
    }
    let plane = r.f64s::<4>()?;
    let (mut name, mut label) = (String::new(), String::new());
    if r.peek(3) == STR_MARKER {
        name = r.utf16()?;
        label = r.utf16()?;
    }
    Ok(Val::SectionPlane { plane, name, label, hidden: db.hidden != 0 })
}

fn read_skfont(ar: &mut Ar, r: &mut Cur) -> L<Val> {
    ar.read_object(r, Some("CAttributeContainer"))?;
    if ar.has_pid {
        let mask = r.u8()?;
        r.raw(mask.count_ones() as usize);
    }
    r.utf16()?;
    r.raw(15);
    Ok(Val::Font)
}

fn path_ref(ar: &mut Ar, r: &mut Cur) -> L<Option<i64>> {
    let tag = r.peek_u16()?;
    let mut new = tag == 0xFFFF || (tag != 0x7FFF && tag & 0x8000 != 0);
    if tag == 0x7FFF {
        let big = r.data.get(r.pos + 2..r.pos + 6).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).ok_or_else(|| LErr::Other("unpack".into()))?;
        new = big & 0x80000000 != 0;
    }
    if new {
        let o = ar.read_object(r, None)?;
        return Ok(o.slot);
    }
    entity_ref(r)
}

fn connection_paths(ar: &mut Ar, r: &mut Cur) -> L<()> {
    entity_ref(r)?;
    for _ in 0..2 {
        let count = r.u32()?;
        if count > 1000 {
            return perr(format!("implausible dimension path length {}", r.ctx()));
        }
        for _ in 0..count {
            path_ref(ar, r)?;
        }
    }
    Ok(())
}

fn read_dimlinear(ar: &mut Ar, r: &mut Cur) -> L<Val> {
    preamble(ar, r)?;
    let db = drawbase(ar, r)?;
    let text = r.utf16()?;
    ar.read_object(r, Some("CSkFont"))?;
    let b37 = r.raw(37);
    let c1 = entity_ref(r)?;
    connection_paths(ar, r)?;
    let b32 = r.raw(32);
    let c2 = entity_ref(r)?;
    connection_paths(ar, r)?;
    let b72 = r.raw(72);
    let un = |b: &[u8], at: usize, n: usize| -> L<Vec<u8>> { b.get(at..at + n).map(|s| s.to_vec()).ok_or_else(|| LErr::Other("unpack requires a buffer".into())) };
    let offset = f64::from_le_bytes(un(b72, 52, 8)?.try_into().unwrap());
    let p3 = |v: Vec<u8>| -> [f64; 3] { [f64::from_le_bytes(v[0..8].try_into().unwrap()), f64::from_le_bytes(v[8..16].try_into().unwrap()), f64::from_le_bytes(v[16..24].try_into().unwrap())] };
    let mut a = None;
    let mut b = None;
    if u32::from_le_bytes(un(b37, 5, 4)?.try_into().unwrap()) == 1 {
        a = Some(p3(un(b37, 13, 24)?));
    }
    if u32::from_le_bytes(un(b32, 0, 4)?.try_into().unwrap()) == 1 {
        b = Some(p3(un(b32, 8, 24)?));
    }
    Ok(Val::Dim { db, text, connect: (c1, c2), offset, a, b })
}

fn read_text(ar: &mut Ar, r: &mut Cur) -> L<Val> {
    preamble(ar, r)?;
    let db = drawbase(ar, r)?;
    ar.read_object(r, Some("CSkFont"))?;
    let mut p = r.pos;
    let idx;
    loop {
        let Some(i) = find_from(r.data, &STR_MARKER, p, r.pos + 512) else {
            return perr(format!("text delimiter not found {}", r.ctx()));
        };
        if i >= 11 {
            let blk = &r.data[i - 11..i];
            let arrow = u32::from_le_bytes(blk[6..10].try_into().unwrap());
            if blk[..4] == [1, 0, 0, 0] && arrow <= 8 && blk[10] == 1 {
                idx = i;
                break;
            }
        }
        p = i + 3;
    }
    let mut point = None;
    let mut label = None;
    let mid = &r.data[r.pos.min(idx)..idx];
    let needle = [1u8, 0, 0, 0, 4, 0, 0, 0];
    if let Some(off) = mid.windows(8).position(|w| w == needle) {
        if off + 32 <= mid.len() {
            let f = |at: usize| f64::from_le_bytes(mid[at..at + 8].try_into().unwrap());
            point = Some([f(off + 8), f(off + 16), f(off + 24)]);
            let lo = off + 32 + 12;
            if lo + 24 <= mid.len() {
                let lp = [f(lo), f(lo + 8), f(lo + 16)];
                if lp.iter().any(|c| c.abs() > 1e-12) {
                    label = Some(lp);
                }
            }
        }
    }
    r.raw(idx.saturating_sub(r.pos));
    let text = r.utf16()?;
    r.raw(5);
    while r.peek(2) == [0xFF, 0x7F] {
        let val = r.data.get(r.pos + 2..r.pos + 6).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).ok_or_else(|| LErr::Other("unpack".into()))?;
        if val & 0x80000000 != 0 {
            break;
        }
        r.raw(6);
    }
    Ok(Val::Text { text, db, p: point, lp: label })
}

fn read_entity_list(ar: &mut Ar, r: &mut Cur, count: usize, owner: &str) -> L<Vec<Ent>> {
    ar.burn_stack.push(0);
    let res = read_entity_list_inner(ar, r, count, owner);
    ar.burn_stack.pop();
    res
}

fn read_entity_list_inner(ar: &mut Ar, r: &mut Cur, count: usize, owner: &str) -> L<Vec<Ent>> {
    let mut ents: Vec<Ent> = Vec::new();
    while ents.len() < count {
        let p = r.pos;
        if owner == "def" && ar.burn_stack.last().is_some_and(|&b| b != 0) {
            let z = r.data.get(p..p + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).ok_or_else(|| LErr::Other("unpack".into()))?;
            if z == 0 && r.data.get(p + 22..p + 25) == Some(&STR_MARKER[..]) {
                break;
            }
        }
        let prev = ar.in_entity_list;
        ar.in_entity_list = true;
        let res = ar.read_object(r, None);
        ar.in_entity_list = prev;
        match res {
            Ok(o) => ents.push((o.slot, o.name, o.value)),
            Err(LErr::Parse(m)) => {
                if owner == "root" {
                    r.pos = p;
                    break;
                }
                if owner == "def" && ar.burn_stack.last().is_some_and(|&b| b != 0) {
                    r.pos = p;
                    break;
                }
                return Err(LErr::Parse(m));
            }
            Err(e) => return Err(e),
        }
    }
    Ok(ents)
}

fn read_definition(ar: &mut Ar, r: &mut Cur) -> L<Val> {
    preamble(ar, r)?;
    r.raw(if ar.ver >= 17 { 22 } else { 20 });
    let nlayers = r.u32()?;
    if nlayers > 10000 {
        return perr(format!("implausible def layer count {}", r.ctx()));
    }
    let mut got = 0;
    while got < nlayers {
        if r.peek_u16()? == 0 {
            r.pos += 2;
            continue;
        }
        ar.read_object(r, Some("CLayer"))?;
        got += 1;
    }
    let mut decl = r.u16()? as u32;
    if decl == 0x7FFF {
        decl = r.u32()?;
    }
    let _ = decl;
    let mut count: Option<u32> = None;
    if ar.ver >= 20 {
        let pos = r.pos;
        count = retry_count_after_v20_filler(r, pos, 5_000_000, Some(ar));
    }
    let mut count = match count {
        Some(c) => c,
        None => {
            r.u32()?;
            r.u32()?
        }
    };
    if count > 5_000_000 || count == 0 {
        let at = r.pos.wrapping_sub(4);
        if let Some(retry) = retry_count_after_v20_filler(r, at, 5_000_000, Some(ar)) {
            count = retry;
        }
    }
    if count > 5_000_000 {
        return perr(format!("implausible def entity count {}", r.ctx()));
    }
    let ents = read_entity_list(ar, r, count as usize, "def")?;
    let mut nrel = r.u32()?;
    if nrel > 100000 {
        let at = r.pos - 4;
        if let Some(retry) = retry_count_after_v20_filler(r, at, 100_000, Some(ar)) {
            nrel = retry;
        }
    }
    if nrel > 100000 {
        return perr(format!("definition list misaligned {}", r.ctx()));
    }
    for _ in 0..nrel {
        ar.read_object(r, Some("CRelationship"))?;
    }
    r.u16()?;
    if r.peek(19).get(16..19) != Some(&STR_MARKER[..]) {
        for skip in 1..5 {
            let at = r.pos + skip;
            if r.data.get(at + 16..at + 19) == Some(&STR_MARKER[..]) {
                r.pos = at;
                break;
            }
        }
    }
    let guid = hex_upper(r.raw(16));
    let name = r.utf16()?;
    r.utf16()?;
    r.utf16()?;
    r.u32()?;
    let mut tpos: Option<usize> = None;
    let thumb_slot = ar.class_slot.get("CThumbnail").copied();
    for off in 0..96 {
        let p = r.pos + off;
        if r.data.get(p..p + 2) == Some(&[0xFF, 0xFF][..]) && r.data.get(p + 4..p + 6) == Some(&[0x0A, 0x00][..]) && r.data.get(p + 6..p + 16) == Some(&b"CThumbnail"[..]) {
            tpos = Some(p);
            break;
        }
        if thumb_slot.is_some_and(|ts| is_class_ref(r.data, p, ts)) {
            tpos = Some(p);
            break;
        }
    }
    let Some(tpos) = tpos else {
        return perr(format!("definition tail: thumbnail not found {}", r.ctx()));
    };
    let gap = r.raw(tpos - r.pos);
    let behavior = if gap.len() >= 9 { gap[gap.len() - 9] } else { 0 };
    ar.read_object(r, Some("CThumbnail"))?;
    Ok(Val::Definition { name, guid, ents, faces_camera: behavior & 1 != 0, shadows_face_sun: behavior & 2 != 0 })
}

fn read_instance(ar: &mut Ar, r: &mut Cur) -> L<Val> {
    let pre = preamble(ar, r)?;
    let db = drawbase(ar, r)?;
    let d = ar.read_object(r, Some("CComponentDefinition"))?;
    if !d.is("CComponentDefinition") {
        return perr(format!("instance definition ref is {} {}", d.name.as_deref().unwrap_or("None"), r.ctx()));
    }
    let xf = r.f64s::<13>()?;
    let name = r.utf16()?;
    let guid = if ar.ver >= 14 { hex_upper(r.raw(16)) } else { String::new() };
    Ok(Val::Instance { db, def: d.slot, xf, name, guid, attrs: pre.attrs })
}

// ---- the walk driver ---------------------------------------------------------------

/// `is_legacy(header)`.
pub fn is_legacy(header: &[u8]) -> bool {
    if !header.starts_with(&[0xFF, 0xFE, 0xFF, 0x0E]) {
        return false;
    }
    if find_from(header, b"PK\x03\x04", 0, 0x100).is_some() {
        return false;
    }
    find_from(header, b"CVersionMap", 0, 0x200).is_some()
}

/// `re.search(rb'\xff\xff..' + pack('<H', len) + name, data, re.DOTALL)`.
fn find_class_record(data: &[u8], name: &[u8]) -> Option<usize> {
    let mut needle = (name.len() as u16).to_le_bytes().to_vec();
    needle.extend_from_slice(name);
    let mut from = 0;
    while let Some(k) = find_from(data, &needle, from, data.len()) {
        if k >= 4 && data[k - 4] == 0xFF && data[k - 3] == 0xFF {
            return Some(k - 4);
        }
        from = k + 1;
    }
    None
}

struct Walked<'a> {
    ar: Ar<'a>,
    root: Vec<Ent>,
    layers: Vec<(i64, Rc<Val>)>,
    materials: Vec<(i64, Rc<Val>)>,
    end_pos: usize,
}

fn walk(data: &[u8]) -> L<Walked<'_>> {
    // re.search(rb'\{(\d+)\.', data[:0x60].replace(b'\x00', b''))
    let head: Vec<u8> = data[..data.len().min(0x60)].iter().copied().filter(|&b| b != 0).collect();
    let mut ver: Option<i32> = None;
    for i in 0..head.len() {
        if head[i] == b'{' {
            let digits: Vec<u8> = head[i + 1..].iter().copied().take_while(|b| b.is_ascii_digit()).collect();
            if !digits.is_empty() && head.get(i + 1 + digits.len()) == Some(&b'.') {
                ver = std::str::from_utf8(&digits).ok().and_then(|s| s.parse().ok());
                break;
            }
        }
    }
    let Some(ver) = ver else { return perr("no version string in header") };
    let (start, mat_count);
    if let Some(s) = find_class_record(data, b"CMaterial") {
        start = s;
        let mc = u32::from_le_bytes(data.get(start.wrapping_sub(4)..start).ok_or_else(|| LErr::Other("unpack".into()))?.try_into().unwrap());
        if mc > 100000 {
            return perr("implausible material count");
        }
        mat_count = mc;
    } else {
        let Some(lm) = find_class_record(data, b"CLayer") else { return perr("no CMaterial or CLayer class record found") };
        mat_count = 0;
        start = lm.wrapping_sub(if ver >= 17 { 9 } else { 8 });
    }
    let bases: Vec<i64> = if mat_count >= 2 { vec![bootstrap_two_materials(data, ver, start)?] } else { probe_layer_anchor_bases(data, ver, start, mat_count)? };
    let mut last: Option<LErr> = None;
    for base in bases {
        match walk_model(data, ver, start, mat_count, base) {
            Ok(w) => return Ok(w),
            Err(e) => last = Some(e),
        }
    }
    match last {
        Some(e) => Err(e),
        None => perr("no viable slot base candidate"),
    }
}

fn bootstrap_two_materials(data: &[u8], ver: i32, mat_hdr: usize) -> L<i64> {
    let mut boot = Ar::new(data, ver);
    boot.next_slot = 1 << 20;
    boot.walk_base = 1 << 20;
    let mut r = Cur::new(data, mat_hdr);
    boot.read_object(&mut r, Some("CMaterial"))?;
    let tag = r.peek_u16()?;
    if tag == 0xFFFF || tag & 0x8000 == 0 {
        return perr("cannot bootstrap the slot base");
    }
    Ok((tag & 0x7FFF) as i64)
}

fn probe_layer_anchor_bases(data: &[u8], ver: i32, start: usize, mat_count: u32) -> L<Vec<i64>> {
    let mut boot = Ar::new(data, ver);
    let b0: i64 = 1 << 20;
    boot.next_slot = b0;
    boot.walk_base = b0;
    let mut r = Cur::new(data, start);
    for _ in 0..mat_count {
        boot.read_object(&mut r, Some("CMaterial"))?;
    }
    r.u32()?;
    if ver >= 17 {
        r.u8()?;
    }
    let layer_count = r.u32()?;
    if !(1..=100000).contains(&layer_count) {
        return perr("implausible layer count in base probe");
    }
    let mut layer_slots = Vec::new();
    for _ in 0..layer_count {
        layer_slots.push(boot.read_object(&mut r, Some("CLayer"))?.slot);
    }
    let o = boot.read_object(&mut r, None)?;
    if !o.is("premodel") {
        return perr(format!("base probe: anchor resolved to {}", o.name.as_deref().unwrap_or("None")));
    }
    let s = o.slot.unwrap_or(0);
    Ok(layer_slots.into_iter().filter_map(|rel| rel.map(|rel| s - (rel - b0))).filter(|&b| 0 < b && b < b0).collect())
}

fn walk_model(data: &[u8], ver: i32, start: usize, mat_count: u32, base: i64) -> L<Walked<'_>> {
    let mut ar = Ar::new(data, ver);
    ar.next_slot = base;
    ar.walk_base = base;
    let mut r = Cur::new(data, start);
    let mut materials: Vec<(i64, Rc<Val>)> = Vec::new();
    for _ in 0..mat_count {
        let o = ar.read_object(&mut r, Some("CMaterial"))?;
        if let (Some(s), Some(v)) = (o.slot, o.value) {
            materials.push((s, v));
        }
    }
    r.u32()?;
    if ver >= 17 {
        r.u8()?;
    }
    let layer_count = r.u32()?;
    if layer_count > 100000 {
        return perr("implausible layer count");
    }
    let mut layers: Vec<(i64, Rc<Val>)> = Vec::new();
    while (layers.len() as u32) < layer_count {
        let tag = r.peek_u16()?;
        if tag == 0 {
            r.pos += 2;
            continue;
        }
        if tag != 0xFFFF && tag & 0x8000 == 0 {
            break;
        }
        let o = ar.read_object(&mut r, Some("CLayer"))?;
        match (o.slot, o.value) {
            (Some(s), Some(v)) => layers.push((s, v)),
            _ => continue,
        }
    }
    let lay_cls = ar.class_slot.get("CLayer").copied();
    loop {
        let tag = r.peek_u16()?;
        if tag == 0 {
            r.pos += 2;
            continue;
        }
        if lay_cls.is_some_and(|c| tag as i64 == (0x8000 | c)) {
            let o = ar.read_object(&mut r, Some("CLayer"))?;
            if let (Some(s), Some(v)) = (o.slot, o.value) {
                layers.push((s, v));
            }
            continue;
        }
        break;
    }
    let anchor = ar.read_object(&mut r, None)?;
    if !anchor.is("CLayer") {
        return perr(format!("definition-list anchor is {}, not a layer", anchor.name.as_deref().unwrap_or("None")));
    }
    let mut def_count = r.u32()?;
    if def_count > 1_000_000 {
        let at = r.pos - 4;
        if let Some(retry) = retry_count_after_v20_filler(&mut r, at, 1_000_000, Some(&ar)) {
            def_count = retry;
        }
    }
    if def_count > 1_000_000 {
        return perr("implausible definition count");
    }
    for _ in 0..def_count {
        ar.read_object(&mut r, Some("CComponentDefinition"))?;
    }
    let def_cls = ar.class_slot.get("CComponentDefinition").copied();
    loop {
        let tag = r.peek_u16()?;
        let mut is_def = def_cls.is_some_and(|c| tag as i64 == (0x8000 | c));
        if !is_def && tag == 0xFFFF && r.peek(26).get(6..26) == Some(&b"CComponentDefinition"[..]) {
            is_def = true;
        }
        if !is_def {
            break;
        }
        ar.read_object(&mut r, None)?;
    }
    let mut root_count = r.u32()?;
    if root_count > 5_000_000 {
        let at = r.pos - 4;
        if let Some(retry) = retry_count_after_v20_filler(&mut r, at, 5_000_000, Some(&ar)) {
            root_count = retry;
        }
    }
    if root_count > 5_000_000 {
        return perr("implausible root entity count");
    }
    let root = read_entity_list(&mut ar, &mut r, root_count as usize, "root")?;
    let end_pos = r.pos;
    Ok(Walked { ar, root, layers, materials, end_pos })
}

// ---- pages ---------------------------------------------------------------------------

const PAGE_ALLOWED_FLAGS: [u32; 2] = [0xc00, 0xc20];
const PAGE_HIDDEN_LAYERS_BIT: u32 = 0x20;

/// `_skip_typed_ref`.
fn skip_typed_ref(ar: &mut Ar, r: &mut Cur, cls: &str, reader: fn(&mut Ar, &mut Cur) -> L<Val>) -> L<()> {
    let tag = r.peek_u16()?;
    if tag == 0 {
        r.u16()?;
        return Ok(());
    }
    if tag == 0xFFFF {
        r.u16()?;
        let _schema = r.u16()?;
        let namelen = r.u16()? as usize;
        if namelen > 40 {
            return perr("implausible class name length");
        }
        let raw = r.raw(namelen);
        if !raw.is_ascii() {
            return oerr("'ascii' codec can't decode byte");
        }
        let name = std::str::from_utf8(raw).unwrap();
        if name != cls {
            return perr(format!("expected {cls} decl, got {name}"));
        }
        if !ar.class_slot.contains_key(cls) {
            let s = ar.alloc(Slot::Class { name: Rc::from(cls) });
            ar.class_slot.insert(Rc::from(cls), s);
        }
        reader(ar, r)?;
        return Ok(());
    }
    if tag & 0x8000 != 0 {
        let cslot = (tag & 0x7FFF) as i64;
        ar.slots.entry(cslot).or_insert_with(|| Slot::Class { name: Rc::from(cls) });
        match ar.slots.get(&cslot) {
            Some(Slot::Class { name }) if &**name == cls => {}
            _ => return perr(format!("unexpected {cls} class-ref")),
        }
        r.u16()?;
        reader(ar, r)?;
        return Ok(());
    }
    r.u16()?;
    Ok(())
}

/// `_scan_pages(data, ar)`.
fn scan_pages(data: &[u8], ar: &mut Ar, from: usize) -> Vec<RawPage> {
    let mut pages = Vec::new();
    let mut pos = from;
    let n = data.len();
    while let Some(idx) = find_from(data, &STR_MARKER, pos, n) {
        pos = idx + 1;
        let nlen_pos = idx + 3;
        if nlen_pos >= n {
            break;
        }
        let nlen = data[nlen_pos] as usize;
        if nlen == 0 || nlen > 40 {
            continue;
        }
        let name_start = nlen_pos + 1;
        let name_end = name_start + 2 * nlen;
        if name_end + 7 > n {
            continue;
        }
        if data[name_end..name_end + 3] != STR_MARKER || data[name_end + 3] != 0 {
            continue;
        }
        let flags_pos = name_end + 4;
        let flags = u32::from_le_bytes(data[flags_pos..flags_pos + 4].try_into().unwrap());
        if !PAGE_ALLOWED_FLAGS.contains(&flags) {
            continue;
        }
        let name = utf16le_replace(&data[name_start..name_end]);
        let mut r = Cur::new(data, flags_pos + 4);
        let attempt = (|| -> L<Vec<i64>> {
            skip_typed_ref(ar, &mut r, "CCamera", read_camera)?;
            skip_typed_ref(ar, &mut r, "CDib", read_dib)?;
            let mut hidden_ids = Vec::new();
            if flags & PAGE_HIDDEN_LAYERS_BIT != 0 {
                loop {
                    let save = r.pos;
                    let v = r.u16()? as i64;
                    match ar.slots.get(&v) {
                        Some(Slot::Obj { name, .. }) if &**name == "CLayer" => hidden_ids.push(v),
                        _ => {
                            r.pos = save;
                            break;
                        }
                    }
                }
                if r.u32()? != 1 {
                    return perr("unexpected hidden-layers marker");
                }
            }
            Ok(hidden_ids)
        })();
        if let Ok(ids) = attempt {
            pages.push(RawPage { name, eye: None, target: None, up: None, fov: 35.0, parallel: false, ortho_height: 0.0, hidden_layer_ids: ids });
        }
    }
    pages
}

// ---- adapter to the full_parse shape ---------------------------------------------------

fn lerr_to_other(e: &LErr) -> String {
    e.message().to_string()
}

fn add_edge(b: &mut Builder, slot: i64, e: &Val, slots: &HashMap<i64, Slot>) -> L<()> {
    if b.edges.contains_key(&slot) {
        return Ok(());
    }
    let Val::Edge { db, curve, v1, v2, .. } = e else { return oerr("KeyError: 'v1'") };
    for vs in [v1, v2].into_iter().flatten() {
        if let Some(Slot::Obj { value: Some(v), .. }) = slots.get(vs) {
            if !b.vertices.contains_key(vs) {
                match &**v {
                    Val::Vertex(xyz) => b.vertices.insert(*vs, *xyz),
                    _ => return oerr("KeyError: 'xyz'"),
                }
            }
        }
    }
    b.edges.insert(slot, (*v1, *v2));
    let flags = (if db.soft != 0 { 0x08 } else { 0 }) | (if db.smooth != 0 { 0x10 } else { 0 }) | (if db.hidden != 0 { 0x01 } else { 0 });
    if flags != 0 {
        b.edge_flags.insert(slot, flags);
    }
    if db.layer != 0 {
        b.edge_layers.insert(slot, db.layer as i64);
    }
    if let Some(c) = curve {
        b.edge_curves.insert(slot, *c);
    }
    Ok(())
}

fn fill_builder(b: &mut Builder, ents: &[Ent], slots: &HashMap<i64, Slot>) -> L<()> {
    for (s, _n, v) in ents {
        let Some(v) = v else { continue };
        let Some(s) = *s else { continue };
        match &**v {
            Val::Edge { .. } => add_edge(b, s, v, slots)?,
            Val::Face { db, plane, loops, back_mat, attrs } => {
                let mut out_loops: Vec<Vec<CoEdge>> = Vec::new();
                for lp in loops {
                    let Some(lp) = lp else { return oerr("TypeError: 'NoneType' object is not subscriptable") };
                    let Val::Loop { uses } = &**lp else { return oerr("KeyError: 'uses'") };
                    let mut lp_out = Vec::new();
                    for u in uses {
                        let Some(u) = u else { return oerr("TypeError: 'NoneType' object is not subscriptable") };
                        let Val::EdgeUse { edge, sense } = &**u else { return oerr("KeyError: 'edge'") };
                        let es = *edge;
                        let ent = es.and_then(|e| slots.get(&e));
                        let Some(Slot::Obj { value: Some(ev), .. }) = ent else { continue };
                        let es = es.unwrap();
                        add_edge(b, es, ev, slots)?;
                        lp_out.push((es, if *sense != 0 { -1 } else { 1 }));
                    }
                    out_loops.push(lp_out);
                }
                let mut face = RawFace {
                    loops: out_loops,
                    normal: [plane[0], plane[1], plane[2]],
                    material_id: if db.mat != 0 { Some(db.mat as i64) } else { None },
                    back_material_id: if *back_mat != 0 { Some(*back_mat as i64) } else { None },
                    uv_transform: None,
                    uv_transform_back: None,
                    uv_projected: false,
                    uv_projected_back: false,
                    hidden: db.hidden != 0,
                    layer: Some(db.layer as i64),
                };
                if let Some(a) = attrs {
                    if let Val::Attrs { children } = &**a {
                        for (_cn, cv) in children {
                            if let Some(cv) = cv {
                                if let Val::Ftc { front, back, front_projected, back_projected } = &**cv {
                                    face.uv_transform = Some(*front);
                                    face.uv_transform_back = Some(*back);
                                    face.uv_projected = *front_projected;
                                    face.uv_projected_back = *back_projected;
                                }
                            }
                        }
                    }
                }
                b.faces.insert(s, face);
            }
            Val::Instance { db, def, xf, name, guid, attrs } => {
                let dicts = attr_dicts(attrs);
                let properties = dicts.iter().find(|(k, _)| k == "dynamic_attributes").map(|(_, e)| e.iter().map(|(k, v)| (k.clone(), v.stringify())).collect()).unwrap_or_default();
                b.instances.push(RawInst {
                    name: Some(name.clone()),
                    ref_idx: *def,
                    ref_guid: Some(String::new()),
                    matrix: xf.to_vec(),
                    material_id: if db.mat != 0 { Some(db.mat as i64) } else { None },
                    hidden: db.hidden != 0,
                    layer_id: if db.layer != 0 { Some(db.layer as i64) } else { None },
                    properties,
                    attribute_dictionaries: dicts,
                });
                let _ = guid;
            }
            Val::Image { db, def, xform, guid } => {
                b.instances.push(RawInst {
                    name: Some(String::new()),
                    ref_idx: *def,
                    ref_guid: Some(guid.clone()),
                    matrix: xform.to_vec(),
                    material_id: if db.mat != 0 { Some(db.mat as i64) } else { None },
                    hidden: db.hidden != 0,
                    layer_id: if db.layer != 0 { Some(db.layer as i64) } else { None },
                    properties: Vec::new(),
                    attribute_dictionaries: Vec::new(),
                });
            }
            Val::SectionPlane { plane, name, label, hidden } => b.section_planes.push(SectionPlane { plane: *plane, name: name.clone(), label: label.clone(), hidden: *hidden }),
            Val::Text { text, db, p, lp } => b.texts.push(TextEntity { text: text.clone(), hidden: db.hidden != 0, point: *p, label_point: *lp }),
            Val::Dim { db, text, connect, offset, a, b: bpt } => {
                let mut pts = [*a, *bpt];
                for (i, cs) in [connect.0, connect.1].iter().enumerate() {
                    if pts[i].is_some() {
                        continue;
                    }
                    if let Some(Slot::Obj { value: Some(v), .. }) = cs.and_then(|c| slots.get(&c)) {
                        if let Val::Vertex(xyz) = &**v {
                            pts[i] = Some(*xyz);
                        }
                    }
                }
                b.dimensions.push(Dimension { text: text.clone(), hidden: db.hidden != 0, a: pts[0], b: pts[1], offset: *offset, plane_x: None, normal: None });
            }
            Val::CLine { point, direction, start, end } => b.construction_lines.push(ConstructionLine { point: *point, direction: *direction, start: *start, end: *end }),
            Val::CPoint { pos } => b.construction_points.push(ConstructionPoint { position: *pos }),
            _ => {}
        }
    }
    Ok(())
}

fn attr_dicts(attrs: &Option<Rc<Val>>) -> AttrDict {
    let mut out: AttrDict = Vec::new();
    if let Some(a) = attrs {
        if let Val::Attrs { children } = &**a {
            for (_n, v) in children {
                if let Some(v) = v {
                    if let Val::Dict { name, entries } = &**v {
                        if name.is_empty() {
                            continue;
                        }
                        match out.iter_mut().find(|(k, _)| k == name) {
                            Some(slot) => slot.1 = entries.clone(),
                            None => out.push((name.clone(), entries.clone())),
                        }
                    }
                }
            }
        }
    }
    out
}

/// `full_parse_legacy`.
pub fn full_parse_legacy(data: &[u8]) -> Result<RawParse> {
    let mut version = "unknown".to_string();
    if let Some(second) = find_from(data, &STR_MARKER, 4, data.len()) {
        let text = utf16le_ignore(&data[second + 4..(second + 100).min(data.len())]);
        if let (Some(a), Some(b)) = (text.find('{'), text.find('}')) {
            if b >= a {
                version = text[a..=b].to_string();
            }
        }
    }
    let mut walked = walk(data).map_err(|e| SkpError::new(format!("legacy .skp parse failed: {}", lerr_to_other(&e)), "legacy_walk"))?;
    let mut rp = RawParse { version, ..Default::default() };

    // images are definitions placed by a CImage
    let mut image_def_ids: std::collections::HashSet<i64> = std::collections::HashSet::new();
    for s in &walked.ar.alloc_order {
        if let Some(Slot::Obj { value: Some(v), .. }) = walked.ar.slots.get(s) {
            if let Val::Image { def: Some(d), .. } = &**v {
                image_def_ids.insert(*d);
            }
        }
    }
    for (s, v) in &walked.materials {
        let Val::Material(m) = &**v else { continue };
        let trans = if m.use_opacity != 0 { (1.0 - m.opacity).clamp(0.0, 1.0) } else { 1.0 };
        let mut texture = None;
        if m.textured {
            let dib = m.tex_dib.and_then(|d| walked.ar.slots.get(&d));
            let tex_data = match dib {
                Some(Slot::Obj { value: Some(v), .. }) => match &**v {
                    Val::Dib { data } => Some(data.clone()),
                    _ => None,
                },
                _ => None,
            };
            let ext = if tex_data.as_ref().is_some_and(|d| d.starts_with(b"\x89PNG")) { ".png" } else { ".jpg" };
            let fname = if m.tex_file.is_empty() { format!("{}{}", m.name, ext) } else { m.tex_file.clone() };
            texture = Some(Texture { filename: fname, width: m.tex_w, height: m.tex_h, data: tex_data });
        }
        let si = rp.material_store.len();
        rp.material_store.push(RawMaterial {
            name: m.name.clone(),
            color: [m.rgba[0] as i64, m.rgba[1] as i64, m.rgba[2] as i64, m.rgba[3] as i64],
            transparency: trans,
            colorized: m.colorized,
            colorize_type: if m.colorized { 1 } else { 0 },
            texture,
        });
        rp.materials.insert(m.name.clone(), si);
        rp.material_id_to_name.insert(*s, m.name.clone());
    }
    for (s, v) in &walked.layers {
        let Val::Layer { name, hidden, rgba } = &**v else { continue };
        rp.layer_colors.insert(name.clone(), [rgba[0] as i64, rgba[1] as i64, rgba[2] as i64]);
        rp.layer_hidden.insert(name.clone(), *hidden != 0);
        rp.layer_id_to_name.insert(*s, name.clone());
    }
    if !rp.layer_colors.contains_key(&"Layer0".to_string()) {
        rp.layer_colors.insert("Layer0".into(), [136, 136, 136]);
    }
    rp.layer_hidden.entry("Layer0".into()).or_insert(false);

    let order: Vec<i64> = walked.ar.alloc_order.clone();
    for s in order {
        let Some(Slot::Obj { name, value: Some(v) }) = walked.ar.slots.get(&s) else { continue };
        if &**name != "CComponentDefinition" {
            continue;
        }
        let Val::Definition { name: dname, guid, ents, faces_camera, shadows_face_sun } = &**v else { continue };
        let mut b = Builder::default();
        fill_builder(&mut b, ents, &walked.ar.slots).map_err(|e| {
            let mut err = SkpError::new(format!("Failed while building component definitions: {}", lerr_to_other(&e)), "legacy_defs");
            err.definition_id = Some(s);
            err
        })?;
        rp.defs.insert(
            Some(s),
            RawDef { guid: guid.clone(), name: Some(dname.clone()), always_faces_camera: *faces_camera, shadows_face_sun: *shadows_face_sun, is_image: image_def_ids.contains(&s), builder: b },
        );
    }
    let mut root = Builder::default();
    fill_builder(&mut root, &walked.root, &walked.ar.slots).map_err(|e| SkpError::new(format!("legacy .skp parse failed: {}", lerr_to_other(&e)), "legacy_walk"))?;
    rp.dimensions = root.dimensions.iter().filter(|d| d.a.is_some() && d.b.is_some()).cloned().collect();
    rp.root = root;
    rp.pages = scan_pages(data, &mut walked.ar, walked.end_pos);
    Ok(rp)
}
