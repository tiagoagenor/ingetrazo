// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Section planes (`core/section.py`): an origin + a unit normal, a name and
//! a symbol. The active plane cuts the display; the cut is display-only
//! scaffolding, never topology.
//!
//! Ported here: the plane itself, its side test, its uid counter, the fill
//! it resolves to, the symbol the next plane takes and the `.igz` form. Not
//! ported: the geometry of the cut (`cut_chords`, `fill_triangles`,
//! `hatch_segments`), which works on the mesh and belongs with the section
//! renderer.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::{Hypot, V3};

use crate::pyconv::{err, get, index, number, object, py_float, py_int, qvec3, truthy, v3_value, PyErr, PyResult};

/// How a cut can be filled: solid, hatched at 45°, cross-hatched, or not.
pub const FILL_PATTERNS: [&str; 4] = ["solid", "hatch", "cross", "none"];
/// Hatch spacing when neither the plane nor anything else says: 5 cm.
pub const DEFAULT_HATCH_SPACING: f64 = 0.05;

/// `core.section._last_uid` and `_take_uid`: the module-level counter that
/// hands out stable plane identities. It is an explicit value here so two
/// documents (or two tests) never share one by accident.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UidCounter {
    pub last: i64,
}

impl UidCounter {
    /// A fresh uid, or `wanted` (from a loaded file) while keeping the
    /// counter ahead so later fresh uids never collide.
    pub fn take(&mut self, wanted: Option<i64>) -> i64 {
        match wanted {
            None => {
                self.last += 1;
                self.last
            }
            Some(w) => {
                if w > self.last {
                    self.last = w;
                }
                w
            }
        }
    }
}

/// The letter the next section plane takes: A, B, C… then AA, AB. `symbols`
/// are the existing planes' symbols (`None` = no symbol).
pub fn next_symbol(symbols: &[Option<&str>]) -> String {
    let taken: Vec<String> = symbols.iter().map(|s| s.unwrap_or("").trim().to_uppercase()).collect();
    let mut i: i64 = 0;
    loop {
        let (mut n, mut label) = (i, String::new());
        loop {
            label.insert(0, (b'A' + (n % 26) as u8) as char);
            n = n / 26 - 1;
            if n < 0 {
                break;
            }
        }
        if !taken.contains(&label) {
            return label;
        }
        i += 1;
    }
}

/// `effective_fill`: `(pattern, color, spacing)` the cut of a plane is
/// filled with — the plane's own settings over the style's. `fill` is the
/// plane's `fill` (a dict, or `None`); the style's Section Fill switch is
/// the master: off, nothing is filled. The pattern is whatever the fill
/// holds under `"pattern"` (a name, normally).
pub fn effective_fill(fill: Option<&Value>, section_fill: bool, style_color: [f64; 3]) -> PyResult<(Value, Vec<f64>, f64)> {
    let own = match fill {
        Some(v) if truthy(v) => v,
        _ => &Value::Object(Vec::new()),
    };
    let pattern = if !section_fill {
        Value::Str("none".into())
    } else {
        get(own, "pattern")?.cloned().unwrap_or_else(|| Value::Str("solid".into()))
    };
    // tuple(own.get("color") or style.section_fill_color)
    let color = match get(own, "color")? {
        Some(c) if truthy(c) => match c {
            Value::Array(items) => items.iter().map(number).collect::<PyResult<Vec<f64>>>()?,
            Value::Str(_) | Value::Object(_) => return err("NotModelled", "color is not a list of numbers"),
            _ => return err("TypeError", "object is not iterable"),
        },
        _ => style_color.to_vec(),
    };
    // float(own.get("spacing") or DEFAULT_HATCH_SPACING)
    let spacing = match get(own, "spacing")? {
        Some(s) if truthy(s) => py_float(s)?,
        _ => DEFAULT_HATCH_SPACING,
    };
    Ok((pattern, color, spacing))
}

/// Origin + unit normal (+ name/symbol). `active` = the cutting one;
/// `hidden` = its frame is not drawn (nor picked) — the cut, if it is the
/// active one, stays.
#[derive(Clone, Debug, PartialEq)]
pub struct SectionPlane {
    pub point: V3,
    pub normal: V3,
    pub name: String,
    pub symbol: String,
    pub active: bool,
    /// Stable identity for scenes (.igz SavedView references).
    pub uid: i64,
    pub hidden: bool,
    /// This cut's own fill, over the display style's: a dict with each of
    /// `"pattern"`, `"color"`, `"spacing"` optional. `None`: the style's.
    pub fill: Option<Value>,
}

impl SectionPlane {
    /// `SectionPlane(point, normal, name, symbol, active, uid, hidden)`: a
    /// null normal becomes +Z, the rest is normalised; `uid` is taken from
    /// the counter.
    pub fn new(
        point: V3,
        normal: V3,
        name: &str,
        symbol: &str,
        active: bool,
        uid: Option<i64>,
        hidden: bool,
        uids: &mut UidCounter,
        h: Hypot,
    ) -> SectionPlane {
        let n = if (normal.length(h) as f64) < 1e-12 { V3(0.0, 0.0, 1.0) } else { normal };
        SectionPlane {
            point,
            normal: n.normalized(h),
            name: name.to_string(),
            symbol: symbol.to_string(),
            active,
            hidden,
            fill: None,
            uid: uids.take(uid),
        }
    }

    /// Reverse: the cut hides the other side.
    pub fn flip(&mut self) {
        self.normal = self.normal.neg();
    }

    /// Signed distance: > 0 is the HIDDEN side (the normal's side).
    pub fn side(&self, p: V3) -> f32 {
        self.normal.dot(p.sub(self.point))
    }

    pub fn to_dict(&self) -> Value {
        let mut items = vec![("point", v3_value(self.point)), ("normal", v3_value(self.normal)), ("uid", Value::int(self.uid))];
        if !self.name.is_empty() {
            items.push(("name", Value::Str(self.name.clone())));
        }
        if !self.symbol.is_empty() {
            items.push(("symbol", Value::Str(self.symbol.clone())));
        }
        if self.active {
            items.push(("active", Value::Bool(true)));
        }
        if self.hidden {
            items.push(("hidden", Value::Bool(true))); // older readers ignore it
        }
        if let Some(Value::Object(entries)) = self.fill.as_ref().filter(|f| truthy(f)) {
            // {k: (list(v) if k == "color" else v)}: a tuple becomes a list.
            items.push(("fill", Value::Object(entries.clone())));
        }
        object(items)
    }

    /// `SectionPlane.from_dict`. `name` and `symbol` that are not text are
    /// `NotModelled`.
    pub fn from_dict(raw: &Value, uids: &mut UidCounter, h: Hypot) -> PyResult<SectionPlane> {
        let point = qvec3(index(raw, "point")?)?;
        let normal = qvec3(index(raw, "normal")?)?;
        let text = |key: &str| -> PyResult<String> {
            match get(raw, key)? {
                None => Ok(String::new()),
                Some(Value::Str(s)) => Ok(s.clone()),
                Some(v) if !truthy(v) => Ok(String::new()),
                Some(_) => Err(PyErr { kind: "NotModelled", msg: format!("{key} is not text") }),
            }
        };
        let name = text("name")?;
        let symbol = text("symbol")?;
        let active = get(raw, "active")?.is_some_and(truthy);
        let hidden = get(raw, "hidden")?.is_some_and(truthy);
        let wanted = match get(raw, "uid")? {
            None | Some(Value::Null) => None,
            Some(v) => Some(py_int(v)?),
        };
        let mut sp = SectionPlane::new(point, normal, &name, &symbol, active, wanted, hidden, uids, h);
        if let Some(fill @ Value::Object(_)) = get(raw, "fill")? {
            let mut clean: Vec<(String, Value)> = Vec::new();
            if let Some(Value::Str(p)) = fill.get("pattern") {
                if FILL_PATTERNS.contains(&p.as_str()) {
                    clean.push(("pattern".into(), Value::Str(p.clone())));
                }
            }
            if let Some(Value::Array(c)) = fill.get("color") {
                if c.len() >= 3 {
                    let rgb = c[..3].iter().map(|x| py_float(x).map(Value::Float)).collect::<PyResult<Vec<Value>>>()?;
                    clean.push(("color".into(), Value::Array(rgb)));
                }
            }
            // isinstance(spacing, (int, float)) and spacing > 0 (a bool is an int)
            let positive = match fill.get("spacing") {
                Some(Value::Float(x)) if *x > 0.0 => Some(*x),
                Some(v @ Value::Int(s)) if !s.starts_with('-') && s != "0" => Some(py_float(v)?),
                Some(Value::Bool(true)) => Some(1.0),
                _ => None,
            };
            if let Some(x) = positive {
                clean.push(("spacing".into(), Value::Float(x)));
            }
            sp.fill = if clean.is_empty() { None } else { Some(Value::Object(clean)) };
        }
        Ok(sp)
    }
}
