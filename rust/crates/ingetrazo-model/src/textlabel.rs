// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Leader text (`core/textlabel.py`): a label pointing at the model. It
//! records the world `anchor` it points at and an `offset` to where the
//! label floats; like a dimension it is an annotation, not geometry.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::V3;

use crate::pyconv::{get, object, qvec3, text_or_none, v3_value, PyErr, PyResult};

#[derive(Clone, Debug, PartialEq)]
pub struct TextLabel {
    pub anchor: V3,
    /// Displacement from the anchor to the label.
    pub offset: V3,
    pub text: String,
    /// Layer (tag) the label lives on; `None` = default layer.
    pub layer: Option<String>,
}

impl TextLabel {
    /// Where the label text floats (anchor + offset).
    pub fn position(&self) -> V3 {
        self.anchor.add(self.offset)
    }

    pub fn to_dict(&self) -> Value {
        let mut items = vec![("anchor", v3_value(self.anchor)), ("offset", v3_value(self.offset)), ("text", Value::Str(self.text.clone()))];
        if let Some(layer) = self.layer.as_ref().filter(|l| !l.is_empty()) {
            items.push(("layer", Value::Str(layer.clone())));
        }
        object(items)
    }

    /// `TextLabel.from_dict`. A `text` that is not a string (Python keeps
    /// whatever object it is) is `NotModelled`.
    pub fn from_dict(data: &Value) -> PyResult<TextLabel> {
        let zero = Value::Array(vec![Value::int(0), Value::int(0), Value::int(0)]);
        let anchor = qvec3(get(data, "anchor")?.unwrap_or(&zero))?;
        let offset = qvec3(get(data, "offset")?.unwrap_or(&zero))?;
        let text = match get(data, "text")? {
            None => String::new(),
            Some(Value::Str(s)) => s.clone(),
            Some(_) => return Err(PyErr { kind: "NotModelled", msg: "text is not a string".into() }),
        };
        let layer = text_or_none(get(data, "layer")?)?;
        Ok(TextLabel { anchor, offset, text, layer })
    }
}
