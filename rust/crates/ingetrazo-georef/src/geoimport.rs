// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `georef/geoimport.py`: import georeferenced alignments from KML / KMZ and
//! GeoJSON as plain `(lat, lon)` features. Pure parsing; the caller projects
//! them through the scene datum into [`crate::geopath::GeoPath`]s.
//!
//! KML needs an XML reader (Python's `xml.etree`): [`xml`] is a small
//! well-formedness-checking one covering what KML uses (namespaces, entities,
//! CDATA, comments, processing instructions, the XML declaration, simple
//! internal-subset entities). Malformed input is `[]`, as Python's
//! `ParseError` is.

use std::io::Read;
use std::path::Path;

use ingetrazo_formats::pyjson::{loads, Value};

use crate::pyconv::{
    err, float_str, get, is_py_space, py_split_ws, py_str, py_strip, truthy, unpack, PyResult,
};

/// `GeoFeature`: ordered `(lat, lon)` points, open/closed, a name.
#[derive(Clone, Debug, PartialEq)]
pub struct GeoFeature {
    pub points: Vec<(f64, f64)>,
    pub closed: bool,
    pub name: String,
}

/// `_drop_closing_dup`: a closed ring repeats its first vertex last; drop it.
fn drop_closing_dup(mut points: Vec<(f64, f64)>, closed: bool) -> Vec<(f64, f64)> {
    if closed && points.len() > 1 {
        let (a, b) = (points[0], points[points.len() - 1]);
        if (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9 {
            points.pop();
        }
    }
    points
}

// ---- A small XML reader ------------------------------------------------------

pub mod xml {
    //! Enough of `xml.etree.ElementTree.fromstring` for KML: the tree in
    //! document order with local names, leading text and children.

    use std::collections::HashMap;

    /// An element: local name (namespace stripped), the text before its first
    /// child, and its children.
    #[derive(Clone, Debug, Default)]
    pub struct Elem {
        pub local: String,
        pub text: String,
        pub children: Vec<usize>,
    }

    /// The parsed tree; `elems[0]` is the root.
    #[derive(Clone, Debug, Default)]
    pub struct Tree {
        pub elems: Vec<Elem>,
    }

    impl Tree {
        /// `root.iter()`: the element and its descendants in document order.
        pub fn iter(&self, from: usize) -> Vec<usize> {
            let mut out = Vec::new();
            let mut stack = vec![from];
            while let Some(i) = stack.pop() {
                out.push(i);
                for &c in self.elems[i].children.iter().rev() {
                    stack.push(c);
                }
            }
            out
        }
    }

    fn valid_char(c: char) -> bool {
        matches!(c, '\t' | '\n' | '\r' | '\u{20}'..='\u{d7ff}' | '\u{e000}'..='\u{fffd}' | '\u{10000}'..='\u{10ffff}')
    }

    fn name_start(c: char) -> bool {
        matches!(c,
            ':' | 'A'..='Z' | '_' | 'a'..='z' | '\u{c0}'..='\u{d6}' | '\u{d8}'..='\u{f6}' | '\u{f8}'..='\u{2ff}'
            | '\u{370}'..='\u{37d}' | '\u{37f}'..='\u{1fff}' | '\u{200c}'..='\u{200d}' | '\u{2070}'..='\u{218f}'
            | '\u{2c00}'..='\u{2fef}' | '\u{3001}'..='\u{d7ff}' | '\u{f900}'..='\u{fdcf}' | '\u{fdf0}'..='\u{fffd}'
            | '\u{10000}'..='\u{effff}')
    }

    fn name_char(c: char) -> bool {
        name_start(c) || matches!(c, '-' | '.' | '0'..='9' | '\u{b7}' | '\u{300}'..='\u{36f}' | '\u{203f}'..='\u{2040}')
    }

    fn is_ws(c: char) -> bool {
        matches!(c, ' ' | '\t' | '\n' | '\r')
    }

    struct P<'a> {
        s: &'a str,
        pos: usize,
        entities: HashMap<String, String>,
    }

    type R<T> = Result<T, String>;

    fn bad<T>(msg: &str) -> R<T> {
        Err(msg.to_string())
    }

    impl<'a> P<'a> {
        fn rest(&self) -> &'a str {
            &self.s[self.pos..]
        }
        fn starts(&self, t: &str) -> bool {
            self.rest().starts_with(t)
        }
        fn peek(&self) -> Option<char> {
            self.rest().chars().next()
        }
        fn skip_ws(&mut self) -> bool {
            let before = self.pos;
            while let Some(c) = self.peek() {
                if is_ws(c) {
                    self.pos += c.len_utf8();
                } else {
                    break;
                }
            }
            self.pos > before
        }

        fn name(&mut self) -> R<&'a str> {
            let start = self.pos;
            match self.peek() {
                Some(c) if name_start(c) => self.pos += c.len_utf8(),
                _ => return bad("not a name"),
            }
            while let Some(c) = self.peek() {
                if name_char(c) {
                    self.pos += c.len_utf8();
                } else {
                    break;
                }
            }
            Ok(&self.s[start..self.pos])
        }

        fn comment(&mut self) -> R<()> {
            // at "<!--"
            self.pos += 4;
            let Some(i) = self.rest().find("-->") else { return bad("unterminated comment") };
            let body = &self.rest()[..i];
            if body.contains("--") || body.ends_with('-') {
                return bad("-- in comment");
            }
            if !body.chars().all(valid_char) {
                return bad("invalid character");
            }
            self.pos += i + 3;
            Ok(())
        }

        fn pi(&mut self) -> R<()> {
            // at "<?"
            self.pos += 2;
            let target = self.name()?;
            if target.eq_ignore_ascii_case("xml") || target.contains(':') {
                return bad("reserved processing instruction target");
            }
            let Some(i) = self.rest().find("?>") else { return bad("unterminated PI") };
            let body = &self.rest()[..i];
            if !body.is_empty() && !body.starts_with(is_ws) {
                return bad("PI target must be followed by whitespace");
            }
            if !body.chars().all(valid_char) {
                return bad("invalid character");
            }
            self.pos += i + 2;
            Ok(())
        }

        fn xml_decl(&mut self) -> R<()> {
            // at "<?xml" followed by whitespace
            self.pos += 5;
            let mut seen_version = false;
            let mut stage = 0; // 0 version, 1 encoding, 2 standalone, 3 done
            loop {
                let had_ws = self.skip_ws();
                if self.starts("?>") {
                    self.pos += 2;
                    break;
                }
                if !had_ws {
                    return bad("bad XML declaration");
                }
                let n = self.name()?;
                self.skip_ws();
                if self.peek() != Some('=') {
                    return bad("bad XML declaration");
                }
                self.pos += 1;
                self.skip_ws();
                let q = match self.peek() {
                    Some(q @ ('"' | '\'')) => q,
                    _ => return bad("bad XML declaration"),
                };
                self.pos += 1;
                let Some(i) = self.rest().find(q) else { return bad("bad XML declaration") };
                let val = &self.rest()[..i];
                self.pos += i + 1;
                match n {
                    "version" if stage == 0 => {
                        if val.is_empty() || !val.chars().all(|c| c.is_ascii_digit() || c == '.') {
                            return bad("bad version");
                        }
                        seen_version = true;
                        stage = 1;
                    }
                    "encoding" if stage <= 1 && seen_version => {
                        if val.is_empty() {
                            return bad("bad encoding");
                        }
                        stage = 2;
                    }
                    "standalone" if stage <= 2 && seen_version => {
                        if val != "yes" && val != "no" {
                            return bad("bad standalone");
                        }
                        stage = 3;
                    }
                    _ => return bad("bad XML declaration"),
                }
            }
            if !seen_version {
                return bad("no version");
            }
            Ok(())
        }

        /// `&name;` or `&#N;` at `self.pos` (on the `&`), appended to `out`.
        fn reference(&mut self, out: &mut String, in_attr: bool) -> R<()> {
            self.pos += 1;
            let Some(i) = self.rest().find(';') else { return bad("unterminated reference") };
            let body = &self.rest()[..i];
            self.pos += i + 1;
            if let Some(num) = body.strip_prefix('#') {
                let code = if let Some(h) = num.strip_prefix('x') {
                    if h.is_empty() || !h.chars().all(|c| c.is_ascii_hexdigit()) {
                        return bad("bad character reference");
                    }
                    u32::from_str_radix(h, 16).ok()
                } else {
                    if num.is_empty() || !num.chars().all(|c| c.is_ascii_digit()) {
                        return bad("bad character reference");
                    }
                    num.parse::<u32>().ok()
                };
                match code.and_then(char::from_u32) {
                    Some(c) if valid_char(c) => out.push(c),
                    _ => return bad("bad character reference"),
                }
                return Ok(());
            }
            match body {
                "lt" => out.push('<'),
                "gt" => out.push('>'),
                "amp" => out.push('&'),
                "quot" => out.push('"'),
                "apos" => out.push('\''),
                _ => match self.entities.get(body) {
                    Some(v) => {
                        if in_attr && v.contains('<') {
                            return bad("< in attribute value");
                        }
                        out.push_str(v);
                    }
                    None => return bad("undefined entity"),
                },
            }
            Ok(())
        }

        fn doctype(&mut self) -> R<()> {
            // at "<!DOCTYPE"
            self.pos += 9;
            if !self.skip_ws() {
                return bad("bad DOCTYPE");
            }
            self.name()?;
            // External id and/or internal subset.
            loop {
                self.skip_ws();
                match self.peek() {
                    Some('>') => {
                        self.pos += 1;
                        return Ok(());
                    }
                    Some('[') => {
                        self.pos += 1;
                        self.internal_subset()?;
                        self.skip_ws();
                        if self.peek() != Some('>') {
                            return bad("bad DOCTYPE");
                        }
                        self.pos += 1;
                        return Ok(());
                    }
                    Some(q @ ('"' | '\'')) => {
                        self.pos += 1;
                        let Some(i) = self.rest().find(q) else { return bad("bad DOCTYPE") };
                        self.pos += i + 1;
                    }
                    Some(c) if name_start(c) => {
                        self.name()?;
                    }
                    _ => return bad("bad DOCTYPE"),
                }
            }
        }

        fn internal_subset(&mut self) -> R<()> {
            loop {
                self.skip_ws();
                if self.peek() == Some(']') {
                    self.pos += 1;
                    return Ok(());
                }
                if self.starts("<!--") {
                    self.comment()?;
                } else if self.starts("<?") {
                    self.pi()?;
                } else if self.starts("<!ENTITY") {
                    self.pos += 8;
                    if !self.skip_ws() {
                        return bad("bad ENTITY");
                    }
                    if self.peek() == Some('%') {
                        return bad("parameter entities are not supported");
                    }
                    let n = self.name()?.to_string();
                    if !self.skip_ws() {
                        return bad("bad ENTITY");
                    }
                    let q = match self.peek() {
                        Some(q @ ('"' | '\'')) => q,
                        _ => return bad("external entities are not supported"),
                    };
                    self.pos += 1;
                    let Some(i) = self.rest().find(q) else { return bad("bad ENTITY") };
                    let val = self.rest()[..i].to_string();
                    self.pos += i + 1;
                    if val.contains('&') || val.contains('%') || !val.chars().all(valid_char) {
                        return bad("entity values with references are not supported");
                    }
                    self.skip_ws();
                    if self.peek() != Some('>') {
                        return bad("bad ENTITY");
                    }
                    self.pos += 1;
                    self.entities.entry(n).or_insert(val);
                } else if self.starts("<!ELEMENT") || self.starts("<!ATTLIST") || self.starts("<!NOTATION") {
                    let mut quote: Option<char> = None;
                    loop {
                        let Some(c) = self.peek() else { return bad("unterminated declaration") };
                        self.pos += c.len_utf8();
                        match (quote, c) {
                            (Some(q), c) if c == q => quote = None,
                            (None, '"' | '\'') => quote = Some(c),
                            (None, '>') => break,
                            _ => {}
                        }
                    }
                } else {
                    return bad("unsupported DOCTYPE content");
                }
            }
        }
    }

    struct Open {
        idx: usize,
        qname: String,
        scope: usize, // number of bindings in force (stack length)
    }

    /// Parse `text` into a tree, or an error message (Python's `ParseError`).
    pub fn parse(text: &str) -> R<Tree> {
        let text = text.strip_prefix('\u{feff}').unwrap_or(text);
        if text.is_empty() {
            return bad("no element found");
        }
        // Newlines are normalised before anything is read.
        let norm;
        let src: &str = if text.contains('\r') {
            norm = text.replace("\r\n", "\n").replace('\r', "\n");
            &norm
        } else {
            text
        };
        let mut p = P { s: src, pos: 0, entities: HashMap::new() };
        let mut tree = Tree::default();

        // Prolog.
        if p.starts("<?xml") && p.rest()[5..].starts_with(is_ws) {
            p.xml_decl()?;
        }
        let mut seen_doctype = false;
        loop {
            p.skip_ws();
            if p.starts("<!--") {
                p.comment()?;
            } else if p.starts("<?") {
                p.pi()?;
            } else if p.starts("<!DOCTYPE") && !seen_doctype {
                seen_doctype = true;
                p.doctype()?;
            } else {
                break;
            }
        }
        if p.peek() != Some('<') {
            return bad("no element found");
        }

        // Namespace bindings: (prefix, bound) pairs, innermost last. Only the
        // set of bound prefixes matters (names are reduced to local parts).
        let mut bindings: Vec<(String, String)> = Vec::new();
        let mut open: Vec<Open> = Vec::new();
        let mut root_done = false;

        loop {
            if root_done {
                // Epilogue: whitespace, comments, PIs.
                p.skip_ws();
                if p.pos >= p.s.len() {
                    break;
                }
                if p.starts("<!--") {
                    p.comment()?;
                } else if p.starts("<?") {
                    p.pi()?;
                } else {
                    return bad("junk after document element");
                }
                continue;
            }
            // Inside an element (or at the root start tag).
            let Some(c) = p.peek() else { return bad("unclosed token") };
            if c == '<' {
                if p.starts("</") {
                    p.pos += 2;
                    let qn = p.name()?.to_string();
                    p.skip_ws();
                    if p.peek() != Some('>') {
                        return bad("bad end tag");
                    }
                    p.pos += 1;
                    let Some(top) = open.pop() else { return bad("unexpected end tag") };
                    if top.qname != qn {
                        return bad("mismatched tag");
                    }
                    bindings.truncate(top.scope);
                    if open.is_empty() {
                        root_done = true;
                    }
                } else if p.starts("<!--") {
                    if open.is_empty() {
                        return bad("comment before root handled in prolog");
                    }
                    p.comment()?;
                } else if p.starts("<![CDATA[") {
                    if open.is_empty() {
                        return bad("CDATA outside the root");
                    }
                    p.pos += 9;
                    let Some(i) = p.rest().find("]]>") else { return bad("unterminated CDATA") };
                    let body = &p.rest()[..i];
                    if !body.chars().all(valid_char) {
                        return bad("invalid character");
                    }
                    let top = open.last().expect("non-empty");
                    if tree.elems[top.idx].children.is_empty() {
                        tree.elems[top.idx].text.push_str(body);
                    }
                    p.pos += i + 3;
                } else if p.starts("<?") {
                    if open.is_empty() {
                        return bad("PI before root handled in prolog");
                    }
                    p.pi()?;
                } else if p.starts("<!") {
                    return bad("bad markup");
                } else {
                    // Start tag.
                    p.pos += 1;
                    let qname = p.name()?.to_string();
                    let scope = bindings.len();
                    // Attributes.
                    let mut attrs: Vec<(String, String)> = Vec::new();
                    let self_closing;
                    loop {
                        let had_ws = p.skip_ws();
                        if p.starts("/>") {
                            p.pos += 2;
                            self_closing = true;
                            break;
                        }
                        if p.peek() == Some('>') {
                            p.pos += 1;
                            self_closing = false;
                            break;
                        }
                        if !had_ws {
                            return bad("bad start tag");
                        }
                        let an = p.name()?.to_string();
                        p.skip_ws();
                        if p.peek() != Some('=') {
                            return bad("attribute without value");
                        }
                        p.pos += 1;
                        p.skip_ws();
                        let q = match p.peek() {
                            Some(q @ ('"' | '\'')) => q,
                            _ => return bad("attribute value not quoted"),
                        };
                        p.pos += 1;
                        let mut val = String::new();
                        loop {
                            let Some(c) = p.peek() else { return bad("unterminated attribute") };
                            if c == q {
                                p.pos += 1;
                                break;
                            }
                            match c {
                                '<' => return bad("< in attribute value"),
                                '&' => p.reference(&mut val, true)?,
                                c if valid_char(c) => {
                                    // Attribute whitespace normalisation.
                                    val.push(if matches!(c, '\t' | '\n') { ' ' } else { c });
                                    p.pos += c.len_utf8();
                                }
                                _ => return bad("invalid character"),
                            }
                        }
                        attrs.push((an, val));
                    }
                    // Namespace declarations first.
                    for (an, val) in &attrs {
                        if an == "xmlns" {
                            bindings.push((String::new(), val.clone()));
                        } else if let Some(pfx) = an.strip_prefix("xmlns:") {
                            if pfx.is_empty() || pfx.contains(':') || pfx == "xmlns" || pfx == "xml" || val.is_empty() {
                                return bad("bad namespace declaration");
                            }
                            bindings.push((pfx.to_string(), val.clone()));
                        }
                    }
                    let bound = |pfx: &str, b: &[(String, String)]| -> bool {
                        pfx == "xml" || b.iter().any(|(k, _)| k == pfx)
                    };
                    let uri_of = |pfx: &str, b: &[(String, String)]| -> String {
                        if pfx == "xml" {
                            return "xml".to_string();
                        }
                        b.iter().rev().find(|(k, _)| k == pfx).map(|(_, v)| v.clone()).unwrap_or_default()
                    };
                    // Element name.
                    let local = match qname.split_once(':') {
                        None => qname.clone(),
                        Some((pfx, loc)) => {
                            if pfx.is_empty() || loc.is_empty() || loc.contains(':') || pfx == "xmlns" || !bound(pfx, &bindings)
                            {
                                bindings.truncate(scope);
                                return bad("bad element name or unbound prefix");
                            }
                            loc.to_string()
                        }
                    };
                    // Attribute names: unique after namespace expansion.
                    let mut seen: Vec<(String, String)> = Vec::new();
                    for (an, _) in &attrs {
                        if an == "xmlns" || an.starts_with("xmlns:") {
                            if seen.iter().any(|(u, l)| u == "xmlns" && l == an) {
                                return bad("duplicate attribute");
                            }
                            seen.push(("xmlns".to_string(), an.clone()));
                            continue;
                        }
                        let (uri, loc) = match an.split_once(':') {
                            None => (String::new(), an.clone()),
                            Some((pfx, loc)) => {
                                if pfx.is_empty() || loc.is_empty() || loc.contains(':') || !bound(pfx, &bindings) {
                                    return bad("bad attribute name or unbound prefix");
                                }
                                (uri_of(pfx, &bindings), loc.to_string())
                            }
                        };
                        if seen.iter().any(|(u, l)| *u == uri && *l == loc) {
                            return bad("duplicate attribute");
                        }
                        seen.push((uri, loc));
                    }
                    let idx = tree.elems.len();
                    tree.elems.push(Elem { local, text: String::new(), children: Vec::new() });
                    if let Some(parent) = open.last() {
                        tree.elems[parent.idx].children.push(idx);
                    }
                    if self_closing {
                        bindings.truncate(scope);
                        if open.is_empty() {
                            root_done = true;
                        }
                    } else {
                        open.push(Open { idx, qname, scope });
                    }
                }
            } else {
                // Character data.
                if open.is_empty() {
                    return bad("text outside the root");
                }
                let mut buf = String::new();
                while let Some(c) = p.peek() {
                    if c == '<' {
                        break;
                    }
                    if c == '&' {
                        p.reference(&mut buf, false)?;
                    } else if c == ']' && p.starts("]]>") {
                        return bad("]]> in text");
                    } else if valid_char(c) {
                        buf.push(c);
                        p.pos += c.len_utf8();
                    } else {
                        return bad("invalid character");
                    }
                }
                let top = open.last().expect("non-empty");
                if tree.elems[top.idx].children.is_empty() {
                    tree.elems[top.idx].text.push_str(&buf);
                }
            }
        }
        if !open.is_empty() || tree.elems.is_empty() {
            return bad("no element found");
        }
        Ok(tree)
    }
}

// ---- KML ---------------------------------------------------------------------

/// `_parse_coords`: KML `coordinates` text, whitespace-separated
/// `lon,lat[,alt]`, to `(lat, lon)`.
fn parse_coords(text: &str) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    for tok in py_split_ws(text) {
        let parts: Vec<&str> = tok.split(',').collect();
        if parts.len() >= 2 {
            if let (Some(lat), Some(lon)) = (float_str(parts[1]), float_str(parts[0])) {
                out.push((lat, lon));
            }
        }
    }
    out
}

/// `_first_coords`: the first `coordinates` descendant with text.
fn first_coords(tree: &xml::Tree, elem: usize) -> Vec<(f64, f64)> {
    for d in tree.iter(elem) {
        let e = &tree.elems[d];
        if e.local == "coordinates" && !e.text.is_empty() {
            return parse_coords(&e.text);
        }
    }
    Vec::new()
}

/// `parse_kml`: LineString becomes an open feature, Polygon a closed one.
pub fn parse_kml(text: &str) -> Vec<GeoFeature> {
    let Ok(tree) = xml::parse(text) else { return Vec::new() };
    let mut feats = Vec::new();
    for pm in tree.iter(0) {
        if tree.elems[pm].local != "Placemark" {
            continue;
        }
        let mut name = String::new();
        for &ch in &tree.elems[pm].children {
            let e = &tree.elems[ch];
            if e.local == "name" && !e.text.is_empty() {
                name = py_strip(&e.text).to_string();
            }
        }
        for geom in tree.iter(pm) {
            match tree.elems[geom].local.as_str() {
                "LineString" => {
                    let pts = first_coords(&tree, geom);
                    if pts.len() >= 2 {
                        feats.push(GeoFeature { points: pts, closed: false, name: name.clone() });
                    }
                }
                "Polygon" => {
                    let pts = drop_closing_dup(first_coords(&tree, geom), true);
                    if pts.len() >= 3 {
                        feats.push(GeoFeature { points: pts, closed: true, name: name.clone() });
                    }
                }
                _ => {}
            }
        }
    }
    feats
}

// ---- KMZ ---------------------------------------------------------------------

fn u16le(b: &[u8], o: usize) -> Option<usize> {
    Some(u16::from_le_bytes([*b.get(o)?, *b.get(o + 1)?]) as usize)
}
fn u32le(b: &[u8], o: usize) -> Option<usize> {
    Some(u32::from_le_bytes([*b.get(o)?, *b.get(o + 1)?, *b.get(o + 2)?, *b.get(o + 3)?]) as usize)
}

/// `parse_kmz` over the archive's bytes: the first `.kml` entry (`doc.kml`
/// when present), decoded as UTF-8 with replacement. `Err` for an unreadable
/// archive (Python's `BadZipFile`).
pub fn parse_kmz_bytes(zip: &[u8]) -> Result<Vec<GeoFeature>, String> {
    const BAD: &str = "BadZipFile";
    // End of central directory.
    let lo = zip.len().saturating_sub(22 + 65535);
    let mut eocd = None;
    if zip.len() >= 22 {
        let mut i = zip.len() - 22;
        loop {
            if zip[i..].starts_with(b"PK\x05\x06") {
                eocd = Some(i);
                break;
            }
            if i == lo {
                break;
            }
            i -= 1;
        }
    }
    let eocd = eocd.ok_or(BAD)?;
    let count = u16le(zip, eocd + 10).ok_or(BAD)?;
    let cd_size = u32le(zip, eocd + 12).ok_or(BAD)?;
    let cd_off = u32le(zip, eocd + 16).ok_or(BAD)?;
    // The central directory sits right before the EOCD (any prefix before the
    // archive shifts every offset).
    let concat = eocd.checked_sub(cd_size + cd_off).ok_or(BAD)?;
    let mut pos = cd_off + concat;
    struct Entry {
        name: String,
        method: usize,
        csize: usize,
        local: usize,
    }
    let mut entries = Vec::new();
    for _ in 0..count {
        if !zip.get(pos..).ok_or(BAD)?.starts_with(b"PK\x01\x02") {
            return Err(BAD.to_string());
        }
        let flags = u16le(zip, pos + 8).ok_or(BAD)?;
        let method = u16le(zip, pos + 10).ok_or(BAD)?;
        let csize = u32le(zip, pos + 20).ok_or(BAD)?;
        let nlen = u16le(zip, pos + 28).ok_or(BAD)?;
        let xlen = u16le(zip, pos + 30).ok_or(BAD)?;
        let clen = u16le(zip, pos + 32).ok_or(BAD)?;
        let local = u32le(zip, pos + 42).ok_or(BAD)?;
        let raw = zip.get(pos + 46..pos + 46 + nlen).ok_or(BAD)?;
        let name = if flags & 0x800 != 0 {
            String::from_utf8_lossy(raw).into_owned()
        } else {
            // cp437: ASCII is the same; the high half only matters for
            // non-ASCII names, which cannot be `doc.kml`.
            raw.iter().map(|&b| if b < 0x80 { b as char } else { '\u{fffd}' }).collect()
        };
        entries.push(Entry { name, method, csize, local: local + concat });
        pos += 46 + nlen + xlen + clen;
    }
    let kmls: Vec<&Entry> = entries.iter().filter(|e| e.name.to_lowercase().ends_with(".kml")).collect();
    if kmls.is_empty() {
        return Ok(Vec::new());
    }
    let pick = if kmls.iter().any(|e| e.name == "doc.kml") { "doc.kml" } else { kmls[0].name.as_str() };
    let entry = entries.iter().rev().find(|e| e.name == pick).expect("picked from entries");
    let lh = entry.local;
    if !zip.get(lh..).ok_or(BAD)?.starts_with(b"PK\x03\x04") {
        return Err(BAD.to_string());
    }
    let nlen = u16le(zip, lh + 26).ok_or(BAD)?;
    let xlen = u16le(zip, lh + 28).ok_or(BAD)?;
    let start = lh + 30 + nlen + xlen;
    let data = zip.get(start..start + entry.csize).ok_or(BAD)?;
    let bytes = match entry.method {
        0 => data.to_vec(),
        8 => {
            let mut out = Vec::new();
            flate2::read::DeflateDecoder::new(data).read_to_end(&mut out).map_err(|e| e.to_string())?;
            out
        }
        m => return Err(format!("NotImplementedError: compression type {m}")),
    };
    Ok(parse_kml(&String::from_utf8_lossy(&bytes)))
}

// ---- GeoJSON -----------------------------------------------------------------

/// `[(c[1], c[0]) for c in coords if len(c) >= 2]` over a JSON value.
///
/// A coordinate that is a list of two or more numbers (or booleans, which are
/// Python ints) gives a point; one shorter than two is skipped, whatever it
/// is (a string's characters and a dict's keys all are). Python holds any
/// other value as a "coordinate" without complaint; those are `NotModelled`.
fn lat_lon_pairs(coords: &Value) -> PyResult<Vec<(f64, f64)>> {
    let mut out = Vec::new();
    for c in unpack(coords)? {
        match &c {
            Value::Array(a) => {
                if a.len() >= 2 {
                    let num = |v: &Value| match v {
                        Value::Int(_) | Value::Float(_) | Value::Bool(_) => crate::pyconv::number(v),
                        _ => err("NotModelled", "a coordinate that is not a number"),
                    };
                    out.push((num(&a[1])?, num(&a[0])?));
                }
            }
            Value::Str(t) => {
                if t.chars().count() >= 2 {
                    return err("NotModelled", "a coordinate that is a string");
                }
            }
            Value::Object(o) => {
                if o.len() >= 2 {
                    return err("KeyError", "1");
                }
            }
            _ => return err("TypeError", "object of this type has no len()"),
        }
    }
    Ok(out)
}

/// `c or []`.
fn or_empty(v: Option<&Value>) -> Value {
    match v {
        Some(x) if truthy(x) => x.clone(),
        _ => Value::Array(Vec::new()),
    }
}

fn add_line(coords: &Value, name: &str, feats: &mut Vec<GeoFeature>) -> PyResult<()> {
    let pts = lat_lon_pairs(coords)?;
    if pts.len() >= 2 {
        feats.push(GeoFeature { points: pts, closed: false, name: name.to_string() });
    }
    Ok(())
}

fn add_ring(coords: &Value, name: &str, feats: &mut Vec<GeoFeature>) -> PyResult<()> {
    let pts = drop_closing_dup(lat_lon_pairs(coords)?, true);
    if pts.len() >= 3 {
        feats.push(GeoFeature { points: pts, closed: true, name: name.to_string() });
    }
    Ok(())
}

/// `c[0]` of a JSON sequence.
fn first_item(v: &Value) -> PyResult<Value> {
    match v {
        Value::Array(a) => match a.first() {
            Some(x) => Ok(x.clone()),
            None => err("IndexError", "list index out of range"),
        },
        Value::Str(s) => match s.chars().next() {
            Some(c) => Ok(Value::Str(c.to_string())),
            None => err("IndexError", "string index out of range"),
        },
        Value::Object(_) => err("KeyError", "0"),
        _ => err("TypeError", "object is not subscriptable"),
    }
}

fn geom(g: &Value, name: &str, feats: &mut Vec<GeoFeature>) -> PyResult<()> {
    if !matches!(g, Value::Object(_)) {
        return Ok(());
    }
    let t = g.get("type");
    let c = g.get("coordinates");
    let ty = match t {
        Some(Value::Str(s)) => s.as_str(),
        _ => "",
    };
    match ty {
        "LineString" => add_line(&or_empty(c), name, feats)?,
        "MultiLineString" => {
            for line in unpack(&or_empty(c))? {
                add_line(&line, name, feats)?;
            }
        }
        "Polygon" => {
            if let Some(cv) = c {
                if truthy(cv) {
                    add_ring(&first_item(cv)?, name, feats)?;
                }
            }
        }
        "MultiPolygon" => {
            for poly in unpack(&or_empty(c))? {
                if truthy(&poly) {
                    add_ring(&first_item(&poly)?, name, feats)?;
                }
            }
        }
        "GeometryCollection" => {
            let subs = match g.get("geometries") {
                Some(v) => unpack(v)?,
                None => Vec::new(),
            };
            for sub in subs {
                geom(&sub, name, feats)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn obj(o: &Value, feats: &mut Vec<GeoFeature>) -> PyResult<()> {
    if !matches!(o, Value::Object(_)) {
        return Ok(());
    }
    match o.get("type") {
        Some(Value::Str(t)) if t == "FeatureCollection" => {
            let items = match o.get("features") {
                Some(v) => unpack(v)?,
                None => Vec::new(),
            };
            for f in items {
                obj(&f, feats)?;
            }
        }
        Some(Value::Str(t)) if t == "Feature" => {
            // (o.get("properties") or {}).get("name") or ""
            let name_val = match o.get("properties") {
                Some(p) if truthy(p) => get(p, "name")?.cloned(),
                _ => None,
            };
            let name = match name_val {
                Some(v) if truthy(&v) => py_str(&v),
                _ => String::new(),
            };
            if let Some(g) = o.get("geometry") {
                geom(g, &name, feats)?;
            }
        }
        _ => geom(o, "", feats)?,
    }
    Ok(())
}

/// `parse_geojson`: LineString / Polygon and their Multi* variants (and
/// geometry collections). Unreadable JSON is `[]`. `Err` for what Python lets
/// escape from a malformed document (`AttributeError`, `TypeError`,
/// `IndexError`...) or holds as non-numbers (`NotModelled`).
pub fn parse_geojson(text: &str) -> PyResult<Vec<GeoFeature>> {
    let Ok(data) = loads(text) else { return Ok(Vec::new()) };
    let mut feats = Vec::new();
    obj(&data, &mut feats)?;
    Ok(feats)
}

/// `load_features` after the file is read: dispatch on the extension
/// (lower-case, with the dot), sniffing the content for an unknown one.
/// `.kmz` has no text form: use [`parse_kmz_bytes`].
pub fn load_features_text(ext: &str, text: &str) -> PyResult<Vec<GeoFeature>> {
    match ext {
        ".kml" => Ok(parse_kml(text)),
        ".geojson" | ".json" => parse_geojson(text),
        _ => {
            if text.trim_start_matches(is_py_space).starts_with('<') {
                Ok(parse_kml(text))
            } else {
                parse_geojson(text)
            }
        }
    }
}

/// `load_features(path)`: read a KML / KMZ / GeoJSON file.
pub fn load_features(path: &Path) -> Result<Vec<GeoFeature>, String> {
    let ext = path
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()))
        .unwrap_or_default();
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if ext == ".kmz" {
        return parse_kmz_bytes(&bytes);
    }
    let text = String::from_utf8_lossy(&bytes);
    load_features_text(&ext, &text).map_err(|e| format!("{}: {}", e.kind, e.msg))
}
