// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! A small strict XML reader with the behaviour `formats/dae.py` leans on
//! from `xml.etree.ElementTree.parse` (expat underneath):
//!
//! - tags and attribute names come out Clark-expanded (`{uri}local`), so the
//!   importer asks for `{COLLADA ns}name` exactly as Python does; namespace
//!   declarations are not attributes;
//! - an element keeps the text before its first child (`el.text`; `None` when
//!   there is none); comments and processing instructions leave no trace, and
//!   CDATA is plain text; the text after a child (`tail`) is dropped;
//! - `\r\n` and `\r` read as `\n`; attribute values turn tabs and line breaks
//!   into spaces; the five predefined entities and character references are
//!   expanded, any other entity is an error;
//! - a document that is not well formed is an error.
//!
//! No DTD is read: a `<!DOCTYPE>` is skipped; its internal general entities with
//! plain text values are kept, any other entity declaration is refused, and
//! nothing is ever fetched. Elements live in an
//! arena in document order, so `root.iter()` is ascending ids.

use std::collections::HashMap;

/// One element.
#[derive(Clone, Debug)]
pub struct Element {
    /// `{uri}local` (or `local` when it has no namespace).
    pub tag: String,
    pub attrs: Vec<(String, String)>,
    /// `el.text`.
    pub text: Option<String>,
    pub children: Vec<usize>,
}

/// A parsed document: elements by id, the root first.
#[derive(Clone, Debug, Default)]
pub struct Doc {
    pub nodes: Vec<Element>,
}

impl Doc {
    pub const ROOT: usize = 0;

    pub fn el(&self, id: usize) -> &Element {
        &self.nodes[id]
    }

    /// `el.get(name)`.
    pub fn get(&self, id: usize, name: &str) -> Option<&str> {
        self.nodes[id].attrs.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }

    /// `el.find(tag)`: the first direct child with that expanded tag.
    pub fn find(&self, id: usize, tag: &str) -> Option<usize> {
        self.nodes[id].children.iter().copied().find(|&c| self.nodes[c].tag == tag)
    }

    /// `el.findall(tag)`.
    pub fn findall(&self, id: usize, tag: &str) -> Vec<usize> {
        self.nodes[id].children.iter().copied().filter(|&c| self.nodes[c].tag == tag).collect()
    }

    /// `el.iter(tag)`: `id` and its descendants in document order.
    pub fn iter_tag(&self, id: usize, tag: &str) -> Vec<usize> {
        self.iter_all(id).into_iter().filter(|&i| self.nodes[i].tag == tag).collect()
    }

    /// `el.iter()`.
    pub fn iter_all(&self, id: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut stack = vec![id];
        while let Some(i) = stack.pop() {
            out.push(i);
            stack.extend(self.nodes[i].children.iter().rev());
        }
        out
    }
}

/// `_tag(el)`: the tag without its namespace.
pub fn local(tag: &str) -> &str {
    tag.rsplit_once('}').map(|(_, l)| l).unwrap_or(tag)
}

fn is_xml_char(c: char) -> bool {
    matches!(c, '\u{9}' | '\u{a}' | '\u{d}' | '\u{20}'..='\u{d7ff}' | '\u{e000}'..='\u{fffd}' | '\u{10000}'..='\u{10ffff}')
}

fn is_name_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_' || c == ':' || (c as u32) >= 0xc0 && c != '\u{d7}' && c != '\u{f7}' && is_xml_char(c)
}

fn is_name_char(c: char) -> bool {
    is_name_start(c) || c.is_ascii_digit() || c == '-' || c == '.' || c == '\u{b7}'
}

/// Decodes the bytes the way expat chooses an encoding: a byte order mark,
/// else the `<?xml ... encoding=...?>` declaration, else UTF-8.
fn decode(data: &[u8]) -> Result<String, String> {
    let bad = || "not well-formed (invalid token): line 1, column 0".to_string();
    let utf16 = |bytes: &[u8], be: bool| -> Result<String, String> {
        if !bytes.len().is_multiple_of(2) {
            return Err(bad());
        }
        let units: Vec<u16> = bytes.chunks(2).map(|c| if be { u16::from_be_bytes([c[0], c[1]]) } else { u16::from_le_bytes([c[0], c[1]]) }).collect();
        String::from_utf16(&units).map_err(|_| bad())
    };
    if let Some(rest) = data.strip_prefix(&[0xef, 0xbb, 0xbf]) {
        return std::str::from_utf8(rest).map(str::to_string).map_err(|_| bad());
    }
    if let Some(rest) = data.strip_prefix(&[0xff, 0xfe]) {
        return utf16(rest, false);
    }
    if let Some(rest) = data.strip_prefix(&[0xfe, 0xff]) {
        return utf16(rest, true);
    }
    if data.starts_with(&[b'<', 0, b'?', 0]) {
        return utf16(data, false);
    }
    if data.starts_with(&[0, b'<', 0, b'?']) {
        return utf16(data, true);
    }
    let mut enc = "utf-8".to_string();
    if data.starts_with(b"<?xml") {
        let end = data.windows(2).position(|w| w == b"?>").unwrap_or(data.len());
        let decl = String::from_utf8_lossy(&data[..end]).to_string();
        if let Some(at) = decl.find("encoding") {
            let rest = decl[at + 8..].trim_start();
            if let Some(rest) = rest.strip_prefix('=') {
                let rest = rest.trim_start();
                if let Some(q) = rest.chars().next().filter(|c| *c == '"' || *c == '\'') {
                    if let Some(len) = rest[1..].find(q) {
                        enc = rest[1..1 + len].to_ascii_lowercase();
                    }
                }
            }
        }
    }
    match enc.as_str() {
        "utf-8" | "utf8" => std::str::from_utf8(data).map(str::to_string).map_err(|_| bad()),
        "iso-8859-1" | "latin-1" | "latin1" => Ok(data.iter().map(|&b| b as char).collect()),
        "us-ascii" | "ascii" => {
            if data.is_ascii() {
                Ok(data.iter().map(|&b| b as char).collect())
            } else {
                Err(bad())
            }
        }
        // expat's own set is UTF-8, UTF-16, ISO-8859-1 and US-ASCII; Python adds
        // its single-byte codecs, which are not here.
        other => Err(format!("unknown encoding: {other}")),
    }
}

struct Parser<'a> {
    s: &'a [char],
    i: usize,
    /// Internal general entities of the DOCTYPE: name to replacement text
    /// (character references already resolved).
    entities: HashMap<String, String>,
}

/// The predefined entity `name` stands for.
fn predefined(name: &str) -> Option<char> {
    match name {
        "lt" => Some('<'),
        "gt" => Some('>'),
        "amp" => Some('&'),
        "apos" => Some('\''),
        "quot" => Some('"'),
        _ => None,
    }
}

/// The character a `#N` / `#xH` reference body names.
fn char_ref(body: &str) -> Option<char> {
    let code = if let Some(h) = body.strip_prefix("#x") {
        u32::from_str_radix(h, 16).ok().filter(|_| !h.is_empty() && h.chars().all(|c| c.is_ascii_hexdigit()))
    } else if let Some(d) = body.strip_prefix('#') {
        d.parse::<u32>().ok().filter(|_| !d.is_empty() && d.chars().all(|c| c.is_ascii_digit()))
    } else {
        None
    };
    code.and_then(char::from_u32).filter(|&c| is_xml_char(c))
}

/// An entity's literal value with its character references resolved.
fn resolve_char_refs(raw: &str) -> Option<String> {
    let mut out = String::new();
    let mut rest = raw;
    while let Some(at) = rest.find("&#") {
        out.push_str(&rest[..at]);
        let end = rest[at..].find(';')? + at;
        out.push(char_ref(&rest[at + 1..end])?);
        rest = &rest[end + 1..];
    }
    out.push_str(rest);
    Some(out)
}

/// The text an entity's replacement stands for: predefined entities expanded,
/// anything else refused.
fn expand_replacement(value: &str) -> Option<String> {
    let mut out = String::new();
    let mut rest = value;
    while let Some(at) = rest.find('&') {
        out.push_str(&rest[..at]);
        let end = rest[at..].find(';')? + at;
        out.push(predefined(&rest[at + 1..end])?);
        rest = &rest[end + 1..];
    }
    out.push_str(rest);
    Some(out)
}

impl Parser<'_> {
    fn pos(&self) -> (usize, usize) {
        let mut line = 1;
        let mut col = 0;
        for &c in &self.s[..self.i.min(self.s.len())] {
            if c == '\n' {
                line += 1;
                col = 0;
            } else {
                col += 1;
            }
        }
        (line, col)
    }

    fn err<T>(&self, what: &str) -> Result<T, String> {
        let (l, c) = self.pos();
        Err(format!("{what}: line {l}, column {c}"))
    }

    fn starts(&self, lit: &str) -> bool {
        lit.chars().enumerate().all(|(k, c)| self.s.get(self.i + k) == Some(&c))
    }

    fn skip_ws(&mut self) -> bool {
        let from = self.i;
        while self.i < self.s.len() && matches!(self.s[self.i], ' ' | '\t' | '\n' | '\r') {
            self.i += 1;
        }
        self.i > from
    }

    fn name(&mut self) -> Result<String, String> {
        let start = self.i;
        match self.s.get(self.i) {
            Some(&c) if is_name_start(c) => self.i += 1,
            _ => return self.err("not well-formed (invalid token)"),
        }
        while self.i < self.s.len() && is_name_char(self.s[self.i]) {
            self.i += 1;
        }
        Ok(self.s[start..self.i].iter().collect())
    }

    /// Skips up to and past `end`.
    fn skip_to(&mut self, end: &str) -> Result<(), String> {
        while self.i < self.s.len() {
            if self.starts(end) {
                self.i += end.chars().count();
                return Ok(());
            }
            self.i += 1;
        }
        self.err("unclosed token")
    }

    fn comment(&mut self) -> Result<(), String> {
        self.i += 4;
        while self.i < self.s.len() {
            if self.starts("--") {
                if self.starts("-->") {
                    self.i += 3;
                    return Ok(());
                }
                return self.err("not well-formed (invalid token)");
            }
            self.i += 1;
        }
        self.err("unclosed token")
    }

    fn pi(&mut self) -> Result<(), String> {
        self.i += 2;
        let target = self.name()?;
        if target.eq_ignore_ascii_case("xml") {
            return self.err("XML or text declaration not at start of entity");
        }
        self.skip_to("?>")
    }

    /// `<!ENTITY name "value">` (internal, general): the first declaration of
    /// a name wins. Parameter and external entities are refused.
    fn entity_decl(&mut self) -> Result<(), String> {
        self.i += 8;
        self.skip_ws();
        if self.s.get(self.i) == Some(&'%') {
            return self.err("parameter entities are not supported");
        }
        let name = self.name()?;
        self.skip_ws();
        let q = match self.s.get(self.i) {
            Some(&c) if c == '"' || c == '\'' => c,
            _ => return self.err("external entities are not supported"),
        };
        self.i += 1;
        let from = self.i;
        while self.i < self.s.len() && self.s[self.i] != q {
            self.i += 1;
        }
        if self.i >= self.s.len() {
            return self.err("unclosed token");
        }
        let raw: String = self.s[from..self.i].iter().collect();
        self.i += 1;
        self.skip_ws();
        if self.s.get(self.i) != Some(&'>') {
            return self.err("not well-formed (invalid token)");
        }
        self.i += 1;
        let Some(value) = resolve_char_refs(&raw).filter(|v| !v.contains('<')) else { return self.err("not well-formed (invalid token)") };
        self.entities.entry(name).or_insert(value);
        Ok(())
    }

    fn doctype(&mut self) -> Result<(), String> {
        self.i += 9;
        let mut depth = 0;
        let mut quote: Option<char> = None;
        while self.i < self.s.len() {
            let c = self.s[self.i];
            if let Some(q) = quote {
                if c == q {
                    quote = None;
                }
            } else if c == '"' || c == '\'' {
                quote = Some(c);
            } else if c == '[' {
                depth += 1;
            } else if c == ']' {
                depth -= 1;
            } else if depth > 0 && self.starts("<!ENTITY") {
                self.entity_decl()?;
                continue;
            } else if depth > 0 && self.starts("<!--") {
                self.comment()?;
                continue;
            } else if c == '>' && depth <= 0 {
                self.i += 1;
                return Ok(());
            }
            self.i += 1;
        }
        self.err("unclosed token")
    }

    /// `&...;` at `self.i` (on the `&`).
    fn reference(&mut self) -> Result<String, String> {
        let start = self.i;
        self.i += 1;
        let mut body = String::new();
        while let Some(&c) = self.s.get(self.i) {
            self.i += 1;
            if c == ';' {
                let ch: Option<String> = match predefined(&body) {
                    Some(c) => Some(c.to_string()),
                    None if body.starts_with('#') => char_ref(&body).map(|c| c.to_string()),
                    None => self.entities.get(&body).and_then(|v| expand_replacement(v)),
                };
                return match ch {
                    Some(c) => Ok(c),
                    None => {
                        self.i = start;
                        if body.starts_with('#') {
                            self.err("reference to invalid character number")
                        } else {
                            self.err("undefined entity")
                        }
                    }
                };
            }
            if !(is_name_char(c) || c == '#') {
                break;
            }
            body.push(c);
        }
        self.i = start;
        self.err("not well-formed (invalid token)")
    }

    fn attr_value(&mut self) -> Result<String, String> {
        let q = match self.s.get(self.i) {
            Some(&c) if c == '"' || c == '\'' => c,
            _ => return self.err("not well-formed (invalid token)"),
        };
        self.i += 1;
        let mut out = String::new();
        loop {
            let Some(&c) = self.s.get(self.i) else { return self.err("unclosed token") };
            if c == q {
                self.i += 1;
                return Ok(out);
            }
            match c {
                '<' => return self.err("not well-formed (invalid token)"),
                '&' => out.push_str(&self.reference()?),
                '\t' | '\n' | '\r' => {
                    out.push(' ');
                    self.i += 1;
                }
                _ => {
                    out.push(c);
                    self.i += 1;
                }
            }
        }
    }
}

struct Open {
    id: usize,
    raw: String,
    scopes: usize,
}

/// `ET.parse(...).getroot()` on the file's bytes.
pub fn parse(data: &[u8]) -> Result<Doc, String> {
    let text = decode(data)?;
    let mut norm = String::with_capacity(text.len());
    let mut it = text.chars().peekable();
    while let Some(c) = it.next() {
        if c == '\r' {
            if it.peek() == Some(&'\n') {
                it.next();
            }
            norm.push('\n');
        } else {
            norm.push(c);
        }
    }
    let chars: Vec<char> = norm.chars().collect();
    if let Some(pos) = chars.iter().position(|&c| !is_xml_char(c)) {
        let mut p = Parser { s: &chars, i: pos, entities: HashMap::new() };
        p.i = pos;
        return p.err("not well-formed (invalid token)");
    }
    let mut p = Parser { s: &chars, i: 0, entities: HashMap::new() };
    let mut doc = Doc::default();
    // Namespace scopes: (prefix, uri) pairs, innermost last.
    let mut scopes: Vec<(String, String)> = vec![("xml".to_string(), "http://www.w3.org/XML/1998/namespace".to_string())];
    let mut stack: Vec<Open> = Vec::new();
    let mut seen_root = false;
    let mut done = false;
    // The XML declaration, only at the very start.
    if p.starts("<?xml") && chars.get(5).is_some_and(|c| c.is_whitespace()) {
        p.skip_to("?>")?;
    }
    loop {
        if stack.is_empty() {
            p.skip_ws();
        }
        if p.i >= chars.len() {
            break;
        }
        if p.starts("<!--") {
            p.comment()?;
        } else if p.starts("<?") {
            p.pi()?;
        } else if p.starts("<![CDATA[") {
            let Some(open) = stack.last() else { return p.err("syntax error") };
            let id = open.id;
            p.i += 9;
            let from = p.i;
            while p.i < chars.len() && !p.starts("]]>") {
                p.i += 1;
            }
            if p.i >= chars.len() {
                return p.err("unclosed CDATA section");
            }
            let t: String = chars[from..p.i].iter().collect();
            p.i += 3;
            if doc.nodes[id].children.is_empty() && !t.is_empty() {
                doc.nodes[id].text.get_or_insert_with(String::new).push_str(&t);
            }
        } else if p.starts("<!DOCTYPE") {
            if seen_root || !stack.is_empty() {
                return p.err("not well-formed (invalid token)");
            }
            p.doctype()?;
        } else if p.starts("</") {
            let Some(open) = stack.pop() else { return p.err("junk after document element") };
            p.i += 2;
            let name = p.name()?;
            p.skip_ws();
            if name != open.raw {
                return p.err("mismatched tag");
            }
            if p.s.get(p.i) != Some(&'>') {
                return p.err("not well-formed (invalid token)");
            }
            p.i += 1;
            scopes.truncate(open.scopes);
            if stack.is_empty() {
                done = true;
            }
        } else if p.starts("<") {
            if done {
                return p.err("junk after document element");
            }
            p.i += 1;
            let raw = p.name()?;
            let mut attrs: Vec<(String, String)> = Vec::new();
            let scope_mark = scopes.len();
            let mut empty = false;
            loop {
                let had_ws = p.skip_ws();
                match p.s.get(p.i) {
                    Some('>') => {
                        p.i += 1;
                        break;
                    }
                    Some('/') if p.s.get(p.i + 1) == Some(&'>') => {
                        p.i += 2;
                        empty = true;
                        break;
                    }
                    Some(_) if had_ws => {
                        let an = p.name()?;
                        p.skip_ws();
                        if p.s.get(p.i) != Some(&'=') {
                            return p.err("not well-formed (invalid token)");
                        }
                        p.i += 1;
                        p.skip_ws();
                        let v = p.attr_value()?;
                        if attrs.iter().any(|(k, _)| *k == an) {
                            return p.err("duplicate attribute");
                        }
                        attrs.push((an, v));
                    }
                    _ => return p.err("not well-formed (invalid token)"),
                }
            }
            // Namespace declarations out of the attributes, then expansion.
            let mut plain: Vec<(String, String)> = Vec::new();
            for (k, v) in attrs {
                if k == "xmlns" {
                    scopes.push((String::new(), v));
                } else if let Some(pre) = k.strip_prefix("xmlns:") {
                    if pre.is_empty() || v.is_empty() {
                        return p.err("not well-formed (invalid token)");
                    }
                    scopes.push((pre.to_string(), v));
                } else {
                    plain.push((k, v));
                }
            }
            let lookup = |pre: &str| scopes.iter().rev().find(|(k, _)| k == pre).map(|(_, u)| u.clone());
            let expand = |name: &str, is_attr: bool| -> Result<String, String> {
                match name.split_once(':') {
                    Some((pre, loc)) if !pre.is_empty() && !loc.is_empty() && !loc.contains(':') => match lookup(pre) {
                        Some(uri) => Ok(format!("{{{uri}}}{loc}")),
                        None => Err("unbound prefix".to_string()),
                    },
                    Some(_) => Err("not well-formed (invalid token)".to_string()),
                    None if is_attr => Ok(name.to_string()),
                    None => match lookup("") {
                        Some(uri) if !uri.is_empty() => Ok(format!("{{{uri}}}{name}")),
                        _ => Ok(name.to_string()),
                    },
                }
            };
            let tag = match expand(&raw, false) {
                Ok(t) => t,
                Err(m) => return p.err(&m),
            };
            let mut expanded = Vec::with_capacity(plain.len());
            for (k, v) in plain {
                match expand(&k, true) {
                    Ok(k) => {
                        if expanded.iter().any(|(e, _): &(String, String)| *e == k) {
                            return p.err("duplicate attribute");
                        }
                        expanded.push((k, v));
                    }
                    Err(m) => return p.err(&m),
                }
            }
            let id = doc.nodes.len();
            doc.nodes.push(Element { tag, attrs: expanded, text: None, children: Vec::new() });
            if let Some(parent) = stack.last() {
                doc.nodes[parent.id].children.push(id);
            }
            seen_root = true;
            if empty {
                scopes.truncate(scope_mark);
                if stack.is_empty() {
                    done = true;
                }
            } else {
                stack.push(Open { id, raw, scopes: scope_mark });
            }
        } else {
            // Character data.
            let Some(open) = stack.last() else { return p.err("syntax error") };
            let id = open.id;
            let mut t = String::new();
            while p.i < chars.len() && chars[p.i] != '<' {
                if chars[p.i] == '&' {
                    t.push_str(&p.reference()?);
                } else {
                    if p.starts("]]>") {
                        return p.err("not well-formed (invalid token)");
                    }
                    t.push(chars[p.i]);
                    p.i += 1;
                }
            }
            if doc.nodes[id].children.is_empty() && !t.is_empty() {
                doc.nodes[id].text.get_or_insert_with(String::new).push_str(&t);
            }
        }
    }
    if !stack.is_empty() {
        return p.err("no element found");
    }
    if doc.nodes.is_empty() {
        return p.err("no element found");
    }
    Ok(doc)
}

/// Element ids by their `id` attribute (the last element with an id wins, as
/// a dict filled in document order).
pub fn ids(doc: &Doc) -> HashMap<String, usize> {
    let mut by_id = HashMap::new();
    for i in 0..doc.nodes.len() {
        if let Some(v) = doc.get(i, "id") {
            by_id.insert(v.to_string(), i);
        }
    }
    by_id
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(s: &str) -> Doc {
        parse(s.as_bytes()).unwrap()
    }

    #[test]
    fn text_is_the_data_before_the_first_child() {
        let d = ok("<a>x<!-- c --><b/>tail<![CDATA[<y>]]></a>");
        assert_eq!(d.el(0).text.as_deref(), Some("x"));
        let d = ok("<a><![CDATA[<y>]]>&amp;&#65;&#x42;<b/></a>");
        assert_eq!(d.el(0).text.as_deref(), Some("<y>&AB"));
        assert_eq!(ok("<a><b/></a>").el(0).text, None);
        assert_eq!(ok("<a></a>").el(0).text, None);
    }

    #[test]
    fn namespaces_expand_like_etree() {
        let d = ok("<?xml version=\"1.0\"?><c:a xmlns:c=\"u\" xmlns=\"v\" c:k=\"1\" j=\"2\"><b/><x xmlns=\"\"><b/></x></c:a>");
        assert_eq!(d.el(0).tag, "{u}a");
        assert_eq!(d.el(0).attrs, vec![("{u}k".to_string(), "1".to_string()), ("j".to_string(), "2".to_string())]);
        assert_eq!(d.el(1).tag, "{v}b");
        assert_eq!(d.el(3).tag, "b");
    }

    #[test]
    fn line_ends_and_attribute_whitespace() {
        let d = ok("<a k=\"1\t2\r\n3&#10;4\">a\r\nb\rc</a>");
        assert_eq!(d.get(0, "k"), Some("1 2 3\n4"));
        assert_eq!(d.el(0).text.as_deref(), Some("a\nb\nc"));
    }

    #[test]
    fn malformed_documents_are_refused() {
        for bad in [
            "", "<a>", "<a></b>", "<a/><b/>", "<a k=1/>", "<a k=\"1\" k=\"2\"/>", "<a>&nope;</a>", "<a>]]></a>", "<p:a/>", "<a><!-- -- --></a>",
            "<a/>junk", "text", "<a>\u{1}</a>", "<!DOCTYPE a [<!ENTITY % e \"x\">]><a/>", "<!DOCTYPE a [<!ENTITY e SYSTEM \"x\">]><a/>", "<!DOCTYPE a [<!ENTITY e \"<b/>\">]><a>&e;</a>", "<a><?xml x?></a>", "<a k=\"<\"/>",
        ] {
            assert!(parse(bad.as_bytes()).is_err(), "{bad:?}");
        }
        assert!(parse(b"<!DOCTYPE a><a/>").is_ok());
        assert!(parse(b"<a/>  <!-- x -->\n<?pi x?>").is_ok());
    }

    #[test]
    fn internal_entities_expand() {
        let d = ok("<!DOCTYPE a [<!ENTITY x \"y &amp; &#65;\"><!ENTITY x \"second\">]><a k=\"[&x;]\">&x;!</a>");
        assert_eq!(d.get(0, "k"), Some("[y & A]"));
        assert_eq!(d.el(0).text.as_deref(), Some("y & A!"));
    }

    #[test]
    fn encodings() {
        assert!(parse(b"\xef\xbb\xbf<a>\xc3\xa9</a>").is_ok());
        let d = parse(b"<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?><a>\xe9</a>").unwrap();
        assert_eq!(d.el(0).text.as_deref(), Some("\u{e9}"));
        let mut u16: Vec<u8> = vec![0xff, 0xfe];
        for c in "<a>\u{e9}</a>".encode_utf16() {
            u16.extend(c.to_le_bytes());
        }
        assert_eq!(parse(&u16).unwrap().el(0).text.as_deref(), Some("\u{e9}"));
        assert!(parse(b"<a>\xff</a>").is_err());
    }
}
