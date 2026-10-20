// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
// Port of openskp (MIT, Copyright (c) 2024-2026 Ahsan Mehmood): see NOTICE-openskp.md.
//! A small namespace-aware XML reader for the `material.xml` and `style.xml`
//! of a .skp container, standing where `defusedxml.ElementTree` stands: the
//! document must be well formed, an entity declaration is refused, and only
//! the five predefined entities and character references are expanded.

use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
pub struct Elem {
    /// Namespace URI of the element's name ("" for none).
    pub ns: String,
    pub name: String,
    /// Attributes as written (a prefixed attribute keeps its prefix).
    pub attrs: Vec<(String, String)>,
    /// The text before the first child.
    pub text: Option<String>,
    pub children: Vec<Elem>,
}

impl Elem {
    pub fn get(&self, key: &str) -> Option<&str> {
        self.attrs.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    /// `elem.find("{ns}name")`: the first direct child.
    pub fn child(&self, ns: &str, name: &str) -> Option<&Elem> {
        self.children.iter().find(|c| c.ns == ns && c.name == name)
    }

    pub fn children_named<'a>(&'a self, ns: &'a str, name: &'a str) -> impl Iterator<Item = &'a Elem> {
        self.children.iter().filter(move |c| c.ns == ns && c.name == name)
    }

    /// `elem.find(".//{ns}name")`: the first descendant, depth first.
    pub fn descendant(&self, ns: &str, name: &str) -> Option<&Elem> {
        for c in &self.children {
            if c.ns == ns && c.name == name {
                return Some(c);
            }
            if let Some(d) = c.descendant(ns, name) {
                return Some(d);
            }
        }
        None
    }
}

struct P<'a> {
    s: &'a str,
    pos: usize,
}

type R<T> = Result<T, String>;

fn decode(bytes: &[u8]) -> R<String> {
    if let Some(rest) = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]) {
        return String::from_utf8(rest.to_vec()).map_err(|e| e.to_string());
    }
    if bytes.starts_with(&[0xFF, 0xFE]) || bytes.starts_with(&[0xFE, 0xFF]) {
        let le = bytes[0] == 0xFF;
        let units: Vec<u16> = bytes[2..].as_chunks::<2>().0.iter().map(|c| if le { u16::from_le_bytes(*c) } else { u16::from_be_bytes(*c) }).collect();
        return String::from_utf16(&units).map_err(|e| e.to_string());
    }
    // An XML declaration naming latin-1 is the only other encoding seen.
    let head = String::from_utf8_lossy(&bytes[..bytes.len().min(200)]).to_ascii_lowercase();
    if head.starts_with("<?xml") && (head.contains("iso-8859-1") || head.contains("latin1")) {
        return Ok(bytes.iter().map(|&b| b as char).collect());
    }
    String::from_utf8(bytes.to_vec()).map_err(|e| e.to_string())
}

impl<'a> P<'a> {
    fn rest(&self) -> &'a str {
        &self.s[self.pos..]
    }
    fn starts(&self, t: &str) -> bool {
        self.rest().starts_with(t)
    }
    fn skip_ws(&mut self) {
        let r = self.rest();
        self.pos += r.len() - r.trim_start_matches([' ', '\t', '\r', '\n']).len();
    }
    fn skip_to(&mut self, end: &str) -> R<()> {
        match self.rest().find(end) {
            Some(i) => {
                self.pos += i + end.len();
                Ok(())
            }
            None => Err(format!("unterminated construct, expected {end}")),
        }
    }
    fn name(&mut self) -> R<&'a str> {
        let r = self.rest();
        let n = r.find(|c: char| c.is_whitespace() || matches!(c, '=' | '>' | '/' | '<' | '"' | '\'')).unwrap_or(r.len());
        if n == 0 {
            return Err("not well-formed (invalid token)".into());
        }
        self.pos += n;
        Ok(&r[..n])
    }
}

fn unescape(raw: &str) -> R<String> {
    if !raw.contains('&') {
        return Ok(raw.to_string());
    }
    let mut out = String::with_capacity(raw.len());
    let mut rest = raw;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        rest = &rest[i + 1..];
        let end = rest.find(';').ok_or("unterminated entity reference")?;
        let ent = &rest[..end];
        match ent {
            "lt" => out.push('<'),
            "gt" => out.push('>'),
            "amp" => out.push('&'),
            "quot" => out.push('"'),
            "apos" => out.push('\''),
            _ => {
                let code = if let Some(h) = ent.strip_prefix("#x").or_else(|| ent.strip_prefix("#X")) {
                    u32::from_str_radix(h, 16).ok()
                } else if let Some(d) = ent.strip_prefix('#') {
                    d.parse::<u32>().ok()
                } else {
                    return Err(format!("undefined entity &{ent};"));
                };
                out.push(code.and_then(char::from_u32).ok_or("invalid character reference")?);
            }
        }
        rest = &rest[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

fn resolve(prefixes: &[HashMap<String, String>], qname: &str, is_attr: bool) -> R<(String, String)> {
    let (prefix, local) = match qname.split_once(':') {
        Some((p, l)) => (Some(p), l),
        None => (None, qname),
    };
    let key = prefix.unwrap_or("");
    if prefix.is_none() && is_attr {
        return Ok((String::new(), local.to_string()));
    }
    for m in prefixes.iter().rev() {
        if let Some(uri) = m.get(key) {
            return Ok((uri.clone(), local.to_string()));
        }
    }
    if prefix == Some("xml") {
        return Ok(("http://www.w3.org/XML/1998/namespace".into(), local.to_string()));
    }
    if prefix.is_some() {
        return Err(format!("unbound prefix: {qname}"));
    }
    Ok((String::new(), local.to_string()))
}

fn parse_element(p: &mut P, prefixes: &mut Vec<HashMap<String, String>>, depth: usize) -> R<Elem> {
    if depth > 256 {
        return Err("nesting too deep".into());
    }
    p.pos += 1; // <
    let qname = p.name()?.to_string();
    let mut raw_attrs: Vec<(String, String)> = Vec::new();
    let self_closing;
    loop {
        p.skip_ws();
        if p.starts("/>") {
            p.pos += 2;
            self_closing = true;
            break;
        }
        if p.starts(">") {
            p.pos += 1;
            self_closing = false;
            break;
        }
        let an = p.name()?.to_string();
        p.skip_ws();
        if !p.starts("=") {
            return Err("not well-formed (attribute without value)".into());
        }
        p.pos += 1;
        p.skip_ws();
        let quote = p.rest().chars().next().ok_or("unexpected end")?;
        if quote != '"' && quote != '\'' {
            return Err("attribute value not quoted".into());
        }
        p.pos += 1;
        let close = p.rest().find(quote).ok_or("unterminated attribute")?;
        let val = &p.rest()[..close];
        if val.contains('<') {
            return Err("'<' in attribute value".into());
        }
        let val = unescape(val)?;
        p.pos += close + 1;
        if raw_attrs.iter().any(|(k, _)| *k == an) {
            return Err("duplicate attribute".into());
        }
        raw_attrs.push((an, val));
    }
    let mut decl = HashMap::new();
    for (k, v) in &raw_attrs {
        if k == "xmlns" {
            decl.insert(String::new(), v.clone());
        } else if let Some(pfx) = k.strip_prefix("xmlns:") {
            decl.insert(pfx.to_string(), v.clone());
        }
    }
    prefixes.push(decl);
    let (ns, name) = resolve(prefixes, &qname, false)?;
    let mut attrs = Vec::new();
    for (k, v) in raw_attrs {
        if k == "xmlns" || k.starts_with("xmlns:") {
            continue;
        }
        if k.contains(':') {
            let (uri, local) = resolve(prefixes, &k, true)?;
            attrs.push((format!("{{{uri}}}{local}"), v));
        } else {
            attrs.push((k, v));
        }
    }
    let mut el = Elem { ns, name, attrs, text: None, children: Vec::new() };
    if !self_closing {
        let mut text = String::new();
        loop {
            let r = p.rest();
            let lt = r.find('<').ok_or("unclosed element")?;
            let chunk = &r[..lt];
            let chunk = unescape(chunk)?;
            if el.children.is_empty() {
                text.push_str(&chunk);
            }
            p.pos += lt;
            if p.starts("</") {
                p.pos += 2;
                let end = p.name()?;
                if end != qname {
                    return Err("mismatched tag".into());
                }
                p.skip_ws();
                if !p.starts(">") {
                    return Err("not well-formed".into());
                }
                p.pos += 1;
                break;
            } else if p.starts("<!--") {
                p.pos += 4;
                p.skip_to("-->")?;
            } else if p.starts("<![CDATA[") {
                p.pos += 9;
                let i = p.rest().find("]]>").ok_or("unterminated CDATA")?;
                if el.children.is_empty() {
                    text.push_str(&p.rest()[..i]);
                }
                p.pos += i + 3;
            } else if p.starts("<?") {
                p.pos += 2;
                p.skip_to("?>")?;
            } else {
                let child = parse_element(p, prefixes, depth + 1)?;
                el.children.push(child);
            }
        }
        if !text.is_empty() {
            el.text = Some(text);
        }
    }
    prefixes.pop();
    Ok(el)
}

/// `ET.fromstring(data)`: the root element.
pub fn parse(bytes: &[u8]) -> R<Elem> {
    let text = decode(bytes)?;
    let mut p = P { s: &text, pos: 0 };
    let mut prefixes: Vec<HashMap<String, String>> = Vec::new();
    loop {
        p.skip_ws();
        if p.starts("<?") {
            p.pos += 2;
            p.skip_to("?>")?;
        } else if p.starts("<!--") {
            p.pos += 4;
            p.skip_to("-->")?;
        } else if p.starts("<!DOCTYPE") {
            // defusedxml refuses entity declarations; a bare DOCTYPE is fine.
            let r = p.rest();
            let end = if let Some(open) = r.find('[') {
                let gt = r.find('>').unwrap_or(r.len());
                if open < gt {
                    r.find("]>").map(|i| i + 2).ok_or("unterminated DOCTYPE")?
                } else {
                    gt + 1
                }
            } else {
                r.find('>').ok_or("unterminated DOCTYPE")? + 1
            };
            if r[..end.min(r.len())].contains("<!ENTITY") {
                return Err("EntitiesForbidden".into());
            }
            p.pos += end;
        } else if p.starts("<") {
            break;
        } else {
            return Err("syntax error: no element found".into());
        }
    }
    let root = parse_element(&mut p, &mut prefixes, 0)?;
    loop {
        p.skip_ws();
        if p.rest().is_empty() {
            break;
        }
        if p.starts("<!--") {
            p.pos += 4;
            p.skip_to("-->")?;
        } else if p.starts("<?") {
            p.pos += 2;
            p.skip_to("?>")?;
        } else {
            return Err("junk after document element".into());
        }
    }
    Ok(root)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_a_namespaced_material() {
        let x = br#"<?xml version="1.0" encoding="UTF-8"?>
<mat:materials xmlns:mat="urn:m"><mat:material name="A &amp; B" colorRed="1" useTrans="1">
<mat:texture textureFilename="t.png" xScale="2.5"/></mat:material></mat:materials>"#;
        let root = parse(x).unwrap();
        let m = root.descendant("urn:m", "material").unwrap();
        assert_eq!(m.get("name"), Some("A & B"));
        assert_eq!(m.child("urn:m", "texture").unwrap().get("xScale"), Some("2.5"));
        assert!(root.descendant("urn:other", "material").is_none());
    }

    #[test]
    fn refuses_entities_and_bad_documents() {
        assert!(parse(b"<!DOCTYPE a [<!ENTITY x \"y\">]><a>&x;</a>").is_err());
        assert!(parse(b"<a><b></a>").is_err());
        assert!(parse(b"<a/><b/>").is_err());
        assert!(parse(b"<p:a/>").is_err());
        assert_eq!(parse(b"<a>&#65;&#x42;</a>").unwrap().text.as_deref(), Some("AB"));
    }
}
