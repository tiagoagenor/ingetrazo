// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
// Port of openskp (MIT, Copyright (c) 2024-2026 Ahsan Mehmood): see NOTICE-openskp.md.
//! The tag-length-value records of `model.dat` (`_core.parse_tlv_recursive`).
//!
//! A tag is two bytes, written here as the 16-bit number of its hex name
//! (`F401` is `0xF401`, bytes `F4 01`); the length is a little-endian `u32`.

/// The tags whose payload is itself a run of records
/// (`_core.CONTAINER_TAGS`, with the Image placement pair).
pub fn is_container(tag: u16) -> bool {
    matches!(
        tag,
        0xF401 | 0xF701 | 0xD430 | 0xD530 | 0xC832
            | 0x7C15 | 0x8813 | 0x8913 | 0x8A13 | 0x8B13 | 0x8C13 | 0x8D13 | 0x4C1D | 0x6419
            | 0xF901 | 0x7017 | 0x7117 | 0xD007 | 0xC409 | 0x9411 | 0x9511 | 0x0F01
            | 0x384A | 0xB80B | 0x9713 | 0x2C4C | 0xAC0D | 0xAE0D | 0xF601 | 0xF801
            | 0x983A | 0x993A | 0x8C3C | 0x8D3C
            | 0x9013 | 0x401F
    )
}

/// The containers of an entity's attribute blob (`_PROP_CONTAINER_TAGS`).
pub fn is_prop_container(tag: u16) -> bool {
    matches!(tag, 0xDD05 | 0xB536 | 0xB136 | 0xB236 | 0xB336 | 0xB036 | 0xA438 | 0xAE38)
}

/// One record. A container that parsed into children has an empty payload.
#[derive(Debug)]
pub struct Node<'a> {
    pub offset: usize,
    pub tag: u16,
    pub size: usize,
    pub children: Vec<Node<'a>>,
    pub payload: &'a [u8],
}

pub fn u32_at(d: &[u8], at: usize) -> Option<u32> {
    d.get(at..at + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

pub fn f64_at(d: &[u8], at: usize) -> Option<f64> {
    d.get(at..at + 8).map(|b| f64::from_le_bytes(b.try_into().unwrap()))
}

/// `parse_var_int(data, offset, length)`: a little-endian unsigned number of
/// `length` bytes (only the low 8 are kept). `None` where Python raises
/// `IndexError`.
pub fn var_int(d: &[u8], offset: usize, length: usize) -> Option<u64> {
    let mut v = 0u64;
    for i in 0..length {
        let b = *d.get(offset + i)?;
        if i < 8 {
            v |= (b as u64) << (8 * i);
        }
    }
    Some(v)
}

/// How deep records may nest: Python's recursion limit makes a deeper tree a
/// `RecursionError` there, and would overflow the stack here.
pub const MAX_DEPTH: usize = 900;

/// `parse_tlv_recursive(data, start, end, container_tags)`; an `Err` where
/// Python raises `RecursionError`.
pub fn try_parse<'a>(data: &'a [u8], start: usize, end: usize, containers: fn(u16) -> bool) -> Result<Vec<Node<'a>>, String> {
    parse_at(data, start, end, containers, 0)
}

fn parse_at<'a>(data: &'a [u8], start: usize, end: usize, containers: fn(u16) -> bool, depth: usize) -> Result<Vec<Node<'a>>, String> {
    if depth > MAX_DEPTH {
        return Err("RecursionError: maximum recursion depth exceeded".into());
    }
    let mut pos = start;
    let mut out = Vec::new();
    while pos + 6 <= end {
        let tag = u16::from_be_bytes([data[pos], data[pos + 1]]);
        let size = u32::from_le_bytes([data[pos + 2], data[pos + 3], data[pos + 4], data[pos + 5]]) as usize;
        if pos + 6 + size > end {
            break;
        }
        let children = if containers(tag) && size > 0 { parse_at(data, pos + 6, pos + 6 + size, containers, depth + 1)? } else { Vec::new() };
        let payload = if children.is_empty() { &data[pos + 6..pos + 6 + size] } else { &data[0..0] };
        out.push(Node { offset: pos, tag, size, children, payload });
        pos += 6 + size;
    }
    Ok(out)
}

/// [`try_parse`] where a tree that deep is taken as empty (an entity's
/// attribute blob).
pub fn parse<'a>(data: &'a [u8], start: usize, end: usize, containers: fn(u16) -> bool) -> Vec<Node<'a>> {
    try_parse(data, start, end, containers).unwrap_or_default()
}

/// `_flat_headers`: the `(tag, offset, size)` of the direct children only.
pub fn flat_headers(data: &[u8], start: usize, end: usize) -> Vec<(u16, usize, usize)> {
    let mut pos = start;
    let mut out = Vec::new();
    while pos + 6 <= end {
        let tag = u16::from_be_bytes([data[pos], data[pos + 1]]);
        let size = u32::from_le_bytes([data[pos + 2], data[pos + 3], data[pos + 4], data[pos + 5]]) as usize;
        if pos + 6 + size > end {
            break;
        }
        out.push((tag, pos, size));
        pos += 6 + size;
    }
    out
}

/// `_tlv_flat`: `[(tag, body)]` of a payload read as a flat run.
pub fn flat(payload: &[u8]) -> Vec<(u16, &[u8])> {
    let mut pos = 0;
    let mut out = Vec::new();
    while pos + 6 <= payload.len() {
        let tag = u16::from_be_bytes([payload[pos], payload[pos + 1]]);
        let size = u32::from_le_bytes([payload[pos + 2], payload[pos + 3], payload[pos + 4], payload[pos + 5]]) as usize;
        if pos + 6 + size > payload.len() {
            break;
        }
        out.push((tag, &payload[pos + 6..pos + 6 + size]));
        pos += 6 + size;
    }
    out
}

/// `find_child_tag(nodes, target)`: the first node with the tag, depth first.
pub fn find_child_tag<'n, 'a>(nodes: &'n [Node<'a>], target: u16) -> Option<&'n Node<'a>> {
    for n in nodes {
        if n.tag == target {
            return Some(n);
        }
        if let Some(r) = find_child_tag(&n.children, target) {
            return Some(r);
        }
    }
    None
}

/// `find_all_nodes_rec`: every node with the tag, in pre-order (a match is
/// searched inside too).
pub fn find_all<'n, 'a>(nodes: &'n [Node<'a>], target: u16, out: &mut Vec<&'n Node<'a>>) {
    for n in nodes {
        if n.tag == target {
            out.push(n);
        }
        find_all(&n.children, target, out);
    }
}

/// `extract_entity_id(node)`: the persistent id, from a `DE05` or the `DE05`
/// inside a `DC05`, searched through the children.
pub fn extract_entity_id(node: &Node) -> Result<Option<i64>, String> {
    for child in &node.children {
        if child.tag == 0xDE05 {
            return var_int(child.payload, 0, child.payload.len()).map(|v| Some(v as i64)).ok_or_else(|| "IndexError".to_string());
        }
        if child.tag == 0xDC05 {
            let p = child.payload;
            if p.starts_with(&[0xDE, 0x05]) {
                let n = u32_at(p, 2).ok_or("struct.error")? as usize;
                return var_int(p, 6, n).map(|v| Some(v as i64)).ok_or_else(|| "IndexError".to_string());
            }
        }
    }
    for child in &node.children {
        if let Some(r) = extract_entity_id(child)? {
            return Ok(Some(r));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(tag: [u8; 2], body: &[u8]) -> Vec<u8> {
        let mut v = tag.to_vec();
        v.extend((body.len() as u32).to_le_bytes());
        v.extend(body);
        v
    }

    #[test]
    fn nests_containers_and_keeps_leaf_payloads() {
        let leaf = rec([0xDE, 0x05], &[7]);
        let tree = rec([0xC4, 0x09], &leaf);
        let nodes = parse(&tree, 0, tree.len(), is_container);
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].tag, 0xC409);
        assert!(nodes[0].payload.is_empty());
        assert_eq!(nodes[0].children[0].payload, [7]);
        assert_eq!(extract_entity_id(&nodes[0]).unwrap(), Some(7));
    }

    #[test]
    fn a_container_without_records_keeps_its_bytes() {
        let tree = rec([0xC4, 0x09], &[1, 2, 3]);
        let nodes = parse(&tree, 0, tree.len(), is_container);
        assert!(nodes[0].children.is_empty());
        assert_eq!(nodes[0].payload, [1, 2, 3]);
    }

    #[test]
    fn a_record_that_overruns_stops_the_run() {
        let mut t = rec([0x01, 0x02], &[1]);
        t.extend([0x01, 0x02, 9, 0, 0, 0, 1]);
        assert_eq!(parse(&t, 0, t.len(), is_container).len(), 1);
    }

    #[test]
    fn var_ints_are_little_endian() {
        assert_eq!(var_int(&[1, 1], 0, 2), Some(257));
        assert_eq!(var_int(&[1], 0, 2), None);
        assert_eq!(var_int(&[], 0, 0), Some(0));
    }
}
