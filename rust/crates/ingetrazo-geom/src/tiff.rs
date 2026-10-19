// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `georef.geotiff`'s strip/tile decompressors, byte for byte: TIFF LZW
//! (`_lzw_decode`: MSB-first codes, early change, a corrupt stream keeps
//! what was decoded) and PackBits (`_unpackbits`).

/// `_lzw_decode`. Every decoded entry lands contiguous in the output, so
/// the table holds (start, length) spans of the output itself: the entry
/// "previous + first byte of this one" is exactly the previous span grown
/// by one. Codes below 256 are their byte; 256 and 257 are control codes.
pub fn lzw_decode(data: &[u8]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(data.len() * 3);
    let mut table: Vec<(usize, usize)> = Vec::with_capacity(4096);
    let mut bits: u32 = 9;
    let mut prev: Option<(usize, usize)> = None;
    let mut acc: u64 = 0;
    let mut nacc: u32 = 0;
    let mut pos = 0usize;
    let n = data.len();
    loop {
        while nacc < bits && pos < n {
            acc = (acc << 8) | data[pos] as u64;
            pos += 1;
            nacc += 8;
        }
        if nacc < bits {
            break;
        }
        let code = ((acc >> (nacc - bits)) & ((1u64 << bits) - 1)) as usize;
        nacc -= bits;
        acc &= (1u64 << nacc) - 1;
        if code == 256 {
            table.clear();
            bits = 9;
            prev = None;
            continue;
        }
        if code == 257 {
            break;
        }
        let start = out.len();
        let size = 258 + table.len(); // the Python's len(table)
        if code < 256 {
            out.push(code as u8);
        } else if code < size {
            let (s, l) = table[code - 258];
            out.extend_from_within(s..s + l);
        } else if let (Some((ps, pl)), true) = (prev, code == size) {
            let first = out[ps];
            out.extend_from_within(ps..ps + pl);
            out.push(first);
        } else {
            break;
        }
        let len = out.len() - start;
        if let Some((ps, pl)) = prev {
            // prev + this entry's first byte: the span right before it, + 1.
            table.push((ps, pl + 1));
        }
        prev = Some((start, len));
        if 258 + table.len() + 1 >= (1usize << bits) && bits < 12 {
            bits += 1;
        }
    }
    out
}

/// `_unpackbits`.
pub fn unpackbits(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 2);
    let (mut i, n) = (0usize, data.len());
    while i < n {
        let h = data[i] as usize;
        i += 1;
        if h < 128 {
            let end = (i + h + 1).min(n);
            out.extend_from_slice(&data[i.min(n)..end]);
            i += h + 1;
        } else if h > 128 {
            if i < n {
                out.extend(std::iter::repeat_n(data[i], 257 - h));
            }
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packbits_literal_and_run() {
        assert_eq!(unpackbits(&[2, 1, 2, 3, 254, 7, 128]), vec![1, 2, 3, 7, 7, 7]);
    }

    #[test]
    fn lzw_stops_at_end_of_information() {
        // 9-bit codes: 256 (clear), 'A' (65), 257 (end), MSB first.
        let codes = [256u32, 65, 257];
        let mut bits = Vec::new();
        for c in codes {
            for k in (0..9).rev() {
                bits.push((c >> k) & 1);
            }
        }
        while bits.len() % 8 != 0 {
            bits.push(0);
        }
        let bytes: Vec<u8> = bits.chunks(8).map(|ch| ch.iter().fold(0u8, |a, &b| (a << 1) | b as u8)).collect();
        assert_eq!(lzw_decode(&bytes), b"A".to_vec());
    }
}
