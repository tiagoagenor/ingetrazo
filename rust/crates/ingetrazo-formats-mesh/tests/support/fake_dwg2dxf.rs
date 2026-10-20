// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! A stand-in for LibreDWG's `dwg2dxf`, for the bridge tests only. Called as
//! `dwg2dxf -y -o <out> <in>`; the "DWG" it reads is a little script of
//! `key=value` lines: `stdout=<hex>`, `stderr=<hex>`, `write=<hex>` (the bytes
//! of the output file; absent writes none), `sleep=<ms>`, `rc=<n>`,
//! `argv=<path>` (record the arguments, one per line, in that file).

use std::io::Write;

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = args.iter().position(|a| a == "-o").and_then(|i| args.get(i + 1)).expect("-o <out>");
    let input = args.last().expect("input");
    let spec = std::fs::read_to_string(input).unwrap_or_default();
    let mut rc = 0;
    for line in spec.lines() {
        let Some((k, v)) = line.split_once('=') else { continue };
        match k {
            "stdout" => std::io::stdout().write_all(&unhex(v)).unwrap(),
            "stderr" => std::io::stderr().write_all(&unhex(v)).unwrap(),
            "write" => std::fs::write(out, unhex(v)).unwrap(),
            "sleep" => std::thread::sleep(std::time::Duration::from_millis(v.parse().unwrap())),
            "rc" => rc = v.parse().unwrap(),
            "argv" => std::fs::write(v, args.join("\n")).unwrap(),
            _ => {}
        }
    }
    std::process::exit(rc);
}
