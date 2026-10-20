// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `ingetrazo-cli`: IngeTrazo without a window.
//!
//!     ingetrazo-cli golden replay [--root golden] [--id <id>] [--verbose]
//!     ingetrazo-cli golden sig <file.igz>
//!
//! Replays the golden corpus as far as the Rust engine goes and prints,
//! per script, the first point where it parts from Python; then a summary
//! of what blocks the most scripts. Exit status 1 if a script is broken or
//! its start state differs (a real divergence), 0 otherwise.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;

use ingetrazo_golden::{file_signature, ids, load, replay, Outcome};

fn usage() -> ExitCode {
    eprintln!("usage: ingetrazo-cli golden replay [--root <golden dir>] [--id <id>] [--verbose]");
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() == 3 && args[0] == "golden" && args[1] == "sig" {
        // The L2 signature of an .igz, as JSON (to diff against Python's).
        return match std::fs::read(&args[2]).map_err(|e| e.to_string()).and_then(|raw| file_signature(&raw)) {
            Ok(sig) => {
                println!("{}", ingetrazo_golden::sig::canon_dumps(&sig));
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("{e}");
                ExitCode::from(1)
            }
        };
    }
    if args.len() < 2 || args[0] != "golden" || args[1] != "replay" {
        return usage();
    }
    let mut root = PathBuf::from("golden");
    let mut only: Option<String> = None;
    let mut verbose = false;
    let mut it = args[2..].iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--root" => match it.next() {
                Some(r) => root = PathBuf::from(r),
                None => return usage(),
            },
            "--id" => match it.next() {
                Some(i) => only = Some(i.clone()),
                None => return usage(),
            },
            "--verbose" | "-v" => verbose = true,
            _ => return usage(),
        }
    }
    let all = match only {
        Some(i) => vec![i],
        None => ids(&root),
    };
    if all.is_empty() {
        eprintln!("no scripts under {}", root.join("ops").display());
        return ExitCode::from(2);
    }
    let (mut ok, mut start_ok, mut bad, mut further) = (0usize, 0usize, 0usize, 0usize);
    let mut blockers: BTreeMap<String, usize> = BTreeMap::new();
    let mut parts: BTreeMap<String, usize> = BTreeMap::new();
    for id in &all {
        let outcome = match load(&root, id) {
            Ok(s) => replay(&s),
            Err(e) => Outcome::Broken(e),
        };
        match &outcome {
            Outcome::Ok => {
                ok += 1;
                start_ok += 1;
            }
            Outcome::Unsupported { at, op } => {
                start_ok += 1;
                if *at > 0 {
                    further += 1;
                }
                *blockers.entry(op.clone()).or_default() += 1;
            }
            Outcome::Diverges { .. } => {
                start_ok += 1;
                bad += 1;
            }
            Outcome::Start(ps) => {
                bad += 1;
                for p in ps {
                    *parts.entry(p.clone()).or_default() += 1;
                }
            }
            Outcome::Broken(_) => bad += 1,
        }
        let real = matches!(outcome, Outcome::Start(_) | Outcome::Broken(_) | Outcome::Diverges { .. });
        if verbose || real {
            match &outcome {
                Outcome::Ok => println!("ok        {id}"),
                Outcome::Unsupported { at, op } => println!("stops     {id}: step {at}, op {op} not in Rust yet"),
                Outcome::Diverges { at, op, why } => println!("DIVERGES  {id}: step {at}, op {op}: {why}"),
                Outcome::Start(ps) => println!("DIFFERS   {id}: start signature ({})", ps.join(", ")),
                Outcome::Broken(e) => println!("BROKEN    {id}: {e}"),
            }
        }
    }
    println!();
    println!("{} scripts: {ok} replay to the end, {further} more run some steps before an op the engine lacks, {start_ok} start state identical, {bad} differ or broken", all.len());
    if !parts.is_empty() {
        println!("start signature parts that differ: {parts:?}");
    }
    let mut ranked: Vec<(&String, &usize)> = blockers.iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    if !ranked.is_empty() {
        println!("first op the Rust engine lacks, by scripts it blocks:");
        for (op, n) in ranked.iter().take(25) {
            println!("  {n:5}  {op}");
        }
    }
    if bad > 0 {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}
