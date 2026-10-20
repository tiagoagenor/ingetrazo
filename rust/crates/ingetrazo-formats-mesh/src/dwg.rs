// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The DWG to DXF bridge (docs/rust-rewrite-plan.md, M5.6): `formats/dwg_bridge.py`.
//!
//! DWG is never parsed in-process: the LibreDWG `dwg2dxf` command-line tool
//! runs as an external converter and its DXF goes through the DXF importer
//! ([`crate::dxf`]). The Python module reads no DWG header (there is no
//! `AC10xx` table in it): the converter decides what it can open, so none is
//! invented here.
//!
//! Search order for the tool, as in Python: the bundle beside the install
//! (`vendor/libredwg/bin`), the sibling IngeCAD checkout's bundle (a
//! development-machine courtesy), then the system `PATH`. A binary built for
//! another OS is ignored (a plain macOS checkout carries the Linux ELF).
//!
//! The command line is `dwg2dxf -y -o <tmp>/converted.dxf <file.dwg>`, run
//! with [`std::process::Command`] (no shell), a 300 second timeout and the
//! output written in a fresh `ingetrazo-dwg-*` temp directory. The verdict is
//! whether the output exists and is not empty, never the exit code. Two
//! repairs on the converter's DXF follow ([`strip_null_handles`],
//! [`dedupe_handles`]); the Rust reader is not as strict as ezdxf, but they
//! stay: the file the bridge hands over is the same.

use std::collections::HashSet;
use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use ingetrazo_math::Hypot;
use ingetrazo_model::scene::Scene;

use crate::dxf::import::{load_dxf_bytes, DxfStats};
use crate::obj::Stitch;

/// Seconds a conversion may take (`_TIMEOUT`).
pub const TIMEOUT_SECS: u64 = 300;
/// Prefix of the temp directory (`tempfile.mkdtemp(prefix=...)`).
pub const TEMP_PREFIX: &str = "ingetrazo-dwg-";
/// Name of the converted file inside it.
pub const TEMP_NAME: &str = "converted.dxf";

/// The strings the UI shows around the bridge (`views/main_window.py`); a
/// caller translates them with `ingetrazo-i18n`'s `tr`, as Python does.
pub mod messages {
    pub const IMPORT_FAILED_TITLE: &str = "Import DWG failed";
    pub const CONVERTING: &str = "Converting DWG…";
    pub const FILE_FILTER: &str = "AutoCAD DWG (*.dwg);;All files (*)";
    pub const NOT_AVAILABLE_DIALOG: &str = "The LibreDWG converter (dwg2dxf) is not available in this installation.";
    pub const NOT_AVAILABLE_HINT: &str = "To work on the drawing anyway: save it as DXF from your CAD program (or convert it with a free DWG → DXF converter such as ODA File Converter), then File ▸ Import ▸ AutoCAD DXF.";
    /// What `dwg_to_dxf` raises when no tool is found.
    pub const NOT_AVAILABLE: &str = "LibreDWG (dwg2dxf) is not available";
}

/// `DwgBridgeError`: a conversion failed or no converter is available. Its
/// `Display` is the message Python's exception carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DwgBridgeError(pub String);

impl fmt::Display for DwgBridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DwgBridgeError {}

fn err<T>(msg: impl Into<String>) -> Result<T, DwgBridgeError> {
    Err(DwgBridgeError(msg.into()))
}

/// `sys.platform` as the bridge sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    Windows,
    Linux,
    MacOs,
    /// Any other OS: every file counts as native.
    Other,
}

impl Platform {
    pub fn current() -> Platform {
        if cfg!(target_os = "windows") {
            Platform::Windows
        } else if cfg!(target_os = "macos") {
            Platform::MacOs
        } else if cfg!(target_os = "linux") {
            Platform::Linux
        } else {
            Platform::Other
        }
    }

    /// `_NATIVE_MAGIC[sys.platform]`.
    pub fn native_magic(self) -> &'static [&'static [u8]] {
        match self {
            Platform::MacOs => &[b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xfe\xed\xfa\xce", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca"],
            Platform::Linux => &[b"\x7fELF"],
            Platform::Windows => &[b"MZ"],
            Platform::Other => &[],
        }
    }
}

/// `_is_native_executable`: the file starts with the magic of the platform.
/// An unknown platform accepts anything; an unreadable file is rejected.
pub fn is_native_executable(path: &Path, platform: Platform) -> bool {
    let magic = platform.native_magic();
    if magic.is_empty() {
        return true;
    }
    let mut head = Vec::new();
    match std::fs::File::open(path).and_then(|f| f.take(4).read_to_end(&mut head)) {
        Ok(_) => magic.iter().any(|m| head.starts_with(m)),
        Err(_) => false,
    }
}

/// Where the bridge looks for tools and how long it waits for them.
#[derive(Debug, Clone)]
pub struct DwgBridge {
    /// `vendor/libredwg/bin` of the install.
    pub vendor_bin: PathBuf,
    /// The sibling IngeCAD checkout's bundle.
    pub sibling_bin: PathBuf,
    /// Directories of the system `PATH` (`shutil.which`).
    pub path_dirs: Vec<PathBuf>,
    /// Extensions Windows tries on a bare name (`PATHEXT`).
    pub path_ext: Vec<String>,
    pub platform: Platform,
    pub timeout: Duration,
}

impl DwgBridge {
    /// The Python defaults for an install rooted at `app_root` (the directory
    /// holding `resources/` and `i18n/`), with the process `PATH`.
    pub fn from_app_root(app_root: &Path) -> DwgBridge {
        let platform = Platform::current();
        let path_dirs = std::env::var_os("PATH").map(|p| std::env::split_paths(&p).collect()).unwrap_or_default();
        let path_ext = if platform == Platform::Windows {
            std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_string()).split(';').filter(|s| !s.is_empty()).map(str::to_string).collect()
        } else {
            Vec::new()
        };
        // app_root().parent.parent / "ingecad" / "vendor" / "libredwg" / "bin"
        let grand = app_root.parent().and_then(Path::parent).unwrap_or(app_root);
        DwgBridge {
            vendor_bin: app_root.join("vendor").join("libredwg").join("bin"),
            sibling_bin: grand.join("ingecad").join("vendor").join("libredwg").join("bin"),
            path_dirs,
            path_ext,
            platform,
            timeout: Duration::from_secs(TIMEOUT_SECS),
        }
    }

    /// `shutil.which(name)` over [`Self::path_dirs`]. Unlike Python on
    /// Windows, the current directory is not searched first.
    fn which(&self, name: &str) -> Option<PathBuf> {
        let candidates: Vec<String> = if self.platform == Platform::Windows {
            if Path::new(name).extension().is_some() && self.path_ext.iter().any(|e| name.to_lowercase().ends_with(&e.to_lowercase())) {
                vec![name.to_string()]
            } else {
                self.path_ext.iter().map(|e| format!("{name}{e}")).collect()
            }
        } else {
            vec![name.to_string()]
        };
        for dir in &self.path_dirs {
            for c in &candidates {
                let p = dir.join(c);
                if is_executable_file(&p) {
                    return Some(p);
                }
            }
        }
        None
    }

    /// `_find_tool`: Windows ships `<name>.exe` (with its DLLs beside it),
    /// everywhere else the bare name.
    pub fn find_tool(&self, name: &str) -> Option<PathBuf> {
        let exe = if self.platform == Platform::Windows { format!("{name}.exe") } else { name.to_string() };
        for base in [&self.vendor_bin, &self.sibling_bin] {
            let bundled = base.join(&exe);
            if bundled.is_file() && is_native_executable(&bundled, self.platform) {
                return Some(bundled);
            }
        }
        if let Some(system) = self.which(name) {
            if is_native_executable(&system, self.platform) {
                return Some(system);
            }
        }
        None
    }

    pub fn find_dwg2dxf(&self) -> Option<PathBuf> {
        self.find_tool("dwg2dxf")
    }

    pub fn have_dwg_support(&self) -> bool {
        self.find_dwg2dxf().is_some()
    }

    /// `_run`: run the converter and return its stderr so the caller can
    /// inspect warnings. The output's existence is the verdict: LibreDWG often
    /// exits non-zero on recoverable warnings while still writing a usable file.
    pub fn run(&self, cmd: &[String], out_path: &Path) -> Result<String, DwgBridgeError> {
        let Some((program, args)) = cmd.split_first() else {
            return err("converter timed out: ");
        };
        let mut child = match Command::new(program).args(args).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
            Ok(c) => c,
            Err(e) => return err(format!("could not run the converter {program}: {e}")),
        };
        // Drain both pipes on threads so a chatty converter never blocks.
        let mut so = child.stdout.take().expect("piped");
        let mut se = child.stderr.take().expect("piped");
        let t_out = std::thread::spawn(move || {
            let mut b = Vec::new();
            let _ = so.read_to_end(&mut b);
            b
        });
        let t_err = std::thread::spawn(move || {
            let mut b = Vec::new();
            let _ = se.read_to_end(&mut b);
            b
        });
        let start = Instant::now();
        let status = loop {
            match child.try_wait() {
                Ok(Some(s)) => break s,
                Ok(None) => {
                    if start.elapsed() >= self.timeout {
                        let _ = child.kill();
                        let _ = child.wait();
                        // The pipes close with the child; the readers end.
                        let _ = t_out.join();
                        let _ = t_err.join();
                        return err(format!("converter timed out: {}", cmd.join(" ")));
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(e) => return err(format!("could not wait for the converter: {e}")),
            }
        };
        // The converter echoes names from the drawing into its warnings in the
        // file's own codepage, not UTF-8; its log is diagnostics, never a
        // reason to fail.
        let stdout = String::from_utf8_lossy(&t_out.join().unwrap_or_default()).into_owned();
        let stderr = String::from_utf8_lossy(&t_err.join().unwrap_or_default()).into_owned();
        let usable = std::fs::metadata(out_path).map(|m| m.is_file() && m.len() > 0).unwrap_or(false);
        if !usable {
            let src = if !stderr.is_empty() {
                &stderr
            } else {
                &stdout
            };
            let lines = py_splitlines(py_strip(src));
            let tail = &lines[lines.len().saturating_sub(3)..];
            let rc = status.code().map(|c| c.to_string()).unwrap_or_else(|| "-1".to_string());
            let detail = if tail.is_empty() { format!("rc={rc}") } else { tail.join(" | ") };
            return err(format!("conversion produced no output: {detail}"));
        }
        Ok(stderr)
    }

    /// `dwg_to_dxf`: convert a DWG to a temporary DXF and return its path. The
    /// temp file lands in a fresh ASCII-only directory (argv encoding is a
    /// known gotcha family), so the output side stays plain even when the
    /// drawing's name carries accents. Release it with [`discard_temp_dxf`].
    pub fn dwg_to_dxf(&self, dwg_path: &Path) -> Result<PathBuf, DwgBridgeError> {
        let Some(tool) = self.find_dwg2dxf() else {
            return err(messages::NOT_AVAILABLE);
        };
        let out_dir = make_temp_dir().map_err(|e| DwgBridgeError(format!("could not create a temporary directory: {e}")))?;
        let out_dxf = out_dir.join(TEMP_NAME);
        let cmd = convert_command(&tool, &out_dxf, dwg_path);
        let result = self.run(&cmd, &out_dxf).and_then(|_| {
            strip_null_handles(&out_dxf).map_err(|e| DwgBridgeError(format!("could not repair the converted DXF: {e}")))?;
            dedupe_handles(&out_dxf).map_err(|e| DwgBridgeError(format!("could not repair the converted DXF: {e}")))?;
            Ok(())
        });
        match result {
            Ok(()) => Ok(out_dxf),
            Err(e) => {
                // Python leaks the directory on failure; there is nothing in
                // it worth keeping.
                let _ = std::fs::remove_dir_all(&out_dir);
                Err(e)
            }
        }
    }

    /// The whole import the main window runs: convert, feed the DXF importer
    /// with the user's chosen `scale`, discard the temp. The group is named
    /// after the DWG (`name=shown.stem`), not after `converted`.
    pub fn import_dwg(&self, scene: &mut Scene, dwg_path: &Path, scale: f64, h: Hypot, stitch: Option<Stitch>) -> Result<DxfStats, DwgBridgeError> {
        let dxf = self.dwg_to_dxf(dwg_path)?;
        let result = std::fs::read(&dxf).map_err(|e| DwgBridgeError(e.to_string())).and_then(|data| {
            let stem = dwg_path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            let temp_stem = Path::new(TEMP_NAME).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            load_dxf_bytes(scene, &data, &temp_stem, scale, Some(&stem), h, stitch).map_err(DwgBridgeError)
        });
        discard_temp_dxf(&dxf);
        result
    }
}

/// The converter's argv: `[tool, "-y", "-o", out, dwg]`.
pub fn convert_command(tool: &Path, out_dxf: &Path, dwg: &Path) -> Vec<String> {
    vec![tool.to_string_lossy().into_owned(), "-y".to_string(), "-o".to_string(), out_dxf.to_string_lossy().into_owned(), dwg.to_string_lossy().into_owned()]
}

fn is_executable_file(p: &Path) -> bool {
    if !p.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(p).map(|m| m.permissions().mode() & 0o111 != 0).unwrap_or(false)
    }
    #[cfg(not(unix))]
    {
        true
    }
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// `tempfile.mkdtemp(prefix="ingetrazo-dwg-")`: a new, empty directory.
fn make_temp_dir() -> std::io::Result<PathBuf> {
    let base = std::env::temp_dir();
    let mut last = None;
    for _ in 0..100 {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        let n = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = base.join(format!("{TEMP_PREFIX}{:x}{:x}{:x}", std::process::id(), nanos & 0xffff_ffff, n));
        match std::fs::create_dir(&dir) {
            Ok(()) => {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let _ = std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700));
                }
                return Ok(dir);
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => last = Some(e),
            Err(e) => return Err(e),
        }
    }
    Err(last.unwrap_or_else(|| std::io::Error::other("no free temp name")))
}

/// `discard_temp_dxf`: remove the directory [`DwgBridge::dwg_to_dxf`] made,
/// recognised by prefix and no other way (fixture paths must survive).
pub fn discard_temp_dxf(dxf: &Path) {
    if let Some(parent) = dxf.parent() {
        if parent.file_name().is_some_and(|n| n.to_string_lossy().starts_with(TEMP_PREFIX)) {
            let _ = std::fs::remove_dir_all(parent);
        }
    }
}

// ---- Python string helpers ----------------------------------------------------

/// Whitespace of Python's `str.strip()` / `isspace()` on a Latin-1 or wider char.
fn py_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\x0b' | '\x0c' | '\r' | '\x1c'..='\x1f' | '\u{85}' | '\u{a0}') || (c > '\u{ff}' && c.is_whitespace())
}

/// `str.strip()`.
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(py_space)
}

/// `str.splitlines()`.
pub fn py_splitlines(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut it = s.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        if matches!(c, '\n' | '\r' | '\x0b' | '\x0c' | '\x1c' | '\x1d' | '\x1e' | '\u{85}' | '\u{2028}' | '\u{2029}') {
            out.push(&s[start..i]);
            let mut end = i + c.len_utf8();
            if c == '\r' {
                if let Some(&(j, '\n')) = it.peek() {
                    it.next();
                    end = j + 1;
                }
            }
            start = end;
        }
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

// ---- The two DXF repairs -------------------------------------------------------

/// `_read_dxf_lines`: split on `\n` only, latin-1 (one char per byte).
fn read_dxf_lines(path: &Path) -> std::io::Result<Vec<String>> {
    let bytes = std::fs::read(path)?;
    Ok(bytes.iter().map(|&b| b as char).collect::<String>().split('\n').map(str::to_string).collect())
}

fn write_dxf_lines(path: &Path, lines: &[String]) -> std::io::Result<()> {
    let joined = lines.join("\n");
    // Every char is below 256 (read from bytes, or an ASCII handle).
    let bytes: Vec<u8> = joined.chars().map(|c| c as u32 as u8).collect();
    std::fs::write(path, bytes)
}

/// `_strip_null_handles`: drop `(5, 0)` tag pairs. LibreDWG 0.14 emits ENDBLK
/// entities with handle 0, which ezdxf rejects even in recover mode.
pub fn strip_null_handles(path: &Path) -> std::io::Result<()> {
    let lines = read_dxf_lines(path)?;
    let (out, dropped) = strip_null_handles_lines(&lines);
    if dropped > 0 {
        write_dxf_lines(path, &out)?;
    }
    Ok(())
}

fn strip_null_handles_lines(lines: &[String]) -> (Vec<String>, usize) {
    let mut out = Vec::with_capacity(lines.len());
    let mut dropped = 0;
    let mut i = 0;
    while i + 1 < lines.len() {
        if py_strip(&lines[i]) == "5" && py_strip(&lines[i + 1]) == "0" {
            dropped += 1;
            i += 2;
            continue;
        }
        out.push(lines[i].clone());
        out.push(lines[i + 1].clone());
        i += 2;
    }
    out.extend(lines[i..].iter().cloned());
    (out, dropped)
}

/// Sections whose objects carry a handle in group 5. HEADER is out on purpose:
/// there group 5 is the value of `$HANDSEED`.
pub const HANDLE_SECTIONS: [&str; 4] = ["TABLES", "BLOCKS", "ENTITIES", "OBJECTS"];

/// Handles above this are already corrupt.
pub const MAX_SANE_HANDLE: i128 = 1 << 32;

/// Python's `int(s, 16)` on an already stripped string; a value that does not
/// fit saturates (it is only ever compared with [`MAX_SANE_HANDLE`]).
fn py_int16(s: &str) -> Option<i128> {
    let (neg, rest) = match s.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    let rest = rest.strip_prefix("0x").or_else(|| rest.strip_prefix("0X")).unwrap_or(rest);
    if rest.is_empty() || rest.starts_with('_') || rest.ends_with('_') || rest.contains("__") {
        return None;
    }
    let mut v: i128 = 0;
    for c in rest.chars() {
        if c == '_' {
            continue;
        }
        let d = c.to_digit(16)?;
        v = v.saturating_mul(16).saturating_add(d as i128);
    }
    Some(if neg { -v } else { v })
}

/// `_dedupe_handles`: give a fresh handle to objects that reuse one; returns
/// how many. LibreDWG emits some objects (LAYOUT, GROUP...) with a handle that
/// already belongs to a table record (LibreDWG#1356). The first user keeps the
/// handle, later claimants are renumbered above every handle in the file.
pub fn dedupe_handles(path: &Path) -> std::io::Result<usize> {
    let mut lines = read_dxf_lines(path)?;
    let n = dedupe_handles_lines(&mut lines);
    if n > 0 {
        write_dxf_lines(path, &lines)?;
    }
    Ok(n)
}

/// The in-memory body of [`dedupe_handles`]; `lines` are rewritten only when
/// there are collisions. Returns the number of collisions.
pub fn dedupe_handles_lines(lines: &mut [String]) -> usize {
    let mut spots: Vec<(usize, String)> = Vec::new();
    let mut seed_at: Option<usize> = None;
    let mut section: Option<String> = None;
    let mut expecting = false;
    let mut i = 0;
    while i + 1 < lines.len() {
        let code = py_strip(&lines[i]);
        let value = py_strip(&lines[i + 1]);
        if code == "0" {
            if value == "ENDSEC" {
                section = None;
            }
            expecting = !matches!(value, "SECTION" | "ENDSEC" | "EOF");
        } else if code == "2" && section.is_none() {
            section = Some(value.to_string());
        } else if code == "9" && value == "$HANDSEED" && i + 3 < lines.len() {
            seed_at = if py_strip(&lines[i + 2]) == "5" { Some(i + 3) } else { None };
        } else if code == "5" && expecting && section.as_deref().is_some_and(|s| HANDLE_SECTIONS.contains(&s)) {
            spots.push((i + 1, value.to_string()));
            expecting = false;
        }
        i += 2;
    }

    let mut seen: HashSet<&str> = HashSet::new();
    let mut collisions: Vec<usize> = Vec::new();
    for (idx, handle) in &spots {
        if !seen.insert(handle.as_str()) {
            collisions.push(*idx);
        }
    }
    if collisions.is_empty() {
        return 0;
    }

    let mut used: HashSet<String> = spots.iter().map(|(_, h)| h.clone()).collect();
    let max_sane = spots.iter().filter_map(|(_, h)| py_int16(h)).filter(|&n| n < MAX_SANE_HANDLE).max();
    let mut nxt: i128 = max_sane.unwrap_or(0) + 1;
    for idx in &collisions {
        // Python's format(n, "X") of a negative int would be "-3"; a fresh
        // handle is never below 1 in practice, but follow it.
        while used.contains(&fmt_hex_upper(nxt)) {
            nxt += 1;
        }
        let fresh = fmt_hex_upper(nxt);
        used.insert(fresh.clone());
        lines[*idx] = fresh;
        nxt += 1;
    }
    if let Some(at) = seed_at {
        lines[at] = fmt_hex_upper(nxt);
    }
    collisions.len()
}

fn fmt_hex_upper(n: i128) -> String {
    if n < 0 {
        format!("-{:X}", -n)
    } else {
        format!("{n:X}")
    }
}
