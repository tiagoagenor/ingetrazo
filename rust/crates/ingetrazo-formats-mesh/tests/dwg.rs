// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The DWG bridge against `formats/dwg_bridge.py`
//! (scripts/golden/make_dwg_fixtures.py): constants, messages, the native
//! magic table, the two DXF repairs byte for byte and the messages raised for
//! converter outcomes. The converter is `fake-dwg2dxf` (tests/support), a tiny
//! compiled helper driven by the "DWG" it is given; a real `dwg2dxf` is only
//! used when `INGETRAZO_DWG_E2E` names a `.dwg` (see the last test).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_formats_mesh::dwg::{convert_command, dedupe_handles, discard_temp_dxf, is_native_executable, messages, strip_null_handles, DwgBridge, Platform, TEMP_NAME, TEMP_PREFIX, TIMEOUT_SECS};
use ingetrazo_math::Hypot;
use ingetrazo_model::scene::Scene;

fn fixture() -> Value {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/dwg/cases.json");
    loads(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn arr(v: &Value) -> &[Value] {
    match v {
        Value::Array(a) => a,
        other => panic!("{other:?}"),
    }
}

fn s(v: &Value) -> &str {
    match v {
        Value::Str(x) => x,
        other => panic!("{other:?}"),
    }
}

fn unhex(h: &str) -> Vec<u8> {
    (0..h.len() / 2).map(|i| u8::from_str_radix(&h[2 * i..2 * i + 2], 16).unwrap()).collect()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

static N: AtomicU32 = AtomicU32::new(0);

/// A fresh scratch directory for one test.
fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ingetrazo-dwgtest-{}-{}-{tag}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn fake_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_fake-dwg2dxf"))
}

/// Copy the fake converter in as `<dir>/dwg2dxf[.exe]`.
fn install_fake(dir: &Path, platform: Platform) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    let name = if platform == Platform::Windows { "dwg2dxf.exe" } else { "dwg2dxf" };
    let to = dir.join(name);
    std::fs::copy(fake_exe(), &to).unwrap();
    to
}

/// A bridge that sees only `vendor` (and the given PATH dirs).
fn bridge(root: &Path, path_dirs: Vec<PathBuf>) -> DwgBridge {
    DwgBridge { vendor_bin: root.join("vendor"), sibling_bin: root.join("sibling"), path_dirs, path_ext: vec![".exe".to_string()], platform: Platform::current(), timeout: Duration::from_secs(60) }
}

fn spec(lines: &[String]) -> String {
    lines.join("\n") + "\n"
}

// ---- parity with the Python module --------------------------------------------

#[test]
fn constants_and_messages_match_python() {
    let f = fixture();
    assert_eq!(f.get("timeout_seconds").unwrap().as_i64(), Some(TIMEOUT_SECS as i64));
    assert_eq!(s(f.get("temp_prefix").unwrap()), TEMP_PREFIX);
    assert_eq!(s(f.get("temp_name").unwrap()), TEMP_NAME);
    assert_eq!(s(f.get("not_available").unwrap()), messages::NOT_AVAILABLE);
    let max = f.get("max_sane_handle").unwrap().as_i64().unwrap();
    assert_eq!(max as i128, ingetrazo_formats_mesh::dwg::MAX_SANE_HANDLE);
    let secs: Vec<&str> = arr(f.get("handle_sections").unwrap()).iter().map(s).collect();
    let mut mine = ingetrazo_formats_mesh::dwg::HANDLE_SECTIONS.to_vec();
    mine.sort_unstable();
    assert_eq!(secs, mine);
    // The default search locations: vendor/libredwg/bin and the IngeCAD sibling.
    let root = Path::new("/a/b/c/ingetrazo");
    let b = DwgBridge::from_app_root(root);
    let join = |base: &Path, rel: &Value| arr(rel).iter().fold(base.to_path_buf(), |p, x| p.join(s(x)));
    assert_eq!(b.vendor_bin, join(root, f.get("vendor_rel").unwrap()));
    assert_eq!(b.sibling_bin, join(Path::new("/a/b"), f.get("sibling_rel").unwrap()));
    assert_eq!(b.timeout, Duration::from_secs(TIMEOUT_SECS));
}

#[test]
fn the_ui_strings_are_the_python_sources() {
    // The sources are asserted present in views/main_window.py by the fixture
    // script; here they are pinned character for character.
    assert_eq!(messages::IMPORT_FAILED_TITLE, "Import DWG failed");
    assert_eq!(messages::CONVERTING, "Converting DWG\u{2026}");
    assert_eq!(messages::FILE_FILTER, "AutoCAD DWG (*.dwg);;All files (*)");
    assert_eq!(messages::NOT_AVAILABLE_DIALOG, "The LibreDWG converter (dwg2dxf) is not available in this installation.");
    assert_eq!(messages::NOT_AVAILABLE_HINT, "To work on the drawing anyway: save it as DXF from your CAD program (or convert it with a free DWG \u{2192} DXF converter such as ODA File Converter), then File \u{25B8} Import \u{25B8} AutoCAD DXF.");
}

#[test]
fn native_magic_table_matches_python() {
    let f = fixture();
    let table = f.get("native_magic").unwrap();
    for (key, platform) in [("darwin", Platform::MacOs), ("linux", Platform::Linux), ("win32", Platform::Windows)] {
        let want: Vec<Vec<u8>> = arr(table.get(key).unwrap()).iter().map(|h| unhex(s(h))).collect();
        let got: Vec<Vec<u8>> = platform.native_magic().iter().map(|m| m.to_vec()).collect();
        assert_eq!(got, want, "{key}");
    }
}

#[test]
fn the_dxf_repairs_match_python_byte_for_byte() {
    let dir = scratch("repairs");
    let f = fixture();
    let cases = arr(f.get("repairs").unwrap());
    assert!(cases.len() >= 10);
    for c in cases {
        let name = s(c.get("name").unwrap());
        let input = unhex(s(c.get("input").unwrap()));
        let p = dir.join("r.dxf");
        std::fs::write(&p, &input).unwrap();
        strip_null_handles(&p).unwrap();
        assert_eq!(hex(&std::fs::read(&p).unwrap()), s(c.get("stripped").unwrap()), "strip {name}");
        std::fs::write(&p, &input).unwrap();
        let n = dedupe_handles(&p).unwrap();
        assert_eq!(n as i64, c.get("dedupe_count").unwrap().as_i64().unwrap(), "count {name}");
        assert_eq!(hex(&std::fs::read(&p).unwrap()), s(c.get("deduped").unwrap()), "dedupe {name}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn converter_outcomes_map_to_python_messages() {
    let dir = scratch("runs");
    let b = bridge(&dir, vec![]);
    let f = fixture();
    for c in arr(f.get("runs").unwrap()) {
        let name = s(c.get("name").unwrap());
        let out = dir.join("o.dxf");
        let _ = std::fs::remove_file(&out);
        let mut lines = vec![format!("stdout={}", s(c.get("stdout").unwrap())), format!("stderr={}", s(c.get("stderr").unwrap())), format!("rc={}", c.get("rc").unwrap().as_i64().unwrap())];
        if let Some(Value::Str(w)) = c.get("write") {
            lines.push(format!("write={w}"));
        }
        let input = dir.join("in.dwg");
        std::fs::write(&input, spec(&lines)).unwrap();
        let cmd = convert_command(&fake_exe(), &out, &input);
        let got = b.run(&cmd, &out);
        match c.get("ok") {
            Some(Value::Bool(true)) => {
                let want = String::from_utf8(unhex(s(c.get("stderr").unwrap()))).unwrap();
                assert_eq!(got.unwrap(), want, "{name}");
            }
            _ => assert_eq!(got.unwrap_err().to_string(), s(c.get("message").unwrap()), "{name}"),
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// ---- timeout, arguments, discovery ----------------------------------------------

#[test]
fn a_slow_converter_times_out_with_the_python_message() {
    let dir = scratch("timeout");
    let mut b = bridge(&dir, vec![]);
    b.timeout = Duration::from_millis(400);
    let out = dir.join("o.dxf");
    let input = dir.join("in.dwg");
    std::fs::write(&input, "sleep=30000\nwrite=30\n").unwrap();
    let cmd = convert_command(&fake_exe(), &out, &input);
    let t0 = std::time::Instant::now();
    let e = b.run(&cmd, &out).unwrap_err();
    assert!(t0.elapsed() < Duration::from_secs(20), "the child must be killed");
    let want = s(fixture().get("timed_out_template").unwrap()).replace("{cmd}", &cmd.join(" "));
    assert_eq!(e.to_string(), want);
    assert!(e.to_string().starts_with("converter timed out: "));
    // The Python sample has the same shape: the executable, then the arguments.
    assert!(s(fixture().get("timed_out_sample").unwrap()).starts_with("converter timed out: <exe> "));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_command_line_is_y_o_out_in_with_no_shell() {
    let tool = Path::new("/opt/libredwg/bin/dwg2dxf");
    let out = Path::new("/tmp/ingetrazo-dwg-x/converted.dxf");
    let dwg = Path::new("/home/u/plan with spaces; rm -rf.dwg");
    assert_eq!(convert_command(tool, out, dwg), vec![tool.to_string_lossy().into_owned(), "-y".to_string(), "-o".to_string(), out.to_string_lossy().into_owned(), dwg.to_string_lossy().into_owned()]);

    // And what the process really receives: one argv entry per element.
    let dir = scratch("argv");
    let record = dir.join("argv.txt");
    let input = dir.join("odd name; & echo.dwg");
    std::fs::write(&input, format!("argv={}\nwrite=30\n", record.display())).unwrap();
    let b = bridge(&dir, vec![]);
    let o = dir.join("o.dxf");
    b.run(&convert_command(&fake_exe(), &o, &input), &o).unwrap();
    let got: Vec<String> = std::fs::read_to_string(&record).unwrap().split('\n').map(str::to_string).collect();
    assert_eq!(got, vec!["-y".to_string(), "-o".to_string(), o.display().to_string(), input.display().to_string()]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn discovery_prefers_the_bundle_then_the_sibling_then_path() {
    let root = scratch("find");
    let plat = Platform::current();
    let on_path = root.join("onpath");
    let mut b = bridge(&root, vec![on_path.clone()]);
    assert!(b.find_dwg2dxf().is_none());
    assert!(!b.have_dwg_support());

    let sys = install_fake(&on_path, plat);
    assert_eq!(b.find_dwg2dxf(), Some(sys.clone()));
    assert!(b.have_dwg_support());

    let sib = install_fake(&root.join("sibling"), plat);
    assert_eq!(b.find_dwg2dxf(), Some(sib));
    let vend = install_fake(&root.join("vendor"), plat);
    assert_eq!(b.find_dwg2dxf(), Some(vend));

    // A PATH entry is only used when it is the sole candidate.
    b.vendor_bin = root.join("nowhere");
    b.sibling_bin = root.join("nowhere2");
    assert_eq!(b.find_dwg2dxf(), Some(sys));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_binary_of_another_os_is_ignored() {
    let root = scratch("magic");
    let plat = Platform::current();
    let name = if plat == Platform::Windows { "dwg2dxf.exe" } else { "dwg2dxf" };
    std::fs::create_dir_all(root.join("vendor")).unwrap();
    // A script is a file, but not a native executable of any OS here.
    std::fs::write(root.join("vendor").join(name), b"#!/bin/sh\nexit 0\n").unwrap();
    let b = bridge(&root, vec![root.join("vendor")]);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let p = root.join("vendor").join(name);
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    if plat != Platform::Other {
        assert!(b.find_dwg2dxf().is_none());
    }
    // The wrong OS's magic is rejected, the right one accepted.
    let p = root.join("magic.bin");
    std::fs::write(&p, b"\x7fELF....").unwrap();
    assert!(is_native_executable(&p, Platform::Linux));
    assert!(!is_native_executable(&p, Platform::MacOs));
    assert!(!is_native_executable(&p, Platform::Windows));
    assert!(is_native_executable(&p, Platform::Other));
    std::fs::write(&p, b"MZ\x90\x00").unwrap();
    assert!(is_native_executable(&p, Platform::Windows));
    assert!(!is_native_executable(&p, Platform::Linux));
    std::fs::write(&p, b"\xcf\xfa\xed\xfe").unwrap();
    assert!(is_native_executable(&p, Platform::MacOs));
    assert!(!is_native_executable(&root.join("missing"), Platform::Linux));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn windows_looks_for_the_exe_and_linux_for_the_bare_name() {
    let root = scratch("winname");
    let bin = root.join("vendor");
    std::fs::create_dir_all(&bin).unwrap();
    std::fs::write(bin.join("dwg2dxf.exe"), b"MZ").unwrap();
    let mut b = bridge(&root, vec![]);
    b.platform = Platform::Windows;
    assert_eq!(b.find_dwg2dxf(), Some(bin.join("dwg2dxf.exe")));
    b.platform = Platform::Linux;
    assert_eq!(b.find_dwg2dxf(), None);
    // PATH on Windows tries the PATHEXT extensions on a bare name.
    let onpath = root.join("p");
    std::fs::create_dir_all(&onpath).unwrap();
    std::fs::write(onpath.join("dwg2dxf.exe"), b"MZ").unwrap();
    // A PATH search wants the execute bit where there is one (`os.access`).
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(onpath.join("dwg2dxf.exe"), std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let mut w = bridge(&root, vec![onpath.clone()]);
    w.vendor_bin = root.join("none");
    w.platform = Platform::Windows;
    assert_eq!(w.find_dwg2dxf(), Some(onpath.join("dwg2dxf.exe")));
    let _ = std::fs::remove_dir_all(&root);
}

// ---- the whole bridge ----------------------------------------------------------

const DXF_LINE: &str = "0\nSECTION\n2\nENTITIES\n0\nLINE\n8\n0\n10\n0.0\n20\n0.0\n30\n0.0\n11\n10.0\n21\n0.0\n31\n0.0\n0\nENDSEC\n0\nEOF\n";

fn temp_dirs_now() -> usize {
    std::fs::read_dir(std::env::temp_dir()).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with(TEMP_PREFIX)).count()
}

#[test]
fn a_missing_converter_is_the_python_error() {
    let root = scratch("missing");
    let b = bridge(&root, vec![]);
    let e = b.dwg_to_dxf(&root.join("x.dwg")).unwrap_err();
    assert_eq!(e.to_string(), "LibreDWG (dwg2dxf) is not available");
    let mut scene = Scene::new();
    assert!(b.import_dwg(&mut scene, &root.join("x.dwg"), 1.0, Hypot::Msvc, None).is_err());
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn convert_repairs_import_and_clean_up() {
    let root = scratch("bridge");
    install_fake(&root.join("vendor"), Platform::current());
    let b = bridge(&root, vec![]);
    // A DXF with the LibreDWG scars: an ENDBLK with handle 0 and a reused handle.
    let dxf = format!("0\nSECTION\n2\nBLOCKS\n0\nENDBLK\n5\n0\n0\nENDSEC\n{DXF_LINE}");
    let hexed = hex(dxf.as_bytes());
    let dwg = root.join("Obra Cañería.dwg");
    std::fs::write(&dwg, format!("stderr={}\nrc=1\nwrite={hexed}\n", hex(b"warning: recovered\n"))).unwrap();

    let before = temp_dirs_now();
    let out = b.dwg_to_dxf(&dwg).unwrap();
    assert_eq!(out.file_name().unwrap(), TEMP_NAME);
    assert!(out.parent().unwrap().file_name().unwrap().to_string_lossy().starts_with(TEMP_PREFIX));
    assert!(out.parent().unwrap().to_string_lossy().is_ascii());
    let text = std::fs::read_to_string(&out).unwrap();
    assert!(text.contains("ENDBLK") && !text.contains("\n5\n0\n"), "null handle stripped: {text}");
    discard_temp_dxf(&out);
    assert!(!out.exists() && !out.parent().unwrap().exists());

    // Import: the group carries the USER's name, the temp goes away.
    std::fs::write(&dwg, format!("rc=1\nwrite={hexed}\n")).unwrap();
    let mut scene = Scene::new();
    let stats = b.import_dwg(&mut scene, &dwg, 1.0, Hypot::Msvc, None).unwrap();
    assert!(stats.edges >= 1, "{stats:?}");
    assert_eq!(scene.groups[0].name, "Obra Cañería");
    assert!(temp_dirs_now() <= before, "temp directories leaked");

    // Failure: no output, the temp directory is removed, the message is Python's.
    std::fs::write(&dwg, format!("stderr={}\nrc=2\n", hex(b"bad file\n"))).unwrap();
    let e = b.dwg_to_dxf(&dwg).unwrap_err();
    assert_eq!(e.to_string(), "conversion produced no output: bad file");
    assert!(temp_dirs_now() <= before, "temp directories leaked");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn discard_only_touches_our_own_temp_dirs() {
    let root = scratch("discard");
    let keeper = root.join("fixtures").join("x.dxf");
    std::fs::create_dir_all(keeper.parent().unwrap()).unwrap();
    std::fs::write(&keeper, "data").unwrap();
    discard_temp_dxf(&keeper);
    assert!(keeper.exists(), "fixture paths must survive");
    let _ = std::fs::remove_dir_all(&root);
}

/// End to end with the REAL LibreDWG: set `INGETRAZO_DWG_E2E` to a `.dwg`
/// (and have `dwg2dxf` on PATH or bundled). Skipped, and said so, otherwise.
#[test]
fn real_dwg2dxf_end_to_end_when_asked() {
    let Some(dwg) = std::env::var_os("INGETRAZO_DWG_E2E") else {
        eprintln!("skipped: INGETRAZO_DWG_E2E is not set");
        return;
    };
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3).unwrap().to_path_buf();
    let b = DwgBridge::from_app_root(&root);
    assert!(b.have_dwg_support(), "dwg2dxf not found");
    let mut scene = Scene::new();
    let stats = b.import_dwg(&mut scene, Path::new(&dwg), 1.0, Hypot::Msvc, None).unwrap();
    assert!(stats.entities > 0, "{stats:?}");
}
