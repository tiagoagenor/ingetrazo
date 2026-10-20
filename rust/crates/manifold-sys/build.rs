// SPDX-License-Identifier: GPL-3.0-or-later
//! Builds Manifold statically from source.
//!
//! The source is a pinned GitHub release tarball (and the Clipper2 commit
//! Manifold's CMake pins for its CrossSection, which `manifoldc` needs),
//! each checked against a SHA-256 before anything is extracted. They are
//! cached under `$MANIFOLD_SYS_CACHE` (default: `OUT_DIR`), so a CI cache
//! of that folder makes the build offline. To build with no network at all
//! point `MANIFOLD_SRC_DIR` / `CLIPPER2_SRC_DIR` at unpacked copies of the
//! same versions (a packager's vendored sources).
//!
//! The version is the one the Python app's `manifold3d` wheel is built from
//! (`requirements.txt`: manifold3d >= 3.5.3; the app runs 3.5.4), so both
//! give the same booleans.

use std::env;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

const MANIFOLD_VERSION: &str = "3.5.4";
const MANIFOLD_URL: &str = "https://github.com/elalish/manifold/archive/refs/tags/v3.5.4.tar.gz";
const MANIFOLD_SHA256: &str = "ce2f8b7798409b9f07e092c0ec0524ed0806431e01aea4020b65d0e934ea9438";

// The commit `cmake/manifoldDeps.cmake` of v3.5.4 fetches.
const CLIPPER2_URL: &str = "https://github.com/AngusJohnson/Clipper2/archive/46f639177fe418f9689e8ddb74f08a870c71f5b4.tar.gz";
const CLIPPER2_SHA256: &str = "73f5783e0c88299976334f48e3e356c756b96212c652ecc8bceec3bd99455bcb";

fn sha256_hex(data: &[u8]) -> String {
    Sha256::digest(data).iter().map(|b| format!("{b:02x}")).collect()
}

/// The unpacked source: `$env_dir` if set, else the checked tarball unpacked
/// under `cache/<name>` (its single top-level folder).
fn source(env_dir: &str, name: &str, url: &str, sha: &str, cache: &Path) -> PathBuf {
    println!("cargo:rerun-if-env-changed={env_dir}");
    if let Ok(dir) = env::var(env_dir) {
        return PathBuf::from(dir);
    }
    let dest = cache.join(name);
    let marker = dest.join(".unpacked");
    if marker.exists() {
        return fs::read_to_string(&marker).map(|s| PathBuf::from(s.trim())).expect("read marker");
    }
    let tarball = cache.join(format!("{name}.tar.gz"));
    let data = match fs::read(&tarball) {
        Ok(d) if sha256_hex(&d) == sha => d,
        _ => {
            let mut buf = Vec::new();
            ureq::get(url)
                .call()
                .unwrap_or_else(|e| panic!("download {url}: {e}"))
                .into_reader()
                .read_to_end(&mut buf)
                .expect("read download");
            let got = sha256_hex(&buf);
            assert_eq!(got, sha, "{url}: SHA-256 is {got}, expected {sha}");
            fs::create_dir_all(cache).expect("create cache dir");
            fs::write(&tarball, &buf).expect("cache tarball");
            buf
        }
    };
    let _ = fs::remove_dir_all(&dest);
    fs::create_dir_all(&dest).expect("create source dir");
    let mut archive = tar::Archive::new(flate2::read::GzDecoder::new(&data[..]));
    archive.unpack(&dest).expect("unpack tarball");
    let top: Vec<_> = fs::read_dir(&dest).expect("read source dir").filter_map(|e| e.ok()).collect();
    assert_eq!(top.len(), 1, "{name}: expected one top-level folder");
    let root = top[0].path();
    fs::write(&marker, root.to_string_lossy().as_bytes()).expect("write marker");
    root
}

fn find_libs(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            find_libs(&p, out);
        } else if let Some(ext) = p.extension() {
            if ext == "a" || ext == "lib" {
                out.push(p);
            }
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=shim.cpp");
    println!("cargo:rerun-if-env-changed=MANIFOLD_SYS_CACHE");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cache = env::var("MANIFOLD_SYS_CACHE").map(PathBuf::from).unwrap_or_else(|_| out.join("src-cache"));

    let manifold = source("MANIFOLD_SRC_DIR", "manifold", MANIFOLD_URL, MANIFOLD_SHA256, &cache);
    let clipper2 = source("CLIPPER2_SRC_DIR", "clipper2", CLIPPER2_URL, CLIPPER2_SHA256, &cache);
    let version = fs::read_to_string(manifold.join("CMakeLists.txt")).expect("Manifold CMakeLists.txt");
    let patch = MANIFOLD_VERSION.rsplit('.').next().unwrap();
    assert!(version.contains(&format!("set(MANIFOLD_VERSION_PATCH {patch})")), "unexpected Manifold source version");

    // Sequential (no TBB), no tests/Python/JS bindings, static libraries.
    // Clipper2 comes from the pinned source dir, never from the network.
    let mut config = cmake::Config::new(&manifold);
    if env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        // The flags the cmake crate passes replace CMake's defaults, /EHsc
        // among them; Manifold and Clipper2 throw and catch.
        config.cxxflag("/EHsc");
    }
    let dst = config
        .define("CMAKE_CXX_STANDARD", "17")
        .profile("Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("MANIFOLD_TEST", "OFF")
        .define("MANIFOLD_PYBIND", "OFF")
        .define("MANIFOLD_JSBIND", "OFF")
        .define("MANIFOLD_CBIND", "ON")
        .define("MANIFOLD_CROSS_SECTION", "ON")
        .define("MANIFOLD_PAR", "OFF")
        .define("MANIFOLD_DOWNLOADS", "ON")
        .define("MANIFOLD_USE_BUILTIN_CLIPPER2", "ON")
        .define("FETCHCONTENT_SOURCE_DIR_CLIPPER2", clipper2.to_str().unwrap())
        .define("FETCHCONTENT_FULLY_DISCONNECTED", "ON")
        .build_target("manifoldc")
        .build();

    let mut libs = Vec::new();
    find_libs(&dst.join("build"), &mut libs);
    assert!(!libs.is_empty(), "no static libraries were built under {}", dst.display());
    let mut dirs: Vec<PathBuf> = libs.iter().filter_map(|l| l.parent().map(Path::to_path_buf)).collect();
    dirs.sort();
    dirs.dedup();
    for d in &dirs {
        println!("cargo:rustc-link-search=native={}", d.display());
    }

    // The one function the C API lacks, next to it.
    cc::Build::new().cpp(true).std("c++17").file("shim.cpp").include(manifold.join("include")).compile("ingetrazo_manifold_shim");

    // Dependents first, for GNU ld.
    let has = |name: &str| libs.iter().any(|l| l.file_stem().is_some_and(|s| s.to_string_lossy().trim_start_matches("lib") == name));
    for name in ["manifoldc", "manifold", "Clipper2"] {
        assert!(has(name), "{name} was not built");
        println!("cargo:rustc-link-lib=static={name}");
    }
    println!("cargo:version={MANIFOLD_VERSION}");
}
