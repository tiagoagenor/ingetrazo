// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Where the application finds its data (M4.9): the `resources/` tree
//! (components, textures, colors, styles, mime, ui) and what sits beside it
//! (`materials/presets`, `vendor/libredwg`). One function, [`app_root`],
//! answers for every part of the program, in every layout:
//!
//! | layout | executable | root |
//! |---|---|---|
//! | Windows installer, portable zip | `ingetrazo\ingetrazo.exe` | the folder of the exe |
//! | Linux tree, AppImage, Snap, Flatpak | `lib/ingetrazo/ingetrazo` | the folder of the binary |
//! | Linux `bin/ingetrazo` wrapper, or a symlink to it | `bin/ingetrazo` | `../lib/ingetrazo` |
//! | macOS bundle | `IngeTrazo.app/Contents/MacOS/ingetrazo` | `Contents/Resources` |
//! | `cargo run`, `cargo build` | `rust/target/{debug,release}/ingetrazo` | the repository root (the nearest folder above that holds `resources/`) |
//!
//! `INGETRAZO_HOME` overrides all of it (a folder holding `resources/`).
//! The root is the folder that *contains* `resources/`, as in the packaging
//! scripts (`packaging/windows/build.ps1`, `packaging/macos/build.sh`,
//! `packaging/linux/install-tree.sh`): `app_root().join("resources/textures/x.png")`.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The environment variable that names the root by hand.
pub const HOME_VAR: &str = "INGETRAZO_HOME";

/// The folder that holds `resources/` (see the module documentation).
/// Computed once. When no layout matches, the folder of the executable: a
/// missing resource is then an ordinary missing file, not a panic.
pub fn app_root() -> &'static Path {
    static ROOT: OnceLock<PathBuf> = OnceLock::new();
    ROOT.get_or_init(|| {
        let env = std::env::var_os(HOME_VAR).map(PathBuf::from);
        let exe = std::env::current_exe().ok();
        resolve_root(exe.as_deref(), env.as_deref(), &|p| p.join("resources").is_dir())
    })
}

/// `app_root()/rel`: a file of the install (`"resources/textures/brick.png"`).
pub fn resource_path(rel: impl AsRef<Path>) -> PathBuf {
    app_root().join(rel)
}

/// The rule, apart from the process so a test can feed it layouts.
/// `has_resources(dir)`: whether `dir` holds a `resources/` folder.
pub fn resolve_root(exe: Option<&Path>, env: Option<&Path>, has_resources: &dyn Fn(&Path) -> bool) -> PathBuf {
    if let Some(home) = env {
        if has_resources(home) {
            return home.to_path_buf();
        }
    }
    let Some(exe) = exe else { return PathBuf::from(".") };
    // A symlink to the binary (`/usr/bin/ingetrazo`) is followed first.
    let real = simplify(std::fs::canonicalize(exe).unwrap_or_else(|_| exe.to_path_buf()));
    for path in [real.as_path(), exe] {
        let Some(dir) = path.parent() else { continue };
        // Beside the binary: Windows, `lib/ingetrazo/`, AppImage, Snap, Flatpak.
        if has_resources(dir) {
            return dir.to_path_buf();
        }
        // macOS: Contents/MacOS/ingetrazo -> Contents/Resources.
        let mac = dir.join("..").join("Resources");
        if has_resources(&mac) {
            return normalize(&mac);
        }
        // A `bin/` wrapper over `lib/ingetrazo/`.
        let lib = dir.join("..").join("lib").join("ingetrazo");
        if has_resources(&lib) {
            return normalize(&lib);
        }
        // `cargo run`: walk up to the repository root.
        for up in dir.ancestors().skip(1) {
            if has_resources(up) {
                return up.to_path_buf();
            }
        }
    }
    exe.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."))
}

/// Windows' `canonicalize` answers `\\?\C:\...`: the plain form is what the
/// rest of the program (and a message to the user) expects.
fn simplify(p: PathBuf) -> PathBuf {
    #[cfg(windows)]
    {
        if let Some(rest) = p.to_str().and_then(|s| s.strip_prefix(r"\\?\")) {
            if !rest.starts_with("UNC\\") {
                return PathBuf::from(rest);
            }
        }
    }
    p
}

/// The images a document names by a path on disk (a texture of the
/// library, `resources/textures/brick.png`) are read from the install: a
/// relative path is relative to [`app_root`], an absolute one is itself. What
/// the document embeds comes first.
pub struct InstallTextures<T>(pub T);

impl<T: ingetrazo_render::TextureSource> ingetrazo_render::TextureSource for InstallTextures<T> {
    fn load(&mut self, path: &str) -> Option<ingetrazo_render::textures::TextureImage> {
        if let Some(image) = self.0.load(path) {
            return Some(image);
        }
        let p = Path::new(path);
        let file = if p.is_absolute() { p.to_path_buf() } else { app_root().join(p) };
        ingetrazo_render::document::decode_image(&std::fs::read(file).ok()?)
    }
}

/// `a/b/../c` as `a/c` (the paths here exist, and `..` follows no link).
fn normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            std::path::Component::ParentDir => {
                out.pop();
            }
            other => out.push(other),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// A made-up file system: the folders that hold `resources/`.
    fn with(dirs: &[&str]) -> impl Fn(&Path) -> bool {
        let set: HashSet<PathBuf> = dirs.iter().map(PathBuf::from).collect();
        move |p: &Path| set.contains(&normalize(p))
    }

    fn root(exe: &str, env: Option<&str>, dirs: &[&str]) -> PathBuf {
        resolve_root(Some(Path::new(exe)), env.map(Path::new), &with(dirs))
    }

    #[test]
    fn next_to_the_executable() {
        // Windows installer; Linux lib/ingetrazo, AppImage, Snap, Flatpak.
        assert_eq!(root("/prog/IngeTrazo/ingetrazo.exe", None, &["/prog/IngeTrazo"]), PathBuf::from("/prog/IngeTrazo"));
        assert_eq!(root("/usr/lib/ingetrazo/ingetrazo", None, &["/usr/lib/ingetrazo"]), PathBuf::from("/usr/lib/ingetrazo"));
    }

    #[test]
    fn inside_a_macos_bundle() {
        let exe = "/Applications/IngeTrazo.app/Contents/MacOS/ingetrazo";
        assert_eq!(root(exe, None, &["/Applications/IngeTrazo.app/Contents/Resources"]), PathBuf::from("/Applications/IngeTrazo.app/Contents/Resources"));
    }

    #[test]
    fn a_bin_wrapper_finds_the_lib_folder() {
        assert_eq!(root("/usr/bin/ingetrazo", None, &["/usr/lib/ingetrazo"]), PathBuf::from("/usr/lib/ingetrazo"));
    }

    #[test]
    fn cargo_run_walks_up_to_the_repository() {
        let exe = "/work/ingetrazo/rust/target/release/ingetrazo";
        assert_eq!(root(exe, None, &["/work/ingetrazo"]), PathBuf::from("/work/ingetrazo"));
        // The nearest one wins.
        assert_eq!(root(exe, None, &["/work/ingetrazo", "/work/ingetrazo/rust"]), PathBuf::from("/work/ingetrazo/rust"));
    }

    #[test]
    fn the_environment_overrides_and_nothing_found_falls_back_to_the_exe_folder() {
        assert_eq!(root("/x/ingetrazo", Some("/home/me/data"), &["/home/me/data", "/x"]), PathBuf::from("/home/me/data"));
        // A variable that names a folder with no resources/ is ignored.
        assert_eq!(root("/x/ingetrazo", Some("/nope"), &["/x"]), PathBuf::from("/x"));
        assert_eq!(root("/x/ingetrazo", None, &[]), PathBuf::from("/x"));
    }

    #[test]
    fn this_checkout_resolves_to_the_repository() {
        // The test binary lives in rust/target/<profile>/deps: the walk up
        // reaches the repository root, whose resources/ is committed.
        let r = app_root();
        assert!(r.join("resources").is_dir(), "{} has no resources/", r.display());
    }
}
