// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Windows packaging metadata (M9.1): the executable icon and the version
//! resource, and a GUI-subsystem release binary (no console window).

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../../resources/icons/ingetrazo.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("../../../resources/icons/ingetrazo.ico");
        res.set("ProductName", "IngeTrazo");
        res.set("FileDescription", "IngeTrazo");
        res.set("CompanyName", "Ing. Marco Sumari Tellez");
        res.set("LegalCopyright", "GPL-3.0-or-later");
        res.set("OriginalFilename", "ingetrazo.exe");
        if let Err(e) = res.compile() {
            // Without the Windows SDK's resource compiler the binary still
            // works, only without the icon: warn, do not fail the build.
            println!("cargo:warning=could not embed the Windows resources: {e}");
        }
    }
    let msvc = std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");
    if msvc && std::env::var("PROFILE").as_deref() == Ok("release") {
        println!("cargo:rustc-link-arg-bins=/SUBSYSTEM:WINDOWS");
        println!("cargo:rustc-link-arg-bins=/ENTRY:mainCRTStartup");
    }
}
