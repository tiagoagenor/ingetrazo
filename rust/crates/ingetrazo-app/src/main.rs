// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `ingetrazo [file.igz]`: the IngeTrazo application.

use std::path::PathBuf;

use eframe::egui;
use eframe::egui_wgpu;

const USAGE: &str = "usage: ingetrazo [--novsync] [--bench-orbit <seconds>] [file.igz]";

fn main() -> eframe::Result {
    let mut path: Option<PathBuf> = None;
    let mut novsync = false;
    let mut bench: Option<f64> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--novsync" => novsync = true,
            "--bench-orbit" => match args.next().and_then(|s| s.parse::<f64>().ok()) {
                Some(secs) if secs > 0.0 => {
                    bench = Some(secs);
                    // The benchmark measures the real cost of a frame, not the monitor.
                    novsync = true;
                }
                _ => {
                    eprintln!("ingetrazo: --bench-orbit needs a number of seconds\n{USAGE}");
                    std::process::exit(2);
                }
            },
            other if other.starts_with('-') => eprintln!("ingetrazo: ignoring unknown option {other}"),
            other => path = Some(PathBuf::from(other)),
        }
    }

    let mut wgpu_options = egui_wgpu::WgpuConfiguration::default();
    if novsync {
        wgpu_options.surface.present_mode = egui_wgpu::wgpu::PresentMode::AutoNoVsync;
    }
    let options = eframe::NativeOptions {
        // The app id is the Wayland app id and the X11 WM class: the Linux
        // .desktop files say `StartupWMClass=IngeTrazo`.
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 800.0]).with_title("IngeTrazo").with_app_id("IngeTrazo"),
        renderer: eframe::Renderer::Wgpu,
        wgpu_options,
        ..Default::default()
    };
    eframe::run_native(
        "ingetrazo",
        options,
        Box::new(move |cc| {
            cc.egui_ctx.enable_accesskit();
            Ok(Box::new(ingetrazo_app::app::App::with_bench(cc, path, bench)))
        }),
    )
}
