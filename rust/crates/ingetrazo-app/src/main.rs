// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `ingetrazo [file.igz]`: the IngeTrazo application.

use std::path::PathBuf;

use eframe::egui;
use eframe::egui_wgpu;

fn main() -> eframe::Result {
    let mut path: Option<PathBuf> = None;
    let mut novsync = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "-h" | "--help" => {
                println!("usage: ingetrazo [--novsync] [file.igz]");
                return Ok(());
            }
            "--novsync" => novsync = true,
            other if other.starts_with('-') => eprintln!("ingetrazo: ignoring unknown option {other}"),
            other => path = Some(PathBuf::from(other)),
        }
    }

    let mut wgpu_options = egui_wgpu::WgpuConfiguration::default();
    if novsync {
        wgpu_options.surface.present_mode = egui_wgpu::wgpu::PresentMode::AutoNoVsync;
    }
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 800.0]).with_title("IngeTrazo"),
        renderer: eframe::Renderer::Wgpu,
        wgpu_options,
        ..Default::default()
    };
    eframe::run_native(
        "ingetrazo",
        options,
        Box::new(move |cc| {
            cc.egui_ctx.enable_accesskit();
            Ok(Box::new(ingetrazo_app::app::App::new(cc, path)))
        }),
    )
}
