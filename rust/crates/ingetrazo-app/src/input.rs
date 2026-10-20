// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! From egui's events to the session's [`Input`]: what a Qt widget's mouse
//! and key events would have been. egui reports a modifier as a state, not as
//! a key: here its changes become the key presses and releases the tools
//! watch (Shift locks the inference, Ctrl copies, Alt cycles the inference
//! mode). It reports no double-click either: a second press at the same
//! spot, soon enough, is one.

use eframe::egui::{Event, Key, Modifiers, MouseWheelUnit, PointerButton, Pos2, Rect};
use ingetrazo_tools::base::{KEY_ALT, KEY_BACKSPACE, KEY_CONTROL, KEY_DELETE, KEY_DOWN, KEY_ESCAPE, KEY_LEFT, KEY_RETURN, KEY_RIGHT, KEY_SHIFT, KEY_UP, MOD_ALT, MOD_CONTROL, MOD_SHIFT};

use crate::session::{Btn, Input};

/// Wheel notches in egui's smooth-scroll points (`egui` turns a notch into
/// 50 points).
pub const POINTS_PER_NOTCH: f32 = 50.0;
/// `QApplication.doubleClickInterval()`, seconds.
pub const DOUBLE_CLICK_SECS: f64 = 0.4;
/// `(ev.position() - pos).manhattanLength() < 8`.
pub const DOUBLE_CLICK_PX: f64 = 8.0;

/// egui's modifiers as Qt's bits.
pub fn qt_mods(m: &Modifiers) -> u32 {
    let mut bits = 0;
    if m.shift {
        bits |= MOD_SHIFT;
    }
    if m.ctrl || m.command {
        bits |= MOD_CONTROL;
    }
    if m.alt {
        bits |= MOD_ALT;
    }
    bits
}

/// egui's named keys as Qt's key codes; a letter or a digit under Ctrl is its
/// ASCII code (the keys a chord is made of).
pub fn qt_key(key: Key) -> Option<i32> {
    Some(match key {
        Key::Escape => KEY_ESCAPE,
        Key::Enter => KEY_RETURN,
        Key::Backspace => KEY_BACKSPACE,
        Key::Delete => KEY_DELETE,
        Key::ArrowLeft => KEY_LEFT,
        Key::ArrowRight => KEY_RIGHT,
        Key::ArrowUp => KEY_UP,
        Key::ArrowDown => KEY_DOWN,
        other => {
            let name = other.name();
            let mut chars = name.chars();
            match (chars.next(), chars.next()) {
                (Some(c), None) if c.is_ascii_alphanumeric() => c.to_ascii_uppercase() as i32,
                _ => return None,
            }
        }
    })
}

/// The state a frame's events are read against.
#[derive(Debug, Default)]
pub struct Translator {
    mods: u32,
    /// Where each button went down: only a press that began in the viewport
    /// has a release in it.
    down_left: bool,
    down_middle: bool,
    last_press: Option<(f64, f64, f64)>,
    pos: Option<Pos2>,
}

impl Translator {
    pub fn new() -> Translator {
        Translator::default()
    }

    /// The events of one frame, `modifiers` the state at its end, `now` the
    /// frame's time in seconds, `rect` the viewport (events are reported
    /// relative to its top-left corner), `hovered` whether the pointer is over it.
    pub fn translate(&mut self, events: &[Event], modifiers: Modifiers, now: f64, rect: Rect, hovered: bool) -> Vec<Input> {
        let mut out = Vec::new();
        let mods = qt_mods(&modifiers);
        // A modifier is a key: its change is its press or its release.
        for (key, bit) in [(KEY_SHIFT, MOD_SHIFT), (KEY_CONTROL, MOD_CONTROL), (KEY_ALT, MOD_ALT)] {
            let (was, is) = (self.mods & bit != 0, mods & bit != 0);
            if is && !was {
                out.push(Input::Key { key, mods, pressed: true });
            } else if was && !is {
                out.push(Input::Key { key, mods, pressed: false });
            }
        }
        self.mods = mods;
        let rel = |p: Pos2| ((p.x - rect.min.x) as f64, (p.y - rect.min.y) as f64);
        for event in events {
            match event {
                Event::PointerMoved(pos) => {
                    self.pos = Some(*pos);
                    if rect.contains(*pos) || self.down_left || self.down_middle {
                        let (x, y) = rel(*pos);
                        out.push(Input::Move { x, y, mods });
                    }
                }
                Event::PointerButton { pos, button, pressed, .. } => {
                    self.pos = Some(*pos);
                    let (x, y) = rel(*pos);
                    let b = match button {
                        PointerButton::Primary => Btn::Left,
                        PointerButton::Middle => Btn::Middle,
                        _ => continue,
                    };
                    if *pressed {
                        if !(rect.contains(*pos) && hovered) {
                            continue;
                        }
                        match b {
                            Btn::Left => self.down_left = true,
                            Btn::Middle => self.down_middle = true,
                        }
                        if b == Btn::Left {
                            let double = self.last_press.is_some_and(|(t, lx, ly)| now - t <= DOUBLE_CLICK_SECS && (x - lx).abs() + (y - ly).abs() < DOUBLE_CLICK_PX);
                            if double {
                                self.last_press = None;
                                out.push(Input::DoubleClick { x, y, mods });
                                continue;
                            }
                            self.last_press = Some((now, x, y));
                        }
                        out.push(Input::Press { button: b, x, y, mods });
                    } else {
                        let began = match b {
                            Btn::Left => std::mem::take(&mut self.down_left),
                            Btn::Middle => std::mem::take(&mut self.down_middle),
                        };
                        if began {
                            out.push(Input::Release { button: b, x, y, mods });
                        }
                    }
                }
                Event::MouseWheel { unit, delta, .. } if hovered => {
                    let notches = match unit {
                        MouseWheelUnit::Line => delta.y,
                        MouseWheelUnit::Point => delta.y / POINTS_PER_NOTCH,
                        MouseWheelUnit::Page => delta.y * 10.0,
                    };
                    if let Some(p) = self.pos {
                        let (x, y) = rel(p);
                        out.push(Input::Wheel { x, y, notches: notches as f64 });
                    }
                }
                Event::Key { key, pressed, repeat, modifiers: km, .. } => {
                    let Some(code) = qt_key(*key) else { continue };
                    let named = matches!(*key, Key::Escape | Key::Enter | Key::Backspace | Key::Delete | Key::ArrowLeft | Key::ArrowRight | Key::ArrowUp | Key::ArrowDown);
                    // Letters and digits are typed text; under Ctrl they are
                    // the key of a chord.
                    if !named && !(km.ctrl || km.command || km.alt) {
                        continue;
                    }
                    if *repeat && *key != Key::Backspace {
                        continue;
                    }
                    out.push(Input::Key { key: code, mods: qt_mods(km), pressed: *pressed });
                }
                Event::Text(text) => {
                    let printable: String = text.chars().filter(|c| !c.is_control()).collect();
                    if !printable.is_empty() {
                        out.push(Input::Text { text: printable, mods });
                    }
                }
                _ => {}
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect() -> Rect {
        Rect::from_min_size(Pos2::new(100.0, 50.0), eframe::egui::vec2(800.0, 600.0))
    }

    fn press(pos: Pos2, pressed: bool) -> Event {
        Event::PointerButton { pos, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE }
    }

    #[test]
    fn a_press_is_in_the_viewports_pixels_and_a_second_one_is_a_double_click() {
        let mut t = Translator::new();
        let p = Pos2::new(150.0, 80.0);
        let ev = [press(p, true), press(p, false), press(p, true), press(p, false)];
        let out = t.translate(&ev, Modifiers::NONE, 1.0, rect(), true);
        assert_eq!(
            out,
            vec![
                Input::Press { button: Btn::Left, x: 50.0, y: 30.0, mods: 0 },
                Input::Release { button: Btn::Left, x: 50.0, y: 30.0, mods: 0 },
                Input::DoubleClick { x: 50.0, y: 30.0, mods: 0 },
                Input::Release { button: Btn::Left, x: 50.0, y: 30.0, mods: 0 },
            ]
        );
        // A press long after is single again.
        let out = t.translate(&[press(p, true)], Modifiers::NONE, 9.0, rect(), true);
        assert!(matches!(out[0], Input::Press { .. }));
    }

    #[test]
    fn a_press_outside_the_viewport_has_no_release_in_it() {
        let mut t = Translator::new();
        let outside = Pos2::new(10.0, 10.0);
        let out = t.translate(&[press(outside, true), press(outside, false)], Modifiers::NONE, 0.0, rect(), false);
        assert!(out.is_empty());
    }

    #[test]
    fn a_modifier_changing_is_a_key_press_or_release() {
        let mut t = Translator::new();
        let shift = Modifiers { shift: true, ..Modifiers::NONE };
        let out = t.translate(&[], shift, 0.0, rect(), true);
        assert_eq!(out, vec![Input::Key { key: KEY_SHIFT, mods: MOD_SHIFT, pressed: true }]);
        let out = t.translate(&[], Modifiers::NONE, 0.1, rect(), true);
        assert_eq!(out, vec![Input::Key { key: KEY_SHIFT, mods: 0, pressed: false }]);
    }

    #[test]
    fn typed_text_and_named_keys_arrive_as_they_are() {
        let mut t = Translator::new();
        let ev = [
            Event::Text("5".into()),
            Event::Key { key: Key::Enter, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE },
            // A letter under no modifier is text, not a key.
            Event::Key { key: Key::L, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE },
        ];
        let out = t.translate(&ev, Modifiers::NONE, 0.0, rect(), true);
        assert_eq!(out, vec![Input::Text { text: "5".into(), mods: 0 }, Input::Key { key: KEY_RETURN, mods: 0, pressed: true }]);
    }
}
