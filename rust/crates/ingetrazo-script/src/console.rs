// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The script console (M8.2): what `plugins/python_console.py` was, without a
//! widget. A UI owns the text boxes and drives a [`Console`]:
//!
//! * [`Console::submit`] runs the text (a line, or several: see
//!   [`Console::is_complete`], which tells the input box whether Enter should
//!   run or open a new line) and answers a [`Submission`]: the lines to print,
//!   coloured by [`Role`]. Every run is **one undo step**; a run that fails
//!   is rolled back whole (the document is as it was); a run that only reads
//!   leaves no undo entry (the Python console popped it back off, here the
//!   engine never pushes it).
//! * Variables and functions defined in one submission are there in the next
//!   (the Python console's persistent scope).
//! * [`Console::history_prev`] / [`Console::history_next`] walk the submitted
//!   inputs (Up/Down), keeping the half-typed draft.
//! * [`Console::completions`] answers names for a prefix: the API functions
//!   and properties (taken from the engine, so they never go stale), Rhai
//!   keywords and the variables of the session.
//! * [`Console::cancel_token`] stops a long script (see
//!   [`crate::ScriptEngine::cancel_token`]); the run's wall-clock and
//!   operation limits are the engine's [`Limits`].
//!
//! Scripts are text the user typed or loaded: the sandbox of the engine
//! applies (no files, network or processes; `import` does not exist).

use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use ingetrazo_model::scene::Scene;
use ingetrazo_ops::command::{Cx, History};
use ingetrazo_ops::edit_cmds::Entity;

use crate::{Limits, ScriptEngine, ScriptError};

/// What a line of the transcript is, for colouring (the Python console's
/// palette roles).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// The input, echoed.
    Echo,
    /// What `print` / `debug` wrote.
    Output,
    /// The value of the script's last expression.
    Value,
    /// An error, with its position.
    Error,
    /// Notes from the console itself (rollback, running a file, welcome).
    Muted,
    /// The API help.
    Help,
    /// A heading.
    Title,
    /// A script file starting to run.
    Run,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Line {
    pub role: Role,
    pub text: String,
}

fn line(role: Role, text: impl Into<String>) -> Line {
    Line { role, text: text.into() }
}

/// What one submission did.
#[derive(Clone, Debug)]
pub struct Submission {
    /// The lines to append to the transcript, in order (the echo first).
    pub lines: Vec<Line>,
    /// The script ran to its end.
    pub ok: bool,
    /// The document changed: one new undo step was pushed.
    pub changed: bool,
    /// The user (or the host) stopped it with the cancel token.
    pub cancelled: bool,
    /// The error, when it did not run to its end.
    pub error: Option<ScriptError>,
    /// What `print` wrote (also in `lines`).
    pub output: String,
}

impl Submission {
    fn empty() -> Submission {
        Submission { lines: Vec::new(), ok: true, changed: false, cancelled: false, error: None, output: String::new() }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CompletionKind {
    Function,
    /// A function whose first argument is an object: `f.area()`.
    Method,
    /// `f.color`, `v.x`.
    Property,
    Keyword,
    Variable,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Completion {
    pub name: String,
    pub kind: CompletionKind,
    /// The signatures, one per line, tidied for display (empty for keywords
    /// and variables).
    pub signature: String,
}

const KEYWORDS: &[&str] = &[
    "let", "const", "if", "else", "while", "loop", "for", "in", "break", "continue", "return", "fn", "switch", "true", "false", "throw", "try",
    "catch", "do", "until", "print", "debug",
];

/// The most lines the transcript keeps.
const TRANSCRIPT_MAX: usize = 5000;

pub struct Console {
    engine: ScriptEngine,
    history: Vec<String>,
    /// `history.len()` = not browsing.
    cursor: usize,
    draft: String,
    transcript: Vec<Line>,
    api: Option<Vec<Completion>>,
}

impl Default for Console {
    fn default() -> Self {
        Console::new(Limits::default())
    }
}

impl Console {
    pub fn new(limits: Limits) -> Console {
        let mut engine = ScriptEngine::new(limits);
        engine.set_persistent(true);
        let mut c = Console { engine, history: Vec::new(), cursor: 0, draft: String::new(), transcript: Vec::new(), api: None };
        c.push_welcome();
        c
    }

    fn push_welcome(&mut self) {
        self.transcript.push(line(Role::Title, "IngeTrazo - script console"));
        self.transcript.push(line(
            Role::Muted,
            "Rhai scripts over the open document. Each run is one undo step (Ctrl+Z); an error rolls it back. The API help lists what a script can call.",
        ));
        self.transcript.push(line(Role::Muted, ""));
    }

    /// The engine, to set limits or register more API.
    pub fn engine_mut(&mut self) -> &mut ScriptEngine {
        self.api = None;
        &mut self.engine
    }

    /// The UI selection, as `selection()` answers it.
    pub fn set_selection(&mut self, selection: Vec<Entity>) {
        self.engine.set_selection(selection);
    }

    /// Set it (from the UI thread's Esc key or a watchdog) to stop the
    /// running script.
    pub fn cancel_token(&self) -> Arc<AtomicBool> {
        self.engine.cancel_token()
    }

    /// Forget the session's variables and functions (the Clear button keeps
    /// them; opening another document should not).
    pub fn reset_session(&mut self) {
        self.engine.reset_scope();
    }

    // ---- transcript ---------------------------------------------------

    /// Everything printed so far (for repainting in another palette).
    pub fn transcript(&self) -> &[Line] {
        &self.transcript
    }

    /// The Clear button: the transcript restarts at the welcome.
    pub fn clear_transcript(&mut self) {
        self.transcript.clear();
        self.push_welcome();
    }

    fn append(&mut self, lines: &[Line]) {
        self.transcript.extend_from_slice(lines);
        if self.transcript.len() > TRANSCRIPT_MAX {
            let drop = self.transcript.len() - TRANSCRIPT_MAX;
            self.transcript.drain(..drop);
        }
    }

    /// The API reference as help lines (the "API help" button).
    pub fn help(&mut self) -> Vec<Line> {
        let text = crate::api::REFERENCE.trim_end();
        let lines = vec![line(Role::Help, text.to_string())];
        self.append(&lines);
        lines
    }

    // ---- history ------------------------------------------------------

    /// The submitted inputs, oldest first.
    pub fn history(&self) -> &[String] {
        &self.history
    }

    /// Up: the previous input. `current` is what the box holds now; it is
    /// kept as the draft when browsing starts. `None` at the oldest.
    pub fn history_prev(&mut self, current: &str) -> Option<&str> {
        if self.history.is_empty() {
            return None;
        }
        if self.cursor >= self.history.len() {
            self.cursor = self.history.len();
            self.draft = current.to_string();
        }
        if self.cursor == 0 {
            return None;
        }
        self.cursor -= 1;
        Some(&self.history[self.cursor])
    }

    /// Down: the next input, and past the newest the draft. `None` when not
    /// browsing.
    pub fn history_next(&mut self) -> Option<&str> {
        if self.cursor >= self.history.len() {
            return None;
        }
        self.cursor += 1;
        if self.cursor == self.history.len() {
            Some(&self.draft)
        } else {
            Some(&self.history[self.cursor])
        }
    }

    fn remember(&mut self, text: &str) {
        if self.history.last().map(String::as_str) != Some(text) {
            self.history.push(text.to_string());
        }
        self.cursor = self.history.len();
        self.draft.clear();
    }

    // ---- input --------------------------------------------------------

    /// Whether `text` is a whole script (Enter runs it) or an unfinished one
    /// (an open bracket, string or comment, or a dangling operator: Enter
    /// should add a line).
    pub fn is_complete(text: &str) -> bool {
        complete(text)
    }

    /// Run what the user typed.
    pub fn submit(&mut self, scene: &mut Scene, cx: &mut Cx, history: &mut History, text: &str) -> Submission {
        let text = text.trim_end();
        if text.trim().is_empty() {
            return Submission::empty();
        }
        self.remember(text);
        let mut lines = Vec::new();
        for (i, l) in text.lines().enumerate() {
            lines.push(line(Role::Echo, format!("{} {l}", if i == 0 { ">>>" } else { "..." })));
        }
        self.execute(scene, cx, history, text, lines)
    }

    /// Run a loaded script (the "Run script file" button; the UI reads the
    /// file). Not added to the input history.
    pub fn run_script(&mut self, scene: &mut Scene, cx: &mut Cx, history: &mut History, name: &str, source: &str) -> Submission {
        let lines = vec![line(Role::Run, format!("- running {name}"))];
        self.execute(scene, cx, history, source, lines)
    }

    /// Try a script on a copy of the document (see
    /// [`ScriptEngine::dry_run`]): nothing is printed to the transcript,
    /// nothing changes. For hosts that vet a script before running it.
    pub fn dry_run(&mut self, scene: &Scene, cx: &Cx, source: &str) -> Result<crate::Report, ScriptError> {
        self.engine.dry_run(scene, cx, source)
    }

    fn execute(&mut self, scene: &mut Scene, cx: &mut Cx, history: &mut History, source: &str, mut lines: Vec<Line>) -> Submission {
        let before = history.top_serial();
        let undo_before = history.undo_len();
        let result = self.engine.run(scene, cx, history, source);
        let mut sub = Submission::empty();
        match result {
            Ok(report) => {
                push_output(&mut lines, &report.output);
                if !report.value.is_empty() {
                    lines.push(line(Role::Value, report.value.clone()));
                }
                sub.output = report.output;
            }
            Err(e) => {
                push_output(&mut lines, &e.output);
                lines.push(line(Role::Error, error_block(&e, source)));
                lines.push(line(Role::Muted, if e.cancelled { "(cancelled - the document is unchanged)" } else { "(rolled back - the document is unchanged)" }));
                sub.ok = false;
                sub.cancelled = e.cancelled;
                sub.output = e.output.clone();
                sub.error = Some(e);
            }
        }
        sub.changed = history.undo_len() > undo_before || history.top_serial() != before;
        self.append(&lines);
        sub.lines = lines;
        sub
    }

    // ---- completion ---------------------------------------------------

    /// The names starting with `prefix` (case-sensitive), functions and
    /// properties first, sorted. Includes the variables of the session.
    pub fn completions(&mut self, prefix: &str) -> Vec<Completion> {
        if self.api.is_none() {
            self.api = Some(build_api(&mut self.engine));
        }
        let mut out: Vec<Completion> = self.api.as_ref().into_iter().flatten().filter(|c| c.name.starts_with(prefix)).cloned().collect();
        for kw in KEYWORDS {
            if kw.starts_with(prefix) {
                out.push(Completion { name: (*kw).to_string(), kind: CompletionKind::Keyword, signature: String::new() });
            }
        }
        for v in self.engine.scope_names() {
            if v.starts_with(prefix) {
                out.push(Completion { name: v, kind: CompletionKind::Variable, signature: String::new() });
            }
        }
        out.sort_by(|a, b| (a.kind, &a.name).cmp(&(b.kind, &b.name)));
        out
    }

    /// The identifier being typed at byte offset `cursor` of `text`:
    /// (start of it, the prefix). The UI feeds the prefix to
    /// [`Console::completions`] and replaces `start..cursor` with the pick.
    pub fn completion_prefix(text: &str, cursor: usize) -> (usize, &str) {
        let cursor = cursor.min(text.len());
        let head = &text[..cursor];
        let start = head.rfind(|c: char| !(c.is_alphanumeric() || c == '_')).map(|i| i + head[i..].chars().next().map_or(1, char::len_utf8)).unwrap_or(0);
        (start, &head[start..])
    }
}

fn push_output(lines: &mut Vec<Line>, output: &str) {
    let out = output.trim_end();
    if !out.is_empty() {
        lines.push(line(Role::Output, out.to_string()));
    }
}

/// "line 3, column 5: message", the offending source line and a caret.
pub fn error_block(e: &ScriptError, source: &str) -> String {
    let message = strip_position(&e.message);
    let mut s = match (e.line, e.column) {
        (Some(l), Some(c)) => format!("line {l}, column {c}: {message}"),
        (Some(l), None) => format!("line {l}: {message}"),
        _ => message.to_string(),
    };
    if let Some(l) = e.line {
        if let Some(src) = source.lines().nth(l.saturating_sub(1)) {
            let gutter = format!("{l} | ");
            s.push_str(&format!("\n{gutter}{src}"));
            if let Some(c) = e.column {
                s.push_str(&format!("\n{}^", " ".repeat(gutter.len() + c.saturating_sub(1))));
            }
        }
    }
    s
}

/// Rhai appends " (line N, position M)" to most messages; the console prints
/// the position itself.
fn strip_position(m: &str) -> &str {
    match m.rfind(" (line ") {
        Some(i) if m.ends_with(')') => &m[..i],
        _ => m,
    }
}

// ---- completeness scanner ----------------------------------------------

fn complete(text: &str) -> bool {
    let mut stack: Vec<char> = Vec::new();
    let mut chars = text.chars().peekable();
    let mut last_sig: Option<char> = None;
    let mut block_comment = 0usize;
    while let Some(c) = chars.next() {
        if block_comment > 0 {
            if c == '*' && chars.peek() == Some(&'/') {
                chars.next();
                block_comment -= 1;
            } else if c == '/' && chars.peek() == Some(&'*') {
                chars.next();
                block_comment += 1;
            }
            continue;
        }
        match c {
            '/' if chars.peek() == Some(&'/') => {
                for n in chars.by_ref() {
                    if n == '\n' {
                        break;
                    }
                }
            }
            '/' if chars.peek() == Some(&'*') => {
                chars.next();
                block_comment = 1;
            }
            '"' => {
                if !skip_quoted(&mut chars, '"', true) {
                    return false;
                }
                last_sig = Some('"');
            }
            '\'' => {
                if !skip_quoted(&mut chars, '\'', false) {
                    return false;
                }
                last_sig = Some('\'');
            }
            '`' => {
                if !skip_quoted(&mut chars, '`', true) {
                    return false;
                }
                last_sig = Some('`');
            }
            '(' | '[' | '{' => {
                stack.push(c);
                last_sig = Some(c);
            }
            ')' | ']' | '}' => {
                stack.pop();
                last_sig = Some(c);
            }
            c if c.is_whitespace() => {}
            c => last_sig = Some(c),
        }
    }
    if block_comment > 0 || !stack.is_empty() {
        return false;
    }
    !matches!(last_sig, Some('+' | '-' | '*' | '/' | '%' | '=' | ',' | '.' | '&' | '|' | '^' | '<' | '>'))
}

/// Consume up to the closing `quote`; `false` when the input ends first.
/// Single-quoted char literals end at the line's end instead (an apostrophe
/// in a comment-less typo must not swallow the rest).
fn skip_quoted(chars: &mut std::iter::Peekable<std::str::Chars<'_>>, quote: char, multiline: bool) -> bool {
    while let Some(c) = chars.next() {
        if c == '\\' {
            chars.next();
        } else if c == quote || (c == '\n' && !multiline) {
            return true;
        }
    }
    false
}

// ---- API names ----------------------------------------------------------

fn build_api(engine: &mut ScriptEngine) -> Vec<Completion> {
    let mut by_key: BTreeMap<(String, CompletionKind), Vec<String>> = BTreeMap::new();
    for sig in engine.signatures() {
        let Some(open) = sig.find('(') else { continue };
        let raw_name = &sig[..open];
        let (name, kind) = if let Some(p) = raw_name.strip_prefix("get$").or_else(|| raw_name.strip_prefix("set$")) {
            (p.to_string(), CompletionKind::Property)
        } else if raw_name.chars().next().is_some_and(|c| c.is_alphabetic() || c == '_') {
            let kind = if first_param_is_object(&sig[open..]) { CompletionKind::Method } else { CompletionKind::Function };
            (raw_name.to_string(), kind)
        } else {
            continue; // operators
        };
        let entry = by_key.entry((name, kind)).or_default();
        if kind != CompletionKind::Property {
            let t = tidy(&sig);
            if !entry.contains(&t) {
                entry.push(t);
            }
        }
    }
    by_key.into_iter().map(|((name, kind), mut sigs)| {
        sigs.sort();
        Completion { name, kind, signature: sigs.join("\n") }
    }).collect()
}

const OBJECT_TYPES: &[&str] = &["Face", "Edge", "Group", "Vec3"];

fn first_param_is_object(params: &str) -> bool {
    let inner = params.trim_start_matches('(');
    let first = inner.split([',', ')']).next().unwrap_or("");
    let ty = first.trim().trim_start_matches("_:").trim().trim_start_matches("&mut ").trim();
    OBJECT_TYPES.contains(&ty)
}

/// Rhai prints Rust paths and `Result` wrappers; show what a script sees.
fn tidy(sig: &str) -> String {
    let mut s = sig.to_string();
    const WRAP_START: &str = "core::result::Result<";
    const WRAP_END: &str = ",alloc::boxed::Box<rhai::types::error::EvalAltResult>>";
    if let Some(i) = s.find(WRAP_START) {
        if s.ends_with(WRAP_END) {
            let inner = s[i + WRAP_START.len()..s.len() - WRAP_END.len()].to_string();
            s.truncate(i);
            s.push_str(&inner);
        }
    }
    for (from, to) in [
        ("alloc::vec::Vec<rhai::types::dynamic::Dynamic>", "array"),
        ("rhai::types::dynamic::Dynamic", "any"),
        ("types::dynamic::Dynamic", "any"),
        ("alloc::string::String", "string"),
        ("ingetrazo_math::qt::V3", "Vec3"),
        ("ingetrazo_script::convert::GroupH", "Group"),
        ("ingetrazo_script::convert::FaceH", "Face"),
        ("ingetrazo_script::convert::EdgeH", "Edge"),
        ("_: ", ""),
    ] {
        s = s.replace(from, to);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completeness() {
        assert!(complete("let a = 1;"));
        assert!(complete("faces().len()"));
        assert!(!complete("if true {"));
        assert!(!complete("let a = [1,"));
        assert!(!complete("print(\"abc"));
        assert!(complete("print(\"a{b\")"));
        assert!(complete("let a = 1; // {"));
        assert!(!complete("/* open"));
        assert!(complete("/* closed */ 1"));
        assert!(!complete("let x = 1 +"));
        assert!(!complete("let s = `multi"));
        assert!(complete("let s = '{';"));
    }

    #[test]
    fn prefix_at_cursor() {
        assert_eq!(Console::completion_prefix("let a = add_p", 13), (8, "add_p"));
        assert_eq!(Console::completion_prefix("foo(fa", 6), (4, "fa"));
        assert_eq!(Console::completion_prefix("x.", 2), (2, ""));
    }

    #[test]
    fn tidy_signatures() {
        let t = tidy("make_group(_: types::dynamic::Dynamic, _: string) -> core::result::Result<ingetrazo_script::convert::GroupH,alloc::boxed::Box<rhai::types::error::EvalAltResult>>");
        assert_eq!(t, "make_group(any, string) -> Group");
    }
}
