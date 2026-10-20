// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! IngeTrazo's AI assistant (M8.4), the port of `plugins/ai_assistant.py` and
//! `core/ai.py`.
//!
//! The Python assistant chatted with a model, took the ONE fenced Python
//! block of each reply, `exec`'d it against live objects as one undoable step
//! and sent the result (stdout, errors, a viewport screenshot) back, round
//! after round. This one is the same loop over Rhai:
//!
//! 1. [`prompt::system_prompt`] teaches the model the script API, built from
//!    the engine's own registry so it cannot drift.
//! 2. [`Assistant::send`] sends the conversation ([`provider`]: Anthropic's
//!    Messages API, or any OpenAI-compatible endpoint; one blocking call per
//!    turn behind the [`provider::Transport`] trait), strips the model's
//!    thoughts, takes the ```` ```rhai ```` block.
//! 3. The block is first run **on a copy** of the document
//!    ([`Console::dry_run`]): if it fails, nothing happens to the document
//!    and the error (message, line, column, the offending line) goes back to
//!    the model. If it passes it runs for real through the console
//!    ([`Console::run_script`]), as ONE undo step of the user's history.
//! 4. The result goes back as the next turn; the loop ends when a reply has
//!    no block, after [`Config::max_rounds`] rounds, or after
//!    [`Config::max_failed_rounds`] failures in a row.
//!
//! The key is the user's (`ia/api_key` in the Python app's settings); it is
//! kept in an [`provider::ApiKey`], sent only as a request header, and
//! scrubbed from any error text. Not ported: viewport screenshots to the
//! model (the UI can attach them later) and the MCP bridge (2.1).

pub mod prompt;
pub mod provider;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use ingetrazo_model::scene::Scene;
use ingetrazo_ops::command::{Cx, History};
use ingetrazo_script::console::{error_block, Console};

pub use provider::{ApiKey, Message, Provider, Retry, Role, Transport, UreqTransport};

/// Everything the loop needs to know.
#[derive(Clone, Debug)]
pub struct Config {
    pub provider: Provider,
    /// Empty = the provider's default ([`Provider::default_model`]).
    pub model: String,
    pub key: ApiKey,
    /// Base URL of the local provider (Ollama...).
    pub local_url: String,
    /// Replies per request (the cap for multi-step work that keeps working).
    pub max_rounds: usize,
    /// Failed blocks in a row before control goes back to the user.
    pub max_failed_rounds: usize,
    /// Output tokens per turn (0 = the provider's default).
    pub max_tokens: u32,
    pub retry: Retry,
}

impl Config {
    /// A config from the settings the Python app kept (`ia/api_key`,
    /// `ia/proveedor`, `ia/modelo`, `ia/ollama_url`): `provider` None = auto,
    /// by the key's prefix.
    pub fn new(key: &str, provider: Option<Provider>, model: &str, local_url: &str) -> Config {
        Config {
            provider: provider.unwrap_or_else(|| Provider::detect(key)),
            model: model.trim().to_string(),
            key: ApiKey::new(key),
            local_url: if local_url.trim().is_empty() { "http://localhost:11434".to_string() } else { local_url.trim().to_string() },
            max_rounds: 12,
            max_failed_rounds: 3,
            max_tokens: 0,
            retry: Retry::default(),
        }
    }

    pub fn effective_model(&self) -> &str {
        if self.model.is_empty() {
            self.provider.default_model()
        } else {
            &self.model
        }
    }

    /// Why a request cannot even start (known before any network call).
    pub fn missing_key_hint(&self) -> Option<String> {
        if self.provider != Provider::Ollama && self.key.is_empty() {
            Some(format!("Falta la clave de API de {}: consíguela en {}.", self.provider.id(), self.provider.key_url()))
        } else {
            None
        }
    }
}

/// What the UI hears while a request runs.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// A turn was sent; waiting for the model.
    Thinking,
    /// The provider is busy: retry `attempt` of `total` in `wait`.
    Retry { attempt: usize, total: usize, wait: Duration, reason: String },
    /// The model's reply (thoughts removed).
    Reply(String),
    /// The block passed the dry run and is being run for real.
    Running,
    /// What went back to the model after a block.
    Feedback(String),
    /// A note from the loop (a nudge, a cut-off reply).
    Note(String),
}

/// Why the loop stopped without a final answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// `max_rounds` replies in this request: "continue" resumes.
    StepLimit,
    /// `max_failed_rounds` blocks failed in a row.
    TooManyFailures,
    /// The user cancelled.
    Cancelled,
}

/// How a request ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The model answered without a block: its text is the summary.
    Done { summary: String, changed: bool },
    Stopped { reason: StopReason, changed: bool },
    /// The provider failed (after retries); the message is readable.
    Failed { message: String, changed: bool },
}

impl Outcome {
    /// At least one block changed the document (each is an undo step).
    pub fn changed(&self) -> bool {
        match self {
            Outcome::Done { changed, .. } | Outcome::Stopped { changed, .. } | Outcome::Failed { changed, .. } => *changed,
        }
    }
}

/// Output longer than this is cut in the middle before it goes back (a
/// print-happy recipe would ride EVERY turn).
const FEEDBACK_OUTPUT_MAX: usize = 1500;

/// One model turn, ready to send. It owns everything it needs and is `Send`,
/// so the UI can run [`Turn::run`] on a worker thread while the document and
/// the console stay where they are.
#[derive(Clone, Debug)]
pub struct Turn {
    provider: Provider,
    model: String,
    key: ApiKey,
    local_url: String,
    system: String,
    messages: Vec<Message>,
    max_tokens: u32,
    retry: Retry,
}

impl Turn {
    /// The blocking network call (with the provider's busy-retries).
    /// `on_retry(attempt, total, wait, reason)` hears each retry.
    pub fn run(&self, transport: &dyn Transport, sleep: &dyn Fn(Duration), on_retry: &mut dyn FnMut(usize, usize, Duration, &str)) -> Result<String, String> {
        provider::chat(transport, self.provider, &self.model, &self.key, &self.system, &self.messages, &self.local_url, self.max_tokens, &self.retry, sleep, on_retry)
    }
}

/// What to do after a reply was handled.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    /// Another model turn is needed: call [`Assistant::turn`] again.
    Next,
    Finished(Outcome),
}

#[derive(Default)]
struct RunState {
    round: usize,
    failed: usize,
    nudged: bool,
    changed: bool,
    prompt: String,
}

/// The conversation and its loop.
///
/// Two ways to drive it. [`Assistant::send`] does the whole request on the
/// calling thread (simple; tests). A UI that must not block calls
/// [`Assistant::start`], then repeats: `turn()` -> run it on a worker
/// ([`Turn::run`]) -> back on the UI thread, `reply(result, ...)` until it
/// answers [`Step::Finished`].
pub struct Assistant<T: Transport> {
    pub config: Config,
    transport: T,
    convo: Vec<Message>,
    system: Option<String>,
    cancel: Arc<AtomicBool>,
    sleep: Box<dyn Fn(Duration)>,
    state: RunState,
}

impl<T: Transport> Assistant<T> {
    pub fn new(config: Config, transport: T) -> Assistant<T> {
        Assistant { config, transport, convo: Vec::new(), system: None, cancel: Arc::new(AtomicBool::new(false)), sleep: Box::new(std::thread::sleep), state: RunState::default() }
    }

    /// Replace how retries wait (tests pass a no-op).
    pub fn set_sleep(&mut self, sleep: impl Fn(Duration) + 'static) {
        self.sleep = Box::new(sleep);
    }

    /// Set it to stop the request at the next turn boundary.
    pub fn cancel_token(&self) -> Arc<AtomicBool> {
        self.cancel.clone()
    }

    /// The conversation so far.
    pub fn conversation(&self) -> &[Message] {
        &self.convo
    }

    /// The "new chat" button.
    pub fn reset(&mut self) {
        self.convo.clear();
        self.system = None;
    }

    /// Begin a request: the user's message joins the conversation.
    /// `console` supplies the API the system prompt describes.
    pub fn start(&mut self, prompt: &str, console: &mut Console) {
        self.cancel.store(false, Ordering::Relaxed);
        if self.system.is_none() {
            self.system = Some(prompt::system_prompt(console));
        }
        self.convo.push(Message::user(prompt));
        self.state = RunState { prompt: prompt.to_string(), ..RunState::default() };
    }

    /// The next network turn. An `Err` is a reason the request cannot be
    /// sent at all (no key); show it and stop.
    pub fn turn(&self) -> Result<Turn, String> {
        let cfg = &self.config;
        if let Some(hint) = cfg.missing_key_hint() {
            return Err(hint);
        }
        Ok(Turn {
            provider: cfg.provider,
            model: cfg.effective_model().to_string(),
            key: cfg.key.clone(),
            local_url: cfg.local_url.clone(),
            system: self.system.clone().unwrap_or_default(),
            messages: prompt::compact_messages(&self.convo, 2),
            max_tokens: if cfg.max_tokens == 0 { cfg.provider.max_tokens() } else { cfg.max_tokens },
            retry: cfg.retry.clone(),
        })
    }

    /// Handle what the network answered: run the model's block (vetted on a
    /// copy first) and decide whether another turn follows.
    pub fn reply(&mut self, answer: Result<String, String>, scene: &mut Scene, cx: &mut Cx, history: &mut History, console: &mut Console, on_event: &mut dyn FnMut(Event)) -> Step {
        let changed = self.state.changed;
        let text = match answer {
            Ok(t) => t,
            Err(message) => return Step::Finished(Outcome::Failed { message, changed }),
        };
        if self.cancel.load(Ordering::Relaxed) {
            return Step::Finished(Outcome::Stopped { reason: StopReason::Cancelled, changed });
        }
        let text = prompt::strip_thoughts(&text);
        self.convo.push(Message::assistant(text.clone()));
        on_event(Event::Reply(text.clone()));

        let Some(code) = prompt::extract_code(&text) else {
            if prompt::truncated_code(&text) {
                // Cut by the token limit mid-recipe: half a block must
                // neither run nor end the loop silently.
                if self.state.round < self.config.max_rounds {
                    self.state.round += 1;
                    on_event(Event::Note("The reply was cut off mid-code: asking for a shorter, complete block.".into()));
                    self.convo.push(Message::user(
                        "Tu respuesta se cortó a mitad del bloque ```rhai (límite de tokens). NO continúes donde quedaste: reenvía UN bloque completo y más corto, avanzando solo la primera parte del trabajo; el resto irá en turnos siguientes.",
                    ));
                    return Step::Next;
                }
                return Step::Finished(Outcome::Stopped { reason: StopReason::StepLimit, changed });
            }
            if self.state.round == 0 && !self.state.nudged && prompt::asks_to_build(&self.state.prompt) && !text.chars().rev().take(80).any(|c| c == '?') {
                // A model that narrates a build it never sent: one push.
                self.state.nudged = true;
                on_event(Event::Note("No code came back: asking for the script.".into()));
                self.convo.push(Message::user("No incluiste ningún bloque ```rhai: NADA se ejecutó y el modelo no cambió. Escribe ahora el script completo en UN bloque ```rhai (sin describirlo antes)."));
                return Step::Next;
            }
            return Step::Finished(Outcome::Done { summary: text, changed });
        };

        if self.state.round >= self.config.max_rounds {
            // The convo survives: any new message resumes from here.
            return Step::Finished(Outcome::Stopped { reason: StopReason::StepLimit, changed });
        }
        self.state.round += 1;

        let (feedback, ran) = run_block(&code, scene, cx, history, console, on_event);
        self.state.changed |= ran.changed;
        self.state.failed = if ran.ok { 0 } else { self.state.failed + 1 };
        on_event(Event::Feedback(feedback.clone()));
        self.convo.push(Message::user(feedback));
        if self.state.failed >= self.config.max_failed_rounds {
            return Step::Finished(Outcome::Stopped { reason: StopReason::TooManyFailures, changed: self.state.changed });
        }
        Step::Next
    }

    /// One whole request on the calling thread: blocks on the network.
    pub fn send(&mut self, prompt: &str, scene: &mut Scene, cx: &mut Cx, history: &mut History, console: &mut Console, on_event: &mut dyn FnMut(Event)) -> Outcome {
        self.start(prompt, console);
        loop {
            if self.cancel.load(Ordering::Relaxed) {
                return Outcome::Stopped { reason: StopReason::Cancelled, changed: self.state.changed };
            }
            let turn = match self.turn() {
                Ok(t) => t,
                Err(message) => return Outcome::Failed { message, changed: self.state.changed },
            };
            on_event(Event::Thinking);
            let answer = turn.run(&self.transport, &*self.sleep, &mut |attempt, total, wait, reason| on_event(Event::Retry { attempt, total, wait, reason: reason.to_string() }));
            if let Step::Finished(o) = self.reply(answer, scene, cx, history, console, on_event) {
                return o;
            }
        }
    }
}

/// Vet the block on a copy, then run it for real. The text that goes back to
/// the model, and what happened.
fn run_block(code: &str, scene: &mut Scene, cx: &mut Cx, history: &mut History, console: &mut Console, on_event: &mut dyn FnMut(Event)) -> (String, Ran) {
    let mut parts: Vec<String> = Vec::new();
    // Dry run: a script that fails here never touches the document.
    if let Err(e) = console.dry_run(scene, cx, code) {
        push_output(&mut parts, &e.output);
        parts.push(format!("ERROR (nada se ejecutó; el documento sigue igual): {}", error_block(&e, code)));
        return (format!("Resultado de la ejecución:\n{}", parts.join("\n")), Ran { ok: false, changed: false });
    }
    on_event(Event::Running);
    let sub = console.run_script(scene, cx, history, "assistant", code);
    push_output(&mut parts, &sub.output);
    if let Some(e) = &sub.error {
        // Not expected after a clean dry run, but the real document is the
        // judge; it was rolled back.
        parts.push(format!("ERROR (todo revertido): {}", error_block(e, code)));
    }
    parts.push(format!("(cambió el modelo: {})", sub.changed));
    (format!("Resultado de la ejecución:\n{}", parts.join("\n")), Ran { ok: sub.ok, changed: sub.changed })
}

struct Ran {
    ok: bool,
    changed: bool,
}

fn push_output(parts: &mut Vec<String>, output: &str) {
    let out = output.trim_end();
    if out.is_empty() {
        return;
    }
    let chars: Vec<char> = out.chars().collect();
    if chars.len() > FEEDBACK_OUTPUT_MAX {
        let head: String = chars[..700].iter().collect();
        let tail: String = chars[chars.len() - 700..].iter().collect();
        parts.push(format!("{head}\n… (recortado) …\n{tail}"));
    } else {
        parts.push(out.to_string());
    }
}
