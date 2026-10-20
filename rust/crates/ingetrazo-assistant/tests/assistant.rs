// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The assistant loop (M8.4) against a fake transport: no network, no real
//! key. The model's replies are scripted; what is checked is what the loop
//! sends, what it runs and what it leaves in the document.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::time::Duration;

use ingetrazo_assistant::prompt::{self, compact_messages, extract_code, strip_thoughts, truncated_code};
use ingetrazo_assistant::provider::{HttpRequest, HttpResponse, TransportError};
use ingetrazo_assistant::{Assistant, Config, Event, Message, Outcome, Provider, Retry, Step, StopReason, Transport};
use ingetrazo_model::scene::Scene;
use ingetrazo_ops::command::{Cx, History};
use ingetrazo_script::console::Console;
use serde_json::{json, Value};

const KEY: &str = "sk-ant-TEST-not-a-real-key-0000";

type Answers = Rc<RefCell<VecDeque<Result<HttpResponse, TransportError>>>>;
type Seen = Rc<RefCell<Vec<HttpRequest>>>;

struct Fake {
    answers: Answers,
    seen: Seen,
}

impl Transport for Fake {
    fn post(&self, req: &HttpRequest) -> Result<HttpResponse, TransportError> {
        self.seen.borrow_mut().push(req.clone());
        self.answers.borrow_mut().pop_front().expect("the fake ran out of answers: the loop asked for more turns than scripted")
    }
}

struct Rig {
    asst: Assistant<Fake>,
    answers: Answers,
    seen: Seen,
    scene: Scene,
    cx: Cx,
    hist: History,
    console: Console,
    events: Vec<Event>,
}

fn anthropic(text: &str) -> Result<HttpResponse, TransportError> {
    Ok(HttpResponse { status: 200, body: json!({"content": [{"type": "text", "text": text}]}).to_string() })
}

fn block(code: &str) -> String {
    format!("Voy a dibujarlo.\n```rhai\n{code}\n```")
}

impl Rig {
    fn new(config: Config) -> Rig {
        let answers: Answers = Rc::default();
        let seen: Seen = Rc::default();
        let mut asst = Assistant::new(config, Fake { answers: answers.clone(), seen: seen.clone() });
        asst.set_sleep(|_| {});
        Rig { asst, answers, seen, scene: Scene::new(), cx: Cx::default(), hist: History::new(), console: Console::default(), events: Vec::new() }
    }

    fn anthropic() -> Rig {
        Rig::new(Config::new(KEY, None, "", ""))
    }

    fn queue(&self, a: Result<HttpResponse, TransportError>) {
        self.answers.borrow_mut().push_back(a);
    }

    fn ask(&mut self, prompt: &str) -> Outcome {
        let events = &mut self.events;
        self.asst.send(prompt, &mut self.scene, &mut self.cx, &mut self.hist, &mut self.console, &mut |e| events.push(e))
    }

    fn faces(&self) -> usize {
        self.cx.mesh(&self.scene).faces.len()
    }

    fn body(&self, n: usize) -> Value {
        serde_json::from_str(&self.seen.borrow()[n].body).unwrap()
    }
}

const QUAD: &str = "add_face([vec3(0,0,0), vec3(2,0,0), vec3(2,2,0), vec3(0,2,0)]);";

#[test]
fn the_prompt_is_built_from_the_registered_api() {
    let mut c = Console::default();
    let p = prompt::system_prompt(&mut c);
    // The hand-written reference...
    assert!(p.contains("add_polygon(points)"));
    // ...and the registry's exact signatures.
    assert!(p.contains("add_polygon(array)"), "{p}");
    assert!(p.contains("make_group(any, string) -> Group"), "{p}");
    assert!(p.contains("```rhai"));
    assert!(!p.to_lowercase().contains("python"));
    // Every callable the registry has is in it.
    for comp in c.completions("") {
        if matches!(comp.kind, ingetrazo_script::console::CompletionKind::Function) && comp.name != "print" && comp.name != "debug" {
            assert!(p.contains(&format!("{}(", comp.name)), "{} missing", comp.name);
        }
    }
}

#[test]
fn the_prompts_example_really_runs() {
    let mut rig = Rig::anthropic();
    let s = rig.console.submit(&mut rig.scene, &mut rig.cx, &mut rig.hist, prompt::EXAMPLE);
    assert!(s.ok, "{:?}", s.error);
    assert!(s.changed);
    assert_eq!(s.output, "caras: 6\n");
    assert_eq!(rig.scene.groups.len(), 1);
}

#[test]
fn reply_plumbing() {
    let t = "Antes\n```rhai\nlet a = 1;\n```\nDespués";
    assert_eq!(extract_code(t).as_deref(), Some("let a = 1;"));
    assert_eq!(extract_code("```rs\nx\n```").as_deref(), Some("x"));
    assert_eq!(extract_code("```\nbare\n```").as_deref(), Some("bare"));
    assert_eq!(extract_code("```python\nprint(1)\n```"), None);
    assert_eq!(extract_code("sin código"), None);
    assert_eq!(extract_code("```rhai\n\n```"), None);
    assert!(truncated_code("```rhai\nlet a = "));
    assert!(!truncated_code("```rhai\nlet a = 1;\n```"));
    assert!(!truncated_code("nada"));
    assert_eq!(strip_thoughts("<think>hmm</think>Hola"), "Hola");
    assert_eq!(strip_thoughts("Hola<thinking>sin cerrar"), "Hola");
    assert_eq!(strip_thoughts("pensé</thought>Hola"), "Hola");

    let convo = vec![
        Message::user("uno"),
        Message::assistant("a\n```rhai\nlet x = 1;\n```\nb"),
        Message::user("r1"),
        Message::assistant("```rhai\nlet y = 2;\n```"),
        Message::user("r2"),
        Message::assistant("```rhai\nlet z = 3;\n```"),
    ];
    let c = compact_messages(&convo, 2);
    assert_eq!(c[1].text, format!("a\n{}\nb", prompt::CODE_STUB));
    assert!(c[3].text.contains("let y = 2;") && c[5].text.contains("let z = 3;"));
    assert_eq!(convo[1].text, "a\n```rhai\nlet x = 1;\n```\nb", "the stored convo is untouched");
}

#[test]
fn a_request_runs_the_block_as_one_undo_step_and_answers() {
    let mut rig = Rig::anthropic();
    rig.queue(anthropic(&block(&format!("{QUAD}\n{QUAD}\nprint(\"hecho\");"))));
    rig.queue(anthropic("Listo: dibujé un cuadrado."));
    let out = rig.ask("dibuja un cuadrado de 2 m");
    assert_eq!(out, Outcome::Done { summary: "Listo: dibujé un cuadrado.".into(), changed: true });
    assert!(rig.faces() >= 1);
    assert_eq!(rig.hist.undo_len(), 1, "one block, one undo step");
    // The feedback the model got.
    let second = rig.body(1);
    let msgs = second["messages"].as_array().unwrap();
    let last = msgs.last().unwrap()["content"][0]["text"].as_str().unwrap();
    assert!(last.starts_with("Resultado de la ejecución:"), "{last}");
    assert!(last.contains("hecho") && last.contains("(cambió el modelo: true)"), "{last}");
    assert!(rig.events.iter().any(|e| matches!(e, Event::Running)));
    // Undo takes it all back.
    assert!(rig.hist.undo(&mut rig.scene, &mut rig.cx));
    assert_eq!(rig.faces(), 0);
}

#[test]
fn the_request_is_anthropics_messages_api_with_the_default_model_and_no_key_in_the_body() {
    let mut rig = Rig::anthropic();
    rig.queue(anthropic("Hola."));
    assert!(matches!(rig.ask("hola"), Outcome::Done { changed: false, .. }));
    let req = rig.seen.borrow()[0].clone();
    assert_eq!(req.url, "https://api.anthropic.com/v1/messages");
    assert!(req.headers.iter().any(|(k, v)| k == "x-api-key" && v == KEY));
    assert!(req.headers.iter().any(|(k, v)| k == "anthropic-version" && v == "2023-06-01"));
    let body = rig.body(0);
    assert_eq!(body["model"], "claude-sonnet-5");
    assert_eq!(body["max_tokens"], 8192);
    assert!(body["system"].as_str().unwrap().contains("```rhai"));
    assert!(!req.body.contains(KEY));
    // Nothing a log line could print shows the key.
    let shown = format!("{:?} {:?} {:?}", req, rig.asst.config, rig.asst.config.key);
    assert!(!shown.contains(KEY) && !shown.contains("TEST-not-a-real"), "{shown}");
}

#[test]
fn a_chosen_model_wins_over_the_default() {
    let mut rig = Rig::new(Config::new(KEY, None, "claude-opus-5-5", ""));
    rig.queue(anthropic("Hola."));
    rig.ask("hola");
    assert_eq!(rig.body(0)["model"], "claude-opus-5-5");
}

#[test]
fn other_providers_use_the_openai_compatible_endpoint() {
    let mut rig = Rig::new(Config::new("gsk_TESTTESTTESTTEST", None, "", ""));
    rig.queue(Ok(HttpResponse { status: 200, body: json!({"choices": [{"message": {"content": "Hola."}}]}).to_string() }));
    let out = rig.ask("hola");
    assert_eq!(out, Outcome::Done { summary: "Hola.".into(), changed: false });
    let req = rig.seen.borrow()[0].clone();
    assert_eq!(req.url, "https://api.groq.com/openai/v1/chat/completions");
    assert!(req.headers.iter().any(|(k, v)| k == "Authorization" && v == "Bearer gsk_TESTTESTTESTTEST"));
    let body = rig.body(0);
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(body["max_tokens"], 8192);

    let mut local = Rig::new(Config::new("", None, "", "http://localhost:11434"));
    local.queue(Ok(HttpResponse { status: 200, body: json!({"choices": [{"message": {"content": "ok"}}]}).to_string() }));
    local.ask("hola");
    assert_eq!(local.seen.borrow()[0].url, "http://localhost:11434/v1/chat/completions");
    assert!(local.seen.borrow()[0].headers.iter().all(|(k, _)| k != "Authorization"));
    assert_eq!(Provider::detect("sk-or-x"), Provider::OpenRouter);
    assert_eq!(Provider::detect("sk-proj-x"), Provider::OpenAi);
    assert_eq!(Provider::detect("AIzaX"), Provider::Gemini);
}

#[test]
fn a_failing_script_never_touches_the_document_and_the_error_goes_back() {
    let mut rig = Rig::anthropic();
    // It draws, THEN fails: the dry run must catch it before the document does.
    rig.queue(anthropic(&block(&format!("{QUAD}\nlet n = 3;\nno_existe(n);"))));
    rig.queue(anthropic(&block(QUAD)));
    rig.queue(anthropic("Listo."));
    let out = rig.ask("dibuja un cuadrado");
    assert_eq!(out, Outcome::Done { summary: "Listo.".into(), changed: true });
    assert_eq!(rig.hist.undo_len(), 1, "only the good block reached the document");
    // The second request carries the error with line and column.
    let body = rig.body(1);
    let msgs = body["messages"].as_array().unwrap();
    let fb = msgs.last().unwrap()["content"][0]["text"].as_str().unwrap();
    assert!(fb.contains("ERROR (nada se ejecutó"), "{fb}");
    assert!(fb.contains("line 3, column "), "{fb}");
    assert!(fb.contains("3 | no_existe(n);"), "{fb}");
    // The failed block's code was not applied to the live document first.
    assert_eq!(rig.faces(), 1);
}

#[test]
fn repeated_failures_stop_the_loop_with_the_document_untouched() {
    let mut rig = Rig::anthropic();
    for _ in 0..3 {
        rig.queue(anthropic(&block("boom();")));
    }
    let out = rig.ask("dibuja algo");
    assert_eq!(out, Outcome::Stopped { reason: StopReason::TooManyFailures, changed: false });
    assert_eq!(rig.seen.borrow().len(), 3);
    assert_eq!(rig.hist.undo_len(), 0);
    assert_eq!(rig.faces(), 0);
}

#[test]
fn a_success_resets_the_failure_count_and_the_round_cap_stops_a_runaway() {
    let mut cfg = Config::new(KEY, None, "", "");
    cfg.max_rounds = 3;
    let mut rig = Rig::new(cfg);
    for _ in 0..4 {
        rig.queue(anthropic(&block("print(1);")));
    }
    let out = rig.ask("dibuja algo");
    assert_eq!(out, Outcome::Stopped { reason: StopReason::StepLimit, changed: false });
    assert_eq!(rig.seen.borrow().len(), 4);

    let mut rig = Rig::anthropic();
    rig.queue(anthropic(&block("boom();")));
    rig.queue(anthropic(&block("boom();")));
    rig.queue(anthropic(&block("print(1);")));
    rig.queue(anthropic(&block("boom();")));
    rig.queue(anthropic(&block("boom();")));
    rig.queue(anthropic("fin"));
    assert!(matches!(rig.ask("dibuja algo"), Outcome::Done { .. }));
}

#[test]
fn session_variables_carry_over_between_blocks() {
    let mut rig = Rig::anthropic();
    rig.queue(anthropic(&block("let ancho = 2.0;")));
    rig.queue(anthropic(&block("print(ancho * 2.0);")));
    rig.queue(anthropic("fin"));
    rig.ask("calcula");
    let fb = &rig.asst.conversation()[4].text;
    assert!(fb.contains("4.0"), "{fb}");
}

#[test]
fn a_busy_provider_is_retried_and_the_key_is_scrubbed_from_errors() {
    let mut cfg = Config::new(KEY, None, "", "");
    cfg.retry = Retry { backoff: vec![Duration::ZERO, Duration::ZERO] };
    let mut rig = Rig::new(cfg);
    rig.queue(Ok(HttpResponse { status: 503, body: "high demand".into() }));
    rig.queue(Err(TransportError("timed out".into())));
    rig.queue(anthropic("Hola."));
    assert!(matches!(rig.ask("hola"), Outcome::Done { .. }));
    let retries: Vec<_> = rig.events.iter().filter(|e| matches!(e, Event::Retry { .. })).collect();
    assert_eq!(retries.len(), 2);

    let mut rig = Rig::anthropic();
    rig.queue(Ok(HttpResponse { status: 401, body: format!("invalid x-api-key: {KEY}") }));
    let Outcome::Failed { message, changed } = rig.ask("hola") else { panic!("expected a failure") };
    assert!(!changed);
    assert!(message.starts_with("HTTP 401"), "{message}");
    assert!(!message.contains(KEY) && message.contains("***"), "{message}");
    assert_eq!(rig.seen.borrow().len(), 1, "a 401 is not retried");
}

#[test]
fn no_key_fails_before_any_network_call() {
    let mut rig = Rig::new(Config::new("", Some(Provider::Anthropic), "", ""));
    let Outcome::Failed { message, .. } = rig.ask("hola") else { panic!("expected a failure") };
    assert!(message.contains("clave"), "{message}");
    assert!(rig.seen.borrow().is_empty());
}

#[test]
fn a_cut_off_reply_and_a_narrated_build_each_get_one_push() {
    let mut rig = Rig::anthropic();
    rig.queue(anthropic("```rhai\nadd_face(["));
    rig.queue(anthropic(&block(QUAD)));
    rig.queue(anthropic("Listo."));
    assert!(matches!(rig.ask("dibuja un cuadrado"), Outcome::Done { changed: true, .. }));
    assert!(rig.events.iter().any(|e| matches!(e, Event::Note(n) if n.contains("cut off"))));

    let mut rig = Rig::anthropic();
    rig.queue(anthropic("Se añadió el cuadrado."));
    rig.queue(anthropic(&block(QUAD)));
    rig.queue(anthropic("Listo."));
    assert!(matches!(rig.ask("dibuja un cuadrado"), Outcome::Done { changed: true, .. }));
    // A plain question is answered, not pushed.
    let mut rig = Rig::anthropic();
    rig.queue(anthropic("Son 3 caras."));
    assert!(matches!(rig.ask("¿cuántas caras hay?"), Outcome::Done { changed: false, .. }));
    assert_eq!(rig.seen.borrow().len(), 1);
}

#[test]
fn cancelling_stops_at_the_next_turn() {
    let mut rig = Rig::anthropic();
    rig.queue(anthropic(&block(QUAD)));
    let token = rig.asst.cancel_token();
    // The "user" presses Stop as soon as the first block has run.
    let events = &mut rig.events;
    let out = rig.asst.send("dibuja", &mut rig.scene, &mut rig.cx, &mut rig.hist, &mut rig.console, &mut |e| {
        if matches!(e, Event::Feedback(_)) {
            token.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        events.push(e);
    });
    assert_eq!(out, Outcome::Stopped { reason: StopReason::Cancelled, changed: true });
    assert_eq!(rig.seen.borrow().len(), 1);
}

#[test]
fn the_step_api_lets_a_ui_keep_the_document_while_the_network_runs_elsewhere() {
    fn assert_send<T: Send>() {}
    assert_send::<ingetrazo_assistant::Turn>();

    let mut rig = Rig::anthropic();
    rig.queue(anthropic(&block(QUAD)));
    rig.queue(anthropic("Listo."));
    rig.asst.start("dibuja un cuadrado", &mut rig.console);
    let mut steps = 0;
    loop {
        let turn = rig.asst.turn().unwrap();
        // This is the part a worker thread would do.
        let answer = turn.run(&Fake { answers: rig.answers.clone(), seen: rig.seen.clone() }, &|_| {}, &mut |_, _, _, _| {});
        let step = rig.asst.reply(answer, &mut rig.scene, &mut rig.cx, &mut rig.hist, &mut rig.console, &mut |_| {});
        steps += 1;
        if let Step::Finished(o) = step {
            assert!(matches!(o, Outcome::Done { changed: true, .. }));
            break;
        }
    }
    assert_eq!(steps, 2);
}

#[test]
fn what_the_assistant_ran_shows_in_the_console_transcript_not_its_input_history() {
    let mut rig = Rig::anthropic();
    rig.queue(anthropic(&block("print(\"visible\");")));
    rig.queue(anthropic("fin"));
    rig.ask("calcula");
    assert!(rig.console.transcript().iter().any(|l| l.text == "visible"));
    assert!(rig.console.history().is_empty());
}
