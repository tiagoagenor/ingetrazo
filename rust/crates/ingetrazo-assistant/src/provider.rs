// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The provider layer of `core/ai.py`: ONE API key, the provider named by its
//! prefix (`gsk_` Groq, `sk-ant-` Anthropic, `AIza` Gemini, `sk-or-`
//! OpenRouter, `sk-` OpenAI) plus a local Ollama / OpenAI-compatible server.
//! Anthropic speaks its native Messages API; the others their
//! OpenAI-compatible chat endpoint: two wire formats cover all.
//!
//! Nothing here logs or prints the key: it travels only in a request header
//! and [`HttpRequest`]'s `Debug` hides it.

use std::time::Duration;

use serde_json::{json, Value};

/// The API key. Its `Debug` and `Display` never show it.
#[derive(Clone, Default)]
pub struct ApiKey(String);

impl ApiKey {
    pub fn new(key: impl Into<String>) -> ApiKey {
        ApiKey(key.into().trim().to_string())
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub(crate) fn expose(&self) -> &str {
        &self.0
    }
    /// `text` with the key (if it occurs) blotted out: provider errors
    /// sometimes echo it.
    pub(crate) fn redact(&self, text: &str) -> String {
        if self.0.len() >= 8 {
            text.replace(&self.0, "***")
        } else {
            text.to_string()
        }
    }
}

impl std::fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0.is_empty() { "ApiKey(<empty>)" } else { "ApiKey(<hidden>)" })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provider {
    Groq,
    Anthropic,
    OpenAi,
    Gemini,
    OpenRouter,
    DeepSeek,
    /// A local server (Ollama, LM Studio, vLLM...) at the configured URL.
    Ollama,
}

impl Provider {
    pub const ALL: [Provider; 7] = [Provider::Groq, Provider::Anthropic, Provider::OpenAi, Provider::Gemini, Provider::OpenRouter, Provider::DeepSeek, Provider::Ollama];

    /// The key's prefix names the provider; no key means a local server.
    pub fn detect(key: &str) -> Provider {
        let key = key.trim();
        if key.is_empty() {
            Provider::Ollama
        } else if key.starts_with("gsk_") {
            Provider::Groq
        } else if key.starts_with("sk-ant-") {
            Provider::Anthropic
        } else if key.starts_with("AIza") {
            Provider::Gemini
        } else if key.starts_with("sk-or-") {
            Provider::OpenRouter
        } else if key.starts_with("sk-") {
            Provider::OpenAi
        } else {
            Provider::Anthropic
        }
    }

    /// The model used when none is chosen (the Python app's defaults).
    pub fn default_model(self) -> &'static str {
        match self {
            Provider::Groq => "openai/gpt-oss-120b",
            Provider::Anthropic => "claude-sonnet-5",
            Provider::OpenAi => "gpt-4o",
            Provider::Gemini => "gemini-2.5-flash",
            Provider::OpenRouter => "anthropic/claude-sonnet-5",
            Provider::DeepSeek => "deepseek-chat",
            Provider::Ollama => "qwen2.5-coder:7b",
        }
    }

    /// The settings name (`ia/proveedor`).
    pub fn id(self) -> &'static str {
        match self {
            Provider::Groq => "groq",
            Provider::Anthropic => "anthropic",
            Provider::OpenAi => "openai",
            Provider::Gemini => "gemini",
            Provider::OpenRouter => "openrouter",
            Provider::DeepSeek => "deepseek",
            Provider::Ollama => "ollama",
        }
    }

    pub fn from_id(id: &str) -> Option<Provider> {
        Provider::ALL.into_iter().find(|p| p.id() == id)
    }

    /// Where the user gets a key.
    pub fn key_url(self) -> &'static str {
        match self {
            Provider::Groq => "https://console.groq.com/keys",
            Provider::Anthropic => "https://console.anthropic.com/settings/keys",
            Provider::OpenAi => "https://platform.openai.com/api-keys",
            Provider::Gemini => "https://aistudio.google.com/app/apikey",
            Provider::OpenRouter => "https://openrouter.ai/keys",
            Provider::DeepSeek => "https://platform.deepseek.com/api_keys",
            Provider::Ollama => "https://ollama.com/download",
        }
    }

    fn openai_base(self) -> Option<&'static str> {
        Some(match self {
            Provider::Groq => "https://api.groq.com/openai/v1",
            Provider::OpenAi => "https://api.openai.com/v1",
            Provider::Gemini => "https://generativelanguage.googleapis.com/v1beta/openai",
            Provider::OpenRouter => "https://openrouter.ai/api/v1",
            Provider::DeepSeek => "https://api.deepseek.com/v1",
            Provider::Anthropic | Provider::Ollama => return None,
        })
    }

    /// Output budget per turn: Gemini 2.5 bills hidden thinking against it.
    pub fn max_tokens(self) -> u32 {
        match self {
            Provider::Gemini => 16384,
            _ => 8192,
        }
    }
}

/// The API root of the local provider: `http://host:port` serves under
/// `/v1`; one given with its version is taken as is.
pub fn local_base(url: &str) -> String {
    let base = url.trim().trim_end_matches('/');
    let versioned = base.rsplit('/').next().is_some_and(|last| {
        let mut ch = last.chars();
        ch.next() == Some('v') && ch.clone().next().is_some_and(|c| c.is_ascii_digit()) && ch.all(|c| c.is_ascii_alphanumeric())
    });
    if versioned {
        base.to_string()
    } else {
        format!("{base}/v1")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub text: String,
}

impl Message {
    pub fn user(text: impl Into<String>) -> Message {
        Message { role: Role::User, text: text.into() }
    }
    pub fn assistant(text: impl Into<String>) -> Message {
        Message { role: Role::Assistant, text: text.into() }
    }
}

/// One POST: where, with which headers, what body.
#[derive(Clone)]
pub struct HttpRequest {
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: String,
}

impl std::fmt::Debug for HttpRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<&str> = self.headers.iter().map(|(k, _)| k.as_str()).collect();
        f.debug_struct("HttpRequest").field("url", &self.url).field("headers", &names).field("body_bytes", &self.body.len()).finish()
    }
}

#[derive(Clone, Debug)]
pub struct HttpResponse {
    pub status: u16,
    pub body: String,
}

/// The network failed (no answer at all).
#[derive(Clone, Debug)]
pub struct TransportError(pub String);

/// How a request reaches the provider. The app uses [`UreqTransport`]; tests
/// use a fake that never touches the network.
pub trait Transport {
    fn post(&self, req: &HttpRequest) -> Result<HttpResponse, TransportError>;
}

/// The real thing: a blocking `ureq` call. Run it on a worker thread, never
/// on the UI thread.
pub struct UreqTransport {
    pub timeout: Duration,
}

impl Default for UreqTransport {
    fn default() -> Self {
        UreqTransport { timeout: Duration::from_secs(180) }
    }
}

impl Transport for UreqTransport {
    fn post(&self, req: &HttpRequest) -> Result<HttpResponse, TransportError> {
        let agent = ureq::AgentBuilder::new().timeout(self.timeout).build();
        let mut r = agent.post(&req.url);
        for (k, v) in &req.headers {
            r = r.set(k, v);
        }
        match r.send_string(&req.body) {
            Ok(resp) => {
                let status = resp.status();
                Ok(HttpResponse { status, body: resp.into_string().map_err(|e| TransportError(e.to_string()))? })
            }
            Err(ureq::Error::Status(status, resp)) => Ok(HttpResponse { status, body: resp.into_string().unwrap_or_default() }),
            Err(e) => Err(TransportError(e.to_string())),
        }
    }
}

const USER_AGENT: &str = concat!("IngeTrazo/", env!("CARGO_PKG_VERSION"));
const ANTHROPIC_URL: &str = "https://api.anthropic.com/v1/messages";

fn role_name(r: Role) -> &'static str {
    match r {
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

/// The request for one chat turn.
pub fn build_request(provider: Provider, model: &str, key: &ApiKey, system: &str, messages: &[Message], local_url: &str, max_tokens: u32) -> HttpRequest {
    let mut headers = vec![("Content-Type".to_string(), "application/json".to_string()), ("User-Agent".to_string(), USER_AGENT.to_string())];
    if provider == Provider::Anthropic {
        headers.push(("x-api-key".into(), key.expose().to_string()));
        headers.push(("anthropic-version".into(), "2023-06-01".into()));
        let msgs: Vec<Value> = messages.iter().map(|m| json!({"role": role_name(m.role), "content": [{"type": "text", "text": m.text}]})).collect();
        let body = json!({"model": model, "max_tokens": max_tokens, "system": system, "messages": msgs});
        return HttpRequest { url: ANTHROPIC_URL.to_string(), headers, body: body.to_string() };
    }
    let base = provider.openai_base().map(str::to_string).unwrap_or_else(|| local_base(local_url));
    if !key.is_empty() {
        headers.push(("Authorization".into(), format!("Bearer {}", key.expose())));
    }
    let mut msgs = vec![json!({"role": "system", "content": system})];
    msgs.extend(messages.iter().map(|m| json!({"role": role_name(m.role), "content": m.text})));
    // OpenAI's newer models reject max_tokens; the others keep the old name.
    let limit = if provider == Provider::OpenAi { "max_completion_tokens" } else { "max_tokens" };
    let mut body = json!({"model": model, "messages": msgs});
    body[limit] = json!(max_tokens);
    HttpRequest { url: format!("{base}/chat/completions"), headers, body: body.to_string() }
}

/// The reply text of a 2xx body.
pub fn parse_reply(provider: Provider, body: &str) -> Result<String, String> {
    let v: Value = serde_json::from_str(body).map_err(|e| format!("the reply is not JSON: {e}"))?;
    if provider == Provider::Anthropic {
        let blocks = v.get("content").and_then(Value::as_array).ok_or_else(|| short(body))?;
        return Ok(blocks.iter().filter_map(|b| b.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join(""));
    }
    let first = v.get("choices").and_then(Value::as_array).and_then(|c| c.first()).ok_or_else(|| short(body))?;
    Ok(first.pointer("/message/content").and_then(Value::as_str).unwrap_or("").to_string())
}

fn short(body: &str) -> String {
    body.chars().take(400).collect()
}

/// Statuses worth another try: busy or rate-limited.
pub const RETRY_STATUSES: [u16; 6] = [429, 500, 502, 503, 504, 529];

/// How a chat turn is retried.
#[derive(Clone, Debug)]
pub struct Retry {
    /// Pauses before retry 1, 2, ... (the last repeats). Its length is the
    /// number of retries.
    pub backoff: Vec<Duration>,
}

impl Default for Retry {
    fn default() -> Self {
        Retry { backoff: [2u64, 4, 8, 16].into_iter().map(Duration::from_secs).collect() }
    }
}

impl Retry {
    pub fn none() -> Retry {
        Retry { backoff: Vec::new() }
    }
}

/// One blocking chat turn with the readable errors of the Python app. A busy
/// provider is retried; `on_retry(n, total, wait, reason)` hears each time.
#[allow(clippy::too_many_arguments)]
pub fn chat(
    transport: &dyn Transport,
    provider: Provider,
    model: &str,
    key: &ApiKey,
    system: &str,
    messages: &[Message],
    local_url: &str,
    max_tokens: u32,
    retry: &Retry,
    sleep: &dyn Fn(Duration),
    on_retry: &mut dyn FnMut(usize, usize, Duration, &str),
) -> Result<String, String> {
    let req = build_request(provider, model, key, system, messages, local_url, max_tokens);
    let total = retry.backoff.len();
    let mut attempt = 0usize;
    loop {
        let reason = match transport.post(&req) {
            Ok(resp) if (200..300).contains(&resp.status) => return parse_reply(provider, &resp.body).map_err(|e| key.redact(&e)),
            Ok(resp) => {
                let reason = key.redact(&format!("HTTP {}: {}", resp.status, short(&resp.body)));
                if !RETRY_STATUSES.contains(&resp.status) || attempt >= total {
                    return Err(if attempt > 0 { format!("{reason} (after {} attempts)", attempt + 1) } else { reason });
                }
                reason
            }
            Err(e) => {
                let reason = key.redact(&format!("no connection: {}", e.0));
                if attempt >= total {
                    return Err(reason);
                }
                reason
            }
        };
        let wait = retry.backoff[attempt.min(total - 1)];
        attempt += 1;
        on_retry(attempt, total, wait, &reason);
        sleep(wait);
    }
}
