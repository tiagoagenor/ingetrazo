// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The assistant's system prompt and the reply plumbing of `core/ai.py`
//! (`extract_code`, `strip_thoughts`, `truncated_code`, `compact_messages`).
//!
//! The Python prompt taught `box()`/`revolve()`/`house()` helpers bound in a
//! Python scope. This one teaches the Rhai API instead, and builds the API
//! section from the engine itself ([`ingetrazo_script::api::REFERENCE`] for
//! the meaning, the registered signatures for the exact names and arities),
//! so a function added to the API reaches the model with no edit here. The
//! prose stays Spanish, as the Python prompt and the conversation are.

use ingetrazo_script::console::{CompletionKind, Console};

use crate::provider::{Message, Role};

/// A recipe that works; the tests run it against the engine, so the prompt
/// never teaches code that fails.
pub const EXAMPLE: &str = r#"fn caja(x, y, z, w, d, h) {
    let p = [vec3(x, y, z), vec3(x + w, y, z), vec3(x + w, y + d, z), vec3(x, y + d, z)];
    let q = [vec3(x, y, z + h), vec3(x + w, y, z + h), vec3(x + w, y + d, z + h), vec3(x, y + d, z + h)];
    let caras = [add_face([p[3], p[2], p[1], p[0]]), add_face(q)];
    for i in 0..4 {
        let j = (i + 1) % 4;
        caras.push(add_face([p[i], p[j], q[j], q[i]]));
    }
    set_color(caras, 0.8, 0.4, 0.2);
    make_group(caras, "Caja")
}
let g = caja(0.0, 0.0, 0.0, 6.0, 4.0, 3.0);
print(`caras: ${g.faces().len()}`);"#;

const UNITS: &str = "IngeTrazo es un modelador 3D libre de empujar/tirar: Z arriba, unidades en METROS; los ángulos de rotate() van en GRADOS.";

const CONTRACT: &str = "Para ACTUAR sobre el modelo incluye EXACTAMENTE UN bloque ```rhai por respuesta. Antes de tocar el modelo se prueba en una copia; si falla no se ejecuta nada y recibirás el error con su línea y columna para que lo corrijas. Tras cada bloque recibirás su resultado (lo que imprimiste con print y si el modelo cambió) — revísalo e itera. Cuando el pedido esté terminado, responde SIN bloque de código con un resumen corto. SIN bloque no se ejecuta nada: nunca describas como hecho lo que no has ejecutado.";

const HOW_IT_RUNS: &str = "Cada bloque es UN paso de undo: si lanza un error se revierte POR COMPLETO, y el Ctrl+Z del usuario deshace tu paso entero. Las variables y funciones que definas siguen disponibles en los bloques siguientes — no las redefinas. No existen import, eval, archivos ni red. NO explores la API: lo que necesitas está en este documento; si algo falta, escríbelo directo y lee el error. Un pedido sencillo (una casa, una mesa, un poste) va COMPLETO de una sola vez, con detalles razonables aunque no te los pidan; los grandes, por pasos. Para saber cuánto has dibujado imprime face_count() (caras sueltas) o g.faces().len() (las de un grupo). La figura de persona de un documento nuevo se llama «Ingeniero» y es la ESCALA (1,70 m): no la borres ni la muevas salvo que te lo pidan.";

const RHAI_PRIMER: &str = r#"Rhai en pocas líneas: `let x = 1.5;` variables (los enteros y los decimales se mezclan en + - *, pero 3 / 2 es 1: escribe 3.0 / 2.0); arrays `[a, b]`, `a.push(x)`, `a.len()`, `a[i]`; mapas `#{color: [0.8, 0.1, 0.1]}`; `for i in 0..n { }`, `for p in lista { }`, `if c { } else { }`; funciones `fn f(a, b) { a + b }` (no ven las variables de fuera, sí las funciones de la API); texto con "..." o con plantillas `x = ${x}` entre acentos graves; closures `|p| p + vec3(0, 0, 1)` en `.map(...)` y `.filter(...)`. Un punto se escribe `vec3(x, y, z)` o `[x, y, z]`."#;

/// The system prompt for the API the engine in `console` registers.
pub fn system_prompt(console: &mut Console) -> String {
    let functions = function_list(console);
    [
        format!("Eres el asistente de modelado de IngeTrazo. {UNITS} Conversas en español, breve y claro."),
        CONTRACT.to_string(),
        RHAI_PRIMER.to_string(),
        format!("== API de scripts ==\n{}", ingetrazo_script::api::REFERENCE.trim()),
        format!("== Funciones registradas (firmas exactas) ==\n{functions}"),
        HOW_IT_RUNS.to_string(),
        format!("== Ejemplo: una caja de 6 × 4 × 3 m ==\n```rhai\n{EXAMPLE}\n```"),
        "El código de tus bloques viejos se resume como \"[receta ya ejecutada]\"; su efecto sigue en el modelo.".to_string(),
    ]
    .join("\n\n")
}

/// One line per function: the signature as a script sees it.
fn function_list(console: &mut Console) -> String {
    let mut lines = Vec::new();
    for c in console.completions("") {
        if matches!(c.kind, CompletionKind::Function | CompletionKind::Method) && c.name != "print" && c.name != "debug" {
            lines.extend(c.signature.lines().map(str::to_string));
        }
    }
    lines.sort();
    lines.dedup();
    lines.join("\n")
}

// ---- replies --------------------------------------------------------------

const FENCE: &str = "```";
/// Fence tags that open a Rhai block, best first. A bare fence is the last
/// resort: some models forget the language.
const TAGS: [&str; 5] = ["rhai", "rs", "rust", "js", ""];

/// A fenced block: where the fence opens, the code's range, where it ends.
struct Block {
    open: usize,
    code: (usize, usize),
    end: usize,
}

fn find_block(text: &str) -> Option<Block> {
    for tag in TAGS {
        let mut from = 0;
        while let Some(i) = text[from..].find(FENCE) {
            let open = from + i;
            let after = open + FENCE.len();
            let rest = &text[after..];
            let line_end = rest.find('\n').unwrap_or(rest.len());
            let info = rest[..line_end].trim();
            if info.eq_ignore_ascii_case(tag) {
                let code_start = (after + line_end + 1).min(text.len());
                let close = text[code_start..].find(FENCE)?;
                let code_end = code_start + close;
                return Some(Block { open, code: (code_start, code_end), end: code_end + FENCE.len() });
            }
            // Skip this fence's block, whatever it is.
            from = match text[after + line_end..].find(FENCE) {
                Some(j) => after + line_end + j + FENCE.len(),
                None => break,
            };
        }
    }
    None
}

/// The code of the first fenced Rhai block of a reply.
pub fn extract_code(text: &str) -> Option<String> {
    let b = find_block(text)?;
    let code = text[b.code.0..b.code.1].trim_matches('\n');
    (!code.trim().is_empty()).then(|| code.to_string())
}

/// A reply that opens a fence and never closes it: cut by the token limit.
/// Half a recipe was on its way; it must not count as "no code, done".
pub fn truncated_code(text: &str) -> bool {
    for tag in TAGS.iter().take(4) {
        let marker = format!("{FENCE}{tag}");
        let mut from = 0;
        while let Some(i) = text[from..].find(&marker) {
            let start = from + i + marker.len();
            // The tag must end its line (``` + rhai, not ```rhaiX).
            let line = text[start..].split('\n').next().unwrap_or("");
            if line.trim().is_empty() {
                return !text[start..].contains(FENCE);
            }
            from = start;
        }
    }
    false
}

/// What an executed recipe's code collapses to in old turns.
pub const CODE_STUB: &str = "[receta ya ejecutada — código omitido]";

/// The convo diet: the fenced block of every assistant turn but the last
/// `keep_last` is replaced by [`CODE_STUB`] (its effect is already in the
/// document); prose and the result turns stay. Returns copies.
pub fn compact_messages(messages: &[Message], keep_last: usize) -> Vec<Message> {
    let coded: Vec<usize> = messages.iter().enumerate().filter(|(_, m)| m.role == Role::Assistant && extract_code(&m.text).is_some()).map(|(i, _)| i).collect();
    let stub: Vec<usize> = coded[..coded.len().saturating_sub(keep_last)].to_vec();
    messages
        .iter()
        .enumerate()
        .map(|(i, m)| {
            if stub.contains(&i) {
                if let Some(b) = find_block(&m.text) {
                    let text = format!("{}{CODE_STUB}{}", &m.text[..b.open], &m.text[b.end..]);
                    return Message { role: m.role, text: text.trim().to_string() };
                }
            }
            m.clone()
        })
        .collect()
}

const THOUGHT_TAGS: [&str; 3] = ["thought", "thinking", "think"];

/// Remove the `<thought>`/`<thinking>`/`<think>` blocks some models leak
/// into their visible text: internal monologue that would be resent every
/// turn. Handles an unclosed opener (drop to the end) and a stray closer
/// whose opener fell in an earlier chunk (drop the prefix).
pub fn strip_thoughts(text: &str) -> String {
    let mut text = text.to_string();
    for tag in THOUGHT_TAGS {
        let (open, close) = (format!("<{tag}>"), format!("</{tag}>"));
        while let Some(s) = text.find(&open) {
            let rest = match text[s..].find(&close) {
                Some(e) => text[s + e + close.len()..].to_string(),
                None => String::new(),
            };
            text = format!("{}{rest}", &text[..s]);
        }
        if let Some(e) = text.find(&close) {
            if !text[..e].contains(&open) {
                text = text[e + close.len()..].to_string();
            }
        }
    }
    text.trim().to_string()
}

const BUILD_WORDS: [&str; 26] = [
    "dibuj", "crea", "haz", "hac", "modela", "constru", "añad", "agreg", "pon", "gener", "diseñ", "levant", "arma", "traza", "draw", "build", "make", "create", "add", "colo", "pinta", "mueve", "gira", "escala", "borra", "elimina",
];

/// Whether the user's message asks for something to be modelled (so a reply
/// without code is a model that did not do its job).
pub fn asks_to_build(prompt: &str) -> bool {
    let low = prompt.to_lowercase();
    BUILD_WORDS.iter().any(|w| low.contains(w))
}
