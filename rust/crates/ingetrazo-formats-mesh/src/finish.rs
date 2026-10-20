// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! A material's finish (`core/finish.py`): how its surface answers light,
//! chosen in the document or guessed from the material's and its picture's
//! names.
//!
//! The guess is Python's regular expressions, run here by a small matcher
//! over the same alternatives: `(?:^|[^a-záéíóúñç])(?:…)` with
//! `re.IGNORECASE` on the name with `-` made `_`. Case folding is the simple
//! lower-casing of each character; the exotic folds of `re` (the Kelvin sign
//! for `k`, the long s) are not applied.

/// `FINISHES`.
pub const FINISHES: [&str; 6] = ["matte", "satin", "gloss", "metal", "glass", "water"];

/// `PBR[finish]`: `(roughness, metallic, transmission, ior)`.
pub fn pbr(finish: &str) -> Option<(f64, f64, f64, f64)> {
    Some(match finish {
        "matte" => (0.9, 0.0, 0.0, 1.5),
        "satin" => (0.45, 0.0, 0.0, 1.5),
        "gloss" => (0.12, 0.0, 0.0, 1.5),
        "metal" => (0.3, 1.0, 0.0, 1.5),
        "glass" => (0.02, 0.0, 1.0, 1.45),
        "water" => (0.03, 0.0, 0.0, 1.33),
        _ => return None,
    })
}

/// The letters the pattern's negated class and look-aheads name.
fn is_letter(c: char) -> bool {
    matches!(c, 'a'..='z' | 'á' | 'é' | 'í' | 'ó' | 'ú' | 'ñ' | 'ç')
}

/// One element of an alternative.
enum El {
    /// A literal character (lower case).
    Ch(char),
    /// `[xyz]`, a set of lower-case characters.
    Set(Vec<char>),
    /// `[xyz]?`.
    OptSet(Vec<char>),
    /// `(?![a-záéíóúñç])`.
    NotLetter,
}

/// Parses the tiny regex dialect of the table below.
fn parse(alt: &str) -> Vec<El> {
    let chars: Vec<char> = alt.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '[' {
            let end = chars[i..].iter().position(|&c| c == ']').unwrap() + i;
            let set = chars[i + 1..end].to_vec();
            i = end + 1;
            if chars.get(i) == Some(&'?') {
                out.push(El::OptSet(set));
                i += 1;
            } else {
                out.push(El::Set(set));
            }
        } else if chars[i..].starts_with(&['(', '?', '!', 'L', ')']) {
            out.push(El::NotLetter);
            i += 5;
        } else {
            out.push(El::Ch(chars[i]));
            i += 1;
        }
    }
    out
}

/// The alternatives of core/finish.py `_WORDS`, `(?![a-záéíóúñç])` written
/// `(?!L)`, the groups `(?:lago|lake|mar|sea)(?!…)` expanded.
const WORDS: [(&str, &[&str]); 5] = [
    (
        "water",
        &[
            "agua", "water", "pool", "piscina", "pileta", "espejo[_ ]de[_ ]agua", "lago(?!L)", "lake(?!L)", "mar(?!L)", "sea(?!L)", "ocean",
            "oc[eé]ano", "[aá]gua",
        ],
    ),
    ("glass", &["vidri", "glass", "cristal", "crystal", "vitral", "window[_ ]?pane", "mampara", "vidro"]),
    (
        "metal",
        &[
            "metal", "acero", "steel", "a[cç]o(?!L)", "alumin", "hierro", "iron", "ferro", "chrom", "cromo", "inox", "bronce", "bronze", "cobre",
            "copper", "lat[oó]n", "brass", "galvaniz", "zinc", "zinco", "titani",
        ],
    ),
    (
        "gloss",
        &[
            "m[aá]rmol", "marble", "m[aá]rmore", "granit", "porcel", "cer[aá]mic", "ceramic", "azulej", "tile", "lacad", "lacquer", "laca", "pulid",
            "polish", "brillant", "glossy", "pl[aá]stic", "acr[ií]lic", "esmalt", "enamel", "lacquered", "gleam",
        ],
    ),
    ("satin", &["madera", "wood", "madeira", "parquet", "parquet", "laminad", "laminate", "cuero", "leather", "couro", "satin", "satinad", "semi"]),
];

fn matches_at(els: &[El], text: &[char], at: usize) -> bool {
    let Some((first, rest)) = els.split_first() else { return true };
    match first {
        El::Ch(c) => text.get(at) == Some(c) && matches_at(rest, text, at + 1),
        El::Set(set) => text.get(at).is_some_and(|c| set.contains(c)) && matches_at(rest, text, at + 1),
        El::OptSet(set) => {
            (text.get(at).is_some_and(|c| set.contains(c)) && matches_at(rest, text, at + 1)) || matches_at(rest, text, at)
        }
        El::NotLetter => !text.get(at).is_some_and(|&c| is_letter(c)) && matches_at(rest, text, at),
    }
}

/// Simple case folding of one character.
fn fold(c: char) -> char {
    let mut it = c.to_lowercase();
    match (it.next(), it.next()) {
        (Some(l), None) => l,
        _ => c,
    }
}

/// `rx.search(low)` for the finish's pattern.
fn search(alts: &[Vec<El>], low: &[char]) -> bool {
    for start in 0..=low.len() {
        // `(?:^|[^a-záéíóúñç])` then the alternative.
        let mut at = Vec::new();
        if start == 0 {
            at.push(0);
        }
        if start < low.len() && !is_letter(low[start]) {
            at.push(start + 1);
        }
        for a in at {
            if alts.iter().any(|els| matches_at(els, low, a)) {
                return true;
            }
        }
    }
    false
}

/// `guess(name, picture, opacity)`.
pub fn guess(name: Option<&str>, picture: Option<&str>, opacity: Option<f64>) -> &'static str {
    let table: Vec<(&'static str, Vec<Vec<El>>)> = WORDS.iter().map(|(f, alts)| (*f, alts.iter().map(|a| parse(a)).collect())).collect();
    for text in [name, picture].into_iter().flatten() {
        if text.is_empty() {
            continue;
        }
        let low: Vec<char> = text.replace('-', "_").chars().map(fold).collect();
        for (finish, alts) in &table {
            if search(alts, &low) {
                return finish;
            }
        }
    }
    if opacity.is_some_and(|o| o < 0.999) {
        return "glass";
    }
    "matte"
}

/// `resolve(chosen, name, picture, opacity)`: the user's choice, or the guess
/// when it is auto, absent or unknown.
pub fn resolve(chosen: Option<&str>, name: Option<&str>, picture: Option<&str>, opacity: Option<f64>) -> String {
    match chosen {
        Some(c) if FINISHES.contains(&c) => c.to_string(),
        _ => guess(name, picture, opacity).to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guesses_from_names() {
        assert_eq!(guess(Some("water_calm"), None, None), "water");
        assert_eq!(guess(Some("[Water Pool Light]"), None, None), "water");
        assert_eq!(guess(Some("Metal_10_1K"), None, None), "metal");
        assert_eq!(guess(Some("Marble_07"), None, None), "gloss");
        assert_eq!(guess(Some("mar"), None, None), "water");
        assert_eq!(guess(Some("marble"), None, None), "gloss");
        assert_eq!(guess(Some("Vidrio pulido"), None, None), "glass");
        assert_eq!(guess(Some("MADERA clara"), None, None), "satin");
        assert_eq!(guess(Some("xmadera"), None, None), "matte");
        assert_eq!(guess(Some("acero"), None, None), "metal");
        assert_eq!(guess(Some("aço"), None, None), "metal");
        assert_eq!(guess(Some("plain"), None, Some(0.5)), "glass");
        assert_eq!(guess(Some("plain"), Some("wood.png"), None), "satin");
        assert_eq!(guess(None, None, None), "matte");
    }
}
