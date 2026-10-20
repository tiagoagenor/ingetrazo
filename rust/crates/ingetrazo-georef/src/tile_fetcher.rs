// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `georef/tile_fetcher.py`: the request planning of the cache-first tile
//! fetcher, as pure data. [`TilePlanner`] is `TileFetcher`'s bookkeeping (the
//! zoom guard, the in-flight set, the queue that drains eight at a time);
//! the network itself is the [`HttpGet`] trait the app implements, and the
//! decoded image / signals are the app's. Nothing here does I/O.

use std::collections::{HashSet, VecDeque};

use crate::tiles::TileSource;

/// `TileFetcher._MAX_INFLIGHT`.
pub const MAX_INFLIGHT: usize = 8;

/// The app version the Python `_USER_AGENT` embeds (`core.version`).
pub const APP_VERSION: &str = "0.5.8";

/// `_USER_AGENT`: `IngeTrazo/<version> (https://github.com/ingelibre/ingetrazo)`.
pub fn user_agent(version: &str) -> String {
    format!("IngeTrazo/{version} (https://github.com/ingelibre/ingetrazo)")
}

/// `(source_id, x, y, z)`.
pub type TileKey = (String, i64, i64, i64);

/// One GET to perform: the URL, the User-Agent header, and whether redirects
/// may be followed (`NoLessSafeRedirectPolicy`: never from https to http).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileRequest {
    pub key: TileKey,
    pub url: String,
    pub user_agent: String,
}

/// The HTTP client the app provides (Qt-net, `ureq`...). The reply bytes, or
/// the error string `QNetworkReply.errorString()` would give.
pub trait HttpGet {
    fn get(&self, url: &str, user_agent: &str) -> Result<Vec<u8>, String>;
}

/// What [`TilePlanner::request`] decided.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RequestOutcome {
    /// `z < 0` or `z > source.max_zoom`: refused, `None` in Python.
    Refused,
    /// The disk cache held a decodable tile: Python returns the image.
    CacheHit,
    /// Already in flight or queued: the reply will signal. `None` in Python.
    AlreadyActive,
    /// A download slot was free: perform this request now.
    Start(TileRequest),
    /// All slots busy: queued until [`TilePlanner::finished`] frees one.
    Queued,
}

/// `_on_finished`'s verdict on a reply.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TileOutcome {
    /// `tileFailed(reason)`.
    Failed(String),
    /// Cache the bytes, then `tileReady`.
    Ready,
}

/// `TileFetcher`'s state: in-flight keys and the pending queue.
#[derive(Clone, Debug)]
pub struct TilePlanner {
    pub user_agent: String,
    inflight: Vec<TileKey>,
    queue: VecDeque<(TileSource, i64, i64, i64)>,
    queued: HashSet<TileKey>,
}

impl TilePlanner {
    pub fn new(user_agent: String) -> TilePlanner {
        TilePlanner { user_agent, inflight: Vec::new(), queue: VecDeque::new(), queued: HashSet::new() }
    }

    pub fn inflight_len(&self) -> usize {
        self.inflight.len()
    }

    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    /// The in-flight keys, oldest first (Python's dict order).
    pub fn inflight_keys(&self) -> Vec<TileKey> {
        self.inflight.clone()
    }

    /// The queued downloads, next first.
    pub fn queue_keys(&self) -> Vec<TileKey> {
        self.queue.iter().map(|(s, x, y, z)| (s.id.clone(), *x, *y, *z)).collect()
    }

    pub fn is_inflight(&self, key: &TileKey) -> bool {
        self.inflight.contains(key)
    }

    pub fn is_queued(&self, key: &TileKey) -> bool {
        self.queued.contains(key)
    }

    fn start(&mut self, source: &TileSource, x: i64, y: i64, z: i64) -> TileRequest {
        let key = (source.id.clone(), x, y, z);
        self.inflight.push(key.clone());
        TileRequest { key, url: source.url(x, y, z), user_agent: self.user_agent.clone() }
    }

    /// `request`: `cache_decodes` is called (once, after the zoom guard) to
    /// look the tile up on disk and say whether the cached bytes decode to an
    /// image; a corrupt entry falls through to a download.
    pub fn request(
        &mut self,
        source: &TileSource,
        x: i64,
        y: i64,
        z: i64,
        cache_decodes: impl FnOnce() -> bool,
    ) -> RequestOutcome {
        if z < 0 || z > source.max_zoom {
            return RequestOutcome::Refused;
        }
        let key = (source.id.clone(), x, y, z);
        if cache_decodes() {
            return RequestOutcome::CacheHit;
        }
        if self.inflight.contains(&key) || self.queued.contains(&key) {
            return RequestOutcome::AlreadyActive;
        }
        if self.inflight.len() < MAX_INFLIGHT {
            RequestOutcome::Start(self.start(source, x, y, z))
        } else {
            self.queue.push_back((source.clone(), x, y, z));
            self.queued.insert(key);
            RequestOutcome::Queued
        }
    }

    /// `_pump`: start queued downloads up to the concurrency limit.
    pub fn pump(&mut self) -> Vec<TileRequest> {
        let mut out = Vec::new();
        while !self.queue.is_empty() && self.inflight.len() < MAX_INFLIGHT {
            let (source, x, y, z) = self.queue.pop_front().expect("non-empty");
            self.queued.remove(&(source.id.clone(), x, y, z));
            out.push(self.start(&source, x, y, z));
        }
        out
    }

    /// The bookkeeping half of `_on_finished`: the key leaves the in-flight
    /// set and the freed slot starts the next queued downloads.
    pub fn finished(&mut self, key: &TileKey) -> Vec<TileRequest> {
        if let Some(i) = self.inflight.iter().position(|k| k == key) {
            self.inflight.remove(i);
        }
        self.pump()
    }

    /// `cancel_all`: drop every in-flight download and the queue; returns the
    /// keys to abort.
    pub fn cancel_all(&mut self) -> Vec<TileKey> {
        let aborted = std::mem::take(&mut self.inflight);
        self.queue.clear();
        self.queued.clear();
        aborted
    }
}

/// The verdict half of `_on_finished`: a network error, an empty reply or
/// bytes that do not decode as an image fail the tile; otherwise it is
/// cached and ready.
pub fn classify_reply(error: Option<&str>, data: &[u8], decodes: bool) -> TileOutcome {
    if let Some(e) = error {
        return TileOutcome::Failed(e.to_string());
    }
    if data.is_empty() || !decodes {
        return TileOutcome::Failed("undecodable tile data".to_string());
    }
    TileOutcome::Ready
}
