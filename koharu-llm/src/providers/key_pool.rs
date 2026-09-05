//! Rotating pool of API keys.
//!
//! Free Gemini keys cap out at a few hundred requests a day — roughly two
//! volumes — after which every request fails with RESOURCE_EXHAUSTED. Holding
//! several keys and switching on exhaustion turns that from a manual
//! copy-key-into-Settings-and-reload chore into something the provider handles
//! mid-request, so the page that triggered the limit still gets translated.

use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// How long a key whose *daily* quota is gone is skipped. Free quotas reset once
/// a day; this only needs to be long enough to stop the pool from spinning on
/// dead keys during one session.
pub const DAILY_COOLDOWN: Duration = Duration::from_secs(6 * 60 * 60);

/// How long a key that merely tripped a per-minute rate limit is skipped. It
/// still has daily quota, so it must come back quickly.
pub const RATE_LIMIT_COOLDOWN: Duration = Duration::from_secs(90);

/// Anything shorter than this is not a key — a stray word, a bare `=`, a
/// truncated paste.
const MIN_KEY_LEN: usize = 20;

/// Environment variable holding keys for a provider, e.g.
/// `KOHARU_GEMINI_API_KEYS=key1,key2`.
fn keys_env_var(provider: &str) -> String {
    format!("KOHARU_{}_API_KEYS", provider.to_ascii_uppercase().replace('-', "_"))
}

/// Environment variable pointing at a key file.
fn key_file_env_var(provider: &str) -> String {
    format!("KOHARU_{}_KEY_FILE", provider.to_ascii_uppercase().replace('-', "_"))
}

/// Places a key file may live, in priority order.
///
/// The key file lives in the project directory: one obvious location, versioned
/// alongside the code (and gitignored), with nothing in a hidden app-data folder
/// to shadow it. An installed build that is not run from a checkout has no
/// project directory, so it must supply keys through `KOHARU_<PROVIDER>_API_KEYS`
/// or `KOHARU_<PROVIDER>_KEY_FILE`.
pub fn key_file_candidates(provider: &str) -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    // Explicit override always wins.
    if let Some(path) = std::env::var_os(key_file_env_var(provider)) {
        candidates.push(PathBuf::from(path));
    }

    if let Some(root) = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent() {
        candidates.push(root.join(format!("{provider}_keys.txt")));
        candidates.push(root.join(".env"));
    }

    candidates
}

/// Pull keys out of free-form text.
///
/// Accepts a bare key per line, several keys separated by commas or whitespace,
/// and `NAME=key1,key2` so that a `.env` file works unchanged. `#` starts a
/// comment.
pub fn parse_keys(text: &str) -> Vec<String> {
    let mut keys = Vec::new();

    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        // `NAME=value` — keep only the value side.
        let payload = match line.split_once('=') {
            Some((name, value)) if !name.contains(char::is_whitespace) => value,
            _ => line,
        };

        for token in payload.split([',', ';', ' ', '\t', '"', '\'']) {
            let token = token.trim();
            if token.len() >= MIN_KEY_LEN {
                keys.push(token.to_string());
            }
        }
    }

    keys
}

/// Collect keys for `provider` from the environment and the first key file
/// found, in that order. `preferred` (typically the key saved in Settings) is
/// placed first so an explicit choice still wins.
pub fn collect_keys(provider: &str, preferred: Option<&str>) -> Vec<String> {
    let mut keys: Vec<String> = Vec::new();

    if let Some(value) = preferred {
        // Settings accepts several keys pasted together, too.
        keys.extend(parse_keys(value));
        // A single key shorter than MIN_KEY_LEN would be dropped by the parser;
        // trust an explicit value even so.
        let trimmed = value.trim();
        if keys.is_empty() && !trimmed.is_empty() {
            keys.push(trimmed.to_string());
        }
    }

    if let Ok(value) = std::env::var(keys_env_var(provider)) {
        keys.extend(parse_keys(&value));
    }

    for path in key_file_candidates(provider) {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let found = parse_keys(&text);
        if !found.is_empty() {
            tracing::info!(
                path = %path.display(),
                count = found.len(),
                provider,
                "loaded API keys from file"
            );
            keys.extend(found);
        }
        break;
    }

    // Dedup while keeping the order the keys were written in — the "start from
    // key N" setting indexes this list, so sorting it would point the user at a
    // different key than the one they counted in their file.
    let mut seen = std::collections::HashSet::new();
    keys.retain(|key| seen.insert(key.clone()));
    keys
}

struct Slot {
    key: String,
    /// When the key was retired, and for how long.
    resting_until: Option<Instant>,
}

pub struct ApiKeyPool {
    provider: &'static str,
    slots: Mutex<Vec<Slot>>,
    current: Mutex<usize>,
}

/// Live view of a pool, for display while a batch is running.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyPoolStatus {
    /// Keys configured in total.
    pub total: u32,
    /// Keys not currently resting.
    pub available: u32,
    /// 1-based position of the key in use, matching what the user sees in
    /// Settings.
    pub current: u32,
}

impl ApiKeyPool {
    pub fn new(provider: &'static str, keys: Vec<String>) -> Self {
        Self::starting_at(provider, keys, 1)
    }

    /// Build a pool that begins at `start_index` (1-based, clamped into range).
    ///
    /// Keys are per-day limited, so a run started later in the day often needs
    /// to skip the keys already spent; this avoids one wasted 429 per key.
    pub fn starting_at(provider: &'static str, keys: Vec<String>, start_index: u32) -> Self {
        let total = keys.len();
        let start = if total == 0 {
            0
        } else {
            (start_index.max(1) as usize - 1).min(total - 1)
        };
        if start > 0 {
            tracing::info!(provider, start_index = start + 1, total, "key pool starts mid-list");
        }
        Self {
            provider,
            slots: Mutex::new(
                keys.into_iter()
                    .map(|key| Slot { key, resting_until: None })
                    .collect(),
            ),
            current: Mutex::new(start),
        }
    }

    /// Current pool state, or `None` when the pool holds no keys.
    pub fn status(&self) -> Option<KeyPoolStatus> {
        let slots = self.slots.lock().ok()?;
        if slots.is_empty() {
            return None;
        }
        let current = *self.current.lock().ok()?;
        Some(KeyPoolStatus {
            total: slots.len() as u32,
            available: slots.iter().filter(|s| Self::usable(s)).count() as u32,
            current: current as u32 + 1,
        })
    }

    pub fn len(&self) -> usize {
        self.slots.lock().map(|s| s.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The key to use right now, or `None` when every key is still resting.
    ///
    /// Falls forward to any recovered key rather than only inspecting the
    /// current slot: when the pool was fully spent, `current` sits on the key
    /// that died *last*, which is also the last to come back, so checking only
    /// that slot would keep reporting exhaustion while earlier keys are alive
    /// again.
    pub fn active(&self) -> Option<String> {
        let slots = self.slots.lock().ok()?;
        let mut current = self.current.lock().ok()?;
        let total = slots.len();
        if total == 0 {
            return None;
        }

        for step in 0..total {
            let candidate = (*current + step) % total;
            if Self::usable(&slots[candidate]) {
                *current = candidate;
                return Some(slots[candidate].key.clone());
            }
        }
        None
    }

    fn usable(slot: &Slot) -> bool {
        match slot.resting_until {
            None => true,
            Some(until) => Instant::now() >= until,
        }
    }

    /// Rest the active key for `cooldown` and move to the next usable one.
    ///
    /// Use [`RATE_LIMIT_COOLDOWN`] for a per-minute limit and [`DAILY_COOLDOWN`]
    /// when the daily quota is gone — resting a merely rate-limited key for
    /// hours would throw away most of the pool during a burst.
    ///
    /// Returns the next key, or `None` when every key is resting.
    pub fn rotate(&self, cooldown: Duration) -> Option<String> {
        let mut slots = self.slots.lock().ok()?;
        let mut current = self.current.lock().ok()?;
        let total = slots.len();
        if total == 0 {
            return None;
        }

        if let Some(slot) = slots.get_mut(*current) {
            let until = Instant::now() + cooldown;
            // Keep the longer rest if this key was already retired for the day.
            slot.resting_until = Some(match slot.resting_until {
                Some(existing) if existing > until => existing,
                _ => until,
            });
        }

        for step in 1..=total {
            let candidate = (*current + step) % total;
            if Self::usable(&slots[candidate]) {
                *current = candidate;
                let remaining = slots.iter().filter(|s| Self::usable(s)).count();
                tracing::warn!(
                    provider = self.provider,
                    key_index = candidate + 1,
                    total,
                    remaining,
                    cooldown_secs = cooldown.as_secs(),
                    "API key hit its limit, rotated to the next key"
                );
                return Some(slots[candidate].key.clone());
            }
        }

        tracing::error!(
            provider = self.provider,
            total,
            "every API key in the pool is resting"
        );
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidates_point_at_the_project_directory() {
        let candidates = key_file_candidates("gemini");
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        assert!(
            candidates.contains(&repo.join("gemini_keys.txt")),
            "project key file must be a candidate: {candidates:?}"
        );
        assert!(candidates.contains(&repo.join(".env")));
        // Nothing outside the project, so a stray file in an app-data folder
        // cannot silently shadow the one in the repo.
        assert!(
            candidates.iter().all(|p| p.starts_with(&repo)),
            "unexpected candidate outside the project: {candidates:?}"
        );
    }

    #[test]
    fn parses_bare_keys_env_lines_and_comments() {
        let text = "\
# gemini keys
AIzaSyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA1
AIzaSyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA2  # second
GEMINI_API_KEYS=AIzaSyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA3,AIzaSyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4
short
";
        let keys = parse_keys(text);
        assert_eq!(keys.len(), 4, "{keys:?}");
        assert!(keys.iter().all(|k| k.starts_with("AIzaSy")));
        assert!(!keys.iter().any(|k| k == "short"));
    }

    #[test]
    fn parses_quoted_env_value() {
        let keys = parse_keys("KOHARU_GEMINI_API_KEYS=\"AIzaSyBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB1\"");
        assert_eq!(keys, vec!["AIzaSyBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB1".to_string()]);
    }

    #[test]
    fn rotate_walks_the_pool_then_gives_up() {
        let pool = ApiKeyPool::new("gemini", vec!["a".into(), "b".into(), "c".into()]);
        assert_eq!(pool.active().as_deref(), Some("a"));
        assert_eq!(pool.rotate(DAILY_COOLDOWN).as_deref(), Some("b"));
        assert_eq!(pool.active().as_deref(), Some("b"));
        assert_eq!(pool.rotate(DAILY_COOLDOWN).as_deref(), Some("c"));
        // Whole pool now spent.
        assert_eq!(pool.rotate(DAILY_COOLDOWN), None);
        assert_eq!(pool.active(), None);
    }

    #[test]
    fn rotate_on_a_single_key_pool_reports_exhaustion() {
        let pool = ApiKeyPool::new("gemini", vec!["only".into()]);
        assert_eq!(pool.active().as_deref(), Some("only"));
        assert_eq!(pool.rotate(DAILY_COOLDOWN), None);
    }

    #[test]
    fn empty_pool_is_reported_as_empty() {
        let pool = ApiKeyPool::new("gemini", Vec::new());
        assert!(pool.is_empty());
        assert_eq!(pool.active(), None);
        assert_eq!(pool.rotate(DAILY_COOLDOWN), None);
    }

    #[test]
    fn a_short_cooldown_lets_the_key_come_back() {
        let pool = ApiKeyPool::new("gemini", vec!["a".into(), "b".into()]);
        // Both keys trip a per-minute limit with a cooldown that has already
        // elapsed by the time we look again.
        assert_eq!(pool.rotate(Duration::ZERO).as_deref(), Some("b"));
        assert_eq!(pool.rotate(Duration::ZERO).as_deref(), Some("a"));
        // Neither key was retired for the day, so the pool is still usable.
        assert!(pool.active().is_some());
    }

    #[test]
    fn active_falls_forward_to_a_recovered_key() {
        let pool = ApiKeyPool::new("gemini", vec!["a".into(), "b".into()]);
        // "a" rests briefly; the pool moves to "b".
        assert_eq!(pool.rotate(Duration::from_millis(50)).as_deref(), Some("b"));
        // "b" is then out for the day while "a" is still resting, so nothing is
        // usable and `current` stays parked on "b".
        assert_eq!(pool.rotate(DAILY_COOLDOWN), None);

        std::thread::sleep(Duration::from_millis(80));

        // "a" has recovered. It must be found even though `current` points at
        // "b", which is out for hours.
        assert_eq!(pool.active().as_deref(), Some("a"));
    }

    #[test]
    fn a_daily_rest_is_not_shortened_by_a_later_rate_limit() {
        // Single key, so rotation cannot move away and the rest is observable.
        let pool = ApiKeyPool::new("gemini", vec!["only".into()]);
        assert_eq!(pool.rotate(DAILY_COOLDOWN), None);
        // A later rate-limit rotation must not overwrite the longer rest.
        assert_eq!(pool.rotate(Duration::ZERO), None);
        assert_eq!(pool.active(), None, "daily rest was shortened");
    }

    #[test]
    fn collect_keys_keeps_written_order_and_dedups() {
        let first = "AIzaSyCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC9";
        let second = "AIzaSyDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD8";
        // `second` sorts before `first`; order must still follow the input, and
        // the repeat must collapse.
        let keys = collect_keys(
            "gemini-test-provider",
            Some(&format!("{first},{second},{first}")),
        );
        assert_eq!(keys, vec![first.to_string(), second.to_string()]);
    }

    #[test]
    fn starting_at_skips_spent_keys_and_clamps() {
        let keys = vec!["a".into(), "b".into(), "c".into()];
        let pool = ApiKeyPool::starting_at("gemini", keys.clone(), 3);
        assert_eq!(pool.active().as_deref(), Some("c"));
        assert_eq!(pool.status().unwrap().current, 3);

        // Out of range in either direction lands inside the list.
        assert_eq!(
            ApiKeyPool::starting_at("gemini", keys.clone(), 99).active().as_deref(),
            Some("c")
        );
        assert_eq!(
            ApiKeyPool::starting_at("gemini", keys, 0).active().as_deref(),
            Some("a")
        );
    }

    #[test]
    fn status_tracks_availability_as_keys_rest() {
        let pool = ApiKeyPool::new("gemini", vec!["a".into(), "b".into(), "c".into()]);
        assert_eq!(
            pool.status(),
            Some(KeyPoolStatus { total: 3, available: 3, current: 1 })
        );
        pool.rotate(DAILY_COOLDOWN);
        assert_eq!(
            pool.status(),
            Some(KeyPoolStatus { total: 3, available: 2, current: 2 })
        );
        assert_eq!(ApiKeyPool::new("gemini", Vec::new()).status(), None);
    }
}
