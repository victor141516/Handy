use std::time::Duration;

pub(super) const POLL_INTERVAL: Duration = Duration::from_millis(250);
pub(super) const RECORDER_PROGRESS_LOG_INTERVAL: Duration = Duration::from_secs(60);

pub(super) const MAX_SELECTION_DEPTH: usize = 8;
pub(super) const MAX_FOCUS_DEPTH: usize = 8;
pub(super) const MAX_DESCENDANT_TEXT_DEPTH: usize = 2;
pub(super) const MAX_NODES: usize = 2_500;
pub(super) const MAX_DESCENDANT_NODES: usize = 80;

pub(super) const MAX_RECORDED_EVENTS: usize = 240;
pub(super) const MAX_RECORDED_CONTEXT_BYTES: usize = 512 * 1024;
pub(super) const MAX_CONTEXT_FIELD_CHARS: usize = 256;
pub(super) const MAX_CONTEXT_TEXT_CHARS: usize = 8_000;
pub(super) const MAX_CONTEXT_ARRAY_ITEMS: usize = 64;
