use super::limits::{
    MAX_RECORDED_CONTEXT_BYTES, MAX_RECORDED_EVENTS, POLL_INTERVAL, RECORDER_PROGRESS_LOG_INTERVAL,
};
use super::platform::PlatformAccessibilityReader;
use super::{push_deduplicated_event, AccessibilityContextEvent};
use crate::memory_diagnostics::log_memory_snapshot;
use log::{info, warn};
use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

struct ActiveRecorder {
    stop: Arc<AtomicBool>,
    handle: JoinHandle<Vec<AccessibilityContextEvent>>,
}

impl ActiveRecorder {
    fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let handle = thread::spawn(move || record_loop(thread_stop));

        Self { stop, handle }
    }

    fn stop(self) -> Vec<AccessibilityContextEvent> {
        self.stop.store(true, Ordering::Release);
        match self.handle.join() {
            Ok(events) => events,
            Err(_) => {
                warn!("Accessibility context recording thread panicked");
                Vec::new()
            }
        }
    }
}

static ACTIVE_RECORDER: Lazy<Mutex<Option<ActiveRecorder>>> = Lazy::new(|| Mutex::new(None));

pub(crate) fn start_recording_context() {
    let previous = ACTIVE_RECORDER
        .lock()
        .ok()
        .and_then(|mut active| active.take());
    if let Some(recorder) = previous {
        let _ = recorder.stop();
    }

    if !super::is_accessibility_context_supported() {
        return;
    }

    if let Ok(mut active) = ACTIVE_RECORDER.lock() {
        *active = Some(ActiveRecorder::start());
    }
}

pub(crate) fn stop_recording_context() -> Vec<AccessibilityContextEvent> {
    let recorder = ACTIVE_RECORDER
        .lock()
        .ok()
        .and_then(|mut active| active.take());
    recorder.map(ActiveRecorder::stop).unwrap_or_default()
}

fn record_loop(stop: Arc<AtomicBool>) -> Vec<AccessibilityContextEvent> {
    let Some(mut reader) = PlatformAccessibilityReader::new() else {
        return Vec::new();
    };

    let started = Instant::now();
    let mut events = Vec::new();
    let mut last_snapshot = None;
    let mut poll_count = 0_u64;
    let mut changed_snapshot_count = 0_u64;
    let mut dropped_event_count = 0_u64;
    let mut slowest_poll = Duration::ZERO;
    let mut next_progress_log = RECORDER_PROGRESS_LOG_INTERVAL;

    log_memory_snapshot("accessibility recorder started");

    while !stop.load(Ordering::Acquire) {
        let poll_started = Instant::now();
        poll_count += 1;
        if let Some(snapshot) = reader.current_reading() {
            if last_snapshot.as_ref() != Some(&snapshot) {
                changed_snapshot_count += 1;
                dropped_event_count += push_recorded_event(
                    &mut events,
                    AccessibilityContextEvent {
                        elapsed_ms: duration_ms(started.elapsed()),
                        app_name: snapshot.app_name.clone(),
                        window_title: snapshot.window_title.clone(),
                        element: snapshot.element.clone(),
                        text: snapshot.text.clone(),
                    },
                ) as u64;
                last_snapshot = Some(snapshot);
            }
        } else {
            last_snapshot = None;
        }

        slowest_poll = slowest_poll.max(poll_started.elapsed());
        if started.elapsed() >= next_progress_log {
            info!(
                "Accessibility context recorder progress: elapsed={:.1}s, polls={}, changed_snapshots={}, retained_events={}, retained_bytes={}, dropped_events={}, slowest_poll_ms={}",
                started.elapsed().as_secs_f64(),
                poll_count,
                changed_snapshot_count,
                events.len(),
                recorded_events_bytes(&events),
                dropped_event_count,
                slowest_poll.as_millis(),
            );
            log_memory_snapshot("accessibility recorder progress");
            next_progress_log += RECORDER_PROGRESS_LOG_INTERVAL;
        }

        thread::sleep(POLL_INTERVAL);
    }

    info!(
        "Accessibility context recorder stopped: elapsed={:.1}s, polls={}, changed_snapshots={}, retained_events={}, retained_bytes={}, dropped_events={}, slowest_poll_ms={}",
        started.elapsed().as_secs_f64(),
        poll_count,
        changed_snapshot_count,
        events.len(),
        recorded_events_bytes(&events),
        dropped_event_count,
        slowest_poll.as_millis(),
    );
    log_memory_snapshot("accessibility recorder stopped");
    events
}

fn push_recorded_event(
    events: &mut Vec<AccessibilityContextEvent>,
    event: AccessibilityContextEvent,
) -> usize {
    let original_len = events.len();
    push_deduplicated_event(events, event);
    let mut dropped = original_len.saturating_add(1).saturating_sub(events.len());

    while events.len() > MAX_RECORDED_EVENTS
        || recorded_events_bytes(events) > MAX_RECORDED_CONTEXT_BYTES
    {
        events.remove(0);
        dropped += 1;
    }
    dropped
}

fn recorded_events_bytes(events: &[AccessibilityContextEvent]) -> usize {
    events.iter().map(accessibility_event_bytes).sum()
}

fn accessibility_event_bytes(event: &AccessibilityContextEvent) -> usize {
    event.app_name.len()
        + event.window_title.len()
        + event.element.len()
        + event.text.len()
        + std::mem::size_of::<AccessibilityContextEvent>()
}

fn duration_ms(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

#[cfg(test)]
mod tests {
    use super::{push_recorded_event, recorded_events_bytes, AccessibilityContextEvent};
    use crate::accessibility_context::limits::{MAX_RECORDED_CONTEXT_BYTES, MAX_RECORDED_EVENTS};

    #[test]
    fn recorded_events_are_bounded() {
        let mut events = Vec::new();
        for index in 0..(MAX_RECORDED_EVENTS + 5) {
            push_recorded_event(
                &mut events,
                AccessibilityContextEvent {
                    elapsed_ms: index as u64,
                    app_name: "Notes".to_string(),
                    window_title: "Draft".to_string(),
                    element: "Selected text".to_string(),
                    text: format!("selected text {index}"),
                },
            );
        }

        assert_eq!(events.len(), MAX_RECORDED_EVENTS);
        assert_eq!(events[0].elapsed_ms, 5);
    }

    #[test]
    fn recorded_events_are_bounded_by_retained_bytes() {
        let mut events = Vec::new();
        for index in 0..100 {
            push_recorded_event(
                &mut events,
                AccessibilityContextEvent {
                    elapsed_ms: index,
                    app_name: "Notes".to_string(),
                    window_title: format!("Draft {index}"),
                    element: "Selected text".to_string(),
                    text: "x".repeat(8_000),
                },
            );
        }

        assert!(recorded_events_bytes(&events) <= MAX_RECORDED_CONTEXT_BYTES);
        assert!(events.len() < 100);
    }
}
