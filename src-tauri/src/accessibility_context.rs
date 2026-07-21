#[path = "accessibility_context/limits.rs"]
mod limits;
#[path = "accessibility_context/platform/mod.rs"]
mod platform;
#[path = "accessibility_context/recorder.rs"]
mod recorder;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccessibilityContextEvent {
    pub elapsed_ms: u64,
    pub app_name: String,
    pub window_title: String,
    pub element: String,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ReadingSnapshot {
    pub app_name: String,
    pub window_title: String,
    pub element: String,
    pub text: String,
}

pub(crate) use recorder::{start_recording_context, stop_recording_context};

pub(crate) fn is_accessibility_context_supported() -> bool {
    platform::SUPPORTED
}

#[cfg(test)]
pub(crate) fn deduplicate_accessibility_events(
    events: &[AccessibilityContextEvent],
) -> Vec<AccessibilityContextEvent> {
    let mut deduplicated = Vec::with_capacity(events.len());
    for event in events {
        push_deduplicated_event(&mut deduplicated, event.clone());
    }
    deduplicated
}

pub(crate) fn deduplicate_owned_accessibility_events(
    events: Vec<AccessibilityContextEvent>,
) -> Vec<AccessibilityContextEvent> {
    let mut deduplicated = Vec::with_capacity(events.len());
    for event in events {
        push_deduplicated_event(&mut deduplicated, event);
    }
    deduplicated
}

pub(super) fn push_deduplicated_event(
    events: &mut Vec<AccessibilityContextEvent>,
    event: AccessibilityContextEvent,
) {
    while events
        .last()
        .is_some_and(|previous| event_expands_previous(previous, &event))
    {
        events.pop();
    }

    events.push(event);
}

fn event_expands_previous(
    previous: &AccessibilityContextEvent,
    next: &AccessibilityContextEvent,
) -> bool {
    if normalized_accessibility_text(&previous.app_name)
        != normalized_accessibility_text(&next.app_name)
        || normalized_accessibility_text(&previous.window_title)
            != normalized_accessibility_text(&next.window_title)
    {
        return false;
    }

    let previous_text = normalized_accessibility_text(&previous.text);
    let next_text = normalized_accessibility_text(&next.text);

    !previous_text.is_empty()
        && next_text.len() > previous_text.len()
        && contains_expanded_text(&next_text, &previous_text)
}

fn contains_expanded_text(next_text: &str, previous_text: &str) -> bool {
    if next_text.starts_with(previous_text) || next_text.ends_with(previous_text) {
        return true;
    }

    next_text.match_indices(previous_text).any(|(index, _)| {
        let before = next_text[..index].chars().next_back();
        let after = next_text[index + previous_text.len()..].chars().next();

        before.is_some_and(is_text_boundary) && after.is_some_and(is_text_boundary)
    })
}

fn is_text_boundary(character: char) -> bool {
    !character.is_alphanumeric()
}

fn normalized_accessibility_text(value: &str) -> String {
    value
        .replace('\u{00a0}', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::{
        deduplicate_accessibility_events, deduplicate_owned_accessibility_events,
        AccessibilityContextEvent,
    };

    fn event(
        elapsed_ms: u64,
        app_name: &str,
        window_title: &str,
        text: &str,
    ) -> AccessibilityContextEvent {
        AccessibilityContextEvent {
            elapsed_ms,
            app_name: app_name.to_string(),
            window_title: window_title.to_string(),
            element: "Selected text".to_string(),
            text: text.to_string(),
        }
    }

    #[test]
    fn deduplicate_accessibility_events_keeps_latest_expanded_selection() {
        let events = vec![
            event(100, "Notes", "Draft", "first"),
            event(200, "Notes", "Draft", "first second"),
            event(300, "Notes", "Draft", "first second third"),
        ];

        assert_eq!(
            deduplicate_accessibility_events(&events),
            vec![event(300, "Notes", "Draft", "first second third")]
        );
    }

    #[test]
    fn deduplicate_accessibility_events_handles_selection_expanding_backward() {
        let events = vec![
            event(100, "Notes", "Draft", "second"),
            event(200, "Notes", "Draft", "first second"),
        ];

        assert_eq!(
            deduplicate_accessibility_events(&events),
            vec![event(200, "Notes", "Draft", "first second")]
        );
    }

    #[test]
    fn deduplicate_accessibility_events_preserves_different_locations() {
        let events = vec![
            event(100, "Notes", "Draft", "first"),
            event(200, "Safari", "Draft", "first second"),
            event(300, "Notes", "Other Draft", "first second third"),
        ];

        assert_eq!(deduplicate_accessibility_events(&events), events);
    }

    #[test]
    fn deduplicate_accessibility_events_does_not_match_inside_words() {
        let events = vec![
            event(100, "Notes", "Draft", "art"),
            event(200, "Notes", "Draft", "partial"),
        ];

        assert_eq!(deduplicate_accessibility_events(&events), events);
    }

    #[test]
    fn owned_deduplication_moves_payload_strings_without_copying_them() {
        let events = vec![event(100, "Notes", "Draft", "a fairly large selected text")];
        let original_text_allocation = events[0].text.as_ptr();

        let deduplicated = deduplicate_owned_accessibility_events(events);

        assert_eq!(deduplicated.len(), 1);
        assert_eq!(deduplicated[0].text.as_ptr(), original_text_allocation);
    }
}
