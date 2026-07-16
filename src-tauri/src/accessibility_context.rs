#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccessibilityContextEvent {
    pub elapsed_ms: u64,
    pub app_name: String,
    pub window_title: String,
    pub element: String,
    pub text: String,
}

pub(crate) fn deduplicate_accessibility_events(
    events: &[AccessibilityContextEvent],
) -> Vec<AccessibilityContextEvent> {
    let mut deduplicated = Vec::with_capacity(events.len());
    for event in events {
        push_deduplicated_event(&mut deduplicated, event.clone());
    }
    deduplicated
}

fn push_deduplicated_event(
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

#[cfg(target_os = "macos")]
mod platform {
    use super::{push_deduplicated_event, AccessibilityContextEvent};
    use core_foundation::array::CFArray;
    use core_foundation::attributed_string::CFAttributedString;
    use core_foundation::base::{CFType, CFTypeRef, TCFType};
    use core_foundation::boolean::CFBoolean;
    use core_foundation::dictionary::CFDictionary;
    use core_foundation::number::CFNumber;
    use core_foundation::string::{CFString, CFStringRef};
    use core_foundation_sys::array::{CFArrayGetTypeID, CFArrayRef};
    use core_foundation_sys::attributed_string::CFAttributedStringGetString;
    use core_foundation_sys::base::{Boolean, CFGetTypeID, CFHash, CFRelease, CFTypeID};
    use core_foundation_sys::dictionary::{
        CFDictionaryGetTypeID, CFDictionaryGetValueIfPresent, CFDictionaryRef,
    };
    use core_foundation_sys::number::{CFBooleanGetTypeID, CFBooleanGetValue};
    use core_graphics::window::{
        copy_window_info, kCGNullWindowID, kCGWindowAlpha, kCGWindowLayer,
        kCGWindowListExcludeDesktopElements, kCGWindowListOptionOnScreenOnly, kCGWindowName,
        kCGWindowOwnerName, kCGWindowOwnerPID,
    };
    use log::{debug, warn};
    use once_cell::sync::Lazy;
    use std::collections::HashSet;
    use std::ffi::c_void;
    use std::ptr;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread::{self, JoinHandle};
    use std::time::{Duration, Instant};

    const AX_SUCCESS: i32 = 0;
    const POLL_INTERVAL: Duration = Duration::from_millis(250);
    const MAX_SELECTION_DEPTH: usize = 8;
    const MAX_FOCUS_DEPTH: usize = 8;
    const MAX_DESCENDANT_TEXT_DEPTH: usize = 2;
    const MAX_NODES: usize = 2_500;
    const MAX_DESCENDANT_NODES: usize = 80;

    #[repr(C)]
    struct __AXUIElement(c_void);

    type AXUIElementRef = *const __AXUIElement;

    #[link(name = "ApplicationServices", kind = "framework")]
    unsafe extern "C" {
        static kAXTrustedCheckOptionPrompt: CFStringRef;

        fn AXIsProcessTrustedWithOptions(options: CFDictionaryRef) -> Boolean;
        fn AXUIElementCreateApplication(pid: i32) -> AXUIElementRef;
        fn AXUIElementCreateSystemWide() -> AXUIElementRef;
        fn AXUIElementGetTypeID() -> CFTypeID;
        fn AXUIElementCopyAttributeValue(
            element: AXUIElementRef,
            attribute: CFStringRef,
            value: *mut CFTypeRef,
        ) -> i32;
        fn AXUIElementCopyParameterizedAttributeValue(
            element: AXUIElementRef,
            parameterized_attribute: CFStringRef,
            parameter: CFTypeRef,
            value: *mut CFTypeRef,
        ) -> i32;
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct WindowContext {
        pid: i32,
        app_name: String,
        window_title: String,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct ReadingSnapshot {
        app_name: String,
        window_title: String,
        element: String,
        text: String,
    }

    struct ActiveRecorder {
        stop: Arc<AtomicBool>,
        handle: JoinHandle<Vec<AccessibilityContextEvent>>,
    }

    impl ActiveRecorder {
        fn start() -> Option<Self> {
            if !accessibility_trusted(true) {
                warn!(
                    "Accessibility context recording skipped: accessibility permission is missing"
                );
                return None;
            }

            let stop = Arc::new(AtomicBool::new(false));
            let thread_stop = Arc::clone(&stop);
            let handle = thread::spawn(move || record_loop(thread_stop));

            Some(Self { stop, handle })
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

    pub fn start_recording_context() {
        let previous = ACTIVE_RECORDER
            .lock()
            .ok()
            .and_then(|mut active| active.take());
        if let Some(recorder) = previous {
            let _ = recorder.stop();
        }

        let recorder = ActiveRecorder::start();
        if let Ok(mut active) = ACTIVE_RECORDER.lock() {
            *active = recorder;
        }
    }

    pub fn stop_recording_context() -> Vec<AccessibilityContextEvent> {
        let recorder = ACTIVE_RECORDER
            .lock()
            .ok()
            .and_then(|mut active| active.take());
        recorder.map(ActiveRecorder::stop).unwrap_or_default()
    }

    fn record_loop(stop: Arc<AtomicBool>) -> Vec<AccessibilityContextEvent> {
        let started = Instant::now();
        let mut events = Vec::new();
        let mut last_snapshot: Option<ReadingSnapshot> = None;

        while !stop.load(Ordering::Acquire) {
            if let Some(snapshot) = current_reading() {
                if last_snapshot.as_ref() != Some(&snapshot) {
                    push_deduplicated_event(
                        &mut events,
                        AccessibilityContextEvent {
                            elapsed_ms: duration_ms(started.elapsed()),
                            app_name: snapshot.app_name.clone(),
                            window_title: snapshot.window_title.clone(),
                            element: snapshot.element.clone(),
                            text: snapshot.text.clone(),
                        },
                    );
                    last_snapshot = Some(snapshot);
                }
            } else {
                last_snapshot = None;
            }

            thread::sleep(POLL_INTERVAL);
        }

        debug!(
            "Accessibility context recorder captured {} event(s)",
            events.len()
        );
        events
    }

    fn duration_ms(duration: Duration) -> u64 {
        duration.as_millis().min(u128::from(u64::MAX)) as u64
    }

    fn accessibility_trusted(prompt: bool) -> bool {
        unsafe {
            let key = CFString::wrap_under_get_rule(kAXTrustedCheckOptionPrompt);
            let value = if prompt {
                CFBoolean::true_value()
            } else {
                CFBoolean::false_value()
            };
            let options = CFDictionary::from_CFType_pairs(&[(key, value)]);
            AXIsProcessTrustedWithOptions(options.as_concrete_TypeRef()) != 0
        }
    }

    fn current_reading() -> Option<ReadingSnapshot> {
        let window_context = active_window_context()?;
        let focused_element = focused_element_for(window_context.pid);
        let focused_window = focused_window_for(window_context.pid, focused_element.as_ref());
        let window_title = focused_window
            .as_ref()
            .and_then(window_title)
            .or_else(|| non_empty(window_context.window_title.clone()))
            .unwrap_or_else(|| "(unknown window)".to_string());

        if let Some(focused) = focused_element.as_ref() {
            if let Some(text) = selected_text(focused) {
                return Some(reading(
                    &window_context,
                    &window_title,
                    "Selected text",
                    text,
                ));
            }

            let mut visited = HashSet::new();
            if let Some((_, text)) = find_selected_text(
                focused,
                0,
                MAX_SELECTION_DEPTH.min(6),
                MAX_NODES,
                &mut visited,
            ) {
                return Some(reading(
                    &window_context,
                    &window_title,
                    "Selected text",
                    text,
                ));
            }
        }

        if let Some(window) = focused_window.as_ref() {
            let mut visited = HashSet::new();
            if let Some((_, text)) =
                find_selected_text(window, 0, MAX_SELECTION_DEPTH, MAX_NODES, &mut visited)
            {
                return Some(reading(
                    &window_context,
                    &window_title,
                    "Selected text",
                    text,
                ));
            }
        }

        if focused_element
            .as_ref()
            .is_some_and(is_selection_only_text_surface)
            || is_selection_only_application(&window_context.app_name)
        {
            return None;
        }

        if let Some(focused) = focused_element.as_ref() {
            if let Some(text) = focused_element_text(focused) {
                return Some(reading(
                    &window_context,
                    &window_title,
                    &element_name(focused),
                    text,
                ));
            }
        }

        if let Some(window) = focused_window.as_ref() {
            let mut visited = HashSet::new();
            if let Some((element, text)) =
                find_focused_reading(window, 0, MAX_FOCUS_DEPTH, MAX_NODES, &mut visited)
            {
                return Some(reading(
                    &window_context,
                    &window_title,
                    &element_name(&element),
                    text,
                ));
            }
        }

        None
    }

    fn reading(
        context: &WindowContext,
        window_title: &str,
        element: &str,
        text: String,
    ) -> ReadingSnapshot {
        ReadingSnapshot {
            app_name: context.app_name.clone(),
            window_title: window_title.to_string(),
            element: element.to_string(),
            text,
        }
    }

    fn active_window_context() -> Option<WindowContext> {
        let options = kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements;
        let windows = copy_window_info(options, kCGNullWindowID)?;
        let current_pid = std::process::id() as i32;

        for item in windows.get_all_values() {
            if item.is_null()
                || unsafe { CFGetTypeID(item as CFTypeRef) } != unsafe { CFDictionaryGetTypeID() }
            {
                continue;
            }

            let dict = item as CFDictionaryRef;
            let layer = unsafe { dict_i32(dict, kCGWindowLayer) }.unwrap_or(0);
            let alpha = unsafe { dict_f64(dict, kCGWindowAlpha) }.unwrap_or(1.0);
            let pid = unsafe { dict_i32(dict, kCGWindowOwnerPID) }?;

            if layer != 0 || alpha <= 0.0 || pid == current_pid {
                continue;
            }

            let app_name = unsafe { dict_string(dict, kCGWindowOwnerName) }
                .filter(|name| !name.trim().is_empty())
                .unwrap_or_else(|| format!("pid {}", pid));
            let window_title = unsafe { dict_string(dict, kCGWindowName) }.unwrap_or_default();

            return Some(WindowContext {
                pid,
                app_name,
                window_title,
            });
        }

        None
    }

    unsafe fn dict_value(dict: CFDictionaryRef, key: CFStringRef) -> Option<CFTypeRef> {
        let mut value: *const c_void = ptr::null();
        if CFDictionaryGetValueIfPresent(dict, key as *const c_void, &mut value) != 0
            && !value.is_null()
        {
            Some(value as CFTypeRef)
        } else {
            None
        }
    }

    unsafe fn dict_string(dict: CFDictionaryRef, key: CFStringRef) -> Option<String> {
        dict_value(dict, key).and_then(|value| text_from_type_ref(value))
    }

    unsafe fn dict_i32(dict: CFDictionaryRef, key: CFStringRef) -> Option<i32> {
        let value = dict_value(dict, key)?;
        if CFGetTypeID(value) != CFNumber::type_id() {
            return None;
        }
        let number = CFNumber::wrap_under_get_rule(value as _);
        number.to_i32()
    }

    unsafe fn dict_f64(dict: CFDictionaryRef, key: CFStringRef) -> Option<f64> {
        let value = dict_value(dict, key)?;
        if CFGetTypeID(value) != CFNumber::type_id() {
            return None;
        }
        let number = CFNumber::wrap_under_get_rule(value as _);
        number.to_f64()
    }

    fn focused_element_for(pid: i32) -> Option<CFType> {
        let app = create_application_element(pid)?;
        if let Some(focused) = copy_attribute(&app, "AXFocusedUIElement").filter(is_ax_element) {
            return Some(focused);
        }

        let system_wide = create_system_wide_element()?;
        copy_attribute(&system_wide, "AXFocusedUIElement").filter(is_ax_element)
    }

    fn focused_window_for(pid: i32, focused_element: Option<&CFType>) -> Option<CFType> {
        if let Some(focused) = focused_element {
            if let Some(window) = copy_attribute(focused, "AXWindow").filter(is_ax_element) {
                return Some(window);
            }
        }

        let app = create_application_element(pid)?;
        copy_attribute(&app, "AXFocusedWindow").filter(is_ax_element)
    }

    fn create_application_element(pid: i32) -> Option<CFType> {
        let element = unsafe { AXUIElementCreateApplication(pid) };
        if element.is_null() {
            None
        } else {
            Some(unsafe { CFType::wrap_under_create_rule(element as CFTypeRef) })
        }
    }

    fn create_system_wide_element() -> Option<CFType> {
        let element = unsafe { AXUIElementCreateSystemWide() };
        if element.is_null() {
            None
        } else {
            Some(unsafe { CFType::wrap_under_create_rule(element as CFTypeRef) })
        }
    }

    fn is_ax_element(value: &CFType) -> bool {
        unsafe { CFGetTypeID(value.as_CFTypeRef()) == AXUIElementGetTypeID() }
    }

    fn ax_ptr(element: &CFType) -> AXUIElementRef {
        element.as_CFTypeRef() as AXUIElementRef
    }

    fn copy_attribute(element: &CFType, attribute: &'static str) -> Option<CFType> {
        let attribute = CFString::from_static_string(attribute);
        let mut value: CFTypeRef = ptr::null();
        let error = unsafe {
            AXUIElementCopyAttributeValue(
                ax_ptr(element),
                attribute.as_concrete_TypeRef(),
                &mut value,
            )
        };

        if error == AX_SUCCESS && !value.is_null() {
            Some(unsafe { CFType::wrap_under_create_rule(value) })
        } else {
            None
        }
    }

    fn copy_parameterized_attribute(
        element: &CFType,
        attribute: &'static str,
        parameter: &CFType,
    ) -> Option<CFType> {
        let attribute = CFString::from_static_string(attribute);
        let mut value: CFTypeRef = ptr::null();
        let error = unsafe {
            AXUIElementCopyParameterizedAttributeValue(
                ax_ptr(element),
                attribute.as_concrete_TypeRef(),
                parameter.as_CFTypeRef(),
                &mut value,
            )
        };

        if error == AX_SUCCESS && !value.is_null() {
            Some(unsafe { CFType::wrap_under_create_rule(value) })
        } else {
            None
        }
    }

    fn selected_text(element: &CFType) -> Option<String> {
        copy_attribute(element, "AXSelectedText")
            .as_ref()
            .and_then(accessibility_text)
            .and_then(non_blank)
            .or_else(|| selected_text_from_text_marker_range(element))
            .or_else(|| selected_text_from_text_range(element))
    }

    fn selected_text_from_text_marker_range(element: &CFType) -> Option<String> {
        let marker_range = copy_attribute(element, "AXSelectedTextMarkerRange")?;
        copy_parameterized_attribute(element, "AXStringForTextMarkerRange", &marker_range)
            .as_ref()
            .and_then(accessibility_text)
            .and_then(non_blank)
    }

    fn selected_text_from_text_range(element: &CFType) -> Option<String> {
        if let Some(range) = copy_attribute(element, "AXSelectedTextRange") {
            if let Some(text) = string_for_text_range(element, &range) {
                return Some(text);
            }
        }

        let ranges = copy_attribute(element, "AXSelectedTextRanges")?;
        let pieces = array_values(&ranges)
            .into_iter()
            .filter_map(|range| string_for_text_range(element, &range))
            .collect::<Vec<_>>();
        non_blank(pieces.join("\n"))
    }

    fn string_for_text_range(element: &CFType, range: &CFType) -> Option<String> {
        copy_parameterized_attribute(element, "AXStringForRange", range)
            .as_ref()
            .and_then(accessibility_text)
            .and_then(non_blank)
    }

    fn focused_element_text(element: &CFType) -> Option<String> {
        let mut seen = HashSet::new();
        let mut pieces = Vec::new();

        if let Some(text) = selected_text(element) {
            push_unique_text(&mut seen, &mut pieces, text);
        }

        for attribute in [
            "AXSelectedText",
            "AXTitle",
            "AXValue",
            "AXLabel",
            "AXDescription",
            "AXPlaceholderValue",
            "AXHelp",
        ] {
            if let Some(text) = copy_attribute(element, attribute)
                .as_ref()
                .and_then(accessibility_text)
            {
                push_unique_text(&mut seen, &mut pieces, text);
            }
        }

        if pieces.is_empty() {
            let mut visited = HashSet::new();
            collect_descendant_text(
                element,
                0,
                MAX_DESCENDANT_TEXT_DEPTH,
                MAX_DESCENDANT_NODES,
                &mut visited,
                &mut seen,
                &mut pieces,
            );
        }

        non_blank(pieces.join("\n"))
    }

    fn collect_descendant_text(
        element: &CFType,
        depth: usize,
        max_depth: usize,
        max_nodes: usize,
        visited: &mut HashSet<usize>,
        seen: &mut HashSet<String>,
        pieces: &mut Vec<String>,
    ) {
        if depth >= max_depth || visited.len() >= max_nodes {
            return;
        }

        for child in child_elements(element) {
            if !visited.insert(element_hash(&child)) {
                continue;
            }

            for attribute in ["AXTitle", "AXValue", "AXLabel", "AXDescription"] {
                if let Some(text) = copy_attribute(&child, attribute)
                    .as_ref()
                    .and_then(accessibility_text)
                {
                    push_unique_text(seen, pieces, text);
                }
            }

            collect_descendant_text(
                &child,
                depth + 1,
                max_depth,
                max_nodes,
                visited,
                seen,
                pieces,
            );
        }
    }

    fn find_selected_text(
        element: &CFType,
        depth: usize,
        max_depth: usize,
        max_nodes: usize,
        visited: &mut HashSet<usize>,
    ) -> Option<(CFType, String)> {
        if depth > max_depth || visited.len() >= max_nodes {
            return None;
        }

        if !visited.insert(element_hash(element)) {
            return None;
        }

        if let Some(text) = selected_text(element) {
            return Some((element.clone(), text));
        }

        for child in child_elements(element) {
            if let Some(result) =
                find_selected_text(&child, depth + 1, max_depth, max_nodes, visited)
            {
                return Some(result);
            }
        }

        None
    }

    fn find_focused_reading(
        element: &CFType,
        depth: usize,
        max_depth: usize,
        max_nodes: usize,
        visited: &mut HashSet<usize>,
    ) -> Option<(CFType, String)> {
        if depth > max_depth || visited.len() >= max_nodes {
            return None;
        }

        if !visited.insert(element_hash(element)) {
            return None;
        }

        if bool_attribute(element, "AXFocused") == Some(true)
            || bool_attribute(element, "AXSelected") == Some(true)
        {
            if is_selection_only_text_surface(element) {
                return None;
            }

            if let Some(text) = focused_element_text(element) {
                return Some((element.clone(), text));
            }
        }

        for child in child_elements(element) {
            if let Some(reading) =
                find_focused_reading(&child, depth + 1, max_depth, max_nodes, visited)
            {
                return Some(reading);
            }
        }

        None
    }

    fn child_elements(element: &CFType) -> Vec<CFType> {
        let mut result = Vec::new();
        let mut seen = HashSet::new();

        for attribute in [
            "AXVisibleChildren",
            "AXChildren",
            "AXRows",
            "AXColumns",
            "AXCells",
            "AXItems",
            "AXContents",
            "AXTabs",
        ] {
            let Some(value) = copy_attribute(element, attribute) else {
                continue;
            };

            if is_ax_element(&value) {
                append_unique_ax(value, &mut seen, &mut result);
                continue;
            }

            for child in array_values(&value) {
                if is_ax_element(&child) {
                    append_unique_ax(child, &mut seen, &mut result);
                }
            }
        }

        result
    }

    fn append_unique_ax(element: CFType, seen: &mut HashSet<usize>, result: &mut Vec<CFType>) {
        if seen.insert(element_hash(&element)) {
            result.push(element);
        }
    }

    fn array_values(value: &CFType) -> Vec<CFType> {
        if unsafe { CFGetTypeID(value.as_CFTypeRef()) } != unsafe { CFArrayGetTypeID() } {
            return Vec::new();
        }

        let array =
            unsafe { CFArray::<CFType>::wrap_under_get_rule(value.as_CFTypeRef() as CFArrayRef) };

        array.iter().map(|item| item.clone()).collect()
    }

    fn element_name(element: &CFType) -> String {
        let role = text_attribute(element, "AXRole").unwrap_or_else(|| "AXElement".to_string());
        if let Some(subrole) = text_attribute(element, "AXSubrole") {
            format!("{}/{}", role, subrole)
        } else {
            role
        }
    }

    fn window_title(window: &CFType) -> Option<String> {
        text_attribute(window, "AXTitle").and_then(non_blank)
    }

    fn text_attribute(element: &CFType, attribute: &'static str) -> Option<String> {
        copy_attribute(element, attribute)
            .as_ref()
            .and_then(accessibility_text)
            .and_then(non_blank)
    }

    fn bool_attribute(element: &CFType, attribute: &'static str) -> Option<bool> {
        let value = copy_attribute(element, attribute)?;
        let type_id = unsafe { CFGetTypeID(value.as_CFTypeRef()) };

        if type_id == unsafe { CFBooleanGetTypeID() } {
            Some(unsafe { CFBooleanGetValue(value.as_CFTypeRef() as _) })
        } else if type_id == CFNumber::type_id() {
            let number = unsafe { CFNumber::wrap_under_get_rule(value.as_CFTypeRef() as _) };
            number.to_i32().map(|value| value != 0)
        } else {
            None
        }
    }

    fn is_selection_only_application(app_name: &str) -> bool {
        matches!(
            normalize(app_name).as_str(),
            "Visual Studio Code" | "Code" | "Code - Insiders" | "VSCodium"
        )
    }

    fn is_selection_only_text_surface(element: &CFType) -> bool {
        if bool_attribute(element, "AXEditable") == Some(true) {
            return true;
        }

        let role = text_attribute(element, "AXRole")
            .map(|role| role.to_ascii_lowercase())
            .unwrap_or_default();
        let subrole = text_attribute(element, "AXSubrole")
            .map(|subrole| subrole.to_ascii_lowercase())
            .unwrap_or_default();

        role.contains("textfield")
            || role.contains("textarea")
            || role.contains("text area")
            || role.contains("combobox")
            || subrole.contains("textfield")
            || subrole.contains("textarea")
            || subrole.contains("text area")
    }

    fn accessibility_text(value: &CFType) -> Option<String> {
        unsafe { text_from_type_ref(value.as_CFTypeRef()) }
    }

    unsafe fn text_from_type_ref(value: CFTypeRef) -> Option<String> {
        if value.is_null() {
            return None;
        }

        let type_id = CFGetTypeID(value);
        if type_id == CFString::type_id() {
            let string = CFString::wrap_under_get_rule(value as _);
            return Some(string.to_string());
        }

        if type_id == CFNumber::type_id() {
            let number = CFNumber::wrap_under_get_rule(value as _);
            return number
                .to_i64()
                .map(|value| value.to_string())
                .or_else(|| number.to_f64().map(|value| value.to_string()));
        }

        if type_id == CFAttributedString::type_id() {
            let string_ref = CFAttributedStringGetString(value as _);
            if !string_ref.is_null() {
                let string = CFString::wrap_under_get_rule(string_ref);
                return Some(string.to_string());
            }
        }

        if type_id == CFArrayGetTypeID() {
            let array = CFArray::<CFType>::wrap_under_get_rule(value as CFArrayRef);
            let pieces = array
                .iter()
                .filter_map(|item| text_from_type_ref(item.as_CFTypeRef()))
                .filter_map(non_blank)
                .collect::<Vec<_>>();
            return non_blank(pieces.join(" "));
        }

        None
    }

    fn push_unique_text(seen: &mut HashSet<String>, pieces: &mut Vec<String>, text: String) {
        let normalized = normalize(&text);
        if !normalized.is_empty() && seen.insert(normalized) {
            pieces.push(text);
        }
    }

    fn element_hash(element: &CFType) -> usize {
        unsafe { CFHash(element.as_CFTypeRef()) }
    }

    fn normalize(value: &str) -> String {
        value
            .replace('\u{00a0}', " ")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn non_blank(value: String) -> Option<String> {
        if value.trim().is_empty() {
            None
        } else {
            Some(value)
        }
    }

    fn non_empty(value: String) -> Option<String> {
        if value.is_empty() {
            None
        } else {
            Some(value)
        }
    }

    #[allow(dead_code)]
    fn release_created_value(value: CFTypeRef) {
        if !value.is_null() {
            unsafe { CFRelease(value) };
        }
    }

    #[cfg(test)]
    mod tests {
        use super::is_selection_only_application;

        #[test]
        fn code_editors_are_selection_only_applications() {
            assert!(is_selection_only_application("Visual Studio Code"));
            assert!(is_selection_only_application("Code"));
            assert!(is_selection_only_application("Code - Insiders"));
            assert!(is_selection_only_application("VSCodium"));
        }

        #[test]
        fn regular_applications_are_not_selection_only_applications() {
            assert!(!is_selection_only_application("Safari"));
            assert!(!is_selection_only_application("Notes"));
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod platform {
    use super::AccessibilityContextEvent;

    pub fn start_recording_context() {}

    pub fn stop_recording_context() -> Vec<AccessibilityContextEvent> {
        Vec::new()
    }
}

pub use platform::{start_recording_context, stop_recording_context};

#[cfg(test)]
mod tests {
    use super::{deduplicate_accessibility_events, AccessibilityContextEvent};

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

        let deduplicated = deduplicate_accessibility_events(&events);

        assert_eq!(
            deduplicated,
            vec![event(300, "Notes", "Draft", "first second third")]
        );
    }

    #[test]
    fn deduplicate_accessibility_events_handles_selection_expanding_backward() {
        let events = vec![
            event(100, "Notes", "Draft", "second"),
            event(200, "Notes", "Draft", "first second"),
        ];

        let deduplicated = deduplicate_accessibility_events(&events);

        assert_eq!(
            deduplicated,
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

        let deduplicated = deduplicate_accessibility_events(&events);

        assert_eq!(deduplicated, events);
    }

    #[test]
    fn deduplicate_accessibility_events_does_not_match_inside_words() {
        let events = vec![
            event(100, "Notes", "Draft", "art"),
            event(200, "Notes", "Draft", "partial"),
        ];

        let deduplicated = deduplicate_accessibility_events(&events);

        assert_eq!(deduplicated, events);
    }
}
