mod implementation {
    use crate::accessibility_context::limits::{
        MAX_CONTEXT_ARRAY_ITEMS, MAX_CONTEXT_FIELD_CHARS, MAX_CONTEXT_TEXT_CHARS,
        MAX_DESCENDANT_NODES, MAX_DESCENDANT_TEXT_DEPTH, MAX_FOCUS_DEPTH, MAX_NODES,
        MAX_SELECTION_DEPTH,
    };
    use crate::accessibility_context::ReadingSnapshot;
    use core_foundation::array::CFArray;
    use core_foundation::attributed_string::CFAttributedString;
    use core_foundation::base::{CFType, CFTypeRef, TCFType};
    use core_foundation::boolean::CFBoolean;
    use core_foundation::dictionary::CFDictionary;
    use core_foundation::number::CFNumber;
    use core_foundation::string::{CFString, CFStringRef};
    use core_foundation_sys::array::{CFArrayGetTypeID, CFArrayRef};
    use core_foundation_sys::attributed_string::CFAttributedStringGetString;
    use core_foundation_sys::base::{Boolean, CFGetTypeID, CFHash, CFRange, CFRelease, CFTypeID};
    use core_foundation_sys::dictionary::{
        CFDictionaryGetTypeID, CFDictionaryGetValueIfPresent, CFDictionaryRef,
    };
    use core_foundation_sys::number::{CFBooleanGetTypeID, CFBooleanGetValue};
    use core_foundation_sys::string::{kCFStringEncodingUTF8, CFStringGetBytes, CFStringGetLength};
    use core_graphics::window::{
        copy_window_info, kCGNullWindowID, kCGWindowAlpha, kCGWindowLayer,
        kCGWindowListExcludeDesktopElements, kCGWindowListOptionOnScreenOnly, kCGWindowName,
        kCGWindowOwnerName, kCGWindowOwnerPID,
    };
    use log::warn;
    use std::collections::HashSet;
    use std::ffi::c_void;
    use std::ptr;

    const AX_SUCCESS: i32 = 0;

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

    pub(crate) struct PlatformAccessibilityReader;

    impl PlatformAccessibilityReader {
        pub(crate) fn new() -> Option<Self> {
            if accessibility_trusted(true) {
                Some(Self)
            } else {
                warn!(
                    "Accessibility context recording skipped: accessibility permission is missing"
                );
                None
            }
        }

        pub(crate) fn current_reading(&mut self) -> Option<ReadingSnapshot> {
            current_reading()
        }
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
            app_name: truncate_text(&context.app_name, MAX_CONTEXT_FIELD_CHARS),
            window_title: truncate_text(window_title, MAX_CONTEXT_FIELD_CHARS),
            element: truncate_text(element, MAX_CONTEXT_FIELD_CHARS),
            text: truncate_text(&text, MAX_CONTEXT_TEXT_CHARS),
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

        value_from_ax_copy_result(error, value)
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

        value_from_ax_copy_result(error, value)
    }

    fn value_from_ax_copy_result(error: i32, value: CFTypeRef) -> Option<CFType> {
        if error == AX_SUCCESS && !value.is_null() {
            Some(unsafe { CFType::wrap_under_create_rule(value) })
        } else {
            release_created_value(value);
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
                if collected_text_chars(&pieces) >= MAX_CONTEXT_TEXT_CHARS {
                    break;
                }
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
            if collected_text_chars(pieces) >= MAX_CONTEXT_TEXT_CHARS {
                return;
            }

            if !visited.insert(element_hash(&child)) {
                continue;
            }

            for attribute in ["AXTitle", "AXValue", "AXLabel", "AXDescription"] {
                if let Some(text) = copy_attribute(&child, attribute)
                    .as_ref()
                    .and_then(accessibility_text)
                {
                    push_unique_text(seen, pieces, text);
                    if collected_text_chars(pieces) >= MAX_CONTEXT_TEXT_CHARS {
                        return;
                    }
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
            return non_blank(bounded_cf_string(
                value as CFStringRef,
                MAX_CONTEXT_TEXT_CHARS,
            ));
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
                return non_blank(bounded_cf_string(string_ref, MAX_CONTEXT_TEXT_CHARS));
            }
        }

        if type_id == CFArrayGetTypeID() {
            let array = CFArray::<CFType>::wrap_under_get_rule(value as CFArrayRef);
            let pieces = array
                .iter()
                .take(MAX_CONTEXT_ARRAY_ITEMS)
                .filter_map(|item| text_from_type_ref(item.as_CFTypeRef()))
                .filter_map(non_blank)
                .collect::<Vec<_>>();
            return non_blank(truncate_text(&pieces.join(" "), MAX_CONTEXT_TEXT_CHARS));
        }

        None
    }

    fn push_unique_text(seen: &mut HashSet<String>, pieces: &mut Vec<String>, text: String) {
        let used_chars = collected_text_chars(pieces);
        let separator_chars = usize::from(!pieces.is_empty());
        if used_chars.saturating_add(separator_chars) >= MAX_CONTEXT_TEXT_CHARS {
            return;
        }

        let text = truncate_text(&text, MAX_CONTEXT_TEXT_CHARS - used_chars - separator_chars);
        let normalized = normalize(&text);
        if !normalized.is_empty() && seen.insert(normalized) {
            pieces.push(text);
        }
    }

    fn collected_text_chars(pieces: &[String]) -> usize {
        pieces
            .iter()
            .map(|piece| piece.chars().count())
            .sum::<usize>()
            .saturating_add(pieces.len().saturating_sub(1))
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

    fn truncate_text(value: &str, max_chars: usize) -> String {
        if value.chars().count() <= max_chars {
            return value.to_string();
        }

        let mut truncated = value
            .chars()
            .take(max_chars.saturating_sub(1))
            .collect::<String>();
        truncated.push('…');
        truncated
    }

    unsafe fn bounded_cf_string(value: CFStringRef, max_utf16_units: usize) -> String {
        let total_units = CFStringGetLength(value).max(0);
        let requested_units = total_units.min(max_utf16_units as isize);
        if requested_units == 0 {
            return String::new();
        }

        let range = CFRange::init(0, requested_units);
        let mut bytes_required = 0;
        let converted_units = CFStringGetBytes(
            value,
            range,
            kCFStringEncodingUTF8,
            0,
            false as Boolean,
            ptr::null_mut(),
            0,
            &mut bytes_required,
        );
        if converted_units <= 0 || bytes_required <= 0 {
            return String::new();
        }

        let mut buffer = vec![0_u8; bytes_required as usize];
        let mut bytes_used = 0;
        let converted_units = CFStringGetBytes(
            value,
            range,
            kCFStringEncodingUTF8,
            0,
            false as Boolean,
            buffer.as_mut_ptr(),
            buffer.len() as isize,
            &mut bytes_used,
        );
        buffer.truncate(bytes_used.max(0) as usize);
        let mut result = String::from_utf8_lossy(&buffer).into_owned();
        if converted_units < total_units {
            result.push('…');
        }
        result
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

    fn release_created_value(value: CFTypeRef) {
        if !value.is_null() {
            unsafe { CFRelease(value) };
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{
            bounded_cf_string, is_selection_only_application, push_unique_text, truncate_text,
            CFString, HashSet, TCFType, MAX_CONTEXT_TEXT_CHARS,
        };

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

        #[test]
        fn truncated_text_preserves_unicode_boundary() {
            assert_eq!(truncate_text("abcdef", 4), "abc…");
            assert_eq!(truncate_text("áéíóú", 4), "áéí…");
        }

        #[test]
        fn core_foundation_strings_are_copied_with_a_hard_limit() {
            let value = CFString::new(&"x".repeat(100_000));
            let bounded = unsafe { bounded_cf_string(value.as_concrete_TypeRef(), 32) };

            assert_eq!(bounded.chars().count(), 33);
            assert!(bounded.ends_with('…'));
        }

        #[test]
        fn collected_text_parts_have_an_aggregate_limit() {
            let mut seen = HashSet::new();
            let mut pieces = Vec::new();
            push_unique_text(
                &mut seen,
                &mut pieces,
                "a".repeat(MAX_CONTEXT_TEXT_CHARS - 10),
            );
            push_unique_text(&mut seen, &mut pieces, "b".repeat(1_000));
            push_unique_text(&mut seen, &mut pieces, "ignored".to_string());

            let chars = pieces
                .iter()
                .map(|piece| piece.chars().count())
                .sum::<usize>()
                + pieces.len().saturating_sub(1);
            assert!(chars <= MAX_CONTEXT_TEXT_CHARS);
        }
    }
}

pub(crate) use implementation::PlatformAccessibilityReader;
