use crate::accessibility_context::limits::{
    MAX_CONTEXT_ARRAY_ITEMS, MAX_CONTEXT_FIELD_CHARS, MAX_CONTEXT_TEXT_CHARS, MAX_DESCENDANT_NODES,
    MAX_DESCENDANT_TEXT_DEPTH, MAX_FOCUS_DEPTH, MAX_NODES, MAX_SELECTION_DEPTH,
};
use crate::accessibility_context::ReadingSnapshot;
use log::warn;
use std::collections::HashSet;
use std::path::Path;
use windows::core::PWSTR;
use windows::Win32::Foundation::{CloseHandle, HANDLE, HWND};
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CoUninitialize, CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED,
};
use windows::Win32::System::Threading::{
    OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION,
};
use windows::Win32::UI::Accessibility::{
    CUIAutomation, IUIAutomation, IUIAutomationElement, IUIAutomationLegacyIAccessiblePattern,
    IUIAutomationSelectionItemPattern, IUIAutomationTextPattern, IUIAutomationTreeWalker,
    IUIAutomationValuePattern, UIA_DocumentControlTypeId, UIA_EditControlTypeId,
    UIA_LegacyIAccessiblePatternId, UIA_SelectionItemPatternId, UIA_TextPatternId,
    UIA_ValuePatternId,
};
use windows::Win32::UI::WindowsAndMessaging::{
    GetForegroundWindow, GetWindowTextLengthW, GetWindowTextW, GetWindowThreadProcessId,
};

struct ComApartment;

impl ComApartment {
    fn initialize() -> Option<Self> {
        let result = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
        if result.is_err() {
            warn!("Accessibility context recording skipped: COM initialization failed: {result:?}");
            None
        } else {
            Some(Self)
        }
    }
}

impl Drop for ComApartment {
    fn drop(&mut self) {
        unsafe { CoUninitialize() };
    }
}

pub(crate) struct PlatformAccessibilityReader {
    // This guard must be dropped after all UI Automation interfaces.
    automation: IUIAutomation,
    walker: IUIAutomationTreeWalker,
    _com: ComApartment,
}

impl PlatformAccessibilityReader {
    pub(crate) fn new() -> Option<Self> {
        let com = ComApartment::initialize()?;
        let automation = unsafe {
            CoCreateInstance::<_, IUIAutomation>(&CUIAutomation, None, CLSCTX_INPROC_SERVER)
        }
        .map_err(|error| {
            warn!("Accessibility context recording skipped: UI Automation initialization failed: {error}");
            error
        })
        .ok()?;
        let walker = unsafe { automation.ControlViewWalker() }
            .map_err(|error| {
                warn!("Accessibility context recording skipped: UI Automation tree walker failed: {error}");
                error
            })
            .ok()?;

        Some(Self {
            automation,
            walker,
            _com: com,
        })
    }

    pub(crate) fn current_reading(&mut self) -> Option<ReadingSnapshot> {
        let context = active_window_context()?;
        let window = unsafe { self.automation.ElementFromHandle(context.hwnd) }.ok()?;
        let focused = unsafe { self.automation.GetFocusedElement() }.ok();

        let window_title = non_empty(context.window_title.clone())
            .or_else(|| element_text(&window, |element| unsafe { element.CurrentName() }))
            .unwrap_or_else(|| "(unknown window)".to_string());

        if let Some(focused) = focused.as_ref() {
            if let Some(text) = selected_text(focused) {
                return Some(reading(&context, &window_title, "Selected text", text));
            }

            let mut visited = 0;
            if let Some(text) = find_selected_text(
                &self.walker,
                focused,
                0,
                MAX_SELECTION_DEPTH.min(6),
                MAX_NODES,
                &mut visited,
            ) {
                return Some(reading(&context, &window_title, "Selected text", text));
            }
        }

        let mut visited = 0;
        if let Some(text) = find_selected_text(
            &self.walker,
            &window,
            0,
            MAX_SELECTION_DEPTH,
            MAX_NODES,
            &mut visited,
        ) {
            return Some(reading(&context, &window_title, "Selected text", text));
        }

        if focused.as_ref().is_some_and(is_selection_only_surface)
            || is_selection_only_application(&context.app_name)
        {
            return None;
        }

        if let Some(focused) = focused.as_ref() {
            if let Some(text) = focused_element_text(&self.walker, focused) {
                return Some(reading(
                    &context,
                    &window_title,
                    &element_name(focused),
                    text,
                ));
            }
        }

        let mut visited = 0;
        find_focused_reading(
            &self.walker,
            &window,
            0,
            MAX_FOCUS_DEPTH,
            MAX_NODES,
            &mut visited,
        )
        .map(|(element, text)| reading(&context, &window_title, &element_name(&element), text))
    }
}

struct WindowContext {
    hwnd: HWND,
    app_name: String,
    window_title: String,
}

fn active_window_context() -> Option<WindowContext> {
    let hwnd = unsafe { GetForegroundWindow() };
    if hwnd.is_invalid() {
        return None;
    }

    let mut pid = 0_u32;
    unsafe { GetWindowThreadProcessId(hwnd, Some(&mut pid)) };
    if pid == 0 || pid == std::process::id() {
        return None;
    }

    Some(WindowContext {
        hwnd,
        app_name: process_name(pid).unwrap_or_else(|| format!("pid {pid}")),
        window_title: window_title(hwnd),
    })
}

fn process_name(pid: u32) -> Option<String> {
    let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) }.ok()?;
    let handle = ProcessHandle(handle);
    let mut buffer = vec![0_u16; 32_768];
    let mut length = buffer.len() as u32;
    unsafe {
        QueryFullProcessImageNameW(
            handle.0,
            PROCESS_NAME_WIN32,
            PWSTR(buffer.as_mut_ptr()),
            &mut length,
        )
    }
    .ok()?;
    buffer.truncate(length as usize);
    let path = String::from_utf16_lossy(&buffer);
    Path::new(&path)
        .file_stem()
        .and_then(|name| name.to_str())
        .map(str::to_string)
        .and_then(non_blank)
}

struct ProcessHandle(HANDLE);

impl Drop for ProcessHandle {
    fn drop(&mut self) {
        let _ = unsafe { CloseHandle(self.0) };
    }
}

fn window_title(hwnd: HWND) -> String {
    let length = unsafe { GetWindowTextLengthW(hwnd) };
    if length <= 0 {
        return String::new();
    }

    let mut buffer = vec![0_u16; length as usize + 1];
    let copied = unsafe { GetWindowTextW(hwnd, &mut buffer) };
    if copied <= 0 {
        String::new()
    } else {
        String::from_utf16_lossy(&buffer[..copied as usize])
    }
}

fn selected_text(element: &IUIAutomationElement) -> Option<String> {
    let pattern =
        unsafe { element.GetCurrentPatternAs::<IUIAutomationTextPattern>(UIA_TextPatternId) }
            .ok()?;
    let ranges = unsafe { pattern.GetSelection() }.ok()?;
    let length = unsafe { ranges.Length() }
        .ok()?
        .clamp(0, MAX_CONTEXT_ARRAY_ITEMS as i32);
    let mut pieces = Vec::new();
    let mut used_chars = 0;

    for index in 0..length {
        if used_chars >= MAX_CONTEXT_TEXT_CHARS {
            break;
        }
        let range = unsafe { ranges.GetElement(index) }.ok()?;
        let remaining = MAX_CONTEXT_TEXT_CHARS - used_chars;
        let text = unsafe { range.GetText(remaining.min(i32::MAX as usize) as i32) }
            .ok()
            .map(|value| value.to_string())
            .and_then(non_blank);
        if let Some(text) = text {
            used_chars += text.chars().count() + usize::from(!pieces.is_empty());
            pieces.push(text);
        }
    }

    non_blank(truncate_text(&pieces.join("\n"), MAX_CONTEXT_TEXT_CHARS))
}

fn find_selected_text(
    walker: &IUIAutomationTreeWalker,
    element: &IUIAutomationElement,
    depth: usize,
    max_depth: usize,
    max_nodes: usize,
    visited: &mut usize,
) -> Option<String> {
    if depth > max_depth || *visited >= max_nodes {
        return None;
    }
    *visited += 1;

    if let Some(text) = selected_text(element) {
        return Some(text);
    }

    for child in child_elements(walker, element) {
        if let Some(text) =
            find_selected_text(walker, &child, depth + 1, max_depth, max_nodes, visited)
        {
            return Some(text);
        }
    }
    None
}

fn find_focused_reading(
    walker: &IUIAutomationTreeWalker,
    element: &IUIAutomationElement,
    depth: usize,
    max_depth: usize,
    max_nodes: usize,
    visited: &mut usize,
) -> Option<(IUIAutomationElement, String)> {
    if depth > max_depth || *visited >= max_nodes {
        return None;
    }
    *visited += 1;

    if is_focused_or_selected(element) && !is_selection_only_surface(element) {
        if let Some(text) = focused_element_text(walker, element) {
            return Some((element.clone(), text));
        }
    }

    for child in child_elements(walker, element) {
        if let Some(reading) =
            find_focused_reading(walker, &child, depth + 1, max_depth, max_nodes, visited)
        {
            return Some(reading);
        }
    }
    None
}

fn child_elements(
    walker: &IUIAutomationTreeWalker,
    element: &IUIAutomationElement,
) -> Vec<IUIAutomationElement> {
    let mut children = Vec::new();
    let mut current = unsafe { walker.GetFirstChildElement(element) }.ok();
    while let Some(child) = current {
        current = unsafe { walker.GetNextSiblingElement(&child) }.ok();
        children.push(child);
    }
    children
}

fn focused_element_text(
    walker: &IUIAutomationTreeWalker,
    element: &IUIAutomationElement,
) -> Option<String> {
    let mut seen = HashSet::new();
    let mut pieces = Vec::new();

    if let Some(text) = selected_text(element) {
        push_unique_text(&mut seen, &mut pieces, text);
    }
    if let Some(text) = element_text(element, |element| unsafe { element.CurrentName() }) {
        push_unique_text(&mut seen, &mut pieces, text);
    }
    if let Some(text) = value_pattern_text(element) {
        push_unique_text(&mut seen, &mut pieces, text);
    }
    if let Some(text) = element_text(element, |element| unsafe { element.CurrentHelpText() }) {
        push_unique_text(&mut seen, &mut pieces, text);
    }
    if let Some(text) = legacy_accessible_text(element) {
        push_unique_text(&mut seen, &mut pieces, text);
    }

    if pieces.is_empty() {
        let mut visited = 0;
        collect_descendant_text(
            walker,
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

#[allow(clippy::too_many_arguments)]
fn collect_descendant_text(
    walker: &IUIAutomationTreeWalker,
    element: &IUIAutomationElement,
    depth: usize,
    max_depth: usize,
    max_nodes: usize,
    visited: &mut usize,
    seen: &mut HashSet<String>,
    pieces: &mut Vec<String>,
) {
    if depth >= max_depth || *visited >= max_nodes {
        return;
    }

    for child in child_elements(walker, element) {
        if collected_text_chars(pieces) >= MAX_CONTEXT_TEXT_CHARS || *visited >= max_nodes {
            return;
        }
        *visited += 1;

        if let Some(text) = element_text(&child, |element| unsafe { element.CurrentName() }) {
            push_unique_text(seen, pieces, text);
        }
        if let Some(text) = value_pattern_text(&child) {
            push_unique_text(seen, pieces, text);
        }
        if let Some(text) = element_text(&child, |element| unsafe { element.CurrentHelpText() }) {
            push_unique_text(seen, pieces, text);
        }

        collect_descendant_text(
            walker,
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

fn value_pattern_text(element: &IUIAutomationElement) -> Option<String> {
    let pattern =
        unsafe { element.GetCurrentPatternAs::<IUIAutomationValuePattern>(UIA_ValuePatternId) }
            .ok()?;
    unsafe { pattern.CurrentValue() }
        .ok()
        .map(|value| value.to_string())
        .and_then(non_blank)
}

fn legacy_accessible_text(element: &IUIAutomationElement) -> Option<String> {
    let pattern = unsafe {
        element.GetCurrentPatternAs::<IUIAutomationLegacyIAccessiblePattern>(
            UIA_LegacyIAccessiblePatternId,
        )
    }
    .ok()?;
    let mut pieces = Vec::new();
    if let Some(name) = unsafe { pattern.CurrentName() }
        .ok()
        .map(|value| value.to_string())
        .and_then(non_blank)
    {
        pieces.push(name);
    }
    if let Some(value) = unsafe { pattern.CurrentValue() }
        .ok()
        .map(|value| value.to_string())
        .and_then(non_blank)
    {
        pieces.push(value);
    }
    non_blank(pieces.join("\n"))
}

fn is_focused_or_selected(element: &IUIAutomationElement) -> bool {
    if unsafe { element.CurrentHasKeyboardFocus() }
        .ok()
        .is_some_and(|value| value.as_bool())
    {
        return true;
    }

    unsafe {
        element.GetCurrentPatternAs::<IUIAutomationSelectionItemPattern>(UIA_SelectionItemPatternId)
    }
    .ok()
    .and_then(|pattern| unsafe { pattern.CurrentIsSelected() }.ok())
    .is_some_and(|value| value.as_bool())
}

fn is_selection_only_application(app_name: &str) -> bool {
    matches!(
        normalize(app_name).to_ascii_lowercase().as_str(),
        "visual studio code" | "code" | "code - insiders" | "vscodium"
    )
}

fn is_selection_only_surface(element: &IUIAutomationElement) -> bool {
    let control_type = unsafe { element.CurrentControlType() }.ok();
    if control_type == Some(UIA_EditControlTypeId) {
        return true;
    }

    if control_type == Some(UIA_DocumentControlTypeId) {
        return unsafe {
            element.GetCurrentPatternAs::<IUIAutomationValuePattern>(UIA_ValuePatternId)
        }
        .ok()
        .and_then(|pattern| unsafe { pattern.CurrentIsReadOnly() }.ok())
        .is_some_and(|read_only| !read_only.as_bool());
    }

    false
}

fn element_name(element: &IUIAutomationElement) -> String {
    element_text(element, |element| unsafe {
        element.CurrentLocalizedControlType()
    })
    .unwrap_or_else(|| "UI Automation element".to_string())
}

fn element_text<F>(element: &IUIAutomationElement, read: F) -> Option<String>
where
    F: FnOnce(&IUIAutomationElement) -> windows::core::Result<windows::core::BSTR>,
{
    read(element)
        .ok()
        .map(|value| value.to_string())
        .and_then(non_blank)
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

#[cfg(test)]
mod tests {
    use super::{is_selection_only_application, truncate_text};

    #[test]
    fn code_editors_are_selection_only_applications() {
        assert!(is_selection_only_application("Visual Studio Code"));
        assert!(is_selection_only_application("Code"));
        assert!(is_selection_only_application("Code - Insiders"));
        assert!(is_selection_only_application("VSCodium"));
    }

    #[test]
    fn regular_applications_are_not_selection_only_applications() {
        assert!(!is_selection_only_application("Notepad"));
        assert!(!is_selection_only_application("chrome"));
    }

    #[test]
    fn truncated_text_preserves_unicode_boundary() {
        assert_eq!(truncate_text("abcdef", 4), "abc…");
        assert_eq!(truncate_text("áéíóú", 4), "áéí…");
    }
}
