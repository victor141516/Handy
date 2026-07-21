use crate::accessibility_context::ReadingSnapshot;

pub(crate) struct PlatformAccessibilityReader;

impl PlatformAccessibilityReader {
    pub(crate) fn new() -> Option<Self> {
        None
    }

    pub(crate) fn current_reading(&mut self) -> Option<ReadingSnapshot> {
        None
    }
}
