#[cfg(target_os = "macos")]
mod macos;
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
mod unsupported;
#[cfg(target_os = "windows")]
mod windows;

#[cfg(target_os = "macos")]
pub(super) use macos::PlatformAccessibilityReader;
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub(super) use unsupported::PlatformAccessibilityReader;
#[cfg(target_os = "windows")]
pub(super) use windows::PlatformAccessibilityReader;

pub(super) const SUPPORTED: bool = cfg!(any(target_os = "macos", target_os = "windows"));
