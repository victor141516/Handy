use log::{debug, info};

#[derive(Clone, Copy, Debug)]
pub(crate) struct ProcessMemorySnapshot {
    pub resident_bytes: u64,
    pub resident_peak_bytes: u64,
    pub virtual_bytes: u64,
}

pub(crate) fn log_memory_snapshot(label: &str) {
    match process_memory_snapshot() {
        Some(snapshot) => {
            let available = available_memory_bytes()
                .map(|bytes| format!("{:.1} MiB", bytes_to_mib(bytes)))
                .unwrap_or_else(|| "unavailable".to_string());
            info!(
                "memory snapshot [{label}]: resident={:.1} MiB, peak_resident={:.1} MiB, virtual={:.1} MiB, system_available={available}",
                bytes_to_mib(snapshot.resident_bytes),
                bytes_to_mib(snapshot.resident_peak_bytes),
                bytes_to_mib(snapshot.virtual_bytes),
            );
        }
        None => debug!("memory snapshot [{label}]: unavailable on this platform"),
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn available_memory_bytes() -> Option<u64> {
    use std::ffi::{c_int, c_uint};
    use std::mem::{size_of, MaybeUninit};

    const KERN_SUCCESS: c_int = 0;
    const HOST_VM_INFO64: c_int = 4;

    #[repr(C)]
    struct VmStatistics64 {
        free_count: c_uint,
        active_count: c_uint,
        inactive_count: c_uint,
        wire_count: c_uint,
        zero_fill_count: u64,
        reactivations: u64,
        pageins: u64,
        pageouts: u64,
        faults: u64,
        cow_faults: u64,
        lookups: u64,
        hits: u64,
        purges: u64,
        purgeable_count: c_uint,
        speculative_count: c_uint,
        decompressions: u64,
        compressions: u64,
        swapins: u64,
        swapouts: u64,
        compressor_page_count: c_uint,
        throttled_count: c_uint,
        external_page_count: c_uint,
        internal_page_count: c_uint,
        total_uncompressed_pages_in_compressor: u64,
    }

    unsafe extern "C" {
        static mach_task_self_: c_uint;
        fn mach_host_self() -> c_uint;
        fn mach_port_deallocate(task: c_uint, name: c_uint) -> c_int;
        fn host_page_size(host: c_uint, page_size: *mut c_uint) -> c_int;
        fn host_statistics64(
            host: c_uint,
            flavor: c_int,
            info: *mut c_int,
            count: *mut c_uint,
        ) -> c_int;
    }

    let host = unsafe { mach_host_self() };
    let mut page_size = 0;
    let mut stats = MaybeUninit::<VmStatistics64>::zeroed();
    let mut count = (size_of::<VmStatistics64>() / size_of::<c_int>()) as c_uint;
    let page_result = unsafe { host_page_size(host, &mut page_size) };
    let stats_result = unsafe {
        host_statistics64(
            host,
            HOST_VM_INFO64,
            stats.as_mut_ptr().cast::<c_int>(),
            &mut count,
        )
    };
    let _ = unsafe { mach_port_deallocate(mach_task_self_, host) };
    if page_result != KERN_SUCCESS || stats_result != KERN_SUCCESS || page_size == 0 {
        return None;
    }

    let stats = unsafe { stats.assume_init() };
    let reclaimable_pages = u64::from(stats.free_count)
        + u64::from(stats.inactive_count)
        + u64::from(stats.speculative_count)
        + u64::from(stats.purgeable_count);
    Some(reclaimable_pages.saturating_mul(u64::from(page_size)))
}

#[cfg(target_os = "linux")]
pub(crate) fn available_memory_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    meminfo
        .lines()
        .find(|line| line.starts_with("MemAvailable:"))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse::<u64>().ok())
        .map(|kib| kib * 1024)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub(crate) fn available_memory_bytes() -> Option<u64> {
    None
}

fn bytes_to_mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

#[cfg(target_os = "macos")]
fn process_memory_snapshot() -> Option<ProcessMemorySnapshot> {
    use std::ffi::{c_int, c_uint};
    use std::mem::{size_of, MaybeUninit};

    type KernReturn = c_int;
    type MachPort = c_uint;
    type MachMsgTypeNumber = c_uint;

    const KERN_SUCCESS: KernReturn = 0;
    const MACH_TASK_BASIC_INFO: c_int = 20;

    #[repr(C)]
    struct TimeValue {
        seconds: c_int,
        microseconds: c_int,
    }

    #[repr(C)]
    struct MachTaskBasicInfo {
        virtual_size: u64,
        resident_size: u64,
        resident_size_max: u64,
        user_time: TimeValue,
        system_time: TimeValue,
        policy: c_int,
        suspend_count: c_int,
    }

    unsafe extern "C" {
        static mach_task_self_: MachPort;
        fn task_info(
            target_task: MachPort,
            flavor: c_int,
            task_info_out: *mut c_int,
            task_info_out_count: *mut MachMsgTypeNumber,
        ) -> KernReturn;
    }

    let mut info = MaybeUninit::<MachTaskBasicInfo>::zeroed();
    let mut count = (size_of::<MachTaskBasicInfo>() / size_of::<c_uint>()) as MachMsgTypeNumber;
    let result = unsafe {
        task_info(
            mach_task_self_,
            MACH_TASK_BASIC_INFO,
            info.as_mut_ptr().cast::<c_int>(),
            &mut count,
        )
    };
    if result != KERN_SUCCESS {
        return None;
    }

    let info = unsafe { info.assume_init() };
    Some(ProcessMemorySnapshot {
        resident_bytes: info.resident_size,
        resident_peak_bytes: info.resident_size_max,
        virtual_bytes: info.virtual_size,
    })
}

#[cfg(target_os = "linux")]
fn process_memory_snapshot() -> Option<ProcessMemorySnapshot> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let kib = |field: &str| -> Option<u64> {
        status
            .lines()
            .find(|line| line.starts_with(field))
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|value| value.parse::<u64>().ok())
            .map(|value| value * 1024)
    };

    Some(ProcessMemorySnapshot {
        resident_bytes: kib("VmRSS:")?,
        resident_peak_bytes: kib("VmHWM:").unwrap_or(0),
        virtual_bytes: kib("VmSize:").unwrap_or(0),
    })
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn process_memory_snapshot() -> Option<ProcessMemorySnapshot> {
    None
}

#[cfg(test)]
mod tests {
    use super::{available_memory_bytes, bytes_to_mib, process_memory_snapshot};

    #[test]
    fn formats_binary_mebibytes() {
        assert_eq!(bytes_to_mib(1024 * 1024), 1.0);
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn reads_current_process_memory() {
        let snapshot = process_memory_snapshot().expect("process memory should be available");
        assert!(snapshot.resident_bytes > 0);
        assert!(snapshot.virtual_bytes >= snapshot.resident_bytes);
        assert!(available_memory_bytes().is_some_and(|bytes| bytes > 0));
    }
}
