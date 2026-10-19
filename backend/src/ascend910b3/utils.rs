//! Shared host FFI boundaries: checked sizes, borrowed pointers and output buffers.
//!
//! This module is enabled with the Ascend910B3 backend feature. Its tests exercise
//! host boundaries without executing device operators.

use super::status::Ascend910B3Status;
use crate::BackendError;

pub(super) fn evaluation_count(num_vars: usize) -> Result<usize, BackendError> {
    if num_vars >= usize::BITS as usize {
        return Err(BackendError::InvalidParameters(
            "MLE variable count exceeds the host shift range".into(),
        ));
    }
    Ok(1usize << num_vars)
}

/// Check both arithmetic overflow and Rust's maximum addressable slice length.
pub(super) fn byte_count(count: usize, width: usize) -> Result<usize, BackendError> {
    count
        .checked_mul(width)
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| BackendError::InvalidParameters("host packet byte length overflow".into()))
}

/// The ABI represents every empty borrowed array by a null pointer.
pub(super) fn input_pointer<T>(values: &[T]) -> *const T {
    if values.is_empty() {
        std::ptr::null()
    } else {
        values.as_ptr()
    }
}

/// Allocate one host output buffer and expose only bytes initialized on success.
///
/// A successful bridge call must write all `length` bytes before returning and
/// make them host-ready. On failure the bridge may leave any bytes unwritten;
/// they are never read or exposed as initialized Rust values.
pub(super) fn collect_output(
    operation: &'static str,
    length: usize,
    fill: impl FnOnce(*mut u8) -> Ascend910B3Status,
) -> Result<Vec<u8>, BackendError> {
    if length > isize::MAX as usize {
        return Err(BackendError::InvalidParameters(format!(
            "{operation} output exceeds the host addressable length"
        )));
    }
    let mut output = Vec::new();
    output.try_reserve_exact(length).map_err(|error| {
        BackendError::InvalidState(format!("{operation} output allocation failed: {error}"))
    })?;
    let pointer = if length == 0 {
        std::ptr::null_mut()
    } else {
        output.as_mut_ptr()
    };
    fill(pointer).into_result(operation)?;
    // SAFETY: capacity is at least length, the allocation stays alive throughout
    // fill, and successful native calls completely initialize host-ready bytes.
    unsafe { output.set_len(length) };
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_check_shift_arithmetic_and_addressable_lengths() {
        assert_eq!(evaluation_count(0).unwrap(), 1);
        assert_eq!(evaluation_count(2).unwrap(), 4);
        assert_eq!(
            evaluation_count(usize::BITS as usize - 1).unwrap(),
            usize::MAX / 2 + 1
        );
        assert!(matches!(
            evaluation_count(usize::BITS as usize),
            Err(BackendError::InvalidParameters(_))
        ));
        assert!(matches!(
            evaluation_count(usize::MAX),
            Err(BackendError::InvalidParameters(_))
        ));
        assert_eq!(byte_count(0, 32).unwrap(), 0);
        assert_eq!(byte_count(3, 32).unwrap(), 96);
        assert_eq!(
            byte_count(isize::MAX as usize, 1).unwrap(),
            isize::MAX as usize
        );
        assert!(matches!(
            byte_count(usize::MAX, 32),
            Err(BackendError::InvalidParameters(_))
        ));
        assert!(matches!(
            byte_count(isize::MAX as usize + 1, 1),
            Err(BackendError::InvalidParameters(_))
        ));
    }
}
