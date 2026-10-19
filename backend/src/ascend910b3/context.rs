//! Shared native execution-context ownership and final synchronized destruction.

use super::ffi;
use std::ptr::NonNull;

pub(crate) struct Ascend910B3Context {
    pub(crate) raw: NonNull<ffi::Ascend910B3NativeContext>,
}

impl Drop for Ascend910B3Context {
    fn drop(&mut self) {
        // SAFETY: no backend or resource retains this context. Native destruction
        // synchronizes and drains retired allocations before closing the context.
        let status = unsafe { ffi::hp_ascend910b3_v1_context_destroy(self.raw.as_ptr()) };
        if status.into_result("context_destroy").is_err() {
            std::process::abort();
        }
    }
}
