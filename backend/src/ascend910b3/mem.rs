//! Device allocation ownership; native stream completion governs reclamation.

use super::{context::Ascend910B3Context, ffi};
use std::{ffi::c_void, ptr::NonNull, rc::Rc};

pub(crate) struct Ascend910B3DevicePointer {
    pub(crate) raw: NonNull<c_void>,
    pub(crate) context: Rc<Ascend910B3Context>,
}

impl Drop for Ascend910B3DevicePointer {
    fn drop(&mut self) {
        // SAFETY: this is the last logical owner. Its context is still alive;
        // retirement transfers reclamation to native completion-aware ownership.
        let status = unsafe {
            ffi::hp_ascend910b3_v1_buffer_retire(self.context.raw.as_ptr(), self.raw.as_ptr())
        };
        if status.into_result("buffer_retire").is_err() {
            std::process::abort();
        }
    }
}
