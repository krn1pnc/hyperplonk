//! Status values shared by the host C ABI and the Rust error boundary.

use crate::BackendError;

/// A C-compatible status, never an exception or a Rust discriminated enum.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Ascend910B3Status {
    pub kind: i32,
    pub native_code: i32,
}

const ASCEND910B3_STATUS_SUCCESS: i32 = 0;
const ASCEND910B3_STATUS_INVALID_PARAMETERS: i32 = 1;
const ASCEND910B3_STATUS_INVALID_STATE: i32 = 2;
const ASCEND910B3_STATUS_NOT_IMPLEMENTED: i32 = 3;

impl Ascend910B3Status {
    /// Consume a native status, accepting only an unqualified success.
    ///
    /// `operation` is the ABI symbol suffix after `hp_ascend910b3_v1_`.
    /// Errors preserve that operation, the status kind and the native code.
    pub(crate) fn into_result(self, operation: &'static str) -> Result<(), BackendError> {
        let Self { kind, native_code } = self;
        match kind {
            ASCEND910B3_STATUS_SUCCESS if native_code == 0 => Ok(()),
            ASCEND910B3_STATUS_INVALID_PARAMETERS => Err(BackendError::InvalidParameters(format!(
                "{operation} failed (kind={kind}, native_code={native_code})"
            ))),
            ASCEND910B3_STATUS_NOT_IMPLEMENTED => Err(BackendError::InvalidState(format!(
                "{operation} not implemented (kind=3, native_code={native_code})"
            ))),
            ASCEND910B3_STATUS_INVALID_STATE | ASCEND910B3_STATUS_SUCCESS => {
                Err(BackendError::InvalidState(format!(
                    "{operation} failed (kind={kind}, native_code={native_code})"
                )))
            },
            _ => Err(BackendError::InvalidState(format!(
                "{operation} failed (kind={kind}, native_code={native_code})"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_zero_success_is_successful() {
        assert!(Ascend910B3Status {
            kind: 0,
            native_code: 0,
        }
        .into_result("mle_upload")
        .is_ok());
        for (kind, native_code) in [(0, 7), (2, 0), (2, -42), (3, 0), (3, 9), (-1, 0), (99, 0)] {
            assert!(matches!(
                Ascend910B3Status { kind, native_code }.into_result("mle_upload"),
                Err(BackendError::InvalidState(_))
            ));
        }
    }

    #[test]
    fn invalid_parameters_preserve_error_category() {
        for native_code in [0, -42, 7] {
            assert!(matches!(
                Ascend910B3Status {
                    kind: 1,
                    native_code,
                }
                .into_result("mle_upload"),
                Err(BackendError::InvalidParameters(_))
            ));
        }
    }
}
