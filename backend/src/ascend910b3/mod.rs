//! Ascend910B3 BLS12-381 backend using a statically linked host C ABI.
//!
//! The `ascend910b3` feature is never part of the default CPU build; request it
//! explicitly (`--features ascend910b3`, or `hyperplonk/ascend910b3`). It links
//! the real native bridge and has no CPU fallback. This feature gates the native
//! backend, not the BLS12-381 dependency, which is also available in CPU-only
//! builds and host tests. The crate root gates this entire module; child modules
//! inherit that boundary.
//!
//! `backend/build.rs` owns all native linkage. It first honours the optional
//! prebuilt `HYPERPLONK_ASCEND910B3_LIB_DIR`, requiring both
//! `libhyperplonk_ascend910b3_bridge.a` and `libhyperplonk_ascend910b3_math.a`.
//! Otherwise it configures the original `native/CMakeLists.txt` project in
//! `OUT_DIR` against the installed CANN (`ASCEND_HOME_PATH`, optional `ASC_DIR`),
//! builds the real bridge and math targets once per build directory, and links
//! them with their transitive CANN libraries and runtime search path. Native
//! sources and CANN environment variables are declared as rerun inputs. `ffi`
//! keeps the authoritative `#[link(name = "hyperplonk_ascend910b3_bridge")]`
//! attribute; the build script only contributes search paths and the math/CANN
//! dependencies, so the bridge archive is never linked twice.
//!
//! Each backend uses one host thread and one native stream. Resources retain
//! their context through `Rc`, so dropping the backend before its handles is
//! safe. Handles and backends are neither Send nor Sync. Device outputs may be
//! pending; successful host numerical outputs are ready when the call returns.
//! Native buffer retirement and final context synchronization own physical
//! reclamation; Rust exposes no manual synchronization or collection API.
//!
//! The backend entry point is defined here; `context` owns the shared execution
//! environment lifetime and `mem` owns device buffer retirement. Each algorithm
//! module defines its own polynomial, prepared parameter or prover-state handle.
//! Shared size checks and host FFI pointer/output helpers live in `utils`;
//! numerical packets are encoded in `encoding`. SumCheck descriptions and their
//! validation remain local to `sum_check`.
//! Dimensions are checked when resources are first created; derived MLEs retain
//! that invariant. Native output packet sizes use checked arithmetic before
//! exposing writable pointers. Ordinary Rust containers enforce their own
//! allocation limits rather than translating every capacity failure.
//!
//! Host packets use canonical little-endian field integers and explicit affine
//! point coordinates, not device layouts or the protocol's serialization.
//!
//! Native parameter failures map to [`crate::BackendError::InvalidParameters`];
//! state failures, NotImplemented and malformed numerical outputs map to
//! [`crate::BackendError::InvalidState`]. Diagnostics retain the operation and
//! native status code. A failed destructor or successful creation with a null
//! resource pointer violates the ownership contract and aborts the process.

use self::context::Ascend910B3Context;
use crate::BackendError;
use std::{ptr::NonNull, rc::Rc};

mod context;
mod encoding;
mod ffi;
#[cfg(all(test, feature = "cpu"))]
mod integration_tests;
mod mem;
mod multilinear_kzg;
mod multilinear_polynomial;
mod status;
mod sum_check;
mod utils;

/// Single-threaded, single-stream Ascend910B3 execution entry point.
///
/// Enable `ascend910b3` and link the native bridge. Every resource retains its
/// context through `Rc`; dropping this entry point does not close a context still
/// used by handles. Host numerical results are ready on return.
pub struct Ascend910B3Backend {
    pub(crate) context: Rc<Ascend910B3Context>,
}

impl Ascend910B3Backend {
    /// Create a native context with one stream for a nonnegative device ID.
    ///
    /// Requires the `ascend910b3` feature and the linked native bridge. The
    /// context remains alive until both this entry point and all its handles are
    /// dropped; no manual close is needed.
    pub fn new(device_id: i32) -> Result<Self, BackendError> {
        if device_id < 0 {
            return Err(BackendError::InvalidParameters("negative device ID".into()));
        }
        let mut raw = std::ptr::null_mut();
        // SAFETY: out is a live writable pointer slot; no Rust resources are
        // exposed. On failure the native implementation cleans partial creation.
        let status = unsafe { ffi::hp_ascend910b3_v1_context_create(device_id, &mut raw) };
        status.into_result("context_create")?;
        let raw = NonNull::new(raw).unwrap_or_else(|| std::process::abort());
        Ok(Self {
            context: Rc::new(Ascend910B3Context { raw }),
        })
    }

    pub(crate) fn check_context(
        &self,
        context: &Rc<Ascend910B3Context>,
    ) -> Result<(), BackendError> {
        if Rc::ptr_eq(&self.context, context) {
            Ok(())
        } else {
            Err(BackendError::InvalidParameters(
                "resource belongs to another context".into(),
            ))
        }
    }
}

pub use multilinear_kzg::Ascend910B3MultilinearKzgPreparedProverParam;
pub use multilinear_polynomial::Ascend910B3Mle;
pub use sum_check::Ascend910B3SumCheckProverState;
