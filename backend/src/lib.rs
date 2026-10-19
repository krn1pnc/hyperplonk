// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.
//
// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Mathematical objects, common host mathematics, and backend execution
//! interfaces.
//!
//! The `cpu` and `ascend910b3` features gate backend implementations, not the
//! `ark-poly` or `ark-bls12-381` dependencies, which are always available.
//! Backend-specific tests follow the same implementation feature gates.

#[cfg(feature = "ascend910b3")]
pub mod ascend910b3;
pub mod common;
#[cfg(feature = "cpu")]
pub mod cpu;
pub mod errors;
pub mod multilinear_kzg;
pub mod multilinear_polynomial;
pub mod sum_check;
pub mod virtual_polynomial;

pub use errors::BackendError;
pub use multilinear_kzg::MultilinearKzgBackend;
pub use multilinear_polynomial::{Mle, MleBackend};
pub use sum_check::SumCheckProver;
pub use virtual_polynomial::{VPAuxInfo, VirtualPolynomial};
