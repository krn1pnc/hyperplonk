// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.
//
// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! CPU execution using immutable, shared dense multilinear polynomials.

pub mod multilinear;
pub mod multilinear_kzg;
pub mod multilinear_polynomial;
pub mod sum_check;
pub mod virtual_polynomial;

pub use ark_poly::DenseMultilinearExtension;
pub use multilinear::{fix_last_variables, identity_permutation_mles, merge_polynomials};

/// CPU implementation using immutable, shared dense MLEs.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuBackend;
