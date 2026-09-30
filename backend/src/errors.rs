// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.
//
// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Error module.
//!
//! Errors from structural and numerical backend operations.

use ark_std::string::String;
use displaydoc::Display;

/// A `enum` specifying the possible failure modes of backend operations.
#[derive(Display, Debug)]
pub enum BackendError {
    /// Invalid parameters: {0}
    InvalidParameters(String),
    /// Invalid execution state: {0}
    InvalidState(String),
}
