// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Error module.

use ark_serialize::SerializationError;
use ark_std::string::String;
use backend::BackendError;
use displaydoc::Display;
use subroutines::{PCSError, PolyIOPErrors};
use transcript::TranscriptError;

/// A `enum` specifying the possible failure modes of hyperplonk.
#[derive(Display, Debug)]
pub enum HyperPlonkErrors {
    /// Invalid Prover: {0}
    InvalidProver(String),
    /// Invalid Verifier: {0}
    InvalidVerifier(String),
    /// Invalid Proof: {0}
    InvalidProof(String),
    /// Invalid parameters: {0}
    InvalidParameters(String),
    /// An error during (de)serialization: {0}
    SerializationError(SerializationError),
    /// PolyIOP error {0}
    PolyIOPErrors(PolyIOPErrors),
    /// PCS error {0}
    PCSErrors(PCSError),
    /// Transcript error {0}
    TranscriptError(TranscriptError),
}

impl From<SerializationError> for HyperPlonkErrors {
    fn from(e: ark_serialize::SerializationError) -> Self {
        Self::SerializationError(e)
    }
}

impl From<PolyIOPErrors> for HyperPlonkErrors {
    fn from(e: PolyIOPErrors) -> Self {
        Self::PolyIOPErrors(e)
    }
}

impl From<PCSError> for HyperPlonkErrors {
    fn from(e: PCSError) -> Self {
        Self::PCSErrors(e)
    }
}

impl From<TranscriptError> for HyperPlonkErrors {
    fn from(e: TranscriptError) -> Self {
        Self::TranscriptError(e)
    }
}

impl From<BackendError> for HyperPlonkErrors {
    fn from(e: BackendError) -> Self {
        match e {
            BackendError::InvalidParameters(message) => Self::InvalidParameters(message),
            BackendError::InvalidState(message) => Self::InvalidProver(message),
        }
    }
}
