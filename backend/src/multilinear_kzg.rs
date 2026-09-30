// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.
//
// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Raw multilinear KZG execution interface, independent of protocol wrappers.

use crate::{BackendError, MleBackend};
use ark_ec::pairing::Pairing;

/// Computes multilinear KZG commitments and openings without exposing MLE data.
///
/// Preparation consumes the host parameters and returns backend-owned
/// resources. Polynomial inputs remain immutable. Parameters and polynomials
/// must belong to compatible backend contexts. Returned commitments, proofs,
/// and scalar values must be ready for host consumption; polynomial data need
/// not be host-resident. Trimming and verification are independent of this
/// execution backend.
pub trait MultilinearKzgBackend<E: Pairing>: MleBackend<E::ScalarField> {
    type PreparedProverParam;

    /// Prepare the host parameters once, transferring their ownership. A device
    /// backend must finish using host upload buffers before releasing them.
    fn prepare_prover_param(
        &self,
        num_vars: usize,
        powers_of_g: impl ExactSizeIterator<Item = Vec<E::G1Affine>>,
    ) -> Result<Self::PreparedProverParam, BackendError>;

    /// Generate a commitment for a polynomial.
    fn commit(
        &self,
        prover_param: &Self::PreparedProverParam,
        polynomial: &Self::Mle,
    ) -> Result<E::G1Affine, BackendError>;

    /// Commit each input independently, preserving order and duplicates. Empty
    /// input returns an empty list; supported polynomial dimensions may differ.
    fn multi_commit(
        &self,
        prover_param: &Self::PreparedProverParam,
        polynomials: &[Self::Mle],
    ) -> Result<Vec<E::G1Affine>, BackendError>;

    /// On input a polynomial `p` and a point `point`, outputs the quotient
    /// commitments and the evaluation at that point.
    /// A complete point is required, including `[]` for constants.
    fn open(
        &self,
        prover_param: &Self::PreparedProverParam,
        polynomial: &Self::Mle,
        point: &[E::ScalarField],
    ) -> Result<(Vec<E::G1Affine>, E::ScalarField), BackendError>;
}
