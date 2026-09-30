// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

mod errors;
mod multilinear_kzg;
mod structs;

pub mod prelude;

use crate::PCSError;
use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::Rng;
use std::{borrow::Borrow, fmt::Debug, hash::Hash};
use transcript::IOPTranscript;

use backend::{MultilinearKzgBackend, SumCheckProver};

/// This trait defines APIs for polynomial commitment schemes.
/// Note that for our usage of PCS, we do not require the hiding property.
pub trait PolynomialCommitmentScheme<E: Pairing> {
    /// Host prover parameters returned by `trim`, before backend preparation.
    type ProverParam: Clone + Sync;
    /// Verifier parameters
    type VerifierParam: Clone + CanonicalSerialize + CanonicalDeserialize;
    /// Structured reference string
    type SRS: Clone + Debug;
    /// Polynomial input domain
    type Point: Clone + Ord + Debug + Sync + Hash + PartialEq + Eq;
    /// Polynomial Evaluation
    type Evaluation: Field;
    /// Commitments
    type Commitment: Clone + CanonicalSerialize + CanonicalDeserialize + Debug + PartialEq + Eq;
    /// Proofs
    type Proof: Clone + CanonicalSerialize + CanonicalDeserialize + Debug + PartialEq + Eq;
    /// Batch proofs
    type BatchProof;

    /// Build SRS for testing.
    ///
    /// `supported_num_vars` is the number of variables.
    ///
    /// WARNING: THIS FUNCTION IS FOR TESTING PURPOSE ONLY.
    /// THE OUTPUT SRS SHOULD NOT BE USED IN PRODUCTION.
    fn gen_srs_for_testing<R: Rng>(
        rng: &mut R,
        supported_num_vars: usize,
    ) -> Result<Self::SRS, PCSError>;

    /// Trim the universal parameters to specialize the public parameters.
    /// `supported_num_vars` is the number of variables.
    /// ## Note on function signature
    /// Usually, data structure like SRS and ProverParam are huge and users
    /// might wish to keep them in heap using different kinds of smart pointers
    /// (instead of only in stack) therefore our `impl Borrow<_>` interface
    /// allows for passing in any pointer type, e.g.: `trim(srs: &Self::SRS,
    /// ..)` or `trim(srs: Box<Self::SRS>, ..)` or `trim(srs: Arc<Self::SRS>,
    /// ..)` etc.
    fn trim(
        srs: impl Borrow<Self::SRS>,
        supported_num_vars: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError>;

    /// Generate a commitment for a polynomial.
    /// Uses already prepared backend parameters.
    fn commit<B: MultilinearKzgBackend<E>>(
        backend: &B,
        prover_param: &B::PreparedProverParam,
        poly: &B::Mle,
    ) -> Result<Self::Commitment, PCSError>;

    /// Generate independent commitments in input order.
    fn multi_commit<B: MultilinearKzgBackend<E>>(
        backend: &B,
        prover_param: &B::PreparedProverParam,
        polynomials: &[B::Mle],
    ) -> Result<Vec<Self::Commitment>, PCSError>;

    /// On input a polynomial `p` and a point `point`, outputs a proof for the
    /// same.
    /// A complete point and prepared backend parameters are required.
    fn open<B: MultilinearKzgBackend<E>>(
        backend: &B,
        prover_param: &B::PreparedProverParam,
        polynomial: &B::Mle,
        point: &Self::Point,
    ) -> Result<(Self::Proof, Self::Evaluation), PCSError>;

    /// Input a list of multilinear polynomial handles, and a same number of
    /// points, and a transcript, compute a multi-opening for all the polynomials.
    /// Uses backend numerical operations with prepared parameters.
    ///
    /// Input lists must have equal nonzero lengths. All polynomials and points
    /// must share a positive dimension.
    fn multi_open<B: MultilinearKzgBackend<E> + SumCheckProver<E::ScalarField>>(
        backend: &B,
        prover_param: &B::PreparedProverParam,
        polynomials: &[B::Mle],
        points: &[Self::Point],
        evals: &[Self::Evaluation],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<Self::BatchProof, PCSError>;

    /// Verifies that `value` is the evaluation at `x` of the polynomial
    /// committed inside `comm`.
    fn verify(
        verifier_param: &Self::VerifierParam,
        commitment: &Self::Commitment,
        point: &Self::Point,
        value: &E::ScalarField,
        proof: &Self::Proof,
    ) -> Result<bool, PCSError>;

    /// Verifies that `value_i` is the evaluation at `x_i` of the polynomial
    /// `poly_i` committed inside `comm`.
    ///
    /// Commitments, points, and evaluations in the proof must have equal,
    /// nonzero lengths, and all points must share a positive dimension.
    fn batch_verify(
        verifier_param: &Self::VerifierParam,
        commitments: &[Self::Commitment],
        points: &[Self::Point],
        batch_proof: &Self::BatchProof,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<bool, PCSError>;
}

/// API definitions for structured reference string
pub trait StructuredReferenceString<E: Pairing>: Sized {
    /// Prover parameters
    type ProverParam;
    /// Verifier parameters
    type VerifierParam;

    /// Trim the universal parameters to specialize the public parameters
    /// for polynomials to the given `supported_num_vars`, and
    /// returns committer key and verifier key.
    ///
    /// `supported_num_vars` should be in range `0..=params.num_vars`.
    fn trim(
        &self,
        supported_num_vars: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError>;

    /// Build SRS for testing.
    ///
    /// `supported_num_vars` is the number of variables.
    ///
    /// WARNING: THIS FUNCTION IS FOR TESTING PURPOSE ONLY.
    /// THE OUTPUT SRS SHOULD NOT BE USED IN PRODUCTION.
    fn gen_srs_for_testing<R: Rng>(
        rng: &mut R,
        supported_num_vars: usize,
    ) -> Result<Self, PCSError>;
}
