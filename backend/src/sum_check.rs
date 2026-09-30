// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.
//
// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! SumCheck numerical prover execution interface.

use crate::{BackendError, MleBackend, VirtualPolynomial};
use ark_ff::PrimeField;

/// Trait for sum check protocol prover side APIs.
///
/// Each initialization creates independent working state without modifying the
/// input polynomial. The protocol driver owns the transcript and challenge
/// history; returned round evaluations must be ready for host consumption.
pub trait SumCheckProver<F: PrimeField>: MleBackend<F> {
    type ProverState;

    /// Initialize the prover state to argue for the sum of the input polynomial
    /// over {0,1}^`num_vars`.
    /// Requires positive variable-count and degree metadata.
    fn prover_init(
        &self,
        polynomial: &VirtualPolynomial<F, Self::Mle>,
    ) -> Result<Self::ProverState, BackendError>;

    /// Receive the preceding round's challenge, generate prover message
    /// evaluations, and update the prover state for the next round.
    /// The first round requires `None`; subsequent rounds require `Some`.
    ///
    /// Main algorithm used is from section 3.2 of [XZZPS19](https://eprint.iacr.org/2019/317.pdf#subsection.3.2).
    fn prove_round_and_update_state(
        &self,
        state: &mut Self::ProverState,
        challenge: &Option<F>,
    ) -> Result<Vec<F>, BackendError>;
}
