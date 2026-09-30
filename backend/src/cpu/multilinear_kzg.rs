// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.
//
// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! CPU-owned raw KZG resources and numerical execution.

use super::CpuBackend;
use crate::{BackendError, MultilinearKzgBackend};
use ark_ec::{pairing::Pairing, scalar_mul::variable_base::VariableBaseMSM, CurveGroup};
use ark_std::{end_timer, start_timer, Zero};
use rayon::prelude::*;

/// Owned CPU SRS layers. Fields are private to the execution implementation.
pub struct CpuPreparedProverParam<E: Pairing> {
    num_vars: usize,
    powers_of_g: Vec<Vec<E::G1Affine>>,
}

impl<E: Pairing> MultilinearKzgBackend<E> for CpuBackend {
    type PreparedProverParam = CpuPreparedProverParam<E>;

    fn prepare_prover_param(
        &self,
        num_vars: usize,
        powers_of_g: impl ExactSizeIterator<Item = Vec<E::G1Affine>>,
    ) -> Result<Self::PreparedProverParam, BackendError> {
        // The host trim contract supplies N+1 owned layers, in decreasing dimensions.
        Ok(CpuPreparedProverParam {
            num_vars,
            powers_of_g: powers_of_g.collect(),
        })
    }

    /// Generate a commitment for a polynomial.
    ///
    /// This function takes `2^num_vars` number of scalar multiplications over
    /// G1.
    fn commit(
        &self,
        prover_param: &Self::PreparedProverParam,
        polynomial: &Self::Mle,
    ) -> Result<E::G1Affine, BackendError> {
        let commit_timer = start_timer!(|| "commit");
        if prover_param.num_vars < polynomial.num_vars {
            return Err(BackendError::InvalidParameters(format!(
                "MlE length ({}) exceeds param limit ({})",
                polynomial.num_vars, prover_param.num_vars
            )));
        }
        let ignored = prover_param.num_vars - polynomial.num_vars;
        let bases = &prover_param.powers_of_g[ignored];
        let msm_timer = start_timer!(|| format!("msm of size {}", bases.len()));
        let commitment = E::G1::msm_unchecked(bases, &polynomial.evaluations).into_affine();
        end_timer!(msm_timer);
        end_timer!(commit_timer);
        Ok(commitment)
    }

    fn multi_commit(
        &self,
        prover_param: &Self::PreparedProverParam,
        polynomials: &[Self::Mle],
    ) -> Result<Vec<E::G1Affine>, BackendError> {
        polynomials
            .par_iter()
            .map(|polynomial| self.commit(prover_param, polynomial))
            .collect()
    }

    /// On input a polynomial `p` and a point `point`, outputs the quotient
    /// commitments and the evaluation at that point. This function does not
    /// need to take the evaluation value as an input.
    ///
    /// It proceeds with `num_var` number of rounds:
    /// - at round i, indexed from 1, we compute an MSM for `2^{num_var - i}`
    ///   number of G1 elements.
    fn open(
        &self,
        prover_param: &Self::PreparedProverParam,
        polynomial: &Self::Mle,
        point: &[E::ScalarField],
    ) -> Result<(Vec<E::G1Affine>, E::ScalarField), BackendError> {
        let open_timer = start_timer!(|| format!("open mle with {} variable", polynomial.num_vars));
        let nv = polynomial.num_vars;
        if nv > prover_param.num_vars {
            return Err(BackendError::InvalidParameters(format!(
                "Polynomial num_vars {} exceed the limit {}",
                nv, prover_param.num_vars
            )));
        }
        if nv != point.len() {
            return Err(BackendError::InvalidParameters(format!(
                "Polynomial num_vars {} does not match point len {}",
                nv,
                point.len()
            )));
        }
        if nv == 0 {
            end_timer!(open_timer);
            return Ok((Vec::new(), polynomial.evaluations[0]));
        }

        // the first `ignored` SRS vectors are unused for opening.
        let ignored = prover_param.num_vars - nv + 1;
        let mut f = polynomial.evaluations.clone();
        let mut q = vec![E::ScalarField::zero(); f.len() / 2];
        let mut proofs = Vec::with_capacity(nv);
        for (i, (&coordinate, gi)) in point
            .iter()
            .zip(&prover_param.powers_of_g[ignored..ignored + nv])
            .enumerate()
        {
            let round_timer = start_timer!(|| format!("{}-th round", i));
            let eval_timer = start_timer!(|| format!("{}-th round eval", i));
            let cur_dim = f.len() / 2;
            q.truncate(cur_dim);
            for b in 0..cur_dim {
                let low = f[b << 1];
                // q[b] = f[1, b] - f[0, b]
                q[b] = f[(b << 1) + 1] - low;
                // Forward traversal never overwrites a pair not yet read.
                // f[b] = f[0, b] + q[b] * coordinate
                f[b] = low + q[b] * coordinate;
            }
            f.truncate(cur_dim);
            end_timer!(eval_timer);
            // this is a MSM over G1 and is likely to be the bottleneck
            let msm_timer = start_timer!(|| format!("msm of size {} at round {}", gi.len(), i));
            proofs.push(E::G1::msm_unchecked(gi, &q).into_affine());
            end_timer!(msm_timer);
            end_timer!(round_timer);
        }
        end_timer!(open_timer);
        Ok((proofs, f[0]))
    }
}
