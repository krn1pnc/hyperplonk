// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.
//
// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Prover subroutines for a SumCheck protocol.

use super::{CpuBackend, DenseMultilinearExtension};
use crate::{BackendError, SumCheckProver, VirtualPolynomial};
use ark_ff::{batch_inversion, PrimeField};
use ark_poly::MultilinearExtension;
use ark_std::{end_timer, start_timer};
use rayon::prelude::*;
use std::sync::Arc;

/// CPU working state, accessible only to its prover implementation.
pub struct CpuSumCheckProverState<F: PrimeField> {
    /// the current round number
    round: usize,
    num_variables: usize,
    max_degree: usize,
    products: Vec<(F, Vec<usize>)>,
    flattened_ml_extensions: Vec<Arc<DenseMultilinearExtension<F>>>,
    /// points with precomputed barycentric weights for extrapolating smaller
    /// degree uni-polys to `max_degree + 1` evaluations.
    extrapolation_aux: Vec<(Vec<F>, Vec<F>)>,
}

impl<F: PrimeField> SumCheckProver<F> for CpuBackend {
    type ProverState = CpuSumCheckProverState<F>;

    /// Initialize the prover state to argue for the sum of the input polynomial
    /// over {0,1}^`num_vars`.
    fn prover_init(
        &self,
        polynomial: &VirtualPolynomial<F, Self::Mle>,
    ) -> Result<Self::ProverState, BackendError> {
        let start = start_timer!(|| "sum check prover init");
        if polynomial.aux_info.num_variables == 0 || polynomial.aux_info.max_degree == 0 {
            return Err(BackendError::InvalidParameters(
                "SumCheck requires positive variable count and degree.".to_string(),
            ));
        }
        polynomial
            .aux_info
            .max_degree
            .checked_add(1)
            .ok_or_else(|| {
                BackendError::InvalidParameters("SumCheck degree is too large.".to_string())
            })?;
        end_timer!(start);

        Ok(CpuSumCheckProverState {
            round: 0,
            num_variables: polynomial.aux_info.num_variables,
            max_degree: polynomial.aux_info.max_degree,
            products: polynomial.products.clone(),
            flattened_ml_extensions: polynomial.flattened_ml_extensions().to_vec(),
            extrapolation_aux: (1..polynomial.aux_info.max_degree)
                .map(|degree| {
                    let points = (0..1 + degree as u64).map(F::from).collect::<Vec<_>>();
                    let weights = barycentric_weights(&points);
                    (points, weights)
                })
                .collect(),
        })
    }

    /// Receive message from verifier, generate prover message, and proceed to
    /// next round.
    ///
    /// Main algorithm used is from section 3.2 of [XZZPS19](https://eprint.iacr.org/2019/317.pdf#subsection.3.2).
    fn prove_round_and_update_state(
        &self,
        state: &mut Self::ProverState,
        challenge: &Option<F>,
    ) -> Result<Vec<F>, BackendError> {
        // let start =
        //     start_timer!(|| format!("sum check prove {}-th round and update state",
        // self.round));

        if state.round >= state.num_variables {
            return Err(BackendError::InvalidState(
                "Prover is not active".to_string(),
            ));
        }

        // let fix_argument = start_timer!(|| "fix argument");

        // Step 1:
        // fix argument and evaluate f(x) over x_m = r; where r is the challenge
        // for the current round, and m is the round number, indexed from 1
        //
        // i.e.:
        // at round m <= n, for each mle g(x_1, ... x_n) within the flattened_mle
        // which has already been evaluated to
        //
        //    g(r_1, ..., r_{m-1}, x_m ... x_n)
        //
        // eval g over r_m, and mutate g to g(r_1, ... r_m,, x_{m+1}... x_n)

        if let Some(chal) = challenge {
            if state.round == 0 {
                return Err(BackendError::InvalidState(
                    "first round should be prover first.".to_string(),
                ));
            }
            // update prover's state to the partial evaluated polynomial
            state
                .flattened_ml_extensions
                .par_iter_mut()
                .for_each(|mle| *mle = Arc::new(mle.fix_variables(&[*chal])));
        } else if state.round > 0 {
            return Err(BackendError::InvalidState(
                "verifier message is empty".to_string(),
            ));
        }
        // end_timer!(fix_argument);

        state.round += 1;

        let mut products_sum = vec![F::zero(); state.max_degree + 1];

        // Step 2: generate sum for the partial evaluated polynomial:
        // f(r_1, ... r_m,, x_{m+1}... x_n)

        state.products.iter().for_each(|(coefficient, products)| {
            let mut sum = (0..(1 << (state.num_variables - state.round)))
                .into_par_iter()
                .fold(
                    || {
                        (
                            vec![(F::zero(), F::zero()); products.len()],
                            vec![F::zero(); products.len() + 1],
                        )
                    },
                    |(mut buf, mut acc), b| {
                        buf.iter_mut()
                            .zip(products.iter())
                            .for_each(|((eval, step), f)| {
                                let table = &state.flattened_ml_extensions[*f].evaluations;
                                *eval = table[b << 1];
                                *step = table[(b << 1) + 1] - table[b << 1];
                            });
                        acc[0] += buf.iter().map(|(eval, _)| eval).product::<F>();
                        acc[1..].iter_mut().for_each(|acc| {
                            buf.iter_mut().for_each(|(eval, step)| *eval += step as &_);
                            *acc += buf.iter().map(|(eval, _)| eval).product::<F>();
                        });
                        (buf, acc)
                    },
                )
                .map(|(_, partial)| partial)
                .reduce(
                    || vec![F::zero(); products.len() + 1],
                    |mut sum, partial| {
                        sum.iter_mut()
                            .zip(partial.iter())
                            .for_each(|(sum, partial)| *sum += partial);
                        sum
                    },
                );
            sum.iter_mut().for_each(|sum| *sum *= coefficient);
            let extraploation = (0..(state.max_degree - products.len()))
                .into_par_iter()
                .map(|i| {
                    let (points, weights) = &state.extrapolation_aux[products.len() - 1];
                    let at = F::from((products.len() + 1 + i) as u64);
                    extrapolate(points, weights, &sum, &at)
                })
                .collect::<Vec<_>>();
            products_sum
                .iter_mut()
                .zip(sum.iter().chain(extraploation.iter()))
                .for_each(|(products_sum, sum)| *products_sum += sum);
        });

        Ok(products_sum)
    }
}

fn barycentric_weights<F: PrimeField>(points: &[F]) -> Vec<F> {
    let mut weights = points
        .iter()
        .enumerate()
        .map(|(j, point_j)| {
            points
                .iter()
                .enumerate()
                .filter(|&(i, _point_i)| i != j)
                .map(|(_i, point_i)| *point_j - point_i)
                .reduce(|acc, value| acc * value)
                .unwrap_or_else(F::one)
        })
        .collect::<Vec<_>>();
    batch_inversion(&mut weights);
    weights
}

fn extrapolate<F: PrimeField>(points: &[F], weights: &[F], evals: &[F], at: &F) -> F {
    let (coeffs, sum_inv) = {
        let mut coeffs = points.iter().map(|point| *at - point).collect::<Vec<_>>();
        batch_inversion(&mut coeffs);
        coeffs.iter_mut().zip(weights).for_each(|(coeff, weight)| {
            *coeff *= weight;
        });
        let sum_inv = coeffs.iter().sum::<F>().inverse().unwrap_or_default();
        (coeffs, sum_inv)
    };
    coeffs
        .iter()
        .zip(evals)
        .map(|(coeff, eval)| *coeff * eval)
        .sum::<F>()
        * sum_inv
}

#[cfg(test)]
mod test {
    use super::*;
    use ark_bls12_381::Fr;

    #[test]
    fn round_lifecycle_preserves_challenge_boundaries() -> Result<(), BackendError> {
        let mle = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            2,
            [2u64, 3, 5, 7].map(Fr::from).to_vec(),
        ));
        let mut poly = VirtualPolynomial::new_from_mle(&mle, Fr::from(2u64));
        poly.add_mle_list([mle.clone(), mle.clone()], Fr::from(3u64))?;
        let mut state = CpuBackend.prover_init(&poly)?;
        let challenge = Some(Fr::from(2u64));

        assert!(matches!(
            CpuBackend.prove_round_and_update_state(&mut state, &challenge),
            Err(BackendError::InvalidState(_))
        ));
        let first = CpuBackend.prove_round_and_update_state(&mut state, &None)?;
        assert_eq!(first, [101u64, 194, 317].map(Fr::from));
        assert!(matches!(
            CpuBackend.prove_round_and_update_state(&mut state, &None),
            Err(BackendError::InvalidState(_))
        ));
        let second = CpuBackend.prove_round_and_update_state(&mut state, &challenge)?;
        assert_eq!(second, [56u64, 261, 616].map(Fr::from));
        assert!(matches!(
            CpuBackend.prove_round_and_update_state(&mut state, &challenge),
            Err(BackendError::InvalidState(_))
        ));
        assert_eq!(mle.evaluations, [2u64, 3, 5, 7].map(Fr::from));
        Ok(())
    }
}
