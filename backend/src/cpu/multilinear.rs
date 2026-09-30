// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.
//
// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Dense CPU identity permutations, merging, and high-bit suffix folding.

use super::DenseMultilinearExtension;
use crate::{common::indexing::get_batched_nv, BackendError};
use ark_ff::{Field, PrimeField};
use ark_poly::MultilinearExtension;
use rayon::prelude::*;
use std::sync::Arc;

/// A list of MLEs that represents an identity permutation
pub fn identity_permutation_mles<F: PrimeField>(
    num_vars: usize,
    num_chunks: usize,
) -> Vec<Arc<DenseMultilinearExtension<F>>> {
    let mut res = vec![];
    for i in 0..num_chunks {
        let shift = (i * (1 << num_vars)) as u64;
        let s_id_vec = (shift..shift + (1u64 << num_vars)).map(F::from).collect();
        res.push(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars, s_id_vec,
        )));
    }
    res
}

/// merge a set of polynomials. Returns an error if the
/// polynomials do not share a same number of nvs.
pub fn merge_polynomials<F: PrimeField>(
    polynomials: &[Arc<DenseMultilinearExtension<F>>],
) -> Result<Arc<DenseMultilinearExtension<F>>, BackendError> {
    let nv = polynomials[0].num_vars();
    for poly in polynomials.iter() {
        if nv != poly.num_vars() {
            return Err(BackendError::InvalidParameters(
                "num_vars do not match for polynomials".to_string(),
            ));
        }
    }

    let merged_nv = get_batched_nv(nv, polynomials.len());
    let mut scalars = Vec::with_capacity(1 << merged_nv);
    for poly in polynomials.iter() {
        scalars.extend_from_slice(&poly.evaluations);
    }
    scalars.resize(1 << merged_nv, F::zero());
    Ok(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
        merged_nv, scalars,
    )))
}

/// Fix a high-bit suffix without changing the input polynomial.
pub fn fix_last_variables<F: PrimeField>(
    poly: &DenseMultilinearExtension<F>,
    partial_point: &[F],
) -> DenseMultilinearExtension<F> {
    let evaluations =
        fix_last_variables_evaluations(poly.num_vars, &poly.evaluations, partial_point);
    DenseMultilinearExtension::from_evaluations_vec(
        poly.num_vars - partial_point.len(),
        evaluations,
    )
}

/// Fix the high-bit suffix in reverse coordinate order.
/// Requires a valid 2^num_vars table and no more than num_vars coordinates.
fn fix_last_variables_evaluations<F: Field>(
    num_vars: usize,
    evaluations: &[F],
    partial_point: &[F],
) -> Vec<F> {
    assert!(
        partial_point.len() <= num_vars,
        "invalid size of partial point"
    );
    let mut points = partial_point.iter().rev();
    let Some(first) = points.next() else {
        return evaluations.to_vec();
    };
    let mut result = fix_last_variable_helper(evaluations, num_vars, first);
    for (i, point) in points.enumerate() {
        result = fix_last_variable_helper(&result, num_vars - i - 1, point);
    }
    result
}

fn fix_last_variable_helper<F: Field>(data: &[F], nv: usize, point: &F) -> Vec<F> {
    let half_len = 1 << (nv - 1);
    let mut res = vec![F::zero(); half_len];

    // evaluate single variable of partial point from left to right
    res.par_iter_mut().enumerate().for_each(|(i, x)| {
        *x = data[i] + (data[i + half_len] - data[i]) * point;
    });

    res
}
