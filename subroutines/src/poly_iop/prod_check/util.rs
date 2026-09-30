// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! This module implements useful functions for the product check protocol.

use crate::{IOPProof, PolyIOP, PolyIOPErrors, ZeroCheck};
use ark_ff::PrimeField;
use ark_std::{end_timer, start_timer};
use backend::{SumCheckProver, VirtualPolynomial};
use transcript::IOPTranscript;

/// generate the zerocheck proof for the virtual polynomial
///    prod(x) - p1(x) * p2(x) + alpha * [frac(x) * g1(x) * ... * gk(x) - f1(x)
/// * ... * fk(x)] where p1(x) = (1-x1) * frac(x2, ..., xn, 0) + x1 * prod(x2,
///   ..., xn, 0), p2(x) = (1-x1) * frac(x2, ..., xn, 1) + x1 * prod(x2, ...,
///   xn, 1)
///
/// Returns proof.
///
/// Cost: O(N)
pub(super) fn prove_zero_check<F, B>(
    backend: &B,
    fxs: &[B::Mle],
    gxs: &[B::Mle],
    frac_poly: &B::Mle,
    product_polys: (&B::Mle, [B::Mle; 2]),
    alpha: &F,
    transcript: &mut IOPTranscript<F>,
) -> Result<IOPProof<F>, PolyIOPErrors>
where
    F: PrimeField,
    B: SumCheckProver<F>,
{
    let start = start_timer!(|| "zerocheck in product check");
    let (prod_x, product_factors) = product_polys;

    // compute Q(x)
    // prod(x)
    let mut q_x = VirtualPolynomial::new_from_mle(prod_x, F::one());

    //   prod(x)
    // - p1(x) * p2(x)
    q_x.add_mle_list(product_factors, -F::one())?;

    //   prod(x)
    // - p1(x) * p2(x)
    // + alpha * frac(x) * g1(x) * ... * gk(x)
    let mut mle_list = gxs.to_vec();
    mle_list.push(frac_poly.clone());
    q_x.add_mle_list(mle_list, *alpha)?;

    //   prod(x)
    // - p1(x) * p2(x)
    // + alpha * frac(x) * g1(x) * ... * gk(x)
    // - alpha * f1(x) * ... * fk(x)]
    q_x.add_mle_list(fxs.to_vec(), -*alpha)?;

    let iop_proof = <PolyIOP<F> as ZeroCheck<F>>::prove(backend, &q_x, transcript)?;

    end_timer!(start);
    Ok(iop_proof)
}
