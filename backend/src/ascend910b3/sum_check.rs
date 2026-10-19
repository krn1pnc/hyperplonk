//! Numerical SumCheck state forwarding without mirroring native round state.
//!
//! Product descriptions preserve factor order and repeated multiplicands. Their
//! host arrays are borrowed only during initialization, never device layouts.

use super::{
    context::Ascend910B3Context,
    encoding::{decode_fr, encode_fr},
    ffi,
    utils::{byte_count, collect_output, evaluation_count, input_pointer},
    Ascend910B3Backend,
};
use crate::SumCheckProver;
use crate::{BackendError, Mle, VirtualPolynomial};
use ark_bls12_381::Fr;
use std::{
    ptr::{self, NonNull},
    rc::Rc,
};

struct Ascend910B3ProductDescription {
    coefficients: Vec<u8>,
    offsets: Vec<usize>,
    indices: Vec<usize>,
}

/// Validate publicly mutable VP metadata before exposing its indices to native.
///
/// Empty whole sums are valid with positive dimension and degree; empty products
/// are not. `offsets` always contains at least its initial zero, even when the
/// coefficients and indices arrays must be passed as null to native.
fn encode_products<P: Mle<Fr>>(
    poly: &VirtualPolynomial<Fr, P>,
) -> Result<Ascend910B3ProductDescription, BackendError> {
    let num_vars = poly.aux_info.num_variables;
    let max_degree = poly.aux_info.max_degree;
    if num_vars == 0 || max_degree == 0 {
        return Err(BackendError::InvalidParameters(
            "SumCheck requires positive variable count and maximum degree".into(),
        ));
    }
    byte_count(evaluation_count(num_vars)?, 32)?;
    let round_count = max_degree.checked_add(1).ok_or_else(|| {
        BackendError::InvalidParameters("SumCheck round evaluation count overflow".into())
    })?;
    byte_count(round_count, 32)?;

    let factors = poly.flattened_ml_extensions();
    if factors.iter().any(|factor| factor.num_vars() != num_vars) {
        return Err(BackendError::InvalidParameters(
            "SumCheck factor dimensions do not match virtual polynomial metadata".into(),
        ));
    }
    // Each existing product contains an Fr and a Vec, bounding these capacities.
    let coefficient_bytes = poly.products.len() * 32;
    let offset_count = poly.products.len() + 1;
    let mut index_count = 0usize;
    for (_, indices) in &poly.products {
        if indices.is_empty() || indices.len() > max_degree {
            return Err(BackendError::InvalidParameters(
                "SumCheck product must have between one and max_degree multiplicands".into(),
            ));
        }
        if indices.iter().any(|&index| index >= factors.len()) {
            return Err(BackendError::InvalidParameters(
                "SumCheck product references an out-of-range factor".into(),
            ));
        }
        index_count += indices.len();
    }

    let mut coefficients = Vec::with_capacity(coefficient_bytes);
    let mut offsets = Vec::with_capacity(offset_count);
    let mut indices = Vec::with_capacity(index_count);
    offsets.push(0);
    for (coefficient, product) in &poly.products {
        coefficients.extend_from_slice(&encode_fr(coefficient));
        indices.extend_from_slice(product);
        offsets.push(indices.len());
    }
    Ok(Ascend910B3ProductDescription {
        coefficients,
        offsets,
        indices,
    })
}

/// Independently advancing opaque SumCheck prover state retaining its context.
///
/// Native state owns the numerical-input lifetime; input MLE handles and virtual
/// polynomials need not survive initialization. Host round outputs are ready on
/// return. This uniquely owned handle is neither Send nor Sync.
pub struct Ascend910B3SumCheckProverState {
    pub(crate) raw: NonNull<ffi::Ascend910B3NativeSumCheckProverState>,
    pub(crate) context: Rc<Ascend910B3Context>,
    pub(crate) round_evaluation_count: usize,
}

impl SumCheckProver<Fr> for Ascend910B3Backend {
    type ProverState = Ascend910B3SumCheckProverState;

    fn prover_init(
        &self,
        polynomial: &VirtualPolynomial<Fr, Self::Mle>,
    ) -> Result<Self::ProverState, BackendError> {
        let products = encode_products(polynomial)?;
        let factors = polynomial.flattened_ml_extensions();
        // encode_products already validated this count and its output packet.
        let round_evaluation_count = polynomial.aux_info.max_degree + 1;
        let mut addresses = Vec::with_capacity(factors.len());
        for factor in factors {
            self.check_mle(factor)?;
            addresses.push(factor.allocation.raw.as_ptr());
        }
        let mut raw = ptr::null_mut();
        // SAFETY: validated product offsets/indices preserve repeated factors
        // and first-insertion address order. Every factor belongs to this live
        // context. Host arrays are borrowed only until return; successful state
        // independently owns all numerical lifetimes needed by later rounds.
        // Offsets always include the initial 0, even for zero products; other
        // zero-count arrays are null. Failure cleans and nulls partial state.
        unsafe {
            ffi::hp_ascend910b3_v1_sumcheck_prover_state_create(
                self.context.raw.as_ptr(),
                polynomial.aux_info.num_variables,
                polynomial.aux_info.max_degree,
                input_pointer(&addresses),
                addresses.len(),
                input_pointer(&products.coefficients),
                products.offsets.as_ptr(),
                polynomial.products.len(),
                input_pointer(&products.indices),
                products.indices.len(),
                &mut raw,
            )
            .into_result("sumcheck_prover_state_create")?;
        }
        let raw = NonNull::new(raw).unwrap_or_else(|| std::process::abort());
        Ok(Ascend910B3SumCheckProverState {
            raw,
            context: Rc::clone(&self.context),
            round_evaluation_count,
        })
    }

    fn prove_round_and_update_state(
        &self,
        state: &mut Self::ProverState,
        challenge: &Option<Fr>,
    ) -> Result<Vec<Fr>, BackendError> {
        self.check_context(&state.context)?;
        let challenge = challenge.as_ref().map(encode_fr);
        let challenge_pointer = challenge
            .as_ref()
            .map_or(ptr::null(), |value| value.as_ptr());
        let output_length = byte_count(state.round_evaluation_count, 32)?;
        let bytes = collect_output("sumcheck_prover_state_round", output_length, |out| {
            // SAFETY: unique mutable state access uses its owning context;
            // optional canonical challenge is live for this call and the output
            // has exactly the cached round count. Native enforces None/Some and
            // exhaustion without advancing on validation errors. Success writes
            // every host-ready packet; errors never expose uninitialized bytes.
            unsafe {
                ffi::hp_ascend910b3_v1_sumcheck_prover_state_round(
                    self.context.raw.as_ptr(),
                    state.raw.as_ptr(),
                    challenge_pointer,
                    out,
                    state.round_evaluation_count,
                )
            }
        })?;
        bytes.chunks_exact(32).map(decode_fr).collect()
    }
}

impl Drop for Ascend910B3SumCheckProverState {
    fn drop(&mut self) {
        // SAFETY: this uniquely owned state and its retained context remain live
        // for destruction. Native code must preserve any queued numerical uses.
        let status = unsafe {
            ffi::hp_ascend910b3_v1_sumcheck_prover_state_destroy(
                self.context.raw.as_ptr(),
                self.raw.as_ptr(),
            )
        };
        if status.into_result("sumcheck_prover_state_destroy").is_err() {
            std::process::abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ascend910b3::encoding::decode_fr;
    use ark_ff::{AdditiveGroup, Field};

    /// Host metadata only: this neither allocates native resources nor stores
    /// polynomial evaluations or emulates any numerical kernel.
    #[derive(Clone)]
    struct Ascend910B3MetadataHandle {
        identity: usize,
        num_vars: usize,
    }

    impl Mle<Fr> for Ascend910B3MetadataHandle {
        type Id = usize;

        fn num_vars(&self) -> usize {
            self.num_vars
        }

        fn id(&self) -> Self::Id {
            self.identity
        }
    }

    fn polynomial() -> VirtualPolynomial<Fr, Ascend910B3MetadataHandle> {
        let a = Ascend910B3MetadataHandle {
            identity: 11,
            num_vars: 2,
        };
        let b = Ascend910B3MetadataHandle {
            identity: 22,
            num_vars: 2,
        };
        let mut poly = VirtualPolynomial::new(2);
        poly.add_mle_list([a.clone(), a, b.clone()], Fr::from(3u64))
            .unwrap();
        poly.add_mle_list([b], Fr::from(5u64)).unwrap();
        poly
    }

    /// Interpret the actual production packet at two factor assignments; this
    /// catches missing powers and incorrect coefficient/index/offset order.
    fn evaluate_description(description: &Ascend910B3ProductDescription, factors: &[Fr]) -> Fr {
        let coefficients: Vec<_> = description
            .coefficients
            .chunks_exact(32)
            .map(|bytes| decode_fr(bytes).unwrap())
            .collect();
        assert_eq!(description.coefficients.len(), coefficients.len() * 32);
        assert_eq!(description.offsets.len(), coefficients.len() + 1);
        assert_eq!(description.offsets.first(), Some(&0));
        assert_eq!(description.offsets.last(), Some(&description.indices.len()));
        coefficients
            .iter()
            .zip(description.offsets.windows(2))
            .fold(Fr::ZERO, |sum, (coefficient, bounds)| {
                let product = description.indices[bounds[0]..bounds[1]]
                    .iter()
                    .fold(Fr::ONE, |product, &index| product * factors[index]);
                sum + *coefficient * product
            })
    }

    fn assert_invalid(result: Result<Ascend910B3ProductDescription, BackendError>) {
        assert!(matches!(result, Err(BackendError::InvalidParameters(_))));
    }

    #[test]
    fn products_preserve_semantics_order_and_repeated_factors() {
        let poly = polynomial();
        let handles = poly.flattened_ml_extensions();
        assert_eq!(handles.len(), 2);
        assert_eq!(handles[0].id(), 11);
        assert_eq!(handles[1].id(), 22);
        let description = encode_products(&poly).unwrap();
        assert_eq!(description.offsets, [0, 3, 4]);
        assert_eq!(description.indices, [0, 0, 1, 1]);
        assert_eq!(
            evaluate_description(&description, &[Fr::from(2u64), Fr::from(7u64)]),
            Fr::from(119u64)
        );
        assert_eq!(
            evaluate_description(&description, &[Fr::from(5u64), Fr::from(11u64)]),
            Fr::from(880u64)
        );
    }

    #[test]
    fn malformed_public_products_and_metadata_are_rejected() {
        let mut bad_index = polynomial();
        bad_index.products[0].1[0] = 2;
        assert_invalid(encode_products(&bad_index));

        let mut empty_product = polynomial();
        empty_product.products[0].1.clear();
        assert_invalid(encode_products(&empty_product));

        let mut small_degree = polynomial();
        small_degree.aux_info.max_degree = 2;
        assert_invalid(encode_products(&small_degree));

        let mut wrong_dimension = polynomial();
        wrong_dimension.aux_info.num_variables = 3;
        assert_invalid(encode_products(&wrong_dimension));

        let mut zero_dimension = polynomial();
        zero_dimension.aux_info.num_variables = 0;
        assert_invalid(encode_products(&zero_dimension));

        let mut zero_degree = polynomial();
        zero_degree.aux_info.max_degree = 0;
        assert_invalid(encode_products(&zero_degree));

        let mut overflowing_degree = polynomial();
        overflowing_degree.aux_info.max_degree = usize::MAX;
        assert_invalid(encode_products(&overflowing_degree));
    }

    #[test]
    fn empty_sum_with_positive_degree_has_zero_semantics() {
        let mut poly = VirtualPolynomial::<Fr, Ascend910B3MetadataHandle>::new(2);
        poly.aux_info.max_degree = 3;
        let description = encode_products(&poly).unwrap();
        assert!(description.coefficients.is_empty());
        assert!(description.indices.is_empty());
        assert_eq!(description.offsets, [0]);
        for values in [[2u64, 7u64], [5, 11]] {
            assert_eq!(
                evaluate_description(&description, &values.map(Fr::from)),
                Fr::ZERO
            );
        }
    }
}
