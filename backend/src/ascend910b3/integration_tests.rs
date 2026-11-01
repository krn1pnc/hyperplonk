//! Hardware integration tests for the Ascend910B3 backend.
//!
//! Every test needs a real device: with the `ascend910b3` feature enabled this
//! module compiles only against the linked native bridge, and an unavailable
//! device is a real failure rather than a skip. The oracle is [`CpuBackend`]
//! over the same BLS12-381 `Fr` inputs, plus closed-form host computations for
//! each operation; comparisons are exact because finite-field arithmetic has no
//! dtype tolerance. Polynomial outputs are checked through scalar
//! `evaluate_mle` calls at hypercube corners and non-Boolean points, so no
//! device-to-host table download is needed.
//!
//! ACL device IDs are logical indices into the visible device set. A host that
//! exposes only physical card 6 (for example `ASCEND_RT_VISIBLE_DEVICES=6`)
//! therefore selects it as logical device 0; the tests never hardcode a
//! physical card. `HYPERPLONK_ASCEND910B3_TEST_DEVICE` may name another logical
//! index.
//!
//! ACL runtime lifecycle is process-global even though contexts are independent,
//! so device tests run under one serialized lock. The lock avoids cross-test
//! interference; inside a test, simultaneous independent contexts are still
//! exercised deliberately.

use super::{
    Ascend910B3Backend, Ascend910B3Mle, Ascend910B3MultilinearKzgPreparedProverParam,
};
use crate::cpu::CpuBackend;
use crate::{
    BackendError, Mle, MleBackend, MultilinearKzgBackend, SumCheckProver, VirtualPolynomial,
};
use ark_bls12_381::{Bls12_381, Fq, Fr, G1Affine, G1Projective};
use ark_ec::{scalar_mul::variable_base::VariableBaseMSM, CurveGroup, PrimeGroup};
use ark_ff::{AdditiveGroup, Field, PrimeField};
use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};

/// CPU oracle polynomial type.
type CpuMle = <CpuBackend as MleBackend<Fr>>::Mle;
/// CPU oracle prepared KZG parameters.
type CpuKzg = <CpuBackend as MultilinearKzgBackend<Bls12_381>>::PreparedProverParam;
type TestResult = Result<(), BackendError>;

/// Serializes device tests around the process-global ACL runtime lifecycle.
static DEVICE_LOCK: Mutex<()> = Mutex::new(());

fn device_guard() -> MutexGuard<'static, ()> {
    DEVICE_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn test_device_id() -> i32 {
    match std::env::var("HYPERPLONK_ASCEND910B3_TEST_DEVICE") {
        Ok(value) => value.parse().unwrap_or_else(|error| {
            panic!(
                "HYPERPLONK_ASCEND910B3_TEST_DEVICE must be a logical ACL device index, \
                 got {value:?}: {error}"
            )
        }),
        Err(std::env::VarError::NotPresent) => 0,
        Err(error) => panic!("cannot read HYPERPLONK_ASCEND910B3_TEST_DEVICE: {error}"),
    }
}

fn ascend_backend() -> Ascend910B3Backend {
    let device = test_device_id();
    Ascend910B3Backend::new(device).unwrap_or_else(|error| {
        panic!(
            "Ascend910B3 backend is required for these hardware integration tests; \
             logical device {device} failed: {error:?}"
        )
    })
}

/// Deterministic PRNG; only used to derive reproducible full-field inputs.
struct TestRng(u64);

impl TestRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    /// Reduce 32 pseudo-random bytes into the scalar field, so draws cover the
    /// whole canonical range rather than only small integers.
    fn fr(&mut self) -> Fr {
        let mut bytes = [0u8; 32];
        for chunk in bytes.chunks_exact_mut(8) {
            chunk.copy_from_slice(&self.next_u64().to_le_bytes());
        }
        Fr::from_le_bytes_mod_order(&bytes)
    }
}

/// Low-bit-first evaluation table with deterministic full-field values.
///
/// Boundaries at both ends make tail handling observable: index 0 is zero and
/// the final entry is `p - 1` whenever the table has at least two entries.
fn field_table(num_vars: usize, seed: u64) -> Vec<Fr> {
    let length = 1usize << num_vars;
    let mut rng = TestRng::new(seed ^ (num_vars as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
    let mut values: Vec<Fr> = (0..length).map(|_| rng.fr()).collect();
    if length > 1 {
        values[0] = Fr::ZERO;
        values[length - 1] = -Fr::ONE;
        if length > 3 {
            values[1] = Fr::ONE;
        }
    }
    values
}

/// Same distribution with every zero replaced, for denominators.
fn nonzero_table(num_vars: usize, seed: u64) -> Vec<Fr> {
    field_table(num_vars, seed)
        .into_iter()
        .map(|value| if value == Fr::ZERO { Fr::ONE } else { value })
        .collect()
}

fn boolean_point(num_vars: usize, index: usize) -> Vec<Fr> {
    (0..num_vars)
        .map(|bit| {
            if (index >> bit) & 1 == 1 {
                Fr::ONE
            } else {
                Fr::ZERO
            }
        })
        .collect()
}

/// Hypercube corners (all corners for small dimensions) and non-Boolean points.
fn comparison_points(num_vars: usize, seed: u64) -> Vec<Vec<Fr>> {
    if num_vars == 0 {
        return vec![Vec::new()];
    }
    let mut points = Vec::new();
    if num_vars <= 3 {
        points.extend((0..(1usize << num_vars)).map(|index| boolean_point(num_vars, index)));
    } else {
        let last = (1usize << num_vars) - 1;
        for index in [0, 1, last >> 1, last - 1, last] {
            points.push(boolean_point(num_vars, index));
        }
    }
    let mut rng = TestRng::new(seed ^ 0x5DEE_CE66_D);
    for _ in 0..3 {
        points.push((0..num_vars).map(|_| rng.fr()).collect());
    }
    points.push(
        (0..num_vars)
            .map(|bit| if bit % 2 == 0 { -Fr::ONE } else { Fr::ONE })
            .collect(),
    );
    points
}

/// Multilinear extension evaluation in low-bit-first table order.
fn multilinear_eval(table: &[Fr], point: &[Fr]) -> Fr {
    assert_eq!(table.len(), 1usize << point.len());
    let mut values = table.to_vec();
    for &coordinate in point {
        let half = values.len() / 2;
        for index in 0..half {
            let low = values[index << 1];
            let high = values[(index << 1) + 1];
            values[index] = low + coordinate * (high - low);
        }
        values.truncate(half);
    }
    values[0]
}

fn cpu_mle(num_vars: usize, values: &[Fr]) -> CpuMle {
    CpuBackend
        .mle_from_evaluations(num_vars, values.to_vec())
        .expect("CPU MLE upload")
}

fn npu_mle(backend: &Ascend910B3Backend, num_vars: usize, values: &[Fr]) -> Ascend910B3Mle {
    backend
        .mle_from_evaluations(num_vars, values.to_vec())
        .expect("Ascend MLE upload")
}

/// The two backends and an independent host table must agree exactly.
fn assert_mle_matches(
    backend: &Ascend910B3Backend,
    actual: &Ascend910B3Mle,
    cpu: &CpuMle,
    expected: &[Fr],
    points: &[Vec<Fr>],
) {
    assert_eq!(actual.num_vars(), cpu.num_vars());
    assert_eq!(expected.len(), 1usize << actual.num_vars());
    for point in points {
        let expected_value = multilinear_eval(expected, point);
        let cpu_value = CpuBackend
            .evaluate_mle(cpu, point)
            .expect("CPU evaluation");
        let actual_value = backend
            .evaluate_mle(actual, point)
            .expect("Ascend evaluation");
        assert_eq!(cpu_value, expected_value, "CPU disagrees at {point:?}");
        assert_eq!(actual_value, expected_value, "Ascend disagrees at {point:?}");
    }
}

fn assert_invalid_parameters<T>(result: Result<T, BackendError>) {
    match result {
        Err(BackendError::InvalidParameters(_)) => {}
        Err(other) => panic!("expected InvalidParameters, got {other:?}"),
        Ok(_) => panic!("expected InvalidParameters, got Ok"),
    }
}

fn assert_invalid_state<T>(result: Result<T, BackendError>) {
    match result {
        Err(BackendError::InvalidState(_)) => {}
        Err(other) => panic!("expected InvalidState, got {other:?}"),
        Ok(_) => panic!("expected InvalidState, got Ok"),
    }
}

fn build_virtual_polynomial<P: Mle<Fr> + Clone>(
    num_vars: usize,
    products: &[(Fr, Vec<usize>)],
    factors: &[P],
) -> Result<VirtualPolynomial<Fr, P>, BackendError> {
    let mut polynomial = VirtualPolynomial::new(num_vars);
    for (coefficient, indices) in products {
        polynomial.add_mle_list(
            indices.iter().map(|&index| factors[index].clone()),
            *coefficient,
        )?;
    }
    Ok(polynomial)
}

/// Materialize host tables in the virtual polynomial's first-insertion order,
/// so product indices can be interpreted independently of handle creation order.
fn flattened_factor_tables<P: Mle<Fr> + Clone>(
    polynomial: &VirtualPolynomial<Fr, P>,
    factors: &[P],
    tables: &[Vec<Fr>],
) -> Vec<Vec<Fr>> {
    let mut by_identity = HashMap::new();
    for (factor, table) in factors.iter().zip(tables) {
        by_identity.insert(factor.id(), table.clone());
    }
    polynomial
        .flattened_ml_extensions()
        .iter()
        .map(|factor| {
            by_identity
                .get(&factor.id())
                .expect("flattened factor was supplied by the test")
                .clone()
        })
        .collect()
}

/// Run exactly `num_vars` rounds; the final challenge is host-side and unused.
fn sumcheck_transcript<B: SumCheckProver<Fr>>(
    backend: &B,
    state: &mut B::ProverState,
    num_vars: usize,
    challenges: &[Fr],
) -> Result<Vec<Vec<Fr>>, BackendError> {
    assert_eq!(challenges.len(), num_vars);
    let mut messages = Vec::with_capacity(num_vars);
    for round in 0..num_vars {
        let challenge = if round == 0 {
            None
        } else {
            Some(challenges[round - 1])
        };
        messages.push(backend.prove_round_and_update_state(state, &challenge)?);
    }
    Ok(messages)
}

/// Lagrange interpolation of a univariate message given at `0..=degree`.
fn eval_univariate(evaluations: &[Fr], at: Fr) -> Fr {
    let mut result = Fr::ZERO;
    for (index, &value) in evaluations.iter().enumerate() {
        let mut numerator = Fr::ONE;
        let mut denominator = Fr::ONE;
        for other in 0..evaluations.len() {
            if other == index {
                continue;
            }
            numerator *= at - Fr::from(other as u64);
            denominator *= Fr::from(index as u64) - Fr::from(other as u64);
        }
        result += value
            * numerator
            * denominator
                .inverse()
                .expect("interpolation points are distinct");
    }
    result
}

/// Virtual polynomial value at an arbitrary point.
fn virtual_value(products: &[(Fr, Vec<usize>)], tables: &[Vec<Fr>], point: &[Fr]) -> Fr {
    products
        .iter()
        .map(|(coefficient, indices)| {
            let factor = indices
                .iter()
                .map(|&index| multilinear_eval(&tables[index], point))
                .product::<Fr>();
            *coefficient * factor
        })
        .sum()
}

/// Total over the Boolean hypercube, read directly from the tables.
fn cube_sum(products: &[(Fr, Vec<usize>)], tables: &[Vec<Fr>], num_vars: usize) -> Fr {
    let mut total = Fr::ZERO;
    for index in 0..(1usize << num_vars) {
        let mut value = Fr::ZERO;
        for (coefficient, indices) in products {
            let mut factor = Fr::ONE;
            for &table in indices {
                factor *= tables[table][index];
            }
            value += *coefficient * factor;
        }
        total += value;
    }
    total
}

/// Independent checks for one full sumcheck transcript: message length, sum
/// consistency across rounds, and the final claim at the host challenge.
fn assert_sumcheck_messages(
    products: &[(Fr, Vec<usize>)],
    tables: &[Vec<Fr>],
    num_vars: usize,
    max_degree: usize,
    challenges: &[Fr],
    messages: &[Vec<Fr>],
) {
    assert_eq!(messages.len(), num_vars);
    for message in messages {
        assert_eq!(message.len(), max_degree + 1);
    }
    assert_eq!(
        messages[0][0] + messages[0][1],
        cube_sum(products, tables, num_vars),
        "first round does not sum to the claimed total"
    );
    for round in 1..num_vars {
        assert_eq!(
            messages[round][0] + messages[round][1],
            eval_univariate(&messages[round - 1], challenges[round - 1]),
            "round {round} does not match the previous challenge"
        );
    }
    assert_eq!(
        eval_univariate(&messages[num_vars - 1], challenges[num_vars - 1]),
        virtual_value(products, tables, challenges),
        "final claim does not match the virtual polynomial"
    );
}

fn srs_layers(num_vars: usize, seed: u64) -> Vec<Vec<G1Affine>> {
    let mut rng = TestRng::new(seed);
    (0..=num_vars)
        .map(|layer| {
            let count = 1usize << (num_vars - layer);
            (0..count)
                .map(|_| (G1Projective::generator() * rng.fr()).into_affine())
                .collect()
        })
        .collect()
}

fn cpu_prepared(num_vars: usize, layers: &[Vec<G1Affine>]) -> Result<CpuKzg, BackendError> {
    <CpuBackend as MultilinearKzgBackend<Bls12_381>>::prepare_prover_param(
        &CpuBackend,
        num_vars,
        layers.to_vec().into_iter(),
    )
}

fn ascend_prepared(
    backend: &Ascend910B3Backend,
    num_vars: usize,
    layers: &[Vec<G1Affine>],
) -> Result<Ascend910B3MultilinearKzgPreparedProverParam, BackendError> {
    <Ascend910B3Backend as MultilinearKzgBackend<Bls12_381>>::prepare_prover_param(
        backend,
        num_vars,
        layers.to_vec().into_iter(),
    )
}

fn cpu_commit(params: &CpuKzg, polynomial: &CpuMle) -> Result<G1Affine, BackendError> {
    <CpuBackend as MultilinearKzgBackend<Bls12_381>>::commit(&CpuBackend, params, polynomial)
}

fn ascend_commit(
    backend: &Ascend910B3Backend,
    params: &Ascend910B3MultilinearKzgPreparedProverParam,
    polynomial: &Ascend910B3Mle,
) -> Result<G1Affine, BackendError> {
    <Ascend910B3Backend as MultilinearKzgBackend<Bls12_381>>::commit(
        backend, params, polynomial,
    )
}

fn cpu_open(
    params: &CpuKzg,
    polynomial: &CpuMle,
    point: &[Fr],
) -> Result<(Vec<G1Affine>, Fr), BackendError> {
    <CpuBackend as MultilinearKzgBackend<Bls12_381>>::open(
        &CpuBackend, params, polynomial, point,
    )
}

fn ascend_open(
    backend: &Ascend910B3Backend,
    params: &Ascend910B3MultilinearKzgPreparedProverParam,
    polynomial: &Ascend910B3Mle,
    point: &[Fr],
) -> Result<(Vec<G1Affine>, Fr), BackendError> {
    <Ascend910B3Backend as MultilinearKzgBackend<Bls12_381>>::open(
        backend, params, polynomial, point,
    )
}

fn independent_commitment(
    layers: &[Vec<G1Affine>],
    param_num_vars: usize,
    num_vars: usize,
    values: &[Fr],
) -> G1Affine {
    let bases = &layers[param_num_vars - num_vars];
    assert_eq!(bases.len(), values.len());
    G1Projective::msm_unchecked(bases, values).into_affine()
}

/// Safe Ark construction of an on-curve point outside the prime-order subgroup,
/// used to exercise SRS ingress subgroup validation.
fn on_curve_non_subgroup_point() -> G1Affine {
    for candidate in 1u64..4096 {
        let x = Fq::from(candidate);
        if let Some(point) = G1Affine::get_point_from_x_unchecked(x, true) {
            if !point.is_in_correct_subgroup_assuming_on_curve() {
                return point;
            }
        }
    }
    panic!("no on-curve non-subgroup point found in the searched x range");
}

#[test]
fn linear_combination_and_evaluate_match_cpu_for_all_dimensions() -> TestResult {
    let _guard = device_guard();
    let backend = ascend_backend();

    for num_vars in 0..=8 {
        let inputs = [
            field_table(num_vars, 0xA11CE),
            field_table(num_vars, 0xB0B),
            field_table(num_vars, 0xC0FFEE),
        ];
        let coefficients = [Fr::ZERO, -Fr::ONE, Fr::from(0xDEAD_BEEF_u64)];
        let length = 1usize << num_vars;
        let mut expected = vec![Fr::ZERO; length];
        for (input, coefficient) in inputs.iter().zip(coefficients) {
            for (value, entry) in expected.iter_mut().zip(input) {
                *value += coefficient * entry;
            }
        }

        let cpu_handles: Vec<CpuMle> = inputs
            .iter()
            .map(|values| cpu_mle(num_vars, values))
            .collect();
        let ascend_handles: Vec<Ascend910B3Mle> = inputs
            .iter()
            .map(|values| npu_mle(&backend, num_vars, values))
            .collect();
        let cpu_result = CpuBackend.linear_combination(&cpu_handles, &coefficients)?;
        let ascend_result = backend.linear_combination(&ascend_handles, &coefficients)?;
        assert_mle_matches(
            &backend,
            &ascend_result,
            &cpu_result,
            &expected,
            &comparison_points(num_vars, 0x51),
        );
        if num_vars == 0 {
            assert_eq!(backend.evaluate_mle(&ascend_result, &[])?, expected[0]);
        }

        // Even a zero combination keeps the variable-count metadata.
        let zero = backend.linear_combination(
            &ascend_handles,
            &[Fr::ZERO, Fr::ZERO, Fr::ZERO],
        )?;
        assert_eq!(zero.num_vars(), num_vars);
        assert_eq!(backend.evaluate_mle(&zero, &comparison_points(num_vars, 0x52)[0])?, Fr::ZERO);

        // Inputs stay immutable when used through both backends.
        for (input, handle) in inputs.iter().zip(&ascend_handles) {
            assert_mle_matches(
                &backend,
                handle,
                &cpu_mle(num_vars, input),
                input,
                &comparison_points(num_vars, 0x53),
            );
        }
    }

    let one_values = field_table(1, 1);
    let two_values = field_table(2, 2);
    let one = npu_mle(&backend, 1, &one_values);
    let two = npu_mle(&backend, 2, &two_values);
    assert_invalid_parameters(backend.linear_combination(&[one.clone()], &[]));
    assert_invalid_parameters(backend.linear_combination(
        &[one.clone(), two],
        &[Fr::ONE, Fr::ONE],
    ));
    assert_invalid_parameters(backend.linear_combination(&[], &[]));
    assert_invalid_parameters(backend.evaluate_mle(&one, &[]));

    let cpu_one = cpu_mle(1, &one_values);
    let cpu_two = cpu_mle(2, &two_values);
    assert_invalid_parameters(CpuBackend.linear_combination(&[cpu_one.clone()], &[]));
    assert_invalid_parameters(CpuBackend.linear_combination(
        &[cpu_one.clone(), cpu_two],
        &[Fr::ONE, Fr::ONE],
    ));
    assert_invalid_parameters(<CpuBackend as MleBackend<Fr>>::linear_combination(
        &CpuBackend,
        &[],
        &[],
    ));
    assert_invalid_parameters(CpuBackend.evaluate_mle(&cpu_one, &[]));
    Ok(())
}

#[test]
fn eq_polynomial_matches_closed_form_and_cpu() -> TestResult {
    let _guard = device_guard();
    let backend = ascend_backend();

    for num_vars in 1..=8 {
        let mut rng = TestRng::new(0xE9 ^ num_vars as u64);
        let mut r: Vec<Fr> = (0..num_vars).map(|_| rng.fr()).collect();
        r[0] = Fr::ZERO;
        r[num_vars - 1] = -Fr::ONE;

        let expected: Vec<Fr> = (0..(1usize << num_vars))
            .map(|index| {
                (0..num_vars)
                    .map(|bit| {
                        if (index >> bit) & 1 == 1 {
                            r[bit]
                        } else {
                            Fr::ONE - r[bit]
                        }
                    })
                    .product::<Fr>()
            })
            .collect();

        let cpu_result = CpuBackend.build_eq_x_r(&r)?;
        let ascend_result = backend.build_eq_x_r(&r)?;
        assert_mle_matches(
            &backend,
            &ascend_result,
            &cpu_result,
            &expected,
            &comparison_points(num_vars, 0xEA),
        );
    }

    assert_invalid_parameters(backend.build_eq_x_r(&[]));
    assert_invalid_parameters(<CpuBackend as MleBackend<Fr>>::build_eq_x_r(
        &CpuBackend, &[],
    ));
    Ok(())
}

#[test]
fn fraction_polynomial_matches_cpu_and_rejects_zero_denominators() -> TestResult {
    let _guard = device_guard();
    let backend = ascend_backend();

    for num_vars in 0..=8 {
        let numerator_values = [
            field_table(num_vars, 0xF1),
            field_table(num_vars, 0xF2),
            field_table(num_vars, 0xF3),
        ];
        let denominator_values = [
            nonzero_table(num_vars, 0xD1),
            nonzero_table(num_vars, 0xD2),
            nonzero_table(num_vars, 0xD3),
        ];
        let length = 1usize << num_vars;
        let mut expected = vec![Fr::ZERO; length];
        for index in 0..length {
            let numerator = numerator_values
                .iter()
                .map(|values| values[index])
                .product::<Fr>();
            let denominator = denominator_values
                .iter()
                .map(|values| values[index])
                .product::<Fr>();
            expected[index] = numerator
                * denominator
                    .inverse()
                    .expect("constructed denominator is nonzero");
        }

        let cpu_numerators: Vec<CpuMle> = numerator_values
            .iter()
            .map(|values| cpu_mle(num_vars, values))
            .collect();
        let cpu_denominators: Vec<CpuMle> = denominator_values
            .iter()
            .map(|values| cpu_mle(num_vars, values))
            .collect();
        let ascend_numerators: Vec<Ascend910B3Mle> = numerator_values
            .iter()
            .map(|values| npu_mle(&backend, num_vars, values))
            .collect();
        let ascend_denominators: Vec<Ascend910B3Mle> = denominator_values
            .iter()
            .map(|values| npu_mle(&backend, num_vars, values))
            .collect();

        let cpu_result = CpuBackend.compute_frac_poly(&cpu_numerators, &cpu_denominators)?;
        let ascend_result = backend.compute_frac_poly(&ascend_numerators, &ascend_denominators)?;
        assert_mle_matches(
            &backend,
            &ascend_result,
            &cpu_result,
            &expected,
            &comparison_points(num_vars, 0xF4),
        );

        // Inputs remain immutable.
        assert_mle_matches(
            &backend,
            &ascend_numerators[0],
            &cpu_mle(num_vars, &numerator_values[0]),
            &numerator_values[0],
            &comparison_points(num_vars, 0xF5),
        );
    }

    for num_vars in 0..=3 {
        let length = 1usize << num_vars;
        let numerator_values = field_table(num_vars, 0xCC);
        let mut denominator_values = nonzero_table(num_vars, 0xDD);
        for position in [0usize, length - 1] {
            denominator_values[position] = Fr::ZERO;
            let cpu_denominator = cpu_mle(num_vars, &denominator_values);
            let ascend_denominator = npu_mle(&backend, num_vars, &denominator_values);
            // x / 0
            assert_invalid_parameters(CpuBackend.compute_frac_poly(
                &[cpu_mle(num_vars, &numerator_values)],
                &[cpu_denominator.clone()],
            ));
            assert_invalid_parameters(backend.compute_frac_poly(
                &[npu_mle(&backend, num_vars, &numerator_values)],
                &[ascend_denominator.clone()],
            ));
            // 0 / 0
            let mut zero_numerator = numerator_values.clone();
            zero_numerator[position] = Fr::ZERO;
            assert_invalid_parameters(CpuBackend.compute_frac_poly(
                &[cpu_mle(num_vars, &zero_numerator)],
                &[cpu_denominator],
            ));
            assert_invalid_parameters(backend.compute_frac_poly(
                &[npu_mle(&backend, num_vars, &zero_numerator)],
                &[ascend_denominator],
            ));
            denominator_values[position] = Fr::ONE;
        }

        // A zero produced only by multiplying factors is still a zero denominator.
        let mut first_factor = nonzero_table(num_vars, 0xD4);
        first_factor[0] = Fr::ZERO;
        let second_factor = nonzero_table(num_vars, 0xD5);
        assert_invalid_parameters(CpuBackend.compute_frac_poly(
            &[cpu_mle(num_vars, &numerator_values)],
            &[
                cpu_mle(num_vars, &first_factor),
                cpu_mle(num_vars, &second_factor),
            ],
        ));
        assert_invalid_parameters(backend.compute_frac_poly(
            &[npu_mle(&backend, num_vars, &numerator_values)],
            &[
                npu_mle(&backend, num_vars, &first_factor),
                npu_mle(&backend, num_vars, &second_factor),
            ],
        ));
    }

    // Structural misuse is rejected by this backend before any device call; the
    // CPU implementation documents the same caller obligation but does not check it.
    let values = field_table(2, 7);
    let handle = npu_mle(&backend, 2, &values);
    assert_invalid_parameters(backend.compute_frac_poly(&[handle.clone()], &[]));
    assert_invalid_parameters(backend.compute_frac_poly(
        &[handle.clone()],
        &[handle.clone(), handle.clone()],
    ));
    assert_invalid_parameters(backend.compute_frac_poly(&[], &[]));
    Ok(())
}

#[test]
fn fraction_zero_detection_spans_grid_stride_tiles_without_poisoning() -> TestResult {
    let _guard = device_guard();
    let backend = ascend_backend();

    // Consumer boundary: 2^12 = 4096 evaluations span more than one
    // grid-stride tile (40 * 48 = 1920 lanes), so zero detection must accumulate
    // over every tile. A zero in the first tile cannot be overwritten by later
    // nonzero tiles, and a nonzero last tile with padded lanes must not report a
    // false zero. This is not an operation-count assertion.
    let num_vars = 12;
    let length = 1usize << num_vars;
    let mut numerator_values = nonzero_table(num_vars, 0xF0);
    numerator_values[0] = Fr::ONE; // x / 0 at the first lane
    numerator_values[length - 1] = Fr::from(9u64);
    let denominator_values = nonzero_table(num_vars, 0xF1);
    assert_ne!(denominator_values[length - 1], Fr::ZERO);

    let cpu_numerator = cpu_mle(num_vars, &numerator_values);
    let ascend_numerator = npu_mle(&backend, num_vars, &numerator_values);
    let cpu_denominator = cpu_mle(num_vars, &denominator_values);
    let ascend_denominator = npu_mle(&backend, num_vars, &denominator_values);

    for zero_position in [0usize, length - 1] {
        let mut zeroed = denominator_values.clone();
        zeroed[zero_position] = Fr::ZERO;
        let cpu_zero = cpu_mle(num_vars, &zeroed);
        let ascend_zero = npu_mle(&backend, num_vars, &zeroed);

        // x / 0
        assert_invalid_parameters(CpuBackend.compute_frac_poly(
            &[cpu_numerator.clone()],
            &[cpu_zero.clone()],
        ));
        assert_invalid_parameters(backend.compute_frac_poly(
            &[ascend_numerator.clone()],
            &[ascend_zero.clone()],
        ));

        // 0 / 0
        let mut zero_numerator_values = numerator_values.clone();
        zero_numerator_values[zero_position] = Fr::ZERO;
        assert_invalid_parameters(CpuBackend.compute_frac_poly(
            &[cpu_mle(num_vars, &zero_numerator_values)],
            &[cpu_zero.clone()],
        ));
        assert_invalid_parameters(backend.compute_frac_poly(
            &[npu_mle(&backend, num_vars, &zero_numerator_values)],
            &[ascend_zero.clone()],
        ));

        // A single factor carrying the only zero is enough.
        assert_invalid_parameters(CpuBackend.compute_frac_poly(
            &[cpu_numerator.clone()],
            &[cpu_denominator.clone(), cpu_zero],
        ));
        assert_invalid_parameters(backend.compute_frac_poly(
            &[ascend_numerator.clone()],
            &[ascend_denominator.clone(), ascend_zero],
        ));
    }

    // The rejected calls are validation failures, not context poisoning: the
    // same context must produce an exact zero-free result afterwards.
    let expected: Vec<Fr> = numerator_values
        .iter()
        .zip(&denominator_values)
        .map(|(numerator, denominator)| {
            *numerator
                * denominator
                    .inverse()
                    .expect("constructed denominator is nonzero")
        })
        .collect();
    let cpu_result = CpuBackend.compute_frac_poly(&[cpu_numerator], &[cpu_denominator])?;
    let ascend_result = backend.compute_frac_poly(&[ascend_numerator], &[ascend_denominator])?;
    assert_mle_matches(
        &backend,
        &ascend_result,
        &cpu_result,
        &expected,
        &comparison_points(num_vars, 0xF2),
    );
    let combined = backend.linear_combination(&[ascend_result], &[Fr::ONE])?;
    assert_mle_matches(
        &backend,
        &combined,
        &cpu_mle(num_vars, &expected),
        &expected,
        &comparison_points(num_vars, 0xF3),
    );
    Ok(())
}

#[test]
fn product_recurrence_and_factors_match_cpu() -> TestResult {
    let _guard = device_guard();
    let backend = ascend_backend();

    for num_vars in 1..=8 {
        let values = field_table(num_vars, 0x9A ^ num_vars as u64);
        let expected_product = expected_product_table(&values);
        let expected_factors = expected_factor_tables(&values, &expected_product);

        let cpu_frac = cpu_mle(num_vars, &values);
        let ascend_frac = npu_mle(&backend, num_vars, &values);
        let cpu_product = CpuBackend.compute_product_poly(&cpu_frac)?;
        let ascend_product = backend.compute_product_poly(&ascend_frac)?;
        assert_mle_matches(
            &backend,
            &ascend_product,
            &cpu_product,
            &expected_product,
            &comparison_points(num_vars, 0x9B),
        );

        let cpu_factors = CpuBackend.compute_product_factors(&cpu_frac, &cpu_product)?;
        let ascend_factors = backend.compute_product_factors(&ascend_frac, &ascend_product)?;
        for (index, expected) in expected_factors.iter().enumerate() {
            assert_mle_matches(
                &backend,
                &ascend_factors[index],
                &cpu_factors[index],
                expected,
                &comparison_points(num_vars, 0x9C + index as u64),
            );
        }
    }

    let constant = field_table(0, 3);
    assert_invalid_parameters(backend.compute_product_poly(&npu_mle(&backend, 0, &constant)));
    assert_invalid_parameters(CpuBackend.compute_product_poly(&cpu_mle(0, &constant)));
    assert_invalid_parameters(backend.compute_product_factors(
        &npu_mle(&backend, 0, &constant),
        &npu_mle(&backend, 0, &constant),
    ));
    assert_invalid_parameters(CpuBackend.compute_product_factors(
        &cpu_mle(0, &constant),
        &cpu_mle(0, &constant),
    ));
    assert_invalid_parameters(backend.compute_product_factors(
        &npu_mle(&backend, 1, &field_table(1, 5)),
        &npu_mle(&backend, 2, &field_table(2, 6)),
    ));
    assert_invalid_parameters(CpuBackend.compute_product_factors(
        &cpu_mle(1, &field_table(1, 5)),
        &cpu_mle(2, &field_table(2, 6)),
    ));
    Ok(())
}

/// Independent implementation of `compute_product_poly` from its recurrence.
fn expected_product_table(frac: &[Fr]) -> Vec<Fr> {
    let size = frac.len();
    let mut evaluations = Vec::with_capacity(size);
    for pair in frac.chunks_exact(2) {
        evaluations.push(pair[0] * pair[1]);
    }
    for child in (0..size - 2).step_by(2) {
        evaluations.push(evaluations[child] * evaluations[child + 1]);
    }
    evaluations.push(Fr::ZERO);
    evaluations
}

/// Independent implementation of `compute_product_factors`.
fn expected_factor_tables(frac: &[Fr], product: &[Fr]) -> [Vec<Fr>; 2] {
    let mut first = Vec::with_capacity(frac.len());
    let mut second = Vec::with_capacity(frac.len());
    for evaluations in [frac, product] {
        for pair in evaluations.chunks_exact(2) {
            first.push(pair[0]);
            second.push(pair[1]);
        }
    }
    [first, second]
}

#[test]
fn nums_and_denoms_match_cpu_with_nonidentity_permutations() -> TestResult {
    let _guard = device_guard();
    let backend = ascend_backend();
    let columns = 3usize;

    for num_vars in 0..=8 {
        let length = 1usize << num_vars;
        let f_values: Vec<Vec<Fr>> = (0..columns)
            .map(|column| field_table(num_vars, 0x21 + column as u64))
            .collect();
        let g_values: Vec<Vec<Fr>> = (0..columns)
            .map(|column| nonzero_table(num_vars, 0x31 + column as u64))
            .collect();
        let perm_values: Vec<Vec<Fr>> = (0..columns)
            .map(|column| {
                let base = field_table(num_vars, 0x41 + column as u64);
                if num_vars == 0 {
                    base
                } else {
                    // Cyclic shifts are nonidentity permutations of the table.
                    // length >= 2 here, so this shift never wraps to a full
                    // rotation even for column + 1 == length.
                    let shift = 1 + column % (length - 1);
                    (0..length)
                        .map(|index| base[(index + shift) % length])
                        .collect()
                }
            })
            .collect();
        if num_vars > 0 {
            for column in 0..columns {
                assert_ne!(perm_values[column], field_table(num_vars, 0x41 + column as u64));
            }
        }

        let cpu_f: Vec<CpuMle> = f_values
            .iter()
            .map(|values| cpu_mle(num_vars, values))
            .collect();
        let cpu_g: Vec<CpuMle> = g_values
            .iter()
            .map(|values| cpu_mle(num_vars, values))
            .collect();
        let cpu_p: Vec<CpuMle> = perm_values
            .iter()
            .map(|values| cpu_mle(num_vars, values))
            .collect();
        let ascend_f: Vec<Ascend910B3Mle> = f_values
            .iter()
            .map(|values| npu_mle(&backend, num_vars, values))
            .collect();
        let ascend_g: Vec<Ascend910B3Mle> = g_values
            .iter()
            .map(|values| npu_mle(&backend, num_vars, values))
            .collect();
        let ascend_p: Vec<Ascend910B3Mle> = perm_values
            .iter()
            .map(|values| npu_mle(&backend, num_vars, values))
            .collect();

        let mut rng = TestRng::new(0x55 ^ num_vars as u64);
        let challenge_pairs = [
            (rng.fr(), rng.fr()),
            (Fr::ZERO, rng.fr()),
            (rng.fr(), Fr::ZERO),
            (Fr::ZERO, Fr::ZERO),
        ];
        for (beta, gamma) in challenge_pairs {
            let mut expected_numerators = Vec::with_capacity(columns);
            let mut expected_denominators = Vec::with_capacity(columns);
            for column in 0..columns {
                let mut numerators = Vec::with_capacity(length);
                let mut denominators = Vec::with_capacity(length);
                for row in 0..length {
                    numerators.push(
                        f_values[column][row]
                            + beta * Fr::from((column * length + row) as u64)
                            + gamma,
                    );
                    denominators.push(
                        g_values[column][row] + beta * perm_values[column][row] + gamma,
                    );
                }
                expected_numerators.push(numerators);
                expected_denominators.push(denominators);
            }

            let (cpu_numerators, cpu_denominators) = CpuBackend.compute_nums_and_denoms(
                &beta, &gamma, &cpu_f, &cpu_g, &cpu_p,
            )?;
            let (ascend_numerators, ascend_denominators) = backend.compute_nums_and_denoms(
                &beta, &gamma, &ascend_f, &ascend_g, &ascend_p,
            )?;
            assert_eq!(ascend_numerators.len(), columns);
            assert_eq!(ascend_denominators.len(), columns);
            for column in 0..columns {
                assert_mle_matches(
                    &backend,
                    &ascend_numerators[column],
                    &cpu_numerators[column],
                    &expected_numerators[column],
                    &comparison_points(num_vars, 0x56),
                );
                assert_mle_matches(
                    &backend,
                    &ascend_denominators[column],
                    &cpu_denominators[column],
                    &expected_denominators[column],
                    &comparison_points(num_vars, 0x57),
                );
            }
        }

        // Inputs stay immutable.
        for column in 0..columns {
            assert_mle_matches(
                &backend,
                &ascend_f[column],
                &cpu_mle(num_vars, &f_values[column]),
                &f_values[column],
                &comparison_points(num_vars, 0x58),
            );
        }
    }

    // Structural misuse is rejected before any device call.
    let base = npu_mle(&backend, 2, &field_table(2, 0x35));
    assert_invalid_parameters(backend.compute_nums_and_denoms(
        &Fr::ONE,
        &Fr::ONE,
        &[base.clone()],
        &[base.clone(), base.clone()],
        &[base.clone()],
    ));
    assert_invalid_parameters(backend.compute_nums_and_denoms(
        &Fr::ONE,
        &Fr::ONE,
        &[],
        &[],
        &[],
    ));
    Ok(())
}

#[test]
fn commit_matches_cpu_and_independent_msm() -> TestResult {
    let _guard = device_guard();
    let backend = ascend_backend();

    for param_vars in [0usize, 1, 2, 4] {
        let layers = srs_layers(param_vars, 0xC0 ^ param_vars as u64);
        let cpu_params = cpu_prepared(param_vars, &layers)?;
        let ascend_params = ascend_prepared(&backend, param_vars, &layers)?;
        for num_vars in 0..=param_vars {
            let values = field_table(num_vars, 0xC1 ^ num_vars as u64);
            let expected = independent_commitment(&layers, param_vars, num_vars, &values);
            let cpu_handle = cpu_mle(num_vars, &values);
            let ascend_handle = npu_mle(&backend, num_vars, &values);
            assert_eq!(cpu_commit(&cpu_params, &cpu_handle)?, expected);
            assert_eq!(
                ascend_commit(&backend, &ascend_params, &ascend_handle)?,
                expected
            );
        }
    }
    Ok(())
}

#[test]
fn multi_commit_preserves_order_duplicates_and_empty_batch() -> TestResult {
    let _guard = device_guard();
    let backend = ascend_backend();

    let param_vars = 4usize;
    let layers = srs_layers(param_vars, 0x77);
    let cpu_params = cpu_prepared(param_vars, &layers)?;
    let ascend_params = ascend_prepared(&backend, param_vars, &layers)?;

    let constant = field_table(0, 0x01);
    let two_a = field_table(2, 0x02);
    let two_b = field_table(2, 0x04);
    let four = field_table(4, 0x05);

    let cpu_constant = cpu_mle(0, &constant);
    let cpu_two_a = cpu_mle(2, &two_a);
    let cpu_duplicate = cpu_mle(2, &two_a);
    let cpu_two_b = cpu_mle(2, &two_b);
    let cpu_four = cpu_mle(4, &four);
    let ascend_constant = npu_mle(&backend, 0, &constant);
    let ascend_two_a = npu_mle(&backend, 2, &two_a);
    let ascend_duplicate = npu_mle(&backend, 2, &two_a);
    let ascend_two_b = npu_mle(&backend, 2, &two_b);
    let ascend_four = npu_mle(&backend, 4, &four);
    assert_ne!(ascend_two_a.id(), ascend_duplicate.id());
    assert_ne!(ascend_two_a.id(), ascend_two_b.id());

    let cpu_polynomials = vec![
        cpu_constant,
        cpu_two_a.clone(),
        cpu_two_a,
        cpu_duplicate,
        cpu_two_b,
        cpu_four,
    ];
    let ascend_polynomials = vec![
        ascend_constant,
        ascend_two_a.clone(),
        ascend_two_a,
        ascend_duplicate,
        ascend_two_b,
        ascend_four,
    ];
    let cpu_commitments = CpuBackend.multi_commit(&cpu_params, &cpu_polynomials)?;
    let ascend_commitments = backend.multi_commit(&ascend_params, &ascend_polynomials)?;

    let expected = vec![
        independent_commitment(&layers, param_vars, 0, &constant),
        independent_commitment(&layers, param_vars, 2, &two_a),
        independent_commitment(&layers, param_vars, 2, &two_a),
        independent_commitment(&layers, param_vars, 2, &two_a),
        independent_commitment(&layers, param_vars, 2, &two_b),
        independent_commitment(&layers, param_vars, 4, &four),
    ];
    assert_eq!(cpu_commitments, expected);
    assert_eq!(ascend_commitments, expected);

    assert!(CpuBackend.multi_commit(&cpu_params, &[])?.is_empty());
    assert!(backend.multi_commit(&ascend_params, &[])?.is_empty());
    Ok(())
}

#[test]
fn commit_and_open_reject_oversized_dimensions_and_mismatched_points() -> TestResult {
    let _guard = device_guard();
    let backend = ascend_backend();

    let param_vars = 2usize;
    let layers = srs_layers(param_vars, 0x88);
    let cpu_params = cpu_prepared(param_vars, &layers)?;
    let ascend_params = ascend_prepared(&backend, param_vars, &layers)?;

    for num_vars in [1usize, 2, 3] {
        let values = field_table(num_vars, 0x89);
        let cpu_handle = cpu_mle(num_vars, &values);
        let ascend_handle = npu_mle(&backend, num_vars, &values);
        let point: Vec<Fr> = (0..num_vars).map(|i| Fr::from(i as u64 + 2)).collect();

        if num_vars > param_vars {
            assert_invalid_parameters(cpu_commit(&cpu_params, &cpu_handle));
            assert_invalid_parameters(ascend_commit(&backend, &ascend_params, &ascend_handle));
            assert_invalid_parameters(CpuBackend.multi_commit(&cpu_params, &[cpu_handle.clone()]));
            assert_invalid_parameters(backend.multi_commit(&ascend_params, &[ascend_handle.clone()]));
            assert_invalid_parameters(cpu_open(&cpu_params, &cpu_handle, &point));
            assert_invalid_parameters(ascend_open(
                &backend,
                &ascend_params,
                &ascend_handle,
                &point,
            ));
        } else {
            let short = &point[..point.len() - 1];
            assert_invalid_parameters(cpu_open(&cpu_params, &cpu_handle, short));
            assert_invalid_parameters(ascend_open(&backend, &ascend_params, &ascend_handle, short));
            if num_vars == param_vars {
                // The valid point must still be accepted, not over-rejected.
                let (_cpu_proofs, cpu_eval) = cpu_open(&cpu_params, &cpu_handle, &point)?;
                let (_ascend_proofs, ascend_eval) =
                    ascend_open(&backend, &ascend_params, &ascend_handle, &point)?;
                assert_eq!(ascend_eval, cpu_eval);
            }
        }
    }
    Ok(())
}

#[test]
fn open_matches_cpu_with_exact_quotients_and_evaluations() -> TestResult {
    let _guard = device_guard();
    let backend = ascend_backend();

    let param_vars = 4usize;
    let layers = srs_layers(param_vars, 0x0C);
    let cpu_params = cpu_prepared(param_vars, &layers)?;
    let ascend_params = ascend_prepared(&backend, param_vars, &layers)?;

    for num_vars in 0..=param_vars {
        let values = field_table(num_vars, 0x0D ^ num_vars as u64);
        let cpu_handle = cpu_mle(num_vars, &values);
        let ascend_handle = npu_mle(&backend, num_vars, &values);
        for point in comparison_points(num_vars, 0x0E) {
            let (cpu_proofs, cpu_evaluation) = cpu_open(&cpu_params, &cpu_handle, &point)?;
            let (ascend_proofs, ascend_evaluation) =
                ascend_open(&backend, &ascend_params, &ascend_handle, &point)?;
            assert_eq!(cpu_evaluation, multilinear_eval(&values, &point));
            assert_eq!(ascend_evaluation, cpu_evaluation);
            assert_eq!(ascend_proofs, cpu_proofs);
            assert_eq!(ascend_proofs.len(), num_vars);
        }
    }
    Ok(())
}

#[test]
fn srs_ingress_rejects_invalid_layer_shapes_and_points() -> TestResult {
    let _guard = device_guard();
    let backend = ascend_backend();

    let param_vars = 2usize;
    let layers = srs_layers(param_vars, 0x5A);

    // Layer count: exactly num_vars + 1 layers.
    assert_invalid_parameters(ascend_prepared(
        &backend,
        param_vars,
        &layers[..param_vars],
    ));
    let mut extra_layer = layers.clone();
    extra_layer.push(layers[0].clone());
    assert_invalid_parameters(ascend_prepared(&backend, param_vars, &extra_layer));
    assert_invalid_parameters(ascend_prepared(&backend, param_vars + 1, &layers));

    // Layer lengths: 2^(num_vars - layer).
    let mut short_layer = layers.clone();
    short_layer[0].pop();
    assert_invalid_parameters(ascend_prepared(&backend, param_vars, &short_layer));
    let mut long_layer = layers.clone();
    long_layer[param_vars].push(layers[0][0]);
    assert_invalid_parameters(ascend_prepared(&backend, param_vars, &long_layer));

    // Off-curve finite point (safe unchecked Ark construction).
    let mut off_curve = layers.clone();
    off_curve[0][1] = G1Affine::new_unchecked(Fq::from(1u64), Fq::from(1u64));
    assert!(!off_curve[0][1].is_on_curve());
    assert_invalid_parameters(ascend_prepared(&backend, param_vars, &off_curve));

    // On-curve but outside the prime-order subgroup.
    let non_subgroup = on_curve_non_subgroup_point();
    assert!(non_subgroup.is_on_curve());
    assert!(!non_subgroup.is_in_correct_subgroup_assuming_on_curve());
    let mut non_subgroup_layers = layers;
    non_subgroup_layers[param_vars][0] = non_subgroup;
    assert_invalid_parameters(ascend_prepared(
        &backend,
        param_vars,
        &non_subgroup_layers,
    ));
    Ok(())
}

#[test]
fn identity_srs_base_is_accepted_and_commits_exactly() -> TestResult {
    let _guard = device_guard();
    let backend = ascend_backend();

    let param_vars = 2usize;
    let mut layers = srs_layers(param_vars, 0x6B);
    // The identity is a valid subgroup element; the host packet tag must round-trip.
    layers[0][3] = G1Affine::identity();
    layers[1][1] = G1Affine::identity();
    layers[param_vars][0] = G1Affine::identity();

    let cpu_params = cpu_prepared(param_vars, &layers)?;
    let ascend_params = ascend_prepared(&backend, param_vars, &layers)?;
    for num_vars in 0..=param_vars {
        let values = field_table(num_vars, 0x6C ^ num_vars as u64);
        let expected = independent_commitment(&layers, param_vars, num_vars, &values);
        let cpu_handle = cpu_mle(num_vars, &values);
        let ascend_handle = npu_mle(&backend, num_vars, &values);
        assert_eq!(cpu_commit(&cpu_params, &cpu_handle)?, expected);
        assert_eq!(
            ascend_commit(&backend, &ascend_params, &ascend_handle)?,
            expected
        );
    }
    Ok(())
}

#[test]
fn independent_contexts_isolate_resources_without_masking_multicontext() -> TestResult {
    let _guard = device_guard();
    let first = ascend_backend();
    let second = ascend_backend();

    let values = field_table(2, 0x2A);
    let first_mle = npu_mle(&first, 2, &values);
    let second_mle = npu_mle(&second, 2, &values);
    let point = comparison_points(2, 0x2B)[0].clone();
    let expected = multilinear_eval(&values, &point);

    assert_eq!(first.evaluate_mle(&first_mle, &point)?, expected);
    assert_eq!(second.evaluate_mle(&second_mle, &point)?, expected);
    assert_invalid_parameters(first.evaluate_mle(&second_mle, &point));
    assert_invalid_parameters(second.evaluate_mle(&first_mle, &point));
    assert_invalid_parameters(first.linear_combination(
        &[first_mle.clone(), second_mle.clone()],
        &[Fr::ONE, Fr::ONE],
    ));

    let param_vars = 2usize;
    let layers = srs_layers(param_vars, 0x2C);
    let first_params = ascend_prepared(&first, param_vars, &layers)?;
    let second_params = ascend_prepared(&second, param_vars, &layers)?;
    let expected_commitment_value = independent_commitment(&layers, param_vars, 2, &values);
    assert_eq!(
        ascend_commit(&first, &first_params, &first_mle)?,
        expected_commitment_value
    );
    assert_eq!(
        ascend_commit(&second, &second_params, &second_mle)?,
        expected_commitment_value
    );
    assert_invalid_parameters(ascend_commit(&first, &first_params, &second_mle));
    assert_invalid_parameters(ascend_commit(&first, &second_params, &first_mle));
    assert_invalid_parameters(ascend_commit(&second, &first_params, &second_mle));
    assert_invalid_parameters(ascend_open(&first, &second_params, &first_mle, &point));

    // Both contexts remain usable after the cross-context rejections.
    assert_eq!(first.evaluate_mle(&first_mle, &point)?, expected);
    assert_eq!(second.evaluate_mle(&second_mle, &point)?, expected);

    // A prover state is bound to the context that created it.
    let mut first_polynomial = VirtualPolynomial::new(1);
    first_polynomial.add_mle_list(
        [npu_mle(&first, 1, &field_table(1, 0x2D))],
        Fr::ONE,
    )?;
    let mut state = first.prover_init(&first_polynomial)?;
    assert_invalid_parameters(second.prove_round_and_update_state(&mut state, &None));
    let first_round = first.prove_round_and_update_state(&mut state, &None)?;
    assert_eq!(first_round.len(), 2);
    Ok(())
}

#[test]
fn resources_survive_ordered_drops_and_same_device_reuse() -> TestResult {
    let _guard = device_guard();
    let backend = ascend_backend();

    let param_vars = 2usize;
    let layers = srs_layers(param_vars, 0x3A);
    let params = ascend_prepared(&backend, param_vars, &layers)?;
    let values = field_table(2, 0x3B);
    let factor_tables: Vec<Vec<Fr>> = (0..3)
        .map(|index| field_table(2, 0x3C + index as u64))
        .collect();
    let factor_handles: Vec<Ascend910B3Mle> = factor_tables
        .iter()
        .map(|table| npu_mle(&backend, 2, table))
        .collect();
    let products = [(Fr::from(2u64), vec![0usize, 1]), (Fr::from(3u64), vec![2])];
    let polynomial = build_virtual_polynomial(2, &products, &factor_handles)?;
    let challenges = [Fr::from(5u64), Fr::from(7u64)];

    let cpu_transcript = {
        let cpu_handles: Vec<CpuMle> = factor_tables
            .iter()
            .map(|table| cpu_mle(2, table))
            .collect();
        let cpu_polynomial = build_virtual_polynomial(2, &products, &cpu_handles)?;
        let mut state = CpuBackend.prover_init(&cpu_polynomial)?;
        sumcheck_transcript(&CpuBackend, &mut state, 2, &challenges)?
    };

    let mut state = backend.prover_init(&polynomial)?;
    // The prepared SRS can retire while the prover state still owns its inputs.
    drop(params);
    let mut messages = vec![backend.prove_round_and_update_state(&mut state, &None)?];
    // Input handles and the virtual polynomial may also go away after init.
    drop(polynomial);
    drop(factor_handles);
    messages.push(backend.prove_round_and_update_state(&mut state, &Some(challenges[0]))?);
    assert_eq!(messages, cpu_transcript);
    drop(state);

    // The same context remains usable for unrelated work.
    let alive = npu_mle(&backend, 2, &values);
    assert_mle_matches(
        &backend,
        &alive,
        &cpu_mle(2, &values),
        &values,
        &comparison_points(2, 0x3D),
    );

    // Dropping the entry point while handles remain retires safely; a fresh
    // context on the same device must initialize and compute afterwards.
    let extra = npu_mle(&backend, 1, &field_table(1, 0x3E));
    drop(backend);
    drop(alive);
    drop(extra);

    let next = ascend_backend();
    let next_values = field_table(2, 0x3F);
    let next_handle = npu_mle(&next, 2, &next_values);
    assert_mle_matches(
        &next,
        &next_handle,
        &cpu_mle(2, &next_values),
        &next_values,
        &comparison_points(2, 0x40),
    );
    Ok(())
}

#[test]
fn sumcheck_input_handles_can_be_dropped_after_prover_init() -> TestResult {
    let _guard = device_guard();
    let backend = ascend_backend();

    let num_vars = 2usize;
    let tables: Vec<Vec<Fr>> = (0..3)
        .map(|index| field_table(num_vars, 0x44 + index as u64))
        .collect();
    let products = [
        (Fr::from(2u64), vec![0usize, 0]),
        (Fr::from(5u64), vec![1, 2]),
    ];
    let ascend_handles: Vec<Ascend910B3Mle> = tables
        .iter()
        .map(|table| npu_mle(&backend, num_vars, table))
        .collect();
    let polynomial = build_virtual_polynomial(num_vars, &products, &ascend_handles)?;
    let challenges = [Fr::from(8u64), Fr::from(9u64)];

    let cpu_transcript = {
        let cpu_handles: Vec<CpuMle> = tables
            .iter()
            .map(|table| cpu_mle(num_vars, table))
            .collect();
        let cpu_polynomial = build_virtual_polynomial(num_vars, &products, &cpu_handles)?;
        let mut state = CpuBackend.prover_init(&cpu_polynomial)?;
        sumcheck_transcript(&CpuBackend, &mut state, num_vars, &challenges)?
    };

    let mut state = backend.prover_init(&polynomial)?;
    // Native state owns every numerical input after creation.
    drop(polynomial);
    drop(ascend_handles);
    let transcript = sumcheck_transcript(&backend, &mut state, num_vars, &challenges)?;
    assert_eq!(transcript, cpu_transcript);
    Ok(())
}

#[test]
fn sumcheck_round_messages_match_cpu_for_mixed_products() -> TestResult {
    let _guard = device_guard();
    let backend = ascend_backend();

    let num_vars = 3usize;
    let shared = field_table(num_vars, 0x61);
    let tables: Vec<Vec<Fr>> = vec![
        shared.clone(),
        field_table(num_vars, 0x62),
        field_table(num_vars, 0x63),
        field_table(num_vars, 0x64),
        field_table(num_vars, 0x65),
        shared, // equal contents, distinct handle identity
    ];
    let products = vec![
        (Fr::from(2u64), vec![0usize, 0]),
        (Fr::from(3u64), vec![0, 1]),
        (Fr::from(5u64), vec![1, 5]),
        (Fr::from(7u64), vec![2, 3, 4]),
        (Fr::from(11u64), vec![5]),
        (Fr::ZERO, vec![0, 2, 3]),
    ];

    let cpu_factors: Vec<CpuMle> = tables
        .iter()
        .map(|table| cpu_mle(num_vars, table))
        .collect();
    let ascend_factors: Vec<Ascend910B3Mle> = tables
        .iter()
        .map(|table| npu_mle(&backend, num_vars, table))
        .collect();
    assert_ne!(ascend_factors[0].id(), ascend_factors[5].id());

    let cpu_polynomial = build_virtual_polynomial(num_vars, &products, &cpu_factors)?;
    let ascend_polynomial = build_virtual_polynomial(num_vars, &products, &ascend_factors)?;
    assert_eq!(ascend_polynomial.aux_info.max_degree, 3);
    // Dedup is by handle identity, and repeated multiplicands stay in the product.
    assert_eq!(ascend_polynomial.flattened_ml_extensions().len(), 6);
    assert_eq!(ascend_polynomial.products[0].1, vec![0, 0]);
    assert_eq!(ascend_polynomial.products[1].1, vec![0, 1]);
    assert_eq!(ascend_polynomial.products[2].1, vec![1, 2]);
    assert_eq!(ascend_polynomial.products[3].1, vec![3, 4, 5]);
    assert_eq!(ascend_polynomial.products[4].1, vec![2]);
    assert_eq!(ascend_polynomial.products[5].1, vec![0, 3, 4]);

    let ascend_factor_tables =
        flattened_factor_tables(&ascend_polynomial, &ascend_factors, &tables);
    let cpu_factor_tables = flattened_factor_tables(&cpu_polynomial, &cpu_factors, &tables);
    assert_eq!(ascend_factor_tables, cpu_factor_tables);

    let mut rng = TestRng::new(0x66);
    let challenge_sets = [
        [rng.fr(), rng.fr(), rng.fr()],
        [Fr::ZERO, Fr::ONE, -Fr::ONE],
    ];
    for challenges in challenge_sets {
        let cpu_transcript = {
            let mut state = CpuBackend.prover_init(&cpu_polynomial)?;
            sumcheck_transcript(&CpuBackend, &mut state, num_vars, &challenges)?
        };
        let mut state = backend.prover_init(&ascend_polynomial)?;
        let ascend_transcript =
            sumcheck_transcript(&backend, &mut state, num_vars, &challenges)?;
        assert_eq!(ascend_transcript, cpu_transcript);
        assert_sumcheck_messages(
            &ascend_polynomial.products,
            &ascend_factor_tables,
            num_vars,
            3,
            &challenges,
            &ascend_transcript,
        );
    }
    Ok(())
}

#[test]
fn sumcheck_repeated_factors_mixed_arity_max_degree_34_are_independent() -> TestResult {
    let _guard = device_guard();
    let backend = ascend_backend();

    let num_vars = 3usize;
    let factor_tables: Vec<Vec<Fr>> = (0..7)
        .map(|index| field_table(num_vars, 0x71 + index as u64))
        .collect();
    let mut products: Vec<(Fr, Vec<usize>)> = Vec::new();
    for arity in 1..=34usize {
        let indices: Vec<usize> = (0..arity).map(|offset| (offset * 3 + arity) % 7).collect();
        products.push((Fr::from(arity as u64 + 1), indices));
    }
    products.push((Fr::ZERO, vec![0, 0, 0]));

    let cpu_factors: Vec<CpuMle> = factor_tables
        .iter()
        .map(|table| cpu_mle(num_vars, table))
        .collect();
    let ascend_factors: Vec<Ascend910B3Mle> = factor_tables
        .iter()
        .map(|table| npu_mle(&backend, num_vars, table))
        .collect();
    let cpu_polynomial = build_virtual_polynomial(num_vars, &products, &cpu_factors)?;
    let ascend_polynomial = build_virtual_polynomial(num_vars, &products, &ascend_factors)?;

    assert_eq!(ascend_polynomial.aux_info.max_degree, 34);
    assert_eq!(ascend_polynomial.flattened_ml_extensions().len(), 7);
    assert!(ascend_polynomial
        .products
        .iter()
        .any(|(_, indices)| indices.len() == 34));
    assert!(ascend_polynomial.products.iter().any(|(_, indices)| {
        indices
            .iter()
            .enumerate()
            .any(|(position, index)| indices[..position].contains(index))
    }));
    for arity in 1..=34usize {
        assert!(ascend_polynomial
            .products
            .iter()
            .any(|(_, indices)| indices.len() == arity));
    }

    let ascend_factor_tables =
        flattened_factor_tables(&ascend_polynomial, &ascend_factors, &factor_tables);
    let cpu_factor_tables =
        flattened_factor_tables(&cpu_polynomial, &cpu_factors, &factor_tables);
    assert_eq!(ascend_factor_tables.len(), 7);
    assert_eq!(ascend_factor_tables, cpu_factor_tables);

    let challenges = [Fr::from(2u64), Fr::from(3u64), Fr::from(5u64)];
    let cpu_transcript = {
        let mut state = CpuBackend.prover_init(&cpu_polynomial)?;
        sumcheck_transcript(&CpuBackend, &mut state, num_vars, &challenges)?
    };

    // Two states created from the same virtual polynomial advance independently.
    let mut first_state = backend.prover_init(&ascend_polynomial)?;
    let mut second_state = backend.prover_init(&ascend_polynomial)?;
    let first_message = backend.prove_round_and_update_state(&mut first_state, &None)?;
    assert_eq!(first_message, cpu_transcript[0]);
    let second_transcript =
        sumcheck_transcript(&backend, &mut second_state, num_vars, &challenges)?;
    assert_eq!(second_transcript, cpu_transcript);

    let mut first_transcript = vec![first_message];
    for round in 1..num_vars {
        first_transcript.push(backend.prove_round_and_update_state(
            &mut first_state,
            &Some(challenges[round - 1]),
        )?);
    }
    assert_eq!(first_transcript, cpu_transcript);
    assert_sumcheck_messages(
        &ascend_polynomial.products,
        &ascend_factor_tables,
        num_vars,
        34,
        &challenges,
        &second_transcript,
    );

    // Inputs remain immutable across both states.
    for (index, table) in factor_tables.iter().enumerate() {
        assert_mle_matches(
            &backend,
            &ascend_factors[index],
            &cpu_factors[index],
            table,
            &comparison_points(num_vars, 0x72 + index as u64),
        );
    }
    Ok(())
}

#[test]
fn sumcheck_rejects_zero_dimension_and_zero_degree() -> TestResult {
    let _guard = device_guard();
    let backend = ascend_backend();

    let constant = field_table(0, 0x91);
    let mut cpu_zero_dimension: VirtualPolynomial<Fr, CpuMle> = VirtualPolynomial::new(0);
    cpu_zero_dimension.add_mle_list([cpu_mle(0, &constant)], Fr::ONE)?;
    let mut ascend_zero_dimension: VirtualPolynomial<Fr, Ascend910B3Mle> = VirtualPolynomial::new(0);
    ascend_zero_dimension.add_mle_list([npu_mle(&backend, 0, &constant)], Fr::ONE)?;
    assert_invalid_parameters(CpuBackend.prover_init(&cpu_zero_dimension));
    assert_invalid_parameters(backend.prover_init(&ascend_zero_dimension));

    let cpu_zero_degree: VirtualPolynomial<Fr, CpuMle> = VirtualPolynomial::new(2);
    let ascend_zero_degree: VirtualPolynomial<Fr, Ascend910B3Mle> = VirtualPolynomial::new(2);
    assert_invalid_parameters(CpuBackend.prover_init(&cpu_zero_degree));
    assert_invalid_parameters(backend.prover_init(&ascend_zero_degree));
    Ok(())
}

#[test]
fn sumcheck_empty_whole_polynomial_messages_and_call_order_contract() -> TestResult {
    let _guard = device_guard();
    let backend = ascend_backend();

    // An empty whole sum with positive metadata is valid and returns d+1 zeros
    // for every round.
    let mut cpu_empty: VirtualPolynomial<Fr, CpuMle> = VirtualPolynomial::new(2);
    cpu_empty.aux_info.max_degree = 2;
    let mut ascend_empty: VirtualPolynomial<Fr, Ascend910B3Mle> = VirtualPolynomial::new(2);
    ascend_empty.aux_info.max_degree = 2;
    let empty_challenges = [Fr::from(3u64), Fr::from(4u64)];
    let cpu_empty_transcript = {
        let mut state = CpuBackend.prover_init(&cpu_empty)?;
        sumcheck_transcript(&CpuBackend, &mut state, 2, &empty_challenges)?
    };
    let mut empty_state = backend.prover_init(&ascend_empty)?;
    let ascend_empty_transcript =
        sumcheck_transcript(&backend, &mut empty_state, 2, &empty_challenges)?;
    assert_eq!(ascend_empty_transcript, cpu_empty_transcript);
    for message in &ascend_empty_transcript {
        assert_eq!(message.len(), 3);
        assert!(message.iter().all(|value| *value == Fr::ZERO));
    }
    assert_sumcheck_messages(&[], &[], 2, 2, &empty_challenges, &ascend_empty_transcript);

    // An individual empty product is invalid; only an empty whole sum is valid.
    let values = field_table(2, 0x81);
    let handle = npu_mle(&backend, 2, &values);
    let mut empty_product: VirtualPolynomial<Fr, Ascend910B3Mle> = VirtualPolynomial::new(2);
    empty_product.aux_info.max_degree = 2;
    empty_product.products.push((Fr::ONE, Vec::new()));
    assert_invalid_parameters(backend.prover_init(&empty_product));

    // Product arity may not exceed the declared maximum degree.
    let mut arity: VirtualPolynomial<Fr, Ascend910B3Mle> = VirtualPolynomial::new(2);
    arity.add_mle_list([handle.clone()], Fr::ONE)?;
    arity.products.push((Fr::ONE, vec![0, 0]));
    assert_invalid_parameters(backend.prover_init(&arity));

    // Call-order and exhaustion failures never advance the state.
    let cpu_handle = cpu_mle(2, &values);
    let mut cpu_polynomial = VirtualPolynomial::new_from_mle(&cpu_handle, Fr::from(2u64));
    cpu_polynomial.add_mle_list(
        [cpu_handle.clone(), cpu_handle.clone()],
        Fr::from(3u64),
    )?;
    let mut ascend_polynomial = VirtualPolynomial::new_from_mle(&handle, Fr::from(2u64));
    ascend_polynomial.add_mle_list([handle.clone(), handle.clone()], Fr::from(3u64))?;
    let challenges = [Fr::from(6u64), Fr::from(7u64)];
    let cpu_transcript = {
        let mut state = CpuBackend.prover_init(&cpu_polynomial)?;
        sumcheck_transcript(&CpuBackend, &mut state, 2, &challenges)?
    };

    let mut cpu_state = CpuBackend.prover_init(&cpu_polynomial)?;
    assert_invalid_state(CpuBackend.prove_round_and_update_state(
        &mut cpu_state,
        &Some(challenges[0]),
    ));
    assert_eq!(
        CpuBackend.prove_round_and_update_state(&mut cpu_state, &None)?,
        cpu_transcript[0]
    );
    assert_invalid_state(CpuBackend.prove_round_and_update_state(&mut cpu_state, &None));
    assert_eq!(
        CpuBackend.prove_round_and_update_state(&mut cpu_state, &Some(challenges[0]))?,
        cpu_transcript[1]
    );
    assert_invalid_state(CpuBackend.prove_round_and_update_state(
        &mut cpu_state,
        &Some(challenges[1]),
    ));

    let mut state = backend.prover_init(&ascend_polynomial)?;
    assert_invalid_state(backend.prove_round_and_update_state(
        &mut state,
        &Some(challenges[0]),
    ));
    assert_eq!(
        backend.prove_round_and_update_state(&mut state, &None)?,
        cpu_transcript[0]
    );
    assert_invalid_state(backend.prove_round_and_update_state(&mut state, &None));
    assert_eq!(
        backend.prove_round_and_update_state(&mut state, &Some(challenges[0]))?,
        cpu_transcript[1]
    );
    assert_invalid_state(backend.prove_round_and_update_state(
        &mut state,
        &Some(challenges[1]),
    ));
    assert_invalid_state(backend.prove_round_and_update_state(&mut state, &None));

    // A state cannot be advanced by a different context.
    let other = ascend_backend();
    let mut foreign_state = backend.prover_init(&ascend_polynomial)?;
    assert_invalid_parameters(other.prove_round_and_update_state(&mut foreign_state, &None));
    assert_eq!(
        backend.prove_round_and_update_state(&mut foreign_state, &None)?,
        cpu_transcript[0]
    );
    Ok(())
}
