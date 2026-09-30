// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.
//
// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! CPU resident MLE adaptation and numerical operations.

use super::{CpuBackend, DenseMultilinearExtension};
use crate::{common::eq::build_eq_x_r_vec, BackendError, Mle, MleBackend};
use ark_ff::{batch_inversion, PrimeField};
use ark_poly::Polynomial;
use std::sync::Arc;

impl<F: PrimeField> Mle<F> for Arc<DenseMultilinearExtension<F>> {
    type Id = usize;

    fn num_vars(&self) -> usize {
        self.as_ref().num_vars
    }

    fn id(&self) -> Self::Id {
        // The container keeps the allocation alive for as long as this ID is
        // indexed. The address is only compared, never dereferenced.
        Arc::as_ptr(self) as usize
    }
}

impl<F: PrimeField> MleBackend<F> for CpuBackend {
    type Mle = Arc<DenseMultilinearExtension<F>>;

    fn mle_from_evaluations(
        &self,
        num_vars: usize,
        evaluations: Vec<F>,
    ) -> Result<Self::Mle, BackendError> {
        if num_vars >= usize::BITS as usize || evaluations.len() != 1usize << num_vars {
            return Err(BackendError::InvalidParameters(format!(
                "evaluation count {} does not match {} variables",
                evaluations.len(),
                num_vars
            )));
        }
        Ok(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            evaluations,
        )))
    }

    fn linear_combination(
        &self,
        polynomials: &[Self::Mle],
        coefficients: &[F],
    ) -> Result<Self::Mle, BackendError> {
        let Some(first) = polynomials.first() else {
            return Err(BackendError::InvalidParameters(
                "linear combination requires nonempty polynomials".to_string(),
            ));
        };
        if polynomials.len() != coefficients.len()
            || polynomials
                .iter()
                .any(|poly| poly.num_vars != first.num_vars)
        {
            return Err(BackendError::InvalidParameters(
                "linear combination requires matching counts and dimensions".to_string(),
            ));
        }
        let mut evaluations = vec![F::zero(); first.evaluations.len()];
        for (poly, &coefficient) in polynomials.iter().zip(coefficients) {
            if coefficient.is_zero() {
                continue;
            }
            for (value, evaluation) in evaluations.iter_mut().zip(&poly.evaluations) {
                *value += coefficient * evaluation;
            }
        }
        Ok(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            first.num_vars,
            evaluations,
        )))
    }

    fn evaluate_mle(&self, mle: &Self::Mle, point: &[F]) -> Result<F, BackendError> {
        if mle.num_vars >= usize::BITS as usize || mle.evaluations.len() != 1usize << mle.num_vars {
            return Err(BackendError::InvalidParameters(format!(
                "evaluation count {} does not match {} variables",
                mle.evaluations.len(),
                mle.num_vars
            )));
        }
        if point.len() != mle.num_vars {
            return Err(BackendError::InvalidParameters(format!(
                "wrong number of variables {} vs {}",
                mle.num_vars,
                point.len()
            )));
        }
        Ok(mle.evaluate(&point.to_vec()))
    }

    fn build_eq_x_r(&self, r: &[F]) -> Result<Self::Mle, BackendError> {
        let evaluations = build_eq_x_r_vec(r)?;
        Ok(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            r.len(),
            evaluations,
        )))
    }

    fn compute_frac_poly(
        &self,
        fxs: &[Self::Mle],
        gxs: &[Self::Mle],
    ) -> Result<Self::Mle, BackendError> {
        let mut numerator = fxs[0].evaluations.clone();
        for fx in &fxs[1..] {
            for (value, factor) in numerator.iter_mut().zip(&fx.evaluations) {
                *value *= factor;
            }
        }
        let mut denominator = gxs[0].evaluations.clone();
        for gx in &gxs[1..] {
            for (value, factor) in denominator.iter_mut().zip(&gx.evaluations) {
                *value *= factor;
            }
        }
        if denominator.iter().any(|value| value.is_zero()) {
            return Err(BackendError::InvalidParameters(
                "gxs has zero entries in the boolean hypercube".to_string(),
            ));
        }
        batch_inversion(&mut denominator);
        for (value, inverse) in numerator.iter_mut().zip(denominator) {
            *value *= inverse;
        }
        Ok(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            fxs[0].num_vars,
            numerator,
        )))
    }

    fn compute_product_poly(&self, frac_poly: &Self::Mle) -> Result<Self::Mle, BackendError> {
        if frac_poly.num_vars == 0 {
            return Err(BackendError::InvalidParameters(
                "product polynomial requires at least one variable".to_string(),
            ));
        }
        let size = frac_poly.evaluations.len();
        // ===================================
        // prod(x)
        // ===================================
        //
        // `prod(x)` can be computed via recursing the following formula for 2^n-1
        // times
        //
        // With x = (u, z), where z is the last coordinate:
        // `prod(u,z) :=
        //      [(1-z)*frac(0,u) + z*prod(0,u)] *
        //      [(1-z)*frac(1,u) + z*prod(1,u)]`
        //
        // At any given step, the right hand side of the equation
        // is available via either frac_poly or the current view of evaluations.
        let mut evaluations = Vec::with_capacity(size);
        // The last coordinate decides whether children come from frac or prod.
        // For z = 0, the children are frac(0,u) and frac(1,u): adjacent entries
        // in the low-bit-first evaluation table.
        for pair in frac_poly.evaluations.chunks_exact(2) {
            evaluations.push(pair[0] * pair[1]);
        }
        // For z = 1, the children are prod(0,u) and prod(1,u), whose evaluations
        // have already been computed.
        for child in (0..size - 2).step_by(2) {
            evaluations.push(evaluations[child] * evaluations[child + 1]);
        }
        // prod(1, 1, ..., 1) := 0
        evaluations.push(F::zero());
        Ok(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            frac_poly.num_vars,
            evaluations,
        )))
    }

    fn compute_nums_and_denoms(
        &self,
        beta: &F,
        gamma: &F,
        fxs: &[Self::Mle],
        gxs: &[Self::Mle],
        perms: &[Self::Mle],
    ) -> Result<(Vec<Self::Mle>, Vec<Self::Mle>), BackendError> {
        let num_vars = fxs[0].num_vars;
        let size = fxs[0].evaluations.len();
        let mut numerators = Vec::with_capacity(fxs.len());
        let mut denominators = Vec::with_capacity(fxs.len());
        for (column, ((fx, gx), perm)) in fxs.iter().zip(gxs).zip(perms).enumerate() {
            let mut numerator = Vec::with_capacity(size);
            let mut denominator = Vec::with_capacity(size);
            let mut identity_term = *beta * F::from((column * size) as u64);
            for (&f, (&g, &p)) in fx
                .evaluations
                .iter()
                .zip(gx.evaluations.iter().zip(&perm.evaluations))
            {
                numerator.push(f + identity_term + gamma);
                denominator.push(g + *beta * p + gamma);
                identity_term += beta;
            }
            numerators.push(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                num_vars, numerator,
            )));
            denominators.push(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                denominator,
            )));
        }
        Ok((numerators, denominators))
    }

    fn compute_product_factors(
        &self,
        frac_poly: &Self::Mle,
        prod_poly: &Self::Mle,
    ) -> Result<[Self::Mle; 2], BackendError> {
        let num_vars = frac_poly.num_vars;
        if num_vars == 0 {
            return Err(BackendError::InvalidParameters(
                "product factors require at least one variable".to_string(),
            ));
        }
        if num_vars != prod_poly.num_vars {
            return Err(BackendError::InvalidParameters(format!(
                "product factor inputs have different variable counts: {} vs {}",
                num_vars, prod_poly.num_vars
            )));
        }
        // With x = (u, z), where z is the last coordinate:
        // compute p1(u,z) = (1-z) * frac(0,u) + z * prod(0,u)
        // compute p2(u,z) = (1-z) * frac(1,u) + z * prod(1,u)
        let mut p1_evals = Vec::with_capacity(frac_poly.evaluations.len());
        let mut p2_evals = Vec::with_capacity(frac_poly.evaluations.len());
        for evaluations in [&frac_poly.evaluations, &prod_poly.evaluations] {
            for pair in evaluations.chunks_exact(2) {
                p1_evals.push(pair[0]);
                p2_evals.push(pair[1]);
            }
        }
        Ok([p1_evals, p2_evals].map(|evaluations| {
            Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                evaluations,
            ))
        }))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ark_bls12_381::Fr;

    fn sample_mle() -> Arc<DenseMultilinearExtension<Fr>> {
        Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            3,
            [2u64, 3, 5, 7, 11, 13, 17, 19]
                .into_iter()
                .map(Fr::from)
                .collect(),
        ))
    }

    fn mle(num_vars: usize, values: &[u64]) -> Arc<DenseMultilinearExtension<Fr>> {
        Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            values.iter().copied().map(Fr::from).collect(),
        ))
    }

    #[test]
    fn owned_evaluations_preserve_layout_and_validate_dimensions() -> Result<(), BackendError> {
        let poly = CpuBackend
            .mle_from_evaluations(2, [2u64, 3, 5, 7].into_iter().map(Fr::from).collect())?;
        assert_eq!(
            CpuBackend.evaluate_mle(&poly, &[Fr::from(2u64), Fr::from(3u64)])?,
            Fr::from(19u64)
        );
        let shared = poly.clone();
        let equal_contents = mle(2, &[2, 3, 5, 7]);
        assert_eq!(poly.id(), shared.id());
        assert_ne!(poly.id(), equal_contents.id());
        let constant = CpuBackend.mle_from_evaluations(0, vec![Fr::from(11u64)])?;
        assert_eq!(CpuBackend.evaluate_mle(&constant, &[])?, Fr::from(11u64));
        for (num_vars, length) in [(0, 0), (1, 1), (2, 3), (usize::BITS as usize, 0)] {
            assert!(matches!(
                CpuBackend.mle_from_evaluations(num_vars, vec![Fr::from(0u64); length]),
                Err(BackendError::InvalidParameters(_))
            ));
            let invalid = Arc::new(DenseMultilinearExtension {
                num_vars,
                evaluations: vec![Fr::from(0u64); length],
            });
            assert!(matches!(
                CpuBackend.evaluate_mle(&invalid, &vec![Fr::from(0u64); num_vars]),
                Err(BackendError::InvalidParameters(_))
            ));
        }
        Ok(())
    }

    #[test]
    fn linear_combination_preserves_values_and_inputs() -> Result<(), BackendError> {
        let p = mle(2, &[2, 3, 5, 7]);
        let q = mle(2, &[11, 13, 17, 19]);
        let combined = CpuBackend.linear_combination(
            &[p.clone(), q.clone(), p.clone()],
            &[Fr::from(2u64), Fr::from(3u64), -Fr::from(2u64)],
        )?;
        let point = [Fr::from(2u64), Fr::from(3u64)];
        assert_eq!(CpuBackend.evaluate_mle(&combined, &point)?, Fr::from(99u64));
        assert_eq!(CpuBackend.evaluate_mle(&p, &point)?, Fr::from(19u64));
        assert_eq!(CpuBackend.evaluate_mle(&q, &point)?, Fr::from(33u64));
        Ok(())
    }

    #[test]
    fn linear_combination_preserves_zero_dimension_metadata() -> Result<(), BackendError> {
        let p = mle(2, &[2, 3, 5, 7]);
        for coefficients in [
            [Fr::from(1u64), -Fr::from(1u64)],
            [Fr::from(0u64), Fr::from(0u64)],
        ] {
            let zero = CpuBackend.linear_combination(&[p.clone(), p.clone()], &coefficients)?;
            assert_eq!(zero.num_vars, 2);
            assert_eq!(
                CpuBackend.evaluate_mle(&zero, &[Fr::from(2u64), Fr::from(3u64)])?,
                Fr::from(0u64)
            );
        }
        let constant = CpuBackend.linear_combination(
            &[mle(0, &[11]), mle(0, &[3])],
            &[Fr::from(2u64), -Fr::from(3u64)],
        )?;
        assert_eq!(CpuBackend.evaluate_mle(&constant, &[])?, Fr::from(13u64));
        Ok(())
    }

    #[test]
    fn linear_combination_rejects_invalid_shapes() {
        for (polynomials, coefficients) in [
            (vec![], vec![]),
            (vec![mle(1, &[2, 3])], vec![]),
            (
                vec![mle(1, &[2, 3]), mle(2, &[2, 3, 5, 7])],
                vec![Fr::from(1u64); 2],
            ),
        ] {
            assert!(matches!(
                CpuBackend.linear_combination(&polynomials, &coefficients),
                Err(BackendError::InvalidParameters(_))
            ));
        }
    }

    #[test]
    fn fraction_matches_boolean_ratio_without_mutating_inputs() -> Result<(), BackendError> {
        let fxs = [mle(2, &[0, 2, 3, 4]), mle(2, &[2, 3, 5, 7])];
        let gxs = [mle(2, &[1, 2, 3, 4]), mle(2, &[7, 5, 6, 7])];
        let originals: Vec<_> = fxs
            .iter()
            .chain(&gxs)
            .map(|p| p.evaluations.clone())
            .collect();
        let frac = CpuBackend.compute_frac_poly(&fxs, &gxs)?;
        assert_eq!(frac.num_vars, 2);
        let expected: Vec<_> = [0u64, 6, 15, 28]
            .into_iter()
            .zip([7u64, 10, 18, 28])
            .map(|(n, d)| Fr::from(n) / Fr::from(d))
            .collect();
        assert_eq!(frac.evaluations, expected);
        for (input, original) in fxs.iter().chain(&gxs).zip(originals) {
            assert_eq!(input.evaluations, original);
        }
        Ok(())
    }

    #[test]
    fn fraction_supports_scalars_and_rejects_zero_denominators() -> Result<(), BackendError> {
        let frac = CpuBackend.compute_frac_poly(&[mle(0, &[12])], &[mle(0, &[4])])?;
        assert_eq!(frac.num_vars, 0);
        assert_eq!(CpuBackend.evaluate_mle(&frac, &[])?, Fr::from(3u64));
        for (num_vars, f, g) in [(0, vec![0], vec![0]), (1, vec![0, 2], vec![0, 3])] {
            assert!(matches!(
                CpuBackend.compute_frac_poly(&[mle(num_vars, &f)], &[mle(num_vars, &g)]),
                Err(BackendError::InvalidParameters(_))
            ));
        }
        Ok(())
    }

    #[test]
    fn product_tree_preserves_recurrence_layout_and_total_product() -> Result<(), BackendError> {
        for (num_vars, values, expected) in [
            (1, vec![2, 3], vec![6, 0]),
            (2, vec![2, 3, 5, 7], vec![6, 35, 210, 0]),
            (
                3,
                vec![2, 3, 5, 7, 11, 13, 17, 19],
                vec![6, 35, 143, 323, 210, 46189, 9699690, 0],
            ),
        ] {
            let frac = mle(num_vars, &values);
            let prod = CpuBackend.compute_product_poly(&frac)?;
            assert_eq!(prod.num_vars, num_vars);
            assert_eq!(
                prod.evaluations,
                expected.into_iter().map(Fr::from).collect::<Vec<_>>()
            );
            let [p1, p2] = CpuBackend.compute_product_factors(&frac, &prod)?;
            for row in 0..prod.evaluations.len() - 1 {
                assert_eq!(
                    prod.evaluations[row],
                    p1.evaluations[row] * p2.evaluations[row]
                );
            }
            assert_eq!(
                frac.evaluations,
                values.into_iter().map(Fr::from).collect::<Vec<_>>()
            );
        }
        assert!(matches!(
            CpuBackend.compute_product_poly(&mle(0, &[2])),
            Err(BackendError::InvalidParameters(_))
        ));
        Ok(())
    }

    #[test]
    fn permutation_terms_preserve_column_offsets_and_inputs() -> Result<(), BackendError> {
        let fxs: Vec<_> = (0..3)
            .map(|i| mle(2, &[2 + i, 3 + i, 5 + i, 7 + i]))
            .collect();
        let gxs: Vec<_> = (0..3)
            .map(|i| mle(2, &[0, 11 + i, 13 + i, 17 + i]))
            .collect();
        let perms: Vec<_> = (0..3)
            .map(|i| mle(2, &[11 - 4 * i, 10 - 4 * i, 9 - 4 * i, 8 - 4 * i]))
            .collect();
        let originals: Vec<_> = fxs
            .iter()
            .chain(&gxs)
            .chain(&perms)
            .map(|p| p.evaluations.clone())
            .collect();
        for (beta, gamma) in [
            (Fr::from(2u64), Fr::from(5u64)),
            (Fr::from(0u64), Fr::from(5u64)),
            (-Fr::from(2u64), Fr::from(0u64)),
            (Fr::from(0u64), Fr::from(0u64)),
        ] {
            let (nums, denoms) =
                CpuBackend.compute_nums_and_denoms(&beta, &gamma, &fxs, &gxs, &perms)?;
            assert_eq!(nums.len(), 3);
            assert_eq!(denoms.len(), 3);
            for column in 0..3 {
                assert_eq!(nums[column].num_vars, 2);
                assert_eq!(denoms[column].num_vars, 2);
                for row in 0..4 {
                    assert_eq!(
                        nums[column].evaluations[row],
                        fxs[column].evaluations[row]
                            + beta * Fr::from((4 * column + row) as u64)
                            + gamma
                    );
                    assert_eq!(
                        denoms[column].evaluations[row],
                        gxs[column].evaluations[row]
                            + beta * perms[column].evaluations[row]
                            + gamma
                    );
                }
            }
        }
        for (input, original) in fxs.iter().chain(&gxs).chain(&perms).zip(originals) {
            assert_eq!(input.evaluations, original);
        }
        Ok(())
    }

    #[test]
    fn permutation_terms_support_scalar_columns() -> Result<(), BackendError> {
        let fxs = [mle(0, &[2]), mle(0, &[3]), mle(0, &[5])];
        let gxs = [mle(0, &[7]), mle(0, &[11]), mle(0, &[13])];
        let perms = [mle(0, &[2]), mle(0, &[0]), mle(0, &[1])];
        let (nums, denoms) = CpuBackend.compute_nums_and_denoms(
            &Fr::from(2u64),
            &Fr::from(3u64),
            &fxs,
            &gxs,
            &perms,
        )?;
        assert_eq!(
            nums.iter().map(|p| p.evaluations[0]).collect::<Vec<_>>(),
            [5u64, 8, 12].map(Fr::from)
        );
        assert_eq!(
            denoms.iter().map(|p| p.evaluations[0]).collect::<Vec<_>>(),
            [14u64, 14, 18].map(Fr::from)
        );
        for poly in nums.iter().chain(&denoms) {
            assert_eq!(poly.num_vars, 0);
        }
        Ok(())
    }

    #[test]
    fn product_factors_preserve_bit_order_and_inputs() -> Result<(), BackendError> {
        let frac = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            2,
            [2u64, 3, 5, 7].map(Fr::from).to_vec(),
        ));
        let prod = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            2,
            [11u64, 13, 17, 19].map(Fr::from).to_vec(),
        ));
        let [p1, p2] = CpuBackend.compute_product_factors(&frac, &prod)?;
        assert_eq!(p1.evaluations, [2u64, 5, 11, 17].map(Fr::from));
        assert_eq!(p2.evaluations, [3u64, 7, 13, 19].map(Fr::from));
        assert_eq!(p1.num_vars, 2);
        assert_eq!(p2.num_vars, 2);

        let point = vec![Fr::from(3u64), Fr::from(5u64)];
        for (factor, first) in [p1, p2].iter().zip([0u64, 1]) {
            let source_point = vec![Fr::from(first), point[0]];
            let a = frac.evaluate(&source_point);
            let b = prod.evaluate(&source_point);
            assert_eq!(factor.evaluate(&point), a + (b - a) * point[1]);
        }
        assert_eq!(frac.evaluations, [2u64, 3, 5, 7].map(Fr::from));
        assert_eq!(prod.evaluations, [11u64, 13, 17, 19].map(Fr::from));
        Ok(())
    }

    #[test]
    fn product_factors_support_one_variable() -> Result<(), BackendError> {
        let frac = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            1,
            [2u64, 3].map(Fr::from).to_vec(),
        ));
        let prod = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            1,
            [5u64, 7].map(Fr::from).to_vec(),
        ));
        let [p1, p2] = CpuBackend.compute_product_factors(&frac, &prod)?;
        assert_eq!(p1.evaluations, [2u64, 5].map(Fr::from));
        assert_eq!(p2.evaluations, [3u64, 7].map(Fr::from));
        Ok(())
    }

    #[test]
    fn product_factors_reject_invalid_dimensions() {
        let frac = sample_mle();
        let prod = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            1,
            [2u64, 3].map(Fr::from).to_vec(),
        ));
        assert!(matches!(
            CpuBackend.compute_product_factors(&frac, &prod),
            Err(BackendError::InvalidParameters(_))
        ));
        let constant = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            0,
            vec![Fr::from(2u64)],
        ));
        assert!(matches!(
            CpuBackend.compute_product_factors(&constant, &constant),
            Err(BackendError::InvalidParameters(_))
        ));
    }

    #[test]
    fn invalid_dimensions_are_errors() {
        let mle = sample_mle();
        for point_length in [0, 2, 4] {
            assert!(matches!(
                CpuBackend.evaluate_mle(&mle, &vec![Fr::from(0u64); point_length]),
                Err(BackendError::InvalidParameters(_))
            ));
        }
        assert!(matches!(
            <CpuBackend as MleBackend<Fr>>::build_eq_x_r(&CpuBackend, &[]),
            Err(BackendError::InvalidParameters(_))
        ));
    }
}
