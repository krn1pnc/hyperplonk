// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.
//
// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! CPU virtual-polynomial evaluation and Dense random construction.

use super::{CpuBackend, DenseMultilinearExtension};
use crate::{common::indexing::bit_decompose, BackendError, MleBackend, VirtualPolynomial};
use ark_ff::PrimeField;
use ark_std::{
    end_timer,
    rand::{Rng, RngCore},
    start_timer,
};
use std::sync::Arc;

/// Sample a random list of multilinear polynomials.
/// Returns
/// - the list of polynomials,
/// - its sum of polynomial evaluations over the boolean hypercube.
fn random_mle_list<F: PrimeField, R: RngCore>(
    nv: usize,
    degree: usize,
    rng: &mut R,
) -> (Vec<Arc<DenseMultilinearExtension<F>>>, F) {
    let start = start_timer!(|| "sample random mle list");
    let mut multiplicands = Vec::with_capacity(degree);
    for _ in 0..degree {
        multiplicands.push(Vec::with_capacity(1 << nv))
    }
    let mut sum = F::zero();

    for _ in 0..(1 << nv) {
        let mut product = F::one();

        for e in multiplicands.iter_mut() {
            let val = F::rand(rng);
            e.push(val);
            product *= val;
        }
        sum += product;
    }

    let list = multiplicands
        .into_iter()
        .map(|x| Arc::new(DenseMultilinearExtension::from_evaluations_vec(nv, x)))
        .collect();

    end_timer!(start);
    (list, sum)
}

/// Build a random list of MLEs whose product is zero on the boolean hypercube.
/// Returns an error when `degree` is zero.
fn random_zero_mle_list<F: PrimeField, R: RngCore>(
    nv: usize,
    degree: usize,
    rng: &mut R,
) -> Result<Vec<Arc<DenseMultilinearExtension<F>>>, BackendError> {
    if degree == 0 {
        return Err(BackendError::InvalidParameters(
            "zero polynomial product requires at least one multiplicand".to_string(),
        ));
    }
    let start = start_timer!(|| "sample random zero mle list");

    let mut multiplicands = Vec::with_capacity(degree);
    for _ in 0..degree {
        multiplicands.push(Vec::with_capacity(1 << nv))
    }
    for _ in 0..(1 << nv) {
        multiplicands[0].push(F::zero());
        for e in multiplicands.iter_mut().skip(1) {
            e.push(F::rand(rng));
        }
    }

    let list = multiplicands
        .into_iter()
        .map(|x| Arc::new(DenseMultilinearExtension::from_evaluations_vec(nv, x)))
        .collect();

    end_timer!(start);
    Ok(list)
}

impl CpuBackend {
    /// Evaluate the virtual polynomial at point `point`.
    /// Returns an error is point.len() does not match `num_variables`.
    ///
    /// CPU-only evaluator for tests and diagnostics, not a backend requirement.
    /// Evaluate each unique factor once, retaining repeated factors'
    /// multiplicity.
    pub fn evaluate_vp<F: PrimeField>(
        &self,
        poly: &VirtualPolynomial<F, Arc<DenseMultilinearExtension<F>>>,
        point: &[F],
    ) -> Result<F, BackendError> {
        if poly.aux_info.num_variables != point.len() {
            return Err(BackendError::InvalidParameters(format!(
                "wrong number of variables {} vs {}",
                poly.aux_info.num_variables,
                point.len()
            )));
        }
        let evaluations = poly
            .flattened_ml_extensions()
            .iter()
            .map(|mle| self.evaluate_mle(mle, point))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(poly
            .products
            .iter()
            .map(|(coefficient, factors)| {
                *coefficient * factors.iter().map(|&i| evaluations[i]).product::<F>()
            })
            .sum())
    }
}

// Random generation and printing remain explicitly CPU-only.
impl<F: PrimeField> VirtualPolynomial<F, Arc<DenseMultilinearExtension<F>>> {
    /// Sample a random virtual polynomial, return the polynomial and its sum.
    pub fn rand<R: RngCore>(
        nv: usize,
        num_multiplicands_range: (usize, usize),
        num_products: usize,
        rng: &mut R,
    ) -> Result<(Self, F), BackendError> {
        let start = start_timer!(|| "sample random virtual polynomial");

        let mut sum = F::zero();
        let mut poly = VirtualPolynomial::new(nv);
        for _ in 0..num_products {
            let num_multiplicands =
                rng.gen_range(num_multiplicands_range.0..num_multiplicands_range.1);
            let (product, product_sum) = random_mle_list(nv, num_multiplicands, rng);
            let coefficient = F::rand(rng);
            poly.add_mle_list(product.into_iter(), coefficient)?;
            sum += product_sum * coefficient;
        }

        end_timer!(start);
        Ok((poly, sum))
    }

    /// Sample a random virtual polynomial that evaluates to zero everywhere
    /// over the boolean hypercube.
    pub fn rand_zero<R: RngCore>(
        nv: usize,
        num_multiplicands_range: (usize, usize),
        num_products: usize,
        rng: &mut R,
    ) -> Result<Self, BackendError> {
        let mut poly = VirtualPolynomial::new(nv);
        for _ in 0..num_products {
            let num_multiplicands =
                rng.gen_range(num_multiplicands_range.0..num_multiplicands_range.1);
            let product = random_zero_mle_list(nv, num_multiplicands, rng)?;
            let coefficient = F::rand(rng);
            poly.add_mle_list(product.into_iter(), coefficient)?;
        }

        Ok(poly)
    }

    /// Print out the evaluation map for testing. Panic if the num_vars > 5.
    pub fn print_evals(&self) {
        if self.aux_info.num_variables > 5 {
            panic!("this function is used for testing only. cannot print more than 5 num_vars")
        }
        for i in 0..1 << self.aux_info.num_variables {
            let point = bit_decompose(i, self.aux_info.num_variables);
            let point_fr: Vec<F> = point.iter().map(|&x| F::from(x)).collect();
            println!(
                "{} {}",
                i,
                CpuBackend.evaluate_vp(self, point_fr.as_ref()).unwrap()
            )
        }
        println!()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::common::eq::eq_eval;
    use ark_bls12_381::Fr;
    use ark_ff::{One, UniformRand, Zero};
    use ark_poly::Polynomial;
    use ark_std::test_rng;
    use std::sync::Arc;

    fn sample_mle() -> Arc<DenseMultilinearExtension<Fr>> {
        Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            3,
            [2u64, 3, 5, 7, 11, 13, 17, 19]
                .into_iter()
                .map(Fr::from)
                .collect(),
        ))
    }

    #[test]
    fn virtual_evaluation_preserves_shared_factors() -> Result<(), BackendError> {
        let mle = sample_mle();
        let mut poly = VirtualPolynomial::new_from_mle(&mle, Fr::from(2u64));
        poly.add_mle_list([mle.clone(), mle.clone()], Fr::from(3u64))?;
        let point = vec![Fr::from(2u64), Fr::from(3u64), Fr::from(5u64)];
        let r = [Fr::from(7u64), Fr::from(11u64), Fr::from(13u64)];
        let value = mle.evaluate(&point);
        let expected = Fr::from(2u64) * value + Fr::from(3u64) * value * value;
        assert_eq!(CpuBackend.evaluate_vp(&poly, &point)?, expected);
        let eq = CpuBackend.build_eq_x_r(&r)?;
        assert_eq!(eq.evaluate(&point), eq_eval(&point, &r)?);
        Ok(())
    }

    #[test]
    fn invalid_virtual_dimensions_are_errors() {
        let mle = sample_mle();
        let poly = VirtualPolynomial::new_from_mle(&mle, Fr::from(1u64));
        assert!(matches!(
            CpuBackend.evaluate_vp(&poly, &[]),
            Err(BackendError::InvalidParameters(_))
        ));
    }

    #[test]
    fn test_handle_identity_and_repeated_factors() -> Result<(), BackendError> {
        let mle = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            2,
            vec![
                Fr::from(2u64),
                Fr::from(3u64),
                Fr::from(5u64),
                Fr::from(7u64),
            ],
        ));
        let equal_contents = Arc::new(mle.as_ref().clone());
        let mut poly = VirtualPolynomial::new_from_mle(&mle, Fr::from(2u64));
        poly.add_mle_list(
            [mle.clone(), mle.clone(), equal_contents.clone()],
            Fr::from(3u64),
        )?;

        let point = vec![Fr::from(2u64), Fr::from(3u64)];
        let value = mle.evaluate(&point);
        let expected = Fr::from(2u64) * value + Fr::from(3u64) * value * value * value;
        assert_eq!(CpuBackend.evaluate_vp(&poly, &point)?, expected);

        let mut cloned = poly.clone();
        cloned.mul_by_mle(mle, Fr::from(5u64))?;
        assert_eq!(
            CpuBackend.evaluate_vp(&cloned, &point)?,
            expected * value * Fr::from(5u64)
        );
        assert_eq!(CpuBackend.evaluate_vp(&poly, &point)?, expected);
        Ok(())
    }

    #[test]
    fn test_random_zero_multiplicands() -> Result<(), BackendError> {
        let mut rng = test_rng();
        assert!(matches!(
            random_zero_mle_list::<Fr, _>(2, 0, &mut rng),
            Err(BackendError::InvalidParameters(_))
        ));
        assert!(matches!(
            VirtualPolynomial::<Fr, _>::rand(2, (0, 1), 1, &mut rng),
            Err(BackendError::InvalidParameters(_))
        ));
        assert!(matches!(
            VirtualPolynomial::<Fr, _>::rand_zero(2, (0, 1), 1, &mut rng),
            Err(BackendError::InvalidParameters(_))
        ));
        let empty = VirtualPolynomial::<Fr, _>::rand_zero(2, (0, 1), 0, &mut rng)?;
        assert_eq!(
            CpuBackend.evaluate_vp(&empty, &[Fr::zero(), Fr::one()])?,
            Fr::zero()
        );
        let poly = VirtualPolynomial::<Fr, _>::rand_zero(2, (1, 3), 2, &mut rng)?;
        for x in [Fr::zero(), Fr::one()] {
            for y in [Fr::zero(), Fr::one()] {
                assert_eq!(CpuBackend.evaluate_vp(&poly, &[x, y])?, Fr::zero());
            }
        }
        Ok(())
    }

    #[test]
    fn test_virtual_polynomial_additions() -> Result<(), BackendError> {
        let mut rng = test_rng();
        for nv in 2..5 {
            for num_products in 2..5 {
                let base: Vec<Fr> = (0..nv).map(|_| Fr::rand(&mut rng)).collect();

                let (a, _a_sum) =
                    VirtualPolynomial::<Fr, _>::rand(nv, (2, 3), num_products, &mut rng)?;
                let (b, _b_sum) =
                    VirtualPolynomial::<Fr, _>::rand(nv, (2, 3), num_products, &mut rng)?;
                let c = &a + &b;

                assert_eq!(
                    CpuBackend.evaluate_vp(&a, base.as_ref())?
                        + CpuBackend.evaluate_vp(&b, base.as_ref())?,
                    CpuBackend.evaluate_vp(&c, base.as_ref())?
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_virtual_polynomial_mul_by_mle() -> Result<(), BackendError> {
        let mut rng = test_rng();
        for nv in 2..5 {
            for num_products in 2..5 {
                let base: Vec<Fr> = (0..nv).map(|_| Fr::rand(&mut rng)).collect();

                let (a, _a_sum) =
                    VirtualPolynomial::<Fr, _>::rand(nv, (2, 3), num_products, &mut rng)?;
                let (b, _b_sum) = random_mle_list(nv, 1, &mut rng);
                let b_mle = b[0].clone();
                let coeff = Fr::rand(&mut rng);
                let b_vp = VirtualPolynomial::new_from_mle(&b_mle, coeff);

                let mut c = a.clone();

                c.mul_by_mle(b_mle, coeff)?;

                assert_eq!(
                    CpuBackend.evaluate_vp(&a, base.as_ref())?
                        * CpuBackend.evaluate_vp(&b_vp, base.as_ref())?,
                    CpuBackend.evaluate_vp(&c, base.as_ref())?
                );
            }
        }

        Ok(())
    }
}
