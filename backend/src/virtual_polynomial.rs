// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.
//
// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! This module defines our main mathematical object `VirtualPolynomial`; and
//! various functions associated with it.
//!
//! Generic sums of products of shared multilinear polynomial handles.

use crate::{BackendError, Mle};
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use ark_std::{end_timer, start_timer};
use std::{cmp::max, collections::HashMap, fmt, marker::PhantomData, ops::Add};

#[rustfmt::skip]
/// A virtual polynomial is a sum of products of multilinear polynomials;
/// where each multilinear polynomial is represented by a shared handle `P`.
///
/// * Number of products n = `polynomial.products.len()`,
/// * Number of multiplicands of ith product m_i =
///   `polynomial.products[i].1.len()`,
/// * Coefficient of ith product c_i = `polynomial.products[i].0`
///
/// The resulting polynomial is
///
/// $$ \sum_{i=0}^{n} c_i \cdot \prod_{j=0}^{m_i} P_{ij} $$
///
/// Example:
///  f = c0 * f0 * f1 * f2 + c1 * f3 * f4
/// where f0 ... f4 are multilinear polynomials
///
/// - flattened_ml_extensions stores shared handles for
///   f0, f1, f2, f3 and f4
/// - products is 
///     \[ 
///         (c0, \[0, 1, 2\]), 
///         (c1, \[3, 4\]) 
///     \]
/// - polynomial_lookup maps each handle identity to its storage index
///
#[derive(Clone, PartialEq)]
pub struct VirtualPolynomial<F: PrimeField, P: Mle<F>> {
    /// Aux information about the multilinear polynomial
    pub aux_info: VPAuxInfo<F>,
    /// list of reference to products (as usize) of multilinear extension
    pub products: Vec<(F, Vec<usize>)>,
    /// Stores shared multilinear polynomial handles which product multiplicands
    /// can refer to.
    flattened_ml_extensions: Vec<P>,
    /// Identity-based index; contents are never hashed.
    polynomial_lookup: HashMap<P::Id, usize>,
}

impl<F: PrimeField, P: Mle<F>> Default for VirtualPolynomial<F, P> {
    fn default() -> Self {
        Self::new(0)
    }
}

impl<F: PrimeField, P: Mle<F> + fmt::Debug> fmt::Debug for VirtualPolynomial<F, P>
where
    P::Id: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VirtualPolynomial")
            .field("aux_info", &self.aux_info)
            .field("products", &self.products)
            .field("flattened_ml_extensions", &self.flattened_ml_extensions)
            .field("polynomial_lookup", &self.polynomial_lookup)
            .finish()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, CanonicalSerialize)]
/// Auxiliary information about the multilinear polynomial
pub struct VPAuxInfo<F: PrimeField> {
    /// max number of multiplicands in each product
    pub max_degree: usize,
    /// number of variables of the polynomial
    pub num_variables: usize,
    /// Associated field
    #[doc(hidden)]
    pub phantom: PhantomData<F>,
}

impl<F: PrimeField, P: Mle<F>> Add for &VirtualPolynomial<F, P> {
    type Output = VirtualPolynomial<F, P>;
    fn add(self, other: &VirtualPolynomial<F, P>) -> Self::Output {
        let start = start_timer!(|| "virtual poly add");
        let mut res = self.clone();
        for products in other.products.iter() {
            let cur = products
                .1
                .iter()
                .map(|&x| other.flattened_ml_extensions[x].clone());

            res.add_mle_list(cur, products.0)
                .expect("add product failed");
        }
        end_timer!(start);
        res
    }
}

impl<F: PrimeField, P: Mle<F>> VirtualPolynomial<F, P> {
    /// Creates an empty virtual polynomial with `num_variables`.
    pub fn new(num_variables: usize) -> Self {
        VirtualPolynomial {
            aux_info: VPAuxInfo {
                max_degree: 0,
                num_variables,
                phantom: PhantomData,
            },
            products: Vec::new(),
            flattened_ml_extensions: Vec::new(),
            polynomial_lookup: HashMap::new(),
        }
    }

    /// Shared factors in first-insertion order. Product indices refer to this
    /// slice; changes must go through `add_mle_list` or `mul_by_mle`.
    pub fn flattened_ml_extensions(&self) -> &[P] {
        &self.flattened_ml_extensions
    }

    /// Creates an new virtual polynomial from a MLE and its coefficient.
    pub fn new_from_mle(mle: &P, coefficient: F) -> Self {
        let mut hm = HashMap::new();
        hm.insert(mle.id(), 0);

        VirtualPolynomial {
            aux_info: VPAuxInfo {
                // The max degree is the max degree of any individual variable
                max_degree: 1,
                num_variables: mle.num_vars(),
                phantom: PhantomData,
            },
            // here `0` points to the first polynomial of `flattened_ml_extensions`
            products: vec![(coefficient, vec![0])],
            flattened_ml_extensions: vec![mle.clone()],
            polynomial_lookup: hm,
        }
    }

    /// Add a product of list of multilinear extensions to self
    /// Returns an error if the list is empty, or the MLE has a different
    /// `num_vars` from self.
    ///
    /// The MLEs will be multiplied together, and then multiplied by the scalar
    /// `coefficient`.
    pub fn add_mle_list(
        &mut self,
        mle_list: impl IntoIterator<Item = P>,
        coefficient: F,
    ) -> Result<(), BackendError> {
        let mle_list: Vec<P> = mle_list.into_iter().collect();

        if mle_list.is_empty() {
            return Err(BackendError::InvalidParameters(
                "input mle_list is empty".to_string(),
            ));
        }

        for mle in &mle_list {
            if mle.num_vars() != self.aux_info.num_variables {
                return Err(BackendError::InvalidParameters(format!(
                    "product has a multiplicand with wrong number of variables {} vs {}",
                    mle.num_vars(),
                    self.aux_info.num_variables
                )));
            }
        }

        self.aux_info.max_degree = max(self.aux_info.max_degree, mle_list.len());
        let mut indexed_product = Vec::with_capacity(mle_list.len());
        for mle in mle_list {
            let mle_id = mle.id();
            if let Some(index) = self.polynomial_lookup.get(&mle_id) {
                indexed_product.push(*index)
            } else {
                let curr_index = self.flattened_ml_extensions.len();
                self.flattened_ml_extensions.push(mle);
                self.polynomial_lookup.insert(mle_id, curr_index);
                indexed_product.push(curr_index);
            }
        }
        self.products.push((coefficient, indexed_product));
        Ok(())
    }

    /// Multiple the current VirtualPolynomial by an MLE:
    /// - add the MLE to the MLE list;
    /// - multiple each product by MLE and its coefficient.
    ///
    /// Returns an error if the MLE has a different `num_vars` from self.
    pub fn mul_by_mle(&mut self, mle: P, coefficient: F) -> Result<(), BackendError> {
        let start = start_timer!(|| "mul by mle");

        if mle.num_vars() != self.aux_info.num_variables {
            return Err(BackendError::InvalidParameters(format!(
                "product has a multiplicand with wrong number of variables {} vs {}",
                mle.num_vars(),
                self.aux_info.num_variables
            )));
        }

        let mle_id = mle.id();

        // check if this mle already exists in the virtual polynomial
        let mle_index = match self.polynomial_lookup.get(&mle_id) {
            Some(&p) => p,
            None => {
                self.polynomial_lookup
                    .insert(mle_id, self.flattened_ml_extensions.len());
                self.flattened_ml_extensions.push(mle);
                self.flattened_ml_extensions.len() - 1
            },
        };

        for (prod_coef, indices) in self.products.iter_mut() {
            // - add the MLE to the MLE list;
            // - multiple each product by MLE and its coefficient.
            indices.push(mle_index);
            *prod_coef *= coefficient;
        }

        // increase the max degree by one as the MLE has degree 1.
        self.aux_info.max_degree += 1;
        end_timer!(start);
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ark_bls12_381::Fr;
    use std::sync::Arc;

    // An immutable host polynomial storage handle exposing only its metadata.
    // Clones share storage; equal contents in distinct allocations remain distinct.
    #[derive(Debug, PartialEq)]
    struct Storage {
        num_vars: usize,
        evaluations: Vec<Fr>,
    }

    #[derive(Clone, Debug, PartialEq)]
    struct MetadataHandle(Arc<Storage>);

    impl MetadataHandle {
        fn new(num_vars: usize, evaluations: &[u64]) -> Self {
            assert_eq!(evaluations.len(), 1 << num_vars);
            Self(Arc::new(Storage {
                num_vars,
                evaluations: evaluations.iter().copied().map(Fr::from).collect(),
            }))
        }
    }

    impl Mle<Fr> for MetadataHandle {
        type Id = usize;
        fn num_vars(&self) -> usize {
            self.0.num_vars
        }
        fn id(&self) -> Self::Id {
            Arc::as_ptr(&self.0) as usize
        }
    }

    #[test]
    fn handle_identity_and_repeated_factors() -> Result<(), BackendError> {
        let mle = MetadataHandle::new(2, &[2, 3, 5, 7]);
        let equal_contents = MetadataHandle::new(2, &[2, 3, 5, 7]);
        let mut poly = VirtualPolynomial::new_from_mle(&mle, Fr::from(2u64));
        poly.add_mle_list(
            [mle.clone(), mle.clone(), equal_contents.clone()],
            Fr::from(3u64),
        )?;
        let factors = poly.flattened_ml_extensions();
        assert_eq!(factors.len(), 2);
        assert!(Arc::ptr_eq(&factors[0].0, &mle.0));
        assert!(Arc::ptr_eq(&factors[1].0, &equal_contents.0));
        assert_ne!(factors[0].id(), factors[1].id());
        assert_eq!(poly.products[1].1, vec![0, 0, 1]);
        assert_eq!(poly.aux_info.max_degree, 3);
        let mut cloned = poly.clone();
        cloned.mul_by_mle(mle, Fr::from(5u64))?;
        assert_eq!(cloned.flattened_ml_extensions().len(), 2);
        assert_eq!(cloned.products[1].1, vec![0, 0, 1, 0]);
        assert_eq!(cloned.products[1].0, Fr::from(15u64));
        assert_eq!(poly.products[1].1, vec![0, 0, 1]);
        Ok(())
    }

    #[test]
    fn invalid_factors() {
        let mut poly = VirtualPolynomial::<Fr, MetadataHandle>::new(2);
        let wrong = MetadataHandle::new(1, &[2, 3]);
        assert!(matches!(
            poly.add_mle_list([], Fr::from(1u64)),
            Err(BackendError::InvalidParameters(_))
        ));
        assert!(matches!(
            poly.add_mle_list([wrong.clone()], Fr::from(1u64)),
            Err(BackendError::InvalidParameters(_))
        ));
        assert!(matches!(
            poly.mul_by_mle(wrong, Fr::from(1u64)),
            Err(BackendError::InvalidParameters(_))
        ));
    }

    #[test]
    fn add_mle_list_error_preserves_state() -> Result<(), BackendError> {
        let original = MetadataHandle::new(2, &[2, 3, 5, 7]);
        let added = MetadataHandle::new(2, &[11, 13, 17, 19]);
        let wrong = MetadataHandle::new(1, &[23, 29]);
        let mut poly = VirtualPolynomial::new_from_mle(&original, Fr::from(2u64));
        let before = poly.clone();
        for factors in [
            vec![],
            vec![wrong.clone(), added.clone()],
            vec![added.clone(), wrong],
        ] {
            assert!(matches!(
                poly.add_mle_list(factors, Fr::from(3u64)),
                Err(BackendError::InvalidParameters(_))
            ));
            assert_eq!(poly, before);
            assert!(Arc::ptr_eq(
                &poly.flattened_ml_extensions()[0].0,
                &original.0
            ));
        }
        poly.add_mle_list([added.clone(), added], Fr::from(3u64))?;
        assert_eq!(poly.flattened_ml_extensions().len(), 2);
        assert_eq!(poly.products[1].1, vec![1, 1]);
        Ok(())
    }
}
