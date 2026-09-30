// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.
//
// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Resident multilinear polynomial metadata and execution interface.

use crate::BackendError;
use ark_ff::PrimeField;
use std::hash::Hash;

/// A shared handle to an immutable multilinear polynomial over `F`.
///
/// Cloning a handle must share storage, not copy polynomial data. Its identity
/// and variable count must remain stable while any clone is alive. Distinct
/// logical polynomials must have distinct identities, even if their contents
/// agree. Device implementations must distinguish contexts and view layouts.
///
/// Only metadata is exposed: structural operations on a virtual polynomial do
/// not require host access to its evaluations.
pub trait Mle<F: PrimeField>: Clone {
    type Id: Copy + Eq + Hash;

    fn num_vars(&self) -> usize;
    fn id(&self) -> Self::Id;
}

/// Computes with MLEs without requiring host access to their evaluations.
///
/// Polynomial outputs belong to this backend and inputs remain immutable.
/// A scalar result must be ready for host consumption when its call returns.
pub trait MleBackend<F: PrimeField> {
    type Mle: Mle<F>;

    /// Construct a resident MLE from an owned low-bit-first evaluation table.
    /// The table must contain exactly 2^num_vars values, including one value
    /// for constants.
    fn mle_from_evaluations(
        &self,
        num_vars: usize,
        evaluations: Vec<F>,
    ) -> Result<Self::Mle, BackendError>;

    /// Sum `coefficients[i] * polynomials[i]` into a new resident MLE.
    /// Inputs must be nonempty, equally sized, and have matching dimensions.
    /// Even a zero result retains their variable count.
    fn linear_combination(
        &self,
        polynomials: &[Self::Mle],
        coefficients: &[F],
    ) -> Result<Self::Mle, BackendError>;

    /// Evaluate at a complete point, including the empty point for constants.
    fn evaluate_mle(&self, mle: &Self::Mle, point: &[F]) -> Result<F, BackendError>;

    /// This function builds the eq(x, r) polynomial for any given r.
    ///
    /// Evaluate
    /// `eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))`
    /// over r, which is
    /// `eq(x,r) = \prod_i=1^num_var (x_i * r_i + (1-x_i)*(1-r_i))`.
    /// An empty `r` is rejected.
    fn build_eq_x_r(&self, r: &[F]) -> Result<Self::Mle, BackendError>;

    /// Compute multilinear fractional polynomial s.t. frac(x) = f1(x) * ... * fk(x)
    /// / (g1(x) * ... * gk(x)) for all x \in {0,1}^n.
    ///
    /// The caller must provide nonempty, equally sized lists whose polynomials
    /// have the same number of variables. Zero denominator evaluations return
    /// an error.
    fn compute_frac_poly(
        &self,
        fxs: &[Self::Mle],
        gxs: &[Self::Mle],
    ) -> Result<Self::Mle, BackendError>;

    /// Compute the product polynomial `prod(x)` such that
    /// `prod(u,z) = p1(u,z) * p2(u,z)` on the Boolean hypercube {0,1}^n, where
    /// `p1(u,z) = (1-z)*frac(0,u) + z*prod(0,u)` and
    /// `p2(u,z) = (1-z)*frac(1,u) + z*prod(1,u)`.
    /// Here `u` contains the first n-1 coordinates and `z` is the last coordinate.
    ///
    /// Requires at least one variable. The final table entry is zero.
    /// Cost: linear in N, where N = 2^n.
    fn compute_product_poly(&self, frac_poly: &Self::Mle) -> Result<Self::Mle, BackendError>;

    /// Returns two lists of MLEs:
    /// - numerators = (a1, ..., ak)
    /// - denominators = (b1, ..., bk)
    ///
    /// where
    /// - beta and gamma are challenges,
    /// - (f1, ..., fk), (g1, ..., gk),
    /// - (s_id1, ..., s_idk), (perm1, ..., permk) are MLEs.
    ///
    /// - ai(x) is the MLE for `fi(x) + beta * s_id_i(x) + gamma`.
    /// - bi(x) is the MLE for `gi(x) + beta * perm_i(x) + gamma`.
    ///
    /// The caller must provide nonempty, equally sized lists whose polynomials
    /// have the same number of variables.
    #[allow(clippy::type_complexity)]
    fn compute_nums_and_denoms(
        &self,
        beta: &F,
        gamma: &F,
        fxs: &[Self::Mle],
        gxs: &[Self::Mle],
        perms: &[Self::Mle],
    ) -> Result<(Vec<Self::Mle>, Vec<Self::Mle>), BackendError>;

    /// Build `[p1, p2]` for the product recurrence, preserving variable count.
    /// In evaluation-table order, p1 takes even entries and p2 odd entries,
    /// concatenating frac's entries before prod's. Both inputs must have the
    /// same positive variable count.
    fn compute_product_factors(
        &self,
        frac_poly: &Self::Mle,
        prod_poly: &Self::Mle,
    ) -> Result<[Self::Mle; 2], BackendError>;
}
