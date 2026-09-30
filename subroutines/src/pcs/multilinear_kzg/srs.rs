// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Implementing Structured Reference Strings for multilinear polynomial KZG
use crate::{PCSError, StructuredReferenceString};
use ark_ec::{pairing::Pairing, scalar_mul::ScalarMul, AffineRepr, CurveGroup};
use ark_ff::{Field, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{
    collections::LinkedList, end_timer, format, rand::Rng, start_timer, string::ToString, vec::Vec,
    UniformRand,
};
use backend::{common::eq::eq_eval, MultilinearKzgBackend};
use core::iter::FromIterator;

/// Generate eq(t,x), a product of multilinear polynomials with fixed t.
/// eq(a,b) is the multilinear extension of equality on {0,1}^num_vars:
/// it evaluates to 1 when a and b in {0,1}^num_vars are equal.
/// Returns one full-dimensional Boolean evaluation table for each factor
/// eq(t_i, x_i). Coordinate zero is the low table bit; an empty point
/// returns no factor tables.
fn build_eq_factor_tables<F: Field>(t: &[F]) -> Vec<Vec<F>> {
    let start = start_timer!(|| "eq extension");

    let dim = t.len();
    let mut result = Vec::with_capacity(dim);
    for (i, &ti) in t.iter().enumerate() {
        let mut poly = Vec::with_capacity(1 << dim);
        for x in 0..(1 << dim) {
            let xi = if x >> i & 1 == 1 { F::one() } else { F::zero() };
            let ti_xi = ti * xi;
            poly.push(ti_xi + ti_xi - xi - ti + F::one());
        }
        result.push(poly);
    }

    end_timer!(start);
    result
}

/// fix first `pad` variables of `evaluations` represented in evaluation form
/// to zero by stride projection.
/// Requires a nonempty power-of-two table and pad <= log2(evaluations.len()).
fn fix_first_variables_to_zero<F: Field>(evaluations: &[F], pad: usize) -> Vec<F> {
    if pad == 0 {
        return evaluations.to_vec();
    }
    let nv = ark_std::log2(evaluations.len()) as usize - pad;
    (0..(1 << nv)).map(|x| evaluations[x << pad]).collect()
}

/// Evaluations over {0,1}^n for G1 or G2
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct Evaluations<C: AffineRepr> {
    /// The evaluations.
    pub evals: Vec<C>,
}

/// Universal Parameter
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct MultilinearUniversalParams<E: Pairing> {
    /// number of variables
    pub num_vars: usize,
    /// `pp_{0}`, `pp_{1}`, ...,pp_{nu_vars} defined
    /// by XZZPD19 where pp_{nv-0}=g and
    /// pp_{nv-i}=g^{eq((t_1,..t_i),(X_1,..X_i))}
    pub powers_of_g: Vec<Evaluations<E::G1Affine>>,
    /// generator for G1
    pub g: E::G1Affine,
    /// generator for G2
    pub h: E::G2Affine,
    /// h^randomness: h^t1, h^t2, ..., **h^{t_nv}**
    pub h_mask: Vec<E::G2Affine>,
}

/// Prover Parameters
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct MultilinearProverParam<E: Pairing> {
    /// number of variables
    pub num_vars: usize,
    /// `pp_{0}`, `pp_{1}`, ...,pp_{nu_vars} defined
    /// by XZZPD19 where pp_{nv-0}=g and
    /// pp_{nv-i}=g^{eq((t_1,..t_i),(X_1,..X_i))}
    pub powers_of_g: Vec<Evaluations<E::G1Affine>>,
    /// generator for G1
    pub g: E::G1Affine,
    /// generator for G2
    pub h: E::G2Affine,
}

impl<E: Pairing> MultilinearProverParam<E> {
    /// Consume host layers into backend-owned prepared prover resources.
    pub fn prepare<B: MultilinearKzgBackend<E>>(
        self,
        backend: &B,
    ) -> Result<B::PreparedProverParam, PCSError> {
        Ok(backend.prepare_prover_param(
            self.num_vars,
            self.powers_of_g.into_iter().map(|layer| layer.evals),
        )?)
    }
}

/// Verifier Parameters
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct MultilinearVerifierParam<E: Pairing> {
    /// number of variables
    pub num_vars: usize,
    /// generator of G1
    pub g: E::G1Affine,
    /// generator of G2
    pub h: E::G2Affine,
    /// h^randomness: h^t1, h^t2, ..., **h^{t_nv}**
    pub h_mask: Vec<E::G2Affine>,
}

impl<E: Pairing> StructuredReferenceString<E> for MultilinearUniversalParams<E> {
    type ProverParam = MultilinearProverParam<E>;
    type VerifierParam = MultilinearVerifierParam<E>;

    /// Trim the universal parameters to specialize the public parameters
    /// for multilinear polynomials to the given `supported_num_vars`, and
    /// returns committer key and verifier key. `supported_num_vars` should
    /// be in range `0..=params.num_vars`.
    fn trim(
        &self,
        supported_num_vars: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError> {
        if supported_num_vars > self.num_vars {
            return Err(PCSError::InvalidParameters(format!(
                "SRS does not support target number of vars {}",
                supported_num_vars
            )));
        }

        let to_reduce = self.num_vars - supported_num_vars;
        let ck = Self::ProverParam {
            powers_of_g: self.powers_of_g[to_reduce..].to_vec(),
            g: self.g,
            h: self.h,
            num_vars: supported_num_vars,
        };
        let vk = Self::VerifierParam {
            num_vars: supported_num_vars,
            g: self.g,
            h: self.h,
            h_mask: self.h_mask[to_reduce..].to_vec(),
        };
        Ok((ck, vk))
    }

    /// Build SRS for testing.
    /// WARNING: THIS FUNCTION IS FOR TESTING PURPOSE ONLY.
    /// THE OUTPUT SRS SHOULD NOT BE USED IN PRODUCTION.
    fn gen_srs_for_testing<R: Rng>(rng: &mut R, num_vars: usize) -> Result<Self, PCSError> {
        if num_vars == 0 {
            return Err(PCSError::InvalidParameters(
                "constant polynomial not supported".to_string(),
            ));
        }

        let total_timer = start_timer!(|| "SRS generation");

        let pp_generation_timer = start_timer!(|| "Prover Param generation");

        let g = E::G1::rand(rng);
        let h = E::G2::rand(rng);

        let mut powers_of_g = Vec::new();

        let t: Vec<_> = (0..num_vars).map(|_| E::ScalarField::rand(rng)).collect();

        let mut eq: LinkedList<Vec<E::ScalarField>> =
            LinkedList::from_iter(build_eq_factor_tables(&t));
        let mut eq_arr = LinkedList::new();
        let mut base = eq.pop_back().unwrap();

        for i in (0..num_vars).rev() {
            eq_arr.push_front(fix_first_variables_to_zero(&base, i));
            if i != 0 {
                let mul = eq.pop_back().unwrap();
                base = base
                    .into_iter()
                    .zip(mul.into_iter())
                    .map(|(a, b)| a * b)
                    .collect();
            }
        }

        let mut pp_powers = Vec::new();
        for i in 0..num_vars {
            let eq = eq_arr.pop_front().unwrap();
            let pp_k_powers = (0..(1 << (num_vars - i))).map(|x| eq[x]);
            pp_powers.extend(pp_k_powers);
        }

        let pp_g = g.batch_mul(&pp_powers);

        let mut start = 0;
        for i in 0..num_vars {
            let size = 1 << (num_vars - i);
            let pp_k_g = Evaluations {
                evals: pp_g[start..(start + size)].to_vec(),
            };
            // check correctness of pp_k_g
            let t_eval_0 = eq_eval(&vec![E::ScalarField::zero(); num_vars - i], &t[i..num_vars])
                .map_err(|_| {
                    PCSError::InvalidParameters("x and y have different length".to_string())
                })?;
            assert_eq!((g * t_eval_0).into(), pp_k_g.evals[0]);
            powers_of_g.push(pp_k_g);
            start += size;
        }
        let gg = Evaluations {
            evals: [g.into_affine()].to_vec(),
        };
        powers_of_g.push(gg);

        end_timer!(pp_generation_timer);

        let vp_generation_timer = start_timer!(|| "VP generation");
        let h_mask = h.batch_mul(&t);
        end_timer!(vp_generation_timer);
        end_timer!(total_timer);
        Ok(Self {
            num_vars,
            powers_of_g,
            g: g.into_affine(),
            h: h.into_affine(),
            h_mask,
        })
    }
}

#[cfg(all(test, feature = "cpu"))]
mod test {
    use super::*;
    use crate::{MultilinearKzgPCS, PolynomialCommitmentScheme};
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_poly::{DenseMultilinearExtension, Polynomial};
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use backend::cpu::CpuBackend;
    use std::sync::Arc;

    #[test]
    fn serialized_srs_trims_matching_keys() -> Result<(), PCSError> {
        type PCS = MultilinearKzgPCS<Bls12_381>;
        let mut rng = StdRng::seed_from_u64(7);
        let srs = PCS::gen_srs_for_testing(&mut rng, 3)?;
        let (full_ck, full_vk) = srs.trim(3)?;
        let backend = CpuBackend;
        let full_ck = full_ck.prepare(&backend)?;
        let mut bytes = Vec::new();
        srs.serialize_compressed(&mut bytes).unwrap();
        let mut reader = bytes.as_slice();
        let restored =
            MultilinearUniversalParams::<Bls12_381>::deserialize_compressed(&mut reader).unwrap();
        assert!(reader.is_empty());

        for num_vars in 0..=3 {
            let (ck, vk) = restored.trim(num_vars)?;
            let ck = ck.prepare(&backend)?;
            let polynomial = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                (0..(1u64 << num_vars))
                    .map(|i| Fr::from(i * i + 2))
                    .collect(),
            ));
            let commitment = PCS::commit(&backend, &ck, &polynomial)?;
            assert_eq!(commitment, PCS::commit(&backend, &full_ck, &polynomial)?);
            let point = (0..num_vars)
                .map(|i| Fr::from(i as u64 + 3))
                .collect::<Vec<_>>();
            let (proof, value) = PCS::open(&backend, &ck, &polynomial, &point)?;
            assert_eq!(value, polynomial.evaluate(&point));
            assert!(PCS::verify(&vk, &commitment, &point, &value, &proof)?);
            assert!(PCS::verify(&full_vk, &commitment, &point, &value, &proof)?);
        }
        assert!(matches!(
            restored.trim(4),
            Err(PCSError::InvalidParameters(_))
        ));
        Ok(())
    }
}
