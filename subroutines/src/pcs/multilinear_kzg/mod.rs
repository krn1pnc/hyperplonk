// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Main module for multilinear KZG commitment scheme

pub(crate) mod batching;
pub(crate) mod srs;

use crate::{
    BatchProof, Commitment, MultilinearProverParam, MultilinearUniversalParams,
    MultilinearVerifierParam, PCSError, PolynomialCommitmentScheme, StructuredReferenceString,
};
use ark_ec::{pairing::Pairing, scalar_mul::ScalarMul, AffineRepr, CurveGroup};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{
    borrow::Borrow, end_timer, format, marker::PhantomData, rand::Rng, start_timer, vec::Vec, One,
};
use backend::{MultilinearKzgBackend, SumCheckProver};
use std::ops::Mul;
use transcript::IOPTranscript;

use self::batching::{batch_verify_internal, multi_open_internal};

/// KZG Polynomial Commitment Scheme on multilinear polynomials.
pub struct MultilinearKzgPCS<E: Pairing> {
    #[doc(hidden)]
    phantom: PhantomData<E>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug, PartialEq, Eq)]
/// proof of opening
pub struct MultilinearKzgProof<E: Pairing> {
    /// Evaluation of quotients
    pub proofs: Vec<E::G1Affine>,
}

impl<E: Pairing> PolynomialCommitmentScheme<E> for MultilinearKzgPCS<E> {
    // Parameters
    type ProverParam = MultilinearProverParam<E>;
    type VerifierParam = MultilinearVerifierParam<E>;
    type SRS = MultilinearUniversalParams<E>;
    // Evaluation domain
    type Point = Vec<E::ScalarField>;
    type Evaluation = E::ScalarField;
    // Commitments and proofs
    type Commitment = Commitment<E>;
    type Proof = MultilinearKzgProof<E>;
    type BatchProof = BatchProof<E, Self>;

    /// Build SRS for testing.
    ///
    /// `supported_num_vars` is the number of variables.
    ///
    /// WARNING: THIS FUNCTION IS FOR TESTING PURPOSE ONLY.
    /// THE OUTPUT SRS SHOULD NOT BE USED IN PRODUCTION.
    fn gen_srs_for_testing<R: Rng>(
        rng: &mut R,
        supported_num_vars: usize,
    ) -> Result<Self::SRS, PCSError> {
        MultilinearUniversalParams::<E>::gen_srs_for_testing(rng, supported_num_vars)
    }

    /// Trim the universal parameters to specialize the public parameters.
    /// `supported_num_vars` is the number of variables.
    fn trim(
        srs: impl Borrow<Self::SRS>,
        supported_num_vars: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError> {
        let (ml_ck, ml_vk) = srs.borrow().trim(supported_num_vars)?;

        Ok((ml_ck, ml_vk))
    }

    /// Generate a commitment for a polynomial.
    fn commit<B: MultilinearKzgBackend<E>>(
        backend: &B,
        prover_param: &B::PreparedProverParam,
        poly: &B::Mle,
    ) -> Result<Self::Commitment, PCSError> {
        Ok(Commitment(backend.commit(prover_param, poly)?))
    }

    fn multi_commit<B: MultilinearKzgBackend<E>>(
        backend: &B,
        prover_param: &B::PreparedProverParam,
        polynomials: &[B::Mle],
    ) -> Result<Vec<Self::Commitment>, PCSError> {
        Ok(backend
            .multi_commit(prover_param, polynomials)?
            .into_iter()
            .map(Commitment)
            .collect())
    }

    /// On input a polynomial `p` and a point `point`, outputs a proof for the
    /// same. This function does not need to take the evaluation value as an
    /// input.
    fn open<B: MultilinearKzgBackend<E>>(
        backend: &B,
        prover_param: &B::PreparedProverParam,
        polynomial: &B::Mle,
        point: &Self::Point,
    ) -> Result<(Self::Proof, Self::Evaluation), PCSError> {
        let (proofs, evaluation) = backend.open(prover_param, polynomial, point)?;
        Ok((MultilinearKzgProof { proofs }, evaluation))
    }

    /// Input a list of multilinear polynomial handles, and a same number of
    /// points, and a transcript, compute a multi-opening for all the polynomials.
    fn multi_open<B: MultilinearKzgBackend<E> + SumCheckProver<E::ScalarField>>(
        backend: &B,
        prover_param: &B::PreparedProverParam,
        polynomials: &[B::Mle],
        points: &[Self::Point],
        evals: &[Self::Evaluation],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<Self::BatchProof, PCSError> {
        multi_open_internal::<E, Self, B>(
            backend,
            prover_param,
            polynomials,
            points,
            evals,
            transcript,
        )
    }

    /// Verifies that `value` is the evaluation at `x` of the polynomial
    /// committed inside `comm`.
    ///
    /// This function takes
    /// - num_var number of pairing product.
    /// - num_var number of MSM
    fn verify(
        verifier_param: &Self::VerifierParam,
        commitment: &Self::Commitment,
        point: &Self::Point,
        value: &E::ScalarField,
        proof: &Self::Proof,
    ) -> Result<bool, PCSError> {
        verify_internal(verifier_param, commitment, point, value, proof)
    }

    /// Verifies that `value_i` is the evaluation at `x_i` of the polynomial
    /// `poly_i` committed inside `comm`.
    ///
    /// Requires equal, nonzero commitment, point, and proof-evaluation counts
    /// and a shared positive point dimension.
    fn batch_verify(
        verifier_param: &Self::VerifierParam,
        commitments: &[Self::Commitment],
        points: &[Self::Point],
        batch_proof: &Self::BatchProof,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<bool, PCSError> {
        batch_verify_internal(verifier_param, commitments, points, batch_proof, transcript)
    }
}

/// Verifies that `value` is the evaluation at `x` of the polynomial
/// committed inside `comm`.
///
/// This function takes
/// - num_var number of pairing product.
/// - num_var number of MSM
fn verify_internal<E: Pairing>(
    verifier_param: &MultilinearVerifierParam<E>,
    commitment: &Commitment<E>,
    point: &[E::ScalarField],
    value: &E::ScalarField,
    proof: &MultilinearKzgProof<E>,
) -> Result<bool, PCSError> {
    let verify_timer = start_timer!(|| "verify");
    let num_var = point.len();

    if num_var > verifier_param.num_vars {
        return Err(PCSError::InvalidParameters(format!(
            "point length ({}) exceeds param limit ({})",
            num_var, verifier_param.num_vars
        )));
    }

    let prepare_inputs_timer = start_timer!(|| "prepare pairing inputs");

    let h_mul = verifier_param.h.into_group().batch_mul(point);

    let ignored = verifier_param.num_vars - num_var;
    let h_vec: Vec<_> = (0..num_var)
        .map(|i| verifier_param.h_mask[ignored + i].into_group() - h_mul[i])
        .collect();
    let h_vec: Vec<E::G2Affine> = E::G2::normalize_batch(&h_vec);
    end_timer!(prepare_inputs_timer);

    let pairing_product_timer = start_timer!(|| "pairing product");

    let mut pairings: Vec<_> = proof
        .proofs
        .iter()
        .map(|&x| E::G1Prepared::from(x))
        .zip(h_vec.into_iter().take(num_var).map(E::G2Prepared::from))
        .collect();

    pairings.push((
        E::G1Prepared::from(
            (verifier_param.g.mul(*value) - commitment.0.into_group()).into_affine(),
        ),
        E::G2Prepared::from(verifier_param.h),
    ));

    let ps = pairings.iter().map(|(p, _)| p.clone());
    let hs = pairings.iter().map(|(_, h)| h.clone());

    let res = E::multi_pairing(ps, hs) == ark_ec::pairing::PairingOutput(E::TargetField::one());

    end_timer!(pairing_product_timer);
    end_timer!(verify_timer);
    Ok(res)
}

#[cfg(all(test, feature = "cpu"))]
mod test {
    use super::*;
    use ark_bls12_381::Bls12_381;
    use ark_ec::pairing::Pairing;
    use ark_poly::{DenseMultilinearExtension, MultilinearExtension, Polynomial};
    use ark_std::{
        rand::{rngs::StdRng, SeedableRng},
        test_rng,
        vec::Vec,
        UniformRand, Zero,
    };
    use backend::cpu::CpuBackend;
    use std::sync::Arc;

    type E = Bls12_381;
    type Fr = <E as Pairing>::ScalarField;

    fn test_single_helper<R: Rng>(
        params: &MultilinearUniversalParams<E>,
        poly: &Arc<DenseMultilinearExtension<Fr>>,
        rng: &mut R,
    ) -> Result<(), PCSError> {
        let nv = poly.num_vars();
        assert_ne!(nv, 0);
        let (ck, vk) = MultilinearKzgPCS::trim(params, nv)?;
        let backend = CpuBackend;
        let ck = ck.prepare(&backend)?;
        let point: Vec<_> = (0..nv).map(|_| Fr::rand(rng)).collect();
        let com = MultilinearKzgPCS::commit(&backend, &ck, poly)?;
        let (proof, value) = MultilinearKzgPCS::open(&backend, &ck, poly, &point)?;

        assert!(MultilinearKzgPCS::verify(
            &vk, &com, &point, &value, &proof
        )?);

        let value = Fr::rand(rng);
        assert!(!MultilinearKzgPCS::verify(
            &vk, &com, &point, &value, &proof
        )?);

        Ok(())
    }

    #[test]
    fn test_single_commit() -> Result<(), PCSError> {
        let mut rng = test_rng();

        let params = MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, 10)?;

        // normal polynomials
        let poly1 = Arc::new(DenseMultilinearExtension::rand(8, &mut rng));
        test_single_helper(&params, &poly1, &mut rng)?;

        // single-variate polynomials
        let poly2 = Arc::new(DenseMultilinearExtension::rand(1, &mut rng));
        test_single_helper(&params, &poly2, &mut rng)?;

        Ok(())
    }

    #[test]
    fn test_multi_commit_mixed_dimensions_and_order() -> Result<(), PCSError> {
        let mut rng = StdRng::seed_from_u64(11);
        let params = MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, 3)?;
        let (ck, vk) = MultilinearKzgPCS::<E>::trim(&params, 3)?;
        let backend = CpuBackend;
        let poly = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            2,
            [2u64, 3, 5, 7].into_iter().map(Fr::from).collect(),
        ));
        let polynomials = vec![
            poly.clone(),
            Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                0,
                vec![Fr::from(13u64)],
            )),
            Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                1,
                vec![Fr::from(17u64), Fr::from(19u64)],
            )),
            Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                3,
                (23u64..31).map(Fr::from).collect(),
            )),
            Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                2,
                vec![Fr::zero(); 4],
            )),
            poly,
        ];
        let expected = polynomials
            .iter()
            .map(|poly| {
                let bases = &ck.powers_of_g[ck.num_vars - poly.num_vars].evals;
                let sum: <E as Pairing>::G1 = bases
                    .iter()
                    .zip(&poly.evaluations)
                    .map(|(base, scalar)| base.mul(*scalar))
                    .sum();
                Commitment(sum.into_affine())
            })
            .collect::<Vec<_>>();
        let ck = ck.prepare(&backend)?;
        let commitments = MultilinearKzgPCS::<E>::multi_commit(&backend, &ck, &polynomials)?;
        assert_eq!(commitments, expected);
        for (poly, commitment) in polynomials.iter().zip(&commitments) {
            let point = (2..2 + poly.num_vars as u64)
                .map(Fr::from)
                .collect::<Vec<_>>();
            let (proof, value) = MultilinearKzgPCS::<E>::open(&backend, &ck, poly, &point)?;
            assert_eq!(value, poly.evaluate(&point));
            assert!(MultilinearKzgPCS::<E>::verify(
                &vk, commitment, &point, &value, &proof
            )?);
        }
        Ok(())
    }

    #[test]
    fn test_multi_commit_empty_and_singleton() -> Result<(), PCSError> {
        let mut rng = StdRng::seed_from_u64(12);
        let params = MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, 1)?;
        let (ck, _) = MultilinearKzgPCS::<E>::trim(&params, 1)?;
        let backend = CpuBackend;
        let g = ck.g;
        let ck = ck.prepare(&backend)?;
        assert_eq!(
            MultilinearKzgPCS::<E>::multi_commit(&backend, &ck, &[])?,
            Vec::<Commitment<E>>::new()
        );
        let constant = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            0,
            vec![Fr::from(29u64)],
        ));
        assert_eq!(
            MultilinearKzgPCS::<E>::multi_commit(&backend, &ck, &[constant])?,
            vec![Commitment(g.mul(Fr::from(29u64)).into_affine())]
        );
        Ok(())
    }

    #[test]
    fn test_multi_commit_rejects_unsupported_dimension() -> Result<(), PCSError> {
        let mut rng = StdRng::seed_from_u64(13);
        let params = MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, 2)?;
        let (ck, _) = MultilinearKzgPCS::<E>::trim(&params, 1)?;
        let backend = CpuBackend;
        let ck = ck.prepare(&backend)?;
        let supported = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            1,
            vec![Fr::from(2u64), Fr::from(3u64)],
        ));
        let unsupported = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            2,
            vec![Fr::from(5u64); 4],
        ));
        assert!(matches!(
            MultilinearKzgPCS::<E>::multi_commit(
                &backend,
                &ck,
                &[supported.clone(), unsupported, supported]
            ),
            Err(PCSError::InvalidParameters(_))
        ));
        Ok(())
    }

    #[test]
    fn opening_preserves_polynomial_across_points() -> Result<(), PCSError> {
        let mut rng = StdRng::seed_from_u64(17);
        let srs = MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, 3)?;
        let (host_param, verifier_param) = MultilinearKzgPCS::<E>::trim(&srs, 3)?;
        let backend = CpuBackend;
        let prepared = host_param.prepare(&backend)?;
        drop(srs);
        let polynomial = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            2,
            [2u64, 3, 5, 7].into_iter().map(Fr::from).collect(),
        ));
        let commitment = MultilinearKzgPCS::<E>::commit(&backend, &prepared, &polynomial)?;
        for (coordinates, expected) in [([0u64, 0], 2u64), ([1, 1], 7), ([2, 3], 19)] {
            let point = coordinates.into_iter().map(Fr::from).collect::<Vec<_>>();
            let (proof, value) =
                MultilinearKzgPCS::<E>::open(&backend, &prepared, &polynomial, &point)?;
            assert_eq!(value, Fr::from(expected));
            assert!(MultilinearKzgPCS::<E>::verify(
                &verifier_param,
                &commitment,
                &point,
                &value,
                &proof
            )?);
        }
        assert_eq!(
            MultilinearKzgPCS::<E>::commit(&backend, &prepared, &polynomial)?,
            commitment
        );
        assert!(matches!(
            MultilinearKzgPCS::<E>::open(&backend, &prepared, &polynomial, &vec![Fr::one()]),
            Err(PCSError::InvalidParameters(_))
        ));
        Ok(())
    }

    #[test]
    fn setup_commit_verify_constant_polynomial() {
        let mut rng = test_rng();

        // normal polynomials
        assert!(MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, 0).is_err());
    }
}
