// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Main module for the ZeroCheck protocol.

use std::fmt::Debug;

use crate::{PolyIOP, PolyIOPErrors, SumCheck};
use ark_ff::PrimeField;
use ark_std::{end_timer, start_timer};
use backend::{common::eq::eq_eval, SumCheckProver, VirtualPolynomial};
use transcript::IOPTranscript;

/// A zero check IOP subclaim for `f(x)` consists of the following:
///   - the initial challenge vector r which is used to build eq(x, r) in
///     SumCheck
///   - the random vector `v` to be evaluated
///   - the claimed evaluation of `f(v)`
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ZeroCheckSubClaim<F: PrimeField> {
    // the evaluation point
    pub point: Vec<F>,
    /// the expected evaluation
    pub expected_evaluation: F,
    // the initial challenge r which is used to build eq(x, r)
    pub init_challenge: Vec<F>,
}

/// A ZeroCheck for `f(x)` proves that `f(x) = 0` for all `x \in {0,1}^num_vars`
/// It is derived from SumCheck.
pub trait ZeroCheck<F: PrimeField>: SumCheck<F> {
    type ZeroCheckSubClaim: Clone + Debug + Default + PartialEq;
    type ZeroCheckProof: Clone + Debug + Default + PartialEq;

    /// Initialize the system with a transcript
    ///
    /// This function is optional -- in the case where a ZeroCheck is
    /// an building block for a more complex protocol, the transcript
    /// may be initialized by this complex protocol, and passed to the
    /// ZeroCheck prover/verifier.
    fn init_transcript() -> Self::Transcript;

    /// initialize the prover to argue for the sum of polynomial over
    /// {0,1}^`num_vars` is zero.
    fn prove<B: SumCheckProver<F>>(
        backend: &B,
        poly: &VirtualPolynomial<F, B::Mle>,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::ZeroCheckProof, PolyIOPErrors>;

    /// verify the claimed sum using the proof
    fn verify(
        proof: &Self::ZeroCheckProof,
        aux_info: &Self::VPAuxInfo,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::ZeroCheckSubClaim, PolyIOPErrors>;
}

impl<F: PrimeField> ZeroCheck<F> for PolyIOP<F> {
    type ZeroCheckSubClaim = ZeroCheckSubClaim<F>;
    type ZeroCheckProof = Self::SumCheckProof;

    fn init_transcript() -> Self::Transcript {
        IOPTranscript::<F>::new(b"Initializing ZeroCheck transcript")
    }

    fn prove<B: SumCheckProver<F>>(
        backend: &B,
        poly: &VirtualPolynomial<F, B::Mle>,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::ZeroCheckProof, PolyIOPErrors> {
        let start = start_timer!(|| "zero check prove");

        let length = poly.aux_info.num_variables;
        let r = transcript.get_and_append_challenge_vectors(b"0check r", length)?;
        // Input poly f(x) and a random vector r, output
        //      \hat f(x) = f(x) eq(x, r)
        // where
        //      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
        //
        // This function is used in ZeroCheck.
        let eq = backend.build_eq_x_r(&r)?;
        let mut f_hat = poly.clone();
        f_hat.mul_by_mle(eq, F::one())?;
        let res = <Self as SumCheck<F>>::prove(backend, &f_hat, transcript);

        end_timer!(start);
        res
    }

    fn verify(
        proof: &Self::ZeroCheckProof,
        fx_aux_info: &Self::VPAuxInfo,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::ZeroCheckSubClaim, PolyIOPErrors> {
        let start = start_timer!(|| "zero check verify");
        // hat_fx's max degree is increased by eq(x, r).degree() which is 1
        let mut hat_fx_aux_info = fx_aux_info.clone();
        hat_fx_aux_info.max_degree = fx_aux_info.max_degree.checked_add(1).ok_or_else(|| {
            PolyIOPErrors::InvalidParameters("ZeroCheck degree is too large.".to_string())
        })?;
        // generate `r` and pass it to the caller for correctness check
        let length = fx_aux_info.num_variables;
        let r = transcript.get_and_append_challenge_vectors(b"0check r", length)?;

        // check that the sum is zero
        let sum_subclaim =
            <Self as SumCheck<F>>::verify(F::zero(), proof, &hat_fx_aux_info, transcript)?;

        // expected_eval = sumcheck.expect_eval/eq(v, r)
        // where v = sum_check_sub_claim.point
        let eq_x_r_eval = eq_eval(&sum_subclaim.point, &r)?;
        let expected_evaluation = sum_subclaim.expected_evaluation / eq_x_r_eval;

        end_timer!(start);
        Ok(ZeroCheckSubClaim {
            point: sum_subclaim.point,
            expected_evaluation,
            init_challenge: r,
        })
    }
}

#[cfg(all(test, feature = "cpu"))]
mod test {

    use crate::{IOPProof, PolyIOP, PolyIOPErrors, ZeroCheck};
    use ark_bls12_381::Fr;
    use ark_poly::DenseMultilinearExtension;
    use ark_std::test_rng;
    use backend::{cpu::CpuBackend, VirtualPolynomial};
    use std::sync::Arc;

    #[test]
    fn test_zerocheck_rejects_malformed_shape() -> Result<(), PolyIOPErrors> {
        let mle = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            2,
            vec![Fr::from(0u64); 4],
        ));
        let poly = VirtualPolynomial::new_from_mle(&mle, Fr::from(1u64));
        let mut transcript = <PolyIOP<Fr> as ZeroCheck<Fr>>::init_transcript();
        transcript.append_message(b"testing", b"initializing transcript for testing")?;
        let proof = <PolyIOP<Fr> as ZeroCheck<Fr>>::prove(&CpuBackend, &poly, &mut transcript)?;
        let mut malformed = vec![IOPProof::default()];
        for rounds in [0, 1, 3] {
            let mut candidate = proof.clone();
            candidate.proofs.resize(rounds, proof.proofs[0].clone());
            malformed.push(candidate);
        }
        for point_length in [0, 1, 3] {
            let mut candidate = proof.clone();
            candidate.point.resize(point_length, Fr::from(0u64));
            malformed.push(candidate);
        }
        for round in 0..2 {
            for width in [0, 1, 2, 4] {
                let mut candidate = proof.clone();
                candidate.proofs[round]
                    .evaluations
                    .resize(width, Fr::from(0u64));
                malformed.push(candidate);
            }
        }
        for candidate in malformed {
            let mut transcript = <PolyIOP<Fr> as ZeroCheck<Fr>>::init_transcript();
            transcript.append_message(b"testing", b"shape rejection")?;
            assert!(matches!(
                <PolyIOP<Fr> as ZeroCheck<Fr>>::verify(&candidate, &poly.aux_info, &mut transcript,),
                Err(PolyIOPErrors::InvalidProof(_))
            ));
        }
        for (num_variables, max_degree) in [(0, 1), (2, usize::MAX), (2, usize::MAX - 1)] {
            let mut aux_info = poly.aux_info.clone();
            aux_info.num_variables = num_variables;
            aux_info.max_degree = max_degree;
            let mut transcript = <PolyIOP<Fr> as ZeroCheck<Fr>>::init_transcript();
            transcript.append_message(b"testing", b"metadata rejection")?;
            assert!(matches!(
                <PolyIOP<Fr> as ZeroCheck<Fr>>::verify(&proof, &aux_info, &mut transcript),
                Err(PolyIOPErrors::InvalidParameters(_))
            ));
        }
        let mut transcript = <PolyIOP<Fr> as ZeroCheck<Fr>>::init_transcript();
        transcript.append_message(b"testing", b"initializing transcript for testing")?;
        let subclaim =
            <PolyIOP<Fr> as ZeroCheck<Fr>>::verify(&proof, &poly.aux_info, &mut transcript)?;
        assert_eq!(
            CpuBackend.evaluate_vp(&poly, &subclaim.point)?,
            subclaim.expected_evaluation
        );
        Ok(())
    }

    #[test]
    fn test_zerocheck_zero_degree_becomes_positive_hat_degree() -> Result<(), PolyIOPErrors> {
        let poly = VirtualPolynomial::<Fr, Arc<DenseMultilinearExtension<Fr>>>::new(2);
        let mut transcript = <PolyIOP<Fr> as ZeroCheck<Fr>>::init_transcript();
        transcript.append_message(b"testing", b"initializing transcript for testing")?;
        let proof = <PolyIOP<Fr> as ZeroCheck<Fr>>::prove(&CpuBackend, &poly, &mut transcript)?;
        let mut transcript = <PolyIOP<Fr> as ZeroCheck<Fr>>::init_transcript();
        transcript.append_message(b"testing", b"initializing transcript for testing")?;
        let subclaim =
            <PolyIOP<Fr> as ZeroCheck<Fr>>::verify(&proof, &poly.aux_info, &mut transcript)?;
        assert_eq!(subclaim.expected_evaluation, Fr::from(0u64));
        assert_eq!(
            CpuBackend.evaluate_vp(&poly, &subclaim.point)?,
            subclaim.expected_evaluation
        );
        Ok(())
    }

    fn test_zerocheck(
        nv: usize,
        num_multiplicands_range: (usize, usize),
        num_products: usize,
    ) -> Result<(), PolyIOPErrors> {
        let mut rng = test_rng();

        {
            // good path: zero virtual poly
            let poly =
                VirtualPolynomial::rand_zero(nv, num_multiplicands_range, num_products, &mut rng)?;

            let mut transcript = <PolyIOP<Fr> as ZeroCheck<Fr>>::init_transcript();
            transcript.append_message(b"testing", b"initializing transcript for testing")?;
            let proof = <PolyIOP<Fr> as ZeroCheck<Fr>>::prove(&CpuBackend, &poly, &mut transcript)?;

            let poly_info = poly.aux_info.clone();
            let mut transcript = <PolyIOP<Fr> as ZeroCheck<Fr>>::init_transcript();
            transcript.append_message(b"testing", b"initializing transcript for testing")?;
            let zero_subclaim =
                <PolyIOP<Fr> as ZeroCheck<Fr>>::verify(&proof, &poly_info, &mut transcript)?;
            assert!(
                CpuBackend.evaluate_vp(&poly, &zero_subclaim.point)?
                    == zero_subclaim.expected_evaluation,
                "wrong subclaim"
            );
        }

        {
            // bad path: random virtual poly whose sum is not zero
            let (poly, _sum) = VirtualPolynomial::<Fr, _>::rand(
                nv,
                num_multiplicands_range,
                num_products,
                &mut rng,
            )?;

            let mut transcript = <PolyIOP<Fr> as ZeroCheck<Fr>>::init_transcript();
            transcript.append_message(b"testing", b"initializing transcript for testing")?;
            let proof = <PolyIOP<Fr> as ZeroCheck<Fr>>::prove(&CpuBackend, &poly, &mut transcript)?;

            let poly_info = poly.aux_info.clone();
            let mut transcript = <PolyIOP<Fr> as ZeroCheck<Fr>>::init_transcript();
            transcript.append_message(b"testing", b"initializing transcript for testing")?;

            assert!(
                <PolyIOP<Fr> as ZeroCheck<Fr>>::verify(&proof, &poly_info, &mut transcript)
                    .is_err()
            );
        }

        Ok(())
    }

    #[test]
    fn test_trivial_polynomial() -> Result<(), PolyIOPErrors> {
        let nv = 1;
        let num_multiplicands_range = (4, 5);
        let num_products = 1;

        test_zerocheck(nv, num_multiplicands_range, num_products)
    }
    #[test]
    fn test_normal_polynomial() -> Result<(), PolyIOPErrors> {
        let nv = 5;
        let num_multiplicands_range = (4, 9);
        let num_products = 5;

        test_zerocheck(nv, num_multiplicands_range, num_products)
    }

    #[test]
    fn zero_polynomial_should_error() -> Result<(), PolyIOPErrors> {
        let nv = 0;
        let num_multiplicands_range = (4, 13);
        let num_products = 5;

        assert!(test_zerocheck(nv, num_multiplicands_range, num_products).is_err());
        Ok(())
    }
}
