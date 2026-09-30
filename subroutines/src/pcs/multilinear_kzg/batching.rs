// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Sumcheck based batch opening and verify commitment.

use crate::{Commitment, IOPProof, PCSError, PolyIOP, PolynomialCommitmentScheme, SumCheck};
use ark_ec::{pairing::Pairing, scalar_mul::variable_base::VariableBaseMSM, CurveGroup};
use ark_ff::PrimeField;
use backend::{
    common::eq::{build_eq_x_r_vec, eq_eval},
    Mle, MultilinearKzgBackend, SumCheckProver, VPAuxInfo, VirtualPolynomial,
};

use ark_std::{end_timer, log2, start_timer, One, Zero};
use std::{collections::BTreeMap, marker::PhantomData};
use transcript::IOPTranscript;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchProof<E: Pairing, PCS: PolynomialCommitmentScheme<E>> {
    /// A sum check proof proving tilde g's sum
    pub(crate) sum_check_proof: IOPProof<E::ScalarField>,
    /// f_i(point_i)
    pub f_i_eval_at_point_i: Vec<E::ScalarField>,
    /// proof for g'(a_2)
    pub(crate) g_prime_proof: PCS::Proof,
}

// The equality polynomial on zero variables has the single value one.
// Keep this batch-local: the common helper requires a nonempty point.
fn batch_weights<F: PrimeField>(point: &[F]) -> Result<Vec<F>, PCSError> {
    if point.is_empty() {
        Ok(vec![F::one()])
    } else {
        Ok(build_eq_x_r_vec(point)?)
    }
}

/// Steps:
/// 1. get challenge point t from transcript
/// 2. build eq(t,i) for i in [0..k]
/// 3. build \tilde g_i(b) = eq(t, i) * f_i(b)
/// 4. compute \tilde eq_i(b) = eq(b, point_i)
/// 5. run sumcheck on \sum_i=1..k \tilde eq_i * \tilde g_i
/// 6. build g'(X) = \sum_i=1..k \tilde eq_i(a2) * \tilde g_i(X) where (a2) is
///    the sumcheck's point 7. open g'(X) at point (a2)
pub(super) fn multi_open_internal<E, PCS, B>(
    backend: &B,
    prover_param: &B::PreparedProverParam,
    polynomials: &[B::Mle],
    points: &[Vec<E::ScalarField>],
    evals: &[E::ScalarField],
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<BatchProof<E, PCS>, PCSError>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E, Point = Vec<E::ScalarField>, Evaluation = E::ScalarField>,
    B: MultilinearKzgBackend<E> + SumCheckProver<E::ScalarField>,
{
    let open_timer = start_timer!(|| format!("multi open {} points", points.len()));
    let k = polynomials.len();
    if k == 0 || points.len() != k || evals.len() != k {
        return Err(PCSError::InvalidParameters(
            "batch polynomials, points, and evaluations must have equal nonzero lengths"
                .to_string(),
        ));
    }
    let num_var = polynomials[0].num_vars();
    if num_var == 0
        || polynomials.iter().any(|poly| poly.num_vars() != num_var)
        || points.iter().any(|point| point.len() != num_var)
    {
        return Err(PCSError::InvalidParameters(
            "batch polynomials and points must have the same positive dimension".to_string(),
        ));
    }
    for eval_point in points.iter() {
        transcript.append_serializable_element(b"eval_point", eval_point)?;
    }
    for eval in evals.iter() {
        transcript.append_field_element(b"eval", eval)?;
    }

    let ell = log2(k) as usize;

    // challenge point t
    let t = transcript.get_and_append_challenge_vectors("t".as_ref(), ell)?;

    // eq(t, i) for i in [0..k]
    let eq_t_i_list = batch_weights(&t)?;

    // \tilde g_i(b) = eq(t, i) * f_i(b)
    let timer = start_timer!(|| format!("compute tilde g for {} points", points.len()));
    // combine the polynomials that have same opening point first to reduce the
    // cost of sum check later.
    let point_indices = points
        .iter()
        .fold(BTreeMap::<_, _>::new(), |mut indices, point| {
            let idx = indices.len();
            indices.entry(point).or_insert(idx);
            indices
        });
    let deduped_points =
        BTreeMap::from_iter(point_indices.iter().map(|(point, idx)| (*idx, *point)))
            .into_values()
            .collect::<Vec<_>>();
    let mut groups: Vec<Vec<usize>> = (0..point_indices.len()).map(|_| Vec::new()).collect();
    for (index, point) in points.iter().enumerate() {
        groups[point_indices[point]].push(index);
    }
    let mut group_polynomials = Vec::new();
    let mut group_coefficients = Vec::new();
    let merged_tilde_gs = groups
        .iter()
        .map(|indices| {
            if let [index] = indices.as_slice() {
                return backend.linear_combination(
                    &polynomials[*index..=*index],
                    &eq_t_i_list[*index..=*index],
                );
            }
            group_polynomials.clear();
            group_coefficients.clear();
            group_polynomials.extend(indices.iter().map(|&index| polynomials[index].clone()));
            group_coefficients.extend(indices.iter().map(|&index| eq_t_i_list[index]));
            backend.linear_combination(&group_polynomials, &group_coefficients)
        })
        .collect::<Result<Vec<_>, _>>()?;
    end_timer!(timer);

    let timer = start_timer!(|| format!("compute tilde eq for {} points", points.len()));
    let tilde_eqs = deduped_points
        .iter()
        .map(|point| backend.build_eq_x_r(point))
        .collect::<Result<Vec<_>, _>>()?;
    end_timer!(timer);

    // built the virtual polynomial for SumCheck
    let timer = start_timer!(|| format!("sum check prove of {} variables", num_var));

    let step = start_timer!(|| "add mle");
    let mut sum_check_vp = VirtualPolynomial::new(num_var);
    for (merged_tilde_g, tilde_eq) in merged_tilde_gs.iter().zip(tilde_eqs.into_iter()) {
        sum_check_vp.add_mle_list([merged_tilde_g.clone(), tilde_eq], E::ScalarField::one())?;
    }
    end_timer!(step);

    let proof = match <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::prove(
        backend,
        &sum_check_vp,
        transcript,
    ) {
        Ok(p) => p,
        Err(_e) => {
            // cannot wrap IOPError with PCSError due to cyclic dependency
            return Err(PCSError::InvalidProver(
                "Sumcheck in batch proving Failed".to_string(),
            ));
        },
    };

    end_timer!(timer);

    // a2 := sumcheck's point
    let a2 = &proof.point;

    // build g'(X) = \sum_i=1..k \tilde eq_i(a2) * \tilde g_i(X) where (a2) is the
    // sumcheck's point \tilde eq_i(a2) = eq(a2, point_i)
    let step = start_timer!(|| "evaluate at a2");
    let coefficients = deduped_points
        .iter()
        .map(|point| {
            eq_eval(a2, point).map_err(|_| {
                PCSError::InvalidParameters("x and y have different length".to_string())
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let g_prime = backend.linear_combination(&merged_tilde_gs, &coefficients)?;
    end_timer!(step);

    let step = start_timer!(|| "pcs open");
    let (g_prime_proof, _g_prime_eval) = PCS::open(backend, prover_param, &g_prime, a2)?;
    end_timer!(step);

    end_timer!(open_timer);

    Ok(BatchProof {
        sum_check_proof: proof,
        f_i_eval_at_point_i: evals.to_vec(),
        g_prime_proof,
    })
}

/// Steps:
/// 1. get challenge point t from transcript
/// 2. verify the claimed weighted sum via SumCheck
/// 3. build g' commitment at the verifier's SumCheck point
/// 4. verify commitment
pub(super) fn batch_verify_internal<E, PCS>(
    verifier_param: &PCS::VerifierParam,
    f_i_commitments: &[Commitment<E>],
    points: &[Vec<E::ScalarField>],
    proof: &BatchProof<E, PCS>,
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<bool, PCSError>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<
        E,
        Point = Vec<E::ScalarField>,
        Evaluation = E::ScalarField,
        Commitment = Commitment<E>,
    >,
{
    let open_timer = start_timer!(|| "batch verification");
    let k = f_i_commitments.len();
    if k == 0 || points.len() != k || proof.f_i_eval_at_point_i.len() != k {
        return Err(PCSError::InvalidParameters(
            "batch commitments, points, and evaluations must have equal nonzero lengths"
                .to_string(),
        ));
    }
    let num_var = points[0].len();
    if num_var == 0 || points.iter().any(|point| point.len() != num_var) {
        return Err(PCSError::InvalidParameters(
            "batch points must have the same positive dimension".to_string(),
        ));
    }
    for eval_point in points.iter() {
        transcript.append_serializable_element(b"eval_point", eval_point)?;
    }
    for eval in proof.f_i_eval_at_point_i.iter() {
        transcript.append_field_element(b"eval", eval)?;
    }

    let ell = log2(k) as usize;

    // challenge point t
    let t = transcript.get_and_append_challenge_vectors("t".as_ref(), ell)?;

    let eq_t_list = batch_weights(&t)?;

    // ensure \sum_i eq(t, <i>) * f_i_evals matches the sum via SumCheck
    let mut sum = E::ScalarField::zero();
    for (i, &e) in eq_t_list.iter().enumerate().take(k) {
        sum += e * proof.f_i_eval_at_point_i[i];
    }
    let aux_info = VPAuxInfo {
        max_degree: 2,
        num_variables: num_var,
        phantom: PhantomData,
    };
    let subclaim = match <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::verify(
        sum,
        &proof.sum_check_proof,
        &aux_info,
        transcript,
    ) {
        Ok(p) => p,
        Err(_e) => {
            // cannot wrap IOPError with PCSError due to cyclic dependency
            return Err(PCSError::InvalidProver(
                "Sumcheck in batch verification failed".to_string(),
            ));
        },
    };
    let tilde_g_eval = subclaim.expected_evaluation;

    // sum check point (a2)
    // Both aggregation and the final opening use verifier-derived challenges,
    // not the proof's cached point coordinates.
    let a2 = &subclaim.point;
    // build g' commitment
    let step = start_timer!(|| "build homomorphic commitment");
    let mut scalars = Vec::with_capacity(k);
    let mut bases = Vec::with_capacity(k);
    for (i, point) in points.iter().enumerate() {
        let eq_i_a2 = eq_eval(a2, point).map_err(|_| {
            PCSError::InvalidParameters("x and y have different length".to_string())
        })?;
        scalars.push(eq_i_a2 * eq_t_list[i]);
        bases.push(f_i_commitments[i].0);
    }
    let g_prime_commit = E::G1::msm_unchecked(&bases, &scalars);
    end_timer!(step);

    // verify commitment
    let res = PCS::verify(
        verifier_param,
        &Commitment(g_prime_commit.into_affine()),
        a2,
        &tilde_g_eval,
        &proof.g_prime_proof,
    )?;

    end_timer!(open_timer);
    Ok(res)
}

#[cfg(all(test, feature = "cpu"))]
mod test {
    use super::*;
    use crate::{MultilinearKzgPCS, MultilinearUniversalParams, StructuredReferenceString};
    use ark_bls12_381::Bls12_381 as E;
    use ark_ec::pairing::Pairing;
    use ark_poly::{DenseMultilinearExtension, MultilinearExtension, Polynomial};
    use ark_std::{rand::Rng, test_rng, vec::Vec, One, UniformRand};
    use backend::{common::indexing::get_batched_nv, cpu::CpuBackend};
    use std::sync::Arc;

    type Fr = <E as Pairing>::ScalarField;

    fn batch_transcript() -> IOPTranscript<Fr> {
        let mut transcript = IOPTranscript::new(b"test transcript");
        transcript
            .append_field_element(b"init", &Fr::zero())
            .unwrap();
        transcript
    }

    fn assert_rejected<T>(
        call: impl FnOnce(&mut IOPTranscript<Fr>) -> Result<T, PCSError>,
    ) -> PCSError {
        let mut transcript = batch_transcript();
        call(&mut transcript).err().expect("invalid batch accepted")
    }

    fn test_multi_open_helper<R: Rng>(
        ml_params: &MultilinearUniversalParams<E>,
        polys: &[Arc<DenseMultilinearExtension<Fr>>],
        rng: &mut R,
    ) -> Result<(), PCSError> {
        let merged_nv = get_batched_nv(polys[0].num_vars(), polys.len());
        let (ml_ck, ml_vk) = ml_params.trim(merged_nv)?;
        let backend = CpuBackend;
        let ml_ck = ml_ck.prepare(&backend)?;

        let mut points = Vec::new();
        for poly in polys.iter() {
            let point = (0..poly.num_vars())
                .map(|_| Fr::rand(rng))
                .collect::<Vec<Fr>>();
            points.push(point);
        }
        if points.len() == 3 {
            // Exercise the shared-point aggregation optimization as well.
            points[2] = points[0].clone();
        }

        let evals = polys
            .iter()
            .zip(points.iter())
            .map(|(f, p)| f.evaluate(p))
            .collect::<Vec<_>>();

        let commitments = MultilinearKzgPCS::multi_commit(&backend, &ml_ck, polys)?;

        let mut transcript = batch_transcript();
        let mut batch_proof = MultilinearKzgPCS::multi_open(
            &backend,
            &ml_ck,
            polys,
            &points,
            &evals,
            &mut transcript,
        )?;

        // good path
        let mut transcript = batch_transcript();
        assert!(MultilinearKzgPCS::batch_verify(
            &ml_vk,
            &commitments,
            &points,
            &batch_proof,
            &mut transcript
        )?);

        // Cached coordinates are not the verifier's final opening point.
        // The real PCS proof still opens at the transcript-derived point.
        for coordinate in &mut batch_proof.sum_check_proof.point {
            *coordinate += Fr::one();
        }
        let mut transcript = batch_transcript();
        assert!(MultilinearKzgPCS::batch_verify(
            &ml_vk,
            &commitments,
            &points,
            &batch_proof,
            &mut transcript,
        )?);

        Ok(())
    }

    #[test]
    fn test_multi_open() -> Result<(), PCSError> {
        let mut rng = test_rng();

        let ml_params = MultilinearUniversalParams::<E>::gen_srs_for_testing(&mut rng, 5)?;
        for num_poly in 1..=3 {
            for nv in 1..=3 {
                let polys1: Vec<_> = (0..num_poly)
                    .map(|_| Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)))
                    .collect();
                test_multi_open_helper(&ml_params, &polys1, &mut rng)?;
            }
        }

        Ok(())
    }

    #[test]
    fn test_batch_input_validation() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let params = MultilinearUniversalParams::<E>::gen_srs_for_testing(&mut rng, 3)?;
        let (ck, vk) = params.trim(2)?;
        let backend = CpuBackend;
        let ck = ck.prepare(&backend)?;
        let polys = (0..2)
            .map(|_| Arc::new(DenseMultilinearExtension::rand(2, &mut rng)))
            .collect::<Vec<_>>();
        let points = (0..2)
            .map(|_| (0..2).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let evals = polys
            .iter()
            .zip(&points)
            .map(|(poly, point)| poly.evaluate(point))
            .collect::<Vec<_>>();
        let commitments = MultilinearKzgPCS::multi_commit(&backend, &ck, &polys)?;
        let mut proof = MultilinearKzgPCS::multi_open(
            &backend,
            &ck,
            &polys,
            &points,
            &evals,
            &mut batch_transcript(),
        )?;

        assert_rejected(|transcript| {
            MultilinearKzgPCS::multi_open(&backend, &ck, &[], &[], &[], transcript)
        });
        let saved_evals = proof.f_i_eval_at_point_i.clone();
        proof.f_i_eval_at_point_i.clear();
        assert_rejected(|transcript| {
            MultilinearKzgPCS::batch_verify(&vk, &[], &[], &proof, transcript)
        });
        proof.f_i_eval_at_point_i = saved_evals;

        // Vary each list independently on both sides; no zip truncation is valid.
        for length in [0, 1, 3] {
            let mut bad_polys = polys.clone();
            bad_polys.resize(length, polys[0].clone());
            assert_rejected(|transcript| {
                MultilinearKzgPCS::multi_open(
                    &backend, &ck, &bad_polys, &points, &evals, transcript,
                )
            });

            let mut bad_points = points.clone();
            bad_points.resize(length, points[0].clone());
            assert_rejected(|transcript| {
                MultilinearKzgPCS::multi_open(
                    &backend,
                    &ck,
                    &polys,
                    &bad_points,
                    &evals,
                    transcript,
                )
            });
            assert_rejected(|transcript| {
                MultilinearKzgPCS::batch_verify(&vk, &commitments, &bad_points, &proof, transcript)
            });

            let mut bad_evals = evals.clone();
            bad_evals.resize(length, evals[0]);
            assert_rejected(|transcript| {
                MultilinearKzgPCS::multi_open(
                    &backend, &ck, &polys, &points, &bad_evals, transcript,
                )
            });
            proof.f_i_eval_at_point_i = bad_evals;
            assert_rejected(|transcript| {
                MultilinearKzgPCS::batch_verify(&vk, &commitments, &points, &proof, transcript)
            });
            proof.f_i_eval_at_point_i = evals.clone();

            let mut bad_commitments = commitments.clone();
            bad_commitments.resize(length, commitments[0].clone());
            assert_rejected(|transcript| {
                MultilinearKzgPCS::batch_verify(&vk, &bad_commitments, &points, &proof, transcript)
            });
        }

        // Check every position, including zero-dimensional polynomials and points.
        for index in 0..polys.len() {
            for dimension in [0, 1, 3] {
                let mut bad_polys = polys.clone();
                bad_polys[index] = Arc::new(DenseMultilinearExtension::rand(dimension, &mut rng));
                assert_rejected(|transcript| {
                    MultilinearKzgPCS::multi_open(
                        &backend, &ck, &bad_polys, &points, &evals, transcript,
                    )
                });
                let mut bad_points = points.clone();
                bad_points[index].resize(dimension, Fr::zero());
                assert_rejected(|transcript| {
                    MultilinearKzgPCS::multi_open(
                        &backend,
                        &ck,
                        &polys,
                        &bad_points,
                        &evals,
                        transcript,
                    )
                });
                assert_rejected(|transcript| {
                    MultilinearKzgPCS::batch_verify(
                        &vk,
                        &commitments,
                        &bad_points,
                        &proof,
                        transcript,
                    )
                });
            }
        }
        let zero_polys = vec![Arc::new(DenseMultilinearExtension::rand(0, &mut rng)); polys.len()];
        let zero_points = vec![vec![]; points.len()];
        assert_rejected(|transcript| {
            MultilinearKzgPCS::multi_open(
                &backend,
                &ck,
                &zero_polys,
                &zero_points,
                &evals,
                transcript,
            )
        });
        assert_rejected(|transcript| {
            MultilinearKzgPCS::batch_verify(&vk, &commitments, &zero_points, &proof, transcript)
        });

        // SumCheck rejects malformed cached-point, round-count, and message-width
        // shapes, including widths in later rounds.
        let sum_check_proof = proof.sum_check_proof.clone();
        for length in [0, 1, 3] {
            proof.sum_check_proof = sum_check_proof.clone();
            proof.sum_check_proof.point.resize(length, Fr::zero());
            let error = assert_rejected(|transcript| {
                MultilinearKzgPCS::batch_verify(&vk, &commitments, &points, &proof, transcript)
            });
            assert!(matches!(error, PCSError::InvalidProver(_)));

            proof.sum_check_proof = sum_check_proof.clone();
            proof
                .sum_check_proof
                .proofs
                .resize(length, sum_check_proof.proofs[0].clone());
            let error = assert_rejected(|transcript| {
                MultilinearKzgPCS::batch_verify(&vk, &commitments, &points, &proof, transcript)
            });
            assert!(matches!(error, PCSError::InvalidProver(_)));
        }
        for round in 0..sum_check_proof.proofs.len() {
            for width in [0, 2, 4] {
                proof.sum_check_proof = sum_check_proof.clone();
                proof.sum_check_proof.proofs[round]
                    .evaluations
                    .resize(width, Fr::zero());
                let error = assert_rejected(|transcript| {
                    MultilinearKzgPCS::batch_verify(&vk, &commitments, &points, &proof, transcript)
                });
                assert!(matches!(error, PCSError::InvalidProver(_)));
            }
        }

        Ok(())
    }
}
