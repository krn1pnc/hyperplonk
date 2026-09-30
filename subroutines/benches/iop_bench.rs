// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

use ark_bls12_381::{Bls12_381, Fr};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_std::test_rng;
use backend::{
    cpu::{identity_permutation_mles, CpuBackend},
    VPAuxInfo, VirtualPolynomial,
};
use std::{
    marker::PhantomData,
    sync::Arc,
    time::{Duration, Instant},
};
use subroutines::{
    MultilinearKzgPCS, PermutationCheck, PolyIOP, PolyIOPErrors, PolynomialCommitmentScheme,
    ProductCheck, SumCheck, ZeroCheck,
};

type Kzg = MultilinearKzgPCS<Bls12_381>;

fn main() -> Result<(), PolyIOPErrors> {
    bench_permutation_check()?;
    println!("\n\n");
    bench_sum_check()?;
    println!("\n\n");
    bench_prod_check()?;
    println!("\n\n");
    bench_zero_check()
}

fn bench_sum_check() -> Result<(), PolyIOPErrors> {
    let mut rng = test_rng();
    for degree in 2..4 {
        for nv in 4..25 {
            let repetition = if nv < 10 {
                100
            } else if nv < 20 {
                50
            } else {
                10
            };

            let (poly, asserted_sum) =
                VirtualPolynomial::rand(nv, (degree, degree + 1), 2, &mut rng)?;
            let poly_info = poly.aux_info.clone();
            let mut prove_time = Duration::ZERO;
            let mut verify_time = Duration::ZERO;
            for _ in 0..repetition {
                let start = Instant::now();
                let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
                let proof =
                    <PolyIOP<Fr> as SumCheck<Fr>>::prove(&CpuBackend, &poly, &mut transcript)?;
                prove_time += start.elapsed();

                let start = Instant::now();
                let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
                let _subclaim = <PolyIOP<Fr> as SumCheck<Fr>>::verify(
                    asserted_sum,
                    &proof,
                    &poly_info,
                    &mut transcript,
                )?;
                verify_time += start.elapsed();
            }
            println!(
                "sum check proving time for {} variables and {} degree: {} ns",
                nv,
                degree,
                prove_time.as_nanos() / repetition as u128
            );
            println!(
                "sum check verification time for {} variables and {} degree: {} ns",
                nv,
                degree,
                verify_time.as_nanos() / repetition as u128
            );

            println!("====================================");
        }
    }
    Ok(())
}

fn bench_zero_check() -> Result<(), PolyIOPErrors> {
    let mut rng = test_rng();
    for degree in 2..4 {
        for nv in 4..20 {
            let repetition = if nv < 10 {
                100
            } else if nv < 20 {
                50
            } else {
                10
            };

            let poly = VirtualPolynomial::rand_zero(nv, (degree, degree + 1), 2, &mut rng)?;
            let poly_info = poly.aux_info.clone();
            let mut prove_time = Duration::ZERO;
            let mut verify_time = Duration::ZERO;
            for _ in 0..repetition {
                let start = Instant::now();
                let mut transcript = <PolyIOP<Fr> as ZeroCheck<Fr>>::init_transcript();
                transcript.append_message(b"testing", b"initializing transcript for testing")?;
                let proof =
                    <PolyIOP<Fr> as ZeroCheck<Fr>>::prove(&CpuBackend, &poly, &mut transcript)?;
                prove_time += start.elapsed();

                let start = Instant::now();
                let mut transcript = <PolyIOP<Fr> as ZeroCheck<Fr>>::init_transcript();
                transcript.append_message(b"testing", b"initializing transcript for testing")?;
                let _zero_subclaim =
                    <PolyIOP<Fr> as ZeroCheck<Fr>>::verify(&proof, &poly_info, &mut transcript)?;
                verify_time += start.elapsed();
            }
            println!(
                "zero check proving time for {} variables and {} degree: {} ns",
                nv,
                degree,
                prove_time.as_nanos() / repetition as u128
            );
            println!(
                "zero check verification time for {} variables and {} degree: {} ns",
                nv,
                degree,
                verify_time.as_nanos() / repetition as u128
            );

            println!("====================================");
        }
    }
    Ok(())
}

fn bench_permutation_check() -> Result<(), PolyIOPErrors> {
    let mut rng = test_rng();

    for nv in 4..20 {
        let repetition = if nv < 10 {
            100
        } else if nv < 20 {
            50
        } else {
            10
        };

        let srs = Kzg::gen_srs_for_testing(&mut rng, nv + 1)?;
        let (pcs_param, _) = Kzg::trim(&srs, nv + 1)?;
        let backend = CpuBackend;
        let pcs_param = pcs_param.prepare(&backend)?;

        let ws = vec![Arc::new(DenseMultilinearExtension::rand(nv, &mut rng))];

        // identity map
        let perms = identity_permutation_mles(nv, 1);

        let poly_info = VPAuxInfo {
            max_degree: 2,
            num_variables: nv,
            phantom: PhantomData,
        };
        let mut prove_time = Duration::ZERO;
        let mut verify_time = Duration::ZERO;
        for _ in 0..repetition {
            let start = Instant::now();
            let mut transcript =
                <PolyIOP<Fr> as PermutationCheck<Bls12_381, Kzg>>::init_transcript();
            transcript.append_message(b"testing", b"initializing transcript for testing")?;
            let (proof, ..) = <PolyIOP<Fr> as PermutationCheck<Bls12_381, Kzg>>::prove(
                &backend,
                &pcs_param,
                &ws,
                &ws,
                &perms,
                &mut transcript,
            )?;
            prove_time += start.elapsed();

            let start = Instant::now();
            let mut transcript =
                <PolyIOP<Fr> as PermutationCheck<Bls12_381, Kzg>>::init_transcript();
            transcript.append_message(b"testing", b"initializing transcript for testing")?;
            let _perm_check_sum_claim = <PolyIOP<Fr> as PermutationCheck<Bls12_381, Kzg>>::verify(
                &proof,
                &poly_info,
                &mut transcript,
            )?;
            verify_time += start.elapsed();
        }
        println!(
            "permutation check proving time for {} variables: {} ns",
            nv,
            prove_time.as_nanos() / repetition as u128
        );
        println!(
            "permutation check verification time for {} variables: {} ns",
            nv,
            verify_time.as_nanos() / repetition as u128
        );
        println!("====================================");
    }

    Ok(())
}

fn bench_prod_check() -> Result<(), PolyIOPErrors> {
    let mut rng = test_rng();

    for nv in 4..20 {
        let repetition = if nv < 10 {
            100
        } else if nv < 20 {
            50
        } else {
            10
        };

        let srs = Kzg::gen_srs_for_testing(&mut rng, nv + 1)?;
        let (pcs_param, _) = Kzg::trim(&srs, nv + 1)?;
        let backend = CpuBackend;
        let pcs_param = pcs_param.prepare(&backend)?;

        let f: DenseMultilinearExtension<Fr> = DenseMultilinearExtension::rand(nv, &mut rng);
        let mut g = f.clone();
        g.evaluations.reverse();
        let fs = vec![Arc::new(f)];
        let gs = vec![Arc::new(g)];

        let poly_info = VPAuxInfo {
            max_degree: 2,
            num_variables: nv,
            phantom: PhantomData,
        };
        let mut prove_time = Duration::ZERO;
        let mut verify_time = Duration::ZERO;
        for _ in 0..repetition {
            let start = Instant::now();
            let mut transcript = <PolyIOP<Fr> as ProductCheck<Bls12_381, Kzg>>::init_transcript();
            transcript.append_message(b"testing", b"initializing transcript for testing")?;
            let (proof, ..) = <PolyIOP<Fr> as ProductCheck<Bls12_381, Kzg>>::prove(
                &backend,
                &pcs_param,
                &fs,
                &gs,
                &mut transcript,
            )?;
            prove_time += start.elapsed();

            let start = Instant::now();
            let mut transcript = <PolyIOP<Fr> as ProductCheck<Bls12_381, Kzg>>::init_transcript();
            transcript.append_message(b"testing", b"initializing transcript for testing")?;
            let _prod_check_sum_claim = <PolyIOP<Fr> as ProductCheck<Bls12_381, Kzg>>::verify(
                &proof,
                &poly_info,
                &mut transcript,
            )?;
            verify_time += start.elapsed();
        }
        println!(
            "product check proving time for {} variables: {} ns",
            nv,
            prove_time.as_nanos() / repetition as u128
        );
        println!(
            "product check verification time for {} variables: {} ns",
            nv,
            verify_time.as_nanos() / repetition as u128
        );

        println!("====================================");
    }

    Ok(())
}
