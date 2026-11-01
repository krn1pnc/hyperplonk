//! Standalone SumCheck benchmark: CPU oracle versus the Ascend910B3 backend.
//!
//! The workload is the repository's generic sum-of-products virtual polynomial
//! over the full 255-bit BLS12-381 scalar field. Factor tables are seeded
//! deterministically from `(num_vars, degree)`, uploaded exactly once per
//! backend, and never read back from the device. Every warm-up and timed proof
//! is compared byte-for-byte with a CPU-prover reference proof that is computed
//! once, before any backend timing.
//!
//! The device prover returns host-ready round messages, so each round timer
//! contains the device round plus a synchronization, the transcript append and
//! the Fiat-Shamir challenge derivation. These are not pure device times and
//! the output never labels them as such.
//!
//! The whole-proof window per sample is continuous: it starts before
//! `prover_init`, covers transcript setup and every round, and ends after the
//! explicit prover-state drop, which is where a backend releases staged or
//! retired device resources and finishes its teardown synchronization.
//! Verification and the CPU oracle stay outside that window. `cold_init_ms`
//! measures only the first `prover_init` and excludes the cold state drop.
//! SumCheck is not MSM; the report marks `not_msm: true` and contains no MSM
//! figures.
//!
//! Usage:
//! ```text
//! cargo run --release -p subroutines --example sumcheck_bench -- \
//!     --backend cpu --num-vars 20 --degree 2 --warmups 1 --samples 3
//! cargo run --release -p subroutines --example sumcheck_bench --features ascend910b3 -- \
//!     --backend npu --num-vars 25 --degree 34 --warmups 1 --samples 3
//! ```

use ark_bls12_381::Fr;
use ark_ff::UniformRand;
use ark_serialize::CanonicalSerialize;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use backend::{cpu::CpuBackend, MleBackend, SumCheckProver, VPAuxInfo, VirtualPolynomial};
use std::time::{Duration, Instant};
use subroutines::{IOPProof, IOPProverMessage, PolyIOP, SumCheck};

#[cfg(feature = "ascend910b3")]
use backend::ascend910b3::Ascend910B3Backend;

/// Allowed variable counts and degrees for this focused CLI.
const ALLOWED_NUM_VARS: [usize; 2] = [20, 25];
const ALLOWED_DEGREES: [usize; 3] = [2, 3, 34];
const DEFAULT_WARMUPS: usize = 1;
const DEFAULT_SAMPLES: usize = 3;

#[derive(Clone, Copy, PartialEq, Eq)]
enum BackendKind {
    Cpu,
    Npu,
}

impl BackendKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Npu => "npu",
        }
    }
}

struct Args {
    num_vars: usize,
    degree: usize,
    warmups: usize,
    samples: usize,
    backend: BackendKind,
}

struct TermFamily {
    factor_count: usize,
    products: Vec<(Fr, Vec<usize>)>,
    label: &'static str,
}

#[derive(Clone)]
struct Summary {
    samples: Vec<f64>,
    minimum: f64,
    median: f64,
    maximum: f64,
}

struct Report {
    backend_name: &'static str,
    backend_identity: String,
    num_vars: usize,
    degree: usize,
    round_count: usize,
    factor_count: usize,
    product_count: usize,
    term_family: &'static str,
    seed: u64,
    digest: String,
    warmups: usize,
    samples: usize,
    cold_upload_ms: f64,
    cold_init_ms: f64,
    init_timing: Summary,
    round_timing: Summary,
    total_timing: Summary,
    per_round_median: Vec<f64>,
    verified_proofs: usize,
}

fn print_usage() {
    println!(
        "usage: sumcheck_bench --backend <cpu|npu> --num-vars <20|25> --degree <2|3|34> \
         [--warmups <n>] [--samples <n>]"
    );
    println!("  --backend   execution backend; npu requires the ascend910b3 feature");
    println!("  --num-vars  sum-check variable count, 20 or 25");
    println!("  --degree    maximum product arity, 2, 3 or 34");
    println!("  --warmups   untimed warm-up proofs, default {DEFAULT_WARMUPS}");
    println!("  --samples   timed proofs, default {DEFAULT_SAMPLES}");
    println!("The JSON report is written to stdout; progress goes to stderr.");
}

fn next_value(
    arguments: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, String> {
    arguments
        .next()
        .ok_or_else(|| format!("missing value after {flag}"))
}

fn parse_usize(flag: &str, value: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("{flag} expects a nonnegative integer, got {value:?}"))
}

fn parse_args() -> Result<Args, String> {
    let mut num_vars = None;
    let mut degree = None;
    let mut warmups = None;
    let mut samples = None;
    let mut backend = None;
    let mut arguments = std::env::args().skip(1);
    while let Some(flag) = arguments.next() {
        match flag.as_str() {
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            "--num-vars" => {
                num_vars = Some(parse_usize(&flag, &next_value(&mut arguments, &flag)?)?);
            }
            "--degree" => {
                degree = Some(parse_usize(&flag, &next_value(&mut arguments, &flag)?)?);
            }
            "--warmups" => {
                warmups = Some(parse_usize(&flag, &next_value(&mut arguments, &flag)?)?);
            }
            "--samples" => {
                samples = Some(parse_usize(&flag, &next_value(&mut arguments, &flag)?)?);
            }
            "--backend" => {
                let value = next_value(&mut arguments, &flag)?;
                backend = Some(match value.as_str() {
                    "cpu" => BackendKind::Cpu,
                    "npu" => BackendKind::Npu,
                    other => {
                        return Err(format!("--backend must be cpu or npu, got {other:?}"));
                    }
                });
            }
            other => return Err(format!("unknown argument {other:?}; try --help")),
        }
    }

    let num_vars = num_vars.ok_or("missing required --num-vars")?;
    if !ALLOWED_NUM_VARS.contains(&num_vars) {
        return Err(format!(
            "--num-vars must be one of {ALLOWED_NUM_VARS:?}, got {num_vars}"
        ));
    }
    let degree = degree.ok_or("missing required --degree")?;
    if !ALLOWED_DEGREES.contains(&degree) {
        return Err(format!(
            "--degree must be one of {ALLOWED_DEGREES:?}, got {degree}"
        ));
    }
    let warmups = warmups.unwrap_or(DEFAULT_WARMUPS);
    if warmups == 0 {
        return Err("--warmups must be at least 1".to_string());
    }
    let samples = samples.unwrap_or(DEFAULT_SAMPLES);
    if samples == 0 {
        return Err("--samples must be at least 1".to_string());
    }
    let backend = backend.ok_or("missing required --backend")?;
    Ok(Args {
        num_vars,
        degree,
        warmups,
        samples,
        backend,
    })
}

/// Working input family. Degree 2 is the paper's simple two-factor product;
/// the higher degrees are the repository's generic multi-product HyperPlonk
/// shape with repeated multiplicands and several coefficients.
fn term_family(degree: usize) -> Result<TermFamily, String> {
    let (label, products) = match degree {
        2 => (
            "degree_2_simple_product",
            vec![(Fr::from(1u64), vec![0usize, 1usize])],
        ),
        3 => (
            "generic_multi_product_repeated_factors",
            vec![
                (Fr::from(1u64), vec![0usize, 1usize, 2usize]),
                (Fr::from(3u64), vec![0usize, 0usize, 1usize]),
            ],
        ),
        34 => (
            "generic_multi_product_repeated_factors",
            vec![
                (Fr::from(1u64), (0..34).map(|index| index % 3).collect()),
                (Fr::from(3u64), vec![0usize, 0usize, 1usize]),
                (Fr::from(5u64), vec![2usize]),
            ],
        ),
        other => return Err(format!("unsupported degree {other}")),
    };
    let factor_count = products
        .iter()
        .flat_map(|(_, indices)| indices.iter().copied())
        .max()
        .map(|index| index + 1)
        .unwrap_or(0);
    Ok(TermFamily {
        factor_count,
        products,
        label,
    })
}

fn workload_seed(num_vars: usize, degree: usize) -> u64 {
    0x5355_4d43_4845_454b ^ ((num_vars as u64) << 8) ^ degree as u64
}

fn factor_tables(num_vars: usize, factor_count: usize, seed: u64) -> Vec<Vec<Fr>> {
    let entries = 1usize << num_vars;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..factor_count)
        .map(|_| (0..entries).map(|_| Fr::rand(&mut rng)).collect())
        .collect()
}

fn build_polynomial<B: MleBackend<Fr>>(
    backend: &B,
    num_vars: usize,
    tables: &[Vec<Fr>],
    products: &[(Fr, Vec<usize>)],
) -> Result<VirtualPolynomial<Fr, B::Mle>, String> {
    let factors = tables
        .iter()
        .map(|values| {
            backend
                .mle_from_evaluations(num_vars, values.clone())
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut polynomial = VirtualPolynomial::new(num_vars);
    for (coefficient, indices) in products {
        polynomial
            .add_mle_list(
                indices.iter().map(|&index| factors[index].clone()),
                *coefficient,
            )
            .map_err(|error| error.to_string())?;
    }
    Ok(polynomial)
}

fn serialized<T: CanonicalSerialize>(value: &T) -> Vec<u8> {
    let mut bytes = Vec::new();
    value
        .serialize_compressed(&mut bytes)
        .expect("canonical serialization");
    bytes
}

fn encoded_field_elements(values: &[Fr]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for value in values {
        bytes.extend_from_slice(&serialized(value));
    }
    bytes
}

fn encoded_messages(proof: &IOPProof<Fr>) -> Vec<Vec<u8>> {
    proof.proofs.iter().map(serialized).collect()
}

fn fnv1a(state: &mut u64, bytes: &[u8]) {
    for &byte in bytes {
        *state ^= byte as u64;
        *state = state.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

/// Digest of the exact inputs: dimensions, product coefficients/indices and
/// every full-width field element of every factor table.
fn digest_manifest(
    num_vars: usize,
    degree: usize,
    tables: &[Vec<Fr>],
    products: &[(Fr, Vec<usize>)],
) -> String {
    let mut state = 0xcbf2_9ce4_8422_2325u64;
    fnv1a(&mut state, b"hyperplonk_sumcheck_bench_v1");
    fnv1a(&mut state, &(num_vars as u64).to_le_bytes());
    fnv1a(&mut state, &(degree as u64).to_le_bytes());
    fnv1a(&mut state, &(tables.len() as u64).to_le_bytes());
    for table in tables {
        fnv1a(&mut state, &(table.len() as u64).to_le_bytes());
        for value in table {
            fnv1a(&mut state, &serialized(value));
        }
    }
    fnv1a(&mut state, &(products.len() as u64).to_le_bytes());
    for (coefficient, indices) in products {
        fnv1a(&mut state, &serialized(coefficient));
        fnv1a(&mut state, &(indices.len() as u64).to_le_bytes());
        for &index in indices {
            fnv1a(&mut state, &(index as u64).to_le_bytes());
        }
    }
    format!("fnv1a64:{state:016x}")
}

/// Reference proof from the CPU backend. This is the correctness oracle and is
/// computed once, outside every backend timing window.
fn cpu_reference(
    num_vars: usize,
    tables: &[Vec<Fr>],
    products: &[(Fr, Vec<usize>)],
) -> Result<IOPProof<Fr>, String> {
    let cpu = CpuBackend;
    let polynomial = build_polynomial(&cpu, num_vars, tables, products)?;
    let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
    <PolyIOP<Fr> as SumCheck<Fr>>::prove(&cpu, &polynomial, &mut transcript)
        .map_err(|error| error.to_string())
}

/// Exact comparison with the oracle plus host verification of the measured
/// proof. No device table is consulted; only host-ready round messages are used.
fn verify_against_oracle(
    proof: &IOPProof<Fr>,
    oracle: &IOPProof<Fr>,
    aux_info: &VPAuxInfo<Fr>,
    claimed_sum: Fr,
) -> Result<(), String> {
    if encoded_messages(proof) != encoded_messages(oracle) {
        return Err("prover round message bytes differ from the CPU oracle".to_string());
    }
    if encoded_field_elements(&proof.point) != encoded_field_elements(&oracle.point) {
        return Err("Fiat-Shamir challenges differ from the CPU oracle".to_string());
    }
    let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
    let subclaim =
        <PolyIOP<Fr> as SumCheck<Fr>>::verify(claimed_sum, proof, aux_info, &mut transcript)
            .map_err(|error| format!("host verification of the measured proof failed: {error}"))?;
    if subclaim.point != proof.point {
        return Err("verifier challenges differ from the proof point".to_string());
    }
    Ok(())
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

fn summarize(samples: Vec<f64>) -> Summary {
    Summary {
        minimum: samples.iter().copied().fold(f64::INFINITY, f64::min),
        maximum: samples.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        median: median(&samples),
        samples,
    }
}

fn cpu_identity() -> String {
    format!(
        "{{\"name\":\"cpu\",\"execution\":\"rayon\",\"threads\":{},\"config_source\":\"runtime\"}}",
        rayon::current_num_threads()
    )
}

#[cfg(feature = "ascend910b3")]
fn npu_identity() -> String {
    String::from(
        "{\"name\":\"ascend910b3\",\"execution\":\"native_single_stream\",\"device_id\":0,\
         \"soc\":\"Ascend910B3 (DAV_2201)\",\"documented_aiv_cores\":40,\
         \"documented_aic_cores\":20,\"documented_ub_bytes\":196352,\
         \"cann_toolkit\":\"9.1.0\",\"config_source\":\"documented_target_config_not_measured\"}",
    )
}

/// Measure one backend: cold upload/init, warm-up proofs, timed proofs and the
/// per-round breakdown. All warm-up and timed proofs are verified first.
#[allow(clippy::too_many_arguments)]
fn benchmark<B: MleBackend<Fr> + SumCheckProver<Fr>>(
    backend: &B,
    backend_name: &'static str,
    backend_identity: String,
    args: &Args,
    family: &TermFamily,
    polynomial: &VirtualPolynomial<Fr, B::Mle>,
    oracle: &IOPProof<Fr>,
    claimed_sum: Fr,
    cold_upload: Duration,
    digest: &str,
) -> Result<Report, String> {
    let round_count = oracle.proofs.len();
    let aux_info = polynomial.aux_info.clone();

    // Cold initialization on the already resident inputs. This times
    // `prover_init` only; the explicit cold prover-state drop happens after the
    // timer stops, so `cold_init_ms` is not a full cold-lifecycle number.
    let cold_start = Instant::now();
    let cold_state = backend
        .prover_init(polynomial)
        .map_err(|error| error.to_string())?;
    let cold_init = cold_start.elapsed();
    drop(cold_state);

    let mut init_samples = Vec::with_capacity(args.samples);
    let mut round_samples = Vec::with_capacity(args.samples);
    let mut total_samples = Vec::with_capacity(args.samples);
    let mut round_observations: Vec<Vec<f64>> = vec![Vec::with_capacity(args.samples); round_count];
    let mut verified_proofs = 0usize;

    for run in 0..args.warmups + args.samples {
        // Continuous whole-proof lifecycle for one sample: starts before
        // `prover_init`, covers transcript setup and every round, and stops only
        // after the explicit prover-state drop below. Verification is outside.
        let total_start = Instant::now();
        let init_start = Instant::now();
        let mut state = backend
            .prover_init(polynomial)
            .map_err(|error| error.to_string())?;
        let init_elapsed = init_start.elapsed();

        // This window covers only the transcript aux-info append, every round
        // message, transcript append, challenge derivation and the host-visible
        // result synchronization inside each prover call. It excludes
        // `prover_init` (measured above) and prover-state teardown (included in
        // the continuous lifecycle window instead).
        let proof_start = Instant::now();
        let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
        transcript
            .append_serializable_element(b"aux info", &aux_info)
            .map_err(|error| error.to_string())?;
        let mut challenge = None;
        let mut proofs = Vec::with_capacity(round_count);
        let mut point = Vec::with_capacity(round_count);
        let mut round_durations = Vec::with_capacity(round_count);
        for _ in 0..round_count {
            let round_start = Instant::now();
            let evaluations = backend
                .prove_round_and_update_state(&mut state, &challenge)
                .map_err(|error| error.to_string())?;
            let message = IOPProverMessage::new(evaluations);
            transcript
                .append_serializable_element(b"prover msg", &message)
                .map_err(|error| error.to_string())?;
            let next = transcript
                .get_and_append_challenge(b"Internal round")
                .map_err(|error| error.to_string())?;
            round_durations.push(millis(round_start.elapsed()));
            point.push(next);
            proofs.push(message);
            challenge = Some(next);
        }
        let rounds_elapsed = proof_start.elapsed();
        let proof = IOPProof { point, proofs };
        // Tear down the prover state inside the measured lifecycle window so the
        // backend's staging/retirement/free work and any teardown synchronization
        // are part of the whole-proof number. This runs before the lifecycle
        // elapsed time is read and before verification.
        drop(state);
        let total_elapsed = total_start.elapsed();
        verify_against_oracle(&proof, oracle, &aux_info, claimed_sum)?;
        verified_proofs += 1;

        if run >= args.warmups {
            init_samples.push(millis(init_elapsed));
            round_samples.push(millis(rounds_elapsed));
            // The whole-proof figure is the actual continuous lifecycle time, not
            // the sum of the separately reported init and round windows.
            total_samples.push(millis(total_elapsed));
            for (slot, duration) in round_observations.iter_mut().zip(&round_durations) {
                slot.push(*duration);
            }
            eprintln!(
                "sample {}: init {:.3} ms, rounds {:.3} ms, proof {:.3} ms",
                run - args.warmups,
                millis(init_elapsed),
                millis(rounds_elapsed),
                millis(total_elapsed)
            );
        }
    }

    Ok(Report {
        backend_name,
        backend_identity,
        num_vars: args.num_vars,
        degree: args.degree,
        round_count,
        factor_count: family.factor_count,
        product_count: family.products.len(),
        term_family: family.label,
        seed: workload_seed(args.num_vars, args.degree),
        digest: digest.to_string(),
        warmups: args.warmups,
        samples: args.samples,
        cold_upload_ms: millis(cold_upload),
        cold_init_ms: millis(cold_init),
        per_round_median: round_observations
            .iter()
            .map(|observations| median(observations))
            .collect(),
        init_timing: summarize(init_samples),
        round_timing: summarize(round_samples),
        total_timing: summarize(total_samples),
        verified_proofs,
    })
}

impl Report {
    fn to_json(&self) -> String {
        let samples = |summary: &Summary| {
            summary
                .samples
                .iter()
                .map(|value| format!("{value:.6}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let per_round = self
            .per_round_median
            .iter()
            .map(|value| format!("{value:.6}"))
            .collect::<Vec<_>>()
            .join(", ");

        let mut lines = Vec::new();
        lines.push("{".to_string());
        lines.push("  \"schema\": \"hyperplonk_sumcheck_bench_v1\",".to_string());
        lines.push(format!(
            "  \"backend\": \"{}\",",
            self.backend_name
        ));
        lines.push(format!("  \"backend_identity\": {},", self.backend_identity));
        lines.push("  \"workload\": {".to_string());
        lines.push("    \"kind\": \"sumcheck\",".to_string());
        lines.push("    \"not_msm\": true,".to_string());
        lines.push("    \"field\": \"BLS12-381 Fr (255-bit prime)\",".to_string());
        lines.push(format!("    \"num_vars\": {},", self.num_vars));
        lines.push(format!("    \"degree\": {},", self.degree));
        lines.push(format!("    \"rounds\": {},", self.round_count));
        lines.push(format!(
            "    \"round_evaluations\": {},",
            self.degree + 1
        ));
        lines.push(format!("    \"factor_count\": {},", self.factor_count));
        lines.push(format!("    \"product_count\": {},", self.product_count));
        lines.push(format!("    \"term_family\": \"{}\",", self.term_family));
        lines.push(format!("    \"input_seed\": \"0x{:016x}\",", self.seed));
        lines.push(format!("    \"input_digest\": \"{}\"", self.digest));
        lines.push("  },".to_string());
        lines.push(format!(
            "  \"configuration\": {{\"warmups\": {}, \"samples\": {}}},",
            self.warmups, self.samples
        ));
        lines.push(format!(
            "  \"cold\": {{\"upload_ms\": {:.6}, \"init_ms\": {:.6}}},",
            self.cold_upload_ms, self.cold_init_ms
        ));
        lines.push("  \"timings_ms\": {".to_string());
        lines.push(format!(
            "    \"init\": {{\"samples\": [{}], \"min\": {:.6}, \"median\": {:.6}, \"max\": {:.6}}},",
            samples(&self.init_timing),
            self.init_timing.minimum,
            self.init_timing.median,
            self.init_timing.maximum
        ));
        lines.push(format!(
            "    \"rounds\": {{\"samples\": [{}], \"min\": {:.6}, \"median\": {:.6}, \"max\": {:.6}}},",
            samples(&self.round_timing),
            self.round_timing.minimum,
            self.round_timing.median,
            self.round_timing.maximum
        ));
        lines.push(format!(
            "    \"proof_total\": {{\"samples\": [{}], \"min\": {:.6}, \"median\": {:.6}, \"max\": {:.6}}},",
            samples(&self.total_timing),
            self.total_timing.minimum,
            self.total_timing.median,
            self.total_timing.maximum
        ));
        lines.push(format!("    \"per_round_median\": [{per_round}]"));
        lines.push("  },".to_string());
        lines.push("  \"verification\": {".to_string());
        lines.push(
            "    \"cpu_oracle\": \"CpuBackend SumCheck proof computed once before any backend timing\","
                .to_string(),
        );
        lines.push(format!(
            "    \"verified_proofs\": {},",
            self.verified_proofs
        ));
        lines
            .push("    \"round_message_bytes_equal_to_cpu_oracle\": true,".to_string());
        lines.push("    \"fiat_shamir_challenges_equal_to_cpu_oracle\": true,".to_string());
        lines.push("    \"host_sumcheck_verify_passed\": true,".to_string());
        lines.push("    \"device_tables_read_back\": false".to_string());
        lines.push("  },".to_string());
        lines.push("  \"timing_scope\": {".to_string());
        lines.push(
            "    \"cold_upload\": \"first resident upload of every factor table for this backend\","
                .to_string(),
        );
        lines.push(
            "    \"cold_init\": \"first prover_init on the resident inputs; excludes the cold \
             prover-state drop\","
                .to_string(),
        );
        lines.push("    \"init\": \"prover_init per sample\",".to_string());
        lines.push(
            "    \"rounds\": \"all N prover rounds including round message, transcript append, \
             challenge derivation and host-visible result synchronization; excludes prover_init \
             and prover-state teardown\","
                .to_string(),
        );
        lines.push(
            "    \"proof_total\": \"continuous per-sample lifecycle: prover_init + transcript + \
             all N rounds + explicit prover-state drop, which is where a backend releases staged \
             or retired device resources and finishes teardown synchronization; excludes \
             verification\","
                .to_string(),
        );
        lines.push("    \"pure_device_time\": false,".to_string());
        lines.push(
            "    \"excluded\": [\"verification\", \"cpu_oracle\", \"digest\", \"context_open\", \
             \"cold_prover_state_drop\"]"
                .to_string(),
        );
        lines.push("  },".to_string());
        lines.push("  \"paper_reference\": {".to_string());
        lines.push(
            "    \"primary\": \"https://www.usenix.org/system/files/usenixsecurity26-lu-tao.pdf\","
                .to_string(),
        );
        lines.push(
            "    \"artifact\": \"https://doi.org/10.5281/zenodo.20337923\",".to_string(),
        );
        lines.push(
            "    \"evaluated_field\": \"BabyBear p=2^31-2^27+1, degree-8 extension X^8-11\","
                .to_string(),
        );
        lines.push(
            "    \"evaluated_hardware\": \"NVIDIA H100 80GB\",".to_string(),
        );
        lines.push("    \"degree_2_n25_ms\": 4.04,".to_string());
        lines.push("    \"directly_comparable\": false,".to_string());
        lines.push(
            "    \"note\": \"different 255-bit prime field and hardware; the reference publishes \
             no N=20 or standalone high-degree sum-check number\""
                .to_string(),
        );
        lines.push("  }".to_string());
        lines.push("}".to_string());
        lines.push(String::new());
        lines.join("\n")
    }
}

#[cfg(feature = "ascend910b3")]
fn run_npu(
    args: &Args,
    family: &TermFamily,
    tables: &[Vec<Fr>],
    oracle: &IOPProof<Fr>,
    claimed_sum: Fr,
    digest: &str,
) -> Result<Report, String> {
    let ascend = Ascend910B3Backend::new(0)
        .map_err(|error| format!("failed to open Ascend910B3 device 0: {error}"))?;
    let upload_start = Instant::now();
    let polynomial = build_polynomial(&ascend, args.num_vars, tables, &family.products)?;
    let cold_upload = upload_start.elapsed();
    benchmark(
        &ascend,
        "npu",
        npu_identity(),
        args,
        family,
        &polynomial,
        oracle,
        claimed_sum,
        cold_upload,
        digest,
    )
}

#[cfg(not(feature = "ascend910b3"))]
fn run_npu(
    _args: &Args,
    _family: &TermFamily,
    _tables: &[Vec<Fr>],
    _oracle: &IOPProof<Fr>,
    _claimed_sum: Fr,
    _digest: &str,
) -> Result<Report, String> {
    Err("this binary was built without the ascend910b3 feature; \
         rebuild with --features ascend910b3"
        .to_string())
}

fn run(args: &Args) -> Result<Report, String> {
    let family = term_family(args.degree)?;
    let seed = workload_seed(args.num_vars, args.degree);
    let tables = factor_tables(args.num_vars, family.factor_count, seed);
    let digest = digest_manifest(args.num_vars, args.degree, &tables, &family.products);

    eprintln!(
        "CPU oracle: degree {} over 2^{} variables, {} factor(s), {} product(s), inputs {digest}",
        args.degree,
        args.num_vars,
        family.factor_count,
        family.products.len()
    );
    let oracle = cpu_reference(args.num_vars, &tables, &family.products)?;
    if oracle.proofs.len() != args.num_vars {
        return Err(format!(
            "CPU oracle returned {} rounds, expected {}",
            oracle.proofs.len(),
            args.num_vars
        ));
    }
    let claimed_sum = <PolyIOP<Fr> as SumCheck<Fr>>::extract_sum(&oracle);

    match args.backend {
        BackendKind::Cpu => {
            let cpu = CpuBackend;
            let upload_start = Instant::now();
            let polynomial = build_polynomial(&cpu, args.num_vars, &tables, &family.products)?;
            let cold_upload = upload_start.elapsed();
            benchmark(
                &cpu,
                BackendKind::Cpu.as_str(),
                cpu_identity(),
                args,
                &family,
                &polynomial,
                &oracle,
                claimed_sum,
                cold_upload,
                &digest,
            )
        }
        BackendKind::Npu => run_npu(args, &family, &tables, &oracle, claimed_sum, &digest),
    }
}

fn main() {
    match run_with_args() {
        Ok(()) => {}
        Err(error) => {
            eprintln!("sumcheck_bench: {error}");
            std::process::exit(1);
        }
    }
}

fn run_with_args() -> Result<(), String> {
    let args = parse_args()?;
    let report = run(&args)?;
    print!("{}", report.to_json());
    Ok(())
}
