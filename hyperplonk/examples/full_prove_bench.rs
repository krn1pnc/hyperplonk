//! Full HyperPlonk prove-and-verify benchmark: CPU oracle versus Ascend910B3.
//!
//! This driver builds a generic, non-degenerate vanilla Plonk fixture through
//! the public `MockCircuit` surface, preprocesses it once per backend, computes
//! one CPU oracle proof outside every timing window, and then measures complete
//! `<PolyIOP<Fr> as HyperPlonkSNARK<..>>::prove` calls on the requested backend.
//! Every warm-up and measured proof is compared with the oracle and verified by
//! the host CPU verifier before it is accepted; any mismatch or verification
//! error aborts the run with a non-zero exit status. There are no retries and
//! no excluded samples.
//!
//! Timing boundary for `prove`:
//!
//! ```text
//! let start = Instant::now();
//! let proof = <PolyIOP<Fr> as HyperPlonkSNARK<Bls12_381, Pcs, B>>::prove(
//!     backend, pk, public_inputs, witnesses)?;
//! let elapsed = start.elapsed();
//! ```
//!
//! The window therefore contains the witness clone/encode/host-to-device
//! upload, the witness commitments, the gate zero-check, the permutation and
//! product checks, every `evaluate_mle` query, the `frac`/`prod` commitments,
//! the batched `multi_open` (whose final `PCS::open` synchronizes the device
//! stream), all host Fiat-Shamir work, the proof assembly inside `prove`, and
//! the local drops before `prove` returns. It excludes only SRS setup, backend
//! context creation, `preprocess`, fixture construction, the CPU oracle, host
//! verification and warm-up proofs; each of those is reported separately.
//!
//! The proof object has no canonical serialization in this crate, so the
//! reported boundary is the ready proof object, never serialized bytes. The
//! proof comparison covers every publicly reachable field; the batch-opening
//! inner sum-check messages and the quotient opening are `pub(crate)` and are
//! additionally covered by the real host verification of every proof.
//!
//! Fixture: `CustomizedGates::vanilla_plonk_gate()` with all five selector
//! terms active on every row. Rows are generated in identical pairs, which
//! repeats witness values so that a non-identity permutation can swap
//! equal-valued positions within each witness column; every swap is validated
//! to be bijective, non-identity, and value-consistent. `q_L`, `q_R`, `q_M` and
//! `q_C` are independent non-zero full-range field elements and the output
//! selector `q_O` is solved from the unchanged gate equation (with a non-zero
//! `w_2`) and required to be non-zero, so no gate path is zeroed and `w_0`
//! differs from `w_2` on every row.
//!
//! Usage (defaults are a small CPU smoke run; large scales are never run
//! automatically):
//!
//! ```text
//! cargo run --release -p hyperplonk --example full_prove_bench -- \
//!     --backend cpu --num-vars 8 --warmups 1 --samples 3
//! cargo run --release -p hyperplonk --example full_prove_bench \
//!     --features ascend910b3 -- \
//!     --backend npu --num-vars 20 --warmups 1 --samples 3 --seed 0x485046554c4c5052
//! ```
//!
//! `--num-vars` accepts 2..=25; N >= 22 needs an explicit memory feasibility
//! check before it is safe to run. `--srs-file` loads a canonical
//! (`serialize_compressed`) universal SRS and validates its dimensions; if the
//! path does not exist the SRS is generated deterministically from the seed and
//! written once with `create_new`, never overwriting an existing file. Loading
//! a file uses checked canonical deserialization, which validates every group
//! element.

use std::io::{BufReader, BufWriter, Write};
use std::time::{Duration, Instant};

use ark_bls12_381::{Bls12_381, Fr};
use ark_ff::{One, UniformRand, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::{rngs::StdRng, SeedableRng};
use backend::cpu::CpuBackend;
use hyperplonk::{CustomizedGates, HyperPlonkSNARK, MockCircuit, SelectorColumn, WitnessColumn};
use subroutines::{
    MultilinearKzgPCS, MultilinearUniversalParams, PolyIOP, PolynomialCommitmentScheme,
};

#[cfg(feature = "ascend910b3")]
use backend::ascend910b3::Ascend910B3Backend;

type Pcs = MultilinearKzgPCS<Bls12_381>;
type Srs = MultilinearUniversalParams<Bls12_381>;
/// The index type is projection-named because `HyperPlonkIndex` is not part of
/// the crate's public re-exports.
type Index = <PolyIOP<Fr> as HyperPlonkSNARK<Bls12_381, Pcs, CpuBackend>>::Index;
type Proof = <PolyIOP<Fr> as HyperPlonkSNARK<Bls12_381, Pcs, CpuBackend>>::Proof;
type CpuProvingKey = <PolyIOP<Fr> as HyperPlonkSNARK<Bls12_381, Pcs, CpuBackend>>::ProvingKey;
type VerifyingKey = <PolyIOP<Fr> as HyperPlonkSNARK<Bls12_381, Pcs, CpuBackend>>::VerifyingKey;
#[cfg(feature = "ascend910b3")]
type NpuProvingKey =
    <PolyIOP<Fr> as HyperPlonkSNARK<Bls12_381, Pcs, Ascend910B3Backend>>::ProvingKey;

const MIN_NUM_VARS: usize = 2;
const MAX_NUM_VARS: usize = 25;
const DEFAULT_NUM_VARS: usize = 8;
const DEFAULT_WARMUPS: usize = 1;
const DEFAULT_SAMPLES: usize = 3;
const DEFAULT_SEED: u64 = 0x4850_4655_4c4c_5f50;
const SRS_SEED_DOMAIN: u64 = 0x5352_535f_5345_4544;
const FIXTURE_SEED_DOMAIN: u64 = 0x4850_5f46_4958_5455;
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
#[allow(dead_code)]
const NPU_FEATURE_ERROR: &str = concat!(
    "backend npu requires the ascend910b3 feature; build with ",
    "`cargo run --release -p hyperplonk --example full_prove_bench ",
    "--features ascend910b3 -- --backend npu ...`"
);

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
    backend: BackendKind,
    num_vars: usize,
    warmups: usize,
    samples: usize,
    seed: u64,
    srs_file: Option<String>,
}

/// Everything derived from the public gate and the seeded randomness. The
/// circuit is retained so both backends preprocess exactly the same index.
struct Fixture {
    circuit: MockCircuit<Fr>,
    num_vars: usize,
    num_constraints: usize,
    num_witness_columns: usize,
    num_selector_columns: usize,
    gate_degree: usize,
    num_pub_input: usize,
    permutation_len: usize,
    nontrivial_permutation_positions: usize,
    input_digest: String,
}

struct SrsSetup {
    srs: Srs,
    source: &'static str,
    path: Option<String>,
    num_vars: usize,
    setup_ms: f64,
    generation_ms: Option<f64>,
    load_ms: Option<f64>,
    seed: Option<u64>,
}

#[derive(Default)]
struct Measurement {
    warmup_ms: Vec<f64>,
    prove_ms: Vec<f64>,
    verify_ms: Vec<f64>,
}

struct Summary {
    samples: Vec<f64>,
    median: f64,
    min: f64,
    max: f64,
    total: f64,
}

struct BackendOutcome {
    actual_backend: &'static str,
    identity_json: String,
    context_create_ms: f64,
    preprocess_backend_ms: f64,
    preprocess_cpu_oracle_ms: Option<f64>,
    oracle_prove_ms: f64,
    oracle_verify_ms: f64,
    vk_equivalence: Option<&'static str>,
    measurement: Measurement,
    negative_control_ms: f64,
    negative_control_rejected: bool,
}

fn print_usage() {
    println!(
        "usage: full_prove_bench --backend <cpu|npu> [--num-vars <2..25>] \
         [--warmups <n>] [--samples <n>] [--seed <u64>] [--srs-file <path>]"
    );
    println!("  --backend   execution backend; npu requires the ascend910b3 feature");
    println!(
        "  --num-vars  number of variables, default {DEFAULT_NUM_VARS}; N >= 22 needs a memory check"
    );
    println!("  --warmups   untimed warm-up proofs, default {DEFAULT_WARMUPS}");
    println!("  --samples   timed proofs, default {DEFAULT_SAMPLES}");
    println!("  --seed      fixture/SRS seed, decimal or 0x-prefixed hex");
    println!(
        "  --srs-file  canonical universal SRS; loaded and dimension-validated, \
         or generated and written once if the path does not exist"
    );
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

fn parse_seed(value: &str) -> Result<u64, String> {
    let parsed = match value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        Some(hex) => u64::from_str_radix(hex, 16),
        None => value.parse::<u64>(),
    };
    parsed.map_err(|_| {
        format!("--seed expects a decimal or 0x-prefixed hexadecimal u64, got {value:?}")
    })
}

fn parse_args() -> Result<Args, String> {
    let mut backend = None;
    let mut num_vars = None;
    let mut warmups = None;
    let mut samples = None;
    let mut seed = None;
    let mut srs_file = None;
    let mut arguments = std::env::args().skip(1);
    while let Some(flag) = arguments.next() {
        match flag.as_str() {
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
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
            "--num-vars" => {
                num_vars = Some(parse_usize(&flag, &next_value(&mut arguments, &flag)?)?);
            }
            "--warmups" => {
                warmups = Some(parse_usize(&flag, &next_value(&mut arguments, &flag)?)?);
            }
            "--samples" => {
                samples = Some(parse_usize(&flag, &next_value(&mut arguments, &flag)?)?);
            }
            "--seed" => {
                seed = Some(parse_seed(&next_value(&mut arguments, &flag)?)?);
            }
            "--srs-file" => {
                srs_file = Some(next_value(&mut arguments, &flag)?);
            }
            other => return Err(format!("unknown argument {other:?}; try --help")),
        }
    }

    let num_vars = num_vars.unwrap_or(DEFAULT_NUM_VARS);
    if !(MIN_NUM_VARS..=MAX_NUM_VARS).contains(&num_vars) {
        return Err(format!(
            "--num-vars must be within {MIN_NUM_VARS}..={MAX_NUM_VARS}, got {num_vars}; \
             N >= 22 needs a memory feasibility check before running"
        ));
    }
    let samples = samples.unwrap_or(DEFAULT_SAMPLES);
    if samples == 0 {
        return Err("--samples must be at least 1".to_string());
    }
    let warmups = warmups.unwrap_or(DEFAULT_WARMUPS);
    let seed = seed.unwrap_or(DEFAULT_SEED);
    let backend = backend.unwrap_or(BackendKind::Cpu);

    Ok(Args {
        backend,
        num_vars,
        warmups,
        samples,
        seed,
        srs_file,
    })
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

impl Summary {
    fn new(values: &[f64]) -> Self {
        let (min, max) = if values.is_empty() {
            (0.0, 0.0)
        } else {
            (
                values.iter().copied().fold(f64::INFINITY, f64::min),
                values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            )
        };
        Self {
            samples: values.to_vec(),
            median: median(values),
            min,
            max,
            total: values.iter().sum(),
        }
    }

    fn json(&self) -> String {
        let samples = self
            .samples
            .iter()
            .map(|value| format!("{value:.6}"))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"samples\":[{samples}],\"median\":{:.6},\"min\":{:.6},\"max\":{:.6},\"total\":{:.6}}}",
            self.median, self.min, self.max, self.total
        )
    }
}

fn option_f64(value: Option<f64>) -> String {
    value
        .map(|inner| format!("{inner:.6}"))
        .unwrap_or_else(|| "null".to_string())
}

fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other if (other as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", other as u32));
            }
            other => out.push(other),
        }
    }
    out
}

fn push_field(out: &mut String, first: &mut bool, key: &str, value: &str) {
    if !*first {
        out.push(',');
    }
    *first = false;
    out.push('"');
    out.push_str(&json_escape(key));
    out.push_str("\":");
    out.push_str(value);
}

fn fnv1a(state: &mut u64, bytes: &[u8]) {
    for &byte in bytes {
        *state ^= byte as u64;
        *state = state.wrapping_mul(FNV_PRIME);
    }
}

fn fnv1a_usize(state: &mut u64, value: usize) {
    fnv1a(state, &(value as u64).to_le_bytes());
}

fn fnv1a_fr(state: &mut u64, value: &Fr, buffer: &mut Vec<u8>) -> Result<(), String> {
    buffer.clear();
    value
        .serialize_compressed(&mut *buffer)
        .map_err(|error| format!("serialize field element for the input digest: {error:?}"))?;
    fnv1a(state, buffer);
    Ok(())
}

fn nonzero_random(rng: &mut StdRng) -> Fr {
    loop {
        let value = Fr::rand(rng);
        if !value.is_zero() {
            return value;
        }
    }
}

/// Build a generic, fully active vanilla Plonk instance of `2^num_vars` rows.
///
/// Rows are generated in identical pairs: both rows of a pair carry the same
/// three witness values and the same five selector values. The permutation
/// swaps the two equal-valued rows of every pair inside each witness column,
/// which is a bijection whose every cycle has matching witness values. All
/// fixture validation (gate satisfaction, bijectivity, non-identity, connected
/// values, public-input layout) runs here, outside every timing window.
fn build_fixture(num_vars: usize, seed: u64) -> Result<Fixture, String> {
    let gate = CustomizedGates::vanilla_plonk_gate();
    let num_constraints = 1usize << num_vars;
    let num_witness_columns = gate.num_witness_columns();
    let num_selector_columns = gate.num_selector_columns();
    let gate_degree = gate.degree();
    if num_witness_columns != 3 || num_selector_columns != 5 || gate_degree != 3 {
        return Err(format!(
            "vanilla gate layout changed: d={num_witness_columns}, s={num_selector_columns}, \
             degree={gate_degree}"
        ));
    }

    // A minimal mock instance supplies the (unnameable) index type and the
    // vanilla parameters. Every size-dependent field is replaced below; only
    // the gate function and `num_pub_input = min(4, num_constraints)` are kept.
    let mut circuit = MockCircuit::<Fr>::new(4, &gate);
    if circuit.index.params.num_pub_input != 4 {
        return Err("MockCircuit did not produce the expected four public inputs".to_string());
    }
    circuit.index.params.num_constraints = num_constraints;

    let mut rng = StdRng::seed_from_u64(seed ^ FIXTURE_SEED_DOMAIN);
    let mut digest_state = FNV_OFFSET_BASIS;
    let mut digest_buffer: Vec<u8> = Vec::with_capacity(64);
    fnv1a(&mut digest_state, b"hyperplonk-full-prove-fixture-v1");
    for value in [
        num_vars,
        num_constraints,
        num_witness_columns,
        num_selector_columns,
        gate_degree,
        4,
    ] {
        fnv1a_usize(&mut digest_state, value);
    }
    fnv1a(&mut digest_state, &seed.to_le_bytes());

    let row_pairs = num_constraints / 2;
    let mut witnesses: Vec<WitnessColumn<Fr>> =
        vec![WitnessColumn::<Fr>::default(); num_witness_columns];
    let mut selectors: Vec<SelectorColumn<Fr>> =
        vec![SelectorColumn::<Fr>::default(); num_selector_columns];

    for _ in 0..row_pairs {
        // Full-range field elements; `w0 != w2` rules out the degenerate
        // `w0 = w2` gate path on every row and a non-zero `w2` lets the output
        // selector carry the residual.
        let row: [Fr; 3] = loop {
            let candidate = [
                Fr::rand(&mut rng),
                Fr::rand(&mut rng),
                Fr::rand(&mut rng),
            ];
            if candidate[0] != candidate[2] && !candidate[2].is_zero() {
                break candidate;
            }
        };

        // q_L, q_R, q_M and q_C are independent non-zero full-range values; the
        // output selector solves the unchanged gate equation
        //     q_L w0 + q_R w1 + q_O w2 + q_M w0 w1 + q_C = 0
        // and is required to be non-zero, so the output path stays active and
        // every selector term is non-zero.
        let (q_left, q_right, q_output, q_mul, q_const) = loop {
            let q_left = nonzero_random(&mut rng);
            let q_right = nonzero_random(&mut rng);
            let q_mul = nonzero_random(&mut rng);
            let q_const = nonzero_random(&mut rng);
            let residual = q_left * row[0] + q_right * row[1] + q_mul * row[0] * row[1] + q_const;
            let q_output = -residual / row[2];
            if !q_output.is_zero() {
                break (q_left, q_right, q_output, q_mul, q_const);
            }
        };
        let selector_row = [q_left, q_right, q_output, q_mul, q_const];

        for _ in 0..2 {
            for (column, value) in row.iter().enumerate() {
                witnesses[column].append(*value);
                fnv1a_fr(&mut digest_state, value, &mut digest_buffer)?;
            }
            for (column, value) in selector_row.iter().enumerate() {
                selectors[column].append(*value);
                fnv1a_fr(&mut digest_state, value, &mut digest_buffer)?;
            }
        }
    }

    for (column, witness) in witnesses.iter().enumerate() {
        if witness.coeff_ref().len() != num_constraints {
            return Err(format!(
                "witness column {column} has length {}, expected {num_constraints}",
                witness.coeff_ref().len()
            ));
        }
    }

    // Non-identity wiring: swap the two equal-valued rows of every pair within
    // each witness column. Labels follow the repository convention
    // `column * num_constraints + row`.
    let permutation_len = num_witness_columns * num_constraints;
    let mut permutation: Vec<usize> = (0..permutation_len).collect();
    for column in 0..num_witness_columns {
        for pair in 0..row_pairs {
            let first = column * num_constraints + 2 * pair;
            permutation.swap(first, first + 1);
        }
    }

    // Bijectivity of the actual label vector.
    let mut seen = vec![false; permutation_len];
    for &target in permutation.iter() {
        if target >= permutation_len {
            return Err(format!(
                "permutation label {target} is outside 0..{permutation_len}"
            ));
        }
        if seen[target] {
            return Err(format!("permutation label {target} appears twice"));
        }
        seen[target] = true;
    }
    if seen.iter().any(|value| !*value) {
        return Err("permutation is not surjective".to_string());
    }

    let nontrivial_permutation_positions = permutation
        .iter()
        .enumerate()
        .filter(|&(position, &target)| position != target)
        .count();
    if nontrivial_permutation_positions == 0 {
        return Err("permutation is the identity; non-identity copies are required".to_string());
    }

    // Every connected pair must carry equal witness values in its column.
    for position in 0..permutation_len {
        let target = permutation[position];
        let (column, row) = (position / num_constraints, position % num_constraints);
        let (target_column, target_row) = (target / num_constraints, target % num_constraints);
        if witnesses[column].coeff_ref()[row] != witnesses[target_column].coeff_ref()[target_row] {
            return Err(format!(
                "permutation connects unequal witness values at position {position}"
            ));
        }
    }

    fnv1a_usize(&mut digest_state, permutation_len);
    for &value in permutation.iter() {
        fnv1a(&mut digest_state, &(value as u64).to_le_bytes());
    }

    let num_pub_input = circuit.index.params.num_pub_input;
    if !num_pub_input.is_power_of_two() {
        return Err(format!(
            "public input length {num_pub_input} is not a power of two"
        ));
    }
    let public_inputs = witnesses[0].coeff_ref()[..num_pub_input].to_vec();
    for value in public_inputs.iter() {
        fnv1a_fr(&mut digest_state, value, &mut digest_buffer)?;
    }

    let permutation_fr: Vec<Fr> = permutation
        .iter()
        .map(|&value| Fr::from(value as u64))
        .collect();

    circuit.witnesses = witnesses;
    circuit.index.selectors = selectors;
    circuit.index.permutation = permutation_fr;
    circuit.public_inputs = public_inputs;

    if circuit.index.params.num_variables() != num_vars {
        return Err("fixture index reports the wrong number of variables".to_string());
    }
    if !circuit.is_satisfied() {
        return Err("constructed fixture violates the vanilla gate".to_string());
    }
    if circuit.public_inputs.len() != num_pub_input
        || circuit.public_inputs.as_slice() != &circuit.witnesses[0].coeff_ref()[..num_pub_input]
    {
        return Err("public inputs do not match the witness prefix".to_string());
    }

    Ok(Fixture {
        circuit,
        num_vars,
        num_constraints,
        num_witness_columns,
        num_selector_columns,
        gate_degree,
        num_pub_input,
        permutation_len,
        nontrivial_permutation_positions,
        input_digest: format!("fnv1a64:{digest_state:016x}"),
    })
}

fn validate_srs(srs: &Srs, num_vars: usize, origin: &str) -> Result<(), String> {
    if srs.num_vars < num_vars {
        return Err(format!(
            "SRS {origin} supports {} variables, but --num-vars is {num_vars}",
            srs.num_vars
        ));
    }
    if srs.powers_of_g.len() != srs.num_vars + 1 {
        return Err(format!(
            "SRS {origin} has {} power layers, expected {}",
            srs.powers_of_g.len(),
            srs.num_vars + 1
        ));
    }
    let expected_first = 1usize
        .checked_shl(srs.num_vars as u32)
        .ok_or_else(|| format!("SRS {origin} reports an unrepresentable num_vars"))?;
    if srs.powers_of_g.first().map(|layer| layer.evals.len()) != Some(expected_first) {
        return Err(format!(
            "SRS {origin} first power layer has the wrong length for {} variables",
            srs.num_vars
        ));
    }
    if srs.powers_of_g.last().map(|layer| layer.evals.len()) != Some(1) {
        return Err(format!(
            "SRS {origin} last power layer must contain the generator"
        ));
    }
    if srs.h_mask.len() != srs.num_vars {
        return Err(format!(
            "SRS {origin} has {} G2 mask points, expected {}",
            srs.h_mask.len(),
            srs.num_vars
        ));
    }
    Ok(())
}

fn load_or_generate_srs(
    num_vars: usize,
    seed: u64,
    path: Option<&str>,
) -> Result<SrsSetup, String> {
    let start = Instant::now();
    let srs_seed = seed ^ SRS_SEED_DOMAIN;

    if let Some(path) = path {
        match std::fs::File::open(path) {
            Ok(file) => {
                let load_start = Instant::now();
                let mut reader = BufReader::new(file);
                let srs = Srs::deserialize_compressed(&mut reader).map_err(|error| {
                    format!("deserialize canonical SRS from {path:?}: {error:?}")
                })?;
                let load_ms = millis(load_start.elapsed());
                validate_srs(&srs, num_vars, path)?;
                // A reusable universal SRS may be larger than requested; report
                // its actual dimension. Capture it before `srs` is moved.
                let actual_num_vars = srs.num_vars;
                Ok(SrsSetup {
                    srs,
                    source: "loaded_canonical_file",
                    path: Some(path.to_string()),
                    num_vars: actual_num_vars,
                    setup_ms: millis(start.elapsed()),
                    generation_ms: None,
                    load_ms: Some(load_ms),
                    seed: None,
                })
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                // Refuse to generate unless the parent directory already exists
                // and is a directory, so an unwritable tree cannot trigger a
                // large generation that could never be stored. This is only a
                // parent-shape check, not a claim about POSIX writability; the
                // `create_new` below stays the authority and never overwrites.
                if let Some(parent) = std::path::Path::new(path).parent() {
                    if !parent.as_os_str().is_empty() {
                        let metadata = std::fs::metadata(parent).map_err(|error| {
                            format!(
                                "inspect SRS parent directory {parent:?} before generation: {error}"
                            )
                        })?;
                        if !metadata.is_dir() {
                            return Err(format!("SRS parent {parent:?} is not a directory"));
                        }
                    }
                }
                let generation_start = Instant::now();
                let mut rng = StdRng::seed_from_u64(srs_seed);
                let srs = <Pcs as PolynomialCommitmentScheme<Bls12_381>>::gen_srs_for_testing(
                    &mut rng,
                    num_vars,
                )
                .map_err(|error| format!("generate SRS: {error}"))?;
                let generation_ms = millis(generation_start.elapsed());
                validate_srs(&srs, num_vars, path)?;
                let actual_num_vars = srs.num_vars;

                // Never overwrite an existing SRS: `create_new` fails if the
                // path appeared after the open above.
                let file = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)
                    .map_err(|error| {
                        format!("create canonical SRS file {path:?} without overwriting: {error}")
                    })?;
                let mut writer = BufWriter::new(file);
                srs.serialize_compressed(&mut writer)
                    .map_err(|error| format!("write canonical SRS to {path:?}: {error:?}"))?;
                writer.flush().map_err(|error| {
                    format!("flush canonical SRS file {path:?}: {error}")
                })?;

                Ok(SrsSetup {
                    srs,
                    source: "generated_and_written_canonical_file",
                    path: Some(path.to_string()),
                    num_vars: actual_num_vars,
                    setup_ms: millis(start.elapsed()),
                    generation_ms: Some(generation_ms),
                    load_ms: None,
                    seed: Some(srs_seed),
                })
            }
            Err(error) => Err(format!("open SRS file {path:?}: {error}")),
        }
    } else {
        let mut rng = StdRng::seed_from_u64(srs_seed);
        let srs =
            <Pcs as PolynomialCommitmentScheme<Bls12_381>>::gen_srs_for_testing(&mut rng, num_vars)
                .map_err(|error| format!("generate SRS: {error}"))?;
        let generation_ms = millis(start.elapsed());
        validate_srs(&srs, num_vars, "<memory>")?;
        let actual_num_vars = srs.num_vars;
        Ok(SrsSetup {
            srs,
            source: "generated_in_memory",
            path: None,
            num_vars: actual_num_vars,
            setup_ms: generation_ms,
            generation_ms: Some(generation_ms),
            load_ms: None,
            seed: Some(srs_seed),
        })
    }
}

fn cpu_preprocess(index: &Index, srs: &Srs) -> Result<(CpuProvingKey, VerifyingKey), String> {
    <PolyIOP<Fr> as HyperPlonkSNARK<Bls12_381, Pcs, CpuBackend>>::preprocess(
        &CpuBackend,
        index,
        srs,
    )
    .map_err(|error| format!("CPU preprocess: {error}"))
}

fn cpu_prove(
    backend: &CpuBackend,
    pk: &CpuProvingKey,
    public_inputs: &[Fr],
    witnesses: &[WitnessColumn<Fr>],
) -> Result<Proof, String> {
    <PolyIOP<Fr> as HyperPlonkSNARK<Bls12_381, Pcs, CpuBackend>>::prove(
        backend,
        pk,
        public_inputs,
        witnesses,
    )
    .map_err(|error| format!("CPU prove: {error}"))
}

fn verify_proof(
    vk: &VerifyingKey,
    public_inputs: &[Fr],
    proof: &Proof,
) -> Result<bool, String> {
    <PolyIOP<Fr> as HyperPlonkSNARK<Bls12_381, Pcs, CpuBackend>>::verify(
        vk,
        public_inputs,
        proof,
    )
    .map_err(|error| format!("host verification: {error}"))
}

/// Exact comparison of every publicly reachable proof field. Whole-proof
/// `PartialEq` is not available for this instantiation (its derived bounds
/// require `PCS: PartialEq`, which `MultilinearKzgPCS` does not implement), and
/// the batch-opening internals are `pub(crate)`; every proof is additionally
/// verified by the host verifier, which covers those parts.
fn compare_proofs(measured: &Proof, oracle: &Proof) -> Result<(), String> {
    if measured.witness_commits != oracle.witness_commits {
        return Err("witness commitments differ from the CPU oracle".to_string());
    }
    if measured.zero_check_proof != oracle.zero_check_proof {
        return Err("gate zero-check proof differs from the CPU oracle".to_string());
    }
    if measured.perm_check_proof.zero_check_proof != oracle.perm_check_proof.zero_check_proof {
        return Err("permutation zero-check proof differs from the CPU oracle".to_string());
    }
    if measured.perm_check_proof.prod_x_comm != oracle.perm_check_proof.prod_x_comm {
        return Err("product polynomial commitment differs from the CPU oracle".to_string());
    }
    if measured.perm_check_proof.frac_comm != oracle.perm_check_proof.frac_comm {
        return Err("fractional polynomial commitment differs from the CPU oracle".to_string());
    }
    if measured.batch_openings.f_i_eval_at_point_i
        != oracle.batch_openings.f_i_eval_at_point_i
    {
        return Err("batch-opening evaluations differ from the CPU oracle".to_string());
    }
    Ok(())
}

fn assert_vk_equivalence(left: &VerifyingKey, right: &VerifyingKey) -> Result<(), String> {
    if left.params != right.params {
        return Err("CPU and device verifying-key parameters differ".to_string());
    }
    if left.selector_commitments != right.selector_commitments {
        return Err("CPU and device selector commitments differ".to_string());
    }
    if left.perm_commitments != right.perm_commitments {
        return Err("CPU and device permutation commitments differ".to_string());
    }
    let mut left_param = Vec::new();
    left.pcs_param
        .serialize_compressed(&mut left_param)
        .map_err(|error| format!("serialize CPU verifier parameter: {error:?}"))?;
    let mut right_param = Vec::new();
    right
        .pcs_param
        .serialize_compressed(&mut right_param)
        .map_err(|error| format!("serialize device verifier parameter: {error:?}"))?;
    if left_param != right_param {
        return Err("CPU and device PCS verifier parameters differ".to_string());
    }
    Ok(())
}

/// Time and validate `warmups + samples` proofs. Every proof is compared with
/// the oracle and verified before it is accepted; the first failure aborts.
fn measure_proofs(
    warmups: usize,
    samples: usize,
    prove_once: &mut dyn FnMut() -> Result<Proof, String>,
    verify_once: &mut dyn FnMut(&Proof) -> Result<bool, String>,
    oracle: &Proof,
) -> Result<Measurement, String> {
    let mut measurement = Measurement::default();
    for run in 0..warmups + samples {
        let start = Instant::now();
        let proof = prove_once()?;
        let prove_ms = millis(start.elapsed());

        compare_proofs(&proof, oracle)?;

        let verify_start = Instant::now();
        let verified = verify_once(&proof)?;
        let verify_ms = millis(verify_start.elapsed());
        if !verified {
            return Err(format!(
                "proof {run} ({}) failed host verification",
                if run < warmups { "warm-up" } else { "measured" }
            ));
        }

        measurement.verify_ms.push(verify_ms);
        if run < warmups {
            measurement.warmup_ms.push(prove_ms);
            eprintln!("warm-up {run}: prove {prove_ms:.3} ms, verify {verify_ms:.3} ms");
        } else {
            measurement.prove_ms.push(prove_ms);
            eprintln!(
                "sample {}: prove {prove_ms:.3} ms, verify {verify_ms:.3} ms",
                run - warmups
            );
        }
    }
    Ok(measurement)
}

/// Untimed host-verifier negative control: a proof for the honest public input
/// must be rejected when verified against a tampered public input.
fn negative_control(
    vk: &VerifyingKey,
    fixture: &Fixture,
    oracle: &Proof,
) -> Result<(f64, bool), String> {
    let mut tampered = fixture.circuit.public_inputs.clone();
    if tampered.is_empty() {
        return Err("no public inputs to tamper with".to_string());
    }
    tampered[0] = tampered[0] + Fr::one();
    let start = Instant::now();
    let outcome = verify_proof(vk, &tampered, oracle);
    let elapsed = millis(start.elapsed());
    match outcome {
        Ok(true) => Err(
            "negative control failed: host verification accepted a tampered public input"
                .to_string(),
        ),
        _ => Ok((elapsed, true)),
    }
}

fn cpu_identity(threads: usize) -> String {
    format!(
        "{{\"name\":\"cpu\",\"execution\":\"rayon\",\"threads\":{threads},\"config_source\":\"runtime\"}}"
    )
}

#[cfg(feature = "ascend910b3")]
fn npu_identity() -> String {
    String::from(
        "{\"name\":\"ascend910b3\",\"execution\":\"native_single_stream\",\"device_id\":0,\
         \"soc\":\"Ascend910B3 (DAV_2201)\",\"documented_aiv_cores\":40,\
         \"documented_aic_cores\":20,\"documented_ub_bytes\":196352,\
         \"config_source\":\"documented_target_config_not_measured\"}",
    )
}

fn run_cpu(
    args: &Args,
    fixture: &Fixture,
    srs_setup: &SrsSetup,
    fixture_build_ms: f64,
) -> Result<(), String> {
    let backend = CpuBackend;
    let context_create_ms = 0.0;

    let preprocess_start = Instant::now();
    let (pk, vk) = cpu_preprocess(&fixture.circuit.index, &srs_setup.srs)?;
    let preprocess_ms = millis(preprocess_start.elapsed());

    let oracle_start = Instant::now();
    let oracle = cpu_prove(
        &backend,
        &pk,
        &fixture.circuit.public_inputs,
        &fixture.circuit.witnesses,
    )?;
    let oracle_prove_ms = millis(oracle_start.elapsed());

    let oracle_verify_start = Instant::now();
    if !verify_proof(&vk, &fixture.circuit.public_inputs, &oracle)? {
        return Err("CPU oracle proof failed host verification".to_string());
    }
    let oracle_verify_ms = millis(oracle_verify_start.elapsed());

    let mut prove = || {
        cpu_prove(
            &backend,
            &pk,
            &fixture.circuit.public_inputs,
            &fixture.circuit.witnesses,
        )
    };
    let mut verify =
        |proof: &Proof| verify_proof(&vk, &fixture.circuit.public_inputs, proof);
    let measurement =
        measure_proofs(args.warmups, args.samples, &mut prove, &mut verify, &oracle)?;

    let (negative_control_ms, negative_control_rejected) =
        negative_control(&vk, fixture, &oracle)?;

    let threads = rayon::current_num_threads();
    let outcome = BackendOutcome {
        actual_backend: "cpu",
        identity_json: cpu_identity(threads),
        context_create_ms,
        preprocess_backend_ms: preprocess_ms,
        preprocess_cpu_oracle_ms: None,
        oracle_prove_ms,
        oracle_verify_ms,
        vk_equivalence: None,
        measurement,
        negative_control_ms,
        negative_control_rejected,
    };
    emit_report(args, fixture, fixture_build_ms, srs_setup, &outcome);
    Ok(())
}

#[cfg(feature = "ascend910b3")]
fn npu_preprocess(
    backend: &Ascend910B3Backend,
    index: &Index,
    srs: &Srs,
) -> Result<(NpuProvingKey, VerifyingKey), String> {
    let (pk, vk) = <PolyIOP<Fr> as HyperPlonkSNARK<Bls12_381, Pcs, Ascend910B3Backend>>::preprocess(
        backend, index, srs,
    )
    .map_err(|error| format!("device preprocess: {error}"))?;
    Ok((pk, vk))
}

#[cfg(feature = "ascend910b3")]
fn npu_prove(
    backend: &Ascend910B3Backend,
    pk: &NpuProvingKey,
    public_inputs: &[Fr],
    witnesses: &[WitnessColumn<Fr>],
) -> Result<Proof, String> {
    <PolyIOP<Fr> as HyperPlonkSNARK<Bls12_381, Pcs, Ascend910B3Backend>>::prove(
        backend,
        pk,
        public_inputs,
        witnesses,
    )
    .map_err(|error| format!("device prove: {error}"))
}

#[cfg(feature = "ascend910b3")]
fn run_npu_impl(
    args: &Args,
    fixture: &Fixture,
    srs_setup: &SrsSetup,
    fixture_build_ms: f64,
) -> Result<(), String> {
    let context_start = Instant::now();
    let backend = Ascend910B3Backend::new(0)
        .map_err(|error| format!("create Ascend910B3 context: {error}"))?;
    let context_create_ms = millis(context_start.elapsed());

    let device_preprocess_start = Instant::now();
    let (device_pk, device_vk) = npu_preprocess(&backend, &fixture.circuit.index, &srs_setup.srs)?;
    let device_preprocess_ms = millis(device_preprocess_start.elapsed());

    let cpu_preprocess_start = Instant::now();
    let (cpu_pk, cpu_vk) = cpu_preprocess(&fixture.circuit.index, &srs_setup.srs)?;
    let cpu_preprocess_ms = millis(cpu_preprocess_start.elapsed());

    assert_vk_equivalence(&device_vk, &cpu_vk)?;

    let oracle_start = Instant::now();
    let oracle = cpu_prove(
        &CpuBackend,
        &cpu_pk,
        &fixture.circuit.public_inputs,
        &fixture.circuit.witnesses,
    )?;
    let oracle_prove_ms = millis(oracle_start.elapsed());
    // The CPU prepared key is no longer needed; release it before the measured
    // loop so the device run does not pay for an extra host copy of the SRS.
    drop(cpu_pk);

    let oracle_verify_start = Instant::now();
    if !verify_proof(&cpu_vk, &fixture.circuit.public_inputs, &oracle)? {
        return Err("CPU oracle proof failed host verification".to_string());
    }
    let oracle_verify_ms = millis(oracle_verify_start.elapsed());

    let mut prove = || {
        npu_prove(
            &backend,
            &device_pk,
            &fixture.circuit.public_inputs,
            &fixture.circuit.witnesses,
        )
    };
    let mut verify =
        |proof: &Proof| verify_proof(&device_vk, &fixture.circuit.public_inputs, proof);
    let measurement =
        measure_proofs(args.warmups, args.samples, &mut prove, &mut verify, &oracle)?;

    let (negative_control_ms, negative_control_rejected) =
        negative_control(&device_vk, fixture, &oracle)?;

    let outcome = BackendOutcome {
        actual_backend: "ascend910b3",
        identity_json: npu_identity(),
        context_create_ms,
        preprocess_backend_ms: device_preprocess_ms,
        preprocess_cpu_oracle_ms: Some(cpu_preprocess_ms),
        oracle_prove_ms,
        oracle_verify_ms,
        vk_equivalence: Some("equivalent"),
        measurement,
        negative_control_ms,
        negative_control_rejected,
    };
    emit_report(args, fixture, fixture_build_ms, srs_setup, &outcome);
    Ok(())
}

#[cfg(feature = "ascend910b3")]
fn run_npu(
    args: &Args,
    fixture: &Fixture,
    srs_setup: &SrsSetup,
    fixture_build_ms: f64,
) -> Result<(), String> {
    run_npu_impl(args, fixture, srs_setup, fixture_build_ms)
}

#[cfg(not(feature = "ascend910b3"))]
fn run_npu(
    args: &Args,
    fixture: &Fixture,
    srs_setup: &SrsSetup,
    fixture_build_ms: f64,
) -> Result<(), String> {
    let _ = (args, fixture, srs_setup, fixture_build_ms);
    Err(NPU_FEATURE_ERROR.to_string())
}

fn emit_report(
    args: &Args,
    fixture: &Fixture,
    fixture_build_ms: f64,
    srs_setup: &SrsSetup,
    outcome: &BackendOutcome,
) {
    let threads = rayon::current_num_threads();
    let mut compiled_features: Vec<&str> = Vec::new();
    if cfg!(feature = "cpu") {
        compiled_features.push("cpu");
    }
    if cfg!(feature = "ascend910b3") {
        compiled_features.push("ascend910b3");
    }
    if cfg!(feature = "print-trace") {
        compiled_features.push("print-trace");
    }
    let features_json = format!(
        "[{}]",
        compiled_features
            .iter()
            .map(|feature| format!("\"{feature}\""))
            .collect::<Vec<_>>()
            .join(",")
    );

    let prove_summary = Summary::new(&outcome.measurement.prove_ms);
    let warmup_summary = Summary::new(&outcome.measurement.warmup_ms);
    let verify_summary = Summary::new(&outcome.measurement.verify_ms);

    let backend_json = format!(
        "{{\"requested\":\"{}\",\"actual\":\"{}\",\"identity\":{},\"compiled_features\":{}}}",
        args.backend.as_str(),
        outcome.actual_backend,
        outcome.identity_json,
        features_json
    );
    let workload_json = format!(
        "{{\"kind\":\"hyperplonk_full_prove\",\"field\":\"BLS12-381 Fr (255-bit prime)\",\
         \"gate\":\"vanilla_plonk_gate\",\"num_vars\":{},\"num_constraints\":{},\
         \"num_witness_columns\":{},\"num_selector_columns\":{},\"gate_degree\":{},\
         \"num_pub_input\":{},\"row_pairs\":{},\"permutation_len\":{},\
         \"nontrivial_permutation_positions\":{},\"seed\":\"0x{:016x}\",\
         \"input_digest\":\"{}\",\
         \"input_digest_scope\":\"exact selectors, witnesses, permutation and public inputs \
         actually proved; SRS reported separately\"}}",
        fixture.num_vars,
        fixture.num_constraints,
        fixture.num_witness_columns,
        fixture.num_selector_columns,
        fixture.gate_degree,
        fixture.num_pub_input,
        fixture.num_constraints / 2,
        fixture.permutation_len,
        fixture.nontrivial_permutation_positions,
        args.seed,
        fixture.input_digest
    );
    let srs_json = format!(
        "{{\"source\":\"{}\",\"path\":{},\"num_vars\":{},\"target_num_vars\":{},\
         \"setup_ms\":{:.6},\
         \"generation_ms\":{},\"load_ms\":{},\"generation_seed\":{},\
         \"validation\":\"canonical_checked_dimensions_and_layer_sizes\"}}",
        srs_setup.source,
        srs_setup
            .path
            .as_ref()
            .map(|path| format!("\"{}\"", json_escape(path)))
            .unwrap_or_else(|| "null".to_string()),
        srs_setup.num_vars,
        args.num_vars,
        srs_setup.setup_ms,
        option_f64(srs_setup.generation_ms),
        option_f64(srs_setup.load_ms),
        srs_setup
            .seed
            .map(|seed| format!("\"0x{seed:016x}\""))
            .unwrap_or_else(|| "null".to_string())
    );
    let configuration_json = format!(
        "{{\"warmups\":{},\"samples\":{},\"cpu_threads\":{},\
         \"cpu_thread_pool\":\"rayon_current_num_threads\"}}",
        args.warmups, args.samples, threads
    );
    let phase_json = format!(
        "{{\"fixture_build\":{:.6},\"context_create\":{:.6},\"preprocess_backend\":{:.6},\
         \"preprocess_cpu_oracle\":{},\"cpu_oracle_prove\":{:.6},\
         \"cpu_oracle_verify\":{:.6},\"warmup_prove\":{},\"verification\":{},\
         \"negative_control\":{:.6}}}",
        fixture_build_ms,
        outcome.context_create_ms,
        outcome.preprocess_backend_ms,
        option_f64(outcome.preprocess_cpu_oracle_ms),
        outcome.oracle_prove_ms,
        outcome.oracle_verify_ms,
        warmup_summary.json(),
        verify_summary.json(),
        outcome.negative_control_ms
    );
    let vk_equivalence = match outcome.vk_equivalence {
        Some(value) => format!("\"{value}\""),
        None => "\"not_applicable_cpu_backend\"".to_string(),
    };
    let correctness_json = format!(
        "{{\"oracle_proof_verified\":true,\"verified_proofs\":{},\
         \"warmups_also_verified\":{},\"proof_comparison\":\"exact on every publicly reachable \
         field\",\"compared_fields\":[\"witness_commits\",\"zero_check_proof\",\
         \"perm_check_proof.zero_check_proof\",\"perm_check_proof.prod_x_comm\",\
         \"perm_check_proof.frac_comm\",\"batch_openings.f_i_eval_at_point_i\"],\
         \"vk_equivalence\":{},\
         \"verification\":\"host CPU pairing via multilinear KZG batch_verify\",\
         \"negative_control\":{{\"kind\":\"tampered_public_input\",\"rejected\":{},\
         \"elapsed_ms\":{:.6}}},\
         \"limitations\":[\"whole-proof PartialEq is not available for this instantiation: the \
         derived bounds require PCS: PartialEq, which MultilinearKzgPCS does not implement\",\
         \"batch-opening inner sum-check messages and the quotient opening are pub(crate) and \
         unreachable from an external example\",\
         \"proof serialization is not implemented; the timing boundary is the ready proof \
         object, not serialized bytes\"]}}",
        outcome.measurement.verify_ms.len(),
        args.warmups,
        vk_equivalence,
        outcome.negative_control_rejected,
        outcome.negative_control_ms
    );
    let boundary_json = String::from(concat!(
        "{\"prove_wall_scope\":\"external Instant immediately before the entire \
         <PolyIOP<Fr> as HyperPlonkSNARK<...>>::prove call through the returned proof object; \
         includes witness clone/encode/H2D, witness commitments, gate and permutation \
         zero-checks, all evaluate_mle queries, frac/prod commitments, batched multi_open \
         (final PCS::open synchronizes), host Fiat-Shamir, proof assembly and internal drops\",",
        "\"excluded_from_prove_wall\":[\"srs_setup\",\"context_create\",\"preprocess\",",
        "\"fixture_build\",\"cpu_oracle\",\"verification\",\"warmup_proofs\"],",
        "\"proof_serialization\":\"not_implemented\",",
        "\"proof_boundary\":\"proof_object_ready_not_serialized_bytes\"}"
    ));

    let mut report = String::from("{");
    let mut first = true;
    push_field(
        &mut report,
        &mut first,
        "schema",
        "\"hyperplonk_full_prove_bench_v1\"",
    );
    push_field(&mut report, &mut first, "backend", &backend_json);
    push_field(&mut report, &mut first, "workload", &workload_json);
    push_field(&mut report, &mut first, "srs", &srs_json);
    push_field(&mut report, &mut first, "configuration", &configuration_json);
    push_field(
        &mut report,
        &mut first,
        "prove_wall_ms",
        &prove_summary.json(),
    );
    push_field(&mut report, &mut first, "phase_ms", &phase_json);
    push_field(&mut report, &mut first, "boundary", &boundary_json);
    push_field(&mut report, &mut first, "correctness", &correctness_json);
    report.push('}');
    println!("{report}");
}

fn run(args: &Args) -> Result<(), String> {
    // Reject an unsupported backend before any fixture or SRS work, so an
    // NPU invocation without the compiled feature has no resource side effects
    // (no fixture build, no SRS generation, load, or file write).
    if args.backend == BackendKind::Npu && !cfg!(feature = "ascend910b3") {
        return Err(NPU_FEATURE_ERROR.to_string());
    }

    let fixture_start = Instant::now();
    let fixture = build_fixture(args.num_vars, args.seed)?;
    let fixture_build_ms = millis(fixture_start.elapsed());
    eprintln!(
        "fixture: N={} constraints={} d={} s={} gate_degree={} digest={}",
        fixture.num_vars,
        fixture.num_constraints,
        fixture.num_witness_columns,
        fixture.num_selector_columns,
        fixture.gate_degree,
        fixture.input_digest
    );

    let srs_setup = load_or_generate_srs(args.num_vars, args.seed, args.srs_file.as_deref())?;
    eprintln!(
        "srs: source={} num_vars={} setup {:.3} ms",
        srs_setup.source, srs_setup.num_vars, srs_setup.setup_ms
    );

    match args.backend {
        BackendKind::Cpu => run_cpu(args, &fixture, &srs_setup, fixture_build_ms),
        BackendKind::Npu => run_npu(args, &fixture, &srs_setup, fixture_build_ms),
    }
}

fn main() {
    let result = parse_args().and_then(|args| run(&args));
    if let Err(error) = result {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
