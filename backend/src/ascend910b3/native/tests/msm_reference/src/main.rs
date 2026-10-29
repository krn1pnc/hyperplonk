use ark_bls12_381::{Fr, G1Affine, G1Projective};
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{PrimeField};
use ark_std::{rand::{rngs::StdRng, seq::SliceRandom, SeedableRng}, UniformRand, Zero};
use std::{fs::{self, File}, io::{BufWriter, Write}, path::PathBuf, time::Instant};

fn encode_field<F: PrimeField>(value: F, bytes: &mut [u8]) {
    for (limb, chunk) in value.into_bigint().as_ref().iter().zip(bytes.chunks_exact_mut(8)) {
        chunk.copy_from_slice(&limb.to_le_bytes());
    }
}
fn encode(p: &G1Affine) -> [u8; 96] {
    let mut bytes = [0; 96];
    if !p.is_zero() {
        encode_field(p.x, &mut bytes[..48]);
        encode_field(p.y, &mut bytes[48..]);
    }
    bytes
}
fn msm(bases: &[G1Affine], scalars: &[Fr]) -> (G1Affine, f64) {
    let start = Instant::now();
    let result = G1Projective::msm_unchecked(bases, scalars).into_affine();
    (result, start.elapsed().as_secs_f64())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 { return Err("usage: msm_reference OUTPUT_DIRECTORY COUNT".into()); }
    let out = PathBuf::from(&args[1]);
    fs::create_dir_all(&out)?;
    let n: usize = args[2].parse()?;
    if n == 0 || n > u32::MAX as usize { return Err("count must be in 1..=UINT32_MAX".into()); }
    let seed = 20261103 + n as u64;
    rayon::ThreadPoolBuilder::new().num_threads(16).build_global()?;
    let preparation = Instant::now();
    // Distinct nonzero subgroup points G, 2G, ..., nG; n < subgroup order.
    // Bound normalization temporaries to 2^18 rather than duplicating n projectives.
    let generator = G1Affine::generator();
    let mut p = G1Projective::zero();
    let mut bases = Vec::with_capacity(n);
    let mut batch = Vec::with_capacity(1 << 18);
    while bases.len() < n {
        batch.clear();
        for _ in 0..(n - bases.len()).min(1 << 18) {
            p += generator;
            batch.push(p);
        }
        bases.extend(G1Projective::normalize_batch(&batch));
    }
    drop(batch);
    let mut rng = StdRng::seed_from_u64(seed);
    bases.shuffle(&mut rng);
    // UniformRand samples the full Fr domain using rejection, not small scalars.
    let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
    let mut b = BufWriter::with_capacity(1 << 20, File::create(out.join("bases.bin"))?);
    for point in &bases { b.write_all(&encode(point))?; }
    b.flush()?;
    let mut s = BufWriter::with_capacity(1 << 20, File::create(out.join("scalars.bin"))?);
    for scalar in &scalars {
        let mut bytes = [0; 32];
        encode_field(*scalar, &mut bytes);
        s.write_all(&bytes)?;
    }
    s.flush()?;
    drop(b); drop(s);
    let preparation_s = preparation.elapsed().as_secs_f64();
    eprintln!("INPUTS_READY n={n} seed={seed} preparation_s={preparation_s:.9}");
    // This full independent arkworks MSM is a reference call, NOT the warmup.
    let (expected, reference_s) = msm(&bases, &scalars);
    let encoded = encode(&expected);
    fs::write(out.join("expected.bin"), &encoded)?;
    eprintln!("REFERENCE elapsed_s={reference_s:.9}");
    let (warm, warmup_s) = msm(&bases, &scalars);
    assert_eq!(warm, expected, "warmup differs from independent reference");
    eprintln!("WARMUP elapsed_s={warmup_s:.9} exact=true");
    let mut samples = [0.0; 3];
    for i in 0..3 {
        let (result, elapsed) = msm(&bases, &scalars);
        assert_eq!(result, expected, "sample differs from independent reference");
        samples[i] = elapsed;
        eprintln!("SAMPLE index={i} elapsed_s={elapsed:.9} exact=true");
    }
    let mut sorted = samples; sorted.sort_by(f64::total_cmp);
    let mut hex = String::with_capacity(192);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in encoded {
        hex.push(HEX[(byte >> 4) as usize] as char);
        hex.push(HEX[(byte & 15) as usize] as char);
    }
    let report = format!("{{\"count\":{n},\"threads\":16,\"seed\":{seed},\"generator_method\":\"shuffle G..nG with StdRng seed=20261103+n, then full-domain UniformRand Fr from same RNG\",\"normalization_batch\":262144,\"preparation_s\":{preparation_s:.9},\"reference_s\":{reference_s:.9},\"reference_is_warmup\":false,\"warmup\":1,\"samples\":3,\"warmup_s\":{warmup_s:.9},\"sample_s\":{samples:?},\"median_s\":{:.9},\"exact_match\":true,\"result_hex\":\"{hex}\",\"boundary\":\"resident decoded affine bases and Fr scalars through full msm_unchecked and final into_affine; excludes input generation, IO, decoding, initial input allocations and result serialization; includes internal MSM allocations\",\"implementation\":\"arkworks 0.6.0 BLS12-381 VariableBaseMSM release, 16 Rayon workers\"}}\n", sorted[1]);
    fs::write(out.join("cpu-calibration.json"), &report)?;
    print!("{report}");
    Ok(())
}
