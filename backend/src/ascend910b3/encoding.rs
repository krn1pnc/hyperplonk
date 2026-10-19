//! Canonical host packets, not device layouts or transcript serialization.
//!
//! Fields are unsigned little-endian canonical integers (Fr: 32 bytes, Fq: 48
//! bytes). G1 packets contain a one-byte tag followed by x and y in that Fq
//! format. In particular, these are not the standard BLS point encodings.

use crate::BackendError;
use ark_bls12_381::{Fq, Fr, G1Affine};
use ark_ec::AffineRepr;
use ark_ff::{BigInt, PrimeField};

pub(crate) fn encode_fr(value: &Fr) -> [u8; 32] {
    let mut bytes = [0; 32];
    for (chunk, limb) in bytes.chunks_exact_mut(8).zip(value.into_bigint().0) {
        chunk.copy_from_slice(&limb.to_le_bytes());
    }
    bytes
}

/// Pack one canonical host packet without allocating once per field element.
pub(super) fn encode_fields(values: &[Fr]) -> Vec<u8> {
    // Fr occupies 32 bytes, so a valid input slice already bounds this capacity.
    let mut bytes = Vec::with_capacity(values.len() * 32);
    for value in values {
        bytes.extend_from_slice(&encode_fr(value));
    }
    bytes
}

pub(crate) fn decode_fr(bytes: &[u8]) -> Result<Fr, BackendError> {
    if bytes.len() != 32 {
        return Err(BackendError::InvalidState(
            "invalid Fr packet length".into(),
        ));
    }
    let mut limbs = [0; 4];
    for (limb, chunk) in limbs.iter_mut().zip(bytes.chunks_exact(8)) {
        let mut encoded = [0; 8];
        encoded.copy_from_slice(chunk);
        *limb = u64::from_le_bytes(encoded);
    }
    Fr::from_bigint(BigInt::<4>::new(limbs))
        .ok_or_else(|| BackendError::InvalidState("noncanonical Fr packet".into()))
}

fn encode_fq(value: &Fq) -> [u8; 48] {
    let mut bytes = [0; 48];
    for (chunk, limb) in bytes.chunks_exact_mut(8).zip(value.into_bigint().0) {
        chunk.copy_from_slice(&limb.to_le_bytes());
    }
    bytes
}

fn decode_fq(bytes: &[u8]) -> Result<Fq, BackendError> {
    if bytes.len() != 48 {
        return Err(BackendError::InvalidState(
            "invalid Fq packet length".into(),
        ));
    }
    let mut limbs = [0; 6];
    for (limb, chunk) in limbs.iter_mut().zip(bytes.chunks_exact(8)) {
        let mut encoded = [0; 8];
        encoded.copy_from_slice(chunk);
        *limb = u64::from_le_bytes(encoded);
    }
    Fq::from_bigint(BigInt::<6>::new(limbs))
        .ok_or_else(|| BackendError::InvalidState("noncanonical Fq packet".into()))
}

pub(crate) fn encode_g1(value: &G1Affine) -> [u8; 97] {
    let mut bytes = [0; 97];
    if let Some((x, y)) = value.xy() {
        bytes[1..49].copy_from_slice(&encode_fq(&x));
        bytes[49..97].copy_from_slice(&encode_fq(&y));
    } else {
        bytes[0] = 1;
    }
    bytes
}

pub(crate) fn decode_g1(bytes: &[u8]) -> Result<G1Affine, BackendError> {
    if bytes.len() != 97 {
        return Err(BackendError::InvalidState(
            "invalid G1 packet length".into(),
        ));
    }
    match bytes[0] {
        1 if bytes[1..].iter().all(|&byte| byte == 0) => Ok(G1Affine::identity()),
        0 => {
            // Strict field decoding precedes unchecked construction: unlike new(),
            // this cannot assert on an untrusted native point.
            let point =
                G1Affine::new_unchecked(decode_fq(&bytes[1..49])?, decode_fq(&bytes[49..97])?);
            if point.is_zero()
                || !point.is_on_curve()
                || !point.is_in_correct_subgroup_assuming_on_curve()
            {
                return Err(BackendError::InvalidState(
                    "invalid finite G1 packet".into(),
                ));
            }
            Ok(point)
        },
        _ => Err(BackendError::InvalidState(
            "invalid G1 tag or infinity coordinates".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ec::CurveGroup;
    use ark_ff::{AdditiveGroup, Field};

    fn assert_invalid<T>(result: Result<T, BackendError>) {
        assert!(matches!(result, Err(BackendError::InvalidState(_))));
    }

    #[test]
    fn fr_canonical_roundtrips_and_little_endian_one() {
        let mixed = Fr::from_bigint(BigInt::<4>::new([5, 1, 1, 1])).unwrap();
        for value in [Fr::ZERO, Fr::ONE, -Fr::ONE, mixed] {
            assert_eq!(decode_fr(&encode_fr(&value)).unwrap(), value);
        }
        let mut one = [0; 32];
        one[0] = 1;
        assert_eq!(encode_fr(&Fr::ONE), one);
        let encoded = encode_fr(&mixed);
        assert_eq!(encoded[0], 5);
        for offset in [8, 16, 24] {
            assert_eq!(encoded[offset], 1);
        }
    }

    #[test]
    fn fr_rejects_noncanonical_values_and_lengths() {
        let mut modulus = [0; 32];
        for (chunk, limb) in modulus.chunks_exact_mut(8).zip(Fr::MODULUS.0) {
            chunk.copy_from_slice(&limb.to_le_bytes());
        }
        assert_invalid(decode_fr(&modulus));
        assert_invalid(decode_fr(&[0xff; 32]));
        assert_invalid(decode_fr(&[0; 31]));
        assert_invalid(decode_fr(&[0; 33]));
    }

    #[test]
    fn fq_canonical_roundtrips_and_little_endian_one() {
        let mixed = Fq::from_bigint(BigInt::<6>::new([5, 1, 2, 3, 4, 1])).unwrap();
        for value in [Fq::ZERO, Fq::ONE, -Fq::ONE, mixed] {
            assert_eq!(decode_fq(&encode_fq(&value)).unwrap(), value);
        }
        let mut one = [0; 48];
        one[0] = 1;
        assert_eq!(encode_fq(&Fq::ONE), one);
        let encoded = encode_fq(&mixed);
        for (offset, value) in [(0, 5), (8, 1), (16, 2), (24, 3), (32, 4), (40, 1)] {
            assert_eq!(encoded[offset], value);
        }
    }

    #[test]
    fn fq_rejects_noncanonical_values_and_lengths() {
        let mut modulus = [0; 48];
        for (chunk, limb) in modulus.chunks_exact_mut(8).zip(Fq::MODULUS.0) {
            chunk.copy_from_slice(&limb.to_le_bytes());
        }
        assert_invalid(decode_fq(&modulus));
        assert_invalid(decode_fq(&[0xff; 48]));
        assert_invalid(decode_fq(&[0; 47]));
        assert_invalid(decode_fq(&[0; 49]));
    }

    #[test]
    fn g1_roundtrips_identity_and_fixed_subgroup_points() {
        let generator = G1Affine::generator();
        for point in [
            G1Affine::identity(),
            generator,
            -generator,
            (generator * Fr::from(2u64)).into_affine(),
            (generator * Fr::from(7u64)).into_affine(),
        ] {
            assert_eq!(decode_g1(&encode_g1(&point)).unwrap(), point);
        }
        let mut infinity = [0; 97];
        infinity[0] = 1;
        assert_eq!(encode_g1(&G1Affine::identity()), infinity);
    }

    #[test]
    fn g1_rejects_bad_tags_coordinates_and_lengths() {
        assert_invalid(decode_g1(&[0; 97])); // finite-tag identity
        let mut packet = [0; 97];
        packet[0] = 2;
        assert_invalid(decode_g1(&packet));
        packet[0] = 1;
        packet[1] = 1;
        assert_invalid(decode_g1(&packet));
        packet[1] = 0;
        packet[96] = 1;
        assert_invalid(decode_g1(&packet));

        let generator = encode_g1(&G1Affine::generator());
        let mut modulus = [0; 48];
        for (chunk, limb) in modulus.chunks_exact_mut(8).zip(Fq::MODULUS.0) {
            chunk.copy_from_slice(&limb.to_le_bytes());
        }
        for coordinate in [1..49, 49..97] {
            let mut invalid = generator;
            invalid[coordinate].copy_from_slice(&modulus);
            assert_invalid(decode_g1(&invalid));
        }
        let mut off_curve = [0; 97];
        off_curve[1] = 1;
        off_curve[49] = 1;
        assert_invalid(decode_g1(&off_curve));
        assert_invalid(decode_g1(&[0; 96]));
        assert_invalid(decode_g1(&[0; 98]));
    }

    #[test]
    fn g1_rejects_fixed_on_curve_non_subgroup_point() {
        let point = G1Affine::new_unchecked(Fq::ZERO, Fq::from(2u64));
        assert!(!point.is_zero());
        assert!(point.is_on_curve());
        assert!(!point.is_in_correct_subgroup_assuming_on_curve());
        assert_invalid(decode_g1(&encode_g1(&point)));
    }
}
