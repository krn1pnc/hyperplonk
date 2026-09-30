// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

use ark_std::log2;

/// Decompose an integer into a binary vector in little endian.
pub fn bit_decompose(input: u64, num_var: usize) -> Vec<bool> {
    let mut res = Vec::with_capacity(num_var);
    let mut i = input;
    for _ in 0..num_var {
        res.push(i & 1 == 1);
        i >>= 1;
    }
    res
}

/// Return the number of variables that one need for an MLE to
/// batch the list of MLEs
#[inline]
pub fn get_batched_nv(num_var: usize, polynomials_len: usize) -> usize {
    num_var + log2(polynomials_len) as usize
}

#[cfg(test)]
mod test {
    use super::bit_decompose;
    use ark_std::{rand::RngCore, test_rng};

    /// Project a little endian binary vector into an integer.
    fn project(input: &[bool]) -> u64 {
        let mut res = 0;
        for &e in input.iter().rev() {
            res <<= 1;
            res += e as u64;
        }
        res
    }

    #[test]
    fn test_decomposition() {
        let mut rng = test_rng();
        for _ in 0..100 {
            let t = rng.next_u64();
            let b = bit_decompose(t, 64);
            let r = project(&b);
            assert_eq!(t, r)
        }
    }
}
