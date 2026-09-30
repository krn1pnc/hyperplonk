// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_std::log2;

/// A column of witnesses of length `#constraints`
#[derive(Debug, Clone, Default)]
pub struct WitnessColumn<F: PrimeField>(pub(crate) Vec<F>);

impl<F: PrimeField> WitnessColumn<F> {
    /// the number of variables of the multilinear polynomial that presents a
    /// column.
    pub fn get_nv(&self) -> usize {
        log2(self.0.len()) as usize
    }

    /// Append a new element to the witness column
    pub fn append(&mut self, new_element: F) {
        self.0.push(new_element)
    }

    pub fn coeff_ref(&self) -> &[F] {
        self.0.as_ref()
    }
}

impl<F: PrimeField> From<&WitnessColumn<F>> for DenseMultilinearExtension<F> {
    fn from(witness: &WitnessColumn<F>) -> Self {
        let nv = witness.get_nv();
        Self::from_evaluations_slice(nv, witness.0.as_ref())
    }
}
