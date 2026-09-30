// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_std::log2;

/// A column of selectors of length `#constraints`
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SelectorColumn<F: PrimeField>(pub(crate) Vec<F>);

impl<F: PrimeField> SelectorColumn<F> {
    /// the number of variables of the multilinear polynomial that presents a
    /// column.
    pub fn get_nv(&self) -> usize {
        log2(self.0.len()) as usize
    }

    /// Append a new element to the selector column
    pub fn append(&mut self, new_element: F) {
        self.0.push(new_element)
    }
}

impl<F: PrimeField> From<&SelectorColumn<F>> for DenseMultilinearExtension<F> {
    fn from(witness: &SelectorColumn<F>) -> Self {
        let nv = witness.get_nv();
        Self::from_evaluations_slice(nv, witness.0.as_ref())
    }
}
