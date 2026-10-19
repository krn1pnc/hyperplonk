//! Immutable resident MLE operations forwarded to the native single-stream bridge.

use super::{
    encoding::{decode_fr, encode_fields, encode_fr},
    ffi,
    mem::Ascend910B3DevicePointer,
    utils::{byte_count, collect_output, evaluation_count, input_pointer},
    Ascend910B3Backend,
};
use crate::{BackendError, Mle, MleBackend};
use ark_bls12_381::Fr;
use std::{
    ffi::c_void,
    ptr::{self, NonNull},
    rc::Rc,
};

/// Shared immutable device MLE with host variable-count metadata.
///
/// Cloning shares storage and context ownership, not numerical data. The final
/// owner retires the buffer; native stream completion governs physical release.
/// This handle is neither Send nor Sync and may outlive its backend entry point.
#[derive(Clone)]
pub struct Ascend910B3Mle {
    pub(crate) allocation: Rc<Ascend910B3DevicePointer>,
    pub(crate) num_vars: usize,
}

impl Mle<Fr> for Ascend910B3Mle {
    type Id = usize;
    fn num_vars(&self) -> usize {
        self.num_vars
    }
    fn id(&self) -> Self::Id {
        Rc::as_ptr(&self.allocation) as usize
    }
}

impl Ascend910B3Backend {
    pub(crate) fn check_mle(&self, mle: &Ascend910B3Mle) -> Result<(), BackendError> {
        self.check_context(&mle.allocation.context)
    }

    /// Take the unique ownership transferred by a successful native output.
    ///
    /// # Safety
    /// `address` must be a fresh, uniquely transferred allocation in this context
    /// for the given dimension. It must not alias another independently owned
    /// output. Null after native success violates the ABI and aborts.
    /// The dimension must have been validated on creation or inherited from an
    /// existing MLE.
    pub(crate) unsafe fn own_mle(&self, address: *mut c_void, num_vars: usize) -> Ascend910B3Mle {
        let raw = NonNull::new(address).unwrap_or_else(|| std::process::abort());
        Ascend910B3Mle {
            allocation: Rc::new(Ascend910B3DevicePointer {
                raw,
                context: Rc::clone(&self.context),
            }),
            num_vars,
        }
    }

    fn mle_addresses(
        &self,
        polynomials: &[Ascend910B3Mle],
        num_vars: usize,
    ) -> Result<Vec<ffi::Ascend910B3DeviceAddress>, BackendError> {
        let mut addresses = Vec::with_capacity(polynomials.len());
        for polynomial in polynomials {
            self.check_mle(polynomial)?;
            if polynomial.num_vars != num_vars {
                return Err(BackendError::InvalidParameters(
                    "MLE dimensions must match".to_string(),
                ));
            }
            addresses.push(polynomial.allocation.raw.as_ptr());
        }
        Ok(addresses)
    }

    fn nonempty_dimension(&self, polynomials: &[Ascend910B3Mle]) -> Result<usize, BackendError> {
        let first = polynomials.first().ok_or_else(|| {
            BackendError::InvalidParameters("MLE input list must be nonempty".to_string())
        })?;
        Ok(first.num_vars)
    }
}

impl MleBackend<Fr> for Ascend910B3Backend {
    type Mle = Ascend910B3Mle;

    fn mle_from_evaluations(
        &self,
        num_vars: usize,
        evaluations: Vec<Fr>,
    ) -> Result<Self::Mle, BackendError> {
        if evaluations.len() != evaluation_count(num_vars)? {
            return Err(BackendError::InvalidParameters(
                "evaluation table length must equal 2^num_vars".to_string(),
            ));
        }
        let bytes = encode_fields(&evaluations);
        let mut out = ptr::null_mut();
        // SAFETY: packet and output slot remain live for the call; native borrows
        // host inputs only until return and transfers a unique nonnull allocation.
        unsafe {
            ffi::hp_ascend910b3_v1_mle_upload(
                self.context.raw.as_ptr(),
                num_vars,
                input_pointer(&bytes),
                evaluations.len(),
                &mut out,
            )
            .into_result("mle_upload")?;
            Ok(self.own_mle(out, num_vars))
        }
    }

    fn linear_combination(
        &self,
        polynomials: &[Self::Mle],
        coefficients: &[Fr],
    ) -> Result<Self::Mle, BackendError> {
        let num_vars = self.nonempty_dimension(polynomials)?;
        if polynomials.len() != coefficients.len() {
            return Err(BackendError::InvalidParameters(
                "coefficient count must match MLE count".to_string(),
            ));
        }
        let addresses = self.mle_addresses(polynomials, num_vars)?;
        let coefficients = encode_fields(coefficients);
        let mut out = ptr::null_mut();
        // SAFETY: all handles belong to this live context, arrays have the
        // declared count, and native returns a unique allocation without borrowing them.
        unsafe {
            ffi::hp_ascend910b3_v1_mle_linear_combination(
                self.context.raw.as_ptr(),
                num_vars,
                input_pointer(&addresses),
                input_pointer(&coefficients),
                addresses.len(),
                &mut out,
            )
            .into_result("mle_linear_combination")?;
            Ok(self.own_mle(out, num_vars))
        }
    }

    fn evaluate_mle(&self, mle: &Self::Mle, point: &[Fr]) -> Result<Fr, BackendError> {
        self.check_mle(mle)?;
        if point.len() != mle.num_vars {
            return Err(BackendError::InvalidParameters(
                "evaluation requires a complete point".to_string(),
            ));
        }
        let point = encode_fields(point);
        let bytes = collect_output("mle_evaluate", 32, |out| {
            // SAFETY: matching-context MLE and complete point remain live;
            // success initializes all 32 output bytes and makes them host-ready.
            unsafe {
                ffi::hp_ascend910b3_v1_mle_evaluate(
                    self.context.raw.as_ptr(),
                    mle.allocation.raw.as_ptr(),
                    mle.num_vars,
                    input_pointer(&point),
                    out,
                )
            }
        })?;
        decode_fr(&bytes)
    }

    fn build_eq_x_r(&self, r: &[Fr]) -> Result<Self::Mle, BackendError> {
        if r.is_empty() {
            return Err(BackendError::InvalidParameters(
                "eq polynomial requires a nonempty point".to_string(),
            ));
        }
        let num_vars = r.len();
        byte_count(evaluation_count(num_vars)?, 32)?;
        let r = encode_fields(r);
        let mut out = ptr::null_mut();
        // SAFETY: complete canonical input is borrowed only for this call;
        // success transfers one unique allocation of the declared dimension.
        unsafe {
            ffi::hp_ascend910b3_v1_mle_build_eq_x_r(
                self.context.raw.as_ptr(),
                input_pointer(&r),
                num_vars,
                &mut out,
            )
            .into_result("mle_build_eq_x_r")?;
            Ok(self.own_mle(out, num_vars))
        }
    }

    fn compute_frac_poly(
        &self,
        fxs: &[Self::Mle],
        gxs: &[Self::Mle],
    ) -> Result<Self::Mle, BackendError> {
        let num_vars = self.nonempty_dimension(fxs)?;
        if fxs.len() != gxs.len() {
            return Err(BackendError::InvalidParameters(
                "fraction input counts must match".to_string(),
            ));
        }
        let numerators = self.mle_addresses(fxs, num_vars)?;
        let denominators = self.mle_addresses(gxs, num_vars)?;
        let mut out = ptr::null_mut();
        // SAFETY: equal nonempty matching-context arrays stay live through the
        // call; native checks all zero denominators before reporting success.
        unsafe {
            ffi::hp_ascend910b3_v1_mle_compute_frac_poly(
                self.context.raw.as_ptr(),
                num_vars,
                input_pointer(&numerators),
                input_pointer(&denominators),
                numerators.len(),
                &mut out,
            )
            .into_result("mle_compute_frac_poly")?;
            Ok(self.own_mle(out, num_vars))
        }
    }

    fn compute_product_poly(&self, frac_poly: &Self::Mle) -> Result<Self::Mle, BackendError> {
        self.check_mle(frac_poly)?;
        if frac_poly.num_vars == 0 {
            return Err(BackendError::InvalidParameters(
                "product polynomial requires positive dimension".to_string(),
            ));
        }
        let mut out = ptr::null_mut();
        // SAFETY: immutable input belongs to this context; success transfers a
        // distinct allocation preserving its dimension.
        unsafe {
            ffi::hp_ascend910b3_v1_mle_compute_product_poly(
                self.context.raw.as_ptr(),
                frac_poly.allocation.raw.as_ptr(),
                frac_poly.num_vars,
                &mut out,
            )
            .into_result("mle_compute_product_poly")?;
            Ok(self.own_mle(out, frac_poly.num_vars))
        }
    }

    fn compute_nums_and_denoms(
        &self,
        beta: &Fr,
        gamma: &Fr,
        fxs: &[Self::Mle],
        gxs: &[Self::Mle],
        perms: &[Self::Mle],
    ) -> Result<(Vec<Self::Mle>, Vec<Self::Mle>), BackendError> {
        let num_vars = self.nonempty_dimension(fxs)?;
        if fxs.len() != gxs.len() || fxs.len() != perms.len() {
            return Err(BackendError::InvalidParameters(
                "numerator/denominator input counts must match".to_string(),
            ));
        }
        let fxs = self.mle_addresses(fxs, num_vars)?;
        let gxs = self.mle_addresses(gxs, num_vars)?;
        let perms = self.mle_addresses(perms, num_vars)?;
        let beta = encode_fr(beta);
        let gamma = encode_fr(gamma);
        let mut numerator_slots = vec![ptr::null_mut(); fxs.len()];
        let mut denominator_slots = vec![ptr::null_mut(); fxs.len()];
        // Allocate owners before native creates any resources. If wrapping is
        // interrupted, already established owners retain their normal RAII drop.
        let mut numerators = Vec::with_capacity(fxs.len());
        let mut denominators = Vec::with_capacity(fxs.len());
        // SAFETY: matching arrays and challenges are live, both output arrays
        // contain count slots. Failure cleans partial resources and clears slots;
        // success transfers count independent unique allocations per array.
        unsafe {
            ffi::hp_ascend910b3_v1_mle_compute_nums_and_denoms(
                self.context.raw.as_ptr(),
                num_vars,
                beta.as_ptr(),
                gamma.as_ptr(),
                input_pointer(&fxs),
                input_pointer(&gxs),
                input_pointer(&perms),
                fxs.len(),
                numerator_slots.as_mut_ptr(),
                denominator_slots.as_mut_ptr(),
            )
            .into_result("mle_compute_nums_and_denoms")?;
            for address in numerator_slots {
                numerators.push(self.own_mle(address, num_vars));
            }
            for address in denominator_slots {
                denominators.push(self.own_mle(address, num_vars));
            }
        }
        Ok((numerators, denominators))
    }

    fn compute_product_factors(
        &self,
        frac_poly: &Self::Mle,
        prod_poly: &Self::Mle,
    ) -> Result<[Self::Mle; 2], BackendError> {
        self.check_mle(frac_poly)?;
        self.check_mle(prod_poly)?;
        if frac_poly.num_vars == 0 || frac_poly.num_vars != prod_poly.num_vars {
            return Err(BackendError::InvalidParameters(
                "product factors require matching positive dimensions".to_string(),
            ));
        }
        let mut p1 = ptr::null_mut();
        let mut p2 = ptr::null_mut();
        // SAFETY: both inputs belong to this context and have matching positive
        // dimensions. Native cleans partial outputs on error and transfers two
        // independent allocations on success; each is immediately RAII-owned.
        unsafe {
            ffi::hp_ascend910b3_v1_mle_compute_product_factors(
                self.context.raw.as_ptr(),
                frac_poly.allocation.raw.as_ptr(),
                prod_poly.allocation.raw.as_ptr(),
                frac_poly.num_vars,
                &mut p1,
                &mut p2,
            )
            .into_result("mle_compute_product_factors")?;
            let p1 = self.own_mle(p1, frac_poly.num_vars);
            let p2 = self.own_mle(p2, frac_poly.num_vars);
            Ok([p1, p2])
        }
    }
}
