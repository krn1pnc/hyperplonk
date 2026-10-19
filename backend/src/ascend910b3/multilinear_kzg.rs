//! Raw multilinear KZG calls; protocol serialization and transcripts stay outside this module.

use super::{
    context::Ascend910B3Context,
    encoding::{decode_fr, decode_g1, encode_fields, encode_g1},
    ffi,
    utils::{byte_count, collect_output, evaluation_count, input_pointer},
    Ascend910B3Backend,
};
use crate::{BackendError, MultilinearKzgBackend};
use ark_bls12_381::{Bls12_381, Fr, G1Affine};
use std::{
    ptr::{self, NonNull},
    rc::Rc,
};

/// Opaque host-managed multilinear KZG parameters retaining their native context.
///
/// Native data layout is not exposed. This uniquely owned, single-threaded handle
/// may outlive the backend and is destroyed before its final context reference.
pub struct Ascend910B3MultilinearKzgPreparedProverParam {
    pub(crate) raw: NonNull<ffi::Ascend910B3NativeMultilinearKzgPreparedProverParam>,
    pub(crate) context: Rc<Ascend910B3Context>,
}

impl MultilinearKzgBackend<Bls12_381> for Ascend910B3Backend {
    type PreparedProverParam = Ascend910B3MultilinearKzgPreparedProverParam;

    fn prepare_prover_param(
        &self,
        num_vars: usize,
        mut powers_of_g: impl ExactSizeIterator<Item = Vec<G1Affine>>,
    ) -> Result<Self::PreparedProverParam, BackendError> {
        let first_layer_count = evaluation_count(num_vars)?;
        byte_count(first_layer_count, 97)?;
        let layer_count = num_vars + 1;
        if powers_of_g.len() != layer_count {
            return Err(BackendError::InvalidParameters(
                "SRS must contain num_vars + 1 layers".to_string(),
            ));
        }
        let mut raw = ptr::null_mut();
        // SAFETY: context and output slot stay live. Failure clears any partial
        // resource; success transfers a unique host management object.
        unsafe {
            ffi::hp_ascend910b3_v1_multilinear_kzg_params_create(
                self.context.raw.as_ptr(),
                num_vars,
                &mut raw,
            )
            .into_result("multilinear_kzg_params_create")?;
        }
        let raw = NonNull::new(raw).unwrap_or_else(|| std::process::abort());
        let params = Ascend910B3MultilinearKzgPreparedProverParam {
            raw,
            context: Rc::clone(&self.context),
        };
        // One scratch allocation is reused; each owned layer is dropped after
        // its synchronous host borrow ends, never collecting the entire SRS.
        let mut scratch = Vec::new();
        for layer_index in 0..layer_count {
            let layer = powers_of_g.next().ok_or_else(|| {
                BackendError::InvalidParameters(
                    "SRS iterator ended before its declared length".to_string(),
                )
            })?;
            let point_count = first_layer_count >> layer_index;
            if layer.len() != point_count {
                return Err(BackendError::InvalidParameters(format!(
                    "SRS layer {layer_index} has incorrect point count"
                )));
            }
            // Every layer is bounded by the already checked first-layer packet.
            let length = point_count * 97;
            scratch.clear();
            scratch.reserve(length);
            for point in &layer {
                scratch.extend_from_slice(&encode_g1(point));
            }
            // SAFETY: params is RAII-owned in this context, scratch contains
            // exactly point_count canonical packets and is borrowed only until
            // return. An upload failure drops params through its dedicated destroy.
            unsafe {
                ffi::hp_ascend910b3_v1_multilinear_kzg_params_upload_layer(
                    self.context.raw.as_ptr(),
                    params.raw.as_ptr(),
                    layer_index,
                    input_pointer(&scratch),
                    point_count,
                )
                .into_result("multilinear_kzg_params_upload_layer")?;
            }
        }
        if powers_of_g.next().is_some() {
            return Err(BackendError::InvalidParameters(
                "SRS iterator exceeded its declared length".to_string(),
            ));
        }
        Ok(params)
    }

    fn commit(
        &self,
        prover_param: &Self::PreparedProverParam,
        polynomial: &Self::Mle,
    ) -> Result<G1Affine, BackendError> {
        self.check_context(&prover_param.context)?;
        self.check_mle(polynomial)?;
        let bytes = collect_output("multilinear_kzg_commit", 97, |out| {
            // SAFETY: live prepared params and immutable MLE share this context;
            // success writes one complete host-ready G1 packet.
            unsafe {
                ffi::hp_ascend910b3_v1_multilinear_kzg_commit(
                    self.context.raw.as_ptr(),
                    prover_param.raw.as_ptr(),
                    polynomial.allocation.raw.as_ptr(),
                    polynomial.num_vars,
                    out,
                )
            }
        })?;
        decode_g1(&bytes)
    }

    fn multi_commit(
        &self,
        prover_param: &Self::PreparedProverParam,
        polynomials: &[Self::Mle],
    ) -> Result<Vec<G1Affine>, BackendError> {
        self.check_context(&prover_param.context)?;
        if polynomials.is_empty() {
            // This local empty result makes no native synchronization promise.
            return Ok(Vec::new());
        }
        let output_length = byte_count(polynomials.len(), 97)?;
        let mut addresses = Vec::with_capacity(polynomials.len());
        let mut dimensions = Vec::with_capacity(polynomials.len());
        for polynomial in polynomials {
            self.check_mle(polynomial)?;
            addresses.push(polynomial.allocation.raw.as_ptr());
            dimensions.push(polynomial.num_vars);
        }
        let bytes = collect_output("multilinear_kzg_multi_commit", output_length, |out| {
            // SAFETY: every input has this context; dimensions and addresses
            // retain order and duplicates and contain the declared count.
            // Native writes all count host-ready packets on success.
            unsafe {
                ffi::hp_ascend910b3_v1_multilinear_kzg_multi_commit(
                    self.context.raw.as_ptr(),
                    prover_param.raw.as_ptr(),
                    input_pointer(&addresses),
                    input_pointer(&dimensions),
                    addresses.len(),
                    out,
                )
            }
        })?;
        bytes.chunks_exact(97).map(decode_g1).collect()
    }

    fn open(
        &self,
        prover_param: &Self::PreparedProverParam,
        polynomial: &Self::Mle,
        point: &[Fr],
    ) -> Result<(Vec<G1Affine>, Fr), BackendError> {
        self.check_context(&prover_param.context)?;
        self.check_mle(polynomial)?;
        if point.len() != polynomial.num_vars {
            return Err(BackendError::InvalidParameters(
                "opening requires a complete point".to_string(),
            ));
        }
        let point = encode_fields(point);
        let quotient_length = byte_count(polynomial.num_vars, 97)?;
        let output_length = quotient_length.checked_add(32).ok_or_else(|| {
            BackendError::InvalidParameters("opening output length overflow".to_string())
        })?;
        let bytes = collect_output("multilinear_kzg_open", output_length, |out| {
            // SAFETY: collect_output reserves quotient_length + 32 bytes; these
            // disjoint regions hold all quotient packets and the scalar packet.
            // For constants only the quotient pointer and point are null.
            // Native success completely writes both regions and makes them ready.
            unsafe {
                let quotients = if quotient_length == 0 {
                    ptr::null_mut()
                } else {
                    out
                };
                ffi::hp_ascend910b3_v1_multilinear_kzg_open(
                    self.context.raw.as_ptr(),
                    prover_param.raw.as_ptr(),
                    polynomial.allocation.raw.as_ptr(),
                    polynomial.num_vars,
                    input_pointer(&point),
                    quotients,
                    out.add(quotient_length),
                )
            }
        })?;
        let quotients = bytes[..quotient_length]
            .chunks_exact(97)
            .map(decode_g1)
            .collect::<Result<Vec<_>, _>>()?;
        let evaluation = decode_fr(&bytes[quotient_length..])?;
        Ok((quotients, evaluation))
    }
}

impl Drop for Ascend910B3MultilinearKzgPreparedProverParam {
    fn drop(&mut self) {
        // SAFETY: the unique native owner is destroyed once, with its context
        // retained throughout this call. Native code owns asynchronous retirement.
        let status = unsafe {
            ffi::hp_ascend910b3_v1_multilinear_kzg_params_destroy(
                self.context.raw.as_ptr(),
                self.raw.as_ptr(),
            )
        };
        if status
            .into_result("multilinear_kzg_params_destroy")
            .is_err()
        {
            std::process::abort();
        }
    }
}
