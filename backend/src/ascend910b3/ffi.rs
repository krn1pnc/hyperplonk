//! Version-one host ABI; these packets are not device memory layouts.
//!
//! Fr is canonical LE32, G1 is tag + x LE48 + y LE48 (97 bytes). A tag of
//! zero denotes a finite subgroup point; one denotes infinity with zero coordinates.
//! Array lengths count elements, not bytes. Empty arrays use null pointers, except
//! product offsets, which always contain product_count + 1 entries starting at zero.
//! All host buffers are borrowed only until return, including upload inputs.
//!
//! Successful host numerical results must be completely written and ready. Device
//! outputs may be pending but must be distinct, immutable and uniquely transferred.
//! Failure must clean partial native resources and null every resource output slot.
//! No exception may cross this ABI. Opaque SumCheck state owns the lifetime of all
//! its numerical inputs after creation. Native retirement reclaims buffers only
//! after successful stream-completion observation; context destruction synchronizes
//! and drains retirement without resetting another live context.

use super::status::Ascend910B3Status;
use std::ffi::c_void;

#[repr(C)]
pub(crate) struct Ascend910B3NativeContext {
    _private: [u8; 0],
}
#[repr(C)]
pub(crate) struct Ascend910B3NativeMultilinearKzgPreparedProverParam {
    _private: [u8; 0],
}
#[repr(C)]
pub(crate) struct Ascend910B3NativeSumCheckProverState {
    _private: [u8; 0],
}
pub(crate) type Ascend910B3DeviceAddress = *mut c_void;

#[link(name = "hyperplonk_ascend910b3_bridge", kind = "static")]
extern "C" {
    pub(crate) fn hp_ascend910b3_v1_context_create(
        device_id: i32,
        out: *mut *mut Ascend910B3NativeContext,
    ) -> Ascend910B3Status;
    pub(crate) fn hp_ascend910b3_v1_context_destroy(
        ctx: *mut Ascend910B3NativeContext,
    ) -> Ascend910B3Status;
    pub(crate) fn hp_ascend910b3_v1_buffer_retire(
        ctx: *mut Ascend910B3NativeContext,
        address: Ascend910B3DeviceAddress,
    ) -> Ascend910B3Status;
    pub(crate) fn hp_ascend910b3_v1_mle_upload(
        ctx: *mut Ascend910B3NativeContext,
        num_vars: usize,
        evaluations: *const u8,
        evaluation_count: usize,
        out: *mut Ascend910B3DeviceAddress,
    ) -> Ascend910B3Status;
    pub(crate) fn hp_ascend910b3_v1_mle_linear_combination(
        ctx: *mut Ascend910B3NativeContext,
        num_vars: usize,
        polynomials: *const Ascend910B3DeviceAddress,
        coefficients: *const u8,
        count: usize,
        out: *mut Ascend910B3DeviceAddress,
    ) -> Ascend910B3Status;
    pub(crate) fn hp_ascend910b3_v1_mle_evaluate(
        ctx: *mut Ascend910B3NativeContext,
        polynomial: Ascend910B3DeviceAddress,
        num_vars: usize,
        point: *const u8,
        out_evaluation: *mut u8,
    ) -> Ascend910B3Status;
    pub(crate) fn hp_ascend910b3_v1_mle_build_eq_x_r(
        ctx: *mut Ascend910B3NativeContext,
        r: *const u8,
        num_vars: usize,
        out: *mut Ascend910B3DeviceAddress,
    ) -> Ascend910B3Status;
    pub(crate) fn hp_ascend910b3_v1_mle_compute_frac_poly(
        ctx: *mut Ascend910B3NativeContext,
        num_vars: usize,
        numerators: *const Ascend910B3DeviceAddress,
        denominators: *const Ascend910B3DeviceAddress,
        count: usize,
        out: *mut Ascend910B3DeviceAddress,
    ) -> Ascend910B3Status;
    pub(crate) fn hp_ascend910b3_v1_mle_compute_product_poly(
        ctx: *mut Ascend910B3NativeContext,
        frac: Ascend910B3DeviceAddress,
        num_vars: usize,
        out: *mut Ascend910B3DeviceAddress,
    ) -> Ascend910B3Status;
    pub(crate) fn hp_ascend910b3_v1_mle_compute_nums_and_denoms(
        ctx: *mut Ascend910B3NativeContext,
        num_vars: usize,
        beta: *const u8,
        gamma: *const u8,
        fxs: *const Ascend910B3DeviceAddress,
        gxs: *const Ascend910B3DeviceAddress,
        permutations: *const Ascend910B3DeviceAddress,
        count: usize,
        out_numerators: *mut Ascend910B3DeviceAddress,
        out_denominators: *mut Ascend910B3DeviceAddress,
    ) -> Ascend910B3Status;
    pub(crate) fn hp_ascend910b3_v1_mle_compute_product_factors(
        ctx: *mut Ascend910B3NativeContext,
        frac: Ascend910B3DeviceAddress,
        product: Ascend910B3DeviceAddress,
        num_vars: usize,
        out_p1: *mut Ascend910B3DeviceAddress,
        out_p2: *mut Ascend910B3DeviceAddress,
    ) -> Ascend910B3Status;
    pub(crate) fn hp_ascend910b3_v1_multilinear_kzg_params_create(
        ctx: *mut Ascend910B3NativeContext,
        num_vars: usize,
        out: *mut *mut Ascend910B3NativeMultilinearKzgPreparedProverParam,
    ) -> Ascend910B3Status;
    pub(crate) fn hp_ascend910b3_v1_multilinear_kzg_params_upload_layer(
        ctx: *mut Ascend910B3NativeContext,
        params: *mut Ascend910B3NativeMultilinearKzgPreparedProverParam,
        layer_index: usize,
        points: *const u8,
        point_count: usize,
    ) -> Ascend910B3Status;
    pub(crate) fn hp_ascend910b3_v1_multilinear_kzg_params_destroy(
        ctx: *mut Ascend910B3NativeContext,
        params: *mut Ascend910B3NativeMultilinearKzgPreparedProverParam,
    ) -> Ascend910B3Status;
    pub(crate) fn hp_ascend910b3_v1_multilinear_kzg_commit(
        ctx: *mut Ascend910B3NativeContext,
        params: *const Ascend910B3NativeMultilinearKzgPreparedProverParam,
        polynomial: Ascend910B3DeviceAddress,
        num_vars: usize,
        out_commitment: *mut u8,
    ) -> Ascend910B3Status;
    pub(crate) fn hp_ascend910b3_v1_multilinear_kzg_multi_commit(
        ctx: *mut Ascend910B3NativeContext,
        params: *const Ascend910B3NativeMultilinearKzgPreparedProverParam,
        polynomials: *const Ascend910B3DeviceAddress,
        num_vars: *const usize,
        count: usize,
        out_commitments: *mut u8,
    ) -> Ascend910B3Status;
    pub(crate) fn hp_ascend910b3_v1_multilinear_kzg_open(
        ctx: *mut Ascend910B3NativeContext,
        params: *const Ascend910B3NativeMultilinearKzgPreparedProverParam,
        polynomial: Ascend910B3DeviceAddress,
        num_vars: usize,
        point: *const u8,
        out_quotients: *mut u8,
        out_evaluation: *mut u8,
    ) -> Ascend910B3Status;
    pub(crate) fn hp_ascend910b3_v1_sumcheck_prover_state_create(
        ctx: *mut Ascend910B3NativeContext,
        num_vars: usize,
        max_degree: usize,
        factors: *const Ascend910B3DeviceAddress,
        factor_count: usize,
        coefficients: *const u8,
        product_offsets: *const usize,
        product_count: usize,
        factor_indices: *const usize,
        index_count: usize,
        out: *mut *mut Ascend910B3NativeSumCheckProverState,
    ) -> Ascend910B3Status;
    pub(crate) fn hp_ascend910b3_v1_sumcheck_prover_state_round(
        ctx: *mut Ascend910B3NativeContext,
        state: *mut Ascend910B3NativeSumCheckProverState,
        challenge_or_null: *const u8,
        out_evaluations: *mut u8,
        evaluation_count: usize,
    ) -> Ascend910B3Status;
    pub(crate) fn hp_ascend910b3_v1_sumcheck_prover_state_destroy(
        ctx: *mut Ascend910B3NativeContext,
        state: *mut Ascend910B3NativeSumCheckProverState,
    ) -> Ascend910B3Status;
}
