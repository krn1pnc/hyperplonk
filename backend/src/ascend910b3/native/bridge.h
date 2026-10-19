#pragma once
#include <stdint.h>
#include <stddef.h>

/* Host exchange ABI, not device layout. Fr is canonical LE32; G1 is a tag
 * (0 finite, 1 infinity), x LE48 and y LE48; infinity coordinates are zero.
 * All buffers are borrowed only until return. Zero-count arrays are null,
 * except product_offsets always contains product_count+1 entries starting at 0.
 * Host outputs must be fully written and ready on success. Byte widths are:
 * evaluations 32, commitments 97, point/r 32*num_vars, SRS 97*point_count,
 * open quotients 97*num_vars, rounds 32*evaluation_count. All counts and byte
 * lengths must fit size_t and PTRDIFF_MAX. No exception may cross this ABI.
 * Resource outputs are unique and nonnull on success; failure cleans partial
 * resources and clears output slots. Inputs are immutable. Contexts are single
 * host-thread/single-stream. State owns all data needed after create returns.
 * Retire transfers release responsibility to a native retired queue; buffers may
 * be physically freed only after aclrtStreamQuery succeeds and reports COMPLETE.
 * Opaque resources must retain any data they still need after buffer retirement.
 * Host-result return synchronizes/reclaims; context_destroy finally drains work.
 * No Rust pending flag, manual collection or MemPoolAsync API is required.
 * Context/parameter/state objects are host metadata, not device allocations.
 * Independent contexts must not reset one another; native owns shared runtime
 * lifetime. Zero denominators, including 0/0, are InvalidParameters.
 * The first round requires a null challenge, later rounds a nonnull challenge;
 * invalid ordering or exhaustion returns InvalidState without advancing state.
 * An execution failure may invalidate state; subsequent rounds return InvalidState.
 */
#define ASCEND910B3_STATUS_SUCCESS 0
#define ASCEND910B3_STATUS_INVALID_PARAMETERS 1
#define ASCEND910B3_STATUS_INVALID_STATE 2
#define ASCEND910B3_STATUS_NOT_IMPLEMENTED 3

typedef struct Ascend910B3Status { int32_t kind; int32_t native_code; } Ascend910B3Status;
typedef struct Ascend910B3NativeContext Ascend910B3NativeContext;
typedef struct Ascend910B3NativeMultilinearKzgPreparedProverParam Ascend910B3NativeMultilinearKzgPreparedProverParam;
typedef struct Ascend910B3NativeSumCheckProverState Ascend910B3NativeSumCheckProverState;
typedef void* Ascend910B3DeviceAddress;

#ifdef __cplusplus
extern "C" {
#endif
Ascend910B3Status hp_ascend910b3_v1_context_create(int32_t device_id, Ascend910B3NativeContext** out);
Ascend910B3Status hp_ascend910b3_v1_context_destroy(Ascend910B3NativeContext* ctx);
Ascend910B3Status hp_ascend910b3_v1_buffer_retire(Ascend910B3NativeContext* ctx, Ascend910B3DeviceAddress address);
Ascend910B3Status hp_ascend910b3_v1_mle_upload(Ascend910B3NativeContext* ctx, size_t num_vars, const uint8_t* evaluations, size_t evaluation_count, Ascend910B3DeviceAddress* out);
Ascend910B3Status hp_ascend910b3_v1_mle_linear_combination(Ascend910B3NativeContext* ctx, size_t num_vars, const Ascend910B3DeviceAddress* polynomials, const uint8_t* coefficients, size_t count, Ascend910B3DeviceAddress* out);
Ascend910B3Status hp_ascend910b3_v1_mle_evaluate(Ascend910B3NativeContext* ctx, Ascend910B3DeviceAddress polynomial, size_t num_vars, const uint8_t* point, uint8_t* out_evaluation);
Ascend910B3Status hp_ascend910b3_v1_mle_build_eq_x_r(Ascend910B3NativeContext* ctx, const uint8_t* r, size_t num_vars, Ascend910B3DeviceAddress* out);
Ascend910B3Status hp_ascend910b3_v1_mle_compute_frac_poly(Ascend910B3NativeContext* ctx, size_t num_vars, const Ascend910B3DeviceAddress* numerators, const Ascend910B3DeviceAddress* denominators, size_t count, Ascend910B3DeviceAddress* out);
Ascend910B3Status hp_ascend910b3_v1_mle_compute_product_poly(Ascend910B3NativeContext* ctx, Ascend910B3DeviceAddress frac, size_t num_vars, Ascend910B3DeviceAddress* out);
Ascend910B3Status hp_ascend910b3_v1_mle_compute_nums_and_denoms(Ascend910B3NativeContext* ctx, size_t num_vars, const uint8_t* beta, const uint8_t* gamma, const Ascend910B3DeviceAddress* fxs, const Ascend910B3DeviceAddress* gxs, const Ascend910B3DeviceAddress* permutations, size_t count, Ascend910B3DeviceAddress* out_numerators, Ascend910B3DeviceAddress* out_denominators);
Ascend910B3Status hp_ascend910b3_v1_mle_compute_product_factors(Ascend910B3NativeContext* ctx, Ascend910B3DeviceAddress frac, Ascend910B3DeviceAddress product, size_t num_vars, Ascend910B3DeviceAddress* out_p1, Ascend910B3DeviceAddress* out_p2);
Ascend910B3Status hp_ascend910b3_v1_multilinear_kzg_params_create(Ascend910B3NativeContext* ctx, size_t num_vars, Ascend910B3NativeMultilinearKzgPreparedProverParam** out);
Ascend910B3Status hp_ascend910b3_v1_multilinear_kzg_params_upload_layer(Ascend910B3NativeContext* ctx, Ascend910B3NativeMultilinearKzgPreparedProverParam* params, size_t layer_index, const uint8_t* points, size_t point_count);
Ascend910B3Status hp_ascend910b3_v1_multilinear_kzg_params_destroy(Ascend910B3NativeContext* ctx, Ascend910B3NativeMultilinearKzgPreparedProverParam* params);
Ascend910B3Status hp_ascend910b3_v1_multilinear_kzg_commit(Ascend910B3NativeContext* ctx, const Ascend910B3NativeMultilinearKzgPreparedProverParam* params, Ascend910B3DeviceAddress polynomial, size_t num_vars, uint8_t* out_commitment);
Ascend910B3Status hp_ascend910b3_v1_multilinear_kzg_multi_commit(Ascend910B3NativeContext* ctx, const Ascend910B3NativeMultilinearKzgPreparedProverParam* params, const Ascend910B3DeviceAddress* polynomials, const size_t* num_vars, size_t count, uint8_t* out_commitments);
Ascend910B3Status hp_ascend910b3_v1_multilinear_kzg_open(Ascend910B3NativeContext* ctx, const Ascend910B3NativeMultilinearKzgPreparedProverParam* params, Ascend910B3DeviceAddress polynomial, size_t num_vars, const uint8_t* point, uint8_t* out_quotients, uint8_t* out_evaluation);
Ascend910B3Status hp_ascend910b3_v1_sumcheck_prover_state_create(Ascend910B3NativeContext* ctx, size_t num_vars, size_t max_degree, const Ascend910B3DeviceAddress* factors, size_t factor_count, const uint8_t* coefficients, const size_t* product_offsets, size_t product_count, const size_t* factor_indices, size_t index_count, Ascend910B3NativeSumCheckProverState** out);
Ascend910B3Status hp_ascend910b3_v1_sumcheck_prover_state_round(Ascend910B3NativeContext* ctx, Ascend910B3NativeSumCheckProverState* state, const uint8_t* challenge_or_null, uint8_t* out_evaluations, size_t evaluation_count);
Ascend910B3Status hp_ascend910b3_v1_sumcheck_prover_state_destroy(Ascend910B3NativeContext* ctx, Ascend910B3NativeSumCheckProverState* state);
#ifdef __cplusplus
}
#endif
