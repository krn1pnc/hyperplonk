#pragma once

#include "field.h"

#include <acl/acl.h>

#include <cstddef>
#include <cstdint>

// Host-side launch interface for the resident MLE operation kernels.
//
// Every function here only enqueues work on `stream`: no host copy, no
// allocation, no synchronization and no resource ownership. All Fr pointers are
// borrowed device memory that must stay alive until the stream completes, and
// every table is canonical BLS12-381 Fr (little-endian 32 bytes, value < r).
// Table index bit i is variable i (low-bit-first), matching
// ark_poly::DenseMultilinearExtension.
//
// The ABI entries in bridge_mle.cpp validate context/dimensions/ownership and
// then call these launchers. KZG reuses the fold launcher for open; resident
// protocol state that keeps its own packed Montgomery tables reuses the same
// engine primitives through its own FieldVector friend instead of new math.

namespace hyperplonk::ascend910b3::detail {

// Cached AIV core count of the target platform (0 when it cannot be resolved).
// The launchers below use it for their contiguous block partition, so callers
// never pass a grid shape. It is pure host control metadata: no ACL state.
uint32_t mle_aiv_cores() noexcept;

// Single-variable low-bit-first canonical fold of a power-of-two table.
//   count = 2^num_vars, num_vars >= 1
//   folded[b]   = input[2b] + coordinate * (input[2b+1] - input[2b])
//   quotient[b] = input[2b+1] - input[2b]          (only when quotient != null)
// `folded` and `quotient` hold 2^(num_vars-1) entries. They must be pairwise
// disjoint and must not alias `input`, which is never written. `coordinate` is
// one device-resident canonical Fr for this level.
// Returns false without enqueueing when the pointer/dimension combination is
// degenerate or the platform reports no AIV core. Callers can chain levels by
// passing this level's `folded` as the next `input`; quotient stays valid for
// later enqueues on the same stream.
bool launch_mle_fold(aclrtStream stream, const Fr* input, size_t num_vars,
                     const Fr* coordinate, Fr* folded, Fr* quotient);

// Fold every variable of one table: result[0] = f(point), point[0] being the
// lowest table bit. `point_device` holds num_vars canonical Fr. `result` is one
// canonical Fr on the device. scratch_a needs 2^(num_vars-1) elements
// (required when num_vars >= 2) and scratch_b 2^(num_vars-2), which is
// required only when num_vars >= 3 and is unused/optional for num_vars < 3
// (num_vars == 2 also unused); both are scratch only and are disjoint from
// polynomial/result. num_vars >= 1.
bool launch_mle_evaluate(aclrtStream stream, size_t num_vars,
                         const Fr* polynomial, const Fr* point_device,
                         Fr* result, Fr* scratch_a, Fr* scratch_b);

// out[j] = sum_i coefficients[i] * polynomials[i][j] over 2^num_vars entries.
// `addresses` is a device array of count uint64 device addresses; `coefficients`
// is a device array of count canonical Fr. count >= 1.
bool launch_mle_linear_combination(aclrtStream stream, size_t num_vars,
                                   const uint64_t* addresses,
                                   const Fr* coefficients, size_t count,
                                   Fr* out);

// out[idx] = prod_i (idx_bit_i ? r[i] : 1 - r[i]) for idx < 2^num_vars.
// r_device holds num_vars canonical Fr; num_vars >= 1.
bool launch_mle_build_eq_x_r(aclrtStream stream, const Fr* r_device,
                             size_t num_vars, Fr* out);

// Number of per-block flag records launch_mle_frac writes for a 2^num_vars
// table (0 when the dimension is unsupported). Each record is 48 int32 = 192
// bytes, so the caller sizes its flag buffer as blocks * 192 before enqueueing.
uint32_t mle_frac_flag_blocks(size_t num_vars);

// out[j] = prod_i numerators[i][j] / prod_i denominators[i][j] over
// 2^num_vars entries. numerator/denominator addresses are device arrays of
// count uint64 addresses (count >= 1), inputs are immutable.
//
// Zero denominators (including 0/0) are reported through `flags` instead of a
// host copy of any table. Blocks are numbered 0..B-1 and each block owns 48
// int32 counters (192 bytes) covering every tile in its contiguous range: a
// counter is incremented once per tile whose lane denominator product is zero,
// so a zero in an early tile is never hidden by a later nonzero tile. Padding
// lanes beyond the last real element are treated as nonzero (canonical one), so
// a partial tail tile can never report a false zero. Bytes are little-endian
// int32; the caller rejects the call when any of flags[0, B*192) is nonzero.
// Returns B, or 0 when nothing was enqueued.
uint32_t launch_mle_frac(aclrtStream stream, size_t num_vars,
                         const uint64_t* numerator_addresses,
                         const uint64_t* denominator_addresses, size_t count,
                         Fr* out, uint8_t* flags);

// Product tree of `frac` in the recurrence layout used by
// compute_product_poly: out[0..2^(num_vars-1)) holds adjacent pair products,
// each later stage multiplies the previous stage's adjacent results, and the
// final entry is zero. num_vars >= 1; out has 2^num_vars entries and does not
// alias frac.
bool launch_mle_compute_product(aclrtStream stream, size_t num_vars,
                                const Fr* frac, Fr* out);

// numerator[c][j]   = fxs[c][j]   + beta * (c * 2^num_vars + j) + gamma
// denominator[c][j] = gxs[c][j]   + beta * perms[c][j]        + gamma
// beta/gamma are device canonical Fr; the five uint64 arrays are device arrays
// of count addresses (inputs, then the two output tables). count >= 1.
bool launch_mle_nums_and_denoms(aclrtStream stream, size_t num_vars,
                                const Fr* beta, const Fr* gamma,
                                const uint64_t* fx_addresses,
                                const uint64_t* gx_addresses,
                                const uint64_t* perm_addresses, size_t count,
                                const uint64_t* numerator_out,
                                const uint64_t* denominator_out);

// p1 = [frac even lanes | product even lanes]
// p2 = [frac odd lanes  | product odd lanes]
// Both outputs hold 2^num_vars entries (the low table bit selects the lane
// parity); num_vars >= 1; p1/p2 are disjoint from the immutable inputs and each
// other.
bool launch_mle_product_factors(aclrtStream stream, size_t num_vars,
                                const Fr* frac, const Fr* product, Fr* p1,
                                Fr* p2);

} // namespace hyperplonk::ascend910b3::detail
