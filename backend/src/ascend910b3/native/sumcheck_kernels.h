#pragma once

#include "field.h"

#include <acl/acl.h>

#include <cstddef>
#include <cstdint>

// Host-side launch interface for the resident SumCheck prover kernels.
//
// Every function only enqueues work on `stream`: no host copy, no allocation,
// no synchronization and no resource ownership. All pointers are borrowed
// device memory that must stay alive until the stream completes. MLE tables are
// canonical BLS12-381 Fr (little-endian 32 bytes, value < r) laid out
// low-bit-first (bit i is variable i), matching
// ark_poly::DenseMultilinearExtension and the shared MLE fold convention.
//
// Working tables remain canonical numeric Fr. Grid weights are Montgomery;
// each private coefficient is c*R^(arity+1), scaled once on device at create.
// Generic message products have scale R^(1-arity); the unchanged reduce first
// combines the scaled coefficient and then projects with literal canonical 1.
// Coordinates are canonical; fold converts the scalar to rR once internally.
//
// Three lane variants are compiled. `lanes` selects the variant and `chunk`
// (see sumcheck_chunk_*) is the number of 0..degree evaluations one pass can
// keep resident. Wider lanes give more lanes per instruction but hold fewer
// grid rows in UB; beyond `chunk` the host splits t into several passes.

namespace hyperplonk::ascend910b3::detail {

// Wide / medium / narrow Vector lane tiles and their resident t-grid capacity.
inline constexpr uint32_t sumcheck_lane_wide = 48;
inline constexpr uint32_t sumcheck_lane_medium = 32;
inline constexpr uint32_t sumcheck_lane_narrow = 16;
inline constexpr uint32_t sumcheck_chunk_wide = 4;
inline constexpr uint32_t sumcheck_chunk_medium = 15;
inline constexpr uint32_t sumcheck_chunk_narrow = 48;
// One round-message evaluation / one partial record is one canonical-Mont Fr.
inline constexpr uint64_t sumcheck_fr_bytes = 32;
// Convert/fold launches choose their own contiguous grid: one block per 32-byte
// lane tile, never more blocks than the platform's real AIV count and never
// more than this DAV2201 cap. The grid is a parallelism hint only and never
// changes results; message/reduce grids are chosen by the host.
inline constexpr uint32_t sumcheck_block_cap = 40;

// Lane variant and resident grid capacity chosen for one maximum degree.
// max_degree >= 1 (degree 0 is rejected by the SumCheck contract). Wider lanes
// are preferred; a degree whose grid does not fit falls back to narrower lanes
// and then to several t passes.
struct SumcheckShape {
    uint32_t lanes = 0;
    uint32_t chunk = 0;   // resident t values per pass
    uint32_t chunks = 0;  // ceil((max_degree + 1) / chunk)
};
constexpr SumcheckShape sumcheck_shape(size_t max_degree) noexcept {
    SumcheckShape shape{};
    if (max_degree == 0) return shape;
    const uint64_t grid = uint64_t(max_degree) + 1;
    if (grid <= sumcheck_chunk_wide) {
        shape.lanes = sumcheck_lane_wide;
        shape.chunk = sumcheck_chunk_wide;
    } else if (grid <= sumcheck_chunk_medium) {
        shape.lanes = sumcheck_lane_medium;
        shape.chunk = sumcheck_chunk_medium;
    } else {
        shape.lanes = sumcheck_lane_narrow;
        shape.chunk = sumcheck_chunk_narrow;
    }
    shape.chunks = uint32_t((grid + shape.chunk - 1) / shape.chunk);
    return shape;
}

// Retain the original canonical table for round zero. Each distinct factor
// owns only shrinking A/B slots: n=1 has neither, n=2 has A only.
constexpr uint64_t sumcheck_slot_a_bytes(size_t num_vars) noexcept {
    return num_vars <= 1 ? 0 : (uint64_t(1) << (num_vars - 1)) * sumcheck_fr_bytes;
}
constexpr uint64_t sumcheck_slot_b_bytes(size_t num_vars) noexcept {
    return num_vars <= 2 ? 0 : (uint64_t(1) << (num_vars - 2)) * sumcheck_fr_bytes;
}
constexpr uint64_t sumcheck_slot_b_offset(size_t num_vars) noexcept {
    return sumcheck_slot_a_bytes(num_vars);
}
constexpr uint64_t sumcheck_table_stride(size_t num_vars) noexcept {
    return sumcheck_slot_a_bytes(num_vars) + sumcheck_slot_b_bytes(num_vars);
}
constexpr uint64_t sumcheck_round_slot_offset(size_t num_vars, size_t round) noexcept {
    return (round & 1) ? 0 : sumcheck_slot_b_offset(num_vars);
}
constexpr uint64_t sumcheck_previous_slot_offset(size_t num_vars, size_t round) noexcept {
    return sumcheck_round_slot_offset(num_vars, round - 1);
}
// Low-bit-fold input exponent at round r >= 1: 2^(num_vars - r + 1) entries.
constexpr uint32_t sumcheck_fold_exponent(size_t num_vars, size_t round) noexcept {
    return uint32_t(num_vars - round + 1);
}
// Rest entries (table pairs) of round r's subject tables: 2^(num_vars - r - 1).
constexpr uint64_t sumcheck_rest_count(size_t num_vars, size_t round) noexcept {
    return uint64_t(1) << (num_vars - round - 1);
}
// Grid used by launch_sumcheck_message: one block per lane tile, capped by the
// platform AIV count. `blocks` is also the partial-record stride of the round.
constexpr uint32_t sumcheck_message_blocks(uint64_t rest_count, uint32_t lanes,
                                            uint32_t aiv_cores) noexcept {
    if (lanes == 0 || aiv_cores == 0) return 0;
    const uint64_t tiles = (rest_count + lanes - 1) / lanes;
    if (tiles == 0) return 0;
    return uint32_t(tiles < aiv_cores ? tiles : aiv_cores);
}
// Partial records for one product at product-index p, grid point t, block b:
//   partial[((p * (degree + 1)) + t) * blocks + b]
// Resident size is product_count * (degree + 1) * max_blocks records, where
// max_blocks is the largest grid the state will use (round zero rest count).
constexpr uint64_t sumcheck_partial_bytes(size_t product_count, size_t degree,
                                          uint32_t max_blocks) noexcept {
    return uint64_t(product_count) * (uint64_t(degree) + 1) * max_blocks *
           sumcheck_fr_bytes;
}

// Canonical -> Montgomery conversion, used only for the grid weights.
bool launch_sumcheck_convert(aclrtStream stream, const Fr* input, Fr* output,
                             uint64_t count, uint32_t lanes);

// Scale each c by R^(arity+1), using runtime arities, once at create.
bool launch_sumcheck_scale_coefficients(aclrtStream stream, const Fr* canonical,
                                        const uint32_t* arities, Fr* scaled,
                                        uint32_t product_count, uint32_t lanes);

// Canonical low-bit-first fold: even + mont(odd-even,rR).
// Input contains 2^exponent records; output contains 2^(exponent-1).
bool launch_sumcheck_fold(aclrtStream stream, const Fr* input, const Fr* coordinate,
                          Fr* output, uint32_t exponent, uint32_t lanes);

// Generic natural-layout message. Each immutable binding record is 32 bytes:
// a little-endian absolute uint64 GM address followed by 24 zero bytes.
// Repeated records preserve multiplicity. No common arena base is implied.
// Partial layout: [product][grid point][block], with scale R^(1-arity).
bool launch_sumcheck_message(aclrtStream stream, const uint64_t* absolute_offsets,
                             uint32_t arity, uint64_t rest_count,
                             uint32_t t_begin, uint32_t t_count, Fr* partial,
                             uint32_t product_index, uint32_t degree,
                             const Fr* chunk_weight, uint32_t blocks,
                             uint32_t lanes);

// Reduce the shared partial layout using c*R^(arity+1); final literal-1
// Montgomery projection is unchanged. Empty sums write canonical zeros.
bool launch_sumcheck_reduce(aclrtStream stream, const Fr* partial,
                            const Fr* coefficients, Fr* output,
                            uint32_t product_count, uint32_t degree,
                            uint32_t blocks, uint32_t lanes);

} // namespace hyperplonk::ascend910b3::detail
