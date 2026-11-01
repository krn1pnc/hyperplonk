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
// The prover keeps its working tables in the *private Montgomery domain*: a
// table value is x*R mod p stored as canonical radix digits. The only interface
// boundaries are canonical Fr, so the caller never supplies a Montgomery
// coordinate:
//   * `coordinate` is one canonical Fr; the fold kernel converts it with R^2
//     (Mont(r) = r*R) before use and writes folded Montgomery values.
//   * product coefficients and the 0..degree grid weights are converted
//     once at create by `launch_sumcheck_convert` and stay Montgomery resident.
//   * `launch_sumcheck_reduce` converts the final d+1 values back to canonical
//     Fr before the host download, so host bytes are ordinary canonical Fr.
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

// Factor-table arena geometry for one distinct factor (num_vars >= 1, which the
// SumCheck contract requires). The state registers one
// arena buffer holding `factor_count * sumcheck_table_stride(num_vars)` bytes;
// factor f owns the byte range [f * stride, (f + 1) * stride). Within it:
//   parity 0 table at +0           capacity 2^num_vars entries
//   parity 1 table at +2^n*32      capacity 2^(num_vars-1) entries
// Round r (0 <= r < num_vars) reads parity r % 2 and, for r >= 1, folds from
// parity (r - 1) % 2 into parity r % 2 with 2^(num_vars - r + 1) inputs and
// 2^(num_vars - r) outputs.
constexpr uint64_t sumcheck_parity_bytes(size_t num_vars) noexcept {
    return (uint64_t(1) << num_vars) * sumcheck_fr_bytes;
}
constexpr uint64_t sumcheck_table_stride(size_t num_vars) noexcept {
    return (uint64_t(3) << (num_vars - 1)) * sumcheck_fr_bytes;
}
constexpr uint64_t sumcheck_parity_offset(size_t num_vars, size_t round) noexcept {
    return (round & 1) ? sumcheck_parity_bytes(num_vars) : 0;
}
constexpr uint64_t sumcheck_fold_src_offset(size_t num_vars, size_t round) noexcept {
    return sumcheck_parity_offset(num_vars, round - 1);
}
constexpr uint64_t sumcheck_fold_dst_offset(size_t num_vars, size_t round) noexcept {
    return sumcheck_parity_offset(num_vars, round);
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

// Canonical -> private Montgomery conversion of `count` contiguous entries
// (natural low-bit-first order). `input` and `output` are disjoint; the output
// holds canonical-length digits of x*R mod p. count 0 enqueues nothing.
// Create-time uses: each factor table (count = 2^n) into arena parity 0, the
// product coefficient array (count = product_count) and the grid weights
// 0..degree (count = degree + 1).
bool launch_sumcheck_convert(aclrtStream stream, const Fr* input, Fr* output,
                             uint64_t count, uint32_t lanes);

// One low-bit-first fold level in the private Montgomery domain:
//   out[b] = in[2b] + Mont(r) * (in[2b+1] - in[2b])   (Montgomery homomorphic)
// which is the canonical fold of the represented values (`coordinate` is one
// device-resident canonical Fr). `input` has 2^exponent entries, `output`
// 2^(exponent-1); exponent >= 1 and output must not alias input (the state
// keeps disjoint parity regions).
bool launch_sumcheck_fold(aclrtStream stream, const Fr* input, const Fr* coordinate,
                          Fr* output, uint32_t exponent, uint32_t lanes);

// One product's round message over the current subject tables.
// `arena` is the factor-table arena; the product's multiplicands are described
// by `offsets`: `arity` records of one 64-bit arena offset each, stored 32 bytes
// apart (records may repeat the same factor). Element b of a subject table is
// addressed from that factor's parity base, which the kernel derives as
//   arena + offset + parity_offset
// where `parity_offset` is sumcheck_parity_offset(num_vars, round). For a rest
// entry b and grid point t:
//   partial[t][block] = sum over the block's rest entries of
//       prod_j ( even_j(b) + t * (odd_j(b) - even_j(b)) )
// evaluated in the Montgomery domain, where even/odd are table elements 2b and
// 2b+1 of the round's subject table (low bit first). `rest_count` is the number
// of rest entries and `t_count` the number of grid points in this pass.
// `chunk_weight` is Mont(t_begin) (one Montgomery Fr, replicated inside); it is
// unused and may be null when t_begin == 0.
// Results are written as Montgomery Fr for t in [t_begin, t_begin + t_count):
//   partial[((product_index * (degree + 1)) + t) * blocks + GetBlockIdx()]
// `blocks` must equal the launch grid (sumcheck_message_blocks) and be >= 1.
bool launch_sumcheck_message(aclrtStream stream, const void* arena,
                             const uint64_t* offsets, uint32_t arity,
                             uint64_t parity_offset, uint64_t rest_count,
                             uint32_t t_begin, uint32_t t_count, Fr* partial,
                             uint32_t product_index, uint32_t degree,
                             const Fr* chunk_weight, uint32_t blocks,
                             uint32_t lanes);

// Reduce one round's block partials to the d+1 canonical evaluations:
//   out[t] = sum_p Mont(c_p) * sum_b partial[p][t][b]
// `coefficients` holds product_count Montgomery coefficients (canonical digits
// of c*R). The grid is parallel over t; every grid block writes its own t and
// product_count == 0 writes canonical zeros without reading `partial`.
bool launch_sumcheck_reduce(aclrtStream stream, const Fr* partial,
                            const Fr* coefficients, Fr* output,
                            uint32_t product_count, uint32_t degree,
                            uint32_t blocks, uint32_t lanes);

} // namespace hyperplonk::ascend910b3::detail
