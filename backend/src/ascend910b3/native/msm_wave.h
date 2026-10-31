#pragma once

// Target-private whole-wave affine reduction contract. This header is internal
// to the library (never included by msm.h / the public Rust FFI): it declares
// the frozen wave geometry, the SINGLE canonical arena allocator and the
// device control/diagnostic blocks. The six launch entry points implemented by
// msm_wave.asc are declared next to the other internal launchers in
// msm_internal.h, so this header stays free of acl/acl.h and can be consumed by
// the host-only plan tests that link msm_plan.cpp without ACL headers.
//
// A "wave" fuses `fuse_windows` consecutive scalar windows into one metadata
// capture, one 26-level affine bucket tree and one per-window weighted reduce.
// The arena holds only what the tree cannot recompute cheaply: captured
// {digit, point_id} records, captured RowPtr/base_offsets, the op stream, the
// step-major chain prefixes, the two node ping-pong buffers, the per-window
// bucket tuples, the per-block plan-stage summaries and the device
// control/diagnostic blocks.
//
// Arena order is exactly the WaveBounds member order. Every region starts at
// the running offset of the previous region and every size is already
// ceil-aligned to msm_wave_region_alignment, so the offset of region k is the
// plain prefix sum of the aligned sizes (wave_regions(), one inline
// definition) and the arena total is the 512-multiple sum of those sizes.
// wave_scratch_bounds() is DEFINED ONCE, in msm_plan.cpp.

#include <cstddef>
#include <cstdint>

#include "msm_plan.h"

namespace hyperplonk::ascend910b3::detail {

// Level 0..25; a bucket holds at most plan.count points, so depth <= 25 for the
// declared shapes (ceil(log2(count))). Arrays are sized one slot larger.
inline constexpr uint32_t msm_wave_level_slots = 26;
inline constexpr uint32_t msm_wave_core_slots = 40;
inline constexpr uint32_t msm_wave_window_slots = 32;  // 8-bit windows: 32 slots
inline constexpr uint32_t msm_wave_lanes = 64;
// Frozen chain geometry of the production controller: L_MAX 2048 ops per block
// lane and chain granularity 1, i.e. every op starts its own one-op chain. A
// level splits its ops into at most aiv_cores*lanes one-op chains (40*64 = 2560
// on the target), so two ops are never grouped into one chain and the frozen
// geometry is the same for every shape.
inline constexpr uint32_t msm_wave_l_max = 2048;
inline constexpr uint32_t msm_wave_chain_granularity = 1;
inline constexpr uint64_t msm_wave_region_alignment = 512;
inline constexpr uint32_t msm_wave_control_bytes = 4096;
inline constexpr uint32_t msm_wave_diag_bytes = 4096;
// One captured ScheduleMeta header (task_count[6], merge_levels, order_offset).
inline constexpr uint64_t msm_wave_slot_meta_bytes = 64;
// One 512 B plan-stage summary record per AIV block: u64 local node total,
// u32 local depth, u32 reserved and 32 * u32 per-window caps (zero filled).
// The region is allocated for every usable block, i.e. aiv_cores * 512 B
// (40 * 512 = 20480 B on the target), and block k owns record k.
inline constexpr uint64_t msm_wave_plan_summary_bytes = 512;
// Fixed algorithm-workspace budget of the private wave arena. The bundle width
// is the largest W <= plan.windows whose canonical arena fits; a count whose
// minimal W=1 arena exceeds it keeps W=1 and reports the real bytes so the
// caller's ordinary device-capacity preflight decides.
inline constexpr uint64_t msm_wave_budget_bytes = 9ull * 1024 * 1024 * 1024;
inline constexpr uint64_t msm_wave_control_magic = 0x575641564c333331ULL;  // "WAVL331"
inline constexpr uint32_t msm_wave_none = 0xffffffffu;

// Production shape of one bundle. `fuse_windows` is the ACTUAL bundle width,
// not the plan maximum: the arena is allocated for plan.wave_windows and a tail
// bundle uses the same regions with a smaller shape.
struct WaveShape {
    uint32_t count = 0;              // points per window (plan.count)
    uint32_t window_bits = 0;        // plan.window_bits
    uint32_t windows = 0;            // plan.windows (total)
    uint32_t buckets = 0;            // 1u << window_bits
    uint32_t aiv_cores = 0;          // min(plan.platform.aiv_cores, msm_wave_core_slots)
    uint32_t fuse_windows = 0;       // W: fused windows, >= 1
    uint32_t l_max = 0;              // msm_wave_l_max
    uint32_t chain_granularity = 0;  // msm_wave_chain_granularity
};

// Per-region BYTES, already ceil-aligned to msm_wave_region_alignment. The
// round-33/36 prototype also carried an out_a comparison buffer; the production
// path has no compare arm, so the arena is twelve regions.
//
//   records        = W * count * 8          captured {digit, point_id}
//   rows           = W * (buckets + 1) * 8  captured RowPtr
//   base_offsets   = W * (buckets + 1) * 8  base1 per bucket + sentinel
//   op_src         = (W*count/2) * 4        child index per op
//   op_dst         = (W*count/2) * 4        destination node index
//   prefixes       = P_SLOTS * 128          step-major chain products
//   nodes_a        = NODE_SLOTS * 96        ping-pong A
//   nodes_b        = NODE_SLOTS * 96        ping-pong B
//   out_b          = W * buckets * 192      per-window bucket tuples (Arm B)
//   control        = 4096
//   diag           = 4096
//   plan_summaries = aiv_cores * 512        per-block plan-stage summary
// with A_ub = floor(W*count/2), NODE_SLOTS = ceil(W*(count+buckets)/2) + 1 and
// P_SLOTS = min(A_ub, aiv_cores*lanes*l_max).
//
// P_SLOTS is exact for the CYCLIC step-major prefix: the prefix slot of the
// work step (j, chain c) is j*C + c, which is the op's own index inside the
// sub-wave, so every stored/loaded slot is <= ops_s - 1 for a sub-wave that
// publishes ops_s <= min(A_ub, chunk) ops (chunk = aiv_cores*lanes*l_max).
// Only the valid leading lanes of a step are transferred, so the region needs
// no chain-major rectangle and no padding beyond the published ops; the bound
// does not depend on the chain granularity and an empty level (A_ub == 0 or 0
// ops published) uses a zero-length region without any pair traffic.
struct WaveBounds {
    uint64_t records = 0;
    uint64_t rows = 0;
    uint64_t base_offsets = 0;
    uint64_t op_src = 0;
    uint64_t op_dst = 0;
    uint64_t prefixes = 0;
    uint64_t nodes_a = 0;
    uint64_t nodes_b = 0;
    uint64_t out_b = 0;
    uint64_t control = 0;
    uint64_t diag = 0;
    uint64_t plan_summaries = 0;
    uint64_t total = 0;  // sum of the aligned sizes (512 multiple)
};

// Byte offsets of every region inside the arena (prefix sum of aligned sizes).
struct WaveRegions {
    uint64_t records = 0;
    uint64_t rows = 0;
    uint64_t base_offsets = 0;
    uint64_t op_src = 0;
    uint64_t op_dst = 0;
    uint64_t prefixes = 0;
    uint64_t nodes_a = 0;
    uint64_t nodes_b = 0;
    uint64_t out_b = 0;
    uint64_t control = 0;
    uint64_t diag = 0;
    uint64_t plan_summaries = 0;
};

// Single shared derivation of the arena offsets from the aligned sizes.
inline WaveRegions wave_regions(const WaveBounds& b) noexcept {
    WaveRegions r{};
    uint64_t offset = 0;
    r.records = offset;       offset += b.records;
    r.rows = offset;          offset += b.rows;
    r.base_offsets = offset;  offset += b.base_offsets;
    r.op_src = offset;        offset += b.op_src;
    r.op_dst = offset;        offset += b.op_dst;
    r.prefixes = offset;      offset += b.prefixes;
    r.nodes_a = offset;       offset += b.nodes_a;
    r.nodes_b = offset;       offset += b.nodes_b;
    r.out_b = offset;         offset += b.out_b;
    r.control = offset;       offset += b.control;
    r.diag = offset;          offset += b.diag;
    r.plan_summaries = offset; offset += b.plan_summaries;
    return r;
}

// Canonical arena allocator (defined in msm_plan.cpp). Returns the all-zero
// bounds when the shape is invalid or any size/add/alignment overflows; the
// caller must never clamp.
WaveBounds wave_scratch_bounds(const WaveShape& shape) noexcept;

// Production shape of one bundle of a validated plan. `fuse_windows` must be
// in [1, plan.wave_windows].
WaveShape msm_wave_shape(const MsmPlan& plan, uint32_t fuse_windows) noexcept;

// Arena of one bundle, i.e. wave_scratch_bounds(msm_wave_shape(plan,n)).
inline WaveBounds msm_wave_bounds(const MsmPlan& plan, uint32_t fuse_windows) noexcept {
    return wave_scratch_bounds(msm_wave_shape(plan, fuse_windows));
}

// Number of sub-wave launches per level: ceil(A_ub / (aiv_cores*lanes*l_max))
// with A_ub = floor(W*count/2). Every level publishes ops <= A_ub on device and
// the kernel self-gates `subwave*chunk >= ops_total`, so this is an exact upper
// bound (inactive levels launch only no-op blocks).
inline uint32_t msm_wave_subwave_bound(const WaveShape& s) noexcept {
    if (!s.count || !s.fuse_windows || !s.aiv_cores || !s.l_max) return 0;
    const uint32_t cores = s.aiv_cores > msm_wave_core_slots ? msm_wave_core_slots : s.aiv_cores;
    const uint64_t ops = uint64_t(s.count) * s.fuse_windows / 2;
    const uint64_t chunk = uint64_t(cores) * msm_wave_lanes * s.l_max;
    const uint64_t subwaves = ops / chunk + (ops % chunk != 0);
    return static_cast<uint32_t>(subwaves ? subwaves : 1);
}

// Device-written control block (4096 B). Every value is produced on device and
// read only after completion; no host copy happens inside the compute path.
struct WaveControl {
    uint64_t magic = 0;           // msm_wave_control_magic
    uint32_t count = 0;
    uint32_t window_bits = 0;
    uint32_t windows = 0;
    uint32_t buckets = 0;
    uint32_t fuse_windows = 0;
    uint32_t l_max = 0;
    uint32_t depth_max = 0;       // max depth(u,b)
    uint32_t chain_granularity = 0;
    uint64_t node_count = 0;      // global sum of cap(u,b)
    uint32_t processed_ops = 0;   // ops written by the level build
    uint32_t finalized_buckets = 0;  // nonzero reduction buckets (W*(buckets-1))
    uint32_t status = 0;          // 0 clean; msm_wave_error_* bit otherwise
    uint32_t status_detail = 0;   // offending level / block / shape value
    uint32_t ops[msm_wave_level_slots]{};               // exact A_l per level
    uint32_t parity[msm_wave_window_slots]{};           // per fused window parity
    uint32_t window_records[msm_wave_window_slots]{};   // rows[buckets]
    uint32_t window_node_count[msm_wave_window_slots]{};  // per window sentinel
    uint32_t block_ops[msm_wave_core_slots]{};          // per-slab op counts
    uint32_t reserved[(msm_wave_control_bytes -
                       (64 + 4 * msm_wave_level_slots + 12 * msm_wave_window_slots +
                        4 * msm_wave_core_slots)) / 4]{};
};
static_assert(sizeof(WaveControl) == msm_wave_control_bytes, "wave control block size changed");
static_assert(offsetof(WaveControl, status) == 56, "wave control status offset changed");
static_assert(offsetof(WaveControl, ops) == 64, "wave control ops offset changed");
static_assert(offsetof(WaveControl, parity) == 64 + 4 * msm_wave_level_slots,
              "wave control parity offset changed");
static_assert(offsetof(WaveControl, block_ops) ==
                  64 + 4 * msm_wave_level_slots + 12 * msm_wave_window_slots,
              "wave control block_ops offset changed");

// Device-detected geometry failures. A kernel that cannot honour the frozen
// geometry sets a bit and skips the offending work instead of clamping.
inline constexpr uint32_t msm_wave_error_chain_over_l_max = 1u << 0;
inline constexpr uint32_t msm_wave_error_build_capacity = 1u << 1;
inline constexpr uint32_t msm_wave_error_control_shape = 1u << 2;
inline constexpr uint32_t msm_wave_error_control_magic = 1u << 3;
inline constexpr uint32_t msm_wave_error_geometry = 1u << 4;

// Device-written diagnostics (4096 B): per-block and per-level slots only, no
// atomics. The production path has no compare arm, so checked/mismatch/
// first_mismatch slots stay zero and are retained only to keep the block
// layout identical to the validated prototype.
struct WaveDiag {
    uint32_t inversion_calls_per_block[msm_wave_core_slots]{};
    uint32_t idle_lane_steps_per_block[msm_wave_core_slots]{};
    uint32_t finalized_per_block[msm_wave_core_slots]{};
    uint32_t checked_per_block[msm_wave_core_slots]{};
    uint32_t mismatch_per_block[msm_wave_core_slots]{};
    uint32_t first_mismatch_per_block[msm_wave_core_slots]{};
    uint32_t error_per_block[msm_wave_core_slots]{};
    uint32_t processed_ops_per_level[msm_wave_level_slots]{};
    uint32_t blocks_used_per_level[msm_wave_level_slots]{};
    uint32_t subwaves_per_level[msm_wave_level_slots]{};
    uint32_t chains_per_level[msm_wave_level_slots]{};
    uint32_t node_count_per_window[msm_wave_window_slots]{};
    uint32_t reserved[(msm_wave_diag_bytes -
                      4 * (7 * msm_wave_core_slots + 4 * msm_wave_level_slots +
                           msm_wave_window_slots)) / 4]{};
};
static_assert(sizeof(WaveDiag) == msm_wave_diag_bytes, "wave diag block size changed");
static_assert(offsetof(WaveDiag, processed_ops_per_level) == 4 * 7 * msm_wave_core_slots,
              "wave diag level table offset changed");
static_assert(offsetof(WaveDiag, node_count_per_window) ==
                  4 * (7 * msm_wave_core_slots + 4 * msm_wave_level_slots),
              "wave diag window table offset changed");

// The six wave launchers are declared in msm_internal.h (they need aclrtStream
// and this header must stay usable by ACL-free host plan tests).
//
// capture: for bundle slot `fuse_index`, copies the current window's selected
// records, RowPtr, record parity and the current ScheduleMeta 64-byte header
// into the arena and WaveSlotMeta[u]; it must run before the next window
// overwrites those plan regions.
// plan: one launch per bundle, after every capture of that bundle; its two
// private kernels publish the 64-byte linear partition entries in the build
// regions and then one 512-byte summary record per block (local node total,
// local depth and per-window caps) into plan_summaries.
// build: per level slot 0..25, publishes ops/diag for that level.
// affine: per level and sub-wave msm_wave_subwave_bound().
// carry: per level, after all sub-waves of that level.
// finalize: one launch per bundle, after level 25, publishing out_b per window.

}  // namespace hyperplonk::ascend910b3::detail
