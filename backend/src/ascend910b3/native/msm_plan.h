#pragma once
#include <cstddef>
#include <cstdint>

namespace hyperplonk::ascend910b3::detail {
// Wave controller budget on 196352B UB: FieldVector<Fq,64,false> arena
// 118016B + private 65536B staging on the same pipe + the 8192B SDK planning
// reserve, i.e. 191744B inclusive of the reserve; 4608B stay unaddressed.
inline constexpr uint64_t msm_required_ub_bytes = 191744;
inline constexpr uint32_t msm_point_lanes = 64;

// Frozen 8-bit digit geometry of the target-private MSM metadata path. All
// records are two exact u32 words; nothing here touches field arithmetic.
inline constexpr uint32_t msm_radix_chunk_records = 8192;
inline constexpr uint32_t msm_radix_partition_records = 8192;
inline constexpr uint32_t msm_radix_max_datacopypad_bytes = 32768;
inline constexpr uint32_t msm_radix_max_passes = 3;
inline constexpr uint32_t msm_radix_max_bits = 8;
inline constexpr uint32_t msm_radix_max_bins = 1u << msm_radix_max_bits;
inline constexpr uint32_t msm_radix_control_bytes = 64;
inline constexpr uint32_t msm_schedule_categories = 65;
inline constexpr uint32_t msm_schedule_levels = 6;
inline constexpr uint32_t msm_schedule_meta_header_bytes = 1104;
inline constexpr uint32_t msm_schedule_tile_cat_bytes = msm_schedule_categories * 4;
inline constexpr uint32_t msm_schedule_tile_level_bytes = msm_schedule_levels * 8;
inline constexpr uint32_t msm_schedule_meta_tile_bytes =
    2 * msm_schedule_tile_cat_bytes + 2 * msm_schedule_tile_level_bytes;
static_assert(msm_schedule_tile_cat_bytes == 260, "frozen schedule category tile");
static_assert(msm_schedule_tile_level_bytes == 48, "frozen schedule level tile");
static_assert(msm_schedule_meta_tile_bytes == 616, "frozen ScheduleMeta tile stride");

struct MsmPlatform { uint32_t aiv_cores = 0; uint64_t ub_bytes = 0; };
// Region order is the frozen allocation order; offsets follow the enum order.
// The weighted tail uses two fixed compact buffers (one point tile per usable
// AIV, ceil(/4) for the tree), the fused-wave arena is one private MsmRegion
// whose twelve sub-region offsets come from wave_regions() in msm_wave.h, and
// WaveSlotMeta keeps one captured 64-byte ScheduleMeta header per fused slot.
// MontgomeryBases stays last so the prepared-scratch boundary is unchanged.
enum class MsmRegion : uint32_t {
    RecordsA, RecordsB, RadixFragments, RadixHistogram, RadixDescriptors,
    RadixDigitTotals, RadixControl, RowStarts, RowLocal, RowPtr, RowBlockMin,
    RowBlockCarry, ScheduleMeta, TaskPrefix, WeightedA, WeightedB, Windows,
    WaveArena, WaveSlotMeta, MontgomeryBases, Count
};
inline constexpr uint32_t msm_region_count = static_cast<uint32_t>(MsmRegion::Count);
// Compact weighted-tree capacities: one 192-byte point tile per usable AIV
// (msm_wave_core_slots = 40, so 2560 tuples) and ceil(/4) for the tree level.
inline constexpr uint32_t msm_weighted_fan_in = 4;
inline constexpr uint32_t msm_weighted_capacity = msm_point_lanes * 40;

// Balanced 1..8-bit LSD digit plan: ceil(c/8) levels, q+1 bits for the first
// c%levels levels. Route 1 (9..16 bits) uses high/low decision metadata.
struct MsmRadixPassPlan {
    uint32_t route = 0;
    uint32_t passes = 0;
    uint32_t pass_bits[msm_radix_max_passes]{};
    uint32_t max_bits = 0;
    uint32_t bins = 0;
    uint32_t high_bits = 0;
    uint32_t low_bits = 0;
};
constexpr MsmRadixPassPlan make_msm_radix_pass_plan(uint32_t window_bits) noexcept {
    MsmRadixPassPlan plan{};
    if (window_bits < 2 || window_bits > 20) return plan;
    plan.passes = (window_bits + 7) / 8;
    const uint32_t q = window_bits / plan.passes, r = window_bits % plan.passes;
    for (uint32_t i = 0; i < msm_radix_max_passes; ++i)
        plan.pass_bits[i] = i < plan.passes ? q + (i < r ? 1u : 0u) : 0u;
    plan.route = plan.passes - 1;
    plan.max_bits = q + (r ? 1u : 0u);
    plan.bins = 1u << plan.max_bits;
    plan.low_bits = plan.pass_bits[0];
    plan.high_bits = plan.passes > 1 ? plan.pass_bits[1] : 0u;
    return plan;
}

// Device routing decision produced by the radix group kernels.
struct MsmRadixControl {
    uint32_t final_parity = 0; // 0 selects RecordsA, 1 selects RecordsB
    uint32_t fallback = 0;     // 1 means the generic low-digit pass owns the buffers
    uint32_t max_partition = 0;
    uint32_t reserved = 0;
    uint32_t unused[(msm_radix_control_bytes - 16) / 4]{};
};
static_assert(sizeof(MsmRadixControl) == msm_radix_control_bytes, "frozen RadixControl size");

// Byte offsets of the device schedule metadata that consumers read on device.
struct MsmScheduleLayout {
    uint64_t task_count = 0;
    uint64_t merge_levels = 48;
    uint64_t order_offset = 56;
    uint64_t cat_hist = 64;
    uint64_t cat_start = 584;
    uint64_t tile_cat = msm_schedule_meta_header_bytes;
    uint64_t tile_cat_off = 0;
    uint64_t tile_level_sum = 0;
    uint64_t tile_level_off = 0;
    uint64_t bytes = 0;
};
constexpr MsmScheduleLayout msm_schedule_layout(uint32_t row_blocks) noexcept {
    MsmScheduleLayout layout{};
    layout.tile_cat_off = layout.tile_cat + uint64_t(msm_schedule_tile_cat_bytes) * row_blocks;
    layout.tile_level_sum = layout.tile_cat_off + uint64_t(msm_schedule_tile_cat_bytes) * row_blocks;
    layout.tile_level_off = layout.tile_level_sum + uint64_t(msm_schedule_tile_level_bytes) * row_blocks;
    const uint64_t end = layout.tile_level_off + uint64_t(msm_schedule_tile_level_bytes) * row_blocks;
    layout.bytes = (end + 63) / 64 * 64;
    return layout;
}

struct MsmPlan {
    MsmPlatform platform{};
    uint64_t count = 0, workspace_bytes = 0, dense_records = 0, fragment_records = 0;
    uint64_t host_workspace_bytes = 0, host_prefix_offset = 0;
    // Order-entry capacity of TaskPrefix (level-0 tasks plus one padding entry
    // per bucket); sizes the host order array and bounds make_msm_schedule.
    uint64_t task_capacity = 0;
    // Private fused-wave arena: bytes of the canonical twelve-region bundle for
    // wave_windows windows (0 for an empty plan) and the chosen bundle width.
    uint64_t wave_arena_bytes = 0;
    uint64_t region_offset[msm_region_count]{}, region_size[msm_region_count]{};
    uint32_t window_bits = 0, windows = 0, buckets = 0, radix_passes = 0;
    uint32_t chunks = 0, merge_levels = 0, row_blocks = 0;
    uint32_t weighted_segments = 0, weighted_segment_length = 0;
    uint32_t wave_windows = 0;
    uint32_t radix_route = 0, radix_pass_bits[msm_radix_max_passes]{};
    uint32_t radix_max_bits = 0, radix_bins = 0, radix_high_bits = 0, radix_low_bits = 0;
};
struct MsmSchedule {
    uint64_t task_count[6]{};
    uint32_t merge_levels = 0;
    uint64_t order_offset = 0;
};
struct MsmPreparedSizes {
    uint64_t reusable_bytes = 0;
    uint64_t scratch_bytes = 0;
    uint64_t host_workspace_bytes = 0;
};
// Pure metadata queries: failures leave the caller's output unchanged.
bool validate_msm_plan(const MsmPlan&) noexcept;
bool query_msm_prepared_sizes(const MsmPlan&, MsmPreparedSizes&) noexcept;
bool make_msm_plan(size_t count, const MsmPlatform&, uint32_t requested_window_bits,
                   MsmPlan& out) noexcept;
bool make_msm_schedule(const MsmPlan&, const uint64_t* row_ptr,
                      uint64_t* task_prefix, MsmSchedule& out) noexcept;
} // namespace hyperplonk::ascend910b3::detail
