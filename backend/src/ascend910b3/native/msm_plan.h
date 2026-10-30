#pragma once
#include <cstddef>
#include <cstdint>

namespace hyperplonk::ascend910b3::detail {
// Vector field arena88512B + G1 banks/IO/tables93696B + SDK8192B reserve.
// Includes the cached packed24 2p table (3072B); 5952B remains on 196352B UB.
inline constexpr uint64_t msm_required_ub_bytes = 190400;
struct MsmPlatform { uint32_t aiv_cores = 0; uint64_t ub_bytes = 0; };
enum class MsmRegion : uint32_t {
    RecordsA, RecordsB, RadixFragments, RadixHistogram, RadixDescriptors,
    RadixDigitTotals, RowStarts, RowLocal, RowPtr, RowBlockMin,
    RowBlockCarry, TaskPrefix, Tasks, PartialA, PartialB, Buckets, Windows,
    MontgomeryBases, Count
};
inline constexpr uint32_t msm_region_count = static_cast<uint32_t>(MsmRegion::Count);
struct MsmPlan {
    MsmPlatform platform{};
    uint64_t count = 0, workspace_bytes = 0, dense_records = 0, fragment_records = 0;
    uint64_t host_workspace_bytes = 0, host_prefix_offset = 0;
    uint64_t partial_capacity[2]{};
    uint64_t region_offset[msm_region_count]{}, region_size[msm_region_count]{};
    uint32_t window_bits = 0, windows = 0, buckets = 0, radix_passes = 0;
    uint32_t chunks = 0, merge_levels = 0, row_blocks = 0;
    uint32_t weighted_segments = 0, weighted_levels = 0;
};
struct MsmSchedule {
    uint64_t task_count[6]{};
    uint32_t merge_levels = 0;
    uint64_t order_offset = 0;
};
bool make_msm_plan(size_t count, const MsmPlatform&, uint32_t requested_window_bits,
                   MsmPlan& out) noexcept;
bool make_msm_schedule(const MsmPlan&, const uint64_t* row_ptr,
                      uint64_t* task_prefix, MsmSchedule& out) noexcept;
} // namespace hyperplonk::ascend910b3::detail
