#include "msm_plan.h"
#include "msm_wave.h"
#include <limits>

namespace hyperplonk::ascend910b3::detail {
namespace {
constexpr uint64_t limit = static_cast<uint64_t>(PTRDIFF_MAX) < SIZE_MAX
    ? static_cast<uint64_t>(PTRDIFF_MAX) : SIZE_MAX;
uint64_t ceil_div(uint64_t n, uint64_t d) noexcept { return n / d + (n % d != 0); }
bool add(uint64_t a, uint64_t b, uint64_t& out) noexcept {
    if (a > limit || b > limit - a) return false;
    out = a + b; return true;
}
bool mul(uint64_t a, uint64_t b, uint64_t& out) noexcept {
    if (b && a > limit / b) return false;
    out = a * b; return true;
}
bool round_up(uint64_t n, uint64_t k, uint64_t& out) noexcept {
    return mul(ceil_div(n, k), k, out);
}
uint32_t levels(uint64_t n, uint64_t fan_in = 64) noexcept {
    uint32_t l = 0;
    while (n > 1) { n = ceil_div(n,fan_in); ++l; }
    return l;
}
// Arena order of the twelve production sub-regions (msm_wave.h). The
// plan_summaries region is appended after the diagnostic block, so every other
// region keeps the offset it had before it was added.
constexpr uint32_t wave_records = 0, wave_rows = 1, wave_base_offsets = 2,
                   wave_op_src = 3, wave_op_dst = 4, wave_prefixes = 5,
                   wave_nodes_a = 6, wave_nodes_b = 7, wave_out_b = 8,
                   wave_control = 9, wave_diag = 10, wave_plan_summaries = 11,
                   wave_region_count_ = 12;
}
// Single canonical definition of the private wave arena. Every size, add and
// alignment is overflow-checked; an invalid shape yields the all-zero bounds
// instead of a clamped layout.
WaveBounds wave_scratch_bounds(const WaveShape& shape) noexcept {
    const WaveBounds none{};
    const uint64_t N = shape.count, B = shape.buckets, W = shape.fuse_windows;
    const uint64_t g = shape.chain_granularity, l_max = shape.l_max;
    if (!N || !B || !W || !g || !l_max || !shape.aiv_cores) return none;
    if (shape.aiv_cores > msm_wave_core_slots) return none;
    if (shape.window_bits == 0 || shape.window_bits > 31) return none;
    if ((uint64_t{1} << shape.window_bits) != B) return none;
    uint64_t raw[wave_region_count_]{};
    uint64_t value = 0;
    if (!mul(W, N, value) || !mul(value, 8, raw[wave_records])) return none;
    if (!mul(W, B + 1, value) || !mul(value, 8, raw[wave_rows])) return none;
    raw[wave_base_offsets] = raw[wave_rows];
    // A_ub = floor(W*count/2) bounds every level's op count.
    if (!mul(W, N, value)) return none;
    const uint64_t a_ub = value / 2;
    if (!mul(a_ub, 4, raw[wave_op_src])) return none;
    raw[wave_op_dst] = raw[wave_op_src];
    // P_SLOTS = min(A_ub, aiv_cores*lanes*l_max). The step-major prefix of the
    // cyclic chain schedule stores slot (j*C + c), which is the op's own index
    // inside the sub-wave, so a sub-wave that publishes ops_s <= min(A_ub,
    // chunk) ops touches only slots [0, ops_s - 1]: the valid lanes of a step
    // are a leading contiguous prefix and are transferred as such. No
    // chain-major rectangle, no padding past the published ops, and the bound
    // is the same for every chain granularity.
    if (!mul(uint64_t(shape.aiv_cores), msm_wave_lanes, value)) return none;
    uint64_t chunk = 0;
    if (!mul(value, l_max, chunk)) return none;
    const uint64_t p_slots = a_ub < chunk ? a_ub : chunk;
    if (!mul(p_slots, 128, raw[wave_prefixes])) return none;
    // NODE_SLOTS = ceil(W*(count+buckets)/2) + 1
    if (!mul(W, N + B, value) || !add(ceil_div(value, 2), 1, value) ||
        !mul(value, 96, raw[wave_nodes_a])) return none;
    raw[wave_nodes_b] = raw[wave_nodes_a];
    // out_b = W * buckets * 192
    if (!mul(W, B, value) || !mul(value, 192, raw[wave_out_b])) return none;
    raw[wave_control] = msm_wave_control_bytes;
    raw[wave_diag] = msm_wave_diag_bytes;
    // plan_summaries = aiv_cores * 512 (one 512 B record per AIV block).
    if (!mul(uint64_t(shape.aiv_cores), msm_wave_plan_summary_bytes,
             raw[wave_plan_summaries])) return none;
    uint64_t aligned[wave_region_count_]{}, total = 0;
    for (uint32_t i = 0; i < wave_region_count_; ++i) {
        if (!round_up(raw[i], msm_wave_region_alignment, aligned[i])) return none;
        if (!add(total, aligned[i], total)) return none;
    }
    WaveBounds b{};
    b.records = aligned[wave_records];
    b.rows = aligned[wave_rows];
    b.base_offsets = aligned[wave_base_offsets];
    b.op_src = aligned[wave_op_src];
    b.op_dst = aligned[wave_op_dst];
    b.prefixes = aligned[wave_prefixes];
    b.nodes_a = aligned[wave_nodes_a];
    b.nodes_b = aligned[wave_nodes_b];
    b.out_b = aligned[wave_out_b];
    b.control = aligned[wave_control];
    b.diag = aligned[wave_diag];
    b.plan_summaries = aligned[wave_plan_summaries];
    b.total = total;
    return b;
}
WaveShape msm_wave_shape(const MsmPlan& plan, uint32_t fuse_windows) noexcept {
    WaveShape shape{};
    shape.count = static_cast<uint32_t>(plan.count);  // make_msm_plan caps count
    shape.window_bits = plan.window_bits;
    shape.windows = plan.windows;
    shape.buckets = plan.buckets;
    shape.aiv_cores = plan.platform.aiv_cores > msm_wave_core_slots
        ? msm_wave_core_slots : plan.platform.aiv_cores;
    shape.fuse_windows = fuse_windows;
    shape.l_max = msm_wave_l_max;
    shape.chain_granularity = msm_wave_chain_granularity;
    return shape;
}
bool make_msm_plan(size_t count, const MsmPlatform& platform, uint32_t bits,
                   MsmPlan& out) noexcept {
    if (count > UINT32_MAX || !platform.aiv_cores || platform.ub_bytes < msm_required_ub_bytes ||
        (bits && (bits < 2 || bits > 20))) return false;
    MsmPlan p{}; p.platform = platform; p.count = count;
    if (!bits) {
        uint32_t ell = 0;
        for (uint64_t v = count ? count - 1 : 0; v; v >>= 1) ++ell;
        bits = ell - ell / 3 + 2;
        if (bits > 20) bits = 20;
    }
    p.window_bits = bits;
    if (!count) { out = p; return true; }
    const MsmRadixPassPlan digits = make_msm_radix_pass_plan(bits);
    p.windows = (255 + bits - 1) / bits;
    p.buckets = uint32_t{1} << bits;
    p.radix_route = digits.route;
    p.radix_passes = digits.passes;
    for (uint32_t i = 0; i < msm_radix_max_passes; ++i) p.radix_pass_bits[i] = digits.pass_bits[i];
    p.radix_max_bits = digits.max_bits;
    p.radix_bins = digits.bins;
    p.radix_high_bits = digits.high_bits;
    p.radix_low_bits = digits.low_bits;
    p.chunks = static_cast<uint32_t>(ceil_div(count, msm_radix_chunk_records));
    p.merge_levels = levels(ceil_div(count,64));
    p.row_blocks = static_cast<uint32_t>(ceil_div(uint64_t(p.buckets)+1,256));
    // Fit one packed-coordinate point tile per usable AIV, without making tiny
    // segments pay predominantly for absolute-weight restoration. The usable
    // core count is capped at msm_weighted_capacity/msm_point_lanes (40) so the
    // fixed compact WeightedA/B capacities can never be exceeded on a wider
    // device; the launcher may still use more blocks than that.
    const uint32_t weighted_cores = platform.aiv_cores > msm_weighted_capacity/msm_point_lanes
        ? msm_weighted_capacity/msm_point_lanes : platform.aiv_cores;
    const uint64_t ideal_segment = ceil_div(uint64_t(p.buckets)-1,
                                            uint64_t(weighted_cores)*msm_point_lanes);
    const uint32_t segment_length = ideal_segment < 16 ? 16 : static_cast<uint32_t>(ideal_segment);
    p.weighted_segment_length = segment_length;
    p.weighted_segments = static_cast<uint32_t>(ceil_div(uint64_t(p.buckets)-1,segment_length));
    uint64_t padded_rows = 0, fragment_bound = 0, temp = 0, prefix_bytes = 0;
    // Each digit gains at most seven 8-record padding records per chunk.
    if (!round_up(count,8,p.dense_records) ||
        !mul(uint64_t{7} * p.radix_bins, p.chunks, temp) ||
        !add(count,temp,fragment_bound) || !round_up(fragment_bound,8,p.fragment_records) ||
        !round_up(uint64_t(p.buckets)+1,8,padded_rows) || !mul(padded_rows,8,temp)) return false;
    p.host_prefix_offset = temp;
    uint64_t order_bytes = 0;
    if (!add(ceil_div(count,64),p.buckets,p.task_capacity) ||
        !mul(temp,p.merge_levels+1,prefix_bytes) ||
        !mul(p.task_capacity,sizeof(uint32_t),order_bytes) ||
        !add(prefix_bytes,order_bytes,prefix_bytes) ||
        !round_up(prefix_bytes,64,prefix_bytes) ||
        !add(p.host_prefix_offset,prefix_bytes,p.host_workspace_bytes)) return false;
    // Largest legal bundle width whose canonical arena fits the fixed private
    // budget, scanning plan.windows downward. The control block has 32 window
    // slots, so widths above msm_wave_window_slots are never legal. A minimal
    // W=1 arena that still exceeds the budget is kept and reported rather than
    // rejected: the caller's ordinary device-capacity preflight decides.
    const uint32_t width_cap = p.windows < msm_wave_window_slots
        ? p.windows : msm_wave_window_slots;
    for (uint32_t w = width_cap; w >= 1; --w) {
        const WaveBounds bounds = wave_scratch_bounds(msm_wave_shape(p, w));
        if (!bounds.total) return false;
        if (bounds.total <= msm_wave_budget_bytes) {
            p.wave_windows = w; p.wave_arena_bytes = bounds.total; break;
        }
    }
    if (!p.wave_windows) {
        const WaveBounds bounds = wave_scratch_bounds(msm_wave_shape(p, 1));
        if (!bounds.total) return false;
        p.wave_windows = 1; p.wave_arena_bytes = bounds.total;
    }
    if (p.wave_arena_bytes % msm_wave_region_alignment) return false;
    uint64_t cursor = 0;
    auto region = [&](MsmRegion r, uint64_t units, uint64_t bytes,
                      uint64_t alignment) noexcept {
        const auto i = static_cast<uint32_t>(r);
        if (!mul(units,bytes,p.region_size[i])) return false;
        if (!p.region_size[i]) return true;
        if (!round_up(cursor,alignment,cursor)) return false;
        p.region_offset[i] = cursor;
        return add(cursor,p.region_size[i],cursor);
    };
    // The weighted tree only ever needs one point tile per usable AIV and one
    // quarter of that for the next level: fixed compact capacities, never the
    // old per-bucket partial task arrays.
    constexpr uint64_t weighted_a_units = msm_weighted_capacity;
    constexpr uint64_t weighted_b_units =
        (weighted_a_units + msm_weighted_fan_in - 1) / msm_weighted_fan_in;
    if (!region(MsmRegion::RecordsA,p.dense_records,8,64) ||
        !region(MsmRegion::RecordsB,p.dense_records,8,64) ||
        !region(MsmRegion::RadixFragments,p.fragment_records,8,64) ||
        !region(MsmRegion::RadixHistogram,p.chunks,p.radix_bins*4ull,64) ||
        !region(MsmRegion::RadixDescriptors,p.chunks,p.radix_bins*32ull,64) ||
        !region(MsmRegion::RadixDigitTotals,p.radix_bins,64,64) ||
        !region(MsmRegion::RadixControl,1,msm_radix_control_bytes,64) ||
        !region(MsmRegion::RowStarts,uint64_t(p.buckets)+1,64,64) ||
        !region(MsmRegion::RowLocal,padded_rows,8,64) ||
        !region(MsmRegion::RowPtr,padded_rows,8,64) ||
        !region(MsmRegion::RowBlockMin,p.row_blocks,64,64) ||
        !region(MsmRegion::RowBlockCarry,p.row_blocks,64,64) ||
        !region(MsmRegion::ScheduleMeta,1,msm_schedule_layout(p.row_blocks).bytes,64) ||
        !region(MsmRegion::TaskPrefix,prefix_bytes,1,64) ||
        !region(MsmRegion::WeightedA,weighted_a_units,192,64) ||
        !region(MsmRegion::WeightedB,weighted_b_units,192,64) ||
        !region(MsmRegion::Windows,p.windows,192,64) ||
        !region(MsmRegion::WaveArena,1,p.wave_arena_bytes,msm_wave_region_alignment) ||
        !region(MsmRegion::WaveSlotMeta,p.wave_windows,msm_wave_slot_meta_bytes,
                msm_wave_region_alignment) ||
        !region(MsmRegion::MontgomeryBases,count,96,64) ||
        !round_up(cursor,64,p.workspace_bytes)) return false;
    out = p; return true;
}
bool validate_msm_plan(const MsmPlan& a) noexcept {
    MsmPlan b{};
    if (a.count > SIZE_MAX ||
        !make_msm_plan(static_cast<size_t>(a.count),a.platform,a.window_bits,b)) return false;
    if (a.platform.aiv_cores != b.platform.aiv_cores || a.platform.ub_bytes != b.platform.ub_bytes ||
        a.count != b.count || a.workspace_bytes != b.workspace_bytes ||
        a.dense_records != b.dense_records || a.fragment_records != b.fragment_records ||
        a.host_workspace_bytes != b.host_workspace_bytes || a.host_prefix_offset != b.host_prefix_offset ||
        a.task_capacity != b.task_capacity ||
        a.wave_arena_bytes != b.wave_arena_bytes || a.wave_windows != b.wave_windows ||
        a.window_bits != b.window_bits || a.windows != b.windows || a.buckets != b.buckets ||
        a.radix_passes != b.radix_passes || a.chunks != b.chunks || a.merge_levels != b.merge_levels ||
        a.row_blocks != b.row_blocks || a.weighted_segments != b.weighted_segments ||
        a.weighted_segment_length != b.weighted_segment_length ||
        a.radix_route != b.radix_route || a.radix_max_bits != b.radix_max_bits ||
        a.radix_bins != b.radix_bins || a.radix_high_bits != b.radix_high_bits ||
        a.radix_low_bits != b.radix_low_bits) return false;
    for (unsigned i=0;i<msm_radix_max_passes;++i)
        if (a.radix_pass_bits[i]!=b.radix_pass_bits[i]) return false;
    for (unsigned i=0;i<msm_region_count;++i)
        if (a.region_offset[i]!=b.region_offset[i] || a.region_size[i]!=b.region_size[i]) return false;
    return true;
}
bool query_msm_prepared_sizes(const MsmPlan& p,MsmPreparedSizes& out) noexcept {
    if (!validate_msm_plan(p)) return false;
    MsmPreparedSizes sizes{};
    if (p.count) {
        const auto cache=static_cast<unsigned>(MsmRegion::MontgomeryBases);
        sizes.reusable_bytes=p.region_size[cache];
        // The old workspace rounds its final region up; odd counts have32B padding.
        sizes.scratch_bytes=p.region_offset[cache];
        sizes.host_workspace_bytes=p.host_workspace_bytes;
    }
    out=sizes;
    return true;
}
bool make_msm_schedule(const MsmPlan& p, const uint64_t* rows,
                      uint64_t* prefix, MsmSchedule& out) noexcept {
    MsmSchedule s{};
    if (!p.count) { out = s; return true; }
    if (!rows || !prefix || p.merge_levels > 5 || !p.buckets) return false;
    const uint64_t stride = ((uint64_t(p.buckets)+8)/8)*8;
    const uint64_t rb = (uint64_t(p.buckets)+1)*8;
    const uint64_t pb = p.region_size[static_cast<uint32_t>(MsmRegion::TaskPrefix)];
    const uintptr_t r = reinterpret_cast<uintptr_t>(rows), q = reinterpret_cast<uintptr_t>(prefix);
    if (r > UINTPTR_MAX-rb || q > UINTPTR_MAX-pb || (r < q+pb && q < r+rb) ||
        rows[0] || rows[p.buckets] != p.count) return false;
    uint64_t largest = 0;
    for (uint32_t b = 0; b < p.buckets; ++b) {
        if (rows[b] > p.count || rows[b+1] > p.count || rows[b] > rows[b+1]) return false;
        if (b) {
            const uint64_t n = ceil_div(rows[b+1]-rows[b],64);
            if (n > largest) largest = n;
        }
    }
    s.merge_levels = levels(largest);
    if (s.merge_levels > p.merge_levels) return false;
    uint64_t divisor = 64;
    for (uint32_t k = 0; k <= s.merge_levels; ++k) {
        uint64_t total = 0;
        auto* dst = prefix + stride*k;
        for (uint32_t b = 0; b < p.buckets; ++b) {
            dst[b] = total;
            if (b) total += ceil_div(rows[b+1]-rows[b],divisor);
        }
        for (uint64_t b = p.buckets; b < stride; ++b) dst[b] = total;
        s.task_count[k] = total;
        divisor *= 64;
    }
    s.order_offset = p.host_prefix_offset*(s.merge_levels+1);
    auto* order = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(prefix)+s.order_offset);
    uint64_t histogram[65]{}, start[65]{};
    for (uint32_t b = 1; b < p.buckets; ++b) {
        const uint64_t length = rows[b+1]-rows[b];
        histogram[64] += length/64;
        if (length%64) ++histogram[length%64];
    }
    uint64_t position = 0;
    for (uint32_t length = 64; length; --length) {
        start[length] = position;
        position += histogram[length];
    }
    for (uint32_t b = 1; b < p.buckets; ++b) {
        const uint64_t length = rows[b+1]-rows[b], full = length/64;
        const uint64_t base = prefix[b];
        for (uint64_t task = 0; task < full; ++task)
            order[start[64]++] = static_cast<uint32_t>(base+task);
        if (length%64)
            order[start[length%64]++] = static_cast<uint32_t>(base+full);
    }
    out = s; return true;
}
} // namespace hyperplonk::ascend910b3::detail
