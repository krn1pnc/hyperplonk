#include "msm_plan.h"
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
}
bool make_msm_plan(size_t count, const MsmPlatform& platform, uint32_t bits,
                   MsmPlan& out) noexcept {
    if (count > UINT32_MAX || !platform.aiv_cores || platform.ub_bytes < msm_required_ub_bytes ||
        (bits && (bits < 2 || bits > 16))) return false;
    MsmPlan p{}; p.platform = platform; p.count = count;
    if (!bits) {
        uint32_t ell = 0;
        for (uint64_t v = count ? count - 1 : 0; v; v >>= 1) ++ell;
        bits = ell - ell / 3 + 2;
        if (bits > 16) bits = 16;
    }
    p.window_bits = bits;
    if (!count) { out = p; return true; }
    p.windows = (255 + bits - 1) / bits;
    p.buckets = uint32_t{1} << bits;
    p.radix_passes = (bits + 3) / 4;
    p.chunks = static_cast<uint32_t>(ceil_div(count,2048));
    p.merge_levels = levels(ceil_div(count,64));
    p.row_blocks = static_cast<uint32_t>(ceil_div(uint64_t(p.buckets)+1,256));
    // Fit one 48-lane segment tile per available AIV, without making tiny
    // segments pay predominantly for absolute-weight restoration.
    const uint64_t ideal_segment = ceil_div(uint64_t(p.buckets)-1,uint64_t(platform.aiv_cores)*48);
    const uint32_t segment_length = ideal_segment < 16 ? 16 : static_cast<uint32_t>(ideal_segment);
    p.weighted_segments = static_cast<uint32_t>(ceil_div(uint64_t(p.buckets)-1,segment_length));
    // The weighted tail has much less parallel work than bucket partials.
    p.weighted_levels = levels(p.weighted_segments,4);
    uint64_t padded_rows = 0, fragment_bound = 0, temp = 0, prefix_bytes = 0;
    if (!round_up(count,8,p.dense_records) || !mul(112,p.chunks,temp) ||
        !add(count,temp,fragment_bound) || !round_up(fragment_bound,8,p.fragment_records) ||
        !round_up(uint64_t(p.buckets)+1,8,padded_rows) || !mul(padded_rows,8,temp)) return false;
    p.host_prefix_offset = temp;
    uint64_t order_bytes = 0;
    if (!add(ceil_div(count,64),p.buckets,p.partial_capacity[0]) ||
        !mul(temp,p.merge_levels+1,prefix_bytes) ||
        !mul(p.partial_capacity[0],sizeof(uint32_t),order_bytes) ||
        !add(prefix_bytes,order_bytes,prefix_bytes) ||
        !round_up(prefix_bytes,64,prefix_bytes) ||
        !add(p.host_prefix_offset,prefix_bytes,p.host_workspace_bytes)) return false;
    if ((p.merge_levels || p.weighted_levels) &&
        !add(ceil_div(count,4096),p.buckets,p.partial_capacity[1])) return false;
    uint64_t cursor = 0;
    auto region = [&](MsmRegion r, uint64_t units, uint64_t bytes) noexcept {
        const auto i = static_cast<uint32_t>(r);
        if (!mul(units,bytes,p.region_size[i])) return false;
        if (!p.region_size[i]) return true;
        if (!round_up(cursor,64,cursor)) return false;
        p.region_offset[i] = cursor;
        return add(cursor,p.region_size[i],cursor);
    };
    if (!region(MsmRegion::RecordsA,p.dense_records,8) ||
        !region(MsmRegion::RecordsB,p.dense_records,8) ||
        !region(MsmRegion::RadixFragments,p.fragment_records,8) ||
        !region(MsmRegion::RadixHistogram,p.chunks,64) ||
        !region(MsmRegion::RadixDescriptors,p.chunks,64*16) ||
        !region(MsmRegion::RadixDigitTotals,16,64) ||
        !region(MsmRegion::RowStarts,uint64_t(p.buckets)+1,64) ||
        !region(MsmRegion::RowLocal,padded_rows,8) ||
        !region(MsmRegion::RowPtr,padded_rows,8) ||
        !region(MsmRegion::RowBlockMin,p.row_blocks,64) ||
        !region(MsmRegion::RowBlockCarry,p.row_blocks,64) ||
        !region(MsmRegion::TaskPrefix,prefix_bytes,1) ||
        !region(MsmRegion::Tasks,p.partial_capacity[0],64) ||
        !region(MsmRegion::PartialA,p.partial_capacity[0],192) ||
        !region(MsmRegion::PartialB,p.partial_capacity[1],192) ||
        !region(MsmRegion::Buckets,p.buckets,192) ||
        !region(MsmRegion::Windows,p.windows,192) ||
        !region(MsmRegion::MontgomeryBases,count,96) ||
        !round_up(cursor,64,p.workspace_bytes)) return false;
    out = p; return true;
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
