#include "msm_internal.h"
#include "msm_wave.h"
#include "tiling/platform/platform_ascendc.h"

namespace hyperplonk::ascend910b3::detail {
namespace {
struct Range { uintptr_t begin, end; };
bool range(const void* p, uint64_t bytes, uintptr_t alignment, Range& out) {
    const uintptr_t start = reinterpret_cast<uintptr_t>(p);
    if (!p || start % alignment || bytes > UINTPTR_MAX-start) return false;
    out = {start,start+static_cast<uintptr_t>(bytes)}; return true;
}
bool overlaps(const Range& a,const Range& b) { return a.begin < b.end && b.begin < a.end; }
uint8_t* region(void* w,const MsmPlan& p,MsmRegion r) {
    return static_cast<uint8_t*>(w)+p.region_offset[static_cast<unsigned>(r)];
}
bool checked_range(const void* p,uint64_t bytes,uintptr_t alignment,Range& out) {
    return bytes <= SIZE_MAX && bytes <= static_cast<uint64_t>(PTRDIFF_MAX) &&
        range(p,bytes,alignment,out);
}
bool same_platform(const MsmPlatform& a,const MsmPlatform& b) {
    return a.aiv_cores==b.aiv_cores && a.ub_bytes==b.ub_bytes;
}
aclError execution_context(const MsmPlan& p,aclrtContext& context) {
    aclError status=aclrtGetCurrentContext(&context);
    if (status!=ACL_SUCCESS) return status;
    if (!context) return ACL_ERROR_INVALID_PARAM;
    // The plan's platform is caller metadata, not proof of the current device.
    const char* soc=aclrtGetSocName();
    const char target[]="Ascend910B3";
    if (!soc) return ACL_ERROR_INVALID_PARAM;
    for (size_t i=0;i<sizeof(target);++i)
        if (soc[i]!=target[i]) return ACL_ERROR_INVALID_PARAM;
    MsmPlan actual{};
    if (!build_msm_plan(static_cast<size_t>(p.count),actual,p.window_bits) ||
        !same_platform(p.platform,actual.platform)) return ACL_ERROR_INVALID_PARAM;
    return ACL_SUCCESS;
}
bool disjoint(const Range* ranges,size_t count) {
    for (size_t i=0;i<count;++i)
        for (size_t j=0;j<i;++j)
            if (overlaps(ranges[i],ranges[j])) return false;
    return true;
}
aclError compute_msm(aclrtStream stream,const void* cache,const Fr* scalars,
                     G1Affine* out,void* workspace,void* pinned_host_workspace,
                     const MsmPlan& p);
}
bool build_msm_plan(size_t count,MsmPlan& out,uint32_t bits) {
    auto* platform=platform_ascendc::PlatformAscendCManager::GetInstance("Ascend910B3");
    if (!platform) return false;
    MsmPlatform info{}; info.aiv_cores=platform->GetCoreNumAiv();
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,info.ub_bytes);
    return make_msm_plan(count,info,bits,out);
}
aclError run_msm(aclrtStream stream,const G1Affine* bases,const Fr* scalars,
                 G1Affine* out,void* workspace,size_t workspace_bytes,
                 void* pinned_host_workspace,size_t host_workspace_bytes,const MsmPlan& p) {
    Range output{}, input_points{}, input_scalars{}, device{}, host{};
    if (!stream || !validate_msm_plan(p) ||
        !checked_range(out,96,32,output)) return ACL_ERROR_INVALID_PARAM;
    aclrtContext context=nullptr;
    const aclError status=execution_context(p,context);
    if (status!=ACL_SUCCESS) return status;
    if (!p.count) return compute_msm(stream,nullptr,scalars,out,nullptr,nullptr,p);
    if (workspace_bytes < p.workspace_bytes || host_workspace_bytes < p.host_workspace_bytes ||
        !checked_range(bases,p.count*96,32,input_points) ||
        !checked_range(scalars,p.count*32,32,input_scalars) ||
        !checked_range(workspace,workspace_bytes,64,device) ||
        !checked_range(pinned_host_workspace,host_workspace_bytes,64,host) ||
        overlaps(output,input_points) || overlaps(output,input_scalars) || overlaps(output,device) ||
        overlaps(device,input_points) || overlaps(device,input_scalars) ||
        overlaps(host,output) || overlaps(host,input_points) || overlaps(host,input_scalars) ||
        overlaps(host,device)) return ACL_ERROR_INVALID_PARAM;
    void* cache=region(workspace,p,MsmRegion::MontgomeryBases);
    launch_msm_prepare_bases(stream,bases,cache,p);
    return compute_msm(stream,cache,scalars,out,workspace,pinned_host_workspace,p);
}
namespace {
// The only post-preparation compute implementation. Cache is independent of
// scratch, so prepared callers never form the old final-region cache address.
// Every dynamic count, ordering offset and record parity is produced and
// consumed on device: no H2D, D2H or host synchronization happens online. The
// pinned host workspace stays allocated and validated but is never touched here.
//
// Windows are processed in bundles of plan.wave_windows: each bundle captures
// its windows' metadata (including the 64-byte ScheduleMeta header into
// WaveSlotMeta BEFORE the next window overwrites it), runs one 26-level wave,
// then reduces every captured window from its own arena slot. A tail bundle
// uses the same allocated arena with its actual width.
aclError compute_msm(aclrtStream stream,const void* cache,const Fr* scalars,
                     G1Affine* out,void* workspace,void* pinned_host_workspace,
                     const MsmPlan& p) {
    (void)pinned_host_workspace;
    if (!p.count) {
        launch_msm_finish(stream,nullptr,out,p); return ACL_SUCCESS;
    }
    if (!p.wave_windows || p.wave_windows > p.windows) return ACL_ERROR_INVALID_PARAM;
    const uint32_t width = p.wave_windows;
    uint8_t* base = static_cast<uint8_t*>(workspace);
    uint8_t* arena = base + p.region_offset[static_cast<uint32_t>(MsmRegion::WaveArena)];
    uint8_t* slot_meta = base + p.region_offset[static_cast<uint32_t>(MsmRegion::WaveSlotMeta)];
    uint8_t* window_slots = base + p.region_offset[static_cast<uint32_t>(MsmRegion::Windows)];
    const uint64_t arena_bytes = p.region_size[static_cast<uint32_t>(MsmRegion::WaveArena)];
    const uint64_t slot_bytes = p.region_size[static_cast<uint32_t>(MsmRegion::WaveSlotMeta)];
    // Pure host geometry preflight: no allocation, transfer or synchronization.
    for (uint32_t first = 0; first < p.windows; first += width) {
        const uint32_t n = p.windows - first < width ? p.windows - first : width;
        const WaveShape shape = msm_wave_shape(p,n);
        const WaveBounds bounds = wave_scratch_bounds(shape);
        if (!bounds.total || bounds.total > arena_bytes ||
            uint64_t(n) * msm_wave_slot_meta_bytes > slot_bytes ||
            !msm_wave_subwave_bound(shape))
            return ACL_ERROR_INVALID_PARAM;
    }
    for (uint32_t first = 0; first < p.windows; first += width) {
        const uint32_t n = p.windows - first < width ? p.windows - first : width;
        const WaveShape shape = msm_wave_shape(p,n);
        const WaveBounds bounds = wave_scratch_bounds(shape);
        const WaveRegions regions = wave_regions(bounds);
        const uint32_t subwaves = msm_wave_subwave_bound(shape);
        for (uint32_t u = 0; u < n; ++u) {
            launch_msm_extract(stream,scalars,workspace,p,first+u);
            launch_msm_radix_group(stream,workspace,p);
            launch_msm_rows(stream,workspace,p);
            launch_msm_schedule(stream,workspace,p);
            launch_msm_wave_capture(stream,workspace,arena,p,bounds,u);
        }
        launch_msm_wave_plan(stream,arena,p,bounds,n);
        for (uint32_t level = 0; level < msm_wave_level_slots; ++level) {
            launch_msm_wave_build(stream,arena,p,bounds,n,level);
            for (uint32_t sub = 0; sub < subwaves; ++sub)
                launch_msm_wave_affine(stream,cache,arena,p,bounds,n,level,sub);
            launch_msm_wave_carry(stream,cache,arena,p,bounds,n,level);
        }
        launch_msm_wave_finalize(stream,cache,arena,p,bounds,n);
        for (uint32_t u = 0; u < n; ++u)
            launch_msm_window_reduce(stream,workspace,p,
                arena + regions.out_b + uint64_t(u)*p.buckets*192,
                window_slots + uint64_t(first+u)*192,
                slot_meta + uint64_t(u)*msm_wave_slot_meta_bytes);
    }
    launch_msm_finish(stream,workspace,out,p);
    return ACL_SUCCESS;
}
}
aclError prepare_msm_srs(aclrtStream stream,const G1Affine* bases,size_t bases_bytes,
                         void* cache,size_t cache_bytes,const MsmPlan& p,
                         std::optional<PreparedSrsView>& out) {
    out.reset();
    MsmPreparedSizes sizes{};
    if (!stream || !query_msm_prepared_sizes(p,sizes)) return ACL_ERROR_INVALID_PARAM;
    if (p.count) {
        Range input{}, destination{};
        if (bases_bytes < sizes.reusable_bytes || cache_bytes < sizes.reusable_bytes ||
            !checked_range(bases,bases_bytes,32,input) ||
            !checked_range(cache,cache_bytes,64,destination) ||
            overlaps(input,destination)) return ACL_ERROR_INVALID_PARAM;
    } else if (bases_bytes || cache_bytes) {
        return ACL_ERROR_INVALID_PARAM;
    }
    aclrtContext context=nullptr;
    aclError status=execution_context(p,context);
    if (status!=ACL_SUCCESS) return status;
    if (p.count) launch_msm_prepare_bases(stream,bases,cache,p);
    // Completion is required even for an empty cache: earlier stream work may
    // still borrow caller resources. A failed drain never publishes a view.
    status=aclrtSynchronizeStream(stream);
    if (status!=ACL_SUCCESS) return status;
    out=PreparedSrsView(p.count ? cache : nullptr,sizes.reusable_bytes,
                        p.count ? cache_bytes : 0,p.count,p.platform,context);
    return ACL_SUCCESS;
}
aclError run_msm_prepared(aclrtStream stream,const PreparedSrsView& prepared,
                          const Fr* scalars,size_t scalar_count,G1Affine* out,
                          void* scratch,size_t scratch_bytes,
                          void* pinned_host_workspace,size_t host_workspace_bytes,
                          const MsmPlan& p) {
    MsmPreparedSizes sizes{};
    Range ranges[5]{};
    if (!stream || !query_msm_prepared_sizes(p,sizes) || scalar_count!=p.count ||
        prepared.count()!=p.count ||
        prepared.format()!=PreparedSrsFormat::PositiveAffineMontgomeryV1 ||
        !same_platform(prepared.platform(),p.platform) ||
        prepared.payload_bytes()!=sizes.reusable_bytes ||
        prepared.capacity_bytes()<prepared.payload_bytes() ||
        !checked_range(out,96,32,ranges[0])) return ACL_ERROR_INVALID_PARAM;
    if (p.count) {
        if (scratch_bytes<sizes.scratch_bytes || host_workspace_bytes<sizes.host_workspace_bytes ||
            !checked_range(prepared.cache(),prepared.capacity_bytes(),64,ranges[1]) ||
            !checked_range(scalars,p.count*32,32,ranges[2]) ||
            !checked_range(scratch,scratch_bytes,64,ranges[3]) ||
            !checked_range(pinned_host_workspace,host_workspace_bytes,64,ranges[4]) ||
            !disjoint(ranges,5)) return ACL_ERROR_INVALID_PARAM;
    } else if (prepared.cache() || prepared.capacity_bytes() ||
               scratch_bytes || host_workspace_bytes) {
        return ACL_ERROR_INVALID_PARAM;
    }
    aclrtContext context=nullptr;
    const aclError status=execution_context(p,context);
    if (status!=ACL_SUCCESS) return status;
    if (context!=prepared.context()) return ACL_ERROR_INVALID_PARAM;
    return compute_msm(stream,prepared.cache(),scalars,out,scratch,pinned_host_workspace,p);
}
} // namespace hyperplonk::ascend910b3::detail
