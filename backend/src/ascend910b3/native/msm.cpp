#include "msm_internal.h"
#include "tiling/platform/platform_ascendc.h"

namespace hyperplonk::ascend910b3::detail {
namespace {
bool same_plan(const MsmPlan& a, const MsmPlan& b) {
    if (a.platform.aiv_cores != b.platform.aiv_cores || a.platform.ub_bytes != b.platform.ub_bytes ||
        a.count != b.count || a.workspace_bytes != b.workspace_bytes ||
        a.dense_records != b.dense_records || a.fragment_records != b.fragment_records ||
        a.host_workspace_bytes != b.host_workspace_bytes || a.host_prefix_offset != b.host_prefix_offset ||
        a.window_bits != b.window_bits || a.windows != b.windows || a.buckets != b.buckets ||
        a.radix_passes != b.radix_passes || a.chunks != b.chunks || a.merge_levels != b.merge_levels ||
        a.row_blocks != b.row_blocks || a.weighted_segments != b.weighted_segments ||
        a.weighted_levels != b.weighted_levels) return false;
    for (unsigned i=0;i<2;++i) if (a.partial_capacity[i]!=b.partial_capacity[i]) return false;
    for (unsigned i=0;i<msm_region_count;++i)
        if (a.region_offset[i]!=b.region_offset[i] || a.region_size[i]!=b.region_size[i]) return false;
    return true;
}
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
    MsmPlan expected{}; Range output{}, input_points{}, input_scalars{}, device{}, host{};
    if (!stream || p.count > SIZE_MAX ||
        !make_msm_plan(static_cast<size_t>(p.count),p.platform,p.window_bits,expected) ||
        !same_plan(p,expected) || !range(out,96,32,output)) return ACL_ERROR_INVALID_PARAM;
    if (!p.count) {
        launch_msm_finish(stream,nullptr,out,p); return ACL_SUCCESS;
    }
    if (workspace_bytes < p.workspace_bytes || host_workspace_bytes < p.host_workspace_bytes ||
        !range(bases,p.count*96,32,input_points) || !range(scalars,p.count*32,32,input_scalars) ||
        !range(workspace,workspace_bytes,64,device) ||
        !range(pinned_host_workspace,host_workspace_bytes,64,host) ||
        overlaps(output,input_points) || overlaps(output,input_scalars) || overlaps(output,device) ||
        overlaps(device,input_points) || overlaps(device,input_scalars) ||
        overlaps(host,output) || overlaps(host,input_points) || overlaps(host,input_scalars) ||
        overlaps(host,device)) return ACL_ERROR_INVALID_PARAM;
    auto* rows=static_cast<uint64_t*>(pinned_host_workspace);
    auto* prefix=reinterpret_cast<uint64_t*>(static_cast<uint8_t*>(pinned_host_workspace)+p.host_prefix_offset);
    const size_t row_bytes=8*(static_cast<size_t>(p.buckets)+1);
    launch_msm_prepare_bases(stream,bases,workspace,p);
    for (uint32_t w=0;w<p.windows;++w) {
        launch_msm_extract(stream,scalars,workspace,p,w);
        for (uint32_t pass=0;pass<p.radix_passes;++pass)
            launch_msm_radix_pass(stream,workspace,p,pass);
        launch_msm_rows(stream,workspace,p);
        aclError status=aclrtMemcpyAsync(rows,row_bytes,region(workspace,p,MsmRegion::RowPtr),
                                         row_bytes,ACL_MEMCPY_DEVICE_TO_HOST,stream);
        if (status!=ACL_SUCCESS) return status;
        status=aclrtSynchronizeStream(stream);
        if (status!=ACL_SUCCESS) return status;
        MsmSchedule schedule{};
        if (!make_msm_schedule(p,rows,prefix,schedule)) return ACL_ERROR_INTERNAL_ERROR;
        const bool empty=schedule.task_count[0]==0;
        if (!empty) {
            const size_t bytes=static_cast<size_t>(p.host_prefix_offset)*(schedule.merge_levels+1);
            status=aclrtMemcpyAsync(region(workspace,p,MsmRegion::TaskPrefix),
                                    p.region_size[static_cast<unsigned>(MsmRegion::TaskPrefix)],
                                    prefix,bytes,ACL_MEMCPY_HOST_TO_DEVICE,stream);
            if (status!=ACL_SUCCESS) return status;
            launch_msm_tasks(stream,workspace,p,0,schedule.task_count[0]);
            launch_msm_partial_sums(stream,workspace,p,schedule.task_count[0]);
            for (uint32_t level=1;level<=schedule.merge_levels;++level) {
                launch_msm_tasks(stream,workspace,p,level,schedule.task_count[level]);
                launch_msm_partial_merge(stream,workspace,p,level,schedule.task_count[level]);
            }
        }
        launch_msm_window_reduce(stream,workspace,p,w,schedule.merge_levels,empty);
    }
    launch_msm_finish(stream,workspace,out,p);
    return ACL_SUCCESS;
}
} // namespace hyperplonk::ascend910b3::detail
