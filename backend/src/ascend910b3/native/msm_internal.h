#pragma once
#include "msm.h"
namespace hyperplonk::ascend910b3::detail {
void launch_msm_extract(aclrtStream, const Fr*, void*, const MsmPlan&, uint32_t window);
void launch_msm_radix_pass(aclrtStream, void*, const MsmPlan&, uint32_t pass);
void launch_msm_rows(aclrtStream, void*, const MsmPlan&);
void launch_msm_tasks(aclrtStream, void*, const MsmPlan&, uint32_t level, uint64_t tasks);
void launch_msm_partial_sums(aclrtStream, const G1Affine*, void*, const MsmPlan&, uint64_t tasks);
void launch_msm_partial_merge(aclrtStream, void*, const MsmPlan&, uint32_t level, uint64_t tasks);
void launch_msm_window_reduce(aclrtStream, void*, const MsmPlan&, uint32_t window,
                              uint32_t final_merge_level, bool empty);
void launch_msm_finish(aclrtStream, const void*, G1Affine*, const MsmPlan&);
// Internal point materialization is implemented in the accumulation unit.
void launch_msm_bucket_materialize(aclrtStream, void*, const MsmPlan&, uint32_t level);
}
