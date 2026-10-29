#pragma once
#include "g1.h"
#include "msm_plan.h"
#include "acl/acl.h"

namespace hyperplonk::ascend910b3::detail {
bool build_msm_plan(size_t count, MsmPlan& out, uint32_t window_bits = 0);
// Borrowed canonical device inputs. Scratch must not be shared concurrently.
// CPU metadata scheduling synchronizes each window; final output is still async.
// Pinned host scratch, device scratch and inputs must live until stream completes,
// including when an ACL failure leaves previously submitted work outstanding.
aclError run_msm(aclrtStream stream, const G1Affine* bases, const Fr* scalars,
                 G1Affine* out, void* workspace, size_t workspace_bytes,
                 void* pinned_host_workspace, size_t host_workspace_bytes,
                 const MsmPlan& plan);
} // namespace hyperplonk::ascend910b3::detail
