#pragma once
#include "g1.h"
#include "msm_plan.h"
#include "acl/acl.h"
#include <optional>

namespace hyperplonk::ascend910b3::detail {
bool build_msm_plan(size_t count, MsmPlan& out, uint32_t window_bits = 0);
// Borrowed canonical device inputs. Scratch must not be shared concurrently.
// Metadata scheduling and every dynamic count stay on device; final output is async.
// Pinned host scratch, device scratch and inputs must live until stream completes,
// including when an ACL failure leaves previously submitted work outstanding.
aclError run_msm(aclrtStream stream, const G1Affine* bases, const Fr* scalars,
                 G1Affine* out, void* workspace, size_t workspace_bytes,
                 void* pinned_host_workspace, size_t host_workspace_bytes,
                 const MsmPlan& plan);

enum class PreparedSrsFormat : uint32_t { PositiveAffineMontgomeryV1 = 1 };
// Borrowed immutable cache, not an allocation/context owner. Only a successfully
// completed cold preparation can create a view; copies copy metadata only.
class PreparedSrsView final {
public:
    PreparedSrsView(const PreparedSrsView&) = default;
    PreparedSrsView& operator=(const PreparedSrsView&) = default;
    const void* cache() const noexcept { return cache_; }
    uint64_t payload_bytes() const noexcept { return payload_bytes_; }
    uint64_t capacity_bytes() const noexcept { return capacity_bytes_; }
    uint64_t count() const noexcept { return count_; }
    PreparedSrsFormat format() const noexcept { return format_; }
    const MsmPlatform& platform() const noexcept { return platform_; }
    aclrtContext context() const noexcept { return context_; }
private:
    PreparedSrsView(const void* cache,uint64_t payload,uint64_t capacity,
                    uint64_t count,const MsmPlatform& platform,aclrtContext context) noexcept
        : cache_(cache),payload_bytes_(payload),capacity_bytes_(capacity),count_(count),
          platform_(platform),context_(context) {}
    const void* cache_;
    uint64_t payload_bytes_,capacity_bytes_,count_;
    PreparedSrsFormat format_ = PreparedSrsFormat::PositiveAffineMontgomeryV1;
    MsmPlatform platform_;
    aclrtContext context_;
    friend aclError prepare_msm_srs(aclrtStream,const G1Affine*,size_t,void*,size_t,
                                    const MsmPlan&,std::optional<PreparedSrsView>&);
};
// Clears out first and publishes only after a real successful stream completion.
// Bases/cache/stream/context remain borrowed on every error until safely drained.
// Stream and allocations must belong to the current context; no context switching.
aclError prepare_msm_srs(aclrtStream stream,const G1Affine* canonical_bases,
                         size_t bases_bytes,void* cache,size_t cache_bytes,
                         const MsmPlan& plan,std::optional<PreparedSrsView>& out);
// No canonical bases, preparation, allocations or numerical transfers online.
// Each reader owns disjoint scalar/output/device+pinned scratch ranges. The cache
// and every borrowed resource must outlive completion, including error returns.
// Final output remains asynchronous, exactly as for run_msm.
aclError run_msm_prepared(aclrtStream stream,const PreparedSrsView& prepared,
                          const Fr* scalars,size_t scalar_count,G1Affine* out,
                          void* scratch,size_t scratch_bytes,
                          void* pinned_host_workspace,size_t host_workspace_bytes,
                          const MsmPlan& plan);
} // namespace hyperplonk::ascend910b3::detail
