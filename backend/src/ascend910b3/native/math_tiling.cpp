#include "math_tiling.h"

#include "tiling/matrix/matmul_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace hyperplonk::ascend910b3::detail {
namespace {
bool make_tiling(const platform_ascendc::PlatformAscendC& platform,
                 int32_t n, int32_t k, AscendC::tiling::TCubeTiling& out)
{
    using namespace matmul_tiling;
    MatmulApiTiling tiling(platform);
    if (tiling.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8, false) != 0 ||
        tiling.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8, false) != 0 ||
        tiling.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_INT32) != 0 ||
        tiling.SetBias(false) != 0 ||
        tiling.SetShape(32, n, k) != 0 ||
        tiling.SetOrgShape(32, n, k) != 0 ||
        tiling.SetFixSplit(32, 32, 32) != 0) {
        return false;
    }
    return tiling.GetTiling(out) == 0;
}
} // namespace

bool build_math_tiling(AscendC::tiling::TCubeTiling& dot,
                       AscendC::tiling::TCubeTiling& scale,
                       uint32_t& aiv_cores)
{
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance("Ascend910B3");
    if (platform == nullptr) {
        return false;
    }
    const uint32_t cores = platform->GetCoreNumAiv();
    uint64_t ub_bytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_bytes);
    // Fixed 88512-byte Vector arena plus 8192-byte SDK runtime reservation.
    constexpr uint64_t required_ub_bytes = 96704;
    if (cores == 0 || platform->GetCoreNumAic() == 0 || ub_bytes < required_ub_bytes) {
        return false;
    }
    AscendC::tiling::TCubeTiling dot_result{};
    AscendC::tiling::TCubeTiling scale_result{};
    if (!make_tiling(*platform, 32, 256, dot_result) ||
        !make_tiling(*platform, 64, 32, scale_result)) {
        return false;
    }
    dot = dot_result;
    scale = scale_result;
    aiv_cores = cores;
    return true;
}
} // namespace hyperplonk::ascend910b3::detail
