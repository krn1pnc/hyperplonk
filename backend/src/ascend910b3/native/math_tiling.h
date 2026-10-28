#ifndef HYPERPLONK_ASCEND910B3_MATH_TILING_H
#define HYPERPLONK_ASCEND910B3_MATH_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace hyperplonk::ascend910b3::detail {

// Publishes outputs only after both exact signed-INT8 tilings and resources validate.
bool build_math_tiling(AscendC::tiling::TCubeTiling& dot,
                       AscendC::tiling::TCubeTiling& scale,
                       uint32_t& aiv_cores);

} // namespace hyperplonk::ascend910b3::detail
#endif
