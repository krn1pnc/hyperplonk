#pragma once

#include "field.h"
#include <acl/acl.h>
#include "kernel_tiling/kernel_tiling.h"

namespace hyperplonk::ascend910b3::detail {

inline constexpr size_t workspace_bytes = 26112;
// Borrowed device buffers, canonical Fr, no input/output aliasing. Caller checks
// sizes and retains all resources until stream completion. Enqueue-only: no ACL
// copies, allocation, synchronization, or ownership changes occur here.
void launch_add(aclrtStream, const Fr*, const Fr*, Fr*, size_t, uint32_t);
void launch_sub(aclrtStream, const Fr*, const Fr*, Fr*, size_t, uint32_t);
void launch_mul(aclrtStream, const Fr*, const Fr*, Fr*, size_t, uint32_t);
void launch_inverse(aclrtStream, const Fr*, Fr*, uint8_t*, size_t, uint32_t);
void launch_inner_product(aclrtStream, const Fr*, const Fr*, Fr*, size_t, void*, const AscendC::tiling::TCubeTiling*, uint32_t);
void launch_scalar_vector_mul(aclrtStream, const Fr*, const Fr*, Fr*, size_t, void*, const AscendC::tiling::TCubeTiling*, uint32_t);
// Packing/Cube primitive shared with verification. Unlike field operations this
// accepts arbitrary 32-byte patterns, including noncanonical all-255 limbs.
// valid_count <= 256; padding is mathematical unsigned zero. Leaves signed S,
// signed row/column sums in the documented workspace partitions.
void launch_dot_matrix(aclrtStream, const Fr*, const Fr*, void*, const AscendC::tiling::TCubeTiling*, uint32_t valid_count);

} // namespace hyperplonk::ascend910b3::detail
