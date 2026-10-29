#pragma once

#include "field.h"
#include <acl/acl.h>
#include "kernel_tiling/kernel_tiling.h"

namespace hyperplonk::ascend910b3::detail {

inline constexpr size_t workspace_bytes = 26112;
// Borrowed canonical device fields, no input/output aliasing. Caller checks sizes
// and retains resources until completion; nonempty calls require cores > 0.
// Enqueue-only: no runtime copies, allocation, synchronization or ownership changes.
enum class FieldOperation : uint32_t { add, sub, neg, mul, square, inverse, wide };
template<class Field>
void launch_field(aclrtStream, const void*, const void*, Field*, uint8_t*,
                  size_t, uint32_t, FieldOperation);

template<class Field>
inline void launch_add(aclrtStream s, const Field* a, const Field* b, Field* out,
                       size_t n, uint32_t cores) {
    launch_field<Field>(s, a, b, out, nullptr, n, cores, FieldOperation::add);
}
template<class Field>
inline void launch_sub(aclrtStream s, const Field* a, const Field* b, Field* out,
                       size_t n, uint32_t cores) {
    launch_field<Field>(s, a, b, out, nullptr, n, cores, FieldOperation::sub);
}
template<class Field>
inline void launch_mul(aclrtStream s, const Field* a, const Field* b, Field* out,
                       size_t n, uint32_t cores) {
    launch_field<Field>(s, a, b, out, nullptr, n, cores, FieldOperation::mul);
}
template<class Field>
inline void launch_inverse(aclrtStream s, const Field* a, Field* out, uint8_t* mask,
                           size_t n, uint32_t cores) {
    launch_field<Field>(s, a, nullptr, out, mask, n, cores, FieldOperation::inverse);
}
template<class Field>
inline void launch_neg(aclrtStream s, const Field* a, Field* out, size_t n, uint32_t cores) {
    launch_field<Field>(s, a, nullptr, out, nullptr, n, cores, FieldOperation::neg);
}
template<class Field>
inline void launch_square(aclrtStream s, const Field* a, Field* out, size_t n, uint32_t cores) {
    launch_field<Field>(s, a, nullptr, out, nullptr, n, cores, FieldOperation::square);
}
template<class Field>
inline void launch_reduce_wide(aclrtStream s, const WideInteger<typename Field::Parameters>* a,
                               Field* out, size_t n, uint32_t cores) {
    launch_field<Field>(s, a, nullptr, out, nullptr, n, cores, FieldOperation::wide);
}
void launch_inner_product(aclrtStream, const Fr*, const Fr*, Fr*, size_t, void*, const AscendC::tiling::TCubeTiling*, uint32_t);
void launch_scalar_vector_mul(aclrtStream, const Fr*, const Fr*, Fr*, size_t, void*, const AscendC::tiling::TCubeTiling*, uint32_t);
// Packing/Cube primitive shared with verification. Unlike field operations this
// accepts arbitrary 32-byte patterns, including noncanonical all-255 limbs.
// valid_count <= 256; padding is mathematical unsigned zero. Leaves signed S,
// signed row/column sums in the documented workspace partitions.
void launch_dot_matrix(aclrtStream, const Fr*, const Fr*, void*, const AscendC::tiling::TCubeTiling*, uint32_t valid_count);

} // namespace hyperplonk::ascend910b3::detail
