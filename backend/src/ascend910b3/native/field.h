#pragma once

#include <cstddef>
#include <cstdint>

namespace hyperplonk::ascend910b3 {

// Canonical BLS12-381 scalar, not Montgomery storage. Words are least significant first.
struct Fr { uint32_t words[8]; };
struct FrLimbs { uint8_t bytes[32]; };
// Full unsigned integer, including carry above the 512-bit product.
struct FrWide { uint8_t bytes[68]; };
// Signed Cube output; not field elements until correction and reduction.
struct LimbProducts { int32_t values[32 * 32]; };

static_assert(sizeof(Fr) == 32);
static_assert(sizeof(FrLimbs) == 32);

namespace detail {
inline bool checked_field_bytes(size_t count, size_t& bytes) {
    if (count > size_t(PTRDIFF_MAX) / sizeof(Fr)) return false;
    bytes = count * sizeof(Fr);
    return true;
}
} // namespace detail
static_assert(sizeof(FrWide) == 68);

} // namespace hyperplonk::ascend910b3
