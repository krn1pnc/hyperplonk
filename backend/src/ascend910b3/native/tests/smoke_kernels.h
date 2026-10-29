#pragma once

#include <acl/acl.h>
#include <cstdint>

// Test-only launch boundary: no device headers in the ordinary C++ harness.
namespace field_smoke {
constexpr uint32_t tile_elements = 32;
enum Operation : uint32_t { add, sub, neg, mul, square, inverse, wide };
}
namespace hyperplonk::ascend910b3::test {
void launch_fr_smoke(uint32_t blocks, aclrtStream stream, uint8_t* lhs, uint8_t* rhs,
                     uint8_t* out, uint8_t* mask, uint64_t count, uint32_t op);
void launch_fq_smoke(uint32_t blocks, aclrtStream stream, uint8_t* lhs, uint8_t* rhs,
                     uint8_t* out, uint8_t* mask, uint64_t count, uint32_t op);
void launch_g1_elementwise_smoke(uint32_t blocks, aclrtStream stream, uint8_t* lhs,
    uint8_t* rhs, uint8_t* affine_rhs, uint8_t* factors, uint8_t* out, uint64_t count, uint32_t op);
void launch_g1_inspect_smoke(uint32_t blocks, aclrtStream stream, uint8_t* points,
    uint8_t* rhs_jacobian, uint8_t* rhs_affine, uint8_t* affine_out, uint8_t* flags, uint64_t count);
void launch_g1_diagnostics_smoke(uint32_t blocks, aclrtStream stream, uint8_t* affine,
    uint8_t* jacobian, uint8_t* flags, uint64_t count);
void launch_g1_generator_smoke(aclrtStream stream, uint8_t* jacobian, uint8_t* affine);
void launch_g1_chain_partials(uint32_t blocks, aclrtStream stream, uint8_t* bases,
    uint8_t* scalars, uint8_t* partials, uint32_t count, uint32_t window_bits);
void launch_g1_chain_segments(uint32_t blocks, aclrtStream stream, uint8_t* partials,
    uint8_t* segments, uint32_t window_bits);
void launch_g1_chain_finish(aclrtStream stream, uint8_t* segments, uint8_t* out,
    uint32_t windows, uint32_t window_bits);
} // namespace hyperplonk::ascend910b3::test
