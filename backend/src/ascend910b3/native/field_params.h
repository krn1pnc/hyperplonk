#pragma once

#include <cstdint>

namespace hyperplonk::ascend910b3 {

// BLS12-381 primes; little-endian radix-2^32 words. For R=2^(32N),
// barrett_mu=floor(R^2/p) in radix 2^16, wide_factor=R^2 mod p.
struct Bls12381FrParams {
    static constexpr unsigned word_count = 8;
    static constexpr unsigned modulus_bits = 255;
    static constexpr uint32_t modulus[word_count] = {
        0x00000001, 0xffffffff, 0xfffe5bfe, 0x53bda402,
        0x09a1d805, 0x3339d808, 0x299d7d48, 0x73eda753};
    static constexpr uint16_t barrett_mu[2 * word_count + 1] = {
        0x6393, 0x0c0d, 0x7a02, 0x4273, 0xad71, 0xbe4b, 0x3eb4, 0x6504,
        0x8ed3, 0x07e0, 0xdcb7, 0x38b5, 0x377c, 0xfede, 0x94ed, 0x3550, 0x0002};
    static constexpr uint32_t wide_factor[word_count] = {
        0xf3f29c6d, 0xc999e990, 0x87925c23, 0x2b6cedcb,
        0x7254398f, 0x05d31496, 0x9f59ff11, 0x0748d9d9};
};

struct Bls12381FqParams {
    static constexpr unsigned word_count = 12;
    static constexpr unsigned modulus_bits = 381;
    static constexpr uint32_t modulus[word_count] = {
        0xffffaaab, 0xb9feffff, 0xb153ffff, 0x1eabfffe,
        0xf6b0f624, 0x6730d2a0, 0xf38512bf, 0x64774b84,
        0x434bacd7, 0x4b1ba7b6, 0x397fe69a, 0x1a0111ea};
    static constexpr uint16_t barrett_mu[2 * word_count + 1] = {
        0xba2e, 0x6591, 0x07f5, 0x13e2, 0xc07b, 0x58f1, 0x67a0, 0x9971,
        0x79d3, 0x2867, 0x71e0, 0xdf47, 0xa94b, 0xf6a0, 0x741f, 0x1b82,
        0xba29, 0xc7a6, 0x1b0c, 0x2810, 0x45ce, 0xcc9e, 0xd2f3, 0xd835, 0x0009};
    static constexpr uint32_t wide_factor[word_count] = {
        0x1c341746, 0xf4df1f34, 0x09d104f1, 0x0a76e6a6,
        0x4c95b6d5, 0x8de5476c, 0x939d83c0, 0x67eb88a9,
        0xb519952d, 0x9a793e85, 0x92cae3aa, 0x11988fe5};
};

} // namespace hyperplonk::ascend910b3
