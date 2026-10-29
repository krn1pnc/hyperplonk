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

// UB Montgomery domains use radix 2^12; public radix-2^32 constants above do not change.
template<class Params> struct FieldVectorParams;

template<class Params> struct CanonicalRadix12Params {
    static constexpr unsigned radix_bits = 12;
    static constexpr unsigned digit_count = (Params::modulus_bits + radix_bits - 1) / radix_bits;
    struct Digits {
        int32_t values[digit_count]{};
#if defined(__NPU_DEVICE__)
        __aicore__
#endif
        constexpr int32_t operator[](unsigned i) const { return values[i]; }
    };
    static constexpr Digits modulus() {
        Digits result{};
        for (unsigned j = 0; j < digit_count; ++j) {
            const unsigned bit = radix_bits * j, w = bit / 32, shift = bit % 32;
            uint64_t value = uint64_t(Params::modulus[w]) >> shift;
            if (shift + radix_bits > 32 && w + 1 < Params::word_count)
                value |= uint64_t(Params::modulus[w + 1]) << (32 - shift);
            result.values[j] = int32_t(value & 4095);
        }
        return result;
    }
    static constexpr Digits power_of_two(unsigned exponent) {
        Digits result{};
        result.values[0] = 1;
        const auto p = modulus();
        for (unsigned bit = 0; bit < exponent; ++bit) {
            int32_t carry = 0;
            for (unsigned j = 0; j < digit_count; ++j) {
                const int32_t value = 2 * result.values[j] + carry;
                result.values[j] = value & 4095;
                carry = value >> radix_bits;
            }
            bool ge = carry != 0;
            if (!ge) {
                for (unsigned j = digit_count; j-- > 0;) {
                    if (result.values[j] != p.values[j]) {
                        ge = result.values[j] > p.values[j];
                        break;
                    }
                    if (j == 0) ge = true;
                }
            }
            if (ge) {
                int32_t borrow = 0;
                for (unsigned j = 0; j < digit_count; ++j) {
                    const int32_t value = result.values[j] + 4096 - p.values[j] - borrow;
                    result.values[j] = value & 4095;
                    borrow = 1 - (value >> radix_bits);
                }
            }
        }
        return result;
    }
    static constexpr int32_t negative_inverse() {
        unsigned inverse = 1;
        for (unsigned bit = 1; bit < radix_bits; ++bit)
            if ((Params::modulus[0] * inverse) & (1U << bit)) inverse += 1U << bit;
        return int32_t((-inverse) & 4095);
    }
    static constexpr auto modulus_digits = modulus();
    static constexpr auto r_digits = power_of_two(radix_bits * digit_count);
    static constexpr auto r2_digits = power_of_two(2 * radix_bits * digit_count);
    static constexpr int32_t n0prime = negative_inverse();
};

template<> struct FieldVectorParams<Bls12381FrParams> : CanonicalRadix12Params<Bls12381FrParams> {};
template<> struct FieldVectorParams<Bls12381FqParams> : CanonicalRadix12Params<Bls12381FqParams> {};

} // namespace hyperplonk::ascend910b3
