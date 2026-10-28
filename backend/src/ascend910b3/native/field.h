#pragma once

#include "field_params.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

// The ASC translation unit supplies __aicore__ before its first inclusion.
#ifndef HP_FIELD_DEVICE
#define HP_FIELD_DEVICE
#endif
#define HP_FIELD_INLINE HP_FIELD_DEVICE inline

namespace hyperplonk::ascend910b3 {

template<class Params> struct LimbBytes { uint8_t bytes[4 * Params::word_count]; };
// An unsigned double-width integer plus a full 32-bit accumulation carry.
template<class Params> struct WideInteger { uint8_t bytes[8 * Params::word_count + 4]; };

namespace field_detail {
template<class Params> constexpr bool valid_parameters() {
    if (Params::word_count == 0 || Params::word_count > 12) return false;
    unsigned bits = 0;
    for (unsigned i = 0; i < Params::word_count; ++i) {
        uint32_t word = Params::modulus[i];
        unsigned local_bits = 0;
        while (word) { ++local_bits; word >>= 1; }
        if (local_bits) bits = 32 * i + local_bits;
    }
    return (Params::modulus[0] & 1) && bits == Params::modulus_bits &&
           bits > 32 * Params::word_count - 16;
}

// Column multiplication: at most 25 radix-2^16 products plus carry per
// column, well within uint64_t. nout may deliberately truncate the product.
HP_FIELD_INLINE void product(const uint16_t* a, unsigned na, const uint16_t* b,
                             unsigned nb, uint16_t* out, unsigned nout) {
    uint64_t carry = 0;
    for (unsigned col = 0; col < nout; ++col) {
        uint64_t sum = carry;
        const unsigned begin = col >= nb ? col - nb + 1 : 0;
        const unsigned end = col + 1 < na ? col + 1 : na;
        for (unsigned i = begin; i < end; ++i) sum += uint64_t(a[i]) * b[col - i];
        out[col] = uint16_t(sum);
        carry = sum >> 16;
    }
}
} // namespace field_detail

// Canonical integer storage, not Montgomery. Raw native writers must provide
// values < p; external encodings go through from_le_bytes. No resource ownership.
template<class Params> struct FieldElement {
    using Parameters = Params;
    static constexpr unsigned word_count = Params::word_count;
    static constexpr unsigned byte_count = 4 * word_count;
    static_assert(field_detail::valid_parameters<Params>(), "invalid fixed field parameters");

    uint32_t words[word_count]{};

    HP_FIELD_INLINE static FieldElement zero() { return {}; }
    HP_FIELD_INLINE static FieldElement one() {
        FieldElement out{};
        out.words[0] = 1;
        return out;
    }
    HP_FIELD_INLINE bool is_zero() const {
        uint32_t bits = 0;
        for (unsigned i = 0; i < word_count; ++i) bits |= words[i];
        return bits == 0;
    }
    HP_FIELD_INLINE bool operator==(const FieldElement& other) const {
        uint32_t difference = 0;
        for (unsigned i = 0; i < word_count; ++i) difference |= words[i] ^ other.words[i];
        return difference == 0;
    }
    HP_FIELD_INLINE bool operator!=(const FieldElement& other) const { return !(*this == other); }
    HP_FIELD_INLINE FieldElement operator+(const FieldElement& other) const {
        FieldElement out{};
        uint64_t carry = 0;
        for (unsigned i = 0; i < word_count; ++i) {
            const uint64_t sum = uint64_t(words[i]) + other.words[i] + carry;
            out.words[i] = uint32_t(sum);
            carry = sum >> 32;
        }
        // Canonical inputs sum to <2p. A storage carry also requires subtracting
        // p modulo R; this remains correct for primes without a spare high bit.
        if (carry || at_least_modulus(out)) subtract_modulus(out);
        return out;
    }
    HP_FIELD_INLINE FieldElement operator-(const FieldElement& other) const {
        FieldElement out{};
        uint64_t borrow = 0;
        for (unsigned i = 0; i < word_count; ++i) {
            const uint64_t rhs = uint64_t(other.words[i]) + borrow;
            out.words[i] = uint32_t(uint64_t(words[i]) - rhs);
            borrow = uint64_t(words[i]) < rhs;
        }
        if (borrow) {
            uint64_t carry = 0;
            for (unsigned i = 0; i < word_count; ++i) {
                const uint64_t sum = uint64_t(out.words[i]) + Params::modulus[i] + carry;
                out.words[i] = uint32_t(sum);
                carry = sum >> 32;
            }
        }
        return out;
    }
    HP_FIELD_INLINE FieldElement operator-() const { return zero() - *this; }
    HP_FIELD_INLINE FieldElement operator*(const FieldElement& other) const {
        uint16_t a[digit_count], b[digit_count], wide[2 * digit_count];
        digits(*this, a);
        digits(other, b);
        field_detail::product(a, digit_count, b, digit_count, wide, 2 * digit_count);
        return barrett(wide);
    }
    HP_FIELD_INLINE FieldElement square() const { return *this * *this; }
    HP_FIELD_INLINE bool inverse(FieldElement& out) const {
        if (is_zero()) return false;
        uint32_t exponent[word_count];
        uint64_t borrow = 2;
        for (unsigned i = 0; i < word_count; ++i) {
            exponent[i] = uint32_t(uint64_t(Params::modulus[i]) - borrow);
            borrow = uint64_t(Params::modulus[i]) < borrow;
        }
        FieldElement result = one();
        for (int bit = int(Params::modulus_bits) - 1; bit >= 0; --bit) {
            result = result.square();
            if ((exponent[bit / 32] >> (bit % 32)) & 1) result = result * *this;
        }
        out = result;
        return true;
    }
    HP_FIELD_INLINE static bool from_le_bytes(const uint8_t* input, FieldElement& out) {
        FieldElement candidate{};
        for (unsigned i = 0; i < byte_count; ++i)
            candidate.words[i / 4] |= uint32_t(input[i]) << (8 * (i % 4));
        if (at_least_modulus(candidate)) return false;
        out = candidate;
        return true;
    }
    HP_FIELD_INLINE void to_le_bytes(uint8_t* output) const {
        for (unsigned i = 0; i < byte_count; ++i)
            output[i] = uint8_t(words[i / 4] >> (8 * (i % 4)));
    }
    HP_FIELD_INLINE LimbBytes<Params> to_limbs() const {
        LimbBytes<Params> out{};
        to_le_bytes(out.bytes);
        return out;
    }
    HP_FIELD_INLINE static FieldElement reduce_wide(const WideInteger<Params>& input) {
        uint16_t low[2 * digit_count];
        for (unsigned i = 0; i < 2 * digit_count; ++i)
            low[i] = uint16_t(input.bytes[2 * i]) | (uint16_t(input.bytes[2 * i + 1]) << 8);
        const FieldElement reduced = barrett(low);
        constexpr unsigned high_offset = 2 * byte_count;
        const uint32_t high = uint32_t(input.bytes[high_offset]) |
            (uint32_t(input.bytes[high_offset + 1]) << 8) |
            (uint32_t(input.bytes[high_offset + 2]) << 16) |
            (uint32_t(input.bytes[high_offset + 3]) << 24);
        if (high == 0) return reduced;
        const uint16_t h[2] = {uint16_t(high), uint16_t(high >> 16)};
        uint16_t factor[digit_count], product[2 * digit_count];
        for (unsigned i = 0; i < word_count; ++i) {
            factor[2 * i] = uint16_t(Params::wide_factor[i]);
            factor[2 * i + 1] = uint16_t(Params::wide_factor[i] >> 16);
        }
        field_detail::product(h, 2, factor, digit_count, product, 2 * digit_count);
        return reduced + barrett(product);
    }

private:
    static constexpr unsigned digit_count = 2 * word_count;
    HP_FIELD_INLINE static bool at_least_modulus(const FieldElement& value) {
        for (unsigned i = word_count; i-- > 0;) {
            if (value.words[i] != Params::modulus[i]) return value.words[i] > Params::modulus[i];
        }
        return true;
    }
    HP_FIELD_INLINE static void subtract_modulus(FieldElement& value) {
        uint64_t borrow = 0;
        for (unsigned i = 0; i < word_count; ++i) {
            const uint64_t rhs = uint64_t(Params::modulus[i]) + borrow;
            const uint64_t lhs = value.words[i];
            value.words[i] = uint32_t(lhs - rhs);
            borrow = lhs < rhs;
        }
    }
    HP_FIELD_INLINE static void digits(const FieldElement& value, uint16_t* out) {
        for (unsigned i = 0; i < word_count; ++i) {
            out[2 * i] = uint16_t(value.words[i]);
            out[2 * i + 1] = uint16_t(value.words[i] >> 16);
        }
    }
    HP_FIELD_INLINE static FieldElement barrett(const uint16_t* x) {
        constexpr unsigned d = digit_count;
        uint16_t p[d + 1]{};
        for (unsigned i = 0; i < word_count; ++i) {
            p[2 * i] = uint16_t(Params::modulus[i]);
            p[2 * i + 1] = uint16_t(Params::modulus[i] >> 16);
        }
        uint16_t q2[2 * (d + 1)];
        field_detail::product(x + d - 1, d + 1, Params::barrett_mu, d + 1, q2, 2 * (d + 1));
        uint16_t qp[d + 1];
        field_detail::product(q2 + d + 1, d + 1, p, d, qp, d + 1);
        uint16_t r[d + 1];
        uint32_t borrow = 0;
        for (unsigned i = 0; i < d + 1; ++i) {
            const uint32_t rhs = uint32_t(qp[i]) + borrow;
            r[i] = uint16_t(uint32_t(x[i]) - rhs);
            borrow = uint32_t(x[i]) < rhs;
        }
        // Normalized Barrett estimate: 0 <= r < 3p, retaining digit d.
        for (unsigned correction = 0; correction < 2; ++correction) {
            bool greater = true;
            for (unsigned i = d + 1; i-- > 0;) {
                if (r[i] != p[i]) { greater = r[i] > p[i]; break; }
            }
            if (greater) {
                borrow = 0;
                for (unsigned i = 0; i < d + 1; ++i) {
                    const uint32_t rhs = uint32_t(p[i]) + borrow;
                    const uint32_t lhs = r[i];
                    r[i] = uint16_t(lhs - rhs);
                    borrow = lhs < rhs;
                }
            }
        }
        FieldElement out{};
        for (unsigned i = 0; i < word_count; ++i)
            out.words[i] = uint32_t(r[2 * i]) | (uint32_t(r[2 * i + 1]) << 16);
        return out;
    }
};

using Fr = FieldElement<Bls12381FrParams>;
using Fq = FieldElement<Bls12381FqParams>;
using FrLimbs = LimbBytes<Bls12381FrParams>;
using FqLimbs = LimbBytes<Bls12381FqParams>;
using FrWide = WideInteger<Bls12381FrParams>;
using FqWide = WideInteger<Bls12381FqParams>;
// Signed Cube output, not a field value or a generic Fq tensor primitive.
struct LimbProducts { int32_t values[32 * 32]; };

static_assert(std::is_standard_layout<Fr>::value && std::is_trivially_copyable<Fr>::value);
static_assert(std::is_standard_layout<Fq>::value && std::is_trivially_copyable<Fq>::value);
static_assert(sizeof(Fr) == 32 && alignof(Fr) == 4 && offsetof(Fr, words) == 0);
static_assert(sizeof(Fq) == 48 && alignof(Fq) == 4 && offsetof(Fq, words) == 0);
static_assert(sizeof(FrLimbs) == 32 && sizeof(FqLimbs) == 48);
static_assert(sizeof(FrWide) == 68 && sizeof(FqWide) == 100);

namespace detail {
template<class Field> inline bool checked_field_bytes(size_t count, size_t& bytes) {
    if (count > size_t(PTRDIFF_MAX) / sizeof(Field)) return false;
    bytes = count * sizeof(Field);
    return true;
}
} // namespace detail
} // namespace hyperplonk::ascend910b3

#undef HP_FIELD_INLINE
