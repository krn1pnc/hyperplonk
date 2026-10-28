#pragma once

#include "field.h"

// The ASC translation unit supplies __aicore__; ordinary host tests need no SDK.
#ifndef HP_FIELD_DEVICE
#define HP_FIELD_DEVICE
#endif
#define HP_FR_INLINE HP_FIELD_DEVICE inline

namespace hyperplonk::ascend910b3 {
namespace field_detail {
HP_FR_INLINE Fr modulus() {
    return {{0x00000001, 0xffffffff, 0xfffe5bfe, 0x53bda402,
             0x09a1d805, 0x3339d808, 0x299d7d48, 0x73eda753}};
}
HP_FR_INLINE bool ge(const Fr& a, const Fr& b) {
    for (int i = 7; i >= 0; --i) {
        if (a.words[i] != b.words[i]) return a.words[i] > b.words[i];
    }
    return true;
}
HP_FR_INLINE Fr subtract(const Fr& a, const Fr& b) {
    Fr r{};
    uint64_t borrow = 0;
    for (unsigned i = 0; i < 8; ++i) {
        const uint64_t rhs = uint64_t(b.words[i]) + borrow;
        r.words[i] = uint32_t(uint64_t(a.words[i]) - rhs);
        borrow = uint64_t(a.words[i]) < rhs;
    }
    return r;
}
HP_FR_INLINE void digits(const Fr& a, uint16_t* out) {
    for (unsigned i = 0; i < 8; ++i) {
        out[2 * i] = uint16_t(a.words[i]);
        out[2 * i + 1] = uint16_t(a.words[i] >> 16);
    }
}
// Column multiplication with fixed bounded carry; all accumulations fit uint64_t.
HP_FR_INLINE void product(const uint16_t* a, unsigned na, const uint16_t* b,
                          unsigned nb, uint16_t* out, unsigned nout) {
    uint64_t carry = 0;
    for (unsigned col = 0; col < nout; ++col) {
        uint64_t sum = carry;
        for (unsigned i = 0; i < na; ++i) {
            if (col >= i && col - i < nb) sum += uint64_t(a[i]) * b[col - i];
        }
        out[col] = uint16_t(sum);
        carry = sum >> 16;
    }
}
HP_FR_INLINE Fr barrett(const uint16_t* x) {
    const uint16_t mu[17] = {0x6393,0x0c0d,0x7a02,0x4273,0xad71,0xbe4b,
        0x3eb4,0x6504,0x8ed3,0x07e0,0xdcb7,0x38b5,0x377c,0xfede,0x94ed,0x3550,0x0002};
    uint16_t p[17]{};
    digits(modulus(), p);
    uint16_t q2[34]{};
    product(x + 15, 17, mu, 17, q2, 34);
    uint16_t qp[17]{};
    product(q2 + 17, 17, p, 16, qp, 17);
    uint16_t r[17]{};
    uint32_t borrow = 0;
    for (unsigned i = 0; i < 17; ++i) {
        const uint32_t rhs = uint32_t(qp[i]) + borrow;
        r[i] = uint16_t(uint32_t(x[i]) - rhs);
        borrow = uint32_t(x[i]) < rhs;
    }
    // Barrett's estimate leaves 0 <= r < 3p, including the 17th digit.
    for (unsigned correction = 0; correction < 2; ++correction) {
        bool greater = true;
        for (int i = 16; i >= 0; --i) {
            if (r[i] != p[i]) { greater = r[i] > p[i]; break; }
        }
        if (greater) {
            borrow = 0;
            for (unsigned i = 0; i < 17; ++i) {
                const uint32_t rhs = uint32_t(p[i]) + borrow;
                const uint32_t lhs = r[i];
                r[i] = uint16_t(lhs - rhs);
                borrow = lhs < rhs;
            }
        }
    }
    Fr result{};
    for (unsigned i = 0; i < 8; ++i) result.words[i] = uint32_t(r[2*i]) | (uint32_t(r[2*i+1]) << 16);
    return result;
}
} // namespace field_detail

HP_FR_INLINE bool from_le_bytes(const uint8_t* input, Fr& out) {
    Fr candidate{};
    for (unsigned i = 0; i < 32; ++i) candidate.words[i/4] |= uint32_t(input[i]) << (8*(i%4));
    if (field_detail::ge(candidate, field_detail::modulus())) return false;
    out = candidate;
    return true;
}
HP_FR_INLINE void to_le_bytes(const Fr& a, uint8_t* output) {
    for (unsigned i = 0; i < 32; ++i) output[i] = uint8_t(a.words[i/4] >> (8*(i%4)));
}
HP_FR_INLINE FrLimbs to_limbs(const Fr& a) {
    FrLimbs result{};
    to_le_bytes(a, result.bytes);
    return result;
}
HP_FR_INLINE Fr add(const Fr& a, const Fr& b) {
    Fr result{};
    uint64_t carry = 0;
    for (unsigned i = 0; i < 8; ++i) {
        const uint64_t sum = uint64_t(a.words[i]) + b.words[i] + carry;
        result.words[i] = uint32_t(sum);
        carry = sum >> 32;
    }
    // Canonical operands are < 2^255, so their sum cannot overflow 256 bits.
    if (field_detail::ge(result, field_detail::modulus())) result = field_detail::subtract(result, field_detail::modulus());
    return result;
}
HP_FR_INLINE Fr sub(const Fr& a, const Fr& b) {
    if (field_detail::ge(a,b)) return field_detail::subtract(a,b);
    return field_detail::subtract(field_detail::modulus(), field_detail::subtract(b,a));
}
HP_FR_INLINE Fr neg(const Fr& a) { return sub(Fr{}, a); }
HP_FR_INLINE Fr mul(const Fr& a, const Fr& b) {
    uint16_t ad[16], bd[16], wide[32];
    field_detail::digits(a, ad);
    field_detail::digits(b, bd);
    field_detail::product(ad,16,bd,16,wide,32);
    return field_detail::barrett(wide);
}
HP_FR_INLINE Fr square(const Fr& a) { return mul(a,a); }
HP_FR_INLINE bool inverse(const Fr& a, Fr& out) {
    uint32_t bits = 0;
    for (unsigned i = 0; i < 8; ++i) bits |= a.words[i];
    if (bits == 0) return false;
    Fr exponent = field_detail::modulus();
    exponent.words[0] -= 2;
    // p's low word is 1: subtracting 2 borrows once into word 1.
    --exponent.words[1];
    Fr result{{1,0,0,0,0,0,0,0}};
    for (int bit = 254; bit >= 0; --bit) {
        result = square(result);
        if ((exponent.words[bit/32] >> (bit%32)) & 1) result = mul(result,a);
    }
    out = result;
    return true;
}
HP_FR_INLINE Fr reduce_wide(const FrWide& input) {
    uint16_t low[32];
    for (unsigned i = 0; i < 32; ++i) low[i] = uint16_t(input.bytes[2*i]) | (uint16_t(input.bytes[2*i+1]) << 8);
    const uint32_t high = uint32_t(input.bytes[64]) | (uint32_t(input.bytes[65]) << 8) |
        (uint32_t(input.bytes[66]) << 16) | (uint32_t(input.bytes[67]) << 24);
    const Fr r2{{0xf3f29c6d,0xc999e990,0x87925c23,0x2b6cedcb,0x7254398f,0x05d31496,0x9f59ff11,0x0748d9d9}};
    const Fr h{{high,0,0,0,0,0,0,0}};
    return add(field_detail::barrett(low), mul(h,r2));
}
} // namespace hyperplonk::ascend910b3
#undef HP_FR_INLINE
