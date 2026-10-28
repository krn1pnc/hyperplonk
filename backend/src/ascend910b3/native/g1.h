#pragma once

#include "field.h"

#define HP_G1_INLINE HP_FIELD_DEVICE inline

namespace hyperplonk::ascend910b3 {

// Trusted native values: coordinates must be canonical Fq, and arithmetic
// requires valid curve points. Production SRS inputs also require prior subgroup
// validation: is_on_curve() is not a subgroup check and does not clear cofactors.
// These compact values are not the 97-byte interchange packet. Branches and
// small-integer multiplication are variable-time; no constant-time guarantee.
// ASC must first include kernel_operator.h and define HP_FIELD_DEVICE __aicore__.
struct G1Jacobian;
struct G1Affine {
    Fq x{};
    Fq y{};

    HP_G1_INLINE static G1Affine identity() { return {}; }
    HP_G1_INLINE static G1Affine generator() {
        return {
            {{0xdb22c6bb, 0xfb3af00a, 0xf97a1aef, 0x6c55e83f,
              0x171bac58, 0xa14e3a3f, 0x9774b905, 0xc3688c4f,
              0x4fa9ac0f, 0x2695638c, 0x3197d794, 0x17f1d3a7}},
            {{0x46c5e7e1, 0x0caa2329, 0xa2888ae4, 0xd03cc744,
              0x2c04b3ed, 0x00db18cb, 0xd5d00af6, 0xfcf5e095,
              0x741d8ae4, 0xa09e30ed, 0xe3aaa0f1, 0x08b3f481}}
        };
    }
    HP_G1_INLINE bool is_identity() const { return x.is_zero() && y.is_zero(); }
    HP_G1_INLINE bool is_on_curve() const {
        if (is_identity()) return true;
        const Fq two = Fq::one() + Fq::one();
        return y.square() == x.square() * x + (two + two);
    }
    HP_G1_INLINE bool operator==(const G1Affine& other) const {
        return x == other.x && y == other.y;
    }
    HP_G1_INLINE bool operator!=(const G1Affine& other) const { return !(*this == other); }
    HP_G1_INLINE G1Affine operator-() const {
        if (is_identity()) return identity();
        return {x, -y};
    }
    HP_G1_INLINE G1Jacobian to_jacobian() const;
};

struct G1Jacobian {
    Fq x{};
    Fq y{{1}};
    Fq z{};

    HP_G1_INLINE static G1Jacobian identity() { return {}; }
    HP_G1_INLINE bool is_identity() const { return z.is_zero(); }
    HP_G1_INLINE bool is_on_curve() const {
        if (is_identity()) return true;
        const Fq zz = z.square();
        const Fq z6 = zz.square() * zz;
        const Fq twice_z6 = z6 + z6;
        return y.square() == x.square() * x + (twice_z6 + twice_z6);
    }
    HP_G1_INLINE bool operator==(const G1Jacobian& other) const {
        if (is_identity()) return other.is_identity();
        if (other.is_identity()) return false;
        const Fq zz1 = z.square();
        const Fq zz2 = other.z.square();
        if (x * zz2 != other.x * zz1) return false;
        return y * other.z * zz2 == other.y * z * zz1;
    }
    HP_G1_INLINE bool operator!=(const G1Jacobian& other) const { return !(*this == other); }
    HP_G1_INLINE G1Jacobian operator-() const {
        if (is_identity()) return identity();
        return {x, -y, z};
    }

    // cuZK BLS12-381 Jacobian formulas, revision
    // 88e3295cada379abe06240df23ec6c9fc0f648b2. Keep all old-input
    // dependencies in temporaries before writing back, including p += p.
    HP_G1_INLINE G1Jacobian& operator+=(const G1Jacobian& other) {
        if (is_identity()) {
            *this = other.is_identity() ? identity() : other;
            return *this;
        }
        if (other.is_identity()) return *this;
        const Fq zz1 = z.square();
        const Fq zz2 = other.z.square();
        const Fq u1 = x * zz2;
        const Fq u2 = other.x * zz1;
        const Fq s1 = y * other.z * zz2;
        const Fq s2 = other.y * z * zz1;
        if (u1 == u2) {
            if (s1 == s2) double_in_place();
            else *this = identity();
            return *this;
        }
        const Fq h = u2 - u1;
        const Fq i = (h + h).square();
        const Fq j = h * i;
        const Fq delta_s = s2 - s1;
        const Fq r = delta_s + delta_s;
        const Fq v = u1 * i;
        const Fq next_x = r.square() - j - (v + v);
        const Fq s1j = s1 * j;
        const Fq next_y = r * (v - next_x) - (s1j + s1j);
        const Fq next_z = ((z + other.z).square() - zz1 - zz2) * h;
        if (next_z.is_zero()) *this = identity();
        else { x = next_x; y = next_y; z = next_z; }
        return *this;
    }
    HP_G1_INLINE void mixed_add_assign(const G1Affine& other) {
        if (is_identity()) {
            *this = other.to_jacobian();
            return;
        }
        if (other.is_identity()) return;
        const Fq zz = z.square();
        const Fq u2 = other.x * zz;
        const Fq s2 = z * other.y * zz;
        if (x == u2) {
            if (y == s2) double_in_place();
            else *this = identity();
            return;
        }
        const Fq h = u2 - x;
        const Fq hh = h.square();
        const Fq twice_hh = hh + hh;
        const Fq i = twice_hh + twice_hh;
        const Fq j = h * i;
        const Fq delta_s = s2 - y;
        const Fq r = delta_s + delta_s;
        const Fq v = x * i;
        const Fq next_x = r.square() - j - (v + v);
        const Fq yj = y * j;
        const Fq next_y = r * (v - next_x) - (yj + yj);
        const Fq next_z = (z + h).square() - zz - hh;
        if (next_z.is_zero()) *this = identity();
        else { x = next_x; y = next_y; z = next_z; }
    }
    HP_G1_INLINE void double_in_place() {
        if (is_identity()) { *this = identity(); return; }
        const Fq a = x.square();
        const Fq b = y.square();
        const Fq c = b.square();
        const Fq half_d = (x + b).square() - a - c;
        const Fq d = half_d + half_d;
        const Fq e = (a + a) + a;
        const Fq f = e.square();
        const Fq next_x = f - (d + d);
        const Fq twice_c = c + c;
        const Fq four_c = twice_c + twice_c;
        const Fq eight_c = four_c + four_c;
        const Fq next_y = e * (d - next_x) - eight_c;
        const Fq next_z = (y + y) * z;
        if (next_z.is_zero()) *this = identity();
        else { x = next_x; y = next_y; z = next_z; }
    }
    HP_G1_INLINE G1Jacobian mul_by_u64(uint64_t n) const {
        if (is_identity() || n == 0) return identity();
        if (n == 1) return *this;
        G1Jacobian result = identity();
        G1Jacobian base = *this;
        while (n != 0) {
            if (n & 1) result += base;
            n >>= 1;
            if (n != 0) base.double_in_place();
        }
        return result;
    }
    HP_G1_INLINE G1Affine to_affine() const {
        if (is_identity()) return G1Affine::identity();
        if (z == Fq::one()) return {x, y};
        Fq zi{};
        // A trusted canonical nonzero Fq is invertible; do not manufacture
        // an identity fallback on a violated native-input precondition.
        z.inverse(zi);
        const Fq zi2 = zi.square();
        return {x * zi2, y * zi2 * zi};
    }
};

HP_G1_INLINE G1Jacobian G1Affine::to_jacobian() const {
    if (is_identity()) return G1Jacobian::identity();
    return {x, y, Fq::one()};
}

static_assert(std::is_standard_layout<G1Affine>::value &&
              std::is_trivially_copyable<G1Affine>::value);
static_assert(std::is_standard_layout<G1Jacobian>::value &&
              std::is_trivially_copyable<G1Jacobian>::value);
static_assert(sizeof(G1Affine) == 96 && alignof(G1Affine) == 4);
static_assert(sizeof(G1Jacobian) == 144 && alignof(G1Jacobian) == 4);
static_assert(offsetof(G1Affine, x) == 0 && offsetof(G1Affine, y) == 48);
static_assert(offsetof(G1Jacobian, x) == 0 && offsetof(G1Jacobian, y) == 48 &&
              offsetof(G1Jacobian, z) == 96);

} // namespace hyperplonk::ascend910b3

#undef HP_G1_INLINE
