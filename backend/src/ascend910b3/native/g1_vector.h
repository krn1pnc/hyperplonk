#pragma once

#include "field.h"
#include "field_vector.h"
#include "msm_plan.h"

#if defined(__NPU_DEVICE__)
#define HP_G1_VECTOR_HEAVY __attribute__((noinline))
#else
#define HP_G1_VECTOR_HEAVY
#endif

namespace hyperplonk::ascend910b3 {

// 48 independent Montgomery Jacobians. Public point handles are 0, 3 and 14;
// scratch fields 6..10 and backup 11..13 are private to each operation. These
// field slots are NOT the FieldVector engine's difference/modulus banks 6/7.
class G1Vector {
public:
    using Point = uint32_t;
    static constexpr uint32_t tile_size = FieldVector<Fq>::tile_size;
    static constexpr uint32_t ub_bytes = FieldVector<Fq>::arena_bytes + 79872;
    static constexpr uint64_t absent = UINT64_MAX;

    __aicore__ inline G1Vector() {
        fq_.pipe_.InitBuffer(extra_, extra_bytes);
        extra_memory_ = extra_.Get<uint8_t>();
    }

    __aicore__ inline void set_identity(Point p) {
        fq_.fill_bank(slot(p), 0);
        mont_one(slot(p + 1));
        fq_.fill_bank(slot(p + 2), 0);
    }

    __aicore__ inline void copy_point(Point dst, Point src) {
        if (dst == src) return;
        for (uint32_t c = 0; c < 3; ++c) copy_field(slot(dst + c), slot(src + c));
    }

    // Distinct public handles never overlap. rhs is preserved; p += p uses the
    // dedicated doubling formula. In particular auxiliary 14..16 is untouched
    // unless it is lhs (or the explicitly mutable base of mul_by_u64).
    __aicore__ inline void add_assign(Point lhs, Point rhs) {
        if (lhs == rhs) double_in_place(lhs);
        else add_impl<false>(lhs, rhs, false);
    }

    // Distinct handles only. rhs must be the unchanged result of load_affine:
    // each lane has Z=MontOne or is normalized identity. All exceptional point
    // cases remain exact; only the finite-rhs Z=1 algebra is specialized.
    __aicore__ inline void add_assign_affine(Point lhs, Point rhs) {
        add_impl<true>(lhs, rhs, false);
    }

    HP_G1_VECTOR_HEAVY __aicore__ inline void double_in_place(Point p) {
        auto a = slot(6), b = slot(7), c = slot(8);
        auto d = slot(9), e = slot(10);
        square(a, slot(p));
        square(b, slot(p + 1));
        square(c, b);
        add(d, slot(p), b);
        square(d, d);
        sub(d, d, a);
        sub(d, d, c);
        add(d, d, d);
        add(e, a, a);
        add(e, e, a);
        square(a, e);
        // Consume old Y and Z before overwriting either coordinate.
        add(b, slot(p + 1), slot(p + 1));
        mul(slot(p + 2), b, slot(p + 2));
        add(b, d, d);
        sub(slot(p), a, b);
        sub(b, d, slot(p));
        mul(slot(p + 1), e, b);
        add(c, c, c);
        add(c, c, c);
        add(c, c, c);
        sub(slot(p + 1), slot(p + 1), c);
        normalize_identity(p);
    }

    // dst/base must be distinct public handles. base may be destroyed. Scalar
    // weight loads are public metadata; add_impl also reads derived Boolean
    // control aggregates, never Fq coordinate words or Scalar Fq arithmetic.
    HP_G1_VECTOR_HEAVY __aicore__ inline void mul_by_u64(
        Point dst, Point base, const uint64_t* weights, uint32_t valid) {
        set_identity(dst);
        uint64_t remaining = 0;
        for (uint32_t lane = 0; lane < valid; ++lane) remaining |= weights[lane];
        uint32_t bit = 0;
        while (remaining != 0) {
            fq_.sync<AscendC::HardEvent::V_S>();
            for (uint32_t lane = 0; lane < tile_size; ++lane)
                fq_.tmp(14).SetValue(lane,
                    lane < valid ? int32_t((weights[lane] >> bit) & 1U) : 0);
            fq_.sync<AscendC::HardEvent::S_V>();
            add_impl<false>(dst, base, true);
            remaining >>= 1;
            ++bit;
            if (remaining != 0) double_in_place(base);
        }
    }

    __aicore__ inline void load_affine(
        Point dst, GM_ADDR src, const uint64_t* byte_offsets, uint32_t valid) {
        load(dst, src, byte_offsets, valid, true);
    }

    __aicore__ inline void load_points(
        Point dst, GM_ADDR src, const uint64_t* byte_offsets, uint32_t valid) {
        load(dst, src, byte_offsets, valid, false);
    }

    HP_G1_VECTOR_HEAVY __aicore__ inline void store_points(
        Point src, GM_ADDR dst, const uint64_t* byte_offsets, uint32_t valid) {
        canonical_one(slot(6));
        for (uint32_t coordinate = 0; coordinate < 3; ++coordinate) {
            mul(slot(7), slot(src + coordinate), slot(6));
            fq_.pack_words(fq_.i32(coordinate * word_soa_bytes), slot(7));
        }
        interleave(false);
        fq_.sync<AscendC::HardEvent::V_MTE3>();
        for (uint32_t lane = 0; lane < valid; ++lane) {
            if (byte_offsets[lane] != absent)
                fq_.copy_out(raw()[lane * jacobian_bytes],
                             dst + byte_offsets[lane], jacobian_bytes);
        }
        fq_.sync<AscendC::HardEvent::MTE3_V>();
    }

    // Final normalization only: inverse_montgomery clobbers engine banks 4/5,
    // which are input-point Y/Z, and they are dead at this stage. Copy all three
    // source coordinates before invoking it; no source-handle alias assumption
    // is needed by the inverse. Only lane zero is written to the public output.
    HP_G1_VECTOR_HEAVY __aicore__ inline void store_affine(Point src, GM_ADDR dst) {
        copy_point(11, src);
        fq_.inverse_montgomery(slot(6), slot(13));
        square(slot(7), slot(6));
        mul(slot(8), slot(11), slot(7));
        mul(slot(9), slot(12), slot(7));
        mul(slot(9), slot(9), slot(6));
        canonical_one(slot(6));
        mul(slot(7), slot(8), slot(6));
        fq_.pack_words(fq_.i32(0), slot(7));
        mul(slot(7), slot(9), slot(6));
        fq_.pack_words(fq_.i32(word_soa_bytes), slot(7));
        interleave(true);
        fq_.sync<AscendC::HardEvent::V_MTE3>();
        fq_.copy_out(raw(), dst, affine_bytes);
        fq_.sync<AscendC::HardEvent::MTE3_V>();
    }

    // Both metadata readers share one 3 KiB buffer. The returned tensor remains
    // valid only until the next metadata read. No point operation uses it.
    __aicore__ inline AscendC::LocalTensor<uint8_t> read_meta(
        GM_ADDR src, uint32_t bytes) {
        clear_meta();
        if (bytes != 0) fq_.copy_in(meta(), src, bytes);
        fq_.sync<AscendC::HardEvent::MTE2_S>();
        return meta();
    }

    __aicore__ inline AscendC::LocalTensor<uint8_t> read_meta_scattered(
        GM_ADDR src, const uint64_t* offsets, uint32_t valid, uint32_t bytes) {
        clear_meta();
        if (bytes != 0) {
            for (uint32_t lane = 0; lane < valid; ++lane) {
                if (offsets[lane] != absent)
                    fq_.copy_in(meta()[lane * metadata_stride],
                                src + offsets[lane], bytes);
            }
        }
        fq_.sync<AscendC::HardEvent::MTE2_S>();
        return meta();
    }

private:
    using I32 = AscendC::LocalTensor<int32_t>;
    using U32 = AscendC::LocalTensor<uint32_t>;
    using Bytes = AscendC::LocalTensor<uint8_t>;
    using Params = FieldVectorParams<Fq::Parameters>;
    static constexpr uint32_t digits = Params::digit_count;
    static constexpr uint32_t field_bytes = digits * tile_size * sizeof(int32_t);
    static constexpr uint32_t word_soa_bytes = Fq::word_count * tile_size * 4;
    static constexpr uint32_t affine_bytes = 96;
    static constexpr uint32_t jacobian_bytes = 192;
    static constexpr uint32_t extra_banks = 11;
    static constexpr uint32_t raw_offset = extra_banks * field_bytes;
    static constexpr uint32_t raw_bytes = tile_size * jacobian_bytes;
    static constexpr uint32_t metadata_offset = raw_offset + raw_bytes;
    static constexpr uint32_t metadata_stride = 64;
    static constexpr uint32_t metadata_bytes = tile_size * metadata_stride;
    static constexpr uint32_t extra_bytes = metadata_offset + metadata_bytes;
    static constexpr uint32_t zero_row_offset = 3 * word_soa_bytes;
    static_assert(tile_size == 48 && field_bytes == 6144 && extra_bytes == 79872);
    static_assert(raw_bytes == 9216 && metadata_bytes == 3072);
    static_assert(FieldVector<Fq>::arena_bytes + extra_bytes == ub_bytes);
    static_assert(ub_bytes + 8192 == detail::msm_required_ub_bytes);
    static_assert(zero_row_offset + tile_size * 4 <= FieldVector<Fq>::raw_mask);
    static_assert((raw_offset | metadata_offset | extra_bytes) % 32 == 0);

    __aicore__ inline I32 slot(uint32_t index) {
        return index < 6 ? fq_.bank(index) :
            extra_memory_[(index - 6) * field_bytes].ReinterpretCast<int32_t>();
    }
    __aicore__ inline I32 digit(I32 field, uint32_t index) {
        return field[index * tile_size];
    }
    __aicore__ inline Bytes raw() { return extra_memory_[raw_offset]; }
    __aicore__ inline Bytes meta() { return extra_memory_[metadata_offset]; }
    __aicore__ inline void copy_field(I32 dst, I32 src) {
        // tmp15 is the permanent radix mask; only transient tmp0 is available.
        fq_.fill(fq_.tmp(0), 1);
        fq_.multiply_rows(dst, src, fq_.tmp(0), true);
    }
    __aicore__ inline void mont_one(I32 dst) {
        for (uint32_t j = 0; j < digits; ++j)
            fq_.fill(digit(dst, j), Params::r_digits[j]);
    }
    __aicore__ inline void canonical_one(I32 dst) {
        fq_.fill_bank(dst, 0);
        fq_.fill(digit(dst, 0), 1);
    }
    __aicore__ inline void add(I32 dst, I32 a, I32 b) { fq_.add_mod(dst, a, b); }
    __aicore__ inline void sub(I32 dst, I32 a, I32 b) { fq_.sub_mod(dst, a, b); }
    __aicore__ inline void mul(I32 dst, I32 a, I32 b) { fq_.montgomery(dst, a, b); }
    __aicore__ inline void square(I32 dst, I32 a) { mul(dst, a, a); }

    // Exact digit-sum test, not integer Compare/Select or Scalar lane reads.
    // tmp12 is the reduction workspace; dst may be tmp12 or tmp13.
    __aicore__ inline void nonzero(I32 dst, I32 field) {
        fq_.fill(fq_.tmp(12), 0);
        for (uint32_t j = 0; j < digits; ++j)
            fq_.add(fq_.tmp(12), fq_.tmp(12), digit(field, j));
        fq_.adds(dst, fq_.tmp(12), 1048575);
        fq_.shift_right(dst, dst, 20);
    }
    __aicore__ inline void complement(I32 mask) {
        fq_.muls(mask, mask, -1);
        fq_.adds(mask, mask, 1);
    }
    __aicore__ inline void zero(I32 dst, I32 field) {
        nonzero(dst, field);
        complement(dst);
    }
    // Exact tile vote on finalized 0/1 masks, not on Fq coordinate words.
    // Each of the eight aggregates covers six lanes and lies in [0,6].
    // This coordinate-derived branch is data-dependent, not constant-time.
    __aicore__ inline bool any_same_vote(I32 same) {
        auto aggregate = fq_.tmp(11);
        const AscendC::BinaryRepeatParams strides{1, 1, 1, 6, 6, 6};
        AscendC::Add<int32_t, true>(
            aggregate, same, same[16], uint64_t(16), uint8_t(1), strides);
        fq_.vector_barrier();
        AscendC::Add<int32_t, true>(
            aggregate, aggregate, same[32], uint64_t(16), uint8_t(1), strides);
        fq_.vector_barrier();
        AscendC::Add<int32_t, true>(
            aggregate, aggregate, aggregate[8], uint64_t(8), uint8_t(1), strides);
        fq_.vector_barrier();
        fq_.sync<AscendC::HardEvent::V_S>();
        uint32_t control = 0;
        for (uint32_t i = 0; i < 8; ++i)
            control |= static_cast<uint32_t>(aggregate.GetValue(i));
        // Complete Scalar control reads before any subsequent Vector reuse.
        fq_.sync<AscendC::HardEvent::S_V>();
        return control != 0;
    }
    __aicore__ inline void select_field(I32 dst, I32 yes, I32 mask) {
        // Engine bank6 is dead arithmetic scratch, not G1 slot6. No source or
        // destination field aliases it, and exceptional masks remain live.
        auto difference = fq_.bank(6);
        fq_.subtract_rows(difference, yes, dst);
        fq_.multiply_rows(difference, difference, mask, true);
        fq_.add_rows(dst, dst, difference);
    }
    __aicore__ inline void select_point(Point dst, Point yes, I32 mask) {
        for (uint32_t c = 0; c < 3; ++c) select_field(slot(dst + c), slot(yes + c), mask);
    }
    __aicore__ inline void normalize_identity(Point p) {
        nonzero(fq_.tmp(13), slot(p + 2));
        fq_.multiply_rows(slot(p), slot(p), fq_.tmp(13), true);
        fq_.multiply_rows(slot(p + 2), slot(p + 2), fq_.tmp(13), true);
        // R is built in engine scratch only after all field arithmetic finishes.
        // The point, its backup, and every live exceptional mask are disjoint.
        auto one = fq_.bank(6);
        mont_one(one);
        fq_.subtract_rows(slot(p + 1), slot(p + 1), one);
        fq_.multiply_rows(slot(p + 1), slot(p + 1), fq_.tmp(13), true);
        fq_.add_rows(slot(p + 1), slot(p + 1), one);
    }

    template<bool affine_rhs>
    HP_G1_VECTOR_HEAVY __aicore__ inline void add_impl(
        Point lhs, Point rhs, bool masked) {
        copy_point(11, lhs);
        auto a_zero = fq_.tmp(6), b_zero = fq_.tmp(7), same = fq_.tmp(8);
        zero(a_zero, slot(13));
        zero(b_zero, slot(rhs + 2));
        if (masked) {
            // Inactive rhs behaves as identity; suppress lhs-identity and same
            // selections too, so even an identity lhs is unchanged in that lane.
            fq_.mul(a_zero, a_zero, fq_.tmp(14));
            complement(b_zero);
            fq_.mul(b_zero, b_zero, fq_.tmp(14));
            complement(b_zero);
        }
        auto zz1 = slot(6), zz2 = slot(7), u1 = slot(8);
        auto h = slot(9), s1 = slot(10);
        square(zz1, slot(13));
        if constexpr (affine_rhs) {
            copy_field(u1, slot(11));
        } else {
            square(zz2, slot(rhs + 2));
            mul(u1, slot(11), zz2);
        }
        mul(h, slot(rhs), zz1);
        if constexpr (affine_rhs) {
            copy_field(s1, slot(12));
        } else {
            mul(s1, slot(12), slot(rhs + 2));
            mul(s1, s1, zz2);
        }
        mul(zz1, zz1, slot(13));
        mul(zz1, zz1, slot(rhs + 1)); // ZZ1 becomes S2, then delta, then R.
        sub(h, h, u1);
        sub(zz1, zz1, s1);
        zero(same, h);
        zero(fq_.tmp(9), zz1);
        if constexpr (affine_rhs) {
            // Explicit opposite-point mask, excluding identities below.
            fq_.copy(fq_.tmp(10), fq_.tmp(9));
            complement(fq_.tmp(10));
            fq_.mul(fq_.tmp(10), same, fq_.tmp(10));
        }
        fq_.mul(same, same, fq_.tmp(9));
        fq_.copy(fq_.tmp(9), a_zero);
        complement(fq_.tmp(9));
        fq_.mul(same, same, fq_.tmp(9));
        if constexpr (affine_rhs)
            fq_.mul(fq_.tmp(10), fq_.tmp(10), fq_.tmp(9));
        fq_.copy(fq_.tmp(9), b_zero);
        complement(fq_.tmp(9));
        fq_.mul(same, same, fq_.tmp(9));
        if constexpr (affine_rhs)
            fq_.mul(fq_.tmp(10), fq_.tmp(10), fq_.tmp(9));
        // 2*Z1*Z2*H is algebraically the existing cuZK next-Z expression.
        if constexpr (affine_rhs) {
            mul(slot(lhs + 2), slot(13), h);
        } else {
            mul(slot(lhs + 2), slot(13), slot(rhs + 2));
            mul(slot(lhs + 2), slot(lhs + 2), h);
        }
        add(slot(lhs + 2), slot(lhs + 2), slot(lhs + 2));
        add(zz2, h, h);
        square(zz2, zz2); // I; the old ZZ2 is now dead.
        mul(h, h, zz2); // J.
        add(zz1, zz1, zz1); // R.
        mul(u1, u1, zz2); // V.
        square(slot(lhs), zz1);
        sub(slot(lhs), slot(lhs), h);
        add(zz2, u1, u1);
        sub(slot(lhs), slot(lhs), zz2);
        sub(u1, u1, slot(lhs));
        mul(slot(lhs + 1), zz1, u1);
        mul(s1, s1, h);
        add(s1, s1, s1);
        sub(slot(lhs + 1), slot(lhs + 1), s1);
        // Backup is still the original lhs here. Always select identity rhs
        // BEFORE the true-vote path can overwrite it. Doubling uses only G1
        // scratch6..10 and tmp0..5/12/13, preserving masks tmp6..10 and tmp14/15.
        select_point(lhs, 11, b_zero);
        if (any_same_vote(same)) {
            double_in_place(11);
            select_point(lhs, 11, same);
        }
        if constexpr (affine_rhs) {
            // Opposite finite points become canonical identity below. Keep this
            // explicit rather than relying only on the formula's zero H.
            complement(fq_.tmp(10));
            fq_.multiply_rows(slot(lhs + 2), slot(lhs + 2), fq_.tmp(10), true);
        }
        select_point(lhs, rhs, a_zero);
        normalize_identity(lhs);
    }

    HP_G1_VECTOR_HEAVY __aicore__ inline void load(
        Point dst, GM_ADDR src, const uint64_t* byte_offsets,
        uint32_t valid, bool affine) {
        const uint32_t stride = affine ? affine_bytes : jacobian_bytes;
        AscendC::Duplicate(raw().ReinterpretCast<int32_t>(), int32_t(0), raw_bytes / 4);
        fq_.vector_barrier();
        fq_.sync<AscendC::HardEvent::V_MTE2>();
        for (uint32_t lane = 0; lane < valid; ++lane) {
            if (byte_offsets[lane] != absent)
                fq_.copy_in(raw()[lane * stride], src + byte_offsets[lane], stride);
        }
        fq_.sync<AscendC::HardEvent::MTE2_V>();
        for (uint32_t j = 0; j < digits; ++j)
            fq_.fill(digit(slot(6), j), Params::r2_digits[j]);
        const uint32_t coordinates = affine ? 2 : 3;
        for (uint32_t c = 0; c < coordinates; ++c) {
            fq_.sync<AscendC::HardEvent::V_S>();
            auto indices = fq_.u32(FieldVector<Fq>::offsets);
            for (uint32_t w = 0; w < Fq::word_count; ++w)
                for (uint32_t lane = 0; lane < tile_size; ++lane)
                    indices.SetValue(w * tile_size + lane,
                                     lane * stride + c * 48 + w * 4);
            fq_.sync<AscendC::HardEvent::S_V>();
            fq_.gather(fq_.u32(FieldVector<Fq>::soa_lhs),
                       raw().ReinterpretCast<uint32_t>(), indices,
                       Fq::word_count * tile_size);
            fq_.unpack(slot(dst + c), fq_.u32(FieldVector<Fq>::soa_lhs));
            mul(slot(dst + c), slot(dst + c), slot(6));
        }
        if (affine) {
            nonzero(fq_.tmp(6), slot(dst));
            nonzero(fq_.tmp(7), slot(dst + 1));
            fq_.mul(fq_.tmp(0), fq_.tmp(6), fq_.tmp(7));
            fq_.add(fq_.tmp(6), fq_.tmp(6), fq_.tmp(7));
            fq_.sub(fq_.tmp(6), fq_.tmp(6), fq_.tmp(0));
            for (uint32_t j = 0; j < digits; ++j)
                fq_.muls(digit(slot(dst + 2), j), fq_.tmp(6), Params::r_digits[j]);
        }
        normalize_identity(dst);
    }

    // Three packed word-SoA arrays reside only in FieldVector's raw region,
    // never in live coordinate banks. Gather produces whole aligned records.
    // 1152-entry chunks fit the 1200-entry offset table, with aligned starts.
    HP_G1_VECTOR_HEAVY __aicore__ inline void interleave(bool affine) {
        fq_.fill(fq_.i32(zero_row_offset), 0);
        const uint32_t record_words = affine ? affine_bytes / 4 : jacobian_bytes / 4;
        const uint32_t total = tile_size * record_words;
        constexpr uint32_t chunk = 1152;
        for (uint32_t first = 0; first < total; first += chunk) {
            const uint32_t count = total - first < chunk ? total - first : chunk;
            fq_.sync<AscendC::HardEvent::V_S>();
            auto indices = fq_.u32(FieldVector<Fq>::offsets);
            for (uint32_t i = 0; i < count; ++i) {
                const uint32_t linear = first + i;
                const uint32_t lane = linear / record_words;
                const uint32_t w = linear % record_words;
                const uint32_t offset = w < 36 ?
                    (w / 12) * word_soa_bytes + (w % 12) * tile_size * 4 + lane * 4 :
                    zero_row_offset + lane * 4;
                indices.SetValue(i, offset);
            }
            fq_.sync<AscendC::HardEvent::S_V>();
            fq_.gather(raw()[first * 4].ReinterpretCast<uint32_t>(),
                       fq_.u32(0), indices, count);
        }
    }

    __aicore__ inline void clear_meta() {
        // All previous Scalar metadata reads finish before Vector reuses it.
        fq_.sync<AscendC::HardEvent::S_V>();
        AscendC::Duplicate(meta().ReinterpretCast<int32_t>(), int32_t(0), metadata_bytes / 4);
        fq_.vector_barrier();
        fq_.sync<AscendC::HardEvent::V_MTE2>();
    }

    FieldVector<Fq> fq_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> extra_;
    Bytes extra_memory_;
};

} // namespace hyperplonk::ascend910b3

#undef HP_G1_VECTOR_HEAVY
