#pragma once

#include "field.h"
#include "field_vector.h"
#include "msm_plan.h"
#include <type_traits>

#if defined(__NPU_DEVICE__)
#define HP_G1_VECTOR_HEAVY __attribute__((noinline))
#else
#define HP_G1_VECTOR_HEAVY
#endif

namespace hyperplonk::ascend910b3 {

// Independent Montgomery Jacobians. Public point handles are 0, 3 and 14;
// scratch fields 6..10 and backup 11..13 are private to each operation. Packed
// coordinates contain exact normalized radix-12 digits; arithmetic stays I32.
template<uint32_t Lanes = 48, bool Packed = false>
class G1VectorEngine {
    friend struct SquareProbeAccess;
    using Engine = FieldVector<Fq, Lanes, Packed>;
    using Coordinate = std::conditional_t<Packed, int16_t, int32_t>;
    using C = AscendC::LocalTensor<Coordinate>;
public:
    using Point = uint32_t;
    static constexpr uint32_t tile_size = Engine::tile_size;
    static constexpr uint64_t absent = UINT64_MAX;

    __aicore__ inline G1VectorEngine() {
        fq_.pipe_.InitBuffer(extra_, extra_bytes);
        extra_memory_ = extra_.template Get<uint8_t>();
        // Immutable public tables are built once per core, not per point load.
        for (uint32_t j = 0; j < digits; ++j)
            fill_coordinate(digit(cached_mont_one(), j), Params::r_digits[j]);
        for (uint32_t j = 0; j < Engine::packed_digits; ++j)
            fq_.fill(digit(cached_packed_2p(), j),
                     Engine::packed_double_modulus.values[j]);
        fq_.template sync<AscendC::HardEvent::V_S>();
        auto affine_indices = load_indices(true);
        auto jacobian_indices = load_indices(false);
        for (uint32_t word = 0; word < Fq::word_count; ++word)
            for (uint32_t lane = 0; lane < tile_size; ++lane) {
                affine_indices.SetValue(word * tile_size + lane,
                                        lane * affine_bytes + word * 4);
                jacobian_indices.SetValue(word * tile_size + lane,
                                          lane * jacobian_bytes + word * 4);
            }
        fq_.template sync<AscendC::HardEvent::S_V>();
        if constexpr (Packed) prepare_widen_indices();
    }

    __aicore__ inline void set_identity(Point p) {
        fill_field(slot(p), 0);
        mont_one(slot(p + 1));
        fill_field(slot(p + 2), 0);
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
            fq_.template sync<AscendC::HardEvent::V_S>();
            for (uint32_t lane = 0; lane < tile_size; ++lane)
                fq_.tmp(14).SetValue(lane,
                    lane < valid ? int32_t((weights[lane] >> bit) & 1U) : 0);
            fq_.template sync<AscendC::HardEvent::S_V>();
            add_impl<false>(dst, base, true);
            remaining >>= 1;
            ++bit;
            if (remaining != 0) double_in_place(base);
        }
    }

    __aicore__ inline void load_affine(
        Point dst, GM_ADDR src, const uint64_t* byte_offsets, uint32_t valid) {
        load<false, true>(dst, src, byte_offsets, valid);
    }

    __aicore__ inline void load_points(
        Point dst, GM_ADDR src, const uint64_t* byte_offsets, uint32_t valid) {
        load<false, false>(dst, src, byte_offsets, valid);
    }

    // Internal MSM records contain normalized packed Montgomery integers <2p,
    // not radix-12 banks. Public canonical loaders retain their conversion.
    __aicore__ inline void load_affine_montgomery(
        Point dst, GM_ADDR src, const uint64_t* byte_offsets, uint32_t valid) {
        load<true, true>(dst, src, byte_offsets, valid);
    }

    __aicore__ inline void load_points_montgomery(
        Point dst, GM_ADDR src, const uint64_t* byte_offsets, uint32_t valid) {
        load<true, false>(dst, src, byte_offsets, valid);
    }

    HP_G1_VECTOR_HEAVY __aicore__ inline void store_points(
        Point src, GM_ADDR dst, const uint64_t* byte_offsets, uint32_t valid) {
        store<false, false>(src, dst, byte_offsets, valid);
    }

    HP_G1_VECTOR_HEAVY __aicore__ inline void store_points_montgomery(
        Point src, GM_ADDR dst, const uint64_t* byte_offsets, uint32_t valid) {
        store<true, false>(src, dst, byte_offsets, valid);
    }

    // Preconversion only: src is an unchanged affine load, so finite Z=R.
    // Pack X/Y directly without inversion; normalized identity becomes (0,0).
    // All source coordinates and the other public point handles are preserved.
    HP_G1_VECTOR_HEAVY __aicore__ inline void store_affine_montgomery(
        Point src, GM_ADDR dst, const uint64_t* byte_offsets, uint32_t valid) {
        store<true, true>(src, dst, byte_offsets, valid);
    }

    // Final normalization only: inverse_montgomery clobbers engine banks 4/5,
    // which are input-point Y/Z, and they are dead at this stage. Copy all three
    // source coordinates before invoking it; no source-handle alias assumption
    // is needed by the inverse. Only lane zero is written to the public output.
    HP_G1_VECTOR_HEAVY __aicore__ inline void store_affine(Point src, GM_ADDR dst) {
        static_assert(!Packed, "final affine normalization requires the full I32 engine");
        copy_point(11, src);
        fq_.inverse_montgomery(slot(6), slot(13));
        square(slot(7), slot(6));
        mul(slot(8), slot(11), slot(7));
        mul(slot(9), slot(12), slot(7));
        mul(slot(9), slot(9), slot(6));
        canonical_one(slot(6));
        // Public affine coordinates must leave the redundant domain.
        fq_.montgomery(slot(7), slot(8), slot(6));
        fq_.pack_words(fq_.i32(0), slot(7));
        fq_.montgomery(slot(7), slot(9), slot(6));
        fq_.pack_words(fq_.i32(word_soa_bytes), slot(7));
        interleave(true);
        fq_.template sync<AscendC::HardEvent::V_MTE3>();
        fq_.copy_out(raw(), dst, affine_bytes);
        fq_.template sync<AscendC::HardEvent::MTE3_V>();
    }

    // Both metadata readers share one 3 KiB buffer. The returned tensor remains
    // valid only until the next metadata read. No point operation uses it.
    __aicore__ inline AscendC::LocalTensor<uint8_t> read_meta(
        GM_ADDR src, uint32_t bytes) {
        clear_meta();
        if (bytes != 0) fq_.copy_in(meta(), src, bytes);
        fq_.template sync<AscendC::HardEvent::MTE2_S>();
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
        fq_.template sync<AscendC::HardEvent::MTE2_S>();
        return meta();
    }

private:
    using I32 = AscendC::LocalTensor<int32_t>;
    using U32 = AscendC::LocalTensor<uint32_t>;
    using Bytes = AscendC::LocalTensor<uint8_t>;
    using Params = FieldVectorParams<Fq::Parameters>;
    static constexpr uint32_t digits = Params::digit_count;
    static constexpr uint32_t field_bytes = digits * tile_size * sizeof(Coordinate);
    static constexpr uint8_t coordinate_stride = tile_size * sizeof(Coordinate) / 32;
    static constexpr uint8_t i32_stride = tile_size * sizeof(int32_t) / 32;
    static constexpr uint32_t word_soa_bytes = Fq::word_count * tile_size * 4;
    static constexpr uint32_t affine_bytes = 96;
    static constexpr uint32_t jacobian_bytes = 192;
    static constexpr uint32_t extra_banks = Packed ? 17 : 11;
    static constexpr uint32_t raw_offset = extra_banks * field_bytes;
    static constexpr uint32_t raw_bytes = tile_size * jacobian_bytes;
    static constexpr uint32_t metadata_offset = raw_offset + raw_bytes;
    static constexpr uint32_t metadata_stride = 64;
    static constexpr uint32_t metadata_bytes = tile_size * metadata_stride;
    static constexpr uint32_t mont_one_offset = metadata_offset + metadata_bytes;
    static constexpr uint32_t affine_indices_offset = mont_one_offset + field_bytes;
    static constexpr uint32_t jacobian_indices_offset = affine_indices_offset + word_soa_bytes;
    static constexpr uint32_t packed_2p_offset = jacobian_indices_offset + word_soa_bytes;
    static constexpr uint32_t packed_2p_bytes =
        Engine::packed_digits * tile_size * sizeof(int32_t);
    static constexpr uint32_t extra_bytes = packed_2p_offset + packed_2p_bytes;
    static constexpr uint32_t zero_row_offset = 3 * word_soa_bytes;
public:
    static constexpr uint32_t ub_bytes = Engine::arena_bytes + extra_bytes;
private:
    static_assert(tile_size >= 32 && tile_size <= 64 && tile_size % 16 == 0,
                  "G1 row masks require 32, 48 or 64 lanes");
    static_assert(Packed || tile_size != 48 || extra_bytes == 93696);
    static_assert(!Packed || tile_size != 64 || extra_bytes == 100352);
    static_assert(Packed || tile_size != 48 || ub_bytes + 8192 == 190400);
    static_assert(!Packed || tile_size != 64 || ub_bytes + 8192 == 186112);
    static_assert(ub_bytes + 8192 <= detail::msm_required_ub_bytes);
    static_assert(zero_row_offset + tile_size * 4 <= Engine::offsets);
    static_assert((raw_offset | metadata_offset | packed_2p_offset | extra_bytes) % 32 == 0);

    __aicore__ inline C slot(uint32_t index) {
        if constexpr (Packed) {
            return extra_memory_[index * field_bytes].template ReinterpretCast<Coordinate>();
        } else {
            return index < 6 ? fq_.bank(index) :
                extra_memory_[(index - 6) * field_bytes].template ReinterpretCast<Coordinate>();
        }
    }
    template<class T>
    __aicore__ inline AscendC::LocalTensor<T> digit(
        AscendC::LocalTensor<T> field, uint32_t index) {
        return field[index * tile_size];
    }
    __aicore__ inline Bytes raw() { return extra_memory_[raw_offset]; }
    __aicore__ inline Bytes meta() { return extra_memory_[metadata_offset]; }
    __aicore__ inline C cached_mont_one() {
        return extra_memory_[mont_one_offset].template ReinterpretCast<Coordinate>();
    }
    __aicore__ inline I32 cached_packed_2p() {
        return extra_memory_[packed_2p_offset].template ReinterpretCast<int32_t>();
    }
    __aicore__ inline U32 load_indices(bool affine) {
        return extra_memory_[affine ? affine_indices_offset : jacobian_indices_offset]
            .template ReinterpretCast<uint32_t>();
    }
    // Every helper returns with the normal I32 lane mask, including I16 rows.
    __aicore__ inline void restore_i32_mask() {
        if constexpr (Packed)
            AscendC::SetVectorMask<int32_t, AscendC::MaskMode::NORMAL>(int32_t(tile_size));
    }
    // Interleave borrows this engine table, then restores these reusable indices.
    // No coordinate values are read by Scalar and no additional UB is needed.
    __aicore__ inline void prepare_widen_indices() {
        static_assert(Packed, "only packed coordinates need widening indices");
        fq_.template sync<AscendC::HardEvent::V_S>();
        auto indices = fq_.u32(Engine::offsets);
        for (uint32_t lane = 0; lane < tile_size; ++lane)
            indices.SetValue(lane, (lane / 2) * sizeof(uint32_t));
        fq_.template sync<AscendC::HardEvent::S_V>();
        for (uint32_t row = 1; row < 16; ++row)
            fq_.adds(fq_.i32(Engine::offsets + row * tile_size * 4),
                     fq_.i32(Engine::offsets), int32_t(row * tile_size * 2));
    }
    __aicore__ inline void widen_rows(
        I32 dst, AscendC::LocalTensor<int16_t> src, uint8_t rows) {
        static_assert(Packed, "integer widening belongs to the packed policy");
        // DAV2201 cannot Cast I16 to I32. Gather each aligned packed pair twice;
        // odd lanes then take its high halfword and even lanes its low halfword.
        constexpr uint32_t chunk = 16 * tile_size;
        const uint32_t count = uint32_t(rows) * tile_size;
        auto output = dst.template ReinterpretCast<uint32_t>();
        auto input = src.template ReinterpretCast<uint32_t>();
        for (uint32_t first = 0; first < count; first += chunk) {
            const uint32_t active = count - first < chunk ? count - first : chunk;
            fq_.gather(output[first], input[first / 2],
                       fq_.u32(Engine::offsets), active);
        }
        uint64_t odd_mask[2] = {
            UINT64_C(0xaaaaaaaaaaaaaaaa) >> (64 - tile_size), 0};
        AscendC::ShiftRight(output, output, uint32_t(16), odd_mask, rows,
                            AscendC::UnaryRepeatParams{1, 1, i32_stride, i32_stride});
        fq_.vector_barrier();
        // Digits are in [0,4095]. Mask both U16 halves: low &=4095, high &=0.
        AscendC::And(dst.template ReinterpretCast<uint16_t>(),
                    dst.template ReinterpretCast<uint16_t>(),
                    fq_.tmp(15).template ReinterpretCast<uint16_t>(),
                    uint64_t(2 * tile_size), rows,
                    AscendC::BinaryRepeatParams{1, 1, 1, i32_stride, i32_stride, 0});
        fq_.vector_barrier();
        restore_i32_mask();
    }
    template<class D, class S>
    __aicore__ inline void cast_rows(
        AscendC::LocalTensor<D> dst, AscendC::LocalTensor<S> src, uint8_t rows) {
        // Integer-only exact bridge. All source/destination buffers are disjoint.
        if constexpr (sizeof(D) == sizeof(int32_t) && sizeof(S) == sizeof(int16_t)) {
            widen_rows(dst, src, rows);
        } else {
            static_assert(sizeof(D) == sizeof(int16_t) && sizeof(S) == sizeof(int32_t));
            AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE,
                         uint64_t(tile_size), rows,
                         AscendC::UnaryRepeatParams{
                             1, 1, uint8_t(tile_size * sizeof(D) / 32),
                             uint8_t(tile_size * sizeof(S) / 32)});
            fq_.vector_barrier();
            restore_i32_mask();
        }
    }
    __aicore__ inline void fill_coordinate(C dst, int32_t value, uint8_t rows = 1) {
        AscendC::Duplicate(dst, Coordinate(value), uint64_t(tile_size), rows,
                           1, coordinate_stride);
        fq_.vector_barrier();
        restore_i32_mask();
    }
    __aicore__ inline void fill_field(C dst, int32_t value) {
        fill_coordinate(dst, value, uint8_t(digits));
    }
    __aicore__ inline void copy_field(C dst, C src) {
        AscendC::Adds(dst, src, Coordinate(0), uint64_t(tile_size), uint8_t(digits),
                      AscendC::UnaryRepeatParams{
                          1, 1, coordinate_stride, coordinate_stride});
        fq_.vector_barrier();
        restore_i32_mask();
    }
    __aicore__ inline void add_coordinate_rows(C dst, C a, C b) {
        AscendC::Add(dst, a, b, uint64_t(tile_size), uint8_t(digits),
                     AscendC::BinaryRepeatParams{
                         1, 1, 1, coordinate_stride, coordinate_stride, coordinate_stride});
        fq_.vector_barrier();
        restore_i32_mask();
    }
    __aicore__ inline void subtract_coordinate_rows(C dst, C a, C b) {
        AscendC::Sub(dst, a, b, uint64_t(tile_size), uint8_t(digits),
                     AscendC::BinaryRepeatParams{
                         1, 1, 1, coordinate_stride, coordinate_stride, coordinate_stride});
        fq_.vector_barrier();
        restore_i32_mask();
    }
    __aicore__ inline void multiply_coordinate_rows(C dst, C a, C mask) {
        AscendC::Mul(dst, a, mask, uint64_t(tile_size), uint8_t(digits),
                     AscendC::BinaryRepeatParams{
                         1, 1, 1, coordinate_stride, coordinate_stride, 0});
        fq_.vector_barrier();
        restore_i32_mask();
    }
    __aicore__ inline C coordinate_mask(I32 mask) {
        if constexpr (Packed) {
            auto converted = fq_.tmp(0).template ReinterpretCast<Coordinate>();
            cast_rows(converted, mask, 1);
            return converted;
        } else {
            return mask;
        }
    }
    __aicore__ inline void mask_field(C dst, C src, I32 mask) {
        multiply_coordinate_rows(dst, src, coordinate_mask(mask));
    }
    __aicore__ inline void mont_one(C dst) {
        copy_field(dst, cached_mont_one());
    }
    __aicore__ inline void canonical_one(C dst) {
        fill_field(dst, 0);
        fill_coordinate(digit(dst, 0), 1);
    }
    // Both inputs are captured before any destination write, including aliases.
    // Outside the engine every coordinate digit remains normalized to 0..4095.
    __aicore__ inline void add(C dst, C a, C b) {
        if constexpr (Packed) {
            cast_rows(fq_.template bank<0>(), a, uint8_t(digits));
            cast_rows(fq_.template bank<1>(), b, uint8_t(digits));
            fq_.add_mod_redundant(fq_.template bank<2>(), fq_.template bank<0>(),
                                  fq_.template bank<1>(), cached_packed_2p());
            cast_rows(dst, fq_.template bank<2>(), uint8_t(digits));
        } else {
            fq_.add_mod_redundant(dst, a, b, cached_packed_2p());
        }
    }
    __aicore__ inline void sub(C dst, C a, C b) {
        if constexpr (Packed) {
            cast_rows(fq_.template bank<0>(), a, uint8_t(digits));
            cast_rows(fq_.template bank<1>(), b, uint8_t(digits));
            fq_.sub_mod_redundant(fq_.template bank<2>(), fq_.template bank<0>(),
                                  fq_.template bank<1>(), cached_packed_2p());
            cast_rows(dst, fq_.template bank<2>(), uint8_t(digits));
        } else {
            fq_.sub_mod_redundant(dst, a, b, cached_packed_2p());
        }
    }
    template<bool Canonical>
    __aicore__ inline void multiply_engine(I32 dst, I32 a, I32 b) {
        if constexpr (Canonical) fq_.montgomery(dst, a, b);
        else fq_.montgomery_redundant(dst, a, b);
    }
    template<bool Canonical = false>
    __aicore__ inline void mul(C dst, C a, C b) {
        if constexpr (Packed) {
            cast_rows(fq_.template bank<0>(), a, uint8_t(digits));
            cast_rows(fq_.template bank<1>(), b, uint8_t(digits));
            multiply_engine<Canonical>(fq_.template bank<2>(),
                                       fq_.template bank<0>(), fq_.template bank<1>());
            cast_rows(dst, fq_.template bank<2>(), uint8_t(digits));
        } else {
            multiply_engine<Canonical>(dst, a, b);
        }
    }
    __aicore__ inline void square(C dst, C a) {
        if constexpr (Packed) {
            cast_rows(fq_.template bank<0>(), a, uint8_t(digits));
            fq_.montgomery_square_redundant(fq_.template bank<2>(),
                                            fq_.template bank<0>());
            cast_rows(dst, fq_.template bank<2>(), uint8_t(digits));
        } else {
            fq_.montgomery_square_redundant(dst, a);
        }
    }

    // Exact row-sum tree, also safe with field==workspace. Its first stage
    // consumes both halves before the next stage; an odd middle row survives.
    __aicore__ inline void sum_digit_rows(I32 workspace, I32 field) {
        if constexpr (digits == 1) {
            fq_.copy(workspace, field);
        } else {
            uint32_t half = (digits + 1) / 2;
            AscendC::Add(workspace, field, digit(field, half), uint64_t(tile_size),
                          uint8_t(digits - half), AscendC::BinaryRepeatParams{
                              1, 1, 1, i32_stride, i32_stride, i32_stride});
            fq_.vector_barrier();
            if constexpr (digits % 2)
                fq_.adds(digit(workspace, half - 1), digit(field, half - 1), 0);
            for (uint32_t active = half; active > 1; active = half) {
                half = (active + 1) / 2;
                AscendC::Add(workspace, workspace, digit(workspace, half),
                              uint64_t(tile_size), uint8_t(active - half),
                              AscendC::BinaryRepeatParams{
                                  1, 1, 1, i32_stride, i32_stride, i32_stride});
                fq_.vector_barrier();
                // An odd unpaired row is already in place outside the writes.
            }
        }
    }
    // Zero modulo p has exactly two representatives in [0,2p): 0 and p.
    // Engine bank6 is disjoint from every G1 slot0..16. tmp12 holds the first
    // mask; tmp6..11/13..15 are preserved unless they are the requested dst.
    // dst is a control row, disjoint from bank6, and field is a G1 slot.
    __aicore__ inline void nonzero(I32 dst, C field) {
        auto workspace = fq_.template bank<6>();
        I32 wide;
        if constexpr (Packed) {
            wide = fq_.template bank<0>();
            cast_rows(wide, field, uint8_t(digits));
        } else {
            wide = field;
        }
        sum_digit_rows(workspace, wide);
        // Normalized digits have sum <=32*4095<2^20.
        fq_.adds(fq_.tmp(12), workspace, 1048575);
        fq_.shift_right(fq_.tmp(12), fq_.tmp(12), 20);
        fq_.subtract_rows(workspace, wide, fq_.template bank<7>());
        fq_.multiply_rows(workspace, workspace, workspace, false);
        // Each difference is in [-4095,4095]; 32*4095^2 =
        // 536608800 <2^29. Every tree sum and the final bias stay int32-safe.
        sum_digit_rows(workspace, workspace);
        fq_.adds(workspace, workspace, 536870911);
        fq_.shift_right(workspace, workspace, 29);
        fq_.mul(dst, fq_.tmp(12), workspace);
    }
    __aicore__ inline void complement(I32 mask) {
        fq_.muls(mask, mask, -1);
        fq_.adds(mask, mask, 1);
    }
    __aicore__ inline void zero(I32 dst, C field) {
        nonzero(dst, field);
        complement(dst);
    }
    // Exact tile vote on finalized 0/1 masks, not on Fq coordinate words.
    // Eight aggregates cover every lane, each sum in [0,tile_size/8].
    // This coordinate-derived branch is data-dependent, not constant-time.
    __aicore__ inline bool any_same_vote(I32 same) {
        auto aggregate = fq_.tmp(11);
        const AscendC::BinaryRepeatParams strides{
            1, 1, 1, i32_stride, i32_stride, i32_stride};
        AscendC::Add<int32_t, true>(
            aggregate, same, same[16], uint64_t(16), uint8_t(1), strides);
        fq_.vector_barrier();
        for (uint32_t first = 32; first < tile_size; first += 16) {
            AscendC::Add<int32_t, true>(
                aggregate, aggregate, same[first], uint64_t(16), uint8_t(1), strides);
            fq_.vector_barrier();
        }
        AscendC::Add<int32_t, true>(
            aggregate, aggregate, aggregate[8], uint64_t(8), uint8_t(1), strides);
        fq_.vector_barrier();
        fq_.template sync<AscendC::HardEvent::V_S>();
        uint32_t control = 0;
        for (uint32_t i = 0; i < 8; ++i)
            control |= static_cast<uint32_t>(aggregate.GetValue(i));
        // Complete Scalar control reads before any subsequent Vector reuse.
        fq_.template sync<AscendC::HardEvent::S_V>();
        return control != 0;
    }
    __aicore__ inline void select_field(C dst, C yes, I32 mask) {
        // I16 differences stay within [-4095,4095]; the multiplier is 0/1.
        // Scratch is disjoint from all coordinates and live I32 control rows.
        auto difference = fq_.template bank<6>().template ReinterpretCast<Coordinate>();
        subtract_coordinate_rows(difference, yes, dst);
        multiply_coordinate_rows(difference, difference, coordinate_mask(mask));
        add_coordinate_rows(dst, dst, difference);
    }
    __aicore__ inline void select_point(Point dst, Point yes, I32 mask) {
        for (uint32_t c = 0; c < 3; ++c) select_field(slot(dst + c), slot(yes + c), mask);
    }
    __aicore__ inline void normalize_identity(Point p) {
        nonzero(fq_.tmp(13), slot(p + 2));
        auto finite = coordinate_mask(fq_.tmp(13));
        multiply_coordinate_rows(slot(p), slot(p), finite);
        multiply_coordinate_rows(slot(p + 2), slot(p + 2), finite);
        // Cached R is immutable and disjoint from points and exception masks.
        auto one = cached_mont_one();
        subtract_coordinate_rows(slot(p + 1), slot(p + 1), one);
        multiply_coordinate_rows(slot(p + 1), slot(p + 1), finite);
        add_coordinate_rows(slot(p + 1), slot(p + 1), one);
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
        // Use Z1*Z2*H directly. The former (4X,8Y,2Z) tuple represented
        // the same point; avoiding that projective rescaling saves four
        // modular doublings. Equal/opposite/identity selections are unchanged.
        if constexpr (affine_rhs) {
            mul(slot(lhs + 2), slot(13), h);
        } else {
            mul(slot(lhs + 2), slot(13), slot(rhs + 2));
            mul(slot(lhs + 2), slot(lhs + 2), h);
        }
        square(zz2, h); // H^2; the old ZZ2 is now dead.
        mul(h, h, zz2); // J.
        mul(u1, u1, zz2); // V.
        square(slot(lhs), zz1);
        sub(slot(lhs), slot(lhs), h);
        add(zz2, u1, u1);
        sub(slot(lhs), slot(lhs), zz2);
        sub(u1, u1, slot(lhs));
        mul(slot(lhs + 1), zz1, u1);
        mul(s1, s1, h);
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
            mask_field(slot(lhs + 2), slot(lhs + 2), fq_.tmp(10));
        }
        select_point(lhs, rhs, a_zero);
        normalize_identity(lhs);
    }

    template<bool montgomery, bool affine>
    HP_G1_VECTOR_HEAVY __aicore__ inline void load(
        Point dst, GM_ADDR src, const uint64_t* byte_offsets, uint32_t valid) {
        constexpr uint32_t stride = affine ? affine_bytes : jacobian_bytes;
        AscendC::Duplicate(raw().template ReinterpretCast<int32_t>(), int32_t(0), raw_bytes / 4);
        fq_.vector_barrier();
        fq_.template sync<AscendC::HardEvent::V_MTE2>();
        for (uint32_t lane = 0; lane < valid; ++lane) {
            if (byte_offsets[lane] != absent)
                fq_.copy_in(raw()[lane * stride], src + byte_offsets[lane], stride);
        }
        fq_.template sync<AscendC::HardEvent::MTE2_V>();
        if constexpr (!montgomery) {
            for (uint32_t j = 0; j < digits; ++j)
                fill_coordinate(digit(slot(6), j), Params::r2_digits[j]);
        }
        constexpr uint32_t coordinates = affine ? 2 : 3;
        for (uint32_t c = 0; c < coordinates; ++c) {
            AscendC::Gather(fq_.u32(Engine::soa_lhs),
                            raw().template ReinterpretCast<uint32_t>(), load_indices(affine),
                            uint32_t(c * Fq::word_count * 4), Fq::word_count * tile_size);
            fq_.vector_barrier();
            if constexpr (Packed) {
                fq_.unpack(fq_.template bank<0>(), fq_.u32(Engine::soa_lhs));
                cast_rows(slot(dst + c), fq_.template bank<0>(), uint8_t(digits));
            } else {
                fq_.unpack(slot(dst + c), fq_.u32(Engine::soa_lhs));
            }
            if constexpr (!montgomery)
                mul(slot(dst + c), slot(dst + c), slot(6));
        }
        if constexpr (affine) {
            nonzero(fq_.tmp(6), slot(dst));
            nonzero(fq_.tmp(7), slot(dst + 1));
            fq_.mul(fq_.tmp(0), fq_.tmp(6), fq_.tmp(7));
            fq_.add(fq_.tmp(6), fq_.tmp(6), fq_.tmp(7));
            fq_.sub(fq_.tmp(6), fq_.tmp(6), fq_.tmp(0));
            mont_one(slot(dst + 2));
            mask_field(slot(dst + 2), slot(dst + 2), fq_.tmp(6));
        }
        normalize_identity(dst);
    }

    template<bool montgomery, bool affine>
    HP_G1_VECTOR_HEAVY __aicore__ inline void store(
        Point src, GM_ADDR dst, const uint64_t* byte_offsets, uint32_t valid) {
        if constexpr (!montgomery) canonical_one(slot(6));
        if constexpr (affine) nonzero(fq_.tmp(13), slot(src + 2));
        constexpr uint32_t coordinates = affine ? 2 : 3;
        for (uint32_t c = 0; c < coordinates; ++c) {
            auto field = slot(src + c);
            if constexpr (!montgomery) {
                // Canonical external Jacobian/affine output boundary.
                mul<true>(slot(7), field, slot(6));
                field = slot(7);
            } else if constexpr (affine) {
                if (c == 1) {
                    // Every identity has X=0, Y=R, Z=0. Mask only Y in
                    // private slot7; never modify the caller's source point.
                    mask_field(slot(7), field, fq_.tmp(13));
                    field = slot(7);
                }
            }
            // pack_words writes only its output and engine tmp10. Neither
            // aliases any public coordinate or the live tmp13 finite mask.
            if constexpr (Packed) {
                cast_rows(fq_.template bank<0>(), field, uint8_t(digits));
                fq_.pack_words(fq_.i32(c * word_soa_bytes), fq_.template bank<0>());
            } else {
                fq_.pack_words(fq_.i32(c * word_soa_bytes), field);
            }
        }
        interleave(affine);
        fq_.template sync<AscendC::HardEvent::V_MTE3>();
        constexpr uint32_t stride = affine ? affine_bytes : jacobian_bytes;
        for (uint32_t lane = 0; lane < valid; ++lane) {
            if (byte_offsets[lane] != absent)
                fq_.copy_out(raw()[lane * stride], dst + byte_offsets[lane], stride);
        }
        fq_.template sync<AscendC::HardEvent::MTE3_V>();
    }

    // Three packed word-SoA arrays reside only in FieldVector's raw region,
    // never in live coordinate banks. Gather produces whole aligned records.
    // 1152-entry chunks fit both policies' offset tables, with aligned starts.
    HP_G1_VECTOR_HEAVY __aicore__ inline void interleave(bool affine) {
        fq_.fill(fq_.i32(zero_row_offset), 0);
        const uint32_t record_words = affine ? affine_bytes / 4 : jacobian_bytes / 4;
        const uint32_t total = tile_size * record_words;
        constexpr uint32_t chunk = 1152;
        for (uint32_t first = 0; first < total; first += chunk) {
            const uint32_t count = total - first < chunk ? total - first : chunk;
            fq_.template sync<AscendC::HardEvent::V_S>();
            auto indices = fq_.u32(Engine::offsets);
            for (uint32_t i = 0; i < count; ++i) {
                const uint32_t linear = first + i;
                const uint32_t lane = linear / record_words;
                const uint32_t w = linear % record_words;
                const uint32_t offset = w < 36 ?
                    (w / 12) * word_soa_bytes + (w % 12) * tile_size * 4 + lane * 4 :
                    zero_row_offset + lane * 4;
                indices.SetValue(i, offset);
            }
            fq_.template sync<AscendC::HardEvent::S_V>();
            fq_.gather(raw()[first * 4].template ReinterpretCast<uint32_t>(),
                       fq_.u32(0), indices, count);
        }
        if constexpr (Packed) prepare_widen_indices();
    }

    __aicore__ inline void clear_meta() {
        // All previous Scalar metadata reads finish before Vector reuses it.
        fq_.template sync<AscendC::HardEvent::S_V>();
        AscendC::Duplicate(meta().template ReinterpretCast<int32_t>(), int32_t(0), metadata_bytes / 4);
        fq_.vector_barrier();
        fq_.template sync<AscendC::HardEvent::V_MTE2>();
    }

    Engine fq_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> extra_;
    Bytes extra_memory_;
};

using G1Vector = G1VectorEngine<>;

} // namespace hyperplonk::ascend910b3

#undef HP_G1_VECTOR_HEAVY
