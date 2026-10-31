#pragma once

#include "kernel_operator.h"
#include "field_params.h"

#if defined(__NPU_DEVICE__)
#define HP_FIELD_VECTOR_HEAVY __attribute__((noinline))
#else
#define HP_FIELD_VECTOR_HEAVY
#endif

namespace hyperplonk::ascend910b3 {

template<uint32_t Lanes, bool Packed> class G1VectorEngine;

// UB-only Montgomery representation; the public Field representation is unchanged.
// Each numerical row contains Lanes independent int32 lanes. Scalar loops address
// only public digits, exponent bits, layout indices, and tiles, never field values.
template<class Field, uint32_t Lanes = 48, bool Compact = false> class FieldVector {
    template<uint32_t, bool> friend class G1VectorEngine;
    friend struct SquareProbeAccess;
    using Params = FieldVectorParams<typename Field::Parameters>;
    using I32 = AscendC::LocalTensor<int32_t>;
    using U32 = AscendC::LocalTensor<uint32_t>;
    using Bytes = AscendC::LocalTensor<uint8_t>;
    static constexpr uint32_t lanes = Lanes;
    static constexpr uint32_t digits = Params::digit_count;
    static constexpr uint32_t words = Field::word_count;
    static constexpr uint32_t wide_words = 2 * words + 1;
    static constexpr uint32_t radix_bits = 12;
    static constexpr int32_t radix = 1 << radix_bits;
    static constexpr uint32_t vector_bytes = lanes * sizeof(int32_t);
    static constexpr uint8_t row_stride = vector_bytes / 32;
    static constexpr uint8_t pair_stride = 2 * row_stride;
    static constexpr uint32_t bank_bytes = 32 * vector_bytes;

    // Full mode retains the original capacities and byte layout at 48 lanes.
    // Compact G1 stages process one coordinate at a time: their three raw word
    // banks are disjoint, but all SoA views reuse raw_lhs. The following row is
    // a zero-row workspace, NOT an inverse mask. The index table follows it.
    static constexpr uint32_t raw_lhs = 0;
    static constexpr uint32_t raw_rhs = raw_lhs + (Compact ? 12 : 25) * vector_bytes;
    static constexpr uint32_t raw_out = raw_rhs + 12 * vector_bytes;
    static constexpr uint32_t raw_mask = raw_out + 12 * vector_bytes;
    static constexpr uint32_t soa_lhs = Compact ? raw_lhs : raw_mask + vector_bytes;
    static constexpr uint32_t soa_rhs = Compact ? raw_lhs : soa_lhs + 25 * vector_bytes;
    static constexpr uint32_t soa_out = Compact ? raw_lhs : soa_rhs + 12 * vector_bytes;
    static constexpr uint32_t offsets =
        Compact ? raw_mask + vector_bytes : soa_out + 12 * vector_bytes;
    static constexpr uint32_t banks = offsets + 25 * vector_bytes;
    static constexpr uint32_t accumulator = banks + (Compact ? 5 : 8) * bank_bytes;
    static constexpr uint32_t temporaries = accumulator + 65 * vector_bytes;
    static constexpr uint32_t used_bytes = temporaries + 16 * vector_bytes;
    static_assert(lanes > 0 && lanes <= 64 && lanes % 8 == 0,
                  "Vector rows require 1..64 lanes and whole 32-byte blocks");
    static_assert(digits <= 32 && words <= 12 && wide_words <= 25,
                  "field exceeds the Vector arena layout");
    static_assert((raw_lhs | raw_rhs | raw_out | raw_mask | soa_lhs |
                   soa_rhs | soa_out | offsets | banks | accumulator |
                   temporaries | used_bytes) % 32 == 0, "unaligned Vector arena");
    static_assert(used_bytes == (Compact ? 303 : 461) * vector_bytes,
                  "Vector arena exceeds its reserved UB budget");
    static_assert(Compact || lanes != 48 || used_bytes == 88512,
                  "default Vector arena layout changed");
    static_assert(!Compact || lanes != 64 || used_bytes == 77568,
                  "packed G1 Vector arena layout changed");

public:
    static constexpr uint32_t arena_bytes = used_bytes;
    static constexpr uint32_t tile_size = lanes;

    __aicore__ inline FieldVector() {
        pipe_.InitBuffer(arena_, arena_bytes);
        memory_ = arena_.template Get<uint8_t>();
        // The high halfword of each int32 mask is zero. And/Or below always
        // operate explicitly on all 2*lanes halfwords, not the broken i32 count path.
        fill(tmp(15), radix - 1);
        for (uint32_t j = 0; j < digits; ++j)
            fill(digit(bank<7>(), j), Params::modulus_digits[j]);
    }

    // op: add, sub, neg, mul, square, inverse, wide (0..6).
    // Nonempty calls require canonical, valid input and disjoint output buffers.
    // lhs/rhs may alias each other. No GM scratch or resource ownership is added.
    __aicore__ inline void run(GM_ADDR lhs, GM_ADDR rhs, GM_ADDR out,
                               GM_ADDR mask, uint64_t count, uint32_t op) {
        static_assert(!Compact, "Compact G1 engine has no public field dispatcher");
        if (count == 0) return;
        const uint32_t input_words = op == 6 ? wide_words : words;
        const uint64_t stride = uint64_t(AscendC::GetBlockNum()) * lanes;
        for (uint64_t offset = uint64_t(AscendC::GetBlockIdx()) * lanes;
             offset < count; offset += stride) {
            const uint32_t valid = uint32_t(count - offset < lanes ? count - offset : lanes);
            // Padding lanes must be zero before an exact-byte DMA tail; MTE2
            // must not race the preceding Vector clear of the input workspace.
            AscendC::Duplicate(i32(raw_lhs), int32_t(0), 25 * lanes);
            AscendC::Duplicate(i32(raw_rhs), int32_t(0), 12 * lanes);
            vector_barrier();
            sync<AscendC::HardEvent::V_MTE2>();
            copy_in(bytes(raw_lhs), lhs + offset * input_words * 4,
                    valid * input_words * 4);
            const bool binary = op == 0 || op == 1 || op == 3;
            if (binary)
                copy_in(bytes(raw_rhs), rhs + offset * words * 4, valid * words * 4);
            sync<AscendC::HardEvent::MTE2_V>();
            forward_offsets(input_words);
            gather(u32(soa_lhs), u32(raw_lhs), u32(offsets), input_words * lanes);
            if (op == 6) {
                reduce_wide(bank<2>(), u32(soa_lhs));
            } else {
                unpack(bank<0>(), u32(soa_lhs));
                if (binary) {
                    gather(u32(soa_rhs), u32(raw_rhs), u32(offsets), words * lanes);
                    unpack(bank<1>(), u32(soa_rhs));
                }
                switch (op) {
                case 0: add_mod(bank<2>(), bank<0>(), bank<1>()); break;
                case 1: sub_mod(bank<2>(), bank<0>(), bank<1>()); break;
                case 2:
                    fill_bank(bank<1>(), 0);
                    sub_mod(bank<2>(), bank<1>(), bank<0>());
                    break;
                case 3:
                    load_r2();
                    montgomery(bank<4>(), bank<0>(), bank<3>());
                    montgomery(bank<2>(), bank<4>(), bank<1>());
                    break;
                case 4:
                    load_r2();
                    montgomery_square(bank<4>(), bank<0>());
                    montgomery(bank<2>(), bank<4>(), bank<3>());
                    break;
                case 5: inverse(bank<2>(), bank<0>()); break;
                }
            }
            pack(bank<2>());
            if (op == 5) pack_mask(bank<0>());
            sync<AscendC::HardEvent::V_MTE3>();
            copy_out(bytes(raw_out), out + offset * words * 4, valid * words * 4);
            if (op == 5) copy_out(bytes(raw_mask), mask + offset, valid);
            // All stores finish before this single buffer's next use, including
            // a subsequent grid-stride tile on the same physical AIV.
            sync<AscendC::HardEvent::MTE3_V>();
        }
    }

private:
    __aicore__ inline Bytes bytes(uint32_t offset) { return memory_[offset]; }
    __aicore__ inline I32 i32(uint32_t offset) {
        return bytes(offset).template ReinterpretCast<int32_t>();
    }
    __aicore__ inline U32 u32(uint32_t offset) {
        return bytes(offset).template ReinterpretCast<uint32_t>();
    }
    // Runtime bank selection is needed by the default G1 coordinate policy.
    // Compact access is compile-time checked: logical 3/4/5 do not exist, while
    // product scratch 6 and immutable modulus 7 occupy physical banks 3 and 4.
    __aicore__ inline I32 bank(uint32_t index) {
        static_assert(!Compact, "Compact bank access requires bank<Index>()");
        return i32(banks + index * bank_bytes);
    }
    template<uint32_t Index> __aicore__ inline I32 bank() {
        static_assert(Index < 8 && (!Compact || Index < 3 || Index >= 6),
                      "bank is unavailable in this Vector arena");
        constexpr uint32_t physical = Compact && Index >= 6 ? Index - 3 : Index;
        return i32(banks + physical * bank_bytes);
    }
    __aicore__ inline I32 tmp(uint32_t index) { return i32(temporaries + index * vector_bytes); }
    __aicore__ inline I32 acc() { return i32(accumulator); }
    __aicore__ inline I32 digit(I32 tensor, uint32_t index) { return tensor[index * lanes]; }
    __aicore__ inline U32 as_unsigned(I32 tensor) {
        return tensor.template ReinterpretCast<uint32_t>();
    }
    __aicore__ inline void vector_barrier() { AscendC::PipeBarrier<PIPE_V>(); }
    template<AscendC::HardEvent event> __aicore__ inline void sync() {
        const AscendC::TEventID id = pipe_.FetchEventID(event);
        AscendC::SetFlag<event>(id);
        AscendC::WaitFlag<event>(id);
    }
    __aicore__ inline void fill(I32 dst, int32_t value) {
        AscendC::Duplicate(dst, value, uint64_t(lanes), uint8_t(1), 1, row_stride);
        vector_barrier();
    }
    __aicore__ inline void copy(I32 dst, I32 src) {
        AscendC::Adds(dst, src, int32_t(0), uint64_t(lanes), uint8_t(1),
                      AscendC::UnaryRepeatParams{1, 1, row_stride, row_stride});
        vector_barrier();
    }
    __aicore__ inline void add(I32 dst, I32 a, I32 b) {
        AscendC::Add(dst, a, b, uint64_t(lanes), uint8_t(1),
                     AscendC::BinaryRepeatParams{1, 1, 1, row_stride, row_stride, row_stride});
        vector_barrier();
    }
    __aicore__ inline void sub(I32 dst, I32 a, I32 b) {
        AscendC::Sub(dst, a, b, uint64_t(lanes), uint8_t(1),
                     AscendC::BinaryRepeatParams{1, 1, 1, row_stride, row_stride, row_stride});
        vector_barrier();
    }
    __aicore__ inline void mul(I32 dst, I32 a, I32 b) {
        AscendC::Mul(dst, a, b, uint64_t(lanes), uint8_t(1),
                     AscendC::BinaryRepeatParams{1, 1, 1, row_stride, row_stride, row_stride});
        vector_barrier();
    }
    // Each digit row occupies row_stride 32-byte blocks. Explicit masks/repeats
    // also broadcast the same lane coefficient row with src1RepStride zero.
    __aicore__ inline void multiply_rows(I32 dst, I32 a, I32 b, bool broadcast,
                                        uint8_t repeats = uint8_t(digits)) {
        const AscendC::BinaryRepeatParams strides{
            1, 1, 1, row_stride, row_stride, static_cast<uint8_t>(broadcast ? 0 : row_stride)};
        AscendC::Mul(dst, a, b, uint64_t(lanes), repeats, strides);
        vector_barrier();
    }
    __aicore__ inline void add_rows(I32 dst, I32 a, I32 b,
                                   uint8_t repeats = uint8_t(digits)) {
        AscendC::Add(dst, a, b, uint64_t(lanes), repeats,
                     AscendC::BinaryRepeatParams{1, 1, 1, row_stride, row_stride, row_stride});
        vector_barrier();
    }
    __aicore__ inline void subtract_rows(I32 dst, I32 a, I32 b) {
        AscendC::Sub(dst, a, b, uint64_t(lanes), uint8_t(digits),
                     AscendC::BinaryRepeatParams{1, 1, 1, row_stride, row_stride, row_stride});
        vector_barrier();
    }
    __aicore__ inline void adds(I32 dst, I32 src, int32_t value) {
        AscendC::Adds(dst, src, value, uint64_t(lanes), uint8_t(1),
                      AscendC::UnaryRepeatParams{1, 1, row_stride, row_stride});
        vector_barrier();
    }
    __aicore__ inline void muls(I32 dst, I32 src, int32_t value) {
        AscendC::Muls(dst, src, value, uint64_t(lanes), uint8_t(1),
                      AscendC::UnaryRepeatParams{1, 1, row_stride, row_stride});
        vector_barrier();
    }
    __aicore__ inline void shift_right(I32 dst, I32 src, uint32_t bits) {
        AscendC::ShiftRight(as_unsigned(dst), as_unsigned(src), bits,
                            uint64_t(lanes), uint8_t(1),
                            AscendC::UnaryRepeatParams{1, 1, row_stride, row_stride});
        vector_barrier();
    }
    __aicore__ inline void shift_left(I32 dst, I32 src, uint32_t bits) {
        AscendC::ShiftLeft(as_unsigned(dst), as_unsigned(src), bits,
                           uint64_t(lanes), uint8_t(1),
                           AscendC::UnaryRepeatParams{1, 1, row_stride, row_stride});
        vector_barrier();
    }
    __aicore__ inline void bit_and(I32 dst, I32 a, I32 b) {
        AscendC::And(dst.template ReinterpretCast<uint16_t>(),
                    a.template ReinterpretCast<uint16_t>(),
                    b.template ReinterpretCast<uint16_t>(),
                    uint64_t(2 * lanes), uint8_t(1),
                    AscendC::BinaryRepeatParams{1, 1, 1, row_stride, row_stride, row_stride});
        vector_barrier();
    }
    __aicore__ inline void bit_or(I32 dst, I32 a, I32 b) {
        AscendC::Or(dst.template ReinterpretCast<uint16_t>(),
                   a.template ReinterpretCast<uint16_t>(),
                   b.template ReinterpretCast<uint16_t>(),
                   uint64_t(2 * lanes), uint8_t(1),
                   AscendC::BinaryRepeatParams{1, 1, 1, row_stride, row_stride, row_stride});
        vector_barrier();
    }
    __aicore__ inline void fill_bank(I32 dst, int32_t value) {
        AscendC::Duplicate(dst, value, int32_t(digits * lanes));
        vector_barrier();
    }
    __aicore__ inline void load_r2() {
        static_assert(!Compact, "Compact G1 engine has no R2 working bank");
        for (uint32_t j = 0; j < digits; ++j)
            fill(digit(bank<3>(), j), Params::r2_digits[j]);
    }
    __aicore__ inline void copy_in(Bytes dst, GM_ADDR src, uint32_t size) {
        AscendC::GlobalTensor<uint8_t> gm;
        gm.SetGlobalBuffer(src);
        AscendC::DataCopyPad(dst, gm, AscendC::DataCopyExtParams{1, size, 0, 0, 0},
                            AscendC::DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
    }
    __aicore__ inline void copy_out(Bytes src, GM_ADDR dst, uint32_t size) {
        AscendC::GlobalTensor<uint8_t> gm;
        gm.SetGlobalBuffer(dst);
        AscendC::DataCopyPad(gm, src, AscendC::DataCopyExtParams{1, size, 0, 0, 0});
    }
    __aicore__ inline void gather(U32 dst, U32 src, U32 indices, uint32_t size) {
        AscendC::Gather(dst, src, indices, uint32_t(0), size);
        vector_barrier();
    }

    // The only SetValue calls write public byte offsets. Fence both directions
    // when reusing this table: the previous Gather must finish before Scalar
    // overwrites it, and Scalar writes must finish before the next Gather.
    HP_FIELD_VECTOR_HEAVY __aicore__ inline void forward_offsets(uint32_t width) {
        sync<AscendC::HardEvent::V_S>();
        auto indices = u32(offsets);
        for (uint32_t w = 0; w < width; ++w)
            for (uint32_t lane = 0; lane < lanes; ++lane)
                indices.SetValue(w * lanes + lane, 4 * (lane * width + w));
        sync<AscendC::HardEvent::S_V>();
    }
    HP_FIELD_VECTOR_HEAVY __aicore__ inline void reverse_offsets() {
        sync<AscendC::HardEvent::V_S>();
        auto indices = u32(offsets);
        for (uint32_t lane = 0; lane < lanes; ++lane)
            for (uint32_t w = 0; w < words; ++w)
                indices.SetValue(lane * words + w, 4 * (w * lanes + lane));
        sync<AscendC::HardEvent::S_V>();
    }
    HP_FIELD_VECTOR_HEAVY __aicore__ inline void unpack(I32 dst, U32 src) {
        auto signed_src = src.template ReinterpretCast<int32_t>();
        for (uint32_t j = 0; j < digits; ++j) {
            const uint32_t bit = radix_bits * j;
            const uint32_t w = bit / 32;
            const uint32_t shift = bit % 32;
            if (w >= words) {
                fill(digit(dst, j), 0);
                continue;
            }
            shift_right(digit(dst, j), digit(signed_src, w), shift);
            if (shift + radix_bits > 32 && w + 1 < words) {
                shift_left(tmp(10), digit(signed_src, w + 1), 32 - shift);
                bit_or(digit(dst, j), digit(dst, j), tmp(10));
            }
            bit_and(digit(dst, j), digit(dst, j), tmp(15));
        }
    }
    HP_FIELD_VECTOR_HEAVY __aicore__ inline void pack_words(I32 dst, I32 src) {
        for (uint32_t w = 0; w < words; ++w) fill(digit(dst, w), 0);
        for (uint32_t j = 0; j < digits; ++j) {
            const uint32_t bit = radix_bits * j;
            const uint32_t w = bit / 32;
            const uint32_t shift = bit % 32;
            if (w >= words) break;
            shift_left(tmp(10), digit(src, j), shift);
            bit_or(digit(dst, w), digit(dst, w), tmp(10));
            if (shift + radix_bits > 32 && w + 1 < words) {
                shift_right(tmp(10), digit(src, j), 32 - shift);
                bit_or(digit(dst, w + 1), digit(dst, w + 1), tmp(10));
            }
        }
    }
    HP_FIELD_VECTOR_HEAVY __aicore__ inline void pack(I32 src) {
        pack_words(i32(soa_out), src);
        reverse_offsets();
        gather(u32(raw_out), u32(soa_out), u32(offsets), words * lanes);
    }

    // s is nonnegative and int32-safe; arithmetic scratch never touches G1 masks.
    __aicore__ inline void normalize(I32 low, I32 carry, I32 s) {
        shift_right(carry, s, radix_bits);
        bit_and(low, s, tmp(15));
    }

    // x has n+1 normalized digits. Include the high digit in the subtraction;
    // dropping it before comparison would corrupt CIOS results near 2p.
    // bank(6) and tmp(0..5) are scratch; output can alias either input bank.
    HP_FIELD_VECTOR_HEAVY __aicore__ inline void subtract_modulus(I32 dst, I32 x) {
        auto difference = bank<6>();
        fill(tmp(0), 0);
        for (uint32_t j = 0; j <= digits; ++j) {
            adds(tmp(1), digit(x, j), radix);
            if (j < digits) sub(tmp(1), tmp(1), digit(bank<7>(), j));
            sub(tmp(1), tmp(1), tmp(0));
            shift_right(tmp(2), tmp(1), radix_bits);
            if (j < digits) bit_and(digit(difference, j), tmp(1), tmp(15));
            muls(tmp(0), tmp(2), -1);
            adds(tmp(0), tmp(0), 1);
        }
        subtract_rows(dst, x, difference);
        multiply_rows(dst, dst, tmp(0), true);
        add_rows(dst, dst, difference);
    }
    static constexpr uint32_t packed_digits = (digits + 1) / 2;
    static constexpr uint32_t packed_bits = 2 * radix_bits;
    static constexpr int32_t packed_radix = radix * radix;
    static_assert(4 * packed_digits <= 64 && 2 * packed_digits <= 32,
                  "packed add/sub exceeds the existing scratch arena");

    // Public constants only: double p in packed radix24, including carry.
    // The same construction serves Fr/Fq; G1 caches only its Fq table in UB.
    struct PackedModulusDigits { int32_t values[packed_digits]{}; };
    static constexpr PackedModulusDigits double_packed_modulus() {
        PackedModulusDigits result{};
        int32_t carry = 0;
        for (uint32_t j = 0; j < packed_digits; ++j) {
            int32_t value = Params::modulus_digits.values[2 * j];
            if (2 * j + 1 < digits)
                value += radix * Params::modulus_digits.values[2 * j + 1];
            value = 2 * value + carry;
            result.values[j] = value & (packed_radix - 1);
            carry = value >> packed_bits;
        }
        return result;
    }
    static constexpr auto packed_double_modulus = double_packed_modulus();
    static_assert(Params::modulus_digits.values[digits - 1] < radix / 4,
                  "redundant Montgomery products require 4p < R");

    // Compact pairs of radix12 rows without lane reads. The final unpaired
    // digit, if any, has a zero high half; no source row beyond digits is read.
    __aicore__ inline void pack24(I32 dst, I32 src) {
        if constexpr (digits / 2 != 0) {
            AscendC::Muls(dst, digit(src, 1), radix, uint64_t(lanes),
                          uint8_t(digits / 2),
                          AscendC::UnaryRepeatParams{1, 1, row_stride, pair_stride});
            vector_barrier();
        }
        if constexpr (digits % 2 != 0) fill(digit(dst, packed_digits - 1), 0);
        AscendC::Add(dst, dst, src, uint64_t(lanes), uint8_t(packed_digits),
                     AscendC::BinaryRepeatParams{1, 1, 1, row_stride, row_stride, pair_stride});
        vector_barrier();
    }
    __aicore__ inline void normalize24(I32 low, I32 carry, I32 s) {
        shift_right(carry, s, packed_bits);
        bit_and(low, s, tmp(5));
    }
    // a,b are normalized packed rows. Signed t lies in [-Q,Q-1], so a
    // logical shift of its uint32 representation extracts exactly borrow=0/1.
    // Leaves the final borrow in tmp0; clobbers tmp1/tmp4, not tmp2/tmp3/tmp5.
    __aicore__ inline void subtract24(I32 dst, I32 a, I32 b) {
        fill(tmp(0), 0);
        for (uint32_t j = 0; j < packed_digits; ++j) {
            sub(tmp(1), digit(a, j), digit(b, j));
            sub(tmp(1), tmp(1), tmp(0));
            shift_right(tmp(0), tmp(1), 31);
            muls(tmp(4), tmp(0), packed_radix);
            add(digit(dst, j), tmp(1), tmp(4));
        }
    }
    // src is accumulator sum, disjoint from dst and odd. Stage every odd half
    // before the strided stores; in particular dst may overwrite all of bank6.
    __aicore__ inline void unpack24(I32 dst, I32 src, I32 odd) {
        AscendC::ShiftRight(as_unsigned(odd), as_unsigned(src), radix_bits,
                            uint64_t(lanes), uint8_t(packed_digits),
                            AscendC::UnaryRepeatParams{1, 1, row_stride, row_stride});
        vector_barrier();
        AscendC::And(src.template ReinterpretCast<uint16_t>(),
                    src.template ReinterpretCast<uint16_t>(),
                    tmp(15).template ReinterpretCast<uint16_t>(),
                    uint64_t(2 * lanes), uint8_t(packed_digits),
                    AscendC::BinaryRepeatParams{1, 1, 1, row_stride, row_stride, 0});
        vector_barrier();
        AscendC::Adds(dst, src, int32_t(0), uint64_t(lanes),
                      uint8_t(packed_digits),
                      AscendC::UnaryRepeatParams{1, 1, pair_stride, row_stride});
        vector_barrier();
        if constexpr (digits / 2 != 0) {
            AscendC::Adds(digit(dst, 1), odd, int32_t(0), uint64_t(lanes),
                          uint8_t(digits / 2),
                          AscendC::UnaryRepeatParams{1, 1, pair_stride, row_stride});
            vector_barrier();
        }
    }

    // Pack/copy only after BOTH operands have been staged: a/b may be bank6.
    // The prepacked table is immutable and disjoint from engine scratch.
    template<bool prepacked>
    __aicore__ inline void load_packed_modulus(I32 dst, I32 modulus) {
        if constexpr (prepacked) {
            AscendC::Adds(dst, modulus, int32_t(0), uint64_t(lanes),
                          uint8_t(packed_digits),
                          AscendC::UnaryRepeatParams{1, 1, row_stride, row_stride});
            vector_barrier();
        } else {
            pack24(dst, modulus);
        }
    }

    // R11: only add/sub transiently use radix24. Accumulator rows [0,n),
    // [n,2n), [2n,3n), [3n,4n) hold a,b,sum,odd; bank6 holds m,difference.
    // Pack BOTH operands before touching bank6 or dst: a/b/dst may alias each
    // other or bank6. G1 slots never overlap accumulator, bank7, or tmp0..5.
    // tmp0=borrow/selection, tmp1=work, tmp2=carry, tmp3=add high,
    // tmp4=borrow*Q or m*borrow, tmp5=Q-1. No helper here writes tmp6..15;
    // tmp15 is read-only. Existing digits-wide row helpers are not used on
    // compact tensors, so they cannot cross these packed scratch boundaries.
    // Inputs are in [0,m), where m is p or 2p. They never alias acc.
    template<bool prepacked>
    HP_FIELD_VECTOR_HEAVY __aicore__ inline void add_mod_impl(
        I32 dst, I32 a, I32 b, I32 packed_or_radix_modulus) {
        auto packed_a = acc();
        auto packed_b = digit(acc(), packed_digits);
        auto sum = digit(acc(), 2 * packed_digits);
        auto odd = digit(acc(), 3 * packed_digits);
        auto modulus = bank<6>();
        auto difference = digit(bank<6>(), packed_digits);
        pack24(packed_a, a);
        pack24(packed_b, b);
        load_packed_modulus<prepacked>(modulus, packed_or_radix_modulus);
        fill(tmp(5), packed_radix - 1);
        fill(tmp(2), 0);
        for (uint32_t j = 0; j < packed_digits; ++j) {
            add(tmp(1), digit(packed_a, j), digit(packed_b, j));
            add(tmp(1), tmp(1), tmp(2));
            normalize24(digit(sum, j), tmp(2), tmp(1));
        }
        copy(tmp(3), tmp(2));
        subtract24(difference, sum, modulus);
        // A+B = high*Q^n+sum. It is >=m iff high-final_borrow>=0.
        // Select sum only when this signed difference is negative.
        sub(tmp(1), tmp(3), tmp(0));
        shift_right(tmp(0), tmp(1), 31);
        AscendC::Sub(sum, sum, difference, uint64_t(lanes),
                     uint8_t(packed_digits),
                     AscendC::BinaryRepeatParams{1, 1, 1, row_stride, row_stride, row_stride});
        vector_barrier();
        AscendC::Mul(sum, sum, tmp(0), uint64_t(lanes),
                     uint8_t(packed_digits),
                     AscendC::BinaryRepeatParams{1, 1, 1, row_stride, row_stride, 0});
        vector_barrier();
        AscendC::Add(sum, sum, difference, uint64_t(lanes),
                     uint8_t(packed_digits),
                     AscendC::BinaryRepeatParams{1, 1, 1, row_stride, row_stride, row_stride});
        vector_barrier();
        // The selected result is now entirely in acc, BEFORE bank6-aliasing
        // dst stores can destroy difference. A+B<2m needs one correction.
        unpack24(dst, sum, odd);
    }
    template<bool prepacked>
    HP_FIELD_VECTOR_HEAVY __aicore__ inline void sub_mod_impl(
        I32 dst, I32 a, I32 b, I32 packed_or_radix_modulus) {
        auto packed_a = acc();
        auto packed_b = digit(acc(), packed_digits);
        auto sum = digit(acc(), 2 * packed_digits);
        auto odd = digit(acc(), 3 * packed_digits);
        auto modulus = bank<6>();
        pack24(packed_a, a);
        pack24(packed_b, b);
        load_packed_modulus<prepacked>(modulus, packed_or_radix_modulus);
        fill(tmp(5), packed_radix - 1);
        subtract24(sum, packed_a, packed_b);
        // tmp0 remains the final borrow throughout correction. If it is 1,
        // adding m overflows Q^n once; discard only that final base-Q carry.
        fill(tmp(2), 0);
        for (uint32_t j = 0; j < packed_digits; ++j) {
            mul(tmp(4), digit(modulus, j), tmp(0));
            add(tmp(1), digit(sum, j), tmp(4));
            add(tmp(1), tmp(1), tmp(2));
            normalize24(digit(sum, j), tmp(2), tmp(1));
        }
        // -m<A-B<m, so the corrected result is in [0,m).
        unpack24(dst, sum, odd);
    }
    __aicore__ inline void add_mod(I32 dst, I32 a, I32 b) {
        add_mod_impl<false>(dst, a, b, bank<7>());
    }
    __aicore__ inline void sub_mod(I32 dst, I32 a, I32 b) {
        sub_mod_impl<false>(dst, a, b, bank<7>());
    }
    __aicore__ inline void add_mod_redundant(I32 dst, I32 a, I32 b, I32 packed_2p) {
        add_mod_impl<true>(dst, a, b, packed_2p);
    }
    __aicore__ inline void sub_mod_redundant(I32 dst, I32 a, I32 b, I32 packed_2p) {
        sub_mod_impl<true>(dst, a, b, packed_2p);
    }

    static constexpr int32_t signed_n0prime =
        Params::n0prime <= radix / 2 ? Params::n0prime : Params::n0prime - radix;
    static constexpr uint64_t signed_n0_magnitude =
        signed_n0prime < 0 ? uint64_t(-int64_t(signed_n0prime)) : uint64_t(signed_n0prime);
    static_assert(Params::n0prime > 0 && Params::n0prime < radix,
                  "Montgomery cancellation requires a nonzero radix digit");

    // Highest input digit for normalized integers <2p. Double every public
    // modulus digit and subtract one with borrow: doubling only the top digit
    // would miss carry from below. No __aicore__ Digits::operator[] is called.
    static constexpr uint32_t redundant_top_digit_bound() {
        uint32_t carry = 0;
        uint32_t borrow = 1;
        uint32_t top = 0;
        for (uint32_t j = 0; j < digits; ++j) {
            const uint32_t doubled = 2 * uint32_t(Params::modulus_digits.values[j]) + carry;
            carry = doubled / uint32_t(radix);
            const uint32_t digit_value = doubled % uint32_t(radix);
            top = (digit_value + uint32_t(radix) - borrow) % uint32_t(radix);
            borrow = digit_value < borrow ? 1 : 0;
        }
        return top;
    }
    static constexpr uint32_t safe_signed_cancellation_prefix() {
        constexpr uint64_t mask = radix - 1;
        constexpr uint64_t top = redundant_top_digit_bound();
        uint64_t prior_modulus = 0;
        uint64_t carry_bound = 0;
        for (uint32_t i = 0; i < digits; ++i) {
            uint64_t diagonal_bound = 0;
            for (uint32_t j = 0; j <= i; ++j) {
                const uint64_t a_bound = j == digits - 1 ? top : mask;
                const uint64_t b_bound = i - j == digits - 1 ? top : mask;
                diagonal_bound += a_bound * b_bound;
            }
            if (i != 0) prior_modulus += Params::modulus_digits.values[i];
            // Before q_i: H_i = sum(a_j*b_{i-j}) + sum_{j=1..i}
            // p_j*q_{i-j} + c_{i-1}. Every q is in [0,B-1].
            const uint64_t head_bound =
                diagonal_bound + mask * prior_modulus + carry_bound;
            // Require an exact signed int32 product, never wrapping Muls.
            // All bound arithmetic is uint64 (n<=32, B=4096).
            if (signed_n0_magnitude * head_bound > uint64_t(0x7fffffff)) return i;
            carry_bound =
                (head_bound + mask * uint64_t(Params::modulus_digits.values[0])) / radix;
        }
        return digits;
    }
    static constexpr uint32_t signed_cancellation_prefix = safe_signed_cancellation_prefix();

    // Only the first row supplies q. Congruent signed n0 gives the same low
    // radix digit; the final U16 And also clears the negative product's high
    // halfword. The unmasked specialization is used only for the proven prefix.
    template<bool unmasked>
    __aicore__ inline void montgomery_cancel_column(uint32_t i) {
        auto product = bank<6>();
        auto active = digit(acc(), i);
        if constexpr (unmasked) {
            muls(tmp(4), active, signed_n0prime);
        } else {
            bit_and(tmp(4), active, tmp(15));
            muls(tmp(4), tmp(4), Params::n0prime);
        }
        bit_and(tmp(4), tmp(4), tmp(15));
        multiply_rows(product, bank<7>(), tmp(4), true);
        add_rows(active, active, product);
        shift_right(tmp(2), active, radix_bits);
        add(digit(active, 1), digit(active, 1), tmp(2));
    }
    template<bool unmasked>
    __aicore__ inline void montgomery_iteration(I32 a, I32 b, uint32_t i) {
        auto product = bank<6>();
        auto active = digit(acc(), i);
        multiply_rows(product, a, digit(b, i), true);
        add_rows(active, active, product);
        montgomery_cancel_column<unmasked>(i);
    }

    template<bool canonical>
    __aicore__ inline void montgomery_finish(I32 dst) {
        auto result = digit(acc(), digits);
        for (uint32_t j = 0; j < digits; ++j) {
            shift_right(tmp(2), digit(result, j), radix_bits);
            bit_and(digit(result, j), digit(result, j), tmp(15));
            add(digit(result, j + 1), digit(result, j + 1), tmp(2));
        }
        // 4p<R gives REDC(a*b)<p+4p^2/R<2p<R, so the high
        // normalized row is zero. Finish ALL input reads before writing dst,
        // including dst==a==b. Canonical boundaries subtract p exactly once.
        if constexpr (canonical) {
            subtract_modulus(dst, result);
        } else {
            AscendC::Adds(dst, result, int32_t(0), uint64_t(lanes),
                          uint8_t(digits),
                          AscendC::UnaryRepeatParams{1, 1, row_stride, row_stride});
            vector_barrier();
        }
    }

    // Normalized a,b<2p; inputs are disjoint from acc and engine bank6.
    // dst may alias a/b; canonical dst cannot be bank6 (difference scratch).
    // Engine bank7 remains immutable. Both policies share the product/carry
    // algorithm and preserve tmp6..15.
    template<bool canonical>
    HP_FIELD_VECTOR_HEAVY __aicore__ inline void montgomery_impl(I32 dst, I32 a, I32 b) {
        AscendC::Duplicate(acc(), int32_t(0), 65 * lanes);
        vector_barrier();
        // Radix12 bounds each coefficient by 2*n*4096*4095:
        // Fq1073479680, Fr738017280. Carry only the eliminated coefficient.
        for (uint32_t i = 0; i < signed_cancellation_prefix; ++i)
            montgomery_iteration<true>(a, b, i);
        for (uint32_t i = signed_cancellation_prefix; i < digits; ++i)
            montgomery_iteration<false>(a, b, i);
        montgomery_finish<canonical>(dst);
    }
    __aicore__ inline void montgomery(I32 dst, I32 a, I32 b) {
        montgomery_impl<true>(dst, a, b);
    }
    __aicore__ inline void montgomery_redundant(I32 dst, I32 a, I32 b) {
        montgomery_impl<false>(dst, a, b);
    }

    // Normalized a<2p, disjoint from acc/bank6; exact dst==a is supported.
    // Canonical dst cannot be bank6. Only acc, bank6 and tmp0..5 are scratch;
    // all a reads precede finish, preserving tmp6..15 and immutable tables.
    template<bool canonical>
    HP_FIELD_VECTOR_HEAVY __aicore__ inline void montgomery_square_impl(I32 dst, I32 a) {
        // Diagonals assign every even row; initialize only odds and guard.
        // Fr acc rows45..64 are unused and need not be cleared.
        AscendC::Duplicate(digit(acc(), 1), int32_t(0), uint64_t(lanes),
                           uint8_t(digits), uint16_t(1), pair_stride);
        vector_barrier();
        fill(digit(acc(), 2 * digits), 0);
        AscendC::Mul(acc(), a, a, uint64_t(lanes), uint8_t(digits),
                     AscendC::BinaryRepeatParams{1, 1, 1, pair_stride,
                                                row_stride, row_stride});
        vector_barrier();
        for (uint32_t i = 0; i < digits - 1; ++i) {
            // 2*a_i<=8190, so each product <=33538050 fits int32.
            add(tmp(4), digit(a, i), digit(a, i));
            const uint8_t repeats = uint8_t(digits - i - 1);
            multiply_rows(bank<6>(), digit(a, i + 1), tmp(4), true, repeats);
            auto active = digit(acc(), 2 * i + 1);
            add_rows(active, active, bank<6>(), repeats);
        }
        // The preconvolution has the same low column as ordinary multiplication.
        // Its coefficient bound is 2*n*4096*4095, including cancellation/carry.
        for (uint32_t i = 0; i < signed_cancellation_prefix; ++i)
            montgomery_cancel_column<true>(i);
        for (uint32_t i = signed_cancellation_prefix; i < digits; ++i)
            montgomery_cancel_column<false>(i);
        montgomery_finish<canonical>(dst);
    }
    __aicore__ inline void montgomery_square(I32 dst, I32 a) {
        montgomery_square_impl<true>(dst, a);
    }
    __aicore__ inline void montgomery_square_redundant(I32 dst, I32 a) {
        montgomery_square_impl<false>(dst, a);
    }

    // p-2 is public. Compute its bits from the original u32 parameters, so
    // domains whose low word is 1 correctly borrow across that word boundary.
    __aicore__ inline uint32_t exponent_word(uint32_t index) {
        uint64_t borrow = 2;
        uint32_t result = 0;
        for (uint32_t w = 0; w <= index; ++w) {
            const uint64_t p = Field::Parameters::modulus[w];
            result = uint32_t(p - borrow);
            borrow = p < borrow ? 1 : 0;
        }
        return result;
    }
    // Montgomery-in/Montgomery-out primitive shared with the in-kernel G1 engine.
    // Input is copied before exponentiation; banks4/5 and arithmetic scratch are
    // clobbered. Input may be <2p: each product is canonical after its
    // subtraction, even when the unchanged bank4 base is redundant.
    // This is not a conversion or a host/kernel launch boundary.
    HP_FIELD_VECTOR_HEAVY __aicore__ inline void inverse_montgomery(I32 dst, I32 input) {
        static_assert(!Compact, "Compact G1 engine cannot perform inversion");
        for (uint32_t j = 0; j < digits; ++j)
            copy(digit(bank<4>(), j), digit(input, j));
        for (uint32_t j = 0; j < digits; ++j)
            fill(digit(bank<5>(), j), Params::r_digits[j]);
        for (uint32_t w = words; w-- > 0;) {
            const uint32_t exponent = exponent_word(w);
            for (uint32_t bit = 32; bit-- > 0;) {
                montgomery_square(bank<5>(), bank<5>());
                if ((exponent >> bit) & 1U)
                    montgomery(bank<5>(), bank<5>(), bank<4>());
            }
        }
        for (uint32_t j = 0; j < digits; ++j)
            copy(digit(dst, j), digit(bank<5>(), j));
    }
    HP_FIELD_VECTOR_HEAVY __aicore__ inline void inverse(I32 dst, I32 input) {
        static_assert(!Compact, "Compact G1 engine cannot perform inversion");
        load_r2();
        montgomery(bank<4>(), input, bank<3>());
        inverse_montgomery(bank<5>(), bank<4>());
        fill_bank(bank<1>(), 0);
        fill(digit(bank<1>(), 0), 1);
        montgomery(dst, bank<5>(), bank<1>());
    }

    HP_FIELD_VECTOR_HEAVY __aicore__ inline void reduce_wide(I32 dst, U32 input) {
        static_assert(!Compact, "Compact G1 raw SoA cannot hold a wide field input");
        fill_bank(dst, 0);
        auto signed_input = input.template ReinterpretCast<int32_t>();
        fill(tmp(14), 1);
        // Includes all 17/25 words, not merely the low double-width integer.
        for (uint32_t w = wide_words; w-- > 0;) {
            for (uint32_t bit = 32; bit-- > 0;) {
                shift_right(tmp(12), digit(signed_input, w), bit);
                bit_and(tmp(12), tmp(12), tmp(14));
                copy(tmp(2), tmp(12));
                for (uint32_t j = 0; j < digits; ++j) {
                    add(tmp(1), digit(dst, j), digit(dst, j));
                    add(tmp(1), tmp(1), tmp(2));
                    normalize(digit(acc(), j), tmp(2), tmp(1));
                }
                copy(digit(acc(), digits), tmp(2));
                subtract_modulus(dst, acc());
            }
        }
    }

    HP_FIELD_VECTOR_HEAVY __aicore__ inline void pack_mask(I32 input) {
        static_assert(!Compact, "Compact G1 zero-row workspace is not a mask buffer");
        fill(tmp(12), 0);
        for (uint32_t j = 0; j < digits; ++j)
            add(tmp(12), tmp(12), digit(input, j));
        // 32*(B-1)<2^20. Thus this is exactly 0 for zero and 1 otherwise.
        adds(tmp(12), tmp(12), 1048575);
        shift_right(tmp(12), tmp(12), 20);
        sync<AscendC::HardEvent::V_S>();
        auto indices = u32(offsets);
        constexpr uint32_t packed_words = lanes / 4;
        for (uint32_t part = 0; part < 4; ++part)
            for (uint32_t lane = 0; lane < lanes; ++lane)
                indices.SetValue(part * lanes + lane, 4 * (4 * (lane % packed_words) + part));
        sync<AscendC::HardEvent::S_V>();
        fill(i32(raw_mask), 0);
        for (uint32_t part = 0; part < 4; ++part) {
            gather(as_unsigned(tmp(10)), as_unsigned(tmp(12)), indices[part * lanes], lanes);
            shift_left(tmp(10), tmp(10), 8 * part);
            bit_or(i32(raw_mask), i32(raw_mask), tmp(10));
        }
        // First lanes/4 u32 words contain all lane mask bytes; remaining
        // repetitions are harmless padding, never copied to GM.
    }

    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> arena_;
    Bytes memory_;
};

} // namespace hyperplonk::ascend910b3

#undef HP_FIELD_VECTOR_HEAVY
