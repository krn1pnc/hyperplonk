#pragma once

#include "kernel_operator.h"
#include "field_params.h"

#if defined(__NPU_DEVICE__)
#define HP_FIELD_VECTOR_HEAVY __attribute__((noinline))
#else
#define HP_FIELD_VECTOR_HEAVY
#endif

namespace hyperplonk::ascend910b3 {

class G1Vector;

// UB-only Montgomery representation; the public Field representation is unchanged.
// All numerical tensors contain 48 independent lanes. Scalar loops address only
// public digits, exponent bits, layout indices, and tiles, never field values.
template<class Field> class FieldVector {
    friend class G1Vector;
    using Params = FieldVectorParams<typename Field::Parameters>;
    using I32 = AscendC::LocalTensor<int32_t>;
    using U32 = AscendC::LocalTensor<uint32_t>;
    using Bytes = AscendC::LocalTensor<uint8_t>;
    static constexpr uint32_t lanes = 48;
    static constexpr uint32_t digits = Params::digit_count;
    static constexpr uint32_t words = Field::word_count;
    static constexpr uint32_t wide_words = 2 * words + 1;
    static constexpr uint32_t radix_bits = 12;
    static constexpr int32_t radix = 1 << radix_bits;
    static constexpr uint32_t vector_bytes = lanes * sizeof(int32_t);
    static constexpr uint32_t bank_bytes = 32 * vector_bytes;

    // One allocation, with every offset a multiple of 32 bytes. Capacities are
    // deliberately the same for Fr and Fq, including the full high32 wide input.
    static constexpr uint32_t raw_lhs = 0;
    static constexpr uint32_t raw_rhs = raw_lhs + 4800;
    static constexpr uint32_t raw_out = raw_rhs + 2304;
    static constexpr uint32_t raw_mask = raw_out + 2304;
    static constexpr uint32_t soa_lhs = raw_mask + 192;
    static constexpr uint32_t soa_rhs = soa_lhs + 4800;
    static constexpr uint32_t soa_out = soa_rhs + 2304;
    static constexpr uint32_t offsets = soa_out + 2304;
    static constexpr uint32_t banks = offsets + 4800;
    static constexpr uint32_t accumulator = banks + 8 * bank_bytes;
    static constexpr uint32_t temporaries = accumulator + 65 * vector_bytes;
    static constexpr uint32_t used_bytes = temporaries + 16 * vector_bytes;
    static_assert(digits <= 32 && words <= 12 && wide_words <= 25,
                  "field exceeds the Vector arena layout");
    static_assert((raw_lhs | raw_rhs | raw_out | raw_mask | soa_lhs |
                   soa_rhs | soa_out | offsets | banks | accumulator |
                   temporaries | used_bytes) % 32 == 0, "unaligned Vector arena");
    static_assert(used_bytes == 88512,
                  "Vector arena exceeds its reserved UB budget");

public:
    static constexpr uint32_t arena_bytes = used_bytes;
    static constexpr uint32_t tile_size = lanes;

    __aicore__ inline FieldVector() {
        pipe_.InitBuffer(arena_, arena_bytes);
        memory_ = arena_.template Get<uint8_t>();
        // The high halfword of each int32 mask is zero. And/Or below always
        // operate explicitly on all 96 halfwords, not the broken i32 count path.
        fill(tmp(15), radix - 1);
        for (uint32_t j = 0; j < digits; ++j)
            fill(digit(bank(7), j), Params::modulus_digits[j]);
    }

    // op: add, sub, neg, mul, square, inverse, wide (0..6).
    // Nonempty calls require canonical, valid input and disjoint output buffers.
    // lhs/rhs may alias each other. No GM scratch or resource ownership is added.
    __aicore__ inline void run(GM_ADDR lhs, GM_ADDR rhs, GM_ADDR out,
                               GM_ADDR mask, uint64_t count, uint32_t op) {
        if (count == 0) return;
        const uint32_t input_words = op == 6 ? wide_words : words;
        const uint64_t stride = uint64_t(AscendC::GetBlockNum()) * lanes;
        for (uint64_t offset = uint64_t(AscendC::GetBlockIdx()) * lanes;
             offset < count; offset += stride) {
            const uint32_t valid = uint32_t(count - offset < lanes ? count - offset : lanes);
            // Padding lanes must be zero before an exact-byte DMA tail; MTE2
            // must not race the preceding Vector clear of the input workspace.
            AscendC::Duplicate(i32(raw_lhs), int32_t(0), 4800 / 4);
            AscendC::Duplicate(i32(raw_rhs), int32_t(0), 2304 / 4);
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
                reduce_wide(bank(2), u32(soa_lhs));
            } else {
                unpack(bank(0), u32(soa_lhs));
                if (binary) {
                    gather(u32(soa_rhs), u32(raw_rhs), u32(offsets), words * lanes);
                    unpack(bank(1), u32(soa_rhs));
                }
                switch (op) {
                case 0: add_mod(bank(2), bank(0), bank(1)); break;
                case 1: sub_mod(bank(2), bank(0), bank(1)); break;
                case 2:
                    fill_bank(bank(1), 0);
                    sub_mod(bank(2), bank(1), bank(0));
                    break;
                case 3:
                case 4:
                    load_r2();
                    montgomery(bank(4), bank(0), bank(3));
                    montgomery(bank(2), bank(4), op == 3 ? bank(1) : bank(0));
                    break;
                case 5: inverse(bank(2), bank(0)); break;
                }
            }
            pack(bank(2));
            if (op == 5) pack_mask(bank(0));
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
    __aicore__ inline I32 bank(uint32_t index) { return i32(banks + index * bank_bytes); }
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
        AscendC::Duplicate(dst, value, uint64_t(lanes), uint8_t(1), 1, 6);
        vector_barrier();
    }
    __aicore__ inline void copy(I32 dst, I32 src) {
        AscendC::Adds(dst, src, int32_t(0), uint64_t(lanes), uint8_t(1),
                      AscendC::UnaryRepeatParams{1, 1, 6, 6});
        vector_barrier();
    }
    __aicore__ inline void add(I32 dst, I32 a, I32 b) {
        AscendC::Add(dst, a, b, uint64_t(lanes), uint8_t(1),
                     AscendC::BinaryRepeatParams{1, 1, 1, 6, 6, 6});
        vector_barrier();
    }
    __aicore__ inline void sub(I32 dst, I32 a, I32 b) {
        AscendC::Sub(dst, a, b, uint64_t(lanes), uint8_t(1),
                     AscendC::BinaryRepeatParams{1, 1, 1, 6, 6, 6});
        vector_barrier();
    }
    __aicore__ inline void mul(I32 dst, I32 a, I32 b) {
        AscendC::Mul(dst, a, b, uint64_t(lanes), uint8_t(1),
                     AscendC::BinaryRepeatParams{1, 1, 1, 6, 6, 6});
        vector_barrier();
    }
    // Each digit row is six 32-byte blocks. Explicit mask/repeats also allow
    // broadcasting the same 48-lane coefficient with src1RepStride zero.
    __aicore__ inline void multiply_rows(I32 dst, I32 a, I32 b, bool broadcast) {
        const AscendC::BinaryRepeatParams strides{
            1, 1, 1, 6, 6, static_cast<uint8_t>(broadcast ? 0 : 6)};
        AscendC::Mul(dst, a, b, uint64_t(lanes), uint8_t(digits), strides);
        vector_barrier();
    }
    __aicore__ inline void add_rows(I32 dst, I32 a, I32 b) {
        AscendC::Add(dst, a, b, uint64_t(lanes), uint8_t(digits),
                     AscendC::BinaryRepeatParams{1, 1, 1, 6, 6, 6});
        vector_barrier();
    }
    __aicore__ inline void subtract_rows(I32 dst, I32 a, I32 b) {
        AscendC::Sub(dst, a, b, uint64_t(lanes), uint8_t(digits),
                     AscendC::BinaryRepeatParams{1, 1, 1, 6, 6, 6});
        vector_barrier();
    }
    __aicore__ inline void adds(I32 dst, I32 src, int32_t value) {
        AscendC::Adds(dst, src, value, uint64_t(lanes), uint8_t(1),
                      AscendC::UnaryRepeatParams{1, 1, 6, 6});
        vector_barrier();
    }
    __aicore__ inline void muls(I32 dst, I32 src, int32_t value) {
        AscendC::Muls(dst, src, value, uint64_t(lanes), uint8_t(1),
                      AscendC::UnaryRepeatParams{1, 1, 6, 6});
        vector_barrier();
    }
    __aicore__ inline void shift_right(I32 dst, I32 src, uint32_t bits) {
        AscendC::ShiftRight(as_unsigned(dst), as_unsigned(src), bits,
                            uint64_t(lanes), uint8_t(1),
                            AscendC::UnaryRepeatParams{1, 1, 6, 6});
        vector_barrier();
    }
    __aicore__ inline void shift_left(I32 dst, I32 src, uint32_t bits) {
        AscendC::ShiftLeft(as_unsigned(dst), as_unsigned(src), bits,
                           uint64_t(lanes), uint8_t(1),
                           AscendC::UnaryRepeatParams{1, 1, 6, 6});
        vector_barrier();
    }
    __aicore__ inline void bit_and(I32 dst, I32 a, I32 b) {
        AscendC::And(dst.template ReinterpretCast<uint16_t>(),
                    a.template ReinterpretCast<uint16_t>(),
                    b.template ReinterpretCast<uint16_t>(),
                    uint64_t(2 * lanes), uint8_t(1),
                    AscendC::BinaryRepeatParams{1, 1, 1, 6, 6, 6});
        vector_barrier();
    }
    __aicore__ inline void bit_or(I32 dst, I32 a, I32 b) {
        AscendC::Or(dst.template ReinterpretCast<uint16_t>(),
                   a.template ReinterpretCast<uint16_t>(),
                   b.template ReinterpretCast<uint16_t>(),
                   uint64_t(2 * lanes), uint8_t(1),
                   AscendC::BinaryRepeatParams{1, 1, 1, 6, 6, 6});
        vector_barrier();
    }
    __aicore__ inline void fill_bank(I32 dst, int32_t value) {
        AscendC::Duplicate(dst, value, int32_t(digits * lanes));
        vector_barrier();
    }
    __aicore__ inline void load_r2() {
        for (uint32_t j = 0; j < digits; ++j)
            fill(digit(bank(3), j), Params::r2_digits[j]);
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
        auto difference = bank(6);
        fill(tmp(0), 0);
        for (uint32_t j = 0; j <= digits; ++j) {
            adds(tmp(1), digit(x, j), radix);
            if (j < digits) sub(tmp(1), tmp(1), digit(bank(7), j));
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
    HP_FIELD_VECTOR_HEAVY __aicore__ inline void add_mod(I32 dst, I32 a, I32 b) {
        fill(tmp(2), 0);
        for (uint32_t j = 0; j < digits; ++j) {
            add(tmp(1), digit(a, j), digit(b, j));
            add(tmp(1), tmp(1), tmp(2));
            normalize(digit(acc(), j), tmp(2), tmp(1));
        }
        copy(digit(acc(), digits), tmp(2));
        subtract_modulus(dst, acc());
    }
    HP_FIELD_VECTOR_HEAVY __aicore__ inline void sub_mod(I32 dst, I32 a, I32 b) {
        fill(tmp(0), 0);
        for (uint32_t j = 0; j < digits; ++j) {
            adds(tmp(1), digit(a, j), radix);
            sub(tmp(1), tmp(1), digit(b, j));
            sub(tmp(1), tmp(1), tmp(0));
            shift_right(tmp(2), tmp(1), radix_bits);
            bit_and(digit(dst, j), tmp(1), tmp(15));
            muls(tmp(0), tmp(2), -1);
            adds(tmp(0), tmp(0), 1);
        }
        // The n-digit subtraction is modulo R. If it borrowed, adding p
        // overflows R once; deliberately discard only that final carry.
        fill(tmp(2), 0);
        for (uint32_t j = 0; j < digits; ++j) {
            muls(tmp(4), tmp(0), Params::modulus_digits[j]);
            add(tmp(1), digit(dst, j), tmp(4));
            add(tmp(1), tmp(1), tmp(2));
            normalize(digit(dst, j), tmp(2), tmp(1));
        }
    }

    HP_FIELD_VECTOR_HEAVY __aicore__ inline void montgomery(I32 dst, I32 a, I32 b) {
        auto product = bank(6);
        AscendC::Duplicate(acc(), int32_t(0), 65 * lanes);
        vector_barrier();
        // Radix12 bounds each coefficient by 2*n*4096*4095:
        // Fq1073479680, Fr738017280. Carry only the eliminated coefficient.
        for (uint32_t i = 0; i < digits; ++i) {
            auto active = digit(acc(), i);
            multiply_rows(product, a, digit(b, i), true);
            add_rows(active, active, product);
            bit_and(tmp(4), active, tmp(15));
            muls(tmp(4), tmp(4), Params::n0prime);
            bit_and(tmp(4), tmp(4), tmp(15));
            multiply_rows(product, bank(7), tmp(4), true);
            add_rows(active, active, product);
            shift_right(tmp(2), active, radix_bits);
            add(digit(active, 1), digit(active, 1), tmp(2));
        }
        auto result = digit(acc(), digits);
        for (uint32_t j = 0; j < digits; ++j) {
            shift_right(tmp(2), digit(result, j), radix_bits);
            bit_and(digit(result, j), digit(result, j), tmp(15));
            add(digit(result, j + 1), digit(result, j + 1), tmp(2));
        }
        // Canonical a,b imply result<2p. Retain its high row; finish every
        // operand read before writing dst, including the dst==a==b case.
        subtract_modulus(dst, result);
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
    // clobbered. This is not a conversion or a host/kernel launch boundary.
    HP_FIELD_VECTOR_HEAVY __aicore__ inline void inverse_montgomery(I32 dst, I32 input) {
        for (uint32_t j = 0; j < digits; ++j)
            copy(digit(bank(4), j), digit(input, j));
        for (uint32_t j = 0; j < digits; ++j)
            fill(digit(bank(5), j), Params::r_digits[j]);
        for (uint32_t w = words; w-- > 0;) {
            const uint32_t exponent = exponent_word(w);
            for (uint32_t bit = 32; bit-- > 0;) {
                montgomery(bank(5), bank(5), bank(5));
                if ((exponent >> bit) & 1U)
                    montgomery(bank(5), bank(5), bank(4));
            }
        }
        for (uint32_t j = 0; j < digits; ++j)
            copy(digit(dst, j), digit(bank(5), j));
    }
    HP_FIELD_VECTOR_HEAVY __aicore__ inline void inverse(I32 dst, I32 input) {
        load_r2();
        montgomery(bank(4), input, bank(3));
        inverse_montgomery(bank(5), bank(4));
        fill_bank(bank(1), 0);
        fill(digit(bank(1), 0), 1);
        montgomery(dst, bank(5), bank(1));
    }

    HP_FIELD_VECTOR_HEAVY __aicore__ inline void reduce_wide(I32 dst, U32 input) {
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
        // First 12 u32 words contain all 48 mask bytes; remaining repetitions
        // are harmless padding, never copied to GM.
    }

    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> arena_;
    Bytes memory_;
};

} // namespace hyperplonk::ascend910b3

#undef HP_FIELD_VECTOR_HEAVY
