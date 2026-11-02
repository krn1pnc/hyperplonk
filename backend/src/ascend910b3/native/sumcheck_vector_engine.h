#pragma once
// Device-only shared SumCheck engine. The template name/namespace are the
// immutable FieldVector friend contract; the default allocation is unchanged.
#include "kernel_operator.h"
#include "field.h"
#include "field_vector.h"
#include "sumcheck_kernels.h"

namespace hyperplonk::ascend910b3 {

// One UB-resident Montgomery arena driven through the shared FieldVector
// primitives (same access grant as the G1 engines). Every row is an SoA radix
// digit row set: `digits` rows of `Lanes` int32 plus eight zero padding words
// used as the out-of-range target of the lane-reduction gathers.
template<uint32_t Lanes>
class SumcheckVectorEngine {
    using Engine = FieldVector<Fr, Lanes>;
    using Params = FieldVectorParams<Fr::Parameters>;

public:
    using I32 = AscendC::LocalTensor<int32_t>;
    using U32 = AscendC::LocalTensor<uint32_t>;
    using Bytes = AscendC::LocalTensor<uint8_t>;

    static constexpr uint32_t lanes = Lanes;
    static constexpr uint32_t digits = Params::digit_count;
    static constexpr uint32_t pad_elems = 8;
    static constexpr uint32_t row_elems = digits * Lanes + pad_elems;
    static constexpr uint32_t row_bytes = row_elems * 4;
    // Lane-reduction shifts: pow2(Lanes)/2 down to one (48 -> 32..1, 32 -> 16..1,
    // 16 -> 8..1). Upper lanes are zero padded by the index tables.
    static constexpr uint32_t shift_start = Lanes == 48 ? 32 : (Lanes == 32 ? 16 : 8);
    static constexpr uint32_t nshift = Lanes == 48 ? 6 : (Lanes == 32 ? 5 : 4);
    // Word gather tables: plain lane-major, even of a pair, odd of a pair, the
    // lane-major store permutation, and the lane-zero broadcast used for
    // scalar replicas (same five tables as the shared MLE tile engine).
    static constexpr uint32_t word_tables_bytes = 5 * 8 * Lanes * 4;
    static constexpr uint32_t shift_elems = digits * Lanes;
    static constexpr uint32_t shift_tables_bytes = nshift * shift_elems * 4;
    static constexpr uint32_t index_bytes = word_tables_bytes + shift_tables_bytes;
    // Address/offset staging: one 32-byte record per multiplicand.
    static constexpr uint32_t address_stride = 32;
    static constexpr uint32_t stage_bytes = 2048;
    static constexpr uint32_t address_capacity = stage_bytes / address_stride;

    static constexpr uint32_t ub_total = 196352;
    static constexpr uint32_t ub_reserve = 8192;
    static constexpr uint32_t arena_bytes = Engine::arena_bytes;
    static constexpr uint32_t fixed_rows = 7;  // value, delta, v, one, zero, scratch, scratch2
    static constexpr uint32_t fixed_extra =
        fixed_rows * row_bytes + index_bytes + stage_bytes;
    static constexpr uint32_t chunk_capacity =
        (ub_total - ub_reserve - arena_bytes - fixed_extra) / (2 * row_bytes);
    static constexpr uint32_t row_count = fixed_rows + 2 * chunk_capacity;
    static constexpr uint32_t index_offset = row_count * row_bytes;
    static constexpr uint32_t plain_offset = index_offset;
    static constexpr uint32_t even_offset = plain_offset + 8 * Lanes * 4;
    static constexpr uint32_t odd_offset = even_offset + 8 * Lanes * 4;
    static constexpr uint32_t store_offset = odd_offset + 8 * Lanes * 4;
    static constexpr uint32_t broadcast_offset = store_offset + 8 * Lanes * 4;
    static constexpr uint32_t shift_offset = broadcast_offset + 8 * Lanes * 4;
    static constexpr uint32_t stage_offset = index_offset + index_bytes;
    static constexpr uint32_t extra_bytes =
        (stage_offset + stage_bytes + 31) / 32 * 32;

    static constexpr uint32_t value_row = 0;
    static constexpr uint32_t delta_row = 1;
    static constexpr uint32_t v_row = 2;
    static constexpr uint32_t one_row = 3;
    static constexpr uint32_t zero_row = 4;
    static constexpr uint32_t scratch_row = 5;
    static constexpr uint32_t scratch2_row = 6;
    static constexpr uint32_t prod_row(uint32_t t) { return fixed_rows + t; }
    static constexpr uint32_t acc_row(uint32_t t) {
        return fixed_rows + chunk_capacity + t;
    }

    static constexpr uint32_t declared_chunk =
        Lanes == detail::sumcheck_lane_wide ? detail::sumcheck_chunk_wide :
        Lanes == detail::sumcheck_lane_medium ? detail::sumcheck_chunk_medium :
        detail::sumcheck_chunk_narrow;

    static_assert(Lanes == detail::sumcheck_lane_wide ||
                      Lanes == detail::sumcheck_lane_medium ||
                      Lanes == detail::sumcheck_lane_narrow,
                  "SumCheck supports the published 48/32/16 lane variants");
    static_assert(arena_bytes == 461 * Lanes * 4, "FieldVector arena geometry changed");
    static_assert(row_bytes % 32 == 0, "row must hold whole 32-byte blocks");
    static_assert(index_bytes % 32 == 0 && stage_bytes % 32 == 0,
                  "arena regions must stay block aligned");
    static_assert(arena_bytes + extra_bytes + ub_reserve <= ub_total,
                  "SumCheck resident arena exceeds UB");
    static_assert(chunk_capacity >= declared_chunk,
                  "resident grid smaller than the published chunk capacity");
    static_assert(index_offset % 32 == 0 && stage_offset % 32 == 0,
                  "extra arena offsets must stay block aligned");
    // The private arena is fully accounted: every row, every gather table and
    // the offset staging slot live inside extra_bytes, and the FieldVector
    // arena (banks, scratch, masks) is added on top of it.
    static_assert(extra_bytes >=
                      row_count * row_bytes + index_bytes + stage_bytes,
                  "extra arena must cover rows, gather tables and staging");
    // Real per-variant UB totals: FieldVector arena (banks, scratch, masks) +
    // this engine's rows/gather tables/staging + the 8192B SDK reserve.
    static_assert(Lanes != 48 || arena_bytes + extra_bytes + ub_reserve == 195616u,
                  "wide variant UB total changed");
    static_assert(Lanes != 32 || arena_bytes + extra_bytes + ub_reserve == 193824u,
                  "medium variant UB total changed");
    static_assert(Lanes != 16 || arena_bytes + extra_bytes + ub_reserve == 196256u,
                  "narrow variant UB total changed");
    // Private precondition for the mask normalization below: every gather in
    // this engine starts with a full 64-element repeat before any tail, so a
    // single normalization before the call covers all full repeats. This is an
    // engine-local invariant, not a general contract for arbitrary counts.
    static_assert(8 * Lanes >= 64 && digits * Lanes >= 64,
                  "engine gather counts must start with a full 64-element repeat");

    // Tensor mode retains only eleven working rows, not the generic t-grid.
    static constexpr uint32_t tensor_rows = 11;
    static constexpr uint32_t tensor_extra_bytes =
        tensor_rows * row_bytes + index_bytes + stage_bytes;
    __aicore__ inline explicit SumcheckVectorEngine(bool tensor = false)
        : tensor_(tensor) {
        fq_.pipe_.InitBuffer(extra_, tensor ? tensor_extra_bytes : extra_bytes);
        extra_memory_ = extra_.template Get<uint8_t>();
        // bank<3> becomes the immutable R^2 digit row used by every conversion.
        fq_.load_r2();
        build_word_tables();
        build_shift_tables();
        init_rows();
    }

    // ---------------------------------------------------------------------
    // Row helpers. A "row" is one SoA field element set for all lanes.
    // ---------------------------------------------------------------------
    __aicore__ inline I32 row(uint32_t index) {
        return extra_memory_[index * row_bytes].template ReinterpretCast<int32_t>();
    }
    __aicore__ inline U32 row_words(uint32_t index) {
        return extra_memory_[index * row_bytes].template ReinterpretCast<uint32_t>();
    }

    __aicore__ inline void zero_digits(uint32_t index) { fq_.fill_bank(row(index), 0); }
    __aicore__ inline void copy_digits(uint32_t dst, uint32_t src) {
        for (uint32_t j = 0; j < digits; ++j)
            fq_.copy(fq_.digit(row(dst), j), fq_.digit(row(src), j));
    }
    __aicore__ inline void add_row(uint32_t dst, uint32_t a, uint32_t b) {
        fq_.add_mod(row(dst), row(a), row(b));
    }
    __aicore__ inline void sub_row(uint32_t dst, uint32_t a, uint32_t b) {
        fq_.sub_mod(row(dst), row(a), row(b));
    }
    // Montgomery multiplication: REDC(a*b). dst may alias a or b.
    __aicore__ inline void mul_row(uint32_t dst, uint32_t a, uint32_t b) {
        fq_.montgomery(row(dst), row(a), row(b));
    }
    // canonical (or Montgomery-representation bytes) -> Montgomery value.
    __aicore__ inline void to_montgomery(uint32_t index) {
        fq_.montgomery(row(index), row(index), fq_.template bank<3>());
    }
    // Raw double-width REDC from the passed DotREDC primitive: 44 radix-12
    // SoA rows with T < 512*p*p < p*R. Masked CIOS uses exact n0prime.
    __aicore__ inline void redc_wide_soa(uint32_t dst, I32 input) {
        static_assert(Params::n0prime == 4095, "Fr radix-12 n0prime must be -1 mod 4096");
        static_assert(2 * digits < 65, "wide REDC requires the finish guard row");
        AscendC::Duplicate(fq_.acc(), int32_t(0), 65 * Lanes);
        fq_.vector_barrier();
        AscendC::Adds(fq_.acc(), input, int32_t(0), uint64_t(Lanes),
                      uint8_t(2 * digits),
                      AscendC::UnaryRepeatParams{1, 1, Engine::row_stride,
                                                Engine::row_stride});
        fq_.vector_barrier();
        for (uint32_t i = 0; i < digits; ++i)
            fq_.template montgomery_cancel_column<false>(i);
        fq_.template montgomery_finish<true>(row(dst));
    }
    template<class Buffer>
    __aicore__ inline void init_buffer(Buffer& buffer, uint32_t bytes) {
        fq_.pipe_.InitBuffer(buffer, bytes);
    }
    template<AscendC::HardEvent event> __aicore__ inline void sync() {
        fq_.template sync<event>();
    }
    __aicore__ inline void vector_barrier() { fq_.vector_barrier(); }
    __aicore__ inline void normalize_mask() { normalize_vector_mask(); }
    __aicore__ inline void stage_in(Bytes dst, GM_ADDR src, uint32_t bytes) {
        fq_.template sync<AscendC::HardEvent::V_MTE2>();
        fq_.copy_in(dst, src, bytes);
        fq_.template sync<AscendC::HardEvent::MTE2_V>();
    }
    __aicore__ inline void store_out(Bytes src, GM_ADDR dst, uint32_t bytes) {
        fq_.template sync<AscendC::HardEvent::V_MTE3>();
        fq_.copy_out(src, dst, bytes);
        fq_.template sync<AscendC::HardEvent::MTE3_V>();
    }

    // ---------------------------------------------------------------------
    // Device tables. Built once per kernel launch, before any table work.
    // ---------------------------------------------------------------------
    __aicore__ inline void build_word_tables() {
        fq_.template sync<AscendC::HardEvent::V_S>();
        // These tables live in this engine's extra_ TBuf, not in the FieldVector
        // arena: fq_.u32 is memory_-relative and would alias bank6/7 or tmp0..15.
        auto plain = extra_u32(plain_offset);
        auto even = extra_u32(even_offset);
        auto odd = extra_u32(odd_offset);
        auto store = extra_u32(store_offset);
        auto broadcast = extra_u32(broadcast_offset);
        for (uint32_t word = 0; word < 8; ++word) {
            for (uint32_t lane = 0; lane < Lanes; ++lane) {
                // One 32-byte record per lane.
                plain.SetValue(word * Lanes + lane, 4u * (lane * 8u + word));
                // Two records per rest entry: even element then odd element.
                even.SetValue(word * Lanes + lane, 4u * (lane * 16u + word));
                odd.SetValue(word * Lanes + lane, 4u * (lane * 16u + 8u + word));
                // SoA word rows -> one 32-byte record per lane.
                store.SetValue(lane * 8u + word, 4u * (word * Lanes + lane));
                // Every lane reads lane zero's record: scalar replication.
                broadcast.SetValue(word * Lanes + lane, 4u * word);
            }
        }
        fq_.template sync<AscendC::HardEvent::S_V>();
    }

    __aicore__ inline void build_shift_tables() {
        for (uint32_t s = 0; s < nshift; ++s) {
            auto table = extra_u32(shift_offset + s * shift_elems * 4);
            const uint32_t step = shift_start >> s;
            for (uint32_t d = 0; d < digits; ++d) {
                for (uint32_t lane = 0; lane < Lanes; ++lane) {
                    const uint32_t target = lane + step < Lanes
                                                ? d * Lanes + lane + step
                                                : digits * Lanes;
                    table.SetValue(d * Lanes + lane, 4u * target);
                }
            }
        }
        fq_.template sync<AscendC::HardEvent::S_V>();
    }

    __aicore__ inline void init_rows() {
        // Out-of-range lane targets read these zero padding words.
        for (uint32_t i = 0; i < (tensor_ ? tensor_rows : row_count); ++i) {
            AscendC::Duplicate(
                extra_memory_[i * row_bytes + digits * Lanes * 4]
                    .template ReinterpretCast<int32_t>(),
                int32_t(0), pad_elems);
        }
        fq_.vector_barrier();
        zero_digits(zero_row);
        // Canonical one in every lane, used to leave the Montgomery domain.
        fq_.fill_bank(row(one_row), 0);
        fq_.fill(fq_.digit(row(one_row), 0), 1);
    }

    // ---------------------------------------------------------------------
    // Table I/O.
    // ---------------------------------------------------------------------
    // 32-byte records per lane, one element per lane.
    __aicore__ inline void load_plain(GM_ADDR src, uint32_t valid) {
        AscendC::Duplicate(fq_.i32(Engine::raw_lhs), int32_t(0), 25 * Lanes);
        fq_.vector_barrier();
        fq_.template sync<AscendC::HardEvent::V_MTE2>();
        fq_.copy_in(fq_.bytes(Engine::raw_lhs), src, valid * 32);
        fq_.template sync<AscendC::HardEvent::MTE2_V>();
        gather_with_full_mask(fq_.u32(Engine::soa_lhs), fq_.u32(Engine::raw_lhs),
                              extra_u32(plain_offset), 8 * Lanes);
        fq_.unpack(row(value_row), fq_.u32(Engine::soa_lhs));
    }

    // One rest entry per lane: even element into value_row, odd - even into
    // delta_row. `src` points at the first 64-byte pair of this tile.
    __aicore__ inline void load_pair(GM_ADDR src, uint32_t valid) {
        AscendC::Duplicate(fq_.i32(Engine::raw_lhs), int32_t(0), 25 * Lanes);
        fq_.vector_barrier();
        fq_.template sync<AscendC::HardEvent::V_MTE2>();
        fq_.copy_in(fq_.bytes(Engine::raw_lhs), src, valid * 64);
        fq_.template sync<AscendC::HardEvent::MTE2_V>();
        gather_with_full_mask(fq_.u32(Engine::soa_lhs), fq_.u32(Engine::raw_lhs),
                              extra_u32(even_offset), 8 * Lanes);
        fq_.unpack(row(value_row), fq_.u32(Engine::soa_lhs));
        gather_with_full_mask(fq_.u32(Engine::soa_rhs), fq_.u32(Engine::raw_lhs),
                              extra_u32(odd_offset), 8 * Lanes);
        fq_.unpack(row(delta_row), fq_.u32(Engine::soa_rhs));
        fq_.sub_mod(row(delta_row), row(delta_row), row(value_row));
    }

    // Stores `valid` 32-byte records (one element per lane, natural order).
    __aicore__ inline void store_plain(GM_ADDR dst, uint32_t index, uint32_t valid) {
        fq_.pack_words(fq_.i32(Engine::soa_out), row(index));
        gather_with_full_mask(fq_.u32(Engine::raw_out), fq_.u32(Engine::soa_out),
                              extra_u32(store_offset), 8 * Lanes);
        fq_.template sync<AscendC::HardEvent::V_MTE3>();
        fq_.copy_out(fq_.bytes(Engine::raw_out), dst, valid * 32);
        fq_.template sync<AscendC::HardEvent::MTE3_V>();
    }

    // ---------------------------------------------------------------------
    // Scalar metadata staging (offsets and one 32-byte value).
    // ---------------------------------------------------------------------
    __aicore__ inline void stage_offsets(GM_ADDR src, uint32_t count) {
        fq_.template sync<AscendC::HardEvent::S_MTE2>();
        fq_.copy_in(extra_memory_[mapped_offset(stage_offset)], src, count * address_stride);
        fq_.template sync<AscendC::HardEvent::MTE2_S>();
    }
    __aicore__ inline uint32_t load_u32(GM_ADDR src) {
        fq_.template sync<AscendC::HardEvent::S_MTE2>();
        fq_.copy_in(extra_memory_[mapped_offset(stage_offset)], src, 4);
        fq_.template sync<AscendC::HardEvent::MTE2_S>();
        return extra_u32(stage_offset).GetValue(0);
    }
    __aicore__ inline uint64_t staged_offset(uint32_t index) {
        auto words = extra_u32(stage_offset).template ReinterpretCast<uint64_t>();
        return words.GetValue(index * (address_stride / 8));
    }

    // Replicates the 32-byte device value at `src32` to every lane. The bytes
    // are read as data only (no field arithmetic) and unpacked into an SoA row,
    // so a canonical input yields the canonical value in every lane and a
    // Montgomery input yields the same Montgomery representation. Lane zero's
    // record is broadcast by the gather table, never by scalar word copies.
    __aicore__ inline void replicate_value(uint32_t index, GM_ADDR src32) {
        AscendC::Duplicate(fq_.i32(Engine::raw_lhs), int32_t(0), 25 * Lanes);
        fq_.vector_barrier();
        fq_.template sync<AscendC::HardEvent::V_MTE2>();
        fq_.copy_in(fq_.bytes(Engine::raw_lhs), src32, 32);
        fq_.template sync<AscendC::HardEvent::MTE2_V>();
        gather_with_full_mask(fq_.u32(Engine::soa_lhs), fq_.u32(Engine::raw_lhs),
                              extra_u32(broadcast_offset), 8 * Lanes);
        fq_.unpack(row(index), fq_.u32(Engine::soa_lhs));
    }

    // Loads one 32-byte device value into lane zero only; other lanes stay
    // mathematical zero. Used to multiply a lane-sum by one scalar.
    __aicore__ inline void load_lane_zero(uint32_t index, GM_ADDR src32) {
        AscendC::Duplicate(fq_.i32(Engine::raw_lhs), int32_t(0), 25 * Lanes);
        fq_.vector_barrier();
        fq_.template sync<AscendC::HardEvent::V_MTE2>();
        fq_.copy_in(fq_.bytes(Engine::raw_lhs), src32, 32);
        fq_.template sync<AscendC::HardEvent::MTE2_V>();
        gather_with_full_mask(fq_.u32(Engine::soa_lhs), fq_.u32(Engine::raw_lhs),
                              extra_u32(plain_offset), 8 * Lanes);
        fq_.unpack(row(index), fq_.u32(Engine::soa_lhs));
    }

    // ---------------------------------------------------------------------
    // Lane reduction. After every shift the row keeps the partial sums of the
    // remaining lanes; lane zero ends with the exact field sum of all lanes.
    // ---------------------------------------------------------------------
    __aicore__ inline void lane_sum(uint32_t index) {
        for (uint32_t s = 0; s < nshift; ++s) {
            gather_with_full_mask(extra_u32(scratch_row * row_bytes), row_words(index),
                                  extra_u32(shift_offset + s * shift_elems * 4),
                                  shift_elems);
            fq_.add_mod(row(index), row(index), row(scratch_row));
        }
    }

private:
    // Measured mask-state boundary: a mask-form operation can leave a partial
    // NORMAL vector mask that a following count-form gather then observes. The
    // minimal-primitive A/B reproduced this and showed that normalizing at a
    // separate noinline boundary before the gather restores full writes; this
    // is a state-normalization contract, not a claim about how the compiler
    // emits the gather's own mask handling. Every gather here satisfies the
    // static_assert above (a full 64-element repeat precedes any tail), so one
    // normalization covers all full repeats of the call. The wrapper is inline
    // so each call site keeps its original destination/source/index/count
    // expressions and order.
    __attribute__((noinline)) __aicore__ inline void normalize_vector_mask() {
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
    }

    __aicore__ inline void gather_with_full_mask(U32 dst, U32 src, U32 indices,
                                                 uint32_t size) {
        normalize_vector_mask();
        // Same sequence as the validated minimal-primitive A/B: normalize at the
        // noinline boundary, then a vector-pipe barrier, then the count-form
        // gather with the call site's original operands.
        AscendC::PipeBarrier<PIPE_V>();
        fq_.gather(dst, src, indices, size);
    }

    // Gather tables, shift tables, offset staging and the working rows all live
    // in this engine's extra_ TBuf. fq_.u32/i32/bytes are FieldVector::memory_
    // relative and address only the FieldVector arena (raw/soa/banks/tmp), so
    // every extra_-region view must go through this accessor.
    __aicore__ inline U32 extra_u32(uint32_t offset) {
        return extra_memory_[mapped_offset(offset)].template ReinterpretCast<uint32_t>();
    }
    __aicore__ inline uint32_t mapped_offset(uint32_t offset) {
        return tensor_ && offset >= index_offset
                   ? offset - index_offset + tensor_rows * row_bytes : offset;
    }

    Engine fq_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> extra_;
    Bytes extra_memory_;
    bool tensor_;
};

} // namespace hyperplonk::ascend910b3

