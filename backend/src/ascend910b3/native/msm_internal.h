#pragma once
#include "msm.h"
#include "msm_wave.h"
namespace hyperplonk::ascend910b3::detail {
// Packed integer words of the vector engine's Montgomery one, not radix-12 banks.
struct MsmMontgomeryOneWords {
    uint32_t values[Bls12381FqParams::word_count]{};
#if defined(__NPU_DEVICE__)
    __aicore__
#endif
    constexpr uint32_t operator[](unsigned word) const { return values[word]; }
};
constexpr MsmMontgomeryOneWords pack_msm_montgomery_one() {
    using Params = FieldVectorParams<Bls12381FqParams>;
    MsmMontgomeryOneWords result{};
    for (unsigned j=0;j<Params::digit_count;++j) {
        const unsigned bit=j*Params::radix_bits, word=bit/32, shift=bit%32;
        const uint32_t digit=static_cast<uint32_t>(Params::r_digits.values[j]);
        result.values[word] |= digit<<shift;
        if (shift+Params::radix_bits>32 && word+1<Bls12381FqParams::word_count)
            result.values[word+1] |= digit>>(32-shift);
    }
    return result;
}
inline constexpr auto msm_montgomery_one_words = pack_msm_montgomery_one();
// Every online stage reads its dynamic counts and routing parity on device, so
// none of these signatures carries a CPU-derived count, bookkeeping, or gate.
// The fused-wave metadata contract (geometry, arena allocator, control/diag
// blocks) lives in msm_wave.h; nothing here declares a host-visible count or
// level state. The six wave entry points enqueue one bundle's metadata capture
// and its 26-level tree: they never synchronize, allocate or transfer to host.
// `plan_workspace` is the production MsmPlan workspace (source window regions
// plus the WaveSlotMeta destination) and `wave_arena` is the WaveArena base.
// capture: for bundle slot `fuse_index`, copies the CURRENT window's selected
// records, RowPtr, record parity and the current ScheduleMeta 64-byte header
// into the arena and WaveSlotMeta[u], before the next window overwrites them.
// plan: one launch per bundle, after every capture of that bundle.
// build: per level slot 0..25, publishing that level's ops/diag.
// affine: per level and sub-wave msm_wave_subwave_bound().
// carry: per level, after all sub-waves of that level.
// finalize: one launch per bundle, after level 25, publishing out_b per window.
void launch_msm_wave_capture(aclrtStream, void* plan_workspace, void* wave_arena,
                             const MsmPlan&, const WaveBounds&, uint32_t fuse_index);
void launch_msm_wave_plan(aclrtStream, void* wave_arena, const MsmPlan&,
                          const WaveBounds&, uint32_t fuse_windows);
void launch_msm_wave_build(aclrtStream, void* wave_arena, const MsmPlan&,
                           const WaveBounds&, uint32_t fuse_windows, uint32_t level);
void launch_msm_wave_affine(aclrtStream, const void* cache, void* wave_arena,
                            const MsmPlan&, const WaveBounds&, uint32_t fuse_windows,
                            uint32_t level, uint32_t subwave);
void launch_msm_wave_carry(aclrtStream, const void* cache, void* wave_arena,
                           const MsmPlan&, const WaveBounds&, uint32_t fuse_windows,
                           uint32_t level);
void launch_msm_wave_finalize(aclrtStream, const void* cache, void* wave_arena,
                              const MsmPlan&, const WaveBounds&, uint32_t fuse_windows);
void launch_msm_prepare_bases(aclrtStream, const G1Affine*, void* cache, const MsmPlan&);
void launch_msm_extract(aclrtStream, const Fr*, void*, const MsmPlan&, uint32_t window);
void launch_msm_radix_group(aclrtStream, void*, const MsmPlan&);
void launch_msm_rows(aclrtStream, void*, const MsmPlan&);
void launch_msm_schedule(aclrtStream, void*, const MsmPlan&);
// Per-window weighted reduce of one captured wave slot. `buckets` is the wave
// arena out_b slot of that window and `meta` is its captured 64-byte
// ScheduleMeta header; the weighted segment/tree kernel bodies are unchanged.
// The destination is the final per-window tuple of MsmRegion::Windows.
void launch_msm_window_reduce(aclrtStream, void* workspace, const MsmPlan&,
                              const void* buckets, void* destination, const void* meta);
void launch_msm_finish(aclrtStream, const void*, G1Affine*, const MsmPlan&);
} // namespace hyperplonk::ascend910b3::detail
