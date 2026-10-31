#pragma once
#include "msm.h"
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
// none of these signatures carries a CPU-derived count, level state or gate.
void launch_msm_prepare_bases(aclrtStream, const G1Affine*, void* cache, const MsmPlan&);
void launch_msm_extract(aclrtStream, const Fr*, void*, const MsmPlan&, uint32_t window);
void launch_msm_radix_group(aclrtStream, void*, const MsmPlan&);
void launch_msm_rows(aclrtStream, void*, const MsmPlan&);
void launch_msm_schedule(aclrtStream, void*, const MsmPlan&);
void launch_msm_tasks(aclrtStream, void*, const MsmPlan&, uint32_t level);
void launch_msm_partial_sums(aclrtStream, const void* cache, void* scratch, const MsmPlan&);
void launch_msm_partial_merge(aclrtStream, void*, const MsmPlan&, uint32_t level);
void launch_msm_window_reduce(aclrtStream, void*, const MsmPlan&, uint32_t window);
void launch_msm_finish(aclrtStream, const void*, G1Affine*, const MsmPlan&);
// Internal point materialization is implemented in the accumulation unit.
void launch_msm_bucket_materialize(aclrtStream, void*, const MsmPlan&);
}
