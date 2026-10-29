#pragma once

#include "kernel_operator.h"
#define HP_FIELD_DEVICE __aicore__
#include "../g1.h"

namespace field_smoke {
template<class Field>
__aicore__ inline Field load(AscendC::LocalTensor<uint8_t> local, uint32_t index)
{
    const auto words = local.ReinterpretCast<uint32_t>();
    Field value{};
    for (uint32_t w = 0; w < Field::word_count; ++w)
        value.words[w] = words.GetValue(index * Field::word_count + w);
    return value;
}

template<class Field>
__aicore__ inline void store(AscendC::LocalTensor<uint8_t> local,
                             uint32_t index, const Field& value)
{
    auto words = local.ReinterpretCast<uint32_t>();
    for (uint32_t w = 0; w < Field::word_count; ++w)
        words.SetValue(index * Field::word_count + w, value.words[w]);
}
} // namespace field_smoke
// Test-only compact point transport. Odd records are scalar UB views, not DMA bases.
namespace g1_smoke {
using namespace hyperplonk::ascend910b3;
template<uint32_t InputBytes, uint32_t OutputBytes> class Staging {
public:
    static_assert(InputBytes % 32 == 0 && OutputBytes % 32 == 0);
    static_assert(InputBytes + OutputBytes <= 32768);
    __aicore__ inline Staging() {
        pipe_.InitBuffer(input_, 1, InputBytes);
        pipe_.InitBuffer(output_, 1, OutputBytes);
    }
    __aicore__ inline AscendC::LocalTensor<uint8_t> alloc_input() { return input_.AllocTensor<uint8_t>(); }
    __aicore__ inline void copy_in(AscendC::LocalTensor<uint8_t> local, GM_ADDR source, uint32_t bytes) {
        if (!bytes) return;
        AscendC::GlobalTensor<uint8_t> gm;
        gm.SetGlobalBuffer(source);
        AscendC::DataCopyPad(local, gm, AscendC::DataCopyExtParams{1,bytes,0,0,0},
                            AscendC::DataCopyPadExtParams<uint8_t>{false,0,0,0});
    }
    __aicore__ inline AscendC::LocalTensor<uint8_t> ready_input(AscendC::LocalTensor<uint8_t> local) {
        input_.EnQue(local);
        local = input_.DeQue<uint8_t>();
        const auto event = pipe_.FetchEventID(AscendC::HardEvent::MTE2_S);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(event);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(event);
        return local;
    }
    __aicore__ inline void release(AscendC::LocalTensor<uint8_t> local) { input_.FreeTensor(local); }
    __aicore__ inline AscendC::LocalTensor<uint8_t> alloc_output() { return output_.AllocTensor<uint8_t>(); }
    __aicore__ inline void write(AscendC::LocalTensor<uint8_t> local, GM_ADDR destination, uint32_t bytes) {
        output_.EnQue(local);
        local = output_.DeQue<uint8_t>();
        auto event = pipe_.FetchEventID(AscendC::HardEvent::S_MTE3);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(event);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(event);
        AscendC::GlobalTensor<uint8_t> gm;
        gm.SetGlobalBuffer(destination);
        AscendC::DataCopyPad(gm,local,AscendC::DataCopyExtParams{1,bytes,0,0,0});
        event = pipe_.FetchEventID(AscendC::HardEvent::MTE3_S);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(event);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(event);
        output_.FreeTensor(local);
    }
private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN,1> input_;
    AscendC::TQue<AscendC::QuePosition::VECOUT,1> output_;
};
__aicore__ inline G1Affine load_affine(AscendC::LocalTensor<uint8_t> local, uint32_t i) {
    return {field_smoke::load<Fq>(local,2*i),field_smoke::load<Fq>(local,2*i+1)};
}
__aicore__ inline G1Jacobian load_jacobian(AscendC::LocalTensor<uint8_t> local, uint32_t i) {
    return {field_smoke::load<Fq>(local,3*i),field_smoke::load<Fq>(local,3*i+1),
            field_smoke::load<Fq>(local,3*i+2)};
}
__aicore__ inline void store_affine(AscendC::LocalTensor<uint8_t> local, uint32_t i, const G1Affine& p) {
    field_smoke::store<Fq>(local,2*i,p.x);
    field_smoke::store<Fq>(local,2*i+1,p.y);
}
__aicore__ inline void store_jacobian(AscendC::LocalTensor<uint8_t> local, uint32_t i, const G1Jacobian& p) {
    field_smoke::store<Fq>(local,3*i,p.x);
    field_smoke::store<Fq>(local,3*i+1,p.y);
    field_smoke::store<Fq>(local,3*i+2,p.z);
}
__aicore__ inline void zero_slot(AscendC::LocalTensor<uint8_t> local) {
    auto words = local.ReinterpretCast<uint32_t>();
    for (unsigned i = 0; i < 8; ++i) words.SetValue(i,0);
}
} // namespace g1_smoke
