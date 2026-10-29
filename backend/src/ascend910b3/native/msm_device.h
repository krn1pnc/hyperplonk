#pragma once
#include "kernel_operator.h"
#include "msm_plan.h"

namespace hyperplonk::ascend910b3::detail::msm_device {
// MSM-only DMA staging, following the existing native G1 queue/event lifecycle.
template<uint32_t InputBytes, uint32_t OutputBytes> class Staging {
public:
    static_assert(InputBytes % 32 == 0 && OutputBytes % 32 == 0);
    static_assert(InputBytes + OutputBytes <= 49152);
    static_assert(InputBytes + OutputBytes <= 57344);
    __aicore__ inline Staging() {
        pipe_.InitBuffer(input_,1,InputBytes);
        pipe_.InitBuffer(output_,1,OutputBytes);
    }
    template<AscendC::HardEvent event>
    __aicore__ inline void sync() {
        const auto id=pipe_.FetchEventID(event);
        AscendC::SetFlag<event>(id);
        AscendC::WaitFlag<event>(id);
    }
    __aicore__ inline AscendC::LocalTensor<uint8_t> alloc_input() { return input_.AllocTensor<uint8_t>(); }
    __aicore__ inline void copy_in(AscendC::LocalTensor<uint8_t> local, GM_ADDR source, uint32_t bytes) {
        if (!bytes) return;
        AscendC::GlobalTensor<uint8_t> gm; gm.SetGlobalBuffer(source);
        AscendC::DataCopyPad(local,gm,AscendC::DataCopyExtParams{1,bytes,0,0,0},
                            AscendC::DataCopyPadExtParams<uint8_t>{false,0,0,0});
    }
    __aicore__ inline AscendC::LocalTensor<uint8_t> ready_input(AscendC::LocalTensor<uint8_t> local) {
        input_.EnQue(local); local = input_.DeQue<uint8_t>();
        const auto e = pipe_.FetchEventID(AscendC::HardEvent::MTE2_S);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(e);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(e);
        return local;
    }
    __aicore__ inline AscendC::LocalTensor<uint8_t> read(GM_ADDR source, uint32_t bytes) {
        auto local = alloc_input(); copy_in(local,source,bytes); return ready_input(local);
    }
    __aicore__ inline void release(AscendC::LocalTensor<uint8_t> local) { input_.FreeTensor(local); }
    __aicore__ inline AscendC::LocalTensor<uint8_t> alloc_output() { return output_.AllocTensor<uint8_t>(); }
    __aicore__ inline void write(AscendC::LocalTensor<uint8_t> local, GM_ADDR destination, uint32_t bytes) {
        output_.EnQue(local); local = output_.DeQue<uint8_t>();
        auto e = pipe_.FetchEventID(AscendC::HardEvent::S_MTE3);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(e);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(e);
        AscendC::GlobalTensor<uint8_t> gm; gm.SetGlobalBuffer(destination);
        AscendC::DataCopyPad(gm,local,AscendC::DataCopyExtParams{1,bytes,0,0,0});
        e = pipe_.FetchEventID(AscendC::HardEvent::MTE3_S);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(e);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(e);
        output_.FreeTensor(local);
    }
private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN,1> input_;
    AscendC::TQue<AscendC::QuePosition::VECOUT,1> output_;
};
__aicore__ inline uint32_t min32(uint64_t a, uint32_t b) { return a < b ? static_cast<uint32_t>(a) : b; }
__aicore__ inline void zero(AscendC::LocalTensor<uint8_t> local, uint32_t bytes) {
    auto w=local.ReinterpretCast<uint32_t>(); for(uint32_t i=0;i<bytes/4;++i) w.SetValue(i,0);
}
inline uint8_t* region(void* workspace,const MsmPlan& p,MsmRegion r) {
    return static_cast<uint8_t*>(workspace)+p.region_offset[static_cast<uint32_t>(r)];
}
inline uint32_t blocks(const MsmPlan& p,uint64_t tasks) {
    return tasks < p.platform.aiv_cores ? static_cast<uint32_t>(tasks) : p.platform.aiv_cores;
}
} // namespace hyperplonk::ascend910b3::detail::msm_device
