#include "runtime_audit.h"

#include <acl/acl_rt.h>

namespace {
thread_local bool auditing = false;
thread_local hyperplonk::ascend910b3::test::RuntimeCounts counts;
thread_local bool checking_copies = false;
thread_local const hyperplonk::ascend910b3::test::RuntimeCopyAllowance* copy_allowances = nullptr;
thread_local size_t copy_allowance_count = 0;

void record_copy(const void* dst, const void* src, size_t bytes, aclrtMemcpyKind kind)
{
    if (!auditing) return;
    if (kind == ACL_MEMCPY_HOST_TO_DEVICE) {
        ++counts.h2d_calls;
        counts.h2d_bytes += bytes;
    } else if (kind == ACL_MEMCPY_DEVICE_TO_HOST) {
        ++counts.d2h_calls;
        counts.d2h_bytes += bytes;
    } else {
        ++counts.other_copy_calls;
    }
    if (!checking_copies) return;
    if (bytes != 0 && (kind == ACL_MEMCPY_HOST_TO_DEVICE ||
                       kind == ACL_MEMCPY_DEVICE_TO_HOST)) {
        const bool host_to_device = kind == ACL_MEMCPY_HOST_TO_DEVICE;
        const void* device = host_to_device ? dst : src;
        const void* host = host_to_device ? src : dst;
        for (size_t i = 0; copy_allowances != nullptr && i < copy_allowance_count; ++i) {
            const auto& allowance = copy_allowances[i];
            if (allowance.host_to_device == host_to_device &&
                allowance.device_begin == device && allowance.host_begin == host &&
                bytes <= allowance.max_bytes) {
                return;
            }
        }
    }
    ++counts.unexpected_copy_calls;
}
} // namespace

namespace hyperplonk::ascend910b3::test {
void begin_runtime_audit()
{
    counts = {};
    checking_copies = false;
    copy_allowances = nullptr;
    copy_allowance_count = 0;
    auditing = true;
}

void begin_runtime_audit(const RuntimeCopyAllowance* allowances, size_t count)
{
    counts = {};
    checking_copies = true;
    copy_allowances = allowances;
    copy_allowance_count = count;
    auditing = true;
}

RuntimeCounts end_runtime_audit()
{
    auditing = false;
    checking_copies = false;
    copy_allowances = nullptr;
    copy_allowance_count = 0;
    return counts;
}
} // namespace hyperplonk::ascend910b3::test

extern "C" {
aclError __real_aclrtMemcpy(void*, size_t, const void*, size_t, aclrtMemcpyKind);
aclError __real_aclrtMemcpyAsync(void*, size_t, const void*, size_t, aclrtMemcpyKind, aclrtStream);
aclError __real_aclrtSynchronizeStream(aclrtStream);
aclError __real_aclrtSynchronizeDevice(void);
aclError __real_aclrtMalloc(void**, size_t, aclrtMemMallocPolicy);
aclError __real_aclrtFree(void*);
aclError __real_aclrtMallocHost(void**, size_t);
aclError __real_aclrtFreeHost(void*);

aclError __wrap_aclrtMemcpy(void* dst, size_t capacity, const void* src,
                           size_t bytes, aclrtMemcpyKind kind)
{
    record_copy(dst, src, bytes, kind);
    return __real_aclrtMemcpy(dst, capacity, src, bytes, kind);
}

aclError __wrap_aclrtMemcpyAsync(void* dst, size_t capacity, const void* src,
                                size_t bytes, aclrtMemcpyKind kind, aclrtStream stream)
{
    record_copy(dst, src, bytes, kind);
    return __real_aclrtMemcpyAsync(dst, capacity, src, bytes, kind, stream);
}

aclError __wrap_aclrtSynchronizeStream(aclrtStream stream)
{
    if (auditing) ++counts.stream_sync_calls;
    return __real_aclrtSynchronizeStream(stream);
}

aclError __wrap_aclrtSynchronizeDevice()
{
    if (auditing) ++counts.device_sync_calls;
    return __real_aclrtSynchronizeDevice();
}

aclError __wrap_aclrtMalloc(void** ptr, size_t bytes, aclrtMemMallocPolicy policy)
{
    if (auditing) ++counts.malloc_calls;
    return __real_aclrtMalloc(ptr, bytes, policy);
}

aclError __wrap_aclrtFree(void* ptr)
{
    if (auditing) ++counts.free_calls;
    return __real_aclrtFree(ptr);
}

aclError __wrap_aclrtMallocHost(void** ptr, size_t bytes)
{
    if (auditing) ++counts.host_malloc_calls;
    return __real_aclrtMallocHost(ptr, bytes);
}

aclError __wrap_aclrtFreeHost(void* ptr)
{
    if (auditing) ++counts.host_free_calls;
    return __real_aclrtFreeHost(ptr);
}
} // extern "C"
