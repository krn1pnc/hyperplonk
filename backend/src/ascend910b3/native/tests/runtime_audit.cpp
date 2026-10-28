#include "runtime_audit.h"

#include <acl/acl_rt.h>

namespace {
bool auditing = false;
hyperplonk::ascend910b3::test::RuntimeCounts counts;

void record_copy(size_t bytes, aclrtMemcpyKind kind)
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
}
} // namespace

namespace hyperplonk::ascend910b3::test {
void begin_runtime_audit()
{
    counts = {};
    auditing = true;
}

RuntimeCounts end_runtime_audit()
{
    auditing = false;
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

aclError __wrap_aclrtMemcpy(void* dst, size_t capacity, const void* src,
                           size_t bytes, aclrtMemcpyKind kind)
{
    record_copy(bytes, kind);
    return __real_aclrtMemcpy(dst, capacity, src, bytes, kind);
}

aclError __wrap_aclrtMemcpyAsync(void* dst, size_t capacity, const void* src,
                                size_t bytes, aclrtMemcpyKind kind, aclrtStream stream)
{
    record_copy(bytes, kind);
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
} // extern "C"
