#pragma once

#include <cstddef>
#include <cstdint>

namespace hyperplonk::ascend910b3::test {

// Counts only calls made explicitly by the test and linked static math target,
// not calls made internally by dynamically linked CANN libraries.
struct RuntimeCounts {
    uint64_t h2d_calls = 0;
    uint64_t h2d_bytes = 0;
    uint64_t d2h_calls = 0;
    uint64_t d2h_bytes = 0;
    uint64_t other_copy_calls = 0;
    uint64_t stream_sync_calls = 0;
    uint64_t device_sync_calls = 0;
    uint64_t malloc_calls = 0;
    uint64_t free_calls = 0;
    uint64_t host_malloc_calls = 0;
    uint64_t host_free_calls = 0;
    uint64_t unexpected_copy_calls = 0;
};

struct RuntimeCopyAllowance {
    const void* device_begin;
    const void* host_begin;
    size_t max_bytes;
    bool host_to_device;
};

// Each host thread owns independent counters and borrowed allowance storage;
// concurrent callers cannot overwrite another thread's audit scope.
void begin_runtime_audit();
// Borrows allowances until end_runtime_audit(); an empty list rejects every copy.
// Copies must match both starting addresses, direction, and 0 < bytes <= max_bytes.
void begin_runtime_audit(const RuntimeCopyAllowance* allowances, size_t count);
RuntimeCounts end_runtime_audit();

} // namespace hyperplonk::ascend910b3::test
