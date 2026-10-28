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
};

// The smoke harness enqueues on one host thread; independent streams do not
// imply concurrent host calls into these test-only counters.
void begin_runtime_audit();
RuntimeCounts end_runtime_audit();

} // namespace hyperplonk::ascend910b3::test
