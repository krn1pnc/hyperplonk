// Standalone device metadata gate: extract -> radix group -> CSR -> schedule.
// Compares exact sorted records, RowPtr, ScheduleMeta and the whole TaskPrefix
// (level prefixes plus task-length order) against independent CPU oracles.
// Compute must issue no host transfer, synchronization or allocation; the only
// declared D2H traffic is this test's own post-drain readback.
#include "../g1.h"
#include "../msm.h"
#include "../msm_internal.h"
#include "runtime_audit.h"

#include <acl/acl.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace hp = hyperplonk::ascend910b3;
namespace detail = hp::detail;
namespace audit = hp::test;
using hp::Fr;

bool cleanup_failed = false;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void check(aclError result, const char* operation) {
    if (result != ACL_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(result));
}
void cleanup_check(aclError result, const char* operation) {
    if (result != ACL_SUCCESS) {
        cleanup_failed = true;
        std::cerr << operation << " failed during cleanup: " << result << '\n';
    }
}
uint8_t* align64(void* pointer) {
    return reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(pointer) + 63) & ~uintptr_t(63));
}
size_t round8(size_t n) { return (n + 7) / 8 * 8; }

struct Runtime {
    int32_t device_id = 0;
    aclrtStream stream = nullptr;
    aclrtContext context = nullptr;
    bool initialized = false, selected = false;
    std::vector<void*> device;
    std::vector<std::pair<void*, uint8_t*>> host;

    explicit Runtime(int32_t id) : device_id(id) {
        check(aclInit(nullptr), "aclInit");
        initialized = true;
        try {
            check(aclrtSetDevice(device_id), "aclrtSetDevice");
            selected = true;
            check(aclrtCreateContext(&context, device_id), "aclrtCreateContext");
            check(aclrtCreateStream(&stream), "aclrtCreateStream");
        } catch (...) {
            teardown();
            throw;
        }
    }
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    ~Runtime() { teardown(); }

    void* device_alloc(size_t bytes) {
        if (!bytes) return nullptr;
        void* pointer = nullptr;
        check(aclrtMalloc(&pointer, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
        require(reinterpret_cast<uintptr_t>(pointer) % 64 == 0, "device allocation alignment");
        device.push_back(pointer);
        return pointer;
    }
    uint8_t* host_alloc(size_t bytes) {
        if (!bytes) return nullptr;
        require(bytes <= SIZE_MAX - 63, "host allocation size overflow");
        void* raw = nullptr;
        check(aclrtMallocHost(&raw, bytes + 63), "aclrtMallocHost");
        auto* aligned = align64(raw);
        host.push_back({raw, aligned});
        return aligned;
    }
    void upload(const void* source, void* destination, size_t bytes) {
        if (!bytes) return;
        check(aclrtMemcpyAsync(destination, bytes, source, bytes,
                              ACL_MEMCPY_HOST_TO_DEVICE, stream), "upload");
    }
    void free_device(void* pointer) {
        if (!pointer) return;
        check(aclrtFree(pointer), "aclrtFree");
        device.erase(std::remove(device.begin(), device.end(), pointer), device.end());
    }
    void free_host(uint8_t* aligned) {
        if (!aligned) return;
        for (auto it = host.begin(); it != host.end(); ++it) {
            if (it->second != aligned) continue;
            check(aclrtFreeHost(it->first), "aclrtFreeHost");
            host.erase(it);
            return;
        }
        throw std::runtime_error("releasing unowned pinned host allocation");
    }

private:
    void teardown() {
        bool drained = true;
        if (stream) {
            const auto status = aclrtSynchronizeStream(stream);
            cleanup_check(status, "aclrtSynchronizeStream");
            drained = status == ACL_SUCCESS;
        }
        // An error return is not cancellation: keep every allocation, stream,
        // context and the device selection alive when the drain fails.
        if (!drained) return;
        for (auto it = device.rbegin(); it != device.rend(); ++it)
            cleanup_check(aclrtFree(*it), "aclrtFree");
        device.clear();
        for (auto it = host.rbegin(); it != host.rend(); ++it)
            cleanup_check(aclrtFreeHost(it->first), "aclrtFreeHost");
        host.clear();
        if (stream) {
            cleanup_check(aclrtDestroyStream(stream), "aclrtDestroyStream");
            stream = nullptr;
        }
        if (context) {
            cleanup_check(aclrtDestroyContext(context), "aclrtDestroyContext");
            context = nullptr;
        }
        if (selected) {
            cleanup_check(aclrtResetDevice(device_id), "aclrtResetDevice");
            selected = false;
        }
        if (initialized) {
            cleanup_check(aclFinalize(), "aclFinalize");
            initialized = false;
        }
    }
};

uint8_t* region(uint8_t* workspace, const detail::MsmPlan& plan, detail::MsmRegion name) {
    return workspace + plan.region_offset[static_cast<unsigned>(name)];
}
size_t region_size(const detail::MsmPlan& plan, detail::MsmRegion name) {
    return static_cast<size_t>(plan.region_size[static_cast<unsigned>(name)]);
}
uint32_t read_u32(const uint8_t* p) { uint32_t value; std::memcpy(&value, p, 4); return value; }
uint64_t read_u64(const uint8_t* p) { uint64_t value; std::memcpy(&value, p, 8); return value; }

struct MsmRecord { uint32_t bucket; uint32_t index; };
static_assert(sizeof(MsmRecord) == 8, "MSM record layout");

// Scalar synthesis: only word 0 is nonzero, so every window-0 digit is an exact
// c-bit field of a canonical Fr value.
enum class Pattern { Ramp, Boundary, Zero, LowDigitOne, Bounded };

void fill_scalars(std::vector<Fr>& scalars, size_t n, uint32_t c, Pattern pattern,
                  uint32_t low_bits, uint32_t low_bins) {
    const uint32_t mask = (uint32_t{1} << c) - 1;
    const uint32_t top = (uint32_t{1} << c) >> 1;
    for (size_t i = 0; i < n; ++i) {
        scalars[i] = Fr{};
        uint32_t key = uint32_t(i * 2654435761u);
        switch (pattern) {
        case Pattern::Ramp: break;
        case Pattern::Boundary:
            key = i == 0 ? mask : i == 1 ? top : i == 2 ? 1u : key;
            break;
        case Pattern::Zero: key = 0; break;
        case Pattern::LowDigitOne: key = uint32_t{1} << low_bits; break;
        case Pattern::Bounded:
            key = (uint32_t(i % std::min<uint32_t>(low_bins, 200)) << low_bits)
                | (uint32_t(i) & (low_bins - 1));
            break;
        }
        scalars[i].words[0] = key & mask;
    }
}

// Independent per-bit extract plus stable sort by the full c-bit key.
std::vector<MsmRecord> cpu_extract(const std::vector<Fr>& scalars, uint32_t c) {
    std::vector<MsmRecord> records(scalars.size());
    for (size_t i = 0; i < scalars.size(); ++i) {
        uint32_t digit = 0;
        for (uint32_t bit = 0; bit < c; ++bit) {
            const uint32_t position = bit;
            digit |= ((scalars[i].words[position / 32] >> (position % 32)) & 1u) << bit;
        }
        records[i] = {digit, static_cast<uint32_t>(i)};
    }
    std::stable_sort(records.begin(), records.end(),
                     [](const MsmRecord& a, const MsmRecord& b) { return a.bucket < b.bucket; });
    return records;
}

struct Case {
    size_t n;
    uint32_t window_bits;
    Pattern pattern;
    std::string name;
};

// Count 0 must stay a pure no-op: no region, no transfer, no allocation.
void zero_count_case(Runtime& runtime) {
    detail::MsmPlan plan{};
    require(detail::build_msm_plan(0, plan, 0), "metadata zero-count plan failed");
    require(!plan.workspace_bytes && !plan.chunks && !plan.radix_pass_bits[0] &&
            !plan.radix_route && !plan.buckets, "metadata zero-count plan shape");
    void* dummy = runtime.device_alloc(64);
    audit::begin_runtime_audit(nullptr, 0);
    detail::launch_msm_extract(runtime.stream, nullptr, dummy, plan, 0);
    detail::launch_msm_radix_group(runtime.stream, dummy, plan);
    detail::launch_msm_rows(runtime.stream, dummy, plan);
    detail::launch_msm_schedule(runtime.stream, dummy, plan);
    const auto counts = audit::end_runtime_audit();
    require(!counts.h2d_calls && !counts.d2h_calls && !counts.other_copy_calls &&
            !counts.unexpected_copy_calls && !counts.stream_sync_calls && !counts.device_sync_calls &&
            !counts.malloc_calls && !counts.free_calls && !counts.host_malloc_calls &&
            !counts.host_free_calls, "metadata zero-count pipeline touched the runtime");
    // Four no-op kernels are queued; establish completion before releasing the
    // dummy buffer. A failed drain retains every resource for the owned teardown.
    const aclError drained = aclrtSynchronizeStream(runtime.stream);
    if (drained != ACL_SUCCESS) {
        cleanup_failed = true;
        std::cerr << "metadata zero-count drain failed; retaining resources: " << drained << '\n';
        throw std::runtime_error("metadata zero-count drain failed");
    }
    runtime.free_device(dummy);
    std::cout << "MSM_METADATA_CHECK_PASS case=zero-count\n";
}

void metadata_case(Runtime& runtime, const Case& test_case) {
    constexpr uint32_t partition_records = detail::msm_radix_partition_records;
    detail::MsmPlan plan{};
    require(detail::build_msm_plan(test_case.n, plan, test_case.window_bits),
            "metadata platform plan failed");
    const uint32_t c = plan.window_bits;
    const size_t n = plan.count;
    require(n == test_case.n && c == test_case.window_bits, "metadata plan identity");
    // Balanced digit plan, recomputed independently.
    const uint32_t passes = (c + 7) / 8, q = c / passes, r = c % passes;
    const uint32_t max_bits = q + (r ? 1u : 0u);
    const uint32_t route = c <= 8 ? 0u : (c <= 16 ? 1u : 2u);
    require(plan.radix_passes == passes && plan.radix_route == route
                && plan.radix_max_bits == max_bits && plan.radix_bins == (uint32_t{1} << max_bits)
                && plan.radix_low_bits == q + (r ? 1u : 0u),
            "metadata radix plan shape");
    require(plan.chunks == (n + detail::msm_radix_chunk_records - 1) / detail::msm_radix_chunk_records,
            "metadata radix chunk count");
    const uint32_t low_bits = plan.radix_pass_bits[0];
    const uint32_t low_bins = uint32_t{1} << low_bits;
    const uint32_t high_bits = plan.radix_high_bits;
    const std::string label = "msm_metadata case=" + test_case.name + " n=" + std::to_string(n)
        + " c=" + std::to_string(c) + " route=" + std::to_string(route);

    auto* workspace = static_cast<uint8_t*>(runtime.device_alloc(plan.workspace_bytes));
    std::vector<Fr> scalars(n);
    fill_scalars(scalars, n, c, test_case.pattern, low_bits, low_bins);
    auto* device_scalars = static_cast<Fr*>(runtime.device_alloc(n * sizeof(Fr)));
    runtime.upload(scalars.data(), device_scalars, n * sizeof(Fr));
    check(aclrtSynchronizeStream(runtime.stream), "metadata scalar upload sync");

    // ---- Device metadata pipeline under a strict zero-transfer audit ----
    audit::begin_runtime_audit(nullptr, 0);
    detail::launch_msm_extract(runtime.stream, device_scalars, workspace, plan, 0);
    detail::launch_msm_radix_group(runtime.stream, workspace, plan);
    detail::launch_msm_rows(runtime.stream, workspace, plan);
    detail::launch_msm_schedule(runtime.stream, workspace, plan);
    const auto compute = audit::end_runtime_audit();
    require(!compute.h2d_calls && !compute.h2d_bytes && !compute.d2h_calls && !compute.d2h_bytes &&
            !compute.other_copy_calls && !compute.unexpected_copy_calls &&
            !compute.stream_sync_calls && !compute.device_sync_calls &&
            !compute.malloc_calls && !compute.free_calls &&
            !compute.host_malloc_calls && !compute.host_free_calls,
            label + " metadata pipeline transferred, synchronized or allocated");

    // ---- Declared readback 1: control word (selects the final record buffer) ----
    uint8_t* control = runtime.host_alloc(64);
    {
        const audit::RuntimeCopyAllowance allowance[]{
            {region(workspace, plan, detail::MsmRegion::RadixControl), control, 64, false}};
        audit::begin_runtime_audit(allowance, 1);
        check(aclrtMemcpyAsync(control, 64, region(workspace, plan, detail::MsmRegion::RadixControl),
                              64, ACL_MEMCPY_DEVICE_TO_HOST, runtime.stream), "control readback");
        check(aclrtSynchronizeStream(runtime.stream), "control readback sync");
        const auto counts = audit::end_runtime_audit();
        require(counts.d2h_calls == 1 && counts.d2h_bytes == 64 && !counts.h2d_calls &&
                !counts.h2d_bytes && counts.stream_sync_calls == 1 && !counts.other_copy_calls &&
                !counts.unexpected_copy_calls && !counts.malloc_calls && !counts.free_calls &&
                !counts.host_malloc_calls && !counts.host_free_calls,
                label + " control readback used undeclared traffic");
    }
    const uint32_t parity = read_u32(control);
    const uint32_t fallback = read_u32(control + 4);
    const uint32_t max_partition = read_u32(control + 8);
    require(parity <= 1, label + " final parity range");

    // ---- Declared readback 2: records, RowPtr, ScheduleMeta, TaskPrefix ----
    const size_t record_bytes = round8(n) * sizeof(MsmRecord);
    const size_t row_bytes = round8(plan.buckets + 1) * sizeof(uint64_t);
    const size_t schedule_bytes = region_size(plan, detail::MsmRegion::ScheduleMeta);
    const size_t prefix_bytes = region_size(plan, detail::MsmRegion::TaskPrefix);
    require(schedule_bytes >= detail::msm_schedule_meta_header_bytes && prefix_bytes != 0,
            label + " schedule region capacity");
    const auto final_records = parity ? detail::MsmRegion::RecordsB : detail::MsmRegion::RecordsA;
    uint8_t* records_host = runtime.host_alloc(record_bytes);
    uint8_t* rows_host = runtime.host_alloc(row_bytes);
    uint8_t* schedule_host = runtime.host_alloc(schedule_bytes);
    uint8_t* prefix_host = runtime.host_alloc(prefix_bytes);
    {
        const std::array<audit::RuntimeCopyAllowance, 4> allowances{{
            {region(workspace, plan, final_records), records_host, record_bytes, false},
            {region(workspace, plan, detail::MsmRegion::RowPtr), rows_host, row_bytes, false},
            {region(workspace, plan, detail::MsmRegion::ScheduleMeta), schedule_host,
             schedule_bytes, false},
            {region(workspace, plan, detail::MsmRegion::TaskPrefix), prefix_host, prefix_bytes,
             false}}};
        audit::begin_runtime_audit(allowances.data(), allowances.size());
        struct Copy { const void* device; uint8_t* host; size_t bytes; };
        const std::array<Copy, 4> copies{{
            {region(workspace, plan, final_records), records_host, record_bytes},
            {region(workspace, plan, detail::MsmRegion::RowPtr), rows_host, row_bytes},
            {region(workspace, plan, detail::MsmRegion::ScheduleMeta), schedule_host, schedule_bytes},
            {region(workspace, plan, detail::MsmRegion::TaskPrefix), prefix_host, prefix_bytes}}};
        for (const auto& copy : copies)
            check(aclrtMemcpyAsync(copy.host, copy.bytes, copy.device, copy.bytes,
                                  ACL_MEMCPY_DEVICE_TO_HOST, runtime.stream), "metadata readback");
        check(aclrtSynchronizeStream(runtime.stream), "metadata readback sync");
        const auto counts = audit::end_runtime_audit();
        const uint64_t expected_bytes = record_bytes + row_bytes + schedule_bytes + prefix_bytes;
        require(counts.d2h_calls == 4 && counts.d2h_bytes == expected_bytes && !counts.h2d_calls &&
                !counts.h2d_bytes && counts.stream_sync_calls == 1 && !counts.other_copy_calls &&
                !counts.unexpected_copy_calls && !counts.malloc_calls && !counts.free_calls &&
                !counts.host_malloc_calls && !counts.host_free_calls,
                label + " metadata readback used undeclared traffic");
    }

    // ---- Independent CPU oracle ----
    const auto expected = cpu_extract(scalars, c);
    const auto* dense = reinterpret_cast<const MsmRecord*>(records_host);
    for (size_t i = 0; i < n; ++i)
        require(dense[i].bucket == expected[i].bucket && dense[i].index == expected[i].index,
                label + " sorted record mismatch");
    for (size_t i = n; i < round8(n); ++i)
        require(dense[i].bucket == 0 && dense[i].index == UINT32_MAX, label + " dense tail padding");
    const auto* rows = reinterpret_cast<const uint64_t*>(rows_host);
    std::vector<uint64_t> model_rows(plan.buckets + 1);
    for (uint32_t b = 0; b <= plan.buckets; ++b) {
        model_rows[b] = std::lower_bound(expected.begin(), expected.end(), b,
            [](const MsmRecord& record, uint32_t key) { return record.bucket < key; })
            - expected.begin();
        require(rows[b] == model_rows[b], label + " CSR lower_bound mismatch");
    }
    const size_t stride = round8(plan.buckets + 1);
    for (size_t i = plan.buckets + 1; i < stride; ++i)
        require(rows[i] == n, label + " CSR tail padding");

    std::vector<uint64_t> prefix(prefix_bytes / sizeof(uint64_t), UINT64_MAX);
    detail::MsmSchedule schedule{};
    require(detail::make_msm_schedule(plan, model_rows.data(), prefix.data(), schedule),
            label + " CPU schedule oracle rejected CSR");
    require(read_u64(schedule_host) == schedule.task_count[0] &&
            read_u64(schedule_host + 8) == schedule.task_count[1] &&
            read_u64(schedule_host + 16) == schedule.task_count[2] &&
            read_u64(schedule_host + 24) == schedule.task_count[3] &&
            read_u64(schedule_host + 32) == schedule.task_count[4] &&
            read_u64(schedule_host + 40) == schedule.task_count[5],
            label + " ScheduleMeta task counts");
    require(read_u32(schedule_host + 48) == schedule.merge_levels,
            label + " ScheduleMeta merge levels");
    require(read_u64(schedule_host + 56) == schedule.order_offset,
            label + " ScheduleMeta order offset");
    const size_t written_bytes = stride * (schedule.merge_levels + 1) * sizeof(uint64_t)
        + schedule.task_count[0] * sizeof(uint32_t);
    require(written_bytes <= prefix_bytes, label + " schedule metadata capacity");
    const auto* expected_prefix = reinterpret_cast<const uint8_t*>(prefix.data());
    const auto mismatch = std::mismatch(prefix_host, prefix_host + written_bytes,
                                        expected_prefix);
    if (mismatch.first != prefix_host + written_bytes)
        throw std::runtime_error(label + " TaskPrefix bytes (level prefixes and task-length order)"
            + " first_bad_byte=" + std::to_string(mismatch.first - prefix_host)
            + " actual=" + std::to_string(unsigned(*mismatch.first))
            + " expected=" + std::to_string(unsigned(*mismatch.second)));

    // ---- Route/fallback evidence ----
    if (route == 1) {
        std::vector<uint32_t> partitions(size_t{1} << high_bits, 0);
        for (const auto& record : expected) ++partitions[record.bucket >> low_bits];
        const uint32_t model_max = *std::max_element(partitions.begin(), partitions.end());
        require(fallback == (model_max > partition_records ? 1u : 0u), label + " fallback decision");
        require(max_partition == model_max, label + " maximum coarse partition");
        require(parity == (fallback ? 0u : 1u), label + " parity for route-1 branch");
    } else {
        require(parity == 1, label + " parity for routes 0/2");
        require(fallback == 0, label + " fallback flag on routes 0/2");
    }

    runtime.free_host(control);
    runtime.free_host(records_host);
    runtime.free_host(rows_host);
    runtime.free_host(schedule_host);
    runtime.free_host(prefix_host);
    runtime.free_device(device_scalars);
    runtime.free_device(workspace);
    std::cout << "MSM_METADATA_CHECK_PASS " << label << " fallback=" << fallback
              << " max_partition=" << max_partition << " parity=" << parity
              << " tasks0=" << schedule.task_count[0] << " merge_levels=" << schedule.merge_levels
              << '\n';
}
} // namespace

int main(int argc, char** argv) {
    try {
        int32_t device = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string option(argv[i]);
            if (option == "--help") {
                std::cout << "usage: msm_metadata_check [--device ACL_ID]\n";
                return 0;
            }
            require(i + 1 < argc && option == "--device", "usage: msm_metadata_check [--device ACL_ID]");
            const std::string value(argv[++i]);
            require(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos,
                    "invalid ACL device ID");
            size_t consumed = 0;
            const auto parsed = std::stoll(value, &consumed);
            require(consumed == value.size() && parsed >= 0 &&
                    parsed <= std::numeric_limits<int32_t>::max(), "invalid ACL device ID");
            device = static_cast<int32_t>(parsed);
        }
        std::vector<Case> cases;
        for (size_t n : {1u, 7u, 8u, 9u, 2047u, 2048u, 2049u, 4097u, 65537u})
            for (uint32_t c : {2u, 4u, 8u, 9u, 12u, 16u, 17u, 20u})
                cases.push_back({n, c, Pattern::Ramp, "ramp"});
        for (size_t n : {63u, 64u, 65u, 8191u, 8192u, 8193u})
            cases.push_back({n, 16, Pattern::Boundary, "chunk-boundary"});
        for (uint32_t c = 2; c <= 20; ++c)
            cases.push_back({2049, c, Pattern::Boundary, "boundary-all-c"});
        // Forced fallback: one coarse partition above 8192 records, digit 0 and 1.
        cases.push_back({65537, 16, Pattern::Zero, "fallback-coarse-digit0"});
        cases.push_back({65537, 16, Pattern::LowDigitOne, "fallback-coarse-digit1"});
        // Exact partition guard boundary: 8192 stays fast, 8193 takes the fallback.
        cases.push_back({8192, 16, Pattern::Zero, "partition-limit-fast"});
        cases.push_back({8193, 16, Pattern::Zero, "partition-limit-fallback"});
        // Route-1 fast path with bounded coarse partitions.
        cases.push_back({65537, 16, Pattern::Bounded, "fast-bounded"});
        cases.push_back({4097, 16, Pattern::Bounded, "fast-bounded-small"});

        {
            Runtime runtime(device);
            zero_count_case(runtime);
            for (const auto& test_case : cases) metadata_case(runtime, test_case);
        }
        require(!cleanup_failed, "ACL resource cleanup failed");
        std::cout << "MSM_METADATA_CHECK_ALL_PASS cases=" << cases.size() << '\n';
        return 0;
    } catch (const std::exception& error) {
        audit::end_runtime_audit();
        std::cerr << "MSM_METADATA_CHECK_FAIL: " << error.what() << '\n';
        return 1;
    }
}
