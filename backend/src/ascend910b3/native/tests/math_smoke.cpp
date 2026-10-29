#include "../math_kernels.h"
#include "../g1.h"
#include "../math_tiling.h"
#include "../msm_internal.h"
#include "runtime_audit.h"
#include "smoke_kernels.h"

#include <acl/acl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <type_traits>

namespace {
namespace math = hyperplonk::ascend910b3;
namespace detail = hyperplonk::ascend910b3::detail;
namespace audit = hyperplonk::ascend910b3::test;
using math::Fr;
using Tiling = AscendC::tiling::TCubeTiling;
constexpr std::array<size_t, 20> case_counts{0, 1, 2, 31, 32, 33, 63, 64, 65, 255, 256, 257, 513, 4097, 47, 48, 49, 95, 96, 97};
constexpr uint32_t modulus[8] = {
    0x00000001, 0xffffffff, 0xfffe5bfe, 0x53bda402,
    0x09a1d805, 0x3339d808, 0x299d7d48, 0x73eda753
};
bool cleanup_failed = false;

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

void check(aclError result, const char* operation)
{
    if (result != ACL_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(result));
}

void cleanup_check(aclError result, const char* operation)
{
    if (result != ACL_SUCCESS) {
        cleanup_failed = true;
        std::cerr << operation << " failed during cleanup: " << result << '\n';
    }
}

size_t field_bytes(size_t count)
{
    size_t result = 0;
    require(detail::checked_field_bytes<Fr>(count, result), "invalid field array size");
    return result;
}

bool canonical(const Fr& value)
{
    for (size_t i = 8; i-- > 0;) {
        if (value.words[i] < modulus[i]) return true;
        if (value.words[i] > modulus[i]) return false;
    }
    return false;
}

Fr small(uint32_t value)
{
    Fr out{};
    out.words[0] = value;
    return out;
}

Fr minus_one()
{
    Fr out{};
    std::copy(std::begin(modulus), std::end(modulus), out.words);
    --out.words[0];
    return out;
}

uint8_t limb(const Fr& value, size_t index)
{
    return static_cast<uint8_t>(value.words[index / 4] >> (8 * (index % 4)));
}

std::vector<Fr> read_fields(const std::filesystem::path& path, size_t count)
{
    const size_t bytes = field_bytes(count);
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(input.is_open(), "missing fixture: " + path.string());
    require(input.tellg() == static_cast<std::streamoff>(bytes),
            "incorrect fixture length: " + path.string());
    input.seekg(0);
    std::vector<uint8_t> encoded(bytes);
    if (bytes != 0) {
        input.read(reinterpret_cast<char*>(encoded.data()), static_cast<std::streamsize>(bytes));
        require(static_cast<size_t>(input.gcount()) == bytes, "truncated fixture: " + path.string());
    }
    std::vector<Fr> values(count);
    for (size_t i = 0; i < count; ++i) {
        for (size_t w = 0; w < 8; ++w) {
            uint32_t word = 0;
            for (size_t b = 0; b < 4; ++b)
                word |= uint32_t(encoded[i * 32 + w * 4 + b]) << (8 * b);
            values[i].words[w] = word;
        }
        require(canonical(values[i]), "noncanonical fixture: " + path.string());
    }
    return values;
}

void compare_fields(const Fr* actual, const std::vector<Fr>& expected, const std::string& label)
{
    for (size_t i = 0; i < expected.size(); ++i) {
        require(canonical(actual[i]), label + " noncanonical result at " + std::to_string(i));
        require(std::memcmp(actual + i, &expected[i], sizeof(Fr)) == 0,
                label + " mismatch at " + std::to_string(i));
    }
}

// Host memory used by asynchronous ACL copies is pinned and retained until all
// streams finish. These allocations are test storage, never a production fallback.
class Runtime {
public:
    explicit Runtime(int32_t device) : device_(device)
    {
        check(aclInit(nullptr), "aclInit");
        initialized_ = true;
        try {
            check(aclrtSetDevice(device_), "aclrtSetDevice");
            device_set_ = true;
            check(aclrtCreateContext(&context_, device_), "aclrtCreateContext");
        } catch (...) {
            finish();
            throw;
        }
    }
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    ~Runtime() { finish(); }

    aclrtStream stream()
    {
        aclrtStream result = nullptr;
        check(aclrtCreateStream(&result), "aclrtCreateStream");
        streams_.push_back(result);
        return result;
    }

    template<class T> T* allocate(size_t count)
    {
        if (count == 0) return nullptr;
        require(count <= std::numeric_limits<size_t>::max() / sizeof(T), "allocation size overflow");
        void* result = nullptr;
        check(aclrtMalloc(&result, count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
        device_allocations_.push_back(result);
        return static_cast<T*>(result);
    }

    template<class T> T* host(size_t count)
    {
        if (count == 0) return nullptr;
        require(count <= std::numeric_limits<size_t>::max() / sizeof(T), "host size overflow");
        void* result = nullptr;
        check(aclrtMallocHost(&result, count * sizeof(T)), "aclrtMallocHost");
        host_allocations_.push_back(result);
        return static_cast<T*>(result);
    }

    template<class T> void upload(aclrtStream stream, T* dst, const T* src, size_t count)
    {
        if (count == 0) return;
        T* pinned = host<T>(count);
        std::memcpy(pinned, src, count * sizeof(T));
        check(aclrtMemcpyAsync(dst, count * sizeof(T), pinned, count * sizeof(T),
                              ACL_MEMCPY_HOST_TO_DEVICE, stream), "upload");
    }

    template<class T> T* download(aclrtStream stream, const T* src, size_t count)
    {
        if (count == 0) return nullptr;
        T* pinned = host<T>(count);
        copy_down(stream, pinned, src, count);
        return pinned;
    }

    template<class T> void copy_down(aclrtStream stream, T* dst, const T* src, size_t count)
    {
        if (count == 0) return;
        check(aclrtMemcpyAsync(dst, count * sizeof(T), src, count * sizeof(T),
                              ACL_MEMCPY_DEVICE_TO_HOST, stream), "download");
    }

    // Release only storage allocated by a completed MSM diagnostic/case, keeping
    // the original smoke queue and its tilings alive. Call after its last sync.
    using StorageMark = std::pair<size_t, size_t>;
    StorageMark storage_mark() const { return {device_allocations_.size(), host_allocations_.size()}; }
    void release_storage(StorageMark mark)
    {
        while (device_allocations_.size() > mark.first) {
            check(aclrtFree(device_allocations_.back()), "release completed test device storage");
            device_allocations_.pop_back();
        }
        while (host_allocations_.size() > mark.second) {
            check(aclrtFreeHost(host_allocations_.back()), "release completed test host storage");
            host_allocations_.pop_back();
        }
    }

private:
    void finish()
    {
        for (auto stream : streams_)
            cleanup_check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
        for (auto it = device_allocations_.rbegin(); it != device_allocations_.rend(); ++it)
            cleanup_check(aclrtFree(*it), "aclrtFree");
        device_allocations_.clear();
        for (auto it = host_allocations_.rbegin(); it != host_allocations_.rend(); ++it)
            cleanup_check(aclrtFreeHost(*it), "aclrtFreeHost");
        host_allocations_.clear();
        for (auto it = streams_.rbegin(); it != streams_.rend(); ++it)
            cleanup_check(aclrtDestroyStream(*it), "aclrtDestroyStream");
        streams_.clear();
        if (context_) {
            cleanup_check(aclrtDestroyContext(context_), "aclrtDestroyContext");
            context_ = nullptr;
        }
        if (device_set_) {
            cleanup_check(aclrtResetDevice(device_), "aclrtResetDevice");
            device_set_ = false;
        }
        if (initialized_) {
            cleanup_check(aclFinalize(), "aclFinalize");
            initialized_ = false;
        }
    }
    int32_t device_;
    bool initialized_ = false;
    bool device_set_ = false;
    aclrtContext context_ = nullptr;
    std::vector<aclrtStream> streams_;
    std::vector<void*> device_allocations_;
    std::vector<void*> host_allocations_;
};

struct Queue {
    aclrtStream stream;
    uint8_t* workspace;
    Tiling* dot;
    Tiling* scale;
    uint32_t cores;

    Queue(Runtime& runtime, const Tiling& dot_host, const Tiling& scale_host, uint32_t aiv)
        : stream(runtime.stream()), workspace(runtime.allocate<uint8_t>(detail::workspace_bytes)),
          dot(runtime.allocate<Tiling>(1)), scale(runtime.allocate<Tiling>(1)), cores(aiv)
    {
        require(reinterpret_cast<uintptr_t>(workspace) % 512 == 0, "workspace lacks 512-byte alignment");
        runtime.upload(stream, dot, &dot_host, 1);
        runtime.upload(stream, scale, &scale_host, 1);
    }

    uint32_t blocks(size_t count) const
    {
        return static_cast<uint32_t>(std::min<size_t>(cores, std::max<size_t>(1, (count + 255) / 256)));
    }
};

bool inverse_case(size_t count)
{
    return count == 0 || count == 1 || count == 2 || count == 33 ||
           count == 64 || count == 65 || count == 257 ||
           count == 47 || count == 48 || count == 49 ||
           count == 95 || count == 96 || count == 97;
}

void test_sizes()
{
    size_t bytes = 123;
    require(detail::checked_field_bytes<Fr>(0, bytes) && bytes == 0, "empty size rejected");
    require(detail::checked_field_bytes<Fr>(4097, bytes) && bytes == 4097 * 32, "valid size rejected");
    const size_t limit = static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(Fr);
    require(detail::checked_field_bytes<Fr>(limit, bytes) && bytes == limit * sizeof(Fr),
            "maximum addressable size rejected");
    bytes = 123;
    require(!detail::checked_field_bytes<Fr>(limit + 1, bytes) && bytes == 123,
            "PTRDIFF_MAX overflow accepted or wrote output");
    require(!detail::checked_field_bytes<Fr>(std::numeric_limits<size_t>::max() / 32 + 1, bytes)
            && bytes == 123, "multiplication overflow accepted or wrote output");
    // Deliberately do not pass a fictitious oversized device allocation to a launch.
}
void require_no_runtime_transfers(const audit::RuntimeCounts& counts);

void test_fixture(Runtime& runtime, Queue& queue, const std::filesystem::path& root, size_t count)
{
    const auto path = root / ("case-" + std::to_string(count));
    const auto lhs = read_fields(path / "lhs.bin", count);
    const auto rhs = read_fields(path / "rhs.bin", count);
    const auto scalar = read_fields(path / "scalar.bin", 1);
    const std::array<std::string, 4> names{"add", "sub", "mul", "scale"};
    std::array<std::vector<Fr>, 4> expected;
    for (size_t i = 0; i < names.size(); ++i)
        expected[i] = read_fields(path / (names[i] + ".bin"), count);
    const auto dot_expected = read_fields(path / "inner_product.bin", 1);
    const bool do_inverse = inverse_case(count);
    const auto inverse_input = do_inverse ? read_fields(path / "inverse_input.bin", count) : std::vector<Fr>{};
    const auto inverse_expected = do_inverse ? read_fields(path / "inverse.bin", count) : std::vector<Fr>{};

    Fr* a = runtime.allocate<Fr>(count);
    Fr* b = runtime.allocate<Fr>(count);
    Fr* c = runtime.allocate<Fr>(1);
    runtime.upload(queue.stream, a, lhs.data(), count);
    runtime.upload(queue.stream, b, rhs.data(), count);
    runtime.upload(queue.stream, c, scalar.data(), 1);
    std::array<Fr*, 4> output{};
    for (auto& ptr : output) ptr = runtime.allocate<Fr>(count);
    Fr* dot = runtime.allocate<Fr>(1);
    const Fr poison = minus_one();
    runtime.upload(queue.stream, dot, &poison, 1);
    Fr* inv_input = do_inverse ? runtime.allocate<Fr>(count) : nullptr;
    Fr* inv_output = do_inverse ? runtime.allocate<Fr>(count) : nullptr;
    uint8_t* mask = do_inverse ? runtime.allocate<uint8_t>(count) : nullptr;
    if (do_inverse) runtime.upload(queue.stream, inv_input, inverse_input.data(), count);
    const uint32_t blocks = queue.blocks(count);
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
        audit::begin_runtime_audit();
        detail::launch_add(queue.stream, a, b, output[0], count, blocks);
        detail::launch_sub(queue.stream, a, b, output[1], count, blocks);
        detail::launch_mul(queue.stream, a, b, output[2], count, blocks);
        detail::launch_scalar_vector_mul(queue.stream, count ? c : nullptr, a, output[3], count,
                                        queue.workspace, queue.scale, blocks);
        detail::launch_inner_product(queue.stream, a, b, dot, count, queue.workspace, queue.dot, blocks);
        if (do_inverse)
            detail::launch_inverse(queue.stream, inv_input, inv_output, mask, count, blocks);
        require_no_runtime_transfers(audit::end_runtime_audit());
        std::array<Fr*, 4> observed{};
        for (size_t i = 0; i < output.size(); ++i)
            observed[i] = runtime.download(queue.stream, output[i], count);
        Fr* observed_dot = runtime.download(queue.stream, dot, 1);
        Fr* observed_inverse = do_inverse ? runtime.download(queue.stream, inv_output, count) : nullptr;
        uint8_t* observed_mask = do_inverse ? runtime.download(queue.stream, mask, count) : nullptr;
        check(aclrtSynchronizeStream(queue.stream), "fixture synchronize");
        for (size_t i = 0; i < names.size(); ++i)
            compare_fields(observed[i], expected[i], names[i] + " count=" + std::to_string(count));
        compare_fields(observed_dot, dot_expected, "dot count=" + std::to_string(count));
        if (do_inverse) {
            compare_fields(observed_inverse, inverse_expected, "inverse count=" + std::to_string(count));
            for (size_t i = 0; i < count; ++i)
                require(observed_mask[i] == 1, "nonzero inverse mask mismatch");
        }
    }
    Fr* after_a = runtime.download(queue.stream, a, count);
    Fr* after_b = runtime.download(queue.stream, b, count);
    Fr* after_c = runtime.download(queue.stream, c, 1);
    Fr* after_inv = do_inverse ? runtime.download(queue.stream, inv_input, count) : nullptr;
    check(aclrtSynchronizeStream(queue.stream), "immutability synchronize");
    compare_fields(after_a, lhs, "immutable lhs");
    compare_fields(after_b, rhs, "immutable rhs");
    compare_fields(after_c, scalar, "immutable scalar");
    if (do_inverse) compare_fields(after_inv, inverse_input, "immutable inverse input");
}

void test_inverse_zeros(Runtime& runtime, Queue& queue)
{
    for (const std::vector<Fr>& input : {std::vector<Fr>{small(0)},
                                        std::vector<Fr>{small(0), small(1), minus_one(), small(0)}}) {
        Fr* values = runtime.allocate<Fr>(input.size());
        Fr* out = runtime.allocate<Fr>(input.size());
        uint8_t* mask = runtime.allocate<uint8_t>(input.size());
        runtime.upload(queue.stream, values, input.data(), input.size());
        std::vector<Fr> poison(input.size(), minus_one());
        std::vector<uint8_t> poison_mask(input.size(), 0xa5);
        runtime.upload(queue.stream, out, poison.data(), poison.size());
        runtime.upload(queue.stream, mask, poison_mask.data(), poison_mask.size());
        audit::begin_runtime_audit();
        detail::launch_inverse(queue.stream, values, out, mask, input.size(), queue.blocks(input.size()));
        require_no_runtime_transfers(audit::end_runtime_audit());
        Fr* actual = runtime.download(queue.stream, out, input.size());
        uint8_t* actual_mask = runtime.download(queue.stream, mask, input.size());
        Fr* after = runtime.download(queue.stream, values, input.size());
        check(aclrtSynchronizeStream(queue.stream), "zero inverse synchronize");
        // 0, 1 and -1 are their own expected values under this inverse-with-mask convention.
        compare_fields(actual, input, "zero/mixed inverse independent golden");
        compare_fields(after, input, "immutable mixed inverse input");
        for (size_t i = 0; i < input.size(); ++i)
            require(actual_mask[i] == (input[i].words[0] == 0 && input[i].words[7] == 0 ? 0 : 1),
                    "zero/mixed inverse mask mismatch");
    }
}

void test_readable_examples(Runtime& runtime, Queue& queue)
{
    const std::vector<Fr> a{small(1), small(2), small(3)};
    const std::vector<Fr> b{small(4), small(5), small(6)};
    const std::vector<Fr> values{small(0), small(1), minus_one()};
    Fr* da = runtime.allocate<Fr>(3);
    Fr* db = runtime.allocate<Fr>(3);
    Fr* dv = runtime.allocate<Fr>(3);
    Fr* dc = runtime.allocate<Fr>(1);
    Fr* scaled = runtime.allocate<Fr>(3);
    Fr* dot = runtime.allocate<Fr>(1);
    Fr* added = runtime.allocate<Fr>(1);
    const Fr c = minus_one();
    runtime.upload(queue.stream, da, a.data(), 3);
    runtime.upload(queue.stream, db, b.data(), 3);
    runtime.upload(queue.stream, dv, values.data(), 3);
    runtime.upload(queue.stream, dc, &c, 1);
    audit::begin_runtime_audit();
    detail::launch_inner_product(queue.stream, da, db, dot, 3, queue.workspace, queue.dot, 1);
    detail::launch_scalar_vector_mul(queue.stream, dc, dv, scaled, 3, queue.workspace, queue.scale, 1);
    detail::launch_add(queue.stream, dc, da, added, 1, 1);
    require_no_runtime_transfers(audit::end_runtime_audit());
    Fr* actual_dot = runtime.download(queue.stream, dot, 1);
    Fr* actual_scale = runtime.download(queue.stream, scaled, 3);
    Fr* actual_add = runtime.download(queue.stream, added, 1);
    check(aclrtSynchronizeStream(queue.stream), "readable example synchronize");
    compare_fields(actual_dot, {small(32)}, "dot([1,2,3],[4,5,6])");
    compare_fields(actual_scale, {small(0), minus_one(), small(1)}, "scale(-1,[0,1,-1])");
    compare_fields(actual_add, {small(0)}, "add(-1,1)");
}

void test_matrix(Runtime& runtime, Queue& queue)
{
    for (unsigned pattern = 0; pattern < 4; ++pattern) {
        const uint32_t valid = pattern == 3 ? 31 : 256;
        std::vector<Fr> lhs(valid), rhs(valid);
        for (uint32_t t = 0; t < valid; ++t) {
            for (size_t i = 0; i < 32; ++i) {
                const uint8_t choices[]{0, 127, 128, 255};
                uint8_t u = pattern == 0 ? 255 : pattern == 1 ? 0 : choices[(t + i) % 4];
                uint8_t v = pattern == 0 ? 255 : pattern == 1 ? 0 : choices[(3 * t + i + 1) % 4];
                if (pattern >= 2 && i == 31) { u &= 0x3f; v &= 0x3f; }
                lhs[t].words[i / 4] |= uint32_t(u) << (8 * (i % 4));
                rhs[t].words[i / 4] |= uint32_t(v) << (8 * (i % 4));
            }
        }
        // Full-0xff storage deliberately is not a canonical Fr. This helper
        // exercises packing/GEMM only and accepts arbitrary 32-byte bit patterns;
        // it does not perform a field operation on that representation.
        if (pattern >= 2)
            for (uint32_t t = 0; t < valid; ++t)
                require(canonical(lhs[t]) && canonical(rhs[t]), "matrix canonical input failure");
        Fr* a = runtime.allocate<Fr>(valid);
        Fr* b = runtime.allocate<Fr>(valid);
        runtime.upload(queue.stream, a, lhs.data(), valid);
        runtime.upload(queue.stream, b, rhs.data(), valid);
        detail::launch_dot_matrix(queue.stream, a, b, queue.workspace, queue.dot, valid);
        int32_t* signed_result = runtime.download(queue.stream,
            reinterpret_cast<const int32_t*>(queue.workspace + 16384), 32 * 32);
        int32_t* row_sums = runtime.download(queue.stream,
            reinterpret_cast<const int32_t*>(queue.workspace + 24576), 32);
        int32_t* column_sums = runtime.download(queue.stream,
            reinterpret_cast<const int32_t*>(queue.workspace + 25088), 32);
        Fr* after_a = runtime.download(queue.stream, a, valid);
        Fr* after_b = runtime.download(queue.stream, b, valid);
        check(aclrtSynchronizeStream(queue.stream), "matrix synchronize");
        require(std::memcmp(after_a, lhs.data(), valid * sizeof(Fr)) == 0, "matrix changed lhs storage");
        require(std::memcmp(after_b, rhs.data(), valid * sizeof(Fr)) == 0, "matrix changed rhs storage");
        for (size_t i = 0; i < 32; ++i) {
            int32_t expected_row = -128 * static_cast<int32_t>(256 - valid);
            int32_t expected_column = expected_row;
            for (uint32_t t = 0; t < valid; ++t) {
                expected_row += static_cast<int32_t>(limb(lhs[t], i)) - 128;
                expected_column += static_cast<int32_t>(limb(rhs[t], i)) - 128;
            }
            require(row_sums[i] == expected_row, "matrix signed row sum mismatch");
            require(column_sums[i] == expected_column, "matrix signed column sum mismatch");
            for (size_t j = 0; j < 32; ++j) {
                uint64_t expected = 0;
                int64_t expected_signed = 0;
                for (uint32_t t = 0; t < 256; ++t) {
                    const int64_t u = t < valid ? limb(lhs[t], i) : 0;
                    const int64_t v = t < valid ? limb(rhs[t], j) : 0;
                    expected += static_cast<uint64_t>(u * v);
                    expected_signed += (u - 128) * (v - 128);
                }
                require(signed_result[i * 32 + j] == expected_signed, "raw signed GEMM mismatch");
                const int64_t corrected = int64_t(signed_result[i * 32 + j]) +
                    128 * int64_t(row_sums[i]) + 128 * int64_t(column_sums[j]) + 16384 * int64_t(256);
                require(corrected >= 0 && static_cast<uint64_t>(corrected) == expected,
                        "centered matrix correction mismatch");
                if (pattern == 0) require(corrected == 16646400, "all255 matrix mismatch");
                if (pattern == 1) require(corrected == 0, "zero matrix mismatch");
            }
        }
    }
    std::cout << "MATRIX_INT8_PASS\n";
}

void test_independent_streams(Runtime& runtime, const Tiling& dot, const Tiling& scale,
                              uint32_t cores, const std::filesystem::path& root)
{
    Queue first(runtime, dot, scale, cores);
    Queue second(runtime, dot, scale, cores);
    std::array<Queue*, 2> queues{&first, &second};
    std::array<std::vector<Fr>, 2> a, b, expected_scale, expected_dot;
    std::array<Fr*, 2> da{}, db{}, scalar{}, scaled{}, result{}, actual_scale{}, actual_dot{};
    const std::array<size_t, 2> counts{33, 257};
    for (size_t s = 0; s < 2; ++s) {
        const auto path = root / ("case-" + std::to_string(counts[s]));
        a[s] = read_fields(path / "lhs.bin", counts[s]);
        b[s] = read_fields(path / "rhs.bin", counts[s]);
        auto c = read_fields(path / "scalar.bin", 1);
        expected_scale[s] = read_fields(path / "scale.bin", counts[s]);
        expected_dot[s] = read_fields(path / "inner_product.bin", 1);
        da[s] = runtime.allocate<Fr>(counts[s]);
        db[s] = runtime.allocate<Fr>(counts[s]);
        scalar[s] = runtime.allocate<Fr>(1);
        scaled[s] = runtime.allocate<Fr>(counts[s]);
        result[s] = runtime.allocate<Fr>(1);
        runtime.upload(queues[s]->stream, da[s], a[s].data(), counts[s]);
        runtime.upload(queues[s]->stream, db[s], b[s].data(), counts[s]);
        runtime.upload(queues[s]->stream, scalar[s], c.data(), 1);
    }
    // Interleave both kinds of matrix operation without synchronizing either stream.
    for (size_t s = 0; s < 2; ++s)
        detail::launch_inner_product(queues[s]->stream, da[s], db[s], result[s], counts[s],
                                     queues[s]->workspace, queues[s]->dot, queues[s]->blocks(counts[s]));
    for (size_t s = 0; s < 2; ++s)
        detail::launch_scalar_vector_mul(queues[s]->stream, scalar[s], da[s], scaled[s], counts[s],
                                         queues[s]->workspace, queues[s]->scale, queues[s]->blocks(counts[s]));
    for (size_t s = 0; s < 2; ++s) {
        actual_scale[s] = runtime.download(queues[s]->stream, scaled[s], counts[s]);
        actual_dot[s] = runtime.download(queues[s]->stream, result[s], 1);
    }
    for (size_t s = 0; s < 2; ++s) {
        check(aclrtSynchronizeStream(queues[s]->stream), "independent stream synchronize");
        compare_fields(actual_scale[s], expected_scale[s], "independent stream scale");
        compare_fields(actual_dot[s], expected_dot[s], "independent stream dot");
    }
}

void require_no_runtime_transfers(const audit::RuntimeCounts& counts)
{
    require(counts.h2d_calls == 0 && counts.h2d_bytes == 0 && counts.d2h_calls == 0 &&
            counts.d2h_bytes == 0 && counts.other_copy_calls == 0,
            "resident compute segment performed an explicit runtime copy");
    require(counts.stream_sync_calls == 0 && counts.device_sync_calls == 0,
            "resident compute segment synchronized the host");
    require(counts.malloc_calls == 0 && counts.free_calls == 0,
            "resident compute segment allocated/freed device memory");
    require(counts.host_malloc_calls == 0 && counts.host_free_calls == 0,
            "resident compute segment allocated/freed pinned host memory");
}

void test_resident_chain(Runtime& runtime, const Tiling& dot, const Tiling& scale,
                         uint32_t cores, const std::filesystem::path& root, size_t count)
{
    const auto path = root / ("case-" + std::to_string(count));
    const auto a = read_fields(path / "lhs.bin", count);
    const auto b = read_fields(path / "rhs.bin", count);
    const auto expected = read_fields(path / "chain.bin", 1);
    // Each chain owns its scratch and uploads each tiling exactly once.
    Queue queue(runtime, dot, scale, cores);
    Fr* da = runtime.allocate<Fr>(count);
    Fr* db = runtime.allocate<Fr>(count);
    Fr* scalar = runtime.allocate<Fr>(1);
    Fr* s = runtime.allocate<Fr>(count);
    Fr* t = runtime.allocate<Fr>(count);
    Fr* state[2]{runtime.allocate<Fr>(count), runtime.allocate<Fr>(count)};
    Fr* z = runtime.allocate<Fr>(1);
    Fr* actual = runtime.host<Fr>(1);
    runtime.upload(queue.stream, da, a.data(), count);
    runtime.upload(queue.stream, db, b.data(), count);
    const uint32_t blocks = queue.blocks(count);
    audit::begin_runtime_audit();
    detail::launch_inner_product(queue.stream, da, db, scalar, count, queue.workspace, queue.dot, blocks);
    const Fr* x = da;
    for (unsigned round = 0; round < 10; ++round) {
        detail::launch_scalar_vector_mul(queue.stream, scalar, x, s, count, queue.workspace, queue.scale, blocks);
        detail::launch_add(queue.stream, s, db, t, count, blocks);
        detail::launch_mul(queue.stream, t, da, state[round % 2], count, blocks);
        x = state[round % 2];
    }
    detail::launch_inner_product(queue.stream, x, db, z, count, queue.workspace, queue.dot, blocks);
    const auto compute = audit::end_runtime_audit();
    require_no_runtime_transfers(compute);
    audit::begin_runtime_audit();
    runtime.copy_down(queue.stream, actual, z, 1);
    check(aclrtSynchronizeStream(queue.stream), "resident final synchronize");
    const auto tail = audit::end_runtime_audit();
    require(tail.d2h_calls == 1 && tail.d2h_bytes == sizeof(Fr) && tail.h2d_calls == 0 &&
            tail.h2d_bytes == 0 && tail.other_copy_calls == 0 && tail.stream_sync_calls == 1 &&
            tail.device_sync_calls == 0 && tail.malloc_calls == 0 && tail.free_calls == 0,
            "resident final boundary must be exactly one 32-byte download and one stream synchronize");
    compare_fields(actual, expected, "resident chain count=" + std::to_string(count));
    std::cout << "HBM_RESIDENT_CHAIN_PASS count=" << count << " compute_h2d=0 compute_d2h=0"
                 " compute_sync=0 compute_malloc_free=0 final_d2h=32 final_sync=1\n";
}

std::vector<uint8_t> read_fixture_bytes(const std::filesystem::path& path, size_t bytes)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(input.is_open(), "missing fixture: " + path.string());
    require(input.tellg() == static_cast<std::streamoff>(bytes),
            "incorrect fixture length: " + path.string());
    input.seekg(0);
    std::vector<uint8_t> result(bytes);
    if (bytes != 0) {
        input.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(bytes));
        require(static_cast<size_t>(input.gcount()) == bytes, "truncated fixture: " + path.string());
    }
    return result;
}

template<class Field>
bool generic_canonical(const Field& value)
{
    using Params = typename Field::Parameters;
    for (size_t w = Field::word_count; w-- > 0;) {
        if (value.words[w] < Params::modulus[w]) return true;
        if (value.words[w] > Params::modulus[w]) return false;
    }
    return false;
}

template<class Field>
std::vector<Field> read_generic_fields(const std::filesystem::path& path, size_t count)
{
    size_t bytes = 0;
    require(detail::checked_field_bytes<Field>(count, bytes), "invalid generic fixture size");
    const auto encoded = read_fixture_bytes(path, bytes);
    std::vector<Field> result(count);
    for (size_t i = 0; i < count; ++i) {
        for (size_t w = 0; w < Field::word_count; ++w)
            for (size_t b = 0; b < 4; ++b)
                result[i].words[w] |= uint32_t(encoded[i * sizeof(Field) + w * 4 + b]) << (8 * b);
        require(generic_canonical(result[i]), "noncanonical fixture: " + path.string());
    }
    return result;
}

template<class Field>
void compare_generic_fields(const Field* actual, const std::vector<Field>& expected,
                            const std::string& label)
{
    for (size_t i = 0; i < expected.size(); ++i) {
        require(generic_canonical(actual[i]), label + " noncanonical result at " + std::to_string(i));
        require(std::memcmp(actual + i, &expected[i], sizeof(Field)) == 0,
                label + " mismatch at " + std::to_string(i));
    }
}

uint8_t* device_address(const void* pointer)
{
    return reinterpret_cast<uint8_t*>(const_cast<void*>(pointer));
}

// Exact payload boundaries; retain the allocation base in Runtime.
struct FieldGuard {
    size_t bytes;
    uint8_t* base;
    uint8_t* payload;
    uint8_t* after;
    std::vector<uint8_t> initial;
    FieldGuard(Runtime& runtime, aclrtStream stream, size_t size, const void* source = nullptr)
        : bytes(size), base(runtime.allocate<uint8_t>(size + 128)), payload(base + 64),
          after(runtime.host<uint8_t>(size + 128)), initial(size + 128, 0xcc)
    {
        std::fill(initial.begin(), initial.begin() + 64, 0xa5);
        std::fill(initial.end() - 64, initial.end(), 0x5a);
        if (source && size) std::memcpy(initial.data() + 64, source, size);
        runtime.upload(stream, base, initial.data(), initial.size());
    }
    void read(Runtime& runtime, aclrtStream stream) {
        runtime.copy_down(stream, after, base, initial.size());
    }
    void verify(bool immutable, const std::string& label) const {
        require(std::memcmp(after, initial.data(), 64) == 0 &&
                std::memcmp(after + 64 + bytes, initial.data() + 64 + bytes, 64) == 0,
                label + " guard changed");
        if (immutable)
            require(std::memcmp(after + 64, initial.data() + 64, bytes) == 0,
                    label + " input changed");
    }
    template<class T> T* data() { return reinterpret_cast<T*>(payload); }
};
template<class Field>
bool launch_field_element(aclrtStream stream, const void* lhs, const Field* rhs, Field* out,
                          uint8_t* mask, size_t count, uint32_t cores, field_smoke::Operation op,
                          bool audit_enqueue = true)
{
    require(count == 0 || cores != 0, "generic field test has no AIV cores");
    using Wide = math::WideInteger<typename Field::Parameters>;
    if (audit_enqueue) audit::begin_runtime_audit();
    switch (op) {
    case field_smoke::add:
        detail::launch_add<Field>(stream, static_cast<const Field*>(lhs), rhs, out, count, cores); break;
    case field_smoke::sub:
        detail::launch_sub<Field>(stream, static_cast<const Field*>(lhs), rhs, out, count, cores); break;
    case field_smoke::neg:
        detail::launch_neg<Field>(stream, static_cast<const Field*>(lhs), out, count, cores); break;
    case field_smoke::mul:
        detail::launch_mul<Field>(stream, static_cast<const Field*>(lhs), rhs, out, count, cores); break;
    case field_smoke::square:
        detail::launch_square<Field>(stream, static_cast<const Field*>(lhs), out, count, cores); break;
    case field_smoke::inverse:
        detail::launch_inverse<Field>(stream, static_cast<const Field*>(lhs), out, mask, count, cores); break;
    case field_smoke::wide:
        detail::launch_reduce_wide<Field>(stream, static_cast<const Wide*>(lhs), out, count, cores); break;
    default:
        throw std::runtime_error("invalid field operation");
    }
    if (audit_enqueue) require_no_runtime_transfers(audit::end_runtime_audit());
    return count != 0;
}

template<class Field>
void test_generic_inverse(Runtime& runtime, aclrtStream stream, uint32_t cores,
                          const std::vector<Field>& input, const std::vector<Field>& expected,
                          const std::string& label)
{
    require(input.size() == expected.size(), "inverse golden count mismatch");
    const size_t count = input.size();
    FieldGuard values(runtime, stream, count * sizeof(Field), input.data());
    FieldGuard output(runtime, stream, count * sizeof(Field));
    FieldGuard mask(runtime, stream, count);
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
        require(launch_field_element<Field>(stream, count ? values.payload : nullptr, nullptr,
                    count ? output.data<Field>() : nullptr, count ? mask.payload : nullptr,
                    count, cores, field_smoke::inverse) == (count != 0),
                "inverse empty-launch contract violated");
        values.read(runtime, stream); output.read(runtime, stream); mask.read(runtime, stream);
        check(aclrtSynchronizeStream(stream), "generic inverse synchronize");
        values.verify(true, label); output.verify(false, label); mask.verify(false, label);
        compare_generic_fields(reinterpret_cast<const Field*>(output.after + 64), expected, label);
        for (size_t i = 0; i < count; ++i) {
            bool zero = true;
            for (auto word : input[i].words) zero = zero && word == 0;
            require(mask.after[64 + i] == (zero ? 0 : 1), label + " mask mismatch");
        }
    }
}

template<class Field>
void test_generic_fixture(Runtime& runtime, aclrtStream stream, uint32_t cores,
                          const std::filesystem::path& path, size_t count, const std::string& field,
                          bool inverses = false)
{
    const auto lhs = read_generic_fields<Field>(path / "lhs.bin", count);
    const auto rhs = read_generic_fields<Field>(path / "rhs.bin", count);
    FieldGuard a(runtime, stream, count * sizeof(Field), lhs.data());
    FieldGuard b(runtime, stream, count * sizeof(Field), rhs.data());
    FieldGuard output(runtime, stream, count * sizeof(Field));
    const std::array<const char*, 5> names{"add", "sub", "neg", "mul", "square"};
    const std::array<field_smoke::Operation, 5> ops{
        field_smoke::add, field_smoke::sub, field_smoke::neg, field_smoke::mul, field_smoke::square};
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
        for (size_t op = 0; op < names.size(); ++op) {
            const auto expected = read_generic_fields<Field>(path / (std::string(names[op]) + ".bin"), count);
            require(launch_field_element<Field>(stream, count ? a.payload : nullptr,
                        count ? b.data<Field>() : nullptr, count ? output.data<Field>() : nullptr,
                        nullptr, count, cores, ops[op]) == (count != 0),
                    "generic empty-launch contract violated");
            a.read(runtime, stream); b.read(runtime, stream); output.read(runtime, stream);
            check(aclrtSynchronizeStream(stream), "generic arithmetic synchronize");
            a.verify(true, field + " lhs"); b.verify(true, field + " rhs");
            output.verify(false, field + " output");
            compare_generic_fields(reinterpret_cast<const Field*>(output.after + 64), expected,
                                   field + " " + names[op] + " count=" + std::to_string(count));
        }
    }
    // Cover dst==lhs, dst==rhs, lhs==rhs, and all-equal with guarded payloads.
    for (size_t op = 0; op < names.size(); ++op) {
        const auto golden = read_generic_fields<Field>(path / (std::string(names[op]) + ".bin"), count);
        for (unsigned shape = 0; shape < 4; ++shape) {
            const auto mark = runtime.storage_mark();
            FieldGuard x(runtime, stream, count * sizeof(Field), lhs.data());
            FieldGuard y(runtime, stream, count * sizeof(Field), rhs.data());
            FieldGuard z(runtime, stream, count * sizeof(Field));
            Field* destination = shape == 0 || shape == 3 ? x.data<Field>() :
                                 shape == 1 ? y.data<Field>() : z.data<Field>();
            const Field* right = shape >= 2 ? x.data<Field>() : y.data<Field>();
            auto expected = golden;
            if (shape >= 2) {
                for (size_t i = 0; i < count; ++i) {
                    switch (ops[op]) {
                    case field_smoke::add: expected[i] = lhs[i] + lhs[i]; break;
                    case field_smoke::sub: expected[i] = Field{}; break;
                    case field_smoke::mul: expected[i] = lhs[i] * lhs[i]; break;
                    default: break;
                    }
                }
            }
            launch_field_element<Field>(stream, count ? x.payload : nullptr,
                count ? right : nullptr, count ? destination : nullptr, nullptr, count, cores, ops[op]);
            x.read(runtime, stream); y.read(runtime, stream); z.read(runtime, stream);
            check(aclrtSynchronizeStream(stream), "generic alias synchronize");
            x.verify(destination != x.data<Field>(), field + " alias lhs");
            y.verify(destination != y.data<Field>(), field + " alias rhs");
            z.verify(destination != z.data<Field>(), field + " alias output");
            const auto* actual = destination == x.data<Field>() ? x.after :
                                 destination == y.data<Field>() ? y.after : z.after;
            compare_generic_fields(reinterpret_cast<const Field*>(actual + 64), expected,
                                   field + " alias " + names[op] + " shape=" + std::to_string(shape));
            runtime.release_storage(mark);
        }
    }
    if (inverses) {
        auto input = read_generic_fields<Field>(path / "inverse_input.bin", count);
        auto expected = read_generic_fields<Field>(path / "inverse.bin", count);
        test_generic_inverse(runtime, stream, cores, input, expected, field + " inverse count=" +
                             std::to_string(count));
        if (count == 47 || count == 48 || count == 49 ||
            count == 95 || count == 96 || count == 97) {
            // Distinct status bytes in every packed word, including tile edges.
            auto masked_input = input;
            auto masked_expected = expected;
            for (size_t i = 0; i < count; ++i) {
                if (i % 4 == 0 || i % 4 == 3 || i % 48 == 47) {
                    masked_input[i] = Field{};
                    masked_expected[i] = Field{};
                }
            }
            test_generic_inverse(runtime, stream, cores, masked_input, masked_expected,
                                 field + " packed inverse mask count=" + std::to_string(count));
        }
        if (count == 65) {
            // Three tiles after adding zeros at both ends; poison must be overwritten.
            input.insert(input.begin(), Field{});
            expected.insert(expected.begin(), Field{});
            input.push_back(Field{});
            expected.push_back(Field{});
            test_generic_inverse(runtime, stream, cores, input, expected, field + " mixed zero inverse");
        }
    }
}

template<class Field>
void test_generic_wide(Runtime& runtime, aclrtStream stream, uint32_t cores,
                       const std::filesystem::path& root, const std::string& field)
{
    constexpr size_t width = 2 * sizeof(Field) + 4;
    const auto input = read_fixture_bytes(root / "wide_input.bin", 512 * width);
    const auto all_expected = read_generic_fields<Field>(root / "wide_result.bin", 512);
    for (size_t count : {0u, 1u, 31u, 32u, 33u, 63u, 64u, 65u, 255u, 256u, 257u, 512u, 47u, 48u, 49u, 95u, 96u, 97u}) {
        const auto mark = runtime.storage_mark();
        FieldGuard values(runtime, stream, count * width, input.data());
        FieldGuard output(runtime, stream, count * sizeof(Field));
        require(launch_field_element<Field>(stream, count ? values.payload : nullptr, nullptr,
                    count ? output.data<Field>() : nullptr, nullptr, count, cores,
                    field_smoke::wide) == (count != 0), "wide empty-launch contract violated");
        values.read(runtime, stream); output.read(runtime, stream);
        check(aclrtSynchronizeStream(stream), "generic wide synchronize");
        values.verify(true, field + " wide"); output.verify(false, field + " wide");
        compare_generic_fields(reinterpret_cast<const Field*>(output.after + 64),
                               std::vector<Field>(all_expected.begin(), all_expected.begin() + count),
                               field + " wide count=" + std::to_string(count));
        runtime.release_storage(mark);
    }
}

template<class Field>
void test_generic_field(Runtime& runtime, aclrtStream stream, uint32_t cores,
                        const std::filesystem::path& root, const std::string& field)
{
    for (size_t count : case_counts)
        test_generic_fixture<Field>(runtime, stream, cores, root / ("case-" + std::to_string(count)),
                                    count, field, inverse_case(count));
    for (const std::string directory : {"boundaries", "radix-boundaries"}) {
        const auto path = root / directory;
        const size_t bytes = std::filesystem::file_size(path / "lhs.bin");
        require(bytes % sizeof(Field) == 0, "boundary fixture record size mismatch");
        test_generic_fixture<Field>(runtime, stream, cores, path, bytes / sizeof(Field), field);
    }
    const std::vector<Field> zeros(65);
    test_generic_inverse(runtime, stream, cores, zeros, zeros, field + " all-zero inverse");
    Field pm1{};
    for (unsigned w = 0; w < Field::word_count; ++w) pm1.words[w] = Field::Parameters::modulus[w];
    --pm1.words[0];
    const std::vector<Field> mixed{Field{}, Field::one(), pm1};
    test_generic_inverse(runtime, stream, cores, mixed, mixed, field + " [0,1,p-1] inverse");
    test_generic_wide<Field>(runtime, stream, cores, root, field);
    // Two independent streams enqueue against shared immutable HBM inputs.
    // Only lane64 is nonzero; compare distinct operations before any readback.
    const auto other = runtime.stream();
    std::vector<Field> tail(65);
    tail.back() = pm1;
    const std::vector<Field> factors(65, pm1);
    std::vector<Field> product(65);
    product.back() = Field::one();
    std::vector<Field> sum(65, pm1);
    sum.back() = pm1;
    for (unsigned w = 0; w < Field::word_count; ++w)
        if (sum.back().words[w]-- != 0) break;
    FieldGuard a(runtime, stream, tail.size() * sizeof(Field), tail.data());
    FieldGuard b(runtime, stream, factors.size() * sizeof(Field), factors.data());
    FieldGuard multiplied(runtime, other, product.size() * sizeof(Field));
    FieldGuard added(runtime, stream, sum.size() * sizeof(Field));
    check(aclrtSynchronizeStream(stream), "field shared input setup");
    check(aclrtSynchronizeStream(other), "field second output setup");
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
        launch_field_element<Field>(stream, a.payload, b.data<Field>(), added.data<Field>(),
                                     nullptr, 65, cores, field_smoke::add);
        launch_field_element<Field>(other, a.payload, b.data<Field>(), multiplied.data<Field>(),
                                     nullptr, 65, cores, field_smoke::mul);
    }
    a.read(runtime, stream); b.read(runtime, stream); added.read(runtime, stream);
    multiplied.read(runtime, other);
    check(aclrtSynchronizeStream(stream), "field first stream completion");
    check(aclrtSynchronizeStream(other), "field second stream completion");
    a.verify(true, field); b.verify(true, field);
    added.verify(false, field); multiplied.verify(false, field);
    compare_generic_fields(reinterpret_cast<const Field*>(added.after + 64), sum, field + " tail add");
    compare_generic_fields(reinterpret_cast<const Field*>(multiplied.after + 64), product,
                           field + " lane64 two-stream mul");
    std::cout << "FIELD_ELEMENT_NPU_PASS field=" << field << '\n';
}


void field_audit(const audit::RuntimeCounts& counts, bool diagnostic)
{
    if (!diagnostic) {
        require_no_runtime_transfers(counts);
        return;
    }
    require(counts.h2d_calls == 0 && counts.h2d_bytes == 0 && counts.d2h_calls == 0 &&
            counts.d2h_bytes == 0 && counts.other_copy_calls == 0 &&
            counts.stream_sync_calls == 0 && counts.device_sync_calls == 0 &&
            counts.malloc_calls == 0 && counts.free_calls == 0 &&
            counts.unexpected_copy_calls == 0, "profile performed forbidden runtime activity");
    std::cout << "FIELD_PROFILE_DIAGNOSTIC"
              << " h2d_calls=" << counts.h2d_calls << " h2d_bytes=" << counts.h2d_bytes
              << " d2h_calls=" << counts.d2h_calls << " d2h_bytes=" << counts.d2h_bytes
              << " other_copy_calls=" << counts.other_copy_calls
              << " stream_sync_calls=" << counts.stream_sync_calls
              << " device_sync_calls=" << counts.device_sync_calls
              << " malloc_calls=" << counts.malloc_calls << " free_calls=" << counts.free_calls
              << " host_malloc_calls=" << counts.host_malloc_calls
              << " host_free_calls=" << counts.host_free_calls
              << " unexpected_copy_calls=" << counts.unexpected_copy_calls << '\n';
}

template<class Field>
void measure_field(Runtime& runtime, Queue& queue, const std::filesystem::path& root,
                   const std::string& field, const std::string& operation, size_t count,
                   bool diagnostic)
{
    const auto mark = runtime.storage_mark();
    const bool wide = operation == "wide", inverse = operation == "inverse";
    const bool dot = operation == "inner_product", scale = operation == "scale";
    const bool cube = dot || scale;
    const auto path = root / ("case-" + std::to_string(count));
    std::vector<uint8_t> raw;
    std::vector<Field> lhs;
    if (wide) {
        raw = read_fixture_bytes(root / "wide_input.bin", 512 * (2 * sizeof(Field) + 4));
        raw.resize(count * (2 * sizeof(Field) + 4));
    } else {
        lhs = read_generic_fields<Field>(path / (inverse ? "inverse_input.bin" : "lhs.bin"), count);
    }
    const auto rhs = wide ? std::vector<Field>{} :
        read_generic_fields<Field>(path / "rhs.bin", count);
    auto expected = wide ? read_generic_fields<Field>(root / "wide_result.bin", 512) :
        read_generic_fields<Field>(path / (operation + ".bin"), dot ? 1 : count);
    if (wide) expected.resize(count);
    std::vector<Field> scalar;
    if (scale) scalar = read_generic_fields<Field>(path / "scalar.bin", 1);
    FieldGuard a(runtime, queue.stream, wide ? raw.size() : count * sizeof(Field),
                 wide ? static_cast<const void*>(raw.data()) : lhs.data());
    FieldGuard b(runtime, queue.stream, rhs.size() * sizeof(Field), rhs.data());
    FieldGuard c(runtime, queue.stream, scalar.size() * sizeof(Field), scalar.data());
    FieldGuard out(runtime, queue.stream, expected.size() * sizeof(Field));
    FieldGuard mask(runtime, queue.stream, inverse ? count : 0);
    const field_smoke::Operation op = operation == "add" ? field_smoke::add :
        operation == "sub" ? field_smoke::sub : operation == "neg" ? field_smoke::neg :
        operation == "square" ? field_smoke::square : inverse ? field_smoke::inverse :
        wide ? field_smoke::wide : field_smoke::mul;
    auto enqueue = [&] {
        if constexpr (std::is_same<Field, Fr>::value) {
            if (dot) {
                detail::launch_inner_product(queue.stream, a.data<Fr>(), b.data<Fr>(), out.data<Fr>(),
                    count, queue.workspace, queue.dot, queue.cores);
                return;
            }
            if (scale) {
                detail::launch_scalar_vector_mul(queue.stream, c.data<Fr>(), a.data<Fr>(), out.data<Fr>(),
                    count, queue.workspace, queue.scale, queue.cores);
                return;
            }
        }
        launch_field_element<Field>(queue.stream, a.payload, b.data<Field>(), out.data<Field>(),
                                     inverse ? mask.payload : nullptr, count, queue.cores, op, false);
    };
    check(aclrtSynchronizeStream(queue.stream), "field measurement setup");
    std::vector<double> samples;
    for (unsigned sample = 0; sample < (diagnostic ? 1u : 10u); ++sample) {
        if (!diagnostic) {
            audit::begin_runtime_audit();
            enqueue();
            field_audit(audit::end_runtime_audit(), false);
            check(aclrtSynchronizeStream(queue.stream), "field warmup");
        }
        const auto begin = std::chrono::steady_clock::now();
        audit::begin_runtime_audit();
        enqueue();
        const auto counts = audit::end_runtime_audit();
        check(aclrtSynchronizeStream(queue.stream), "field target");
        const auto end = std::chrono::steady_clock::now();
        field_audit(counts, diagnostic);
        samples.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
        a.read(runtime, queue.stream); b.read(runtime, queue.stream); c.read(runtime, queue.stream);
        out.read(runtime, queue.stream); mask.read(runtime, queue.stream);
        check(aclrtSynchronizeStream(queue.stream), "field measurement verification");
        a.verify(true, "field lhs"); b.verify(true, "field rhs"); c.verify(true, "field scalar");
        out.verify(false, "field output"); mask.verify(false, "field mask");
        compare_generic_fields(reinterpret_cast<const Field*>(out.after + 64), expected,
                               field + " measured " + operation);
        if (inverse)
            for (size_t i = 0; i < count; ++i) {
                bool zero = true;
                for (auto word : lhs[i].words) zero = zero && word == 0;
                require(mask.after[64 + i] == (zero ? 0 : 1), "measured inverse mask mismatch");
            }
    }
    if (!diagnostic) {
        auto sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        const double median = (sorted[4] + sorted[5]) / 2;
        const size_t launches = dot ? 2 + 3 * ((count + 255) / 256) :
            scale ? 1 + 3 * ((count + 31) / 32) : 1;
        std::cout << std::setprecision(12) << "FIELD_PERF {\"field\":\"" << field
                  << "\",\"operation\":\"" << operation << "\",\"count\":" << count
                  << ",\"backend\":\"" << (cube ? "cube-int8-scalar-post" : "vector-i32-r15")
                  << "\",\"timing\":\"resident_completed_api_wall\",\"samples_us\":[";
        for (size_t i = 0; i < samples.size(); ++i) std::cout << (i ? "," : "") << samples[i];
        std::cout << "],\"median_us\":" << median << ",\"elements_per_second\":"
                  << count * 1e6 / median << ",\"aiv_cores\":" << queue.cores
                  << ",\"launches_per_call\":" << launches << ",\"audit\":\"strict_zero\"}\n";
    }
    runtime.release_storage(mark);
}

template<class Field>
void field_bench(Runtime& runtime, Queue& queue, const std::filesystem::path& root,
                 const std::string& field)
{
    for (const std::string operation : {"add", "sub", "neg", "mul", "square"})
        for (size_t count : {33u, 257u, 4097u})
            measure_field<Field>(runtime, queue, root, field, operation, count, false);
    for (size_t count : {33u, 257u})
        measure_field<Field>(runtime, queue, root, field, "inverse", count, false);
    for (size_t count : {33u, 512u})
        measure_field<Field>(runtime, queue, root, field, "wide", count, false);
}

void require_dma_alignment(const void* pointer)
{
    require(!pointer || reinterpret_cast<uintptr_t>(pointer)%32 == 0, "G1 DMA base not 32-byte aligned");
}
std::vector<math::G1Affine> read_g1_affine(const std::filesystem::path& path, size_t count)
{
    const auto coords = read_generic_fields<math::Fq>(path,count*2);
    std::vector<math::G1Affine> out(count);
    for (size_t i = 0; i < count; ++i) out[i] = {coords[2*i],coords[2*i+1]};
    return out;
}
std::vector<math::G1Jacobian> read_g1_jacobian(const std::filesystem::path& path, size_t count)
{
    const auto coords = read_generic_fields<math::Fq>(path,count*3);
    std::vector<math::G1Jacobian> out(count);
    for (size_t i = 0; i < count; ++i) out[i] = {coords[3*i],coords[3*i+1],coords[3*i+2]};
    return out;
}
bool g1_zero(const math::Fq& value)
{
    for (auto word : value.words) if (word) return false;
    return true;
}
bool g1_identity(const math::G1Affine& value) { return g1_zero(value.x) && g1_zero(value.y); }
void compare_g1_affine(const math::G1Affine* actual, const std::vector<math::G1Affine>& expected,
                       const std::string& label)
{
    for (size_t i = 0; i < expected.size(); ++i) {
        require(generic_canonical(actual[i].x) && generic_canonical(actual[i].y),label+" noncanonical affine");
        require(std::memcmp(actual+i,&expected[i],96) == 0,label+" affine mismatch "+std::to_string(i));
    }
}
// Guard payload boundaries, including physical padding (not just logical records).
struct G1Guard {
    uint8_t* allocation = nullptr;
    uint8_t* payload = nullptr;
    std::vector<uint8_t> initial;
    uint8_t* after = nullptr;
    size_t bytes = 0;
    G1Guard(Runtime& runtime, aclrtStream stream, size_t size, const void* source = nullptr) : bytes(size) {
        if (!size) return;
        initial.assign(size+64,0xcc);
        std::fill(initial.begin(),initial.begin()+32,0xa5);
        std::fill(initial.end()-32,initial.end(),0x5a);
        if (source) std::memcpy(initial.data()+32,source,size);
        allocation = runtime.allocate<uint8_t>(size+64);
        payload = allocation+32;
        require_dma_alignment(allocation);
        require_dma_alignment(payload);
        runtime.upload(stream,allocation,initial.data(),initial.size());
    }
    void download(Runtime& runtime, aclrtStream stream) {
        after = runtime.download(stream,allocation,initial.size());
    }
    void verify(bool immutable, const std::string& label) const {
        if (!bytes) return;
        require(std::memcmp(after,initial.data(),32) == 0 &&
                std::memcmp(after+32+bytes,initial.data()+32+bytes,32) == 0,label+" guard corruption");
        if (immutable) require(std::memcmp(after,initial.data(),initial.size()) == 0,label+" input mutation");
    }
};
uint32_t g1_u32(const uint8_t* p)
{
    return uint32_t(p[0]) | (uint32_t(p[1])<<8) | (uint32_t(p[2])<<16) | (uint32_t(p[3])<<24);
}
bool launch_g1_elementwise(aclrtStream stream, uint32_t cores, uint8_t* lhs, uint8_t* rhs,
                          uint8_t* affine, uint8_t* factors, uint8_t* out, size_t count, uint32_t op)
{
    if (!count) return false;
    require(cores && op < 8,"invalid G1 primitive configuration");
    for (auto p : {lhs,rhs,affine,factors,out}) require_dma_alignment(p);
    const uint32_t blocks = uint32_t(std::min<size_t>(cores,(count+1)/2));
    audit::launch_g1_elementwise_smoke(blocks,stream,lhs,rhs,affine,factors,out,count,op);
    return true;
}
void inspect_g1(Runtime& runtime, aclrtStream stream, uint32_t cores, G1Guard& raw,
                G1Guard& rhs, G1Guard& affine_rhs,
                const std::vector<math::G1Affine>& expected,
                const std::vector<math::G1Affine>& expected_rhs, bool canonical_identity,
                const std::string& label)
{
    const size_t count = expected.size();
    if (!count) return;
    G1Guard affine(runtime,stream,count*96), flags(runtime,stream,((count+1)/2)*32);
    for (auto p : {raw.payload,rhs.payload,affine_rhs.payload,affine.payload,flags.payload})
        require_dma_alignment(p);
    audit::launch_g1_inspect_smoke(uint32_t(std::min<size_t>(cores,(count+1)/2)),stream,
        raw.payload,rhs.payload,affine_rhs.payload,affine.payload,flags.payload,count);
    raw.download(runtime,stream);
    affine.download(runtime,stream);
    flags.download(runtime,stream);
    check(aclrtSynchronizeStream(stream),"G1 inspect synchronize");
    raw.verify(!canonical_identity,label+" raw");
    affine.verify(false,label+" affine");
    flags.verify(false,label+" flags");
    compare_g1_affine(reinterpret_cast<const math::G1Affine*>(affine.after+32),expected,label);
    for (size_t i = 0; i < count; ++i) {
        math::G1Jacobian p{};
        std::memcpy(&p,raw.after+32+i*144,144);
        require(generic_canonical(p.x) && generic_canonical(p.y) && generic_canonical(p.z),
                label+" noncanonical Jacobian");
        const bool identity = g1_identity(expected[i]);
        if (canonical_identity && identity) {
            math::Fq one{}; one.words[0] = 1;
            require(g1_zero(p.x) && std::memcmp(&p.y,&one,48) == 0 && g1_zero(p.z),
                    label+" noncanonical identity");
        }
        const bool equal = std::memcmp(&expected[i],&expected_rhs[i],96) == 0;
        const uint32_t expected_flags = 1u | (uint32_t(identity)<<1) | (uint32_t(equal)<<2)
            | (uint32_t(!equal)<<3) | 16u | (uint32_t(identity)<<5)
            | (uint32_t(equal)<<6) | (uint32_t(!equal)<<7);
        require(g1_u32(flags.after+32+(i/2)*32+(i%2)*4) == expected_flags,label+" flags mismatch");
    }
    for (size_t pair = 0; pair < (count+1)/2; ++pair) {
        const size_t used = std::min<size_t>(2,count-pair*2)*4;
        for (size_t b = used; b < 32; ++b)
            require(flags.after[32+pair*32+b] == 0,label+" nonzero flag padding");
    }
}
void test_g1_primitives(Runtime& runtime, aclrtStream stream, uint32_t cores,
                        const std::filesystem::path& fixtures)
{
    const auto root = fixtures/"g1";
    for (const size_t count : {0,1,2,15,16,17,31,32,33,65,63,64,47,48,49,95,96,97}) {
        const auto dir = root/("case-"+std::to_string(count));
        const auto aa = read_g1_affine(dir/"affine_a.bin",count);
        const auto ab = read_g1_affine(dir/"affine_b.bin",count);
        const auto ja = read_g1_jacobian(dir/"jacobian_a.bin",count);
        const auto jb = read_g1_jacobian(dir/"jacobian_b.bin",count);
        const auto equality = read_fixture_bytes(dir/"equal.bin",count*4);
        for (size_t i = 0; i < count; ++i)
            require(g1_u32(equality.data()+4*i) == uint32_t(std::memcmp(&aa[i],&ab[i],96) == 0),
                    "G1 equality fixture mismatch");
        auto factors = read_fixture_bytes(dir/"mul_u64.bin",count*8);
        factors.resize(((count+3)/4)*32,0);
        const std::array<const char*,8> names{"add","add","double","neg","mul","double","affine_b","affine_neg"};
        std::array<std::vector<math::G1Affine>,8> expected;
        for (unsigned op = 0; op < 8; ++op) expected[op] = read_g1_affine(dir/(std::string(names[op])+".bin"),count);
        if (!count) {
            require(!launch_g1_elementwise(stream,cores,nullptr,nullptr,nullptr,nullptr,nullptr,0,0),
                    "empty G1 primitive launched");
            continue;
        }
        G1Guard a(runtime,stream,count*144,ja.data()), b(runtime,stream,count*144,jb.data());
        G1Guard affine_b(runtime,stream,count*96,ab.data()), factor(runtime,stream,factors.size(),factors.data());
        inspect_g1(runtime,stream,cores,a,b,affine_b,aa,ab,false,dir.string()+" original");
        for (uint32_t op = 0; op < 8; ++op) {
            G1Guard out(runtime,stream,count*144);
            require(launch_g1_elementwise(stream,cores,a.payload,b.payload,affine_b.payload,factor.payload,
                                          out.payload,count,op),"G1 primitive did not launch");
            inspect_g1(runtime,stream,cores,out,b,affine_b,expected[op],ab,true,
                       dir.string()+" op="+std::to_string(op));
        }
        for (auto p : {&a,&b,&affine_b,&factor}) p->download(runtime,stream);
        check(aclrtSynchronizeStream(stream),"G1 input synchronize");
        for (auto p : {&a,&b,&affine_b,&factor}) p->verify(true,dir.string());
        std::cout << "G1_PRIMITIVE_NPU_PASS count=" << count << '\n';
    }
    const auto da = read_g1_affine(root/"diagnostics/affine.bin",5);
    const auto dj = read_g1_jacobian(root/"diagnostics/jacobian.bin",5);
    const auto expected_flags = read_fixture_bytes(root/"diagnostics/flags.bin",20);
    G1Guard a(runtime,stream,5*96,da.data()), j(runtime,stream,5*144,dj.data()), flags(runtime,stream,3*32);
    audit::launch_g1_diagnostics_smoke(std::min<uint32_t>(cores,3),stream,a.payload,j.payload,flags.payload,5);
    for (auto p : {&a,&j,&flags}) p->download(runtime,stream);
    check(aclrtSynchronizeStream(stream),"G1 diagnostics synchronize");
    a.verify(true,"diagnostic affine"); j.verify(true,"diagnostic Jacobian"); flags.verify(false,"diagnostic flags");
    for (size_t pair = 0; pair < 3; ++pair) for (size_t byte = 0; byte < 32; ++byte) {
        const size_t point = pair*2+byte/4;
        const uint8_t expected = byte < 8 && point < 5 ? expected_flags[point*4+byte%4] : 0;
        require(flags.after[32+pair*32+byte] == expected,"G1 diagnostic flag/padding mismatch");
    }
    const auto generator = read_g1_affine(root/"generator.bin",1);
    G1Guard ga(runtime,stream,96), gj(runtime,stream,144);
    audit::launch_g1_generator_smoke(stream,gj.payload,ga.payload);
    ga.download(runtime,stream); gj.download(runtime,stream);
    check(aclrtSynchronizeStream(stream),"G1 generator synchronize");
    ga.verify(false,"generator affine"); gj.verify(false,"generator Jacobian");
    compare_g1_affine(reinterpret_cast<const math::G1Affine*>(ga.after+32),generator,"device generator");
    math::G1Jacobian expected_j{generator[0].x,generator[0].y,{}};
    expected_j.z.words[0] = 1;
    require(std::memcmp(gj.after+32,&expected_j,144) == 0,"device generator Jacobian mismatch");
    std::cout << "G1_DIAGNOSTICS_NPU_PASS\nG1_GENERATOR_NPU_PASS\n";
}

void test_g1_chain(Runtime& runtime, aclrtStream stream, uint32_t cores,
                   const std::filesystem::path& dir, size_t count, uint32_t c)
{
    require((c == 3 || c == 4) && cores && (count == 0 || count == 1 || count == 2 ||
            count == 7 || count == 33),"invalid fixed G1 chain configuration");
    const auto bases = read_g1_affine(dir/"bases.bin",count);
    const auto scalars = read_generic_fields<Fr>(dir/"scalars.bin",count);
    const auto expected = read_g1_affine(dir/"expected.bin",1);
    const uint32_t windows = (255+c-1)/c, buckets = 1u << c;
    auto* dbases = runtime.allocate<math::G1Affine>(count);
    auto* dscalars = runtime.allocate<Fr>(count);
    auto* partials = runtime.allocate<math::G1Jacobian>(count ? windows*buckets*2 : 0);
    auto* segments = runtime.allocate<math::G1Jacobian>(count ? windows*2 : 0);
    auto* out = runtime.allocate<math::G1Affine>(1);
    auto* final = runtime.host<math::G1Affine>(1);
    for (const void* p : {static_cast<void*>(dbases),static_cast<void*>(dscalars),
                         static_cast<void*>(partials),static_cast<void*>(segments),static_cast<void*>(out)})
        require_dma_alignment(p);
    runtime.upload(stream,dbases,bases.data(),count);
    runtime.upload(stream,dscalars,scalars.data(),count);
    audit::begin_runtime_audit();
    if (count) {
        audit::launch_g1_chain_partials(std::min(cores,windows*buckets),stream,
            device_address(dbases),device_address(dscalars),device_address(partials),uint32_t(count),c);
        audit::launch_g1_chain_segments(std::min(cores,windows),stream,
            device_address(partials),device_address(segments),c);
    }
    audit::launch_g1_chain_finish(stream,device_address(segments),device_address(out),count ? windows : 0,c);
    const auto compute = audit::end_runtime_audit();
    require_no_runtime_transfers(compute);
    audit::begin_runtime_audit();
    runtime.copy_down(stream,final,out,1);
    check(aclrtSynchronizeStream(stream),"G1 resident final synchronize");
    const auto tail = audit::end_runtime_audit();
    require(tail.d2h_calls == 1 && tail.d2h_bytes == 96 && tail.stream_sync_calls == 1 &&
            tail.h2d_calls == 0 && tail.h2d_bytes == 0 && tail.other_copy_calls == 0 &&
            tail.device_sync_calls == 0 && tail.malloc_calls == 0 && tail.free_calls == 0,
            "G1 resident tail must be one 96-byte download and one stream synchronize");
    compare_g1_affine(final,expected,"resident chain "+dir.string()+"/"+std::to_string(c));
    std::cout << "G1_CUZK_RESIDENT_PASS case=" << dir.filename().string() << " count=" << count
              << " window_bits=" << c << " compute_h2d=0 compute_d2h=0 compute_sync=0"
                 " compute_malloc_free=0 final_d2h=96 final_sync=1\n";
}
void test_g1_chains(Runtime& runtime, aclrtStream stream, uint32_t cores, const std::filesystem::path& fixtures)
{
    const std::array<std::pair<const char*,size_t>,8> cases{{
        {"case-0",0},{"case-1",1},{"case-2",2},{"case-7",7},{"case-33",33},
        {"cancel",2},{"zero-scalars",7},{"identity-bases",7}}};
    for (const auto& item : cases) for (uint32_t c : {3u,4u})
        test_g1_chain(runtime,stream,cores,fixtures/"g1/chain"/item.first,item.second,c);
}

// Independent integer/byte models, never candidate scalar extraction.
size_t msm_round8(size_t n) { return (n+7)/8*8; }
uint8_t* msm_align64(uint8_t* p)
{
    return reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(p)+63)&~uintptr_t(63));
}
uint8_t* msm_region(uint8_t* workspace, const detail::MsmPlan& plan, detail::MsmRegion region)
{
    return workspace+plan.region_offset[static_cast<uint32_t>(region)];
}
uint64_t msm_u64(const uint8_t* p) { uint64_t x; std::memcpy(&x,p,8); return x; }
uint32_t msm_u32(const uint8_t* p) { uint32_t x; std::memcpy(&x,p,4); return x; }
void msm_zero_padding(const uint8_t* p, size_t first, size_t end, const std::string& label)
{
    for (size_t i = first; i < end; ++i) require(p[i] == 0,label+" nonzero padding");
}

// Complete 64-byte boundary spans, retaining original Runtime allocation bases.
struct MsmGuard {
    uint8_t* begin = nullptr;
    uint8_t* payload = nullptr;
    uint8_t* after = nullptr;
    size_t bytes = 0;
    std::vector<uint8_t> initial;
    MsmGuard(Runtime& runtime, aclrtStream stream, size_t size, const void* source = nullptr,
             bool pinned = false) : bytes(size)
    {
        if (!size) return;
        initial.assign(size+128,0xcd);
        std::fill(initial.begin(),initial.begin()+64,0xa5);
        std::fill(initial.end()-64,initial.end(),0x5a);
        if (source) std::memcpy(initial.data()+64,source,size);
        if (pinned) {
            begin = msm_align64(runtime.host<uint8_t>(size+191));
            std::memcpy(begin,initial.data(),initial.size());
        } else {
            begin = msm_align64(runtime.allocate<uint8_t>(size+191));
            runtime.upload(stream,begin,initial.data(),initial.size());
        }
        payload = begin+64;
        after = pinned ? begin : runtime.host<uint8_t>(size+128);
    }
    void download(Runtime& runtime, aclrtStream stream) {
        if (begin && begin != after) runtime.copy_down(stream,after,begin,bytes+128);
    }
    void verify(bool immutable, const std::string& label) const {
        if (!begin) return;
        require(std::memcmp(after,initial.data(),64) == 0 &&
                std::memcmp(after+64+bytes,initial.data()+64+bytes,64) == 0,label+" guard changed");
        if (immutable)
            require(std::memcmp(after+64,initial.data()+64,bytes) == 0,label+" input changed");
    }
};

struct MsmRecord { uint32_t bucket; uint32_t index; };
static_assert(sizeof(MsmRecord) == 8,"MSM record layout");
bool msm_same_record(const MsmRecord& a, const MsmRecord& b)
{
    return a.bucket == b.bucket && a.index == b.index;
}
void test_msm_metadata_case(Runtime& runtime, aclrtStream stream, size_t n, unsigned pattern)
{
    const auto mark = runtime.storage_mark();
    detail::MsmPlan plan{};
    require(detail::build_msm_plan(n,plan,16),"metadata MSM platform initialization failed");
    MsmGuard scratch(runtime,stream,plan.workspace_bytes);
    std::vector<MsmRecord> expected(n);
    for (size_t i = 0; i < n; ++i) {
        uint32_t key = 0;
        if (pattern == 1) key = 0xfedc;
        if (pattern == 2) key = uint32_t((i%2 ? 0xabcd : 0x1234) ^ ((i%16)<<8));
        if (pattern == 3) key = uint32_t((i%23)*0x101);
        if (pattern == 4) key = uint32_t((i*4051+17)&65535);
        expected[i] = {key,i%3 == 0 ? UINT32_MAX-1 : uint32_t((1u<<24)+i)};
    }
    std::vector<MsmRecord> input(msm_round8(n),{0,UINT32_MAX});
    std::copy(expected.begin(),expected.end(),input.begin());
    runtime.upload(stream,reinterpret_cast<MsmRecord*>(msm_region(scratch.payload,plan,
                   detail::MsmRegion::RecordsA)),input.data(),input.size());
    const std::string label = "MSM metadata N="+std::to_string(n)+" pattern="+std::to_string(pattern);
    for (uint32_t pass = 0; pass < plan.radix_passes; ++pass) {
        std::vector<std::array<uint32_t,16>> counts(plan.chunks);
        for (size_t i = 0; i < n; ++i) ++counts[i/2048][(expected[i].bucket>>(4*pass))&15];
        detail::launch_msm_radix_pass(stream,scratch.payload,plan,pass);
        auto* dense = runtime.download(stream,reinterpret_cast<const MsmRecord*>(msm_region(
            scratch.payload,plan,pass%2 ? detail::MsmRegion::RecordsA : detail::MsmRegion::RecordsB)),
            msm_round8(n));
        auto* hist = runtime.download(stream,msm_region(scratch.payload,plan,
            detail::MsmRegion::RadixHistogram),size_t(plan.chunks)*64);
        auto* descriptors = runtime.download(stream,msm_region(scratch.payload,plan,
            detail::MsmRegion::RadixDescriptors),size_t(plan.chunks)*16*64);
        auto* headers = runtime.download(stream,msm_region(scratch.payload,plan,
            detail::MsmRegion::RadixDigitTotals),size_t(16*64));
        auto* fragments = runtime.download(stream,reinterpret_cast<const MsmRecord*>(msm_region(
            scratch.payload,plan,detail::MsmRegion::RadixFragments)),size_t(plan.fragment_records));
        check(aclrtSynchronizeStream(stream),"MSM radix diagnostic synchronize");
        for (size_t chunk = 0; chunk < plan.chunks; ++chunk)
            for (size_t digit = 0; digit < 16; ++digit)
                require(msm_u32(hist+chunk*64+digit*4) == counts[chunk][digit],label+" histogram");
        uint64_t logical_base = 0, physical_base = 0;
        for (size_t digit = 0; digit < 16; ++digit) {
            uint64_t logical = 0, physical = 0;
            const auto* header = headers+digit*64;
            for (size_t chunk = 0; chunk < plan.chunks; ++chunk) {
                const uint64_t count = counts[chunk][digit];
                const auto* descriptor = descriptors+(digit*plan.chunks+chunk)*64;
                require(msm_u64(descriptor) == logical && msm_u64(descriptor+8) == physical &&
                        msm_u32(descriptor+16) == count,label+" fragment descriptor");
                msm_zero_padding(descriptor,20,64,label+" descriptor");
                require((physical_base+physical)%8 == 0 &&
                        physical_base+physical+msm_round8(count) <= plan.fragment_records,
                        label+" physical span");
                size_t rank = 0;
                for (size_t i = chunk*2048; i < std::min(n,(chunk+1)*2048); ++i)
                    if (((expected[i].bucket>>(4*pass))&15) == digit)
                        require(msm_same_record(fragments[physical_base+physical+rank++],expected[i]),
                                label+" stable fragment");
                for (size_t i = count; i < msm_round8(count); ++i)
                    require(fragments[physical_base+physical+i].bucket == 0 &&
                            fragments[physical_base+physical+i].index == UINT32_MAX,label+" fragment tail");
                logical += count; physical += msm_round8(count);
            }
            require(msm_u64(header) == logical && msm_u64(header+8) == physical &&
                    msm_u64(header+16) == logical_base && msm_u64(header+24) == physical_base,
                    label+" digit header");
            msm_zero_padding(header,32,64,label+" header");
            logical_base += logical; physical_base += physical;
        }
        require(logical_base == n && physical_base <= plan.fragment_records,label+" radix totals");
        const uint32_t mask = (1u<<(4*(pass+1)))-1;
        std::stable_sort(expected.begin(),expected.end(),[mask](const auto& a,const auto& b) {
            return (a.bucket&mask) < (b.bucket&mask);
        });
        for (size_t i = 0; i < n; ++i) require(msm_same_record(dense[i],expected[i]),label+" stable dense");
        for (size_t i = n; i < msm_round8(n); ++i)
            require(dense[i].bucket == 0 && dense[i].index == UINT32_MAX,label+" dense tail");
    }
    detail::launch_msm_rows(stream,scratch.payload,plan);
    auto* rows = runtime.download(stream,reinterpret_cast<const uint64_t*>(msm_region(
        scratch.payload,plan,detail::MsmRegion::RowPtr)),msm_round8(plan.buckets+1));
    check(aclrtSynchronizeStream(stream),"MSM CSR diagnostic synchronize");
    std::vector<uint64_t> model_rows(plan.buckets+1);
    for (uint32_t b = 0; b <= plan.buckets; ++b) {
        model_rows[b] = std::lower_bound(expected.begin(),expected.end(),b,
            [](const MsmRecord& record,uint32_t key) { return record.bucket < key; })-expected.begin();
        require(rows[b] == model_rows[b],label+" CSR lower_bound");
    }
    const size_t stride = msm_round8(plan.buckets+1);
    for (size_t i = plan.buckets+1; i < stride; ++i)
        require(rows[i] == n,label+" CSR tail padding");
    std::vector<uint64_t> prefix(stride*(plan.merge_levels+1),UINT64_MAX);
    detail::MsmSchedule schedule{};
    require(detail::make_msm_schedule(plan,rows,prefix.data(),schedule),label+" scheduler rejected CSR");
    uint64_t divisor = 64;
    uint32_t model_levels = 0;
    uint64_t max_tasks = 0;
    for (uint32_t b = 1; b < plan.buckets; ++b)
        max_tasks = std::max(max_tasks,(model_rows[b+1]-model_rows[b]+63)/64);
    for (uint64_t q = max_tasks; q > 1; q = (q+63)/64) ++model_levels;
    require(schedule.merge_levels == model_levels,label+" actual merge levels");
    std::vector<std::vector<uint64_t>> model_prefix(model_levels+1,std::vector<uint64_t>(stride));
    for (uint32_t level = 0; level <= model_levels; ++level) {
        auto& q = model_prefix[level];
        for (uint32_t b = 0; b < plan.buckets; ++b)
            q[b+1] = q[b]+(b ? (model_rows[b+1]-model_rows[b]+divisor-1)/divisor : 0);
        std::fill(q.begin()+plan.buckets+1,q.end(),q[plan.buckets]);
        require(schedule.task_count[level] == q[plan.buckets],label+" task total");
        for (size_t i = 0; i < stride; ++i)
            require(prefix[level*stride+i] == q[i],label+" task prefix");
        divisor *= 64;
    }
    for (uint32_t level = model_levels+1; level < 6; ++level)
        require(schedule.task_count[level] == 0,label+" inactive task count");
    for (size_t i = stride*(model_levels+1); i < prefix.size(); ++i)
        require(prefix[i] == UINT64_MAX,label+" inactive prefix overwritten");
    if (max_tasks) {
        runtime.upload(stream,reinterpret_cast<uint64_t*>(msm_region(scratch.payload,plan,
            detail::MsmRegion::TaskPrefix)),prefix.data(),stride*(model_levels+1));
        for (uint32_t level = 0; level <= model_levels; ++level) {
            detail::launch_msm_tasks(stream,scratch.payload,plan,level,schedule.task_count[level]);
            auto* tasks = runtime.download(stream,msm_region(scratch.payload,plan,
                detail::MsmRegion::Tasks),size_t(schedule.task_count[level])*64);
            check(aclrtSynchronizeStream(stream),"MSM tasks diagnostic synchronize");
            const auto& q = model_prefix[level];
            const auto& previous = level ? model_prefix[level-1] : model_rows;
            for (uint32_t b = 1; b < plan.buckets; ++b)
                for (uint64_t t = q[b]; t < q[b+1]; ++t) {
                    const uint64_t begin = previous[b]+64*(t-q[b]);
                    const uint32_t count = uint32_t(std::min<uint64_t>(64,previous[b+1]-begin));
                    const auto* descriptor = tasks+t*64;
                    require(msm_u64(descriptor) == begin && msm_u32(descriptor+8) == count &&
                            msm_u32(descriptor+12) == b,label+" expanded task descriptor");
                    msm_zero_padding(descriptor,16,64,label+" task descriptor");
                }
        }
    } else {
        // Same empty dispatch condition as production: no upload or point task.
        audit::begin_runtime_audit(nullptr,0);
        detail::launch_msm_window_reduce(stream,scratch.payload,plan,0,0,true);
        const auto empty = audit::end_runtime_audit();
        require_no_runtime_transfers(empty);
        require(empty.host_malloc_calls == 0 && empty.host_free_calls == 0 &&
                empty.unexpected_copy_calls == 0,label+" empty dispatch allocated or copied");
    }
    scratch.download(runtime,stream);
    check(aclrtSynchronizeStream(stream),"MSM metadata guard synchronize");
    scratch.verify(false,label);
    runtime.release_storage(mark);
}
void test_msm_metadata(Runtime& runtime, aclrtStream stream)
{
    for (size_t n : {1,7,8,9,2047,2048,2049,4097})
        for (unsigned pattern = 0; pattern < 5; ++pattern)
            test_msm_metadata_case(runtime,stream,n,pattern);
    std::cout << "MSM_METADATA_NPU_PASS\n";
}

const std::vector<std::pair<std::string,size_t>>& msm_cases()
{
    static const std::vector<std::pair<std::string,size_t>> cases = [] {
        std::vector<std::pair<std::string,size_t>> result;
        for (size_t n : {0,1,2,15,16,17,63,64,65,255,256,257,2047,2048,2049,4097,8193,65536,47,48,49,95,96,97})
            result.emplace_back("uniform-"+std::to_string(n),n);
        for (const auto& item : std::vector<std::pair<std::string,size_t>>{
            {"zero-2049",2049},{"ones-2049",2049},{"hot-65537",65537},
            {"identity-2049",2049},{"cancel-2048",2048},{"alternating-2049",2049},{"boundary-257",257},
            {"mixed-exceptions-65",65},{"mixed-vote-tail-192",192}})
            result.push_back(item);
        return result;
    }();
    return cases;
}
size_t msm_case_count(const std::string& name)
{
    for (const auto& item : msm_cases()) if (item.first == name) return item.second;
    throw std::runtime_error("unknown MSM fixture: "+name);
}
struct MsmFixture {
    std::string name;
    size_t count;
    std::vector<math::G1Affine> bases;
    std::vector<Fr> scalars;
    std::vector<math::G1Affine> expected;
    MsmFixture(const std::filesystem::path& root, const std::string& fixture)
        : name(fixture), count(msm_case_count(fixture)),
          bases(read_g1_affine(root/"msm"/fixture/"bases.bin",count)),
          scalars(read_fields(root/"msm"/fixture/"scalars.bin",count)),
          expected(read_g1_affine(root/"msm"/fixture/"expected.bin",1)) {}
};
detail::MsmPlan msm_plan(size_t count, uint32_t bits)
{
    detail::MsmPlan result{};
    require(detail::build_msm_plan(count,result,bits),"Ascend910B3 MSM platform initialization failed");
    return result;
}
struct MsmBudget {
    uint64_t d2h = 0, h2d = 0, windows = 0, active = 0;
    void add(const MsmBudget& other) {
        d2h += other.d2h; h2d += other.h2d; windows += other.windows; active += other.active;
    }
};
MsmBudget msm_budget(const MsmFixture& fixture, const detail::MsmPlan& plan)
{
    MsmBudget result{};
    if (!fixture.count) return result;
    const uint32_t c = plan.window_bits;
    const uint32_t buckets = 1u<<c;
    result.windows = (255+c-1)/c;
    require(result.windows == plan.windows && buckets == plan.buckets,"MSM budget shape mismatch");
    result.d2h = result.windows*8*(uint64_t(buckets)+1);
    const uint64_t prefix_bytes = msm_round8(size_t(buckets)+1)*8;
    for (uint32_t w = 0; w < result.windows; ++w) {
        std::vector<uint64_t> lengths(buckets);
        for (const auto& scalar : fixture.scalars) {
            uint32_t digit = 0;
            // Per-bit extraction is intentionally independent of the production
            // word/shift implementation, including a partial top window.
            for (uint32_t bit = 0; bit < c && w*c+bit < 255; ++bit) {
                const uint32_t position = w*c+bit;
                digit |= ((scalar.words[position/32]>>(position%32))&1u)<<bit;
            }
            ++lengths[digit];
        }
        uint64_t max_length = 0;
        for (uint32_t b = 1; b < buckets; ++b) max_length = std::max(max_length,lengths[b]);
        if (!max_length) continue;
        ++result.active;
        uint32_t levels = 0;
        for (uint64_t tasks = (max_length+63)/64; tasks > 1; tasks = (tasks+63)/64) ++levels;
        result.h2d += prefix_bytes*(levels+1);
    }
    return result;
}
void msm_require_resources_zero(const audit::RuntimeCounts& counts, const std::string& label)
{
    require(counts.other_copy_calls == 0 && counts.device_sync_calls == 0 &&
            counts.malloc_calls == 0 && counts.free_calls == 0 &&
            counts.host_malloc_calls == 0 && counts.host_free_calls == 0 &&
            counts.unexpected_copy_calls == 0,label+" unexpected copy/allocation/free/device sync:"
            +" other_copy="+std::to_string(counts.other_copy_calls)
            +" device_sync="+std::to_string(counts.device_sync_calls)
            +" device_malloc="+std::to_string(counts.malloc_calls)
            +" device_free="+std::to_string(counts.free_calls)
            +" host_malloc="+std::to_string(counts.host_malloc_calls)
            +" host_free="+std::to_string(counts.host_free_calls)
            +" unexpected_copy="+std::to_string(counts.unexpected_copy_calls));
}
struct MsmInvocation {
    const MsmFixture* fixture;
    const detail::MsmPlan* plan;
    aclrtStream stream;
    MsmGuard* bases;
    MsmGuard* scalars;
    MsmGuard* scratch;
    MsmGuard* host;
    MsmGuard* out;
};
void msm_execute(Runtime& runtime, const std::vector<MsmInvocation>& invocations,
                 const std::string& label, const std::string& marker)
{
    MsmBudget budget{};
    uint64_t count = 0;
    std::vector<aclrtStream> streams;
    std::vector<math::G1Affine*> actual;
    for (const auto& call : invocations) {
        const auto& plan = *call.plan;
        budget.add(msm_budget(*call.fixture,plan));
        count += call.fixture->count;
        if (std::find(streams.begin(),streams.end(),call.stream) == streams.end()) streams.push_back(call.stream);
        actual.push_back(runtime.host<math::G1Affine>(1));
    }
    audit::RuntimeCounts compute{};
    for (const auto& call : invocations) {
        const auto& plan = *call.plan;
        std::array<audit::RuntimeCopyAllowance,2> allowances{};
        if (plan.count) {
            allowances[0] = {msm_region(call.scratch->payload,plan,detail::MsmRegion::RowPtr),
                             call.host->payload,size_t(8*(uint64_t(plan.buckets)+1)),false};
            allowances[1] = {msm_region(call.scratch->payload,plan,detail::MsmRegion::TaskPrefix),
                             call.host->payload+plan.host_prefix_offset,
                             size_t(plan.region_size[static_cast<uint32_t>(detail::MsmRegion::TaskPrefix)]),true};
        }
        // Each call has exactly its own two metadata endpoints, even in replay
        // and two-stream batches. No setup or tail operations enter this window.
        audit::begin_runtime_audit(allowances.data(),plan.count ? 2 : 0);
        check(detail::run_msm(call.stream,reinterpret_cast<const math::G1Affine*>(call.bases->payload),
            reinterpret_cast<const Fr*>(call.scalars->payload),reinterpret_cast<math::G1Affine*>(call.out->payload),
            call.scratch->payload,call.scratch->bytes,call.host->payload,call.host->bytes,*call.plan),"run_msm");
        const auto one = audit::end_runtime_audit();
        msm_require_resources_zero(one,label+" compute");
        compute.d2h_calls += one.d2h_calls; compute.d2h_bytes += one.d2h_bytes;
        compute.h2d_calls += one.h2d_calls; compute.h2d_bytes += one.h2d_bytes;
        compute.stream_sync_calls += one.stream_sync_calls;
    }
    require(compute.d2h_calls == budget.windows && compute.d2h_bytes == budget.d2h &&
            compute.h2d_calls == budget.active && compute.h2d_bytes == budget.h2d &&
            compute.stream_sync_calls == budget.windows,label+" metadata budget mismatch");
    audit::begin_runtime_audit();
    for (size_t i = 0; i < invocations.size(); ++i)
        runtime.copy_down(invocations[i].stream,actual[i],
                         reinterpret_cast<const math::G1Affine*>(invocations[i].out->payload),1);
    for (auto stream : streams) check(aclrtSynchronizeStream(stream),"MSM final synchronize");
    const auto tail = audit::end_runtime_audit();
    msm_require_resources_zero(tail,label+" final");
    require(tail.d2h_calls == invocations.size() && tail.d2h_bytes == 96*invocations.size() &&
            tail.h2d_calls == 0 && tail.h2d_bytes == 0 && tail.stream_sync_calls == streams.size(),
            label+" final download/sync budget mismatch");
    std::vector<MsmGuard*> downloaded;
    for (size_t i = 0; i < invocations.size(); ++i) {
        const auto& call = invocations[i];
        compare_g1_affine(actual[i],call.fixture->expected,label+" "+call.fixture->name);
        for (auto* guard : {call.bases,call.scalars,call.scratch,call.out})
            if (std::find(downloaded.begin(),downloaded.end(),guard) == downloaded.end()) {
                guard->download(runtime,call.stream);
                downloaded.push_back(guard);
            }
    }
    for (auto stream : streams) check(aclrtSynchronizeStream(stream),"MSM guard synchronize");
    for (const auto& call : invocations) {
        call.bases->verify(true,label+" bases"); call.scalars->verify(true,label+" scalars");
        call.scratch->verify(false,label+" device scratch"); call.host->verify(false,label+" pinned scratch");
        call.out->verify(false,label+" out");
    }
    std::cout << marker << " case=" << label << " count=" << count
              << " window_bits=" << invocations.front().plan->window_bits
              << " metadata_d2h=" << budget.d2h << " metadata_h2d=" << budget.h2d
              << " metadata_sync=" << budget.windows
              << " unexpected_copy=0 compute_device_alloc=0 compute_host_alloc=0 compute_free=0"
              << " final_d2h=" << 96*invocations.size() << " final_sync=" << streams.size() << '\n';
}
void test_msm_fixture(Runtime& runtime, aclrtStream stream, const std::filesystem::path& root,
                      const std::string& name, uint32_t bits)
{
    const auto mark = runtime.storage_mark();
    MsmFixture fixture(root,name);
    const auto plan = msm_plan(fixture.count,bits);
    std::cout << "MSM_PLAN case=" << name << " count=" << plan.count
              << " window_bits=" << plan.window_bits << " windows=" << plan.windows
              << " aiv_cores=" << plan.platform.aiv_cores << " ub_bytes=" << plan.platform.ub_bytes
              << " device_workspace_bytes=" << plan.workspace_bytes
              << " host_workspace_bytes=" << plan.host_workspace_bytes << '\n';
    MsmGuard bases(runtime,stream,fixture.count*96,fixture.bases.data());
    MsmGuard scalars(runtime,stream,fixture.count*32,fixture.scalars.data());
    MsmGuard scratch(runtime,stream,plan.workspace_bytes);
    MsmGuard host(runtime,stream,plan.host_workspace_bytes,nullptr,true);
    MsmGuard out(runtime,stream,96);
    msm_execute(runtime,{{&fixture,&plan,stream,&bases,&scalars,&scratch,&host,&out}},
                name,"MSM_CUZK_RESIDENT_PASS");
    runtime.release_storage(mark);
}

void test_msm_preflight(Runtime& runtime, aclrtStream stream, const std::filesystem::path& root)
{
    const auto mark = runtime.storage_mark();
    MsmFixture fixture(root,"uniform-65");
    const auto plan = msm_plan(fixture.count,4);
    MsmGuard bases(runtime,stream,fixture.count*96,fixture.bases.data());
    MsmGuard scalars(runtime,stream,fixture.count*32,fixture.scalars.data());
    MsmGuard scratch(runtime,stream,plan.workspace_bytes);
    MsmGuard host(runtime,stream,plan.host_workspace_bytes,nullptr,true);
    MsmGuard out(runtime,stream,96);
    struct Input {
        aclrtStream stream;
        const math::G1Affine* bases;
        const Fr* scalars;
        math::G1Affine* out;
        void* device;
        size_t device_bytes;
        void* host;
        size_t host_bytes;
        const detail::MsmPlan* plan;
    };
    const Input valid{stream,reinterpret_cast<const math::G1Affine*>(bases.payload),
        reinterpret_cast<const Fr*>(scalars.payload),reinterpret_cast<math::G1Affine*>(out.payload),
        scratch.payload,scratch.bytes,host.payload,host.bytes,&plan};
    const auto reject = [](const Input& input, const std::string& reason) {
        require(detail::run_msm(input.stream,input.bases,input.scalars,input.out,input.device,
                input.device_bytes,input.host,input.host_bytes,*input.plan) == ACL_ERROR_INVALID_PARAM,
                "MSM preflight accepted "+reason);
    };
    audit::begin_runtime_audit(nullptr,0);
    Input bad = valid;
    bad.stream = nullptr; reject(bad,"null stream");
    bad = valid; bad.bases = nullptr; reject(bad,"null bases");
    bad = valid; bad.scalars = nullptr; reject(bad,"null scalars");
    bad = valid; bad.out = nullptr; reject(bad,"null out");
    bad = valid; bad.device = nullptr; reject(bad,"null device scratch");
    bad = valid; bad.host = nullptr; reject(bad,"null host scratch");
    bad = valid; bad.device_bytes = 0; reject(bad,"missing device capacity");
    bad = valid; --bad.device_bytes; reject(bad,"short device scratch");
    bad = valid; bad.host_bytes = 0; reject(bad,"missing host capacity");
    bad = valid; --bad.host_bytes; reject(bad,"short host scratch");
    bad = valid; bad.bases = reinterpret_cast<const math::G1Affine*>(bases.payload+1); reject(bad,"unaligned bases");
    bad = valid; bad.scalars = reinterpret_cast<const Fr*>(scalars.payload+1); reject(bad,"unaligned scalars");
    bad = valid; bad.out = reinterpret_cast<math::G1Affine*>(out.payload+1); reject(bad,"unaligned out");
    bad = valid; bad.device = scratch.payload+32; reject(bad,"32-only aligned device scratch");
    bad = valid; bad.host = host.payload+32; reject(bad,"32-only aligned host scratch");
    bad = valid; bad.out = reinterpret_cast<math::G1Affine*>(scratch.payload); reject(bad,"out/device overlap");
    bad = valid; bad.out = reinterpret_cast<math::G1Affine*>(bases.payload); reject(bad,"out/bases overlap");
    bad = valid; bad.out = reinterpret_cast<math::G1Affine*>(scalars.payload); reject(bad,"out/scalars overlap");
    bad = valid; bad.bases = reinterpret_cast<const math::G1Affine*>(scratch.payload); reject(bad,"bases/device overlap");
    bad = valid; bad.scalars = reinterpret_cast<const Fr*>(scratch.payload); reject(bad,"scalars/device overlap");
    bad = valid; bad.host = scratch.payload; reject(bad,"host/device overlap");
    bad = valid; bad.host = bases.payload; reject(bad,"host/bases overlap");
    bad = valid; bad.host = scalars.payload; reject(bad,"host/scalars overlap");
    bad = valid; bad.host = out.payload; reject(bad,"host/out overlap");
    const uintptr_t last_aligned = std::numeric_limits<uintptr_t>::max()&~uintptr_t(63);
    bad = valid; bad.bases = reinterpret_cast<const math::G1Affine*>(last_aligned); reject(bad,"bases address wrap");
    bad = valid; bad.scalars = reinterpret_cast<const Fr*>(last_aligned); reject(bad,"scalars address wrap");
    bad = valid; bad.out = reinterpret_cast<math::G1Affine*>(last_aligned); reject(bad,"out address wrap");
    bad = valid; bad.device = reinterpret_cast<void*>(last_aligned); reject(bad,"device address wrap");
    bad = valid; bad.host = reinterpret_cast<void*>(last_aligned); reject(bad,"host address wrap");
    // Forge derived fields; a count change within the same rounded shape is a
    // valid new plan, so cross a task-capacity boundary when changing count.
    const auto reject_plan = [&](const detail::MsmPlan& changed) {
        auto input = valid; input.plan = &changed; reject(input,"forged plan");
    };
    auto changed = plan;
    changed.platform.aiv_cores = 0; reject_plan(changed);
    changed = plan; changed.platform.ub_bytes = 65535; reject_plan(changed);
    changed = plan; changed.count += 64; reject_plan(changed);
    changed = plan; ++changed.workspace_bytes; reject_plan(changed);
    changed = plan; ++changed.dense_records; reject_plan(changed);
    changed = plan; ++changed.fragment_records; reject_plan(changed);
    changed = plan; ++changed.host_workspace_bytes; reject_plan(changed);
    changed = plan; ++changed.host_prefix_offset; reject_plan(changed);
    changed = plan; changed.window_bits = 17; reject_plan(changed);
    changed = plan; ++changed.windows; reject_plan(changed);
    changed = plan; ++changed.buckets; reject_plan(changed);
    changed = plan; ++changed.radix_passes; reject_plan(changed);
    changed = plan; ++changed.chunks; reject_plan(changed);
    changed = plan; ++changed.merge_levels; reject_plan(changed);
    changed = plan; ++changed.row_blocks; reject_plan(changed);
    changed = plan; ++changed.weighted_segments; reject_plan(changed);
    changed = plan; ++changed.weighted_levels; reject_plan(changed);
    for (unsigned i = 0; i < 2; ++i) {
        changed = plan; ++changed.partial_capacity[i]; reject_plan(changed);
    }
    for (unsigned i = 0; i < 17; ++i) {
        changed = plan; ++changed.region_offset[i]; reject_plan(changed);
        changed = plan; ++changed.region_size[i]; reject_plan(changed);
    }
    const auto counts = audit::end_runtime_audit();
    require_no_runtime_transfers(counts);
    msm_require_resources_zero(counts,"MSM rejected calls");
    for (auto* guard : {&bases,&scalars,&scratch,&out}) guard->download(runtime,stream);
    check(aclrtSynchronizeStream(stream),"MSM preflight sentinel synchronize");
    for (auto* guard : {&bases,&scalars,&scratch,&host,&out}) guard->verify(true,"MSM preflight sentinel");
    runtime.release_storage(mark);
    std::cout << "MSM_PREFLIGHT_NPU_PASS\n";
}
void test_msm_replay(Runtime& runtime, aclrtStream stream, const std::filesystem::path& root)
{
    const auto mark = runtime.storage_mark();
    std::vector<MsmFixture> fixtures;
    for (const char* name : {"uniform-2049","zero-2049","ones-2049"}) fixtures.emplace_back(root,name);
    const auto plan = msm_plan(2049,0);
    MsmGuard scratch(runtime,stream,plan.workspace_bytes);
    MsmGuard host(runtime,stream,plan.host_workspace_bytes,nullptr,true);
    std::vector<MsmGuard> bases, scalars, outputs;
    bases.reserve(3); scalars.reserve(3); outputs.reserve(3);
    for (const auto& fixture : fixtures) {
        bases.emplace_back(runtime,stream,fixture.count*96,fixture.bases.data());
        scalars.emplace_back(runtime,stream,fixture.count*32,fixture.scalars.data());
        outputs.emplace_back(runtime,stream,96);
    }
    std::vector<MsmInvocation> calls;
    for (size_t i = 0; i < fixtures.size(); ++i)
        calls.push_back({&fixtures[i],&plan,stream,&bases[i],&scalars[i],&scratch,&host,&outputs[i]});
    // No scratch clearing or result downloads between these three run_msm calls.
    msm_execute(runtime,calls,"uniform-zero-ones-2049","MSM_CUZK_REPLAY_PASS");
    runtime.release_storage(mark);
}
void test_msm_two_streams(Runtime& runtime, const std::filesystem::path& root)
{
    const auto mark = runtime.storage_mark();
    const auto a = runtime.stream(), b = runtime.stream();
    MsmFixture fixture(root,"uniform-257");
    const auto pa = msm_plan(fixture.count,3), pb = msm_plan(fixture.count,4);
    MsmGuard bases(runtime,a,fixture.count*96,fixture.bases.data());
    MsmGuard scalars(runtime,a,fixture.count*32,fixture.scalars.data());
    MsmGuard sa(runtime,a,pa.workspace_bytes), sb(runtime,b,pb.workspace_bytes);
    MsmGuard ha(runtime,a,pa.host_workspace_bytes,nullptr,true), hb(runtime,b,pb.host_workspace_bytes,nullptr,true);
    MsmGuard oa(runtime,a,96), ob(runtime,b,96);
    // Publish the shared read-only input before either stream consumes it.
    check(aclrtSynchronizeStream(a),"MSM shared-input setup synchronize");
    // Sequential host submission is intentional: run_msm itself waits per window.
    msm_execute(runtime,{{&fixture,&pa,a,&bases,&scalars,&sa,&ha,&oa},
                         {&fixture,&pb,b,&bases,&scalars,&sb,&hb,&ob}},
                "shared-input-257-c3-c4","MSM_CUZK_TWO_STREAM_PASS");
    runtime.release_storage(mark);
}
void test_msm_suite(Runtime& runtime, aclrtStream stream, const std::filesystem::path& root)
{
    test_msm_metadata(runtime,stream);
    test_msm_preflight(runtime,stream,root);
    for (const auto& item : msm_cases()) {
        if (item.first == "uniform-65536") continue; // Explicit profiling CLI only.
        const uint32_t window = item.first == "boundary-257" ? 4 :
            item.first == "mixed-vote-tail-192" ? 6 : 0;
        test_msm_fixture(runtime,stream,root,item.first,window);
    }
    test_msm_fixture(runtime,stream,root,"uniform-257",3);
    test_msm_fixture(runtime,stream,root,"uniform-257",4);
    test_msm_fixture(runtime,stream,root,"alternating-2049",12);
    test_msm_fixture(runtime,stream,root,"uniform-4097",16);
    test_msm_replay(runtime,stream,root);
    test_msm_two_streams(runtime,root);
}

struct Arguments {
    int32_t device;
    std::filesystem::path fixtures;
    std::string msm_case;
    uint32_t window_bits = 0;
    bool window_override = false;
    std::string field_mode;
};

Arguments arguments(int argc, char** argv)
{
    Arguments result{};
    bool have_device = false;
    bool have_fixtures = false;
    bool have_msm_case = false;
    for (int i = 1; i < argc; ++i) {
        const std::string option(argv[i]);
        require(i + 1 < argc, "usage: native_math_smoke --device <ACL ID> --fixtures <directory>");
        const std::string value(argv[++i]);
        if (option == "--device" && !have_device) {
            require(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos,
                    "invalid ACL device ID");
            size_t consumed = 0;
            const long long parsed = std::stoll(value, &consumed);
            require(consumed == value.size() && parsed >= 0 && parsed <= std::numeric_limits<int32_t>::max(),
                    "invalid ACL device ID");
            result.device = static_cast<int32_t>(parsed);
            have_device = true;
        } else if (option == "--fixtures" && !have_fixtures) {
            result.fixtures = value;
            have_fixtures = true;
        } else if (option == "--field-mode" && result.field_mode.empty()) {
            require(value == "check" || value == "bench" || value == "profile", "invalid field mode");
            result.field_mode = value;
        } else if (option == "--msm-case" && !have_msm_case) {
            (void)msm_case_count(value);
            result.msm_case = value;
            have_msm_case = true;
        } else if (option == "--window-bits" && !result.window_override) {
            require(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos,
                    "window bits must be 0 or 2..16");
            size_t consumed = 0;
            const auto parsed = std::stoul(value,&consumed);
            require(consumed == value.size() && (parsed == 0 || (parsed >= 2 && parsed <= 16)),
                    "window bits must be 0 or 2..16");
            result.window_bits = uint32_t(parsed);
            result.window_override = true;
        } else {
            throw std::runtime_error("unknown or repeated argument: " + option);
        }
    }
    require(have_device && have_fixtures,
            "usage: native_math_smoke --device <ACL ID> --fixtures <directory>");
    require(!result.window_override || have_msm_case,"--window-bits requires --msm-case");
    require(result.field_mode.empty() || (!have_msm_case && !result.window_override),
            "--field-mode conflicts with --msm-case/--window-bits");
    return result;
}
} // namespace

int main(int argc, char** argv)
{
    try {
        const auto options = arguments(argc, argv);
        if (!options.msm_case.empty()) {
            // This branch never builds the unrelated Fr Cube tilings.
            {
                Runtime runtime(options.device);
                const auto stream = runtime.stream();
                const uint32_t bits = options.window_override ? options.window_bits :
                    (options.msm_case == "boundary-257" ? 4 : 0);
                test_msm_fixture(runtime,stream,options.fixtures,options.msm_case,bits);
            }
            require(!cleanup_failed,"ACL resource cleanup failed");
            std::cout << "MSM_NPU_PASS\n";
            return 0;
        }
        test_sizes();
        Tiling dot{};
        Tiling scale{};
        uint32_t aiv_cores = 0;
        require(detail::build_math_tiling(dot, scale, aiv_cores) && aiv_cores != 0,
                "Ascend910B3 math tiling/platform initialization failed");
        {
            Runtime runtime(options.device);
            Queue queue(runtime, dot, scale, aiv_cores);
            if (options.field_mode == "bench") {
                field_bench<Fr>(runtime, queue, options.fixtures, "Fr");
                field_bench<math::Fq>(runtime, queue, options.fixtures / "fq", "Fq");
                for (const std::string operation : {"scale", "inner_product"})
                    for (size_t count : {33u, 257u, 4097u})
                        measure_field<Fr>(runtime, queue, options.fixtures, "Fr", operation, count, false);
            } else if (options.field_mode == "profile") {
                measure_field<Fr>(runtime, queue, options.fixtures, "Fr", "mul", 4097, true);
                measure_field<math::Fq>(runtime, queue, options.fixtures / "fq", "Fq", "mul", 4097, true);
                measure_field<Fr>(runtime, queue, options.fixtures, "Fr", "inner_product", 257, true);
            } else {
                test_matrix(runtime, queue);
                for (size_t count : case_counts) test_fixture(runtime, queue, options.fixtures, count);
                test_inverse_zeros(runtime, queue);
                test_readable_examples(runtime, queue);
                test_independent_streams(runtime, dot, scale, aiv_cores, options.fixtures);
                test_resident_chain(runtime, dot, scale, aiv_cores, options.fixtures, 513);
                test_resident_chain(runtime, dot, scale, aiv_cores, options.fixtures, 4097);
                // Both fields exercise the same production templates.
                test_generic_field<Fr>(runtime, queue.stream, aiv_cores, options.fixtures, "Fr");
                test_generic_field<math::Fq>(runtime, queue.stream, aiv_cores, options.fixtures / "fq", "Fq");
                if (options.field_mode.empty()) {
                    test_g1_primitives(runtime,queue.stream,aiv_cores,options.fixtures);
                    test_g1_chains(runtime,queue.stream,aiv_cores,options.fixtures);
                    test_msm_suite(runtime,queue.stream,options.fixtures);
                }
            }
        }
        require(!cleanup_failed, "ACL resource cleanup failed");
        if (options.field_mode.empty())
            std::cout << "G1_NPU_PASS\nMSM_NPU_PASS\nFIELD_MATH_NPU_PASS\n";
        else if (options.field_mode == "check")
            std::cout << "FIELD_MATH_NPU_PASS\nFIELD_ONLY_NPU_PASS\n";
        return 0;
    } catch (const std::exception& error) {
        // Disable a partially active counter before exception-path cleanup/exit.
        audit::end_runtime_audit();
        std::cerr << "FIELD_MATH_NPU_FAIL: " << error.what() << '\n';
        return 1;
    }
}
