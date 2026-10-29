#include "../math_kernels.h"
#include "../g1.h"
#include "../math_tiling.h"
#include "runtime_audit.h"
#include "smoke_kernels.h"

#include <acl/acl.h>

#include <algorithm>
#include <array>
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
#include <type_traits>

namespace {
namespace math = hyperplonk::ascend910b3;
namespace detail = hyperplonk::ascend910b3::detail;
namespace audit = hyperplonk::ascend910b3::test;
using math::Fr;
using Tiling = AscendC::tiling::TCubeTiling;
constexpr std::array<size_t, 11> case_counts{0, 1, 2, 31, 32, 33, 255, 256, 257, 513, 4097};
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
    return count == 0 || count == 1 || count == 2 || count == 33 || count == 257;
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
        detail::launch_add(queue.stream, a, b, output[0], count, blocks);
        detail::launch_sub(queue.stream, a, b, output[1], count, blocks);
        detail::launch_mul(queue.stream, a, b, output[2], count, blocks);
        detail::launch_scalar_vector_mul(queue.stream, count ? c : nullptr, a, output[3], count,
                                        queue.workspace, queue.scale, blocks);
        detail::launch_inner_product(queue.stream, a, b, dot, count, queue.workspace, queue.dot, blocks);
        if (do_inverse)
            detail::launch_inverse(queue.stream, inv_input, inv_output, mask, count, blocks);
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
        detail::launch_inverse(queue.stream, values, out, mask, input.size(), queue.blocks(input.size()));
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
    detail::launch_inner_product(queue.stream, da, db, dot, 3, queue.workspace, queue.dot, 1);
    detail::launch_scalar_vector_mul(queue.stream, dc, dv, scaled, 3, queue.workspace, queue.scale, 1);
    detail::launch_add(queue.stream, dc, da, added, 1, 1);
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

template<class Field>
bool launch_field_element(aclrtStream stream, const void* lhs, const Field* rhs, Field* out,
                          uint8_t* mask, size_t count, uint32_t cores, field_smoke::Operation op)
{
    // Empty tests exercise this exact path with null buffers; no kernel is enqueued.
    if (count == 0) return false;
    require(cores != 0, "generic field test has no AIV cores");
    const size_t tiles = count / field_smoke::tile_elements +
                         (count % field_smoke::tile_elements != 0);
    const uint32_t blocks = static_cast<uint32_t>(std::min<size_t>(cores, tiles));
    static_assert(std::is_same<Field,Fr>::value || std::is_same<Field,math::Fq>::value);
    if constexpr (std::is_same<Field,Fr>::value)
        audit::launch_fr_smoke(blocks,stream,device_address(lhs),device_address(rhs),
                               device_address(out),mask,count,op);
    else
        audit::launch_fq_smoke(blocks,stream,device_address(lhs),device_address(rhs),
                               device_address(out),mask,count,op);
    return true;
}

template<class Field>
void test_generic_inverse(Runtime& runtime, aclrtStream stream, uint32_t cores,
                          const std::vector<Field>& input, const std::vector<Field>& expected,
                          const std::string& label)
{
    require(input.size() == expected.size(), "inverse golden count mismatch");
    const size_t count = input.size();
    Field* values = runtime.allocate<Field>(count);
    Field* output = runtime.allocate<Field>(count);
    uint8_t* mask = runtime.allocate<uint8_t>(count);
    runtime.upload(stream, values, input.data(), count);
    Field poison{};
    poison.words[0] = 7;
    const std::vector<Field> poison_values(count, poison);
    const std::vector<uint8_t> poison_masks(count, 0xa5);
    runtime.upload(stream, output, poison_values.data(), count);
    runtime.upload(stream, mask, poison_masks.data(), count);
    require(launch_field_element<Field>(stream, values, nullptr, output, mask, count, cores,
                                        field_smoke::inverse) == (count != 0),
            "inverse empty-launch contract violated");
    Field* actual = runtime.download(stream, output, count);
    uint8_t* actual_mask = runtime.download(stream, mask, count);
    Field* after = runtime.download(stream, values, count);
    check(aclrtSynchronizeStream(stream), "generic inverse synchronize");
    compare_generic_fields(actual, expected, label);
    compare_generic_fields(after, input, label + " immutable input");
    for (size_t i = 0; i < count; ++i) {
        bool zero = true;
        for (size_t w = 0; w < Field::word_count; ++w) zero = zero && input[i].words[w] == 0;
        require(actual_mask[i] == (zero ? 0 : 1), label + " mask mismatch at " + std::to_string(i));
    }
}

template<class Field>
void test_generic_fixture(Runtime& runtime, aclrtStream stream, uint32_t cores,
                          const std::filesystem::path& root, size_t count, const std::string& field)
{
    const auto path = root / ("case-" + std::to_string(count));
    const auto lhs = read_generic_fields<Field>(path / "lhs.bin", count);
    const auto rhs = read_generic_fields<Field>(path / "rhs.bin", count);
    Field* a = runtime.allocate<Field>(count);
    Field* b = runtime.allocate<Field>(count);
    Field* output = runtime.allocate<Field>(count);
    runtime.upload(stream, a, lhs.data(), count);
    runtime.upload(stream, b, rhs.data(), count);
    const std::array<const char*, 5> names{"add", "sub", "neg", "mul", "square"};
    const std::array<field_smoke::Operation, 5> ops{
        field_smoke::add, field_smoke::sub, field_smoke::neg, field_smoke::mul, field_smoke::square};
    for (size_t op = 0; op < names.size(); ++op) {
        const auto expected = read_generic_fields<Field>(path / (std::string(names[op]) + ".bin"), count);
        require(launch_field_element<Field>(stream, a, b, output, nullptr, count, cores, ops[op]) ==
                    (count != 0), "generic empty-launch contract violated");
        Field* actual = runtime.download(stream, output, count);
        check(aclrtSynchronizeStream(stream), "generic arithmetic synchronize");
        compare_generic_fields(actual, expected, field + " " + names[op] + " count=" + std::to_string(count));
    }
    Field* after_a = runtime.download(stream, a, count);
    Field* after_b = runtime.download(stream, b, count);
    check(aclrtSynchronizeStream(stream), "generic immutability synchronize");
    compare_generic_fields(after_a, lhs, field + " immutable lhs");
    compare_generic_fields(after_b, rhs, field + " immutable rhs");
    if (inverse_case(count)) {
        auto input = read_generic_fields<Field>(path / "inverse_input.bin", count);
        auto expected = read_generic_fields<Field>(path / "inverse.bin", count);
        test_generic_inverse(runtime, stream, cores, input, expected, field + " inverse count=" +
                             std::to_string(count));
        if (count == 33) {
            // Independent nonzero inverse golden plus zero at both ends: two
            // tiles, a nonaligned mask tail, and poisoned out0/mask0 overwrite.
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
    constexpr size_t count = 512;
    constexpr size_t width = 2 * sizeof(Field) + 4;
    const auto input = read_fixture_bytes(root / "wide_input.bin", count * width);
    const auto expected = read_generic_fields<Field>(root / "wide_result.bin", count);
    uint8_t* values = runtime.allocate<uint8_t>(input.size());
    Field* output = runtime.allocate<Field>(count);
    runtime.upload(stream, values, input.data(), input.size());
    require(!launch_field_element<Field>(stream, nullptr, nullptr, nullptr, nullptr, 0, cores,
                                         field_smoke::wide), "empty wide reduction launched");
    require(launch_field_element<Field>(stream, values, nullptr, output, nullptr, count, cores,
                                        field_smoke::wide), "wide reduction did not launch");
    Field* actual = runtime.download(stream, output, count);
    uint8_t* after = runtime.download(stream, values, input.size());
    check(aclrtSynchronizeStream(stream), "generic wide synchronize");
    compare_generic_fields(actual, expected, field + " wide independent golden");
    require(std::memcmp(after, input.data(), input.size()) == 0, field + " wide input changed");
}

template<class Field>
void test_generic_field(Runtime& runtime, aclrtStream stream, uint32_t cores,
                        const std::filesystem::path& root, const std::string& field)
{
    for (size_t count : case_counts)
        test_generic_fixture<Field>(runtime, stream, cores, root, count, field);
    test_generic_wide<Field>(runtime, stream, cores, root, field);
    std::cout << "FIELD_ELEMENT_NPU_PASS field=" << field << '\n';
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
    for (const size_t count : {0,1,2,15,16,17,31,32,33,65}) {
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

struct Arguments {
    int32_t device;
    std::filesystem::path fixtures;
};

Arguments arguments(int argc, char** argv)
{
    Arguments result{};
    bool have_device = false;
    bool have_fixtures = false;
    for (int i = 1; i < argc; ++i) {
        const std::string option(argv[i]);
        require(i + 1 < argc, "usage: native_math_smoke --device <ACL ID> --fixtures <directory>");
        const std::string value(argv[++i]);
        if (option == "--device" && !have_device) {
            size_t consumed = 0;
            const long long parsed = std::stoll(value, &consumed);
            require(consumed == value.size() && parsed >= 0 && parsed <= std::numeric_limits<int32_t>::max(),
                    "invalid ACL device ID");
            result.device = static_cast<int32_t>(parsed);
            have_device = true;
        } else if (option == "--fixtures" && !have_fixtures) {
            result.fixtures = value;
            have_fixtures = true;
        } else {
            throw std::runtime_error("unknown or repeated argument: " + option);
        }
    }
    require(have_device && have_fixtures,
            "usage: native_math_smoke --device <ACL ID> --fixtures <directory>");
    return result;
}
} // namespace

int main(int argc, char** argv)
{
    try {
        const auto options = arguments(argc, argv);
        test_sizes();
        Tiling dot{};
        Tiling scale{};
        uint32_t aiv_cores = 0;
        require(detail::build_math_tiling(dot, scale, aiv_cores) && aiv_cores != 0,
                "Ascend910B3 math tiling/platform initialization failed");
        {
            Runtime runtime(options.device);
            Queue queue(runtime, dot, scale, aiv_cores);
            test_matrix(runtime, queue);
            for (size_t count : case_counts) test_fixture(runtime, queue, options.fixtures, count);
            test_inverse_zeros(runtime, queue);
            test_readable_examples(runtime, queue);
            test_independent_streams(runtime, dot, scale, aiv_cores, options.fixtures);
            test_resident_chain(runtime, dot, scale, aiv_cores, options.fixtures, 513);
            test_resident_chain(runtime, dot, scale, aiv_cores, options.fixtures, 4097);
            // Preserve the original Fr regression and its audit windows before
            // instantiating the test-only generic Fr/Fq device path.
            test_generic_field<Fr>(runtime, queue.stream, aiv_cores, options.fixtures, "Fr");
            test_generic_field<math::Fq>(runtime, queue.stream, aiv_cores, options.fixtures / "fq", "Fq");
            test_g1_primitives(runtime,queue.stream,aiv_cores,options.fixtures);
            test_g1_chains(runtime,queue.stream,aiv_cores,options.fixtures);
            std::cout << "G1_NPU_PASS\n";
        }
        require(!cleanup_failed, "ACL resource cleanup failed");
        std::cout << "FIELD_MATH_NPU_PASS\n";
        return 0;
    } catch (const std::exception& error) {
        // Disable a partially active counter before exception-path cleanup/exit.
        audit::end_runtime_audit();
        std::cerr << "FIELD_MATH_NPU_FAIL: " << error.what() << '\n';
        return 1;
    }
}
