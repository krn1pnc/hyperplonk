#pragma once
// Sole native runtime framework for the Ascend910B3 host bridge.
//
// Every bridge entry point (bridge.asc / bridge_mle.cpp / bridge_kzg.cpp /
// bridge_sumcheck.cpp) shares this header: one ACL runtime lifetime, one
// execution context type, one retained-device-buffer type and one checked entry
// guard. No peer may create a second context, allocator or ACL owner.
//
// Dataflow contract implemented here:
//  * A context owns the ACL context, one stream and the allocation registry.
//  * Immutable MLE handles are direct canonical-Fr device pointers; the public
//    `Ascend910B3DeviceAddress` value is exactly `BufferRef::data()`.
//  * `retire` removes registry ownership but physical free is deferred until
//    `aclrtStreamQuery` reports `ACL_STREAM_STATUS_COMPLETE`. Internal users
//    (sumcheck state, KZG params) hold `BufferRef` copies, so storage survives a
//    retired public handle.
//  * Async uploads copy the borrowed host bytes into owned pinned staging before
//    returning; staging is reclaimed only after completion observation.
//  * Host-result entry points synchronize and reclaim; `destroy` finally drains.
//  * No exception crosses the C ABI and no device/hot-field arithmetic happens
//    on the host. Only argument validation and control metadata are host-side.
#include "bridge.h"

#include <acl/acl.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

namespace hyperplonk::ascend910b3::native {

using DeviceAddress = Ascend910B3DeviceAddress;

// Host packet widths from bridge.h. Device MLE tables are canonical Fr (32B each).
constexpr size_t kFrBytes = 32;
constexpr size_t kG1AffineBytes = 96;
constexpr size_t kCommitmentPacketBytes = 97;
constexpr size_t kDeviceAlignment = 32;
constexpr size_t kScratchAlignment = 64;
constexpr size_t kPinnedAlignment = 64;
constexpr size_t kMaxAlignment = 4096;

// ---------------------------------------------------------------------------
// Checked size helpers (host control metadata only).
// ---------------------------------------------------------------------------

bool checked_add(size_t a, size_t b, size_t& out) noexcept;
bool checked_mul(size_t a, size_t b, size_t& out) noexcept;

// Canonical Fr table size for `num_vars` variables: (1 << num_vars) * element
// bytes, requiring num_vars < 64 and a byte count addressable by ptrdiff_t.
bool dimension_bytes(size_t num_vars, size_t element_bytes, size_t& out) noexcept;

// A canonical BLS12-381 Fr element: little-endian 32 bytes, value < r.
bool canonical_fr(const uint8_t* bytes) noexcept;
// Every one of `count` consecutive canonical Fr elements (may be null if count 0).
bool canonical_fr_array(const uint8_t* bytes, size_t count) noexcept;

// ---------------------------------------------------------------------------
// Status helpers.
// ---------------------------------------------------------------------------

inline Ascend910B3Status status_success() noexcept { return Ascend910B3Status{0, 0}; }
Ascend910B3Status status_invalid_parameters(int32_t native_code = 0) noexcept;
Ascend910B3Status status_invalid_state(int32_t native_code = 0) noexcept;
inline bool status_ok(Ascend910B3Status status) noexcept {
    return status.kind == 0 && status.native_code == 0;
}
// A failing ACL call: kind InvalidState, native_code the raw aclError.
Ascend910B3Status status_from_acl(aclError error) noexcept;

// Wraps a C entry body: converts an escaped std::bad_alloc / foreign exception
// into a status so nothing crosses the ABI. Entries still clear their resource
// outputs before any work (a failure must null every output slot).
template <typename Fn>
Ascend910B3Status guarded(Fn&& fn) noexcept {
    try {
        return static_cast<Ascend910B3Status>(fn());
    } catch (const std::bad_alloc&) {
        return status_invalid_state(ACL_ERROR_BAD_ALLOC);
    } catch (...) {
        return status_invalid_state(ACL_ERROR_INTERNAL_ERROR);
    }
}

// ---------------------------------------------------------------------------
// RAII storage.
// ---------------------------------------------------------------------------

// Owns one aligned device allocation. Frees immediately in release()/destructor
// and therefore must never be destroyed while outstanding stream work may still
// read it: register it (Context::register_buffer) or synchronize first.
class DeviceStorage {
public:
    DeviceStorage() noexcept = default;
    ~DeviceStorage();
    DeviceStorage(DeviceStorage&& other) noexcept;
    DeviceStorage& operator=(DeviceStorage&& other) noexcept;
    DeviceStorage(const DeviceStorage&) = delete;
    DeviceStorage& operator=(const DeviceStorage&) = delete;

    void* data() const noexcept { return data_; }
    size_t size() const noexcept { return bytes_; }
    bool valid() const noexcept { return data_ != nullptr; }
    explicit operator bool() const noexcept { return valid(); }

    // Frees the raw allocation; returns the ACL result and always clears.
    aclError release() noexcept;
    void swap(DeviceStorage& other) noexcept;

private:
    void* raw_ = nullptr;   // exactly what aclrtMalloc returned
    void* data_ = nullptr;  // alignment-adjusted base within raw_
    size_t bytes_ = 0;
    friend class Context;
};

// Owns pinned host memory used to stage async transfers. Same lifetime rule:
// keep it alive (or hand it to the context) until the transfer is complete.
class PinnedStorage {
public:
    PinnedStorage() noexcept = default;
    ~PinnedStorage();
    PinnedStorage(PinnedStorage&& other) noexcept;
    PinnedStorage& operator=(PinnedStorage&& other) noexcept;
    PinnedStorage(const PinnedStorage&) = delete;
    PinnedStorage& operator=(const PinnedStorage&) = delete;

    void* data() const noexcept { return data_; }
    size_t size() const noexcept { return bytes_; }
    bool valid() const noexcept { return data_ != nullptr; }
    explicit operator bool() const noexcept { return valid(); }

    aclError release() noexcept;
    void swap(PinnedStorage& other) noexcept;

private:
    friend class Context;
    void* raw_ = nullptr;
    void* data_ = nullptr;
    size_t bytes_ = 0;
};

// ---------------------------------------------------------------------------
// Retained device buffer.
// ---------------------------------------------------------------------------

enum class BufferKind : uint32_t {
    MleFr = 1,   // immutable canonical Fr table of (1 << num_vars) elements
    General = 2, // any other registered device allocation
};

class DeviceBlock; // defined by bridge_runtime.cpp

// Copyable shared ownership of one registered device allocation. A BufferRef
// covers the public MLE-handle retirement (buffer_retire) only: it keeps that
// allocation alive while the owning context lives. It does not cover, extend or
// survive physical context destruction (context_destroy). Native objects may
// store a BufferRef (or ContextRef) only when their Rust counterpart also owns
// Rc<Ascend910B3Context>, which is what guarantees destroy runs after every
// retained reference is gone. The data is immutable; no writable transfer.
class BufferRef {
public:
    BufferRef() noexcept = default;
    BufferRef(const BufferRef&) noexcept = default;
    BufferRef(BufferRef&&) noexcept = default;
    BufferRef& operator=(const BufferRef&) noexcept = default;
    BufferRef& operator=(BufferRef&&) noexcept = default;
    ~BufferRef() = default;

    bool valid() const noexcept { return static_cast<bool>(block_); }
    explicit operator bool() const noexcept { return valid(); }
    // Device base address; equals the public Ascend910B3DeviceAddress handle.
    DeviceAddress data() const noexcept;
    DeviceAddress handle() const noexcept { return data(); }
    size_t bytes() const noexcept;
    size_t num_vars() const noexcept;
    BufferKind kind() const noexcept;
    void reset() noexcept { block_.reset(); }

private:
    explicit BufferRef(std::shared_ptr<DeviceBlock> block) noexcept
        : block_(std::move(block)) {}
    std::shared_ptr<DeviceBlock> block_;
    friend class Context;
};

// ---------------------------------------------------------------------------
// Execution context.
// ---------------------------------------------------------------------------

class Context {
public:
    // Binds this context as the current ACL context for the duration of a host
    // entry point and restores the previous current context on exit. A previous
    // null context is not "restored" through an invalid ACL call. Non-nested use
    // per entry point; internal helpers only ensure the context is current.
    class Scope {
    public:
        explicit Scope(Context& context) noexcept;
        ~Scope();
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        Scope(Scope&&) = delete;
        Scope& operator=(Scope&&) = delete;

        bool ok() const noexcept { return status_.kind == 0 && status_.native_code == 0; }
        Ascend910B3Status status() const noexcept { return status_; }

    private:
        aclrtContext previous_ = nullptr;
        bool restore_ = false;
        Ascend910B3Status status_ = Ascend910B3Status{0, 0};
    };

    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(Context&&) = delete;

    int32_t device_id() const noexcept;
    aclrtContext acl_context() const noexcept;
    aclrtStream stream() const noexcept;

    // True once destroy has started; every later operation returns InvalidState.
    bool closed() const noexcept;
    // True after an unrecoverable ACL failure; the first error is preserved.
    bool poisoned() const noexcept;
    aclError first_error() const noexcept;
    // Records an unrecoverable failure; only the first error is kept so cleanup
    // failures never erase the original diagnostic.
    void poison(aclError error) noexcept;

    // Makes this context current without disturbing a matching current context.
    Ascend910B3Status ensure_current() noexcept;
    Scope scope() noexcept;

    // Allocate an unregistered device/host allocation. A zero byte request
    // clears `out` and succeeds. Alignment must be a power of two <= 4096.
    Ascend910B3Status allocate(size_t bytes, size_t alignment, DeviceStorage& out) noexcept;
    Ascend910B3Status allocate_pinned(size_t bytes, size_t alignment, PinnedStorage& out) noexcept;

    // Hands pinned staging to the context. It stays alive until a real completion
    // observation (query COMPLETE or synchronize), so an asynchronous copy that
    // still reads it is never released early.
    Ascend910B3Status retain_pinned(PinnedStorage&& storage) noexcept;

    // Adopt an owned device allocation into the registry. The returned handle is
    // block->data(). On failure the storage is released by the callee.
    Ascend910B3Status register_buffer(DeviceStorage&& storage, size_t num_vars,
                                      BufferKind kind, BufferRef& out) noexcept;
    Ascend910B3Status create_buffer(size_t bytes, size_t num_vars, BufferKind kind,
                                    BufferRef& out) noexcept;

    // Registry lookups. `lookup(address, num_vars, ...)` additionally requires
    // the MLE dimension to match. A retired or foreign address is not found.
    bool lookup(DeviceAddress address, BufferRef& out) noexcept;
    bool lookup(DeviceAddress address, size_t num_vars, BufferRef& out) noexcept;

    // Canonical Fr table upload: allocates, stages through owned pinned memory,
    // enqueues H2D and registers the result. `out` is cleared first.
    Ascend910B3Status upload_mle(size_t num_vars, const uint8_t* evaluations,
                                 size_t evaluation_count, DeviceAddress& out) noexcept;

    // Copies on this context's stream. Caller-owned pointers must outlive the
    // copy; host pointers must stay pinned or be synchronized before release.
    Ascend910B3Status copy_async(void* dst, const void* src, size_t bytes,
                                 aclrtMemcpyKind kind) noexcept;
    Ascend910B3Status copy_to_device_async(void* dst, const void* host_src,
                                           size_t bytes) noexcept;
    Ascend910B3Status copy_to_host_async(void* host_dst, const void* src,
                                         size_t bytes) noexcept;

    // Waits for the stream, then frees every retired allocation and releases all
    // staged pinned memory. On failure the context is poisoned and retirement is
    // kept queued for a later drain instead of being freed unsafely.
    Ascend910B3Status synchronize() noexcept;
    // Query-only observation: frees retired allocations and staging only when the
    // stream really reports complete. Safe to call from any entry point.
    Ascend910B3Status poll_retired() noexcept;
    // Probe completion without releasing anything.
    Ascend910B3Status completion(bool& complete) noexcept;

    // Removes the public handle; physical free is deferred until completion.
    // Unknown or foreign addresses are InvalidParameters. A retained BufferRef
    // keeps the storage usable after retirement (while the context lives).
    Ascend910B3Status retire(DeviceAddress address) noexcept;

    // Contract for ContextRef/BufferRef owners: destroy is the last native use of
    // the handle. Rust objects that can retain device memory (backend, MLE
    // pointer, KZG params, SumCheck state) each own Rc<Ascend910B3Context>, so no
    // retained reference outlives destroy in a supported call sequence.
    // destroy marks the handle closed and only releases retired allocations,
    // staging, the stream and the ACL context after a successful stream
    // synchronize (or when no stream exists). If completion cannot be proven it
    // keeps the handle in a process-lifetime registry root and releases nothing:
    // the whole Context/queue/staging/stream/runtime stays reachable, static
    // destruction cannot free it, and the preserved first error is returned. A
    // later destroy may retry the completion proof.

private:
    // Requires stream completion already observed; frees deferred allocations
    // and releases pinned staging, preserving the first ACL failure.
    Ascend910B3Status reclaim_retired() noexcept;

    struct Impl;
    explicit Context(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend Ascend910B3Status create_context(int32_t, Ascend910B3NativeContext**) noexcept;
    friend Ascend910B3Status destroy_context(Ascend910B3NativeContext*) noexcept;
    friend Ascend910B3Status retire_buffer(Ascend910B3NativeContext*, DeviceAddress) noexcept;
    friend Ascend910B3Status upload_mle(Ascend910B3NativeContext*, size_t, const uint8_t*,
                                        size_t, DeviceAddress*) noexcept;
};

// Shared handle to the native context, so a call cannot race destruction while
// it dereferences the context. Empty on null/unknown/destroyed contexts.
class ContextRef {
public:
    ContextRef() noexcept = default;
    Context& operator*() const noexcept { return *context_; }
    Context* operator->() const noexcept { return context_.get(); }
    explicit operator bool() const noexcept { return static_cast<bool>(context_); }

private:
    explicit ContextRef(std::shared_ptr<Context> context) noexcept
        : context_(std::move(context)) {}
    std::shared_ptr<Context> context_;
    friend Ascend910B3Status acquire_context(Ascend910B3NativeContext*, ContextRef&) noexcept;
};

// Validates the opaque handle against the live-context registry.
Ascend910B3Status acquire_context(Ascend910B3NativeContext* raw, ContextRef& out) noexcept;

// ABI implementation entry points used by bridge.asc. These are noexcept; every
// C++ exception is absorbed and mapped to a status.
Ascend910B3Status create_context(int32_t device_id, Ascend910B3NativeContext** out) noexcept;
Ascend910B3Status destroy_context(Ascend910B3NativeContext* raw) noexcept;
Ascend910B3Status retire_buffer(Ascend910B3NativeContext* raw, DeviceAddress address) noexcept;
Ascend910B3Status upload_mle(Ascend910B3NativeContext* raw, size_t num_vars,
                             const uint8_t* evaluations, size_t evaluation_count,
                             DeviceAddress* out) noexcept;

} // namespace hyperplonk::ascend910b3::native
