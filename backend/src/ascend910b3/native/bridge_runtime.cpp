// Sole native runtime implementation: ACL lifetime, execution context,
// completion-aware retirement and canonical MLE upload.
//
// No numerical work happens here. Every device operation enqueued by this file
// is a control-plane transfer; field/G1 math stays in the kernels owned by the
// other bridge translation units.
#include "bridge_runtime.h"

#include "field_params.h"

#include <atomic>
#include <cstring>
#include <unordered_map>
#include <utility>

namespace hyperplonk::ascend910b3::native {
namespace {

using FrParams = hyperplonk::ascend910b3::Bls12381FrParams;
static_assert(FrParams::word_count == 8, "canonical Fr packet is 32 bytes");
constexpr size_t kFrWords = FrParams::word_count;

bool is_power_of_two(size_t value) noexcept { return value != 0 && (value & (value - 1)) == 0; }

uint32_t little_endian_word(const uint8_t* bytes, size_t word) noexcept {
    const size_t offset = word * 4;
    return static_cast<uint32_t>(bytes[offset]) |
           (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
           (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
           (static_cast<uint32_t>(bytes[offset + 3]) << 24);
}

// ---------------------------------------------------------------------------
// Process-wide ACL lifetime. Reference counted so independently created
// contexts never finalize an initialization they did not own. A repeated
// initialization means some other component owns ACL; we then never finalize.
// ---------------------------------------------------------------------------

} // namespace

void release_runtime() noexcept; // defined below

class RuntimeHandle {
public:
    RuntimeHandle() noexcept = default;
    RuntimeHandle(const RuntimeHandle&) = delete;
    RuntimeHandle& operator=(const RuntimeHandle&) = delete;
    ~RuntimeHandle() { release_runtime(); }
};

namespace {

std::mutex& runtime_mutex() noexcept {
    static std::mutex mutex;
    return mutex;
}
size_t& runtime_references() noexcept {
    static size_t references = 0;
    return references;
}
bool& runtime_owned() noexcept {
    static bool owned = false;
    return owned;
}

// Process-wide conservative admission budget for outstanding stream-marker
// events. RuntimeHandle is created once per Context, so the counter must NOT live
// on it: this function-local static sits beside the runtime reference state and
// is shared by every context (and therefore every device) in this process. The
// CANN 9.1 contract allows 65536 events per device; 4096 keeps a wide margin and
// prevents the marker path from self-starving while uploads are reclaimed only
// by a later synchronize.
constexpr int64_t kMarkerBudget = 4096;

std::atomic<int64_t>& outstanding_markers() noexcept {
    static std::atomic<int64_t> markers{0};
    return markers;
}

bool try_acquire_marker_slot() noexcept {
    std::atomic<int64_t>& markers = outstanding_markers();
    int64_t current = markers.load(std::memory_order_relaxed);
    while (current < kMarkerBudget) {
        if (markers.compare_exchange_weak(current, current + 1,
                                          std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

void release_marker_slot() noexcept {
    // Called only after a successful event destruction; floor at zero so a
    // double release can never inflate later admission.
    std::atomic<int64_t>& markers = outstanding_markers();
    int64_t current = markers.load(std::memory_order_relaxed);
    while (current > 0) {
        if (markers.compare_exchange_weak(current, current - 1,
                                          std::memory_order_relaxed)) {
            return;
        }
    }
}

bool acquire_runtime(std::shared_ptr<RuntimeHandle>& out, aclError& error) noexcept {
    out.reset(); // never destroy a previous handle while holding the mutex
    std::lock_guard<std::mutex> lock(runtime_mutex());
    if (runtime_references() == 0) {
        const aclError status = aclInit(nullptr);
        if (status == ACL_SUCCESS) {
            runtime_owned() = true;
        } else if (status == ACL_ERROR_REPEAT_INITIALIZE) {
            runtime_owned() = false; // externally initialized; never finalize it
        } else {
            error = status;
            return false;
        }
    }
    try {
        out = std::make_shared<RuntimeHandle>();
    } catch (...) {
        error = ACL_ERROR_BAD_ALLOC;
        return false;
    }
    ++runtime_references();
    error = ACL_SUCCESS;
    return true;
}

} // namespace

void release_runtime() noexcept {
    std::lock_guard<std::mutex> lock(runtime_mutex());
    if (runtime_references() == 0) return;
    --runtime_references();
    if (runtime_references() == 0 && runtime_owned()) {
        (void)aclFinalize();
        runtime_owned() = false;
    }
}

// ---------------------------------------------------------------------------
// Completion-aware retirement.
//
// A retired allocation is pushed here by the last owner's destructor. Physical
// freeing is gated by a stream marker recorded only AFTER that last owner queued
// the raw (lazily, by the next poll): a marker recorded earlier could complete
// before a later same-stream use enqueued through a retained BufferRef. The
// marker is ordered after every operation already enqueued on the context
// stream, so its completion proves the storage can no longer be read. A bare
// aclrtStreamQuery is deliberately NOT the release proof: this stack can report
// ACL_STREAM_STATUS_COMPLETE before a just-enqueued task reaches the device
// queue, which previously released metadata (device blocks and pinned upload
// bytes) while a queued kernel could still read them. Pinned staging carries its
// own marker recorded after the transfer that uses it, so staging is released
// only when its own transfer completed and is covered even when no device block
// has been retired yet.
// ---------------------------------------------------------------------------

struct PendingBlock {
    void* raw = nullptr;
    // Marker recorded only after the last owner queued this raw (see
    // take_ready); null means "not marked yet or recording failed", so only a
    // real stream synchronize may free.
    aclrtEvent event = nullptr;
    // One marker attempt per resource: a failed create/admission/record is never
    // retried by later polls. The block then waits for an existing synchronize.
    bool marker_attempted = false;
};

struct PendingStaging {
    std::shared_ptr<PinnedStorage> pinned;
    // Recorded after the transfer that reads these bytes.
    aclrtEvent event = nullptr;
    // The upload path made exactly one attempt; polls never retry it.
    bool marker_attempted = false;
};

struct RetireQueue {
    std::mutex mutex;
    std::vector<PendingBlock> pending;
    std::vector<PendingStaging> staging;
    // Created but not provably recorded events. Recording failure is ambiguous
    // (the record may have been enqueued), so these are never destroyed before a
    // real stream synchronize.
    std::vector<aclrtEvent> unproven;
    std::shared_ptr<RuntimeHandle> runtime;
    aclrtStream stream = nullptr;
    bool attached = true;

    // Called from a destructor after the last owner is gone: never throws. A
    // marker cannot be recorded safely from here, so the entry is queued and the
    // next poll records its one-shot marker (which is then ordered after this
    // enqueue and after every earlier same-stream use). A push failure or a
    // detached queue may only free after completion is proven: a null stream
    // means destroy already synchronized and disowned it, otherwise synchronize
    // must succeed. Without proof the allocation is intentionally abandoned.
    void enqueue(void* raw) noexcept {
        aclrtStream active = nullptr;
        bool free_now = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            active = stream;
            if (!attached) {
                free_now = true;
            } else {
                try {
                    pending.reserve(pending.size() + 1);
                } catch (...) {
                    free_now = true;
                }
                if (!free_now) pending.push_back(PendingBlock{raw, nullptr, false});
            }
        }
        if (!free_now) return;
        if (active == nullptr || aclrtSynchronizeStream(active) == ACL_SUCCESS) {
            (void)aclrtFree(raw);
        }
        // Without a proven synchronize the allocation is intentionally abandoned.
    }

    // Records a one-shot marker on the queue stream. Requires the queue mutex NOT
    // to be held.
    aclrtEvent record_marker() noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        return record_marker_locked();
    }

    // Retains a created event whose recording is ambiguous (or that must not be
    // destroyed yet) until a real stream synchronize.
    void retain_unproven(aclrtEvent event) noexcept {
        if (event == nullptr) return;
        std::lock_guard<std::mutex> lock(mutex);
        try {
            unproven.reserve(unproven.size() + 1);
            unproven.push_back(event);
        } catch (...) {
            // Intentional abandon: never destroy a possibly recorded event.
        }
    }

    // Retains pinned bytes with the marker recorded after the transfer reading
    // them. Capacity is obtained before the owning argument is moved, so a failed
    // reservation cannot free bytes that a queued transfer may still read.
    void retain_staging(std::shared_ptr<PinnedStorage> pinned,
                        aclrtEvent marker) noexcept {
        aclrtStream active = nullptr;
        bool release_now = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            active = stream;
            if (!attached) {
                release_now = true;
            } else {
                try {
                    staging.reserve(staging.size() + 1);
                } catch (...) {
                    release_now = true;
                }
                if (!release_now) {
                    staging.push_back(PendingStaging{});
                    staging.back().pinned = std::move(pinned);
                    staging.back().event = marker;
                    // The upload path already made its single marker attempt.
                    staging.back().marker_attempted = true;
                }
            }
        }
        if (!release_now) return;
        const bool proven =
            active == nullptr || aclrtSynchronizeStream(active) == ACL_SUCCESS;
        if (!proven && pinned) {
            // No completion proof: intentionally abandon the host bytes instead of
            // freeing memory a queued transfer may still read. Allocation-free.
            pinned->abandon();
        }
        if (marker != nullptr) {
            if (proven) {
                destroy_marker(marker);
            } else {
                retain_unproven(marker);
            }
        }
        pinned.reset();
    }

    // Requires the queue mutex to be held.
    aclrtEvent record_marker_locked() noexcept {
        if (stream == nullptr) return nullptr;
        // Admission is conservative and process-wide: when the shared budget is
        // exhausted the caller leaves the resource markerless and it waits for an
        // existing real synchronize. No marker is recorded while the process-wide
        // self-owned count is at the budget, so this path cannot self-starve the
        // device event pool.
        if (!try_acquire_marker_slot()) return nullptr;
        aclrtEvent event = nullptr;
        if (aclrtCreateEvent(&event) != ACL_SUCCESS || event == nullptr) {
            release_marker_slot();
            return nullptr;
        }
        if (aclrtRecordEvent(event, stream) != ACL_SUCCESS) {
            // Ambiguous: the record may already be enqueued. Retain it until a real
            // synchronize instead of pretending the failure proved it unrecorded.
            // Its admission slot stays accounted (conservative if abandoned).
            try {
                unproven.reserve(unproven.size() + 1);
                unproven.push_back(event);
            } catch (...) {
                // Intentional abandon: never destroy a possibly recorded event.
            }
            return nullptr;
        }
        return event;
    }

    // Destroys a marker created by record_marker_locked. The admission slot is
    // returned ONLY when aclrtDestroyEvent succeeds: a failed destruction may
    // leave the event alive, so its slot stays accounted conservatively. This
    // never reports fake success and never adds a synchronization.
    void destroy_marker(aclrtEvent event) noexcept {
        if (event == nullptr) return;
        if (aclrtDestroyEvent(event) == ACL_SUCCESS) {
            release_marker_slot();
        }
    }

    // Moves every deferred resource out of the queue after a proven stream
    // synchronize. All three moves are allocation-free. The caller frees blocks,
    // destroys every marker (block, staging and unproven) and releases pinned
    // bytes by destroying the staging entries.
    void take(std::vector<PendingBlock>& blocks,
              std::vector<PendingStaging>& staging_out,
              std::vector<aclrtEvent>& unproven_out) noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        blocks = std::move(pending);
        pending.clear();
        staging_out = std::move(staging);
        staging.clear();
        unproven_out = std::move(unproven);
        unproven.clear();
    }

    // Moves every block and pinned buffer whose own marker completed (or all of
    // them when `proven` comes from a real aclrtSynchronizeStream) into the
    // caller's ready lists. Completed markers are identified first; only then are
    // missing markers attempted, at most once per resource (marker_attempted). A
    // marker recorded here is only possible after the last owner queued the raw,
    // so it cannot pre-date a future same-stream use through a retained
    // BufferRef. A newly recorded marker is never treated as complete in the same
    // call, and a missing marker, admission refusal or failed/ambiguous query
    // keeps the resource queued until an existing real synchronize: no retry and
    // no query-authorized release.
    void take_ready(bool proven,
                    std::vector<PendingBlock>& ready_blocks,
                    std::vector<std::shared_ptr<PinnedStorage>>& ready_staging,
                    std::vector<aclrtEvent>& ready_events) noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        std::vector<PendingBlock> keep_blocks;
        std::vector<PendingStaging> keep_staging;
        try {
            ready_blocks.reserve(pending.size());
            keep_blocks.reserve(pending.size());
            ready_staging.reserve(staging.size());
            keep_staging.reserve(staging.size());
            ready_events.reserve(staging.size() + unproven.size());
        } catch (...) {
            return; // conservative: release nothing on allocation failure
        }
        if (proven) {
            for (aclrtEvent event : unproven) ready_events.push_back(event);
            unproven.clear();
            for (PendingBlock& block : pending) ready_blocks.push_back(block);
            pending.clear();
            for (PendingStaging& entry : staging) {
                ready_staging.push_back(std::move(entry.pinned));
                if (entry.event != nullptr) ready_events.push_back(entry.event);
            }
            staging.clear();
            return;
        }
        // Pass 1: reap already-completed markers before any new record attempt.
        for (PendingBlock& block : pending) {
            if (block.event != nullptr && marker_completed(false, block.event)) {
                ready_blocks.push_back(block);
            } else {
                keep_blocks.push_back(block);
            }
        }
        pending.swap(keep_blocks);
        for (PendingStaging& entry : staging) {
            if (entry.event != nullptr && marker_completed(false, entry.event)) {
                ready_staging.push_back(std::move(entry.pinned));
                ready_events.push_back(entry.event);
            } else {
                keep_staging.push_back(std::move(entry));
            }
        }
        staging.swap(keep_staging);
        // Pass 2: exactly one marker attempt per still-unmarked resource.
        for (PendingBlock& block : pending) {
            if (block.event == nullptr && !block.marker_attempted) {
                block.marker_attempted = true;
                block.event = record_marker_locked();
            }
        }
        for (PendingStaging& entry : staging) {
            if (entry.event == nullptr && !entry.marker_attempted) {
                entry.marker_attempted = true;
                entry.event = record_marker_locked();
            }
        }
    }

    static bool marker_completed(bool proven, aclrtEvent event) noexcept {
        if (proven) return true;
        if (event == nullptr) return false;
        aclrtEventRecordedStatus status = ACL_EVENT_RECORDED_STATUS_NOT_READY;
        return aclrtQueryEventStatus(event, &status) == ACL_SUCCESS &&
               status == ACL_EVENT_RECORDED_STATUS_COMPLETE;
    }

    void detach() noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        attached = false;
    }

    // After the context destroyed its stream, late destructors must free directly
    // instead of synchronizing a dead handle.
    void drop_stream() noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        stream = nullptr;
    }
};

namespace {

// ---------------------------------------------------------------------------
// Live context registry. Raw handles are validated by membership, never by
// dereferencing a caller-supplied pointer.
// ---------------------------------------------------------------------------

std::mutex& context_mutex() noexcept {
    static std::mutex mutex;
    return mutex;
}
std::unordered_map<Ascend910B3NativeContext*, std::shared_ptr<Context>>& contexts() noexcept {
    // Intentionally never destroyed: static destruction must not run Impl cleanup
    // for a context whose destroy could not prove stream completion (that would
    // free staged pinned memory, registry blocks and the runtime without
    // evidence). A failed destroy keeps its handle in this map as the reachable,
    // safe root; only a completion-proven destroy erases it.
    static auto* registry =
        new std::unordered_map<Ascend910B3NativeContext*, std::shared_ptr<Context>>();
    return *registry;
}

} // namespace

// ---------------------------------------------------------------------------
// Registered device allocation. Defined outside the anonymous namespace so it
// is the same type the published header forward-declares for BufferRef.
// ---------------------------------------------------------------------------

class DeviceBlock {
public:
    DeviceBlock(void* raw, void* data, size_t bytes, size_t num_vars,
                BufferKind kind, std::shared_ptr<RetireQueue> queue) noexcept
        : raw_(raw), data_(data), bytes_(bytes), num_vars_(num_vars), kind_(kind),
          queue_(std::move(queue)) {}

    DeviceBlock(const DeviceBlock&) = delete;
    DeviceBlock& operator=(const DeviceBlock&) = delete;

    // The last reference reaching this destructor retires the storage; the queue
    // frees it only after real stream completion.
    ~DeviceBlock() { queue_->enqueue(raw_); }

    DeviceAddress data() const noexcept { return data_; }
    size_t bytes() const noexcept { return bytes_; }
    size_t num_vars() const noexcept { return num_vars_; }
    BufferKind kind() const noexcept { return kind_; }

private:
    void* raw_;
    void* data_;
    size_t bytes_;
    size_t num_vars_;
    BufferKind kind_;
    std::shared_ptr<RetireQueue> queue_;
};

// ---------------------------------------------------------------------------
// Context implementation.
// ---------------------------------------------------------------------------

struct Context::Impl {
    int32_t device_id = -1;
    aclrtContext acl_ctx = nullptr;
    aclrtStream stream = nullptr;
    std::shared_ptr<RuntimeHandle> runtime;
    std::shared_ptr<RetireQueue> queue;
    std::unordered_map<void*, std::shared_ptr<DeviceBlock>> registry;
    mutable std::mutex mutex;
    bool closed = false;
    bool poisoned = false;
    aclError first_error = ACL_SUCCESS;
};

// ---------------------------------------------------------------------------
// Checked sizes and canonical packets.
// ---------------------------------------------------------------------------

bool checked_add(size_t a, size_t b, size_t& out) noexcept {
    if (b > std::numeric_limits<size_t>::max() - a) return false;
    out = a + b;
    return true;
}

bool checked_mul(size_t a, size_t b, size_t& out) noexcept {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
    out = a * b;
    return true;
}

bool dimension_bytes(size_t num_vars, size_t element_bytes, size_t& out) noexcept {
    out = 0;
    if (element_bytes == 0 || num_vars >= 64) return false;
    const size_t count = static_cast<size_t>(uint64_t{1} << num_vars);
    size_t bytes = 0;
    if (!checked_mul(count, element_bytes, bytes)) return false;
    if (bytes > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max())) return false;
    out = bytes;
    return true;
}

bool canonical_fr(const uint8_t* bytes) noexcept {
    if (bytes == nullptr) return false;
    // Compare from the most significant word; any word below r decides, all
    // equal means the value is exactly r and therefore not canonical.
    for (size_t i = kFrWords; i-- > 0;) {
        const uint32_t word = little_endian_word(bytes, i);
        if (word < FrParams::modulus[i]) return true;
        if (word > FrParams::modulus[i]) return false;
    }
    return false;
}

bool canonical_fr_array(const uint8_t* bytes, size_t count) noexcept {
    if (count == 0) return true;
    if (bytes == nullptr) return false;
    for (size_t i = 0; i < count; ++i)
        if (!canonical_fr(bytes + i * kFrBytes)) return false;
    return true;
}

Ascend910B3Status status_invalid_parameters(int32_t native_code) noexcept {
    return Ascend910B3Status{1, native_code};
}

Ascend910B3Status status_invalid_state(int32_t native_code) noexcept {
    return Ascend910B3Status{2, native_code};
}

Ascend910B3Status status_from_acl(aclError error) noexcept {
    return status_invalid_state(error == ACL_SUCCESS ? ACL_ERROR_INTERNAL_ERROR : error);
}

// ---------------------------------------------------------------------------
// RAII storage.
// ---------------------------------------------------------------------------

DeviceStorage::~DeviceStorage() { (void)release(); }

DeviceStorage::DeviceStorage(DeviceStorage&& other) noexcept { swap(other); }

DeviceStorage& DeviceStorage::operator=(DeviceStorage&& other) noexcept {
    if (this != &other) {
        (void)release();
        swap(other);
    }
    return *this;
}

aclError DeviceStorage::release() noexcept {
    aclError status = ACL_SUCCESS;
    if (raw_ != nullptr) status = aclrtFree(raw_);
    raw_ = nullptr;
    data_ = nullptr;
    bytes_ = 0;
    return status;
}

void DeviceStorage::swap(DeviceStorage& other) noexcept {
    std::swap(raw_, other.raw_);
    std::swap(data_, other.data_);
    std::swap(bytes_, other.bytes_);
}

PinnedStorage::~PinnedStorage() { (void)release(); }

PinnedStorage::PinnedStorage(PinnedStorage&& other) noexcept { swap(other); }

PinnedStorage& PinnedStorage::operator=(PinnedStorage&& other) noexcept {
    if (this != &other) {
        (void)release();
        swap(other);
    }
    return *this;
}

aclError PinnedStorage::release() noexcept {
    aclError status = ACL_SUCCESS;
    if (raw_ != nullptr) status = aclrtFreeHost(raw_);
    raw_ = nullptr;
    data_ = nullptr;
    bytes_ = 0;
    return status;
}

void PinnedStorage::swap(PinnedStorage& other) noexcept {
    std::swap(raw_, other.raw_);
    std::swap(data_, other.data_);
    std::swap(bytes_, other.bytes_);
}

// ---------------------------------------------------------------------------
// BufferRef.
// ---------------------------------------------------------------------------

DeviceAddress BufferRef::data() const noexcept {
    return block_ ? block_->data() : nullptr;
}

size_t BufferRef::bytes() const noexcept { return block_ ? block_->bytes() : 0; }

size_t BufferRef::num_vars() const noexcept {
    return block_ ? block_->num_vars() : 0;
}

BufferKind BufferRef::kind() const noexcept {
    return block_ ? block_->kind() : BufferKind::General;
}

// ---------------------------------------------------------------------------
// Context.
// ---------------------------------------------------------------------------

Context::Context(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

Context::~Context() = default;

int32_t Context::device_id() const noexcept { return impl_->device_id; }

aclrtContext Context::acl_context() const noexcept { return impl_->acl_ctx; }

aclrtStream Context::stream() const noexcept { return impl_->stream; }

bool Context::closed() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->closed;
}

bool Context::poisoned() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->poisoned;
}

aclError Context::first_error() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->first_error;
}

void Context::poison(aclError error) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->poisoned) {
        impl_->poisoned = true;
        impl_->first_error = error == ACL_SUCCESS ? ACL_ERROR_INTERNAL_ERROR : error;
    }
}

Ascend910B3Status Context::ensure_current() noexcept {
    if (closed()) return status_invalid_state();
    const aclrtContext target = impl_->acl_ctx;
    if (target == nullptr) return status_invalid_state();
    aclrtContext current = nullptr;
    aclError status = aclrtGetCurrentContext(&current);
    if (status != ACL_SUCCESS) return status_from_acl(status);
    if (current == target) return status_success();
    status = aclrtSetCurrentContext(target);
    if (status != ACL_SUCCESS) return status_from_acl(status);
    return status_success();
}

Context::Scope::Scope(Context& context) noexcept {
    aclrtContext current = nullptr;
    aclError status = aclrtGetCurrentContext(&current);
    if (status != ACL_SUCCESS) {
        status_ = status_from_acl(status);
        return;
    }
    const aclrtContext target = context.acl_context();
    if (target == nullptr) {
        status_ = status_invalid_state();
        return;
    }
    if (current == target) return;
    status = aclrtSetCurrentContext(target);
    if (status != ACL_SUCCESS) {
        status_ = status_from_acl(status);
        return;
    }
    previous_ = current;
    restore_ = current != nullptr;
}

Context::Scope::~Scope() {
    if (restore_) (void)aclrtSetCurrentContext(previous_);
}

Context::Scope Context::scope() noexcept { return Scope(*this); }

Ascend910B3Status Context::allocate(size_t bytes, size_t alignment,
                                   DeviceStorage& out) noexcept {
    const aclError stale = out.release();
    if (stale != ACL_SUCCESS) return status_from_acl(stale);
    if (closed()) return status_invalid_state();
    if (poisoned()) return status_invalid_state(first_error());
    if (!is_power_of_two(alignment) || alignment > kMaxAlignment)
        return status_invalid_parameters();
    if (bytes == 0) return status_success();
    if (bytes > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
        return status_invalid_parameters();
    size_t total = 0;
    if (!checked_add(bytes, alignment, total)) return status_invalid_parameters();
    const Ascend910B3Status ensured = ensure_current();
    if (!status_ok(ensured)) return ensured;
    void* raw = nullptr;
    const aclError status = aclrtMalloc(&raw, total, ACL_MEM_MALLOC_HUGE_FIRST);
    if (status != ACL_SUCCESS) return status_from_acl(status);
    const uintptr_t base = reinterpret_cast<uintptr_t>(raw);
    const uintptr_t aligned = (base + alignment - 1) & ~static_cast<uintptr_t>(alignment - 1);
    if (aligned < base || aligned + bytes > base + total) {
        (void)aclrtFree(raw);
        return status_invalid_state(ACL_ERROR_INTERNAL_ERROR);
    }
    out.raw_ = raw;
    out.data_ = reinterpret_cast<void*>(aligned);
    out.bytes_ = bytes;
    return status_success();
}

Ascend910B3Status Context::allocate_pinned(size_t bytes, size_t alignment,
                                          PinnedStorage& out) noexcept {
    const aclError stale = out.release();
    if (stale != ACL_SUCCESS) return status_from_acl(stale);
    if (closed()) return status_invalid_state();
    if (poisoned()) return status_invalid_state(first_error());
    if (!is_power_of_two(alignment) || alignment > kMaxAlignment)
        return status_invalid_parameters();
    if (bytes == 0) return status_success();
    if (bytes > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
        return status_invalid_parameters();
    size_t total = 0;
    if (!checked_add(bytes, alignment, total)) return status_invalid_parameters();
    void* raw = nullptr;
    const aclError status = aclrtMallocHost(&raw, total);
    if (status != ACL_SUCCESS) return status_from_acl(status);
    const uintptr_t base = reinterpret_cast<uintptr_t>(raw);
    const uintptr_t aligned = (base + alignment - 1) & ~static_cast<uintptr_t>(alignment - 1);
    if (aligned < base || aligned + bytes > base + total) {
        (void)aclrtFreeHost(raw);
        return status_invalid_state(ACL_ERROR_INTERNAL_ERROR);
    }
    out.raw_ = raw;
    out.data_ = reinterpret_cast<void*>(aligned);
    out.bytes_ = bytes;
    return status_success();
}

// Records a stream marker for a just-finished upload. The marker is ordered
// after every operation already enqueued on this context stream, which includes
// the transfer that reads the staged bytes. Null means the marker could not be
// created/recorded; the resource must then be released only after a real stream
// synchronize. A created-but-ambiguous event is retained by the queue until a
// real synchronize, never destroyed on the assumption that recording failed.
aclrtEvent Context::record_marker() noexcept {
    if (impl_->stream == nullptr) return nullptr;
    const Ascend910B3Status ensured = ensure_current();
    if (!status_ok(ensured)) return nullptr;
    return impl_->queue->record_marker();
}

Ascend910B3Status Context::retain_pinned(PinnedStorage&& storage) noexcept {
    // Resource-lifetime contract: the caller enqueued the transfer that reads
    // these bytes before this call. On any failure path this function must not
    // return owning bytes that a caller could later free while that transfer may
    // still be pending: when completion cannot be proven the bytes are
    // intentionally abandoned (allocation-free) and the failure is still
    // reported. A proven stream synchronize leaves ownership with the caller's
    // object, whose destruction is then safe.
    if (closed()) {
        // No completion can be observed for a closing context; consume the owner
        // so a caller fallback cannot free bytes a queued transfer may read.
        if (storage.valid()) storage.abandon();
        return status_invalid_state();
    }
    if (!storage.valid()) return status_success();
    std::shared_ptr<PinnedStorage> pinned;
    try {
        pinned = std::make_shared<PinnedStorage>(std::move(storage));
    } catch (...) {
        // make_shared allocates before it moves the owner, so on failure `storage`
        // still owns the bytes. Prove the stream drained before letting them be
        // freed; otherwise abandon them and preserve the error.
        const bool proven =
            impl_->stream == nullptr || aclrtSynchronizeStream(impl_->stream) == ACL_SUCCESS;
        if (!proven) storage.abandon();
        return status_invalid_state(ACL_ERROR_BAD_ALLOC);
    }
    // Every caller transfers into these bytes before retaining them, so a marker
    // recorded now completes only after that transfer. The staging is released
    // only when its own marker completed, independent of any device block.
    aclrtEvent marker = record_marker();
    impl_->queue->retain_staging(std::move(pinned), marker);
    return status_success();
}

Ascend910B3Status Context::register_buffer(DeviceStorage&& storage, size_t num_vars,
                                           BufferKind kind, BufferRef& out) noexcept {
    out.reset();
    if (closed()) return status_invalid_state();
    if (poisoned()) return status_invalid_state(first_error());
    if (!storage.valid()) return status_invalid_parameters();
    if (kind == BufferKind::MleFr) {
        size_t expected = 0;
        if (!dimension_bytes(num_vars, kFrBytes, expected) || expected != storage.size())
            return status_invalid_parameters();
    }
    std::shared_ptr<DeviceBlock> block;
    try {
        block = std::make_shared<DeviceBlock>(storage.raw_, storage.data_, storage.size(),
                                              num_vars, kind, impl_->queue);
    } catch (...) {
        // `storage` still owns the allocation and frees it; no work is queued.
        return status_invalid_state(ACL_ERROR_BAD_ALLOC);
    }
    storage.raw_ = nullptr;
    storage.data_ = nullptr;
    storage.bytes_ = 0;
    bool was_closed = false;
    bool registered = false;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        was_closed = impl_->closed;
        if (!was_closed) {
            try {
                registered = impl_->registry.emplace(block->data(), block).second;
            } catch (...) {
                registered = false;
            }
        }
    }
    if (was_closed) {
        block.reset(); // deferred retirement; the context is draining
        return status_invalid_state();
    }
    if (!registered) {
        block.reset();
        return status_invalid_state(ACL_ERROR_INTERNAL_ERROR);
    }
    out = BufferRef(std::move(block));
    return status_success();
}

Ascend910B3Status Context::create_buffer(size_t bytes, size_t num_vars, BufferKind kind,
                                         BufferRef& out) noexcept {
    out.reset();
    DeviceStorage storage;
    const Ascend910B3Status status = allocate(bytes, kDeviceAlignment, storage);
    if (!status_ok(status)) return status;
    return register_buffer(std::move(storage), num_vars, kind, out);
}

bool Context::lookup(DeviceAddress address, BufferRef& out) noexcept {
    out.reset();
    if (address == nullptr) return false;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed) return false;
    const auto it = impl_->registry.find(address);
    if (it == impl_->registry.end()) return false;
    out = BufferRef(it->second);
    return true;
}

bool Context::lookup(DeviceAddress address, size_t num_vars, BufferRef& out) noexcept {
    if (!lookup(address, out)) return false;
    if (out.num_vars() != num_vars) {
        out.reset();
        return false;
    }
    return true;
}

Ascend910B3Status Context::upload_mle(size_t num_vars, const uint8_t* evaluations,
                                      size_t evaluation_count,
                                      DeviceAddress& out) noexcept {
    out = nullptr;
    if (closed()) return status_invalid_state();
    if (poisoned()) return status_invalid_state(first_error());
    size_t bytes = 0;
    if (!dimension_bytes(num_vars, kFrBytes, bytes)) return status_invalid_parameters();
    if (evaluations == nullptr || evaluation_count != bytes / kFrBytes)
        return status_invalid_parameters();
    if (!canonical_fr_array(evaluations, evaluation_count))
        return status_invalid_parameters();
    const Ascend910B3Status ensured = ensure_current();
    if (!status_ok(ensured)) return ensured;

    PinnedStorage staging;
    Ascend910B3Status status = allocate_pinned(bytes, kPinnedAlignment, staging);
    if (!status_ok(status)) return status;
    std::memcpy(staging.data(), evaluations, bytes);

    DeviceStorage device;
    status = allocate(bytes, kDeviceAlignment, device);
    if (!status_ok(status)) return status; // staging frees; nothing enqueued yet

    std::shared_ptr<DeviceBlock> block;
    try {
        block = std::make_shared<DeviceBlock>(device.raw_, device.data_, device.size(),
                                              num_vars, BufferKind::MleFr, impl_->queue);
    } catch (...) {
        return status_invalid_state(ACL_ERROR_BAD_ALLOC); // device frees; nothing enqueued
    }
    device.raw_ = nullptr;
    device.data_ = nullptr;
    device.bytes_ = 0;

    aclError acl_status = aclrtMemcpyAsync(block->data(), block->bytes(), staging.data(), bytes,
                                           ACL_MEMCPY_HOST_TO_DEVICE, impl_->stream);
    if (acl_status != ACL_SUCCESS) {
        // A failed enqueue may still leave previously submitted work outstanding.
        // The central retain_pinned owns the staging hand-off (marker, OOM and
        // failed-sync policy) and consumes the owner; its status must not mask the
        // original copy error.
        (void)retain_pinned(std::move(staging));
        block.reset();
        poison(acl_status);
        return status_from_acl(acl_status);
    }

    const Ascend910B3Status retained = retain_pinned(std::move(staging));
    if (!status_ok(retained)) {
        // retain_pinned either queued the staging with its marker or intentionally
        // abandoned it; the device block then waits for a real synchronize.
        block.reset();
        return retained;
    }

    bool was_closed = false;
    bool registered = false;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        was_closed = impl_->closed;
        if (!was_closed) {
            try {
                registered = impl_->registry.emplace(block->data(), block).second;
            } catch (...) {
                registered = false;
            }
        }
    }
    if (was_closed) {
        block.reset();
        return status_invalid_state();
    }
    if (!registered) {
        block.reset();
        return status_invalid_state(ACL_ERROR_INTERNAL_ERROR);
    }
    DeviceAddress handle = block->data();
    out = handle;
    return status_success();
}

Ascend910B3Status Context::copy_async(void* dst, const void* src, size_t bytes,
                                      aclrtMemcpyKind kind) noexcept {
    if (closed()) return status_invalid_state();
    if (poisoned()) return status_invalid_state(first_error());
    if (bytes == 0) return status_success();
    if (dst == nullptr || src == nullptr) return status_invalid_parameters();
    const Ascend910B3Status ensured = ensure_current();
    if (!status_ok(ensured)) {
        poison(ensured.native_code);
        return ensured;
    }
    const aclError status = aclrtMemcpyAsync(dst, bytes, src, bytes, kind, impl_->stream);
    if (status != ACL_SUCCESS) {
        poison(status);
        return status_from_acl(status);
    }
    return status_success();
}

Ascend910B3Status Context::copy_to_device_async(void* dst, const void* host_src,
                                                size_t bytes) noexcept {
    return copy_async(dst, host_src, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
}

Ascend910B3Status Context::copy_to_host_async(void* host_dst, const void* src,
                                              size_t bytes) noexcept {
    return copy_async(host_dst, src, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
}

Ascend910B3Status Context::reclaim_retired(bool proven) noexcept {
    std::vector<PendingBlock> blocks;
    std::vector<std::shared_ptr<PinnedStorage>> staging;
    std::vector<aclrtEvent> events;
    impl_->queue->take_ready(proven, blocks, staging, events);
    // Pinned bytes and their markers are released only after their own transfer
    // marker completed (or after a real synchronize).
    staging.clear();
    for (aclrtEvent event : events) impl_->queue->destroy_marker(event);
    aclError first = ACL_SUCCESS;
    for (PendingBlock& block : blocks) {
        impl_->queue->destroy_marker(block.event);
        const aclError status = aclrtFree(block.raw);
        if (status != ACL_SUCCESS && first == ACL_SUCCESS) first = status;
    }
    if (first != ACL_SUCCESS) {
        poison(first);
        return status_from_acl(first);
    }
    return status_success();
}

Ascend910B3Status Context::synchronize() noexcept {
    if (closed()) return status_invalid_state();
    const Ascend910B3Status ensured = ensure_current();
    if (!status_ok(ensured)) {
        poison(ensured.native_code);
        return ensured;
    }
    const aclError status = aclrtSynchronizeStream(impl_->stream);
    if (status != ACL_SUCCESS) {
        // Retired allocations and staging stay queued: freeing them now could
        // race outstanding work. The context is poisoned and destroy retries.
        poison(status);
        return status_from_acl(status);
    }
    const Ascend910B3Status drained = reclaim_retired(true);
    if (!status_ok(drained)) return drained;
    if (poisoned()) return status_invalid_state(first_error());
    return status_success();
}

Ascend910B3Status Context::poll_retired() noexcept {
    if (closed()) return status_invalid_state();
    const Ascend910B3Status ensured = ensure_current();
    if (!status_ok(ensured)) {
        poison(ensured.native_code);
        return ensured;
    }
    // No aclrtStreamQuery here: this stack can report ACL_STREAM_STATUS_COMPLETE
    // while a just-enqueued task is still missing from the device queue. Only the
    // per-resource retirement markers authorize a release.
    const Ascend910B3Status drained = reclaim_retired(false);
    if (!status_ok(drained)) return drained;
    if (poisoned()) return status_invalid_state(first_error());
    return status_success();
}

Ascend910B3Status Context::completion(bool& complete) noexcept {
    complete = false;
    if (closed()) return status_invalid_state();
    const Ascend910B3Status ensured = ensure_current();
    if (!status_ok(ensured)) {
        poison(ensured.native_code);
        return ensured;
    }
    aclrtStreamStatus completion_status = ACL_STREAM_STATUS_NOT_READY;
    const aclError status = aclrtStreamQuery(impl_->stream, &completion_status);
    if (status != ACL_SUCCESS) {
        poison(status);
        return status_from_acl(status);
    }
    complete = completion_status == ACL_STREAM_STATUS_COMPLETE;
    return status_success();
}

Ascend910B3Status Context::retire(DeviceAddress address) noexcept {
    if (closed()) return status_invalid_state();
    if (address == nullptr) return status_invalid_parameters();
    std::shared_ptr<DeviceBlock> block;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (impl_->closed) return status_invalid_state();
        const auto it = impl_->registry.find(address);
        if (it == impl_->registry.end()) return status_invalid_parameters();
        block = std::move(it->second);
        impl_->registry.erase(it);
    }
    // Dropping the registry reference only retires; retained BufferRefs keep the
    // storage alive and usable, so no marker is recorded here: a marker recorded
    // now could complete before a later use enqueued through a retained ref. The
    // last owner's destruction queues the raw, and the next poll records its
    // one-shot marker after that enqueue (see take_ready).
    block.reset();
    (void)poll_retired();
    return status_success();
}

// ---------------------------------------------------------------------------
// Context registry and C ABI entry points.
// ---------------------------------------------------------------------------

Ascend910B3Status acquire_context(Ascend910B3NativeContext* raw, ContextRef& out) noexcept {
    out = ContextRef();
    if (raw == nullptr) return status_invalid_parameters();
    std::shared_ptr<Context> context;
    {
        std::lock_guard<std::mutex> lock(context_mutex());
        const auto it = contexts().find(raw);
        if (it == contexts().end()) return status_invalid_state();
        context = it->second;
    }
    out = ContextRef(std::move(context));
    return status_success();
}

Ascend910B3Status create_context(int32_t device_id, Ascend910B3NativeContext** out) noexcept {
    if (out != nullptr) *out = nullptr;
    return guarded([&]() -> Ascend910B3Status {
        if (out == nullptr || device_id < 0) return status_invalid_parameters();
        std::shared_ptr<RuntimeHandle> runtime;
        aclError status = ACL_SUCCESS;
        if (!acquire_runtime(runtime, status)) return status_from_acl(status);

        // Activate the device once per context. We never call aclrtResetDevice:
        // destroying a context must not reset a device another context uses.
        int32_t active_device = -1;
        if (aclrtGetDevice(&active_device) != ACL_SUCCESS || active_device != device_id) {
            status = aclrtSetDevice(device_id);
            if (status != ACL_SUCCESS) return status_from_acl(status);
        }

        aclrtContext previous = nullptr;
        bool have_previous = false;
        if (aclrtGetCurrentContext(&previous) == ACL_SUCCESS && previous != nullptr)
            have_previous = true;

        aclrtContext acl_ctx = nullptr;
        status = aclrtCreateContext(&acl_ctx, device_id);
        if (status != ACL_SUCCESS) return status_from_acl(status); // runtime ref releases
        status = aclrtSetCurrentContext(acl_ctx);
        if (status != ACL_SUCCESS) {
            (void)aclrtDestroyContext(acl_ctx);
            if (have_previous) (void)aclrtSetCurrentContext(previous);
            return status_from_acl(status);
        }
        aclrtStream stream = nullptr;
        status = aclrtCreateStream(&stream);
        if (status != ACL_SUCCESS) {
            (void)aclrtDestroyContext(acl_ctx);
            if (have_previous) (void)aclrtSetCurrentContext(previous);
            return status_from_acl(status);
        }

        std::shared_ptr<Context> context;
        try {
            auto impl = std::make_unique<Context::Impl>();
            impl->device_id = device_id;
            impl->acl_ctx = acl_ctx;
            impl->stream = stream;
            impl->runtime = runtime;
            impl->queue = std::make_shared<RetireQueue>();
            impl->queue->stream = stream;
            impl->queue->runtime = runtime;
            // make_shared cannot reach the private constructor; this factory is
            // the authorized owner instead.
            context = std::shared_ptr<Context>(new Context(std::move(impl)));
        } catch (...) {
            (void)aclrtDestroyStream(stream);
            (void)aclrtDestroyContext(acl_ctx);
            if (have_previous) (void)aclrtSetCurrentContext(previous);
            return status_invalid_state(ACL_ERROR_BAD_ALLOC);
        }

        Ascend910B3NativeContext* handle =
            reinterpret_cast<Ascend910B3NativeContext*>(context.get());
        try {
            std::lock_guard<std::mutex> lock(context_mutex());
            contexts().emplace(handle, context);
        } catch (...) {
            // The context object owns no destructor-side ACL action; tear down the
            // explicit handles here so a failed registration cannot leak them.
            context.reset();
            (void)aclrtDestroyStream(stream);
            (void)aclrtDestroyContext(acl_ctx);
            if (have_previous) (void)aclrtSetCurrentContext(previous);
            return status_invalid_state(ACL_ERROR_BAD_ALLOC);
        }
        *out = handle;
        return status_success();
    });
}

Ascend910B3Status destroy_context(Ascend910B3NativeContext* raw) noexcept {
    return guarded([&]() -> Ascend910B3Status {
        if (raw == nullptr) return status_invalid_parameters();
        std::shared_ptr<Context> context;
        {
            // Locate without removing: the handle must stay reachable until the
            // whole teardown below is proven safe.
            std::lock_guard<std::mutex> lock(context_mutex());
            const auto it = contexts().find(raw);
            if (it == contexts().end()) return status_invalid_state();
            context = it->second;
        }

        Context::Impl& impl = *context->impl_;
        aclError first = ACL_SUCCESS;
        {
            std::lock_guard<std::mutex> lock(impl.mutex);
            impl.closed = true; // no new entry point may use this handle
            if (impl.poisoned) first = impl.first_error;
        }

        aclrtContext previous = nullptr;
        const aclrtContext owned = impl.acl_ctx;
        bool have_previous = false;
        if (aclrtGetCurrentContext(&previous) == ACL_SUCCESS && previous != nullptr &&
            previous != owned)
            have_previous = true;

        // All device calls below must run on this context. A failed bind means we
        // cannot prove safe release, so it is treated like a failed synchronize.
        if (owned != nullptr) {
            const aclError bound = aclrtSetCurrentContext(owned);
            if (bound != ACL_SUCCESS) {
                context->poison(bound);
                if (first == ACL_SUCCESS) first = bound;
                return status_invalid_state(first);
            }
        }

        if (impl.stream != nullptr) {
            const aclError synchronized = aclrtSynchronizeStream(impl.stream);
            if (synchronized != ACL_SUCCESS) {
                context->poison(synchronized);
                if (first == ACL_SUCCESS) first = synchronized;
                // Completion was never observed. Retain the context handle, its
                // registry, retire queue, staged pinned memory, stream and runtime
                // so nothing is released behind an unproven stream. A later
                // destroy attempt may succeed; no "release rather than leak".
                return status_invalid_state(first);
            }
        }

        // Completion (or the absence of a stream) is proven. From here releasing
        // device buffers, staging, the stream and the ACL context is safe.
        {
            std::lock_guard<std::mutex> lock(context_mutex());
            const auto it = contexts().find(raw);
            if (it != contexts().end() && it->second == context) contexts().erase(it);
        }

        // Detach before dropping registry ownership so remaining destructors free
        // immediately instead of queueing for a context that is going away. The
        // stream is disowned first: it was synchronized above, so late destructors
        // must not synchronize a handle that is about to be destroyed.
        impl.queue->detach();
        impl.queue->drop_stream();

        {
            // Exception-free: clear() only destroys shared_ptrs. Blocks still held
            // elsewhere survive and free on their own last release.
            std::lock_guard<std::mutex> lock(impl.mutex);
            impl.registry.clear();
        }

        std::vector<PendingBlock> blocks;
        std::vector<PendingStaging> staging;
        std::vector<aclrtEvent> unproven;
        impl.queue->take(blocks, staging, unproven);
        // The stream synchronize above succeeded, so every marker completed and
        // every staged transfer is done: destroy the event handles, release the
        // pinned bytes (staging destruction) and free the blocks.
        for (PendingStaging& entry : staging) {
            impl.queue->destroy_marker(entry.event);
        }
        staging.clear();
        for (aclrtEvent event : unproven) impl.queue->destroy_marker(event);
        unproven.clear();
        for (PendingBlock& block : blocks) {
            impl.queue->destroy_marker(block.event);
            const aclError freed = aclrtFree(block.raw);
            if (freed != ACL_SUCCESS && first == ACL_SUCCESS) first = freed;
        }

        if (impl.stream != nullptr) {
            const aclError destroyed = aclrtDestroyStream(impl.stream);
            impl.stream = nullptr;
            if (destroyed != ACL_SUCCESS && first == ACL_SUCCESS) first = destroyed;
        }
        if (impl.acl_ctx != nullptr) {
            const aclError destroyed = aclrtDestroyContext(impl.acl_ctx);
            impl.acl_ctx = nullptr;
            if (destroyed != ACL_SUCCESS && first == ACL_SUCCESS) first = destroyed;
        }
        if (have_previous) (void)aclrtSetCurrentContext(previous);

        impl.device_id = -1;
        impl.queue.reset();
        impl.runtime.reset();
        context.reset();
        return first == ACL_SUCCESS ? status_success() : status_invalid_state(first);
    });
}

Ascend910B3Status retire_buffer(Ascend910B3NativeContext* raw, DeviceAddress address) noexcept {
    return guarded([&]() -> Ascend910B3Status {
        ContextRef context;
        Ascend910B3Status status = acquire_context(raw, context);
        if (!status_ok(status)) return status;
        return context->retire(address);
    });
}

Ascend910B3Status upload_mle(Ascend910B3NativeContext* raw, size_t num_vars,
                             const uint8_t* evaluations, size_t evaluation_count,
                             DeviceAddress* out) noexcept {
    if (out != nullptr) *out = nullptr;
    return guarded([&]() -> Ascend910B3Status {
        if (out == nullptr) return status_invalid_parameters();
        ContextRef context;
        Ascend910B3Status status = acquire_context(raw, context);
        if (!status_ok(status)) return status;
        DeviceAddress handle = nullptr;
        status = context->upload_mle(num_vars, evaluations, evaluation_count, handle);
        if (!status_ok(status)) return status;
        *out = handle;
        return status_success();
    });
}

} // namespace hyperplonk::ascend910b3::native
