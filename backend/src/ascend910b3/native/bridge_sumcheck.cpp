// SumCheck prover bridge over the resident arbitrary-degree kernel family.
//
// Ownership: retain distinct immutable canonical originals, and allocate only
// shrinking half/quarter A/B slots. Three immutable absolute-address binding
// arrays select originals/A/B without per-round metadata transfers. Coefficients
// are scaled once on device by R^(arity+1); grid weights stay Montgomery.
// Degree-2, nonempty, all-arity-2 states use the Cube message kernel on the
// canonical NATURAL table every round; the previous-challenge fold is always
// the canonical natural Vector fold, so no packed table ever exists. Every
// other state uses the generic canonical natural Vector message. All
// workspaces and pinned staging are allocated once and retired with state.
//
// A round enqueues at most: one 32B challenge H2D, one fold per distinct factor,
// product_count * ceil((degree+1)/chunk) message passes, one reduction, one
// (degree+1)*32B D2H and then synchronizes. Every argument is validated before
// the first side effect, so a rejected call neither advances the transcript nor
// writes output. A failure after execution started poisons only this state
// (later calls return InvalidState) and keeps every buffer/pinned slot alive
// until the stream is really drained; context teardown is never attempted here.
//
// The live-state registry makes an unknown or foreign raw handle
// InvalidParameters instead of dereferencing an unknown pointer, and the owning
// context is compared by identity.

#include "bridge.h"
#include "bridge_runtime.h"

#include "field.h"
#include "mle_kernels.h"
#include "sumcheck_kernels.h"
#include "sumcheck_tensor.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rt = hyperplonk::ascend910b3::native;
namespace hp = hyperplonk::ascend910b3;
namespace sc = hyperplonk::ascend910b3::detail;

// The C ABI names this tag; only this translation unit makes it concrete. The
// numerical state itself lives in the guarded live-state registry below, keyed
// by this handle, so no caller-supplied pointer is ever dereferenced blindly.
struct Ascend910B3NativeSumCheckProverState {
    unsigned char reserved = 0;
};

namespace {

static_assert(sizeof(hp::Fr) == rt::kFrBytes, "canonical Fr is 32 bytes");

// General (non-MLE) registered buffers carry no dimension.
constexpr size_t kGeneralNumVars = std::numeric_limits<size_t>::max();
// Any single allocation or arena offset must stay addressable by ptrdiff_t.
constexpr uint64_t kAddressLimit =
    static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max());

struct NativeSumcheck {
    rt::ContextRef context;
    size_t num_vars = 0;
    size_t max_degree = 0;
    sc::SumcheckShape shape{};
    uint32_t aiv_cores = 0;
    uint32_t max_blocks = 0;
    size_t distinct_count = 0;
    size_t index_count = 0;
    size_t product_count = 0;
    // Retain originals even after callers retire their public MLE handles.
    std::vector<rt::BufferRef> inputs;
    rt::BufferRef arena;        // distinct_count * (half + quarter) slots
    rt::BufferRef bindings_original;
    rt::BufferRef bindings_slot_a;
    rt::BufferRef bindings_slot_b;
    rt::BufferRef coefficient_arity;
    rt::BufferRef tensor_workspace;
    bool tensor_d2 = false;
    rt::BufferRef coefficients; // c*R^(arity+1), one valid slot when empty
    rt::BufferRef weights;      // degree+1 Mont grid weights 0..degree
    rt::BufferRef partial;      // product_count*(degree+1)*max_blocks records
    rt::BufferRef output;       // degree+1 canonical Fr
    rt::BufferRef coordinate;   // one canonical challenge Fr
    rt::PinnedStorage coordinate_host; // 32B challenge staging
    rt::PinnedStorage output_host;     // (degree+1)*32B result staging
    std::vector<size_t> product_offsets; // product_count + 1 entries
    size_t rounds_done = 0;
    bool failed = false;
    int32_t failure_code = 0;
};

using StatePtr = std::shared_ptr<NativeSumcheck>;
using StateMap =
    std::unordered_map<const Ascend910B3NativeSumCheckProverState*, StatePtr>;

// Never destroyed: an undrained failed teardown must not release state from a
// static destructor. Entry points validate by membership, never by dereference.
std::mutex& state_mutex() noexcept {
    static std::mutex mutex;
    return mutex;
}
StateMap& live_states() noexcept {
    static auto* map = new StateMap();
    return *map;
}

Ascend910B3Status invalid_parameters() noexcept {
    return rt::status_invalid_parameters();
}
Ascend910B3Status invalid_state() noexcept {
    return rt::status_invalid_state(static_cast<int32_t>(ACL_ERROR_INTERNAL_ERROR));
}

// Hand a staging buffer whose transfer may still be queued to the context. On a
// context that cannot accept it the stream is drained first; if even that fails
// the bytes are intentionally leaked so queued work can never read freed host
// memory.
void handoff_staging(rt::Context& context, rt::PinnedStorage&& staging) noexcept {
    if (!staging.valid()) return;
    const Ascend910B3Status retained = context.retain_pinned(std::move(staging));
    if (rt::status_ok(retained) || !staging.valid()) return;
    const Ascend910B3Status drained = context.synchronize();
    if (rt::status_ok(drained)) return;
    (void)new (std::nothrow) rt::PinnedStorage(std::move(staging));
}

// Small host metadata upload: pinned staging -> registered General buffer. The
// staging is handed to the context right after a successful enqueue, so it can
// never be freed while the asynchronous copy still reads it.
Ascend910B3Status upload_bytes(rt::Context& context, const void* host, size_t bytes,
                               rt::BufferRef& out) noexcept {
    out.reset();
    if (bytes == 0) return rt::status_success();
    if (host == nullptr) return invalid_parameters();
    rt::PinnedStorage staging;
    Ascend910B3Status status =
        context.allocate_pinned(bytes, rt::kPinnedAlignment, staging);
    if (!rt::status_ok(status)) return status;
    std::memcpy(staging.data(), host, bytes);
    rt::BufferRef buffer;
    status = context.create_buffer(bytes, kGeneralNumVars, rt::BufferKind::General,
                                   buffer);
    if (!rt::status_ok(status)) return status;
    status = context.copy_to_device_async(buffer.data(), staging.data(), bytes);
    if (!rt::status_ok(status)) return status; // failed copy never reads staging
    handoff_staging(context, std::move(staging));
    out = std::move(buffer);
    return rt::status_success();
}

// Retires every registered allocation the state created if construction does
// not reach publication. `retire` only drops registry ownership; the retained
// BufferRef keeps the storage usable until the shared deferred-free queue can
// observe stream completion.
struct OwnedBuffers {
    rt::Context* context = nullptr;
    std::vector<rt::BufferRef*> buffers;
    bool published = false;

    ~OwnedBuffers() {
        if (published || context == nullptr) return;
        for (rt::BufferRef* ref : buffers) {
            if (ref != nullptr && ref->valid()) (void)context->retire(ref->data());
        }
    }

    void track(rt::BufferRef& ref) { buffers.push_back(&ref); }
};

Ascend910B3Status create_buffer_for(rt::Context& context, size_t bytes,
                                    rt::BufferRef& out) noexcept {
    out.reset();
    if (bytes == 0) return rt::status_success();
    return context.create_buffer(bytes, kGeneralNumVars, rt::BufferKind::General, out);
}

// Resolves the context, validates the state handle by registry membership and
// checks that the state belongs to exactly this live context.
Ascend910B3Status bind_state(Ascend910B3NativeContext* raw_context,
                             Ascend910B3NativeSumCheckProverState* raw_state,
                             rt::ContextRef& cref, StatePtr& state) noexcept {
    state.reset();
    Ascend910B3Status status = rt::acquire_context(raw_context, cref);
    if (!rt::status_ok(status)) return status;
    if (cref->closed()) return rt::status_invalid_state();
    if (cref->poisoned())
        return rt::status_invalid_state(static_cast<int32_t>(cref->first_error()));
    if (raw_state == nullptr) return invalid_parameters();
    {
        std::lock_guard<std::mutex> lock(state_mutex());
        StateMap& map = live_states();
        const auto it = map.find(raw_state);
        if (it == map.end()) return invalid_parameters();
        state = it->second;
    }
    if (!state || !state->context) return invalid_parameters();
    if (&*state->context != &*cref) return invalid_parameters();
    return rt::status_success();
}

// Host metadata: little-endian 32-byte canonical encoding of a small integer.
void store_small_fr(uint8_t* packet, uint64_t value) noexcept {
    for (unsigned i = 0; i < 8; ++i) packet[i] = static_cast<uint8_t>(value >> (8 * i));
}

Ascend910B3Status create_state(Ascend910B3NativeContext* raw_context, size_t num_vars,
                               size_t max_degree, const rt::DeviceAddress* factors,
                               size_t factor_count, const uint8_t* coefficients,
                               const size_t* product_offsets, size_t product_count,
                               const size_t* factor_indices, size_t index_count,
                               Ascend910B3NativeSumCheckProverState** out) {
    if (out == nullptr) return invalid_parameters();
    *out = nullptr;

    rt::ContextRef cref;
    Ascend910B3Status status = rt::acquire_context(raw_context, cref);
    if (!rt::status_ok(status)) return status;
    if (cref->closed()) return rt::status_invalid_state();
    if (cref->poisoned())
        return rt::status_invalid_state(static_cast<int32_t>(cref->first_error()));

    // ---- Pure validation: no device side effects before this block ends. ----
    if (num_vars == 0 || max_degree == 0) return invalid_parameters();
    if (num_vars >= 64) return invalid_parameters();
    if (max_degree > static_cast<size_t>(std::numeric_limits<uint32_t>::max() - 1))
        return invalid_parameters();
    if (product_count > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
        return invalid_parameters();

    size_t table_bytes = 0;
    if (!rt::dimension_bytes(num_vars, rt::kFrBytes, table_bytes))
        return invalid_parameters();
    const uint64_t stride64 = sc::sumcheck_table_stride(num_vars);
    if (stride64 > kAddressLimit) return invalid_parameters();

    const size_t grid = max_degree + 1;
    size_t packet_bytes = 0;
    if (!rt::checked_mul(grid, rt::kFrBytes, packet_bytes)) return invalid_parameters();
    if (static_cast<uint64_t>(packet_bytes) > kAddressLimit)
        return invalid_parameters();

    // Product metadata: offsets monotone from zero, every product arity in
    // 1..max_degree, final offset equals the index count.
    if (product_offsets == nullptr) return invalid_parameters();
    if (product_offsets[0] != 0) return invalid_parameters();
    bool tensor_d2 = max_degree == 2 && product_count > 0;
    std::vector<uint32_t> arities(product_count);
    for (size_t p = 0; p < product_count; ++p) {
        const size_t begin = product_offsets[p];
        const size_t end = product_offsets[p + 1];
        if (end < begin) return invalid_parameters();
        const size_t arity = end - begin;
        if (arity == 0 || arity > max_degree) return invalid_parameters();
        arities[p] = static_cast<uint32_t>(arity);
        tensor_d2 = tensor_d2 && arity == 2;
    }
    if (product_offsets[product_count] != index_count) return invalid_parameters();

    if (factor_count > 0 && factors == nullptr) return invalid_parameters();
    if (index_count > 0 && factor_indices == nullptr) return invalid_parameters();
    if (product_count > 0) {
        if (coefficients == nullptr) return invalid_parameters();
        if (!rt::canonical_fr_array(coefficients, product_count))
            return invalid_parameters();
    }
    for (size_t i = 0; i < index_count; ++i) {
        if (factor_indices[i] >= factor_count) return invalid_parameters();
    }

    const sc::SumcheckShape shape = sc::sumcheck_shape(max_degree);
    if (shape.lanes == 0 || shape.chunk == 0 || shape.chunks == 0)
        return invalid_parameters();
    const uint32_t aiv_cores = sc::mle_aiv_cores();
    if (aiv_cores == 0) return invalid_state();
    const uint64_t round_zero_rest = uint64_t{1} << (num_vars - 1);
    const uint32_t max_blocks =
        sc::sumcheck_message_blocks(round_zero_rest, shape.lanes, aiv_cores);
    if (max_blocks == 0) return invalid_state();

    // Stable dedup by native handle address: repeated products/indices keep
    // their exact multiplicities, only the arena slot list is deduplicated.
    std::vector<rt::BufferRef> distinct;
    std::vector<size_t> slot_of(factor_count, 0);
    {
        std::unordered_map<rt::DeviceAddress, size_t> seen;
        seen.reserve(factor_count);
        for (size_t i = 0; i < factor_count; ++i) {
            const rt::DeviceAddress address = factors[i];
            if (address == nullptr) return invalid_parameters();
            const auto it = seen.find(address);
            if (it != seen.end()) {
                slot_of[i] = it->second;
                continue;
            }
            rt::BufferRef ref;
            if (!cref->lookup(address, num_vars, ref)) return invalid_parameters();
            const size_t slot = distinct.size();
            distinct.push_back(std::move(ref));
            seen.emplace(address, slot);
            slot_of[i] = slot;
        }
    }
    const size_t distinct_count = distinct.size();

    size_t arena_bytes = 0;
    if (distinct_count > 0) {
        if (!rt::checked_mul(static_cast<size_t>(stride64), distinct_count, arena_bytes))
            return invalid_parameters();
        if (static_cast<uint64_t>(arena_bytes) > kAddressLimit)
            return invalid_parameters();
    }
    size_t bindings_bytes = 0;
    if (!rt::checked_mul(index_count, rt::kFrBytes, bindings_bytes))
        return invalid_parameters();
    if (static_cast<uint64_t>(bindings_bytes) > kAddressLimit)
        return invalid_parameters();
    size_t partial_bytes = 0;
    if (product_count > 0) {
        size_t products = 0;
        size_t rows = 0;
        size_t slots = 0;
        if (!rt::checked_mul(product_count, grid, products)) return invalid_parameters();
        if (!rt::checked_mul(products, max_blocks, rows)) return invalid_parameters();
        if (!rt::checked_mul(rows, rt::kFrBytes, slots)) return invalid_parameters();
        if (static_cast<uint64_t>(slots) > kAddressLimit) return invalid_parameters();
        partial_bytes = slots;
    }

    // ---- Resource construction (still no stream work). ----
    const size_t stride = static_cast<size_t>(stride64);
    StatePtr state;
    try {
        state = std::make_shared<NativeSumcheck>();
    } catch (...) {
        return rt::status_invalid_state(static_cast<int32_t>(ACL_ERROR_BAD_ALLOC));
    }
    state->context = cref;
    state->num_vars = num_vars;
    state->max_degree = max_degree;
    state->shape = shape;
    state->aiv_cores = aiv_cores;
    state->max_blocks = max_blocks;
    state->distinct_count = distinct_count;
    state->index_count = index_count;
    state->product_count = product_count;
    state->tensor_d2 = tensor_d2;
    state->inputs = std::move(distinct);
    state->product_offsets.assign(product_offsets, product_offsets + product_count + 1);

    // Transient create-time canonical sources and the failure-cleanup guard are
    // declared after the state so they outlive the guard's retirement pass.
    rt::BufferRef coefficient_source;
    rt::BufferRef weight_source;
    OwnedBuffers owned;
    owned.context = &*cref;
    // Every state slot this create may track plus the two transient canonical
    // sources. Reserve before enqueues so track() cannot allocate during
    // failure cleanup.
    owned.buffers.reserve(16);

    status = create_buffer_for(*cref, arena_bytes, state->arena);
    if (!rt::status_ok(status)) return status;
    owned.track(state->arena);
    status = create_buffer_for(*cref, partial_bytes, state->partial);
    if (!rt::status_ok(status)) return status;
    owned.track(state->partial);
    const size_t coefficient_slots = product_count;
    size_t coefficient_bytes = 0;
    if (!rt::checked_mul(coefficient_slots, rt::kFrBytes, coefficient_bytes))
        return invalid_parameters();
    status = create_buffer_for(*cref, coefficient_bytes, state->coefficients);
    if (!rt::status_ok(status)) return status;
    owned.track(state->coefficients);
    status = create_buffer_for(*cref, packet_bytes, state->weights);
    if (!rt::status_ok(status)) return status;
    owned.track(state->weights);
    status = create_buffer_for(*cref, packet_bytes, state->output);
    if (!rt::status_ok(status)) return status;
    owned.track(state->output);
    status = create_buffer_for(*cref, rt::kFrBytes, state->coordinate);
    if (!rt::status_ok(status)) return status;
    owned.track(state->coordinate);
    if (tensor_d2) {
        const uint64_t message_groups =
            (round_zero_rest + sc::sumcheck_tensor_tile - 1) / sc::sumcheck_tensor_tile;
        const uint32_t groups = static_cast<uint32_t>(
            message_groups < sc::sumcheck_tensor_max_groups
                ? message_groups : sc::sumcheck_tensor_max_groups);
        const uint64_t message_workspace_bytes = sc::sumcheck_tensor_workspace_bytes(groups);
        if (message_workspace_bytes == 0) return invalid_state();
        status = create_buffer_for(*cref, static_cast<size_t>(message_workspace_bytes),
            state->tensor_workspace);
        if (!rt::status_ok(status)) return status;
        owned.track(state->tensor_workspace);
    }
    status = cref->allocate_pinned(rt::kFrBytes, rt::kPinnedAlignment,
                                   state->coordinate_host);
    if (!rt::status_ok(status)) return status;
    status = cref->allocate_pinned(packet_bytes, rt::kPinnedAlignment,
                                   state->output_host);
    if (!rt::status_ok(status)) return status;

    // A handle slot is allocated before any enqueue so a late failure leaves no
    // queued work behind for a handle that was never published.
    std::unique_ptr<Ascend910B3NativeSumCheckProverState> handle;
    try {
        handle = std::make_unique<Ascend910B3NativeSumCheckProverState>();
    } catch (...) {
        return rt::status_invalid_state(static_cast<int32_t>(ACL_ERROR_BAD_ALLOC));
    }

    // ---- One-time coefficient scaling, grid conversion and metadata uploads. ----
    const aclrtStream stream = cref->stream();

    if (product_count > 0) {
        status = upload_bytes(*cref, coefficients, product_count * rt::kFrBytes,
                              coefficient_source);
        if (!rt::status_ok(status)) return status;
        owned.track(coefficient_source);
        status = upload_bytes(*cref, arities.data(), product_count * sizeof(uint32_t),
                              state->coefficient_arity);
        if (!rt::status_ok(status)) return status;
        owned.track(state->coefficient_arity);
        if (!sc::launch_sumcheck_scale_coefficients(
                stream, reinterpret_cast<const hp::Fr*>(coefficient_source.data()),
                reinterpret_cast<const uint32_t*>(state->coefficient_arity.data()),
                reinterpret_cast<hp::Fr*>(state->coefficients.data()),
                static_cast<uint32_t>(product_count), shape.lanes))
            return invalid_state();
    } else {
        // Empty whole sum: the reduce kernel writes canonical zeros and does not
        // read coefficients, but keep a valid zero-filled slot anyway so no
        // launch ever receives an uninitialized/absent pointer.
        static const uint8_t zero_coefficient[rt::kFrBytes] = {};
        status = upload_bytes(*cref, zero_coefficient, rt::kFrBytes,
                              state->coefficients);
        if (!rt::status_ok(status)) return status;
        owned.track(state->coefficients);
    }

    {
        std::vector<uint8_t> weight_host(packet_bytes, 0);
        for (size_t t = 0; t < grid; ++t) store_small_fr(&weight_host[t * rt::kFrBytes], t);
        status = upload_bytes(*cref, weight_host.data(), packet_bytes, weight_source);
        if (!rt::status_ok(status)) return status;
        owned.track(weight_source);
        if (!sc::launch_sumcheck_convert(
                stream, reinterpret_cast<const hp::Fr*>(weight_source.data()),
                reinterpret_cast<hp::Fr*>(state->weights.data()), grid, shape.lanes))
            return invalid_state();
    }

    if (index_count > 0) {
        const auto upload_bindings = [&](rt::BufferRef& bindings, size_t slot_kind)
            -> Ascend910B3Status {
            std::vector<uint8_t> host(bindings_bytes, 0);
            for (size_t i = 0; i < index_count; ++i) {
                const size_t slot = slot_of[factor_indices[i]];
                const void* address = state->inputs[slot].data();
                if (slot_kind != 0) {
                    const size_t offset = slot * stride +
                        (slot_kind == 2 ? static_cast<size_t>(sc::sumcheck_slot_b_offset(num_vars)) : 0);
                    address = static_cast<uint8_t*>(state->arena.data()) + offset;
                }
                store_small_fr(&host[i * rt::kFrBytes],
                               static_cast<uint64_t>(reinterpret_cast<uintptr_t>(address)));
            }
            const Ascend910B3Status uploaded =
                upload_bytes(*cref, host.data(), bindings_bytes, bindings);
            if (rt::status_ok(uploaded)) owned.track(bindings);
            return uploaded;
        };
        status = upload_bindings(state->bindings_original, 0);
        if (!rt::status_ok(status)) return status;
        if (sc::sumcheck_slot_a_bytes(num_vars) != 0) {
            status = upload_bindings(state->bindings_slot_a, 1);
            if (!rt::status_ok(status)) return status;
        }
        if (sc::sumcheck_slot_b_bytes(num_vars) != 0) {
            status = upload_bindings(state->bindings_slot_b, 2);
            if (!rt::status_ok(status)) return status;
        }
    }

    // Setup completes the resident conversions before the state is published;
    // this also reclaims the transient staging handed to the context above.
    status = cref->synchronize();
    if (!rt::status_ok(status)) return status;

    // The canonical sources were only inputs to the one-time conversions. With
    // the stream known complete they can be retired (and freed once these
    // references drop) instead of occupying memory for the state's lifetime.
    if (coefficient_source.valid()) {
        (void)cref->retire(coefficient_source.data());
        coefficient_source.reset();
    }
    if (weight_source.valid()) {
        (void)cref->retire(weight_source.data());
        weight_source.reset();
    }

    try {
        std::lock_guard<std::mutex> lock(state_mutex());
        StateMap& map = live_states();
        if (!map.emplace(handle.get(), state).second) return invalid_state();
    } catch (...) {
        return rt::status_invalid_state(static_cast<int32_t>(ACL_ERROR_BAD_ALLOC));
    }
    // Published: the state and its destroy path now own every allocation.
    owned.published = true;
    *out = handle.release();
    return rt::status_success();
}

Ascend910B3Status round_state(Ascend910B3NativeContext* raw_context,
                              Ascend910B3NativeSumCheckProverState* raw_state,
                              const uint8_t* challenge_or_null, uint8_t* out_evaluations,
                              size_t evaluation_count) noexcept {
    rt::ContextRef cref;
    StatePtr state;
    Ascend910B3Status status = bind_state(raw_context, raw_state, cref, state);
    if (!rt::status_ok(status)) return status;

    if (state->failed) return rt::status_invalid_state(state->failure_code);
    if (out_evaluations == nullptr) return invalid_parameters();
    if (evaluation_count != state->max_degree + 1) return invalid_parameters();

    const size_t round_index = state->rounds_done;
    if (round_index >= state->num_vars) return rt::status_invalid_state();
    if (round_index == 0) {
        // The first message is sent for the None challenge only; a supplied
        // challenge is a transcript-ordering error (InvalidState).
        if (challenge_or_null != nullptr) return rt::status_invalid_state();
    } else {
        // Later rounds fold exactly once with the previous challenge.
        if (challenge_or_null == nullptr) return rt::status_invalid_state();
        // A present challenge is a canonical Fr packet.
        if (!rt::canonical_fr(challenge_or_null)) return invalid_parameters();
    }

    const auto fail = [&](int32_t code) -> Ascend910B3Status {
        state->failed = true;
        state->failure_code =
            code == 0 ? static_cast<int32_t>(ACL_ERROR_INTERNAL_ERROR) : code;
        return rt::status_invalid_state(state->failure_code);
    };

    rt::Context::Scope scope = cref->scope();
    if (!scope.ok()) {
        // No side effect happened yet: report the context problem without
        // advancing or poisoning the transcript state.
        return scope.status();
    }

    const aclrtStream stream = cref->stream();
    const size_t num_vars = state->num_vars;
    const size_t grid = state->max_degree + 1;
    const size_t packet_bytes = grid * rt::kFrBytes;
    const uint32_t degree = static_cast<uint32_t>(state->max_degree);
    const uint32_t lanes = state->shape.lanes;
    const uint64_t stride64 = sc::sumcheck_table_stride(num_vars);
    const size_t stride = static_cast<size_t>(stride64);
    uint8_t* const arena_base = static_cast<uint8_t*>(state->arena.data());

    // Previous-challenge fold: one launch per distinct factor, once per round.
    // A whole sum with no products never reads the tables, so the challenge and
    // the folds are skipped entirely (the reduce still writes real zeros).
    if (round_index > 0 && state->product_count > 0) {
        std::memcpy(state->coordinate_host.data(), challenge_or_null, rt::kFrBytes);
        status = cref->copy_to_device_async(state->coordinate.data(),
                                            state->coordinate_host.data(),
                                            rt::kFrBytes);
        if (!rt::status_ok(status)) return fail(status.native_code);
        const uint32_t exponent =
            sc::sumcheck_fold_exponent(num_vars, round_index);
        const uint64_t source_offset =
            sc::sumcheck_previous_slot_offset(num_vars, round_index);
        const uint64_t destination_offset =
            sc::sumcheck_round_slot_offset(num_vars, round_index);
        for (size_t j = 0; j < state->distinct_count; ++j) {
            uint8_t* slot = arena_base + j * stride;
            const hp::Fr* source = round_index == 1
                ? reinterpret_cast<const hp::Fr*>(state->inputs[j].data())
                : reinterpret_cast<const hp::Fr*>(slot + static_cast<size_t>(source_offset));
            hp::Fr* destination =
                reinterpret_cast<hp::Fr*>(slot + static_cast<size_t>(destination_offset));
            const hp::Fr* coordinate =
                reinterpret_cast<const hp::Fr*>(state->coordinate.data());
            // Canonical natural Vector fold for every route: it converts only
            // the challenge, keeps the table canonical and writes the natural
            // pair order the Cube message kernel also reads.
            if (!sc::launch_sumcheck_fold(stream, source, coordinate, destination,
                                          exponent, lanes))
                return fail(0);
        }
    }

    const uint64_t rest = sc::sumcheck_rest_count(num_vars, round_index);
    const uint64_t tensor_tiles =
        (rest + sc::sumcheck_tensor_tile - 1) / sc::sumcheck_tensor_tile;
    const uint32_t blocks = state->tensor_d2
        ? static_cast<uint32_t>(tensor_tiles < sc::sumcheck_tensor_max_groups
                                   ? tensor_tiles : sc::sumcheck_tensor_max_groups)
        : sc::sumcheck_message_blocks(rest, lanes, state->aiv_cores);
    if (blocks == 0 || blocks > state->max_blocks) return fail(0);

    if (state->product_count > 0) {
        const rt::BufferRef& bindings = round_index == 0
            ? state->bindings_original
            : ((round_index & 1) ? state->bindings_slot_a : state->bindings_slot_b);
        const uint8_t* binding_base = static_cast<const uint8_t*>(bindings.data());
        for (size_t product = 0; product < state->product_count; ++product) {
            const size_t begin = state->product_offsets[product];
            const size_t arity =
                state->product_offsets[product + 1] - begin;
            const uint64_t* offsets = reinterpret_cast<const uint64_t*>(
                binding_base + begin * rt::kFrBytes);
            if (state->tensor_d2) {
                // Natural layout in every round: the Vector fold output and
                // the retained originals both hold sequential pair order.
                if (!sc::launch_sumcheck_tensor_message_d2(
                        stream, offsets, rest,
                        reinterpret_cast<hp::Fr*>(state->partial.data()),
                        static_cast<uint32_t>(product), blocks,
                        state->tensor_workspace.data()))
                    return fail(0);
                continue;
            }
            for (uint32_t pass = 0; pass < state->shape.chunks; ++pass) {
                const uint64_t t_begin64 = static_cast<uint64_t>(pass) * state->shape.chunk;
                const uint32_t t_begin = static_cast<uint32_t>(t_begin64);
                const uint64_t remaining = static_cast<uint64_t>(grid) - t_begin64;
                const uint32_t t_count = static_cast<uint32_t>(
                    remaining < state->shape.chunk ? remaining : state->shape.chunk);
                const hp::Fr* weight = t_begin == 0
                    ? nullptr
                    : reinterpret_cast<const hp::Fr*>(
                          static_cast<const uint8_t*>(state->weights.data()) +
                          t_begin * rt::kFrBytes);
                if (!sc::launch_sumcheck_message(
                        stream, offsets, static_cast<uint32_t>(arity),
                        rest, t_begin, t_count,
                        reinterpret_cast<hp::Fr*>(state->partial.data()),
                        static_cast<uint32_t>(product), degree, weight, blocks, lanes))
                    return fail(0);
            }
        }
    }

    if (!sc::launch_sumcheck_reduce(
            stream, reinterpret_cast<const hp::Fr*>(state->partial.data()),
            reinterpret_cast<const hp::Fr*>(state->coefficients.data()),
            reinterpret_cast<hp::Fr*>(state->output.data()),
            static_cast<uint32_t>(state->product_count), degree, blocks, lanes))
        return fail(0);

    status = cref->copy_to_host_async(state->output_host.data(), state->output.data(),
                                      packet_bytes);
    if (!rt::status_ok(status)) return fail(status.native_code);
    status = cref->synchronize();
    if (!rt::status_ok(status)) return fail(status.native_code);

    if (!rt::canonical_fr_array(static_cast<const uint8_t*>(state->output_host.data()),
                                grid))
        return fail(static_cast<int32_t>(ACL_ERROR_INTERNAL_ERROR));

    std::memcpy(out_evaluations, state->output_host.data(), packet_bytes);
    state->rounds_done = round_index + 1;
    return rt::status_success();
}

// Retires every allocation this state created (never the Rust-owned factor
// handles). Retirement only drops registry ownership; the retained BufferRefs
// let the shared deferred-free queue keep storage usable until it observes real
// stream completion.
void retire_state_buffers(rt::Context& context, NativeSumcheck& state) noexcept {
    const auto retire = [&context](rt::BufferRef& ref) {
        if (ref.valid()) (void)context.retire(ref.data());
    };
    retire(state.arena);
    retire(state.bindings_original);
    retire(state.bindings_slot_a);
    retire(state.bindings_slot_b);
    retire(state.coefficient_arity);
    retire(state.tensor_workspace);
    retire(state.coefficients);
    retire(state.weights);
    retire(state.partial);
    retire(state.output);
    retire(state.coordinate);
}

Ascend910B3Status destroy_state(Ascend910B3NativeContext* raw_context,
                                Ascend910B3NativeSumCheckProverState* raw_state) noexcept {
    if (raw_state == nullptr) return invalid_parameters();

    rt::ContextRef cref;
    Ascend910B3Status status = rt::acquire_context(raw_context, cref);
    if (!rt::status_ok(status)) return status;

    StatePtr state;
    {
        std::lock_guard<std::mutex> lock(state_mutex());
        StateMap& map = live_states();
        const auto it = map.find(raw_state);
        if (it == map.end()) return invalid_parameters();
        if (!it->second || !it->second->context) return invalid_parameters();
        if (&*it->second->context != &*cref) return invalid_parameters();
        state = std::move(it->second);
        map.erase(it);
    }

    // Drain once. Only a proven-complete stream lets the pinned slots die
    // immediately; otherwise they go to the context (or, as the last resort,
    // are intentionally leaked) so queued work can never read freed host
    // memory. A poisoned/closed context is not an error here: the failure was
    // already reported by the call that caused it, and destruction must still
    // release the state safely.
    Ascend910B3Status drain = rt::status_invalid_state();
    if (!cref->closed()) {
        const rt::Context::Scope scope = cref->scope();
        if (scope.ok()) {
            drain = cref->synchronize();
        } else {
            drain = scope.status();
            // Preserve the real ACL bind failure as the context's first
            // diagnostic. A zero native code means no ACL call failed (closed
            // or unavailable context), so no error code is invented.
            if (scope.status().native_code != 0)
                cref->poison(static_cast<aclError>(scope.status().native_code));
        }
    }
    if (!rt::status_ok(drain)) {
        handoff_staging(*cref, std::move(state->coordinate_host));
        handoff_staging(*cref, std::move(state->output_host));
    }
    retire_state_buffers(*cref, *state);
    state.reset();
    delete raw_state;
    return rt::status_success();
}

} // namespace

extern "C" Ascend910B3Status hp_ascend910b3_v1_sumcheck_prover_state_create(
    Ascend910B3NativeContext* ctx, size_t num_vars, size_t max_degree,
    const Ascend910B3DeviceAddress* factors, size_t factor_count,
    const uint8_t* coefficients, const size_t* product_offsets, size_t product_count,
    const size_t* factor_indices, size_t index_count,
    Ascend910B3NativeSumCheckProverState** out) {
    if (out != nullptr) *out = nullptr;
    return rt::guarded([&]() -> Ascend910B3Status {
        return create_state(ctx, num_vars, max_degree,
                            reinterpret_cast<const rt::DeviceAddress*>(factors),
                            factor_count, coefficients, product_offsets, product_count,
                            factor_indices, index_count, out);
    });
}

extern "C" Ascend910B3Status hp_ascend910b3_v1_sumcheck_prover_state_round(
    Ascend910B3NativeContext* ctx, Ascend910B3NativeSumCheckProverState* state,
    const uint8_t* challenge_or_null, uint8_t* out_evaluations,
    size_t evaluation_count) {
    return rt::guarded([&]() -> Ascend910B3Status {
        return round_state(ctx, state, challenge_or_null, out_evaluations,
                           evaluation_count);
    });
}

extern "C" Ascend910B3Status hp_ascend910b3_v1_sumcheck_prover_state_destroy(
    Ascend910B3NativeContext* ctx, Ascend910B3NativeSumCheckProverState* state) {
    return rt::guarded([&]() -> Ascend910B3Status {
        return destroy_state(ctx, state);
    });
}
