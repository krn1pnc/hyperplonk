// Multilinear KZG prover bridge over the accepted unsigned native MSM.
//
// Ownership: prepared params uniquely own their layer plans, immutable
// Montgomery SRS caches (registered, so physical free waits for stream
// completion), one reusable MSM scratch/pinned pair plus output slot sized by
// maximum layer, and one reusable open workspace (half- and quarter-size
// folded-state tables, one quotient table, point upload and host packet
// staging; the first KZG fold reads the resident MLE directly). Caches are
// built once by a cold upload; commit/open never re-upload or convert SRS
// points and no whole-MLE device copy is made.
//
// Dataflow: scalar inputs are the resident canonical-Fr MLE handles (no table
// round trip); fold quotients stay in HBM as the next MSM's scalars; only one
// 96B point per quotient MSM and the final 32B evaluation reach the host.
//
// SRS curve and subgroup membership are caller preconditions, established by
// checked deserialization or generation before preparation. The cold upload
// here only enforces canonical 97-byte packet shape: tag 0/1, infinity with zero
// coordinates, finite values < q, and no finite (0,0). Commit/open results are
// re-validated on the host: canonical coordinates, infinity consistency and the
// standalone is_on_curve() gate (no CPU field fallback or numerical reference).

#include "bridge.h"
#include "bridge_runtime.h"
#include "mle_kernels.h"
#include "msm.h"
#include "msm_plan.h"

#include "field.h"
#include "g1.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace rt = hyperplonk::ascend910b3::native;
namespace hp = hyperplonk::ascend910b3;
namespace msm = hyperplonk::ascend910b3::detail;

namespace {

static_assert(sizeof(hp::Fr) == rt::kFrBytes, "canonical Fr is 32 bytes");
static_assert(sizeof(hp::G1Affine) == rt::kG1AffineBytes, "native G1Affine is 96 bytes");
static_assert(rt::kCommitmentPacketBytes == 1 + 2 * 48, "G1 packet is tag+x+y");

// General (non-MLE) registered buffers carry no dimension.
constexpr size_t kGeneralNumVars = std::numeric_limits<size_t>::max();

// ---------------------------------------------------------------------------
// Params model.
// ---------------------------------------------------------------------------

struct KzgLayer {
    msm::MsmPlan plan{};
    msm::MsmPreparedSizes sizes{};
    rt::BufferRef cache;
    std::optional<msm::PreparedSrsView> view;
};

struct KzgPrepared {
    rt::ContextRef context;
    size_t num_vars = 0;
    std::vector<KzgLayer> layers;
    // Reusable MSM execution slots, sized once for the largest layer.
    rt::BufferRef scratch;
    rt::BufferRef output;
    rt::PinnedStorage msm_host;
    // Reusable open workspace: state_a holds 2^(N-1) Fr (first/even folds),
    // state_b holds 2^(N-2) Fr (odd folds, absent for N < 2), quotient holds
    // 2^(N-1) Fr. The first fold reads the immutable resident MLE directly.
    rt::BufferRef state_a;
    rt::BufferRef state_b;
    rt::BufferRef quotient;
    rt::BufferRef point;
    rt::PinnedStorage point_host;
    rt::PinnedStorage quotient_host;
    // Host packet staging for synchronized result returns.
    rt::PinnedStorage result_host;
    rt::PinnedStorage evaluation_host;
};

} // namespace

// The C ABI names this tag; only this translation unit makes it concrete.
struct Ascend910B3NativeMultilinearKzgPreparedProverParam {
    KzgPrepared kzg;
};

namespace {

// ---------------------------------------------------------------------------
// Host packet helpers (control metadata only).
// ---------------------------------------------------------------------------

// Canonical 97-byte SRS packet -> native canonical G1Affine. Curve/subgroup was
// already validated once by the Rust ingress owner; shape violations are
// InvalidParameters.
bool parse_srs_packet(const uint8_t* packet, hp::G1Affine& out) {
    const uint8_t tag = packet[0];
    bool zero_coordinates = true;
    for (size_t i = 1; i < rt::kCommitmentPacketBytes; ++i) {
        if (packet[i] != 0) {
            zero_coordinates = false;
            break;
        }
    }
    if (tag == 1) {
        if (!zero_coordinates) return false;
        out = hp::G1Affine::identity();
        return true;
    }
    if (tag != 0) return false;
    hp::Fq x{};
    hp::Fq y{};
    if (!hp::Fq::from_le_bytes(packet + 1, x) || !hp::Fq::from_le_bytes(packet + 49, y)) {
        return false;
    }
    // A finite point must not be the (0,0) representation reserved for infinity.
    if (x.is_zero() && y.is_zero()) return false;
    out.x = x;
    out.y = y;
    return true;
}

// Device native G1Affine -> strict 97-byte packet. Rejects noncanonical limbs,
// inconsistent infinity and points off the curve; never repairs the value.
bool encode_result_packet(const hp::G1Affine& raw, uint8_t* out) {
    uint8_t x_bytes[48];
    uint8_t y_bytes[48];
    raw.x.to_le_bytes(x_bytes);
    raw.y.to_le_bytes(y_bytes);
    hp::Fq x{};
    hp::Fq y{};
    if (!hp::Fq::from_le_bytes(x_bytes, x) || !hp::Fq::from_le_bytes(y_bytes, y)) {
        return false;
    }
    if (x.is_zero() && y.is_zero()) {
        std::memset(out, 0, rt::kCommitmentPacketBytes);
        out[0] = 1;
        return true;
    }
    hp::G1Affine point{};
    point.x = x;
    point.y = y;
    if (!point.is_on_curve()) return false;
    out[0] = 0;
    std::memcpy(out + 1, x_bytes, 48);
    std::memcpy(out + 49, y_bytes, 48);
    return true;
}

// ---------------------------------------------------------------------------
// Runtime helpers.
// ---------------------------------------------------------------------------

Ascend910B3Status bind_context(Ascend910B3NativeContext* raw, rt::ContextRef& cref,
                               rt::Context*& ctx) {
    ctx = nullptr;
    if (!raw) return rt::status_invalid_parameters();
    auto status = rt::acquire_context(raw, cref);
    if (!rt::status_ok(status)) return status;
    ctx = &*cref;
    if (ctx->closed()) return rt::status_invalid_state();
    if (ctx->poisoned()) {
        return rt::status_invalid_state(static_cast<int32_t>(ctx->first_error()));
    }
    return rt::status_success();
}

Ascend910B3Status bind_params(Ascend910B3NativeMultilinearKzgPreparedProverParam* params,
                             rt::Context& ctx, KzgPrepared*& out) {
    out = nullptr;
    if (!params) return rt::status_invalid_parameters();
    if (!params->kzg.context) return rt::status_invalid_state();
    if (&*params->kzg.context != &ctx) return rt::status_invalid_parameters();
    out = &params->kzg;
    return rt::status_success();
}

// Allocates and registers one general device buffer with explicit alignment.
Ascend910B3Status device_buffer(rt::Context& ctx, size_t bytes, size_t alignment,
                                rt::BufferRef& out) {
    out.reset();
    if (!bytes) return rt::status_success();
    rt::DeviceStorage storage;
    auto status = ctx.allocate(bytes, alignment, storage);
    if (!rt::status_ok(status)) return status;
    return ctx.register_buffer(std::move(storage), kGeneralNumVars, rt::BufferKind::General, out);
}

// Resolves an immutable MLE handle inside this context with the exact dimension
// byte size. Retained BufferRef keeps the table alive for the whole call.
bool lookup_polynomial(rt::Context& ctx, rt::DeviceAddress address, size_t num_vars,
                       rt::BufferRef& out) {
    out.reset();
    if (!address) return false;
    size_t bytes = 0;
    if (!rt::dimension_bytes(num_vars, rt::kFrBytes, bytes)) return false;
    rt::BufferRef ref;
    if (!ctx.lookup(address, num_vars, ref)) return false;
    if (!ref.valid() || ref.kind() != rt::BufferKind::MleFr) return false;
    if (ref.num_vars() != num_vars || ref.bytes() != bytes) return false;
    out = std::move(ref);
    return true;
}

// Releases every registered buffer and pinned staging owned by one params
// object. Registered blocks become completion-aware retirement entries; the
// caller must have synchronized first when any stream work may be outstanding.
void retire_params(rt::Context& ctx, KzgPrepared& p) noexcept {
    for (KzgLayer& layer : p.layers) {
        if (layer.cache.valid()) {
            (void)ctx.retire(layer.cache.data());
            layer.cache.reset();
        }
        layer.view.reset();
    }
    rt::BufferRef* refs[] = {&p.scratch,  &p.output, &p.state_a,
                             &p.state_b,  &p.quotient, &p.point};
    for (rt::BufferRef* ref : refs) {
        if (ref->valid()) {
            (void)ctx.retire(ref->data());
            ref->reset();
        }
    }
    rt::PinnedStorage* pinned[] = {&p.msm_host, &p.point_host, &p.quotient_host,
                                   &p.result_host, &p.evaluation_host};
    for (rt::PinnedStorage* storage : pinned) {
        if (storage->valid()) (void)storage->release();
    }
}

// Scope guard for a params object that never became visible to the caller.
struct ParamsCleanup {
    rt::Context* ctx = nullptr;
    KzgPrepared* params = nullptr;
    bool armed = false;
    ~ParamsCleanup() {
        if (!armed || ctx == nullptr || params == nullptr) return;
        retire_params(*ctx, *params);
        (void)ctx->poll_retired();
    }
};

} // namespace

extern "C" {

Ascend910B3Status hp_ascend910b3_v1_multilinear_kzg_params_create(
    Ascend910B3NativeContext* raw, size_t num_vars,
    Ascend910B3NativeMultilinearKzgPreparedProverParam** out) {
    if (out) *out = nullptr;
    return rt::guarded([&]() -> Ascend910B3Status {
        if (!out) return rt::status_invalid_parameters();
        // Layer zero holds 2^num_vars points; the MSM plan caps counts at u32.
        if (num_vars > 31) return rt::status_invalid_parameters();
        rt::ContextRef cref;
        rt::Context* ctx = nullptr;
        auto status = bind_context(raw, cref, ctx);
        if (!rt::status_ok(status)) return status;
        auto scope = ctx->scope();
        if (!scope.ok()) return scope.status();

        auto params = std::make_unique<Ascend910B3NativeMultilinearKzgPreparedProverParam>();
        KzgPrepared& p = params->kzg;
        p.context = cref;
        p.num_vars = num_vars;
        p.layers.resize(num_vars + 1);
        ParamsCleanup cleanup{ctx, &p, true};

        size_t max_scratch = 0;
        size_t max_host_workspace = 0;
        size_t count = size_t{1} << num_vars;
        for (size_t i = 0; i <= num_vars; ++i, count >>= 1) {
            KzgLayer& layer = p.layers[i];
            if (!msm::build_msm_plan(count, layer.plan)) return rt::status_invalid_state();
            msm::MsmPreparedSizes sizes{};
            if (!msm::query_msm_prepared_sizes(layer.plan, sizes)) {
                return rt::status_invalid_state();
            }
            layer.sizes = sizes;
            max_scratch = std::max(max_scratch, static_cast<size_t>(sizes.scratch_bytes));
            max_host_workspace =
                std::max(max_host_workspace, static_cast<size_t>(sizes.host_workspace_bytes));
        }

        size_t state_a_bytes = 0;
        size_t state_b_bytes = 0;
        size_t quotient_bytes = 0;
        size_t point_bytes = 0;
        size_t quotient_host_bytes = 0;
        // One-shot open workspace for the largest supported dimension. The first
        // fold reads the resident MLE directly; even rounds fold into the
        // half-size A table (2^(N-1) elements), odd rounds into the quarter-size
        // B table (2^(N-2)). Tables alternate and every later round is smaller,
        // so neither is written beyond its capacity. Quotient keeps the same
        // 2^(N-1) maximum. N < 2 has no valid smaller table and requests none.
        // Nothing here is allocated per commit or per open.
        if (!rt::checked_mul(num_vars, rt::kFrBytes, point_bytes) ||
            !rt::checked_mul(num_vars, rt::kCommitmentPacketBytes, quotient_host_bytes)) {
            return rt::status_invalid_parameters();
        }
        if (num_vars != 0 &&
            !rt::dimension_bytes(num_vars - 1, rt::kFrBytes, state_a_bytes)) {
            return rt::status_invalid_parameters();
        }
        if (num_vars >= 2 &&
            !rt::dimension_bytes(num_vars - 2, rt::kFrBytes, state_b_bytes)) {
            return rt::status_invalid_parameters();
        }
        quotient_bytes = state_a_bytes;

        status = device_buffer(*ctx, max_scratch, rt::kScratchAlignment, p.scratch);
        if (!rt::status_ok(status)) return status;
        status = device_buffer(*ctx, rt::kG1AffineBytes, rt::kScratchAlignment, p.output);
        if (!rt::status_ok(status)) return status;
        if (num_vars != 0) {
            status = device_buffer(*ctx, state_a_bytes, rt::kScratchAlignment, p.state_a);
            if (!rt::status_ok(status)) return status;
            status = device_buffer(*ctx, quotient_bytes, rt::kScratchAlignment, p.quotient);
            if (!rt::status_ok(status)) return status;
            status = device_buffer(*ctx, point_bytes, rt::kScratchAlignment, p.point);
            if (!rt::status_ok(status)) return status;
        }
        if (num_vars >= 2) {
            status = device_buffer(*ctx, state_b_bytes, rt::kScratchAlignment, p.state_b);
            if (!rt::status_ok(status)) return status;
        }
        status = ctx->allocate_pinned(max_host_workspace, rt::kPinnedAlignment, p.msm_host);
        if (!rt::status_ok(status)) return status;
        status = ctx->allocate_pinned(point_bytes, rt::kPinnedAlignment, p.point_host);
        if (!rt::status_ok(status)) return status;
        status = ctx->allocate_pinned(quotient_host_bytes, rt::kPinnedAlignment, p.quotient_host);
        if (!rt::status_ok(status)) return status;
        status = ctx->allocate_pinned(rt::kG1AffineBytes, rt::kPinnedAlignment, p.result_host);
        if (!rt::status_ok(status)) return status;
        status = ctx->allocate_pinned(rt::kFrBytes, rt::kPinnedAlignment, p.evaluation_host);
        if (!rt::status_ok(status)) return status;

        *out = params.release();
        cleanup.armed = false;
        return rt::status_success();
    });
}

Ascend910B3Status hp_ascend910b3_v1_multilinear_kzg_params_upload_layer(
    Ascend910B3NativeContext* raw, Ascend910B3NativeMultilinearKzgPreparedProverParam* params_ptr,
    size_t layer_index, const uint8_t* points, size_t point_count) {
    return rt::guarded([&]() -> Ascend910B3Status {
        rt::ContextRef cref;
        rt::Context* ctx = nullptr;
        auto status = bind_context(raw, cref, ctx);
        if (!rt::status_ok(status)) return status;
        KzgPrepared* p = nullptr;
        status = bind_params(params_ptr, *ctx, p);
        if (!rt::status_ok(status)) return status;
        auto scope = ctx->scope();
        if (!scope.ok()) return scope.status();
        if (layer_index > p->num_vars) return rt::status_invalid_parameters();
        const size_t expected = size_t{1} << (p->num_vars - layer_index);
        if (point_count != expected) return rt::status_invalid_parameters();
        if (!points) return rt::status_invalid_parameters();
        KzgLayer& layer = p->layers[layer_index];
        // Cold prepare happens once: a published view means this layer is ready.
        if (layer.view) return rt::status_invalid_state();

        size_t packet_bytes = 0;
        if (!rt::checked_mul(point_count, rt::kG1AffineBytes, packet_bytes)) {
            return rt::status_invalid_parameters();
        }
        const msm::MsmPreparedSizes& sizes = layer.sizes;
        if (static_cast<size_t>(sizes.reusable_bytes) != packet_bytes) {
            return rt::status_invalid_state();
        }

        // Validate every 97-byte packet and strip the tag into canonical bases.
        auto staging = std::make_unique<rt::PinnedStorage>();
        status = ctx->allocate_pinned(packet_bytes, rt::kPinnedAlignment, *staging);
        if (!rt::status_ok(status)) return status;
        auto* canonical = static_cast<uint8_t*>(staging->data());
        for (size_t i = 0; i < point_count; ++i) {
            hp::G1Affine point{};
            if (!parse_srs_packet(points + i * rt::kCommitmentPacketBytes, point)) {
                return rt::status_invalid_parameters();
            }
            std::memcpy(canonical + i * rt::kG1AffineBytes, &point, sizeof(point));
        }

        // Persistent immutable Montgomery cache for this layer.
        if (!layer.cache.valid()) {
            status = device_buffer(*ctx, static_cast<size_t>(sizes.reusable_bytes),
                                   rt::kScratchAlignment, layer.cache);
            if (!rt::status_ok(status)) return status;
        }

        auto bases = std::make_unique<rt::DeviceStorage>();
        status = ctx->allocate(packet_bytes, rt::kScratchAlignment, *bases);
        if (!rt::status_ok(status)) return status;
        status = ctx->copy_to_device_async(bases->data(), staging->data(), packet_bytes);
        if (!rt::status_ok(status)) return status;

        std::optional<msm::PreparedSrsView> view;
        aclError error = ACL_SUCCESS;
        try {
            error = msm::prepare_msm_srs(
                ctx->stream(), static_cast<const hp::G1Affine*>(bases->data()), packet_bytes,
                layer.cache.data(), layer.cache.bytes(), layer.plan, view);
        } catch (...) {
            // Unexpected host exception after the H2D was enqueued: drain before
            // the temporary staging allocations leave scope.
            const auto drained = ctx->synchronize();
            if (!rt::status_ok(drained)) {
                (void)bases.release();
                (void)staging.release();
            }
            return rt::status_invalid_state(ACL_ERROR_INTERNAL_ERROR);
        }
        if (error != ACL_SUCCESS) {
            // The failed prepare may have left work reading the staging buffers;
            // drain before freeing. A failed drain poisons the context, so the
            // temporary allocations are intentionally retained rather than freed
            // while a device transfer may still reference them.
            const auto drained = ctx->synchronize();
            if (!rt::status_ok(drained)) {
                (void)bases.release();
                (void)staging.release();
            }
            return rt::status_from_acl(error);
        }
        if (!view) return rt::status_invalid_state();
        layer.view = std::move(view);
        // prepare_msm_srs synchronized the stream: the temporary canonical bases
        // and host staging are no longer referenced.
        bases.reset();
        staging.reset();
        return rt::status_success();
    });
}

Ascend910B3Status hp_ascend910b3_v1_multilinear_kzg_params_destroy(
    Ascend910B3NativeContext* raw, Ascend910B3NativeMultilinearKzgPreparedProverParam* params_ptr) {
    return rt::guarded([&]() -> Ascend910B3Status {
        rt::ContextRef cref;
        rt::Context* ctx = nullptr;
        auto status = bind_context(raw, cref, ctx);
        if (!rt::status_ok(status)) return status;
        KzgPrepared* p = nullptr;
        status = bind_params(params_ptr, *ctx, p);
        if (!rt::status_ok(status)) return status;
        auto scope = ctx->scope();
        if (!scope.ok()) return scope.status();

        // Drain first; only a completed stream allows immediate pinned/device
        // release. Registered buffers additionally defer physical free until
        // completion is observed, so the destroy path never frees live storage.
        status = ctx->synchronize();
        if (!rt::status_ok(status)) {
            // Keep the params object and its pinned staging alive; the context
            // owner retries the drain during its own destruction.
            return status;
        }
        retire_params(*ctx, *p);
        delete params_ptr;
        // The stream was synchronized above; retirement is reclaimed now when
        // complete, otherwise the context owner's final drain frees it. A failed
        // reclamation attempt here must not turn a completed drain into an error.
        (void)ctx->poll_retired();
        return rt::status_success();
    });
}

Ascend910B3Status hp_ascend910b3_v1_multilinear_kzg_commit(
    Ascend910B3NativeContext* raw, const Ascend910B3NativeMultilinearKzgPreparedProverParam* params,
    Ascend910B3DeviceAddress polynomial, size_t num_vars, uint8_t* out_commitment) {
    return rt::guarded([&]() -> Ascend910B3Status {
        if (!out_commitment) return rt::status_invalid_parameters();
        rt::ContextRef cref;
        rt::Context* ctx = nullptr;
        auto status = bind_context(raw, cref, ctx);
        if (!rt::status_ok(status)) return status;
        KzgPrepared* p = nullptr;
        status = bind_params(const_cast<Ascend910B3NativeMultilinearKzgPreparedProverParam*>(params),
                             *ctx, p);
        if (!rt::status_ok(status)) return status;
        auto scope = ctx->scope();
        if (!scope.ok()) return scope.status();
        if (num_vars > p->num_vars) return rt::status_invalid_parameters();
        rt::BufferRef polyn;
        if (!lookup_polynomial(*ctx, polynomial, num_vars, polyn)) {
            return rt::status_invalid_parameters();
        }
        // Commit of an nv-variable polynomial uses layer N - nv.
        KzgLayer& layer = p->layers[p->num_vars - num_vars];
        if (!layer.view) return rt::status_invalid_state();
        const size_t count = size_t{1} << num_vars;
        const aclError error = msm::run_msm_prepared(
            ctx->stream(), *layer.view, reinterpret_cast<const hp::Fr*>(polyn.data()), count,
            reinterpret_cast<hp::G1Affine*>(p->output.data()), p->scratch.data(),
            p->scratch.bytes(), p->msm_host.data(), p->msm_host.size(), layer.plan);
        if (error != ACL_SUCCESS) return rt::status_from_acl(error);
        status = ctx->copy_to_host_async(p->result_host.data(), p->output.data(),
                                         rt::kG1AffineBytes);
        if (!rt::status_ok(status)) return status;
        status = ctx->synchronize();
        if (!rt::status_ok(status)) return status;
        hp::G1Affine host{};
        std::memcpy(&host, p->result_host.data(), sizeof(host));
        if (!encode_result_packet(host, out_commitment)) return rt::status_invalid_state();
        return rt::status_success();
    });
}

Ascend910B3Status hp_ascend910b3_v1_multilinear_kzg_multi_commit(
    Ascend910B3NativeContext* raw, const Ascend910B3NativeMultilinearKzgPreparedProverParam* params,
    const Ascend910B3DeviceAddress* polynomials, const size_t* num_vars, size_t count,
    uint8_t* out_commitments) {
    return rt::guarded([&]() -> Ascend910B3Status {
        rt::ContextRef cref;
        rt::Context* ctx = nullptr;
        auto status = bind_context(raw, cref, ctx);
        if (!rt::status_ok(status)) return status;
        KzgPrepared* p = nullptr;
        status = bind_params(const_cast<Ascend910B3NativeMultilinearKzgPreparedProverParam*>(params),
                             *ctx, p);
        if (!rt::status_ok(status)) return status;
        if (count == 0) return rt::status_success();
        if (!polynomials || !num_vars || !out_commitments) {
            return rt::status_invalid_parameters();
        }
        auto scope = ctx->scope();
        if (!scope.ok()) return scope.status();
        size_t total_bytes = 0;
        if (!rt::checked_mul(count, rt::kCommitmentPacketBytes, total_bytes)) {
            return rt::status_invalid_parameters();
        }
        // Validate the whole batch (dimensions, handles, prepared layers) before
        // any device side effect, preserving order and duplicates.
        std::vector<rt::BufferRef> tables;
        tables.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const size_t nv = num_vars[i];
            if (nv > p->num_vars) return rt::status_invalid_parameters();
            rt::BufferRef table;
            if (!lookup_polynomial(*ctx, polynomials[i], nv, table)) {
                return rt::status_invalid_parameters();
            }
            if (!p->layers[p->num_vars - nv].view) return rt::status_invalid_state();
            tables.push_back(std::move(table));
        }
        std::vector<uint8_t> packets(total_bytes, 0);
        for (size_t i = 0; i < count; ++i) {
            const size_t nv = num_vars[i];
            KzgLayer& layer = p->layers[p->num_vars - nv];
            const size_t elements = size_t{1} << nv;
            const aclError error = msm::run_msm_prepared(
                ctx->stream(), *layer.view,
                reinterpret_cast<const hp::Fr*>(tables[i].data()), elements,
                reinterpret_cast<hp::G1Affine*>(p->output.data()), p->scratch.data(),
                p->scratch.bytes(), p->msm_host.data(), p->msm_host.size(), layer.plan);
            if (error != ACL_SUCCESS) return rt::status_from_acl(error);
            status = ctx->copy_to_host_async(p->result_host.data(), p->output.data(),
                                             rt::kG1AffineBytes);
            if (!rt::status_ok(status)) return status;
            status = ctx->synchronize();
            if (!rt::status_ok(status)) return status;
            hp::G1Affine host{};
            std::memcpy(&host, p->result_host.data(), sizeof(host));
            if (!encode_result_packet(host, packets.data() + i * rt::kCommitmentPacketBytes)) {
                return rt::status_invalid_state();
            }
        }
        std::memcpy(out_commitments, packets.data(), total_bytes);
        return rt::status_success();
    });
}

Ascend910B3Status hp_ascend910b3_v1_multilinear_kzg_open(
    Ascend910B3NativeContext* raw, const Ascend910B3NativeMultilinearKzgPreparedProverParam* params,
    Ascend910B3DeviceAddress polynomial, size_t num_vars, const uint8_t* point,
    uint8_t* out_quotients, uint8_t* out_evaluation) {
    return rt::guarded([&]() -> Ascend910B3Status {
        if (!out_evaluation) return rt::status_invalid_parameters();
        rt::ContextRef cref;
        rt::Context* ctx = nullptr;
        auto status = bind_context(raw, cref, ctx);
        if (!rt::status_ok(status)) return status;
        KzgPrepared* p = nullptr;
        status = bind_params(const_cast<Ascend910B3NativeMultilinearKzgPreparedProverParam*>(params),
                             *ctx, p);
        if (!rt::status_ok(status)) return status;
        auto scope = ctx->scope();
        if (!scope.ok()) return scope.status();
        if (num_vars > p->num_vars) return rt::status_invalid_parameters();
        if (num_vars != 0) {
            if (!out_quotients || !point) return rt::status_invalid_parameters();
            if (!rt::canonical_fr_array(point, num_vars)) {
                return rt::status_invalid_parameters();
            }
        }
        rt::BufferRef polyn;
        if (!lookup_polynomial(*ctx, polynomial, num_vars, polyn)) {
            return rt::status_invalid_parameters();
        }

        // A constant polynomial opens to its single resident evaluation.
        if (num_vars == 0) {
            status = ctx->copy_to_host_async(p->evaluation_host.data(), polyn.data(),
                                             rt::kFrBytes);
            if (!rt::status_ok(status)) return status;
            status = ctx->synchronize();
            if (!rt::status_ok(status)) return status;
            if (!rt::canonical_fr(static_cast<const uint8_t*>(p->evaluation_host.data()))) {
                return rt::status_invalid_state();
            }
            std::memcpy(out_evaluation, p->evaluation_host.data(), rt::kFrBytes);
            return rt::status_success();
        }

        // Quotient j of an nv-variable opening uses SRS layer N - nv + 1 + j.
        // Validate every required layer before any device side effect.
        const size_t ignored = p->num_vars - num_vars + 1;
        for (size_t j = 0; j < num_vars; ++j) {
            if (!p->layers[ignored + j].view) return rt::status_invalid_state();
        }

        size_t point_upload_bytes = 0;
        size_t quotient_packet_bytes = 0;
        if (!rt::checked_mul(num_vars, rt::kFrBytes, point_upload_bytes) ||
            !rt::checked_mul(num_vars, rt::kCommitmentPacketBytes, quotient_packet_bytes)) {
            return rt::status_invalid_parameters();
        }

        std::memcpy(p->point_host.data(), point, point_upload_bytes);
        status = ctx->copy_to_device_async(p->point.data(), p->point_host.data(),
                                           point_upload_bytes);
        if (!rt::status_ok(status)) return status;

        // The resident MLE is immutable and feeds the first fold directly; no
        // whole-table copy. Even rounds write the half-size A table, odd rounds
        // the quarter-size B table, and they alternate; each later round is
        // smaller, so neither table is written beyond its allocated capacity.
        const hp::Fr* input = reinterpret_cast<const hp::Fr*>(polyn.data());
        uint8_t* last = nullptr;
        for (size_t j = 0; j < num_vars; ++j) {
            const size_t round_vars = num_vars - j;
            const auto coordinate =
                reinterpret_cast<const hp::Fr*>(p->point.data()) + j;
            uint8_t* folded_into = (j % 2 == 0)
                ? static_cast<uint8_t*>(p->state_a.data())
                : static_cast<uint8_t*>(p->state_b.data());
            if (folded_into == nullptr) return rt::status_invalid_state();
            const bool folded = msm::launch_mle_fold(
                ctx->stream(), input, round_vars, coordinate,
                reinterpret_cast<hp::Fr*>(folded_into),
                reinterpret_cast<hp::Fr*>(p->quotient.data()));
            if (!folded) return rt::status_invalid_state();
            KzgLayer& layer = p->layers[ignored + j];
            const size_t elements = size_t{1} << (round_vars - 1);
            const aclError msm_error = msm::run_msm_prepared(
                ctx->stream(), *layer.view,
                reinterpret_cast<const hp::Fr*>(p->quotient.data()), elements,
                reinterpret_cast<hp::G1Affine*>(p->output.data()), p->scratch.data(),
                p->scratch.bytes(), p->msm_host.data(), p->msm_host.size(), layer.plan);
            if (msm_error != ACL_SUCCESS) return rt::status_from_acl(msm_error);
            status = ctx->copy_to_host_async(
                static_cast<uint8_t*>(p->quotient_host.data()) + j * rt::kCommitmentPacketBytes,
                p->output.data(), rt::kG1AffineBytes);
            if (!rt::status_ok(status)) return status;
            input = reinterpret_cast<const hp::Fr*>(folded_into);
            last = folded_into;
        }
        // Only the final folded scalar (32B) is downloaded.
        status = ctx->copy_to_host_async(p->evaluation_host.data(), last, rt::kFrBytes);
        if (!rt::status_ok(status)) return status;
        status = ctx->synchronize();
        if (!rt::status_ok(status)) return status;

        std::vector<uint8_t> packets(quotient_packet_bytes, 0);
        for (size_t j = 0; j < num_vars; ++j) {
            hp::G1Affine host{};
            std::memcpy(&host,
                        static_cast<const uint8_t*>(p->quotient_host.data()) +
                            j * rt::kCommitmentPacketBytes,
                        sizeof(host));
            if (!encode_result_packet(host, packets.data() + j * rt::kCommitmentPacketBytes)) {
                return rt::status_invalid_state();
            }
        }
        if (!rt::canonical_fr(static_cast<const uint8_t*>(p->evaluation_host.data()))) {
            return rt::status_invalid_state();
        }
        std::memcpy(out_quotients, packets.data(), quotient_packet_bytes);
        std::memcpy(out_evaluation, p->evaluation_host.data(), rt::kFrBytes);
        return rt::status_success();
    });
}

} // extern "C"
