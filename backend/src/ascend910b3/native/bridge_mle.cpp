// MLE bridge entries: the seven resident MLE ABI symbols not owned by
// bridge.asc (upload stays with the runtime owner).
//
// Every entry validates its context, dimensions, handle ownership and counts
// before touching the stream, rejects non-canonical host packets, uploads only
// small control metadata, and never reads a whole MLE table back to the host.
// Outputs are unique registered allocations whose public handle is exactly the
// device base address; failures clear every resource output slot.

#include "mle_kernels.h"

#include "bridge_runtime.h"

#include "tiling/platform/platform_ascendc.h"

#include <cstring>
#include <limits>
#include <utility>
#include <vector>

using namespace hyperplonk::ascend910b3;
using namespace hyperplonk::ascend910b3::native;

namespace hyperplonk::ascend910b3::detail {

uint32_t mle_aiv_cores() noexcept {
    static const uint32_t cores = []() noexcept -> uint32_t {
        auto* platform =
            platform_ascendc::PlatformAscendCManager::GetInstance("Ascend910B3");
        if (platform == nullptr) return 0;
        return static_cast<uint32_t>(platform->GetCoreNumAiv());
    }();
    return cores;
}

} // namespace hyperplonk::ascend910b3::detail

namespace {

using hyperplonk::ascend910b3::detail::launch_mle_build_eq_x_r;
using hyperplonk::ascend910b3::detail::launch_mle_compute_product;
using hyperplonk::ascend910b3::detail::launch_mle_evaluate;
using hyperplonk::ascend910b3::detail::launch_mle_frac;
using hyperplonk::ascend910b3::detail::launch_mle_fold;
using hyperplonk::ascend910b3::detail::launch_mle_linear_combination;
using hyperplonk::ascend910b3::detail::launch_mle_nums_and_denoms;
using hyperplonk::ascend910b3::detail::launch_mle_product_factors;
using hyperplonk::ascend910b3::detail::mle_frac_flag_blocks;

// Host-side MLE element count for a dimension (num_vars < 64, <= PTRDIFF_MAX).
bool table_layout(size_t num_vars, size_t& elements, size_t& bytes) noexcept {
    if (!dimension_bytes(num_vars, kFrBytes, bytes)) return false;
    if (bytes % kFrBytes != 0) return false;
    elements = bytes / kFrBytes;
    return true;
}

// One launch_mle_frac flag record: 48 int32 lane counters (see mle_kernels.h).
constexpr size_t kFlagRecordBytes = 192;

// Registers a unique output table and hands out its public handle.
Ascend910B3Status create_table(Context& context, size_t num_vars, size_t bytes,
                               BufferKind kind, BufferRef& ref,
                               DeviceAddress* slot) {
    Ascend910B3Status status = context.create_buffer(bytes, num_vars, kind, ref);
    if (!status_ok(status)) return status;
    if (slot != nullptr) *slot = ref.data();
    return status_success();
}

Ascend910B3Status create_fr_table(Context& context, size_t num_vars, size_t bytes,
                                  BufferRef& ref, DeviceAddress* slot) {
    return create_table(context, num_vars, bytes, BufferKind::MleFr, ref, slot);
}

// Retires helper allocations; physical release follows real completion.
void retire_all(Context& context, std::vector<BufferRef>& refs) noexcept {
    for (BufferRef& ref : refs) {
        if (ref.valid()) context.retire(ref.data());
        ref.reset();
    }
}

// Small control metadata staged through owned pinned memory. The device copy is
// registered so the kernel and the pinned staging are both kept alive until the
// stream really completes; release() removes the public handle afterwards.
class MetadataTable {
public:
    Ascend910B3Status upload(Context& context, const void* bytes, size_t size) {
        if (size == 0) return status_success();
        Ascend910B3Status status =
            context.allocate_pinned(size, kPinnedAlignment, pinned_);
        if (!status_ok(status)) return status;
        std::memcpy(pinned_.data(), bytes, size);
        status = context.create_buffer(size, 0, BufferKind::General, device_);
        if (!status_ok(status)) return status;
        status = context.copy_to_device_async(device_.data(), pinned_.data(), size);
        if (!status_ok(status)) return status;
        Ascend910B3Status retained = context.retain_pinned(std::move(pinned_));
        if (!status_ok(retained)) {
            // Even a rejected hand-off must not release staged bytes while the
            // enqueued copy can still read them.
            context.synchronize();
            if (device_.valid()) {
                context.retire(device_.data());
                device_.reset();
            }
        }
        return retained;
    }

    DeviceAddress address() const noexcept { return device_.data(); }

    void release(Context& context) noexcept {
        if (device_.valid()) {
            context.retire(device_.data());
            device_.reset();
        }
    }

private:
    PinnedStorage pinned_;
    BufferRef device_;
};

// Device uint64 address table built from validated input handles.
class AddressTable {
public:
    void push(DeviceAddress address) {
        addresses_.push_back(reinterpret_cast<uint64_t>(address));
    }

    Ascend910B3Status upload(Context& context, MetadataTable& staging) {
        return staging.upload(context, addresses_.data(),
                              addresses_.size() * sizeof(uint64_t));
    }

private:
    std::vector<uint64_t> addresses_;
};

Ascend910B3Status lookup_table(Context& context, DeviceAddress address,
                               size_t num_vars, BufferRef& ref) {
    if (!context.lookup(address, num_vars, ref)) return status_invalid_parameters();
    size_t elements = 0, bytes = 0;
    if (!table_layout(num_vars, elements, bytes)) return status_invalid_parameters();
    if (ref.bytes() != bytes) return status_invalid_parameters();
    return status_success();
}

// Validates every handle of one group and appends its addresses to `table`.
// The returned refs keep shared storage alive; registered buffers also stay
// owned by the context until retirement.
Ascend910B3Status lookup_many(Context& context, const DeviceAddress* addresses,
                              size_t num_vars, size_t count,
                              std::vector<BufferRef>& refs,
                              AddressTable& table) {
    for (size_t index = 0; index < count; ++index) {
        BufferRef ref;
        Ascend910B3Status status = lookup_table(context, addresses[index], num_vars, ref);
        if (!status_ok(status)) return status;
        table.push(ref.data());
        refs.push_back(std::move(ref));
    }
    return status_success();
}

} // namespace

// ---------------------------------------------------------------------------
// hp_ascend910b3_v1_mle_linear_combination
// ---------------------------------------------------------------------------
extern "C" Ascend910B3Status hp_ascend910b3_v1_mle_linear_combination(
    Ascend910B3NativeContext* raw_context, size_t num_vars,
    const Ascend910B3DeviceAddress* polynomials, const uint8_t* coefficients,
    size_t count, Ascend910B3DeviceAddress* out) {
    if (out != nullptr) *out = nullptr;
    return guarded([&]() -> Ascend910B3Status {
        ContextRef context_ref;
        Ascend910B3Status status = acquire_context(raw_context, context_ref);
        if (!status_ok(status)) return status;
        Context& context = *context_ref;
        Context::Scope scope = context.scope();
        if (!scope.ok()) return scope.status();
        if (context.closed()) return status_invalid_state();
        if (context.poisoned()) return status_from_acl(context.first_error());
        if (out == nullptr || polynomials == nullptr || coefficients == nullptr ||
            count == 0)
            return status_invalid_parameters();
        size_t elements = 0, bytes = 0;
        if (!table_layout(num_vars, elements, bytes)) return status_invalid_parameters();
        if (!canonical_fr_array(coefficients, count))
            return status_invalid_parameters();

        std::vector<BufferRef> inputs;
        AddressTable address_table;
        status = lookup_many(context, polynomials, num_vars, count, inputs, address_table);
        if (!status_ok(status)) return status;

        MetadataTable address_staging;
        status = address_table.upload(context, address_staging);
        if (!status_ok(status)) return status;
        MetadataTable coefficient_staging;
        status = coefficient_staging.upload(context, coefficients, count * kFrBytes);
        if (!status_ok(status)) return status;

        BufferRef result;
        status = create_fr_table(context, num_vars, bytes, result, out);
        if (!status_ok(status)) return status;

        const bool launched = launch_mle_linear_combination(
            context.stream(), num_vars,
            reinterpret_cast<const uint64_t*>(address_staging.address()),
            reinterpret_cast<const Fr*>(coefficient_staging.address()), count,
            reinterpret_cast<Fr*>(result.data()));
        address_staging.release(context);
        coefficient_staging.release(context);
        if (!launched) {
            context.retire(result.data());
            *out = nullptr;
            return status_invalid_state();
        }
        return status_success();
    });
}

// ---------------------------------------------------------------------------
// hp_ascend910b3_v1_mle_evaluate
// ---------------------------------------------------------------------------
extern "C" Ascend910B3Status hp_ascend910b3_v1_mle_evaluate(
    Ascend910B3NativeContext* raw_context, Ascend910B3DeviceAddress polynomial,
    size_t num_vars, const uint8_t* point, uint8_t* out_evaluation) {
    return guarded([&]() -> Ascend910B3Status {
        ContextRef context_ref;
        Ascend910B3Status status = acquire_context(raw_context, context_ref);
        if (!status_ok(status)) return status;
        Context& context = *context_ref;
        Context::Scope scope = context.scope();
        if (!scope.ok()) return scope.status();
        if (context.closed()) return status_invalid_state();
        if (context.poisoned()) return status_from_acl(context.first_error());
        if (out_evaluation == nullptr) return status_invalid_parameters();
        if (point == nullptr && num_vars != 0) return status_invalid_parameters();
        size_t elements = 0, bytes = 0;
        if (!table_layout(num_vars, elements, bytes)) return status_invalid_parameters();
        if (num_vars != 0 && !canonical_fr_array(point, num_vars))
            return status_invalid_parameters();
        BufferRef input;
        status = lookup_table(context, polynomial, num_vars, input);
        if (!status_ok(status)) return status;

        PinnedStorage host_result;
        status = context.allocate_pinned(kFrBytes, kPinnedAlignment, host_result);
        if (!status_ok(status)) return status;

        if (num_vars == 0) {
            status = context.copy_to_host_async(host_result.data(), input.data(), kFrBytes);
            if (!status_ok(status)) return status;
            status = context.synchronize();
            if (!status_ok(status)) {
                // No completion observation: the D2H target may still be in
                // flight, so the pinned buffer is retained, never freed early.
                context.retain_pinned(std::move(host_result));
                context.poll_retired();
                return status;
            }
            std::memcpy(out_evaluation, host_result.data(), kFrBytes);
            return status_success();
        }

        std::vector<BufferRef> scratch;
        BufferRef scratch_a;
        BufferRef scratch_b;
        BufferRef result;
        MetadataTable point_staging;
        status = point_staging.upload(context, point, num_vars * kFrBytes);
        if (!status_ok(status)) return status;
        if (num_vars >= 2) {
            status = create_table(context, 0, (elements / 2) * kFrBytes,
                                  BufferKind::General, scratch_a, nullptr);
            if (!status_ok(status)) {
                point_staging.release(context);
                return status;
            }
            scratch.push_back(scratch_a);
        }
        if (num_vars >= 3) {
            status = create_table(context, 0, (elements / 4) * kFrBytes,
                                  BufferKind::General, scratch_b, nullptr);
            if (!status_ok(status)) {
                point_staging.release(context);
                retire_all(context, scratch);
                return status;
            }
            scratch.push_back(scratch_b);
        }
        status = create_table(context, 0, kFrBytes, BufferKind::General, result,
                              nullptr);
        if (!status_ok(status)) {
            point_staging.release(context);
            retire_all(context, scratch);
            return status;
        }
        scratch.push_back(result);

        const bool launched = launch_mle_evaluate(
            context.stream(), num_vars, reinterpret_cast<const Fr*>(input.data()),
            reinterpret_cast<const Fr*>(point_staging.address()),
            reinterpret_cast<Fr*>(result.data()),
            scratch_a.valid() ? reinterpret_cast<Fr*>(scratch_a.data()) : nullptr,
            scratch_b.valid() ? reinterpret_cast<Fr*>(scratch_b.data()) : nullptr);
        if (!launched) {
            point_staging.release(context);
            retire_all(context, scratch);
            return status_invalid_state();
        }
        status = context.copy_to_host_async(host_result.data(), result.data(), kFrBytes);
        point_staging.release(context);
        retire_all(context, scratch);
        if (status_ok(status)) status = context.synchronize();
        if (!status_ok(status)) {
            // No completion observation, so the pinned D2H target must outlive
            // any outstanding copy; hand it to the context instead of freeing.
            context.retain_pinned(std::move(host_result));
            context.poll_retired();
            return status;
        }
        std::memcpy(out_evaluation, host_result.data(), kFrBytes);
        return status_success();
    });
}

// ---------------------------------------------------------------------------
// hp_ascend910b3_v1_mle_build_eq_x_r
// ---------------------------------------------------------------------------
extern "C" Ascend910B3Status hp_ascend910b3_v1_mle_build_eq_x_r(
    Ascend910B3NativeContext* raw_context, const uint8_t* r, size_t num_vars,
    Ascend910B3DeviceAddress* out) {
    if (out != nullptr) *out = nullptr;
    return guarded([&]() -> Ascend910B3Status {
        ContextRef context_ref;
        Ascend910B3Status status = acquire_context(raw_context, context_ref);
        if (!status_ok(status)) return status;
        Context& context = *context_ref;
        Context::Scope scope = context.scope();
        if (!scope.ok()) return scope.status();
        if (context.closed()) return status_invalid_state();
        if (context.poisoned()) return status_from_acl(context.first_error());
        if (out == nullptr || r == nullptr) return status_invalid_parameters();
        if (num_vars == 0) return status_invalid_parameters();
        size_t elements = 0, bytes = 0;
        if (!table_layout(num_vars, elements, bytes)) return status_invalid_parameters();
        if (!canonical_fr_array(r, num_vars)) return status_invalid_parameters();

        MetadataTable coordinate_staging;
        status = coordinate_staging.upload(context, r, num_vars * kFrBytes);
        if (!status_ok(status)) return status;

        BufferRef result;
        status = create_fr_table(context, num_vars, bytes, result, out);
        if (!status_ok(status)) {
            coordinate_staging.release(context);
            return status;
        }
        const bool launched = launch_mle_build_eq_x_r(
            context.stream(), reinterpret_cast<const Fr*>(coordinate_staging.address()),
            num_vars, reinterpret_cast<Fr*>(result.data()));
        coordinate_staging.release(context);
        if (!launched) {
            context.retire(result.data());
            *out = nullptr;
            return status_invalid_state();
        }
        return status_success();
    });
}

// ---------------------------------------------------------------------------
// hp_ascend910b3_v1_mle_compute_frac_poly
// ---------------------------------------------------------------------------
extern "C" Ascend910B3Status hp_ascend910b3_v1_mle_compute_frac_poly(
    Ascend910B3NativeContext* raw_context, size_t num_vars,
    const Ascend910B3DeviceAddress* numerators,
    const Ascend910B3DeviceAddress* denominators, size_t count,
    Ascend910B3DeviceAddress* out) {
    if (out != nullptr) *out = nullptr;
    return guarded([&]() -> Ascend910B3Status {
        ContextRef context_ref;
        Ascend910B3Status status = acquire_context(raw_context, context_ref);
        if (!status_ok(status)) return status;
        Context& context = *context_ref;
        Context::Scope scope = context.scope();
        if (!scope.ok()) return scope.status();
        if (context.closed()) return status_invalid_state();
        if (context.poisoned()) return status_from_acl(context.first_error());
        if (out == nullptr || numerators == nullptr || denominators == nullptr ||
            count == 0)
            return status_invalid_parameters();
        size_t elements = 0, bytes = 0;
        if (!table_layout(num_vars, elements, bytes)) return status_invalid_parameters();

        std::vector<BufferRef> inputs;
        AddressTable numerator_table;
        AddressTable denominator_table;
        status = lookup_many(context, numerators, num_vars, count, inputs,
                             numerator_table);
        if (!status_ok(status)) return status;
        status = lookup_many(context, denominators, num_vars, count, inputs,
                             denominator_table);
        if (!status_ok(status)) return status;

        MetadataTable numerator_staging;
        status = numerator_table.upload(context, numerator_staging);
        if (!status_ok(status)) return status;
        MetadataTable denominator_staging;
        status = denominator_table.upload(context, denominator_staging);
        if (!status_ok(status)) return status;

        const uint32_t blocks = mle_frac_flag_blocks(num_vars);
        if (blocks == 0) {
            numerator_staging.release(context);
            denominator_staging.release(context);
            return status_invalid_state();
        }
        const size_t flag_bytes = size_t(blocks) * kFlagRecordBytes;
        BufferRef flags;
        status = context.create_buffer(flag_bytes, 0, BufferKind::General, flags);
        if (!status_ok(status)) {
            numerator_staging.release(context);
            denominator_staging.release(context);
            return status;
        }
        BufferRef result;
        status = create_fr_table(context, num_vars, bytes, result, nullptr);
        if (!status_ok(status)) {
            numerator_staging.release(context);
            denominator_staging.release(context);
            context.retire(flags.data());
            return status;
        }
        PinnedStorage host_flags;
        status = context.allocate_pinned(flag_bytes, kPinnedAlignment, host_flags);
        if (!status_ok(status)) {
            numerator_staging.release(context);
            denominator_staging.release(context);
            context.retire(flags.data());
            context.retire(result.data());
            return status;
        }

        const uint32_t launched = launch_mle_frac(
            context.stream(), num_vars,
            reinterpret_cast<const uint64_t*>(numerator_staging.address()),
            reinterpret_cast<const uint64_t*>(denominator_staging.address()), count,
            reinterpret_cast<Fr*>(result.data()), static_cast<uint8_t*>(flags.data()));
        numerator_staging.release(context);
        denominator_staging.release(context);
        if (launched == 0 || launched != blocks) {
            context.retire(flags.data());
            context.retire(result.data());
            return status_invalid_state();
        }
        status = context.copy_to_host_async(host_flags.data(), flags.data(), flag_bytes);
        if (status_ok(status)) status = context.synchronize();
        if (!status_ok(status)) {
            context.retain_pinned(std::move(host_flags));
            context.poll_retired();
            return status;
        }
        const int32_t* records = static_cast<const int32_t*>(host_flags.data());
        const size_t record_count = flag_bytes / sizeof(int32_t);
        bool denominator_zero = false;
        for (size_t index = 0; index < record_count; ++index) {
            if (records[index] != 0) {
                denominator_zero = true;
                break;
            }
        }
        context.retire(flags.data());
        if (denominator_zero) {
            context.retire(result.data());
            return status_invalid_parameters();
        }
        *out = result.data();
        return status_success();
    });
}

// ---------------------------------------------------------------------------
// hp_ascend910b3_v1_mle_compute_product_poly
// ---------------------------------------------------------------------------
extern "C" Ascend910B3Status hp_ascend910b3_v1_mle_compute_product_poly(
    Ascend910B3NativeContext* raw_context, Ascend910B3DeviceAddress frac,
    size_t num_vars, Ascend910B3DeviceAddress* out) {
    if (out != nullptr) *out = nullptr;
    return guarded([&]() -> Ascend910B3Status {
        ContextRef context_ref;
        Ascend910B3Status status = acquire_context(raw_context, context_ref);
        if (!status_ok(status)) return status;
        Context& context = *context_ref;
        Context::Scope scope = context.scope();
        if (!scope.ok()) return scope.status();
        if (context.closed()) return status_invalid_state();
        if (context.poisoned()) return status_from_acl(context.first_error());
        if (out == nullptr) return status_invalid_parameters();
        if (num_vars == 0) return status_invalid_parameters();
        size_t elements = 0, bytes = 0;
        if (!table_layout(num_vars, elements, bytes)) return status_invalid_parameters();
        BufferRef input;
        status = lookup_table(context, frac, num_vars, input);
        if (!status_ok(status)) return status;

        BufferRef result;
        status = create_fr_table(context, num_vars, bytes, result, out);
        if (!status_ok(status)) return status;
        if (!launch_mle_compute_product(context.stream(), num_vars,
                                        reinterpret_cast<const Fr*>(input.data()),
                                        reinterpret_cast<Fr*>(result.data()))) {
            context.retire(result.data());
            *out = nullptr;
            return status_invalid_state();
        }
        return status_success();
    });
}

// ---------------------------------------------------------------------------
// hp_ascend910b3_v1_mle_compute_nums_and_denoms
// ---------------------------------------------------------------------------
extern "C" Ascend910B3Status hp_ascend910b3_v1_mle_compute_nums_and_denoms(
    Ascend910B3NativeContext* raw_context, size_t num_vars, const uint8_t* beta,
    const uint8_t* gamma, const Ascend910B3DeviceAddress* fxs,
    const Ascend910B3DeviceAddress* gxs,
    const Ascend910B3DeviceAddress* permutations, size_t count,
    Ascend910B3DeviceAddress* out_numerators,
    Ascend910B3DeviceAddress* out_denominators) {
    if (out_numerators != nullptr && out_denominators != nullptr &&
        count <= (size_t(1) << 31)) {
        for (size_t index = 0; index < count; ++index) {
            out_numerators[index] = nullptr;
            out_denominators[index] = nullptr;
        }
    }
    return guarded([&]() -> Ascend910B3Status {
        ContextRef context_ref;
        Ascend910B3Status status = acquire_context(raw_context, context_ref);
        if (!status_ok(status)) return status;
        Context& context = *context_ref;
        Context::Scope scope = context.scope();
        if (!scope.ok()) return scope.status();
        if (context.closed()) return status_invalid_state();
        if (context.poisoned()) return status_from_acl(context.first_error());
        if (beta == nullptr || gamma == nullptr || fxs == nullptr || gxs == nullptr ||
            permutations == nullptr || out_numerators == nullptr ||
            out_denominators == nullptr || count == 0)
            return status_invalid_parameters();
        if (count > (size_t(1) << 31)) return status_invalid_parameters();
        size_t elements = 0, bytes = 0;
        if (!table_layout(num_vars, elements, bytes)) return status_invalid_parameters();
        if (!canonical_fr(beta) || !canonical_fr(gamma))
            return status_invalid_parameters();
        // beta * (column * elements + row) is formed on device; reject inputs
        // whose column offset could overflow that arithmetic.
        const uint64_t total = static_cast<uint64_t>(elements);
        if (total != 0 &&
            count > std::numeric_limits<uint64_t>::max() / total)
            return status_invalid_parameters();

        std::vector<BufferRef> inputs;
        AddressTable fx_table;
        AddressTable gx_table;
        AddressTable perm_table;
        status = lookup_many(context, fxs, num_vars, count, inputs, fx_table);
        if (!status_ok(status)) return status;
        status = lookup_many(context, gxs, num_vars, count, inputs, gx_table);
        if (!status_ok(status)) return status;
        status = lookup_many(context, permutations, num_vars, count, inputs,
                             perm_table);
        if (!status_ok(status)) return status;

        MetadataTable fx_staging;
        status = fx_table.upload(context, fx_staging);
        if (!status_ok(status)) return status;
        MetadataTable gx_staging;
        status = gx_table.upload(context, gx_staging);
        if (!status_ok(status)) return status;
        MetadataTable perm_staging;
        status = perm_table.upload(context, perm_staging);
        if (!status_ok(status)) return status;

        uint8_t scalars[2 * kFrBytes];
        std::memcpy(scalars, beta, kFrBytes);
        std::memcpy(scalars + kFrBytes, gamma, kFrBytes);
        MetadataTable scalar_staging;
        status = scalar_staging.upload(context, scalars, sizeof(scalars));
        if (!status_ok(status)) {
            fx_staging.release(context);
            gx_staging.release(context);
            perm_staging.release(context);
            return status;
        }

        std::vector<BufferRef> outputs;
        outputs.reserve(2 * count);
        AddressTable numerator_table;
        AddressTable denominator_table;
        status = status_success();
        for (size_t index = 0; index < 2 * count && status_ok(status); ++index) {
            BufferRef created;
            status = create_fr_table(context, num_vars, bytes, created, nullptr);
            if (!status_ok(status)) break;
            outputs.push_back(std::move(created));
        }
        if (!status_ok(status)) {
            retire_all(context, outputs);
            fx_staging.release(context);
            gx_staging.release(context);
            perm_staging.release(context);
            scalar_staging.release(context);
            return status;
        }
        for (size_t index = 0; index < count; ++index) {
            numerator_table.push(outputs[index].data());
            denominator_table.push(outputs[count + index].data());
        }
        MetadataTable numerator_out_staging;
        status = numerator_table.upload(context, numerator_out_staging);
        if (!status_ok(status)) {
            retire_all(context, outputs);
            fx_staging.release(context);
            gx_staging.release(context);
            perm_staging.release(context);
            scalar_staging.release(context);
            return status;
        }
        MetadataTable denominator_out_staging;
        status = denominator_table.upload(context, denominator_out_staging);
        if (!status_ok(status)) {
            retire_all(context, outputs);
            fx_staging.release(context);
            gx_staging.release(context);
            perm_staging.release(context);
            scalar_staging.release(context);
            numerator_out_staging.release(context);
            return status;
        }

        const bool launched = launch_mle_nums_and_denoms(
            context.stream(), num_vars,
            reinterpret_cast<const Fr*>(scalar_staging.address()),
            reinterpret_cast<const Fr*>(scalar_staging.address()) + 1,
            reinterpret_cast<const uint64_t*>(fx_staging.address()),
            reinterpret_cast<const uint64_t*>(gx_staging.address()),
            reinterpret_cast<const uint64_t*>(perm_staging.address()), count,
            reinterpret_cast<const uint64_t*>(numerator_out_staging.address()),
            reinterpret_cast<const uint64_t*>(denominator_out_staging.address()));
        fx_staging.release(context);
        gx_staging.release(context);
        perm_staging.release(context);
        scalar_staging.release(context);
        numerator_out_staging.release(context);
        denominator_out_staging.release(context);
        if (!launched) {
            retire_all(context, outputs);
            return status_invalid_state();
        }
        for (size_t index = 0; index < count; ++index) {
            out_numerators[index] = outputs[index].data();
            out_denominators[index] = outputs[count + index].data();
        }
        return status_success();
    });
}

// ---------------------------------------------------------------------------
// hp_ascend910b3_v1_mle_compute_product_factors
// ---------------------------------------------------------------------------
extern "C" Ascend910B3Status hp_ascend910b3_v1_mle_compute_product_factors(
    Ascend910B3NativeContext* raw_context, Ascend910B3DeviceAddress frac,
    Ascend910B3DeviceAddress product, size_t num_vars,
    Ascend910B3DeviceAddress* out_p1, Ascend910B3DeviceAddress* out_p2) {
    if (out_p1 != nullptr) *out_p1 = nullptr;
    if (out_p2 != nullptr) *out_p2 = nullptr;
    return guarded([&]() -> Ascend910B3Status {
        ContextRef context_ref;
        Ascend910B3Status status = acquire_context(raw_context, context_ref);
        if (!status_ok(status)) return status;
        Context& context = *context_ref;
        Context::Scope scope = context.scope();
        if (!scope.ok()) return scope.status();
        if (context.closed()) return status_invalid_state();
        if (context.poisoned()) return status_from_acl(context.first_error());
        if (out_p1 == nullptr || out_p2 == nullptr)
            return status_invalid_parameters();
        if (num_vars == 0) return status_invalid_parameters();
        size_t elements = 0, bytes = 0;
        if (!table_layout(num_vars, elements, bytes)) return status_invalid_parameters();
        BufferRef frac_ref;
        status = lookup_table(context, frac, num_vars, frac_ref);
        if (!status_ok(status)) return status;
        BufferRef product_ref;
        status = lookup_table(context, product, num_vars, product_ref);
        if (!status_ok(status)) return status;

        std::vector<BufferRef> outputs;
        outputs.reserve(2);
        BufferRef first_output;
        status = create_fr_table(context, num_vars, bytes, first_output, nullptr);
        if (!status_ok(status)) return status;
        outputs.push_back(std::move(first_output));
        BufferRef second_output;
        status = create_fr_table(context, num_vars, bytes, second_output, nullptr);
        if (!status_ok(status)) {
            retire_all(context, outputs);
            return status;
        }
        outputs.push_back(std::move(second_output));
        *out_p1 = outputs[0].data();
        *out_p2 = outputs[1].data();
        if (!launch_mle_product_factors(
                context.stream(), num_vars,
                reinterpret_cast<const Fr*>(frac_ref.data()),
                reinterpret_cast<const Fr*>(product_ref.data()),
                reinterpret_cast<Fr*>(outputs[0].data()),
                reinterpret_cast<Fr*>(outputs[1].data()))) {
            retire_all(context, outputs);
            *out_p1 = nullptr;
            *out_p2 = nullptr;
            return status_invalid_state();
        }
        return status_success();
    });
}
