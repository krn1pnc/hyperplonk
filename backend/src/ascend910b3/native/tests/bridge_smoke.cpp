// Actual C ABI smoke test for the Ascend910B3 bridge.
//
// Numerical expectations below are computed by independent host reference code
// (boost::multiprecision) over the documented CPU semantics. They intentionally
// do not call, include or mirror the production field/group/vector helpers.
//
// Covered here: the 20 bridge.h entry points, upload/evaluate/eq/frac/product/
// nums+denoms equality (including an 11/12-variable multi-tile public-index
// regression), product-factors equality, KZG commit/multi-commit/open against a
// direct CPU group sum over fixed valid generator-point layers, complete generic
// sumcheck rounds (degrees 1,2,3,7,34,65), state ordering/error-no-advance, empty
// virtual polynomials, retirement deferral, two-context isolation and the small
// ready-on-return transfer budgets enforced with the existing runtime audit.

#include "../bridge.h"
#include "runtime_audit.h"

#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace audit = hyperplonk::ascend910b3::test;

using Big = boost::multiprecision::cpp_int;
using audit::RuntimeCounts;

namespace abi {
inline constexpr auto buffer_retire = &::hp_ascend910b3_v1_buffer_retire;
inline constexpr auto context_create = &::hp_ascend910b3_v1_context_create;
inline constexpr auto context_destroy = &::hp_ascend910b3_v1_context_destroy;
inline constexpr auto mle_build_eq_x_r = &::hp_ascend910b3_v1_mle_build_eq_x_r;
inline constexpr auto mle_compute_frac_poly = &::hp_ascend910b3_v1_mle_compute_frac_poly;
inline constexpr auto mle_compute_nums_and_denoms =
    &::hp_ascend910b3_v1_mle_compute_nums_and_denoms;
inline constexpr auto mle_compute_product_factors =
    &::hp_ascend910b3_v1_mle_compute_product_factors;
inline constexpr auto mle_compute_product_poly =
    &::hp_ascend910b3_v1_mle_compute_product_poly;
inline constexpr auto mle_evaluate = &::hp_ascend910b3_v1_mle_evaluate;
inline constexpr auto mle_linear_combination =
    &::hp_ascend910b3_v1_mle_linear_combination;
inline constexpr auto mle_upload = &::hp_ascend910b3_v1_mle_upload;
inline constexpr auto multilinear_kzg_commit = &::hp_ascend910b3_v1_multilinear_kzg_commit;
inline constexpr auto multilinear_kzg_multi_commit =
    &::hp_ascend910b3_v1_multilinear_kzg_multi_commit;
inline constexpr auto multilinear_kzg_open = &::hp_ascend910b3_v1_multilinear_kzg_open;
inline constexpr auto multilinear_kzg_params_create =
    &::hp_ascend910b3_v1_multilinear_kzg_params_create;
inline constexpr auto multilinear_kzg_params_destroy =
    &::hp_ascend910b3_v1_multilinear_kzg_params_destroy;
inline constexpr auto multilinear_kzg_params_upload_layer =
    &::hp_ascend910b3_v1_multilinear_kzg_params_upload_layer;
inline constexpr auto sumcheck_prover_state_create =
    &::hp_ascend910b3_v1_sumcheck_prover_state_create;
inline constexpr auto sumcheck_prover_state_destroy =
    &::hp_ascend910b3_v1_sumcheck_prover_state_destroy;
inline constexpr auto sumcheck_prover_state_round =
    &::hp_ascend910b3_v1_sumcheck_prover_state_round;
} // namespace abi

// ---------------------------------------------------------------------------
// Failure reporting and status handling.
// ---------------------------------------------------------------------------

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
    if (!condition) fail(message);
}

std::string status_text(Ascend910B3Status status) {
    return "kind=" + std::to_string(status.kind) +
           " native_code=" + std::to_string(status.native_code);
}

void require_ok(Ascend910B3Status status, const std::string& what) {
    if (status.kind != 0 || status.native_code != 0)
        fail(what + ": " + status_text(status));
}

// Contract errors are InvalidParameters(1) or InvalidState(2). native_code is
// incidental and deliberately not inspected. NOT_IMPLEMENTED(3) always fails.
void require_contract_error(Ascend910B3Status status, const std::string& what) {
    if (status.kind != 1 && status.kind != 2)
        fail(what + ": expected contract error, got " + status_text(status));
}

// ---------------------------------------------------------------------------
// Independent host field arithmetic over BLS12-381 Fr and Fq.
// ---------------------------------------------------------------------------

Big le32_words(std::initializer_list<uint32_t> words) {
    Big value = 0;
    size_t index = 0;
    for (uint32_t word : words) {
        value += Big(word) << (32 * index);
        ++index;
    }
    return value;
}

// Standard BLS12-381 scalar field and base field moduli.
const Big kFrModulus = le32_words({
    0x00000001, 0xffffffff, 0xfffe5bfe, 0x53bda402,
    0x09a1d805, 0x3339d808, 0x299d7d48, 0x73eda753});
const Big kFqModulus = le32_words({
    0xffffaaab, 0xb9feffff, 0xb153ffff, 0x1eabfffe,
    0xf6b0f624, 0x6730d2a0, 0xf38512bf, 0x64774b84,
    0x434bacd7, 0x4b1ba7b6, 0x397fe69a, 0x1a0111ea});

Big mod_big(Big value, const Big& modulus) {
    value %= modulus;
    if (value < 0) value += modulus;
    return value;
}

Big pow_mod(Big base, Big exponent, const Big& modulus) {
    base = mod_big(base, modulus);
    Big result = 1;
    while (exponent > 0) {
        if (boost::multiprecision::bit_test(exponent, 0)) result = (result * base) % modulus;
        exponent >>= 1;
        if (exponent > 0) base = (base * base) % modulus;
    }
    return result;
}

Big fr_norm(Big value) { return mod_big(std::move(value), kFrModulus); }
Big fq_norm(Big value) { return mod_big(std::move(value), kFqModulus); }

struct RefFr {
    Big v = 0;

    static RefFr zero() { return {}; }
    static RefFr one() { return {Big(1)}; }

    RefFr add(const RefFr& other) const { return {fr_norm(v + other.v)}; }
    RefFr sub(const RefFr& other) const { return {fr_norm(v - other.v)}; }
    RefFr mul(const RefFr& other) const { return {fr_norm(v * other.v)}; }
    RefFr inverse() const {
        require(v != 0, "reference inverse of zero");
        return {pow_mod(v, kFrModulus - 2, kFrModulus)};
    }
    bool equals(const RefFr& other) const { return v == other.v; }

    std::array<uint8_t, 32> bytes() const {
        std::array<uint8_t, 32> out{};
        Big value = v;
        for (size_t i = 0; i < out.size(); ++i) {
            const Big low = value & 0xFF;
            out[i] = static_cast<uint8_t>(low.convert_to<unsigned>());
            value >>= 8;
        }
        return out;
    }
};

RefFr fr(uint64_t value) { return {fr_norm(Big(value))}; }
RefFr fr_minus_one() { return {fr_norm(kFrModulus - 1)}; }

// Deterministic full-width canonical Fr value. The high and middle terms stay
// below the modulus, while the seed makes every generated value distinct and
// its low words non-trivial, so a stale pointer or coefficient cannot alias a
// correct one by accident.
RefFr fr_spread(uint64_t seed) {
    const Big high = Big(1) << 249;
    const Big middle = Big(seed + 1) << 157;
    const Big low = Big(seed + 1) * Big(0x9e3779b97f4a7c15ULL) + 0x243f6a8885a308d3ULL;
    return {fr_norm(high + middle + low)};
}

RefFr decode_fr_bytes(const uint8_t* bytes, const std::string& what) {
    Big value = 0;
    for (size_t i = 0; i < 32; ++i) value = (value << 8) + Big(bytes[31 - i]);
    require(value < kFrModulus, what + ": noncanonical Fr result");
    return {value};
}

// ---------------------------------------------------------------------------
// Independent host G1 arithmetic (Jacobian, a = 0, b = 4).
// ---------------------------------------------------------------------------

struct RefPoint {
    bool infinity = true;
    Big x = 0;
    Big y = 0;
};

struct RefJacobian {
    Big x = 0;
    Big y = 0;
    Big z = 0; // zero is the identity
};

RefPoint make_generator() {
    return {
        false,
        le32_words({0xdb22c6bb, 0xfb3af00a, 0xf97a1aef, 0x6c55e83f,
                    0x171bac58, 0xa14e3a3f, 0x9774b905, 0xc3688c4f,
                    0x4fa9ac0f, 0x2695638c, 0x3197d794, 0x17f1d3a7}),
        le32_words({0x46c5e7e1, 0x0caa2329, 0xa2888ae4, 0xd03cc744,
                    0x2c04b3ed, 0x00db18cb, 0xd5d00af6, 0xfcf5e095,
                    0x741d8ae4, 0xa09e30ed, 0xe3aaa0f1, 0x08b3f481})};
}

const RefPoint kGenerator = make_generator();

RefJacobian jacobian_from_point(const RefPoint& point) {
    if (point.infinity) return {};
    return {point.x, point.y, Big(1)};
}

RefJacobian jacobian_double(const RefJacobian& value) {
    if (value.z == 0) return {};
    const Big a = fq_norm(value.x * value.x);
    const Big b = fq_norm(value.y * value.y);
    const Big c = fq_norm(b * b);
    const Big xb = fq_norm(value.x + b);
    const Big d = fq_norm((xb * xb - a - c) * 2);
    const Big e = fq_norm(a + a + a);
    const Big f = fq_norm(e * e);
    const Big x3 = fq_norm(f - d - d);
    const Big y3 = fq_norm(e * (d - x3) - c * 8);
    const Big z3 = fq_norm((value.y + value.y) * value.z);
    if (z3 == 0) return {};
    return {x3, y3, z3};
}

RefJacobian jacobian_add(const RefJacobian& left, const RefJacobian& right) {
    if (left.z == 0) return right;
    if (right.z == 0) return left;
    const Big z1z1 = fq_norm(left.z * left.z);
    const Big z2z2 = fq_norm(right.z * right.z);
    const Big u1 = fq_norm(left.x * z2z2);
    const Big u2 = fq_norm(right.x * z1z1);
    const Big s1 = fq_norm(left.y * right.z * z2z2);
    const Big s2 = fq_norm(right.y * left.z * z1z1);
    if (u1 == u2) {
        if (s1 == s2) return jacobian_double(left);
        return {};
    }
    const Big h = fq_norm(u2 - u1);
    const Big two_h = fq_norm(h + h);
    const Big i = fq_norm(two_h * two_h);
    const Big j = fq_norm(h * i);
    const Big r = fq_norm((s2 - s1) + (s2 - s1));
    const Big v = fq_norm(u1 * i);
    const Big x3 = fq_norm(r * r - j - v - v);
    const Big y3 = fq_norm(r * (v - x3) - (s1 * j) - (s1 * j));
    const Big zsum = fq_norm(left.z + right.z);
    const Big z3 = fq_norm((zsum * zsum - z1z1 - z2z2) * h);
    if (z3 == 0) return {};
    return {x3, y3, z3};
}

RefPoint jacobian_to_point(const RefJacobian& value) {
    if (value.z == 0) return {};
    const Big z_inverse = pow_mod(value.z, kFqModulus - 2, kFqModulus);
    const Big z_inverse2 = fq_norm(z_inverse * z_inverse);
    return {
        false,
        fq_norm(value.x * z_inverse2),
        fq_norm(value.y * z_inverse2 * z_inverse)};
}

RefJacobian scalar_mul_jacobian(const RefPoint& point, const RefFr& scalar) {
    RefJacobian result{};
    RefJacobian base = jacobian_from_point(point);
    for (unsigned bit = 0; bit < 255; ++bit) {
        if (boost::multiprecision::bit_test(scalar.v, bit)) result = jacobian_add(result, base);
        base = jacobian_double(base);
    }
    return result;
}

RefPoint scalar_mul_point(const RefPoint& point, const RefFr& scalar) {
    return jacobian_to_point(scalar_mul_jacobian(point, scalar));
}

RefPoint sum_points(const std::vector<RefPoint>& bases, const std::vector<RefFr>& scalars) {
    require(bases.size() == scalars.size(), "host group sum size mismatch");
    RefJacobian sum{};
    for (size_t i = 0; i < bases.size(); ++i)
        sum = jacobian_add(sum, scalar_mul_jacobian(bases[i], scalars[i]));
    return jacobian_to_point(sum);
}

void write_le_bytes(uint8_t* destination, size_t bytes, Big value) {
    for (size_t i = 0; i < bytes; ++i) {
        const Big low = value & 0xFF;
        destination[i] = static_cast<uint8_t>(low.convert_to<unsigned>());
        value >>= 8;
    }
}

std::array<uint8_t, 97> point_packet(const RefPoint& point) {
    std::array<uint8_t, 97> out{};
    if (point.infinity) {
        out[0] = 1;
        return out;
    }
    write_le_bytes(out.data() + 1, 48, point.x);
    write_le_bytes(out.data() + 49, 48, point.y);
    return out;
}

// ---------------------------------------------------------------------------
// Independent reference tables and CPU semantics.
// ---------------------------------------------------------------------------

struct RefTable {
    size_t num_vars = 0;
    std::vector<RefFr> values;

    RefTable(size_t variables, std::vector<RefFr> table)
        : num_vars(variables), values(std::move(table)) {
        require(values.size() == (size_t(1) << variables), "invalid reference table size");
    }
};

RefTable make_table(size_t variables, std::initializer_list<uint64_t> values) {
    std::vector<RefFr> table;
    table.reserve(values.size());
    for (uint64_t value : values) table.push_back(fr(value));
    return RefTable(variables, std::move(table));
}

RefTable zero_table(size_t variables) {
    return RefTable(variables, std::vector<RefFr>(size_t(1) << variables, RefFr::zero()));
}

std::vector<uint8_t> encode_table(const RefTable& table) {
    std::vector<uint8_t> bytes;
    bytes.reserve(table.values.size() * 32);
    for (const RefFr& value : table.values) {
        const auto encoded = value.bytes();
        bytes.insert(bytes.end(), encoded.begin(), encoded.end());
    }
    return bytes;
}

std::vector<uint8_t> encode_values(const std::vector<RefFr>& values) {
    std::vector<uint8_t> bytes;
    bytes.reserve(values.size() * 32);
    for (const RefFr& value : values) {
        const auto encoded = value.bytes();
        bytes.insert(bytes.end(), encoded.begin(), encoded.end());
    }
    return bytes;
}

RefFr reference_evaluate(const RefTable& table, const std::vector<RefFr>& point) {
    require(point.size() == table.num_vars, "reference evaluate point size");
    RefFr sum = RefFr::zero();
    for (size_t index = 0; index < table.values.size(); ++index) {
        RefFr product = RefFr::one();
        for (size_t bit = 0; bit < table.num_vars; ++bit) {
            const RefFr factor = ((index >> bit) & 1) != 0
                ? point[bit]
                : RefFr::one().sub(point[bit]);
            product = product.mul(factor);
        }
        sum = sum.add(product.mul(table.values[index]));
    }
    return sum;
}

RefTable reference_eq_table(const std::vector<RefFr>& r) {
    std::vector<RefFr> values(size_t(1) << r.size(), RefFr::one());
    for (size_t index = 0; index < values.size(); ++index) {
        RefFr product = RefFr::one();
        for (size_t bit = 0; bit < r.size(); ++bit) {
            const RefFr factor = ((index >> bit) & 1) != 0
                ? r[bit]
                : RefFr::one().sub(r[bit]);
            product = product.mul(factor);
        }
        values[index] = product;
    }
    return RefTable(r.size(), std::move(values));
}

RefTable reference_product_table(const RefTable& frac) {
    require(frac.num_vars > 0, "reference product requires variables");
    const size_t size = frac.values.size();
    std::vector<RefFr> evaluations;
    evaluations.reserve(size);
    for (size_t i = 0; i < size; i += 2)
        evaluations.push_back(frac.values[i].mul(frac.values[i + 1]));
    for (size_t child = 0; child + 1 < size - 1; child += 2)
        evaluations.push_back(evaluations[child].mul(evaluations[child + 1]));
    evaluations.push_back(RefFr::zero());
    require(evaluations.size() == size, "reference product recurrence length");
    return RefTable(frac.num_vars, std::move(evaluations));
}

// Full-range fixture for the per-tile public-index regression. With
// beta = -1, gamma = -2, fx = gx = column*total + row + 10 and
// perm = column*total + (row ^ 1):
//   numerator(row)   = fx + gamma + beta*(column*total + row) = 8,
//   denominator(row) = gx + gamma + beta*perm                = 7/9 by row parity.
// The closed-form MLE at a non-Boolean first coordinate x is therefore 8 and
// 7 + 2*x for every column and every choice of the higher coordinates, so a
// stale base index in a later 48-lane tile cannot hide behind the other
// coordinates. Values are built only from the independent host Fr helpers.
void reference_index_fixture(uint32_t num_vars, uint32_t columns,
                             std::vector<RefTable>& fx,
                             std::vector<RefTable>& perm) {
    require(num_vars < 63 && columns > 0, "invalid public-index fixture");
    const uint64_t total = uint64_t(1) << num_vars;
    fx.clear();
    perm.clear();
    for (uint32_t column = 0; column < columns; ++column) {
        const uint64_t column_base = uint64_t(column) * total;
        std::vector<RefFr> fx_values;
        std::vector<RefFr> perm_values;
        fx_values.reserve(total);
        perm_values.reserve(total);
        for (uint64_t row = 0; row < total; ++row) {
            fx_values.push_back(fr(column_base + row + 10));
            perm_values.push_back(fr(column_base + (row ^ 1)));
        }
        fx.emplace_back(num_vars, std::move(fx_values));
        perm.emplace_back(num_vars, std::move(perm_values));
    }
}

// ---------------------------------------------------------------------------
// ABI helpers.
// ---------------------------------------------------------------------------

Ascend910B3DeviceAddress upload_table(Ascend910B3NativeContext* context,
                                      const RefTable& table) {
    std::vector<uint8_t> bytes = encode_table(table);
    Ascend910B3DeviceAddress address = nullptr;
    require_ok(abi::mle_upload(context, table.num_vars, bytes.data(), table.values.size(),
                               &address),
               "mle_upload");
    require(address != nullptr, "mle_upload returned a null handle");
    // The borrowed host evaluations must not be needed after return.
    std::fill(bytes.begin(), bytes.end(), uint8_t(0xA5));
    return address;
}

RefFr evaluate_checked(Ascend910B3NativeContext* context,
                       Ascend910B3DeviceAddress polynomial,
                       const std::vector<RefFr>& point,
                       const std::string& what) {
    std::vector<uint8_t> point_bytes = encode_values(point);
    std::array<uint8_t, 32> output{};
    require_ok(abi::mle_evaluate(context, polynomial, point.size(),
                                 point_bytes.empty() ? nullptr : point_bytes.data(),
                                 output.data()),
               what);
    return decode_fr_bytes(output.data(), what);
}

void check_table(Ascend910B3NativeContext* context,
                 Ascend910B3DeviceAddress polynomial,
                 const RefTable& expected,
                 const std::string& what) {
    for (size_t index = 0; index < expected.values.size(); ++index) {
        std::vector<RefFr> point(expected.num_vars);
        for (size_t bit = 0; bit < expected.num_vars; ++bit)
            point[bit] = fr(((index >> bit) & 1) != 0 ? 1 : 0);
        const RefFr actual = evaluate_checked(
            context, polynomial, point, what + " boolean " + std::to_string(index));
        require(actual.equals(expected.values[index]),
                what + " table mismatch at " + std::to_string(index));
    }
}

void compare_fr_bytes(const uint8_t* actual, const RefFr& expected,
                      const std::string& what) {
    const auto encoded = expected.bytes();
    require(std::memcmp(actual, encoded.data(), encoded.size()) == 0, what + " mismatch");
}

void compare_packet(const uint8_t* actual, const RefPoint& expected,
                    const std::string& what) {
    const auto encoded = point_packet(expected);
    require(std::memcmp(actual, encoded.data(), encoded.size()) == 0, what + " mismatch");
}

void expect_small_io(const RuntimeCounts& counts, uint64_t max_d2h, uint64_t max_h2d,
                     bool forbid_allocation, const std::string& what) {
    require(counts.unexpected_copy_calls == 0, what + " unexpected copy");
    require(counts.d2h_calls >= 1, what + " host result did not use a D2H transfer");
    require(counts.d2h_bytes > 0 && counts.d2h_bytes <= max_d2h,
            what + " D2H byte budget h2d=" + std::to_string(counts.h2d_bytes) +
                " d2h=" + std::to_string(counts.d2h_bytes));
    require(counts.h2d_bytes <= max_h2d,
            what + " H2D byte budget h2d=" + std::to_string(counts.h2d_bytes));
    if (forbid_allocation)
        require(counts.malloc_calls == 0 && counts.host_malloc_calls == 0,
                what + " allocated inside a resident host-result call");
}

// ---------------------------------------------------------------------------
// Backend MLE operations.
// ---------------------------------------------------------------------------

void test_mle(Ascend910B3NativeContext* context, Ascend910B3NativeContext* other) {
    uint8_t sentinel = 0;
    std::vector<Ascend910B3DeviceAddress> owned;
    auto track = [&](Ascend910B3DeviceAddress address) {
        owned.push_back(address);
        return address;
    };

    const RefTable p = make_table(3, {1, 2, 3, 4, 5, 6, 7, 8});
    const RefTable q = make_table(3, {9, 7, 5, 3, 2, 4, 6, 8});
    const RefTable g = make_table(3, {2, 3, 5, 7, 11, 13, 17, 19});
    const RefTable h = make_table(3, {23, 29, 31, 37, 41, 43, 47, 53});
    const RefTable zeros = zero_table(3);
    const RefTable constant = make_table(0, {42});
    const RefTable small = make_table(2, {2, 5, 7, 11});

    // Upload validation: dimension, noncanonical values and a missing buffer.
    {
        std::vector<uint8_t> wrong_count = encode_table(small);
        Ascend910B3DeviceAddress output = &sentinel;
        require_contract_error(abi::mle_upload(context, 2, wrong_count.data(), 3, &output),
                               "mle_upload wrong evaluation count");
        require(output == nullptr, "mle_upload wrong count did not clear its output");

        std::vector<uint8_t> noncanonical(32, uint8_t(0xFF));
        output = &sentinel;
        require_contract_error(abi::mle_upload(context, 0, noncanonical.data(), 1, &output),
                               "mle_upload noncanonical Fr");
        require(output == nullptr, "mle_upload noncanonical did not clear its output");

        output = &sentinel;
        require_contract_error(abi::mle_upload(context, 0, nullptr, 1, &output),
                               "mle_upload null evaluations");
        require(output == nullptr, "mle_upload null evaluations did not clear its output");
    }

    const Ascend910B3DeviceAddress dp = track(upload_table(context, p));
    const Ascend910B3DeviceAddress dq = track(upload_table(context, q));
    const Ascend910B3DeviceAddress dg = track(upload_table(context, g));
    const Ascend910B3DeviceAddress dh = track(upload_table(context, h));
    const Ascend910B3DeviceAddress dzeros = track(upload_table(context, zeros));
    const Ascend910B3DeviceAddress dconstant = track(upload_table(context, constant));
    const Ascend910B3DeviceAddress dsmall = track(upload_table(context, small));

    check_table(context, dp, p, "mle_upload polynomial p");
    check_table(context, dq, q, "mle_upload polynomial q");
    check_table(context, dconstant, constant, "mle_upload constant");

    // linear_combination: elementwise a*p + b*q; inputs immutable.
    {
        const std::vector<RefFr> coefficients{fr(3), fr(5)};
        std::vector<uint8_t> coefficient_bytes = encode_values(coefficients);
        const std::array<Ascend910B3DeviceAddress, 2> polynomials{dp, dq};
        Ascend910B3DeviceAddress output = &sentinel;
        require_ok(abi::mle_linear_combination(context, 3, polynomials.data(),
                                               coefficient_bytes.data(), 2, &output),
                   "mle_linear_combination");
        require(output != nullptr, "mle_linear_combination returned a null handle");
        std::vector<RefFr> expected;
        for (size_t i = 0; i < p.values.size(); ++i)
            expected.push_back(coefficients[0].mul(p.values[i])
                                   .add(coefficients[1].mul(q.values[i])));
        const RefTable expected_table(3, std::move(expected));
        check_table(context, output, expected_table, "mle_linear_combination table");
        require_ok(abi::buffer_retire(context, output), "retire linear combination");
        check_table(context, dp, p, "mle_linear_combination input p");
        check_table(context, dq, q, "mle_linear_combination input q");

        // Different dimensions between multiplicands are rejected.
        const std::array<Ascend910B3DeviceAddress, 2> mixed{dp, dsmall};
        output = &sentinel;
        require_contract_error(abi::mle_linear_combination(context, 3, mixed.data(),
                                                           coefficient_bytes.data(), 2, &output),
                               "mle_linear_combination mixed dimensions");
        require(output == nullptr, "linear combination dimension error output");

        // Foreign context handles are rejected.
        const RefTable foreign(3, q.values);
        Ascend910B3DeviceAddress foreign_handle = upload_table(other, foreign);
        const std::array<Ascend910B3DeviceAddress, 2> with_foreign{dp, foreign_handle};
        output = &sentinel;
        require_contract_error(abi::mle_linear_combination(context, 3, with_foreign.data(),
                                                           coefficient_bytes.data(), 2, &output),
                               "mle_linear_combination foreign context");
        require(output == nullptr, "linear combination foreign output");
        require_ok(abi::buffer_retire(other, foreign_handle), "retire foreign handle");

        // An empty linear combination is not a valid MLE.
        output = &sentinel;
        require_contract_error(abi::mle_linear_combination(context, 3, nullptr, nullptr, 0,
                                                           &output),
                               "mle_linear_combination empty");
        require(output == nullptr, "linear combination empty output");
    }

    // evaluate: general point, canonical edge values and the constant MLE.
    {
        const std::vector<RefFr> point{fr(2), fr(5), fr(11)};
        const RefFr expected = reference_evaluate(p, point);
        std::vector<uint8_t> point_bytes = encode_values(point);
        std::array<uint8_t, 32> actual{};
        audit::begin_runtime_audit();
        const Ascend910B3Status status =
            abi::mle_evaluate(context, dp, 3, point_bytes.data(), actual.data());
        const RuntimeCounts counts = audit::end_runtime_audit();
        require_ok(status, "mle_evaluate");
        compare_fr_bytes(actual.data(), expected, "mle_evaluate");
        expect_small_io(counts, 32, 3 * 32, false, "mle_evaluate");

        const std::vector<RefFr> edges{fr_minus_one(), fr(4), fr(0)};
        const RefFr edge_expected = reference_evaluate(p, edges);
        std::vector<uint8_t> edge_bytes = encode_values(edges);
        std::array<uint8_t, 32> edge_actual{};
        require_ok(abi::mle_evaluate(context, dp, 3, edge_bytes.data(), edge_actual.data()),
                   "mle_evaluate canonical edges");
        compare_fr_bytes(edge_actual.data(), edge_expected, "mle_evaluate canonical edges");

        std::array<uint8_t, 32> constant_actual{};
        audit::begin_runtime_audit();
        const Ascend910B3Status constant_status =
            abi::mle_evaluate(context, dconstant, 0, nullptr, constant_actual.data());
        const RuntimeCounts constant_counts = audit::end_runtime_audit();
        require_ok(constant_status, "mle_evaluate constant");
        compare_fr_bytes(constant_actual.data(), constant.values[0], "mle_evaluate constant");
        expect_small_io(constant_counts, 32, 0, false, "mle_evaluate constant");

        std::array<uint8_t, 32> rejected{};
        require_contract_error(abi::mle_evaluate(context, dp, 2, point_bytes.data(),
                                                 rejected.data()),
                               "mle_evaluate wrong dimension");
        require_contract_error(abi::mle_evaluate(context, dp, 3, nullptr, rejected.data()),
                               "mle_evaluate null point");
        std::vector<uint8_t> noncanonical_point(3 * 32, uint8_t(0xFF));
        require_contract_error(abi::mle_evaluate(context, dp, 3, noncanonical_point.data(),
                                                 rejected.data()),
                               "mle_evaluate noncanonical point");
        require_ok(abi::mle_evaluate(context, dp, 3, point_bytes.data(), rejected.data()),
                   "mle_evaluate after rejected calls");
        compare_fr_bytes(rejected.data(), expected, "mle_evaluate after rejected calls");
    }

    // build_eq_x_r against the CPU table; null r is rejected.
    {
        const std::vector<RefFr> r{fr(2), fr(5), fr(11)};
        const RefTable expected = reference_eq_table(r);
        std::vector<uint8_t> r_bytes = encode_values(r);
        Ascend910B3DeviceAddress output = &sentinel;
        require_ok(abi::mle_build_eq_x_r(context, r_bytes.data(), 3, &output),
                   "mle_build_eq_x_r");
        require(output != nullptr, "mle_build_eq_x_r returned a null handle");
        check_table(context, output, expected, "mle_build_eq_x_r table");
        require_ok(abi::buffer_retire(context, output), "retire eq table");

        output = &sentinel;
        require_contract_error(abi::mle_build_eq_x_r(context, nullptr, 3, &output),
                               "mle_build_eq_x_r null r");
        require(output == nullptr, "mle_build_eq_x_r null r output");
    }

    // compute_frac_poly: elementwise product ratio; zero denominators fail
    // without poisoning later valid calls, including 0/0.
    {
        const std::array<Ascend910B3DeviceAddress, 2> numerators{dp, dq};
        const std::array<Ascend910B3DeviceAddress, 2> denominators{dg, dh};
        Ascend910B3DeviceAddress fraction = &sentinel;
        require_ok(abi::mle_compute_frac_poly(context, 3, numerators.data(),
                                              denominators.data(), 2, &fraction),
                   "mle_compute_frac_poly");
        require(fraction != nullptr, "mle_compute_frac_poly returned a null handle");
        std::vector<RefFr> expected;
        for (size_t i = 0; i < p.values.size(); ++i) {
            const RefFr numerator = p.values[i].mul(q.values[i]);
            const RefFr denominator = g.values[i].mul(h.values[i]);
            expected.push_back(numerator.mul(denominator.inverse()));
        }
        const RefTable expected_table(3, std::move(expected));
        check_table(context, fraction, expected_table, "mle_compute_frac_poly table");

        const std::array<Ascend910B3DeviceAddress, 1> zero_numerators{dp};
        const std::array<Ascend910B3DeviceAddress, 1> zero_denominators{dzeros};
        Ascend910B3DeviceAddress rejected = &sentinel;
        require_contract_error(abi::mle_compute_frac_poly(context, 3, zero_numerators.data(),
                                                          zero_denominators.data(), 1,
                                                          &rejected),
                               "mle_compute_frac_poly zero denominator");
        require(rejected == nullptr, "frac zero denominator output");

        require_contract_error(abi::mle_compute_frac_poly(context, 3, zero_denominators.data(),
                                                          zero_denominators.data(), 1,
                                                          &rejected),
                               "mle_compute_frac_poly zero over zero");
        require(rejected == nullptr, "frac zero over zero output");

        const std::array<Ascend910B3DeviceAddress, 1> small_denominators{dsmall};
        require_contract_error(abi::mle_compute_frac_poly(context, 3, zero_numerators.data(),
                                                          small_denominators.data(), 1,
                                                          &rejected),
                               "mle_compute_frac_poly mixed dimensions");
        require(rejected == nullptr, "frac mixed dimensions output");

        Ascend910B3DeviceAddress fraction_after = &sentinel;
        require_ok(abi::mle_compute_frac_poly(context, 3, numerators.data(),
                                              denominators.data(), 2, &fraction_after),
                   "mle_compute_frac_poly after errors");
        require(fraction_after != nullptr, "frac after errors returned null");
        check_table(context, fraction_after, expected_table, "mle_compute_frac_poly after errors");
        require_ok(abi::buffer_retire(context, fraction_after), "retire after-error fraction");

        // product recurrence.
        Ascend910B3DeviceAddress product = &sentinel;
        require_ok(abi::mle_compute_product_poly(context, fraction, 3, &product),
                   "mle_compute_product_poly");
        require(product != nullptr, "mle_compute_product_poly returned a null handle");
        const RefTable expected_product = reference_product_table(expected_table);
        check_table(context, product, expected_product, "mle_compute_product_poly table");

        Ascend910B3DeviceAddress rejected_product = &sentinel;
        require_contract_error(abi::mle_compute_product_poly(context, fraction, 0,
                                                             &rejected_product),
                               "mle_compute_product_poly zero variables");
        require(rejected_product == nullptr, "product zero variables output");

        // p1/p2 interleave the frac and product pairs.
        std::vector<RefFr> p1;
        std::vector<RefFr> p2;
        for (const RefTable* table : {&expected_table, &expected_product}) {
            for (size_t i = 0; i < table->values.size(); i += 2) {
                p1.push_back(table->values[i]);
                p2.push_back(table->values[i + 1]);
            }
        }
        const RefTable expected_p1(3, std::move(p1));
        const RefTable expected_p2(3, std::move(p2));
        Ascend910B3DeviceAddress output_p1 = &sentinel;
        Ascend910B3DeviceAddress output_p2 = &sentinel;
        require_ok(abi::mle_compute_product_factors(context, fraction, product, 3,
                                                    &output_p1, &output_p2),
                   "mle_compute_product_factors");
        require(output_p1 != nullptr && output_p2 != nullptr,
                "mle_compute_product_factors returned a null handle");
        check_table(context, output_p1, expected_p1, "mle_compute_product_factors p1");
        check_table(context, output_p2, expected_p2, "mle_compute_product_factors p2");
        require_ok(abi::buffer_retire(context, output_p1), "retire p1");
        require_ok(abi::buffer_retire(context, output_p2), "retire p2");

        Ascend910B3DeviceAddress rejected_p1 = &sentinel;
        Ascend910B3DeviceAddress rejected_p2 = &sentinel;
        require_contract_error(abi::mle_compute_product_factors(context, fraction, dsmall, 3,
                                                                &rejected_p1, &rejected_p2),
                               "mle_compute_product_factors mixed dimensions");
        require(rejected_p1 == nullptr && rejected_p2 == nullptr,
                "product factor dimension error outputs");

        Ascend910B3DeviceAddress after_p1 = &sentinel;
        Ascend910B3DeviceAddress after_p2 = &sentinel;
        require_ok(abi::mle_compute_product_factors(context, fraction, product, 3,
                                                    &after_p1, &after_p2),
                   "mle_compute_product_factors after error");
        check_table(context, after_p1, expected_p1, "product factors after error p1");
        check_table(context, after_p2, expected_p2, "product factors after error p2");
        require_ok(abi::buffer_retire(context, after_p1), "retire after-error p1");
        require_ok(abi::buffer_retire(context, after_p2), "retire after-error p2");

        require_ok(abi::buffer_retire(context, fraction), "retire fraction");
        require_ok(abi::buffer_retire(context, product), "retire product");
    }

    // compute_nums_and_denoms over two columns with beta/gamma and permutation.
    {
        const RefTable f0 = make_table(2, {1, 2, 3, 4});
        const RefTable f1 = make_table(2, {5, 6, 7, 8});
        const RefTable g0 = make_table(2, {2, 3, 4, 5});
        const RefTable g1 = make_table(2, {6, 7, 8, 9});
        const RefTable perm0 = make_table(2, {0, 1, 2, 3});
        const RefTable perm1 = make_table(2, {3, 2, 1, 0});
        const Ascend910B3DeviceAddress df0 = track(upload_table(context, f0));
        const Ascend910B3DeviceAddress df1 = track(upload_table(context, f1));
        const Ascend910B3DeviceAddress dg0 = track(upload_table(context, g0));
        const Ascend910B3DeviceAddress dg1 = track(upload_table(context, g1));
        const Ascend910B3DeviceAddress dperm0 = track(upload_table(context, perm0));
        const Ascend910B3DeviceAddress dperm1 = track(upload_table(context, perm1));

        const RefFr beta = fr(3);
        const RefFr gamma = fr(7);
        const std::array<Ascend910B3DeviceAddress, 2> fxs{df0, df1};
        const std::array<Ascend910B3DeviceAddress, 2> gxs{dg0, dg1};
        const std::array<Ascend910B3DeviceAddress, 2> perms{dperm0, dperm1};
        auto label_bytes = [](const RefFr& value) {
            const auto encoded = value.bytes();
            return std::vector<uint8_t>(encoded.begin(), encoded.end());
        };
        std::vector<uint8_t> beta_bytes = label_bytes(beta);
        std::vector<uint8_t> gamma_bytes = label_bytes(gamma);

        std::array<Ascend910B3DeviceAddress, 2> numerator_out{&sentinel, &sentinel};
        std::array<Ascend910B3DeviceAddress, 2> denominator_out{&sentinel, &sentinel};
        require_ok(abi::mle_compute_nums_and_denoms(context, 2, beta_bytes.data(),
                                                    gamma_bytes.data(), fxs.data(),
                                                    gxs.data(), perms.data(), 2,
                                                    numerator_out.data(),
                                                    denominator_out.data()),
                   "mle_compute_nums_and_denoms");
        require(numerator_out[0] != nullptr && numerator_out[1] != nullptr &&
                    denominator_out[0] != nullptr && denominator_out[1] != nullptr,
                "nums_and_denoms returned null handles");

        const std::array<RefTable, 2> inputs_f{f0, f1};
        const std::array<RefTable, 2> inputs_g{g0, g1};
        const std::array<RefTable, 2> inputs_perm{perm0, perm1};
        for (size_t column = 0; column < 2; ++column) {
            const size_t size = inputs_f[column].values.size();
            RefFr identity_term = beta.mul(fr(column * size));
            std::vector<RefFr> expected_numerator;
            std::vector<RefFr> expected_denominator;
            for (size_t i = 0; i < size; ++i) {
                expected_numerator.push_back(
                    inputs_f[column].values[i].add(identity_term).add(gamma));
                expected_denominator.push_back(
                    inputs_g[column].values[i]
                        .add(beta.mul(inputs_perm[column].values[i]))
                        .add(gamma));
                identity_term = identity_term.add(beta);
            }
            check_table(context, numerator_out[column],
                        RefTable(2, std::move(expected_numerator)),
                        "nums_and_denoms numerator " + std::to_string(column));
            check_table(context, denominator_out[column],
                        RefTable(2, std::move(expected_denominator)),
                        "nums_and_denoms denominator " + std::to_string(column));
        }
        for (size_t column = 0; column < 2; ++column) {
            require_ok(abi::buffer_retire(context, numerator_out[column]),
                       "retire nums numerator");
            require_ok(abi::buffer_retire(context, denominator_out[column]),
                       "retire nums denominator");
        }

        // A dimension mismatch clears every output slot and leaves the
        // subsequent valid path usable.
        const std::array<Ascend910B3DeviceAddress, 2> bad_perms{dperm0, dp};
        numerator_out = std::array<Ascend910B3DeviceAddress, 2>{&sentinel, &sentinel};
        denominator_out = std::array<Ascend910B3DeviceAddress, 2>{&sentinel, &sentinel};
        require_contract_error(abi::mle_compute_nums_and_denoms(context, 2, beta_bytes.data(),
                                                                gamma_bytes.data(), fxs.data(),
                                                                gxs.data(), bad_perms.data(), 2,
                                                                numerator_out.data(),
                                                                denominator_out.data()),
                               "mle_compute_nums_and_denoms mixed dimensions");
        require(numerator_out[0] == nullptr && numerator_out[1] == nullptr &&
                    denominator_out[0] == nullptr && denominator_out[1] == nullptr,
                "nums_and_denoms dimension error outputs not cleared");

        std::vector<uint8_t> bad_beta(32, uint8_t(0xFF));
        numerator_out = std::array<Ascend910B3DeviceAddress, 2>{&sentinel, &sentinel};
        denominator_out = std::array<Ascend910B3DeviceAddress, 2>{&sentinel, &sentinel};
        require_contract_error(abi::mle_compute_nums_and_denoms(context, 2, bad_beta.data(),
                                                                gamma_bytes.data(), fxs.data(),
                                                                gxs.data(), perms.data(), 2,
                                                                numerator_out.data(),
                                                                denominator_out.data()),
                               "mle_compute_nums_and_denoms noncanonical beta");
        require(numerator_out[0] == nullptr && numerator_out[1] == nullptr &&
                    denominator_out[0] == nullptr && denominator_out[1] == nullptr,
                "nums_and_denoms noncanonical error outputs not cleared");

        numerator_out = std::array<Ascend910B3DeviceAddress, 2>{&sentinel, &sentinel};
        denominator_out = std::array<Ascend910B3DeviceAddress, 2>{&sentinel, &sentinel};
        require_ok(abi::mle_compute_nums_and_denoms(context, 2, beta_bytes.data(),
                                                    gamma_bytes.data(), fxs.data(),
                                                    gxs.data(), perms.data(), 2,
                                                    numerator_out.data(),
                                                    denominator_out.data()),
                   "mle_compute_nums_and_denoms after errors");
        require(numerator_out[0] != nullptr && numerator_out[1] != nullptr &&
                    denominator_out[0] != nullptr && denominator_out[1] != nullptr,
                "nums_and_denoms after errors returned null");
        require_ok(abi::buffer_retire(context, numerator_out[0]), "retire numerator 0");
        require_ok(abi::buffer_retire(context, numerator_out[1]), "retire numerator 1");
        require_ok(abi::buffer_retire(context, denominator_out[0]), "retire denominator 0");
        require_ok(abi::buffer_retire(context, denominator_out[1]), "retire denominator 1");
    }

    // Multi-tile public-index regression across whole 2048/4096-entry tables.
    // On 40 AIV blocks each block covers 52/103 rows, so every block spans more
    // than one 48-lane tile. The full-range fixture above makes only the exact
    // per-tile CPU index beta*(column*total + row) produce NUM=8 and
    // DEN=7+2*x at non-Boolean first coordinates; a carried or stale base
    // index in any later tile changes those values. The oracle is the closed
    // form evaluated with the independent host Fr helpers, never the device
    // output. beta = -1 and gamma = -2 are canonical BLS12-381 Fr constants.
    for (const uint32_t num_vars : {uint32_t(11), uint32_t(12)}) {
        constexpr size_t kColumns = 2;
        std::vector<RefTable> fx;
        std::vector<RefTable> perm;
        reference_index_fixture(num_vars, uint32_t(kColumns), fx, perm);

        const RefFr beta = RefFr::zero().sub(RefFr::one());
        const RefFr gamma = RefFr::zero().sub(fr(2));
        const auto beta_encoded = beta.bytes();
        const auto gamma_encoded = gamma.bytes();

        std::array<Ascend910B3DeviceAddress, kColumns> fxs{};
        std::array<Ascend910B3DeviceAddress, kColumns> gxs{};
        std::array<Ascend910B3DeviceAddress, kColumns> perms{};
        for (size_t column = 0; column < kColumns; ++column) {
            fxs[column] = track(upload_table(context, fx[column]));
            gxs[column] = fxs[column];
            perms[column] = track(upload_table(context, perm[column]));
        }

        std::array<Ascend910B3DeviceAddress, kColumns> numerator_out{};
        std::array<Ascend910B3DeviceAddress, kColumns> denominator_out{};
        numerator_out.fill(&sentinel);
        denominator_out.fill(&sentinel);
        require_ok(abi::mle_compute_nums_and_denoms(
                       context, num_vars, beta_encoded.data(), gamma_encoded.data(),
                       fxs.data(), gxs.data(), perms.data(), kColumns,
                       numerator_out.data(), denominator_out.data()),
                   "mle_compute_nums_and_denoms multi-tile " +
                       std::to_string(num_vars));
        for (size_t column = 0; column < kColumns; ++column)
            require(numerator_out[column] != nullptr &&
                        denominator_out[column] != nullptr,
                    "nums_and_denoms multi-tile returned null handles");

        for (const uint64_t first : {uint64_t(2), uint64_t(3)}) {
            std::vector<RefFr> point(num_vars);
            point[0] = fr(first);
            for (uint32_t coordinate = 1; coordinate < num_vars; ++coordinate)
                point[coordinate] = fr(uint64_t(3) + coordinate);
            const RefFr expected_numerator = fr(8);
            const RefFr expected_denominator = fr(7).add(fr(2).mul(point[0]));
            for (size_t column = 0; column < kColumns; ++column) {
                const std::string label =
                    "nums_and_denoms multi-tile " + std::to_string(num_vars) +
                    " column " + std::to_string(column) + " x=" +
                    std::to_string(first);
                require(evaluate_checked(context, numerator_out[column], point,
                                         label + " numerator")
                            .equals(expected_numerator),
                        label + " numerator");
                require(evaluate_checked(context, denominator_out[column], point,
                                         label + " denominator")
                            .equals(expected_denominator),
                        label + " denominator");
            }
        }
        for (size_t column = 0; column < kColumns; ++column) {
            require_ok(abi::buffer_retire(context, numerator_out[column]),
                       "retire multi-tile numerator");
            require_ok(abi::buffer_retire(context, denominator_out[column]),
                       "retire multi-tile denominator");
        }
    }

    // Retirement is asynchronous and queued: no synchronization, unknown and
    // repeated addresses are rejected, and a retired handle is no longer a
    // valid input.
    {
        const RefTable temp = make_table(2, {1, 1, 2, 3});
        const Ascend910B3DeviceAddress handle = upload_table(context, temp);
        audit::begin_runtime_audit();
        const Ascend910B3Status retired = abi::buffer_retire(context, handle);
        const RuntimeCounts counts = audit::end_runtime_audit();
        require_ok(retired, "buffer_retire");
        require(counts.h2d_calls == 0 && counts.h2d_bytes == 0 && counts.d2h_calls == 0 &&
                    counts.d2h_bytes == 0 && counts.other_copy_calls == 0 &&
                    counts.stream_sync_calls == 0 && counts.device_sync_calls == 0,
                "buffer_retire synchronized or copied");

        std::vector<uint8_t> point_bytes = encode_values({fr(3), fr(4)});
        std::array<uint8_t, 32> output{};
        require_contract_error(abi::mle_evaluate(context, handle, 2, point_bytes.data(),
                                                 output.data()),
                               "mle_evaluate retired handle");
        require_contract_error(abi::buffer_retire(context, handle),
                               "buffer_retire repeated");
        require_contract_error(abi::buffer_retire(context, &sentinel),
                               "buffer_retire unknown address");
        require_contract_error(abi::buffer_retire(context, nullptr),
                               "buffer_retire null address");
        const std::vector<RefFr> point{fr(2), fr(5), fr(11)};
        require(evaluate_checked(context, dp, point, "mle_evaluate after retirement")
                    .equals(reference_evaluate(p, point)),
                "mle_evaluate after retirement mismatch");
    }

    for (Ascend910B3DeviceAddress address : owned)
        require_ok(abi::buffer_retire(context, address), "retire owned MLE table");
    std::cout << "BRIDGE_MLE_PASS\n";
}

// ---------------------------------------------------------------------------
// Async metadata and input lifetime across a dependency DAG.
//
// Everything below is submitted without any intermediate observation: uploads,
// linear combinations and retirements only enqueue work. Each intermediate
// public handle is retired immediately after its last consumer was submitted,
// so its device table, its control metadata and the pinned staging used by its
// asynchronous copies become reuse candidates while later kernels are still
// queued. A correct runtime freezes that reuse until the copy/kernel that last
// referenced the storage really completed; a runtime that reclaims on a bare
// stream-complete query lets the next enqueued operation overwrite the stale
// metadata or reads a recycled table, and the final table below is wrong.
//
// The chain mixes arity 2 and arity 1 combinations over full-width canonical
// coefficients. Table values and coefficients are all distinct, so swapping one
// pointer or one coefficient cannot reproduce the expected values. The host
// oracle is the documented elementwise CPU semantics over the same base tables,
// evaluated independently with the reference field arithmetic.
// ---------------------------------------------------------------------------

void test_async_lifetime(Ascend910B3NativeContext* context) {
    uint8_t sentinel = 0;
    constexpr size_t kNumVars = 3;

    const RefTable p0(kNumVars, {fr_spread(101), fr_spread(102), fr_spread(103),
                                 fr_spread(104), fr_spread(105), fr_spread(106),
                                 fr_spread(107), fr_spread(108)});
    const RefTable p1(kNumVars, {fr_spread(201), fr_spread(202), fr_spread(203),
                                 fr_spread(204), fr_spread(205), fr_spread(206),
                                 fr_spread(207), fr_spread(208)});
    const RefTable p2(kNumVars, {fr_spread(301), fr_spread(302), fr_spread(303),
                                 fr_spread(304), fr_spread(305), fr_spread(306),
                                 fr_spread(307), fr_spread(308)});

    // Full-width, pairwise distinct canonical coefficients.
    const RefFr k0 = fr_minus_one();
    const RefFr k1 = fr_spread(0);
    const RefFr k2{fr_norm(kFrModulus - 0x123456789ULL)};
    const RefFr k3 = fr_spread(1);
    const RefFr k4 = fr_spread(2);
    const RefFr k5 = fr_spread(3);
    const RefFr k6 = fr_spread(4);
    const RefFr k7 = fr_spread(5);
    const RefFr k8 = fr_spread(6);
    const RefFr k9 = fr_spread(7);
    const RefFr k10 = fr_spread(8);
    const RefFr k11 = fr_spread(9);
    const RefFr k12 = fr_spread(10);

    auto combine = [](const RefTable& left, const RefTable& right,
                      const RefFr& left_coefficient, const RefFr& right_coefficient) {
        require(left.num_vars == right.num_vars &&
                    left.values.size() == right.values.size(),
                "async lifetime reference combine shape");
        std::vector<RefFr> values;
        values.reserve(left.values.size());
        for (size_t i = 0; i < left.values.size(); ++i)
            values.push_back(left_coefficient.mul(left.values[i])
                                 .add(right_coefficient.mul(right.values[i])));
        return RefTable(left.num_vars, std::move(values));
    };
    auto scale = [](const RefTable& input, const RefFr& coefficient) {
        std::vector<RefFr> values;
        values.reserve(input.values.size());
        for (const RefFr& value : input.values)
            values.push_back(coefficient.mul(value));
        return RefTable(input.num_vars, std::move(values));
    };

    auto linear = [&](std::initializer_list<Ascend910B3DeviceAddress> inputs,
                      std::initializer_list<RefFr> coefficients,
                      const std::string& what) {
        std::vector<Ascend910B3DeviceAddress> polynomials(inputs);
        const std::vector<RefFr> coefficient_values(coefficients);
        std::vector<uint8_t> coefficient_bytes = encode_values(coefficient_values);
        Ascend910B3DeviceAddress output = &sentinel;
        require_ok(abi::mle_linear_combination(context, kNumVars, polynomials.data(),
                                               coefficient_bytes.data(),
                                               coefficient_values.size(), &output),
                   what);
        require(output != nullptr, what + " returned a null handle");
        // The bridge must have copied the borrowed bytes before returning.
        std::fill(coefficient_bytes.begin(), coefficient_bytes.end(), uint8_t(0x5A));
        return output;
    };
    auto retire = [&](Ascend910B3DeviceAddress address, const std::string& what) {
        require_ok(abi::buffer_retire(context, address), what);
    };

    // Independent CPU reference DAG, fully evaluated before the first device
    // upload. Host field arithmetic must not run inside the asynchronous submit
    // sequence below, where it could delay enqueue timing and mask a reuse race.
    const RefTable rA = combine(p0, p1, k0, k1);
    const RefTable rB = scale(p2, k2);
    const RefTable rC = combine(rA, rB, k3, k4);
    const RefTable rD = scale(rA, k5);
    const RefTable rE = combine(rC, rD, k6, k7);
    const RefTable rF = scale(rE, k8);
    const RefTable rG = combine(rF, p0, k9, k10);
    const RefTable rFinal = combine(rG, p1, k11, k12);

    const Ascend910B3DeviceAddress dp0 = upload_table(context, p0);
    const Ascend910B3DeviceAddress dp1 = upload_table(context, p1);
    const Ascend910B3DeviceAddress dp2 = upload_table(context, p2);

    // hA <- p0,p1 (arity 2); hB <- p2 (arity 1). Only the coefficients differ
    // from the later arity-2 metadata blocks, so a stale staging block is
    // observable in the final table.
    const Ascend910B3DeviceAddress hA = linear({dp0, dp1}, {k0, k1}, "async lifetime A");
    const Ascend910B3DeviceAddress hB = linear({dp2}, {k2}, "async lifetime B");
    // hB already consumed p2; no later consumer exists.
    retire(dp2, "retire base p2 after B");

    // hC consumes hA and hB. Retire hB only after this submission, then build
    // hD from hA and retire hA after hD is queued.
    const Ascend910B3DeviceAddress hC = linear({hA, hB}, {k3, k4}, "async lifetime C");
    retire(hB, "retire B after C");

    const Ascend910B3DeviceAddress hD = linear({hA}, {k5}, "async lifetime D");
    retire(hA, "retire A after D");

    // hE consumes the interleaved branches; both intermediates are retired
    // after this call while hE's kernel still has to read them.
    const Ascend910B3DeviceAddress hE = linear({hC, hD}, {k6, k7}, "async lifetime E");
    retire(hC, "retire C after E");
    retire(hD, "retire D after E");

    const Ascend910B3DeviceAddress hF = linear({hE}, {k8}, "async lifetime F");
    retire(hE, "retire E after F");

    // p0 is still needed here; p1 only appears in the final consumer.
    const Ascend910B3DeviceAddress hG = linear({hF, dp0}, {k9, k10}, "async lifetime G");
    retire(hF, "retire F after G");

    const Ascend910B3DeviceAddress hFinal = linear({hG, dp1}, {k11, k12}, "async lifetime final");
    retire(hG, "retire G after final");

    // First and only synchronization point: observe the actual final consumer.
    check_table(context, hFinal, rFinal, "async lifetime final table");
    const std::vector<RefFr> probe{fr_spread(900), fr_minus_one(), fr(123456789)};
    require(evaluate_checked(context, hFinal, probe, "async lifetime nonzero probe")
                .equals(reference_evaluate(rFinal, probe)),
            "async lifetime nonzero probe mismatch");

    retire(hFinal, "retire final after observation");
    retire(dp0, "retire base p0 after final");
    retire(dp1, "retire base p1 after final");
    std::cout << "BRIDGE_ASYNC_LIFETIME_PASS\n";
}

// ---------------------------------------------------------------------------
// KZG preparation, commit, multi-commit and open.
// ---------------------------------------------------------------------------

struct RefOpen {
    std::vector<RefPoint> quotients;
    RefFr evaluation;
};

RefPoint reference_commit(const std::vector<RefPoint>& bases, const RefTable& polynomial) {
    require(bases.size() == polynomial.values.size(), "reference commit size");
    return sum_points(bases, polynomial.values);
}

RefOpen reference_open(const std::vector<std::vector<RefPoint>>& layers, size_t num_vars,
                       const RefTable& polynomial, const std::vector<RefFr>& point) {
    require(polynomial.num_vars <= num_vars, "reference open dimension");
    require(point.size() == polynomial.num_vars, "reference open point size");
    RefOpen result;
    std::vector<RefFr> current = polynomial.values;
    for (size_t round = 0; round < polynomial.num_vars; ++round) {
        const size_t half = current.size() / 2;
        std::vector<RefFr> quotients(half);
        for (size_t i = 0; i < half; ++i)
            quotients[i] = current[2 * i + 1].sub(current[2 * i]);
        const std::vector<RefPoint>& bases =
            layers[num_vars - polynomial.num_vars + 1 + round];
        require(bases.size() == half, "reference open layer size");
        result.quotients.push_back(sum_points(bases, quotients));
        for (size_t i = 0; i < half; ++i)
            current[i] = current[2 * i].add(quotients[i].mul(point[round]));
        current.resize(half);
    }
    result.evaluation = current[0];
    return result;
}

void test_kzg(Ascend910B3NativeContext* context, Ascend910B3NativeContext* other) {
    constexpr size_t kNumVars = 3;
    uint8_t sentinel = 0;
    std::vector<Ascend910B3DeviceAddress> owned;
    auto track = [&](Ascend910B3DeviceAddress address) {
        owned.push_back(address);
        return address;
    };

    // Fixed valid generator-point SRS layers: layer i has 2^(N-i) bases.
    std::vector<std::vector<RefPoint>> layers;
    for (size_t layer = 0; layer <= kNumVars; ++layer) {
        const size_t count = size_t(1) << (kNumVars - layer);
        std::vector<RefPoint> points;
        points.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const uint64_t multiplier = 1 + 32 * layer + i;
            points.push_back(scalar_mul_point(kGenerator, fr(multiplier)));
        }
        layers.push_back(std::move(points));
    }

    // A degree beyond the representable table dimension is rejected, and the
    // resource slot is cleared.
    {
        auto* rejected = reinterpret_cast<Ascend910B3NativeMultilinearKzgPreparedProverParam*>(
            &sentinel);
        require_contract_error(abi::multilinear_kzg_params_create(context, 64, &rejected),
                               "multilinear_kzg_params_create oversized");
        require(rejected == nullptr, "params_create oversized output");
    }

    Ascend910B3NativeMultilinearKzgPreparedProverParam* params = nullptr;
    require_ok(abi::multilinear_kzg_params_create(context, kNumVars, &params),
               "multilinear_kzg_params_create");
    require(params != nullptr, "params_create returned null");

    for (size_t layer = 0; layer <= kNumVars; ++layer) {
        std::vector<uint8_t> packets;
        packets.reserve(layers[layer].size() * 97);
        for (const RefPoint& point : layers[layer]) {
            const auto packet = point_packet(point);
            packets.insert(packets.end(), packet.begin(), packet.end());
        }
        require_ok(abi::multilinear_kzg_params_upload_layer(context, params, layer,
                                                            packets.data(),
                                                            layers[layer].size()),
                   "multilinear_kzg_params_upload_layer");
        std::fill(packets.begin(), packets.end(), uint8_t(0xA5));
    }

    // Invalid layer metadata is rejected without damaging prepared layers.
    {
        const auto packet = point_packet(kGenerator);
        require_contract_error(abi::multilinear_kzg_params_upload_layer(
                                   context, params, kNumVars + 1, packet.data(), 1),
                               "params_upload_layer bad layer index");
        require_contract_error(abi::multilinear_kzg_params_upload_layer(
                                   context, params, 0, packet.data(), 1),
                               "params_upload_layer bad point count");

        // Re-uploading an already published layer is a state error but must
        // leave the prepared cache fully usable (checked by commit/open below).
        std::vector<uint8_t> duplicate;
        duplicate.reserve(layers[0].size() * 97);
        for (const RefPoint& point : layers[0]) {
            const auto encoded = point_packet(point);
            duplicate.insert(duplicate.end(), encoded.begin(), encoded.end());
        }
        require_contract_error(abi::multilinear_kzg_params_upload_layer(
                                   context, params, 0, duplicate.data(), layers[0].size()),
                               "params_upload_layer duplicate layer");
    }

    const RefTable p3 = make_table(3, {1, 2, 3, 4, 5, 6, 7, 8});
    const RefTable p2 = make_table(2, {2, 5, 7, 11});
    const RefTable p0 = make_table(0, {42});
    const RefTable p4 = make_table(4, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    const Ascend910B3DeviceAddress dp3 = track(upload_table(context, p3));
    const Ascend910B3DeviceAddress dp2 = track(upload_table(context, p2));
    const Ascend910B3DeviceAddress dp0 = track(upload_table(context, p0));
    const Ascend910B3DeviceAddress dp4 = track(upload_table(context, p4));

    // Commit, including a constant polynomial, under a small-D2H audit.
    {
        std::array<uint8_t, 97> commitment{};
        audit::begin_runtime_audit();
        const Ascend910B3Status status =
            abi::multilinear_kzg_commit(context, params, dp3, 3, commitment.data());
        const RuntimeCounts counts = audit::end_runtime_audit();
        require_ok(status, "multilinear_kzg_commit");
        compare_packet(commitment.data(), reference_commit(layers[0], p3), "kzg commit p3");
        expect_small_io(counts, 97, 0, true, "multilinear_kzg_commit");

        std::array<uint8_t, 97> constant_commitment{};
        audit::begin_runtime_audit();
        const Ascend910B3Status constant_status =
            abi::multilinear_kzg_commit(context, params, dp0, 0, constant_commitment.data());
        const RuntimeCounts constant_counts = audit::end_runtime_audit();
        require_ok(constant_status, "multilinear_kzg_commit constant");
        compare_packet(constant_commitment.data(), reference_commit(layers[3], p0),
                       "kzg commit constant");
        expect_small_io(constant_counts, 97, 0, true, "multilinear_kzg_commit constant");

        require_contract_error(abi::multilinear_kzg_commit(context, params, dp3, 2,
                                                           commitment.data()),
                               "multilinear_kzg_commit wrong dimension");
        require_contract_error(abi::multilinear_kzg_commit(context, params, dp4, 4,
                                                           commitment.data()),
                               "multilinear_kzg_commit beyond params");

        const RefTable foreign(3, p3.values);
        Ascend910B3DeviceAddress foreign_handle = upload_table(other, foreign);
        require_contract_error(abi::multilinear_kzg_commit(context, params, foreign_handle, 3,
                                                           commitment.data()),
                               "multilinear_kzg_commit foreign polynomial");
        require_contract_error(abi::multilinear_kzg_commit(other, params, dp3, 3,
                                                           commitment.data()),
                               "multilinear_kzg_commit foreign context");
        require_ok(abi::buffer_retire(other, foreign_handle), "retire foreign commit handle");

        require_ok(abi::multilinear_kzg_commit(context, params, dp3, 3, commitment.data()),
                   "multilinear_kzg_commit after errors");
        compare_packet(commitment.data(), reference_commit(layers[0], p3),
                       "kzg commit after errors");
    }

    // Multi-commit preserves order and duplicates.
    {
        const std::array<Ascend910B3DeviceAddress, 4> polynomials{dp3, dp0, dp2, dp3};
        const std::array<size_t, 4> dimensions{3, 0, 2, 3};
        std::vector<uint8_t> commitments(4 * 97);
        audit::begin_runtime_audit();
        const Ascend910B3Status status =
            abi::multilinear_kzg_multi_commit(context, params, polynomials.data(),
                                              dimensions.data(), 4, commitments.data());
        const RuntimeCounts counts = audit::end_runtime_audit();
        require_ok(status, "multilinear_kzg_multi_commit");
        expect_small_io(counts, 4 * 97, 0, true, "multilinear_kzg_multi_commit");
        const std::array<RefPoint, 4> expected{
            reference_commit(layers[0], p3),
            reference_commit(layers[3], p0),
            reference_commit(layers[1], p2),
            reference_commit(layers[0], p3)};
        for (size_t i = 0; i < expected.size(); ++i)
            compare_packet(commitments.data() + i * 97, expected[i],
                           "kzg multi_commit " + std::to_string(i));
    }

    // Open: quotient packets and evaluation against the same direct CPU path.
    {
        const std::vector<RefFr> point{fr(2), fr(5), fr(11)};
        const RefOpen expected = reference_open(layers, kNumVars, p3, point);
        std::vector<uint8_t> quotients(3 * 97);
        std::array<uint8_t, 32> evaluation{};
        std::vector<uint8_t> point_bytes = encode_values(point);
        audit::begin_runtime_audit();
        const Ascend910B3Status status =
            abi::multilinear_kzg_open(context, params, dp3, 3, point_bytes.data(),
                                      quotients.data(), evaluation.data());
        const RuntimeCounts counts = audit::end_runtime_audit();
        require_ok(status, "multilinear_kzg_open");
        for (size_t i = 0; i < expected.quotients.size(); ++i)
            compare_packet(quotients.data() + i * 97, expected.quotients[i],
                           "kzg open quotient " + std::to_string(i));
        compare_fr_bytes(evaluation.data(), expected.evaluation, "kzg open evaluation");
        expect_small_io(counts, 3 * 97 + 32, 3 * 32, true, "multilinear_kzg_open");

        const std::vector<RefFr> point2{fr(7), fr(3)};
        const RefOpen expected2 = reference_open(layers, kNumVars, p2, point2);
        std::vector<uint8_t> quotients2(2 * 97);
        std::array<uint8_t, 32> evaluation2{};
        std::vector<uint8_t> point2_bytes = encode_values(point2);
        require_ok(abi::multilinear_kzg_open(context, params, dp2, 2, point2_bytes.data(),
                                             quotients2.data(), evaluation2.data()),
                   "multilinear_kzg_open p2");
        for (size_t i = 0; i < expected2.quotients.size(); ++i)
            compare_packet(quotients2.data() + i * 97, expected2.quotients[i],
                           "kzg open p2 quotient " + std::to_string(i));
        compare_fr_bytes(evaluation2.data(), expected2.evaluation, "kzg open p2 evaluation");

        // Constant open has no quotients and returns only the evaluation.
        std::array<uint8_t, 32> constant_evaluation{};
        audit::begin_runtime_audit();
        const Ascend910B3Status constant_status =
            abi::multilinear_kzg_open(context, params, dp0, 0, nullptr, nullptr,
                                      constant_evaluation.data());
        const RuntimeCounts constant_counts = audit::end_runtime_audit();
        require_ok(constant_status, "multilinear_kzg_open constant");
        compare_fr_bytes(constant_evaluation.data(), p0.values[0],
                         "kzg open constant evaluation");
        expect_small_io(constant_counts, 32, 0, true, "multilinear_kzg_open constant");

        require_contract_error(abi::multilinear_kzg_open(context, params, dp3, 3, nullptr,
                                                         quotients.data(), evaluation.data()),
                               "multilinear_kzg_open null point");
        std::vector<uint8_t> point4_bytes = encode_values({fr(1), fr(2), fr(3), fr(4)});
        std::vector<uint8_t> quotients4(4 * 97);
        require_contract_error(abi::multilinear_kzg_open(context, params, dp4, 4,
                                                         point4_bytes.data(), quotients4.data(),
                                                         evaluation.data()),
                               "multilinear_kzg_open beyond params");
        require_ok(abi::multilinear_kzg_open(context, params, dp3, 3, point_bytes.data(),
                                             quotients.data(), evaluation.data()),
                   "multilinear_kzg_open after errors");
        compare_fr_bytes(evaluation.data(), expected.evaluation, "kzg open after errors");
    }

    require_ok(abi::multilinear_kzg_params_destroy(context, params),
               "multilinear_kzg_params_destroy");
    for (Ascend910B3DeviceAddress address : owned)
        require_ok(abi::buffer_retire(context, address), "retire KZG MLE table");
    std::cout << "BRIDGE_KZG_PASS\n";
}

// ---------------------------------------------------------------------------
// Sumcheck prover state and generic high-degree rounds.
// ---------------------------------------------------------------------------

struct RefProduct {
    RefFr coefficient;
    std::vector<size_t> indices;
};

struct RefVirtualPolynomial {
    size_t num_vars = 0;
    size_t degree = 0;
    std::vector<std::vector<RefFr>> factors;
    std::vector<RefProduct> products;
};

std::vector<RefFr> reference_round(const std::vector<std::vector<RefFr>>& tables,
                                   const std::vector<RefProduct>& products,
                                   size_t degree) {
    std::vector<RefFr> message(degree + 1, RefFr::zero());
    if (tables.empty()) return message;
    const size_t pairs = tables[0].size() / 2;
    for (size_t x = 0; x <= degree; ++x) {
        const RefFr at = fr(x);
        RefFr total = RefFr::zero();
        for (size_t u = 0; u < pairs; ++u) {
            for (const RefProduct& product : products) {
                RefFr term = product.coefficient;
                for (size_t index : product.indices) {
                    const std::vector<RefFr>& table = tables[index];
                    const RefFr even = table[2 * u];
                    const RefFr odd = table[2 * u + 1];
                    term = term.mul(even.add(at.mul(odd.sub(even))));
                }
                total = total.add(term);
            }
        }
        message[x] = total;
    }
    return message;
}

void fold_tables(std::vector<std::vector<RefFr>>& tables, const RefFr& challenge) {
    for (std::vector<RefFr>& table : tables) {
        std::vector<RefFr> folded(table.size() / 2);
        for (size_t i = 0; i < folded.size(); ++i) {
            const RefFr even = table[2 * i];
            folded[i] = even.add(challenge.mul(table[2 * i + 1].sub(even)));
        }
        table.swap(folded);
    }
}

std::vector<std::vector<RefFr>> reference_messages(
    const RefVirtualPolynomial& polynomial, const std::vector<RefFr>& challenges) {
    std::vector<std::vector<RefFr>> tables = polynomial.factors;
    std::vector<std::vector<RefFr>> messages;
    messages.reserve(polynomial.num_vars);
    for (size_t round = 0; round < polynomial.num_vars; ++round) {
        messages.push_back(reference_round(tables, polynomial.products, polynomial.degree));
        if (round + 1 < polynomial.num_vars) fold_tables(tables, challenges[round]);
    }
    return messages;
}

std::vector<RefProduct> make_products(size_t degree) {
    std::vector<RefProduct> products;
    products.push_back({fr(3), {0}});
    products.push_back({fr(5), {1}});
    if (degree >= 2) products.push_back({fr(7), {0, 1}});
    if (degree >= 3) {
        products.push_back({fr(2), {0, 0, 1}});
        products.push_back({fr(4), {0, 1, 2}});
        products.push_back({fr(0), {2}});
    }
    if (degree >= 7) {
        products.push_back({fr(11), {0, 1, 2, 3, 0}});
        products.push_back({fr(13), {3, 3, 2, 1, 0, 1, 2}});
    }
    if (degree >= 34) {
        std::vector<size_t> indices;
        for (size_t i = 0; i < degree; ++i) indices.push_back(i % 4);
        products.push_back({fr(17), std::move(indices)});
    }
    return products;
}

void encode_product_payload(const RefVirtualPolynomial& polynomial,
                            std::vector<uint8_t>& coefficients,
                            std::vector<size_t>& offsets,
                            std::vector<size_t>& indices) {
    coefficients.clear();
    offsets.clear();
    indices.clear();
    offsets.push_back(0);
    for (const RefProduct& product : polynomial.products) {
        const auto encoded = product.coefficient.bytes();
        coefficients.insert(coefficients.end(), encoded.begin(), encoded.end());
        indices.insert(indices.end(), product.indices.begin(), product.indices.end());
        offsets.push_back(indices.size());
    }
}

Ascend910B3NativeSumCheckProverState* create_state(
    Ascend910B3NativeContext* context, const RefVirtualPolynomial& polynomial,
    const std::vector<Ascend910B3DeviceAddress>& factors) {
    std::vector<uint8_t> coefficients;
    std::vector<size_t> offsets;
    std::vector<size_t> indices;
    encode_product_payload(polynomial, coefficients, offsets, indices);
    Ascend910B3NativeSumCheckProverState* state = nullptr;
    require_ok(abi::sumcheck_prover_state_create(
                   context, polynomial.num_vars, polynomial.degree,
                   factors.empty() ? nullptr : factors.data(), factors.size(),
                   coefficients.empty() ? nullptr : coefficients.data(), offsets.data(),
                   polynomial.products.size(),
                   indices.empty() ? nullptr : indices.data(), indices.size(), &state),
               "sumcheck_prover_state_create");
    require(state != nullptr, "sumcheck_prover_state_create returned null");
    return state;
}

void compare_message(const std::vector<uint8_t>& actual,
                     const std::vector<RefFr>& expected, const std::string& what) {
    require(actual.size() == expected.size() * 32, what + " message size");
    for (size_t i = 0; i < expected.size(); ++i) {
        const auto encoded = expected[i].bytes();
        require(std::memcmp(actual.data() + i * 32, encoded.data(), 32) == 0,
                what + " evaluation " + std::to_string(i));
    }
}

void run_round_checked(Ascend910B3NativeContext* context,
                       Ascend910B3NativeSumCheckProverState* state,
                       const RefVirtualPolynomial& polynomial, size_t round,
                       const std::vector<RefFr>& challenges,
                       const std::vector<std::vector<RefFr>>& expected,
                       const std::string& what) {
    std::vector<uint8_t> output((polynomial.degree + 1) * 32);
    std::array<uint8_t, 32> challenge{};
    const uint8_t* challenge_pointer = nullptr;
    if (round > 0) {
        challenge = challenges[round - 1].bytes();
        challenge_pointer = challenge.data();
    }
    require_ok(abi::sumcheck_prover_state_round(context, state, challenge_pointer,
                                                output.data(), polynomial.degree + 1),
               what);
    compare_message(output, expected[round], what);
}

void test_sumcheck(Ascend910B3NativeContext* context, Ascend910B3NativeContext* other) {
    constexpr size_t kNumVars = 3;
    uint8_t sentinel = 0;

    const RefTable f0 = make_table(3, {1, 2, 3, 4, 5, 6, 7, 8});
    const RefTable f1 = make_table(3, {9, 4, 2, 7, 1, 5, 3, 6});
    const RefTable f2 = make_table(3, {2, 2, 3, 3, 5, 5, 7, 7});
    const RefTable f3 = make_table(3, {11, 13, 17, 19, 23, 29, 31, 37});
    std::vector<Ascend910B3DeviceAddress> factors{
        upload_table(context, f0), upload_table(context, f1),
        upload_table(context, f2), upload_table(context, f3)};
    bool factors_retired = false;

    check_table(context, factors[0], f0, "sumcheck factor 0");
    check_table(context, factors[1], f1, "sumcheck factor 1");
    check_table(context, factors[2], f2, "sumcheck factor 2");
    check_table(context, factors[3], f3, "sumcheck factor 3");

    const std::vector<std::vector<RefFr>> factor_values{
        f0.values, f1.values, f2.values, f3.values};
    const std::vector<RefFr> challenges{fr(1000), fr(2000)};

    // Degree 65 crosses the 48-point message chunk boundary. Degree 2 also audits I/O.
    for (size_t degree : {size_t(1), size_t(2), size_t(3), size_t(7), size_t(65)}) {
        const RefVirtualPolynomial polynomial{
            kNumVars, degree, factor_values, make_products(degree)};
        const std::vector<std::vector<RefFr>> expected =
            reference_messages(polynomial, challenges);
        Ascend910B3NativeSumCheckProverState* state =
            create_state(context, polynomial, factors);
        for (size_t round = 0; round < kNumVars; ++round) {
            if (degree == 2 && round < 2) {
                std::vector<uint8_t> output((degree + 1) * 32);
                std::array<uint8_t, 32> challenge{};
                const uint8_t* challenge_pointer = nullptr;
                if (round > 0) {
                    challenge = challenges[round - 1].bytes();
                    challenge_pointer = challenge.data();
                }
                audit::begin_runtime_audit();
                const Ascend910B3Status status = abi::sumcheck_prover_state_round(
                    context, state, challenge_pointer, output.data(), degree + 1);
                const RuntimeCounts counts = audit::end_runtime_audit();
                require_ok(status, "sumcheck round audited");
                compare_message(output, expected[round], "sumcheck round audited");
                expect_small_io(counts, (degree + 1) * 32, 32, true,
                                "sumcheck round audited");
            } else {
                run_round_checked(context, state, polynomial, round, challenges, expected,
                                  "sumcheck round");
            }
        }
        require_ok(abi::sumcheck_prover_state_destroy(context, state),
                   "sumcheck_prover_state_destroy");
    }

    // Ordering errors never advance the state: the next legitimate call must
    // match the expected message for that round index.
    {
        const RefVirtualPolynomial polynomial{3, 3, factor_values, make_products(3)};
        const std::vector<std::vector<RefFr>> expected =
            reference_messages(polynomial, challenges);
        Ascend910B3NativeSumCheckProverState* state =
            create_state(context, polynomial, factors);
        std::vector<uint8_t> output((polynomial.degree + 1) * 32);
        std::array<uint8_t, 32> challenge{};

        challenge = challenges[0].bytes();
        require_contract_error(abi::sumcheck_prover_state_round(
                                   context, state, challenge.data(), output.data(),
                                   polynomial.degree + 1),
                               "sumcheck first round with challenge");
        run_round_checked(context, state, polynomial, 0, challenges, expected,
                          "sumcheck after first-round rejection");

        require_contract_error(abi::sumcheck_prover_state_round(
                                   context, state, nullptr, output.data(),
                                   polynomial.degree + 1),
                               "sumcheck missing challenge");
        std::vector<uint8_t> noncanonical_challenge(32, uint8_t(0xFF));
        require_contract_error(abi::sumcheck_prover_state_round(
                                   context, state, noncanonical_challenge.data(), output.data(),
                                   polynomial.degree + 1),
                               "sumcheck noncanonical challenge");
        run_round_checked(context, state, polynomial, 1, challenges, expected,
                          "sumcheck after challenge rejections");

        challenge = challenges[1].bytes();
        std::vector<uint8_t> short_output(2 * 32);
        require_contract_error(abi::sumcheck_prover_state_round(
                                   context, state, challenge.data(), short_output.data(), 2),
                               "sumcheck wrong evaluation count");
        require_contract_error(abi::sumcheck_prover_state_round(
                                   other, state, challenge.data(), output.data(),
                                   polynomial.degree + 1),
                               "sumcheck foreign context");
        require_contract_error(abi::sumcheck_prover_state_round(
                                   context, state, challenge.data(), nullptr,
                                   polynomial.degree + 1),
                               "sumcheck null output");
        run_round_checked(context, state, polynomial, 2, challenges, expected,
                          "sumcheck after advance errors");

        require_contract_error(abi::sumcheck_prover_state_round(
                                   context, state, nullptr, output.data(),
                                   polynomial.degree + 1),
                               "sumcheck exhausted with null");
        challenge = challenges[0].bytes();
        require_contract_error(abi::sumcheck_prover_state_round(
                                   context, state, challenge.data(), output.data(),
                                   polynomial.degree + 1),
                               "sumcheck exhausted with challenge");
        require_contract_error(abi::sumcheck_prover_state_round(
                                   context, nullptr, nullptr, output.data(),
                                   polynomial.degree + 1),
                               "sumcheck null state");
        require_ok(abi::sumcheck_prover_state_destroy(context, state),
                   "destroy sumcheck error state");
    }

    // Empty whole virtual polynomial: positive dimension/degree, zero products
    // and zero factors are a valid all-zero lifecycle.
    {
        const RefVirtualPolynomial empty{kNumVars, 2, {}, {}};
        const std::vector<std::vector<RefFr>> expected =
            reference_messages(empty, challenges);
        const std::vector<Ascend910B3DeviceAddress> no_factors;
        Ascend910B3NativeSumCheckProverState* state = create_state(context, empty, no_factors);
        for (size_t round = 0; round < kNumVars; ++round)
            run_round_checked(context, state, empty, round, challenges, expected,
                              "sumcheck empty VP round");
        std::vector<uint8_t> output((empty.degree + 1) * 32);
        require_contract_error(abi::sumcheck_prover_state_round(
                                   context, state, nullptr, output.data(), empty.degree + 1),
                               "sumcheck empty VP exhaustion");
        require_ok(abi::sumcheck_prover_state_destroy(context, state),
                   "destroy empty sumcheck state");
    }

    // Creation contract rejections: dimension, degree, arity, indices, factor
    // dimension/context and coefficient canonicality. Outputs are cleared.
    auto invalid_create = [&](size_t num_vars, size_t degree,
                              const std::vector<Ascend910B3DeviceAddress>& use_factors,
                              const std::vector<RefFr>& coefficients,
                              const std::vector<size_t>& offsets,
                              const std::vector<size_t>& indices,
                              const std::string& what) {
        std::vector<uint8_t> coefficient_bytes = encode_values(coefficients);
        std::vector<size_t> offset_copy = offsets;
        std::vector<size_t> index_copy = indices;
        auto* state = reinterpret_cast<Ascend910B3NativeSumCheckProverState*>(&sentinel);
        const Ascend910B3Status status = abi::sumcheck_prover_state_create(
            context, num_vars, degree,
            use_factors.empty() ? nullptr : use_factors.data(), use_factors.size(),
            coefficient_bytes.empty() ? nullptr : coefficient_bytes.data(), offset_copy.data(),
            offset_copy.size() - 1,
            index_copy.empty() ? nullptr : index_copy.data(), index_copy.size(), &state);
        require_contract_error(status, what);
        require(state == nullptr, what + " output");
    };

    invalid_create(0, 1, {}, {}, {0}, {}, "sumcheck zero variables");
    invalid_create(1, 0, {}, {}, {0}, {}, "sumcheck zero degree");
    invalid_create(3, 1, {factors[0], factors[1]}, {fr(1)}, {0, 2}, {0, 1},
                   "sumcheck product arity above max degree");
    invalid_create(3, 1, {factors[0]}, {fr(1)}, {0, 0}, {}, "sumcheck empty product");
    invalid_create(3, 2, {factors[0], factors[1]}, {fr(1)}, {0, 1}, {4},
                   "sumcheck out-of-range factor");
    invalid_create(3, std::numeric_limits<size_t>::max(), {}, {}, {0}, {},
                   "sumcheck degree overflow");

    const RefTable small_factor = make_table(2, {1, 2, 3, 4});
    const Ascend910B3DeviceAddress small_handle = upload_table(context, small_factor);
    invalid_create(3, 2, {small_handle}, {fr(1)}, {0, 1}, {0},
                   "sumcheck factor dimension mismatch");
    require_ok(abi::buffer_retire(context, small_handle), "retire small sumcheck factor");

    const RefTable foreign_factor(3, f0.values);
    const Ascend910B3DeviceAddress foreign_handle = upload_table(other, foreign_factor);
    invalid_create(3, 2, {foreign_handle}, {fr(1)}, {0, 1}, {0},
                   "sumcheck foreign factor");
    require_ok(abi::buffer_retire(other, foreign_handle), "retire foreign sumcheck factor");

    {
        std::vector<uint8_t> bad_coefficient(32, uint8_t(0xFF));
        std::vector<size_t> offsets{0, 1};
        std::vector<size_t> indices{0};
        auto* state = reinterpret_cast<Ascend910B3NativeSumCheckProverState*>(&sentinel);
        require_contract_error(abi::sumcheck_prover_state_create(
                                   context, 3, 2, factors.data(), factors.size(),
                                   bad_coefficient.data(), offsets.data(), 1, indices.data(), 1,
                                   &state),
                               "sumcheck noncanonical coefficient");
        require(state == nullptr, "sumcheck noncanonical coefficient output");
    }

    // Degree 34: mixed products with a 34-factor product, host metadata
    // scrubbed and factor handles retired after state creation.
    {
        const RefVirtualPolynomial polynomial{3, 34, factor_values, make_products(34)};
        const std::vector<std::vector<RefFr>> expected =
            reference_messages(polynomial, challenges);
        std::vector<uint8_t> coefficients;
        std::vector<size_t> offsets;
        std::vector<size_t> indices;
        encode_product_payload(polynomial, coefficients, offsets, indices);
        Ascend910B3NativeSumCheckProverState* state = nullptr;
        require_ok(abi::sumcheck_prover_state_create(
                       context, polynomial.num_vars, polynomial.degree, factors.data(),
                       factors.size(), coefficients.data(), offsets.data(),
                       polynomial.products.size(), indices.data(), indices.size(), &state),
                   "sumcheck degree 34 create");
        require(state != nullptr, "sumcheck degree 34 create null");
        std::fill(coefficients.begin(), coefficients.end(), uint8_t(0xEE));
        std::fill(offsets.begin(), offsets.end(), size_t(0xEEEE));
        std::fill(indices.begin(), indices.end(), size_t(0xEEEE));
        for (Ascend910B3DeviceAddress address : factors)
            require_ok(abi::buffer_retire(context, address), "retire sumcheck factor");
        factors_retired = true;
        for (size_t round = 0; round < kNumVars; ++round)
            run_round_checked(context, state, polynomial, round, challenges, expected,
                              "sumcheck degree 34 round");
        require_ok(abi::sumcheck_prover_state_destroy(context, state),
                   "destroy degree 34 sumcheck state");
    }

    if (!factors_retired)
        for (Ascend910B3DeviceAddress address : factors)
            require_ok(abi::buffer_retire(context, address), "retire sumcheck factor");
    std::cout << "BRIDGE_SUMCHECK_PASS\n";
}

} // namespace

int main(int argc, char** argv) {
    int32_t device = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--device" && i + 1 < argc) {
            char* end = nullptr;
            const long parsed = std::strtol(argv[++i], &end, 10);
            if (end == nullptr || *end != '\0' ||
                parsed < std::numeric_limits<int32_t>::min() ||
                parsed > std::numeric_limits<int32_t>::max()) {
                std::cerr << "BRIDGE_SMOKE_FAIL invalid device argument\n";
                return 2;
            }
            device = static_cast<int32_t>(parsed);
        } else if (argument == "--help") {
            std::cout << "usage: bridge_smoke [--device N]\n";
            return 0;
        } else {
            std::cerr << "BRIDGE_SMOKE_FAIL unknown argument: " << argument << '\n';
            return 2;
        }
    }

    try {
        uint8_t sentinel = 0;
        require_contract_error(abi::context_create(device, nullptr),
                               "context_create null output");
        require_contract_error(abi::context_destroy(nullptr), "context_destroy null");

        Ascend910B3NativeContext* first = nullptr;
        Ascend910B3NativeContext* second = nullptr;
        require_ok(abi::context_create(device, &first), "context_create first");
        require(first != nullptr, "context_create first null");
        require_ok(abi::context_create(device, &second), "context_create second");
        require(second != nullptr, "context_create second null");

        test_mle(first, second);
        test_async_lifetime(first);
        test_kzg(first, second);
        test_sumcheck(first, second);
        // All ready-on-return and resident-transfer audits above completed.
        std::cout << "BRIDGE_RESOURCE_PASS\n";

        // Independent contexts: a valid handle in the second context still works
        // after the first context is destroyed, and a stale context is rejected.
        const RefTable survivor = make_table(2, {3, 1, 4, 1});
        const Ascend910B3DeviceAddress survivor_handle = upload_table(second, survivor);
        Ascend910B3NativeContext* stale = first;
        require_ok(abi::context_destroy(first), "context_destroy first");
        Ascend910B3DeviceAddress stale_output = &sentinel;
        require_contract_error(abi::mle_upload(stale, 0, nullptr, 1, &stale_output),
                               "destroyed context operation");
        require(stale_output == nullptr, "destroyed context cleared its output");

        const std::vector<RefFr> survivor_point{fr(3), fr(4)};
        require(evaluate_checked(second, survivor_handle, survivor_point,
                                 "second context after first destroy")
                    .equals(reference_evaluate(survivor, survivor_point)),
                "second context evaluation after first destroy");
        require_ok(abi::buffer_retire(second, survivor_handle), "retire survivor");
        require_ok(abi::context_destroy(second), "context_destroy second");
        std::cout << "BRIDGE_CONTEXT_PASS\n";
        std::cout << "BRIDGE_SMOKE_PASS device=" << device << '\n';
    } catch (const std::exception& error) {
        std::cerr << "BRIDGE_SMOKE_FAIL " << error.what() << '\n';
        return 1;
    }
    return 0;
}
