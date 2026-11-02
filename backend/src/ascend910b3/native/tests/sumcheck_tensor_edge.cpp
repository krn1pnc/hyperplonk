// Complete round-by-round oracle for the production SumCheck bridge. Only the
// existing public bridge ABI is used; no kernel, device field or private header.
#include "bridge.h"
#include "runtime_audit.h"

#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Big = boost::multiprecision::cpp_int;
using Packet = std::array<uint8_t, 32>;
namespace audit = hyperplonk::ascend910b3::test;
const Big modulus("0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001");

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void ok(Ascend910B3Status status, const std::string& what) {
    require(status.kind == ASCEND910B3_STATUS_SUCCESS,
            what + " status=" + std::to_string(status.kind) +
            " native=" + std::to_string(status.native_code));
}
// Canonical reduction. Boost.Multiprecision 1.71 expression templates hold
// references into the operands, so a compound expression must never be the
// object that is reduced in place: materialise the argument into this copy
// first and reduce only named Big objects. Call sites below also store each
// intermediate in a named Big for the same reason.
Big norm(const Big& value) {
    Big reduced(value);
    reduced %= modulus;
    if (reduced < 0) reduced += modulus;
    return reduced;
}
Packet encode(Big value) {
    require(value >= 0 && value < modulus, "oracle encoding is not canonical");
    Packet result{};
    for (auto& byte : result) {
        byte = static_cast<uint8_t>((value & 255).convert_to<unsigned>());
        value >>= 8;
    }
    return result;
}
std::vector<uint8_t> encode_table(const std::vector<Big>& values) {
    std::vector<uint8_t> result(values.size() * 32);
    for (size_t i = 0; i < values.size(); ++i) {
        const auto packet = encode(values[i]);
        std::memcpy(result.data() + i * 32, packet.data(), 32);
    }
    return result;
}

// Rejection sampling of 255 random bits, not a small-integer field fixture.
struct Random {
    uint64_t state;
    uint64_t next() {
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    Big field() {
        for (;;) {
            Big value = 0;
            for (unsigned word = 0; word < 4; ++word) {
                uint64_t bits = next();
                if (word == 3) bits &= 0x7fffffffffffffffULL;
                value += Big(bits) << (64 * word);
            }
            if (value < modulus) return value;
        }
    }
    Big nonboolean() {
        for (;;) {
            Big value = field();
            if (value > 1) return value;
        }
    }
};
struct Product {
    Big coefficient;
    std::vector<size_t> factors;
};
using Tables = std::vector<std::vector<Big>>;

// Deliberately ordinary polynomial arithmetic modulo Fr, with no Montgomery,
// byte-radix, four-corner identities, device layout or device constants reused.
std::vector<Big> message(const Tables& tables, const std::vector<Product>& products,
                         size_t degree) {
    std::vector<Big> output(degree + 1, Big(0));
    const size_t pairs = tables.front().size() / 2;
    for (size_t t = 0; t <= degree; ++t) {
        for (size_t b = 0; b < pairs; ++b) {
            for (const auto& product : products) {
                Big term = product.coefficient;
                for (size_t factor : product.factors) {
                    const auto& table = tables[factor];
                    const Big low = table[2 * b];
                    const Big delta = table[2 * b + 1] - table[2 * b];
                    const Big base = low + Big(t) * delta;
                    const Big at = norm(base);
                    const Big scaled = term * at;
                    term = norm(scaled);
                }
                const Big accumulated = output[t] + term;
                output[t] = norm(accumulated);
            }
        }
    }
    return output;
}
void fold(Tables& tables, const Big& coordinate) {
    for (auto& table : tables) {
        std::vector<Big> next(table.size() / 2);
        for (size_t i = 0; i < next.size(); ++i) {
            // Same reason as norm(): never reduce a live compound expression.
            const Big low = table[2 * i];
            const Big delta = table[2 * i + 1] - table[2 * i];
            const Big value = low + coordinate * delta;
            next[i] = norm(value);
        }
        table = std::move(next);
    }
}
struct Context {
    Ascend910B3NativeContext* value = nullptr;
    ~Context() { if (value) (void)hp_ascend910b3_v1_context_destroy(value); }
};
struct State {
    Ascend910B3NativeContext* context;
    Ascend910B3NativeSumCheckProverState* value = nullptr;
    ~State() {
        if (value) (void)hp_ascend910b3_v1_sumcheck_prover_state_destroy(context, value);
    }
};
struct Handles {
    Ascend910B3NativeContext* context;
    std::vector<Ascend910B3DeviceAddress> values;
    ~Handles() {
        for (auto address : values)
            if (address) (void)hp_ascend910b3_v1_buffer_retire(context, address);
    }
    void retire() {
        for (auto& address : values) {
            ok(hp_ascend910b3_v1_buffer_retire(context, address), "retire original");
            address = nullptr;
        }
    }
};
struct Case {
    const char* name;
    size_t n;
    size_t degree;
    unsigned kind; // 0 single d2, 1 repeated/multiple, 2 mixed arity, 3 cubic, 4 empty
};
const Case cases[] = {
    {"n1_d2", 1, 2, 0}, {"n2_d2", 2, 2, 0},
    {"n3_multi", 3, 2, 1}, {"n9_tail", 9, 2, 0},
    {"n12_d2", 12, 2, 0}, {"n12_multi", 12, 2, 1},
    {"n9_mixed", 9, 2, 2}, {"n9_d3", 9, 3, 3},
    {"n2_empty", 2, 2, 4}
};

void rejected(Ascend910B3NativeContext* context, State& state,
              const uint8_t* challenge, uint8_t* destination, size_t count,
              int32_t expected_kind, std::vector<uint8_t>& output,
              const std::string& what) {
    std::fill(output.begin(), output.end(), uint8_t(0xa7));
    audit::begin_runtime_audit();
    const auto status = hp_ascend910b3_v1_sumcheck_prover_state_round(
        context, state.value, challenge, destination, count);
    const auto calls = audit::end_runtime_audit();
    require(status.kind == expected_kind, what + " incorrect status");
    require(std::all_of(output.begin(), output.end(), [](uint8_t b) { return b == 0xa7; }),
            what + " modified output");
    require(calls.h2d_calls == 0 && calls.d2h_calls == 0 && calls.other_copy_calls == 0 &&
            calls.stream_sync_calls == 0 && calls.device_sync_calls == 0 &&
            calls.malloc_calls == 0 && calls.host_malloc_calls == 0,
            what + " had an execution side effect");
}
void run_case(Ascend910B3NativeContext* context, const Case& test, uint64_t seed) {
    const std::string name(test.name);
    Random random{seed};
    Tables tables(3, std::vector<Big>(size_t(1) << test.n));
    for (size_t f = 0; f < 2; ++f) {
        for (auto& value : tables[f]) value = random.field();
        // Keep n=1 fully random; larger fixtures also exercise boundaries.
        if (test.n > 1) {
            tables[f][0] = f == 0 ? Big(0) : Big(modulus - 1);
            tables[f][1] = f == 0 ? Big(modulus - 1) : Big(1);
        }
    }
    tables[2] = tables[0]; // Deliberate duplicate public handle, not another upload.
    std::vector<Product> products;
    if (test.kind == 0) products = {{random.nonboolean(), {2, 1}}};
    if (test.kind == 1) products = {{random.nonboolean(), {2, 1}},
                                  {modulus - 1, {0, 2}},
                                  {random.nonboolean(), {1, 1}}};
    if (test.kind == 2) products = {{Big(3), {0}}, {Big(7), {2, 1}}};
    if (test.kind == 3) products = {{random.nonboolean(), {0, 1, 2}},
                                  {modulus - 1, {1}}};
    Handles originals{context, {}};
    for (size_t f = 0; f < 2; ++f) {
        auto bytes = encode_table(tables[f]);
        Ascend910B3DeviceAddress address = nullptr;
        ok(hp_ascend910b3_v1_mle_upload(context, test.n, bytes.data(), tables[f].size(),
                                       &address), name + " upload");
        require(address != nullptr, name + " null upload");
        originals.values.push_back(address);
        std::fill(bytes.begin(), bytes.end(), uint8_t(0x95));
    }
    std::vector<Ascend910B3DeviceAddress> handles{
        originals.values[0], originals.values[1], originals.values[0]};
    std::vector<size_t> offsets{0}, indices;
    std::vector<Big> coefficients;
    for (const auto& product : products) {
        coefficients.push_back(product.coefficient);
        indices.insert(indices.end(), product.factors.begin(), product.factors.end());
        offsets.push_back(indices.size());
    }
    auto coefficient_bytes = encode_table(coefficients);
    State state{context};
    audit::begin_runtime_audit();
    const auto created = hp_ascend910b3_v1_sumcheck_prover_state_create(
        context, test.n, test.degree, handles.data(), handles.size(),
        coefficient_bytes.empty() ? nullptr : coefficient_bytes.data(), offsets.data(),
        products.size(), indices.empty() ? nullptr : indices.data(), indices.size(), &state.value);
    const auto create_calls = audit::end_runtime_audit();
    ok(created, name + " create");
    require(state.value != nullptr, name + " null state");
    require(create_calls.stream_sync_calls == 1 && create_calls.device_sync_calls == 0,
            name + " create synchronization contract");
    // Drop every original before round zero, then invalidate borrowed host metadata.
    originals.retire();
    std::fill(handles.begin(), handles.end(), nullptr);
    std::fill(offsets.begin(), offsets.end(), size_t(999));
    std::fill(indices.begin(), indices.end(), size_t(999));
    std::fill(coefficient_bytes.begin(), coefficient_bytes.end(), uint8_t(0x95));

    std::vector<uint8_t> output((test.degree + 1) * 32);
    Packet challenge = encode(random.nonboolean());
    rejected(context, state, challenge.data(), output.data(), test.degree + 1,
             ASCEND910B3_STATUS_INVALID_STATE, output, name + " premature challenge");
    rejected(context, state, nullptr, output.data(), test.degree,
             ASCEND910B3_STATUS_INVALID_PARAMETERS, output, name + " wrong count");
    rejected(context, state, nullptr, nullptr, test.degree + 1,
             ASCEND910B3_STATUS_INVALID_PARAMETERS, output, name + " null output");
    for (size_t round = 0; round < test.n; ++round) {
        Big coordinate = 0;
        if (round != 0) {
            rejected(context, state, nullptr, output.data(), test.degree + 1,
                     ASCEND910B3_STATUS_INVALID_STATE, output, name + " missing challenge");
            Packet invalid{};
            invalid.fill(0xff);
            rejected(context, state, invalid.data(), output.data(), test.degree + 1,
                     ASCEND910B3_STATUS_INVALID_PARAMETERS, output, name + " noncanonical challenge");
            coordinate = random.nonboolean();
            challenge = encode(coordinate);
            fold(tables, coordinate);
        }
        const auto expected = message(tables, products, test.degree);
        audit::begin_runtime_audit();
        const auto status = hp_ascend910b3_v1_sumcheck_prover_state_round(
            context, state.value, round == 0 ? nullptr : challenge.data(),
            output.data(), test.degree + 1);
        const auto calls = audit::end_runtime_audit();
        ok(status, name + " round " + std::to_string(round));
        for (size_t t = 0; t <= test.degree; ++t) {
            const auto bytes = encode(expected[t]);
            require(std::memcmp(bytes.data(), output.data() + t * 32, 32) == 0,
                    name + " round=" + std::to_string(round) + " t=" +
                    std::to_string(t) + " byte-exact Fr mismatch");
        }
        const uint64_t h2d = round == 0 || products.empty() ? 0 : 32;
        require(calls.h2d_bytes == h2d && calls.h2d_calls == (h2d == 0 ? 0 : 1) &&
                calls.d2h_bytes == output.size() && calls.d2h_calls == 1 &&
                calls.other_copy_calls == 0 && calls.stream_sync_calls == 1 &&
                calls.device_sync_calls == 0 && calls.malloc_calls == 0 &&
                calls.host_malloc_calls == 0, name + " round transfer/allocation contract");
        std::cout << "ROUND PASS " << name << " round=" << round << '\n';
    }
    rejected(context, state, challenge.data(), output.data(), test.degree + 1,
             ASCEND910B3_STATUS_INVALID_STATE, output, name + " exhausted state");
    ok(hp_ascend910b3_v1_sumcheck_prover_state_destroy(context, state.value), name + " destroy");
    state.value = nullptr;
    std::cout << "CASE PASS " << name << " rounds=" << test.n << " seed=" << seed << '\n';
}
} // namespace

int main(int argc, char** argv) {
    try {
        int device = 0;
        std::string selected;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            require(i + 1 < argc, "expected --device N or --case NAME");
            if (arg == "--device") device = std::stoi(argv[++i]);
            else if (arg == "--case") selected = argv[++i];
            else throw std::runtime_error("unknown argument: " + arg);
        }
        Context context;
        ok(hp_ascend910b3_v1_context_create(device, &context.value), "context create");
        size_t completed = 0;
        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
            if (!selected.empty() && selected != cases[i].name) continue;
            run_case(context.value, cases[i], 0x534354454e534f52ULL + i * 0x10001ULL);
            ++completed;
        }
        require(completed != 0, "unknown case: " + selected);
        ok(hp_ascend910b3_v1_context_destroy(context.value), "context destroy");
        context.value = nullptr;
        std::cout << "SUMCHECK_TENSOR_EDGE PASS cases=" << completed << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "SUMCHECK_TENSOR_EDGE FAIL " << error.what() << '\n';
        return 1;
    }
}
