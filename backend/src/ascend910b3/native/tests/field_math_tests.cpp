#include "field.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace hp = hyperplonk::ascend910b3;
namespace {
struct TestPrime64Params {
    static constexpr unsigned word_count = 2;
    static constexpr unsigned modulus_bits = 64;
    static constexpr uint32_t modulus[2] = {0xffffffc5, 0xffffffff};
    static constexpr uint16_t barrett_mu[5] = {0x003b, 0, 0, 0, 1};
    static constexpr uint32_t wide_factor[2] = {0x0d99, 0};
};
using Prime64 = hp::FieldElement<TestPrime64Params>;
template<class A, class B, class = void> struct CanAdd : std::false_type {};
template<class A, class B> struct CanAdd<A, B, std::void_t<decltype(std::declval<A>() + std::declval<B>())>> : std::true_type {};
template<class A, class B, class = void> struct CanSubtract : std::false_type {};
template<class A, class B> struct CanSubtract<A, B, std::void_t<decltype(std::declval<A>() - std::declval<B>())>> : std::true_type {};
template<class A, class B, class = void> struct CanMultiply : std::false_type {};
template<class A, class B> struct CanMultiply<A, B, std::void_t<decltype(std::declval<A>() * std::declval<B>())>> : std::true_type {};
static_assert(!CanAdd<hp::Fr, hp::Fq>::value && !CanAdd<hp::Fq, hp::Fr>::value);
static_assert(!CanSubtract<hp::Fr, hp::Fq>::value && !CanSubtract<hp::Fq, hp::Fr>::value);
static_assert(!CanMultiply<hp::Fr, hp::Fq>::value && !CanMultiply<hp::Fq, hp::Fr>::value);
static_assert(!std::is_convertible<hp::Fr, hp::Fq>::value && !std::is_convertible<hp::Fq, hp::Fr>::value);
static_assert(!std::is_convertible<uint32_t, hp::Fr>::value && !std::is_convertible<uint32_t, hp::Fq>::value);
static_assert(std::is_aggregate<hp::Fr>::value && std::is_aggregate<hp::Fq>::value);
static_assert(sizeof(Prime64) == 8 && alignof(Prime64) == 4);
constexpr std::array<size_t, 11> counts{0,1,2,31,32,33,255,256,257,513,4097};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
std::vector<uint8_t> read_bytes(const std::filesystem::path& path, size_t count) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(bool(input), "missing fixture: " + path.string());
    require(input.tellg() == static_cast<std::streamoff>(count), "wrong fixture length: " + path.string());
    input.seekg(0);
    std::vector<uint8_t> bytes(count);
    if (count) {
        input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(count));
        require(static_cast<size_t>(input.gcount()) == count, "truncated fixture: " + path.string());
    }
    return bytes;
}
template<class Field> bool canonical(const Field& value) {
    using P = typename Field::Parameters;
    for (unsigned i = Field::word_count; i-- > 0;) {
        if (value.words[i] != P::modulus[i]) return value.words[i] < P::modulus[i];
    }
    return false;
}
template<class Field> void expect(const Field& actual, const Field& expected, const std::string& context) {
    require(canonical(actual), context + ": noncanonical result");
    require(actual == expected, context + ": value mismatch");
}
template<class Field> std::vector<Field> load(const std::filesystem::path& path, size_t count) {
    size_t bytes = 0;
    require(hp::detail::checked_field_bytes<Field>(count, bytes), "fixture size overflow");
    const auto encoded = read_bytes(path, bytes);
    std::vector<Field> values(count);
    for (size_t i = 0; i < count; ++i) {
        const uint8_t* input = encoded.data() + i * sizeof(Field);
        require(Field::from_le_bytes(input, values[i]), "noncanonical fixture: " + path.string());
        std::array<uint8_t, sizeof(Field) + 2> output{};
        output.front() = 0xa5;
        output.back() = 0x5a;
        values[i].to_le_bytes(output.data() + 1);
        require(output.front() == 0xa5 && output.back() == 0x5a, "encoder wrote outside element");
        const auto limbs = values[i].to_limbs();
        for (unsigned j = 0; j < sizeof(Field); ++j) {
            require(output[j + 1] == input[j], "encoding roundtrip mismatch");
            require(limbs.bytes[j] == input[j], "limb encoding mismatch");
            require(uint8_t(values[i].words[j / 4] >> (8 * (j % 4))) == input[j], "word order mismatch");
        }
    }
    return values;
}
template<class Field> void boundaries() {
    using P = typename Field::Parameters;
    const auto zero = Field::zero();
    const auto one = Field::one();
    require(zero.is_zero() && !one.is_zero() && zero != one, "zero/one semantics");
    Field pm1{};
    for (unsigned i = 0; i < Field::word_count; ++i) pm1.words[i] = P::modulus[i];
    --pm1.words[0];
    uint8_t encoded[sizeof(Field)];
    for (unsigned i = 0; i < sizeof(Field); ++i) encoded[i] = uint8_t(P::modulus[i / 4] >> (8 * (i % 4)));
    Field output = pm1;
    require(!Field::from_le_bytes(encoded, output) && output == pm1, "decode modulus changed output or accepted");
    for (auto& byte : encoded) byte = 0xff;
    require(!Field::from_le_bytes(encoded, output) && output == pm1, "decode all-FF changed output or accepted");
    require(!zero.inverse(output) && output == pm1, "inverse zero changed output or succeeded");
    expect(pm1 + one, zero, "(p-1)+1");
    expect(zero - one, pm1, "0-1");
    expect(pm1 * pm1, one, "(p-1)^2");
    expect(-zero, zero, "neg zero");
    expect(-one, pm1, "neg one");
    // Alias-safe failure/success are part of the reference-output API contract.
    output = zero;
    require(!output.inverse(output) && output == zero, "zero inverse alias");
    output = pm1;
    require(output.inverse(output) && output == pm1, "nonzero inverse alias");
    size_t bytes = 987;
    require(hp::detail::checked_field_bytes<Field>(0, bytes) && bytes == 0, "zero checked size");
    require(hp::detail::checked_field_bytes<Field>(1, bytes) && bytes == sizeof(Field), "unit checked size");
    const size_t largest = size_t(PTRDIFF_MAX) / sizeof(Field);
    require(hp::detail::checked_field_bytes<Field>(largest, bytes) && bytes == largest * sizeof(Field), "maximum checked size");
    bytes = 987;
    require(!hp::detail::checked_field_bytes<Field>(largest + 1, bytes) && bytes == 987, "ptrdiff overflow");
    require(!hp::detail::checked_field_bytes<Field>(std::numeric_limits<size_t>::max() / sizeof(Field) + 1, bytes)
            && bytes == 987, "multiplication overflow");
}
template<class Field> void arithmetic_case(const std::filesystem::path& dir, size_t count, bool inverses) {
    const auto lhs = load<Field>(dir / "lhs.bin", count);
    const auto rhs = load<Field>(dir / "rhs.bin", count);
    const auto added = load<Field>(dir / "add.bin", count);
    const auto subtracted = load<Field>(dir / "sub.bin", count);
    const auto multiplied = load<Field>(dir / "mul.bin", count);
    const auto negated = load<Field>(dir / "neg.bin", count);
    const auto squared = load<Field>(dir / "square.bin", count);
    for (size_t i = 0; i < count; ++i) {
        const auto context = dir.string() + "/" + std::to_string(i);
        expect(lhs[i] + rhs[i], added[i], "add " + context);
        expect(lhs[i] - rhs[i], subtracted[i], "sub " + context);
        expect(lhs[i] * rhs[i], multiplied[i], "mul " + context);
        expect(-lhs[i], negated[i], "neg " + context);
        expect(lhs[i].square(), squared[i], "square " + context);
    }
    if (inverses) {
        const auto input = load<Field>(dir / "inverse_input.bin", count);
        const auto expected = load<Field>(dir / "inverse.bin", count);
        for (size_t i = 0; i < count; ++i) {
            Field result{};
            require(input[i].inverse(result), "nonzero inverse failed");
            expect(result, expected[i], "inverse " + dir.string() + "/" + std::to_string(i));
        }
    }
}
template<class Field> void parameters(const std::filesystem::path& dir) {
    using P = typename Field::Parameters;
    const auto modulus = read_bytes(dir / "modulus.bin", sizeof(Field));
    const auto mu = read_bytes(dir / "mu.bin", sizeof(Field) + 2);
    const auto factor = read_bytes(dir / "wide_factor.bin", sizeof(Field));
    for (unsigned i = 0; i < sizeof(Field); ++i) {
        require(modulus[i] == uint8_t(P::modulus[i / 4] >> (8 * (i % 4))), "compiled modulus mismatch");
        require(factor[i] == uint8_t(P::wide_factor[i / 4] >> (8 * (i % 4))), "compiled wide factor mismatch");
    }
    for (unsigned i = 0; i < sizeof(Field) + 2; ++i)
        require(mu[i] == uint8_t(P::barrett_mu[i / 2] >> (8 * (i % 2))), "compiled Barrett mu mismatch");
}
template<class Field> void extras(const std::filesystem::path& root) {
    using P = typename Field::Parameters;
    const auto input = read_bytes(root / "wide_input.bin", 512 * sizeof(hp::WideInteger<P>));
    const auto expected = load<Field>(root / "wide_result.bin", 512);
    for (unsigned i = 0; i < 512; ++i) {
        hp::WideInteger<P> wide{};
        for (unsigned j = 0; j < sizeof(wide); ++j) wide.bytes[j] = input[i * sizeof(wide) + j];
        expect(Field::reduce_wide(wide), expected[i], "wide " + root.string() + "/" + std::to_string(i));
    }
    size_t edge_count = 4;
    for (unsigned bits = 16; bits < P::modulus_bits; bits += 16) edge_count += 3;
    arithmetic_case<Field>(root / "boundaries", edge_count * 4, false);
}
void fr_regression(const std::filesystem::path& root, size_t count) {
    using hp::Fr;
    const auto dir = root / ("case-" + std::to_string(count));
    const auto lhs = load<Fr>(dir / "lhs.bin", count);
    const auto rhs = load<Fr>(dir / "rhs.bin", count);
    const auto scaled = load<Fr>(dir / "scale.bin", count);
    const auto scalar = load<Fr>(dir / "scalar.bin", 1);
    const auto dot = load<Fr>(dir / "inner_product.bin", 1);
    Fr sum{};
    for (size_t i = 0; i < count; ++i) {
        expect(lhs[i] * scalar[0], scaled[i], "scale");
        sum = sum + lhs[i] * rhs[i];
    }
    expect(sum, dot[0], "dot");
    if (count == 513 || count == 4097) {
        auto state = lhs;
        for (unsigned round = 0; round < 10; ++round)
            for (size_t i = 0; i < count; ++i) state[i] = (dot[0] * state[i] + rhs[i]) * lhs[i];
        Fr end{};
        for (size_t i = 0; i < count; ++i) end = end + state[i] * rhs[i];
        expect(end, load<Fr>(dir / "chain.bin", 1)[0], "resident chain");
    }
}
template<class Field> void field_suite(const std::filesystem::path& root, const std::filesystem::path& param_dir,
                                      const char* name) {
    boundaries<Field>();
    parameters<Field>(param_dir);
    for (const size_t count : counts) {
        const bool inverse = count == 0 || count == 1 || count == 2 || count == 33 || count == 257;
        arithmetic_case<Field>(root / ("case-" + std::to_string(count)), count, inverse);
    }
    extras<Field>(root);
    std::cout << "FIELD_ELEMENT_HOST_PASS field=" << name << '\n';
}
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 3 && std::string(argv[1]) == "--fixtures", "usage: field_math_tests --fixtures DIRECTORY");
        const std::filesystem::path root(argv[2]);
        field_suite<hp::Fr>(root, root / "params/fr", "Fr");
        field_suite<hp::Fq>(root / "fq", root / "params/fq", "Fq");
        boundaries<Prime64>();
        parameters<Prime64>(root / "params/prime64");
        arithmetic_case<Prime64>(root / "prime64/case-257", 257, true);
        extras<Prime64>(root / "prime64");
        std::cout << "FIELD_ELEMENT_HOST_PASS field=TestPrime64\n";
        for (const size_t count : counts) fr_regression(root, count);
        std::cout << "FIELD_MATH_HOST_PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FIELD_MATH_HOST_FAIL: " << error.what() << '\n';
        return 1;
    }
}
