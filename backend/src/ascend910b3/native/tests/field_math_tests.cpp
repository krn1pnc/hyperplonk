#include "field.h"
#include "g1.h"

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

template<class A, class B, class = void> struct CanMixedAdd : std::false_type {};
template<class A, class B> struct CanMixedAdd<A, B, std::void_t<
    decltype(std::declval<A&>().mixed_add_assign(std::declval<const B&>()))>> : std::true_type {};
static_assert(std::is_aggregate<hp::G1Affine>::value && std::is_aggregate<hp::G1Jacobian>::value);
static_assert(!std::is_convertible<hp::G1Affine, hp::G1Jacobian>::value);
static_assert(!std::is_convertible<hp::G1Jacobian, hp::G1Affine>::value);
static_assert(CanMixedAdd<hp::G1Jacobian, hp::G1Affine>::value);
static_assert(!CanMixedAdd<hp::G1Jacobian, hp::G1Jacobian>::value);

std::vector<hp::G1Affine> load_affine(const std::filesystem::path& path, size_t count) {
    const auto coords = load<hp::Fq>(path, count * 2);
    std::vector<hp::G1Affine> out(count);
    for (size_t i = 0; i < count; ++i) out[i] = {coords[2*i], coords[2*i+1]};
    return out;
}
std::vector<hp::G1Jacobian> load_jacobian(const std::filesystem::path& path, size_t count) {
    const auto coords = load<hp::Fq>(path, count * 3);
    std::vector<hp::G1Jacobian> out(count);
    for (size_t i = 0; i < count; ++i) out[i] = {coords[3*i], coords[3*i+1], coords[3*i+2]};
    return out;
}
template<class T> std::vector<T> load_integers(const std::filesystem::path& path, size_t count) {
    const auto bytes = read_bytes(path, count * sizeof(T));
    std::vector<T> values(count);
    for (size_t i = 0; i < count; ++i)
        for (size_t j = 0; j < sizeof(T); ++j) values[i] |= T(bytes[i*sizeof(T)+j]) << (8*j);
    return values;
}
bool same_raw(const hp::G1Jacobian& a, const hp::G1Jacobian& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}
void expect_affine(const hp::G1Affine& a, const hp::G1Affine& b, const std::string& context) {
    expect(a.x, b.x, context + " x");
    expect(a.y, b.y, context + " y");
}
void expect_point(const hp::G1Jacobian& actual, const hp::G1Affine& expected, const std::string& context) {
    require(canonical(actual.x) && canonical(actual.y) && canonical(actual.z), context + " canonical coords");
    require(actual.is_on_curve(), context + " off curve");
    if (actual.is_identity())
        require(same_raw(actual, hp::G1Jacobian::identity()), context + " noncanonical identity");
    expect_affine(actual.to_affine(), expected, context);
}
hp::G1Jacobian raw_order_mul(const hp::G1Jacobian& point) {
    hp::G1Jacobian result{};
    for (unsigned bit = 255; bit-- > 0;) {
        result.double_in_place();
        if ((hp::Bls12381FrParams::modulus[bit/32] >> (bit%32)) & 1) result += point;
    }
    return result;
}
unsigned scalar_digit(const hp::Fr& scalar, unsigned window, unsigned c) {
    const unsigned bit = window*c, word = bit/32, shift = bit%32;
    uint32_t digit = scalar.words[word] >> shift;
    if (shift && shift+c > 32 && word+1 < hp::Fr::word_count)
        digit |= scalar.words[word+1] << (32-shift);
    const unsigned available = 255-bit < c ? 255-bit : c;
    return digit & ((1u << available)-1);
}
hp::G1Jacobian weighted_range(const std::vector<hp::G1Jacobian>& buckets, unsigned start, unsigned end) {
    const unsigned a = start < 1 ? 1 : start;
    hp::G1Jacobian running{}, local{};
    if (a > end) return local;
    for (unsigned b = end+1; b-- > a;) {
        running += buckets[b];
        local += running;
    }
    if (a > 1) local += running.mul_by_u64(a-1);
    return local;
}
void g1_host_chains(const std::filesystem::path& root) {
    const std::array<std::pair<const char*, size_t>, 8> cases{{
        {"case-0",0},{"case-1",1},{"case-2",2},{"case-7",7},{"case-33",33},
        {"cancel",2},{"zero-scalars",7},{"identity-bases",7}}};
    for (const auto& item : cases) {
        const auto dir = root / "chain" / item.first;
        const auto bases = load_affine(dir / "bases.bin", item.second);
        const auto scalars = load<hp::Fr>(dir / "scalars.bin", item.second);
        const auto expected = load_affine(dir / "expected.bin", 1)[0];
        for (const unsigned c : {3u,4u}) {
            const unsigned windows = (255+c-1)/c, bucket_count = 1u << c;
            std::vector<std::array<hp::G1Jacobian, 2>> segments(windows);
            for (unsigned w = 0; w < windows; ++w) {
                std::vector<hp::G1Jacobian> buckets(bucket_count);
                for (unsigned b = 0; b < bucket_count; ++b) {
                    hp::G1Jacobian partials[2]{};
                    if (b) for (size_t i = 0; i < item.second; ++i)
                        if (scalar_digit(scalars[i],w,c) == b)
                            partials[i < item.second/2 ? 0 : 1].mixed_add_assign(bases[i]);
                    buckets[b] = partials[0];
                    buckets[b] += partials[1];
                }
                const unsigned split = (bucket_count-1)/2;
                segments[w][0] = weighted_range(buckets,1,split);
                segments[w][1] = weighted_range(buckets,split+1,bucket_count-1);
                require(weighted_range(buckets,3,2).is_identity(), "empty weighted range");
                require(weighted_range(buckets,0,split) == segments[w][0], "zero-start weighted range");
                hp::G1Jacobian direct{};
                for (unsigned b = split+1; b < bucket_count; ++b) direct += buckets[b].mul_by_u64(b);
                require(direct == segments[w][1], "offset weighted range");
            }
            hp::G1Jacobian result{};
            for (unsigned w = windows; w-- > 0;) {
                for (unsigned bit = 0; bit < c; ++bit) result.double_in_place();
                result += segments[w][0];
                result += segments[w][1];
            }
            expect_point(result, expected, "host chain " + dir.string() + "/" + std::to_string(c));
        }
    }
}
void g1_suite(const std::filesystem::path& fixtures) {
    const auto root = fixtures / "g1";
    const auto generator = load_affine(root / "generator.bin",1)[0];
    expect_affine(hp::G1Affine::generator(), generator, "generator");
    require(hp::G1Affine{}.is_identity() && same_raw(hp::G1Jacobian{},hp::G1Jacobian::identity()),
            "default point identity");
    for (const size_t count : {0,1,2,15,16,17,31,32,33,65}) {
        const auto dir = root / ("case-" + std::to_string(count));
        const auto aa = load_affine(dir / "affine_a.bin",count);
        const auto ab = load_affine(dir / "affine_b.bin",count);
        const auto ja = load_jacobian(dir / "jacobian_a.bin",count);
        const auto jb = load_jacobian(dir / "jacobian_b.bin",count);
        const auto mul = load_integers<uint64_t>(dir / "mul_u64.bin",count);
        const auto equal = load_integers<uint32_t>(dir / "equal.bin",count);
        const auto added = load_affine(dir / "add.bin",count);
        const auto doubled = load_affine(dir / "double.bin",count);
        const auto negated = load_affine(dir / "neg.bin",count);
        const auto affine_negated = load_affine(dir / "affine_neg.bin",count);
        const auto multiplied = load_affine(dir / "mul.bin",count);
        for (size_t i = 0; i < count; ++i) {
            const auto a = ja[i], b = jb[i];
            const auto context = dir.string() + "/" + std::to_string(i);
            require((a == b) == bool(equal[i]) && (a != b) == !bool(equal[i]), context + " projective equality");
            require((aa[i] == ab[i]) == bool(equal[i]) && (aa[i] != ab[i]) == !bool(equal[i]),
                    context + " affine equality");
            expect_affine(a.to_affine(),aa[i],context + " normalize");
            expect_point(aa[i].to_jacobian(),aa[i],context + " from affine");
            auto p = a; p += b; expect_point(p,added[i],context + " add");
            p = a; p.mixed_add_assign(ab[i]); expect_point(p,added[i],context + " mixed");
            p = a; p.double_in_place(); expect_point(p,doubled[i],context + " double");
            p = a; p += p; expect_point(p,doubled[i],context + " self add");
            expect_point(-a,negated[i],context + " negate");
            expect_point(a.mul_by_u64(mul[i]),multiplied[i],context + " mul u64");
            expect_affine(-ab[i],affine_negated[i],context + " affine negate");
            require(same_raw(a,ja[i]) && same_raw(b,jb[i]) && aa[i] == a.to_affine()
                    && ab[i] == b.to_affine(), context + " source mutation");
            if (i == 1) require(!same_raw(a,b) && a == b && !(a != b), "scaled equality fixture");
            if (i == 8) {
                p = a; p += b;
                require(same_raw(p,b), "identity add preserves finite representation");
                expect_affine(p.to_affine(),generator,"identity add G");
                p = a; p.mixed_add_assign(ab[i]);
                require(p.z == hp::Fq::one(), "identity mixed Z1");
                expect_affine(p.to_affine(),generator,"identity mixed G");
            }
        }
    }
    const auto da = load_affine(root / "diagnostics/affine.bin",5);
    const auto dj = load_jacobian(root / "diagnostics/jacobian.bin",5);
    const auto flags = load_integers<uint32_t>(root / "diagnostics/flags.bin",5);
    for (size_t i = 0; i < 5; ++i) {
        const unsigned actual = unsigned(da[i].is_identity()) | (unsigned(da[i].is_on_curve()) << 1)
            | (unsigned(dj[i].is_identity()) << 2) | (unsigned(dj[i].is_on_curve()) << 3);
        require(actual == flags[i], "G1 diagnostics flags");
    }
    require(raw_order_mul(generator.to_jacobian()).is_identity(), "[r]G must vanish");
    require(da[2].is_on_curve() && !raw_order_mul(dj[2]).is_identity(), "curve check is not subgroup check");
    g1_host_chains(root);
    std::cout << "G1_HOST_PASS\n";
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
        g1_suite(root);
        std::cout << "FIELD_MATH_HOST_PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FIELD_MATH_HOST_FAIL: " << error.what() << '\n';
        return 1;
    }
}
