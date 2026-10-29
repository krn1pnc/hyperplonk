#include "field.h"
#include "g1.h"
#include "msm_plan.h"

#include <algorithm>
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
constexpr std::array<size_t, 20> counts{0,1,2,31,32,33,63,64,65,255,256,257,513,4097,47,48,49,95,96,97};

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
template<class Field> void vector_parameters(const std::filesystem::path& dir) {
    using P = typename Field::Parameters;
    using V = hp::FieldVectorParams<P>;
    const auto check_digits = [&](const char* name, const auto& digits) {
        const auto bytes = read_bytes(dir / name, 4 * V::digit_count);
        for (unsigned j = 0; j < V::digit_count; ++j) {
            require(digits[j] >= 0 && digits[j] < (1 << 12), "vector digit outside radix");
            for (unsigned b = 0; b < 4; ++b)
                require(bytes[4*j+b] == uint8_t(uint32_t(digits[j]) >> (8*b)),
                        std::string("compiled vector parameter mismatch: ") + name);
        }
    };
    check_digits("vector_modulus.bin", V::modulus_digits);
    check_digits("vector_r.bin", V::r_digits);
    check_digits("vector_r2.bin", V::r2_digits);
    const auto n0prime = read_bytes(dir / "vector_n0prime.bin", 4);
    require(V::n0prime >= 0 && V::n0prime < (1 << 12), "vector n0prime outside radix");
    for (unsigned b = 0; b < 4; ++b)
        require(n0prime[b] == uint8_t(uint32_t(V::n0prime) >> (8*b)), "compiled vector n0prime mismatch");
    require((uint32_t(V::modulus_digits[0]) * uint32_t(V::n0prime) + 1) % (1 << 12) == 0,
            "vector Montgomery inverse identity");
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
    vector_parameters<Field>(param_dir);
    for (const size_t count : counts) {
        const bool inverse = count == 0 || count == 1 || count == 2 || count == 33
                             || count == 64 || count == 65 || count == 257
                             || count == 47 || count == 48 || count == 49
                             || count == 95 || count == 96 || count == 97;
        arithmetic_case<Field>(root / ("case-" + std::to_string(count)), count, inverse);
    }
    extras<Field>(root);
    const auto radix_dir = root / "radix-boundaries";
    const auto count_bytes = read_bytes(radix_dir / "count.bin", 4);
    uint32_t radix_count = 0;
    for (unsigned b = 0; b < 4; ++b) radix_count |= uint32_t(count_bytes[b]) << (8*b);
    arithmetic_case<Field>(radix_dir, radix_count, false);
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
    for (const size_t count : {0,1,2,15,16,17,31,32,33,65,63,64,47,48,49,95,96,97}) {
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

bool same_msm_plan(const hp::detail::MsmPlan& a, const hp::detail::MsmPlan& b) {
    if (a.platform.aiv_cores != b.platform.aiv_cores || a.platform.ub_bytes != b.platform.ub_bytes
        || a.count != b.count || a.workspace_bytes != b.workspace_bytes
        || a.dense_records != b.dense_records || a.fragment_records != b.fragment_records
        || a.host_workspace_bytes != b.host_workspace_bytes || a.host_prefix_offset != b.host_prefix_offset
        || a.window_bits != b.window_bits || a.windows != b.windows || a.buckets != b.buckets
        || a.radix_passes != b.radix_passes || a.chunks != b.chunks || a.merge_levels != b.merge_levels
        || a.row_blocks != b.row_blocks || a.weighted_segments != b.weighted_segments
        || a.weighted_levels != b.weighted_levels) return false;
    for (unsigned i = 0; i < 2; ++i)
        if (a.partial_capacity[i] != b.partial_capacity[i]) return false;
    for (unsigned i = 0; i < hp::detail::msm_region_count; ++i)
        if (a.region_offset[i] != b.region_offset[i] || a.region_size[i] != b.region_size[i]) return false;
    return true;
}

void msm_host_suite() {
    using namespace hp::detail;
    const MsmPlatform platform{20, msm_required_ub_bytes};
    const auto ceil_div = [](uint64_t n, uint64_t d) { return n / d + (n % d != 0); };
    const auto round8 = [&](uint64_t n) { return ceil_div(n, 8) * 8; };
    const auto levels = [&](uint64_t n) {
        uint32_t result = 0;
        while (n > 1) { n = ceil_div(n, 64); ++result; }
        return result;
    };
    const std::array<uint64_t, 9> sizes{0,1,64,65,2048,2049,65537,UINT32_MAX,4097};
    for (const uint64_t n : sizes) for (const uint32_t requested : {0u,2u,3u,4u,12u,16u}) {
        MsmPlan p{};
        require(make_msm_plan(static_cast<size_t>(n), platform, requested, p), "MSM valid plan rejected");
        uint32_t ell = 0;
        for (uint64_t x = n ? n - 1 : 0; x; x >>= 1) ++ell;
        const uint32_t automatic = n ? (ell - ell / 3 + 2 > 16 ? 16 : ell - ell / 3 + 2) : 2;
        const uint32_t c = requested ? requested : automatic;
        require(p.count == n && p.window_bits == c && p.platform.aiv_cores == platform.aiv_cores
                && p.platform.ub_bytes == platform.ub_bytes, "MSM plan identity");
        if (!n) {
            require(!p.workspace_bytes && !p.host_workspace_bytes && !p.host_prefix_offset
                    && !p.windows && !p.buckets && !p.chunks && !p.merge_levels
                    && !p.row_blocks && !p.radix_passes && !p.weighted_segments && !p.weighted_levels
                    && !p.dense_records && !p.fragment_records
                    && !p.partial_capacity[0] && !p.partial_capacity[1], "MSM empty plan shape");
            for (unsigned i = 0; i < msm_region_count; ++i)
                require(!p.region_size[i] && !p.region_offset[i], "MSM empty region");
            continue;
        }
        const uint64_t b = uint64_t{1} << c, chunks = ceil_div(n, 2048);
        const uint64_t t = 8 * round8(b + 1);
        const uint64_t m0 = ceil_div(n, 64) + b, m1 = ceil_div(n, 4096) + b;
        const uint32_t l = levels(ceil_div(n, 64));
        const bool weighted_tree = p.weighted_segments > 1;
        const uint64_t prefix_capacity = ceil_div(t * (l + 1) + 4 * m0, 64) * 64;
        require(p.windows == ceil_div(255, c) && p.buckets == b && p.radix_passes == ceil_div(c, 4)
                && p.chunks == chunks && p.merge_levels == l && p.row_blocks == ceil_div(b + 1, 256)
                && p.weighted_segments >= 1 && p.weighted_segments < b
                && bool(p.weighted_levels) == weighted_tree,
                "MSM plan derived shape");
        require(p.dense_records == round8(n) && p.fragment_records == round8(n + 112 * chunks)
                && p.partial_capacity[0] == m0 && p.partial_capacity[1] == ((l || weighted_tree) ? m1 : 0),
                "MSM plan record and partial capacities");
        const std::array<uint64_t, msm_region_count> bytes{
            8 * round8(n), 8 * round8(n), 8 * round8(n + 112 * chunks),
            64 * chunks, 64 * 16 * chunks, 64 * 16, 64 * (b + 1), t, t,
            64 * ceil_div(b + 1, 256), 64 * ceil_div(b + 1, 256), prefix_capacity,
            64 * m0, 192 * m0, 192 * ((l || weighted_tree) ? m1 : 0), 192 * b, 192 * ceil_div(255, c),
            96 * n};
        uint64_t end = 0;
        for (unsigned i = 0; i < bytes.size(); ++i) {
            require(p.region_size[i] == bytes[i], "MSM region capacity");
            if (!bytes[i]) {
                require(!p.region_offset[i], "MSM zero region offset");
            } else {
                const uint64_t start = p.region_offset[i];
                require(!(start % 64) && start <= p.workspace_bytes
                        && bytes[i] <= p.workspace_bytes - start, "MSM region alignment/bounds");
                for (unsigned j = 0; j < i; ++j)
                    if (bytes[j])
                        require(start + bytes[i] <= p.region_offset[j]
                                || p.region_offset[j] + bytes[j] <= start, "MSM region overlap");
                if (start + bytes[i] > end) end = start + bytes[i];
            }
        }
        require(!(p.workspace_bytes % 64) && p.workspace_bytes >= end
                && p.workspace_bytes <= static_cast<uint64_t>(PTRDIFF_MAX)
                && p.workspace_bytes <= static_cast<uint64_t>(SIZE_MAX)
                && p.host_prefix_offset == t && p.host_workspace_bytes == t + prefix_capacity,
                "MSM total host/device workspace");
    }
    MsmPlan sentinel{};
    require(make_msm_plan(2049, platform, 12, sentinel), "MSM sentinel plan");
    const auto reject_plan = [&](size_t n, MsmPlatform hw, uint32_t c) {
        MsmPlan out = sentinel;
        require(!make_msm_plan(n, hw, c, out) && same_msm_plan(out, sentinel),
                "MSM invalid plan accepted or output changed");
    };
    for (const uint32_t c : {1u,17u,UINT32_MAX}) reject_plan(65, platform, c);
    reject_plan(65, MsmPlatform{0,msm_required_ub_bytes}, 0);
    reject_plan(65, MsmPlatform{20,msm_required_ub_bytes-1}, 0);
    reject_plan(0, MsmPlatform{0,msm_required_ub_bytes}, 0);
    reject_plan(0, MsmPlatform{20,msm_required_ub_bytes-1}, 0);
    if (SIZE_MAX > UINT32_MAX) reject_plan(static_cast<size_t>(UINT32_MAX) + size_t{1}, platform, 0);

    constexpr uint64_t untouched = 0xdeadbeef12345678ULL;
    const auto check_schedule = [&](uint64_t n, const std::array<uint64_t,4>& histogram,
                                    const std::vector<uint64_t>& expected_counts) {
        MsmPlan p{};
        require(make_msm_plan(static_cast<size_t>(n), platform, 2, p), "MSM schedule plan");
        std::array<uint64_t,5> row{};
        for (unsigned b = 0; b < 4; ++b) row[b + 1] = row[b] + histogram[b];
        require(row[4] == n, "MSM test histogram total");
        const size_t stride = static_cast<size_t>(round8(5));
        const size_t capacity = p.region_size[static_cast<unsigned>(MsmRegion::TaskPrefix)];
        std::vector<uint64_t> storage(capacity / 8 + 2, untouched);
        auto* prefix = storage.data() + 1;
        MsmSchedule schedule{};
        require(make_msm_schedule(p, row.data(), prefix, schedule), "MSM schedule rejected");
        require(schedule.merge_levels + 1 == expected_counts.size(), "MSM actual merge levels");
        uint64_t divisor = 64;
        for (unsigned k = 0; k < 6; ++k) {
            require(schedule.task_count[k] == (k < expected_counts.size() ? expected_counts[k] : 0),
                    "MSM actual task count");
            if (k >= expected_counts.size()) continue;
            uint64_t total = 0;
            for (unsigned b = 0; b < 4; ++b) {
                require(prefix[k * stride + b] == total, "MSM bucket task prefix");
                if (b) total += ceil_div(histogram[b], divisor);
            }
            for (size_t b = 4; b < stride; ++b)
                require(prefix[k * stride + b] == total, "MSM task prefix tail padding");
            divisor *= 64;
        }
        require(schedule.order_offset == p.host_prefix_offset * expected_counts.size(),
                "MSM actual order offset");
        const auto* order = reinterpret_cast<const uint32_t*>(
            reinterpret_cast<const uint8_t*>(prefix) + schedule.order_offset);
        struct Task { uint64_t begin, count; unsigned bucket; };
        uint64_t model_size = 0;
        for (unsigned b = 1; b < 4; ++b) model_size += ceil_div(histogram[b], 64);
        const auto model = [&](uint64_t id) {
            for (unsigned b = 1; b < 4; ++b) {
                const uint64_t tasks = ceil_div(histogram[b], 64);
                if (id < tasks) {
                    const uint64_t begin = row[b] + id * 64;
                    return Task{begin, std::min(uint64_t{64}, row[b+1]-begin), b};
                }
                id -= tasks;
            }
            return Task{0,0,0};
        };
        require(model_size == schedule.task_count[0], "MSM row model task count");
        for (uint64_t i = 0; i < model_size; ++i) {
            const uint32_t id = order[i];
            require(id < model_size, "MSM task order range");
            const auto task = model(id);
            require(id >= prefix[task.bucket] && id < prefix[task.bucket+1]
                    && task.begin == row[task.bucket] + 64 * (id-prefix[task.bucket])
                    && task.begin + task.count <= row[task.bucket+1],
                    "MSM original bucket/begin boundaries");
            if (i) {
                const auto previous = order[i-1];
                require(model(previous).count >= task.count, "MSM descending task lengths");
                require(model(previous).count != task.count || previous < id,
                        "MSM stable equal lengths and unique IDs");
            }
        }
        const size_t written = schedule.order_offset + 4 * model_size;
        require(written <= capacity, "MSM actual metadata fits capacity");
        const auto* bytes = reinterpret_cast<const uint8_t*>(prefix);
        const auto* sentinel_bytes = reinterpret_cast<const uint8_t*>(&untouched);
        for (size_t i = written; i < capacity; ++i)
            require(bytes[i] == sentinel_bytes[i % 8], "MSM metadata unwritten tail");
        require(storage.front() == untouched && storage.back() == untouched,
                "MSM full prefix allocation canaries");
    };
    check_schedule(65, {0,65,0,0}, {2,1});
    check_schedule(2048, {0,2048,0,0}, {32,1});
    check_schedule(2049, {0,2049,0,0}, {33,1});
    check_schedule(UINT32_MAX, {0,0,0,UINT32_MAX}, {67108864,1048576,16384,256,4,1});
    check_schedule(65537, {0,0,65537,0}, {1025,17,1});
    check_schedule(130, {1,64,0,65}, {3,2});
    check_schedule(65537, {65537,0,0,0}, {0});
    check_schedule(64, {0,0,64,0}, {1});
    check_schedule(1, {0,0,0,1}, {1});
    check_schedule(385, {0,127,129,129}, {8,3});
    MsmPlan empty{};
    require(make_msm_plan(0, platform, 0, empty), "MSM empty schedule plan");
    MsmSchedule empty_out{};
    empty_out.merge_levels = 5;
    for (auto& count : empty_out.task_count) count = untouched;
    empty_out.order_offset = untouched;
    require(make_msm_schedule(empty, nullptr, nullptr, empty_out)
            && !empty_out.merge_levels && !empty_out.order_offset,
            "MSM empty null schedule rejected");
    for (const auto count : empty_out.task_count) require(!count, "MSM empty task count");

    MsmPlan p{};
    require(make_msm_plan(65, platform, 2, p), "MSM rejection schedule plan");
    const std::array<uint64_t,5> valid{0,0,65,65,65};
    const auto reject_schedule = [&](const uint64_t* row, bool null_prefix) {
        std::vector<uint64_t> prefix(
            p.region_size[static_cast<unsigned>(MsmRegion::TaskPrefix)] / 8, untouched);
        MsmSchedule out{};
        out.merge_levels = 123;
        out.order_offset = untouched;
        for (auto& count : out.task_count) count = untouched;
        require(!make_msm_schedule(p, row, null_prefix ? nullptr : prefix.data(), out),
                "MSM invalid schedule accepted");
        require(out.merge_levels == 123, "MSM failed schedule changed levels");
        require(out.order_offset == untouched, "MSM failed schedule changed order offset");
        for (const auto count : out.task_count) require(count == untouched, "MSM failed schedule changed counts");
        for (const auto value : prefix) require(value == untouched, "MSM failed schedule changed prefix");
    };
    reject_schedule(nullptr, false);
    reject_schedule(valid.data(), true);
    for (const auto& bad : std::array<std::array<uint64_t,5>,4>{{
             {1,1,65,65,65}, {0,32,31,65,65}, {0,0,64,64,64}, {0,0,66,66,65}}})
        reject_schedule(bad.data(), false);
    std::cout << "MSM_HOST_PASS\n";
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
        msm_host_suite();
        std::cout << "FIELD_MATH_HOST_PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FIELD_MATH_HOST_FAIL: " << error.what() << '\n';
        return 1;
    }
}
