#include "field.h"
#include "field_math.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace hp = hyperplonk::ascend910b3;
namespace {
using hp::Fr;
constexpr std::array<uint32_t, 8> modulus = {{1, 0xffffffff, 0xfffe5bfe, 0x53bda402,
    0x09a1d805, 0x3339d808, 0x299d7d48, 0x73eda753}};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
bool equal(const Fr& a, const Fr& b) {
    for (unsigned i = 0; i < 8; ++i) if (a.words[i] != b.words[i]) return false;
    return true;
}
bool canonical(const Fr& a) {
    for (int i = 7; i >= 0; --i) {
        if (a.words[i] != modulus[i]) return a.words[i] < modulus[i];
    }
    return false;
}
void expect(const Fr& actual, const Fr& expected, const std::string& context) {
    require(canonical(actual), context + ": noncanonical result");
    require(equal(actual, expected), context + ": value mismatch");
}
std::vector<Fr> load(const std::filesystem::path& path, size_t count) {
    size_t bytes = 0;
    require(hp::detail::checked_field_bytes(count, bytes), "fixture size overflow");
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    require(bool(file), "missing fixture: " + path.string());
    require(file.tellg() == static_cast<std::streamoff>(bytes), "wrong fixture length: " + path.string());
    file.seekg(0);
    std::vector<Fr> values(count);
    for (size_t i = 0; i < count; ++i) {
        uint8_t input[32]{};
        file.read(reinterpret_cast<char*>(input), sizeof(input));
        require(bool(file), "truncated fixture: " + path.string());
        require(hp::from_le_bytes(input, values[i]), "noncanonical fixture: " + path.string());
        uint8_t output[32]{};
        hp::to_le_bytes(values[i], output);
        const auto limbs = hp::to_limbs(values[i]);
        for (unsigned j = 0; j < 32; ++j) {
            require(output[j] == input[j], "encoding roundtrip mismatch");
            require(limbs.bytes[j] == input[j], "limb encoding mismatch");
            const uint32_t expected = uint32_t(input[j]) << (8 * (j % 4));
            require((values[i].words[j / 4] & (uint32_t(0xff) << (8 * (j % 4)))) == expected,
                    "word little-endian mismatch");
        }
    }
    return values;
}
void boundaries() {
    const Fr zero{};
    const Fr one{{1}};
    Fr pm1{};
    for (unsigned i = 0; i < 8; ++i) pm1.words[i] = modulus[i];
    --pm1.words[0];
    uint8_t pbytes[32]{};
    for (unsigned i = 0; i < 32; ++i) pbytes[i] = uint8_t(modulus[i / 4] >> (8 * (i % 4)));
    Fr output = one;
    require(!hp::from_le_bytes(pbytes, output), "decoder accepted modulus");
    for (auto& byte : pbytes) byte = 0xff;
    require(!hp::from_le_bytes(pbytes, output), "decoder accepted all-FF");
    output = pm1;
    require(!hp::inverse(zero, output), "inverse zero reported success");
    require(equal(output, pm1), "inverse zero modified output");
    expect(hp::add(pm1, one), zero, "(p-1)+1");
    expect(hp::sub(zero, one), pm1, "0-1");
    expect(hp::mul(pm1, pm1), one, "(p-1)^2");
    expect(hp::neg(zero), zero, "neg zero");
    expect(hp::neg(one), pm1, "neg one");
    // Independent fixed constants: 2^512 mod p and (2^512 - 1) mod p.
    const Fr r512{{0xf3f29c6d,0xc999e990,0x87925c23,0x2b6cedcb,
                    0x7254398f,0x05d31496,0x9f59ff11,0x0748d9d9}};
    Fr r512minus1 = r512;
    --r512minus1.words[0];
    hp::FrWide wide{};
    wide.bytes[64] = 1;
    expect(hp::reduce_wide(wide), r512, "wide byte64");
    wide = {};
    for (unsigned i = 0; i < 64; ++i) wide.bytes[i] = 0xff;
    expect(hp::reduce_wide(wide), r512minus1, "maximum 512-bit integer");
    // Independent Python-integer oracle for (2^544 - 1) mod p.
    const Fr r544minus1{{0xefe9ec31,0x0408b03a,0x122df55f,0xc98a431e,
                         0x8277a7be,0xdd74d8f7,0x9163e436,0x1ef2c2b1}};
    for (auto& byte : wide.bytes) byte = 0xff;
    expect(hp::reduce_wide(wide), r544minus1, "maximum 544-bit integer");
    size_t bytes = 987;
    require(hp::detail::checked_field_bytes(0, bytes) && bytes == 0, "zero checked size");
    require(hp::detail::checked_field_bytes(1, bytes) && bytes == 32, "unit checked size");
    const size_t largest = static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) / 32;
    require(hp::detail::checked_field_bytes(largest, bytes) && bytes == largest * 32,
            "maximum checked size");
    bytes = 987;
    require(!hp::detail::checked_field_bytes(largest + 1, bytes) && bytes == 987,
            "ptrdiff overflow changes size or succeeds");
    require(!hp::detail::checked_field_bytes(std::numeric_limits<size_t>::max() / 32 + 1, bytes)
            && bytes == 987, "multiplication overflow changes size or succeeds");
}
void fixture_case(const std::filesystem::path& root, size_t count) {
    const auto dir = root / ("case-" + std::to_string(count));
    const auto lhs = load(dir / "lhs.bin", count);
    const auto rhs = load(dir / "rhs.bin", count);
    const auto added = load(dir / "add.bin", count);
    const auto subtracted = load(dir / "sub.bin", count);
    const auto multiplied = load(dir / "mul.bin", count);
    const auto scaled = load(dir / "scale.bin", count);
    const auto scalar = load(dir / "scalar.bin", 1);
    const auto dot = load(dir / "inner_product.bin", 1);
    Fr sum{};
    for (size_t i = 0; i < count; ++i) {
        const auto context = std::to_string(count) + "/" + std::to_string(i);
        expect(hp::add(lhs[i], rhs[i]), added[i], "add " + context);
        expect(hp::sub(lhs[i], rhs[i]), subtracted[i], "sub " + context);
        expect(hp::mul(lhs[i], rhs[i]), multiplied[i], "mul " + context);
        expect(hp::square(lhs[i]), hp::mul(lhs[i], lhs[i]), "square " + context);
        expect(hp::neg(lhs[i]), hp::sub(Fr{}, lhs[i]), "neg " + context);
        expect(hp::mul(lhs[i], scalar[0]), scaled[i], "scale " + context);
        sum = hp::add(sum, hp::mul(lhs[i], rhs[i]));
    }
    expect(sum, dot[0], "inner product " + std::to_string(count));
    if (count == 0 || count == 1 || count == 2 || count == 33 || count == 257) {
        const auto inputs = load(dir / "inverse_input.bin", count);
        const auto inverses = load(dir / "inverse.bin", count);
        for (size_t i = 0; i < count; ++i) {
            Fr result{};
            require(hp::inverse(inputs[i], result), "nonzero inverse failed");
            expect(result, inverses[i], "inverse golden " + std::to_string(i));
        }
    }
    if (count == 513 || count == 4097) {
        const auto chain = load(dir / "chain.bin", 1);
        auto state = lhs;
        for (unsigned round = 0; round < 10; ++round) {
            for (size_t i = 0; i < count; ++i)
                state[i] = hp::mul(hp::add(hp::mul(dot[0], state[i]), rhs[i]), lhs[i]);
        }
        Fr end{};
        for (size_t i = 0; i < count; ++i) end = hp::add(end, hp::mul(state[i], rhs[i]));
        expect(end, chain[0], "resident chain golden " + std::to_string(count));
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 3 && std::string(argv[1]) == "--fixtures", "usage: field_math_tests --fixtures DIRECTORY");
        boundaries();
        for (size_t count : {0u,1u,2u,31u,32u,33u,255u,256u,257u,513u,4097u})
            fixture_case(argv[2], count);
        std::cout << "FIELD_MATH_HOST_PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FIELD_MATH_HOST_FAIL: " << error.what() << '\n';
        return 1;
    }
}
