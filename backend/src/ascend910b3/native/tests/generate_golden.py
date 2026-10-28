#!/usr/bin/env python3
"""Generate exact canonical field and G1 fixtures using Python integer arithmetic."""
import argparse
from pathlib import Path
import random

P = 52435875175126190479447740508185965837690552500527637822603658699938581184513
Q = 4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787
COUNTS = (0, 1, 2, 31, 32, 33, 255, 256, 257, 513, 4097)
INVERSE_COUNTS = (0, 1, 2, 33, 257)


def write_values(path, values, width=32):
    path.write_bytes(b"".join(value.to_bytes(width, "little") for value in values))


def generate(output):
    rng = random.Random(20261028)
    # Values exercise canonical high bytes and carries/borrows at every word boundary.
    edges = [0, 1, P - 1, P - 2, 1 << 248, (1 << 248) - 1]
    for bits in (8, 16, 32, 64, 96, 128, 160, 192, 224):
        edges.extend(((1 << bits) - 1, 1 << bits, (1 << bits) + 1))
    for count in COUNTS:
        directory = output / f"case-{count}"
        directory.mkdir(parents=True, exist_ok=True)
        lhs = [edges[i] if i < len(edges) else rng.randrange(P) for i in range(count)]
        rhs = [edges[(i + 2) % len(edges)] if i < len(edges) else rng.randrange(P)
               for i in range(count)]
        write_values(directory / "lhs.bin", lhs)
        write_values(directory / "rhs.bin", rhs)
        write_values(directory / "scalar.bin", [P - 1])
        for name, values in (
            ("add", [(a + b) % P for a, b in zip(lhs, rhs)]),
            ("sub", [(a - b) % P for a, b in zip(lhs, rhs)]),
            ("mul", [(a * b) % P for a, b in zip(lhs, rhs)]),
            ("neg", [-a % P for a in lhs]),
            ("square", [a * a % P for a in lhs]),
            ("scale", [a * (P - 1) % P for a in lhs]),
        ):
            write_values(directory / f"{name}.bin", values)
        dot = sum(a * b for a, b in zip(lhs, rhs)) % P
        write_values(directory / "inner_product.bin", [dot])
        if count in INVERSE_COUNTS:
            values = [a if a else 1 for a in lhs]
            write_values(directory / "inverse_input.bin", values)
            write_values(directory / "inverse.bin", [pow(a, P - 2, P) for a in values])
        if count in (513, 4097):
            state = lhs[:]
            for _ in range(10):
                state = [((dot * x + b) * a) % P for x, a, b in zip(state, lhs, rhs)]
            write_values(directory / "chain.bin", [sum(x * b for x, b in zip(state, rhs)) % P])


def field_edges(p):
    values = [0, 1, p - 1, p - 2]
    for bits in range(16, p.bit_length(), 16):
        values.extend(value for value in ((1 << bits) - 1, 1 << bits, (1 << bits) + 1)
                      if value < p)
    return values


def generic_cases(output, p, width, seed, counts):
    rng = random.Random(seed)
    edges = field_edges(p)
    for count in counts:
        directory = output / f"case-{count}"
        directory.mkdir(parents=True, exist_ok=True)
        lhs = [edges[i] if i < len(edges) else rng.randrange(p) for i in range(count)]
        rhs = [edges[(i + 2) % len(edges)] if i < len(edges) else rng.randrange(p)
               for i in range(count)]
        for name, values in (
            ("lhs", lhs), ("rhs", rhs),
            ("add", [(a + b) % p for a, b in zip(lhs, rhs)]),
            ("sub", [(a - b) % p for a, b in zip(lhs, rhs)]),
            ("mul", [(a * b) % p for a, b in zip(lhs, rhs)]),
            ("neg", [-a % p for a in lhs]),
            ("square", [a * a % p for a in lhs]),
        ):
            write_values(directory / f"{name}.bin", values, width)
        if count in INVERSE_COUNTS:
            values = [a if a else 1 for a in lhs]
            write_values(directory / "inverse_input.bin", values, width)
            write_values(directory / "inverse.bin", [pow(a, p - 2, p) for a in values], width)


def extra_fixtures(output, p, width, seed, params):
    params.mkdir(parents=True, exist_ok=True)
    r2 = 1 << (16 * width)
    write_values(params / "modulus.bin", [p], width)
    write_values(params / "mu.bin", [r2 // p], width + 2)
    write_values(params / "wide_factor.bin", [r2 % p], width)
    rng = random.Random(seed)
    wide_width = 2 * width + 4
    wide = [0, p - 1, p, p + 1, r2 - 1, r2, (0x89abcdef << (16 * width)) + p,
            (1 << (8 * wide_width)) - 1]
    wide.extend(rng.getrandbits(8 * wide_width) for _ in range(512 - len(wide)))
    write_values(output / "wide_input.bin", wide, wide_width)
    write_values(output / "wide_result.bin", [a % p for a in wide], width)

    # Independent extra vectors cover every radix boundary without altering the
    # original Fr fixture stream or its Cube/resident-chain regression inputs.
    edges = field_edges(p)
    pairs = [(a, b) for a in edges for b in (0, 1, p - 1, p - 2)]
    directory = output / "boundaries"
    directory.mkdir(parents=True, exist_ok=True)
    for name, values in (
        ("lhs", [a for a, _ in pairs]), ("rhs", [b for _, b in pairs]),
        ("add", [(a + b) % p for a, b in pairs]),
        ("sub", [(a - b) % p for a, b in pairs]),
        ("mul", [(a * b) % p for a, b in pairs]),
        ("neg", [-a % p for a, _ in pairs]),
        ("square", [a * a % p for a, _ in pairs]),
    ):
        write_values(directory / f"{name}.bin", values, width)


def g1_affine_neg(point):
    return None if point is None else (point[0], -point[1] % Q)


def g1_affine_add(lhs, rhs):
    """Independent affine oracle: no projective arithmetic or bucket formulas."""
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    x1, y1 = lhs
    x2, y2 = rhs
    if x1 == x2:
        if (y1 + y2) % Q == 0:
            return None
        slope = 3 * x1 * x1 * pow(2 * y1, -1, Q) % Q
    else:
        slope = (y2 - y1) * pow((x2 - x1) % Q, -1, Q) % Q
    x3 = (slope * slope - x1 - x2) % Q
    return x3, (slope * (x1 - x3) - y1) % Q


def g1_affine_mul(point, scalar):
    """Multiply by an ordinary nonnegative integer, never by a reduced Fr."""
    result = None
    while scalar:
        if scalar & 1:
            result = g1_affine_add(result, point)
        scalar >>= 1
        if scalar:
            point = g1_affine_add(point, point)
    return result


def g1_affine_on_curve(point):
    return point is None or (point[1] * point[1] - point[0] ** 3 - 4) % Q == 0


def write_g1_affine(path, points):
    write_values(path, (coordinate for point in points
                        for coordinate in ((0, 0) if point is None else point)), 48)


def write_g1_jacobian(path, points):
    write_values(path, (coordinate for point in points for coordinate in point), 48)


def g1_scaled(point, index, rng):
    if point is None:
        return 0, 1, 0
    residue = index % 4
    z = (1, 2, Q - 1)[residue] if residue < 3 else rng.randrange(1, Q)
    x, y = point
    return x * z * z % Q, y * z * z * z % Q, z


def generate_g1(output):
    root = output / "g1"
    root.mkdir(parents=True, exist_ok=True)
    # Decimal coordinates are independent of the native generator's LE u32 limbs.
    generator = (
        3685416753713387016781088315183077757961620795782546409894578378688607592378376318836054947676345821548104185464507,
        1339506544944476473020471379941921221584933875938349620426543736416511423956333506472724655353366534992391756441569,
    )
    write_g1_affine(root / "generator.bin", [generator])
    neg_generator = g1_affine_neg(generator)
    twice = g1_affine_mul(generator, 2)
    neg_twice = g1_affine_neg(twice)
    three = g1_affine_mul(generator, 3)
    seventeen = g1_affine_mul(generator, 17)
    two_fifty_seven = g1_affine_mul(generator, 257)
    fixed_a = (None, generator, neg_generator, twice, neg_twice, three,
               seventeen, two_fifty_seven, None)
    fixed_b = (None, generator, generator, neg_twice, neg_generator, None,
               seventeen, generator, generator)
    factors = (0, 1, 2, 31, 32, 33, 255, 256, 257, (1 << 32) - 1,
               1 << 32, (1 << 64) - 1)
    for count in (0, 1, 2, 15, 16, 17, 31, 32, 33, 65):
        directory = root / f"case-{count}"
        directory.mkdir(parents=True, exist_ok=True)
        rng = random.Random(20261101 + count)
        affine_a, affine_b, jacobian_a, jacobian_b = [], [], [], []
        for index in range(count):
            if index < len(fixed_a):
                a, b = fixed_a[index], fixed_b[index]
            else:
                a = g1_affine_mul(generator, rng.randrange(1, P))
                choice = index % 4
                if choice == 0:
                    b = a
                elif choice == 1:
                    b = g1_affine_neg(a)
                elif choice == 2:
                    b = None
                else:
                    b = g1_affine_mul(generator, rng.randrange(1, P))
            ja = g1_scaled(a, index, rng)
            jb = g1_scaled(b, index, rng)
            if index == 0:
                ja, jb = (0, 0, 0), (7, 9, 0)
            elif index == 1:
                # Equal G with deliberately unequal raw Jacobian coordinates.
                ja = g1_scaled(a, 1, rng)
                jb = g1_scaled(b, 2, rng)
            elif index == 8:
                ja = (7, 9, 0)
                jb = g1_scaled(b, 1, rng)
            affine_a.append(a)
            affine_b.append(b)
            jacobian_a.append(ja)
            jacobian_b.append(jb)
        scalars = [factors[index % len(factors)] for index in range(count)]
        write_g1_affine(directory / "affine_a.bin", affine_a)
        write_g1_affine(directory / "affine_b.bin", affine_b)
        write_g1_jacobian(directory / "jacobian_a.bin", jacobian_a)
        write_g1_jacobian(directory / "jacobian_b.bin", jacobian_b)
        write_values(directory / "mul_u64.bin", scalars, 8)
        write_values(directory / "equal.bin",
                     [int(a == b) for a, b in zip(affine_a, affine_b)], 4)
        for name, points in (
            ("add", [g1_affine_add(a, b) for a, b in zip(affine_a, affine_b)]),
            ("double", [g1_affine_add(a, a) for a in affine_a]),
            ("neg", [g1_affine_neg(a) for a in affine_a]),
            ("affine_neg", [g1_affine_neg(b) for b in affine_b]),
            ("mul", [g1_affine_mul(a, n) for a, n in zip(affine_a, scalars)]),
        ):
            write_g1_affine(directory / f"{name}.bin", points)

    diagnostics = root / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    affine = [None, generator, (0, 2), (0, 1),
              (generator[0], (generator[1] + 1) % Q)]
    jacobian = [(0, 1, 0) if point is None else (*point, 1) for point in affine]
    flags = []
    for point, (x, y, z) in zip(affine, jacobian):
        affine_identity = point is None
        affine_on_curve = g1_affine_on_curve(point)
        jacobian_identity = z == 0
        jacobian_on_curve = jacobian_identity or (y * y - x ** 3 - 4 * z ** 6) % Q == 0
        flags.append(int(affine_identity) | (int(affine_on_curve) << 1)
                     | (int(jacobian_identity) << 2) | (int(jacobian_on_curve) << 3))
    assert flags == [15, 10, 10, 0, 0]
    write_g1_affine(diagnostics / "affine.bin", affine)
    write_g1_jacobian(diagnostics / "jacobian.bin", jacobian)
    write_values(diagnostics / "flags.bin", flags, 4)

    base_cycle = (generator, neg_generator, twice, neg_twice, three, None,
                  seventeen, two_fifty_seven)
    scalar_prefix = (P - 1, 1, 0, 1 << 254, (1 << 254) + 1, 7, P - 2)
    chain_cases = [(f"case-{count}", count) for count in (0, 1, 2, 7, 33)]
    chain_cases.extend((("cancel", 2), ("zero-scalars", 7), ("identity-bases", 7)))
    for name, count in chain_cases:
        directory = root / "chain" / name
        directory.mkdir(parents=True, exist_ok=True)
        rng = random.Random(20261102 + count)
        bases = [base_cycle[index % len(base_cycle)] for index in range(count)]
        scalars = [scalar_prefix[index] if index < len(scalar_prefix) else rng.randrange(P)
                   for index in range(count)]
        if name == "cancel":
            bases, scalars = [generator, neg_generator], [1, 1]
        elif name == "zero-scalars":
            scalars = [0] * count
        elif name == "identity-bases":
            bases = [None] * count
        expected = None
        for base, scalar in zip(bases, scalars):
            expected = g1_affine_add(expected, g1_affine_mul(base, scalar))
        write_g1_affine(directory / "bases.bin", bases)
        write_values(directory / "scalars.bin", scalars)
        write_g1_affine(directory / "expected.bin", [expected])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    generate(args.output)
    generic_cases(args.output / "fq", Q, 48, 20261029, COUNTS)
    prime64 = (1 << 64) - 59
    generic_cases(args.output / "prime64", prime64, 8, 20261030, (257,))
    for name, root, p, width, seed in (
        ("fr", args.output, P, 32, 20261028),
        ("fq", args.output / "fq", Q, 48, 20261029),
        ("prime64", args.output / "prime64", prime64, 8, 20261030),
    ):
        extra_fixtures(root, p, width, seed, args.output / "params" / name)
    generate_g1(args.output)


if __name__ == "__main__":
    main()
