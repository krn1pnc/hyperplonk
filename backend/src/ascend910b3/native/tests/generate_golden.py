#!/usr/bin/env python3
"""Generate exact canonical field fixtures using only Python integer arithmetic."""
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


if __name__ == "__main__":
    main()
