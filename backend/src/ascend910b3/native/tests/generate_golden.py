#!/usr/bin/env python3
"""Generate exact canonical Fr fixtures using only Python integer arithmetic."""
import argparse
from pathlib import Path
import random

P = 52435875175126190479447740508185965837690552500527637822603658699938581184513
COUNTS = (0, 1, 2, 31, 32, 33, 255, 256, 257, 513, 4097)
INVERSE_COUNTS = (0, 1, 2, 33, 257)


def write_values(path, values):
    path.write_bytes(b"".join(value.to_bytes(32, "little") for value in values))


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
