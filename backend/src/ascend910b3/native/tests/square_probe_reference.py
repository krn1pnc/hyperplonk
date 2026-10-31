#!/usr/bin/env python3
"""Generate only bounded private radix-12 square fixtures; no SDK/native imports."""
import argparse
import pathlib
import random
import struct

PRIMES = (
    ("fr", 22, int("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001", 16)),
    ("fq", 32, int("1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab", 16)),
)

def generate(output):
    output.mkdir(parents=True, exist_ok=True)
    for name, n, p in PRIMES:
        r = 1 << (12 * n)
        rinv = pow(r, -1, p)
        pinv = pow(p, -1, r)
        values = [0, 1, p - 1, p, p + 1, 2*p - 2, 2*p - 1]
        for bit in range((2*p - 1).bit_length()):
            if bit % 12 not in (0, 1, 11):
                continue
            for delta in (-1, 0, 1):
                a = (1 << bit) + delta
                if 0 <= a < 2*p:
                    values.append(a)
        rng = random.Random(0x26A511 + n)
        values.extend(rng.randrange(2*p) for _ in range(128))
        values = list(dict.fromkeys(values))
        with (output / (name + ".bin")).open("wb") as f:
            f.write(struct.pack("<III", 0x53515032, n, len(values)))
            for a in values:
                t = a*a
                q = (-t*pinv) % r
                u = (t + q*p) // r
                c = (t*rinv) % p
                if not (0 <= u < 2*p and c == u % p):
                    raise ArithmeticError("independent Montgomery oracle bound")
                for value in (a, c, u):
                    f.write(struct.pack("<" + "I"*n,
                        *((value >> (12*j)) & 4095 for j in range(n))))
    (output / "contract.txt").write_text(
        "Private ordinary integers 0<=A<2p; 2p excluded. R=2^(12n). "
        "Triples A, canonical A^2/R mod p, exact U=(A^2+((-A^2*p^-1) mod R)*p)/R.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=pathlib.Path)
    generate(parser.parse_args().output)
