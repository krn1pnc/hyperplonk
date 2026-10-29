# Hyperplonk library
A linear-time FFT-free SNARK proof system (https://eprint.iacr.org/2022/1355.pdf).

## Disclaimer

**DISCLAIMER:** This software is provided "as is" and its security has not been externally audited. Use at your own risk.

## Development environment setup

### Install RUST

We recommend using nix for installing the correct version of rust and
additional libraries:

```bash
> curl -L https://nixos.org/nix/install | sh
```

### Compiling the project for the first time

```bash
> nix-shell
> cargo build
```

### Direnv

We recommend the following tools:

- [nix](https://nixos.org/download.html)
- [direnv](https://direnv.net/docs/installation.html)

Run `direnv allow` at the repo root. You should see dependencies (including Rust) being installed (the first time might take a while). 
Upon modification on `flake.nix`, run `direnv reload` to reflect new dependencies or environment setups.

### Tests

```
> cargo test --release --all
```

### Ascend910B3 internal field and G1 math

`backend/src/ascend910b3/native/` contains independently built BLS12-381
finite-field arithmetic for DAV_2201 (verified with CANN 9.1.0). A single
compile-time `FieldElement<Params>` implements both the scalar field `Fr` and
base field `Fq`, using canonical integers and exact Barrett reduction.
The Fr Cube operations borrow TensorZKP's
[limb/matrix approach](https://www.usenix.org/system/files/usenixsecurity26-lu-tao.pdf),
not its BabyBear field or CUDA implementation.

The existing internal C++ batch launchers provide Fr add, subtract, multiply,
inverse, inner product and shared-scalar vector multiplication. The latter two
use signed INT8 Cube products with exact unsigned correction and integer
reduction. Fq is a host/device value type, not a new production batch API.
`g1.h` adds concrete host/device `G1Affine` and `G1Jacobian` value types over Fq.
This does **not** implement production MSM or the existing protocol bridge: its
NotImplemented behavior, Rust interfaces and C ABI remain unchanged.

```bash
source "$ASCEND_HOME_PATH/set_env.sh"
cmake -S backend/src/ascend910b3/native -B target/ascend910b3-native-math \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$ASCEND_HOME_PATH/aarch64-linux/lib64/cmake"
cmake --build target/ascend910b3-native-math --parallel
python3 backend/src/ascend910b3/native/tests/generate_golden.py \
  --output target/ascend910b3-native-math/fixtures
ctest --test-dir target/ascend910b3-native-math --output-on-failure
target/ascend910b3-native-math/native_math_smoke \
  --device 0 --fixtures target/ascend910b3-native-math/fixtures
```

Use an ACL logical device ID valid in your environment; it need not match the
physical ID printed by `npu-smi`. Missing CANN is a build error, not a CPU
fallback. The host field-test target itself has no ACL dependency.

`field_params.h` fixes the BLS12-381 primes, Barrett reciprocals and wide
reduction factors at compile time; elements contain no parameter pointers or
runtime dispatch. Values have the following least-significant-word-first layout:

| Type | Prime bits | `uint32_t` words | Bytes | Alignment |
|------|------------|------------------|-------|-----------|
| `Fr` | 255 | 8 | 32 | 4 |
| `Fq` | 381 | 12 | 48 | 4 |

Both use canonical integers, not Montgomery storage. The shared class provides
zero/one, equality, add/subtract/negate/multiply/square, inverse, strict
little-endian decoding/encoding, limb bytes and full wide-integer reduction.
Fr/Fq cannot mix implicitly. Raw `words` writers must supply canonical arithmetic
inputs. Decoding rejects values at least the modulus without changing its output;
value-type `inverse(out)` returns false for zero and leaves `out` unchanged.
Arithmetic is not claimed constant-time.

`LimbBytes<Params>` and `WideInteger<Params>` are separate from field values.
Their Fr/Fq aliases occupy 32/48 and 68/100 bytes respectively; all wide bytes,
including the top carry word, participate in reduction. Signed matrix products
remain Fr-specific. Native callers preflight array sizes with
`detail::checked_field_bytes<Fr/Fq>`; nonempty batch device inputs/outputs must
not alias. The Fr batch inverse convention is mask 0/output 0 for zero input,
not a valid inverse.

`G1Affine` stores two Fq coordinates (96 bytes, alignment 4); `G1Jacobian`
stores three (144 bytes, alignment 4). They are resource-free aggregates, not the
97-byte G1 interchange packet. Affine identity is `(0,0)`; Jacobian identity is
recognized by `Z=0`, and point operations produce canonical `(0,1,0)` identity.
Finite Jacobian coordinates mean `(X/Z²,Y/Z³)`, on `y²=x³+4`.

The point formulas follow pinned
[cuZK BLS12-381 G1](https://github.com/Tao-Lu-123/cuZK/blob/88e3295cada379abe06240df23ec6c9fc0f648b2/depends/libff-cuda/curves/bls12_381/bls12_381_g1.cu):
ordinary Jacobian addition, typed Affine mixed addition, doubling, negation,
projective equality, explicit conversions and variable-time `mul_by_u64`.
Self-addition is alias-safe. Equality does not normalize; `to_affine()` uses
one Fq inverse only when Z is neither zero nor one. There is no implicit
Affine/Jacobian conversion or full-Fr scalar multiplication API.

Raw point writers must supply canonical Fq coordinates and valid curve points;
production SRS inputs additionally need prior subgroup validation.
`is_on_curve()` is not a subgroup check: `(0,2)` is on this curve but is not
in the r-order subgroup. Point arithmetic and future cuZK bucket addressing
carry no constant-time guarantee.

Launchers only enqueue work. Callers own the stream, buffers, cached device
tilings and a 512-byte-aligned, 26112-byte workspace per concurrent stream.
They retain every borrowed allocation until completion. Immutable MLE inputs
are never modified or freed by math; intermediate vectors and scalars remain
on device, with ping-pong outputs for evolving state. Future protocol kernels
can consume these device addresses without adding a math ABI.

The smoke program requires `MATRIX_INT8_PASS`, `HBM_RESIDENT_CHAIN_PASS`,
`FIELD_ELEMENT_NPU_PASS field=Fr/Fq`, G1 primitive/diagnostics/generator markers,
all 16 `G1_CUZK_RESIDENT_PASS` markers, `G1_NPU_PASS`, and the final
`FIELD_MATH_NPU_PASS`. Field kernels check exact Python-integer golden across
empty inputs, tails and multiple tiles. Host tests also cover a full-storage
64-bit prime, compiled parameters, strict decoding and zero inverse semantics.
The ACL-free host executable requires `G1_HOST_PASS` before its final
`FIELD_MATH_HOST_PASS`.

Resident-chain cases (513 and 4097 elements) upload inputs once, generate their
scalar on device, perform ten state-update rounds, then download only the
32-byte final result. Test-only forwarding wrappers assert no explicit H2D/D2H,
synchronization or device allocation/free during compute. This audits calls
from the test and static math library, not CANN shared-library internals, and
does not establish residency of the still-unimplemented protocol bridge.

G1 fixtures use an independent Python affine oracle, not the candidate Jacobian
formulas. Primitive tests cover counts 0/1/2/15/16/17/31/32/33/65, scaled
projective equality, inverse points, noncanonical Z=0 inputs, self-aliasing,
full-width u64 factors, non-unit-Z normalization, curve diagnostics and exact
generator coordinates. Pair-owned DMA preserves compact 96/144-byte layouts;
guards, physical scalar padding, flag-slot padding and input immutability are
checked after execution.

The test-only cuZK structure chain uses windows 3 and 4 over eight fixtures:
counts 0/1/2/7/33, cancellation, zero scalars and identity bases. It computes
two bucket partials, two weighted segments per window, then combines windows.
Bases and Fr scalars upload once; intermediate Jacobians stay in HBM between
kernels. Only the final point is normalized and downloaded as 96 native bytes.
Its compute audit requires zero explicit copies, host synchronization and
device allocation/free; the tail requires exactly one download and one stream
synchronization. Empty MSM allocates only its output/pinned result and executes
the identity-output finish path. This is a small correctness/residency test,
not production sorting, ELL/CSR scheduling, a protocol commit implementation,
or evidence of MSM performance.

For mathematical iterations, build/run `field_math_tests` first; it requires
neither SDK initialization nor NPU compilation. This does not replace device
verification. The smoke harness is ordinary C++ (`tests/math_smoke.cpp`).
Its field, G1 primitive and resident-chain kernels compile independently in
`field_smoke.asc`, `g1_smoke.asc` and `g1_chain_smoke.asc`. Test-only launch
declarations in `smoke_kernels.h` keep device headers and launch syntax out of
the harness; they do not add a protocol ABI. Editing a host check no longer
recompiles kernels, and editing a chain kernel leaves the field/primitive units
cached. Shared header changes still rebuild their consumers.

On device only, `G1Jacobian::operator+=` retains a non-inlined call boundary
to prevent heavy Fq/Barrett expressions from expanding repeatedly into nested
bucket loops and triggering expensive LLVM LICM. Fq arithmetic remains optimized
within that function; host inlining, point formulas and layouts are unchanged.
All ASC kernels retain Release `-O3`; no tests or residency audits are skipped.

Measured with CANN 9.1.0, DAV_2201 and four build jobs: a fresh native build took
85.14 seconds, host-harness-only rebuild 2.77 seconds, chain-only rebuild
41.98 seconds, and unchanged rebuild 0.08 seconds. These are build wall times
(excluding configuration, fixture generation and tests), not kernel timings.
The full host/NPU suite, all 16 G1 resident-chain cases and CPU backend
regression pass with this organization; per-kernel performance was not profiled.


### Generate and read the documentation

#### Standard

```
> cargo doc --open
```

### Code formatting

To format your code run

```
> cargo fmt
```

### Updating non-cargo dependencies

Run `nix flake update` if you would like to pin other version edit `flake.nix`
beforehand. Commit the lock file when happy.

To update only a single input specify it as an argument, for example

    nix flake update github:oxalica/rust-overlay

### Benchmarks

To obtain benchmarks, run the script file `scripts/run_benchmarks.sh`. 
We refer to Table 5 and Table 6 in https://eprint.iacr.org/2022/1355.pdf for an example benchmark.
