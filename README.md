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

### Ascend910B3 internal field math

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
This does **not** implement G1, MSM or the existing protocol bridge: its
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

Launchers only enqueue work. Callers own the stream, buffers, cached device
tilings and a 512-byte-aligned, 26112-byte workspace per concurrent stream.
They retain every borrowed allocation until completion. Immutable MLE inputs
are never modified or freed by math; intermediate vectors and scalars remain
on device, with ping-pong outputs for evolving state. Future protocol kernels
can consume these device addresses without adding a math ABI.

The smoke program requires `MATRIX_INT8_PASS`, `HBM_RESIDENT_CHAIN_PASS`,
`FIELD_ELEMENT_NPU_PASS field=Fr`, `FIELD_ELEMENT_NPU_PASS field=Fq` and the
final `FIELD_MATH_NPU_PASS`. Test-only kernels instantiate the same class on
device for add/subtract/negate/multiply/square/inverse and wide reduction,
checking exact Python-integer golden across empty inputs, tails and multiple
tiles. Host tests also cover a full-storage 64-bit prime to exercise the final
addition carry, compiled parameters, strict decoding and zero inverse semantics.

Resident-chain cases (513 and 4097 elements) upload inputs once, generate their
scalar on device, perform ten state-update rounds, then download only the
32-byte final result. Test-only forwarding wrappers assert no explicit H2D/D2H,
synchronization or device allocation/free during compute. This audits calls
from the test and static math library, not CANN shared-library internals, and
does not establish residency of the still-unimplemented protocol bridge.
Fq device correctness does not establish G1/MSM performance.


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
