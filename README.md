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

`backend/src/ascend910b3/native/` contains independently built BLS12-381 Fr
arithmetic for DAV_2201 (verified with CANN 9.1.0). It borrows TensorZKP's
[limb/matrix approach](https://www.usenix.org/system/files/usenixsecurity26-lu-tao.pdf),
not its BabyBear field or CUDA implementation.

The internal C++ launchers provide add, subtract, multiply, inverse, inner
product and shared-scalar vector multiplication. The latter two use signed
INT8 Cube products with exact unsigned correction and integer reduction.
This does **not** implement the existing protocol bridge: its NotImplemented
behavior, Rust interfaces and C ABI remain unchanged.

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

`Fr` is a canonical 32-byte integer (eight least-significant-first words), not
Montgomery storage. `FrLimbs`, `FrWide` and signed matrix products are distinct
types. Native callers preflight sizes with `detail::checked_field_bytes`;
nonempty device inputs/outputs must not alias. Inverse returns a device mask:
zero input produces mask 0/output 0, not a valid inverse.

Launchers only enqueue work. Callers own the stream, buffers, cached device
tilings and a 512-byte-aligned, 26112-byte workspace per concurrent stream.
They retain every borrowed allocation until completion. Immutable MLE inputs
are never modified or freed by math; intermediate vectors and scalars remain
on device, with ping-pong outputs for evolving state. Future protocol kernels
can consume these device addresses without adding a math ABI.

The smoke program requires `MATRIX_INT8_PASS`, `HBM_RESIDENT_CHAIN_PASS` and
`FIELD_MATH_NPU_PASS`. Resident-chain cases upload inputs once, generate their
scalar on device, perform ten state-update rounds, then download only the
32-byte final result. Test-only forwarding wrappers assert no explicit
H2D/D2H, synchronization or device allocation/free during compute. This audits
calls from the test and static math library, not CANN shared-library internals,
and does not establish residency of the still-unimplemented protocol bridge.


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
