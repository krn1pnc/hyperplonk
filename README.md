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

### Ascend910B3 internal field, G1 and MSM math

`backend/src/ascend910b3/native/` contains independently built BLS12-381
finite-field arithmetic for DAV_2201 (verified with CANN 9.1.0). A single
compile-time `FieldElement<Params>` implements both the scalar field `Fr` and
base field `Fq`, using canonical integers and exact Barrett reduction.
The Fr Cube operations borrow TensorZKP's
[limb/matrix approach](https://www.usenix.org/system/files/usenixsecurity26-lu-tao.pdf),
not its BabyBear field or CUDA implementation.

Internal C++ batch function templates provide Fr/Fq add, subtract, negate,
multiply, square, inverse and complete wide reduction. One `FieldVector<Field>`
engine uses exact int32 Vector arithmetic across 48 independent elements,
radix-2^12 digits (Fr22/Fq32), and UB-only deferred-carry CIOS Montgomery multiplication.
Modular add/subtract transiently pack pairs of digits into radix-2^24 carry
words (Fr11/Fq16), then unpack into the same public radix12 banks. This shared
integer-only path halves carry rounds without enlarging UB or changing
Montgomery multiplication; all live G1 exception masks remain untouched.
Fr inner product and shared-scalar vector multiplication retain signed INT8 Cube
products, exact unsigned correction and Scalar packing/reduction.
`g1.h` adds concrete host/device `G1Affine` and `G1Jacobian` value types over Fq.
`msm.h` adds native cuZK-style optimized Pippenger MSM. The protocol bridge is
still NotImplemented: Rust interfaces, C ABI and the 97-byte packet are unchanged.

```bash
source "$ASCEND_HOME_PATH/set_env.sh"
cmake -S backend/src/ascend910b3/native -B target/ascend910b3-native-math \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$ASCEND_HOME_PATH/aarch64-linux/lib64/cmake"
cmake --build target/ascend910b3-native-math --parallel 2
python3 backend/src/ascend910b3/native/tests/generate_golden.py \
  --output target/ascend910b3-native-math/fixtures
ctest --test-dir target/ascend910b3-native-math --output-on-failure
target/ascend910b3-native-math/native_math_smoke \
  --device 0 --fixtures target/ascend910b3-native-math/fixtures
```

Use an ACL logical device ID valid in your environment; it need not match the
physical ID printed by `npu-smi`. Missing CANN is a build error, not a CPU
fallback. The host field-test target itself has no ACL dependency.

`field_params.h` fixes the BLS12-381 primes, Barrett reciprocals, wide reduction
factors and BLS-only `FieldVectorParams` Montgomery constants at compile time;
elements contain no parameter pointers or runtime dispatch. Values retain the
following least-significant-word-first layout:

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
not alias. Both batch inverse paths use mask 0/output 0 for zero input,
not a valid inverse; this differs from the scalar value-type output contract.

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
in the r-order subgroup. Point arithmetic and cuZK bucket addressing
carry no constant-time guarantee.

All field batch launchers only enqueue work. Pointwise Fr/Fq calls borrow the
stream, typed device buffers and queried AIV core limit, use one 96 KiB UB arena
per block, and require no GM workspace. Tiling rejects platforms with less than
104 KiB UB (arena plus SDK reserve); no Scalar fallback is installed.
The Cube APIs additionally borrow cached device tilings and a 512-byte-aligned,
26112-byte workspace per concurrent stream. Callers retain every allocation
until completion. Immutable MLE inputs are never modified or freed by math;
intermediate vectors/scalars remain on device. No protocol math ABI was added.

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
synchronization, device or pinned-host allocation/free during compute. This audits calls
from the test and static math library, not CANN shared-library internals, and
does not establish residency of the still-unimplemented protocol bridge.

Field-only development avoids the unrelated G1/MSM suite:

```bash
target/ascend910b3-native-math/native_math_smoke \
  --device 0 --fixtures target/ascend910b3-native-math/fixtures --field-mode check
# Same resident inputs, ten adjacent same-API warmup/target groups per configuration:
target/ascend910b3-native-math/native_math_smoke \
  --device 0 --fixtures target/ascend910b3-native-math/fixtures --field-mode bench
```

`--field-mode check|bench|profile` is mutually exclusive with `--msm-case` and
`--window-bits`; omitting it retains the complete suite. `check` requires
`FIELD_ONLY_NPU_PASS`, exact Python-integer results, 64-byte guards, input
immutability, radix12 boundaries, counts47/48/49 and95/96/97, inverse masks, all 68/100 wide
bytes, independent streams and resident chains. `bench` emits `FIELD_PERF` JSON
for 44 configurations with all ten raw samples and strict-zero enqueue audits.
`profile` checks Fr/Fq mul4097 and Fr dot257 once each, prints raw counters and
`FIELD_PROFILE_DIAGNOSTIC`, and deliberately does not claim ordinary residency:
known CANN profiler-metadata pinned allocations remain visible.

Historical standalone-field measurements, before the radix12 MSM optimization,
used ACL device0 / Ascend910B3, CANN9.1.0 Release and resident completed API wall
time. The following radix15/64-lane medians in microseconds include
host submission and one completion wait, not device-only kernel time or MSM
speedup. All configurations and samples, including smaller counts and unchanged
Cube controls, are retained in `target/field-vector/{baseline,candidate}.jsonl`
and `target/field-vector/comparison.csv`; the immutable old binary is under
`target/field-scalar-baseline/`.

| Field | Operation | Elements | Scalar µs | Vector µs | Scalar/Vector |
|-------|-----------|----------|-----------|-----------|---------------|
| Fr | add | 4097 | 47.556 | 41.976 | 1.13 |
| Fr | sub | 4097 | 46.176 | 48.916 | 0.94 |
| Fr | neg | 4097 | 45.661 | 48.750 | 0.94 |
| Fr | mul | 4097 | 816.167 | 297.859 | 2.74 |
| Fr | square | 4097 | 810.457 | 293.920 | 2.76 |
| Fr | inverse | 257 | 80100.148 | 25177.919 | 3.18 |
| Fr | wide | 512 | 311.025 | 2118.921 | 0.15 |
| Fq | add | 4097 | 48.846 | 55.956 | 0.87 |
| Fq | sub | 4097 | 50.246 | 58.996 | 0.85 |
| Fq | neg | 4097 | 48.891 | 58.146 | 0.84 |
| Fq | mul | 4097 | 1664.920 | 539.023 | 3.09 |
| Fq | square | 4097 | 1654.154 | 540.178 | 3.06 |
| Fq | inverse | 257 | 243811.122 | 73898.548 | 3.30 |
| Fq | wide | 512 | 639.489 | 4403.991 | 0.15 |

Vectorization is not a uniform speedup: add/sub/neg include layout/launch
overhead, and exact bitwise-Horner wide reduction is about seven times slower
than the old Scalar Barrett path at512. No fallback hides those regressions.
The standalone batch API does not automatically vectorize point operators.
Production MSM now explicitly uses the in-kernel `G1Vector` path described below;
resource-free scalar `G1Jacobian` remains for host/reference and legacy smoke
code. Cube packing/reduction and single-core scheduling remain unchanged.

The diagnostic profiles in `target/field-vector-profile/` associate both mul4097
instances with nonzero `aiv_vec_int32_ratio`, zero fp16/fp32 arithmetic ratios,
and actual Vector-pipeline activity; the dot control has nonzero INT8 Cube
counters. `target/field-vector/hardware-evidence.json` retains the CSV identities
and raw profile audits. This is hardware-counter evidence, not merely an AIV
task label. The installed instruction decoder returns `<not available>` for the
extracted device ELF, so no instruction-by-instruction disassembly is claimed.

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
Production field kernels, test G1 primitives and the legacy resident chain
compile independently in `field_kernels.asc`, `g1_smoke.asc` and
`g1_chain_smoke.asc`. The ordinary C++ harness calls the field function templates
in `math_kernels.h`; test G1 launch declarations remain in `smoke_kernels.h`.
Neither boundary adds a protocol ABI. Host-check edits do not recompile kernels;
shared header changes still rebuild their consumers.

For the legacy device G1 smoke kernels, `G1Jacobian::operator+=` retains a
non-inlined call boundary to prevent heavy Fq/Barrett expressions from expanding
repeatedly into nested bucket loops and triggering expensive LLVM LICM. This
does not describe production MSM, which uses the Vector G1 stages below.
All ASC kernels retain Release `-O3`; no tests or residency audits are skipped.

Measured with CANN 9.1.0, DAV_2201 and four build jobs: a fresh native build took
85.14 seconds, host-harness-only rebuild 2.77 seconds, chain-only rebuild
41.98 seconds, and unchanged rebuild 0.08 seconds. These are build wall times
(excluding configuration, fixture generation and tests), not kernel timings.
The full host/NPU suite, all 16 G1 resident-chain cases and CPU backend
regression pass with this organization; per-kernel performance was not profiled.

The native MSM implementation follows pinned
[cuZK sparse/balanced Pippenger](https://github.com/Tao-Lu-123/cuZK/blob/88e3295cada379abe06240df23ec6c9fc0f648b2/depends/libff-cuda/scalar_multiplication/multiexp.cu),
adapted to DAV_2201 without CUDA ticket atomics: stable integer radix grouping,
CSR bucket rows, bounded 64-point tasks and fan-in-64 bucket-partial reduction.
Level-zero tasks are stably grouped by length across SIMD lanes, but retain
their original point order and partial-output positions. Weighted segments fit
the available AIV lanes with a minimum span of16; their final tree uses fan-in4
to avoid serial work on a nearly empty SIMD tile.
Every nonempty partial, merge and weighted-tree task seeds its accumulator from
the first point instead of adding it to identity. Weighted segments similarly
seed running/local sums while preserving per-lane descending order and tails.
Weighted segments restore absolute bucket weights; only the final result is
normalized. Each input scalar contributes one unsigned digit record per window,
including digit zero; bucket zero is excluded from point accumulation.

Production point stages use `g1_vector.h`: 48 lanes process independent
partial-sum tasks, merge tasks or weighted segments inside each AIV. The shared
Fr/Fq engine batches digit-row products with repeat strides and defers carry
propagation until the eliminated Montgomery coefficient and final carry sweep.
Partial sums use exact mixed-affine addition for freshly loaded affine bases.
Generic and mixed additions retain every exceptional case; an exact tile-wide
vote skips doubling only when no lane needs the equal-point result.
One NPU ingress kernel converts bases into a `96*N`-byte HBM workspace once per
`run_msm` call, amortized across all windows. Internal coordinates remain packed
Montgomery integers in both UB and HBM; public inputs and final output remain
canonical. Ingress time is included in the complete-MSM measurements below.
No per-Fq host launches or numerical host transfers are added.
Bucket materialization copies Montgomery 192-byte tuples and synthesizes
empty-bucket identities; it does not perform Montgomery conversions or field
arithmetic. Its UB tile is 64 records, with public prefix metadata controlling
DMA addresses.

The field arena is 88512 bytes. Eleven extra G1 field banks, a 9216-byte point
buffer and 3072-byte metadata buffer bring explicit UB usage to 168384 bytes.
`msm_required_ub_bytes=176576` includes the 8192-byte SDK reservation; plans below
that queried capacity are rejected before launch. Scalar reads cover public
indices/weights and derived Boolean control aggregates, never coordinate words
or Fq arithmetic. The same-point branch is data-dependent, not constant-time.

The first actual `2^20` stage profile attributed 99.527% of kernel time to
numerical stages: partial sums74.460%, weighted segments15.478%, weighted
tree8.866%, finish0.723%. Host/idle gaps were about29.7ms and metadata copies
about0.695ms. These are diagnostic measurements of the old radix15 Vector
binary, not profiles or timing samples of the current implementation.

A subsequent single-call diagnostic of the Montgomery-resident version attributes
69.782% of task time to partial sums, 14.926% to weighted segments and 7.961% to
the weighted tree. The once-per-call base conversion accounts for 0.775%.
This is stage attribution, not a warmed comparable timing sample.

The following complete-MSM measurements use identical inputs within each size,
distinct subgroup bases and full-width canonical random Fr scalars. Input
generation, allocation, file reads and initial upload are excluded; all device
stages, CPU metadata handoffs, final96-byte D2H and completion are included.
Every warmup and target exactly matched the independent arkworks0.6 result.

| Points | Implementation | Warmups | Timed samples | Reported statistic | Seconds |
|--------|----------------|---------|---------------|--------------------|---------|
| 2^20 | Previous radix15 Vector MSM | 1 | 1 | target | 47.703695 |
| 2^20 | Radix12 before resident-Montgomery IO | 1 | 3 | median | 4.589340 |
| 2^20 | Resident-Montgomery IO checkpoint | 1 | 3 | median | 4.189697 |
| 2^20 | Balanced tasks and weighted reduction checkpoint | 1 | 3 | median | 2.678516 |
| 2^20 | Current seeded accumulators and radix24 add/sub | 1 | 3 | median | 2.240379 |
| 2^20 | CPU,16Rayon workers | 1 | 3 | median | 1.581598 |
| 2^25 | Radix12 before resident-Montgomery IO | 1 | 1 | target | 75.465230 |
| 2^25 | CPU,16Rayon workers | 1 | 3 | median | 43.114322 |

Seeded accumulators and transient radix24 addition/subtraction reduce
million-point latency by 16.36% versus the balanced-reduction checkpoint, but
remain slower than CPU. The current binary passed one warmup and three exact
targets plus full host/native field, G1, MSM, replay and two-stream regressions.
No new `2^25` timing is claimed.
The old NPU and `2^25` figures remain 1+1 diagnostic checkpoints. These do not
establish final performance acceptance: CPU10x and `2^25<1s` remain unmet.
Tested exact-FP32-FMA and dual-base RNS/Cube candidates were slower and are not
integrated. Source, raw records, input/binary identities and rejected candidates
remain under `target/msm-hotspot/`; those generated artifacts are not
tracked. Reproduce inputs and warmed CPU/NPU comparisons with:

```bash
cargo run --release --locked \
  --manifest-path backend/src/ascend910b3/native/tests/msm_reference/Cargo.toml \
  -- target/msm-benchmark-inputs 1048576
mkdir -p target/msm-benchmark-results
target/ascend910b3-native-math/msm_benchmark \
  target/msm-benchmark-inputs target/msm-benchmark-results \
  --count 1048576 --warmups 1 --samples 3
# For 2^25 use 33554432 in both commands and separate input/result directories.
```

The CPU tool performs a separate reference call, one warmup and three targets.
It excludes preparation/serialization and includes arkworks' internal MSM
allocations. Native compute permits no allocations or unexpected explicit
copies. The current `2^20` run transfers8388736B D2H and12440440B H2D metadata
with16 internal syncs; device/pinned scratch is180736576/2949440B, including
the base cache and stable task order. No numerical intermediate is transferred
to the host. The historical pre-cache `2^25` checkpoint used8388736B D2H,
16779264B H2D metadata and1023447680/3146112B scratch; these are not current
plan capacities. Final96B D2H and one completion wait are separate.
Sorting and CSR are unchanged; point-coordinate arithmetic is Vector throughout.



```cpp
using namespace hyperplonk::ascend910b3::detail;
MsmPlan plan;
// Query actual AIV/UB capacity; no Cube tiling is needed.
bool supported = build_msm_plan(count, plan, 0); // 0 = automatic window
// After allocating/uploading inputs and both scratch buffers:
aclError status = run_msm(stream, bases, scalars, out,
                         device_scratch, device_scratch_bytes,
                         pinned_host_scratch, pinned_host_scratch_bytes, plan);
// Check status, then synchronize before consuming out or releasing storage.
```

This internal C++ API borrows HBM inputs (`96*N` bytes of canonical Affine bases,
`32*N` bytes of canonical Fr scalars), a 96-byte device output, and caller-owned
scratch. `msm_plan.h` is ACL-free and computes exact host/device capacities.
Device scratch includes `96*N` bytes for the packed Montgomery base cache;
allocate the queried plan capacity rather than retaining an older byte count.
Pinned scratch also includes the bounded task-order metadata; no point or field
values are added to numerical host transfers.
`N <= UINT32_MAX`; explicit windows are 2–16 bits, with 0 selecting the cuZK-style
automatic heuristic capped at 16. Platform requirements are at least one AIV
and 176576 UB bytes. Input/output addresses must be 32-byte aligned, scratch addresses
64-byte aligned. Allocate host scratch with `aclrtMallocHost`; if aligning an
interior pointer, retain and free the original allocation. Output and scratch
cannot overlap inputs or each other; read-only inputs may alias. Bases must be
valid subgroup points and scalars canonical; the native entrypoint does not
perform subgroup validation or decode external packets.

Unlike the Fr launchers, `run_msm` performs CPU metadata scheduling. For each
of `W=ceil(255/c)` windows it downloads exactly `8*(2^c+1)` bytes of CSR row
pointers and synchronizes the caller's stream. The existing one upload per active
window contains actual-level task prefixes followed by one uint32 original task
index per level-zero task. The CPU derives this length-balancing order from the
CSR rows; point/scalar and sorted-record arrays stay on device. Points, buckets
and partial points never return to the CPU.
No host/device allocation or free occurs inside `run_msm`. The measured million-
point case (`c=16`, `W=16`) transfers8388736B D2H and12440440B H2D metadata with
16 synchronizations. The new order contributes4050808B H2D across all windows;
there is no new copy call or numerical host transfer. Allocate queried pinned
and device capacities rather than reusing the previous smaller scratch sizes.

`ACL_SUCCESS` means internal metadata handoffs succeeded and the finish kernel
was enqueued, **not** that the final point is ready. Keep inputs, output and
both scratch allocations alive until stream completion, including after a
failure with earlier work outstanding. Do not share either scratch buffer
between concurrent calls. Invalid plans, alignment, sizes or overlapping ranges
return `ACL_ERROR_INVALID_PARAM` before enqueue; invalid downloaded CSR returns
`ACL_ERROR_INTERNAL_ERROR`; ACL copy/sync errors propagate unchanged.
For `N=0`, inputs and both scratch pointers may be null with zero capacity:
the finish kernel writes identity without metadata transfers or internal sync.

Production kernels compile separately in `msm_sort.asc`, `msm_rows.asc`,
`msm_accumulate.asc` and `msm_reduce.asc`; orchestration remains ordinary C++.
MSM fixtures use independent Python affine arithmetic with known generator
coefficients, distinct large-case bases and small-case direct-MSM cross-checks.
Run a single case without unrelated Cube tests using:

```bash
target/ascend910b3-native-math/native_math_smoke \
  --device 0 --fixtures target/ascend910b3-native-math/fixtures \
  --msm-case uniform-4097
# Optional override, only with --msm-case:
# --window-bits 16
```

The default smoke adds metadata diagnostics, complete MSM cases, scratch replay,
independent-stream storage, preflight rejection and guard/input-immutability
checks. `uniform-65536` is explicit-only. `MSM_CUZK_RESIDENT_PASS` reports exact
metadata bytes and synchronization counts; its audit permits only the two
metadata address ranges and requires zero other explicit copies and zero
compute allocations/frees. Final output download is separately audited as
96 bytes and one stream sync. `MSM_METADATA_NPU_PASS` and `MSM_NPU_PASS` augment,
not replace, the existing field/G1 markers; host acceptance adds `MSM_HOST_PASS`.

The milestone host/NPU gate passes metadata scenarios, full MSM, replay and
two-stream checks. It forces the mixed-vote-tail-192 fixture to window6,
so only lane47 encounters equal points before inverse cancellation; the
mixed-exceptions-65 fixture exercises nonunit-Z equal/opposite additions.
The original scalar G1 tests and protocol ABI remain unchanged.
On the checked 910B3, the queried platform has
40 AIV cores and 196352 UB bytes. The large single case uses 8005888 bytes of
device scratch and 262400 bytes of pinned host scratch; its measured explicit
metadata budget is 1310880 bytes D2H, 1377600 bytes H2D and 20 stream syncs.

**Pre-Vector profiling limitation with the checked CANN 9.1.0 runtime.**
The seven metric groups and sample-based data for `uniform-4097`,
`uniform-65536` and `hot-65537` are retained under `target/msm-profile/`.
All exported device sequences reach `hp_msm_finish`, but the profiled application
fails the strict host-allocation audit before validating its final output.
An allocation backtrace resolves to CANN's `AscendCFunctionGetMetaInfoKtype`
through `AscendCGetProfkTypeImpl` and the generated launch stub: `uniform-4097`
incurs 627 pinned allocation/free pairs for 627 launches under `msprof`, versus
zero during ordinary compute. `--ascendcl=off` does not remove them. The audit
has not been disabled or given profiler exceptions; temporary stack diagnostics
were removed. Do not interpret the collection script's exit 0 as application
success: inspect `msprof.log` for the actual pass/fail marker.

`target/msm-profile/stage-diagnostics.json` records the qualified device-stage
observations, exact task counts, seven metric CSV paths, sample databases and
CSR-to-next-stage handoff gaps. These are profiler-instrumented diagnostics,
not an accepted performance gate, comparable median MSM latency, or speedup.
CPU scheduler function time was not measured separately. Before Vector G1, the
recorded accumulation/weighted stages spent about 94–95% of reported AIV time
in scalar work. That historical bottleneck evidence is distinct from the new
MSM counter diagnostics and warmed end-to-end comparison above.



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
