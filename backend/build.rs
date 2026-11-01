//! Link and native-build wiring for the explicit `ascend910b3` backend feature.
//!
//! A CPU-only build never enters this script's native paths: it prints only the
//! environment rerun hints and returns. Enabling `ascend910b3` selects the real
//! native bridge, never a CPU fallback:
//!
//! * `HYPERPLONK_ASCEND910B3_LIB_DIR` (optional) supplies prebuilt
//!   `libhyperplonk_ascend910b3_bridge.a` and `libhyperplonk_ascend910b3_math.a`.
//!   Both archives must exist; the build fails instead of linking a silent stub.
//! * Otherwise the original `native/CMakeLists.txt` project is configured and
//!   built inside `OUT_DIR` against the installed CANN (`ASCEND_HOME_PATH`,
//!   optional `ASC_DIR` prefix). The `bridge` target produces the host C ABI
//!   archive from the real runtime/MLE/KZG/sumcheck sources and the accepted
//!   math/MSM target.
//!
//! The Rust `extern "C"` block's `#[link(name = "hyperplonk_ascend910b3_bridge",
//! kind = "static")]` is the authoritative directive for the bridge archive itself;
//! this script never emits it a second time. This script supplies the library
//! search directories, the math dependency, every transitive CANN static/shared
//! library in dependency order, and the runtime search path. Commands are executed
//! directly (no shell interpolation), and all native sources, headers and CANN
//! environment inputs are declared as versioned rerun inputs.

use std::{
    env,
    ffi::OsString,
    path::{Path, PathBuf},
    process::{Command, ExitStatus},
};

const BRIDGE_LIB: &str = "hyperplonk_ascend910b3_bridge";
const MATH_LIB: &str = "hyperplonk_ascend910b3_math";

/// Sources compiled into the single bridge archive, frozen by the implementation
/// contract. The CMake target lists them explicitly and treats any missing file as
/// a configuration error, so this list only drives rerun tracking here.
const BRIDGE_SOURCES: &[&str] = &[
    "bridge.asc",
    "bridge_runtime.cpp",
    "bridge_mle.cpp",
    "bridge_kzg.cpp",
    "bridge_sumcheck.cpp",
    "mle_kernels.asc",
    "sumcheck_kernels.asc",
];

/// Native headers whose edits must rerun this script even if CMake's own
/// dependency scan is bypassed by an incremental cache.
const BRIDGE_HEADERS: &[&str] = &[
    "bridge.h",
    "bridge_runtime.h",
    "field.h",
    "field_vector.h",
    "g1.h",
    "msm.h",
    "msm_plan.h",
    "math_kernels.h",
    "mle_kernels.h",
    "sumcheck_kernels.h",
];

/// Static CANN archives produced by CANN and consumed by the math target.
const CANN_STATIC_LIBS: &[&str] = &["tiling_api", "ascendc_runtime"];
/// Shared CANN libraries, in the same order as the native CMake target.
const CANN_SHARED_LIBS: &[&str] = &[
    "ascendcl",
    "register",
    "platform",
    "unified_dlog",
    "graph_base",
    "c_sec",
    "mmpa",
    "profapi",
    "runtime",
];

fn main() {
    println!("cargo:rerun-if-env-changed=HYPERPLONK_ASCEND910B3_LIB_DIR");
    println!("cargo:rerun-if-env-changed=ASCEND_HOME_PATH");
    println!("cargo:rerun-if-env-changed=ASC_DIR");
    println!("cargo:rerun-if-env-changed=CMAKE");

    if env::var_os("CARGO_FEATURE_ASCEND910B3").is_none() {
        return;
    }

    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("cargo always sets CARGO_MANIFEST_DIR"),
    );
    let native_dir = manifest_dir.join("src/ascend910b3/native");
    // Track every native source and header, including files added by the
    // runtime/MLE/KZG/sumcheck owners after this script was written.
    println!("cargo:rerun-if-changed={}", native_dir.display());
    println!("cargo:rerun-if-changed=build.rs");
    for file in BRIDGE_SOURCES.iter().chain(BRIDGE_HEADERS) {
        println!("cargo:rerun-if-changed={}", native_dir.join(file).display());
    }

    let cann_home = cann_home();
    let cann_lib_dir = cann_home.join("aarch64-linux/lib64");

    if let Some(directory) = env::var_os("HYPERPLONK_ASCEND910B3_LIB_DIR") {
        let directory = PathBuf::from(directory);
        for name in [BRIDGE_LIB, MATH_LIB] {
            let archive = directory.join(format!("lib{name}.a"));
            assert!(
                archive.is_file(),
                "HYPERPLONK_ASCEND910B3_LIB_DIR must contain {}, found no file at {}",
                archive.file_name().unwrap().to_string_lossy(),
                archive.display()
            );
            println!("cargo:rerun-if-changed={}", archive.display());
        }
        emit_link_directives(&directory, &cann_lib_dir);
        return;
    }

    let out_dir =
        PathBuf::from(env::var_os("OUT_DIR").expect("cargo always sets OUT_DIR for build scripts"));
    let build_dir = out_dir.join("ascend910b3-native");
    let cmake = env::var_os("CMAKE").unwrap_or_else(|| OsString::from("cmake"));
    let cann_cmake_prefix = cann_home.join("aarch64-linux/lib64/cmake");
    assert!(
        cann_cmake_prefix.is_dir(),
        "CANN CMake config directory not found at {}; set ASCEND_HOME_PATH to a CANN installation",
        cann_cmake_prefix.display()
    );

    let mut configure = Command::new(&cmake);
    configure
        .arg("-S")
        .arg(&native_dir)
        .arg("-B")
        .arg(&build_dir)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg(format!(
            "-DCMAKE_PREFIX_PATH={}",
            cmake_prefix_paths(&cann_cmake_prefix)
        ));
    run(configure, "configure native bridge");

    let mut build = Command::new(&cmake);
    build
        .arg("--build")
        .arg(&build_dir)
        .arg("--target")
        .arg(BRIDGE_LIB)
        .arg("--parallel");
    run(build, "build native bridge");

    for name in [BRIDGE_LIB, MATH_LIB] {
        let archive = build_dir.join(format!("lib{name}.a"));
        assert!(
            archive.is_file(),
            "native CMake build did not produce {}",
            archive.display()
        );
    }
    emit_link_directives(&build_dir, &cann_lib_dir);
}

fn cann_home() -> PathBuf {
    let home = env::var_os("ASCEND_HOME_PATH").unwrap_or_else(|| {
        panic!(
            "the ascend910b3 feature requires CANN: set ASCEND_HOME_PATH \
             (optionally ASC_DIR for the CMake package prefix)"
        )
    });
    let home = PathBuf::from(home);
    assert!(
        home.is_dir(),
        "ASCEND_HOME_PATH does not name an existing directory: {}",
        home.display()
    );
    home
}

fn cmake_prefix_paths(cann_cmake_prefix: &Path) -> String {
    let mut prefixes = vec![cann_cmake_prefix.to_path_buf()];
    if let Some(asc_dir) = env::var_os("ASC_DIR") {
        let asc_dir = PathBuf::from(asc_dir);
        if !prefixes.contains(&asc_dir) {
            prefixes.push(asc_dir);
        }
    }
    std::env::join_paths(prefixes)
        .expect("CANN CMake prefix paths contain no separator")
        .to_string_lossy()
        .into_owned()
}

/// Emit search directories and the transitive link line: math → CANN static →
/// CANN shared, in resolution order. The bridge archive is *not* repeated here;
/// `ffi`'s `#[link]` attribute owns that directive and finds the archive through
/// these search directories.
fn emit_link_directives(lib_dir: &Path, cann_lib_dir: &Path) {
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!(
        "cargo:rustc-link-search=native={}",
        cann_lib_dir.display()
    );
    println!("cargo:rustc-link-lib=static={MATH_LIB}");
    for name in CANN_STATIC_LIBS {
        println!("cargo:rustc-link-lib=static={name}");
    }
    for name in CANN_SHARED_LIBS {
        println!("cargo:rustc-link-lib=dylib={name}");
    }
    for name in ["dl", "m", "stdc++"] {
        println!("cargo:rustc-link-lib=dylib={name}");
    }
    // Rust binaries statically include the bridge but load CANN shared libraries;
    // record their location so tests and benches run without external wrapping.
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,{}",
        cann_lib_dir.display()
    );
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath-link,{}",
        cann_lib_dir.display()
    );
}

fn run(mut command: Command, operation: &str) {
    let status: ExitStatus = command
        .status()
        .unwrap_or_else(|error| panic!("failed to {operation}: {error}"));
    assert!(status.success(), "failed to {operation} (status {status})");
}
