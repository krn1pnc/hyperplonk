use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=HYPERPLONK_ASCEND910B3_LIB_DIR");
    if env::var_os("CARGO_FEATURE_ASCEND910B3").is_some() {
        if let Some(directory) = env::var_os("HYPERPLONK_ASCEND910B3_LIB_DIR") {
            let directory = PathBuf::from(directory);
            println!("cargo:rustc-link-search=native={}", directory.display());
            println!(
                "cargo:rerun-if-changed={}",
                directory
                    .join("libhyperplonk_ascend910b3_bridge.a")
                    .display()
            );
        }
    }
}
