#!/usr/bin/env bash

#Fail out on error
set -e

# We want the code to panic if there is an integer overflow
export RUSTFLAGS="-C overflow-checks=on"

cargo test --locked --workspace --release
cargo check --locked --workspace --all-targets --no-default-features
cargo test --locked --workspace --no-run \
    --features backend/print-trace,subroutines/print-trace,hyperplonk/print-trace
cargo check --locked --workspace --lib --no-default-features \
    --features backend/print-trace,subroutines/print-trace,hyperplonk/print-trace
cargo bench --locked --workspace --no-run
