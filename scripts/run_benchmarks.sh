#!/usr/bin/env bash

#Fail out on error
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Run the benchmark binary
for threads in 64 32 16 8 4 2 1; do
    RAYON_NUM_THREADS="$threads" cargo bench -p hyperplonk --bench hyperplonk-benches
done