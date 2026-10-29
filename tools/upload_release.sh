#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
repo="yangsy1509/TCB-Detection"
tag="v1.0.0"
command -v gh >/dev/null || { echo "Install GitHub CLI and run: gh auth login"; exit 1; }
[[ -f release_assets/SHA256SUMS ]] || { echo "Run tools/package_dataset.py first."; exit 1; }
if ! gh release view "$tag" --repo "$repo" >/dev/null 2>&1; then
  gh release create "$tag" --repo "$repo" --title "TCB detection dataset v1.0.0" --notes "Full RGB, CTH and CTT NPY dataset. Download/verification instructions are in the README."
fi
gh release upload "$tag" release_assets/tcb-dataset-part-*.zip release_assets/SHA256SUMS --repo "$repo"
echo "Dataset release uploaded."
