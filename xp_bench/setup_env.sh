#!/bin/bash
# One-shot setup for the xp_bench cross-backend benchmarks.
#
# Builds the `all-frameworks` pixi environment: numpy plus GPU torch, jax and cupy
# in a single solve group, so all four backends can be timed in one process. jaxlib
# GPU is a cuda12 build and cupy and torch have cuda12 builds too, which is what
# makes one shared GPU env possible. linux-64 only, because of cupy.
#
# This environment is deliberately not upstreamable, so its manifest block is not
# committed and disappears on every sync from main. Re-run this script afterwards.
# It is idempotent and safe to run on a fresh checkout of any branch.
#
# Run it from the scipy repository, whose manifest it edits. The harness itself lives
# outside that repository, so it takes the workspace from the working directory.
set -euo pipefail

MANIFEST=$(pixi info --json | python3 -c 'import json, sys
print(json.load(sys.stdin)["project_info"]["manifest_path"])')
echo "workspace manifest: $MANIFEST"

# matplotlib is needed for the plots and the PDF report and is not a dependency of
# any existing feature, so it gets its own feature that the environment includes.
if ! python3 - "$MANIFEST" <<'EOF'
import sys, tomllib

features = tomllib.load(open(sys.argv[1], "rb"))["feature"]
sys.exit("bench-plotting" not in features or "matplotlib" not in features["bench-plotting"]["dependencies"])
EOF
then
  echo "adding matplotlib to the bench-plotting feature"
  pixi add --manifest-path "$MANIFEST" --feature bench-plotting --no-install matplotlib
fi

FEATURES=(
  build-deps run-deps test-deps test-cpu bench-common mkl sparse array_api_strict
  jax-cuda torch-cuda marray cuda12 py-cuda cupy bench-plotting
)
echo "declaring the all-frameworks environment"
pixi workspace environment add --manifest-path "$MANIFEST" all-frameworks \
  "${FEATURES[@]/#/--feature=}" --solve-group all-frameworks --force

pixi install --manifest-path "$MANIFEST" -e all-frameworks

echo "all-frameworks env ready:"
pixi run --manifest-path "$MANIFEST" -e all-frameworks python -c "
import cupy, jax, matplotlib, torch
print('  torch cuda', torch.cuda.is_available(),
      '| jax', [d.platform for d in jax.devices()],
      '| cupy devices', cupy.cuda.runtime.getDeviceCount(),
      '| matplotlib', matplotlib.__version__)
"
