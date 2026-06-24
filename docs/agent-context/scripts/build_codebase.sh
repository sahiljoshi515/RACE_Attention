#!/usr/bin/env bash
set -euo pipefail
# build_codebase.sh - regenerate the auto-derived codebase inventory page.
#
# Writes pages/codebase/00-overview.md from the live repo tree: a directory map, a public-symbol
# index (Python classes/defs, C++ pybind m.def, CUDA __global__ kernels), and the README's
# install/usage. The curated narrative pages (cpu-kernels, gpu-kernels, python-api, scaling-module,
# tests-benchmarks, training-scripts, vllm-backend) are authored by hand and verified once; this
# script keeps the inventory + symbol map honest as the code moves. Requires: rg, find, git.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_PATH="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$DOCS_PATH/../.." && pwd)"
OUT="$DOCS_PATH/pages/codebase/00-overview.md"
mkdir -p "$DOCS_PATH/pages/codebase"

cd "$REPO_ROOT"
sha="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"

{
  echo "# Codebase overview (auto-generated inventory)"
  echo
  echo "Repo: github.com/sahiljoshi515/RACE_Attention  |  branch: \`$branch\`  |  HEAD: \`$sha\`"
  echo "Regenerate: scripts/build_codebase.sh  (the narrative codebase/* pages are hand-curated)"
  echo
  echo "## Directory map (excludes arXiv source + docs/)"
  echo
  echo '```'
  find . -type d \
    -not -path './.git*' -not -path './arXiv-2510*' -not -path './docs*' \
    -not -path '*/__pycache__*' -not -path '*/.ipynb_checkpoints*' |
    sed 's|^\./||' | sort | sed 's|[^/]*/|  |g'
  echo '```'
  echo
  echo "## Source files (code only)"
  echo
  echo '```'
  find . -type f \
    \( -name '*.py' -o -name '*.cu' -o -name '*.cpp' -o -name '*.h' -o -name '*.cuh' \) \
    -not -path './.git*' -not -path './arXiv-2510*' -not -path './docs*' \
    -not -path '*/__pycache__*' |
    sed 's|^\./||' | sort
  echo '```'
  echo
  echo "## Public symbol index"
  echo
  echo "Python classes / functions (kernels/, misc/, scaling/):"
  echo '```'
  rg -n --no-heading -g 'kernels/**/*.py' -g 'misc/**/*.py' -g 'scaling/**/*.py' \
    '^[[:space:]]*(class |def )' 2>/dev/null | sed 's|:[[:space:]]*| : |' || echo "(none found)"
  echo '```'
  echo
  echo "CUDA kernels (__global__) and pybind exports (m.def):"
  echo '```'
  rg -n --no-heading -g 'kernels/**/*.cu' '__global__[[:space:]]+\w+[[:space:]]+\w+' 2>/dev/null || true
  rg -n --no-heading -g 'kernels/**/*.cpp' 'm\.def\(' 2>/dev/null || true
  echo '```'
  echo
  echo "## Install & usage (from README.md)"
  echo
  if [[ -f README.md ]]; then
    echo "See README.md. Quickstart: \`pip install -r requirements.txt\`; notebooks/ for runnable"
    echo "examples; CPU kernels build via kernels/cpu (JIT load_ext); CUDA kernels JIT-compile via"
    echo "kernels/gpu/race_cuda_build.py:load_ext() targeting sm_90 (Hopper/H200)."
  fi
  echo
  echo "---"
  echo "Source: live repo tree at HEAD $sha. See the curated pages codebase/{cpu-kernels,gpu-kernels,python-api,scaling-module,tests-benchmarks,training-scripts,vllm-backend} for narrative detail."
} >"$OUT"

echo "wrote $OUT (HEAD $sha)"
