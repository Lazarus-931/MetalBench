#!/usr/bin/env bash
# Bench a kernel on the M4 (lexie) — drop-in for ./bench when the goal is
# M4 perf. Rebuilds metallibs locally, rsyncs them + the .metal sources,
# then runs harness.py on lexie via SSH. Returns the exact stdout the
# remote harness produced, so you can grep "correctness", "speedup", etc.
#
# Usage:
#   ./bench_m4.sh <kernel_name>
#   ./bench_m4.sh <kernel_name> --no-save
set -uo pipefail
cd "$(dirname "$0")"

if [ $# -lt 1 ]; then
  echo "usage: ./bench_m4.sh <kernel_name> [extra harness args]" >&2
  exit 64
fi

NAME="$1"; shift
EXTRA=("$@")

# 1. Build locally (Metal IR is target-independent across M-series).
make --silent all 1>&2 || { echo "[bench_m4] local build failed" >&2; exit 1; }

# 2. Sync just-changed pieces. Keep this fast.
rsync -a --delete --exclude='.git' --exclude='__pycache__' --exclude='.mypy_cache' \
  --exclude='.DS_Store' --exclude='session.json' \
  metal/kernels/ lexie:/tmp/MetalBench/metal/kernels/ 1>&2
rsync -a build/ lexie:/tmp/MetalBench/build/ 1>&2

# 3. Run harness on lexie with SKIP_BUILD so it uses our prebuilt metallibs.
ssh lexie "cd /tmp/MetalBench && source /Users/lexie/mccl-store/.venv/bin/activate && \
  SKIP_BUILD=1 timeout 90 python3 mlx/scripts/harness.py '$NAME' --no-session ${EXTRA[*]:-}" 2>&1
