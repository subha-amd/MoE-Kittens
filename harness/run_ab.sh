#!/usr/bin/env bash
# Run the region A/B on one 8-GPU node, in a container, N times.
#
#   harness/run_ab.sh decode  production,k0d_mega  [runs] [tag]
#   harness/run_ab.sh prefill production,pf4h      [runs] [tag]
#
# Arms are position-balanced and timed inside one process, so the comparison is
# same-run: there is no historical baseline anywhere in the result.
#
# Required environment:
#   MOE_KITTENS_BUILD   output directory of build/build_all.sh (goes on PYTHONPATH)
#   MOE_KITTENS_CORPUS  captured route/weight corpus for the regime (see docs/running.md)
# Optional:
#   MOE_KITTENS_IMAGE   container image (default below)
#   MOE_KITTENS_OUT     result directory root (default $HOME)
#   K0_NTIMED           timed rotations per arm (default 100)
#
# Shared-node rules this script enforces before every run: no other distributed
# job may be present, and all 8 GPUs must report 0% utilization.  Do not remove
# these checks — two concurrent 8-GPU jobs will wedge the node.

set -euo pipefail

regime="${1:?usage: run_ab.sh prefill|decode arms [runs] [tag]}"
arms="${2:?comma-separated arm list required}"
run_count="${3:-5}"
tag="${4:-primary}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${MOE_KITTENS_BUILD:?set MOE_KITTENS_BUILD to the build/build_all.sh output directory}"
CORPUS="${MOE_KITTENS_CORPUS:?set MOE_KITTENS_CORPUS to a captured corpus for this regime}"
IMAGE="${MOE_KITTENS_IMAGE:-rocm/atom-dev:vllm-latest}"
OUTROOT="${MOE_KITTENS_OUT:-$HOME}"
ntimed="${K0_NTIMED:-100}"

case "$regime" in
  prefill)
    T=4096;  tloc_max=40960; padmax=263136; maxtok=4096; maxtok_prod=32768
    blk=128; warp=16; push_grid=1024; transpose_grid=256
    ;;
  decode)
    T=64;    tloc_max=512;   padmax=5088;   maxtok=512;  maxtok_prod=512
    blk=64;  warp=4;  push_grid=16;   transpose_grid=64
    ;;
  *) echo "unknown regime: $regime" >&2; exit 2 ;;
esac

if [[ "$arms" == *k0d_* && "$regime" != "decode" ]]; then
  echo "k0d arms are decode-only; refusing regime=$regime" >&2
  exit 3
fi
if [[ "$arms" == *pf* && "$arms" != *k0d_* && "$regime" != "prefill" ]]; then
  echo "pf arms are prefill-only; refusing regime=$regime" >&2
  exit 3
fi

for run in $(seq 1 "$run_count"); do
  if pgrep -x torchrun >/dev/null || pgrep -x mpirun >/dev/null || pgrep -x orterun >/dev/null; then
    echo "refusing run $run: another distributed job is already running" >&2
    exit 20
  fi
  idle=$(/opt/rocm/bin/rocm-smi --showuse --csv 2>/dev/null | grep -c ',0$' || true)
  if [[ "$idle" -ne 8 ]]; then
    echo "refusing run $run: only $idle/8 GPUs report 0% use" >&2
    exit 21
  fi

  out="$OUTROOT/moekittens_${tag}_${regime}_run${run}"
  log="$out.log"
  if [[ -e "$out" || -e "$log" ]]; then
    echo "refusing to overwrite $out" >&2
    exit 22
  fi
  mkdir -p "$out"
  echo "starting regime=$regime arms=$arms run=$run at $(date -u +%FT%TZ)"

  timeout --signal=TERM "${MOE_KITTENS_TIMEOUT:-2400}" docker run --rm \
    --name "moekittens_${tag}_${regime}_r${run}" \
    --network=host --ipc=host \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
    -v "$REPO:$REPO" -v "$BUILD:$BUILD" -v "$OUTROOT:$OUTROOT" -v /data:/data \
    -e HSA_XNACK=1 -e MORI_GPU_ARCHS=gfx950 \
    -e MORI_SHMEM_HEAP_SIZE=34359738368 \
    -e K0_PYBIND_DIR="$BUILD" \
    -e K0_MORI_KERNELS_DIR=/opt/venv/lib/python3.12/site-packages/mori/_jit-sources/src/ops/kernels \
    -e K0_HIP_SRC="$REPO/kernels/decode/k0_region_kernels.hip" \
    -e K0D_SRC_DIR="$REPO/kernels/decode" \
    -e K0PF3_SRC_DIR="$REPO/kernels/prefill" \
    -e K0_HKP_DIR="$REPO/kernels/hkp" \
    -e K0_PF_HSACO_DIR="$BUILD/hsaco" \
    -e K0_REGION_CORPUS="$CORPUS" \
    -e K0_OUT_DIR="$out" -e K0_ARMS="$arms" \
    -e K0_T="$T" -e K0_T_LOC_MAX="$tloc_max" -e K0_PADMAX="$padmax" \
    -e K0_MAXTOK="$maxtok" -e K0_MAXTOK_PROD="$maxtok_prod" \
    -e K0_BLK="$blk" -e K0_WARP="$warp" \
    -e K0_PF2_PUSH_GRID="$push_grid" -e K0_PF2_TRANSPOSE_GRID="$transpose_grid" \
    -e K0_NTIMED="$ntimed" \
    -e K0_STRESS="${K0_STRESS:-30}" -e K0_XSTRESS="${K0_XSTRESS:-25}" \
    -e K0_SCRUB_MB=1024 \
    -e K0_DECODE_PHASE_PROFILE="${K0_DECODE_PHASE_PROFILE:-0}" \
    -e K0_STAGE_PROFILE="${K0_STAGE_PROFILE:-0}" \
    "$IMAGE" bash -lc \
    "cd '$REPO' && /opt/venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=8 harness/region_ab.py" \
    >"$log" 2>&1

  count=$(find "$out" -maxdepth 1 -name '*rank*.json' -type f | wc -l)
  if [[ "$count" -ne 8 ]]; then
    echo "run $run produced $count/8 rank JSON files — see $log" >&2
    exit 23
  fi
  echo "completed regime=$regime run=$run at $(date -u +%FT%TZ)"
done
