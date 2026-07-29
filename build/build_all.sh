#!/usr/bin/env bash
# Build everything the harness imports, into one directory that goes on PYTHONPATH.
#
# Produces:
#   $OUT/k0_n1g*.so          single-kernel expert GEMM        (CMake)
#   $OUT/k0_n2*.so           two-phase expert GEMM            (CMake)
#   $OUT/tile_plan*.so       decode plan module               (hipcc + pybind)
#   $OUT/k0pf_plan*.so       prefill plan module              (hipcc + pybind)
#   $OUT/k0pf_frozen_plan*.so                                 (hipcc + pybind)
#   $OUT/hsaco/k0pf_gather.hsaco
#   $OUT/hsaco/k0pf_combine.hsaco
#   $OUT/hsaco/k0pf_quant.hsaco
#
# The .hip kernels under kernels/decode and kernels/prefill are NOT built here:
# the harness JIT-compiles them at run time through MoRI's loader, which is how
# they pick up MoRI's own include paths and device shmem globals.
#
# Run this inside the ROCm container, on a machine with hipcc.  GPUs are not
# required — it is a pure cross-compile for gfx950.
#
# Usage:
#   HIPKITTENS_ROOT=/path/to/HipKittens build/build_all.sh [output-dir]

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$HOME/moe-kittens-build}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
HK="${HIPKITTENS_ROOT:?set HIPKITTENS_ROOT to a HipKittens checkout (must contain include/kittens.cuh)}"
MORI_JIT="${MORI_JIT_SOURCES:-/opt/venv/lib/python3.12/site-packages/mori/_jit-sources}"
PYEXE="${PYTHON:-$(command -v python3)}"

mkdir -p "$OUT/hsaco" "$OUT/logs"
EXT="$("$PYEXE" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"

echo "repo            $REPO"
echo "output          $OUT"
echo "rocm            $ROCM_PATH"
echo "hipkittens      $HK"
echo "mori jit source $MORI_JIT"
echo "python          $PYEXE  (ext $EXT)"

FLAGS=(
  --offload-arch=gfx950
  -DKITTENS_CDNA4
  -DHIP_ENABLE_WARP_SYNC_BUILTINS
  -ffast-math
  -Rpass-analysis=kernel-resource-usage
  -mllvm -amdgpu-mfma-vgpr-form=1
)

echo
echo "== expert GEMM modules (CMake) =="
rm -rf "$OUT/cmake"
cmake -S "$REPO/build" -B "$OUT/cmake" \
  -DROCM_PATH="$ROCM_PATH" -DCMAKE_CXX_COMPILER="$ROCM_PATH/bin/hipcc" \
  -DHIPKITTENS_ROOT="$HK" -DPython3_EXECUTABLE="$PYEXE" \
  > "$OUT/logs/cmake_configure.log" 2>&1
echo "configure rc=$?"
make -C "$OUT/cmake" -j"$(nproc)" > "$OUT/logs/cmake_build.log" 2>&1
echo "build     rc=$?"
find "$OUT/cmake" -name '*.so' -exec cp -f {} "$OUT/" \;

echo
echo "== plan modules (hipcc + pybind) =="
PYINC=$("$PYEXE" -m pybind11 --includes)
for mod in tile_plan k0pf_plan k0pf_frozen_plan; do
  # shellcheck disable=SC2086
  "$ROCM_PATH/bin/hipcc" -shared -fPIC -std=c++20 \
    "${FLAGS[@]}" \
    -I"$HK/include" -I"$HK/prototype" -I"$ROCM_PATH/include/hip" \
    $PYINC \
    "$REPO/kernels/plan/$mod.cpp" \
    -o "$OUT/$mod$EXT" > "$OUT/logs/build_$mod.log" 2>&1
  echo "$mod rc=$?"
done

echo
echo "== prebuilt movement kernels (hipcc --genco) =="
# These three are loaded as .hsaco rather than JIT-compiled, so they need the
# union of MoRI's include paths and HipKittens'.
for mod in k0pf_gather k0pf_combine k0pf_quant; do
  "$ROCM_PATH/bin/hipcc" --genco \
    --offload-arch=gfx950 \
    -std=c++20 -O2 \
    -D__HIP_PLATFORM_AMD__ \
    -DHIP_ENABLE_WARP_SYNC_BUILTINS \
    -DMORI_DEVICE_NIC_IONIC \
    -DKITTENS_CDNA4 \
    -ffast-math \
    -I"$HK/include" -I"$HK/prototype" -I"$ROCM_PATH/include/hip" \
    -I"$MORI_JIT" -I"$MORI_JIT/include" -I"$MORI_JIT/src" \
    -I"$MORI_JIT/3rdparty/spdlog/include" \
    -I"$MORI_JIT/3rdparty/msgpack-c/include" \
    -I/usr/lib/x86_64-linux-gnu/openmpi/include \
    "$REPO/kernels/prefill/$mod.hip" -o "$OUT/hsaco/$mod.hsaco" \
    > "$OUT/logs/build_$mod.log" 2>&1
  echo "$mod rc=$?"
done

echo
echo "== artifacts =="
ls -la "$OUT"/*.so "$OUT"/hsaco/*.hsaco 2>/dev/null || echo "nothing produced — check $OUT/logs"

echo
echo "== register pressure and spills =="
grep -hiE 'Function|SGPR|VGPR|AGPR|Spill|Scratch|Occupancy' "$OUT/logs/cmake_build.log" 2>/dev/null | head -40
grep -hi spill "$OUT/logs"/*.log 2>/dev/null | head -20 || echo "(no spill remarks)"
