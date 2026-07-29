# Expert-GEMM pybind modules for gfx950 (CDNA4).
#
#   k0_n1g   single-kernel fused expert GEMM        (kernels/gemm/n1g_*)
#   k0_n2    two-phase expert GEMM, write-once W2   (kernels/gemm/n2_*)
#
# Both targets use identical compile flags, including
# -mllvm -amdgpu-mfma-vgpr-form=1, which selects the vgprcd MFMA form so the
# backend stages MFMA operands in the AGPR file and leaves the accumulators in
# ArchVGPRs.  Changing that flag changes the generated code substantially.
#
#   cmake -S build -B /tmp/moe-kittens-build \
#     -DROCM_PATH=/opt/rocm -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
#     -DHIPKITTENS_ROOT=<path to HipKittens> \
#     -DPython3_EXECUTABLE=$(which python3)
#   make -C /tmp/moe-kittens-build -j16

cmake_minimum_required(VERSION 3.21)
project(moe_kittens_gemm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm installation")
set(HIPKITTENS_ROOT "" CACHE PATH "HipKittens source root")
if(NOT EXISTS "${HIPKITTENS_ROOT}/include/kittens.cuh")
  message(FATAL_ERROR "HIPKITTENS_ROOT must contain include/kittens.cuh")
endif()

set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc" CACHE FILEPATH "hipcc" FORCE)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
find_package(pybind11 REQUIRED)

set(GEMM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/gemm")

set(MOE_KITTENS_FLAGS
  --offload-arch=gfx950
  -DKITTENS_CDNA4
  -DHIP_ENABLE_WARP_SYNC_BUILTINS
  -ffast-math
  -Rpass-analysis=kernel-resource-usage
  -mllvm
  -amdgpu-mfma-vgpr-form=1
)

pybind11_add_module(k0_n1g
  "${GEMM_DIR}/n1g_bindings.cpp"
  "${GEMM_DIR}/n1g_fused_moe.cpp"
)
target_include_directories(k0_n1g PRIVATE
  "${HIPKITTENS_ROOT}/include"
  "${HIPKITTENS_ROOT}/prototype"
  "${ROCM_PATH}/include/hip"
  "${GEMM_DIR}"
)
target_compile_options(k0_n1g PRIVATE ${MOE_KITTENS_FLAGS})

pybind11_add_module(k0_n2
  "${GEMM_DIR}/n2_bindings.cpp"
  "${GEMM_DIR}/n2_phase1.cpp"
  "${GEMM_DIR}/n2_phase2.cpp"
)
target_include_directories(k0_n2 PRIVATE
  "${HIPKITTENS_ROOT}/include"
  "${HIPKITTENS_ROOT}/prototype"
  "${ROCM_PATH}/include/hip"
  "${GEMM_DIR}"
)
target_compile_options(k0_n2 PRIVATE ${MOE_KITTENS_FLAGS})
