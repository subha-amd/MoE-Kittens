// ================================================================================================
// hkp.hpp — umbrella include for the hkp (HipKittens-Parallel) tier: a minimal primitive
// set for multi-GPU MoE kernels on AMD gfx950/CDNA4, distilled from the k0_fused_moe
// decode/prefill arms so existing kernels can be refactored onto it behavior-preservingly.
//
// Tier map (see README.md for the per-abstraction call-site table):
//   hkp_topology.hpp — peer address translation (MoRI p2pPeerPtrs / IRIS heap_bases)
//   hkp_store.hpp    — posted vectorized peer stores + pulls (cache-policy gated)
//   hkp_alloc.hpp    — remote slot reservation + wave destination dedup
//   hkp_sync.hpp     — monotonic-epoch doorbells, bounded waits, local grid barrier
//   hkp_sort.hpp     — destination counting sort stages + owner-combine CSR
//   hkp_quant.hpp    — bf16->FP8 e4m3 group quant + scale layouts + fused zero/transpose
//   hkp_gemm.hpp     — persistent expert-GEMM task decomposition + pipeline policies
// ================================================================================================

#ifndef HKP_HPP
#define HKP_HPP

#define HKP_VERSION 0x000100  // 0.1.0 — paper-core deliverable

#include "hkp_topology.hpp"
#include "hkp_store.hpp"
#include "hkp_alloc.hpp"
#include "hkp_sync.hpp"
#include "hkp_sort.hpp"
#include "hkp_quant.hpp"
#include "hkp_gemm.hpp"

#endif  // HKP_HPP
