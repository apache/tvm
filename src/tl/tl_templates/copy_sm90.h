#pragma once

#include <cute/arch/util.hpp>

#include "common.h"

namespace tl {

using namespace cute;

CUTE_DEVICE static void tma_load(void const* const desc_ptr, uint64_t& smem_mbar,
                                 void const* const smem_ptr, int32_t const& crd0) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3}], [%2];"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "r"(crd0)
      : "memory");
}

CUTE_DEVICE static void tma_load(void const* const desc_ptr, uint64_t& smem_mbar,
                                 void const* const smem_ptr, int32_t const& crd0,
                                 int32_t const& crd1) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4}], [%2];"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "r"(crd0), "r"(crd1)
      : "memory");
}

CUTE_DEVICE static void tma_load(void const* const desc_ptr, uint64_t& smem_mbar,
                                 void const* const smem_ptr, int32_t const& crd0,
                                 int32_t const& crd1, int32_t const& crd2) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5}], [%2];"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "r"(crd0), "r"(crd1), "r"(crd2)
      : "memory");
}

CUTE_DEVICE static void prefetch_tma_descriptor(void const* const desc_ptr) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  asm volatile("prefetch.tensormap [%0];" : : "l"(gmem_int_desc) : "memory");
}

CUTE_DEVICE static void mbarrier_init(uint64_t& smem_barrier, uint32_t arrive_count) {
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_barrier);
  asm volatile(
      "{\n\t"
      "mbarrier.init.shared.b64 [%1], %0; \n"
      "}"
      :
      : "r"(arrive_count), "r"(smem_int_ptr));
}

CUTE_DEVICE static void mbarrier_wait(uint64_t& smem_barrier, int phase_bit) {
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_barrier);
  asm volatile(
      "{\n"
      ".reg .pred                P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared.b64 P1, [%0], %1;\n"
      "@P1                       bra.uni DONE;\n"
      "bra.uni                   LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(smem_int_ptr),
      "r"(phase_bit));
}

CUTE_DEVICE static void mbarrier_arrive(uint64_t& smem_barrier) {
  uint32_t smem_int_ptr = cute::cast_smem_ptr_to_uint(&smem_barrier);
  uint64_t state = 0;
  asm volatile(
      "{\n\t"
      "mbarrier.arrive.shared.b64 %1, [%0];\n\t"
      "}"
      :
      : "r"(smem_int_ptr), "l"(state));
}

CUTE_DEVICE static void mbarrier_arrive_expect_tx(uint64_t& smem_barrier,
                                                  uint32_t transaction_bytes) {
  uint32_t smem_int_ptr = cute::cast_smem_ptr_to_uint(&smem_barrier);
  asm volatile(
      "{\n\t"
      "mbarrier.arrive.expect_tx.shared.b64 _, [%1], %0; \n\t"
      "}"
      :
      : "r"(transaction_bytes), "r"(smem_int_ptr));
}

}  // namespace tl