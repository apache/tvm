#pragma once

#include <cuda.h>

#include "common.h"

namespace tl {

TL_DEVICE void tma_load(const CUtensorMap& descriptor, uint64_t& smem_mbar,
                        void const* const smem_ptr, int32_t const& crd0) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_mbar = smem_ptr_to_uint(&smem_mbar);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3}], [%2];"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "r"(crd0)
      : "memory");
}

TL_DEVICE void tma_load(const CUtensorMap& descriptor, uint64_t& smem_mbar,
                        void const* const smem_ptr, int32_t const& crd0, int32_t const& crd1) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_mbar = smem_ptr_to_uint(&smem_mbar);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4}], [%2];"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "r"(crd0), "r"(crd1)
      : "memory");
}

TL_DEVICE void tma_load(const CUtensorMap& descriptor, uint64_t& smem_mbar,
                        void const* const smem_ptr, int32_t const& crd0, int32_t const& crd1,
                        int32_t const& crd2) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_mbar = smem_ptr_to_uint(&smem_mbar);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5}], [%2];"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "r"(crd0), "r"(crd1), "r"(crd2)
      : "memory");
}

TL_DEVICE void tma_load(const CUtensorMap& descriptor, uint64_t& smem_mbar,
                        void const* const smem_ptr, int32_t const& crd0, int32_t const& crd1,
                        int32_t const& crd2, int32_t const& crd3) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_mbar = smem_ptr_to_uint(&smem_mbar);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5, %6}], [%2];"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "r"(crd0), "r"(crd1), "r"(crd2),
        "r"(crd3)
      : "memory");
}

TL_DEVICE void tma_load(const CUtensorMap& descriptor, uint64_t& smem_mbar,
                        void const* const smem_ptr, int32_t const& crd0, int32_t const& crd1,
                        int32_t const& crd2, int32_t const& crd3, int32_t const& crd4) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_mbar = smem_ptr_to_uint(&smem_mbar);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "r"(crd0), "r"(crd1), "r"(crd2),
        "r"(crd3), "r"(crd4)
      : "memory");
}

TL_DEVICE void tma_store(const CUtensorMap& descriptor, void const* const smem_ptr,
                         int32_t const& crd0) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);

  asm volatile("cp.async.bulk.tensor.1d.global.shared::cta.bulk_group [%0, {%2}], [%1];"
               :
               : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0)
               : "memory");
}

TL_DEVICE void tma_store(const CUtensorMap& descriptor, void const* const smem_ptr,
                         int32_t const& crd0, int32_t const& crd1) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);

  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
               :
               : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1)
               : "memory");
}

TL_DEVICE void tma_store(const CUtensorMap& descriptor, void const* const smem_ptr,
                         int32_t const& crd0, int32_t const& crd1, int32_t const& crd2) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);

  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%2, %3, %4}], [%1];"
               :
               : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1), "r"(crd2)
               : "memory");
}

TL_DEVICE void tma_store(const CUtensorMap& descriptor, void const* const smem_ptr,
                         int32_t const& crd0, int32_t const& crd1, int32_t const& crd2,
                         int32_t const& crd3) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);

  asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.bulk_group [%0, {%2, %3, %4, %5}], [%1];"
               :
               : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3)
               : "memory");
}

TL_DEVICE void tma_store(const CUtensorMap& descriptor, void const* const smem_ptr,
                         int32_t const& crd0, int32_t const& crd1, int32_t const& crd2,
                         int32_t const& crd3, int32_t const& crd4) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);

  asm volatile(
      "cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [%0, {%2, %3, %4, %5, %6}], [%1];"
      :
      : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4)
      : "memory");
}

TL_DEVICE void prefetch_tma_descriptor(const CUtensorMap& descriptor) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  asm volatile("prefetch.tensormap [%0];" : : "l"(gmem_int_desc) : "memory");
}

TL_DEVICE void mbarrier_init(uint64_t& smem_barrier, uint32_t arrive_count) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  asm volatile("mbarrier.init.shared.b64 [%1], %0;" : : "r"(arrive_count), "r"(smem_int_ptr));
}

TL_DEVICE void mbarrier_wait(uint64_t& smem_barrier, int phase_bit) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  asm volatile(
      "{\n"
      ".reg .pred                P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared.b64 P1, [%0], %1;\n"
      "@!P1                      bra.uni LAB_WAIT;\n"
      "}\n" ::"r"(smem_int_ptr),
      "r"(phase_bit));
}

TL_DEVICE void mbarrier_arrive(uint64_t& smem_barrier) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];" : : "r"(smem_int_ptr));
}

TL_DEVICE void mbarrier_expect_tx(uint64_t& smem_barrier, uint32_t transaction_bytes) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  asm volatile("mbarrier.expect_tx.shared.b64 [%1], %0;"
               :
               : "r"(transaction_bytes), "r"(smem_int_ptr));
}

TL_DEVICE void mbarrier_arrive_expect_tx(uint64_t& smem_barrier, uint32_t transaction_bytes) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%1], %0;"
               :
               : "r"(transaction_bytes), "r"(smem_int_ptr));
}

TL_DEVICE void mbarrier_cp_async_arrive(uint64_t& smem_barrier) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  asm volatile("cp.async.mbarrier.arrive.shared.b64 [%0];" : : "r"(smem_int_ptr));
}

TL_DEVICE void syncthreads_partial(uint64_t& smem_barrier) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  uint64_t state;
  asm volatile(
      "{\n"
      ".reg .pred                P1;\n"
      "mbarrier.arrive.shared.b64 %1, [%0];\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.shared.b64 P1, [%0], %1;\n"
      "@!P1                      bra.uni LAB_WAIT;\n"
      "}\n"
      :
      : "r"(smem_int_ptr), "l"(state));
}

}  // namespace tl