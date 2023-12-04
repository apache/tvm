#pragma once

#include "common.h"

namespace tl {

__forceinline__ __device__
void cp_async_commit()
{
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__forceinline__ __device__
void cp_async_wait()
{
  if constexpr (N == 0) {
    asm volatile("cp.async.wait_all;\n" ::);
  } else {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
  }
}

__forceinline__ __device__ unsigned int
cast_smem_ptr_to_int(const void* const smem_ptr)
{
  unsigned int smem_int;
  asm volatile ("{ .reg .u64 smem_int; cvta.to.shared.u64 smem_int, %1; cvt.u32.u64 %0, smem_int; }"
    : "=r"(smem_int) : "l"(smem_ptr));
  return smem_int;
}

template <int N>
__forceinline__ __device__
void cp_async_gs(void const* const smem_addr, void* global_ptr)
{
  static_assert(N == 16 || N == 8 || N == 4);
  unsigned int addr = cast_smem_ptr_to_int(smem_addr);
  if constexpr (N == 16) {
    __asm__ __volatile__(
#if TL_ENABLE_L2_PREFETCH
      "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
      "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
      :: "r"(addr), "l"((void*)(global_ptr)), "n"(N)
    );
  } else {
    __asm__ __volatile__(
#if TL_ENABLE_L2_PREFETCH
      "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;"
#else
      "cp.async.ca.shared.global [%0], [%1], %2;"
#endif
      :: "r"(addr), "l"((void*)(global_ptr)), "n"(N)
    );
  }
}

template <int N>
__forceinline__ __device__
void cp_async_gs_conditional(void const* const smem_addr, void* global_ptr, bool cond)
{
  static_assert(N == 16 || N == 8 || N == 4);
  int bytes = cond ? N : 0;
  unsigned int addr = cast_smem_ptr_to_int(smem_addr);
  if constexpr (N == 16) {
    __asm__ __volatile__(
#if TL_ENABLE_L2_PREFETCH
      "cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;"
#else
      "cp.async.cg.shared.global [%0], [%1], %2, %3;"
#endif
      :: "r"(addr), "l"((void*)(global_ptr)), "n"(N), "r"(bytes)
    );
  } else {
    __asm__ __volatile__(
#if TL_ENABLE_L2_PREFETCH
      "cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;"
#else
      "cp.async.ca.shared.global [%0], [%1], %2, %3;"
#endif
      :: "r"(addr), "l"((void*)(global_ptr)), "n"(N), "r"(bytes)
    );
  }
}

} // namespace tl
