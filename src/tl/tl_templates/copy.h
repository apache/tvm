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

template <int N>
__forceinline__ __device__
void cp_async_gs(void const* const smem_addr, void* global_ptr)
{
  uint32_t addr;
  __asm__ __volatile__(
    "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
    : "=r"(addr)
    : "l"((void*)(smem_addr))
  );
  __asm__ __volatile__(
#if TL_ENABLE_L2_PREFETCH
    "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
    "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
    :: "r"(addr), "l"((void*)(global_ptr)), "n"(N)
  );
}

template <int N>
__forceinline__ __device__
void cp_async_gs_conditional(void const* const smem_addr, void* global_ptr, bool cond)
{
  uint32_t addr;
  __asm__ __volatile__(
    "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
    : "=r"(addr)
    : "l"((void*)(smem_addr))
  );
  int bytes = cond ? N : 0;
  __asm__ __volatile__(
#if TL_ENABLE_L2_PREFETCH
    "cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;"
#else
    "cp.async.cg.shared.global [%0], [%1], %2, %3;"
#endif
    :: "r"(addr), "l"((void*)(global_ptr)), "n"(N), "r"(bytes)
  );

}

} // namespace tl
