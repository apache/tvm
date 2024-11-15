#pragma once

#include "common.h"

using f32 = float;
// using f16 = _Float16;

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;

using index_t = u32;

using ck_tile::int32x4_t;

struct __attribute__((packed)) buffer_resource {
  const void* ptr;
  uint32_t range;
  uint32_t config;
};

CK_TILE_DEVICE int32x4_t make_wave_buffer_resource(const void* ptr, uint32_t size = 0xffffffff) {
  buffer_resource res{ptr, size, CK_TILE_BUFFER_RESOURCE_3RD_DWORD};
  int32x4_t r = __builtin_bit_cast(int32x4_t, res);
  r.x = __builtin_amdgcn_readfirstlane(r.x);
  r.y = __builtin_amdgcn_readfirstlane(r.y);
  r.z = __builtin_amdgcn_readfirstlane(r.z);
  r.w = __builtin_amdgcn_readfirstlane(r.w);
  return r;
}

__device__ void init_m0(uint32_t m0_value) {
  asm volatile("s_mov_b32 m0, %0" : : "s"(m0_value) : "memory");
}

__device__ void inc_m0(uint32_t m0_inc) {
  asm volatile("s_add_u32 m0, %0, m0" : : "n"(m0_inc) : "memory");
}

namespace tl {

// AMDGPU automatically commit memory fence
TL_DEVICE void cp_async_commit() {}

// Global Memory only fence
__device__ void async_gld_fence(index_t cnt) {
  asm volatile("s_waitcnt vmcnt(%0)" : : "n"(cnt) : "memory");
}

// Global Memory and Shared Memory fence
__device__ void async_gld_sld_fence(index_t cnt) {
  asm volatile("s_waitcnt lgkmcnt(%0)" : : "n"(cnt) : "memory");
}

__device__ void wave_barrier() { asm volatile("s_barrier" : : : "memory"); }

template <int N = 0>
TL_DEVICE void cp_async_wait() {
  async_gld_fence(N);
  // async_gld_sld_fence(N);
}

template <bool pre_nop = false>
CK_TILE_DEVICE void async_buffer_load_dword_v(void* smem, int32x4_t rsrc, index_t voffset) {
  auto const lds_ptr_sgpr = __builtin_amdgcn_readfirstlane((reinterpret_cast<uintptr_t>(smem)));
  asm volatile(
      "s_mov_b32 m0, %0; \n\t"
      "buffer_load_dword %1, %2, 0 offen lds;\n\t" ::"s"(lds_ptr_sgpr),
      "v"(voffset), "s"(rsrc)
      : "memory");
}

template <int N>
TL_DEVICE void cp_async_gs(void* lds_base_ptr, void* global_base_ptr) {
  if constexpr(N == 16) {
    *(uint4*)lds_base_ptr = *(uint4*)global_base_ptr;
  } else if constexpr(N == 8) {
    *(uint2*)lds_base_ptr = *(uint2*)global_base_ptr;
  } else if constexpr(N == 4) {
    async_buffer_load_dword_v(lds_base_ptr, make_wave_buffer_resource(((int32_t *)global_base_ptr) - threadIdx.x), threadIdx.x * N /*assume 4 bytes*/);
  }
}


template <int N>
TL_DEVICE void cp_async_gs_conditional(void const* const smem_addr, void* global_ptr, bool cond) {
  static_assert(false, "Not implemented");
}

}  // namespace tl
