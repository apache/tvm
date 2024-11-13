#pragma once

#include "common.h"

using f32 = float;
// using f16 = _Float16;

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;

using index_t = u32;

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x1 __attribute__((ext_vector_type(1)));

typedef f32 f32x8 __attribute__((ext_vector_type(8)));
typedef f32 f32x4 __attribute__((ext_vector_type(4)));
typedef f32 f32x2 __attribute__((ext_vector_type(2)));
typedef f32 f32x1 __attribute__((ext_vector_type(1)));

typedef u32x4 dwordx4_t;
typedef u32x2 dwordx2_t;
typedef u32 dword_t;

typedef uint8_t u8x16 __attribute__((ext_vector_type(16)));
typedef uint8_t u8x8 __attribute__((ext_vector_type(8)));
typedef uint8_t u8x4 __attribute__((ext_vector_type(4)));
typedef uint8_t u8x2 __attribute__((ext_vector_type(2)));
typedef uint8_t u8x1 __attribute__((ext_vector_type(1)));


namespace tl {

TL_DEVICE void cp_async_commit() {}

__device__ void async_gld_fence(index_t cnt) {
  asm volatile("s_waitcnt vmcnt(%0)" : : "n"(cnt) : "memory");
}

template <int N = 0>
TL_DEVICE void cp_async_wait() {
  async_gld_fence(0);
}

#define BUFFER_LOAD_DWORD3 0x00020000  // This is valid for
struct buffer_resource {
  const void* ptr;
  dword_t range;
  dword_t config;
};
__device__ dwordx4_t make_buffer_resource(const void* ptr) {
  buffer_resource res{ptr, 0xffffffff, BUFFER_LOAD_DWORD3};
  return __builtin_bit_cast(dwordx4_t, res);
}

__device__ void llvm_amdgcn_raw_buffer_load_lds(
    dwordx4_t rsrc, __attribute__((address_space(3))) uint32_t* lds_ptr, index_t size,
    index_t voffset, index_t soffset, index_t offset,
    index_t aux) __asm("llvm.amdgcn.raw.buffer.load.lds");

#define SPTR(_ptr_) \
  reinterpret_cast<__attribute__((address_space(3))) uint32_t*>(reinterpret_cast<uintptr_t>(_ptr_))

template <int N, bool pre_nop = false>
TL_DEVICE void cp_async_gs(void* lds_base_ptr, void* global_base_ptr) {
// *(uint4*)lds_base_ptr = *(uint4*)global_base_ptr;
constexpr auto force_buffer_size = N * sizeof(char);
llvm_amdgcn_raw_buffer_load_lds(make_buffer_resource(global_base_ptr),
                                SPTR(lds_base_ptr), sizeof(uint32_t), 0, 0, 0, 0);

//   // Direct loads require that each thread reads and writes exactly a single DWORD.
//   constexpr auto dword_bytes = 4;
//   constexpr auto src_element_space_size = 16;
//   const uint32_t* global_ptr =
//       reinterpret_cast<uint32_t*>(reinterpret_cast<uintptr_t>(global_base_ptr));
//   const int32x4_t src_resource =
//       ck_tile::make_wave_buffer_resource(global_ptr, src_element_space_size);
//   const index_t global_offset_bytes = 16;

//   // LDS pointer must be attributed with the LDS address space.
//   __attribute__((address_space(3))) uint32_t* lds_ptr =
//       reinterpret_cast<__attribute__((address_space(3))) uint32_t*>(
//           reinterpret_cast<uintptr_t>(lds_base_ptr));
//   llvm_amdgcn_raw_buffer_load_lds(src_resource, lds_ptr, sizeof(uint32_t), global_offset_bytes, 0,
//                                   0, 0);
//   cp_async_wait();
  if (threadIdx.x == 0) {
//     // print A
    // for (int i = 0; i < 16; i++) {
    //   printf("%f ", ((float)((half*)global_base_ptr)[i]));
    // }
        printf("\n");
// print A_shared
        for (int i = 0; i < 16; i++) {
        printf("%f ", (float)(((half*)lds_base_ptr)[i]));
        }
        printf("\n");
  }
}

template <int N>
TL_DEVICE void cp_async_gs_conditional(void const* const smem_addr, void* global_ptr, bool cond) {
  static_assert(false, "Not implemented");
}

}  // namespace tl
