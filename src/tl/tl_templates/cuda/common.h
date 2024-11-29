#pragma once

#include <cuda_runtime.h>
#include <cutlass/fast_math.h>
#include <cutlass/numeric_types.h>
#include <math_constants.h>

using cutlass::bfloat16_t;
using cutlass::half_t;
using cutlass::tfloat32_t;

#define hexp cutlass::fast_exp
#define hlog cutlass::fast_log
#define hsqrt cutlass::fast_sqrt
#define htanh cutlass::fast_tanh
#define hpow powf

#define uint unsigned int
#define uchar unsigned char
#define ushort unsigned short

#define TL_DEVICE __forceinline__ __device__

// Pack two half values.
TL_DEVICE unsigned __pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short*)&x);
  unsigned v1 = *((unsigned short*)&y);
  return (v1 << 16) | v0;
}

// Pack two half_t values.
TL_DEVICE unsigned __pack_half2(const half_t x, const half_t y) {
  unsigned v0 = *((unsigned short*)&x);
  unsigned v1 = *((unsigned short*)&y);
  return (v1 << 16) | v0;
}

// Pack two bfloat16_t values.
TL_DEVICE unsigned __pack_half2(const bfloat16_t x, const bfloat16_t y) {
  unsigned v0 = *((unsigned short*)&x);
  unsigned v1 = *((unsigned short*)&y);
  return (v1 << 16) | v0;
}

/// Helper to cast SMEM pointer to unsigned
TL_DEVICE uint32_t smem_ptr_to_uint(void const* const ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// AtomicAdd Functions for FP16
TL_DEVICE void atomicAdd(half_t* address, half_t val) {
  // Use atomicCAS with built-in cuda_fp16 support
  atomicAdd(reinterpret_cast<half*>(address), static_cast<half>(val));
}

// AtomicAdd Functions for FP16
TL_DEVICE void atomicAdd(half_t* address, half_t* val) {
  atomicAdd(reinterpret_cast<half*>(address), static_cast<half>(*val));
}

// AtomicAdd Functions for FP16
TL_DEVICE void atomicAddx2(half_t* address, half_t* val) {
  atomicAdd(reinterpret_cast<half2*>(address), static_cast<half2>(*reinterpret_cast<half2*>(val)));
}

TL_DEVICE void atomicAdd(half_t* address, float val) {
  // Use atomicCAS with built-in cuda_fp16 support
  atomicAdd(reinterpret_cast<half*>(address), __float2half(val));
}

// DP4A
template<typename InDatatype, typename OutDatatype>
TL_DEVICE void DP4A(InDatatype* a, InDatatype* b, OutDatatype* c) {
  const int a_int = *((int*)a);
  const int b_int = *((int*)b);
  const int c_int = *((int*)c);
  *c = __dp4a(a_int, b_int, c_int);
}
