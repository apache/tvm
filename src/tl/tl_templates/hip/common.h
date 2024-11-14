#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
#include <ck_tile/core.hpp>

using ck_tile::half_t;


template <int BlockSize>
struct GemmMPerBlock {
    static constexpr int value = BlockSize;
};

template <int BlockSize>
struct GemmNPerBlock {
    static constexpr int value = BlockSize;
};

template <int BlockSize>
struct GemmKPerBlock {
    static constexpr int value = BlockSize;
};

template <int BlockSize>
struct GemmMPerWave {
    static constexpr int value = BlockSize;
};

template <int BlockSize>
struct GemmNPerWave {
    static constexpr int value = BlockSize;
};

template <int BlockSize>
struct GemmKPerWave {
    static constexpr int value = BlockSize;
};

#define uint unsigned int
#define uchar unsigned char
#define ushort unsigned short

#define TL_DEVICE __forceinline__ __device__

#define half _Float16
#define __float2half_rn(x) half(x)

#define hpow __ocml_pown_f16
#define hsqrt __ocml_sqrt_f16

using float16_t = _Float16;

using float16x2 = __attribute__((__vector_size__(2 * sizeof(float16_t)))) float16_t;
using float16x4 = __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
using float16x8 = __attribute__((__vector_size__(8 * sizeof(float16_t)))) float16_t;
using float16x16 = __attribute__((__vector_size__(16 * sizeof(float16_t)))) float16_t;

using int32x4  = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using float32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using float32x16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;

using int8x4 = __attribute__((__vector_size__(4 * sizeof(int8_t)))) int8_t;

#define max(a, b) (((a) > (b)) ? (a) : (b))
#define min(a, b) (((a) < (b)) ? (a) : (b))

// Pack two half_t values.
TL_DEVICE unsigned __pack_half2(const half_t x, const half_t y) {
  unsigned v0 = *((unsigned short*)&x);
  unsigned v1 = *((unsigned short*)&y);
  return (v1 << 16) | v0;
}
