#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
#include <ck_tile/core.hpp>

using ck_tile::half_t;

#define HIPRT_INF_F __int_as_float(0x7f800000)
#define HIPRT_NEGINF_F __int_as_float(0xff800000)
#define HIPRT_NAN_F __int_as_float(0x7fffffff)
#define HIPRT_MIN_DENORM_F __int_as_float(0x00000001)
#define HIPRT_MAX_NORMAL_F __int_as_float(0x7f7fffff)
#define HIPRT_NEG_ZERO_F __int_as_float(0x80000000)
#define HIPRT_ZERO_F 0.0f
#define HIPRT_ONE_F 1.0f

/* double precision constants */
#define HIPRT_INF __hiloint2double(0x7ff00000, 0x00000000)
#define HIPRT_NAN __hiloint2double(0xfff80000, 0x00000000)

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

// Pack two half_t values.
TL_DEVICE unsigned __pack_half2(const half_t x, const half_t y) {
  unsigned v0 = *((unsigned short*)&x);
  unsigned v1 = *((unsigned short*)&y);
  return (v1 << 16) | v0;
}
