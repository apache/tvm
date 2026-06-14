# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=line-too-long
"""CUDA header generator for codegen.

The header generator is used to generate the header for the CUDA code.
It's controlled by the predefined tags.
The tags are used to identify the utility functions/classes necessary for the codegen.
"""

import tvm_ffi

TAGS = {
    "cuda",
    "cuda/barrier",
    "cooperative_groups",
    "fp16",
    "bf16",
    "fp8",
    "fp6",
    "fp4",
    "int8",
    "math_constants",
    "mma",
    "warp_shuffle",
    "cast_smem_ptr_to_int",
    "get_tmem_addr",
    "gmma_descriptor",
    "smem_descriptor",
    "instr_descriptor",
    "instr_descriptor_block_scaled",
    "get_time_stamp",
    "nvshmem",
    "elect_one_sync",
}


@tvm_ffi.register_global_func("tirx.intrinsics.cuda.header_generator")
def header_generator(tags):
    """Generate the header for the CUDA code."""
    for tag in tags:
        if tag not in TAGS:
            raise ValueError(f"Invalid tag: {tag}")

    header = ""
    if "nvshmem" in tags:
        header += R"""
#include <nvshmem.h>
#include <nvshmemx.h>
"""

    if "cuda/barrier" in tags or "cooperative_groups" in tags:
        header += (
            R"""
#include <cuda/barrier>
#include <cooperative_groups.h>
"""
            + "\n"
        )

    # NVRTC has no host C++ stdlib and no <cuda.h>. Branch on __CUDACC_RTC__ so
    # the same emitted source compiles under both nvcc (offline) and NVRTC
    # (runtime) without any post-processing in tvm.support.nvcc.
    header += """
#ifdef __CUDACC_RTC__
  #include <cuda/std/cstdint>
  using cuda::std::uint8_t;
  using cuda::std::uint16_t;
  using cuda::std::uint32_t;
  using cuda::std::uint64_t;
  using cuda::std::int8_t;
  using cuda::std::int16_t;
  using cuda::std::int32_t;
  using cuda::std::int64_t;

  #include <cuda/std/type_traits>
  namespace std {
    using cuda::std::is_same;
    using cuda::std::is_same_v;
    using cuda::std::is_integral;
    using cuda::std::is_signed;
    using cuda::std::is_unsigned;
    using cuda::std::is_floating_point;
    using cuda::std::enable_if;
    using cuda::std::conditional;
  }

  // NVRTC uses asm/volatile instead of __asm__/__volatile__ (gcc extension).
  #ifndef __asm__
  #define __asm__ asm
  #endif
  #ifndef __volatile__
  #define __volatile__ volatile
  #endif
#else
  #include <cstdint>
  #include <type_traits>
  #include <cuda.h>
#endif
"""

    if "fp16" in tags:
        header += R"""
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
#include <cuda_fp16.h>
__device__ half max(half a, half b)
{
  return __hgt(__half(a), __half(b)) ? a : b;
}
__device__ half min(half a, half b)
{
  return __hlt(__half(a), __half(b)) ? a : b;
}
#endif // __CUDA_ARCH__ >= 530

// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

#define CUDA_UNSUPPORTED_HALF_MATH_BINARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x, half y) {   \
  float tmp_x = __half2float(x);                                          \
  float tmp_y = __half2float(y);                                          \
  float result = FP32_MATH_NAME(tmp_x, tmp_y);                            \
  return __float2half(result);                                            \
}

#define CUDA_UNSUPPORTED_HALF_MATH_UNARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x) {          \
  float tmp_x = __half2float(x);                                         \
  float result = FP32_MATH_NAME(tmp_x);                                  \
  return __float2half(result);                                           \
}

// Some fp16 math functions are not supported in cuda_fp16.h,
// so we define them here to make sure the generated CUDA code
// is valid.
#if defined(__CUDA_ARCH__)
#if (__CUDA_ARCH__ >= 530)
CUDA_UNSUPPORTED_HALF_MATH_BINARY(hpow, powf)
#if ((__CUDACC_VER_MAJOR__ < 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ < 8)))
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htanh, tanhf)
#endif
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htan, tanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hatan, atanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(herf, erf)
#else
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hexp, exp)
#endif
#endif

#undef CUDA_UNSUPPORTED_HALF_MATH_BINARY
#undef CUDA_UNSUPPORTED_HALF_MATH_UNARY
"""

    if "bf16" in tags:
        header += R"""
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#include <cuda_bf16.h>
__device__ nv_bfloat16 max(nv_bfloat16 a, nv_bfloat16 b)
{
  return __hgt(a, b) ? a : b;
}
__device__ nv_bfloat16 min(nv_bfloat16 a, nv_bfloat16 b)
{
  return __hlt(a, b) ? a : b;
}
#endif // __CUDA_ARCH__ >= 800
// Pack two bfloat16 values.
static inline __device__ __host__ unsigned
__pack_nv_bfloat162(const nv_bfloat16 x, const nv_bfloat16 y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Some bfp16 math functions are not supported in cuda_bfp16.h,
// so we define them here to make sure the generated CUDA code
// is valid.
#define CUDA_UNSUPPORTED_HALF_MATH_BINARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ nv_bfloat16 HALF_MATH_NAME(nv_bfloat16 x, nv_bfloat16 y) {   \
  float tmp_x = __bfloat162float(x);                                      \
  float tmp_y = __bfloat162float(y);                                      \
  float result = FP32_MATH_NAME(tmp_x, tmp_y);                            \
  return __float2bfloat16(result);                                        \
}

#define CUDA_UNSUPPORTED_HALF_MATH_UNARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ nv_bfloat16 HALF_MATH_NAME(nv_bfloat16 x) {          \
  float tmp_x = __bfloat162float(x);                                     \
  float result = FP32_MATH_NAME(tmp_x);                                  \
  return __float2bfloat16(result);                                       \
}

CUDA_UNSUPPORTED_HALF_MATH_BINARY(hpow, powf)
#if ((__CUDACC_VER_MAJOR__ < 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ < 8)))
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htanh, tanhf)
#endif
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htan, tanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hatan, atanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(herf, erf)

#undef CUDA_UNSUPPORTED_HALF_MATH_BINARY
#undef CUDA_UNSUPPORTED_HALF_MATH_UNARY
"""

    if "fp8" in tags:
        header += R"""
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
#include <cuda_fp8.h>
using fp8_e4_t = __nv_fp8_e4m3;
using fp8_e4x2_t = __nv_fp8x2_e4m3;
using fp8_e4x4_t = __nv_fp8x4_e4m3;
struct fp8_e4x8_t {
 fp8_e4_t data[8];
};
struct fp8_e4x16_t {
 fp8_e4_t data[16];
};
using fp8_e5_t = __nv_fp8_e5m2;
using fp8_e5x2_t = __nv_fp8x2_e5m2;
using fp8_e5x4_t = __nv_fp8x4_e5m2;
struct fp8_e5x8_t {
 fp8_e5_t data[8];
};
struct fp8_e5x16_t {
 fp8_e5_t data[16];
};
using fp8_e8_t = __nv_fp8_e8m0;
using fp8_e8x2_t = __nv_fp8x2_e8m0;
using fp8_e8x4_t = __nv_fp8x4_e8m0;
struct fp8_e8x8_t {
 fp8_e8_t data[8];
};
struct fp8_e8x16_t {
 fp8_e8_t data[16];
};
#endif // __CUDA_ARCH__ >= 890
"""

    if "fp6" in tags:
        header += R"""
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
#include <cuda_fp6.h>
using fp6_e2_t = __nv_fp6_e2m3;
using fp6_e2x2_t = __nv_fp6x2_e2m3;
using fp6_e2x4_t = __nv_fp6x4_e2m3;
struct fp6_e2x8_t {
 fp6_e2_t data[8];
};
struct fp6_e2x16_t {
 fp6_e2_t data[16];
};
using fp6_e3_t = __nv_fp6_e3m2;
using fp6_e3x2_t = __nv_fp6x2_e3m2;
using fp6_e3x4_t = __nv_fp6x4_e3m2;
struct fp6_e3x8_t {
 fp6_e3_t data[8];
};
struct fp6_e3x16_t {
 fp6_e3_t data[16];
};
#endif // __CUDA_ARCH__ >= 1000
"""

    if "fp4" in tags:
        header += R"""
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#include <cuda_fp4.h>
using fp4_e2_t = __nv_fp4_e2m1;
using fp4_e2x2_t = __nv_fp4x2_e2m1;
using fp4_e2x4_t = __nv_fp4x4_e2m1;
struct fp4_e2x8_t {
 fp4_e2_t data[8];
};
struct fp4_e2x16_t {
 fp4_e2_t data[16];
};
#endif // __CUDA_ARCH__ >= 800
"""

    #########################################################
    # Vector type extensions
    #########################################################
    if "fp16" in tags or "bf16" in tags:
        header += R"""
template <typename T, typename TVec2>
struct __align__(8) half4_bfloat164 {
  T x, y, z, w;
  __host__ __device__ half4_bfloat164() : x(T(0)), y(T(0)), z(T(0)), w(T(0)) {}
  __host__ __device__ half4_bfloat164(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {}
"""
        if "fp8" in tags:
            header += R"""
  __host__ __device__ explicit half4_bfloat164(const __nv_fp8x4_e4m3& fp8x4) {
    if constexpr (std::is_same_v<T, __half>) {
      __nv_fp8x2_e4m3 lo_part, hi_part;
      lo_part.__x = static_cast<__nv_fp8x2_storage_t>(fp8x4.__x & 0xFFFF);
      hi_part.__x = static_cast<__nv_fp8x2_storage_t>((fp8x4.__x >> 16) & 0xFFFF);
      TVec2 lo_half2 = static_cast<TVec2>(lo_part);
      TVec2 hi_half2 = static_cast<TVec2>(hi_part);
      x = reinterpret_cast<T*>(&lo_half2)[0];
      y = reinterpret_cast<T*>(&lo_half2)[1];
      z = reinterpret_cast<T*>(&hi_half2)[0];
      w = reinterpret_cast<T*>(&hi_half2)[1];
    } else {
      __nv_fp8_storage_t elem0_raw = static_cast<__nv_fp8_storage_t>(fp8x4.__x & 0xFF);
      __nv_fp8_storage_t elem1_raw = static_cast<__nv_fp8_storage_t>((fp8x4.__x >> 8) & 0xFF);
      __nv_fp8_storage_t elem2_raw = static_cast<__nv_fp8_storage_t>((fp8x4.__x >> 16) & 0xFF);
      __nv_fp8_storage_t elem3_raw = static_cast<__nv_fp8_storage_t>((fp8x4.__x >> 24) & 0xFF);
      __nv_fp8_e4m3 elem0, elem1, elem2, elem3;
      elem0.__x = elem0_raw;
      elem1.__x = elem1_raw;
      elem2.__x = elem2_raw;
      elem3.__x = elem3_raw;
      x = T(elem0);
      y = T(elem1);
      z = T(elem2);
      w = T(elem3);
    }
  }
  __host__ __device__ explicit operator __nv_fp8x4_e4m3() const {
    __nv_fp8x4_e4m3 result;
    TVec2 lo_half2 = *reinterpret_cast<const TVec2*>(&x);
    TVec2 hi_half2 = *reinterpret_cast<const TVec2*>(&z);
    __nv_fp8x2_e4m3 lo_part(lo_half2), hi_part(hi_half2);
    result.__x =
        (static_cast<__uint32_t>(lo_part.__x) | (static_cast<__uint32_t>(hi_part.__x) << 16));
    return result;
  }
  __host__ __device__ explicit half4_bfloat164(const __nv_fp8x4_e5m2& fp8x4) {
      __nv_fp8x2_e5m2 lo_part, hi_part;
      lo_part.__x = static_cast<__nv_fp8x2_storage_t>(fp8x4.__x & 0xFFFF);
      hi_part.__x = static_cast<__nv_fp8x2_storage_t>((fp8x4.__x >> 16) & 0xFFFF);
      TVec2 lo_half2 = static_cast<TVec2>(lo_part);
      TVec2 hi_half2 = static_cast<TVec2>(hi_part);
      x = reinterpret_cast<T*>(&lo_half2)[0];
      y = reinterpret_cast<T*>(&lo_half2)[1];
      z = reinterpret_cast<T*>(&hi_half2)[0];
      w = reinterpret_cast<T*>(&hi_half2)[1];
  }
  __host__ __device__ explicit operator __nv_fp8x4_e5m2() const {
    __nv_fp8x4_e5m2 result;
    TVec2 lo_half2 = *reinterpret_cast<const TVec2*>(&x);
    TVec2 hi_half2 = *reinterpret_cast<const TVec2*>(&z);
    __nv_fp8x2_e5m2 lo_part(lo_half2), hi_part(hi_half2);
    result.__x =
        (static_cast<__uint32_t>(lo_part.__x) | (static_cast<__uint32_t>(hi_part.__x) << 16));
    return result;
  }
  __host__ __device__ explicit half4_bfloat164(const __nv_fp8x4_e8m0& fp8x4) {
      __nv_fp8x2_e8m0 lo_part, hi_part;
      lo_part.__x = static_cast<__nv_fp8x2_storage_t>(fp8x4.__x & 0xFFFF);
      hi_part.__x = static_cast<__nv_fp8x2_storage_t>((fp8x4.__x >> 16) & 0xFFFF);
      TVec2 lo_half2 = static_cast<TVec2>(lo_part);
      TVec2 hi_half2 = static_cast<TVec2>(hi_part);
      x = reinterpret_cast<T*>(&lo_half2)[0];
      y = reinterpret_cast<T*>(&lo_half2)[1];
      z = reinterpret_cast<T*>(&hi_half2)[0];
      w = reinterpret_cast<T*>(&hi_half2)[1];
  }
  __host__ __device__ explicit operator __nv_fp8x4_e8m0() const {
    __nv_fp8x4_e8m0 result;
    TVec2 lo_half2 = *reinterpret_cast<const TVec2*>(&x);
    TVec2 hi_half2 = *reinterpret_cast<const TVec2*>(&z);
    __nv_fp8x2_e8m0 lo_part(lo_half2), hi_part(hi_half2);
    result.__x =
        (static_cast<__uint32_t>(lo_part.__x) | (static_cast<__uint32_t>(hi_part.__x) << 16));
    return result;
  }
"""
        if "fp4" in tags:
            header += R"""
  __host__ __device__ explicit half4_bfloat164(const __nv_fp4x4_e2m1& fp4x4) {
    if constexpr (std::is_same_v<T, __half>) {
      __nv_fp4x2_storage_t lo_part = static_cast<__nv_fp4x2_storage_t>(fp4x4.__x & 0xFF);
      __nv_fp4x2_storage_t hi_part = static_cast<__nv_fp4x2_storage_t>((fp4x4.__x >> 8) & 0xFF);
      TVec2 lo_half2 = __half2(__nv_cvt_fp4x2_to_halfraw2(lo_part, __NV_E2M1));
      TVec2 hi_half2 = __half2(__nv_cvt_fp4x2_to_halfraw2(hi_part, __NV_E2M1));
      x = reinterpret_cast<T*>(&lo_half2)[0];
      y = reinterpret_cast<T*>(&lo_half2)[1];
      z = reinterpret_cast<T*>(&hi_half2)[0];
      w = reinterpret_cast<T*>(&hi_half2)[1];
    } else {
      __nv_fp4_e2m1 elem0, elem1, elem2, elem3;
      elem0.__x = static_cast<__nv_fp4_storage_t>(fp4x4.__x & 0xF);
      elem1.__x = static_cast<__nv_fp4_storage_t>((fp4x4.__x >> 4) & 0xF);
      elem2.__x = static_cast<__nv_fp4_storage_t>((fp4x4.__x >> 8) & 0xF);
      elem3.__x = static_cast<__nv_fp4_storage_t>((fp4x4.__x >> 12) & 0xF);
      x = T(elem0);
      y = T(elem1);
      z = T(elem2);
      w = T(elem3);
    }
  }
  __host__ __device__ explicit operator __nv_fp4x4_e2m1() const {
    TVec2 lo_half2 = *reinterpret_cast<const TVec2*>(&x);
    TVec2 hi_half2 = *reinterpret_cast<const TVec2*>(&z);
    return __nv_fp4x4_e2m1(lo_half2, hi_half2);
  }
"""
        header += R"""
};
"""
    if "fp16" in tags:
        header += R"""
using half4 = half4_bfloat164<__half, __half2>;
__host__ __device__ half4 make_half4(__half x, __half y, __half z, __half w) {
    return half4(x, y, z, w);
}
"""
    if "bf16" in tags:
        header += R"""
using nv_bfloat164 = half4_bfloat164<nv_bfloat16, nv_bfloat162>;
__host__ __device__ nv_bfloat164 make_nv_bfloat164(nv_bfloat16 x, nv_bfloat16 y, nv_bfloat16 z, nv_bfloat16 w) {
    return nv_bfloat164(x, y, z, w);
}
__host__ __device__ nv_bfloat162 make_nv_bfloat162(nv_bfloat16 x, nv_bfloat16 y) {
    return nv_bfloat162(x, y);
}
"""  # noqa: E501
        if "fp8" in tags:
            header += R"""
__host__ __device__ nv_bfloat162 cast_to_nv_bfloat162(const __nv_fp8x2_e4m3& fp8x2) {
    __nv_fp8_e4m3 elem0, elem1;
    elem0.__x = static_cast<__nv_fp8_storage_t>(fp8x2.__x & 0xFF);
    elem1.__x = static_cast<__nv_fp8_storage_t>((fp8x2.__x >> 8) & 0xFF);
    nv_bfloat16 x = nv_bfloat16(elem0);
    nv_bfloat16 y = nv_bfloat16(elem1);
    return nv_bfloat162(x, y);
}
__host__ __device__ nv_bfloat162 cast_to_nv_bfloat162(const __nv_fp8x2_e5m2& fp8x2) {
    __nv_fp8_e5m2 elem0, elem1;
    elem0.__x = static_cast<__nv_fp8_storage_t>(fp8x2.__x & 0xFF);
    elem1.__x = static_cast<__nv_fp8_storage_t>((fp8x2.__x >> 8) & 0xFF);
    nv_bfloat16 x = nv_bfloat16(elem0);
    nv_bfloat16 y = nv_bfloat16(elem1);
    return nv_bfloat162(x, y);
}
__host__ __device__ nv_bfloat162 cast_to_nv_bfloat162(const __nv_fp8x2_e8m0& fp8x2) {
    __nv_fp8_e8m0 elem0, elem1;
    elem0.__x = static_cast<__nv_fp8_storage_t>(fp8x2.__x & 0xFF);
    elem1.__x = static_cast<__nv_fp8_storage_t>((fp8x2.__x >> 8) & 0xFF);
    nv_bfloat16 x = nv_bfloat16(elem0);
    nv_bfloat16 y = nv_bfloat16(elem1);
    return nv_bfloat162(x, y);
}
    """
    if "fp8" in tags:
        header += R"""
__device__ __nv_fp8x2_e5m2 make___nv_fp8x2_e5m2(__nv_fp8_e5m2 x, __nv_fp8_e5m2 y) {
    __nv_fp8x2_e5m2 result;
    result.__x = (x.__x) | (y.__x << 8);
    return result;
}
__device__ __nv_fp8x4_e5m2 make___nv_fp8x4_e5m2(__nv_fp8_e5m2 a, __nv_fp8_e5m2 b, __nv_fp8_e5m2 c, __nv_fp8_e5m2 d) {
    __nv_fp8x4_e5m2 result;
    result.__x = (a.__x) | (b.__x << 8) | (c.__x << 16) | (d.__x << 24);
    return result;
}
__device__ __nv_fp8x2_e4m3 make___nv_fp8x2_e4m3(__nv_fp8_e4m3 x, __nv_fp8_e4m3 y) {
    __nv_fp8x2_e4m3 result;
    result.__x = (x.__x) | (y.__x << 8);
    return result;
}
__device__ __nv_fp8x4_e4m3 make___nv_fp8x4_e4m3(__nv_fp8_e4m3 a, __nv_fp8_e4m3 b, __nv_fp8_e4m3 c, __nv_fp8_e4m3 d) {
    __nv_fp8x4_e4m3 result;
    result.__x = (a.__x) | (b.__x << 8) | (c.__x << 16) | (d.__x << 24);
    return result;
}
__device__ __nv_fp8x2_e8m0 make___nv_fp8x2_e8m0(__nv_fp8_e8m0 x, __nv_fp8_e8m0 y) {
    __nv_fp8x2_e8m0 result;
    result.__x = (x.__x) | (y.__x << 8);
    return result;
}
__device__ __nv_fp8x4_e8m0 make___nv_fp8x4_e8m0(__nv_fp8_e8m0 a, __nv_fp8_e8m0 b, __nv_fp8_e8m0 c, __nv_fp8_e8m0 d) {
    __nv_fp8x4_e8m0 result;
    result.__x = (a.__x) | (b.__x << 8) | (c.__x << 16) | (d.__x << 24);
    return result;
}
"""  # noqa: E501
    if "fp4" in tags:
        header += R"""
__host__ __device__ nv_bfloat162 cast_to_nv_bfloat162(const __nv_fp4x2_e2m1& fp4x2) {
    __nv_fp4_e2m1 elem0, elem1;
    elem0.__x = static_cast<__nv_fp4_storage_t>(fp4x2.__x & 0xFF);
    elem1.__x = static_cast<__nv_fp4_storage_t>((fp4x2.__x >> 8) & 0xFF);
    nv_bfloat16 x = nv_bfloat16(elem0);
    nv_bfloat16 y = nv_bfloat16(elem1);
    return nv_bfloat162(x, y);
}
"""

    if "int8" in tags:
        header += R"""
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
#include <sm_61_intrinsics.h>

#if defined(__CUDACC_RTC__)
#define __SM_61_INTRINSICS_DECL__ __device__
#else /* !__CUDACC_RTC__ */
#define __SM_61_INTRINSICS_DECL__ static __device__ __inline__
#endif /* __CUDACC_RTC__ */

#ifndef __CUDA_ARCH__
#define __DEF_IF_HOST { }
#else  /* !__CUDA_ARCH__ */
#define __DEF_IF_HOST ;
#endif /* __CUDA_ARCH__ */

__SM_61_INTRINSICS_DECL__ int __dp4a(unsigned int srcA, int srcB, int c) __DEF_IF_HOST
__SM_61_INTRINSICS_DECL__ int __dp4a(int srcA, unsigned int srcB, int c) __DEF_IF_HOST

#undef __DEF_IF_HOST

#if !defined(__CUDACC_RTC__) && defined(__CUDA_ARCH__)
__SM_61_INTRINSICS_DECL__ int __dp4a(unsigned int srcA, int srcB, int c) {
    int ret;
    asm volatile ("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(ret) : "r"(srcA), "r"(srcB), "r"(c));
    return ret;
}

__SM_61_INTRINSICS_DECL__ int __dp4a(int srcA, unsigned int srcB, int c) {
    int ret;
    asm volatile ("dp4a.s32.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(srcA), "r"(srcB), "r"(c));
    return ret;
}
#endif /* !__CUDACC_RTC__ && defined(__CUDA_ARCH__) */

#undef __SM_61_INTRINSICS_DECL__

#endif // __CUDA_ARCH__ >= 610
"""
    if "math_constants" in tags:
        header += R"""
#include <math_constants.h>
"""
    if "mma" in tags:
        header += R"""
#include <mma.h>
"""

    if "warp_shuffle" in tags:
        header += R"""
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#define __shfl_sync(mask, var, lane, width) \
        __shfl((var), (lane), (width))

#define __shfl_down_sync(mask, var, offset, width) \
        __shfl_down((var), (offset), (width))

#define __shfl_up_sync(mask, var, offset, width) \
        __shfl_up((var), (offset), (width))
#endif
"""

    if "cast_smem_ptr_to_int" in tags:
        header += R"""
__forceinline__ __device__ unsigned int cast_smem_ptr_to_int(const void* const smem_ptr) {
  unsigned int smem_int;
  asm volatile ("{ .reg .u64 smem_int; cvta.to.shared.u64 smem_int, %1; cvt.u32.u64 %0, smem_int; }"
    : "=r"(smem_int) : "l"(smem_ptr));
  return smem_int;
}
"""
    header += R"""
#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
#endif
"""

    if "get_tmem_addr" in tags:
        header += R"""
__forceinline__ __device__ uint32_t get_tmem_addr(uint32_t idx, int row_offset, int col_offset) {
  int col_idx = idx & 0xFFFF;
  int row_idx = (idx >> 16) & 0xFFFF;
  col_idx += col_offset;
  row_idx += row_offset;
  col_idx = col_idx & 0xFFFF;
  row_idx = row_idx & 0xFFFF;

  uint32_t new_idx = (row_idx << 16) | col_idx;
  return new_idx;
}
"""

    if "get_time_stamp" in tags:
        header += R"""
__forceinline__ __device__ uint32_t tvm_builtin_get_timestamp() {
  volatile uint32_t ret;
  asm volatile("mov.u32 %0, %globaltimer_lo;" : "=r"(ret));
  return ret;
}
"""

    if "gmma_descriptor" in tags:
        header += R"""
#ifndef HOST_DEVICE
#define HOST_DEVICE __forceinline__ __host__ __device__
#endif
union GmmaDescriptor
{
  HOST_DEVICE constexpr
  GmmaDescriptor() noexcept : desc_(0) {}
  HOST_DEVICE constexpr
  GmmaDescriptor(uint64_t desc) noexcept : desc_(desc) {}
  HOST_DEVICE constexpr
  GmmaDescriptor(GmmaDescriptor const& t) noexcept : desc_(t.desc_) {}
  HOST_DEVICE constexpr
  GmmaDescriptor(GmmaDescriptor && t) noexcept : desc_(t.desc_) {}

  HOST_DEVICE constexpr
  GmmaDescriptor& operator=(GmmaDescriptor const& t) noexcept {
    desc_ = t.desc_;
    return *this;
  }

  HOST_DEVICE constexpr
  GmmaDescriptor& operator=(GmmaDescriptor && t) noexcept {
    desc_ = t.desc_;
    return *this;
  }

  uint64_t desc_;
  uint32_t reg32_[2];
  uint16_t reg16_[4];

  // Bitfield implementation avoids the need for shifts in assignment
  struct {
    // start_address, bit [0,14), 4LSB not included
    uint16_t start_address_ : 14, : 2;        // 14 bits [0,14), 2 bits unused
    // leading dimension byte offset, bit [16,30), 4LSB not included
    // For N: This is the stride from the first col to the second col of the 8x2 brick in INTERLEAVED
    //   Unused for all SWIZZLE_* layouts (and assumed to be 1)
    // For T: This is the stride from the first 8 rows to the next 8 rows.
    uint16_t leading_byte_offset_ : 14, : 2;  // 14 bits [0,14), 2 bits unused
    // stride dimension byte offset, bit [32,46), 4LSB not included
    // For N: This is the stride from the first 8 rows to the next 8 rows.
    // For T: This is the stride fro mthe first 8 cols to the next 8 cols.
    uint16_t stride_byte_offset_ : 14, : 2;   // 14 bits [0,14), 2 bits unused
    // base_offset, bit [49,52)
    // Valid only for SWIZZLE_128B and SWIZZLE_64B
    uint8_t : 1, base_offset_ : 3, : 4;       // 1 bit unused, 3 bits [1,4), 4 bits unused
    // layout type, bit [62,64)
    // SWIZZLE_NONE = 0, SWIZZLE_32B = 3, SWIZZLE_64B = 2, SWIZZLE_128B = 1
    uint8_t : 6, layout_type_ : 2;            // 6 bits unused, 2 bits [6,8)
  } bitfield;

  // Decay to a uint64_t
  HOST_DEVICE constexpr
  operator uint64_t() const noexcept { return desc_; }
};
"""  # noqa: E501

    if "smem_descriptor" in tags:
        header += R"""
#ifndef HOST_DEVICE
#define HOST_DEVICE __forceinline__ __host__ __device__
#endif
union SmemDescriptor
{
  uint64_t desc_ = 0;
  // Bitfield implementation avoids the need for shifts in assignment
  struct {
    // start_address, bit [0,14), 4LSB not included
    uint16_t start_address_ : 14, : 2;                     // 14 bits [0,14), 2 bits unused
    // leading dimension byte offset, bit [16,30), 4LSB not included
    uint16_t leading_byte_offset_ : 14, : 2;               // 14 bits [0,14), 2 bits unused
    // stride dimension byte offset, bit [32,46), 4LSB not included
    uint16_t stride_byte_offset_ : 14, version_ : 2;       // 14 bits [0,14), 2 bits [14,16)
    // base_offset, bit [49,52). leading_byte_offset_mode, bit [52,53).
    uint8_t : 1, base_offset_ : 3, lbo_mode_ : 1, : 3;     // 1 bit unused, 3 bits [1,4), 1 bit [4,5), 3 bits unused
    // layout type, bit [61,64), SWIZZLE_NONE matrix descriptor = 0, SWIZZLE_128B matrix descriptor = 2, SWIZZLE_64B descriptor = 4, SWIZZLE_32B descriptor = 6, SWIZZLE_128B_BASE32B = 1, N/A = 3, N/A = 5, N/A = 7
    uint8_t : 5, layout_type_ : 3;                         // 6 bits unused, 3 bits [5,8)
  };
  // Seperate the field, as we may only update one part of desc
  struct {
    uint32_t lo;
    uint32_t hi;
  };

  // Decay to a uint64_t
  HOST_DEVICE constexpr
  operator uint64_t() const noexcept { return desc_; }
};
"""  # noqa: E501

    if "instr_descriptor" in tags:
        header += R"""
#ifndef HOST_DEVICE
#define HOST_DEVICE __forceinline__ __host__ __device__
#endif
union InstrDescriptor
{
  uint32_t desc_;

  struct {
    // Bitfield implementation avoids the need for shifts in assignment
    uint16_t sparse_id2_    : 2,  // bit [ 0, 2) : Sparse meta data id2
             sparse_flag_   : 1,  // bit [ 2, 3) : 0 = dense. 1 = sparse. 1 value valid only for F32F16/S8/MXF8F6F4
             saturate_      : 1,  // bit [ 3, 4) : 0 = no saturate. 1 = saturate. 1 value valid only for S8
             c_format_      : 2,  // bit [ 4, 6) : 0 = F16. 1 = F32, 2 = S32
                            : 1,  //
             a_format_      : 3,  // bit [ 7,10) : MXF8F6F4Format:0 = E4M3, 1 = E5M2, 3 = E2M3, 4 = E3M2, 5 = E2M1. F32F16Format: 0 = F16, 1 = BF16, 2 = TF32. S8: 0 unsigned 8 bit, 1 signed 8 bit. Boolean MMA: 0 Boolean
             b_format_      : 3,  // bit [10,13) : MXF8F6F4Format:0 = E4M3, 1 = E5M2, 3 = E2M3, 4 = E3M2, 5 = E2M1. F32F16Format: 0 = F16, 1 = BF16, 2 = TF32. S8: 0 unsigned 8 bit, 1 signed 8 bit. Boolean MMA: 0 Boolean
             a_negate_      : 1,  // bit [13,14) : 0 = no negate. 1 = negate. 1 value valid only for F32F16Format and MXF8F6F4Format
             b_negate_      : 1,  // bit [14,15) : 0 = no negate. 1 = negate. 1 value valid only for F32F16Format and MXF8F6F4Format
             a_major_       : 1;  // bit [15,16) : 0 = K-major. 1 = MN-major. Major value of 1 is only valid for E4M3, E5M2, INT8 (signed and unsigned), F16, BF16 and TF32 source formats
    uint16_t b_major_       : 1,  // bit [16,17) : 0 = K-major. 1 = MN-major. Major value of 1 is only valid for E4M3, E5M2, INT8 (signed and unsigned), F16, BF16 and TF32 source formats
             n_dim_         : 6,  // bit [17,23) : 3 LSBs not included. Valid values range from 1 (N=8) to 32 (N=256).  All values are not valid for all instruction formats
                            : 1,  //
             m_dim_         : 5,  // bit [24,29) : 4 LSBs not included. Valid values are: 4 (M=64), 8 (M=128), 16 (M=256)
                            : 1,  //
             max_shift_     : 2;  // bit [30,32) : Maximum shift for WS instruction. Encoded as follows: 0 = no shift, 1 = maximum shift of 8, 2 = maximum shift of 16, 3 = maximum shift of 32.
  };

  // Decay to a uint32_t
  HOST_DEVICE constexpr explicit
  operator uint32_t() const noexcept { return desc_; }
};
"""  # noqa: E501

    if "instr_descriptor_block_scaled" in tags:
        header += R"""
#ifndef HOST_DEVICE
#define HOST_DEVICE __forceinline__ __host__ __device__
#endif
union InstrDescriptorBlockScaled
{
  uint32_t desc_;

  struct {
    // Bitfield implementation avoids the need for shifts in assignment
    uint16_t sparse_id2_    : 2,  // bit [ 0, 2) : Sparse meta data id2
             sparse_flag_   : 1,  // bit [ 2, 3) : 0 = dense. 1 = sparse. 1 value valid only for F32F16/S8/MXF8F6F4
                            : 1,  //
             b_sf_id_       : 2,  // bit [ 4, 6) : Matrix B Scale Factor ID
                            : 1,  //
             a_format_      : 3,  // bit [ 7, 9) : MXF8F6F4Format:0 = E4M3, 1 = E5M2, 3 = E2M3, 4 = E3M2, 5 = E2M1. F32F16Format: 0 = F16, 1 = BF16, 2 = TF32. S8: 0 unsigned 8 bit, 1 signed 8 bit. BMMA: 0 Boolean
             b_format_      : 3,  // bit [10,12) : MXF8F6F4Format:0 = E4M3, 1 = E5M2, 3 = E2M3, 4 = E3M2, 5 = E2M1. F32F16Format: 0 = F16, 1 = BF16, 2 = TF32. S8: 0 unsigned 8 bit, 1 signed 8 bit. BMMA: 0 Boolean
             a_negate_      : 1,  // bit [13,14) : 0 = no negate. 1 = negate. 1 value valid only for F32F16Format and MXF8F6F4Format
             b_negate_      : 1,  // bit [14,15) : 0 = no negate. 1 = negate. 1 value valid only for F32F16Format and MXF8F6F4Format
             a_major_       : 1;  // bit [15,16) : 0 = K-major. 1 = MN-major. Major value of 1 is only valid for E4M3, E5M2, INT8 (signed and unsigned), F16, BF16 and TF32 source formats
    uint16_t b_major_       : 1,  // bit [16,17) : 0 = K-major. 1 = MN-major. Major value of 1 is only valid for E4M3, E5M2, INT8 (signed and unsigned), F16, BF16 and TF32 source formats
             n_dim_         : 6,  // bit [17,23) : 3 LSBs not included. Valid values range from 1 (N=8) to 32 (N=256).  All values are not valid for all instruction formats
             scale_format_  : 1,  // bit [23,24) : 0=E4M3, 1=E8M0
             m_dim_         : 5,  // bit [24,29) : 4 LSBs not included. Valid values are: 4 (M=64), 8 (M=128), 16 (M=256)
             a_sf_id_       : 2,  // bit [29,31) : Matrix A Scale Factor ID
                            : 1;  //
  };

  // Decay to a uint32_t
  HOST_DEVICE constexpr
  operator uint32_t() const noexcept { return desc_; }
};
"""  # noqa: E501

    if "elect_one_sync" in tags:
        header += R"""
__forceinline__ __device__ uint32_t tvm_builtin_elect_one_sync() {{
  uint32_t pred = 0;
  uint32_t laneid = 0;
  asm volatile(
    "{\n"
    ".reg .b32 %%rx;\n"
    ".reg .pred %%px;\n"
    "     elect.sync %%rx|%%px, %2;\n"
    "@%%px mov.s32 %1, 1;\n"
    "     mov.s32 %0, %%rx;\n"
    "}\n"
    : "+r"(laneid), "+r"(pred)
    : "r"(0xFFFFFFFF));
  return pred;
}}
"""
    return header
