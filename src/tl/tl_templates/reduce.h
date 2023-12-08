#pragma once

#include "common.h"

namespace tl {

struct SumOp {
  template <typename T>
  __device__ inline T operator()(T const& x, T const& y) {
    return x + y;
  }
};

struct MaxOp {
  template <typename T>
  __device__ inline T operator()(T const& x, T const& y) {
    return cutlass::fast_max(x, y);
  }
};

template <class Reducer, int threads, int scale>
struct AllReduce {
  static_assert(threads == 1024 or threads == 512 or threads == 256 or threads == 128 or
                threads == 64 or threads == 32 or threads == 16 or threads == 8 or threads == 4 or
                threads == 2);
  static_assert(threads % scale == 0);
  template <typename T>
  static __device__ inline T run(T x, T* red_buf = nullptr) {
    constexpr int offset = threads / 2;
    if constexpr (offset >= 32) {
      red_buf[threadIdx.x] = x;
      __syncthreads();
      x = Reducer()(x, red_buf[threadIdx.x ^ offset]);
    } else {
      x = Reducer()(x, T(__shfl_xor_sync(uint32_t(-1), x, offset)));
    }
    if constexpr (offset == scale) {
      return x;
    } else {
      return AllReduce<Reducer, offset, scale>::run(x, red_buf);
    }
  }
};

}  // namespace tl
