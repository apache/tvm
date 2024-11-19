#pragma once

#include "common.h"

namespace tl {

struct SumOp {
  template <typename T>
  TL_DEVICE T operator()(T const& x, T const& y) {
    return x + y;
  }
};

struct MaxOp {
  template <typename T>
  TL_DEVICE T operator()(T const& x, T const& y) {
    return ck_tile::max(x, y);
  }
};

struct MinOp {
  template <typename T>
  TL_DEVICE T operator()(T const& x, T const& y) {
    return ck_tile::min(x, y);
  }
};

template <class Reducer, int threads, int scale>
struct AllReduce {
  static_assert(threads == 1024 || threads == 512 || threads == 256 || threads == 128 ||
                threads == 64 || threads == 32 || threads == 16 || threads == 8 || threads == 4 ||
                threads == 2);
  static_assert(threads % scale == 0);

  template <typename T>
  static __device__ T run(T x, T* red_buf = nullptr) {
    constexpr int offset = threads / 2;
    constexpr int warpSize = 64;

    if constexpr (offset >= warpSize) {
      __syncthreads();
      red_buf[threadIdx.x] = x;
      __syncthreads();
      x = Reducer()(x, red_buf[threadIdx.x ^ offset]);
    } else {
      x = Reducer()(x, __shfl_xor(x, offset));
    }
    if constexpr (offset == scale) {
      return x;
    } else {
      return AllReduce<Reducer, offset, scale>::run(x, red_buf);
    }
  }
};

}  // namespace tl
