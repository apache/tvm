#pragma once

#include "common.h"

namespace tl
{

struct SumOp {
  template<typename T>
  __device__ inline T operator()(T const& x, T const& y) { return x + y; }
};

struct MaxOp {
  template<typename T>
  __device__ inline T operator()(T const& x, T const& y) { return x > y ? x : y; }
};

template<class Reducer, int threads, int scale>
struct AllReduce {
  static_assert(threads == 32 || threads == 16 || threads == 8 || threads == 4 || threads == 2);
  static_assert(threads % scale == 0);
  template<typename T>
  static __device__ inline T run(T x) {
    constexpr int offset = threads / 2;
    x = Reducer()(x, T(__shfl_xor_sync(uint32_t(-1), x, offset)));
    if constexpr (offset == scale) {
      return x;
    } else {
      return AllReduce<Reducer, offset, scale>::run(x);
    }
  }
};

} // namespace tl
