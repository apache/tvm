#pragma once

#include <hip/hip_runtime.h>
#include <ck_tile/core.hpp>

using ck_tile::bfloat16_t;
using ck_tile::half_t;

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

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
