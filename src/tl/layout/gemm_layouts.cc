/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file layout/gemm_layouts.cc
 * \brief Define Layout used in MMA and other operations.
 *
 */

#include <tvm/tir/stmt_functor.h>

#include <cmath>

#include "layout.h"

namespace tvm {
namespace tl {

static IterVar make_itervar(std::string name, PrimExpr dom) {
  Var var = Var(name);
  return IterVar(Range(0, dom), var, IterVarType::kDataPar);
}

Fragment makeGemmFragment8x8() {
  IterVar i = make_itervar("i", 8);
  IterVar j = make_itervar("j", 8);
  IterVar rep = make_itervar("rep", 1);
  PrimExpr forward_thread = FloorDiv(j->var, 2) + 4 * i;
  PrimExpr index = FloorMod(j->var, 2);
  return Fragment({i, j}, {index}, forward_thread, rep);
}
/*
From https://github.com/RadeonOpenCompute/amd_matrix_instruction_calculator
./matrix_calculator.py --architecture cdna1 --instruction v_mfma_f32_16x16x16f16
--detail-instruction
*/
Fragment makeGemmFragmentAB16x16CDNA() {
  IterVar i = make_itervar("i", 16);
  IterVar j = make_itervar("j", 16);
  IterVar rep = make_itervar("rep", 1);
  PrimExpr forward_thread = 16 * FloorDiv(j->var, 4) + i;
  PrimExpr index = FloorMod(j->var, 4);
  return Fragment({i, j}, {index}, forward_thread, rep);
}

Fragment makeGemmFragmentAB16x16CDNATransposed() {
  IterVar i = make_itervar("i", 16);
  IterVar j = make_itervar("j", 16);
  IterVar rep = make_itervar("rep", 1);
  PrimExpr forward_thread = 16 * FloorDiv(i->var, 4) + j;
  PrimExpr index = FloorMod(i->var, 4);
  return Fragment({i, j}, {index}, forward_thread, rep);
}

Fragment makeGemmFragmentC16x16CDNA() {
  IterVar i = make_itervar("i", 16);
  IterVar j = make_itervar("j", 16);
  IterVar rep = make_itervar("rep", 1);
  PrimExpr forward_thread = 16 * FloorDiv(j->var, 4) + i;
  PrimExpr index = FloorMod(j->var, 4);
  return Fragment({i, j}, {index}, forward_thread, rep);
}


Fragment makeGemmFragment8x8Transposed() {
  IterVar i = make_itervar("i", 8);
  IterVar j = make_itervar("j", 8);
  IterVar rep = make_itervar("rep", 1);
  PrimExpr forward_thread = FloorDiv(i->var, 2) + 4 * j;
  PrimExpr index = FloorMod(i->var, 2);
  return Fragment({i, j}, {index}, forward_thread, rep);
}

Fragment makeGemmFragment8x16() {
  IterVar i = make_itervar("i", 8);
  IterVar j = make_itervar("j", 16);
  IterVar rep = make_itervar("rep", 1);
  PrimExpr forward_thread = FloorDiv(j->var, 4) + 4 * i;
  PrimExpr index = FloorMod(j->var, 4);
  return Fragment({i, j}, {index}, forward_thread, rep);
}

Fragment makeGemmFragmentC_F64(const int block_m, const int block_n, const int warp_m,
                               const int warp_n) {
  ICHECK(block_m % warp_m == 0);
  ICHECK(block_n % warp_n == 0);
  ICHECK(warp_m % 16 == 0);
  ICHECK(warp_n % 16 == 0);
  auto base_layout = makeGemmFragment8x8();
  auto warp_layout = base_layout->Repeat({block_m / warp_m, block_n / warp_n}, true, false);
  auto block_layout = warp_layout->Repeat({warp_m / 8, warp_n / 8}, false, false);
  return block_layout;
}

Fragment makeGemmFragmentC(const int block_m, const int block_n, const int warp_m, const int warp_n,
                           const int element_size) {
  if (element_size == 64) return makeGemmFragmentC_F64(block_m, block_n, warp_m, warp_n);
  ICHECK(block_m % warp_m == 0);
  ICHECK(block_n % warp_n == 0);
  ICHECK(warp_m % 16 == 0) << "warp_m=" << warp_m;
  ICHECK(warp_n % 16 == 0) << "warp_n=" << warp_n;
  auto base_layout = makeGemmFragment8x8()->Repeat({2, 1}, false);
  auto warp_layout = base_layout->Repeat({block_m / warp_m, block_n / warp_n}, true, false);
  auto block_layout = warp_layout->Repeat({warp_m / 16, warp_n / 8}, false, false);
  return block_layout;
}

Fragment makeGemmFragmentCCDNA(const int block_m, const int block_n, const int warp_m, const int warp_n,
                           const int element_size) {
  if (element_size == 64) LOG(FATAL) << "Not supported";
  ICHECK(block_m % warp_m == 0);
  ICHECK(block_n % warp_n == 0);
  ICHECK(warp_m % 16 == 0) << "warp_m=" << warp_m;
  ICHECK(warp_n % 16 == 0) << "warp_n=" << warp_n;
  auto base_layout = makeGemmFragmentC16x16CDNA()->Repeat({1, 1}, false);
  auto warp_layout = base_layout->Repeat({warp_m / 16, warp_n / 16}, false, true);
  auto block_layout = warp_layout->Repeat({block_m / warp_m, block_n / warp_n}, true, false);
  return block_layout;
}

Fragment makeGemmFragmentCHopper(const int block_m, const int block_n, const int warp_m,
                                 const int warp_n, const int element_size) {
  ICHECK(block_m % warp_m == 0);
  // ICHECK(block_n == warp_n);
  ICHECK(warp_m % 16 == 0);
  auto warp_layout =
      makeGemmFragment8x8()->Repeat({2, warp_n / 8}, false, false);     // 16 x N (1 warp)
  auto block_layout = warp_layout->Repeat({block_m / warp_m, block_n / warp_n}, true, false);  // 16*Y x N (Y warp)
  return block_layout->Repeat({warp_m / 16, 1}, false, false);
}

Fragment makeGemmFragmentA(const int block_m, const int block_n, const int block_k,
                           const int warp_m, const int warp_n, const int element_size) {
  // assume not transposed
  ICHECK(block_m % warp_m == 0);
  ICHECK(block_n % warp_n == 0);
  ICHECK(warp_m % 16 == 0);
  ICHECK(block_k % 16 == 0);
  // Only support 8-bit and 16-bit
  ICHECK(element_size == 8 || element_size == 16);
  if (element_size == 8) {
    auto base_layout = makeGemmFragment8x16()->Repeat({2, 2}, false, false);
    auto warp_layout = base_layout->Repeat({block_m / warp_m, 1}, true)->Replicate(block_n / warp_n);
    auto block_layout = warp_layout->Repeat({warp_m / 16, block_k / 32}, false, false);
    return block_layout;
  } else if (element_size == 16) {
    auto base_layout = makeGemmFragment8x8()->Repeat({2, 2}, false, false);
    auto warp_layout = base_layout->Repeat({block_m / warp_m, 1}, true)->Replicate(block_n / warp_n);
    auto block_layout = warp_layout->Repeat({warp_m / 16, block_k / 16}, false, false);
    return block_layout;
  } else {
    ICHECK(0);
    return Fragment();
  }
}

Fragment makeGemmFragmentACDNA(const int block_m, const int block_n, const int block_k,
                           const int warp_m, const int warp_n, bool transposed) {
  // assume not transposed
  ICHECK(block_m % warp_m == 0);
  ICHECK(block_n % warp_n == 0);
  ICHECK(warp_m % 16 == 0);
  ICHECK(block_k % 16 == 0);
  if (transposed) {
    auto base_layout = makeGemmFragmentAB16x16CDNATransposed()->Repeat({1, 1}, false, false);
    auto warp_layout = base_layout->Repeat({warp_m / 16, block_k / 16}, false, false);
    auto block_layout = warp_layout->Repeat({block_m / warp_m, 1}, true, true)->Replicate(block_n / warp_n);
    return block_layout;
  } else {
    auto base_layout = makeGemmFragmentAB16x16CDNA()->Repeat({1, 1}, false, false);
    auto warp_layout = base_layout->Repeat({warp_m / 16, block_k / 16}, false, false);
    auto block_layout =
        warp_layout->Repeat({block_m / warp_m, 1}, true, true)->Replicate(block_n / warp_n);
    return block_layout;
  }
}


Fragment makeGemmFragmentB(const int block_m, const int block_n, const int block_k,
                           const int warp_m, const int warp_n) {
  // transposed
  ICHECK(warp_n % 8 == 0);
  ICHECK(block_k % 16 == 0);
  auto base_layout = makeGemmFragment8x8Transposed()->Repeat({2, 1}, false, false);
  auto warp_layout = base_layout->Replicate(block_m / warp_m)->Repeat({1, block_n / warp_n}, true);
  auto block_layout = warp_layout->Repeat({block_k / 16, warp_n / 8}, false, true);
  return block_layout;
}

Fragment makeGemmFragment32x32(int element_size) {
  IterVar i = make_itervar("i", 32);
  IterVar j = make_itervar("j", 32);
  IterVar rep = make_itervar("rep", 1);
  ICHECK(element_size == 16 || element_size == 32);
  if (element_size == 16) {
    PrimExpr thd = FloorMod(i, 4) + FloorDiv(FloorMod(i, 16), 8) * 4 +
                   FloorDiv(FloorMod(j, 16), 8) * 8 + FloorDiv(i, 16) * 16;
    PrimExpr idx = FloorMod(j, 4) + FloorDiv(j, 16) * 4 + FloorDiv(FloorMod(i, 8), 4) * 8 +
                   FloorDiv(FloorMod(j, 8), 4) * 16;
    return Fragment({i, j}, {idx}, thd, rep);
  } else {
    PrimExpr thd = FloorMod(i, 2) + 2 * FloorDiv(FloorMod(j, 4), 2) +
                   FloorDiv(FloorMod(i, 16), 8) * 4 + FloorDiv(FloorMod(j, 16), 8) * 8 +
                   FloorDiv(i, 16) * 16;
    PrimExpr idx = FloorMod(j, 2) + 2 * FloorDiv(FloorMod(i, 4), 2) + FloorDiv(j, 16) * 4 +
                   FloorDiv(FloorMod(i, 8), 4) * 8 + FloorDiv(FloorMod(j, 8), 4) * 16;
    return Fragment({i, j}, {idx}, thd, rep);
  }
}

Fragment makeGemmVoltaFragmentC(const int block_m, const int block_n, const int warp_m,
                                const int warp_n, int element_size) {
  ICHECK(block_m % warp_m == 0);
  ICHECK(block_n % warp_n == 0);
  ICHECK(warp_m % 32 == 0);
  ICHECK(warp_n % 32 == 0);
  auto base_layout = makeGemmFragment32x32(element_size);
  auto warp_layout = base_layout->Repeat({warp_m / 32, warp_n / 32}, false, false);
  auto block_layout = warp_layout->Repeat({block_m / warp_m, block_n / warp_n}, true);
  return block_layout;
}

Fragment makeGemmVoltaFragmentA(const int block_m, const int block_n, const int block_k,
                                const int warp_m, const int warp_n) {
  // assume not transposed
  ICHECK(block_m % warp_m == 0);
  ICHECK(block_n % warp_n == 0);
  ICHECK(warp_m % 32 == 0);
  ICHECK(block_k % 4 == 0);
  // this is a special case
  IterVar i = make_itervar("i", 32);
  IterVar j = make_itervar("j", 4);
  IterVar rep = make_itervar("rep", 2);
  PrimExpr thd = FloorDiv(FloorMod(i, 16), 8) * 4 + 16 * FloorDiv(i, 16) + FloorMod(i, 4) + 8 * rep;
  PrimExpr idx = j + FloorDiv(FloorMod(i, 8), 4) * 4;
  Fragment base_layout = Fragment({i, j}, {idx}, thd, rep);
  auto warp_layout = base_layout->Repeat({warp_m / 32, block_k / 4}, false, false);
  auto block_layout = warp_layout->Replicate(block_n / warp_n)->Repeat({block_m / warp_m, 1}, true);
  return block_layout;
}

PrimExpr xor2x2(const PrimExpr& i, const PrimExpr& j) { return FloorMod(i + j, 2); }

PrimExpr xor4x4(const PrimExpr& i, const PrimExpr& j) {
  PrimExpr i0 = FloorMod(i, 2);
  PrimExpr j0 = FloorMod(j, 2);
  PrimExpr i1 = FloorDiv(i, 2);
  PrimExpr j1 = FloorDiv(j, 2);
  return 2 * xor2x2(i1, j1) + xor2x2(i0, j0);
}

PrimExpr xor8x8(const PrimExpr& i, const PrimExpr j) {
  PrimExpr i0 = FloorMod(i, 2);
  PrimExpr j0 = FloorMod(j, 2);
  PrimExpr i1 = FloorDiv(i, 2);
  PrimExpr j1 = FloorDiv(j, 2);
  return 2 * xor4x4(i1, j1) + xor2x2(i0, j0);
}

Layout makeHalfBankSwizzleLayout(int stride, int continuous, int element_size) {
  // Swizzle 2 bit
  Var i = InputPlaceholder(0);
  Var j = InputPlaceholder(1);
  int vector_size = 128 / element_size;
  ICHECK(stride % 8 == 0);
  ICHECK(continuous % (vector_size * 4) == 0);
  PrimExpr ts = FloorDiv(i, 8);
  PrimExpr s = FloorMod(i, 8);
  PrimExpr tc = FloorDiv(FloorDiv(j, vector_size), 4);
  PrimExpr c = FloorMod(FloorDiv(j, vector_size), 4);
  PrimExpr vec = FloorMod(j, vector_size);
  PrimExpr c_swizzle = xor4x4(c, FloorDiv(s, 2));
  PrimExpr index = vec + (c_swizzle + s * 4) * vector_size;
  return Layout(Array<PrimExpr>{stride, continuous}, {tc, ts, index});
}

Layout makeFullBankSwizzleLayout(int stride, int continuous, int element_size) {
  // Swizzle 3 bit
  Var i = InputPlaceholder(0);
  Var j = InputPlaceholder(1);
  int vector_size = 128 / element_size;
  ICHECK(stride % 8 == 0);
  ICHECK(continuous % (vector_size * 8) == 0);
  PrimExpr ts = FloorDiv(i, 8);
  PrimExpr s = FloorMod(i, 8);
  PrimExpr tc = FloorDiv(FloorDiv(j, vector_size), 8);
  PrimExpr c = FloorMod(FloorDiv(j, vector_size), 8);
  PrimExpr vec = FloorMod(j, vector_size);
  PrimExpr c_swizzle = xor8x8(c, s);
  PrimExpr index = vec + (c_swizzle + s * 8) * vector_size;
  return Layout(Array<PrimExpr>{stride, continuous}, {tc, ts, index});
}

// Detail implementation please ref to bitblas::tl::mfma_layout::make_mfma_swizzle_layout
Layout makeMatrixCoreSwizzleLayout(int stride, int continuous, int element_size, int kPack=1) {
  const int numBanks = 32;
  const int bankBitWidth = 32;
  const int SIMDWidth = 16;
  const int vecSize = 4 * kPack;
  const int innerDimLength = continuous;
  const int typeWidthInBit = element_size;

  const int elemsPerOneBanksRow = (numBanks * bankBitWidth) / typeWidthInBit;
  const int perPhase = std::max(1, elemsPerOneBanksRow / innerDimLength);
  const int maxPhase = std::min(SIMDWidth / perPhase, innerDimLength / vecSize);

  IterVar row = make_itervar("row", stride);
  IterVar col = make_itervar("col", continuous);
  PrimExpr phase = FloorMod(row / perPhase, maxPhase);
  PrimExpr colOffSwizzled = ((col / vecSize) ^ phase) * vecSize;
  PrimExpr colOffOrdered = FloorMod(col, vecSize);
  PrimExpr colOff = colOffSwizzled + colOffOrdered;

  return Layout(Array{row, col}, {row, colOff});
}

Layout makeGemmABLayoutF64_Kinner(int stride, int continuous) {
  // Swizzle<2, 0, 4>
  Var i = InputPlaceholder(0);
  Var j = InputPlaceholder(1);
  PrimExpr tc = FloorDiv(j, 16);
  PrimExpr ts = FloorDiv(i, 4);
  PrimExpr c = FloorMod(j, 16);
  PrimExpr s = FloorMod(i, 4);
  PrimExpr swizzled_c = FloorDiv(c, 4) * 4 + xor4x4(FloorMod(c, 4), s);
  PrimExpr index = swizzled_c + s * 16;
  return Layout(Array<PrimExpr>{stride, continuous}, {tc, ts, index});
}

Layout makeGemmABLayoutF64_Kouter(int stride, int continuous) {
  // Swizzle<2, 2, 2>
  Var i = InputPlaceholder(0);
  Var j = InputPlaceholder(1);
  PrimExpr tc = FloorDiv(j, 16);
  PrimExpr ts = FloorDiv(i, 4);
  PrimExpr c = FloorMod(j, 16);
  PrimExpr s = FloorMod(i, 4);
  PrimExpr swizzled_c = FloorMod(c, 4) + xor4x4(FloorDiv(c, 4), s) * 4;
  PrimExpr index = swizzled_c + s * 16;
  return Layout(Array<PrimExpr>{stride, continuous}, {tc, ts, index});
}

// The Default Layout for Tensor Access
Layout makeGemmLayoutLinear(int stride, int continuous) {
  IterVar i = make_itervar("i", stride);
  IterVar j = make_itervar("j", continuous);
  return Layout(Array{i, j}, {i * continuous + j});
}

Layout makeGemmABLayoutPadded(int stride, int continuous, int element_size) {
  IterVar i = make_itervar("i", stride);
  IterVar j = make_itervar("j", continuous);
  int padded = continuous;
  // Add 128 bits padding when the last dim is a multiple of 256 bits
  if ((element_size * continuous) % 256 == 0) padded += 128 / element_size;
  return Layout(Array{i, j}, {i * padded + j});
}

Layout MakeGemmVoltaABLayoutCrosswise(int stride, int continuous) {
  ICHECK(stride % 32 == 0 && continuous % 32 == 0);
  IterVar i = make_itervar("i", stride);
  IterVar j = make_itervar("j", continuous);
  PrimExpr vec_contiguous_idx = FloorDiv(j, 4);
  PrimExpr vec_strided_within_tile = FloorMod(vec_contiguous_idx, 8);

  PrimExpr bit2 = FloorMod(FloorDiv(FloorMod(i, 32), 16) + FloorDiv(FloorMod(i, 16), 8) +
                               FloorDiv(vec_strided_within_tile, 4),
                           2);
  PrimExpr bit1 =
      xor2x2(FloorDiv(FloorMod(i, 8), 4), FloorDiv(FloorMod(vec_strided_within_tile, 4), 2));
  PrimExpr permuted_vec_contiguous = FloorDiv(i, 16) * 16 + FloorMod(i, 4) * 4 + bit2 * 2 + bit1;

  PrimExpr offset = FloorMod(j, 4) + permuted_vec_contiguous * 4 + vec_contiguous_idx * stride * 4;
  return Layout(Array{i, j}, {offset});
}

Layout MakeGemmVoltaALayoutCongruous(int stride, int continuous) {
  ICHECK(stride % 4 == 0 && continuous % 64 == 0);
  IterVar i = make_itervar("i", stride);
  IterVar j = make_itervar("j", continuous);
  PrimExpr vec_contiguous_idx = FloorDiv(j, 8);
  PrimExpr vec_strided_idx = i;
  PrimExpr tile_contiguous_idx = FloorDiv(vec_contiguous_idx, 8);
  PrimExpr tile_strided_idx = FloorDiv(vec_strided_idx, 4);
  PrimExpr tile_contiguous_residual = FloorMod(vec_contiguous_idx, 8);
  PrimExpr tile_strided_residual = FloorMod(vec_strided_idx, 4);

  PrimExpr permuted_strided_within_tile = FloorDiv(tile_contiguous_residual, 2);
  PrimExpr permuted_contiguous_within_tile =
      FloorMod(tile_contiguous_residual, 2) * 4 +
      xor4x4(tile_strided_residual, permuted_strided_within_tile);

  PrimExpr element_strided = permuted_strided_within_tile + tile_strided_idx * 4;
  PrimExpr element_contiguous =
      FloorMod(j, 8) + (permuted_contiguous_within_tile + tile_contiguous_idx * 8) * 8;
  PrimExpr offset = element_strided * continuous + element_contiguous;
  return Layout(Array{i, j}, {offset});
}

Layout MakeGemmVoltaBLayoutCongruous(int stride, int continuous) {
  ICHECK(stride % 4 == 0 && continuous % 64 == 0);
  IterVar i = make_itervar("i", stride);
  IterVar j = make_itervar("j", continuous);
  PrimExpr vec_contiguous_idx = FloorDiv(j, 8);
  PrimExpr vec_strided_idx = i;
  PrimExpr tile_contiguous_idx = FloorDiv(vec_contiguous_idx, 8);
  PrimExpr tile_strided_idx = FloorDiv(vec_strided_idx, 4);
  PrimExpr tile_contiguous_residual = FloorMod(vec_contiguous_idx, 8);
  PrimExpr tile_strided_residual = FloorMod(vec_strided_idx, 4);

  PrimExpr permuted_strided_within_tile = FloorMod(tile_contiguous_residual, 4);
  PrimExpr permuted_contiguous_within_tile =
      FloorDiv(tile_contiguous_residual, 4) * 4 +
      xor4x4(tile_strided_residual, permuted_strided_within_tile);

  PrimExpr element_strided = permuted_strided_within_tile + tile_strided_idx * 4;
  PrimExpr element_contiguous =
      FloorMod(j, 8) + (permuted_contiguous_within_tile + tile_contiguous_idx * 8) * 8;
  PrimExpr offset = element_strided * continuous + element_contiguous;
  return Layout(Array{i, j}, {offset});
}

Layout makeGemmVoltaABLayout(int stride, int continuous, bool is_a, int kfactor) {
  if (kfactor == 2) return MakeGemmVoltaABLayoutCrosswise(stride, continuous);
  if (is_a && continuous % 64 == 0) return MakeGemmVoltaALayoutCongruous(stride, continuous);
  if (!is_a && continuous % 64 == 0) return MakeGemmVoltaBLayoutCongruous(stride, continuous);
  return makeGemmABLayoutPadded(stride, continuous, 16);
}

Layout makeGemmABLayout(int stride, int continuous, int element_size, int kfactor) {
  if (element_size == 64) {
    if (kfactor == 1 && continuous % 16 == 0)  // float64 KxN
      return makeGemmABLayoutF64_Kouter(stride, continuous);
    if (kfactor == 2 && continuous % 16 == 0)  // float64 NxK
      return makeGemmABLayoutF64_Kinner(stride, continuous);
    return makeGemmABLayoutPadded(stride, continuous, element_size);
  }
  int vector_size = 128 / element_size;
  if (kfactor == 1 && element_size == 8)  // int8 KxN
    return makeGemmABLayoutPadded(stride, continuous, element_size);
  else if (continuous % (vector_size * 8) == 0)
    return makeFullBankSwizzleLayout(stride, continuous, element_size);
  else if (continuous % (vector_size * 4) == 0)
    return makeHalfBankSwizzleLayout(stride, continuous, element_size);
  else {
    return makeGemmABLayoutPadded(stride, continuous, element_size);
  }
}

Layout makeGemmABLayoutCDNA(int stride, int continuous, int element_size, int kPack) {
  int vector_size = 128 / element_size;
  if (continuous % (vector_size * 4) == 0)
    return makeMatrixCoreSwizzleLayout(stride, continuous, element_size, kPack);
  else {
    return makeGemmABLayoutPadded(stride, continuous, element_size);
  }
}
}  // namespace tl
}  // namespace tvm
