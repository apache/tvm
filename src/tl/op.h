/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
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
 * \file tl/op.h
 * \brief Tile library operations.
 *
 */

#ifndef TVM_TL_OP_H_
#define TVM_TL_OP_H_

#include <tvm/ir/op.h>
#include <tvm/tir/buffer.h>
#include <tvm/arith/analyzer.h>

namespace tvm {
namespace tl {

using namespace tir;

TVM_DLL const Op& copy();

TVM_DLL const Op& gemm();

TVM_DLL const Op& fill();

TVM_DLL const Op& region();

TVM_DLL const Op& reduce();

struct GemmArgs {
  tir::Buffer A, B, C;
  bool trans_A, trans_B;
  int M, N, K;
  enum class GemmWarpPolicy {
    kSquare = 0,
    kFullRow = 1,
    kFullCol = 2,
  } policy;

  static GemmArgs Parse(const Array<PrimExpr>& args, const Map<Var, Buffer>& vmap);

  std::pair<int, int> ComputeWarpPartition(int num_warps) const;
};

struct CopyArgs {
  tir::Buffer src, dst;
  Array<Range> src_range, dst_range;

  static CopyArgs Parse(const Array<PrimExpr>& args);

  Array<IterVar> MakeIterVars() const;
  // ivs: itervars returned by MakeIterVars()
  // src_dst: 0 for src_indices, 1 for dst_indices
  Array<PrimExpr> MakeIndices(const Array<IterVar>& ivs, int src_dst) const;
  PrimExpr MakePredicate(arith::Analyzer* analyzer, const Array<IterVar>& ivs, Array<PrimExpr> extents, int src_dst) const;
  bool CheckRangeEqual() const;
};

struct FillArgs {
  tir::Buffer dst;
  PrimExpr value;

  static FillArgs Parse(const Array<PrimExpr>& args, const Map<Var, Buffer>& vmap);
};

struct ReduceArgs {
  tir::Buffer src, dst;
  int dim;
  enum class ReduceType {
    kSum,
    kMax,
  } type;
  bool clear;
  static ReduceArgs Parse(const Array<PrimExpr>& args, const Map<Var, Buffer>& vmap);

  PrimExpr MakeInitValue() const;
  PrimExpr MakeReduce(const PrimExpr& a, const PrimExpr& b) const;
  std::string MakeCodegenReducer() const;
};

Array<Range> ParseRegionArgs(const tir::CallNode* call);

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_OP_H_
