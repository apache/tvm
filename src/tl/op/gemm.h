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
 * \file tl/op/gemm.h
 * \brief Define gemm operator.
 *
 */

#ifndef TVM_TL_OP_GEMM_H_
#define TVM_TL_OP_GEMM_H_

#include "op.h"

namespace tvm {
namespace tl {

using namespace tir;

class Gemm : public Operator {
 public:
  Gemm(const Array<PrimExpr>& args, const Map<Var, Buffer>& vmap);
  Stmt Lower(const LowerArgs& T, arith::Analyzer* analyzer) const final;
  LayoutMap InferLayout(const LayoutInferArgs& T, InferLevel level) final;
  static const Op& Get();
  enum class GemmWarpPolicy {
    kSquare = 0,
    kFullRow = 1,
    kFullCol = 2,
  } policy;

 private:
  std::pair<int, int> ComputeWarpPartition(int num_warps) const;

  Array<PrimExpr> call_args;
  tir::Buffer A, B, C;
  bool trans_A, trans_B;
  int M, N, K;

  bool completed_ = false;
};

}  // namespace tl
}  // namespace tvm

#endif  //  TVM_TL_OP_GEMM_H_