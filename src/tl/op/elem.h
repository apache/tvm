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
 * \file tl/op/elem.h
 * \brief Define elment-wise operators.
 *
 */

#ifndef TVM_TL_OP_ELEM_H_
#define TVM_TL_OP_ELEM_H_

#include "op.h"

namespace tvm {
namespace tl {

using namespace tir;

class Copy : public Operator {
 public:
  Copy(Array<PrimExpr> args, BufferMap vmap);
  Stmt Lower(const LowerArgs& T, arith::Analyzer* analyzer) const final;
  LayoutMap InferLayout(const LayoutInferArgs& T, InferLevel level) final;

  static const Op& Get();

 protected:
  Stmt LowerBulkCopy(const LowerArgs& T, arith::Analyzer* analyzer) const;
  Array<IterVar> MakeIterVars() const;

  // ivs: itervars returned by MakeIterVars()
  // src_dst: 0 for src_indices, 1 for dst_indices
  Array<PrimExpr> MakeIndices(const Array<IterVar>& ivs, int src_dst) const;

  PrimExpr MakePredicate(arith::Analyzer* analyzer, const Array<IterVar>& ivs,
                         Array<PrimExpr> extents, int src_dst) const;

  Array<PrimExpr> args_;

  Buffer src, dst;
  Array<Range> src_range, dst_range;
};

class Fill : public Operator {
 public:
  Fill(Array<PrimExpr> args, BufferMap vmap);
  Stmt Lower(const LowerArgs& T, arith::Analyzer* analyzer) const final;
  static const Op& Get();

 private:
  tir::Buffer dst;
  PrimExpr value;
};

}  // namespace tl
}  // namespace tvm

#endif  //  TVM_TL_OP_ELEM_H_