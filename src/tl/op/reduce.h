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
 * \file tl/op/reduce.h
 * \brief Define reduce operator.
 *
 */

#ifndef TVM_TL_OP_REDUCE_H_
#define TVM_TL_OP_REDUCE_H_

#include "op.h"

namespace tvm {
namespace tl {

using namespace tir;

class ReduceOp : public Operator {
 public:
  ReduceOp(const Array<PrimExpr>& args, const Map<Var, Buffer>& vmap);
  Stmt Lower(const LowerArgs& T, arith::Analyzer* analyzer) const final;
  LayoutMap InferLayout(const LayoutInferArgs& T, InferLevel level) final;
  static const Op& Get();

 private:
  tir::Buffer src, dst;
  int dim;
  enum class ReduceType {
    kSum,
    kMax,
    kMin,
  } type;
  bool clear;

  PrimExpr MakeInitValue() const;
  PrimExpr MakeReduce(const PrimExpr& a, const PrimExpr& b) const;
  std::string MakeCodegenReducer() const;
};

}  // namespace tl
}  // namespace tvm

#endif  //  TVM_TL_OP_REDUCE_H_