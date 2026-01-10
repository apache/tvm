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

#ifndef TVM_TIR_ANALYSIS_SYMBOLIC_MATCHER_H_
#define TVM_TIR_ANALYSIS_SYMBOLIC_MATCHER_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>

namespace tvm {

namespace arith {
class Analyzer;
}

namespace tir {

/*!
 * \brief Match symbolic vars according to the given PrimExpr, and update the var_remap.
 * Will throw errors if there is a mismatch.
 */
class SymbolicMatcher : ExprFunctor<void(const PrimExpr& n, const PrimExpr& other)> {
 public:
  explicit SymbolicMatcher(arith::Analyzer* analyzer, ffi::Map<tir::Var, PrimExpr>* var_remap)
      : analyzer_(analyzer), var_remap_(var_remap) {}

  void Match(const ffi::Array<PrimExpr>& params, const ffi::Array<PrimExpr>& args);
  void Match(const PrimExpr& param, const PrimExpr& arg);

 private:
  void VisitExpr(const PrimExpr& node, const PrimExpr& other);

  void VisitExpr_(const AddNode* op, const PrimExpr& other) final;
  void VisitExpr_(const SubNode* op, const PrimExpr& other) final;
  void VisitExpr_(const MulNode* op, const PrimExpr& other) final;
  void VisitExpr_(const DivNode* op, const PrimExpr& other) final;
  void VisitExpr_(const ModNode* op, const PrimExpr& other) final;
  void VisitExpr_(const EQNode* op, const PrimExpr& other) final;
  void VisitExpr_(const NENode* op, const PrimExpr& other) final;
  void VisitExpr_(const LTNode* op, const PrimExpr& other) final;
  void VisitExpr_(const LENode* op, const PrimExpr& other) final;
  void VisitExpr_(const GTNode* op, const PrimExpr& other) final;
  void VisitExpr_(const GENode* op, const PrimExpr& other) final;
  void VisitExpr_(const AndNode* op, const PrimExpr& other) final;
  void VisitExpr_(const OrNode* op, const PrimExpr& other) final;
  void VisitExpr_(const MinNode* op, const PrimExpr& other) final;
  void VisitExpr_(const MaxNode* op, const PrimExpr& other) final;
  void VisitExpr_(const FloorDivNode* op, const PrimExpr& other) final;
  void VisitExpr_(const FloorModNode* op, const PrimExpr& other) final;

  void VisitExpr_(const IntImmNode* op, const PrimExpr& other) final;
  void VisitExpr_(const FloatImmNode* op, const PrimExpr& other) final;
  void VisitExpr_(const CastNode* op, const PrimExpr& other) final;
  void VisitExpr_(const VarNode* op, const PrimExpr& rhs) final;
  void VisitExpr_(const SelectNode* op, const PrimExpr& other) final;

  arith::Analyzer* analyzer_;
  ffi::Map<tir::Var, PrimExpr>* var_remap_;
  PrimExpr must_prove_ = Bool(true);
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_ANALYSIS_SYMBOLIC_MATCHER_H_
