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
 * \file simple_analysis.cc
 * \brief Implementation of simple passes
 */
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/analysis.h>

namespace tvm {
namespace tir {

class VarTouchVisitor : public ExprVisitor {
 public:
  explicit VarTouchVisitor(
      std::function<bool(const VarNode*)> var_set)
      : var_set_(var_set) {}

  void VisitExpr(const PrimExpr& e) final {
    if (use_var_) return;
    ExprVisitor::VisitExpr(e);
  }

  void VisitExpr_(const VarNode* op) final {
    Handle(op);
  }

  void VisitExpr_(const LoadNode* op) final {
    Handle(op->buffer_var.get());
    ExprVisitor::VisitExpr_(op);
  }

  void Handle(const VarNode* var) {
    if (var_set_(var)) use_var_ = true;
  }

  bool use_var_{false};

 private:
  std::function<bool(const VarNode*)> var_set_;
};


bool ExprUseVar(const PrimExpr& e,
                std::function<bool(const VarNode*)> var_set) {
  VarTouchVisitor visitor(var_set);
  visitor(e);
  return visitor.use_var_;
}

}  // namespace tir
}  // namespace tvm
