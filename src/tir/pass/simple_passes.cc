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
 * \file simple_passes.cc
 * \brief Implementation of simple passes
 */
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/ir_pass.h>

namespace tvm {
namespace tir {

class IRSideEffect : public ExprVisitor {
 public:
  void VisitExpr(const PrimExpr& e) final {
    if (has_side_effect_) return;
    ExprVisitor::VisitExpr(e);
  }

  void VisitExpr_(const CallNode* op) final {
    if (!op->is_pure()) {
      has_side_effect_ = true; return;
    } else {
      ExprVisitor::VisitExpr_(op);
    }
  }

  bool has_side_effect_{false};
};

bool HasSideEffect(const PrimExpr& e) {
  IRSideEffect v;
  v(e);
  return v.has_side_effect_;
}



class VarTouchVisitor : public ExprVisitor {
 public:
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

  virtual void Handle(const VarNode* var) = 0;

  bool use_var_{false};
};

class ExprUseVarVisitor : public VarTouchVisitor {
 public:
  explicit ExprUseVarVisitor(const VarNode* var)
      : var_(var) {}

  void Handle(const VarNode* var) final {
    if (var == var_) use_var_ = true;
  }
 private:
  const VarNode* var_;
};

class ExprUseVSetVisitor : public VarTouchVisitor {
 public:
  explicit ExprUseVSetVisitor(
      const std::unordered_set<const VarNode*>& vset)
      : vset_(vset) {}

  void Handle(const VarNode* var) final {
    if (vset_.count(var)) use_var_ = true;
  }
 private:
  const std::unordered_set<const VarNode*>& vset_;
};

bool ExprUseVar(const PrimExpr& e, const Var& v) {
  ExprUseVarVisitor visitor(v.get());
  visitor(e);
  return visitor.use_var_;
}

bool ExprUseVar(const PrimExpr& e,
                const std::unordered_set<const VarNode*>& vset) {
  ExprUseVSetVisitor visitor(vset);
  visitor(e);
  return visitor.use_var_;
}

}  // namespace tir
}  // namespace tvm
