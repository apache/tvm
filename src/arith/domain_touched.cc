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
 * \file bound_deducer.cc
 * \brief Utility to deduce bound of expression
 */
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/te/tensor.h>
#include <tvm/runtime/registry.h>

#include <unordered_set>
#include <unordered_map>

namespace tvm {
namespace arith {

using namespace tir;

// Find Read region of the tensor in the stmt.
class FuncTouchedDomain final : public StmtExprVisitor {
 public:
  FuncTouchedDomain(const te::Tensor &tensor, bool consider_calls, bool consider_provides)
    : tensor_(tensor), consider_calls_(consider_calls), consider_provides_(consider_provides)  {}

  Domain Find(const Stmt& stmt) {
    operator()(stmt);
    Domain ret;
    Range none;
    for (size_t i = 0; i < bounds_.size(); ++i) {
      ret.push_back(arith::Union(bounds_[i]).cover_range(none));
    }
    return ret;
  }

  void VisitStmt_(const ForNode *op) final {
    const VarNode* var = op->loop_var.get();
    dom_map_[var] = IntSet::range(
        Range::make_by_min_extent(op->min, op->extent));
    StmtExprVisitor::VisitStmt_(op);
    dom_map_.erase(var);
  }

  void VisitStmt_(const LetStmtNode* op) final {
    dom_map_[op->var.get()] =
        arith::EvalSet(op->value, dom_map_);
    StmtExprVisitor::VisitStmt_(op);
    dom_map_.erase(op->var.get());
  }

  /* TODO: Thread extent unitest not generated.*/
  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent) {
      const IterVarNode* thread_axis = op->node.as<IterVarNode>();
      CHECK(thread_axis);
      const VarNode* var = thread_axis->var.get();
      dom_map_[var] = IntSet::range(Range(make_zero(op->value.dtype()), op->value));
      StmtExprVisitor::VisitStmt_(op);
      dom_map_.erase(var);
    } else {
      StmtExprVisitor::VisitStmt_(op);
    }
  }

  void VisitExpr_(const CallNode* op) final {
    if (consider_calls_ && tensor_->op.same_as(op->func)
        && tensor_->value_index == op->value_index) {
      Touch(op->args);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const ProvideNode* op) final {
    if (consider_provides_ && tensor_->op.same_as(op->func)
        && tensor_->value_index == op->value_index) {
      Touch(op->args);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

 private:
  void Touch(const Array<PrimExpr>& args) {
    if (args.size() > bounds_.size()) {
      bounds_.resize(args.size());
    }
    for (size_t i = 0; i < args.size(); ++i) {
      bounds_[i].emplace_back(EvalSet(args[i], dom_map_));
    }
  }

  const te::Tensor &tensor_;
  bool consider_calls_, consider_provides_;
  std::vector<std::vector<IntSet> > bounds_;
  std::unordered_map<const VarNode*, IntSet> dom_map_;
};

Domain DomainTouched(Stmt stmt,
                     const te::Tensor &tensor,
                     bool consider_calls,
                     bool consider_provides) {
  return FuncTouchedDomain(tensor, consider_calls, consider_provides).Find(stmt);
}

}  // namespace arith
}  // namespace tvm
