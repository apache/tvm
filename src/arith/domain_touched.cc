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
class BufferTouchedDomain final : public StmtExprVisitor {
 public:
  BufferTouchedDomain(const Buffer &buffer,
                      bool consider_loads,
                      bool consider_stores)
      : buffer_(buffer),
        consider_loads_(consider_loads),
        consider_stores_(consider_stores)  {}

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
    if (op->attr_key == tir::attr::thread_extent) {
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

  void VisitExpr_(const BufferLoadNode* op) final {
    if (consider_loads_ && buffer_.same_as(op->buffer)) {
      Touch(op->indices);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    if (consider_stores_ && buffer_.same_as(op->buffer)) {
      Touch(op->indices);
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

  const Buffer &buffer_;
  bool consider_loads_, consider_stores_;
  std::vector<std::vector<IntSet> > bounds_;
  std::unordered_map<const VarNode*, IntSet> dom_map_;
};

Domain DomainTouched(const Stmt& stmt,
                     const Buffer& buffer,
                     bool consider_loads,
                     bool consider_stores) {
  return BufferTouchedDomain(buffer, consider_loads, consider_stores).Find(stmt);
}

TVM_REGISTER_GLOBAL("arith.DomainTouched")
.set_body_typed(DomainTouched);

}  // namespace arith
}  // namespace tvm
