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
#include "../utils.h"

namespace tvm {
namespace tir {

/******** Binding ********/

bool IsAffineBinding(const BlockRealize& realize, const Map<Var, Range>& loop_var_ranges,
                     arith::Analyzer* analyzer) {
  if (loop_var_ranges.empty()) {
    return true;
  }
  Array<arith::IterSumExpr> results = arith::DetectIterMap(
      /*indices=*/realize->iter_values,
      /*input_iters=*/loop_var_ranges,
      /*predicate=*/realize->predicate,
      /*require_bijective=*/false,
      /*analyzer=*/analyzer);
  if (results.empty()) {
    return false;
  }
  for (const arith::IterSumExpr& sum_expr : results) {
    const Array<arith::IterSplitExpr>& args = sum_expr->args;
    if (!args.empty() && !is_one(args[0]->scale)) {
      return false;
    }
  }
  return true;
}

Map<Var, Range> LoopDomainOfSRefTreePath(const StmtSRef& low_inclusive,
                                         const Optional<StmtSRef>& high_exclusive,
                                         const runtime::StorageScope& extra_relax_scope) {
  Map<Var, Range> result;
  const StmtSRefNode* p = low_inclusive.get();
  const StmtSRefNode* limit = static_cast<const StmtSRefNode*>(high_exclusive.get());
  for (; p != limit; p = p->parent) {
    const ForNode* loop = p->StmtAs<ForNode>();
    if (loop == nullptr) {
      break;
    }
    result.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
  }
  if (extra_relax_scope.rank != runtime::StorageRank::kGlobal) {
    for (; p; p = p->parent) {
      if (const ForNode* loop = p->StmtAs<ForNode>()) {
        if (loop->kind == ForKind::kThreadBinding) {
          const String& thread_tag = loop->thread_binding.value()->thread_tag;
          if (CanRelaxStorageUndereThread(extra_relax_scope,
                                          runtime::ThreadScope::Create(thread_tag))) {
            result.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
          }
        }
      }
    }
  }
  return result;
}

Map<Var, PrimExpr> GetBindings(const BlockRealize& realize) {
  const BlockNode* block = realize->block.get();
  const Array<IterVar>& all_lhs = block->iter_vars;
  const Array<PrimExpr>& all_rhs = realize->iter_values;
  ICHECK_EQ(all_lhs.size(), all_rhs.size());
  Map<Var, PrimExpr> result;
  for (int i = 0, n = all_lhs.size(); i < n; ++i) {
    const IterVar& lhs = all_lhs[i];
    const PrimExpr& rhs = all_rhs[i];
    result.Set(lhs->var, rhs);
  }
  return result;
}

/******** Block-loop relation ********/

Array<StmtSRef> GetBlocks(const ScheduleState& self, const String& name, const String& func_name) {
  struct Finder : public StmtVisitor {
    explicit Finder(const ScheduleState& self, const String& name) : self_(self), name_(name) {}

    void VisitStmt_(const BlockNode* block) override {
      if (block->name_hint == name_) {
        auto it = self_->stmt2ref.find(block);
        ICHECK(it != self_->stmt2ref.end());
        results_.push_back(it->second);
      }
      StmtVisitor::VisitStmt_(block);
    }

    const ScheduleState& self_;
    const String& name_;
    Array<StmtSRef> results_;
  };

  BaseFunc func = self->mod->Lookup(func_name);
  const auto* prim_func = TVM_TYPE_AS(prim_func, func, PrimFuncNode);
  Finder finder(self, name);
  finder(prim_func->body);
  return std::move(finder.results_);
}

Array<StmtSRef> GetLoops(const StmtSRef& block_sref) {
  std::vector<StmtSRef> result;
  for (StmtSRefNode* parent = block_sref->parent; parent && parent->stmt->IsInstance<ForNode>();
       parent = parent->parent) {
    result.push_back(GetRef<StmtSRef>(parent));
  }
  return {result.rbegin(), result.rend()};
}

Array<StmtSRef> GetChildBlocks(const ScheduleState& self, const StmtSRef& parent_sref) {
  struct Collector : public StmtVisitor {
   public:
    static Array<StmtSRef> Collect(const ScheduleState& self, const Stmt& stmt) {
      Collector collector(self);
      collector(stmt);
      return std::move(collector.result_);
    }

   private:
    explicit Collector(const ScheduleState& self) : self_(self) {}

    void VisitStmt_(const BlockNode* block) final {
      auto it = self_->stmt2ref.find(block);
      ICHECK(it != self_->stmt2ref.end());
      result_.push_back(it->second);
    }

    const ScheduleState& self_;
    Array<StmtSRef> result_;
  };

  if (parent_sref->stmt->IsInstance<ForNode>()) {
    const auto* loop = static_cast<const ForNode*>(parent_sref->stmt);
    return Collector::Collect(self, loop->body);
  } else if (parent_sref->stmt->IsInstance<BlockNode>()) {
    const auto* block = static_cast<const BlockNode*>(parent_sref->stmt);
    return Collector::Collect(self, block->body);
  }
  ICHECK(false) << "Unreachable";
  throw;
}

}  // namespace tir
}  // namespace tvm
