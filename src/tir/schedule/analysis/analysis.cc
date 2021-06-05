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

/******** Scope ********/

Optional<StmtSRef> GetScopeRoot(const StmtSRef& sref) {
  for (const StmtSRefNode* p = sref->parent; p != nullptr; p = p->parent) {
    if (p->stmt->IsInstance<BlockNode>()) {
      return GetRef<StmtSRef>(p);
    }
  }
  return NullOpt;
}

StmtSRef GetScopeRootAndCheckStagePipeline(const ScheduleState& self, const StmtSRef& sref) {
  class RootBlockError : public ScheduleError {
   public:
    explicit RootBlockError(IRModule mod) : mod_(mod) {}
    IRModule mod() const final { return mod_; }
    String FastErrorString() const final {
      return "ScheduleError: The primitive does not operate on the root block";
    }
    String DetailRenderTemplate() const final {
      return "The primitive does not operate on the root block";
    }
    Array<ObjectRef> LocationsOfInterest() const final { return {}; }
    IRModule mod_;
  };

  class NotStagePipelineError : public ScheduleError {
   public:
    explicit NotStagePipelineError(IRModule mod, Block block) : mod_(mod), block_(block) {}
    IRModule mod() const final { return mod_; }
    String FastErrorString() const final {
      return "ScheduleError: The scope root is not a stage pipeline";
    }
    String DetailRenderTemplate() const final {
      return R"(The scope {0} is not a stage pipeline.
Definition of a scope that is a stage pipeline:
- The region cover property holds for every of its child blocks
- No write-after-read dependency or opaque dependency,
- only read-after-write and write-after-write are allowed
- All the statements in the scope are schedulable statements, i.e. Block and For
)";
    }
    Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }
    IRModule mod_;
    Block block_;
  };

  StmtSRef scope_root_sref{nullptr};
  if (Optional<StmtSRef> opt_scope_root_sref = GetScopeRoot(sref)) {
    scope_root_sref = opt_scope_root_sref.value();
  } else {
    throw RootBlockError(self->mod);
  }
  bool stage_pipeline = self->GetBlockInfo(scope_root_sref).scope->stage_pipeline;
  if (stage_pipeline == false) {
    const BlockNode* block = TVM_SREF_TO_BLOCK(block, scope_root_sref);
    throw NotStagePipelineError(self->mod, GetRef<Block>(block));
  }
  return scope_root_sref;
}

/*!
 * \brief Check the dominant property of a block:
 * the block is the only writer of its output, dominating the reader of its output buffers
 * \param self The schedule state
 * \param block_sref The block whose dominant property is to be checked
 * \return A boolean indicating if the block is a dominant block
 */
bool IsDominantBlock(const BlockScope& self, const StmtSRef& block_sref) {
  // Check whether the input block is the only writer of its outputs
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  const std::unordered_map<Buffer, Array<StmtSRef>, ObjectPtrHash, ObjectPtrEqual>& buffer_writers =
      self->buffer_writers;
  for (const BufferRegion& write_region : block->writes) {
    ICHECK(buffer_writers.count(write_region->buffer))
        << "InternalError: buffer \"" << write_region->buffer->name
        << "\" does not exist in the current scope, when querying block:\n"
        << GetRef<Block>(block);
    if (buffer_writers.at(write_region->buffer).size() != 1) {
      return false;
    }
  }
  return true;
}

bool IsCompleteBlock(const ScheduleState& self, const StmtSRef& block_sref,
                     const StmtSRef& scope_root) {
  BlockScope scope = self->GetBlockScope(scope_root);
  // Cond 1. All block vars are data parallel
  const auto* block = TVM_SREF_TO_BLOCK(block, block_sref);
  for (const IterVar& iter_var : block->iter_vars) {
    if (iter_var->iter_type != kDataPar) {
      return false;
    }
  }
  // Cond 2. Dominant: the block is the only writer of its output,
  // dominating the reader of its output buffers
  if (!IsDominantBlock(scope, block_sref)) {
    return false;
  }
  // Cond 3. No overlap between the buffers the block reads and writes
  std::unordered_set<const BufferNode*> written_buffers;
  written_buffers.reserve(block->writes.size());
  for (const BufferRegion& write : block->writes) {
    written_buffers.insert(write->buffer.get());
  }
  for (const BufferRegion& read : block->reads) {
    if (written_buffers.count(read->buffer.get())) {
      return false;
    }
  }
  return true;
}

void CheckCompleteBlock(const ScheduleState& self, const StmtSRef& block_sref,
                        const StmtSRef& scope_root_sref) {
  class IncompleteBlockError : public ScheduleError {
   public:
    explicit IncompleteBlockError(IRModule mod, Block block) : mod_(mod), block_(block) {}
    String FastErrorString() const final { return "ScheduleError: Incomplete block"; }
    String DetailRenderTemplate() const final {
      return R"(The block {0} is not a complete block.
Definition of a complete block:
1) All block vars are data parallel
2) Dominant: the block is the only writer of its output, dominating the reader of its output buffers
3) No overlap between the buffers the block reads and writes)";
    }
    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }
    IRModule mod_;
    Block block_;
  };

  bool result = IsCompleteBlock(self, block_sref, scope_root_sref);
  if (result == false) {
    const BlockNode* block = TVM_SREF_TO_BLOCK(block, scope_root_sref);
    throw IncompleteBlockError(self->mod, GetRef<Block>(block));
  }
}

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
