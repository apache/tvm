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

/******** IR Module ********/

const PrimFuncNode* GetRootPrimFunc(const IRModule& mod, const StmtNode* root_block,
                                    GlobalVar* result_g_var) {
  for (const auto& kv : mod->functions) {
    const GlobalVar& g_var = kv.first;
    const BaseFunc& base_func = kv.second;
    if (const auto* func = base_func.as<PrimFuncNode>()) {
      if (const auto* realize = func->body.as<BlockRealizeNode>()) {
        if (realize->block.get() == root_block) {
          if (result_g_var != nullptr) {
            *result_g_var = g_var;
          }
          return func;
        }
      }
    }
  }
  LOG(FATAL) << "IndexError: Could not get the corresponding function in the schedule state of the "
                "statement:\n"
             << GetRef<Stmt>(root_block);
  throw;
}

/******** Scope ********/

StmtSRef GetScopeRoot(const ScheduleState& self, const StmtSRef& sref,  //
                      bool require_stage_pipeline,                      //
                      bool require_subtree_compact_dataflow) {
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

  class NotCompactDataFlowError : public ScheduleError {
   public:
    explicit NotCompactDataFlowError(IRModule mod, Stmt subtree_root, Block violate_block)
        : mod_(std::move(mod)),
          subtree_root_(std::move(subtree_root)),
          violate_block_(std::move(violate_block)) {
      ICHECK(subtree_root_->IsInstance<BlockNode>() || subtree_root_->IsInstance<ForNode>());
    }
    String FastErrorString() const final {
      return "ScheduleError: The queried subtree root in SRef tree does not have compact dataflow, "
             "because some of its child block on SRef tree is neither a complete block nor a "
             "reduction block";
    }
    String DetailRenderTemplate() const final {
      return "The queried subtree root {0} in SRef tree does not have compact dataflow, because "
             "its child block {1} on SRef tree is neither a complete block nor a reduction block";
    }
    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {subtree_root_, violate_block_}; }

    IRModule mod_;
    Stmt subtree_root_;
    Block violate_block_;
  };

  StmtSRef scope_root_sref{nullptr};
  StmtSRef scope_root_subtree{nullptr};
  // Step 1. Find the scope root and the subtree that the given sref is in
  {
    const StmtSRefNode* p = sref->parent;
    const StmtSRefNode* subtree = sref.get();
    for (; p != nullptr; subtree = p, p = p->parent) {
      if (p->stmt->IsInstance<BlockNode>()) {
        scope_root_sref = GetRef<StmtSRef>(p);
        scope_root_subtree = GetRef<StmtSRef>(subtree);
        break;
      }
    }
    if (p == nullptr) {
      throw RootBlockError(self->mod);
    }
  }
  // Step 2. Handle `require_stage_pipeline`
  if (require_stage_pipeline) {
    bool stage_pipeline = self->GetBlockInfo(scope_root_sref).scope->stage_pipeline;
    if (stage_pipeline == false) {
      const BlockNode* block = TVM_SREF_TO_BLOCK(block, scope_root_sref);
      throw NotStagePipelineError(self->mod, GetRef<Block>(block));
    }
  }
  // Step 3. Handle `require_subtree_compact_dataflow`
  if (require_subtree_compact_dataflow) {
    Array<StmtSRef> child_block_srefs = GetChildBlockSRefOnSRefTree(self, scope_root_sref);
    for (const StmtSRef& block_sref : child_block_srefs) {
      if (!IsCompleteBlock(self, block_sref, scope_root_sref) &&
          !IsReductionBlock(self, block_sref, scope_root_sref)) {
        const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
        throw NotCompactDataFlowError(self->mod, GetRef<Stmt>(scope_root_subtree->stmt),
                                      GetRef<Block>(block));
      }
    }
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

/*!
 * \brief A helper function that checks whether a given block is a complete block under the scope,
 * or return the condition it violates if it is not a complete block
 * \param self The schedule state
 * \param block_sref The block to be checked
 * \param scope_root_sref The sref to the root block of the scope that `block_sref` is in
 * \return 0 if the block is a complete block, or a positive integer indicating which condition is
 * first violated
 */
int CheckCompleteBlockErrorCode(const ScheduleState& self, const StmtSRef& block_sref,
                                const StmtSRef& scope_root_sref) {
  BlockScope scope = self->GetBlockScope(scope_root_sref);
  // Cond 1. All block vars are data parallel
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  for (const IterVar& iter_var : block->iter_vars) {
    if (iter_var->iter_type != kDataPar) {
      return 1;
    }
  }
  // Cond 2. Dominant: the block is the only writer of its output,
  // dominating the reader of its output buffers
  if (!IsDominantBlock(scope, block_sref)) {
    return 2;
  }
  // Cond 3. No overlap between the buffers the block reads and writes
  std::unordered_set<const BufferNode*> written_buffers;
  written_buffers.reserve(block->writes.size());
  for (const BufferRegion& write : block->writes) {
    written_buffers.insert(write->buffer.get());
  }
  for (const BufferRegion& read : block->reads) {
    if (written_buffers.count(read->buffer.get())) {
      return 3;
    }
  }
  return 0;
}

static const char* kCompleteBlockDefinition = R"(Definition of a complete block:
1) All block vars are data parallel
2) Dominant: the block is the only writer of its output, dominating the reader of its output buffers
3) No overlap between the buffers the block reads and writes)";

static const char* kReductionBlockDefinition = R"(Definition of a reduction block:
1) The block has the `init` statement
2) All the block bindings are quasi-affine expressions
3) All block vars are either data parallel block vars or reduction block vars
4) Dominant: the block is the only writer of its output, dominating the reader of its output buffers
5) The reduction block vars are not used to index the output buffers)";

bool IsCompleteBlock(const ScheduleState& self, const StmtSRef& block_sref,
                     const StmtSRef& scope_root_sref) {
  return CheckCompleteBlockErrorCode(self, block_sref, scope_root_sref) == 0;
}

void CheckCompleteBlock(const ScheduleState& self, const StmtSRef& block_sref,
                        const StmtSRef& scope_root_sref) {
  class IncompleteBlockError : public ScheduleError {
   public:
    explicit IncompleteBlockError(IRModule mod, Block block, int violated_cond)
        : mod_(std::move(mod)), block_(std::move(block)), violated_cond_(violated_cond) {}
    String FastErrorString() const final { return "ScheduleError: Incomplete block"; }
    String DetailRenderTemplate() const final {
      std::ostringstream os;
      os << "The block {0} is not a complete block - it violates condition #" << violated_cond_;
      os << ".\n" << kCompleteBlockDefinition;
      return os.str();
    }
    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }
    IRModule mod_;
    Block block_;
    int violated_cond_;
  };

  int error_code = CheckCompleteBlockErrorCode(self, block_sref, scope_root_sref);
  if (error_code != 0) {
    const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
    throw IncompleteBlockError(self->mod, GetRef<Block>(block), error_code);
  }
}

/*!
 * \brief A helper function that checks whether a given block is a reduction block under the scope,
 * or return the condition it violates if it is not a reduction block
 * \param self The schedule state
 * \param block_sref The block to be checked
 * \param scope_root_sref The sref to the root block of the scope that `block_sref` is in
 * \return 0 if the block is a reduction block, or a positive integer indicating which condition is
 * first violated
 */
int CheckReductionBlockErrorCode(const ScheduleState& self, const StmtSRef& block_sref,
                                 const StmtSRef& scope_root_sref) {
  BlockScope scope = self->GetBlockScope(scope_root_sref);
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  // Cond 1. The block has the `init` statement.
  if (!block->init.defined()) {
    return 1;
  }
  // Cond 2. All the block bindings are quasi-affine expressions.
  if (!self->IsAffineBlockBinding(block_sref)) {
    return 2;
  }
  // Cond 3. All block vars are either data parallel block vars or reduction block vars. Meanwhile,
  // we collect all the reduction block vars.
  std::unordered_set<const VarNode*> reduction_block_vars;
  reduction_block_vars.reserve(block->iter_vars.size());
  for (const IterVar& iter_var : block->iter_vars) {
    if (iter_var->iter_type != kDataPar && iter_var->iter_type != kCommReduce) {
      return 3;
    } else if (iter_var->iter_type == kCommReduce) {
      reduction_block_vars.insert(iter_var->var.get());
    }
  }
  // Cond 4. Dominant: the block is the only writer of its output, dominating the reader of its
  // output buffers.
  if (!IsDominantBlock(scope, block_sref)) {
    return 4;
  }
  // Cond 5. The reduction block vars are not used to index the output buffers.
  std::unordered_set<const BufferNode*> buffer_written;
  buffer_written.reserve(block->writes.size());
  for (const BufferRegion& write_region : block->writes) {
    buffer_written.insert(write_region->buffer.get());
  }
  bool affected = false;
  PreOrderVisit(block->body, [&](const ObjectRef& obj) {
    if (affected) {
      return false;
    }
    if (const auto* store = obj.as<BufferStoreNode>()) {
      ICHECK(buffer_written.count(store->pointer->buffer.get()))
          << "ValueError: The buffer \"" << store->pointer->buffer
          << "\" is written in the block but is not in the block's signature";
      for (const PrimExpr& index : store->pointer->indices) {
        if (UsesVar(index, [&reduction_block_vars](const VarNode* var) {
              return reduction_block_vars.count(var);
            })) {
          affected = true;
          return false;
        }
      }
      return false;
    }
    return true;
  });
  return !affected ? 0 : 5;
}

bool IsReductionBlock(const ScheduleState& self, const StmtSRef& block_sref,
                      const StmtSRef& scope_root_sref) {
  return CheckReductionBlockErrorCode(self, block_sref, scope_root_sref) == 0;
}

void CheckReductionBlock(const ScheduleState& self, const StmtSRef& block_sref,
                         const StmtSRef& scope_root_sref) {
  class NotReductionBlockError : public ScheduleError {
   public:
    explicit NotReductionBlockError(IRModule mod, Block block, int violated_cond)
        : mod_(std::move(mod)), block_(std::move(block)), violated_cond_(violated_cond) {}
    String FastErrorString() const final { return "ScheduleError: Not a reduction block"; }
    String DetailRenderTemplate() const final {
      std::ostringstream os;
      os << "The block {0} is not a reduction block - it violates condition #" << violated_cond_;
      os << ".\n" << kReductionBlockDefinition;
      return os.str();
    }
    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }
    IRModule mod_;
    Block block_;
    int violated_cond_;
  };

  int error_code = CheckReductionBlockErrorCode(self, block_sref, scope_root_sref);
  if (error_code != 0) {
    const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
    throw NotReductionBlockError(self->mod, GetRef<Block>(block), error_code);
  }
}

void CheckCompleteOrReductionBlock(const ScheduleState& self, const StmtSRef& block_sref,
                                   const StmtSRef& scope_root_sref) {
  class NotCompleteOrReductionBlockError : public ScheduleError {
   public:
    explicit NotCompleteOrReductionBlockError(IRModule mod, Block block,
                                              int complete_block_error_code,
                                              int reduction_block_error_code)
        : mod_(mod),
          block_(block),
          complete_block_error_code_(complete_block_error_code),
          reduction_block_error_code_(reduction_block_error_code) {}

    String FastErrorString() const final {
      return "ScheduleError: Not a complete or reduction block";
    }
    String DetailRenderTemplate() const final {
      std::ostringstream os;
      os << "The block {0} is not a complete block - it violates condition #"
         << complete_block_error_code_;
      os << ".\n" << kCompleteBlockDefinition;
      os << "\nThe block is not a reduction block either - it violates condition #"
         << reduction_block_error_code_;
      os << ".\n" << kReductionBlockDefinition;
      return os.str();
    }
    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

    IRModule mod_;
    Block block_;
    int complete_block_error_code_;
    int reduction_block_error_code_;
  };

  int complete_block_error_code = CheckCompleteBlockErrorCode(self, block_sref, scope_root_sref);
  if (complete_block_error_code == 0) {
    return;
  }
  int reduction_block_error_code = CheckReductionBlockErrorCode(self, block_sref, scope_root_sref);
  if (reduction_block_error_code == 0) {
    return;
  }
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  throw NotCompleteOrReductionBlockError(self->mod, GetRef<Block>(block), complete_block_error_code,
                                         reduction_block_error_code);
}

bool IsOutputBlock(const ScheduleState& self, const StmtSRef& block_sref,
                   const StmtSRef& scope_root_sref) {
  const BlockNode* scope_root = TVM_SREF_TO_BLOCK(scope_root, scope_root_sref);
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  std::unordered_set<const BufferNode*> scope_allocated;
  scope_allocated.reserve(scope_root->alloc_buffers.size());
  for (const Buffer& buffer : scope_root->alloc_buffers) {
    scope_allocated.insert(buffer.get());
  }
  for (const BufferRegion& buffer_region : block->writes) {
    if (!scope_allocated.count(buffer_region->buffer.get())) {
      return true;
    }
  }
  return false;
}

void CheckNotOutputBlock(const ScheduleState& self, const StmtSRef& block_sref,
                         const StmtSRef& scope_root_sref) {
  class OutputBlockError : public ScheduleError {
   public:
    explicit OutputBlockError(IRModule mod, Block block) : mod_(mod), block_(block) {}
    String FastErrorString() const final {
      return "ScheduleError: Cannot operate on an output block";
    }
    String DetailRenderTemplate() const final { return "The block {0} is an output block"; }
    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

    IRModule mod_;
    Block block_;
  };
  if (IsOutputBlock(self, block_sref, scope_root_sref)) {
    const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
    throw OutputBlockError(self->mod, GetRef<Block>(block));
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

void CheckAffineBinding(const ScheduleState& self, Block block) {
  class NotAffineBindingError : public ScheduleError {
   public:
    explicit NotAffineBindingError(IRModule mod, Block block)
        : mod_(std::move(mod)), block_(std::move(block)) {}
    String FastErrorString() const final {
      return "ScheduleError: The block is required to have an affine binding";
    }
    String DetailRenderTemplate() const final {
      return "The block {0} is required to have an affine binding";
    }
    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }
    IRModule mod_;
    Block block_;
  };

  if (!self->IsAffineBlockBinding(self->stmt2ref.at(block.get()))) {
    throw NotAffineBindingError(self->mod, std::move(block));
  }
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

bool GetVarsTouchedByBlockIters(const BlockRealize& block_realize,
                                std::unordered_set<const VarNode*>* data_par_vars,
                                std::unordered_set<const VarNode*>* reduce_vars) {
  Block block = block_realize->block;
  ICHECK(block_realize->block.same_as(block))
      << "ValueError: The input `block_realize` is required to be the exact BlockRealize of the "
         "input block";

  bool has_block_vars_of_other_types = false;
  ICHECK_EQ(block->iter_vars.size(), block_realize->iter_values.size());
  int n = static_cast<int>(block->iter_vars.size());
  for (int i = 0; i < n; ++i) {
    const IterVar& iter_var = block->iter_vars[i];
    const PrimExpr& iter_value = block_realize->iter_values[i];
    std::unordered_set<const VarNode*>* set = nullptr;
    if (iter_var->iter_type == IterVarType::kDataPar) {
      set = data_par_vars;
    } else if (iter_var->iter_type == IterVarType::kCommReduce) {
      set = reduce_vars;
    } else {
      has_block_vars_of_other_types = true;
    }

    Array<Var> vars_in_binding = UndefinedVars(iter_value);
    for (const Var& var : vars_in_binding) {
      set->insert(var.get());
    }
  }

  return has_block_vars_of_other_types;
}

/******** Block-loop relation ********/

Array<StmtSRef> GetChildBlockSRefOnSRefTree(const ScheduleState& self,
                                            const StmtSRef& parent_sref) {
  Array<BlockRealize> child_block_realize = GetChildBlockRealizeOnSRefTree(parent_sref);
  Array<StmtSRef> child_block_srefs;
  child_block_srefs.reserve(child_block_realize.size());

  for (BlockRealize realize : child_block_realize) {
    child_block_srefs.push_back(self->stmt2ref.at(realize->block.get()));
  }
  return child_block_srefs;
}

Array<BlockRealize> GetChildBlockRealizeOnSRefTree(const StmtSRef& parent_sref) {
  struct Collector : public StmtVisitor {
    static Array<BlockRealize> Collect(const Stmt& stmt) {
      Collector collector;
      collector(stmt);
      return std::move(collector.result_);
    }

    void VisitStmt_(const BlockRealizeNode* block_realize) final {
      result_.push_back(GetRef<BlockRealize>(block_realize));
    }

    Array<BlockRealize> result_;
  };

  if (parent_sref->stmt->IsInstance<ForNode>()) {
    const auto* loop = static_cast<const ForNode*>(parent_sref->stmt);
    return Collector::Collect(loop->body);
  } else if (parent_sref->stmt->IsInstance<BlockNode>()) {
    const auto* block = static_cast<const BlockNode*>(parent_sref->stmt);
    return Collector::Collect(block->body);
  }
  ICHECK(false) << "Unreachable";
  throw;
}

BlockRealize CheckGetSingleChildBlockRealizeOnSRefTree(const ScheduleState& self,
                                                       const StmtSRef& parent_sref) {
  class NonSingleChildBlockError : public ScheduleError {
   public:
    explicit NonSingleChildBlockError(IRModule mod, const StmtSRef& sref)
        : mod_(std::move(mod)), stmt_(GetRef<Stmt>(sref->stmt)) {
      sref_type_ = stmt_.as<BlockNode>() != nullptr ? "block" : "loop";
    }

    String FastErrorString() const final {
      std::ostringstream os;
      os << "ScheduleError: The " << sref_type_ << " is required to have only one child block";
      return os.str();
    }

    String DetailRenderTemplate() const final {
      std::ostringstream os;
      os << "The " << sref_type_ << " {0} is required to have only one child block";
      return os.str();
    }

    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {stmt_}; }

    IRModule mod_;
    Stmt stmt_;
    String sref_type_;
  };

  Array<BlockRealize> child_block_realize = GetChildBlockRealizeOnSRefTree(parent_sref);
  if (child_block_realize.size() != 1) {
    throw NonSingleChildBlockError(self->mod, parent_sref);
  }
  return child_block_realize[0];
}

BlockRealize GetBlockRealize(const ScheduleState& self, const StmtSRef& block_sref) {
  struct BlockRealizeFinder : public StmtVisitor {
    explicit BlockRealizeFinder(const BlockNode* target_block)
        : target_block(target_block), result(nullptr) {}

    void VisitStmt(const Stmt& stmt) final {
      if (result != nullptr) {
        return;
      }
      StmtVisitor::VisitStmt(stmt);
    }

    void VisitStmt_(const BlockRealizeNode* block_realize) final {
      if (block_realize->block.get() == target_block) {
        result = block_realize;
      }
      // No need to visit recursively, since the deeper BlockRealizes must not be the result.
    }

    const BlockNode* target_block;
    const BlockRealizeNode* result;
  };

  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  if (block_sref->parent == nullptr) {
    const PrimFuncNode* func = GetRootPrimFunc(self->mod, block, nullptr);
    return Downcast<BlockRealize>(func->body);
  } else {
    BlockRealizeFinder finder(block);
    finder(GetRef<Stmt>(block_sref->parent->stmt));
    ICHECK(finder.result != nullptr)
        << "InternalError: Cannot find the BlockRealize of block " << GetRef<Block>(block);
    return GetRef<BlockRealize>(finder.result);
  }
}

/******** Producer-consumer relation ********/

Array<StmtSRef> GetProducers(const StmtSRef& block_sref, const BlockScope& scope) {
  Array<Dependency> deps = scope->GetDepsByDst(block_sref);
  Array<StmtSRef> result;
  result.reserve(deps.size());
  for (const Dependency& dep : deps) {
    result.push_back(dep->src);
  }
  return result;
}

Array<StmtSRef> GetConsumers(const StmtSRef& block_sref, const BlockScope& scope) {
  Array<Dependency> deps = scope->GetDepsBySrc(block_sref);
  Array<StmtSRef> result;
  result.reserve(deps.size());
  for (const Dependency& dep : deps) {
    result.push_back(dep->dst);
  }
  return result;
}

ProducerConsumerSplit ProducerConsumerSplit::Find(
    const ScheduleState& self, const Array<Stmt>& subtrees,
    const Array<StmtSRef>& producer_block_srefs, const Array<StmtSRef>& consumer_block_srefs,
    std::unordered_map<const BlockNode*, const BlockRealizeNode*>* block2realize) {
  class InsertionPointNotFoundError : public ScheduleError {
   public:
    explicit InsertionPointNotFoundError(IRModule mod, int last_producer_position,
                                         int first_consumer_position)
        : mod_(mod),
          last_producer_position_(last_producer_position),
          first_consumer_position_(first_consumer_position) {}

    String FastErrorString() const final {
      return "ScheduleError: Cannot find the insertion point that satisfies the producer-consumer "
             "constraint";
    }

    String DetailRenderTemplate() const final {
      return "Cannot find the insertion point that satisfies the producer-consumer constraint. In "
             "0-based indexing, the last producer appears in subtree " +
             std::to_string(last_producer_position_) +
             ", and the first consumer appears in subtree " +
             std::to_string(first_consumer_position_);
    }

    IRModule mod() const final { return mod_; }

    Array<ObjectRef> LocationsOfInterest() const final { return {}; }

   private:
    IRModule mod_;
    int last_producer_position_;
    int first_consumer_position_;
  };

  class Finder : public StmtVisitor {
   public:
    void VisitStmt_(const BlockRealizeNode* realize) final {
      const BlockNode* block = realize->block.get();
      if (block2realize_) {
        block2realize_->emplace(block, realize);
      }
      if (producer_blocks_.count(block)) {
        ++this->n_producers_visited_;
      }
      if (consumer_blocks_.count(block)) {
        ++this->n_consumers_visited_;
      }
    }

    std::unordered_map<const BlockNode*, const BlockRealizeNode*>* block2realize_;
    std::unordered_set<const StmtNode*> producer_blocks_;
    std::unordered_set<const StmtNode*> consumer_blocks_;
    int n_producers_visited_ = 0;
    int n_consumers_visited_ = 0;
  };

  Finder finder;
  finder.block2realize_ = block2realize;
  // Set up the lookup table for producers
  finder.producer_blocks_.reserve(producer_block_srefs.size());
  for (const StmtSRef& block_sref : producer_block_srefs) {
    finder.producer_blocks_.insert(block_sref->stmt);
  }
  // Set up the lookup table for consumers
  finder.consumer_blocks_.reserve(consumer_block_srefs.size());
  for (const StmtSRef& block_sref : consumer_block_srefs) {
    finder.consumer_blocks_.insert(block_sref->stmt);
  }
  // Visit the subtrees
  int n = subtrees.size();
  int last_producer_position = -1;
  int first_consumer_position = n;
  for (int i = 0; i < n; ++i) {
    int n_producers_visited_before = finder.n_producers_visited_;
    int n_consumers_visited_before = finder.n_consumers_visited_;
    finder(subtrees[i]);
    // Check if the subtree contains at least a producer
    if (finder.n_producers_visited_ != n_producers_visited_before) {
      last_producer_position = i;
    }
    // Check if the subtree contains at least a consumer
    if (finder.n_consumers_visited_ != n_consumers_visited_before) {
      if (first_consumer_position == n) {
        first_consumer_position = i;
      }
    }
  }
  if (last_producer_position >= first_consumer_position) {
    throw InsertionPointNotFoundError(self->mod, last_producer_position, first_consumer_position);
  }
  return ProducerConsumerSplit{last_producer_position,       //
                               first_consumer_position,      //
                               finder.n_producers_visited_,  //
                               finder.n_consumers_visited_};
}

/******** Block-buffer relation ********/

Buffer GetNthAccessBuffer(const ScheduleState& self, const Block& block, int n, bool is_write) {
  class BufferIndexOutOfRangeError : public ScheduleError {
   public:
    explicit BufferIndexOutOfRangeError(IRModule mod, Block block, int buffer_index, bool is_write)
        : mod_(std::move(mod)),
          block_(std::move(block)),
          buffer_index_(buffer_index),
          is_write_(is_write) {}

    String FastErrorString() const final {
      if (is_write_) {
        return "ScheduleError: The input `buffer_index` is out of range. It is required to be in "
               "range "
               "[0, num_write_regions) where `num_write_regions` is the number of buffer regions "
               "written by the block.";
      } else {
        return "ScheduleError: The input `buffer_index` is out of range. It is required to be in "
               "range "
               "[0, num_read_regions) where `num_read_regions` is the number of buffer regions "
               "read by the block.";
      }
    }

    String DetailRenderTemplate() const final {
      std::ostringstream os;
      size_t num = is_write_ ? block_->writes.size() : block_->reads.size();
      std::string access_type = is_write_ ? "write" : "read";
      os << "The block {0} has " << num << " " << access_type
         << " regions, so `buffer_index` is required to be in [0, " << num
         << "). However, the input `buffer_index` is " << buffer_index_
         << ", which is out of the expected range.";
      return os.str();
    }

    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

   private:
    IRModule mod_;
    Block block_;
    int buffer_index_;
    bool is_write_;
  };

  const Array<BufferRegion>& access_region = is_write ? block->writes : block->reads;

  if (n < 0 || static_cast<int>(access_region.size()) <= n) {
    throw BufferIndexOutOfRangeError(self->mod, block, n, is_write);
  }
  return access_region[n]->buffer;
}

/******** Pattern Matcher ********/

/*!
 * \brief PrimExpr pattern matcher.
 *
 * It is different from the pattern matcher in arith/pattern_match.h, which is dedicated
 * for compile-time constant patterns. This pattern matcher can work on dynamic user-specific
 * patterns.
 *
 * The code below shows how to use the pattern matcher.
 *
 * \code
 *
 * Var x("x"), y("y");
 * // use PrimExpr to declare patterns, x, y are holes that can be filled with
 * PatternMatcher pattern_matcher(x + y);
 * // expr = C[i, j] + A[i, k] * B[k, j], which is the expr we want to match
 * pattern_matcher.Match(expr);
 *
 * if (pattern_matcher.Success()) {
 *   pattern_matcher.Eval(x) // C[i, j]
 *   pattern_matcher.Eval(y) // A[i, k] * B[k, j]
 * }
 *
 * \endcode
 */
class PatternMatcher : public ExprVisitor {
 public:
  explicit PatternMatcher(PrimExpr pattern) : pattern_(std::move(pattern)) {}

  void VisitExpr_(const VarNode* op) final {
    auto it = filled_map_.find(op);
    if (it == filled_map_.end()) {
      filled_map_[op] = expr_to_match_;
    } else {
      ExprDeepEqual equal;
      if (it->second.same_as(expr_to_match_) || equal(it->second, expr_to_match_)) return;
      match_success_ = false;
    }
  }

  void VisitExpr_(const LoadNode* op) final {
    const auto* ptr = expr_to_match_.as<LoadNode>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      if (!op->buffer_var.same_as(ptr->buffer_var)) {
        match_success_ = false;
      } else {
        PrimExpr tmp = expr_to_match_;
        expr_to_match_ = ptr->predicate;
        VisitExpr(op->predicate);
        expr_to_match_ = ptr->index;
        VisitExpr(op->index);
        std::swap(expr_to_match_, tmp);
      }
    }
  }

  void VisitExpr_(const LetNode* op) final {
    const auto* ptr = expr_to_match_.as<LetNode>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      PrimExpr tmp = expr_to_match_;
      expr_to_match_ = ptr->var;
      VisitExpr(op->var);
      expr_to_match_ = ptr->value;
      VisitExpr(op->value);
      expr_to_match_ = ptr->body;
      VisitExpr(op->body);
      std::swap(expr_to_match_, tmp);
    }
  }

  void VisitExpr_(const CallNode* op) final {
    const auto* ptr = expr_to_match_.as<CallNode>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      if (!op->op.same_as(ptr->op)) {
        match_success_ = false;
      } else {
        PrimExpr tmp = expr_to_match_;
        for (size_t i = 0; i < op->args.size(); ++i) {
          expr_to_match_ = ptr->args[i];
          VisitExpr(op->args[i]);
        }
        std::swap(expr_to_match_, tmp);
      }
    }
  }

#define TVM_DECLARE_PATTERN_MATCHER_BIN_OP(OpName) \
  void VisitExpr_(const OpName* op) {              \
    const auto* ptr = expr_to_match_.as<OpName>(); \
    if (ptr == nullptr) {                          \
      match_success_ = false;                      \
    } else {                                       \
      PrimExpr current = expr_to_match_;           \
      expr_to_match_ = ptr->a;                     \
      VisitExpr(op->a);                            \
      expr_to_match_ = ptr->b;                     \
      VisitExpr(op->b);                            \
      std::swap(expr_to_match_, current);          \
    }                                              \
  }

  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(AddNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(SubNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(MulNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(DivNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(ModNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(FloorDivNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(FloorModNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(MinNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(MaxNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(EQNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(NENode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(LTNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(LENode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(GTNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(GENode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(AndNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(OrNode);

  void VisitExpr_(const CastNode* op) final {
    const auto* ptr = expr_to_match_.as<CastNode>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      if (!runtime::TypeEqual(op->dtype, ptr->dtype)) {
        match_success_ = false;
      } else {
        PrimExpr tmp = expr_to_match_;
        expr_to_match_ = ptr->value;
        VisitExpr(op->value);
        std::swap(expr_to_match_, tmp);
      }
    }
  }

  void VisitExpr_(const NotNode* op) final {
    const auto* ptr = expr_to_match_.as<NotNode>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      PrimExpr tmp = expr_to_match_;
      expr_to_match_ = ptr->a;
      VisitExpr(op->a);
      std::swap(expr_to_match_, tmp);
    }
  }

  void VisitExpr_(const SelectNode* op) final {
    const auto* ptr = expr_to_match_.as<SelectNode>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      PrimExpr tmp = expr_to_match_;
      expr_to_match_ = ptr->condition;
      VisitExpr(op->condition);
      expr_to_match_ = ptr->true_value;
      VisitExpr(op->true_value);
      expr_to_match_ = ptr->false_value;
      VisitExpr(op->false_value);
      std::swap(expr_to_match_, tmp);
    }
  }

  void VisitExpr_(const RampNode* op) final {
    const auto* ptr = expr_to_match_.as<RampNode>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      if (op->lanes != ptr->lanes) {
        match_success_ = false;
      } else {
        PrimExpr tmp = expr_to_match_;
        expr_to_match_ = ptr->base;
        VisitExpr(op->base);
        expr_to_match_ = ptr->stride;
        VisitExpr(op->stride);
        std::swap(expr_to_match_, tmp);
      }
    }
  }

  void VisitExpr_(const BroadcastNode* op) final {
    const auto* ptr = expr_to_match_.as<BroadcastNode>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      if (op->lanes != ptr->lanes) {
        match_success_ = false;
      } else {
        PrimExpr tmp = expr_to_match_;
        expr_to_match_ = ptr->value;
        VisitExpr(op->value);
        std::swap(expr_to_match_, tmp);
      }
    }
  }

  void VisitExpr_(const ShuffleNode* op) final {
    const auto* ptr = expr_to_match_.as<ShuffleNode>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      if (op->vectors.size() != ptr->vectors.size() || op->indices.size() != ptr->indices.size()) {
        match_success_ = false;
      } else {
        PrimExpr tmp = expr_to_match_;
        for (size_t i = 0; i < op->indices.size(); ++i) {
          expr_to_match_ = ptr->indices[i];
          VisitExpr(op->indices[i]);
        }
        for (size_t i = 0; i < op->vectors.size(); ++i) {
          expr_to_match_ = ptr->vectors[i];
          VisitExpr(op->vectors[i]);
        }
        std::swap(expr_to_match_, tmp);
      }
    }
  }

  void VisitExpr_(const IntImmNode* op) final {
    const auto* ptr = expr_to_match_.as<IntImmNode>();
    match_success_ = ptr != nullptr && op->value == ptr->value;
  }

  void VisitExpr_(const FloatImmNode* op) final {
    const auto* ptr = expr_to_match_.as<FloatImmNode>();
    match_success_ = ptr != nullptr && op->value == ptr->value;
  }

  void VisitExpr_(const StringImmNode* op) final {
    const auto* ptr = expr_to_match_.as<StringImmNode>();
    match_success_ = ptr != nullptr && op->value == ptr->value;
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    const auto* ptr = expr_to_match_.as<BufferLoadNode>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      if (!op->pointer->buffer.same_as(ptr->pointer->buffer) ||
          op->pointer->indices.size() != ptr->pointer->indices.size()) {
        match_success_ = false;
      } else {
        PrimExpr tmp = expr_to_match_;
        for (size_t i = 0; i < op->pointer->indices.size(); ++i) {
          expr_to_match_ = ptr->pointer->indices[i];
          VisitExpr(op->pointer->indices[i]);
        }
        std::swap(expr_to_match_, tmp);
      }
    }
  }

  void Match(const PrimExpr& expr_to_match) {
    this->match_success_ = true;
    this->filled_map_.clear();
    this->expr_to_match_ = expr_to_match;
    this->operator()(pattern_);
  }

  PrimExpr Eval(const Var& var) {
    auto it = filled_map_.find(var.operator->());
    ICHECK(it != filled_map_.end()) << "Unknown pattern variable";
    ICHECK(match_success_) << "Match failed";
    return it->second;
  }

  bool Success() const { return match_success_; }

 private:
  bool match_success_{true};
  PrimExpr pattern_, expr_to_match_;
  std::unordered_map<const VarNode*, PrimExpr> filled_map_;
};

/******** Commutative Reducer ********/

bool MatchReducer(const CommReducer& reducer, const PrimExpr& identity, const PrimExpr& combiner,
                  const BufferLoad& load, PrimExpr* lhs, PrimExpr* rhs) {
  if (!ExprDeepEqual()(reducer->identity_element[0], identity)) {
    return false;
  }
  PatternMatcher pattern_matcher(reducer->result[0]);
  pattern_matcher.Match(combiner);
  if (pattern_matcher.Success()) {
    PrimExpr lhs_tmp = pattern_matcher.Eval(reducer->lhs[0]);
    PrimExpr rhs_tmp = pattern_matcher.Eval(reducer->rhs[0]);
    if (ExprDeepEqual()(load, lhs_tmp)) {
      *lhs = std::move(lhs_tmp);
      *rhs = std::move(rhs_tmp);
    }
    return true;
  }
  return false;
}

bool FromIdentityCombiner(const PrimExpr& identity, const BufferStore& combiner,
                          CommReducer* result_reducer, PrimExpr* lhs, PrimExpr* rhs) {
  BufferLoad load(combiner->pointer->buffer, combiner->pointer->indices);
  // Check reduction patterns.
  for (const TypedPackedFunc<CommReducer(DataType)>& reducer_getter : GetReducerGetters()) {
    CommReducer reducer = reducer_getter(identity.dtype());
    if (MatchReducer(reducer, identity, combiner->value, load, lhs, rhs)) {
      *result_reducer = std::move(reducer);
      return true;
    }
  }
  return false;
}

/******** SRef Tree Related ********/

StmtSRef GetSRefTreeRoot(const StmtSRef& sref) {
  const StmtSRefNode* p = sref.get();
  for (; p->parent != nullptr; p = p->parent) {
  }
  return GetRef<StmtSRef>(p);
}

}  // namespace tir
}  // namespace tvm
