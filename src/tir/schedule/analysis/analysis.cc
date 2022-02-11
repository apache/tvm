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

StmtSRef GetScopeRoot(const ScheduleState& self, const StmtSRef& sref,
                      bool require_stage_pipeline) {
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
  return scope_root_sref;
}

ScopeBlockLoopInfo GetScopeBlockLoopInfo(const Block& scope_block) {
  struct Collector : public StmtVisitor {
    void VisitStmt_(const BlockRealizeNode* realize) final {
      result.realizes.push_back(GetRef<BlockRealize>(realize));
      const Array<IterVar>& iter_vars = realize->block->iter_vars;
      const Array<PrimExpr>& iter_values = realize->iter_values;
      ICHECK_EQ(iter_vars.size(), iter_values.size());
      int n = realize->iter_values.size();
      for (int i = 0; i < n; ++i) {
        const IterVar& iter_var = iter_vars[i];
        const PrimExpr& iter_value = iter_values[i];
        std::unordered_set<const VarNode*>* vars = nullptr;
        if (iter_var->iter_type == IterVarType::kDataPar) {
          vars = &result.spatial_vars;
        } else {
          vars = &result.non_spatial_vars;
        }
        PostOrderVisit(iter_value, [vars](const ObjectRef& obj) {
          if (const VarNode* var = obj.as<VarNode>()) {
            vars->insert(var);
          }
        });
      }
    }

    ScopeBlockLoopInfo result;
  } visitor;
  visitor(scope_block->body);
  return std::move(visitor.result);
}

/*!
 * \brief Check the dominant property of a block:
 * the block is the only writer of its output, dominating the reader of its output buffers
 * \param scope The block-scope of the block to be checked
 * \param block_sref The block whose dominant property is to be checked
 * \return A boolean indicating if the block is a dominant block
 */
bool IsDominantBlock(const BlockScope& scope, const StmtSRef& block_sref) {
  // Check whether the input block is the only writer of its outputs
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  const std::unordered_map<Buffer, Array<StmtSRef>, ObjectPtrHash, ObjectPtrEqual>& buffer_writers =
      scope->buffer_writers;
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
  if (!ContainsOnlyDataParAndReductionBlockIter(block->iter_vars)) {
    return 3;
  }
  // Cond 4. Dominant: the block is the only writer of its output, dominating the reader of its
  // output buffers.
  if (!IsDominantBlock(scope, block_sref)) {
    return 4;
  }
  // Cond 5. The reduction block vars are not used to index the output buffers.
  return ReductionIterNotIndexOutputBuffer(GetRef<Block>(block)) ? 0 : 5;
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

void CheckSubtreeCompactDataflow(const ScheduleState& self, const StmtSRef& subtree_root,
                                 const StmtSRef& scope_root_sref) {
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

  Array<StmtSRef> child_block_srefs = GetChildBlockSRefOnSRefTree(self, subtree_root);
  for (const StmtSRef& block_sref : child_block_srefs) {
    if (!IsCompleteBlock(self, block_sref, scope_root_sref) &&
        !IsReductionBlock(self, block_sref, scope_root_sref)) {
      const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
      throw NotCompactDataFlowError(self->mod, GetRef<Stmt>(subtree_root->stmt),
                                    GetRef<Block>(block));
    }
  }
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

std::vector<IterVarType> GetBlockVarTypes(const StmtSRef& block_sref) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  std::vector<IterVarType> results;
  results.reserve(block->iter_vars.size());
  for (const IterVar& iter_var : block->iter_vars) {
    results.push_back(iter_var->iter_type);
  }
  return results;
}

bool IsWriteCache(const StmtSRef& block_sref) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  if (block->writes.size() != 1) {
    return false;
  }
  const BufferRegion& write_region = block->writes[0];
  for (const BufferRegion& read_region : block->reads) {
    bool exists, surjective, injective, ordered, no_const_read, no_shift_read;
    std::tie(exists, surjective, injective, ordered, no_const_read, no_shift_read) =
        AnalyzeReadWritePattern(read_region, write_region);
    if (!(injective && ordered)) {
      return false;
    }
  }
  return true;
}

/******** Binding ********/

bool IsAffineBinding(const BlockRealize& realize, const Map<Var, Range>& loop_var_ranges,
                     arith::Analyzer* analyzer) {
  if (loop_var_ranges.empty()) {
    return true;
  }
  DiagnosticContext diag_ctx(DiagnosticContext::Default(IRModule()));
  Array<arith::IterSumExpr> results = arith::DetectIterMap(
      /*indices=*/realize->iter_values,
      /*input_iters=*/loop_var_ranges,
      /*predicate=*/realize->predicate,
      /*require_bijective=*/false,
      /*analyzer=*/analyzer,
      /*diag_ctx*/ diag_ctx);
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
          if (CanRelaxStorageUnderThread(extra_relax_scope,
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
    if (set == nullptr) {
      continue;
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

IterVarType GetLoopIterType(const StmtSRef& loop_sref) {
  const ForNode* loop = TVM_SREF_TO_FOR(loop, loop_sref);
  const Var& loop_var = loop->loop_var;
  int n_spatial = 0;
  int n_reduce = 0;
  int n_other = 0;
  auto f_visit = [&loop_var, &n_spatial, &n_reduce, &n_other](const ObjectRef& obj) -> bool {
    if (const auto* realize = obj.as<BlockRealizeNode>()) {
      const BlockNode* block = realize->block.get();
      // Number of block vars and their bindings
      ICHECK_EQ(realize->iter_values.size(), block->iter_vars.size());
      size_t n = realize->iter_values.size();
      for (size_t i = 0; i < n; ++i) {
        const IterVar& iter_var = block->iter_vars[i];
        const PrimExpr& binding = realize->iter_values[i];
        // Categorize the current block var
        int* ref = nullptr;
        if (iter_var->iter_type == IterVarType::kDataPar) {
          ref = &n_spatial;
        } else if (iter_var->iter_type == IterVarType::kCommReduce) {
          ref = &n_reduce;
        } else {
          ref = &n_other;
        }
        // Visit the binding to see if `loop_var` appears
        PostOrderVisit(binding, [&ref, &loop_var](const ObjectRef& obj) -> void {
          if (obj.same_as(loop_var)) {
            (*ref) += 1;
          }
        });
      }
      return false;
    }
    return true;
  };
  PreOrderVisit(loop->body, f_visit);
  if (n_other) {
    return IterVarType::kOpaque;
  } else if (n_spatial && n_reduce) {
    return IterVarType::kOpaque;
  } else if (n_reduce) {
    return IterVarType::kCommReduce;
  } else {
    return IterVarType::kDataPar;
  }
}

StmtSRef GetSRefLowestCommonAncestor(const Array<StmtSRef>& srefs) {
  CHECK(!srefs.empty()) << "ValueError: The input array is required to have at least one sref";

  std::unordered_map<const StmtSRefNode*, size_t> sref_visited_cnt;
  for (const StmtSRef& sref : srefs) {
    const StmtSRefNode* p = sref.get();
    while (p != nullptr) {
      ++sref_visited_cnt[p];
      p = p->parent;
    }
  }
  size_t n_sref = srefs.size();
  const StmtSRefNode* p = srefs[0].get();
  while (p != nullptr && sref_visited_cnt[p] != n_sref) {
    p = p->parent;
  }
  ICHECK(p != nullptr);
  return GetRef<StmtSRef>(p);
}

bool HasBeenMultiLevelTiled(const StmtSRef& block_sref) {
  return tir::GetAnn<String>(block_sref, tir::attr::meta_schedule_tiling_structure).defined();
}

std::pair<Array<StmtSRef>, std::vector<int>> CollectComputeLocation(const ScheduleState& self,
                                                                    const StmtSRef& block_sref) {
  Array<StmtSRef> location_srefs;
  std::vector<int> location_indices;

  // Step 1. Add the "compute-root" candidate. Add the "compute-inline" candidate if the block can
  // be inlined.
  if (CanComputeInline(self, block_sref)) {
    location_srefs.push_back(StmtSRef::InlineMark());
    location_indices.push_back(-2);
  }
  location_srefs.push_back(StmtSRef::RootMark());
  location_indices.push_back(-1);

  // Step 2. If the block has no consumer, there is no more candidate.
  Array<StmtSRef> consumers = GetConsumers(self, block_sref);
  if (consumers.empty()) {
    return std::make_pair(location_srefs, location_indices);
  }

  // Step 3. Get the deepest loop that the input block can be computed at (namely "boundary"). If
  // such a loop cannot be found, there is no more candidate and we just return.
  StmtSRef loop_boundary = consumers.size() > 1 ? GetSRefLowestCommonAncestor(consumers)
                                                : GetRef<StmtSRef>(consumers[0]->parent);
  if (loop_boundary->StmtAs<ForNode>() == nullptr) {
    return std::make_pair(location_srefs, location_indices);
  }

  // Step 4. Collect the loops outside the first consumer and locate the boundary loop. The position
  // of the boundary loop reveals the number of possible additional candidates.
  Array<StmtSRef> loop_srefs = GetLoops(consumers[0]);
  size_t lca_pos =
      std::find(loop_srefs.begin(), loop_srefs.end(), loop_boundary) - loop_srefs.begin();
  ICHECK_LT(lca_pos, loop_srefs.size());
  size_t n_candidate = lca_pos + 1;

  // Step 5. Find the position of the deepest data-parallel loop among the candidate loops. This
  // position is used for removing the unwanted candidates from the perspective of performance.
  std::vector<IterVarType> loop_iter_types;
  loop_iter_types.reserve(n_candidate);
  int i_last_datapar = -1;
  for (size_t i = 0; i < n_candidate; ++i) {
    // TODO(siyuan): improve the performance
    IterVarType iter_type = GetLoopIterType(loop_srefs[i]);
    loop_iter_types.push_back(iter_type);
    if (iter_type == IterVarType::kDataPar) {
      i_last_datapar = i;
    }
  }
  // Step 6. Check and add the candidates in turn according to the following rules:
  //  - skip the unit loops (loops with extent 1);
  //  - do not consider the data-parallel loops after a not-data-parallel loop;
  //  - do not consider the trailing not-data-parallel loops.
  location_srefs.reserve(n_candidate + 2);
  location_indices.reserve(n_candidate + 2);
  bool visited_reduce = false;
  for (size_t i = 0; i < n_candidate; ++i) {
    const int64_t* loop_extent = GetLoopIntExtent(loop_srefs[i]);
    if (loop_extent != nullptr && *loop_extent == 1) {
      continue;
    }

    if (loop_iter_types[i] == IterVarType::kDataPar) {
      if (visited_reduce) {
        break;
      }
    } else {
      visited_reduce = true;
      if (static_cast<int>(i) > i_last_datapar) {
        break;
      }
    }
    if (CanComputeAt(self, block_sref, loop_srefs[i], true)) {
      location_srefs.push_back(loop_srefs[i]);
      location_indices.push_back(i);
    }
  }

  return std::make_pair(location_srefs, location_indices);
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
      if (!op->buffer.same_as(ptr->buffer) || op->indices.size() != ptr->indices.size()) {
        match_success_ = false;
      } else {
        PrimExpr tmp = expr_to_match_;
        for (size_t i = 0; i < op->indices.size(); ++i) {
          expr_to_match_ = ptr->indices[i];
          VisitExpr(op->indices[i]);
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

/******** Reduction Block Related ********/

class InitBodyNotBufferStoreError : public ScheduleError {
 public:
  explicit InitBodyNotBufferStoreError(IRModule mod, Block block, bool init_is_bufferstore,
                                       bool body_is_bufferstore)
      : mod_(std::move(mod)),
        block_(std::move(block)),
        init_is_bufferstore_(init_is_bufferstore),
        body_is_bufferstore_(body_is_bufferstore) {}

  String FastErrorString() const final {
    return "ScheduleError: The `init` and `body` of reduction block are required to be both "
           "BufferStore so that rfactor or cross-thread reduction can be applied";
  }

  String DetailRenderTemplate() const final {
    if (!init_is_bufferstore_ && !body_is_bufferstore_) {
      return "The `init` and `body` of block {0} are required to be BufferStore so that rfactor or "
             "cross-thread reduction can be applied";
    } else if (!init_is_bufferstore_) {
      return "The `init` of block {0} is required to be BufferStore so that rfactor or cross-thread"
             " reduction can be applied";
    } else {
      ICHECK(!body_is_bufferstore_);
      return "The `body` of block {0} is required to be BufferStore so that rfactor or cross-thread"
             " reduction can be applied";
    }
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

  IRModule mod_;
  Block block_;
  bool init_is_bufferstore_;
  bool body_is_bufferstore_;
};

class InitBodyNotSameBufferAccessError : public ScheduleError {
 public:
  explicit InitBodyNotSameBufferAccessError(IRModule mod, Block block)
      : mod_(std::move(mod)), block_(std::move(block)) {}

  String FastErrorString() const final {
    return "ScheduleError: The `init` and `body` of the reduction block are required to have the "
           "same buffer access pattern";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    const auto* init = block_->init.as<BufferStoreNode>();
    const auto* update = block_->body.as<BufferStoreNode>();
    os << "The `init` and `body` of the block {0} is required to have the same buffer access "
          "pattern. However, in block {0} the `init` writes to "
       << init->buffer->name << init->indices << ", and the `body` writes to "
       << update->buffer->name << update->indices;
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

  IRModule mod_;
  Block block_;
};

std::pair<BufferStore, BufferStore> GetBufferStoresFromReductionBlock(
    const Optional<ScheduleState>& self, const Block& block) {
  static constexpr const char* error_str1 =
      "ValueError: The `init` and `body` of the reduction block are required to be both "
      "BufferStore so that rfactor or cross-thread reduction can be applied. However, a reduction "
      "block that doesn't meet this requirement is ";
  static constexpr const char* error_str2 =
      "ValueError: The `init` and `body` of the reduction block are required to have the same "
      "buffer access pattern so that rfactor or cross-thread reduction can be applied. However, a "
      "reduction block that doesn't meet this requirement is ";

  const auto* init = block->init.as<BufferStoreNode>();
  const auto* body = block->body.as<BufferStoreNode>();
  if (!(init && body)) {
    if (self.defined()) {
      throw InitBodyNotBufferStoreError(self.value()->mod, block, init != nullptr, body != nullptr);
    } else {
      LOG(FATAL) << error_str1 << block;
    }
  }
  if (!init->buffer.same_as(body->buffer)) {
    if (self.defined()) {
      throw InitBodyNotSameBufferAccessError(self.value()->mod, block);
    } else {
      LOG(FATAL) << error_str2 << block;
    }
  }
  int ndim = static_cast<int>(init->buffer->shape.size());
  for (int i = 0; i < ndim; ++i) {
    if (!ExprDeepEqual()(init->indices[i], body->indices[i])) {
      if (self.defined()) {
        throw InitBodyNotSameBufferAccessError(self.value()->mod, block);
      } else {
        LOG(FATAL) << error_str2 << block;
      }
    }
  }
  return std::make_pair(GetRef<BufferStore>(init), GetRef<BufferStore>(body));
}

bool ContainsOnlyDataParAndReductionBlockIter(const Array<IterVar>& iters) {
  for (const IterVar& iter_var : iters) {
    if (iter_var->iter_type != kDataPar && iter_var->iter_type != kCommReduce) {
      return false;
    }
  }
  return true;
}

bool ReductionIterNotIndexOutputBuffer(const Block& block) {
  // Step 1. Collect the reduction block iters.
  std::unordered_set<const VarNode*> reduction_block_iters;
  reduction_block_iters.reserve(block->iter_vars.size());
  for (const IterVar& iter_var : block->iter_vars) {
    if (iter_var->iter_type == kCommReduce) {
      reduction_block_iters.insert(iter_var->var.get());
    }
  }
  // Step 2. Check if the reduction block iters are used to index the output buffer.
  std::unordered_set<const BufferNode*> buffer_written;
  buffer_written.reserve(block->writes.size());
  for (const BufferRegion& write_region : block->writes) {
    buffer_written.insert(write_region->buffer.get());
  }
  auto f_uses_reduction_block_var = [&](const PrimExpr& expr) -> bool {
    return UsesVar(expr, [&](const VarNode* var) {  //
      return reduction_block_iters.count(var);
    });
  };
  bool affected = false;
  PreOrderVisit(block->body, [&](const ObjectRef& obj) {
    if (affected) {
      return false;
    }
    const auto* store = obj.as<BufferStoreNode>();
    if (!store) {
      return true;
    }
    ICHECK(buffer_written.count(store->buffer.get()))
        << "ValueError: The buffer \"" << store->buffer
        << "\" is written in the block but is not in the block's signature";
    for (const PrimExpr& index : store->indices) {
      if (f_uses_reduction_block_var(index)) {
        affected = true;
        return false;
      }
    }
    return false;
  });
  return !affected;
}

class NoMatchedReducerError : public ScheduleError {
 public:
  explicit NoMatchedReducerError(IRModule mod, PrimExpr identity, BufferStore combiner)
      : mod_(std::move(mod)), identity_(std::move(identity)), combiner_(std::move(combiner)) {}

  String FastErrorString() const final {
    return "ScheduleError: No matched reducer for the identity and the combiner of this reduction "
           "block. So rfactor and cross-thread reduction cannot be applied.";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "No matched reducer for identity " << identity_ << " and combiner " << combiner_
       << "In this case rfactor cannot be applied. You can check tvm::tir::ReducerRegistry for "
          "default reducers or registering new reducers.";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {}; }

  IRModule mod_;
  PrimExpr identity_;
  BufferStore combiner_;
};

std::tuple<CommReducer, PrimExpr, PrimExpr> GetReducerAndCombinerLhsRhs(
    const Optional<ScheduleState>& self, const PrimExpr& identity, const BufferStore& combiner) {
  CommReducer reducer{nullptr};
  PrimExpr combiner_lhs{nullptr}, combiner_rhs{nullptr};
  bool matched = FromIdentityCombiner(identity, combiner, &reducer, &combiner_lhs, &combiner_rhs);
  if (!matched) {
    if (self.defined()) {
      throw NoMatchedReducerError(self.value()->mod, identity, combiner);
    } else {
      LOG(FATAL) << "ValueError: No matched reducer for the identity and the combiner of the "
                    "reduction block. So rfactor and cross-thread reduction cannot be applied.";
    }
  }
  return std::make_tuple(std::move(reducer), std::move(combiner_lhs), std::move(combiner_rhs));
}

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
  BufferLoad load(combiner->buffer, combiner->indices);
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

/******** Misc ********/

bool HasOp(const Stmt& stmt, const Array<Op>& ops) {
  std::unordered_set<const Object*> op_set;
  op_set.reserve(ops.size());
  for (const Op& op : ops) {
    op_set.insert(op.operator->());
  }
  bool found = false;
  PreOrderVisit(stmt, [&found, &op_set](const ObjectRef& obj) -> bool {
    if (found) {
      return false;
    }
    if (const auto* call = obj.as<CallNode>()) {
      if (op_set.count(call->op.operator->())) {
        found = true;
      }
    }
    return !found;
  });
  return found;
}

bool HasIfThenElse(const Stmt& stmt) {
  bool has_branch = false;
  auto f_visit = [&has_branch](const ObjectRef& obj) -> bool {
    if (has_branch) {
      // stop visiting
      return false;
    }
    if (const auto* realize = obj.as<BlockRealizeNode>()) {
      // Case 1: BlockRealize
      if (!is_one(realize->predicate)) {
        has_branch = true;
      }
    } else if (obj->IsInstance<IfThenElseNode>() || obj->IsInstance<SelectNode>()) {
      // Case 2: IfThenElse / Select
      has_branch = true;
    } else if (const auto* call = obj.as<CallNode>()) {
      // Case 3: Call the `if_then_else` operator
      static const Op& op_if_then_else = Op::Get("tir.if_then_else");
      if (call->op.same_as(op_if_then_else)) {
        has_branch = true;
      }
    }
    return !has_branch;
  };
  PreOrderVisit(stmt, f_visit);
  return has_branch;
}

std::tuple</*exists=*/bool,
           /*surjective=*/bool,
           /*injective=*/bool,
           /*ordered=*/bool,
           /*no_const_read=*/bool,
           /*no_shift_read=*/bool>
AnalyzeReadWritePattern(const BufferRegion& read_region, const BufferRegion& write_region) {
  static constexpr const std::tuple<bool, bool, bool, bool, bool, bool> kNotExist =
      std::make_tuple(false, false, false, false, false, false);
  // Step 1. Extract the write indices
  int w_dim = write_region->buffer->shape.size();
  std::unordered_map<const VarNode*, int> var2idx;
  var2idx.reserve(w_dim);
  for (int i = 0; i < w_dim; ++i) {
    const Range& dom = write_region->region[i];
    if (as_const_int(dom->extent) == nullptr) {
      return kNotExist;
    }
    if (const auto* v = dom->min.as<VarNode>()) {
      var2idx.emplace(v, i);
    } else {
      return kNotExist;
    }
  }
  // Step 2. Map each read index to a write index
  bool no_const_read = true;
  bool no_shift_read = true;
  int r_dim = read_region->buffer->shape.size();
  std::vector<int> mapped(r_dim, -1);
  for (int i = 0; i < r_dim; ++i) {
    const Range& dom = read_region->region[i];
    if (as_const_int(dom->extent) == nullptr) {
      return kNotExist;
    }
    // Case 1. Read index is a constant
    if (as_const_int(dom->min) != nullptr) {
      no_const_read = false;
      continue;
    }
    // Case 2. Read index cannot be recognized as `var +/- const`
    // where `var` is a write index and `const` is an optional constant shift
    Optional<IntImm> opt_const = NullOpt;
    const VarNode* var =
        static_cast<const VarNode*>(AnalyzeVarWithShift(dom->min, &opt_const).get());
    if (var == nullptr || !var2idx.count(var)) {
      return kNotExist;
    }
    // Case 3. Read index is `var +/- const`
    mapped[i] = var2idx.at(var);
    if (opt_const.defined()) {
      no_shift_read = false;
    }
  }
  // Step 3. Check if the mapping is ordered, and count how many times each var is mapped
  std::vector<int> mapped_counter(w_dim, 0);
  bool ordered = true;
  int last_mapped = -1;
  for (int i : mapped) {
    if (i != -1) {
      ++mapped_counter[i];
      if (last_mapped != -1 && last_mapped > i) {
        ordered = false;
      }
      last_mapped = i;
    }
  }
  // Step 4. Check if the mapping is surjective or injective
  // Surjective: each write index is mapped at least once
  // Injective: each write index is mapped at most once
  bool surjective = true;
  bool injective = true;
  for (int cnt : mapped_counter) {
    if (cnt == 0) {
      surjective = false;
    } else if (cnt >= 2) {
      injective = false;
    }
  }
  return std::make_tuple(/*exist=*/true, surjective, injective, ordered, no_const_read,
                         no_shift_read);
}

/******** Storage Scope ********/

void CheckStorageScope(const ScheduleState& self, String storage_scope) {
  class InvalidStorageScopeError : public ScheduleError {
   public:
    explicit InvalidStorageScopeError(IRModule mod, String storage_scope)
        : mod_(std::move(mod)), storage_scope_(std::move(storage_scope)) {}

    String FastErrorString() const final {
      return "ScheduleError: The input storage scope is invalid";
    }

    String DetailRenderTemplate() const final {
      return "The input storage scope \"" + storage_scope_ + "\" is invalid.";
    }

    Array<ObjectRef> LocationsOfInterest() const final { return {}; }
    IRModule mod() const final { return mod_; }

   private:
    IRModule mod_;
    String storage_scope_;
  };

  try {
    runtime::StorageScope::Create(std::string(storage_scope));
  } catch (...) {
    throw InvalidStorageScopeError(self->mod, std::move(storage_scope));
  }
}

bool IsSpatial(const StmtSRef& block_sref) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  for (const IterVar& iter_var : block->iter_vars) {
    if (iter_var->iter_type != IterVarType::kDataPar) {
      return false;
    }
  }
  return true;
}

bool IsTrivialBinding(const ScheduleState& self, const StmtSRef& block_sref) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  Array<StmtSRef> loops = GetLoops(block_sref);
  Array<PrimExpr> binds = GetBlockRealize(self, block_sref)->iter_values;
  if (loops.size() != binds.size()) {
    return false;
  }
  for (int i = 0, n = loops.size(); i < n; ++i) {
    const ForNode* loop = TVM_SREF_TO_FOR(loop, loops[i]);
    if (binds[i].get() != loop->loop_var.get()) {
      return false;
    }
  }
  return true;
}

bool NeedsMultiLevelTiling(const ScheduleState& self, const StmtSRef& block_sref) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  if (block->writes.size() != 1 || block->reads.empty() || IsSpatial(block_sref) ||
      !IsTrivialBinding(self, block_sref)) {
    return false;
  }
  const BufferNode* write_buffer = block->writes[0]->buffer.get();
  // Step 1. Sort out spatial block variables
  std::vector<const VarNode*> spatial_block_vars;
  spatial_block_vars.reserve(block->iter_vars.size());
  for (const IterVar& block_var : block->iter_vars) {
    if (block_var->iter_type == IterVarType::kDataPar) {
      spatial_block_vars.push_back(block_var->var.get());
    }
  }
  // Step 2. Enumerate each read region, check the number of block vars that are not used
  // to index the read region
  int total_unused_block_vars = 0;
  std::unordered_set<const BufferNode*> read_buffers;
  read_buffers.reserve(block->reads.size());
  for (const BufferRegion& buffer_region : block->reads) {
    const BufferNode* buffer = buffer_region->buffer.get();
    const Array<Range>& regions = buffer_region->region;
    // Step 2.1. Duplication of read buffers are not allowed
    if (read_buffers.insert(buffer).second == false) {
      return false;
    }
    // Step 2.2. Skip the reduction buffer
    if (buffer == write_buffer) {
      continue;
    }
    // Step 2.3. Collect the block vars that are used to index the read region
    std::unordered_set<const VarNode*> vars;
    for (const Range& range : regions) {
      if (as_const_int(range->extent) == nullptr) {
        return false;
      }
      for (const Var& var : UndefinedVars(range->min)) {
        vars.insert(var.get());
      }
    }
    // Step 2.4. Check if the block vars are not used to index the read region
    int n_unused_block_vars = 0;
    for (const VarNode* block_var : spatial_block_vars) {
      if (vars.count(block_var) == 0) {
        ++n_unused_block_vars;
      }
    }
    total_unused_block_vars += n_unused_block_vars;
  }
  return total_unused_block_vars >= 1;
}

std::pair<int64_t, int64_t> GetCumulativeSpaceAndReductionLength(const tir::ScheduleState& self,
                                                                 const tir::StmtSRef& block_sref) {
  Array<tir::StmtSRef> loops = tir::GetLoops(block_sref);
  int64_t cum_space_len = 1, cum_reduce_len = 1;
  /*
   * Return (-1, -1) if
   *   1. there is some loop with type other than kDataPar and kCommReduce;
   *   2. there is some loop which is dynamic.
   */
  for (const tir::StmtSRef& loop_sref : loops) {
    tir::IterVarType type = GetLoopIterType(loop_sref);
    if (type == tir::kDataPar) {
      const int64_t* extent = GetLoopIntExtent(loop_sref);
      if (*extent != -1) {
        cum_space_len *= *extent;
      } else {
        return std::make_pair(-1, -1);
      }
    } else if (type == tir::kCommReduce) {
      const int64_t* extent = GetLoopIntExtent(loop_sref);
      if (*extent != -1) {
        cum_reduce_len *= *extent;
      } else {
        return std::make_pair(-1, -1);
      }
    } else {
      return std::make_pair(-1, -1);
    }
  }
  return std::make_pair(cum_space_len, cum_reduce_len);
}

bool NeedsRFactorOrCrossThreadReduction(const tir::ScheduleState& self,   //
                                        const tir::StmtSRef& block_sref,  //
                                        int64_t max_parallel_extent,      //
                                        int64_t max_parallel_basic) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  Array<tir::StmtSRef> loops = tir::GetLoops(block_sref);

  // Cond 1. The block has only one write buffer
  if (block->writes.size() != 1) {
    return false;
  }

  // Cond 2. The block is a reduction block and has trivial binding.
  const StmtSRef& scope_sref = GetScopeRoot(self, block_sref,
                                            /*require_stage_pipeline=*/false);
  if (!IsReductionBlock(self, block_sref, scope_sref)  //
      || !IsTrivialBinding(self, block_sref)           //
      || HasBeenMultiLevelTiled(block_sref)) {
    return false;
  }

  // Cond 3. Every the loop axis must be either spatial axis or reduction axis.
  for (const tir::StmtSRef& loop_sref : loops) {
    const tir::IterVarType& type = GetLoopIterType(loop_sref);
    if (type != tir::kDataPar && type != tir::kCommReduce) {
      return false;
    }
  }

  // Cond 4. Whether there is at least one reduction loop.
  // Cond 5. The loops are continuous, and the body of the innermost loop is exactly the block.
  bool has_reduction_loop = false;
  for (size_t i = 0; i < loops.size(); ++i) {
    // Cond 4.
    if (GetLoopIterType(loops[i]) == tir::kCommReduce) {
      has_reduction_loop = true;
    }

    // Cond 5.
    const ForNode* loop_i = TVM_SREF_TO_FOR(loop_i, loops[i]);
    if (i < loops.size() - 1) {
      const ForNode* loop_i1 = TVM_SREF_TO_FOR(loop_i1, loops[i + 1]);
      if (loop_i->body.get() != loop_i1) {
        return false;
      }
    } else {
      const auto* block_realize = loop_i->body.as<tir::BlockRealizeNode>();
      if (!block_realize || block_realize->block.get() != block) {
        return false;
      }
    }
  }
  if (!has_reduction_loop) {
    return false;
  }

  // Cond 6. Can successfully calculating the cumulative loop length.
  int64_t cum_space_len, cum_reduce_len;
  std::tie(cum_space_len, cum_reduce_len) = GetCumulativeSpaceAndReductionLength(self, block_sref);
  if (cum_space_len == -1 || cum_reduce_len == -1) {
    return false;
  }

  // Cond 7.
  if (NeedsMultiLevelTiling(self, block_sref)) {
    // Do not use rfactor/cross-thread-reduction if we have enough parallelism on spatial loops.
    return !(cum_space_len >= cum_reduce_len || cum_space_len > max_parallel_extent);
  } else if (cum_reduce_len > 1) {
    // Always try rfactor/cross-thread-reduction for other reduction blocks.
    return cum_reduce_len > max_parallel_basic;
  } else {
    return false;
  }
}

}  // namespace tir
}  // namespace tvm
