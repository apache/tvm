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
#include <tvm/arith/int_set.h>

#include "../../transforms/common_subexpr_elim_tools.h"
#include "../../transforms/replace_selected_expr.h"
#include "../utils.h"

namespace tvm {
namespace tir {

/******** Helper Functions/Classes ********/

/*! \brief The auxiliary info used for the insertion point and content of the cache stage. */
struct IndexInfo {
  /*! \brief The target block to perform cache_index */
  StmtSRef target_block;
  /*! \brief Record the common subexpr extract threshold */
  size_t cse_thresh;
  /*! \brief The cache buffer to store the precomputed index */
  std::vector<Buffer> cache_buffer;
  /*! \brief The expr to be precomputed */
  std::vector<PrimExpr> index_exprs;
  /*! \brief The range of the loop vars relating to index computation */
  Map<Var, Range> range_map;
  /*! \brief The binding table of the block var and the loop var */
  Map<Var, PrimExpr> var_binding;
  /*! \brief The block var of the target block */
  std::vector<Array<Var>> origin_block_vars;
  /*! \brief The index to insert the cache stage. */
  size_t loc_pos;
  /*! \brief The cache stage to be inserted. */
  Stmt cache_stage;
  /*! \brief The map used for ScheduleStateNode::Replace. */
  Map<Block, Block> block_reuse;
};

/*!
 * \brief Determine the data type base on the integer range.
 * \param range The range of the integer.
 * \returns A data type that covers the input range.
 */
DataType DetermineDatatype(const arith::IntSet& range) {
  arith::Analyzer ana;
  if (ana.CanProve(range.min() >= INT32_MIN && range.max() <= INT32_MAX)) {
    return DataType::Int(32);
  } else {
    ICHECK(ana.CanProve(range.min() >= make_const(DataType::Int(64), INT64_MIN) &&
                        range.max() <= make_const(DataType::Int(64), INT64_MAX)));
    return DataType::Int(64);
  }
}

/*! \brief Collect the index info to be cached */
class IndexInfoCollector : public StmtExprVisitor {
 public:
  /*!
   * \brief Collect the index info for cache_index and write into the IndexInfo
   * \param self The state of the schedule \param block_sref The sref of the target
   * block of the target buffer being applied cache_index \param scope_sref The sref
   * of the scope block of the target block \param info The index info.
   */
  static void Collect(const ScheduleState& self, const StmtSRef& block_sref,
                      const StmtSRef& scope_sref, IndexInfo* info) {
    IndexInfoCollector collector(self, block_sref, scope_sref, info->cse_thresh);
    collector(GetRef<Stmt>(scope_sref->stmt));
    info->loc_pos = collector.loc_pos_;
    info->index_exprs = collector.exprs_;
    info->range_map = collector.range_map_;
  }

 private:
  /*!
   * \brief Constructor
   * \param self The state of the schedule
   * \param block_sref The sref of the target block of the buffer being applied cache_index
   * \param scope_sref The sref of the scope block of the target block
   * \param cse_thresh The repeat threshold that determines a common subexpr
   */
  IndexInfoCollector(const ScheduleState self, const StmtSRef& block_sref,
                     const StmtSRef& scope_sref, int cse_thresh)
      : self_(self), block_sref_(block_sref), scope_sref_(scope_sref), cse_thresh_(cse_thresh) {}

  void VisitStmt_(const SeqStmtNode* seq_stmt) final {
    for (size_t i = 0; i < seq_stmt->size(); ++i) {
      if (loc_pos_ != -1) {
        break;
      }
      VisitStmt(seq_stmt->seq[i]);
      // `pos` can be assigned only once when we visited `block_sref`
      if (visited_block_ && loc_pos_ == -1 && update_seq_pos_) {
        // The offset of insert position from the block
        loc_pos_ = i;
        return;
      }
    }
  }

  void VisitStmt_(const BlockNode* block) final {
    visiting_target_block = static_cast<bool>(block_sref_->stmt == block);
    StmtVisitor::VisitStmt_(block);
    visiting_target_block = false;
    if (block == scope_sref_->stmt) {
      // The block vistied is the current parent scope
      // Handling cases when no SeqStmt in the scope
      if (visited_block_ && loc_pos_ == -1) {
        loc_pos_ = 0;
      }
    } else if (block_sref_->stmt == block) {
      visited_block_ = true;
    }
    // Update seq pos only at top scope
    if (visited_block_ && self_->stmt2ref.at(block)->parent == scope_sref_.get()) {
      update_seq_pos_ = true;
    }
  }

  void VisitStmt_(const ForNode* loop) final {
    range_map_.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
    StmtVisitor::VisitStmt_(loop);
    // Update seq pos only at top scope
    if (visited_block_ && self_->stmt2ref.at(loop)->parent == scope_sref_.get()) {
      update_seq_pos_ = true;
    }
  }

  void VisitStmt_(const BufferStoreNode* store) final {
    // Only analyze the cache candidate for stores in target block
    if (visiting_target_block) {
      auto IsEligibleComputation = [](const PrimExpr& expr) {
        return (SideEffect(expr) <= CallEffectKind::kPure && CalculateExprComplexity(expr) > 1 &&
                (expr.as<RampNode>() == nullptr) && (expr.as<BroadcastNode>() == nullptr));
      };

      // Analyze sub expr candidates
      ComputationTable table_syntactic_comp_done_by_stmt =
          ComputationsDoneBy::GetComputationsDoneBy(GetRef<Stmt>(store), IsEligibleComputation,
                                                    [](const PrimExpr& expr) { return true; });
      std::vector<std::pair<PrimExpr, size_t>> semantic_comp_done_by_stmt =
          SyntacticToSemanticComputations(table_syntactic_comp_done_by_stmt, true);

      // Analyze the sub expr of a candidate whose repeat time is under cse_thresh_
      for (size_t i = 0; i < semantic_comp_done_by_stmt.size(); i++) {
        std::pair<PrimExpr, size_t>& computation_and_nb = semantic_comp_done_by_stmt[i];
        if (computation_and_nb.second < cse_thresh_) {
          std::vector<PrimExpr> direct_subexprs = DirectSubexpr::GetDirectSubexpressions(
              computation_and_nb.first, IsEligibleComputation,
              [](const PrimExpr& expr) { return true; });
          InsertVectorToSortedSemanticComputations(&semantic_comp_done_by_stmt, direct_subexprs,
                                                   true, computation_and_nb.second);
        }
      }

      // Record the final sub expr with repeat time greater than cse_thresh_
      // In order to make the result stable, sort it by post order and then by complexity
      PostOrderVisit(store->value, [&semantic_comp_done_by_stmt, this](const ObjectRef& node) {
        if (node->IsInstance<PrimExprNode>()) {
          PrimExpr this_expr = Downcast<PrimExpr>(node);
          for (auto& it : semantic_comp_done_by_stmt) {
            if (it.second >= this->cse_thresh_ && EquivalentTerms(this_expr, it.first, true)) {
              auto find_result =
                  std::find_if(this->exprs_.begin(), this->exprs_.end(),
                               [&](PrimExpr expr) { return expr.get() == it.first.get(); });
              if (find_result == this->exprs_.end()) {
                this->exprs_.push_back(it.first);
              }
            }
          }
        }
      });
      auto cmp = [&](const PrimExpr& lhs, const PrimExpr& rhs) -> bool {
        return CalculateExprComplexity(lhs) > CalculateExprComplexity(rhs);
      };
      std::stable_sort(exprs_.begin(), exprs_.end(), cmp);
    }
    StmtVisitor::VisitStmt_(store);
  }

  /*! \brief The schedule class */
  const ScheduleState self_;
  /*! \brief The target block that read the target buffer */
  const StmtSRef& block_sref_;
  /*! \brief The parent scope of the target block */
  const StmtSRef& scope_sref_;
  /*! \brief Record the common subexpr extract threshold */
  size_t cse_thresh_;
  /*! \brief The calculation expr to be precomputed */
  std::vector<PrimExpr> exprs_;
  /*! \brief The flag whether we have visited the target block */
  bool visited_block_{false};
  /*! \brief The flag indicating currently visiting target block */
  bool visiting_target_block{false};
  /*! \brief The index to insert the cache_index stage */
  int loc_pos_{-1};
  /*! \brief The flag indicating the right scope to update seq pos */
  bool update_seq_pos_{false};
  /*! \brief Record the ranges of iter vars */
  Map<Var, Range> range_map_;
};

/*!
 * \brief Create a loop nest that writes precomputed index into index buffer.
 * \param info The cache stage information, which will be updated in the function.
 * \param storage_scope The storage scope of the cached buffer (only used in naming here)
 * \returns A block indicating the body of the loop nesting.
 */
Array<Block> MakeIndexCacheStage(IndexInfo* info, const String& storage_scope) {
  Array<Block> blocks;
  Array<Stmt> bodies;
  bodies.reserve(info->index_exprs.size());
  info->cache_buffer.reserve(info->index_exprs.size());

  // For each index calculation, create a block to pre-compute.
  for (size_t expr_index = 0; expr_index < info->index_exprs.size(); expr_index++) {
    const PrimExpr& index_expr = info->index_exprs[expr_index];

    // Collect the block vars in original index computation
    info->origin_block_vars.push_back({});
    PostOrderVisit(index_expr, [&info, &expr_index](const ObjectRef& node) {
      if (node->IsInstance<VarNode>()) {
        Var iter_var = Downcast<Var>(node);
        const Array<Var>& origin_block_var = info->origin_block_vars[expr_index];
        auto find_result = std::find_if(origin_block_var.begin(), origin_block_var.end(),
                                        [&](Var it) { return it.get() == iter_var.get(); });
        if (find_result == origin_block_var.end()) {
          info->origin_block_vars[expr_index].push_back(iter_var);
        }
      }
    });

    // Collect the loop vars corresponding to collected block vars,
    // which will be used to create new loop vars
    std::vector<Var> iter_vars;
    for (const Var& it : info->origin_block_vars[expr_index]) {
      PostOrderVisit(info->var_binding.at(it), [/*&info,*/ &iter_vars](const ObjectRef& node) {
        if (node->IsInstance<VarNode>()) {
          Var iter_var = Downcast<Var>(node);
          if (std::find_if(iter_vars.begin(), iter_vars.end(),
                           [&](Var it) { return it.get() == iter_var.get(); }) == iter_vars.end()) {
            iter_vars.push_back(iter_var);
          }
        }
      });
    }

    DataType data_type = index_expr.dtype();
    Var index_buffer_var("index_var_" + std::to_string(expr_index),
                         PointerType(PrimType(data_type), storage_scope));
    Array<PrimExpr> buffer_shape;
    for (const Var& it : info->origin_block_vars[expr_index]) {
      buffer_shape.push_back(
          arith::EvalSet(info->var_binding.at(it), arith::AsIntSet(info->range_map)).max() + 1);
    }
    info->cache_buffer.push_back(Buffer(index_buffer_var, data_type, buffer_shape, {1}, {0},
                                        index_buffer_var->name_hint, 0, 0, kDefault));

    // Create loop vars and block vars' binding_value
    std::vector<Var> loop_vars;
    Map<Var, PrimExpr> replace_table;
    for (const Var& it : iter_vars) {
      DataType data_type = DetermineDatatype(arith::IntSet::FromRange(info->range_map.at(it)));
      Var loop_var("ax" + std::to_string(replace_table.size()), data_type);
      loop_vars.push_back(loop_var);
      replace_table.Set(it, loop_var);
    }
    // Create iter_values from the original block.
    std::vector<PrimExpr> iter_values;
    for (const Var& it : info->origin_block_vars[expr_index]) {
      iter_values.push_back(Substitute(info->var_binding.at(it), replace_table));
    }
    // block variables
    Array<IterVar> block_vars;
    // block access region for write buffers
    Region access_region;
    // indices used in block body
    Array<PrimExpr> access_indices;
    Map<Var, PrimExpr> block_var_map;
    // Create block vars, block's accessed region and accessing indices
    for (size_t i = 0; i < info->origin_block_vars[expr_index].size(); i++) {
      const Var& block_var = info->origin_block_vars[expr_index][i];
      Var var("v" + std::to_string(access_indices.size()), block_var.dtype());
      Range range = Range::FromMinExtent(make_zero(block_var.dtype()),
                                         info->range_map.at(iter_vars[i])->extent);
      block_vars.push_back(IterVar(/*dom=*/range,
                                   /*var=*/var,
                                   /*IterVarType=*/kDataPar));

      access_indices.push_back(var);
      access_region.push_back(Range::FromMinExtent(var, make_const(var.dtype(), 1)));
      block_var_map.Set(block_var, var);
    }

    // Create the index computing block
    PrimExpr new_expr = Substitute(index_expr, block_var_map);
    Block block(
        /*iter_vars=*/std::move(block_vars),
        /*reads=*/{},
        /*writes=*/{BufferRegion(info->cache_buffer[expr_index], access_region)},
        /*name_hint=*/"index_" + std::to_string(expr_index),
        /*body=*/
        BufferStore(info->cache_buffer[expr_index], new_expr, access_indices),
        /*init=*/NullOpt,
        /*alloc_buffers=*/{},
        /*match_buffers=*/{},
        /*annotations=*/{});
    blocks.push_back(block);
    // Create the block realize node
    Stmt body = BlockRealize(/*values=*/iter_values,
                             /*predicate=*/const_true(),
                             /*block=*/block);
    // Create surrounding loops
    for (size_t i = loop_vars.size(); i >= 1; --i) {
      body = For(/*loop_var=*/loop_vars[i - 1],
                 /*min=*/0,
                 /*extent=*/info->range_map.at(iter_vars[i - 1])->extent,
                 /*kind=*/ForKind::kSerial,
                 /*body=*/body);
    }
    bodies.push_back(body);
  }

  info->cache_stage = SeqStmt(bodies);
  return blocks;
}

/*!
 * \brief Insert the cache stages into the specific position
 * \param stmt A sequence of statements or a single statement that the new stage is inserted in
 * \param pos The position where the cache stage is inserted
 * \param stage The stage to be inserted
 * \return A SeqStmt, the result after insertion
 */
Stmt InsertIndexStage(const Stmt& stmt, int pos, const Stmt& stage) {
  if (const auto* seq_stmt = stmt.as<SeqStmtNode>()) {
    ObjectPtr<SeqStmtNode> result = make_object<SeqStmtNode>(*seq_stmt);
    result->seq.insert(result->seq.begin() + pos, stage);
    return SeqStmt(result);
  }
  if (pos == 0) {
    return SeqStmt::Flatten<Array<Stmt>>({stage, stmt});
  }
  ICHECK_EQ(pos, 1);
  return SeqStmt::Flatten<Array<Stmt>>({stmt, stage});
}

/*! \brief Mutator for CacheIndex. */
class CacheIndexRewriter : public StmtExprMutator {
 public:
  /*!
   * \brief Rewrite the AST and add stages of writting precomputed index
   * \param scope_sref The parent scope of this mutation
   * \param info The index information
   * \return The new AST rooting at the original parent scope
   */
  static Stmt Rewrite(const StmtSRef& scope_sref, IndexInfo* info) {
    CacheIndexRewriter rewriter(scope_sref, info);
    return rewriter(GetRef<Stmt>(scope_sref->stmt));
  }

 private:
  explicit CacheIndexRewriter(const StmtSRef& scope_sref, IndexInfo* info)
      : scope_sref_(scope_sref), info_(info) {
    cache_indices_.reserve(info_->origin_block_vars.size());
    for (const Array<Var>& group_it : info_->origin_block_vars) {
      cache_indices_.push_back({});
      for (const Var& it : group_it) {
        cache_indices_.back().push_back(it);
      }
    }
  }

  Stmt VisitStmt_(const BlockNode* block) final {
    Block old_stmt = GetRef<Block>(block);
    // Mutate the body
    visiting_target_block = static_cast<bool>(block == info_->target_block->stmt);
    Block stmt = Downcast<Block>(StmtMutator::VisitStmt_(block));
    visiting_target_block = false;

    // Check if it is the block corresponding to the parent scope
    if (block == scope_sref_->stmt) {
      // If so, put buffer allocation and insert cache stages on the parent scope
      ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
      n->body = InsertIndexStage(n->body, info_->loc_pos, info_->cache_stage);
      for (const Buffer& it : info_->cache_buffer) {
        n->alloc_buffers.push_back(it);
      }
      stmt = Block(n);
    }
    info_->block_reuse.Set(old_stmt, stmt);
    return std::move(stmt);
  }

  Stmt VisitStmt_(const BufferStoreNode* store) final {
    Stmt ret_stmt = StmtMutator::VisitStmt_(store);
    // Replace common sub expr for target block, with cached buffer load
    if (visiting_target_block) {
      for (size_t i = 0; i < info_->index_exprs.size(); i++) {
        PrimExpr& computation = info_->index_exprs[i];
        std::function<bool(const PrimExpr&)> predicate_selector =
            [computation](const PrimExpr& current_expr) {
              return (EquivalentTerms(current_expr, computation, true));
            };
        BufferLoad load = BufferLoad(info_->cache_buffer[i], cache_indices_[i]);
        ret_stmt = ReplaceSelectedExpr::ReplaceSelectedExprInStmt(
            ret_stmt, predicate_selector, std::move(load),
            [](const PrimExpr& expr) { return true; });
      }
    }
    return ret_stmt;
  }

  PrimExpr VisitExpr_(const LoadNode* op) final {
    LOG(FATAL) << "Unexpected use of deprecated LoadNode.  Please use BufferLoadNode instead.";
  }

 private:
  /*! \brief The parent scope of the insertion */
  const StmtSRef& scope_sref_;
  /*! \brief The info for inserting cache stage */
  IndexInfo* info_;
  /*! \brief The indices for the cache buffer */
  std::vector<Array<PrimExpr>> cache_indices_;
  /*! \brief Indicating whether cache stage is inserted, only do index replacement afterwards*/
  bool visiting_target_block{false};
};

Array<StmtSRef> CacheIndex(ScheduleState self, const StmtSRef& block_sref,
                           const String& storage_scope, int cse_thresh) {
  /*!
   * Check:
   *   - The index is in the array of block reading region
   *
   * Mutate:
   *   - Allocate new cache buffers under the current scope.
   *   - Precompute the index and store it in cache buffers.
   */

  // Step 0. Checking index, getting the target buffer and the parent scope
  IndexInfo info;
  info.target_block = block_sref;
  CHECK_GE(cse_thresh, 0) << "cse_thresh should not be negative number";
  info.cse_thresh = cse_thresh;
  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false);

  // Step 1. Collect the indexing info of target buffer.
  IndexInfoCollector::Collect(self, block_sref, scope_sref, &info);

  // Step 2. Create cache stages and rewrite the stmt.
  BlockRealize realize = GetBlockRealize(self, block_sref);
  info.var_binding = GetBindings(realize);
  Array<Block> cache_stages = MakeIndexCacheStage(&info, storage_scope);
  Stmt new_scope = CacheIndexRewriter::Rewrite(/*scope_sref=*/scope_sref, /*info=*/&info);

  bool old_stage_pipeline = self->block_info[block_sref].scope->stage_pipeline;

  // Step 3. Replacing and updating flags.
  self->Replace(scope_sref, new_scope, info.block_reuse);
  Array<StmtSRef> result_block_srefs;
  for (const Block& it : cache_stages) {
    StmtSRef result_block_sref = self->stmt2ref.at(it.get());
    result_block_srefs.push_back(result_block_sref);
    BlockInfo& block_info = self->block_info[result_block_sref];

    bool affine_binding = false;
    if (result_block_sref->parent == nullptr) {
      affine_binding = true;
    } else {
      arith::Analyzer analyzer;
      StmtSRef parent_sref = GetRef<StmtSRef>(result_block_sref->parent);
      affine_binding = IsAffineBinding(/*realize=*/GetBlockRealize(self, result_block_sref),
                                       /*loop_var_ranges=*/LoopDomainOfSRefTreePath(parent_sref),
                                       /*analyzer=*/&analyzer);
    }

    block_info.affine_binding = affine_binding;
    block_info.region_cover = true;
    block_info.scope->stage_pipeline = old_stage_pipeline;
  }

  return result_block_srefs;
}

/******** InstructionKind Registration ********/

struct CacheIndexTraits : public UnpackedInstTraits<CacheIndexTraits> {
  static constexpr const char* kName = "CacheIndex";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static Array<BlockRV> UnpackedApplyToSchedule(Schedule sch, BlockRV block, String storage_scope,
                                                Integer cse_thresh) {
    return sch->CacheIndex(block, storage_scope, cse_thresh->value);
  }

  static String UnpackedAsPython(Array<String> outputs, String block, String storage_scope,
                                 Integer cse_thresh) {
    PythonAPICall py("cache_index");
    py.Input("block", block);
    py.Input("storage_scope", storage_scope);
    py.Input("cse_thresh", cse_thresh->value);
    py.OutputList(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(CacheIndexTraits);

}  // namespace tir
}  // namespace tvm
