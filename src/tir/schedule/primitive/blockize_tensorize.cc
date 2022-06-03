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
#include <functional>

#include "../ir_comparator.h"
#include "../utils.h"

namespace tvm {
namespace tir {

/*!
 * \brief ScheduleError that the bindings of the inner block are not divisible by the subspace
 * represented by the outer loops.
 */
class SubspaceNotDivisibleError : public ScheduleError {
 public:
  explicit SubspaceNotDivisibleError(IRModule mod, For scope_loop, Block inner_block)
      : mod_(std::move(mod)),
        scope_loop_(std::move(scope_loop)),
        inner_block_(std::move(inner_block)) {}

  String FastErrorString() const final {
    return "ScheduleError: The bindings of the inner block can not be blockized.";
  }

  String DetailRenderTemplate() const final {
    return "ScheduleError: The bindings of the inner block {0} can not be blockized by the loops "
           "starting at {1}.";
  }

  IRModule mod() const final { return mod_; }

  Array<ObjectRef> LocationsOfInterest() const final { return {inner_block_, scope_loop_}; }

 private:
  IRModule mod_;
  For scope_loop_;
  Block inner_block_;
};

/*!
 * \brief Detect if bindings are a trivial case of the subspace division where we can divide the
 * block iter bindings into two categories:
 *   1. The binding covers no inner loop vars.
 *   2. The binding covers only inner loop vars.
 *
 * The bindings are not required to be quasi-affine.
 *
 * \param iter_vars The input iterators
 * \param bindings The values of iter_vars
 * \param outer_loops Iterators outside the subspace.
 * \param inner_loops Iterators of the subspace
 * \param predicate The predicate constraint on the input iterators.
 * \return The result of the subspace division.
 */
Array<Array<arith::IterMark>> TrivialSubspaceDivision(const Array<IterVar>& iter_vars,
                                                      const Array<PrimExpr>& bindings,
                                                      const Array<Var>& outer_iters,
                                                      const Array<Var>& inner_iters,
                                                      const PrimExpr& predicate) {
  if (!is_one(predicate)) return {};
  Array<Array<arith::IterMark>> res;
  std::unordered_set<const VarNode*> outer_loop_vars;
  std::unordered_set<const VarNode*> inner_loop_vars;

  auto make_uses_var = [](const Array<Var>& vars) -> std::function<bool(const PrimExpr& expr)> {
    std::unordered_set<const VarNode*> var_set;
    var_set.reserve(vars.size());
    for (const Var& var : vars) {
      var_set.insert(var.get());
    }
    return [var_set = std::move(var_set)](const PrimExpr& expr) -> bool {
      return UsesVar(expr, [&var_set](const VarNode* var) {
        return var_set.count(var);  //
      });
    };
  };
  auto use_outer_loop_vars = make_uses_var(outer_iters);
  auto use_inner_loop_vars = make_uses_var(inner_iters);
  arith::IterMark unit_iter_mark(arith::IterSumExpr({}, 0), 1);

  for (size_t i = 0; i < bindings.size(); ++i) {
    bool outer = use_outer_loop_vars(bindings[i]);
    bool inner = use_inner_loop_vars(bindings[i]);
    arith::IterMark iter_mark;
    if (bindings[i]->IsInstance<VarNode>()) {
      iter_mark = arith::IterMark(
          arith::IterSplitExpr(arith::IterMark(bindings[i], iter_vars[i]->dom->extent)),
          iter_vars[i]->dom->extent);
    } else {
      iter_mark = arith::IterMark(arith::IterSumExpr({}, bindings[i]), iter_vars[i]->dom->extent);
    }
    if (outer && !inner) {
      res.push_back({/*outer_iter=*/iter_mark, /*inner_iter=*/unit_iter_mark});
    } else if (inner && !outer) {
      res.push_back({/*outer_iter=*/unit_iter_mark, /*inner_iter=*/iter_mark});
    } else if (!outer && !inner) {
      res.push_back({/*outer_iter=*/unit_iter_mark, /*inner_iter=*/unit_iter_mark});
    } else {
      return {};
    }
  }
  res.push_back({arith::IterMark(arith::IterSumExpr({}, 0), Bool(true)),
                 arith::IterMark(arith::IterSumExpr({}, 0), Bool(true))});
  return res;
}

/*!
 * \brief Generate the blockized init block.
 * \param block The original block with init.
 * \param inner_block_realize The block realize of the inner block after blockize.
 * \param inner_loops The inner loops after blockize.
 * \return The subtree of the init block and its outer loops.
 */
Stmt GenerateBlockizedInit(const Block& block, const BlockRealize& inner_block_realize,
                           const std::vector<const ForNode*>& inner_loops) {
  Array<IterVar> init_block_iters;
  Array<PrimExpr> init_bindings;
  const Block& inner_block = inner_block_realize->block;

  // Step 1: Collect data-parallel block iters
  for (size_t i = 0; i < inner_block->iter_vars.size(); i++) {
    const IterVar& iter_var = inner_block->iter_vars[i];
    const PrimExpr& binding = inner_block_realize->iter_values[i];
    if (iter_var->iter_type == IterVarType::kDataPar &&
        UsesVar(block->init.value(),
                [tgt_var = iter_var->var.get()](const VarNode* var) { return var == tgt_var; })) {
      init_block_iters.push_back(iter_var);
      init_bindings.push_back(binding);
    }
  }

  // Step 2: Collect loops related to iters of the init block
  std::vector<const ForNode*> init_loops;
  for (const ForNode* inner_loop : inner_loops) {
    for (const PrimExpr& init_binding : init_bindings) {
      if (UsesVar(init_binding, [tgt_var = inner_loop->loop_var.get()](const VarNode* var) {
            return var == tgt_var;
          })) {
        init_loops.push_back(inner_loop);
        break;
      }
    }
  }

  // Step 3: Create new block iters for the init block
  Map<Var, PrimExpr> subst_map;
  for (size_t i = 0; i < init_block_iters.size(); i++) {
    IterVar new_iter_var = init_block_iters[i];
    Var old_var = new_iter_var->var;
    Var new_var = old_var.copy_with_suffix("_init");
    new_iter_var.CopyOnWrite()->var = new_var;
    subst_map.Set(old_var, new_var);
    init_block_iters.Set(i, std::move(new_iter_var));
  }

  // Step 4: Generate loop nests and the init block
  Stmt new_init = BlockRealize(
      /*iter_values=*/init_bindings,
      /*predicate=*/inner_block_realize->predicate,
      /*block=*/
      Block{/*iter_vars=*/init_block_iters,
            /*reads=*/{},
            /*writes=*/block->writes,
            /*name_hint=*/block->name_hint + "_init",
            /*body=*/block->init.value(),
            /*init=*/NullOpt});

  // Step 5: Generate the parent loops for the init block
  for (const ForNode* init_loop : init_loops) {
    ObjectPtr<ForNode> new_loop = make_object<ForNode>(*init_loop);
    new_loop->loop_var = init_loop->loop_var.copy_with_suffix("");
    subst_map.Set(init_loop->loop_var, new_loop->loop_var);
    new_loop->body = std::move(new_init);
    new_init = For(new_loop);
  }

  // Step 6: Substitute with new loop variables and block iters to prevent duplication of
  // variables in the outer block.
  new_init = Substitute(new_init, subst_map);

  return new_init;
}

/*!
 * \brief A helper to collect the parent loops of the block. The loops are divided into two groups,
 * 'outer_loops', and 'inner_loops', by a specified loop as the separator. 'outer_loops' are the
 * ancestor loops of the separator loop. 'inner_loops' include the separator loop itself, and its
 * successor loops. It is possible that 'outer_loops' is empty.
 */
class LoopSubspaceCollector {
 public:
  /*!
   * \brief Collect the parent loops of the block and store the result in the corresponding fields.
   * \param block_sref The sref to the target block.
   * \param loop_sref The sref to the separator loop. The loop itself is counted as an inner loop.
   */
  void Collect(const StmtSRef& block_sref, const StmtSRef& loop_sref) {
    bool inner = true;
    for (StmtSRefNode* current_sref = block_sref->parent;
         current_sref && current_sref->stmt->IsInstance<ForNode>();
         current_sref = current_sref->parent) {
      const auto* current_loop = current_sref->StmtAs<ForNode>();
      ICHECK(current_loop);
      if (inner) {
        inner_loops.push_back(current_loop);
        inner_loop_vars.push_back(current_loop->loop_var);
      } else {
        outer_loops.push_back(current_loop);
        outer_loop_vars.push_back(current_loop->loop_var);
      }
      loop_var_domain.Set(current_loop->loop_var,
                          Range::FromMinExtent(current_loop->min, current_loop->extent));
      if (current_sref == loop_sref.get()) inner = false;
    }
  }
  /*! \brief Outer loops which are ancestors of the separator. */
  std::vector<const ForNode*> outer_loops;
  /*! \brief Inner loops which are the separator itself or its successors. */
  std::vector<const ForNode*> inner_loops;
  /*! \brief Loop variables of the outer loops. */
  Array<Var> outer_loop_vars;
  /*! \brief Loop variables of the inner loops. */
  Array<Var> inner_loop_vars;
  /*! \brief Domain of the loop variables. */
  Map<Var, Range> loop_var_domain;
};

/*!
 * \brief Check the bindings of the block iters can be divided by a subspace collected by the
 * collector.
 * \param mod The current IR module.
 * \param block_realize The block realize to be checked.
 * \param collector The collector which has collected the loops of the block.
 * \param analyzer The arithmetic analyzer.
 * \return The result of the subspace division.
 * \throws ScheduleError If the bindings are not divisible by the subspace.
 */
Array<Array<arith::IterMark>> CheckSubspaceDivisible(const IRModule& mod,
                                                     const BlockRealize& block_realize,
                                                     const LoopSubspaceCollector& collector,
                                                     arith::Analyzer* analyzer) {
  const Block& block = block_realize->block;

  Array<Array<arith::IterMark>> division = arith::SubspaceDivide(
      block_realize->iter_values, collector.loop_var_domain, collector.inner_loop_vars,
      block_realize->predicate, arith::IterMapLevel::Surjective, analyzer);

  if (division.empty()) {
    // If we can't do perfect subspace division, check if it is a trivial case of subspace division.
    // In this case, we can still blockize.
    division = TrivialSubspaceDivision(block->iter_vars, block_realize->iter_values,
                                       collector.outer_loop_vars, collector.inner_loop_vars,
                                       block_realize->predicate);
  }
  if (division.empty()) {
    throw SubspaceNotDivisibleError(mod, GetRef<For>(collector.inner_loops.back()), block);
  }
  return division;
}

/*!
 * \brief The binding extractor to compute the bindings of the outer and the inner blocks after
 * blockize.
 */
class BlockizedBindingExtractor {
 public:
  /*!
   * \brief Extract bindings for blockize.
   * \param iter_vars The iter vars of the original inner block.
   * \param division The result of the subspace division.
   */
  void ExtractBindings(const Array<IterVar>& iter_vars,
                       const Array<Array<arith::IterMark>>& division, arith::Analyzer* analyzer) {
    ICHECK_EQ(iter_vars.size() + 1, division.size());
    for (size_t i = 0; i < iter_vars.size(); ++i) {
      const IterVar& iter_var = iter_vars[i];
      arith::IterMark outer_mark = division[i][0];
      arith::IterMark inner_mark = division[i][1];
      const auto* outer_binding =
          TVM_TYPE_AS(outer_binding, outer_mark->source, arith::IterMapExprNode);
      const auto* inner_binding =
          TVM_TYPE_AS(inner_binding, inner_mark->source, arith::IterMapExprNode);

      // After computing the subspace division, bindings[i] can be written as
      // outer_binding * inner_binding->extent + inner_binding
      // The outer block will have binding: iter_outer -> outer_binding
      // The inner block will have binding: iter_inner -> inner_binding
      // The iter in the original block will be substituted with base + iter_inner where
      // base == iter_outer * iter_inner_extent

      if (is_one(division[i][1]->extent)) {  // IsOuter
        // extract this iter var to outer block directly
        outer_bindings.push_back(
            arith::NormalizeIterMapToExpr(GetRef<arith::IterMapExpr>(outer_binding)));
        outer_iter_vars.push_back(iter_var);
      } else {
        // create iter var for the outer block
        const IterVar outer_var(/*dom=*/Range::FromMinExtent(0, division[i][0]->extent),
                                /*var=*/iter_var->var.copy_with_suffix("_o"),
                                /*iter_type=*/iter_var->iter_type);
        outer_bindings.push_back(
            arith::NormalizeIterMapToExpr(GetRef<arith::IterMapExpr>(outer_binding)));
        outer_iter_vars.push_back(outer_var);
        PrimExpr base = is_one(division[i][0]->extent) ? 0 : outer_var * division[i][1]->extent;
        // create iter var for the inner block
        IterVar new_iter(Range::FromMinExtent(0, division[i][1]->extent), Var(iter_var->var),
                         iter_var->iter_type, iter_var->thread_tag, iter_var->span);
        inner_iter_dom_map.Set(new_iter->var, arith::IntSet::FromRange(new_iter->dom));
        analyzer->Bind(new_iter->var, new_iter->dom);
        inner_iter_vars.push_back(new_iter);
        inner_bindings.push_back(
            arith::NormalizeIterMapToExpr(GetRef<arith::IterMapExpr>(inner_binding)));
        inner_iter_subst_map.Set(iter_var->var, base + new_iter->var);
      }
    }
  }
  Map<Var, PrimExpr> inner_iter_subst_map;
  /*! \brief Iters of the outer block. */
  Array<IterVar> outer_iter_vars;
  /*! \brief Iters of the outer block. */
  Array<IterVar> inner_iter_vars;
  /*! \brief Binding values of the outer block. */
  Array<PrimExpr> outer_bindings;
  /*! \brief Binding values of the inner block. */
  Array<PrimExpr> inner_bindings;
  /*! \brief The domain of the inner block iters. */
  Map<Var, arith::IntSet> inner_iter_dom_map;
};

/*!
 * \brief Replacer for the inner block after blockize. Inner block iters will be replaced with
 * base + inner_iter and the expressions after substituion will be simplified if possible.
 */
class InnerIterReplacer : public StmtExprMutator {
 public:
  /*!
   * \brief The constructor
   * \param subst_map The substitution map of the inner block iters.
   * \param analyzer The arithmetic analyzer.
   * \param block_sref_reuse The map to save the block reuse information.
   */
  InnerIterReplacer(Map<Var, PrimExpr> subst_map, arith::Analyzer* analyzer,
                    Map<Block, Block>* block_sref_reuse)
      : subst_map_(std::move(subst_map)),
        analyzer_(analyzer),
        block_sref_reuse_(block_sref_reuse) {}

  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = subst_map_.find(GetRef<Var>(op));
    if (it != subst_map_.end()) {
      return (*it).second;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr(const PrimExpr& op) final {
    PrimExpr result = StmtExprMutator::VisitExpr(op);
    if (!result.same_as(op)) {
      return analyzer_->Simplify(result);
    }
    return result;
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    Stmt result = StmtExprMutator::VisitStmt_(op);
    if (!result.same_as(GetRef<Stmt>(op))) {
      block_sref_reuse_->Set(GetRef<Block>(op), Downcast<Block>(result));
    }
    return result;
  }

 private:
  Map<Var, PrimExpr> subst_map_;
  arith::Analyzer* analyzer_;
  Map<Block, Block>* block_sref_reuse_;
};

/*!
 * \brief Compute the access region of the outer block by relaxing the inner loops.
 * \param buffer_region The original buffer region.
 * \param The range of the inner loops.
 * \return The new buffer region.
 */
BufferRegion RelaxBlockizedInnerIters(const BufferRegion& buffer_region,
                                      const Map<Var, arith::IntSet>& inner_iter_relaxed_range) {
  Array<Range> new_region;
  new_region.reserve(buffer_region->region.size());
  Array<arith::IntSet> relaxed_int_set =
      arith::EvalSet(buffer_region->region, inner_iter_relaxed_range);
  ICHECK(buffer_region->region.size() == buffer_region->buffer->shape.size());
  for (size_t i = 0; i < buffer_region->region.size(); i++) {
    Range max_range = Range::FromMinExtent(0, buffer_region->buffer->shape[i]);
    new_region.push_back(relaxed_int_set[i].CoverRange(max_range));
  }
  return BufferRegion(buffer_region->buffer, std::move(new_region));
}

/*!
 * \brief Generate the outer block after blockize.
 * \param extractor The binding extractor which has extracted the blockized bindings.
 * \param block The original inner block.
 * \param inner_block_realize The block realize of the inner block after blockize.
 * \param inner_loops The inner loops after blockize.
 * \param predicate The outer predicate of the subspace division.
 * \return The block realize of the outer block after blockize.
 */
BlockRealize GenerateBlockizedOuterBlock(const BlockizedBindingExtractor& extractor,
                                         const Block& block, BlockRealize inner_block_realize,
                                         const std::vector<const ForNode*>& inner_loops,
                                         PrimExpr predicate) {
  // Step 1: Generate the init block if needed
  Optional<Stmt> new_init = NullOpt;
  if (block->init.defined()) {
    new_init = GenerateBlockizedInit(block, inner_block_realize, inner_loops);
  }

  // Step 2: Compute the access regions of the outer block by relaxing the inner loops
  Array<BufferRegion> new_reads = block->reads;
  Array<BufferRegion> new_writes = block->writes;

  auto f_mutate = [&](const BufferRegion& buffer_region) {
    return RelaxBlockizedInnerIters(buffer_region, extractor.inner_iter_dom_map);
  };
  new_reads.MutateByApply(f_mutate);
  new_writes.MutateByApply(f_mutate);

  // Step 3: Generate the body of the outer block. The body of the outer block is the inner block
  // realize and its surrounding loops.
  Stmt outer_block_body = inner_block_realize;
  for (const ForNode* loop : inner_loops) {
    ObjectPtr<ForNode> new_loop = make_object<ForNode>(*loop);
    new_loop->body = std::move(outer_block_body);
    outer_block_body = For(new_loop);
  }

  // Step 4: Generate the outer block and block realize.
  return BlockRealize(/*iter_values=*/std::move(extractor.outer_bindings),
                      /*predicate=*/std::move(predicate),
                      /*block=*/
                      Block(/*iter_vars=*/std::move(extractor.outer_iter_vars),  //
                            /*reads=*/std::move(new_reads),                      //
                            /*writes=*/std::move(new_writes),                    //
                            /*name_hint=*/block->name_hint + "_o",               //
                            /*body=*/std::move(outer_block_body),                //
                            /*init=*/std::move(new_init)));
}

StmtSRef Blockize(ScheduleState self, const StmtSRef& loop_sref) {
  const ForNode* loop = TVM_SREF_TO_FOR(loop, loop_sref);
  arith::Analyzer analyzer;

  // Step 1: Check the loop has a single child BlockRealize on the sref tree.
  BlockRealize block_realize = CheckGetSingleChildBlockRealizeOnSRefTree(self, loop_sref);
  Block block = block_realize->block;
  StmtSRef block_sref = self->stmt2ref.at(block.get());

  // Step 2: Collect loops inside and outside loop_sref.
  LoopSubspaceCollector collector;
  collector.Collect(block_sref, loop_sref);

  // Step 3: Calculate subspace division for the inner loops.
  Array<Array<arith::IterMark>> division =
      CheckSubspaceDivisible(self->mod, block_realize, collector, &analyzer);

  // Step 4: Generate bindings for the outer block and the inner block based on the result of
  // the subspace division.
  BlockizedBindingExtractor extractor;
  extractor.ExtractBindings(block->iter_vars, division, &analyzer);
  const PrimExpr& outer_pred = division.back()[0]->extent;
  const PrimExpr& inner_pred = division.back()[1]->extent;

  // Step 5: Substitute the iter vars in the original block with the inner iters after the subspace
  // division
  Map<Block, Block> block_sref_reuse;
  InnerIterReplacer replacer(std::move(extractor.inner_iter_subst_map), &analyzer,
                             &block_sref_reuse);
  Block new_block = Downcast<Block>(replacer(block));

  // Step 6: Generate the inner block.
  bool outer_reduction = false;  // whether there are outer reduction iter vars.
  for (const IterVar& iter_var : extractor.outer_iter_vars) {
    if (iter_var->iter_type == kCommReduce) {
      outer_reduction = true;
    }
  }
  BlockRealizeNode* inner_block_realize = block_realize.CopyOnWrite();
  inner_block_realize->iter_values = extractor.inner_bindings;
  inner_block_realize->predicate = inner_pred;
  inner_block_realize->block = new_block;
  BlockNode* inner_block = inner_block_realize->block.CopyOnWrite();
  inner_block->iter_vars = extractor.inner_iter_vars;
  inner_block->init = NullOpt;
  /* Add write regions to read regions if
   * 1. there are outer reduction iter vars.
   * 2. the init block is defined for current block.
   */
  if (outer_reduction && block->init.defined()) {
    Array<BufferRegion> new_reads;
    for (const BufferRegion& write_access : inner_block->writes) {
      new_reads.push_back(write_access);
    }
    for (const BufferRegion& read_access : inner_block->reads) {
      new_reads.push_back(read_access);
    }
    inner_block->reads = std::move(new_reads);
  }
  block_sref_reuse.Set(block, inner_block_realize->block);

  // Step 6: Generate the outer block.
  BlockRealize outer_realize =
      GenerateBlockizedOuterBlock(extractor, new_block, GetRef<BlockRealize>(inner_block_realize),
                                  collector.inner_loops, outer_pred);
  // Step 7: Do the actual replacement
  self->Replace(loop_sref, outer_realize, block_sref_reuse);

  // Step 8: Update the cached flags
  StmtSRef outer_block_sref = self->stmt2ref.at(outer_realize->block.get());
  StmtSRef scope_root = tir::GetScopeRoot(self, outer_block_sref, /*require_stage_pipeline=*/false);
  bool scope_block_affine_binding = self->IsAffineBlockBinding(scope_root);
  self->UpdateScopeBlockInfo(tir::GetBlockRealize(self, scope_root));
  self->block_info[scope_root].affine_binding = scope_block_affine_binding;
  return outer_block_sref;
}

/*!
 * \brief Update the map from the buffers in the desc to the impl of the tensor
 * intrinsic.
 * \param intrinsic The tensor intrinsic.
 * \param buffer_map The map to be updated.
 */
void RemapTensorIntrinBuffers(
    const TensorIntrin& intrinsic,
    std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>* buffer_map) {
  ICHECK_EQ(intrinsic->desc->params.size(), intrinsic->impl->params.size());
  for (size_t i = 0; i < intrinsic->desc->params.size(); ++i) {
    const Var& lhs_var = intrinsic->desc->params[i];
    const Buffer& lhs_buffer = intrinsic->desc->buffer_map[lhs_var];
    const Var& rhs_var = intrinsic->impl->params[i];
    const Buffer& rhs_buffer = intrinsic->impl->buffer_map[rhs_var];
    (*buffer_map)[rhs_buffer] = lhs_buffer;
  }
}

void Tensorize(ScheduleState self, const StmtSRef& block_or_loop_sref,
               const TensorIntrin& intrinsic) {
  /*!
   * Check:
   *   - Check buffer binding, including type, alignment, shape and etc.
   *   - Check the sub AST is equal to the desc function.
   *
   * Mutate:
   *   - Blockize the sub AST (please refer blockize for details)
   *   - Bind buffers
   *   - Mutate the impl of the tensor intrinsic by replacing its buffers with new
   *     buffers created via match buffer region.
   *   - Replace the sub tree with the mutated function.
   */
  const BlockRealize& desc_block_realize = Downcast<BlockRealize>(intrinsic->desc->body);
  const BlockRealize& impl_block_realize = Downcast<BlockRealize>(intrinsic->impl->body);
  Block impl_block = impl_block_realize->block;

  // Step 1: Blockize the subtree rooted at the given loop if needed
  StmtSRef block_sref{nullptr};
  if (block_or_loop_sref->StmtAs<ForNode>()) {
    block_sref = Blockize(self, block_or_loop_sref);
  } else {
    ICHECK(block_or_loop_sref->StmtAs<BlockNode>());
    block_sref = block_or_loop_sref;
  }
  const BlockRealize& block_realize = GetBlockRealize(self, block_sref);

  // Step 2: Compare the block with the desc of the tensor intrinsic, find the correspondence
  // between buffers in the block and the desc.
  TensorizeComparator comparator(self->mod, /*assert_mode=*/true);
  comparator.VisitStmt(block_realize, desc_block_realize);

  // Step 3: Find the correspondence between buffers in the current AST and the impl of
  // the tensor intrinsic
  // Step 3.1: Map from intrinsic func buffer to desc func buffer
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> intrin_buffer_map;
  RemapTensorIntrinBuffers(intrinsic, &intrin_buffer_map);
  // Step 3.2: Map form intrinsic func buffer to current AST buffer
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_map;
  for (const auto& pair : intrin_buffer_map) {
    auto it = comparator.rhs_buffer_map_.find(pair.second);
    ICHECK(it != comparator.rhs_buffer_map_.end()) << pair.second;
    buffer_map[pair.first] = it->second;
  }

  // Step 4: Create MatchBufferRegion for the params of the impl function of the tensor
  // intrin to make them subregions of the buffer in the original IR.
  std::unordered_map<Buffer, Array<Range>, ObjectPtrHash, ObjectPtrEqual> buffer_region_map;
  for (const BufferRegion& read : impl_block->reads) {
    buffer_region_map.emplace(read->buffer, read->region);
  }
  for (const BufferRegion& write : impl_block->writes) {
    buffer_region_map.emplace(write->buffer, write->region);
  }
  Array<MatchBufferRegion> match_buffer_regions;
  match_buffer_regions.reserve(intrinsic->impl->params.size());
  for (size_t i = 0; i < intrinsic->impl->params.size(); ++i) {
    const auto& param = intrinsic->impl->params[i];
    const auto& buffer = intrinsic->impl->buffer_map.at(param);
    const auto& source = buffer_map.at(buffer);
    // add the detected base indices to each buffer access region of the tensor intrinsic
    Region old_region = buffer_region_map.at(buffer);
    const auto& indices_base = comparator.buffer_indices_.at(source);
    int offset = static_cast<int>(indices_base.size()) - static_cast<int>(old_region.size());
    ICHECK(offset >= 0);
    Region new_region;
    new_region.reserve(source->shape.size());
    for (int i = 0; i < offset; i++) {
      new_region.push_back(Range::FromMinExtent(indices_base[i], 1));
    }
    for (int i = 0; i < static_cast<int>(old_region.size()); i++) {
      new_region.push_back(Range::FromMinExtent(indices_base[i + offset], old_region[i]->extent));
    }
    match_buffer_regions.push_back(MatchBufferRegion(buffer, BufferRegion(source, new_region)));
  }

  // Step 5: Replace the subtree in the original IR with the tensor intrin impl.
  ObjectPtr<BlockNode> new_block_ptr = make_object<BlockNode>(*block_realize->block.get());
  new_block_ptr->body = impl_block->body;
  ICHECK(new_block_ptr->match_buffers.empty());
  new_block_ptr->match_buffers = std::move(match_buffer_regions);
  Block new_block(new_block_ptr);

  self->Replace(block_sref, new_block, {{block_realize->block, new_block}});

  // Step 6: Update the cached flags.
  StmtSRef scope_root = tir::GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false);
  self->UpdateScopeBlockInfo(static_cast<const BlockNode*>(scope_root->stmt)->body);
}

/******** InstructionKind Registration ********/

struct BlockizeTraits : public UnpackedInstTraits<BlockizeTraits> {
  static constexpr const char* kName = "Blockize";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static BlockRV UnpackedApplyToSchedule(Schedule sch, LoopRV loop_rv) {
    return sch->Blockize(loop_rv);
  }

  static String UnpackedAsPython(Array<String> outputs, String loop_rv) {
    PythonAPICall py("blockize");
    py.Input("loop", loop_rv);
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct TensorizeTraits : public UnpackedInstTraits<TensorizeTraits> {
  static constexpr const char* kName = "Tensorize";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, ObjectRef block_or_loop_rv, String intrin) {
    if (const auto* block = block_or_loop_rv.as<BlockRVNode>()) {
      sch->Tensorize(GetRef<BlockRV>(block), intrin);
    } else if (const auto* loop = block_or_loop_rv.as<LoopRVNode>()) {
      sch->Tensorize(GetRef<LoopRV>(loop), intrin);
    } else {
      LOG(FATAL) << "TypeError: Expected Block or Loop, but gets: "
                 << block_or_loop_rv->GetTypeKey();
    }
  }

  static String UnpackedAsPython(Array<String> outputs, String block_or_loop_rv, String intrin) {
    PythonAPICall py("tensorize");
    py.Input("block_or_loop", block_or_loop_rv);
    py.Input("tensor_intrin", intrin);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(BlockizeTraits);
TVM_REGISTER_INST_KIND_TRAITS(TensorizeTraits);

}  // namespace tir
}  // namespace tvm
