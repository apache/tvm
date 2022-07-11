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

template <class T>
bool UsesVar(const T& x, const Var& var) {
  return UsesVar(x, [tgt = var.get()](const VarNode* v) { return v == tgt; });
}

Range RangeFromExtent(const PrimExpr& extent) {
  return Range::FromMinExtent(make_zero(extent->dtype), extent);
}

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
 * \param loops Iterators of the subspace
 * \param predicate The predicate constraint on the input iterators.
 * \return The result of the subspace division.
 */
Array<Array<arith::IterMark>> TrivialSubspaceDivision(const Array<IterVar>& iter_vars,
                                                      const Array<PrimExpr>& bindings,
                                                      const PrimExpr& predicate,
                                                      const Array<Var>& outer_iters,
                                                      const Array<Var>& inner_iters) {
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
 * \brief Subspace division. The space is divided into two subspaces:
 *  1. The subspace represented by the outer loops above `loop_sref` (exclusive).
 *  2. The subspace represented by the inner loops below `loop_sref` (inclusive).
 * \param realize The inner block
 * \param block_sref The sref to the inner block
 * \param loop_sref The loop that is the root of the second subspace.
 * \param loops The loops that represents the second part of the subspace.
 * \param analyzer The arithmetic analyzer to use.
 */
Array<Array<arith::IterMark>> SubspaceDivide(const BlockRealize& realize,
                                             const StmtSRef& block_sref,  //
                                             const StmtSRef& loop_sref,   //
                                             std::vector<const ForNode*>* loops,
                                             arith::Analyzer* analyzer) {
  Array<Var> inner_vars;
  Array<Var> outer_vars;
  Map<Var, Range> loop_var_domain;
  bool inner = true;
  for (StmtSRefNode* sref = block_sref->parent;    //
       sref && sref->stmt->IsInstance<ForNode>();  //
       sref = sref->parent) {
    const ForNode* loop = static_cast<const ForNode*>(sref->stmt);
    if (inner) {
      loops->push_back(loop);
      inner_vars.push_back(loop->loop_var);
    } else {
      outer_vars.push_back(loop->loop_var);
    }
    loop_var_domain.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
    if (sref == loop_sref.get()) {
      inner = false;
    }
  }
  Array<Array<arith::IterMark>> result =
      arith::SubspaceDivide(realize->iter_values, loop_var_domain, inner_vars, realize->predicate,
                            arith::IterMapLevel::Surjective, analyzer);
  if (!result.empty()) {
    return result;
  }
  return TrivialSubspaceDivision(realize->block->iter_vars,
                                 realize->iter_values,  //
                                 realize->predicate,    //
                                 outer_vars, inner_vars);
}

/*!
 * \brief Derive the block bindings for both inner and outer block
 * \param iter_vars The original block iterators to the inner block
 * \param division The subspace division.
 * \param outer_iter_vars The outer block iterators.
 * \param outer_bindings The outer block bindings.
 * \param inner_iter_vars The inner block iterators.
 * \param inner_bindings The inner block bindings.
 * \return A substitution plan to the iterators in the original inner block.
 */
Map<Var, PrimExpr> DeriveBlockBinding(const Array<IterVar>& iter_vars,                //
                                      const Array<Array<arith::IterMark>>& division,  //
                                      Array<IterVar>* outer_iter_vars,                //
                                      Array<PrimExpr>* outer_bindings,                //
                                      Array<IterVar>* inner_iter_vars,                //
                                      Array<PrimExpr>* inner_bindings) {
  using arith::IterMapExpr;
  using arith::IterMapExprNode;
  using arith::NormalizeIterMapToExpr;
  Map<Var, PrimExpr> block_var_subst;
  ICHECK_EQ(iter_vars.size() + 1, division.size());
  for (int i = 0, n = iter_vars.size(); i < n; ++i) {
    const IterVar& iter_var = iter_vars[i];
    arith::IterMark outer_mark = division[i][0];
    arith::IterMark inner_mark = division[i][1];
    IterMapExpr outer_binding = Downcast<IterMapExpr>(outer_mark->source);
    IterMapExpr inner_binding = Downcast<IterMapExpr>(inner_mark->source);
    // After computing the subspace division, bindings[i] can be written as
    // outer_binding * inner_binding->extent + inner_binding
    // The outer block will have binding: iter_outer -> outer_binding
    // The inner block will have binding: iter_inner -> inner_binding
    // The iter in the original block will be substituted with base + iter_inner where
    // base == iter_outer * iter_inner_extent
    if (is_one(inner_mark->extent)) {  // IsOuter
      // extract this iter var to outer block directly
      outer_bindings->push_back(NormalizeIterMapToExpr(outer_binding));
      outer_iter_vars->push_back(iter_var);
      continue;
    }
    // create iter var for the outer block
    IterVar outer_iter(/*dom=*/RangeFromExtent(outer_mark->extent),
                       /*var=*/iter_var->var.copy_with_suffix("_o"),
                       /*iter_type=*/iter_var->iter_type);
    outer_bindings->push_back(NormalizeIterMapToExpr(outer_binding));
    outer_iter_vars->push_back(outer_iter);
    // create iter var for the inner block
    IterVar inner_iter(/*dom=*/RangeFromExtent(inner_mark->extent),
                       /*var=*/iter_var->var.copy_with_suffix("_i"),
                       /*iter_type=*/iter_var->iter_type);
    inner_bindings->push_back(NormalizeIterMapToExpr(inner_binding));
    inner_iter_vars->push_back(inner_iter);
    // substitution
    PrimExpr sub{nullptr};
    if (is_one(outer_mark->extent)) {
      sub = inner_iter->var;
    } else {
      sub = outer_iter * inner_mark->extent + inner_iter->var;
    }
    block_var_subst.Set(iter_var->var, sub);
  }
  return block_var_subst;
}

/*!
 * \brief Generate the inner block for blockization
 * \param iter_vars IterVars used in the inner block.
 * \param iter_values IterVar bindings used in the inner block.
 * \param predicate The predicate of the inner block.
 * \param block The inner block as a template to be created from. This method will modify its
 * `iter_vars`, `init` and `reads` fields.
 * \return The inner block created.
 */
BlockRealize GenerateInner(bool is_write_reduction,
                           const Array<IterVar>& iter_vars,     //
                           const Array<PrimExpr>& iter_values,  //
                           const PrimExpr& predicate,           //
                           Block block) {
  BlockNode* n = block.CopyOnWrite();
  n->iter_vars = iter_vars;
  n->init = NullOpt;
  if (is_write_reduction) {
    Array<BufferRegion> reads;
    reads.reserve(block->writes.size() + block->reads.size());
    reads.insert(reads.end(), block->writes.begin(), block->writes.end());
    reads.insert(reads.end(), block->reads.begin(), block->reads.end());
    n->reads = std::move(reads);
  }
  return BlockRealize(/*iter_values=*/iter_values, /*predicate=*/predicate,
                      /*block=*/block);
}

/*!
 * \brief Generate the init stmt for the outer block
 * \param block The original block with init.
 * \param inner_realize The block realize of the inner block after blockize.
 * \param loops The inner loops after blockize.
 * \return The subtree of the init block and its outer loops.
 */
Stmt GenerateOuterInit(const Stmt& block_init, const BlockRealize& inner_realize,
                       const std::vector<const ForNode*>& loops, String block_name) {
  const Block& inner_block = inner_realize->block;
  Map<Var, PrimExpr> subst_map;
  // Step 1: Create new block vars for the block inside the init stmt of outer block
  // A iter is used in the block if
  // 1) It is data parallel
  // 2) It is used in the original init block
  Array<IterVar> iter_vars;
  Array<PrimExpr> iter_values;
  ICHECK_EQ(inner_block->iter_vars.size(), inner_realize->iter_values.size());
  int n = inner_block->iter_vars.size();
  iter_vars.reserve(n);
  iter_values.reserve(n);
  for (int i = 0; i < n; ++i) {
    const IterVar& old_iter_var = inner_block->iter_vars[i];
    const PrimExpr& iter_value = inner_realize->iter_values[i];
    if (old_iter_var->iter_type == IterVarType::kDataPar &&
        UsesVar(block_init, old_iter_var->var)) {
      ObjectPtr<IterVarNode> new_iter_var = make_object<IterVarNode>(*old_iter_var.get());
      new_iter_var->var = new_iter_var->var.copy_with_suffix("_init");
      subst_map.Set(old_iter_var->var, new_iter_var->var);
      iter_vars.push_back(IterVar(new_iter_var));
      iter_values.push_back(iter_value);
    }
  }
  // Step 2: Generate the block inside init stmt of outer block
  Stmt stmt = BlockRealize(
      /*iter_values=*/iter_values,
      /*predicate=*/inner_realize->predicate,
      /*block=*/
      Block(/*iter_vars=*/iter_vars,
            /*reads=*/{},
            /*writes=*/inner_block->writes,
            /*name_hint=*/block_name,
            /*body=*/block_init,
            /*init=*/NullOpt));
  // Step 3. Create the loop nest on top of the block
  for (const ForNode* loop : loops) {
    bool is_init_loop = false;
    for (const PrimExpr& init_binding : iter_values) {
      if (UsesVar(init_binding, loop->loop_var)) {
        is_init_loop = true;
        break;
      }
    }
    if (is_init_loop) {
      ObjectPtr<ForNode> new_loop = make_object<ForNode>(*loop);
      new_loop->loop_var = loop->loop_var.copy_with_suffix("");
      new_loop->body = std::move(stmt);
      subst_map.Set(loop->loop_var, new_loop->loop_var);
      stmt = For(new_loop);
    }
  }
  // Step 4: Substitute the iter vars and loop vars
  return Substitute(stmt, subst_map);
}

/*!
 * \brief Substitute variables in the stmt, do simplification and track block substitution
 * \param stmt The stmt to be substituted.
 * \param sub The substitution map.
 * \param block_sref_reuse The block substitution happens during the substitution.
 * \param analyzer The analyzer for arithmetic simplification.
 * \return The substituted stmt.
 */
Stmt Substitute(const Stmt& stmt, const Map<Var, PrimExpr>& sub,
                Map<Block, Block>* block_sref_reuse, arith::Analyzer* analyzer) {
  struct Replacer : public StmtExprMutator {
    explicit Replacer(const Map<Var, PrimExpr>& sub, Map<Block, Block>* block_sref_reuse,
                      arith::Analyzer* analyzer)
        : sub_(sub), block_sref_reuse_(block_sref_reuse), analyzer_(analyzer) {}

    PrimExpr VisitExpr(const PrimExpr& op) final {
      PrimExpr result = StmtExprMutator::VisitExpr(op);
      if (!result.same_as(op)) {
        return analyzer_->Simplify(result);
      }
      return result;
    }

    PrimExpr VisitExpr_(const VarNode* op) final {
      if (Optional<PrimExpr> e = sub_.Get(GetRef<Var>(op))) {
        return e.value();
      }
      return StmtExprMutator::VisitExpr_(op);
    }

    Stmt VisitStmt_(const BlockNode* op) final {
      Block src = GetRef<Block>(op);
      Block tgt = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
      if (!src.same_as(tgt)) {
        block_sref_reuse_->Set(src, tgt);
      }
      return tgt;
    }

    const Map<Var, PrimExpr>& sub_;
    Map<Block, Block>* block_sref_reuse_;
    arith::Analyzer* analyzer_;
  };
  return Replacer(sub, block_sref_reuse, analyzer)(stmt);
}

/*!
 * \brief Relax the variables for the given regions
 * \param regions The regions to be relaxed.
 * \param dom_map The variables to be relaxed
 * \return The relaxed regions
 */
Array<BufferRegion> EvalSetRegions(const Array<BufferRegion>& regions,
                                   const Map<Var, arith::IntSet>& dom_map) {
  Array<BufferRegion> results;
  results.reserve(regions.size());
  for (const BufferRegion& buffer_region : regions) {
    const Buffer& buffer = buffer_region->buffer;
    Array<arith::IntSet> relaxed = arith::EvalSet(buffer_region->region, dom_map);
    ICHECK_EQ(relaxed.size(), buffer->shape.size());
    int ndim = buffer->shape.size();
    Array<Range> new_region;
    new_region.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      new_region.push_back(relaxed[i].CoverRange(RangeFromExtent(buffer->shape[i])));
    }
    results.push_back(BufferRegion(buffer, new_region));
  }
  return results;
}

/*!
 * \brief Create the loop nest on top of the given stmt.
 * \param stmt The stmt to be wrapped.
 * \param loops The loop nests
 * \return The wrapped stmt.
 */
Stmt MakeLoopNest(Stmt stmt, const std::vector<const ForNode*>& loops) {
  for (const ForNode* loop : loops) {
    ObjectPtr<ForNode> new_loop = make_object<ForNode>(*loop);
    new_loop->body = std::move(stmt);
    stmt = For(new_loop);
  }
  return stmt;
}

StmtSRef Blockize(ScheduleState self, const StmtSRef& loop_sref) {
  const ForNode* loop = TVM_SREF_TO_FOR(loop, loop_sref);
  arith::Analyzer analyzer;
  // Step 1: Check and get the only block under `loop`.
  BlockRealize block_realize = CheckGetSingleChildBlockRealizeOnSRefTree(self, loop_sref);
  Block block = block_realize->block;
  StmtSRef block_sref = self->stmt2ref.at(block.get());
  // Step 2: Derive subspace division
  std::vector<const ForNode*> loops;
  Array<Array<arith::IterMark>> division =
      SubspaceDivide(block_realize, block_sref, loop_sref, &loops, &analyzer);
  if (division.empty()) {
    throw SubspaceNotDivisibleError(self->mod, GetRef<For>(loops.back()), block);
  }
  PrimExpr outer_predicate = division.back()[0]->extent;
  PrimExpr inner_predicate = division.back()[1]->extent;
  // Step 3. Derive block bindings for both outer and inner block.
  Array<IterVar> outer_iter_vars;
  Array<IterVar> inner_iter_vars;
  Array<PrimExpr> outer_bindings;
  Array<PrimExpr> inner_bindings;
  Map<Var, PrimExpr> block_var_subst =                       //
      DeriveBlockBinding(block->iter_vars, division,         //
                         &outer_iter_vars, &outer_bindings,  //
                         &inner_iter_vars, &inner_bindings);
  // Step 4: Do var substitution to adjust to the new block bindings
  Map<Block, Block> block_sref_reuse;
  Map<Var, arith::IntSet> inner_iter_dom;
  for (const IterVar& iter : inner_iter_vars) {
    inner_iter_dom.Set(iter->var, arith::IntSet::FromRange(iter->dom));
    analyzer.Bind(iter->var, iter->dom);
  }
  Block block_subst =
      Downcast<Block>(Substitute(block, block_var_subst, &block_sref_reuse, &analyzer));
  // Step 5: Generate the inner block. The write regions of the inner blocks will be reduction if
  // 1. The original block has init stmt.
  // 2. There are outer reduction iter vars.
  bool has_outer_reduction = false;
  if (block_subst->init.defined()) {
    for (const IterVar& iter_var : outer_iter_vars) {
      if (iter_var->iter_type == kCommReduce) {
        has_outer_reduction = true;
        break;
      }
    }
  }
  BlockRealize inner_realize = GenerateInner(/*is_write_reduction=*/has_outer_reduction,
                                             /*iter_vars=*/inner_iter_vars,
                                             /*iter_values*/ inner_bindings,
                                             /*predicate=*/inner_predicate,
                                             /*block=*/block_subst);
  block_sref_reuse.Set(block, inner_realize->block);
  // Step 6: Generate the outer block.
  BlockRealize outer_realize(
      /*iter_values=*/std::move(outer_bindings),
      /*predicate=*/std::move(outer_predicate),
      /*block=*/
      Block(/*iter_vars=*/std::move(outer_iter_vars),
            /*reads=*/EvalSetRegions(block_subst->reads, inner_iter_dom),
            /*writes=*/EvalSetRegions(block_subst->writes, inner_iter_dom),
            /*name_hint=*/block_subst->name_hint + "_o",
            /*body=*/MakeLoopNest(inner_realize, loops),
            /*init=*/
            block_subst->init.defined()  //
                ? GenerateOuterInit(block_subst->init.value(), inner_realize, loops,
                                    block_subst->name_hint + "_init")
                : Optional<Stmt>(NullOpt)));
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
