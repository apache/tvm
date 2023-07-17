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
#include <tvm/tir/data_type_rewriter.h>

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

template <class T>
T DeepCopy(const T& stmt) {
  return Downcast<T>(LoadJSON(SaveJSON(stmt)));
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
 * The bindings are not required to be quasi-affine. Trivial block iters are always preserved.
 *
 * \param iter_vars The input iterators
 * \param bindings The values of iter_vars
 * \param predicate The predicate constraint on the input iterators.
 * \param outer_iters The iters of the outer space
 * \param inner_iters The iters of the inner space
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

  for (int i = 0, n = bindings.size(); i < n; ++i) {
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
 * If loop_sref_as_outer is false:
 *  1. The subspace represented by the outer loops above `loop_sref` (exclusive).
 *  2. The subspace represented by the inner loops below `loop_sref` (inclusive).
 * else:
 *  1. The subspace represented by the outer loops above `loop_sref` (inclusive).
 *  2. The subspace represented by the inner loops below `loop_sref` (exclusive).
 * \param realize The inner block
 * \param block_sref The sref to the inner block
 * \param loop_sref The loop that is the root of the second subspace.
 * \param loops The loops that represents the second part of the subspace.
 * \param analyzer The arithmetic analyzer to use.
 * \param preserve_unit_iters Whether or not to preserve unit iterators in block bindings
 * \param loop_sref_as_outer Whether loop_sref is divided into outer or inner
 */
Array<Array<arith::IterMark>> SubspaceDivide(const BlockRealize& realize,
                                             const StmtSRef& block_sref,  //
                                             const StmtSRef& loop_sref,   //
                                             std::vector<const ForNode*>* loops,
                                             arith::Analyzer* analyzer, bool preserve_unit_iters,
                                             bool loop_sref_as_outer = false) {
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
    if ((loop_sref_as_outer && sref->parent == loop_sref.get()) || sref == loop_sref.get()) {
      inner = false;
    }
  }
  Array<Array<arith::IterMark>> result =
      arith::SubspaceDivide(realize->iter_values, loop_var_domain, inner_vars, realize->predicate,
                            arith::IterMapLevel::Surjective, analyzer,
                            /*simplify_trivial_iterators=*/!preserve_unit_iters);
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
 * \param preserve_unit_iters Whether or not to preserve unit iterators in block bindings
 * \return A substitution plan to the iterators in the original inner block.
 */
Map<Var, PrimExpr> DeriveBlockBinding(const Array<IterVar>& iter_vars,                //
                                      const Array<Array<arith::IterMark>>& division,  //
                                      Array<IterVar>* outer_iter_vars,                //
                                      Array<PrimExpr>* outer_bindings,                //
                                      Array<IterVar>* inner_iter_vars,                //
                                      Array<PrimExpr>* inner_bindings,                //
                                      bool preserve_unit_iters, bool reuse_outer = false) {
  using arith::IterMapExpr;
  using arith::IterMapExprNode;
  using arith::NormalizeIterMapToExpr;
  Map<Var, PrimExpr> block_var_subst;
  ICHECK_EQ(iter_vars.size() + 1, division.size());
  arith::Analyzer ana;
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
    // create iter var for the outer block
    IterVar outer_iter;
    if (reuse_outer) {
      outer_iter = outer_iter_vars->operator[](i);
      ICHECK(ana.CanProveEqual(outer_iter->dom->extent, outer_mark->extent));
      ICHECK(
          ana.CanProveEqual(outer_bindings->operator[](i), NormalizeIterMapToExpr(outer_binding)));
    } else {
      outer_iter = IterVar(/*dom=*/RangeFromExtent(outer_mark->extent),
                           /*var=*/iter_var->var.copy_with_suffix("_o"),
                           /*iter_type=*/iter_var->iter_type);
      outer_bindings->push_back(NormalizeIterMapToExpr(outer_binding));
      outer_iter_vars->push_back(outer_iter);
    }
    PrimExpr sub{nullptr};
    if (is_one(inner_mark->extent)) {
      // Skip inner var when extent is 1
      // substitution
      if (is_one(outer_mark->extent) && !preserve_unit_iters) {
        // Simplify outer if not preserve_unit_iters
        sub = make_zero(outer_mark->extent.dtype());
      } else {
        sub = outer_iter;
      }
    } else {
      // create iter var for the inner block
      IterVar inner_iter(/*dom=*/RangeFromExtent(inner_mark->extent),
                         /*var=*/iter_var->var.copy_with_suffix("_i"),
                         /*iter_type=*/iter_var->iter_type);
      inner_bindings->push_back(NormalizeIterMapToExpr(inner_binding));
      inner_iter_vars->push_back(inner_iter);
      // substitution
      if (is_one(outer_mark->extent)) {
        sub = inner_iter->var;
      } else {
        sub = outer_iter * inner_mark->extent + inner_iter->var;
      }
    }
    block_var_subst.Set(iter_var->var, sub);
  }
  return block_var_subst;
}

/*!
 * \brief Generate the inner block for blockization
 * \param is_write_reduction Whether the write regions of the inner block are actually reduction.
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
  Map<Var, Var> subst_map;
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
      return std::move(tgt);
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
 * \brief Get the union of the given regions
 * \param regions The input regions for the union.
 * \return The union regions
 */
Array<BufferRegion> UnionRegions(const Array<BufferRegion>& regions) {
  typedef std::vector<Array<arith::IntSet>> ranges_t;
  std::unordered_map<Buffer, ranges_t, ObjectPtrHash, ObjectPtrEqual> intset_map;
  for (const BufferRegion& buffer_region : regions) {
    const Buffer& buffer = buffer_region->buffer;
    if (intset_map.find(buffer) == intset_map.end()) {
      intset_map[buffer] = {buffer->shape.size(), Array<arith::IntSet>()};
    }
    std::vector<Array<arith::IntSet>> dim_range(buffer->shape.size(), Array<arith::IntSet>());
    for (size_t dim = 0; dim < buffer->shape.size(); ++dim) {
      intset_map[buffer][dim].push_back(arith::IntSet::FromRange(buffer_region->region[dim]));
    }
  }
  Array<BufferRegion> results;
  for (const auto& it : intset_map) {
    const Buffer& buffer = it.first;
    Array<Range> regions;
    for (size_t dim = 0; dim < buffer->shape.size(); ++dim) {
      const arith::IntSet intset = arith::Union(it.second[dim]);
      regions.push_back({intset.min(), intset.max() + 1});
    }
    results.push_back(BufferRegion(buffer, regions));
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

BlockRealize BlockizeImpl(const ScheduleState& self, const StmtSRef& loop_sref,
                          Map<Block, Block>* block_sref_reuse, arith::Analyzer* analyzer,
                          bool preserve_unit_iters) {
  TVM_SREF_TO_FOR(loop_sref);
  // Step 1: Check and get the only block under `loop`.
  BlockRealize block_realize = CheckGetSingleChildBlockRealizeOnSRefTree(self, loop_sref);
  Block block = block_realize->block;
  StmtSRef block_sref = self->stmt2ref.at(block.get());
  // Step 2: Derive subspace division
  std::vector<const ForNode*> loops;
  Array<Array<arith::IterMark>> division =
      SubspaceDivide(block_realize, block_sref, loop_sref, &loops, analyzer, preserve_unit_iters);
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
                         &inner_iter_vars, &inner_bindings,  //
                         preserve_unit_iters);
  // Step 4: Do var substitution to adjust to the new block bindings
  Map<Var, arith::IntSet> inner_iter_dom;
  for (const IterVar& iter : inner_iter_vars) {
    inner_iter_dom.Set(iter->var, arith::IntSet::FromRange(iter->dom));
    analyzer->Bind(iter->var, iter->dom);
  }
  Block block_subst =
      Downcast<Block>(Substitute(block, block_var_subst, block_sref_reuse, analyzer));
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
  block_sref_reuse->Set(block, inner_realize->block);
  // Step 6: Generate the outer block.
  return BlockRealize(
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
}

StmtSRef Blockize(ScheduleState self, const StmtSRef& loop_sref, bool preserve_unit_iters) {
  arith::Analyzer analyzer;
  Map<Block, Block> block_sref_reuse;
  BlockRealize blockized =
      BlockizeImpl(self, loop_sref, &block_sref_reuse, &analyzer, preserve_unit_iters);
  self->Replace(loop_sref, blockized, block_sref_reuse);
  StmtSRef result = self->stmt2ref.at(blockized->block.get());
  StmtSRef scope_root = tir::GetScopeRoot(self, result, /*require_stage_pipeline=*/false);
  bool scope_block_affine_binding = self->IsAffineBlockBinding(scope_root);
  self->UpdateScopeBlockInfo(tir::GetBlockRealize(self, scope_root));
  self->block_info[scope_root].affine_binding = scope_block_affine_binding;
  return result;
}

BlockRealize BlockizeBlocks(const ScheduleState& self, const Array<StmtSRef>& block_srefs,
                            const StmtSRef& lca, Map<Block, Block>* block_sref_reuse,
                            bool preserve_unit_iters) {
  Array<Stmt> seq_body;
  PrimExpr outer_predicate{nullptr};
  Array<IterVar> outer_iter_vars{nullptr};
  Array<PrimExpr> outer_bindings{nullptr};
  Array<BufferRegion> read_regions;
  Array<BufferRegion> write_regions;
  std::string outer_block_name = "outer_";
  Map<Var, Var> loop_var_subst;
  arith::Analyzer analyzer;
  for (const auto& block_sref : block_srefs) {
    auto block_realize = GetBlockRealize(self, block_sref);
    auto block = block_realize->block;
    // Step 1: Derive subspace division
    std::vector<const ForNode*> loops;
    Array<Array<arith::IterMark>> division = SubspaceDivide(block_realize, block_sref, lca, &loops,
                                                            &analyzer, preserve_unit_iters, true);
    if (division.empty()) {
      throw SubspaceNotDivisibleError(self->mod, GetRef<For>(loops.back()), block);
    }
    outer_predicate = division.back()[0]->extent;
    PrimExpr inner_predicate = division.back()[1]->extent;
    // Step 2. Derive block bindings for both outer and inner block.
    Array<IterVar> inner_iter_vars;
    Array<PrimExpr> inner_bindings;
    Map<Var, PrimExpr> block_var_subst =                       //
        DeriveBlockBinding(block->iter_vars, division,         //
                           &outer_iter_vars, &outer_bindings,  //
                           &inner_iter_vars, &inner_bindings,  //
                           preserve_unit_iters, outer_iter_vars.defined());
    // Step 3: Do var substitution to adjust to the new block bindings
    for (size_t i = 0; i < outer_iter_vars.size(); ++i) {
      if (outer_bindings[i].as<Var>()) {
        loop_var_subst.Set(Downcast<Var>(outer_bindings[i]), outer_iter_vars[i]->var);
      }
    }
    Map<Var, arith::IntSet> inner_iter_dom;
    for (const IterVar& iter : inner_iter_vars) {
      Range dom = Substitute(iter->dom, loop_var_subst);
      inner_iter_dom.Set(iter->var, arith::IntSet::FromRange(dom));
      analyzer.Bind(iter->var, dom);
    }
    Block block_subst =
        Downcast<Block>(Substitute(block, block_var_subst, block_sref_reuse, &analyzer));
    auto reads = EvalSetRegions(block_subst->reads, inner_iter_dom);
    auto writes = EvalSetRegions(block_subst->writes, inner_iter_dom);
    read_regions.insert(read_regions.end(), reads.begin(), reads.end());
    write_regions.insert(write_regions.end(), writes.begin(), writes.end());
    outer_block_name += block_subst->name_hint + "_";
    // Step 4: Generate the inner block. No reduction iter vars allowed for the outer loops.
    bool has_outer_reduction = false;
    if (block_subst->init.defined()) {
      for (const IterVar& iter_var : outer_iter_vars) {
        if (iter_var->iter_type == kCommReduce) {
          has_outer_reduction = true;
          break;
        }
      }
    }
    ICHECK(has_outer_reduction == false)
        << "No reduction iter vars allowed for the outer loops when blockize multiple blocks";
    BlockRealize inner_realize = GenerateInner(/*is_write_reduction=*/has_outer_reduction,
                                               /*iter_vars=*/inner_iter_vars,
                                               /*iter_values*/ inner_bindings,
                                               /*predicate=*/inner_predicate,
                                               /*block=*/block_subst);
    block_sref_reuse->Set(block, inner_realize->block);
    Stmt stmt = inner_realize;
    for (const ForNode* loop : loops) {
      ObjectPtr<ForNode> new_loop = make_object<ForNode>(*loop);
      new_loop->body = std::move(stmt);
      new_loop->extent = Substitute(new_loop->extent, loop_var_subst);
      stmt = For(new_loop);
    }
    seq_body.push_back(stmt);
  }
  // Step 5: Generate the outer block.
  return BlockRealize(
      /*iter_values=*/std::move(outer_bindings),
      /*predicate=*/std::move(outer_predicate),
      /*block=*/
      Block(/*iter_vars=*/std::move(outer_iter_vars),
            /*reads=*/UnionRegions(read_regions),
            /*writes=*/UnionRegions(write_regions),
            /*name_hint=*/outer_block_name,
            /*body=*/SeqStmt(seq_body),
            /*init=*/Optional<Stmt>(NullOpt)));
}

class BlockizeRewriter : public StmtMutator {
 public:
  static Stmt Rewrite(const StmtSRef& lca, const Array<StmtSRef>& blocks,
                      const BlockRealize& blockized) {
    BlockizeRewriter rewriter(lca, blocks, blockized);
    return rewriter(GetRef<Stmt>(lca->stmt));
  }

 private:
  explicit BlockizeRewriter(const StmtSRef& lca, const Array<StmtSRef>& blocks,
                            const BlockRealize& blockized)
      : lca_(lca), blocks_(blocks), blockized_(blockized) {}

  Stmt RewriteSeq(const Stmt& stmt) {
    const SeqStmtNode* seq = stmt.as<SeqStmtNode>();
    ICHECK(seq) << "Target blocks must not be nested with each other!";
    int idx_start = -1;
    int last_found_idx = -1;
    size_t cur_idx = 0;
    Array<Stmt> new_seq;
    for (const Stmt& it : seq->seq) {
      target_in_ = false;
      Stmt stmt = StmtMutator::VisitStmt(it);
      if (target_in_) {
        if (idx_start == -1) {
          idx_start = cur_idx;
          new_seq.push_back(blockized_);
        } else {
          ICHECK_EQ(last_found_idx, cur_idx - 1) << "Target blocks must be consecutive!";
        }
        last_found_idx = cur_idx;
      } else {
        new_seq.push_back(it);
      }
      ++cur_idx;
    }
    if (new_seq.size() == 1) return new_seq[0];
    return SeqStmt(new_seq, seq->span);
  }

  Stmt VisitStmt_(const ForNode* loop) final {
    if (loop == lca_->stmt) {
      return For(loop->loop_var, loop->min, loop->extent, loop->kind, RewriteSeq(loop->body),
                 loop->thread_binding, loop->annotations, loop->span);
    }
    return StmtMutator::VisitStmt_(loop);
  }

  Stmt VisitStmt_(const BlockNode* block) final {
    if (block == lca_->stmt) {
      return Block(block->iter_vars, block->reads, block->writes, block->name_hint,
                   RewriteSeq(block->body), block->init, block->alloc_buffers, block->match_buffers,
                   block->annotations, block->span);
    }
    for (const StmtSRef& block_sref : blocks_) {
      if (block_sref->stmt == block) {
        target_in_ = true;
        break;
      }
    }
    return GetRef<Stmt>(block);
  }

  StmtSRef lca_;
  Array<StmtSRef> blocks_;
  BlockRealize blockized_;
  bool target_in_ = false;
};

StmtSRef Blockize(ScheduleState self, const Array<StmtSRef>& blocks, bool preserve_unit_iters) {
  Map<Block, Block> block_sref_reuse;
  auto lca = GetSRefLowestCommonAncestor(blocks);
  BlockRealize blockized =
      BlockizeBlocks(self, blocks, lca, &block_sref_reuse, preserve_unit_iters);
  auto new_root = BlockizeRewriter::Rewrite(lca, blocks, blockized);
  self->Replace(lca, new_root, block_sref_reuse);
  StmtSRef result = self->stmt2ref.at(blockized->block.get());
  StmtSRef scope_root = tir::GetScopeRoot(self, result, /*require_stage_pipeline=*/false);
  self->UpdateScopeBlockInfo(tir::GetBlockRealize(self, scope_root));
  return result;
}

void Tensorize(ScheduleState self, const StmtSRef& sref, const TensorIntrin& intrin,
               bool preserve_unit_iters) {
  // Step 1: Blockize the subtree rooted at the given loop if needed
  BlockRealize block_realize{nullptr};
  Optional<Block> old_block = NullOpt;
  if (sref->stmt->IsInstance<BlockNode>()) {
    block_realize = GetBlockRealize(self, sref);
    old_block = block_realize->block;
  } else if (sref->stmt->IsInstance<ForNode>()) {
    arith::Analyzer analyzer;
    Map<Block, Block> block_sref_reuse;
    block_realize = BlockizeImpl(self, sref, &block_sref_reuse, &analyzer, preserve_unit_iters);
  } else {
    LOG(FATAL) << "TypeError: Tensorize only support For or Block, but gets: "
               << GetRef<Stmt>(sref->stmt);
    throw;
  }
  PrimFunc intrin_desc = intrin->desc;
  PrimFunc intrin_impl = DeepCopy(intrin->impl);

  int index_dtype_bits = -1;
  auto f_update_max_dtype_bits_from_region = [&](const Array<BufferRegion>& buffer_regions) {
    for (const BufferRegion& buffer_region : buffer_regions) {
      for (const auto& range : buffer_region->region) {
        index_dtype_bits = std::max(index_dtype_bits, range->min.dtype().bits());
      }
    }
  };
  f_update_max_dtype_bits_from_region(block_realize->block->reads);
  f_update_max_dtype_bits_from_region(block_realize->block->writes);
  ICHECK(index_dtype_bits > 0);
  intrin_impl = IndexDataTypeNormalizer(DataType::Int(index_dtype_bits)).Rewrite(intrin_impl);
  // Step 2: Structural pattern matching
  TensorizeComparator comparator(self->mod, /*assert_mode=*/true);
  comparator.VisitStmt(block_realize, intrin_desc->body);
  // Step 3: Prepare necessary mapping
  // 1) Buffer mapping from intrin impl buffers to intrin desc buffers.
  // 2) Buffer mapping from intrin impl buffers to buffers in the current AST.
  // 3) Mapping impl buffers to their accessed regions.
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> impl2desc;
  ICHECK_EQ(intrin_desc->params.size(), intrin_impl->params.size());
  for (int i = 0, n = intrin_desc->params.size(); i < n; ++i) {
    const Buffer& desc = intrin_desc->buffer_map[intrin_desc->params[i]];
    const Buffer& impl = intrin_impl->buffer_map[intrin_impl->params[i]];
    impl2desc[impl] = desc;
  }
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> impl2cur;
  for (const auto& pair : impl2desc) {
    const Buffer& impl = pair.first;
    const Buffer& desc = pair.second;
    ICHECK(comparator.rhs_buffer_map_.count(desc));
    impl2cur[impl] = comparator.rhs_buffer_map_[desc];
  }
  std::unordered_map<Buffer, Array<Range>, ObjectPtrHash, ObjectPtrEqual> impl2region;
  Block impl_block = Downcast<BlockRealize>(intrin_impl->body)->block;
  for (const BufferRegion& read : impl_block->reads) {
    impl2region.emplace(read->buffer, read->region);
  }
  for (const BufferRegion& write : impl_block->writes) {
    impl2region.emplace(write->buffer, write->region);
  }
  // Step 4: Create MatchBufferRegion for the params of the impl function of the tensor
  // intrin to make them subregions of the buffer in the original IR.
  Array<MatchBufferRegion> match_buffer_regions;
  match_buffer_regions.reserve(intrin_impl->params.size());
  for (int i = 0, n = intrin_impl->params.size(); i < n; ++i) {
    const Buffer& impl = intrin_impl->buffer_map.at(intrin_impl->params[i]);
    const Buffer& cur = impl2cur.at(impl);
    const Array<Range>& old_region = impl2region.at(impl);
    const std::vector<PrimExpr>& indices_base = comparator.buffer_indices_.at(cur);
    int offset = static_cast<int>(indices_base.size()) - static_cast<int>(old_region.size());
    ICHECK(offset >= 0);
    Array<Range> new_region;
    new_region.reserve(cur->shape.size());
    for (int i = 0; i < offset; i++) {
      PrimExpr min = indices_base[i];
      PrimExpr extent = make_const(min.dtype(), 1);
      new_region.push_back(Range::FromMinExtent(min, extent));
    }
    for (int i = 0; i < static_cast<int>(old_region.size()); i++) {
      PrimExpr min = indices_base[i + offset];
      PrimExpr extent = cast(min.dtype(), old_region[i]->extent);
      new_region.push_back(Range::FromMinExtent(min, extent));
    }
    match_buffer_regions.push_back(MatchBufferRegion(impl, BufferRegion(cur, new_region)));
  }
  // Step 5: Replace the subtree in the original IR with the tensor intrin impl.
  {
    BlockNode* block = block_realize.CopyOnWrite()->block.CopyOnWrite();
    block->body = impl_block->body;
    block->match_buffers = std::move(match_buffer_regions);
    for (const auto& [key, val] : impl_block->annotations) {
      if (block->annotations.count(key) && block->annotations[key] != val) {
        LOG(WARNING) << "Conflict of annotation \"" << key << "\". Tensor intrinsic and schedule "
                     << "has different values : " << block->annotations[key] << " vs " << val << " "
                     << "The value from tensor intrinsic is skipped.";
        continue;
      }
      block->annotations.Set(key, val);
    }
  }
  if (old_block.defined()) {
    self->Replace(sref, block_realize->block, {{old_block.value(), block_realize->block}});
  } else {
    self->Replace(sref, block_realize, {});
  }
  // Step 6: Update the cached flags.
  StmtSRef result = self->stmt2ref.at(block_realize->block.get());
  StmtSRef scope_root = tir::GetScopeRoot(self, result, /*require_stage_pipeline=*/false);
  self->UpdateScopeBlockInfo(scope_root->StmtAs<BlockNode>()->body);
}

/******** InstructionKind Registration ********/

struct BlockizeTraits : public UnpackedInstTraits<BlockizeTraits> {
  static constexpr const char* kName = "Blockize";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static BlockRV UnpackedApplyToSchedule(Schedule sch, ObjectRef target, Bool preserve_unit_iters) {
    if (auto loop = target.as<LoopRV>()) {
      return sch->Blockize(loop.value(), preserve_unit_iters.operator bool());
    } else if (auto blocks = target.as<Array<BlockRV>>()) {
      return sch->Blockize(blocks.value(), preserve_unit_iters.operator bool());
    }
    LOG(FATAL) << "TypeError: expect Loop or list of Blocks, but gets:" << target->GetTypeKey();
  }

  static String UnpackedAsPython(Array<String> outputs, ObjectRef target,
                                 Bool preserve_unit_iters) {
    PythonAPICall py("blockize");
    py.Input("target", target);
    py.Input("preserve_unit_iters", preserve_unit_iters.operator bool());
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
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, ObjectRef block_or_loop_rv, String intrin,
                                      Bool preserve_unit_iters) {
    if (auto block = block_or_loop_rv.as<BlockRV>()) {
      sch->Tensorize(block.value(), intrin, preserve_unit_iters.operator bool());
    } else if (auto loop = block_or_loop_rv.as<LoopRV>()) {
      sch->Tensorize(loop.value(), intrin, preserve_unit_iters.operator bool());
    } else {
      LOG(FATAL) << "TypeError: Expected Block or Loop, but gets: "
                 << block_or_loop_rv->GetTypeKey();
    }
  }

  static String UnpackedAsPython(Array<String> outputs, String block_or_loop_rv, String intrin,
                                 Bool preserve_unit_iters) {
    PythonAPICall py("tensorize");
    py.Input("block_or_loop", block_or_loop_rv);
    py.Input("tensor_intrin", intrin);
    py.Input("preserve_unit_iters", preserve_unit_iters.operator bool());
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(BlockizeTraits);
TVM_REGISTER_INST_KIND_TRAITS(TensorizeTraits);

}  // namespace tir
}  // namespace tvm
