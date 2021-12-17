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

#include "../utils.h"

namespace tvm {
namespace tir {

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
 * \param predicate The predicate constaints on the input iterators.
 * \return The result of the subspace division.
 */
Array<Array<arith::IterMark>> TrivialSubspaceDivision(const Array<IterVar>& iter_vars,
                                                      const Array<PrimExpr>& bindings,
                                                      const Array<Var>& outer_iters,
                                                      const Array<Var>& inner_iters,
                                                      const PrimExpr& predicate) {
  if (!is_one(predicate)) return {};
  std::vector<Array<arith::IterMark>> res;
  std::unordered_set<const VarNode*> outer_loop_vars;
  std::unordered_set<const VarNode*> inner_loop_vars;
  for (const Var& var : outer_iters) {
    outer_loop_vars.insert(var.get());
  }
  for (const Var& var : inner_iters) {
    inner_loop_vars.insert(var.get());
  }
  for (size_t i = 0; i < bindings.size(); ++i) {
    bool outer = UsesVar(
        bindings[i], [&outer_loop_vars](const VarNode* var) { return outer_loop_vars.count(var); });
    bool inner = UsesVar(
        bindings[i], [&inner_loop_vars](const VarNode* var) { return inner_loop_vars.count(var); });
    arith::IterMark iter_mark;
    if (bindings[i]->IsInstance<VarNode>()) {
      iter_mark = arith::IterMark(
          arith::IterSplitExpr(arith::IterMark(bindings[i], iter_vars[i]->dom->extent)),
          iter_vars[i]->dom->extent);
    } else {
      iter_mark = arith::IterMark(arith::IterSumExpr({}, bindings[i]), iter_vars[i]->dom->extent);
    }
    if (outer && !inner) {
      arith::IterMark outer{nullptr};
      const auto& outer_iter = iter_mark;
      arith::IterMark inner_iter(arith::IterSumExpr({}, 0), 1);
      res.push_back(Array<arith::IterMark>({outer_iter, inner_iter}));
    } else if (inner && !outer) {
      const auto& inner_iter = iter_mark;
      arith::IterMark outer_iter(arith::IterSumExpr({}, 0), 1);
      res.push_back(Array<arith::IterMark>({outer_iter, inner_iter}));
    } else if (!outer && !inner) {
      arith::IterMark outer_iter(arith::IterSumExpr({}, 0), 1);
      arith::IterMark inner_iter(arith::IterSumExpr({}, 0), 1);
      res.push_back(Array<arith::IterMark>({outer_iter, inner_iter}));
    } else {
      return {};
    }
  }
  res.push_back({arith::IterMark(arith::IterSumExpr({}, 0), Bool(true)),
                 arith::IterMark(arith::IterSumExpr({}, 0), Bool(true))});
  return res;
}

class SubspaceNotDivisibleError : public ScheduleError {};

/*!
 * \brief Regenerate outer loops of a statement
 * \param
 */
Stmt RegenerateLoops(const std::vector<const ForNode*>& loops, Stmt body) {
  for (const ForNode* loop : loops) {
    ObjectPtr<ForNode> new_loop = make_object<ForNode>(*loop);
    new_loop->body = std::move(body);
    body = For(new_loop);
  }
  return body;
}

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
                [&iter_var](const VarNode* var) { return var == iter_var->var.get(); })) {
      init_block_iters.push_back(iter_var);
      init_bindings.push_back(binding);
    }
  }

  // Step 2: Collect loops related to iters of the init block
  std::vector<const ForNode*> init_loops;
  for (const ForNode* inner_loop : inner_loops) {
    for (const PrimExpr& init_binding : init_bindings) {
      if (UsesVar(init_binding,
                  [inner_loop](const VarNode* var) { return var == inner_loop->loop_var.get(); })) {
        init_loops.push_back(inner_loop);
      }
    }
  }

  // Step 3: Create new block iters for the init block
  Map<Var, PrimExpr> subst_map;
  for (size_t i = 0; i < init_block_iters.size(); i++) {
    IterVar new_iter_var = init_block_iters[i];
    auto* new_init_var_node = new_iter_var.CopyOnWrite();
    Var old_var = new_iter_var->var;
    new_init_var_node->var = old_var.copy_with_suffix("_init");
    subst_map.Set(old_var, new_iter_var->var);
    init_block_iters.Set(i, std::move(new_iter_var));
  }

  // Step 4: Generate loop nests and the init block
  Block init_block{/*iter_vars=*/init_block_iters,            //
                   /*reads=*/{},                              //
                   /*writes=*/block->writes,                  //
                   /*name_hint=*/block->name_hint + "_init",  //
                   /*body=*/block->init.value(),              //
                   /*init=*/NullOpt};
  Stmt new_init = BlockRealize(
          /*iter_values=*/init_bindings,
          /*predicate=*/inner_block_realize->predicate,
                        /*block=*/          std::move(init_block)
          );

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
   * \param loop_sref The sref to the separator loop.
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
 * \param block_realize The block realize to be checked.
 * \param collector The collector which has collected the loops of the block.
 * \param analyzer The arithmetic analyzer.
 * \return The result of the subspace division.
 * \throws ScheduleError If the bindings are not divisible by the subspace.
 */
Array<Array<arith::IterMark>> CheckSubspaceDivisible(const BlockRealize& block_realize,
                                                     const LoopSubspaceCollector& collector,
                                                     arith::Analyzer* analyzer) {
  const Block& block = block_realize->block;

  Array<Array<arith::IterMark>> division =
      arith::SubspaceDivide(block_realize->iter_values, collector.loop_var_domain,
                            collector.inner_loop_vars, block_realize->predicate,
                            /*require_bijective=*/false, analyzer);

  if (division.empty()) {
    // If we can't do perfect subspace division, check if it is a trivial case of subspace division.
    // In this case, we can still blockize.
    division = TrivialSubspaceDivision(block->iter_vars, block_realize->iter_values,
                                       collector.outer_loop_vars, collector.inner_loop_vars,
                                       block_realize->predicate);
  }
  // TODO: raise schedule error
  CHECK(!division.empty()) << "ValueError: The bindings of the block below can not be blockized";
  return division;
}

class BlockizedBindingExtractor {
 public:
  void ExtractBindings(const Array<IterVar>& iter_vars,
                       const Array<Array<arith::IterMark>>& division) {
    ICHECK(iter_vars.size() + 1 == division.size());
    for (size_t i = 0; i < iter_vars.size(); ++i) {
      const IterVar& iter_var = iter_vars[i];
      const arith::IterMapExprNode* outer_binding =
          division[i][0]->source.as<arith::IterMapExprNode>();
      const arith::IterMapExprNode* inner_binding =
          division[i][1]->source.as<arith::IterMapExprNode>();
      ICHECK(outer_binding);
      ICHECK(inner_binding);

      // After computing the subspace division, bindings[i] can be written as
      // outer_binding * inner_binding->extent + inner_binding
      // The outer block will have binding: iter_outer -> outer_binding
      // The inner block will have binding: iter_inner -> iter_outer * inner_binding->extent +
      // inner_binding

      if (is_one(division[i][1]->extent)) {  // IsOuter
        // extract this iter var to outer block directly
        outer_bindings.push_back(
            arith::NormalizeIterMapToExpr(GetRef<arith::IterMapExpr>(outer_binding)));
        outer_iter_vars.push_back(iter_var);
      } else {
        const IterVar outer_var(Range::FromMinExtent(0, division[i][0]->extent),
                                iter_var->var.copy_with_suffix("o"), iter_var->iter_type);
        outer_bindings.push_back(
            arith::NormalizeIterMapToExpr(GetRef<arith::IterMapExpr>(outer_binding)));
        outer_iter_vars.push_back(outer_var);
        // generate a new iter var for outer block
        // TODO: add test case outer extent is zero
        PrimExpr base = is_one(division[i][0]->extent) ? 0 : outer_var * division[i][1]->extent;
        if (const auto* op = division[i][1]->source.as<arith::IterSumExprNode>()) {
          base = base + op->base;
          inner_bindings.push_back(base +
                                   arith::NormalizeIterMapToExpr(arith::IterSumExpr(op->args, 0)));
        } else {
          inner_bindings.push_back(
              base + arith::NormalizeIterMapToExpr(GetRef<arith::IterMapExpr>(inner_binding)));
        }
        inner_iter_vars.push_back(iter_var);
        // bv_iter: inner block iter -> division inner extent
        inner_iter_relaxed_range.Set(iter_var->var,
                                     Range::FromMinExtent(base, division[i][1]->extent));
      }
    }
  }
  /*! \brief Iters of the outer block. */
  Array<IterVar> outer_iter_vars;
  /*! \brief Iters of the outer block. */
  Array<IterVar> inner_iter_vars;
  /*! \brief Binding values of the outer block. */
  Array<PrimExpr> outer_bindings;
  /*! \brief Binding values of the inner block. */
  Array<PrimExpr> inner_bindings;

  /*! \brief The range of the inner block iters Note that this is different from the domain of the
   * inner block iters. */
  Map<Var, Range> inner_iter_relaxed_range;
};


/*!
 * \brief
 */
BufferRegion RelaxBlockizedInnerIters(const BufferRegion& buffer_region,
                                      const Map<Var, Range>& inner_iter_relaxed_range,
                                      arith::Analyzer* analyzer) {
  Array<Range> new_region;
  new_region.reserve(buffer_region->region.size());
  for (const auto& range : buffer_region->region) {
    const Array<arith::IterSumExpr>& res =
        arith::DetectIterMap({range->min}, inner_iter_relaxed_range, true, false, analyzer);
    ICHECK_EQ(res.size(), 1);
    const arith::IterSumExpr& normalized_expr = res[0];
    PrimExpr extent = 1;
    if (normalized_expr->args.size() == 1) {
      ICHECK(analyzer->CanProve(normalized_expr->args[0]->scale - range->extent == 0));
      extent = normalized_expr->args[0]->extent;
    }
    new_region.push_back(Range::FromMinExtent(normalized_expr->base, extent * range->extent));
  }
  return BufferRegion(buffer_region->buffer, std::move(new_region));
};

BlockRealize GenerateBlockizedOuterBlock(const BlockizedBindingExtractor& extractor,
                                         const Block& block, BlockRealize inner_block_realize,
                                         const std::vector<const ForNode*>& inner_loops,
                                         PrimExpr predicate, arith::Analyzer* analyzer) {
  // Step 1: Generate the init block if needed
  Optional<Stmt> new_init = NullOpt;
  if (block->init.defined()) {
    new_init = GenerateBlockizedInit(block, inner_block_realize, inner_loops);
  }

  // Step 2: Compute the access regions of the outer block by relaxing the inner loops
  Array<BufferRegion> new_reads = block->reads;
  Array<BufferRegion> new_writes = block->writes;

  auto f_mutate = [&](const BufferRegion& buffer_region) {
    return RelaxBlockizedInnerIters(buffer_region, extractor.inner_iter_relaxed_range, analyzer);
  };
  new_reads.MutateByApply(f_mutate);
  new_writes.MutateByApply(f_mutate);

  Stmt outer_block_body = RegenerateLoops(inner_loops, inner_block_realize);
  Block outer_block{/*iter_vars=*/extractor.outer_iter_vars,        //
                    /*reads=*/new_reads,                            //
                    /*writes=*/new_writes,                          //
                    /*name_hint=*/"blockized_" + block->name_hint,  //
                    /*body=*/std::move(outer_block_body),           //
                    /*init=*/new_init};
  BlockRealize outer_block_realize{/*iter_values=*/extractor.outer_bindings,
                                   /*predicate=*/std::move(predicate),
                                   /*block=*/std::move(outer_block)};
  return outer_block_realize;
}

StmtSRef Blockize(ScheduleState self, const StmtSRef& loop_sref) {
  /*!
   * Check:
   *   - The sub AST is one-line with only one block
   *
   * Mutate:
   *   - extra block var from the only block
   *   - Update block binding
   */
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
      CheckSubspaceDivisible(block_realize, collector, &analyzer);

  // Step 4: Generate bindings for the outer block and the inner block based on the result of
  // the subspace division.
  BlockizedBindingExtractor extractor;
  extractor.ExtractBindings(block->iter_vars, division);
  const PrimExpr& outer_pred = division.back()[0]->extent;
  const PrimExpr& inner_pred = division.back()[1]->extent;

  // Step 5: Generate the inner block.
  BlockRealizeNode* inner_block_realize = block_realize.CopyOnWrite();
  BlockNode* inner_block = inner_block_realize->block.CopyOnWrite();
  inner_block_realize->iter_values = extractor.inner_bindings;
  inner_block_realize->predicate = inner_pred;
  inner_block->iter_vars = extractor.inner_iter_vars;
  inner_block->init = NullOpt;

  // Step 6: Generate the outer block.
  BlockRealize outer_realize =
      GenerateBlockizedOuterBlock(extractor, block, GetRef<BlockRealize>(inner_block_realize),
                                  collector.inner_loops, outer_pred, &analyzer);
  // Step 7: Do the actual replacement
  self->Replace(loop_sref, outer_realize, {{block, GetRef<Block>(inner_block)}});

  // Step 8: Update the cached flags
  const StmtSRef& outer_block_sref = self->stmt2ref.at(outer_realize->block.get());
  BlockInfo& outer_block_info = self->block_info[outer_block_sref];
  const BlockInfo& inner_block_info = self->block_info.at(block_sref);
  outer_block_info.affine_binding = inner_block_info.affine_binding;
  outer_block_info.region_cover = inner_block_info.region_cover;
  outer_block_info.scope->stage_pipeline = inner_block_info.scope->stage_pipeline;

  return outer_block_sref;
}

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
  friend struct UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(BlockizeTraits);

}  // namespace tir
}  // namespace tvm
