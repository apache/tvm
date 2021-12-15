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
#include "../../../arith/pattern_match.h"
#include "../../ir/functor_common.h"
#include "../utils.h"

namespace tvm {
namespace tir {

/*!
 * \brief Detect if bindings are a trivial case of the subspace division where we can divide the 
 * block iter bindings into two categories:
 *   1. The binding covers no inner loop vars.
 *   2. The binding covers only inner loop vars.
 * This case doesn't require bindings to be quasi-affine.
 *
 * \param
 * \param
 * \return The result of the subspace division.
 */
Array<Array<arith::IterMark>> TrivialSubspaceDivision(const Array<IterVar>& iter_vars,
                                                      const Array<PrimExpr>& bindings,
                                                      const Array<Var>& outer_loops,
                                                      const Array<Var>& inner_loops,
                                                      const PrimExpr& predicate) {
  if (!is_one(predicate)) return {};
  std::vector<Array<arith::IterMark>> res;
  std::unordered_set<const VarNode*> outer_loop_vars;
  std::unordered_set<const VarNode*> inner_loop_vars;
  for (const Var& var : outer_loops) {
    outer_loop_vars.insert(var.get());
  }
  for (const Var& var : inner_loops) {
    inner_loop_vars.insert(var.get());
  }
  for (size_t i = 0; i < bindings.size(); ++i) {
    bool outer = UsesVar(
        bindings[i], [&outer_loop_vars](const VarNode* var) { return outer_loop_vars.count(var); });
    bool inner = UsesVar(
        bindings[i], [&inner_loop_vars](const VarNode* var) { return inner_loop_vars.count(var); });
    bool is_var = bindings[i]->IsInstance<VarNode>();
    if (outer && !inner) {
      arith::IterMark outer{nullptr};
      if (is_var) {
        outer = arith::IterMark(
            arith::IterSplitExpr(arith::IterMark(bindings[i], iter_vars[i]->dom->extent)),
            iter_vars[i]->dom->extent);
      } else {
        outer = arith::IterMark(arith::IterSumExpr({}, bindings[i]), iter_vars[i]->dom->extent);
      }
      arith::IterMark inner(arith::IterSumExpr({}, 0), 1);
      res.push_back(Array<arith::IterMark>({outer, inner}));
    } else if (inner && !outer) {
      arith::IterMark inner{nullptr};
      if (is_var) {
        inner = arith::IterMark(
            arith::IterSplitExpr(arith::IterMark(bindings[i], iter_vars[i]->dom->extent)),
            iter_vars[i]->dom->extent);
      } else {
        inner = arith::IterMark(arith::IterSumExpr({}, bindings[i]), iter_vars[i]->dom->extent);
      }
      arith::IterMark outer(arith::IterSumExpr({}, 0), 1);
      res.push_back(Array<arith::IterMark>({outer, inner}));
    } else if (!outer && !inner) {
      arith::IterMark outer(arith::IterSumExpr({}, 0), 1);
      arith::IterMark inner(arith::IterSumExpr({}, 0), 1);
      res.push_back(Array<arith::IterMark>({outer, inner}));
    } else {
      return {};
    }
  }
  res.push_back({arith::IterMark(arith::IterSumExpr({}, 0), Bool(true)),
                 arith::IterMark(arith::IterSumExpr({}, 0), Bool(true))});
  return res;
}

Stmt GenerateOuterInitBlock(const Array<IterVar>& inner_block_vars, Block block,
                            const std::vector<const ForNode*>& inner_loops,
                            const Array<PrimExpr>& inner_bindings,
                            const Array<Array<arith::IterMark>>& division) {
  std::vector<For> init_loops;
  Stmt new_init;
  std::vector<size_t> init_block_vars;
  std::vector<IterVar> init_block_vars_copy;
  std::vector<PrimExpr> init_bindings;
  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> binding_replace_map;
  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> bv_replace_map;
  std::unordered_map<const IterVarNode*, int> new_block_vars2old_index;
  for (size_t i = 0; i < inner_block_vars.size(); ++i) {
    if (inner_block_vars[i]->iter_type == IterVarType::kDataPar &&
        UsesVar(block->init.value(),
                [v = inner_block_vars[i]->var](const VarNode* var) { return var == v.get(); })) {
      // copy init block vars and ignore reduce block vars
      init_block_vars.push_back(i);
      IterVar init_block_var = inner_block_vars[i];
      init_block_var.CopyOnWrite()->var = inner_block_vars[i]->var.copy_with_suffix("_init");
      init_block_vars_copy.push_back(init_block_var);
      bv_replace_map[inner_block_vars[i]->var] = init_block_var->var;
      new_block_vars2old_index[init_block_var.get()] = i;
    }
  }
  for (const ForNode* inner_loop : inner_loops) {
    for (size_t i = 0; i < init_block_vars.size(); ++i) {
      if (UsesVar(inner_bindings[new_block_vars2old_index[init_block_vars_copy[i].get()]],
                  [v = inner_loop->loop_var](const VarNode* var) { return var == v.get(); })) {
        // copy loops related to init block vars
        For init_loop = GetRef<For>(inner_loop);
        init_loop.CopyOnWrite()->loop_var = inner_loop->loop_var.copy_with_suffix("");
        // replace loop vars with copied loop vars
        binding_replace_map[inner_loop->loop_var] = init_loop->loop_var;
        init_loops.push_back(init_loop);
        break;
      }
    }
  }
  for (size_t i = 0; i < init_block_vars.size(); ++i) {
    init_bindings.push_back(Substitute(inner_bindings[init_block_vars[i]], binding_replace_map));
  }
  new_init = Substitute(Block(/*iter_vars=*/init_block_vars_copy,        //
                              /*reads=*/{},                              //
                              /*writes=*/block->writes,                  //
                              /*name_hint=*/block->name_hint + "_init",  //
                              /*body=*/block->init.value(),              //
                              /*init=*/NullOpt),
                        bv_replace_map);
  new_init = BlockRealize(init_bindings, division.back()[1]->extent, Downcast<Block>(new_init));
  for (const auto& init_loop : init_loops) {
    For new_init_loop = init_loop;
    new_init_loop.CopyOnWrite()->body = new_init;
    new_init = new_init_loop;
  }
  return new_init;
}

// TODO
class SubspaceNotDivisibleError : public ScheduleError {};

std::array<std::pair<Array<IterVar>, Array<PrimExpr>>, 2> GenerateBlockIterVarBindings(
    Block block, const Array<Array<arith::IterMark>>& division,
    std::unordered_map<Var, Range, ObjectPtrHash, ObjectPtrEqual>& bv_iters) {
  Array<IterVar> inner_block_vars, outer_block_vars;
  Array<PrimExpr> inner_bindings, outer_bindings;
  for (size_t i = 0; i < block->iter_vars.size(); ++i) {
    const IterVar& iter_var = block->iter_vars[i];
    const arith::IterMapExprNode* outer_binding =
        division[i][0]->source.as<arith::IterMapExprNode>();
    const arith::IterMapExprNode* inner_binding =
        division[i][1]->source.as<arith::IterMapExprNode>();
    ICHECK(outer_binding);
    ICHECK(inner_binding);
    if (is_one(division[i][1]->extent)) {  // IsOuter
      // extract this iter var to outer block directly
      outer_bindings.push_back(
          arith::NormalizeIterMapToExpr(GetRef<arith::IterMapExpr>(outer_binding)));
      outer_block_vars.push_back(iter_var);
      // bv_iters[iter_var->var] = Range::FromMinExtent(0, division[i][0]->extent);
    } else {
      const IterVar outer_var(Range::FromMinExtent(0, division[i][0]->extent),
                              iter_var->var.copy_with_suffix("o"), iter_var->iter_type);
      outer_bindings.push_back(
          arith::NormalizeIterMapToExpr(GetRef<arith::IterMapExpr>(outer_binding)));
      outer_block_vars.push_back(outer_var);
      // generate a new iter var for outer block
      PrimExpr base = is_one(division[i][0]->extent) ? 0 : outer_var * division[i][1]->extent;
      if (const auto* op = division[i][1]->source.as<arith::IterSumExprNode>()) {
        base = base + op->base;
        inner_bindings.push_back(base +
                                 arith::NormalizeIterMapToExpr(arith::IterSumExpr(op->args, 0)));
      } else {
        inner_bindings.push_back(
            base + arith::NormalizeIterMapToExpr(GetRef<arith::IterMapExpr>(inner_binding)));
      }
      inner_block_vars.push_back(iter_var);
      bv_iters[iter_var->var] = Range::FromMinExtent(base, division[i][1]->extent);
    }
  }
  return {std::make_pair(outer_block_vars, outer_bindings),
          std::make_pair(inner_block_vars, inner_bindings)};
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
 * collector. A ScheduleError is raised if the bindings are not divisible by the subspace.
 * \param block_realize The block realize to be checked.
 * \param collector The collector which has collected the loops of the block.
 * \param analyzer The arithmetic analyzer.
 * \return The result of the subspace division.
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

class BlockizeBlockBuilder {
  
};
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

  // Step 4: Generate a new inner block
  // Step 4.1: Compute outer/inner block bindings
  std::unordered_map<Var, Range, ObjectPtrHash, ObjectPtrEqual>
      bv_iters;  // iter_vars of the inner block
  auto r = GenerateBlockIterVarBindings(block, division, bv_iters);
  Array<IterVar> outer_block_vars = r[0].first;
  Array<PrimExpr> outer_bindings = r[0].second;

  auto* inner_br = block_realize.CopyOnWrite();
  auto* inner_block = inner_br->block.CopyOnWrite();
  {
    inner_br->iter_values = r[1].second;
    inner_br->predicate = division.back()[1]->extent;
    inner_block->iter_vars = r[1].first;
    inner_block->init = NullOpt;
  }

  // Regenerate inner_loops
  Stmt body = GetRef<Stmt>(inner_br);
  for (const auto& inner_loop : collector.inner_loops) {
    auto loop_node = make_object<ForNode>(*inner_loop);
    loop_node->body = body;
    body = For(loop_node);
  }
  // Regenerate init for outer block
  Optional<Stmt> new_init = NullOpt;
  if (block->init.defined()) {
    new_init = GenerateOuterInitBlock(inner_block->iter_vars, block, collector.inner_loops,
                                      inner_br->iter_values, division);
  }

  // Calculate outer block's IO region
  auto rewrite_range = [&](const Range& range) -> Range {
    const Array<arith::IterSumExpr>& res =
        arith::DetectIterMap({range->min}, bv_iters, true, false, &analyzer);
    ICHECK_EQ(res.size(), 1);
    const arith::IterSumExpr& normalized_expr = res[0];
    PrimExpr extent = 1;
    if (normalized_expr->args.size() == 1) {
      CHECK(analyzer.CanProve(normalized_expr->args[0]->scale - range->extent == 0));
      extent = normalized_expr->args[0]->extent;
    }
    return Range::FromMinExtent(normalized_expr->base, extent * range->extent);
  };
  std::vector<BufferRegion> reads, writes;
  auto rewrite_region = [&](std::vector<BufferRegion>* regions, Array<BufferRegion> old_regions) {
    for (auto buffer_region : old_regions) {
      std::vector<Range> region;
      for (const auto& range : buffer_region->region) {
        region.push_back(rewrite_range(range));
      }
      (*regions).emplace_back(buffer_region->buffer, region);
    }
  };
  rewrite_region(&reads, block->reads);
  rewrite_region(&writes, block->writes);
  // Generate a new outer block
  auto outer_block = Block(/*iter_vars=*/outer_block_vars,                 //
                           /*reads=*/reads,                                //
                           /*writes=*/writes,                              //
                           /*name_hint=*/"blockized_" + block->name_hint,  //
                           /*body=*/std::move(body),                       //
                           /*init=*/new_init);
  auto outer_realize = BlockRealize(outer_bindings, division.back()[0]->extent, outer_block);

  // Step x: do actual replace
  self->Replace(loop_sref, outer_realize, {{block, GetRef<Block>(inner_block)}});
  {
    StmtSRef block_sref = self->stmt2ref.at(outer_block.get());
    StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false,
                                       /*require_compact_dataflow*/ false);
    // UpdateScope(self, scope_sref);
  }
  // RecalculateCachedFlags(self.operator->());

  // }
  // TODO(@wuwei): fix affine flags
  // self->Replace(loop_sref, outer_realize, {{block, inner_block}});
  // {
  //   StmtSRef block_sref = self->stmt2ref.at(inner_block.get());
  //   UpdateAffineFlag(self, block_sref);
  // }
  // {
  //   StmtSRef block_sref = self->stmt2ref.at(outer_block.get());
  //   StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false,
  //                                      /*require_compact_dataflow*/false);
  //   UpdateScope(self, scope_sref);
  //   UpdateAffineFlag(self, scope_sref);
  // }
  // {
  //   StmtSRef block_sref = self->stmt2ref.at(outer_block.get());
  //   UpdateScope(self, block_sref);
  //   UpdateAffineFlag(self, block_sref);
  // }

  // // Check loop binding

  // {
  //   struct BindingValidator : public StmtVisitor {
  //     void VisitStmt_(const BlockRealizeNode* realize) final {
  //       StmtSRef& sref = self->stmt2ref.at(realize->block.get());
  //       UpdateAffineFlag(self, sref);
  //       VisitStmt(realize->block->body);
  //     }
  //     ScheduleState self;
  //   };
  //   BindingValidator validator;
  //   validator.self = self;
  //   const PrimFuncNode* func = GetRootPrimFunc(self->mod, GetRootBlock(loop_sref).get(),
  //   nullptr); validator(func->body);
  // }
  return self->stmt2ref.at(outer_block.get());
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

  friend struct UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(BlockizeTraits);

}  // namespace tir
}  // namespace tvm
