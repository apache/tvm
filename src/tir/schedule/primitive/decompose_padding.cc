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
#include "../../transforms/ir_utils.h"
#include "../utils.h"

namespace tvm {
namespace tir {

/*! \brief Information used to create new padding block */
struct PaddingBlockInfo {
  /*! \brief In-bound block iter regions, wrt loop vars. */
  Array<Range> in_bound_region;
  /*! \brief In-bound value, wrt block iter vars. */
  PrimExpr in_bound_value;
  /*! \brief Condition of in-bound write, wrt loop vars. */
  PrimExpr in_bound_predicate;
  /*! \brief Padding value, should be a constant. */
  PrimExpr pad_value;
};

class PaddingPatternMatchError : public ScheduleError {
 public:
  PaddingPatternMatchError(IRModule mod, Block block, const std::string& error_msg)
      : mod_(std::move(mod)), block_(std::move(block)), error_msg_(error_msg) {}

  String FastErrorString() const final {
    return "ScheduleError: decompose_padding expect the block to match padding pattern\n  " +
           error_msg_;
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "ScheduleError: decompose_padding expect the block {0} to match padding pattern\n  "
       << error_msg_;
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

  IRModule mod_;
  Block block_;
  std::string error_msg_;
};

/*!
 * \brief Helper class to analyze and check the padding pattern of the block,
 * then return the padding information.
 */
class PaddingInfoAnalyzer {
 public:
  static PaddingBlockInfo CheckAndGetPaddingInfo(IRModule mod, const BlockRealizeNode* realize,
                                                 const Map<Var, Range>& dom_map,
                                                 arith::Analyzer* analyzer) {
    PaddingInfoAnalyzer padding_analyzer(analyzer);
    if (!padding_analyzer.MatchPadding(realize, dom_map)) {
      throw PaddingPatternMatchError(mod, realize->block, padding_analyzer.error_msg_);
    }
    return padding_analyzer.info_;
  }

 private:
  explicit PaddingInfoAnalyzer(arith::Analyzer* analyzer) : analyzer_(analyzer) {}

  /*! \brief Detect padding pattern and update result. */
  bool MatchPadding(const BlockRealizeNode* realize, const Map<Var, Range>& dom_map) {
    // Step 1. Check match padding computation pattern.
    // A[...] = T.if_then_else(predicate, B[...], imm)
    Block block = realize->block;
    std::unordered_map<const VarNode*, PrimExpr> iter_values;
    for (size_t i = 0; i < realize->iter_values.size(); ++i) {
      Var block_var = block->iter_vars[i]->var;
      iter_values[block_var.get()] = realize->iter_values[i];
    }
    const BufferStoreNode* store = block->body.as<BufferStoreNode>();
    if (!store) {
      SetError("Block body expect a BufferStore to the write buffer");
      return false;
    }
    const CallNode* if_then_else = store->value.as<CallNode>();
    if (!if_then_else || !if_then_else->op.same_as(tir::builtin::if_then_else())) {
      SetError("Value of BufferStore expect to be constrained by a padding predicate");
      return false;
    }
    PrimExpr pad_predicate = Substitute(if_then_else->args[0], iter_values);
    PrimExpr in_bound_value = if_then_else->args[1];
    PrimExpr pad_value = if_then_else->args[2];
    if (!is_const_number(pad_value)) {
      SetError("Pad value should be constant");
      return false;
    }

    // Step 2. Check in-bound computation to be effectiveless.
    if (SideEffect(if_then_else->args[1]) > CallEffectKind::kReadState) {
      SetError("Inbound computation should not have side-effect");
      return false;
    }

    // Step 3. Analyze in-bound write region.
    PrimExpr in_bound_predicate = RewritePredicate(pad_predicate && realize->predicate);
    if (analyzer_->CanProveEqual(in_bound_predicate, 1)) {
      SetError("The in-bound predicate is trivial");
      return false;
    }
    Array<Range> in_bound_region = this->EstimateInBoundRegion(
        /*iter_values=*/realize->iter_values, /*dom_map=*/dom_map,
        /*in_bound_predicate=*/in_bound_predicate);
    if (in_bound_region.empty()) {
      return false;
    }

    // Step 4. Update result information.
    info_.in_bound_value = if_then_else->args[1];
    info_.in_bound_region = in_bound_region;
    info_.in_bound_predicate = in_bound_predicate;
    info_.pad_value = pad_value;
    return true;
  }

  /*! \brief Rewrite predicate to left recursive conjunction, drop likely annotation. */
  PrimExpr RewritePredicate(const PrimExpr& predicate) {
    PrimExpr res = const_true();
    std::function<void(PrimExpr)> update = [&res, &update](PrimExpr e) {
      arith::PVar<PrimExpr> a, b;
      if ((a && b).Match(e)) {
        update(a.Eval());
        update(b.Eval());
      } else {
        if (const CallNode* call = e.as<CallNode>()) {
          if (call->op.same_as(builtin::likely())) {
            e = call->args[0];
          }
        }
        res = res && e;
      }
    };
    update(predicate);
    return analyzer_->Simplify(res);
  }

  /*! \brief Return iteration region of block vars where the padding predicate evals to true. */
  Array<Range> EstimateInBoundRegion(const Array<PrimExpr>& iter_values,
                                     const Map<Var, Range>& dom_map,
                                     const PrimExpr& in_bound_predicate) {
    Array<Range> region;

    auto res = arith::DetectIterMap(iter_values, dom_map, in_bound_predicate,
                                    arith::IterMapLevel::Surjective, analyzer_);
    if (res->indices.empty()) {
      SetError("Block iters are not independent wrt padding condition");
      return {};
    }
    for (const arith::IterSumExpr& sum : res->indices) {
      if (sum->args.empty()) {
        region.push_back(Range::FromMinExtent(sum->base, IntImm(sum->base.dtype(), /* value */ 1)));
      } else {
        ICHECK_EQ(sum->args.size(), 1U);
        if (!analyzer_->CanProveEqual(sum->args[0]->scale, 1)) {
          SetError("Strided iteration is not supported");
          return {};
        }
        region.push_back(Range::FromMinExtent(sum->base, sum->args[0]->extent));
      }
    }
    return region;
  }

  void SetError(const std::string& msg) { error_msg_ = msg; }

  /*! \brief padding info analyse result. */
  PaddingBlockInfo info_;
  /*! \brief current error message. */
  std::string error_msg_;
  /*! \brief arithmetic analyzer. */
  arith::Analyzer* analyzer_;
};

/*! \brief Create block to fill constant pad values into full region */
static std::pair<Stmt, BlockRealize> CreateConstBlock(const BlockRealizeNode* realize,
                                                      const PaddingBlockInfo& info,
                                                      const Array<For>& loops,
                                                      const Stmt& highest_pos_inclusive,
                                                      arith::Analyzer* analyzer) {
  const Block& block = realize->block;
  Array<IterVar> new_iter_vars;
  Map<Var, PrimExpr> repl_dict;

  // create new block itervars
  for (size_t i = 0; i < block->iter_vars.size(); ++i) {
    const IterVar& origin_iter = block->iter_vars[i];
    Var new_var = origin_iter->var.copy_with_suffix("");
    new_iter_vars.push_back(IterVar(origin_iter->dom, new_var, IterVarType::kDataPar));
    repl_dict.Set(origin_iter->var, new_var);
  }

  // rewrite expr helper
  auto rewrite_expr = [&repl_dict, analyzer](const PrimExpr& e) {
    return analyzer->Simplify(Substitute(e, repl_dict));
  };

  // create new write region
  ICHECK_EQ(block->writes.size(), 1U);
  BufferRegion write_region = BufferRegion(
      block->writes[0]->buffer, block->writes[0]->region.Map([rewrite_expr](const Range& r) {
        return Range::FromMinExtent(rewrite_expr(r->min), rewrite_expr(r->extent));
      }));

  // create block to fill const pad values
  BufferStore store = Downcast<BufferStore>(block->body);
  store.CopyOnWrite()->value = info.pad_value;
  store.CopyOnWrite()->indices = store->indices.Map(rewrite_expr);
  Block new_block(/*iter_vars=*/new_iter_vars, /*reads=*/{}, /*writes=*/{write_region},
                  /*name_hint=*/block->name_hint + "_pad_const", /*body=*/std::move(store));

  // create new loop vars
  Array<Var> new_loop_vars;
  for (const For& loop : loops) {
    Var new_var = loop->loop_var.copy_with_suffix("");
    new_loop_vars.push_back(new_var);
    repl_dict.Set(loop->loop_var, new_var);
    if (loop.same_as(highest_pos_inclusive)) {
      break;
    }
  }

  // create new block realize node
  Array<PrimExpr> new_iter_values;
  for (size_t i = 0; i < realize->iter_values.size(); ++i) {
    new_iter_values.push_back(rewrite_expr(realize->iter_values[i]));
  }
  BlockRealize new_realize(/*iter_values=*/new_iter_values,
                           /*predicate=*/rewrite_expr(realize->predicate),
                           /*block=*/new_block);

  // create new loops
  Stmt nest_stmt_root = new_realize;
  for (size_t i = 0; i < new_loop_vars.size(); ++i) {
    For loop = loops[i];
    nest_stmt_root =
        For(new_loop_vars[i], loop->min, loop->extent, ForKind::kSerial, nest_stmt_root);
  }

  return {nest_stmt_root, new_realize};
}

/*! \brief Create block to fill in-bound region values. */
static std::pair<Stmt, BlockRealize> CreateInBoundBlock(const BlockRealizeNode* realize,
                                                        const PaddingBlockInfo& info,

                                                        const Array<For>& loops,
                                                        const Stmt& highest_pos_inclusive,
                                                        arith::Analyzer* analyzer) {
  const Block& block = realize->block;
  Array<IterVar> new_iter_vars;
  Map<Var, PrimExpr> repl_dict;

  // record loop ranges to be mutated
  Map<Var, Range> new_loop_ranges;
  for (const For& loop : loops) {
    new_loop_ranges.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
    if (loop.same_as(highest_pos_inclusive)) {
      break;
    }
  }

  // create new block iter vars and iter bindings
  Array<PrimExpr> new_iter_binding;
  for (size_t i = 0; i < info.in_bound_region.size(); ++i) {
    // add new block itervar
    const IterVar& origin_itervar = block->iter_vars[i];
    Var new_var = origin_itervar->var.copy_with_suffix("");
    Range new_range =
        Range::FromMinExtent(make_const(new_var->dtype, 0), info.in_bound_region[i]->extent);
    new_iter_vars.push_back(IterVar(new_range, new_var, IterVarType::kDataPar));
    repl_dict.Set(origin_itervar->var, new_var + info.in_bound_region[i]->min);

    // update new loop range
    if (auto opt = realize->iter_values[i].as<Var>(); opt && new_loop_ranges.count(opt.value())) {
      // if the block binding is the loop var with single child, mutate loop range
      // instead of insert extra block predicate
      auto loop_var = opt.value();
      new_loop_ranges.Set(loop_var, new_range);
      new_iter_binding.push_back(realize->iter_values[i]);
      repl_dict.Set(loop_var, loop_var + info.in_bound_region[i]->min);
      analyzer->Bind(loop_var, new_range, /*allow_override=*/true);
    } else {
      new_iter_binding.push_back(
          analyzer->Simplify(realize->iter_values[i] - info.in_bound_region[i]->min));
    }
  }

  // rewrite helpers
  auto rewrite_expr = [&repl_dict, analyzer](const PrimExpr& e) {
    return analyzer->Simplify(Substitute(e, repl_dict));
  };
  auto rewrite_region = [rewrite_expr](const Region& region) {
    return region.Map([rewrite_expr](const Range& r) {
      return Range::FromMinExtent(rewrite_expr(r->min), rewrite_expr(r->extent));
    });
  };

  // create new read/write region for in-bound accesses
  Array<BufferRegion> reads, writes;
  for (const BufferRegion& read : block->reads) {
    reads.push_back(BufferRegion(read->buffer, rewrite_region(read->region)));
  }
  for (const BufferRegion& write : block->writes) {
    writes.push_back(BufferRegion(write->buffer, rewrite_region(write->region)));
  }

  // create new block realize node
  BufferStore store = Downcast<BufferStore>(block->body);
  store.CopyOnWrite()->value = rewrite_expr(info.in_bound_value);
  store.CopyOnWrite()->indices = store->indices.Map(rewrite_expr);
  Block new_block(/*iter_vars=*/new_iter_vars, /*reads=*/reads, /*writes=*/writes,
                  /*name_hint=*/block->name_hint, /*body=*/std::move(store));
  PrimExpr new_predicate = rewrite_expr(info.in_bound_predicate);
  BlockRealize new_realize(/*iter_values=*/new_iter_binding, /*predicate=*/new_predicate,
                           /*block=*/new_block);

  // create new loops
  Stmt nest_stmt_root = new_realize;
  for (const For& loop : loops) {
    auto it = new_loop_ranges.find(loop->loop_var);
    PrimExpr min = it == new_loop_ranges.end() ? loop->min : (*it).second->min;
    PrimExpr extent = it == new_loop_ranges.end() ? loop->extent : (*it).second->extent;
    nest_stmt_root = For(loop->loop_var, min, extent, loop->kind, nest_stmt_root,
                         loop->thread_binding, loop->annotations, loop->span);
    if (loop.same_as(highest_pos_inclusive)) {
      break;
    }
  }
  return {nest_stmt_root, new_realize};
}

/*!
 * \brief A helper class to create a new scope that contains decomposed padding blocks.
 */
class DecomposePaddingBlockReplacer : public StmtMutator {
 public:
  /*! \brief Replacement information */
  struct ReplaceDesc {
    /*! \brief loop above which to insert const pad value filling code. */
    For const_filling_pos;
    /*! \brief loop under which to insert in bound value filling code. */
    For in_bound_filling_pos;
    /*! \brief const pad value filling loop. */
    Stmt const_filling_loop;
    /*! \brief highest in bound value filling loop with single child. */
    Stmt in_bound_filling_loop;
    /*! \brief const pad value filling block. */
    BlockRealize const_filling_block;
    /*! \brief in bound value filling block. */
    BlockRealize in_bound_filling_block;
  };

  static Block Replace(Block scope_root, const ReplaceDesc& desc) {
    DecomposePaddingBlockReplacer replacer(desc);
    return Downcast<Block>(replacer(std::move(scope_root)));
  }

 private:
  explicit DecomposePaddingBlockReplacer(const ReplaceDesc& desc) : desc_(desc) {}

  Stmt VisitStmt_(const ForNode* op) final {
    Stmt new_loop;
    if (op == desc_.in_bound_filling_pos.get()) {
      // position to rewrite inbound filling code
      new_loop = desc_.in_bound_filling_loop;
    } else {
      new_loop = StmtMutator::VisitStmt_(op);
    }
    if (op == desc_.const_filling_pos.get()) {
      // position to insert pad value filling code
      return std::move(SeqStmt({desc_.const_filling_loop, new_loop}));
    }
    return std::move(new_loop);
  }

  Stmt VisitStmt_(const SeqStmtNode* seq) final {
    Array<Stmt> new_stmts;
    new_stmts.reserve(seq->seq.size());
    for (const Stmt& old_stmt : seq->seq) {
      new_stmts.push_back(VisitStmt(old_stmt));
    }
    return SeqStmt::Flatten(new_stmts);
  }

 private:
  const ReplaceDesc& desc_;
};

StmtSRef DecomposePaddingImpl(ScheduleState self, const StmtSRef& block_sref,
                              const StmtSRef& loop_sref, bool check_only) {
  /*!
   *  Check
   *    - the block is a compact block
   *    - the loop is an ancester of the block
   *    - the block match padding pattern
   *  Mutate
   *    - generate new block to fill padding values
   *    - trim original block to write non-padding part only
   */
  // Condition Checks and Information Collection
  const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
  const BlockRealizeNode* realize = GetBlockRealize(self, block_sref).get();
  Map<Var, Range> dom_map;
  arith::Analyzer analyzer;

  // Check 1. check the block is complete.
  StmtSRef scope_root_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false);
  CheckCompleteBlock(self, block_sref, scope_root_sref);

  // Check 2. Check loop_sref is an ancestor of block_sref. Also collect
  //   - the highest loop position (inclusive) to insert const pad value filling code above.
  //   - the highest loop position (inclusive) to replace with in-bound value filling code.
  Array<StmtSRef> loop_srefs = GetLoops(block_sref);
  Array<For> loops;
  bool found_const_filling_pos = false;
  bool found_in_bound_filling_pos = false;
  For const_filling_pos = GetRef<For>(loop_sref->StmtAs<ForNode>());
  For in_bound_filling_pos{nullptr};
  for (auto it = loop_srefs.rbegin(); it != loop_srefs.rend(); ++it) {
    For cur_loop = GetRef<For>((*it)->StmtAs<ForNode>());
    Range range = Range::FromMinExtent(cur_loop->min, cur_loop->extent);
    dom_map.Set(cur_loop->loop_var, range);
    analyzer.Bind(cur_loop->loop_var, range);
    loops.push_back(cur_loop);

    if (cur_loop.same_as(const_filling_pos)) {
      ICHECK(!found_const_filling_pos);
      found_const_filling_pos = true;
      if (!found_in_bound_filling_pos) {
        found_in_bound_filling_pos = true;
        in_bound_filling_pos = cur_loop;
      }
    } else if (!found_in_bound_filling_pos) {
      if (!cur_loop->body->IsInstance<ForNode>() &&
          !cur_loop->body->IsInstance<BlockRealizeNode>()) {
        found_in_bound_filling_pos = true;
      } else {
        in_bound_filling_pos = cur_loop;
      }
    }
  }
  ICHECK(in_bound_filling_pos.defined());
  if (!found_const_filling_pos) {
    throw LoopPositionError(self->mod, const_filling_pos, GetRef<Block>(block),
                            "decompose_padding");
  }

  // Check 3. match padding pattern and return padding operation info.
  PaddingBlockInfo info =
      PaddingInfoAnalyzer::CheckAndGetPaddingInfo(self->mod, realize, dom_map, &analyzer);

  // IR Manipulation
  // Step 1. Create const pad value filling part and in-bound value filling part.
  DecomposePaddingBlockReplacer::ReplaceDesc replace_desc;
  replace_desc.const_filling_pos = const_filling_pos;
  replace_desc.in_bound_filling_pos = in_bound_filling_pos;
  std::tie(replace_desc.const_filling_loop, replace_desc.const_filling_block) =
      CreateConstBlock(realize, info, loops, const_filling_pos, &analyzer);
  std::tie(replace_desc.in_bound_filling_loop, replace_desc.in_bound_filling_block) =
      CreateInBoundBlock(realize, info, loops, in_bound_filling_pos, &analyzer);

  // Step 2. Execute IR replacement.
  Block old_scope_root_block = GetRef<Block>(scope_root_sref->StmtAs<BlockNode>());
  Block new_scope_root = DecomposePaddingBlockReplacer::Replace(old_scope_root_block, replace_desc);
  if (check_only) {
    return block_sref;
  }

  // Step 3. Update schedule states.
  self->Replace(scope_root_sref, new_scope_root,
                {{old_scope_root_block, new_scope_root},
                 {GetRef<Block>(block), replace_desc.in_bound_filling_block->block}});
  auto new_block_sref = self->stmt2ref.at(replace_desc.const_filling_block->block.get());

  // Set block info of created const pad value filling block
  BlockInfo& block_info = self->block_info[new_block_sref];
  block_info.affine_binding = true;
  block_info.region_cover = true;
  block_info.stage_pipeline = true;

  // If the const pad value filling block is lifted out of the original subtree,
  // set the region_cover flag as false since region_cover is the property under the subtree.
  bool preserve_stage_pipeline = true;
  for (const StmtSRef& consumer_sref : GetConsumers(self, block_sref)) {
    StmtSRef lca = GetSRefLowestCommonAncestor({consumer_sref, block_sref});
    const StmtSRefNode* parent = new_block_sref->parent;
    bool is_under_lca = false;
    while (parent) {
      if (parent == lca.get()) {
        is_under_lca = true;
        break;
      }
      parent = parent->parent;
    }
    if (!is_under_lca) {
      preserve_stage_pipeline = false;
      self->block_info[consumer_sref].region_cover = false;
    }
  }
  if (!preserve_stage_pipeline) {
    self->block_info[scope_root_sref].stage_pipeline = false;
  }
  return new_block_sref;
}

StmtSRef DecomposePadding(ScheduleState self, const StmtSRef& block_sref,
                          const StmtSRef& loop_sref) {
  return DecomposePaddingImpl(self, block_sref, loop_sref, false);
}

bool CanDecomposePadding(ScheduleState self, const StmtSRef& block_sref,
                         const StmtSRef& loop_sref) {
  try {
    DecomposePaddingImpl(self, block_sref, loop_sref, true);
  } catch (const tvm::runtime::Error& e) {
    return false;
  }
  return true;
}

/******** FFI ********/

TVM_REGISTER_GLOBAL("tir.schedule.CanDecomposePadding")
    .set_body_typed([](Schedule self, BlockRV block_rv, LoopRV loop_rv) {
      return CanDecomposePadding(self->state(), self->GetSRef(block_rv), self->GetSRef(loop_rv));
    });

/******** InstructionKind Registration ********/

struct DecomposPaddingTraits : public UnpackedInstTraits<DecomposPaddingTraits> {
  static constexpr const char* kName = "DecomposePadding";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static BlockRV UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv, LoopRV loop_rv) {
    return sch->DecomposePadding(block_rv, loop_rv);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv, LoopRV loop_rv) {
    PythonAPICall py("decompose_padding");
    py.Input("block", block_rv);
    py.Input("loop", loop_rv);
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(DecomposPaddingTraits);

}  // namespace tir
}  // namespace tvm
