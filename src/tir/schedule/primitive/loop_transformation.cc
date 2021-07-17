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

/*! \brief Append a new predicate to the each child of type BlockRealize (not recursively) */
class BlockPredicateAppender : public StmtMutator {
 public:
  /*!
   * \brief Constructor
   * \param to_append The predicate to be appended to BlockRealizeNode
   */
  explicit BlockPredicateAppender(const PrimExpr& to_append, arith::Analyzer* analyzer)
      : to_append_(to_append) {
    add_predicate_ = !analyzer->CanProve(to_append);
  }

 private:
  // For each direct child of type BlockRealizeNode, append the predicate
  Stmt VisitStmt_(const BlockRealizeNode* realize) final {
    // We do not recursively do this
    if (add_predicate_) {
      ObjectPtr<BlockRealizeNode> n = CopyOnWrite(realize);
      n->predicate = n->predicate && to_append_;
      return BlockRealize(n);
    } else {
      return GetRef<BlockRealize>(realize);
    }
  }

  /*! \brief The predicate to be appended */
  const PrimExpr& to_append_;
  /*! \brief Whether to add predicate */
  bool add_predicate_;
};

/*! \brief Substitute vars and collect the reuse mapping of opaque blocks */
class SubstituteVarAndCollectOpaqueBlock : public StmtExprMutator {
 public:
  explicit SubstituteVarAndCollectOpaqueBlock(std::function<Optional<PrimExpr>(const Var&)> vmap,
                                              Map<Block, Block>* opaque_blocks)
      : vmap_(vmap), opaque_blocks_(opaque_blocks) {}

 private:
  PrimExpr VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);
    if (Optional<PrimExpr> ret = vmap_(var)) {
      return ret.value();
    } else {
      return std::move(var);
    }
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    Stmt res = StmtMutator::VisitStmt_(op);
    if (op->block->iter_vars.empty()) {
      const BlockRealizeNode* realize = TVM_TYPE_AS(realize, res, BlockRealizeNode);
      opaque_blocks_->Set(op->block, realize->block);
    }
    return res;
  }

  /*! \brief The substitute function */
  std::function<Optional<PrimExpr>(const Var&)> vmap_;
  /*! \brief The reuse mapping of opaque blocks */
  Map<Block, Block>* opaque_blocks_;
};

Stmt SubstituteAndCollectOpaqueBlock(Stmt stmt, Map<Block, Block>* opaque_blocks,
                                     std::function<Optional<PrimExpr>(const Var&)> vmap) {
  return SubstituteVarAndCollectOpaqueBlock(vmap, opaque_blocks)(std::move(stmt));
}

/*! \brief Simplify the binding of block realize and update the opaque block reuse mapping */
class IterMapSimplifyBlockBinding : public StmtExprMutator {
 public:
  explicit IterMapSimplifyBlockBinding(const Map<Var, Range>& loop_map,
                                       Map<Block, Block>* opaque_blocks)
      : opaque_blocks_(opaque_blocks), loop_var2extent_(std::move(loop_map)) {}

  static For SimplifyBindings(const Stmt& stmt, const Array<StmtSRef>& loop_srefs,
                              Map<Block, Block>* opaque_blocks) {
    Map<Var, Range> loop_var2extent;
    MapNode* loop_var2extent_mutable = loop_var2extent.CopyOnWrite();
    for (const StmtSRef& sref : loop_srefs) {
      const ForNode* loop = TVM_SREF_TO_FOR(loop, sref);
      loop_var2extent_mutable->at(loop->loop_var) = Range::FromMinExtent(loop->min, loop->extent);
    }
    return Downcast<For>(IterMapSimplifyBlockBinding(loop_var2extent, opaque_blocks)(stmt));
  }

 private:
  Stmt VisitStmt_(const ForNode* op) final {
    loop_var2extent_.Set(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    Stmt res = StmtMutator::VisitStmt_(op);
    loop_var2extent_.erase(op->loop_var);
    return res;
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    // skip opaque block and update mapping
    if (op->iter_values.empty()) {
      Stmt res = StmtMutator::VisitStmt_(op);
      const BlockRealizeNode* realize = res.as<BlockRealizeNode>();
      MapNode* mutable_map = opaque_blocks_->CopyOnWrite();
      for (const std::pair<Block, Block>& entry : *opaque_blocks_) {
        if (entry.second.same_as(op->block)) {
          mutable_map->at(entry.first) = realize->block;
          break;
        }
      }
      return res;
    }
    Array<PrimExpr> v = arith::IterMapSimplify(/*indices=*/op->iter_values,
                                               /*input_iters=*/loop_var2extent_,
                                               /*input_pred=*/op->predicate,
                                               /*require_bijective=*/false);
    if (v.same_as(op->iter_values)) {
      return GetRef<Stmt>(op);
    } else {
      ObjectPtr<BlockRealizeNode> n = CopyOnWrite(op);
      n->iter_values = std::move(v);
      return Stmt(n);
    }
  }

  /*! \brief The reuse mapping */
  Map<Block, Block>* opaque_blocks_;
  /*! \brief The range of loops */
  Map<Var, Range> loop_var2extent_;
};

class HasAnnotationOrThreadBindingError : public ScheduleError {
 public:
  explicit HasAnnotationOrThreadBindingError(IRModule mod, For loop)
      : mod_(mod), loop_(std::move(loop)) {}

  String FastErrorString() const final {
    return "ScheduleError: The primitive can't be applied because the loop has annotation or "
           "thread binding";
  }

  String DetailRenderTemplate() const final {
    return "The primitive can't be applied because the loop {0} has annotation or thread binding";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {loop_}; }

  IRModule mod_;
  For loop_;
};

class OuterNotInnerParent : public ScheduleError {
 public:
  explicit OuterNotInnerParent(IRModule mod, For outer, For inner)
      : mod_(mod), outer_(std::move(outer)), inner_(std::move(inner)) {}

  String FastErrorString() const final {
    return "ScheduleError: The outer loop is not the parent of the inner loop";
  }

  String DetailRenderTemplate() const final {
    return "The loops can't be fused because the outer loop {0} is not the parent of the inner "
           "loop {1}";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {outer_, inner_}; }

  IRModule mod_;
  For outer_;
  For inner_;
};

class NotOnlyChildError : public ScheduleError {
 public:
  explicit NotOnlyChildError(IRModule mod, For outer, For inner)
      : mod_(mod), outer_(std::move(outer)), inner_(std::move(inner)) {}

  String FastErrorString() const final {
    return "ScheduleError: The inner loop is not the only child of outer loop";
  }

  String DetailRenderTemplate() const final {
    return "The loops can't be fused because the inner loop {1} is not the only child of outer "
           "loop {0}.";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {outer_, inner_}; }

  IRModule mod_;
  For outer_;
  For inner_;
};

class LoopNotStartWithZeroError : public ScheduleError {
 public:
  explicit LoopNotStartWithZeroError(IRModule mod, For loop) : mod_(mod), loop_(std::move(loop)) {}

  String FastErrorString() const final {
    return "ScheduleError: The primitive only supports loop starting with 0";
  }

  String DetailRenderTemplate() const final {
    return "The loop {0} does not start with 0, which is not supported";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {loop_}; }

  IRModule mod_;
  For loop_;
};

class NotSingleInferFactorError : public ScheduleError {
 public:
  explicit NotSingleInferFactorError(IRModule mod) : mod_(mod) {}

  String FastErrorString() const final {
    return "ScheduleError: only one factor can be specified as -1 or none";
  }

  String DetailRenderTemplate() const final {
    return "Only one factor can be specified as -1 or none";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {}; }

  IRModule mod_;
};

class WrongFactorProductError : public ScheduleError {
 public:
  explicit WrongFactorProductError(IRModule mod, For loop) : mod_(mod), loop_(std::move(loop)) {}

  String FastErrorString() const final {
    return "ScheduleError: The product of factors is not larger than or equal to the extent of "
           "loop";
  }

  String DetailRenderTemplate() const final {
    return "The product of factors is not larger than or equal to the extent of loop {0}";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {loop_}; }

  IRModule mod_;
  For loop_;
};

Array<StmtSRef> Split(ScheduleState self, const StmtSRef& loop_sref,
                      const Array<PrimExpr>& factors) {
  // Invariance
  // - The total repeat number has not changed for each direct child block with updating predicate.
  // - The execution order has not changed. (The block executes with the same args and the same
  // order with before.
  // Step 1. Check correctness
  const ForNode* loop = TVM_SREF_TO_FOR(loop, loop_sref);
  ICHECK(loop) << "the input sref does not point to a loop";
  if (!loop->annotations.empty() || loop->thread_binding.defined()) {
    throw HasAnnotationOrThreadBindingError(self->mod, GetRef<For>(loop));
  }
  // Currently, loops not starting with 0 are not supported
  arith::Analyzer analyzer;
  if (!analyzer.CanProve(loop->min == 0)) {
    throw LoopNotStartWithZeroError(self->mod, GetRef<For>(loop));
  }
  PrimExpr tot_length = 1;
  int infer_index = -1;
  size_t n = factors.size();
  for (size_t i = 0; i < n; i++) {
    if (!analyzer.CanProve(factors[i] == -1)) {
      tot_length *= factors[i];
    } else if (infer_index != -1) {
      throw NotSingleInferFactorError(self->mod);
    } else {
      infer_index = i;
    }
  }
  // Step 2. infer factors if needed
  Array<PrimExpr> inferred_factors(factors);
  if (infer_index != -1) {
    inferred_factors.Set(infer_index,
                         analyzer.Simplify(floordiv(loop->extent + tot_length - 1, tot_length)));
  } else if (!analyzer.CanProve(tot_length >= loop->extent)) {
    throw WrongFactorProductError(self->mod, GetRef<For>(loop));
  }
  // Step 3. Replace all occurrences of the original loop var with new variables
  std::vector<Var> new_loop_vars;
  new_loop_vars.reserve(n);
  for (size_t i = 0; i < n; i++) {
    new_loop_vars.push_back(loop->loop_var.copy_with_suffix("_" + std::to_string(i)));
  }
  PrimExpr substitute_value = 0;
  for (size_t i = 0; i < n; i++) {
    substitute_value *= inferred_factors[i];
    substitute_value += new_loop_vars[i];
  }
  Map<Block, Block> opaque_block_reuse;
  auto f_substitute = [&](const Var& v) -> Optional<PrimExpr> {
    if (v.same_as(loop->loop_var)) {
      return substitute_value;
    } else {
      return NullOpt;
    }
  };
  Stmt new_stmt =
      SubstituteVarAndCollectOpaqueBlock(f_substitute, &opaque_block_reuse)(std::move(loop->body));
  for (size_t i = 0; i < n; i++) {
    analyzer.Bind(new_loop_vars[i], Range::FromMinExtent(0, inferred_factors[i]));
  }
  // Step 4. Update predicate to guard the loop
  new_stmt =
      BlockPredicateAppender(/*predicate=*/substitute_value < loop->extent, &analyzer)(new_stmt);
  // Step 5. Generate nested loops to replace the original loop and simplify the binding
  for (int i = n - 1; i >= 0; i--) {
    new_stmt = For(new_loop_vars[i], 0, inferred_factors[i], loop->kind, new_stmt);
  }

  new_stmt = IterMapSimplifyBlockBinding::SimplifyBindings(new_stmt, GetLoops(loop_sref),
                                                           &opaque_block_reuse);
  self->Replace(loop_sref, new_stmt, opaque_block_reuse);
  Array<StmtSRef> result_srefs;
  result_srefs.reserve(n);
  for (size_t i = 0; i < n; i++) {
    result_srefs.push_back(self->stmt2ref.at(new_stmt.get()));
    const ForNode* outer_loop = TVM_TYPE_AS(outer_loop, new_stmt, ForNode);
    new_stmt = outer_loop->body;
  }
  return result_srefs;
}

StmtSRef Fuse(ScheduleState self, const Array<StmtSRef>& loop_srefs) {
  // Invariance
  // - The total repeat number has not changed for each direct child block.
  // - The execution order has not changed. (The block executes with the same
  //   args and the same order with before.)
  std::vector<const ForNode*> loops;
  loops.reserve(loop_srefs.size());
  StmtSRef outer_loop_sref{nullptr};
  const ForNode* outer_loop = nullptr;
  arith::Analyzer analyzer;
  // Step 1. check correctness
  for (const StmtSRef& sref : loop_srefs) {
    const auto* loop = sref->StmtAs<ForNode>();
    ICHECK(loop) << "the input sref does not point to a loop";
    if (!loop->annotations.empty() || loop->thread_binding.defined()) {
      throw HasAnnotationOrThreadBindingError(self->mod, GetRef<For>(loop));
    }
    if (outer_loop_sref.defined()) {
      if (sref->parent != outer_loop_sref.get()) {
        throw OuterNotInnerParent(self->mod, GetRef<For>(outer_loop), GetRef<For>(loop));
      }
      if (!outer_loop->body.same_as(GetRef<For>(loop))) {
        throw NotOnlyChildError(self->mod, GetRef<For>(outer_loop), GetRef<For>(loop));
      }
    }
    outer_loop_sref = sref;
    outer_loop = loop;
    if (!analyzer.CanProve(loop->min == 0)) {
      throw LoopNotStartWithZeroError(self->mod, GetRef<For>(loop));
    }
    loops.push_back(loop);
  }
  // Step 2. Create fused loop var and replace the original loop vars
  std::string suffix;
  for (size_t i = 1; i < loops.size(); i++) {
    suffix += "_" + loops[i]->loop_var->name_hint;
  }
  suffix += "_fused";
  Var fused_var = loops[0]->loop_var.copy_with_suffix(suffix);
  Array<PrimExpr> substitute_value;
  substitute_value.resize(loops.size());
  PrimExpr tot = fused_var;
  for (int i = loops.size() - 1; i >= 0; i--) {
    substitute_value.Set(i, floormod(tot, loops[i]->extent));
    tot = floordiv(tot, loops[i]->extent);
  }
  Stmt loop_body = loops.back()->body;
  Map<Block, Block> opaque_block_reuse;
  auto f_substitute = [&](const Var& v) -> Optional<PrimExpr> {
    for (size_t i = 0; i < loops.size(); i++) {
      if (v.same_as(loops[i]->loop_var)) {
        return substitute_value[i];
      }
    }
    return NullOpt;
  };
  Stmt new_stmt =
      SubstituteVarAndCollectOpaqueBlock(f_substitute, &opaque_block_reuse)(std::move(loop_body));
  // Step 3. Generate a loop to replace the original loops
  PrimExpr fused_extent = 1;
  for (size_t i = 0; i < loops.size(); i++) {
    fused_extent *= loops[i]->extent;
  }
  fused_extent = analyzer.Simplify(fused_extent);
  new_stmt = For(fused_var, 0, fused_extent, ForKind::kSerial, new_stmt);
  new_stmt = IterMapSimplifyBlockBinding::SimplifyBindings(new_stmt, GetLoops(loop_srefs[0]),
                                                           &opaque_block_reuse);
  self->Replace(loop_srefs[0], new_stmt, opaque_block_reuse);
  return self->stmt2ref.at(new_stmt.get());
}

}  // namespace tir
}  // namespace tvm
