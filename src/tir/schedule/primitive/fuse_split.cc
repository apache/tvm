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

/*! \brief Append a new predicate to the each children of type BlockRealize (not recursively) */
class PredicateUpdater : public StmtMutator {
 public:
  /*!
   * \brief Constructor
   * \param predicate The predicate to be apppend to BlockRealizeNode
   */
  explicit PredicateUpdater(const PrimExpr& predicate, arith::Analyzer* ana)
      : predicate_(predicate) {
    if (!ana->CanProve(predicate)) {
      add_predicate_ = true;
    }
  }

 private:
  // For each direct child of type BlockRealizeNode, append the predicate
  Stmt VisitStmt_(const BlockRealizeNode* realize) final {
    // We do not recursively do this
    if (add_predicate_) {
      ObjectPtr<BlockRealizeNode> n = CopyOnWrite(realize);
      n->predicate = n->predicate && predicate_;
      return BlockRealize(n);
    } else {
      return GetRef<BlockRealize>(realize);
    }
  }

  /*! \brief The predicate to be added */
  const PrimExpr& predicate_;
  /*! \brief whether to add predicate */
  bool add_predicate_;
};
/*! \brief Substitute vars and collect the reuse mapping of opaque blocks */
class IRSubstituteAndCollectOpaqueBlock : public StmtExprMutator {
 public:
  explicit IRSubstituteAndCollectOpaqueBlock(std::function<Optional<PrimExpr>(const Var&)> vmap,
                                             Map<Block, Block>* opaque_blocks)
      : vmap_(vmap), opaque_blocks_(opaque_blocks) {}

 private:
  PrimExpr VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);
    Optional<PrimExpr> ret = vmap_(var);
    if (ret.defined()) {
      return ret.value();
    } else {
      return std::move(var);
    }
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    Stmt res = StmtMutator::VisitStmt_(op);
    if (op->block->iter_vars.empty()) {
      const BlockRealizeNode* realize = res.as<BlockRealizeNode>();
      opaque_blocks_->Set(op->block, realize->block);
    }
    return res;
  }

  /*! \brief The substitute function */
  std::function<Optional<PrimExpr>(const Var&)> vmap_;
  /*! \brief The reuse mapping */
  Map<Block, Block>* opaque_blocks_;
};

Stmt SubstituteAndCollectOpaqueBlock(Stmt stmt, Map<Block, Block>* opaque_blocks,
                                     std::function<Optional<PrimExpr>(const Var&)> vmap) {
  return IRSubstituteAndCollectOpaqueBlock(vmap, opaque_blocks)(std::move(stmt));
}

/*! \brief Simplify the binding of block realize and update the opaque block reuse mapping*/
class BlockRealizeRewriter : public StmtExprMutator {
 public:
  explicit BlockRealizeRewriter(
      const std::unordered_map<Var, Range, ObjectPtrHash, ObjectPtrEqual>& loop_map,
      Map<Block, Block>* opaque_blocks)
      : opaque_blocks_(opaque_blocks) {
    loop_map_.insert(loop_map.begin(), loop_map.end());
  }

 private:
  Stmt VisitStmt_(const ForNode* op) final {
    loop_map_[op->loop_var] = Range::FromMinExtent(op->min, op->extent);
    Stmt res = StmtMutator::VisitStmt_(op);
    loop_map_.erase(op->loop_var);
    return res;
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    // skip opaque block and update mapping
    if (op->iter_values.empty()) {
      Stmt res = StmtMutator::VisitStmt_(op);
      const BlockRealizeNode* realize = res.as<BlockRealizeNode>();
      for (const std::pair<Block, Block>& entry : *opaque_blocks_) {
        if (entry.second.same_as(op->block)) {
          opaque_blocks_->Set(entry.first, realize->block);
          break;
        }
      }
      return res;
    }
    auto v = arith::IterMapSimplify(op->iter_values, loop_map_, op->predicate, false);
    if (v.same_as(op->iter_values)) {
      return GetRef<Stmt>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->iter_values = std::move(v);
      return Stmt(n);
    }
  }
  /*! \brief The range of loops */
  std::unordered_map<Var, Range, ObjectPtrHash, ObjectPtrEqual> loop_map_;
  /*! \brief The reuse mapping */
  Map<Block, Block>* opaque_blocks_;
};

Stmt SimplifyBindings(const Stmt& stmt, const Array<StmtSRef>& loops,
                      Map<Block, Block>* opaque_blocks) {
  std::unordered_map<Var, Range, ObjectPtrHash, ObjectPtrEqual> loop_map;
  for (const StmtSRef& sref : loops) {
    const auto* loop = sref->StmtAs<ForNode>();
    loop_map[loop->loop_var] = Range::FromMinExtent(loop->min, loop->extent);
  }
  BlockRealizeRewriter rewriter(loop_map, opaque_blocks);
  return rewriter(stmt);
}

class NotLoopError : public ScheduleError {
 public:
  explicit NotLoopError(IRModule mod, String type) : mod_(mod), type_(type) {}

  String FastErrorString() const final {
    return "ScheduleError: this primitive only operates on a "
           "loop";
  }

  String DetailRenderTemplate() const final {
    return "this primitive only operates on a loop, but the StmtSref passed in points to"
           "type: {0} ";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {type_}; }

  IRModule mod_;
  String type_;
};

class HasAnnotationError : public ScheduleError {
 public:
  explicit HasAnnotationError(IRModule mod, For loop) : mod_(mod), loop_(loop) {}

  String FastErrorString() const final {
    return "ScheduleError: The primitive can't be applied because the loop has annotation";
  }

  String DetailRenderTemplate() const final {
    return "The primitive can't be applied because the loop {0} has annotation";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {loop_}; }

  IRModule mod_;
  For loop_;
};

class HasThreadBindingError : public ScheduleError {
 public:
  explicit HasThreadBindingError(IRModule mod, For loop) : mod_(mod), loop_(loop) {}

  String FastErrorString() const final {
    return "ScheduleError: The primitive can't be applied because the loop has thread binding";
  }

  String DetailRenderTemplate() const final {
    return "The primitive can't be applied because the loop {0} has thread binding";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {loop_}; }

  IRModule mod_;
  For loop_;
};

class OuterNotInnerParent : public ScheduleError {
 public:
  explicit OuterNotInnerParent(IRModule mod, For outer, For inner)
      : mod_(mod), outer_(outer), inner_(inner) {}

  String FastErrorString() const final {
    return "ScheduleError: the outer loop is not the parent of the inner loop";
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
      : mod_(mod), outer_(outer), inner_(inner) {}

  String FastErrorString() const final {
    return "ScheduleError: the inner loop is not the only child of outer loop";
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
  explicit LoopNotStartWithZeroError(IRModule mod, For loop) : mod_(mod), loop_(loop) {}

  String FastErrorString() const final {
    return "ScheduleError: the primitive only supports loop starting with 0";
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
  explicit WrongFactorProductError(IRModule mod, For loop) : mod_(mod), loop_(loop) {}

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
  GetScopeRootAndCheckStagePipeline(self, loop_sref);
  const auto* loop = loop_sref->StmtAs<ForNode>();
  if (loop == nullptr) {
    throw NotLoopError(self->mod, loop_sref->stmt->GetTypeKey());
  }
  if (!loop->annotations.empty()) {
    throw HasAnnotationError(self->mod, GetRef<For>(loop));
  }
  if (loop->thread_binding.defined()) {
    throw HasThreadBindingError(self->mod, GetRef<For>(loop));
  }
  // Currently, loops starting with 0 is not supported
  arith::Analyzer analyzer;
  if (!analyzer.CanProve(loop->min == 0)) {
    throw LoopNotStartWithZeroError(self->mod, GetRef<For>(loop));
  }
  PrimExpr tot_length = 1;
  int infer_index = -1;
  for (size_t i = 0; i < factors.size(); i++) {
    if (!analyzer.CanProve(factors[i] == -1)) {
      tot_length *= factors[i];
    } else {
      if (infer_index != -1) {
        throw NotSingleInferFactorError(self->mod);
      } else {
        infer_index = i;
      }
    }
  }
  // Step 2. infer factors if needed
  Array<PrimExpr> inferred_factors(factors);
  if (infer_index != -1) {
    inferred_factors.Set(infer_index,
                         analyzer.Simplify(floordiv(loop->extent + tot_length - 1, tot_length)));
  } else {
    if (!analyzer.CanProve(tot_length >= loop->extent)) {
      throw WrongFactorProductError(self->mod, GetRef<For>(loop));
    }
  }
  // Step 3. Replace all occurrence of the original loop var with new variables
  std::vector<Var> new_loop_vars;
  new_loop_vars.reserve(inferred_factors.size());
  for (size_t i = 0; i < inferred_factors.size(); i++) {
    new_loop_vars.push_back(loop->loop_var.copy_with_suffix("_" + std::to_string(i)));
  }
  PrimExpr substitute_value = 0;
  for (size_t i = 0; i < inferred_factors.size(); i++) {
    substitute_value *= inferred_factors[i];
    substitute_value += new_loop_vars[i];
  }
  Map<Block, Block> opaque_block_reuse;
  auto substitute_function = [&](const Var& v) -> Optional<PrimExpr> {
    if (v.same_as(loop->loop_var)) {
      return substitute_value;
    } else {
      return NullOpt;
    }
  };
  Stmt new_loop_body =
      SubstituteAndCollectOpaqueBlock(loop->body, &opaque_block_reuse, substitute_function);
  for (size_t i = 0; i < inferred_factors.size(); i++) {
    analyzer.Bind(new_loop_vars[i], Range::FromMinExtent(0, inferred_factors[i]));
  }
  // Step 4. Update predicate to guard the loop
  PrimExpr predicate = substitute_value < loop->extent;
  new_loop_body = PredicateUpdater(predicate, &analyzer)(new_loop_body);
  // Step 5. Generate tnested loops to replace the original loop and simplify the binding
  Stmt outer_stmt = new_loop_body;
  for (int i = inferred_factors.size() - 1; i >= 0; i--) {
    outer_stmt = For(new_loop_vars[i], 0, inferred_factors[i], loop->kind, outer_stmt);
  }

  outer_stmt =
      Downcast<For>(SimplifyBindings(outer_stmt, GetLoops(loop_sref), &opaque_block_reuse));
  self->Replace(loop_sref, outer_stmt, opaque_block_reuse);
  Array<StmtSRef> result_srefs;
  result_srefs.reserve(inferred_factors.size());
  for (size_t i = 0; i < inferred_factors.size(); i++) {
    result_srefs.push_back(self->stmt2ref.at(outer_stmt.get()));
    const ForNode* outer_loop = outer_stmt.as<ForNode>();
    ICHECK(outer_loop);
    outer_stmt = outer_loop->body;
  }
  return result_srefs;
}

StmtSRef Fuse(ScheduleState self, const Array<StmtSRef>& loop_srefs) {
  //     Invariance
  //   - The total repeat number has not changed for each direct child block.
  //   - The execution order has not changed. (The block executes with the same
  //     args and the same order with before.)
  std::vector<const ForNode*> loops;
  loops.reserve(loop_srefs.size());
  StmtSRef outer_sref{nullptr};
  const ForNode* outer_loop = nullptr;
  arith::Analyzer analyzer;
  // Step 1. check correctness
  GetScopeRootAndCheckStagePipeline(self, loop_srefs[0]);
  for (const StmtSRef& sref : loop_srefs) {
    const auto* loop = sref->StmtAs<ForNode>();
    if (loop == nullptr) {
      throw NotLoopError(self->mod, sref->stmt->GetTypeKey());
    }
    if (!loop->annotations.empty()) {
      throw HasAnnotationError(self->mod, GetRef<For>(loop));
    }
    if (loop->thread_binding.defined()) {
      throw HasThreadBindingError(self->mod, GetRef<For>(loop));
    }
    if (outer_sref.defined()) {
      if (sref->parent != outer_sref.get()) {
        throw OuterNotInnerParent(self->mod, GetRef<For>(outer_loop), GetRef<For>(loop));
      }
      Array<Stmt> outer_children = GetChildren(GetRef<Stmt>(outer_loop));
      if (outer_children.size() != 1 || outer_children[0].get() != loop) {
        throw NotOnlyChildError(self->mod, GetRef<For>(outer_loop), GetRef<For>(loop));
      }
    }
    outer_sref = sref;
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
  auto substitute_function = [&](const Var& v) -> Optional<PrimExpr> {
    for (size_t i = 0; i < loops.size(); i++) {
      if (v.same_as(loops[i]->loop_var)) {
        return substitute_value[i];
      }
    }
    return NullOpt;
  };
  Stmt new_loop_body =
      SubstituteAndCollectOpaqueBlock(loop_body, &opaque_block_reuse, substitute_function);
  // Step 3. Generate a loop to replace the original loops
  PrimExpr fused_min = 0;
  PrimExpr fused_extent = 1;
  for (size_t i = 0; i < loops.size(); i++) {
    fused_extent *= loops[i]->extent;
  }
  fused_extent = analyzer.Simplify(fused_extent);
  For fused_loop = For(fused_var, fused_min, fused_extent, loops[0]->kind, new_loop_body);
  fused_loop =
      Downcast<For>(SimplifyBindings(fused_loop, GetLoops(loop_srefs[0]), &opaque_block_reuse));
  self->Replace(loop_srefs[0], fused_loop, opaque_block_reuse);
  return self->stmt2ref.at(fused_loop.get());
}

}  // namespace tir
}  // namespace tvm
