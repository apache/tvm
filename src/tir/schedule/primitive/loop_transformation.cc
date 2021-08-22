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
  explicit BlockPredicateAppender(const PrimExpr& to_append) : to_append_(to_append) {}

 private:
  // For each direct child of type BlockRealizeNode, append the predicate
  Stmt VisitStmt_(const BlockRealizeNode* realize) final {
    // We do not recursively do this
    ObjectPtr<BlockRealizeNode> n = CopyOnWrite(realize);
    n->predicate = n->predicate && to_append_;
    return BlockRealize(n);
  }

  /*! \brief The predicate to be appended */
  const PrimExpr& to_append_;
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
    BlockRealize realize = Downcast<BlockRealize>(StmtMutator::VisitStmt_(op));
    if (realize->block->iter_vars.empty()) {
      opaque_blocks_->Set(op->block, realize->block);
    }
    return std::move(realize);
  }

  /*! \brief The substitute function */
  std::function<Optional<PrimExpr>(const Var&)> vmap_;
  /*! \brief The reuse mapping of opaque blocks */
  Map<Block, Block>* opaque_blocks_;
};

/*! \brief Simplify the binding of block realize and update the opaque block reuse mapping */
class IterMapSimplifyBlockBinding : public StmtExprMutator {
 public:
  explicit IterMapSimplifyBlockBinding(MapNode* opaque_blocks, Map<Var, Range> loop_var2extent)
      : opaque_blocks_(opaque_blocks), loop_var2extent_(loop_var2extent) {}

  static For SimplifyBindings(Stmt stmt, const Array<StmtSRef>& loop_srefs,
                              MapNode* opaque_blocks) {
    Map<Var, Range> loop_var2extent;
    for (const StmtSRef& sref : loop_srefs) {
      const ForNode* loop = TVM_SREF_TO_FOR(loop, sref);
      loop_var2extent.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
    }
    return Downcast<For>(
        IterMapSimplifyBlockBinding(opaque_blocks, std::move(loop_var2extent))(std::move(stmt)));
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
      Block block = op->block;
      BlockRealize realize = Downcast<BlockRealize>(StmtMutator::VisitStmt_(op));
      for (const std::pair<ObjectRef, ObjectRef>& entry : *opaque_blocks_) {
        if (entry.second.same_as(block)) {
          opaque_blocks_->at(entry.first) = realize->block;
          break;
        }
      }
      return std::move(realize);
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
  MapNode* opaque_blocks_;
  /*! \brief The range of loops */
  Map<Var, Range> loop_var2extent_;
};

class BlockPropertyError : public ScheduleError {
 public:
  /*!
   * \brief Check that all the blocks under the specific stmt have affine bindings and only have
   *     data-parallel or reduction block iters
   * \param self The state of the schedule
   * \param sref The sref to the specific stmt
   */
  static void CheckBlockIterTypeAndAffineBinding(const ScheduleState& self,
                                                 const StmtSRefNode* sref) {
    class BlockIterTypeAndAffineBindingChecker : public StmtVisitor {
     public:
      explicit BlockIterTypeAndAffineBindingChecker(const ScheduleState& state) : state_(state) {}

     private:
      void VisitStmt_(const BlockNode* op) final {
        for (const IterVar& iter_var : op->iter_vars) {
          if (iter_var->iter_type != kDataPar && iter_var->iter_type != kCommReduce) {
            throw BlockPropertyError(state_->mod, GetRef<Block>(op));
          }
          CheckAffineBinding(state_, GetRef<Block>(op));
        }
      }
      const ScheduleState& state_;
    };

    BlockIterTypeAndAffineBindingChecker checker(self);
    checker(GetRef<Stmt>(sref->stmt));
  }

  explicit BlockPropertyError(IRModule mod, Block block) : mod_(mod), block_(std::move(block)) {}

  String FastErrorString() const final {
    return "ScheduleError: The block under the loops to be reordered have block iter type other "
           "than data-parallel or reduction";
  }

  String DetailRenderTemplate() const final {
    return "The block {0} under the loops to be reordered have block iter type other than "
           "data-parallel or reduction";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

  IRModule mod_;
  Block block_;
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

class LoopMultiAppearanceError : public ScheduleError {
 public:
  explicit LoopMultiAppearanceError(IRModule mod, For loop) : mod_(mod), loop_(std::move(loop)) {}

  String FastErrorString() const final {
    return "ScheduleError: Some loop appears in the input array for multiple times.";
  }

  String DetailRenderTemplate() const final {
    return "Loop {0} appears in the input array for multiple times.";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {loop_}; }

  IRModule mod_;
  For loop_;
};

class LoopsNotAChainError : public ScheduleError {
 public:
  enum class ProblemKind { kNotUnderAScope, kHaveNonSingleBranchStmt };

  explicit LoopsNotAChainError(IRModule mod, Optional<Stmt> problematic_loop, ProblemKind kind)
      : mod_(mod), problematic_loop_(std::move(problematic_loop)), kind_(kind) {}

  String FastErrorString() const final { return "ScheduleError: the loops are not in a chain"; }

  String DetailRenderTemplate() const final {
    std::stringstream ss;
    ss << "The loops are not in a chain because";
    if (kind_ == ProblemKind::kNotUnderAScope) {
      ss << " they are not under the same scope.";
    } else {
      ss << " there is a non-single-branch stmt in between. Problematic stmt: {0}";
    }
    return ss.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final {
    if (kind_ == ProblemKind::kNotUnderAScope) {
      return {};
    } else {
      ICHECK(problematic_loop_.defined());
      return {problematic_loop_.value()};
    }
  }

  IRModule mod_;
  Optional<Stmt> problematic_loop_;
  ProblemKind kind_;
};

class DependentLoopError : public ScheduleError {
 public:
  explicit DependentLoopError(IRModule mod, For loop, String inner_var)
      : mod_(mod), loop_(std::move(loop)), inner_var_(std::move(inner_var)) {}

  String FastErrorString() const final {
    return "ScheduleError: An outer loop's `min` or `extent` is dependent on an inner loop "
           "in the new order";
  }

  String DetailRenderTemplate() const final {
    return "Outer Loop {0}'s `min` or `extent` is dependent on an inner loop " + inner_var_ +
           " in the new order";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {loop_}; }

  IRModule mod_;
  For loop_;
  String inner_var_;
};

Array<StmtSRef> Split(ScheduleState self, const StmtSRef& loop_sref,
                      const Array<PrimExpr>& factors) {
  // Invariance
  // - The total repeat number has not changed for each direct child block with updating predicate.
  // - The execution order has not changed. (The block executes with the same args and the same
  // order with before.
  // Step 1. Check correctness
  const ForNode* loop = TVM_SREF_TO_FOR(loop, loop_sref);
  if (!loop->annotations.empty() || loop->thread_binding.defined()) {
    throw HasAnnotationOrThreadBindingError(self->mod, GetRef<For>(loop));
  }
  // Currently, loops not starting with 0 are not supported
  arith::Analyzer analyzer;
  if (!analyzer.CanProve(loop->min == 0)) {
    throw LoopNotStartWithZeroError(self->mod, GetRef<For>(loop));
  }
  // Step 2. Replace all occurrences of the original loop var with new variables
  int n = factors.size();
  PrimExpr substitute_value = 0;
  std::vector<Var> new_loop_vars;
  new_loop_vars.reserve(n);
  for (int i = 0; i < n; i++) {
    const PrimExpr& factor = factors[i];
    Var var = loop->loop_var.copy_with_suffix("_" + std::to_string(i));
    substitute_value = substitute_value * factor + var;
    analyzer.Bind(var, Range::FromMinExtent(0, factor));
    new_loop_vars.emplace_back(std::move(var));
  }
  Map<Block, Block> opaque_block_reuse;
  Stmt new_stmt = loop->body;
  new_stmt = SubstituteVarAndCollectOpaqueBlock(
      [&](const Var& v) -> Optional<PrimExpr> {
        if (v.same_as(loop->loop_var)) {
          return substitute_value;
        } else {
          return NullOpt;
        }
      },
      &opaque_block_reuse)(std::move(new_stmt));
  // Step 3. Update predicate to guard the loop
  PrimExpr predicate = substitute_value < loop->extent;
  if (!analyzer.CanProve(predicate)) {
    new_stmt = BlockPredicateAppender(/*predicate=*/predicate)(std::move(new_stmt));
  }
  // Step 4. Generate nested loops to replace the original loop and simplify the binding
  for (int i = n - 1; i >= 0; i--) {
    new_stmt = For(new_loop_vars[i], 0, factors[i], ForKind::kSerial, new_stmt);
  }
  new_stmt = IterMapSimplifyBlockBinding::SimplifyBindings(std::move(new_stmt), GetLoops(loop_sref),
                                                           opaque_block_reuse.CopyOnWrite());
  self->Replace(loop_sref, new_stmt, opaque_block_reuse);
  Array<StmtSRef> result_srefs;
  result_srefs.reserve(n);
  for (int i = 0; i < n; i++) {
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
    const ForNode* loop = TVM_SREF_TO_FOR(loop, sref);
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
  int n = loops.size();
  for (int i = 1; i < n; i++) {
    suffix += "_" + loops[i]->loop_var->name_hint;
  }
  suffix += "_fused";
  Var fused_var = loops[0]->loop_var.copy_with_suffix(suffix);
  Array<PrimExpr> substitute_value;
  substitute_value.resize(loops.size());
  PrimExpr tot = fused_var;
  for (int i = static_cast<int>(loops.size()) - 1; i >= 0; i--) {
    substitute_value.Set(i, floormod(tot, loops[i]->extent));
    tot = floordiv(tot, loops[i]->extent);
  }
  Stmt new_stmt = loops.back()->body;
  Map<Block, Block> opaque_block_reuse;
  auto f_substitute = [&](const Var& v) -> Optional<PrimExpr> {
    for (int i = 0; i < n; i++) {
      if (v.same_as(loops[i]->loop_var)) {
        return substitute_value[i];
      }
    }
    return NullOpt;
  };
  new_stmt =
      SubstituteVarAndCollectOpaqueBlock(f_substitute, &opaque_block_reuse)(std::move(new_stmt));
  // Step 3. Generate a loop to replace the original loops
  PrimExpr fused_extent = 1;
  for (int i = 0; i < n; i++) {
    fused_extent *= loops[i]->extent;
  }
  fused_extent = analyzer.Simplify(fused_extent);
  new_stmt = For(fused_var, 0, fused_extent, ForKind::kSerial, new_stmt);
  new_stmt = IterMapSimplifyBlockBinding::SimplifyBindings(
      std::move(new_stmt), GetLoops(loop_srefs[0]), opaque_block_reuse.CopyOnWrite());
  self->Replace(loop_srefs[0], new_stmt, opaque_block_reuse);
  return self->stmt2ref.at(new_stmt.get());
}
/*!
 * \brief Collect an array of loop srefs into a set
 * \param self The schedule state
 * \param ordered_loop_srefs The array of loop srefs
 * \return A set containing all loops in the array
 * \throws ScheduleError If there are duplicate loops in the array
 */
std::unordered_set<const StmtSRefNode*> CollectLoopsIntoSet(
    const ScheduleState& self, const Array<StmtSRef>& ordered_loop_srefs) {
  std::unordered_set<const StmtSRefNode*> loop_srefs;
  loop_srefs.reserve(ordered_loop_srefs.size());
  for (const StmtSRef& loop_sref : ordered_loop_srefs) {
    auto inserted = loop_srefs.insert(loop_sref.get());
    if (!inserted.second) {
      const ForNode* loop = TVM_SREF_TO_FOR(loop, loop_sref);
      throw LoopMultiAppearanceError(self->mod, GetRef<For>(loop));
    }
  }
  return loop_srefs;
}

/*!
 * \brief Get the top and bottom boundary of reorder range (which should be a chain)
 * \param self The schedule state
 * \param loop_srefs The set containing the srefs to the loops to be reordered
 * \return a pair containing the top and bottom boundary of the reorder range
 * \throws ScheduleError If the loops to be reordered is not in a chain
 */
std::pair<const StmtSRefNode*, const StmtSRefNode*> GetBoundaryOfReorderRange(
    const ScheduleState& self, const std::unordered_set<const StmtSRefNode*>& loop_srefs) {
  const StmtSRefNode* top = nullptr;
  const StmtSRefNode* bottom = *loop_srefs.begin();
  std::unordered_set<const StmtSRefNode*> visited;
  bool scope_block_visited = false;
  bool first_traversal = true;
  for (const StmtSRefNode* loop_sref : loop_srefs) {
    if (visited.count(loop_sref)) {
      continue;
    }
    for (const StmtSRefNode* v = loop_sref;; v = v->parent) {
      // Case 1. If `v` corresponds to a block, stop traversal.
      if (v->stmt->IsInstance<BlockNode>()) {
        if (scope_block_visited) {
          throw LoopsNotAChainError(self->mod, NullOpt,
                                    LoopsNotAChainError::ProblemKind::kNotUnderAScope);
        }
        scope_block_visited = true;
        break;
      }
      // Case 2. If `v` corresponds to a previously-visited loop, stop traversal and update
      // `bottom`.
      if (visited.count(v)) {
        if (v != bottom) {
          throw LoopsNotAChainError(self->mod, GetRef<Stmt>(v->stmt),
                                    LoopsNotAChainError::ProblemKind::kHaveNonSingleBranchStmt);
        }
        bottom = loop_sref;
        break;
      }
      // Case 3. Add `v` into `visited`
      visited.insert(v);
      // If it's the first traversal and the loop corresponding to `v` is in the input array,
      // update `top`.
      if (first_traversal && loop_srefs.count(v)) {
        top = v;
      }
    }
    first_traversal = false;
  }
  return std::make_pair(top, bottom);
}

/*!
 * \brief get all the loops in the reorder range
 * \param self The schedule state
 * \param top The top boundary of the reorder range
 * \param bottom The bottom boundary of the reorder range
 * \return an array containing all the loops in the reorder range
 * \throws ScheduleError If some loop in the reorder range is not single-branch
 */
std::vector<const StmtSRefNode*> GetLoopsInReorderRange(const ScheduleState& self,
                                                        const StmtSRefNode* top,
                                                        const StmtSRefNode* bottom) {
  std::vector<const StmtSRefNode*> chain;
  for (const StmtSRefNode* loop_sref = bottom; loop_sref != top;) {
    const StmtSRefNode* parent_loop_sref = loop_sref->parent;
    const ForNode* outer = parent_loop_sref->StmtAs<ForNode>();
    const ForNode* inner = loop_sref->StmtAs<ForNode>();
    ICHECK(outer != nullptr && inner != nullptr);
    if (outer->body.get() != inner) {
      throw LoopsNotAChainError(self->mod, GetRef<For>(outer),
                                LoopsNotAChainError::ProblemKind::kHaveNonSingleBranchStmt);
    }
    chain.push_back(loop_sref);
    loop_sref = parent_loop_sref;
  }
  chain.push_back(top);
  return chain;
}

/*!
 * \brief Construct a loop chain in the new order
 * \param self The schedule state
 * \param chain The loops in the reorder range
 * \param ordered_loop_srefs The loop srefs to be reordered
 * \param loop_srefs The set containing loop srefs to be reordered
 * \return the new loop chain
 * \throws ScheduleError If the domain of an outer loop depends on any of the inner loops after
 * reordering
 */
For ConstructNewLoopChain(const ScheduleState& self, std::vector<const StmtSRefNode*> chain,
                          const Array<StmtSRef>& ordered_loop_srefs,
                          const std::unordered_set<const StmtSRefNode*>& loop_srefs) {
  std::unordered_set<const VarNode*> inner_vars;
  inner_vars.reserve(chain.size());
  For new_loop{nullptr};
  int index = static_cast<int>(ordered_loop_srefs.size()) - 1;
  for (const StmtSRefNode* loop_sref : chain) {
    const ForNode* copy = nullptr;
    if (loop_srefs.count(loop_sref)) {
      copy = ordered_loop_srefs[index]->StmtAs<ForNode>();
      --index;
    } else {
      copy = loop_sref->StmtAs<ForNode>();
    }
    ICHECK(copy != nullptr);
    ObjectPtr<ForNode> n = make_object<ForNode>(*copy);
    if (new_loop.defined()) {
      n->body = new_loop;
    } else {
      n->body = loop_sref->StmtAs<ForNode>()->body;
    }
    const VarNode* used_var = nullptr;
    auto f_contain = [&inner_vars, &used_var](const VarNode* var) {
      if (inner_vars.count(var)) {
        used_var = var;
        return true;
      }
      return false;
    };
    if (UsesVar(copy->min, f_contain) || UsesVar(copy->extent, f_contain)) {
      throw DependentLoopError(self->mod, GetRef<For>(copy), used_var->name_hint);
    }
    inner_vars.insert(copy->loop_var.get());
    new_loop = For(std::move(n));
  }
  return new_loop;
}

void Reorder(ScheduleState self, const Array<StmtSRef>& ordered_loop_srefs) {
  if (ordered_loop_srefs.size() <= 1) {
    return;
  }
  // Step 1. Check uniqueness.and collect the input loop srefs into a set
  std::unordered_set<const StmtSRefNode*> loop_srefs =
      CollectLoopsIntoSet(self, ordered_loop_srefs);
  // Step 2. Gather loops to be reordered
  // For each loop sref in the input sref array, traverse upwards along its parent pointer in the
  // sref tree, and stop on either a block, or a previously-visited loop
  // - the top of the reorder range is the last loop visited in the first traversal which exists in
  //   the input array
  // - the bottom of the reorder range is the last loop in the input array which is not visited in
  // the previous traversals
  auto pair = GetBoundaryOfReorderRange(self, loop_srefs);
  const StmtSRefNode* top = pair.first;
  const StmtSRefNode* bottom = pair.second;
  // Step 3. Collect all loops in the chain and check the loops are single-branch
  std::vector<const StmtSRefNode*> chain = GetLoopsInReorderRange(self, top, bottom);
  // Step 4. Check the block below has all its block_var to be data-parallel or reduction,
  // and the block has an affine binding.
  BlockPropertyError::CheckBlockIterTypeAndAffineBinding(self, bottom);
  // Step 5. Replace the original loops with the reordered loops and check that outer loop is
  // not dependent on inner loop
  For new_loop = ConstructNewLoopChain(self, std::move(chain), ordered_loop_srefs, loop_srefs);
  self->Replace(GetRef<StmtSRef>(top), new_loop, {});
}

/******** Instruction Registration ********/

struct SplitTraits : public UnpackedInstTraits<SplitTraits> {
  static constexpr const char* kName = "Split";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  template <size_t delta>
  static TVM_ALWAYS_INLINE void _SetInputs(const runtime::TVMArgsSetter& setter,
                                           const Array<ObjectRef>& inputs) {
    thread_local ObjectRef loop_rv{nullptr};
    thread_local Array<ObjectRef> factors{nullptr};
    loop_rv = inputs[0];
    factors = Array<ObjectRef>{inputs.begin() + 1, inputs.end()};
    setter(delta, loop_rv);
    setter(delta + 1, factors);
  }

  static Array<LoopRV> UnpackedApplyToSchedule(Schedule sch, LoopRV loop_rv,
                                               Array<Optional<ExprRV>> factors) {
    return sch->Split(loop_rv, factors);
  }

  static String UnpackedAsPython(Array<String> outputs, String loop_rv, Array<ObjectRef> factors) {
    PythonAPICall py("split");
    py.Input("loop", loop_rv);
    py.Input("factors", factors);
    py.OutputList(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct FuseTraits : public UnpackedInstTraits<FuseTraits> {
  static constexpr const char* kName = "Fuse";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  template <size_t delta>
  static TVM_ALWAYS_INLINE void _SetInputs(const runtime::TVMArgsSetter& setter,
                                           const Array<ObjectRef>& inputs) {
    setter(delta, inputs);
  }

  static LoopRV UnpackedApplyToSchedule(Schedule sch, Array<LoopRV> loop_rvs) {
    return sch->Fuse(loop_rvs);
  }

  static String UnpackedAsPython(Array<String> outputs, Array<String> loop_rvs) {
    PythonAPICall py("fuse");
    for (const String& loop_rv : loop_rvs) {
      py.Input("", loop_rv);
    }
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct ReorderTraits : public UnpackedInstTraits<ReorderTraits> {
  static constexpr const char* kName = "Reorder";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  template <size_t delta>
  static TVM_ALWAYS_INLINE void _SetInputs(const runtime::TVMArgsSetter& setter,
                                           const Array<ObjectRef>& inputs) {
    setter(delta, inputs);
  }

  static void UnpackedApplyToSchedule(Schedule sch, Array<LoopRV> loop_rvs) {
    return sch->Reorder(loop_rvs);
  }

  static String UnpackedAsPython(Array<String> outputs, Array<String> loop_rvs) {
    PythonAPICall py("reorder");
    for (const String& loop_rv : loop_rvs) {
      py.Input("", loop_rv);
    }
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(SplitTraits);
TVM_REGISTER_INST_KIND_TRAITS(FuseTraits);
TVM_REGISTER_INST_KIND_TRAITS(ReorderTraits);

}  // namespace tir
}  // namespace tvm
