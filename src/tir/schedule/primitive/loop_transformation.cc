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

class BlockIterTypeError : public ScheduleError {
 public:
  explicit BlockIterTypeError(IRModule mod, Block block)
      : mod_(std::move(mod)), block_(std::move(block)) {}

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
      : mod_(std::move(mod)), loop_(std::move(loop)) {}

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
      : mod_(std::move(mod)), outer_(std::move(outer)), inner_(std::move(inner)) {}

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
      : mod_(std::move(mod)), outer_(std::move(outer)), inner_(std::move(inner)) {}

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
  explicit LoopNotStartWithZeroError(IRModule mod, For loop)
      : mod_(std::move(mod)), loop_(std::move(loop)) {}

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
  explicit NotSingleInferFactorError(IRModule mod) : mod_(std::move(mod)) {}

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
  explicit WrongFactorProductError(IRModule mod, For loop)
      : mod_(std::move(mod)), loop_(std::move(loop)) {}

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
  explicit LoopMultiAppearanceError(IRModule mod, For loop)
      : mod_(std::move(mod)), loop_(std::move(loop)) {}

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

class LoopsNotALineError : public ScheduleError {
 public:
  enum ProblemKind { kNotUnderAScope, kHaveNonSingleBranchStmt };

  explicit LoopsNotALineError(IRModule mod, Optional<Stmt> problematic_loop, ProblemKind kind)
      : mod_(std::move(mod)), problematic_loop_(std::move(problematic_loop)), kind_(kind) {}

  String FastErrorString() const final { return "ScheduleError: the loops are not in a line"; }

  String DetailRenderTemplate() const final {
    std::stringstream ss;
    ss << "The loops are not in a line because";
    if (kind_ == kNotUnderAScope) {
      ss << " they are not under the same scope.";
    } else {
      ss << " there is a non-single-branch stmt in between. Problematic stmt: {0}";
    }
    return ss.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final {
    if (kind_ == kNotUnderAScope) {
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
      : mod_(std::move(mod)), loop_(std::move(loop)), inner_var_(std::move(inner_var)) {}

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

/*!
 * \brief Collect all loops under a specific block scope in the inverse pre-order
 * \param self The state of the schedule
 * \param root_block_sref the sref to the root of block scope
 * \return The array of srefs of all loops under the block scope, in inverse pre-order
 */
std::vector<const StmtSRefNode*> GetLoopsInversePreOrderUnderScope(
    const ScheduleState& self, const StmtSRef& root_block_sref) {
  std::vector<const StmtSRefNode*> loops;
  const BlockNode* root_block = TVM_SREF_TO_BLOCK(root_block, root_block_sref);
  // Gather all the loops under parent_block
  PreOrderVisit(root_block->body, [&loops, self](const ObjectRef& node) {
    // Stops at a new BlockNode
    if (node->IsInstance<BlockNode>()) {
      return false;
    }
    // Collects every ForNode
    if (const auto* loop = node.as<ForNode>()) {
      loops.push_back(self->stmt2ref.at(loop).operator->());
    }
    return true;
  });
  // Reverse to get inverse preorder
  std::reverse(loops.begin(), loops.end());
  return loops;
}
/*!
 * \brief Check that all the blocks under the specific stmt have affine bindings and only have
 *     data-parallel or reduction block iters
 * \param self The state of the schedule
 * \param sref The sref to the specific stmt
 */
void CheckBlockIterTypeAndAffineBinding(const ScheduleState& self, const StmtSRefNode* sref) {
  class BlockIterTypeAndAffineBindingChecker : public StmtVisitor {
   public:
    explicit BlockIterTypeAndAffineBindingChecker(const ScheduleState& state) : state_(state) {}

   private:
    void VisitStmt_(const BlockNode* op) final {
      for (const IterVar& iter_var : op->iter_vars) {
        if (iter_var->iter_type != kDataPar && iter_var->iter_type != kCommReduce) {
          throw BlockIterTypeError(state_->mod, GetRef<Block>(op));
        }
        CheckAffineBinding(state_, GetRef<Block>(op));
      }
    }
    const ScheduleState& state_;
  };

  BlockIterTypeAndAffineBindingChecker checker(self);
  checker(GetRef<Stmt>(sref->stmt));
}

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

void Reorder(ScheduleState self, const Array<StmtSRef>& ordered_loop_srefs) {
  std::unordered_set<const StmtSRefNode*> loop_srefs;
  loop_srefs.reserve(ordered_loop_srefs.size());
  if (ordered_loop_srefs.empty() || ordered_loop_srefs.size() == 1) {
    return;
  }
  // Step 1. check uniqueness
  for (const StmtSRef loop_sref : ordered_loop_srefs) {
    const ForNode* loop = TVM_SREF_TO_FOR(loop, loop_sref);
    // uniqueness check
    auto inserted = loop_srefs.insert(loop_sref.get());
    if (!inserted.second) {
      throw LoopMultiAppearanceError(self->mod, GetRef<For>(loop));
    }
  }
  // Step 2. gather loops to be reordered
  // The algorithm is to scan the inverse preorder of the whole loop tree in the scope.
  // For some Loop x, it is potentially in the reorder range if
  //   - x is in the reorder list
  //   - x has only one child which is a loop and is potentially in the reorder range
  // After the inverse DFS, we can know the exact reorder range
  // `top` and `bottom` denote the boundary of the loop range that need reordering
  const StmtSRefNode* top = nullptr;
  const StmtSRefNode* bottom = nullptr;
  // Maps a parent sref to its child sref
  std::unordered_map<const StmtSRefNode*, const StmtSRefNode*> successor;
  int n_loops_not_found = ordered_loop_srefs.size();
  // Gather all the loops under the block scope
  std::vector<const StmtSRefNode*> inverse_preorder_loops = GetLoopsInversePreOrderUnderScope(
      self, GetScopeRoot(self, ordered_loop_srefs[0], /*require_stage_pipeline=*/true));
  for (const StmtSRefNode* loop : inverse_preorder_loops) {
    bool is_in_reorder_list = loop_srefs.count(loop);
    bool has_successor_in_reorder_list = successor.count(loop);
    if (is_in_reorder_list || has_successor_in_reorder_list) {
      const StmtSRefNode* parent = loop->parent;
      // If the successor of `parent` exists, then `parent` can't be a single-branch loop
      auto inserted = successor.insert({parent, loop});
      if (!inserted.second) {
        throw LoopsNotALineError(self->mod, GetRef<Stmt>(parent->stmt),
                                 LoopsNotALineError::kHaveNonSingleBranchStmt);
      }
      // `bottom` is the first loop encountered
      if (bottom == nullptr) {
        bottom = loop;
      }
      // `top` is the last loop encountered
      if (is_in_reorder_list) {
        top = loop;
        --n_loops_not_found;
      }
    }
  }
  // Step 3. Check loops are in the same block scope
  if (n_loops_not_found != 0) {
    throw LoopsNotALineError(self->mod, NullOpt, LoopsNotALineError::kNotUnderAScope);
  }
  // Step 4. Check that loops are single-branch
  const ForNode* outer_loop = TVM_SREF_TO_FOR(outer_loop, GetRef<StmtSRef>(top));
  for (const StmtSRefNode* loop_sref = top; loop_sref != bottom;) {
    loop_sref = successor[loop_sref];
    const ForNode* inner_loop = TVM_SREF_TO_FOR(inner_loop, GetRef<StmtSRef>(loop_sref));
    if (outer_loop->body.get() != inner_loop) {
      throw LoopsNotALineError(self->mod, GetRef<For>(outer_loop),
                               LoopsNotALineError::kHaveNonSingleBranchStmt);
    }
    outer_loop = inner_loop;
  }
  // Step 5. Check the block below has all its block_var to be data-parallel or reduction
  CheckBlockIterTypeAndAffineBinding(self, bottom);
  // Step 6. Replace the original loops with the reordered loops and check that outer loop is
  // not dependent on inner loop
  std::unordered_set<const VarNode*> inner_vars;
  std::function<Stmt(const StmtSRefNode*, int index)> f_reorder =
      [&bottom, &loop_srefs, &successor, &ordered_loop_srefs, &inner_vars, &self, &f_reorder](
          const StmtSRefNode* loop, int index) -> Stmt {
    const ForNode* copy = loop_srefs.count(loop) ? ordered_loop_srefs[index++]->StmtAs<ForNode>()
                                                 : loop->StmtAs<ForNode>();
    ObjectPtr<ForNode> n = make_object<ForNode>(*copy);
    if (loop == bottom) {
      // stop recursion at bottom loop
      n->body = loop->StmtAs<ForNode>()->body;
    } else {
      // reorder recursively
      n->body = f_reorder(successor.at(loop), index);
    }
    const VarNode* used_var;
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
    return Stmt(std::move(n));
  };
  self->Replace(GetRef<StmtSRef>(top), f_reorder(top, 0), {});
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

  friend struct UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(SplitTraits);
TVM_REGISTER_INST_KIND_TRAITS(FuseTraits);
TVM_REGISTER_INST_KIND_TRAITS(ReorderTraits);

}  // namespace tir
}  // namespace tvm
