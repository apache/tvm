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
      return tvm::cast(var.dtype(), ret.value());
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
  explicit IterMapSimplifyBlockBinding(MapNode* opaque_blocks, Map<Var, Range> loop_var2extent,
                                       bool preserve_unit_iters)
      : opaque_blocks_(opaque_blocks),
        loop_var2extent_(loop_var2extent),
        preserve_unit_iters_(preserve_unit_iters) {}

  static For SimplifyBindings(Stmt stmt, const Array<StmtSRef>& loop_srefs, MapNode* opaque_blocks,
                              bool preserve_unit_iters) {
    Map<Var, Range> loop_var2extent;
    for (const StmtSRef& sref : loop_srefs) {
      const ForNode* loop = TVM_SREF_TO_FOR(sref);
      loop_var2extent.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
    }
    return Downcast<For>(IterMapSimplifyBlockBinding(opaque_blocks, std::move(loop_var2extent),
                                                     preserve_unit_iters)(std::move(stmt)));
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
    Array<PrimExpr> v =
        arith::IterMapSimplify(/*indices=*/op->iter_values,
                               /*input_iters=*/loop_var2extent_,
                               /*input_pred=*/op->predicate,
                               /*check_level=*/arith::IterMapLevel::Surjective,
                               /*analyzer=*/&analzyer_,
                               /*simplify_trivial_iterators=*/!preserve_unit_iters_);
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
  /*! \brief Internal analyzer */
  arith::Analyzer analzyer_;
  /*! \brief Whether or not to simplify unit iterators */
  bool preserve_unit_iters_;
};

class BlockPropertyError : public ScheduleError {
 public:
  /*!
   * \brief Check that all the blocks under the specific stmt have affine bindings
   *     wrt top loop sref and only have data-parallel or reduction block iters
   * \param self The state of the schedule
   * \param sref The sref to the specific stmt
   */
  static void CheckBlockIterTypeAndAffineBinding(const ScheduleState& self, const StmtSRefNode* top,
                                                 const StmtSRefNode* sref) {
    class BlockIterTypeAndAffineBindingChecker : public StmtVisitor {
     public:
      explicit BlockIterTypeAndAffineBindingChecker(const ScheduleState& state,
                                                    const StmtSRefNode* top)
          : state_(state), top_(top) {}

     private:
      void VisitStmt_(const BlockNode* op) final {
        for (const IterVar& iter_var : op->iter_vars) {
          if (iter_var->iter_type != kDataPar && iter_var->iter_type != kCommReduce) {
            throw BlockPropertyError(state_->mod, GetRef<Block>(op));
          }
          Optional<StmtSRef> high_exclusive =
              top_->parent ? GetRef<StmtSRef>(top_->parent) : Optional<StmtSRef>(NullOpt);
          CheckPartialAffineBinding(state_, GetRef<Block>(op), high_exclusive);
        }
      }
      const ScheduleState& state_;
      const StmtSRefNode* top_;
    };

    BlockIterTypeAndAffineBindingChecker checker(self, top);
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
  enum class PrimitiveKind { kFuse, kReorder };
  explicit DependentLoopError(IRModule mod, For loop, String inner_var, PrimitiveKind kind)
      : mod_(mod), loop_(std::move(loop)), inner_var_(std::move(inner_var)), kind_(kind) {}

  String FastErrorString() const final {
    if (kind_ == PrimitiveKind::kReorder) {
      return "ScheduleError: An outer loop's `min` or `extent` is dependent on an inner loop "
             "in the new order";
    } else {
      return "ScheduleError: A loop's `extent` is dependent on another loop";
    }
  }

  String DetailRenderTemplate() const final {
    if (kind_ == PrimitiveKind::kReorder) {
      return "Outer Loop {0}'s `min` or `extent` is dependent on an inner loop " + inner_var_ +
             " in the new order";
    } else {
      return "A loop {0}'s `extent` is dependent on another loop " + inner_var_;
    }
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {loop_}; }

  IRModule mod_;
  For loop_;
  String inner_var_;
  PrimitiveKind kind_;
};

Array<StmtSRef> Split(ScheduleState self, const StmtSRef& loop_sref, const Array<PrimExpr>& factors,
                      bool preserve_unit_iters, bool disable_predication) {
  // Invariance
  // - The total repeat number has not changed for each direct child block with updating predicate.
  // - The execution order has not changed. (The block executes with the same args and the same
  // order with before.
  // Step 1. Check correctness
  const ForNode* loop = TVM_SREF_TO_FOR(loop_sref);
  if (!loop->annotations.empty() || loop->thread_binding.defined()) {
    throw HasAnnotationOrThreadBindingError(self->mod, GetRef<For>(loop));
  }
  // Currently, loops not starting with 0 are not supported
  arith::Analyzer analyzer;
  CheckLoopStartsWithZero(self, loop_sref, &analyzer);

  // Find the most common dtype
  DataType dtype;
  {
    int bits = loop->loop_var.dtype().bits();
    for (const PrimExpr& factor : factors) {
      bits = std::max(bits, factor.dtype().bits());
    }
    dtype = DataType::Int(bits);
  }
  int n = factors.size();
  PrimExpr substitute_value = make_const(dtype, 0);
  std::vector<Var> new_loop_vars;
  new_loop_vars.reserve(n);
  for (int i = 0; i < n; i++) {
    const PrimExpr& factor = factors[i];
    Var var = loop->loop_var.copy_with_suffix("_" + std::to_string(i)).copy_with_dtype(dtype);
    substitute_value = substitute_value * factor + var;
    analyzer.Bind(var, Range::FromMinExtent(make_const(dtype, 0), tvm::cast(dtype, factor)));
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
  if (!disable_predication && !analyzer.CanProve(predicate, arith::ProofStrength::kSymbolicBound)) {
    new_stmt = BlockPredicateAppender(/*predicate=*/predicate)(std::move(new_stmt));
  }
  // Step 4. Generate nested loops to replace the original loop and simplify the binding
  for (int i = n - 1; i >= 0; i--) {
    new_stmt = For(new_loop_vars[i], 0, factors[i], ForKind::kSerial, new_stmt);
  }
  new_stmt = IterMapSimplifyBlockBinding::SimplifyBindings(std::move(new_stmt), GetLoops(loop_sref),
                                                           opaque_block_reuse.CopyOnWrite(),
                                                           preserve_unit_iters);
  self->Replace(loop_sref, new_stmt, opaque_block_reuse);
  Array<StmtSRef> result_srefs;
  result_srefs.reserve(n);
  for (int i = 0; i < n; i++) {
    result_srefs.push_back(self->stmt2ref.at(new_stmt.get()));
    const ForNode* outer_loop = TVM_TYPE_AS(new_stmt, ForNode);
    new_stmt = outer_loop->body;
  }
  return result_srefs;
}

class BufferIndicesMapExtractor : public StmtExprVisitor {
 public:
  explicit BufferIndicesMapExtractor(Var loop_var) : loop_var_(loop_var) {}

  static Map<String, Array<String>> Extract(Var loop_var, Block& block) {
    BufferIndicesMapExtractor extractor(loop_var);
    extractor(std::move(block->body));
    return extractor.buffer_indices_map;
  }

 private:
  void VisitStmt_(const BufferStoreNode* store) final {
    Array<String> indices;
    bool check_ = false;
    for (size_t i = 0; i < store->indices.size(); i++) {
      const VarNode* var_node = store->indices[i].as<VarNode>();
      if (var_node == nullptr) {
        check_ = true;
        break;
      }
      indices.push_back(var_node->name_hint);
    }
    if (buffer_indices_map.find(store->buffer->name) == buffer_indices_map.end() && !check_)
      buffer_indices_map.Set(store->buffer->name, indices);
    StmtExprVisitor::VisitStmt_(store);
  }

  void VisitExpr_(const BufferLoadNode* load) final {
    Array<String> indices;
    bool check_ = false;
    for (size_t i = 0; i < load->indices.size(); i++) {
      const VarNode* var_node = load->indices[i].as<VarNode>();
      if (var_node == nullptr) {
        check_ = true;
        break;
      }
      indices.push_back(var_node->name_hint);
    }
    if (buffer_indices_map.find(load->buffer->name) == buffer_indices_map.end() && !check_)
      buffer_indices_map.Set(load->buffer->name, indices);
    StmtExprVisitor::VisitExpr_(load);
  }

  void VisitStmt_(const BlockNode* op) final { StmtVisitor::VisitStmt_(op); }

  Var loop_var_;
  Map<String, Array<String>> buffer_indices_map;
};

Array<BufferRegion> MutateBufferRegion(Map<String, Array<String>> buffer_indices_map,
                                       Map<String, Range> index_range_map,
                                       Array<BufferRegion> region_arr) {
  // Update the region with new Ranges and return new BufferRegion
  Array<BufferRegion> new_region_arr =
      MutateArray(region_arr, [&buffer_indices_map, &index_range_map](const BufferRegion& region) {
        BufferRegion new_region = region;
        auto it = buffer_indices_map.find(new_region->buffer->name);
        if (it == buffer_indices_map.end()) return new_region;

        Array<String> old_indices = buffer_indices_map[new_region->buffer->name];
        Array<Range> new_ranges;
        for (size_t i = 0; i < old_indices.size(); i++) {
          new_ranges.push_back(index_range_map[old_indices[i]]);
        }
        new_region.CopyOnWrite()->region = std::move(new_ranges);
        return new_region;
      });
  return new_region_arr;
}

class BlockMutator : public StmtExprMutator {
 public:
  explicit BlockMutator(Var new_loop_var, PrimExpr min, PrimExpr extent)
      : new_loop_var_(new_loop_var), min_(min), extent_(extent) {}

 private:
  Stmt VisitStmt_(const BlockNode* _op) final {
    Block new_block = Downcast<Block>(StmtMutator::VisitStmt_(_op));

    // If iter_vars.size() is 0, then the block most probably be an Opaque block
    if (new_block->iter_vars.size() == 0 || inner_iter_var_index == -1) {
      new_block.CopyOnWrite()->name_hint =
          new_block.CopyOnWrite()->name_hint + "_" + new_loop_var_->name_hint;
      return std::move(new_block);
    }

    Var iter_var_ = new_block->iter_vars[inner_iter_var_index]->var;
    inner_iter_var_index = -1;
    // As we are working on cloned block, we need to create new instances of iter_var
    Array<IterVar> new_iter_vars =
        MutateArray(new_block->iter_vars, [this, &iter_var_](const IterVar& iter) {
          auto dtype = iter->var.dtype();
          // Create new Var instance for each IterVar
          Var new_var = Var(iter->var->name_hint, iter->var.dtype());
          IterVar new_iter = iter;
          new_iter.CopyOnWrite()->var = new_var;
          // Change the domain of IterVar corresponding to partitioned loop_var
          if (iter_var_.same_as(iter->var)) {
            new_iter.CopyOnWrite()->dom = Range(tvm::cast(dtype, min_), tvm::cast(dtype, extent_));
          }
          return new_iter;
        });

    // Update the IterVars of new_block
    if (!new_block->iter_vars.same_as(new_iter_vars)) {
      new_block.CopyOnWrite()->iter_vars = std::move(new_iter_vars);
      new_block.CopyOnWrite()->name_hint =
          new_block.CopyOnWrite()->name_hint + "_" + new_loop_var_->name_hint;
    }

    // Get the (iter_var, new Range) map
    Map<String, Range> index_range_map;
    for (size_t i = 0; i < new_block->iter_vars.size(); i++) {
      IterVar iter = new_block->iter_vars[i];
      index_range_map.Set(iter->var->name_hint, iter->dom);
    }

    // Get the (Buffer, indices) map
    Map<String, Array<String>> buffer_indices_map =
        BufferIndicesMapExtractor::Extract(new_loop_var_, new_block);
    Array<BufferRegion> new_writes =
        MutateBufferRegion(buffer_indices_map, index_range_map, new_block->writes);
    if (!new_block->writes.same_as(new_writes)) {
      // Update the writes with new_writes
      new_block.CopyOnWrite()->writes = std::move(new_writes);
    }
    Array<BufferRegion> new_reads =
        MutateBufferRegion(buffer_indices_map, index_range_map, new_block->reads);
    if (!new_block->reads.same_as(new_reads)) {
      // Update the reads with new_reads
      new_block.CopyOnWrite()->reads = std::move(new_reads);
    }

    Map<Var, Var> var_map;
    for (size_t i = 0; i < new_block->iter_vars.size(); i++) {
      var_map.Set(_op->iter_vars[i]->var, new_block->iter_vars[i]->var);
    }

    // Update all instances of old iter_vars in the block with new iter_vars
    auto block_stmt = tir::Substitute(new_block, var_map);
    return std::move(block_stmt);
  }

  Stmt VisitStmt_(const BlockRealizeNode* realize) final {
    Array<PrimExpr> iter_values = realize->iter_values;
    for (size_t i = 0; i < iter_values.size(); i++) {
      if (new_loop_var_.same_as(iter_values[i])) {
        // Get the iter_var index corresponding to loop_var iter_value index
        inner_iter_var_index = i;
        break;
      }
    }
    BlockRealize stmt = Downcast<BlockRealize>(StmtExprMutator::VisitStmt_(realize));
    return std::move(stmt);
  }

  Stmt VisitStmt_(const ForNode* op) final {
    For res = Downcast<For>(StmtMutator::VisitStmt_(op));
    Var new_var = Var(op->loop_var->name_hint, op->loop_var.dtype());

    if (!op->loop_var.same_as(new_var)) {
      // If the partioned loop contains nested for loop, then create new iteration variable instance
      res.CopyOnWrite()->body = std::move(tir::Substitute(res->body, {{op->loop_var, new_var}}));
      res.CopyOnWrite()->loop_var = new_var;
    }
    return res;
  }

  Var new_loop_var_;
  PrimExpr min_, extent_;
  int inner_iter_var_index = -1;
};

const String get_block_name(Stmt loop_body) {
  const BlockRealizeNode* blk_realize = loop_body.as<BlockRealizeNode>();
  if (blk_realize == nullptr) {
    return get_block_name(loop_body.as<ForNode>()->body);
  }
  return blk_realize->block->name_hint;
}

Array<StmtSRef> LoopPartition(ScheduleState self, const StmtSRef& loop_sref,
                              const Array<PrimExpr>& factors, bool preserve_unit_iters) {
  const ForNode* loop = TVM_SREF_TO_FOR(loop_sref);
  if (!loop->annotations.empty() || loop->thread_binding.defined()) {
    throw HasAnnotationOrThreadBindingError(self->mod, GetRef<For>(loop));
  }

  arith::Analyzer analyzer;
  // Find the most common dtype
  DataType dtype;
  {
    int bits = loop->loop_var.dtype().bits();
    for (const PrimExpr& factor : factors) {
      bits = std::max(bits, factor.dtype().bits());
    }
    dtype = DataType::Int(bits);
  }

  String block_name = get_block_name(loop->body) + "_" + loop->loop_var->name_hint;
  int n = factors.size();
  PrimExpr min_value = loop->min;
  PrimExpr extent_value;

  Array<Stmt> block_partitions;
  block_partitions.reserve(n);

  // Iterate over each pair of factors and create partition
  for (int i = 0; i < n; i++) {
    extent_value = analyzer.Simplify(factors[i]);
    Var new_loop_var = loop->loop_var.copy_with_suffix(std::to_string(i)).copy_with_dtype(dtype);
    Stmt loop_body = tir::Substitute(loop->body, {{loop->loop_var, new_loop_var}});

    // Create new block with new reference to each variable/stmt/expr in the existing block
    loop_body = BlockMutator(new_loop_var, min_value, extent_value)(std::move(loop_body));
    // Create new for loop with appropriate range
    auto for_node =
        For(new_loop_var, min_value, extent_value - min_value, ForKind::kSerial, loop_body);

    const auto& partition_block_name = block_name + std::to_string(i) + "_partition";
    // Create partition_block for the partitioned for loop
    BlockRealize partition_block({}, extent_value > 0,
                                 Block({}, {}, {}, partition_block_name, for_node));
    block_partitions.push_back(partition_block);

    min_value = extent_value;
  }

  // Create common block with all the partitioned blocks as its children blocks
  BlockRealize common({}, make_const(DataType::Bool(), 1),
                      Block({}, {}, {}, block_name + "_common", tir::SeqStmt(block_partitions)));

  // Replace existing loop with the newly created common block
  self->Replace(loop_sref, common, {});
  StmtSRef scope_sref = self->stmt2ref.at(common->block.get());
  StmtSRef scope_root = tir::GetScopeRoot(self, scope_sref, /*require_stage_pipeline=*/false);
  bool scope_block_affine_binding = self->IsAffineBlockBinding(scope_root);
  // Update the SRefTree for the newly created common block
  self->UpdateScopeBlockInfo(tir::GetBlockRealize(self, scope_root));
  self->block_info[scope_root].affine_binding = scope_block_affine_binding;

  // Collect the SRef for each partitioned loop and return
  Array<StmtSRef> partition_srefs;
  partition_srefs.reserve(n);
  for (int i = 0; i < n; i++) {
    StmtSRef partition_loop_sref =
        self->stmt2ref.at(block_partitions[i].as<BlockRealizeNode>()->block->body.get());
    partition_srefs.push_back(partition_loop_sref);
  }
  return partition_srefs;
}

class LoopReconstructor : private StmtMutator {
 public:
  explicit LoopReconstructor(Block scope_root, const std::vector<std::vector<For>>& loops)
      : scope_root_(scope_root), loops_(loops) {}

  using StmtMutator::operator();

  /*!
   * \brief Create the new nest loops induced by the given loops
   */
  void MakeNewLoop() {
    Array<Var> new_loop_vars;
    Array<PrimExpr> new_loop_extents;
    Array<Stmt> new_stmts;
    for (size_t i = 0; i < loops_.size(); i++) {
      Map<Var, PrimExpr> var_map;
      for (size_t j = 0; j < loops_[i].size(); j++) {
        if (i == 0) {
          Var merged_var = loops_[i][j]->loop_var.copy_with_suffix("_m");
          new_loop_vars.push_back(merged_var);
          new_loop_extents.push_back(loops_[i][j]->extent);
        }
        var_map.Set(loops_[i][j]->loop_var, new_loop_vars[j]);
      }
      auto new_stmt = Substitute(loops_[i][0]->body, var_map);
      new_stmts.push_back(new_stmt);
      this->need_remove_loop_.push_back(loops_[i].back());
    }
    auto new_loop = For(new_loop_vars[0], Integer(0), new_loop_extents[0], ForKind::kSerial,
                        SeqStmt(std::move(new_stmts)));
    this->new_inner_loop_ = new_loop;
    for (size_t i = 1; i < new_loop_vars.size(); ++i) {
      const Var& loop_var = new_loop_vars[i];
      const PrimExpr& loop_extent = new_loop_extents[i];
      new_loop = For(loop_var, Integer(0), loop_extent, ForKind::kSerial, new_loop);
    }
    this->new_outer_loop_ = new_loop;
  }

 private:
  Stmt VisitStmt_(const BlockNode* block) final {
    if (block != scope_root_.get()) {
      return GetRef<Block>(block);
    }
    return StmtMutator::VisitStmt_(block);
  }

  Stmt VisitStmt_(const ForNode* loop) final {
    if (GetRef<For>(loop) == need_remove_loop_.back()) {
      return new_outer_loop_;
    } else if (std::count(need_remove_loop_.begin(), need_remove_loop_.end(), GetRef<For>(loop))) {
      return Evaluate(0);
    }
    return StmtMutator::VisitStmt_(loop);
  }

  Stmt VisitStmt_(const SeqStmtNode* seq_stmt) final {
    auto ret = Downcast<SeqStmt>(StmtMutator::VisitSeqStmt_(seq_stmt, true));
    Array<Stmt> filtered;
    for (Stmt stmt : ret->seq) {
      if (!is_no_op(stmt)) {
        filtered.push_back(std::move(stmt));
      }
    }
    ret = SeqStmt(filtered);
    if (ret->size() == 0) {
      return Evaluate(0);
    } else if (ret->size() == 1) {
      return ret->seq[0];
    } else {
      return std::move(ret);
    }
  }

 public:
  /*! \brief The root block of the block scope */
  Block scope_root_;
  /*! \brief The given loops to be merge */
  const std::vector<std::vector<For>>& loops_;
  /*! \brief The outermost new loop to replace the original loop */
  For new_outer_loop_{nullptr};
  /*! \brief The innermost new loop to replace the original loop */
  For new_inner_loop_{nullptr};
  /*! \brief The loops to be removed */
  std::vector<For> need_remove_loop_;
};

StmtSRef Merge(ScheduleState self, const Array<StmtSRef>& loop_srefs) {
  // Invariance
  // - The total repeat number has not changed for each direct child block.
  // - The execution order has not changed. (The block executes with the same
  //   args and the same order with before.)
  arith::Analyzer analyzer;
  StmtSRef scope_root_sref;
  StmtSRef lca = GetSRefLowestCommonAncestor(loop_srefs);
  std::vector<std::vector<For>> lca_nest_loops;
  // Step 1. check correctness
  std::vector<For> nest_loop_loops;
  std::vector<PrimExpr> nest_loop_extents;
  for (size_t i = 0; i < loop_srefs.size(); i++) {
    const StmtSRef& sref = loop_srefs[i];
    auto scope_root_sref_ = GetScopeRoot(self, sref, /*require_stage_pipeline=*/false);
    std::vector<PrimExpr> nest_loop_i_extents;
    std::vector<For> nest_loop_i_loops;
    for (auto p = sref.get(); p != lca.get(); p = p->parent) {
      if (auto loop = p->StmtAs<ForNode>()) {
        if (!loop->annotations.empty() || loop->thread_binding.defined()) {
          throw HasAnnotationOrThreadBindingError(self->mod, GetRef<For>(loop));
        }
        CheckLoopStartsWithZero(self, GetRef<StmtSRef>(p), &analyzer);
        nest_loop_i_loops.push_back(GetRef<For>(loop));
        nest_loop_i_extents.push_back(loop->extent);
      }
    }
    lca_nest_loops.push_back(nest_loop_i_loops);
    const ForNode* outer_loop = nullptr;
    for (auto iter = nest_loop_i_loops.rbegin(); iter != nest_loop_i_loops.rend(); ++iter) {
      if (outer_loop && !outer_loop->body.same_as(*iter)) {
        throw NotOnlyChildError(self->mod, GetRef<For>(outer_loop), *iter);
      }
      outer_loop = (*iter).get();
    }
    if (i == 0) {
      scope_root_sref = scope_root_sref_;
      nest_loop_loops = nest_loop_i_loops;
      nest_loop_extents = nest_loop_i_extents;
    } else {
      if (scope_root_sref_.get() != scope_root_sref.get()) {
        LOG(FATAL) << "ScheduleError: Expected the loops to be under the same block scope.";
        throw;
      }
      if (nest_loop_i_extents.size() != nest_loop_extents.size()) {
        LOG(FATAL) << "ScheduleError: Merge loop's nesting depth must be same, but not.";
        throw;
      } else {
        for (size_t j = 0; j < nest_loop_i_extents.size(); j++) {
          if (!analyzer.CanProveEqual(nest_loop_i_extents[j], nest_loop_extents[j])) {
            LOG(FATAL) << "ScheduleError: Merge loop's `extent` must be same, but not."
                       << " extent=[" << j << "," << nest_loop_extents[j] << ","
                       << nest_loop_i_extents[j] << "]";
            throw;
          }
        }
      }
    }
  }
  // Step 2. Create merged loops and replace the original loops
  Block scope_root = GetRef<Block>(scope_root_sref->StmtAs<BlockNode>());
  LoopReconstructor reconstructor(scope_root, lca_nest_loops);
  reconstructor.MakeNewLoop();
  Block new_scope_root = Downcast<Block>(reconstructor(scope_root));
  // Step 3. Do the actual replacement
  self->Replace(scope_root_sref, new_scope_root, {{scope_root, new_scope_root}});
  return self->stmt2ref.at(reconstructor.new_inner_loop_.get());
}

StmtSRef Fuse(ScheduleState self, const Array<StmtSRef>& loop_srefs, bool preserve_unit_iters) {
  // Invariance
  // - The total repeat number has not changed for each direct child block.
  // - The execution order has not changed. (The block executes with the same
  //   args and the same order with before.)
  std::vector<const ForNode*> loops;
  loops.reserve(loop_srefs.size());
  StmtSRef outer_loop_sref{nullptr};
  const ForNode* outer_loop = nullptr;
  arith::Analyzer analyzer;
  std::unordered_set<const VarNode*> outer_loop_vars;
  // Step 1. check correctness
  for (const StmtSRef& sref : loop_srefs) {
    const ForNode* loop = TVM_SREF_TO_FOR(sref);
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
    CheckLoopStartsWithZero(self, sref, &analyzer);
    const VarNode* used_var = nullptr;
    auto f_contain = [&outer_loop_vars, &used_var](const VarNode* var) {
      if (outer_loop_vars.count(var)) {
        used_var = var;
        return true;
      }
      return false;
    };
    if (UsesVar(loop->extent, f_contain)) {
      throw DependentLoopError(self->mod, GetRef<For>(loop), used_var->name_hint,
                               DependentLoopError::PrimitiveKind::kFuse);
    }
    outer_loop_vars.insert(loop->loop_var.get());
    loops.push_back(loop);
  }
  // Step 2. Create fused loop var and replace the original loop vars
  std::string suffix;
  int n = loops.size();
  int bits = loops[0]->loop_var.dtype().bits();
  for (int i = 1; i < n; i++) {
    suffix += "_" + loops[i]->loop_var->name_hint;
    bits = std::max(bits, loops[i]->loop_var.dtype().bits());
  }
  suffix += "_fused";
  Var fused_var = loops[0]->loop_var.copy_with_suffix(suffix).copy_with_dtype(DataType::Int(bits));
  Array<PrimExpr> substitute_value;
  substitute_value.resize(loops.size());
  PrimExpr lower = 1;
  for (int i = static_cast<int>(loops.size()) - 1; i > 0; i--) {
    substitute_value.Set(i, is_one(loops[i]->extent)
                                ? 0
                                : floordiv(floormod(fused_var, lower * loops[i]->extent), lower));
    lower = lower * loops[i]->extent;
  }
  substitute_value.Set(0, is_one(loops[0]->extent) ? 0 : floordiv(fused_var, lower));
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
      std::move(new_stmt), GetLoops(loop_srefs[0]), opaque_block_reuse.CopyOnWrite(),
      preserve_unit_iters);
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
      const ForNode* loop = TVM_SREF_TO_FOR(loop_sref);
      throw LoopMultiAppearanceError(self->mod, GetRef<For>(loop));
    }
  }
  return loop_srefs;
}

/*!
 * \brief Get the top and bottom boundary of reorder range (which should be a chain)
 * \param self The schedule state
 * \param loop_srefs The set containing the srefs to the loops to be reordered
 * \return A pair containing the top and bottom boundary of the reorder range
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
 * \brief Get all the loops in the reorder range
 * \param self The schedule state
 * \param top The top boundary of the reorder range
 * \param bottom The bottom boundary of the reorder range
 * \return An array containing all the loops in the reorder range
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
 * \return The new loop chain
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
      throw DependentLoopError(self->mod, GetRef<For>(copy), used_var->name_hint,
                               DependentLoopError::PrimitiveKind::kReorder);
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
  // Step 1. Check uniqueness and collect the input loop srefs into a set
  std::unordered_set<const StmtSRefNode*> loop_srefs =
      CollectLoopsIntoSet(self, ordered_loop_srefs);
  // Step 2. Gather loops to be reordered
  // For each loop sref in the input sref array, traverse upwards along its parent pointer in the
  // sref tree, and stop on either a block, or a previously-visited loop
  // - the top of the reorder range is the last loop visited in the first traversal which exists in
  //   the input array
  // - the bottom of the reorder range is the last loop in the input array which is not visited in
  // the previous traversals
  auto [top, bottom] = GetBoundaryOfReorderRange(self, loop_srefs);
  // Step 3. Collect all loops in the chain and check the loops are single-branch
  std::vector<const StmtSRefNode*> chain = GetLoopsInReorderRange(self, top, bottom);
  // Step 4. Check the block below has all its block_var to be data-parallel or reduction,
  // and the block has an affine binding wrt top of the loop range.
  BlockPropertyError::CheckBlockIterTypeAndAffineBinding(self, top, bottom);
  // Step 5. Replace the original loops with the reordered loops and check that outer loop is
  // not dependent on inner loop
  For new_loop = ConstructNewLoopChain(self, std::move(chain), ordered_loop_srefs, loop_srefs);
  self->Replace(GetRef<StmtSRef>(top), new_loop, {});
}

StmtSRef AddUnitLoop(ScheduleState self, StmtSRef sref) {
  if (sref->stmt->IsInstance<ForNode>()) {
    For new_loop(Var("u", DataType::Int(32)), 0, 1, ForKind::kSerial, GetRef<Stmt>(sref->stmt));
    self->Replace(sref, new_loop, {});
    return self->stmt2ref.at(new_loop.get());
  }
  class NewLoopCreator : public StmtMutator {
   public:
    explicit NewLoopCreator(const StmtNode* src_block) : src_block_(src_block) {}

    Stmt VisitStmt_(const BlockRealizeNode* realize) final {
      if (realize->block.get() == src_block_) {
        new_loop_ =
            For(Var("u", DataType::Int(32)), 0, 1, ForKind::kSerial, GetRef<BlockRealize>(realize));
        return new_loop_;
      }
      return StmtMutator::VisitStmt_(realize);
    }

    const StmtNode* src_block_;
    For new_loop_{nullptr};
  };

  CHECK(sref->parent != nullptr) << "ValueError: Cannot add loops on top of the root block";
  StmtSRef parent_sref = GetRef<StmtSRef>(sref->parent);
  NewLoopCreator creator(sref->stmt);
  Stmt new_stmt = creator(GetRef<Stmt>(parent_sref->stmt));
  if (new_stmt->IsInstance<ForNode>()) {
    self->Replace(parent_sref, std::move(new_stmt), {});
  } else {
    Block old_parent_block = GetRef<Block>(parent_sref->StmtAs<BlockNode>());
    Block new_parent_block = Downcast<Block>(new_stmt);
    self->Replace(parent_sref, new_stmt, {{old_parent_block, new_parent_block}});
  }
  return self->stmt2ref.at(creator.new_loop_.get());
}

/******** InstructionKind Registration ********/

struct SplitTraits : public UnpackedInstTraits<SplitTraits> {
  static constexpr const char* kName = "Split";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 2;
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
                                               Array<Optional<ExprRV>> factors,
                                               Bool preserve_unit_iters, Bool disable_predication) {
    return sch->Split(loop_rv, factors, preserve_unit_iters.operator bool(),
                      disable_predication.operator bool());
  }

  static String UnpackedAsPython(Array<String> outputs, String loop_rv, Array<ObjectRef> factors,
                                 Bool preserve_unit_iters, Bool disable_predication) {
    PythonAPICall py("split");
    py.Input("loop", loop_rv);
    py.Input("factors", factors);
    py.Input("preserve_unit_iters", preserve_unit_iters.operator bool());
    py.Input("disable_predication", disable_predication.operator bool());
    py.OutputList(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct LoopPartitionTraits : public UnpackedInstTraits<LoopPartitionTraits> {
  static constexpr const char* kName = "LoopPartition";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 1;
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
                                               Array<Optional<ExprRV>> factors,
                                               Bool preserve_unit_iters) {
    return sch->LoopPartition(loop_rv, factors, preserve_unit_iters.operator bool());
  }

  static String UnpackedAsPython(Array<String> outputs, String loop_rv, Array<ObjectRef> factors,
                                 Bool preserve_unit_iters) {
    PythonAPICall py("loop_partition");
    py.Input("loop", loop_rv);
    py.Input("factors", factors);
    py.Input("preserve_unit_iters", preserve_unit_iters.operator bool());
    py.OutputList(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct MergeTraits : public UnpackedInstTraits<MergeTraits> {
  static constexpr const char* kName = "Merge";
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
    return sch->Merge(loop_rvs);
  }

  static String UnpackedAsPython(Array<String> outputs, Array<String> loop_rvs) {
    PythonAPICall py("merge");
    for (const String& loop_rv : loop_rvs) {
      py.Input("", loop_rv);
    }
    py.SingleOutput(outputs);
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
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  template <size_t delta>
  static TVM_ALWAYS_INLINE void _SetInputs(const runtime::TVMArgsSetter& setter,
                                           const Array<ObjectRef>& inputs) {
    setter(delta, inputs);
  }

  static LoopRV UnpackedApplyToSchedule(Schedule sch, Array<LoopRV> loop_rvs,
                                        Bool preserve_unit_iters) {
    return sch->Fuse(loop_rvs, preserve_unit_iters.operator bool());
  }

  static String UnpackedAsPython(Array<String> outputs, Array<String> loop_rvs,
                                 Bool preserve_unit_iters) {
    PythonAPICall py("fuse");
    for (const String& loop_rv : loop_rvs) {
      py.Input("", loop_rv);
    }
    py.Input("preserve_unit_iters", preserve_unit_iters.operator bool());
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

struct AddUnitLoopTraits : public UnpackedInstTraits<AddUnitLoopTraits> {
  static constexpr const char* kName = "AddUnitLoop";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static LoopRV UnpackedApplyToSchedule(Schedule sch, ObjectRef rv) {
    if (auto block = rv.as<BlockRV>()) {
      return sch->AddUnitLoop(block.value());
    } else if (auto loop = rv.as<LoopRV>()) {
      return sch->AddUnitLoop(loop.value());
    } else {
      LOG(FATAL) << "TypeError: AddUnitLoop expects a loop or block";
      throw;
    }
  }

  static String UnpackedAsPython(Array<String> outputs, String rv) {
    PythonAPICall py("add_unit_loop");
    py.Input("block_or_loop", rv);
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(SplitTraits);
TVM_REGISTER_INST_KIND_TRAITS(LoopPartitionTraits);
TVM_REGISTER_INST_KIND_TRAITS(MergeTraits);
TVM_REGISTER_INST_KIND_TRAITS(FuseTraits);
TVM_REGISTER_INST_KIND_TRAITS(ReorderTraits);
TVM_REGISTER_INST_KIND_TRAITS(AddUnitLoopTraits);

}  // namespace tir
}  // namespace tvm
