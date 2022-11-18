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

#include <optional>
#include <variant>

#include "../../../arith/ir_mutator_with_analyzer.h"
#include "../utils.h"

namespace tvm {
namespace tir {

/*! \brief Planning stage prior to rewriting in TransformLayoutRewriter
 *
 * There are four ways that transformation may be handled.  Each
 * updates the buffer shape and the indices used to acces the buffer
 * in BufferStore/BufferLoad nodes, but differ in how they handle the
 * `pad_value`.  In order of preference, the different strategies are
 * as follows:
 *
 * 1. NoPaddingRequired.  The transformation does not introduce
 * padding, so only local changes to update the indices of
 * BufferLoad/BufferStore nodes are required.  No blocks are added,
 * removed, or replaced.
 *
 * 2. ProloguePlan.  The transformation introduces padding, but the
 * analyzed block has no write stages for the transformed buffer.
 * This buffer is an input and the caller is responsible for ensuring
 * that the padding contains the specified `pad_value`.  The generated
 * prologue contains `builtin::assume()` calls that will expose this
 * known value during scheduling/simplification, but will be removed
 * during lowering.
 *
 * 3. ReplacementPlan.  The transformation introduces padding, has at
 * least one write stage for the transformed buffer, and at least one
 * of those write stages writes to all pre-transformation indices
 * following a row-major traversal.  These write stage is rewritten to
 * be row-major traversals of the post-transformation indices, with a
 * `tir::if_then_else` call to write either the specified `pad_value`
 * into padding or the computed value into non-padding.
 *
 * 4. EpiloguePlan.  The transformation introduces padding, has at
 * least one write stage for the transformed buffer, but no write
 * stage can be rewritten to use `tir::if_then_else`.  The
 * transformation still requires the `pad_value` to be written into
 * the padding, so a new block is inserted after the last write stage
 * to explicitly fill the padding.
 *
 */
class TransformLayoutPlanner : private StmtExprVisitor {
 public:
  // Statement to be inserted prior to the analyzed block
  struct ProloguePlan {
    Stmt prologue;
  };

  // Loops within the analyzed block that should be replaced
  struct ReplacementPlan {
    Map<For, Stmt> replacements;
    Map<Block, Block> new_block_to_old;
  };

  // The block to be inserted, along with the location at which it
  // should be inserted.  The location will be either a For or a
  // Block, and will be after all writes the transformed buffer.
  struct EpiloguePlan {
    Stmt insert_after;
    Stmt new_block;
  };

  struct NoPaddingRequired {};

  using TransformPlan =
      std::variant<ProloguePlan, ReplacementPlan, EpiloguePlan, NoPaddingRequired>;

  static TransformPlan Plan(Block block, Buffer old_buffer, Buffer new_buffer, IndexMap index_map,
                            IndexMap inverse, PrimExpr padding_predicate,
                            Optional<IndexMap> pad_value) {
    ICHECK(!pad_value.defined() || pad_value.value()->final_indices.size() == 1)
        << "Internal error: Should be caught by ScheduleError checks prior to this point";
    TransformLayoutPlanner visitor(old_buffer);
    visitor(block);
    return visitor.Finalize(new_buffer, index_map, inverse, padding_predicate, pad_value);
  }

 private:
  struct WriteInfo {
    // The BufferStore object
    BufferStore store;

    // The block realize that contains the store, if any.
    Optional<BlockRealize> innermost_block_realize;

    // The nested loops whose values contribute to the indices used in
    // the store.  Not all loop variables in the loopnest need to
    // contribute, but the first and last must.
    std::vector<For> dependent_loopnest;

    // Whether the padding could be represented as a tir::if_then_else
    // node.  This requires that the surrounding loop iterators
    // iterate over all pre-transformation buffer axes, that there are
    // no data dependencies between loop iterations, and that
    bool contains_row_major_traversal{false};
  };

  explicit TransformLayoutPlanner(Buffer old_buffer) : old_buffer_(old_buffer) {}

  void VisitStmt_(const ForNode* op) override {
    BindLoopVar context(this, GetRef<For>(op));
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const LetStmtNode* op) override {
    BindVariableDefinition context(this, op->var, op->value);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BlockRealizeNode* op) override {
    BindBlockRealize context(this, GetRef<BlockRealize>(op));
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferStoreNode* op) override {
    if (!op->buffer.same_as(old_buffer_)) {
      return;
    }

    std::optional<std::pair<size_t, size_t>> loop_dependency_range = std::nullopt;
    for (const auto& index : op->indices) {
      if (auto index_depth = LoopDependencyRange(index); index_depth.has_value()) {
        if (loop_dependency_range) {
          loop_dependency_range = {
              std::min(loop_dependency_range.value().first, index_depth.value().first),
              std::max(loop_dependency_range.value().second, index_depth.value().second)};
        } else {
          loop_dependency_range = index_depth;
        }
      }
    }

    WriteInfo write_info;
    write_info.store = GetRef<BufferStore>(op);
    if (loop_dependency_range) {
      size_t i = loop_dependency_range.value().first;
      size_t j = loop_dependency_range.value().second;
      ICHECK_LT(i, active_loops_.size());
      ICHECK_LT(j, active_loops_.size());

      write_info.dependent_loopnest = {active_loops_.begin() + i, active_loops_.begin() + j + 1};
    }
    write_info.innermost_block_realize = innermost_block_realize_;

    write_info.contains_row_major_traversal = [&]() -> bool {
      const auto& loopnest = write_info.dependent_loopnest;
      if (loopnest.empty()) {
        return false;
      }

      if (loopnest.size() != old_buffer_->shape.size() || loopnest.size() != op->indices.size()) {
        return false;
      }

      for (size_t i = 0; i < loopnest.size(); i++) {
        const For& loop = loopnest[i];
        const PrimExpr& buffer_dim = old_buffer_->shape[i];
        PrimExpr index = Substitute(op->indices[i], active_var_bindings_);
        bool is_loop_over_axis = index.same_as(loop->loop_var) && is_const_int(loop->min, 0) &&
                                 ExprDeepEqual()(loop->extent, buffer_dim) &&
                                 loop->kind == ForKind::kSerial;
        if (!is_loop_over_axis) {
          return false;
        }
      }

      return true;
    }();

    write_info_.push_back(write_info);

    // Don't need to continue recursing, as the entire goal was to
    // find the BufferStore.
  }

  std::optional<std::pair<size_t, size_t>> LoopDependencyRange(const PrimExpr& expr) const {
    std::optional<std::pair<size_t, size_t>> prev = std::nullopt;
    for (const auto& var : UndefinedVars(expr)) {
      auto it = loop_depth_lookup_.find(var.get());
      if (it != loop_depth_lookup_.end()) {
        if (prev.has_value()) {
          prev = {std::min(prev.value().first, it->second.first),
                  std::max(prev.value().second, it->second.second)};
        } else {
          prev = it->second;
        }
      }
    }

    return prev;
  }

  class BufferStoreReplacer : public StmtExprMutator {
   public:
    BufferStoreReplacer(const WriteInfo& info, const Buffer& new_buffer, PrimExpr padding_predicate,
                        const IndexMap& inverse, const Optional<IndexMap>& pad_value,
                        Map<Block, Block>* new_block_to_old)
        : info(info),
          new_buffer(new_buffer),
          new_indices(inverse->initial_indices.Map([](const Var& var) -> PrimExpr { return var; })),
          padding_predicate(padding_predicate),
          inverse(inverse),
          pad_value(pad_value),
          new_block_to_old(*new_block_to_old) {
      ICHECK_EQ(info.dependent_loopnest.size(), inverse->final_indices.size());
      for (size_t i = 0; i < info.dependent_loopnest.size(); i++) {
        Var var = info.dependent_loopnest[i]->loop_var;
        PrimExpr expr = inverse->final_indices[i];
        var_remap.Set(var, expr);
      }

      DefineBlockUpdates();
    }

    bool is_all_stores_replaced() const { return all_stores_replaced; }

   private:
    void DefineBlockUpdates() {
      if (!info.innermost_block_realize) {
        return;
      }

      BlockRealize block_realize = info.innermost_block_realize.value();
      const auto& block = block_realize->block;
      const Array<PrimExpr>& old_indices = info.store->indices;
      const auto& old_iter_vars = block->iter_vars;

      this->new_iter_vars = old_iter_vars;
      this->new_iter_values = block_realize->iter_values;

      if (old_indices.empty()) {
        return;
      }

      // Find the block iterators that are used to access the buffer.  Must be in the same
      // order as they appear in the indices.
      if (block->iter_vars.size() < old_indices.size()) {
        return;
      }

      size_t block_index_start = 0;
      for (; block_index_start < old_iter_vars.size() - old_indices.size(); block_index_start++) {
        if (old_indices[0].same_as(old_iter_vars[block_index_start]->var)) {
          break;
        }
      }
      if (block_index_start > old_iter_vars.size() - old_indices.size()) {
        return;
      }

      for (size_t i = 0; i < old_indices.size(); i++) {
        if (!old_indices[i].same_as(old_iter_vars[block_index_start + i]->var) ||
            old_iter_vars[block_index_start + i]->iter_type != kDataPar) {
          return;
        }
      }

      // If we got to this point, all indices used to access the
      // buffer are virtual indices defined in the innermost block.
      // Therefore, generate new virtual indices for iterating over
      // the post-transform buffer.

      new_indices = inverse->initial_indices.Map([](Var var) -> PrimExpr {
        std::stringstream ss;
        ss << "v_" << var->name_hint;
        return Var(ss.str(), var.dtype());
      });

      Map<Var, PrimExpr>
          loop_var_to_virtual_var;  // For updating padding_predicate in terms of the new indices
      Array<PrimExpr> new_iter_values;  // For BlockRealize
      Array<IterVar> new_iter_vars;     // For Block

      for (size_t i = 0; i < block_index_start; i++) {
        new_iter_vars.push_back(old_iter_vars[i]);
        new_iter_values.push_back(block_realize->iter_values[i]);
      }

      ICHECK_EQ(new_indices.size(), new_buffer->shape.size());
      for (size_t i = 0; i < new_indices.size(); i++) {
        Var var = inverse->initial_indices[i];
        Var virtual_var = Downcast<Var>(new_indices[i]);
        PrimExpr dim = new_buffer->shape[i];
        new_iter_values.push_back(var);
        new_iter_vars.push_back(
            IterVar(Range::FromMinExtent(make_zero(dim.dtype()), dim), virtual_var, kDataPar));
        loop_var_to_virtual_var.Set(var, virtual_var);
      }

      for (size_t i = block_index_start + old_indices.size(); i < old_iter_vars.size(); i++) {
        new_iter_vars.push_back(old_iter_vars[i]);
        new_iter_values.push_back(block_realize->iter_values[i]);
      }

      ICHECK_EQ(inverse->final_indices.size(), old_indices.size());
      for (size_t i = 0; i < old_indices.size(); i++) {
        Var var = Downcast<Var>(old_indices[i]);
        PrimExpr expr = Substitute(inverse->final_indices[i], loop_var_to_virtual_var);
        var_remap.Set(var, expr);
      }

      padding_predicate = Substitute(padding_predicate, loop_var_to_virtual_var);

      this->new_iter_vars = new_iter_vars;
      this->new_iter_values = new_iter_values;
    }

    Stmt VisitStmt_(const BufferStoreNode* op) final {
      bool can_replace = [&]() -> bool {
        if (!op->buffer.same_as(info.store->buffer)) {
          return false;
        }

        const Array<PrimExpr>& old_indices = info.store->indices;

        ICHECK_EQ(old_indices.size(), op->indices.size());
        ExprDeepEqual expr_equal;
        for (size_t i = 0; i < old_indices.size(); i++) {
          if (!expr_equal(old_indices[i], op->indices[i])) {
            return false;
          }
        }
        return true;
      }();

      BufferStore store = GetRef<BufferStore>(op);
      if (can_replace) {
        PrimExpr pad_value_at_index = pad_value.value()->MapIndices(new_indices)[0];
        store =
            BufferStore(new_buffer, if_then_else(padding_predicate, pad_value_at_index, op->value),
                        new_indices);
      } else {
        all_stores_replaced = false;
      }
      return StmtExprMutator::VisitStmt_(store.get());
    }

    Stmt VisitStmt_(const BlockRealizeNode* op) final {
      BlockRealize realize = Downcast<BlockRealize>(StmtExprMutator::VisitStmt_(op));

      if (op == info.innermost_block_realize.get()) {
        Block block = realize->block;
        if (!block->iter_vars.same_as(this->new_iter_vars)) {
          block.CopyOnWrite()->iter_vars = this->new_iter_vars;
          RecordReplacement(op->block, block);
        }

        if (!block.same_as(realize->block) ||
            !realize->iter_values.same_as(this->new_iter_values)) {
          auto write_ptr = realize.CopyOnWrite();
          write_ptr->block = block;
          write_ptr->iter_values = this->new_iter_values;
        }
      }

      return std::move(realize);
    }

    Stmt VisitStmt_(const BlockNode* op) final {
      Block orig = GetRef<Block>(op);
      Block mutated = Downcast<Block>(StmtExprMutator::VisitStmt_(op));

      RecordReplacement(orig, mutated);
      return std::move(mutated);
    }

    PrimExpr VisitExpr_(const VarNode* op) final {
      Var var = GetRef<Var>(op);
      if (auto opt = var_remap.Get(var)) {
        return opt.value();
      } else {
        return std::move(var);
      }
    }

    void RecordReplacement(Block before, Block after) {
      if (before.same_as(after)) {
        return;
      }

      ICHECK(!new_block_to_old.count(after));

      while (true) {
        if (auto opt = new_block_to_old.Get(before)) {
          before = opt.value();
        } else {
          break;
        }
      }

      new_block_to_old.Set(after, before);
    }

    const WriteInfo& info;
    const Buffer& new_buffer;
    Array<PrimExpr> new_indices;
    Array<IterVar> new_iter_vars;
    Array<PrimExpr> new_iter_values;
    PrimExpr padding_predicate;
    const IndexMap& inverse;
    const Optional<IndexMap>& pad_value;
    Map<Block, Block>& new_block_to_old;
    bool all_stores_replaced{true};

    Map<Var, PrimExpr> var_remap;
  };

  TransformPlan Finalize(Buffer new_buffer, IndexMap index_map, IndexMap inverse,
                         PrimExpr padding_predicate, Optional<IndexMap> pad_value) const {
    if (auto prologue_plan =
            FinalizeProloguePlan(new_buffer, index_map, inverse, padding_predicate, pad_value);
        prologue_plan.has_value()) {
      return prologue_plan.value();
    } else if (auto replacement_plan = FinalizeReplacementPlan(new_buffer, index_map, inverse,
                                                               padding_predicate, pad_value);
               replacement_plan.has_value()) {
      return replacement_plan.value();
    } else if (auto epilogue_plan = FinalizeEpiloguePlan(new_buffer, index_map, inverse,
                                                         padding_predicate, pad_value);
               epilogue_plan.has_value()) {
      return epilogue_plan.value();
    } else {
      return NoPaddingRequired();
    }
  }

  std::optional<ProloguePlan> FinalizeProloguePlan(Buffer new_buffer, IndexMap index_map,
                                                   IndexMap inverse, PrimExpr padding_predicate,
                                                   Optional<IndexMap> pad_value) const {
    if (write_info_.size() || is_zero(padding_predicate) || !pad_value.defined()) {
      return std::nullopt;
    }

    Array<IterVar> iter_vars;
    Array<PrimExpr> iter_values;
    Array<PrimExpr> indices;
    Map<Var, PrimExpr> loop_indices_to_block_indices;
    ICHECK_EQ(inverse->initial_indices.size(), new_buffer->shape.size());
    for (size_t i = 0; i < inverse->initial_indices.size(); i++) {
      const auto& loop_var = inverse->initial_indices[i];
      const auto& dim = new_buffer->shape[i];
      Var block_var("v_" + loop_var->name_hint, loop_var->dtype);
      IterVar iter_var(Range(0, dim), block_var, kDataPar);
      loop_indices_to_block_indices.Set(loop_var, block_var);
      indices.push_back(iter_var->var);
      iter_vars.push_back(iter_var);
      iter_values.push_back(loop_var);
    }
    padding_predicate = Substitute(std::move(padding_predicate), loop_indices_to_block_indices);

    PrimExpr pad_value_at_index = pad_value.value()->MapIndices(indices)[0];
    PrimExpr expr = (!padding_predicate) || (BufferLoad(new_buffer, indices) == pad_value_at_index);
    Stmt stmt = Evaluate(Call(DataType::Bool(), builtin::assume(), {expr}));

    std::stringstream block_name;
    block_name << "buffer_" << new_buffer->name << "_assumptions";
    auto read_region = BufferRegion::FromPoint(new_buffer, indices);
    stmt = BlockRealize(iter_values, Bool(true),
                        Block(iter_vars, {read_region}, {}, block_name.str(), stmt));

    for (size_t rev_i = 0; rev_i < inverse->initial_indices.size(); rev_i++) {
      size_t i = (inverse->initial_indices.size() - 1) - rev_i;
      Var loop_var = inverse->initial_indices[i];
      PrimExpr extent = new_buffer->shape[i];
      stmt = For(loop_var, 0, extent, ForKind::kSerial, stmt);
    }
    return ProloguePlan{stmt};
  }

  std::optional<ReplacementPlan> FinalizeReplacementPlan(Buffer new_buffer, IndexMap index_map,
                                                         IndexMap inverse,
                                                         PrimExpr padding_predicate,
                                                         Optional<IndexMap> pad_value) const {
    if (write_info_.empty() || is_zero(padding_predicate) || !pad_value.defined()) {
      return std::nullopt;
    }

    Map<Block, Block> new_block_to_old;
    auto generate_if_then_else_block = [&](const WriteInfo& info) -> Optional<Stmt> {
      if (!info.contains_row_major_traversal || !pad_value.defined() ||
          is_zero(padding_predicate)) {
        return NullOpt;
      }

      BufferStoreReplacer replacer(info, new_buffer, padding_predicate, inverse, pad_value,
                                   &new_block_to_old);
      Stmt stmt = replacer(info.dependent_loopnest.back()->body);
      if (!replacer.is_all_stores_replaced()) {
        return NullOpt;
      }

      ICHECK_EQ(inverse->initial_indices.size(), new_buffer->shape.size());
      for (size_t rev_i = 0; rev_i < inverse->initial_indices.size(); rev_i++) {
        size_t i = (inverse->initial_indices.size() - 1) - rev_i;
        Var loop_var = inverse->initial_indices[i];
        PrimExpr extent = new_buffer->shape[i];
        stmt = For(loop_var, 0, extent, ForKind::kSerial, stmt);
      }

      return stmt;
    };

    Map<For, Stmt> loop_replacements;

    for (const auto& info : write_info_) {
      if (info.dependent_loopnest.size()) {
        if (auto opt_stmt = generate_if_then_else_block(info)) {
          loop_replacements.Set(info.dependent_loopnest[0], opt_stmt.value());
        }
      }
    }

    if (loop_replacements.size()) {
      return ReplacementPlan{std::move(loop_replacements), std::move(new_block_to_old)};
    } else {
      return std::nullopt;
    }
  }

  std::optional<EpiloguePlan> FinalizeEpiloguePlan(Buffer new_buffer, IndexMap index_map,
                                                   IndexMap inverse, PrimExpr padding_predicate,
                                                   Optional<IndexMap> pad_value) const {
    if (write_info_.empty() || is_zero(padding_predicate) || !pad_value.defined()) {
      return std::nullopt;
    }

    Array<IterVar> iter_vars;
    Array<PrimExpr> iter_values;
    Array<PrimExpr> indices;
    ICHECK_EQ(inverse->initial_indices.size(), new_buffer->shape.size());
    for (size_t i = 0; i < inverse->initial_indices.size(); i++) {
      const auto& loop_var = inverse->initial_indices[i];
      const auto& dim = new_buffer->shape[i];
      Var block_var("v_" + loop_var->name_hint, loop_var->dtype);
      IterVar iter_var(Range(0, dim), block_var, kDataPar);
      indices.push_back(iter_var->var);
      iter_vars.push_back(iter_var);
      iter_values.push_back(loop_var);
    }

    PrimExpr pad_value_at_index = pad_value.value()->MapIndices(indices)[0];
    Stmt stmt = BufferStore(new_buffer, pad_value_at_index, indices);

    std::stringstream block_name;
    block_name << "buffer_" << new_buffer->name << "_padding";
    auto write_region = BufferRegion::FromPoint(new_buffer, indices);
    stmt = BlockRealize(iter_values, padding_predicate,
                        Block(iter_vars, {}, {write_region}, block_name.str(), stmt));

    ICHECK_EQ(inverse->initial_indices.size(), new_buffer->shape.size());
    for (size_t rev_i = 0; rev_i < inverse->initial_indices.size(); rev_i++) {
      size_t i = (inverse->initial_indices.size() - 1) - rev_i;
      Var loop_var = inverse->initial_indices[i];
      PrimExpr extent = new_buffer->shape[i];
      stmt = For(loop_var, 0, extent, ForKind::kSerial, stmt);
    }

    const auto& info = write_info_.back();
    Stmt insert_after = [&]() -> Stmt {
      if (info.dependent_loopnest.size()) {
        return info.dependent_loopnest.front();
      } else if (info.innermost_block_realize) {
        return info.innermost_block_realize.value();
      } else {
        LOG(FATAL) << "Write occured outside of any block/loop";
        return Stmt();
      }
    }();
    return EpiloguePlan{insert_after, stmt};
  }

  struct BindLoopVar {
    BindLoopVar(TransformLayoutPlanner* self, For for_node)
        : self_(self), var_(for_node->loop_var) {
      size_t loop_depth = self_->active_loops_.size();
      self_->loop_depth_lookup_[var_.get()] = {loop_depth, loop_depth};
      self_->active_loops_.push_back(std::move(for_node));
    }
    ~BindLoopVar() {
      self_->active_loops_.pop_back();
      self_->loop_depth_lookup_.erase(var_.get());
    }
    BindLoopVar(const BindLoopVar&) = delete;
    BindLoopVar& operator=(const BindLoopVar&) = delete;
    BindLoopVar(BindLoopVar&&) = delete;
    BindLoopVar& operator=(BindLoopVar&&) = delete;

    TransformLayoutPlanner* self_{nullptr};
    Var var_;
  };

  struct BindVariableDefinition {
    BindVariableDefinition() {}
    BindVariableDefinition(TransformLayoutPlanner* self, Var var, PrimExpr value)
        : self_(self), var_(var) {
      if (auto loop_depth = self->LoopDependencyRange(value); loop_depth.has_value()) {
        self_->loop_depth_lookup_[var_.get()] = loop_depth.value();
        self_->active_var_bindings_[var_.get()] = Substitute(value, self_->active_var_bindings_);
      }
    }
    ~BindVariableDefinition() {
      if (self_) {
        self_->loop_depth_lookup_.erase(var_.get());
        self_->active_var_bindings_.erase(var_.get());
      }
    }
    BindVariableDefinition(const BindVariableDefinition&) = delete;
    BindVariableDefinition& operator=(const BindVariableDefinition&) = delete;
    BindVariableDefinition(BindVariableDefinition&& other) : BindVariableDefinition() {
      swap(other);
    }
    BindVariableDefinition& operator=(BindVariableDefinition&& other) {
      swap(other);
      return *this;
    }
    void swap(BindVariableDefinition& other) {
      std::swap(self_, other.self_);
      std::swap(var_, other.var_);
    }

    TransformLayoutPlanner* self_{nullptr};
    Var var_;
  };

  struct BindBlockRealize {
    BindBlockRealize(TransformLayoutPlanner* self, BlockRealize block_realize) : self_(self) {
      ICHECK_EQ(block_realize->iter_values.size(), block_realize->block->iter_vars.size());
      for (size_t i = 0; i < block_realize->iter_values.size(); i++) {
        bound_vars_.emplace_back(self, block_realize->block->iter_vars[i]->var,
                                 block_realize->iter_values[i]);
      }
      cache_ = std::move(block_realize);
      std::swap(self_->innermost_block_realize_, cache_);
    }
    ~BindBlockRealize() { std::swap(self_->innermost_block_realize_, cache_); }
    BindBlockRealize(const BindBlockRealize&) = delete;
    BindBlockRealize& operator=(const BindBlockRealize&) = delete;
    BindBlockRealize(BindBlockRealize&&) = delete;
    BindBlockRealize& operator=(BindBlockRealize&&) = delete;

    TransformLayoutPlanner* self_{nullptr};
    Optional<BlockRealize> cache_;
    std::vector<BindVariableDefinition> bound_vars_;
  };

  /*! \brief Collected information about each BufferStore */
  std::vector<WriteInfo> write_info_;

  /*! \brief The loop iterators surrounding the current node
   *
   * The outermost loop iterator is `active_loops_.front()`, and the
   * innermost loop iterator is `active_loops_.back()`.
   *
   * Used to fill the `WriteInfo::dependent_loopnest` field.
   */
  std::vector<For> active_loops_;

  /*! \brief Lookup for the outer/inner loops
   *
   * Used to fill the `WriteInfo::dependent_loopnest` field.
   */
  std::unordered_map<const VarNode*, std::pair<size_t, size_t>> loop_depth_lookup_;

  /*! \brief The variable mappings that are currently in-scope
   *
   * Used to determine whether the indices of a BufferStore are a
   * row-major traversal, even if they are rebound in let/block
   * mappings.
   */
  std::unordered_map<const VarNode*, PrimExpr> active_var_bindings_;

  /*! \brief The innermost BlockRealize surrounding the current node
   *
   * Used to fill the `WriteInfo::innermost_block_realize` field..
   */
  Optional<BlockRealize> innermost_block_realize_{NullOpt};

  /*! \brief The buffer to be replaced */
  Buffer old_buffer_;
};

class TransformLayoutRewriter : private arith::IRMutatorWithAnalyzer {
 public:
  /*!
   * \brief Rewrite the access to the buffer after the transformation
   * \param scope_stmt The parent statement that contains all accesses to the target buffer
   * \param old_buffer The target buffer before transformation
   * \param new_buffer The new buffer after transformation
   * \param index_map The transformation applied to the buffer
   * \return The new AST rooting at the original parent scope and the map from the old block to the
   * new block
   */
  static std::pair<Stmt, Map<Block, Block>> Rewrite(
      const Block& scope_stmt, const Buffer& old_buffer, const Buffer& new_buffer,
      const IndexMap& index_map, const IndexMap& inverse, const PrimExpr& padding_predicate,
      const Optional<IndexMap>& pad_value) {
    auto plan = TransformLayoutPlanner::Plan(scope_stmt, old_buffer, new_buffer, index_map, inverse,
                                             padding_predicate, pad_value);

    arith::Analyzer analyzer;
    TransformLayoutRewriter rewriter(old_buffer, new_buffer, index_map, plan, &analyzer);
    Block result = Downcast<Block>(rewriter(scope_stmt));
    if (auto plan_ptr = std::get_if<TransformLayoutPlanner::ProloguePlan>(&plan)) {
      auto write_ptr = result.CopyOnWrite();
      write_ptr->body = SeqStmt({plan_ptr->prologue, write_ptr->body});
    }

    Map<Block, Block> block_sref_reuse;
    for (auto [after, before] : rewriter.new_block_to_old_) {
      while (auto opt = rewriter.new_block_to_old_.Get(before)) {
        before = opt.value();
      }
      while (auto opt = block_sref_reuse.Get(after)) {
        after = opt.value();
      }

      block_sref_reuse.Set(before, after);
    }

    return {result, block_sref_reuse};
  }

 private:
  TransformLayoutRewriter(const Buffer& old_buffer, const Buffer& new_buffer,
                          const IndexMap& index_map,
                          const TransformLayoutPlanner::TransformPlan& plan,
                          arith::Analyzer* analyzer)
      : IRMutatorWithAnalyzer(analyzer),
        old_buffer_(old_buffer),
        new_buffer_(new_buffer),
        index_map_(index_map),
        plan_(plan),
        buffer_data_to_buffer_{{new_buffer->data, new_buffer}} {
    if (auto plan_ptr = std::get_if<TransformLayoutPlanner::ReplacementPlan>(&plan_)) {
      new_block_to_old_ = plan_ptr->new_block_to_old;
    }
  }

  void RewriteBufferAccess(Buffer* buffer, Array<PrimExpr>* indices) {
    *buffer = new_buffer_;
    *indices = index_map_->MapIndices(*indices);
    (*indices).MutateByApply(
        [&](const PrimExpr& e) { return SimplifyNonTrivialExpr(e, analyzer_); });
  }

  using Parent = arith::IRMutatorWithAnalyzer;
  using Parent::VisitExpr_;
  using Parent::VisitStmt_;

  Stmt VisitStmt(const Stmt& stmt) final {
    Stmt output = Parent::VisitStmt(stmt);
    if (auto plan_ptr = std::get_if<TransformLayoutPlanner::EpiloguePlan>(&plan_)) {
      if (plan_ptr->insert_after.same_as(stmt)) {
        return SeqStmt({output, plan_ptr->new_block});
      }
    }
    return output;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    // Some replacements may include the original string, such as
    // replacing `loop` with `{loop, post_proc}`.  In this case, avoid
    // infinite recursion.

    For node = GetRef<For>(op);
    if (auto plan_ptr = std::get_if<TransformLayoutPlanner::ReplacementPlan>(&plan_)) {
      auto it = plan_ptr->replacements.find(node);
      if (it != plan_ptr->replacements.end()) {
        return VisitStmt((*it).second);
      }
    }
    return Parent::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    BufferLoad buffer_load = Downcast<BufferLoad>(Parent::VisitExpr_(op));
    if (buffer_load->buffer.same_as(old_buffer_)) {
      auto* n = buffer_load.CopyOnWrite();
      RewriteBufferAccess(&n->buffer, &n->indices);
    }
    return std::move(buffer_load);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore buffer_store = Downcast<BufferStore>(Parent::VisitStmt_(op));
    if (buffer_store->buffer.same_as(old_buffer_)) {
      auto* n = buffer_store.CopyOnWrite();
      RewriteBufferAccess(&n->buffer, &n->indices);
    }
    return std::move(buffer_store);
  }

  void RewriteAccessRegion(Array<BufferRegion>* old_access_regions,
                           const Array<BufferRegion>& infered_access_regions) {
    auto fmutate = [this, &infered_access_regions](const BufferRegion& buffer_region) {
      if (buffer_region->buffer.same_as(old_buffer_)) {
        ICHECK(infered_access_regions.size() == 1);
        return infered_access_regions[0];
      }
      return buffer_region;
    };
    (*old_access_regions).MutateByApply(fmutate);
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    Block orig = [&]() {
      Block block = GetRef<Block>(op);
      while (true) {
        if (auto it = new_block_to_old_.find(block); it != new_block_to_old_.end()) {
          block = (*it).second;
        } else {
          break;
        }
      }
      return block;
    }();

    Block block = Downcast<Block>(Parent::VisitStmt_(op));

    auto infered_access_regions = GetBlockReadWriteRegion(block, buffer_data_to_buffer_);
    auto* n = block.CopyOnWrite();
    RewriteAccessRegion(&n->reads, infered_access_regions[0]);
    RewriteAccessRegion(&n->writes, infered_access_regions[1]);
    n->alloc_buffers.MutateByApply([this](const Buffer& buffer) {
      if (buffer.same_as(old_buffer_)) {
        return new_buffer_;
      } else {
        return buffer;
      }
    });

    RecordReplacement(orig, block);
    return std::move(block);
  }

  void RecordReplacement(Block before, Block after) {
    if (before.same_as(after)) {
      return;
    }

    ICHECK(!new_block_to_old_.count(after));

    while (true) {
      if (auto opt = new_block_to_old_.Get(before)) {
        before = opt.value();
      } else {
        break;
      }
    }

    new_block_to_old_.Set(after, before);
  }

  const Buffer& old_buffer_;
  const Buffer& new_buffer_;
  const IndexMap& index_map_;
  const TransformLayoutPlanner::TransformPlan& plan_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  Map<Block, Block> new_block_to_old_;
};

class BufferIsSubregionError : public ScheduleError {
 public:
  explicit BufferIsSubregionError(IRModule mod, Buffer buffer) : mod_(mod), buffer_(buffer) {}

  String FastErrorString() const final {
    return "ScheduleError: The input buffer is defined in `match_buffer` of a block, it is expected"
           " to be a function parameter or allocated by a block";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "ScheduleError: The input buffer " << buffer_->name << " is defined in `match_buffer` of "
       << "a block, it is expected to be a function parameter or allocated by a block.";
    return os.str();
  }

  Array<ObjectRef> LocationsOfInterest() const final { return {}; }
  IRModule mod() const final { return mod_; }

 private:
  IRModule mod_;
  Buffer buffer_;
};

class TransformationPaddingIndexMapError : public ScheduleError {
 public:
  TransformationPaddingIndexMapError(IRModule mod, IndexMap pad_value)
      : mod_(mod), pad_value_(pad_value) {}

  String FastErrorString() const final {
    std::ostringstream ss;
    ss << "ScheduleError: The IndexMap specifying pad_value has "
       << pad_value_->final_indices.size() << " outputs, should only have one output";
    return ss.str();
  }

  String DetailRenderTemplate() const final {
    std::ostringstream ss;
    ss << "ScheduleError: Pad value is specified as " << pad_value_ << " which has "
       << pad_value_->final_indices.size() << " outputs, but should only have one output";
    return ss.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {}; }

 private:
  IRModule mod_;
  IndexMap pad_value_;
};

class TransformationPaddingTypeError : public ScheduleError {
 public:
  TransformationPaddingTypeError(IRModule mod, Buffer buffer, IndexMap pad_value)
      : mod_(mod), buffer_(buffer), pad_value_(pad_value) {
    ICHECK_EQ(pad_value_->final_indices.size(), 1);
    pad_value_dtype_ = pad_value_->final_indices[0].dtype();
  }

  String FastErrorString() const final {
    std::ostringstream ss;
    ss << "ScheduleError: Type mismatch " << buffer_->dtype << " vs " << pad_value_dtype_;
    return ss.str();
  }

  String DetailRenderTemplate() const final {
    std::ostringstream ss;
    ss << "ScheduleError: Buffer " << buffer_->name << " has elements of type " << buffer_->dtype
       << ", but the transformation fills padding with " << pad_value_ << ", which is of type "
       << pad_value_dtype_;
    return ss.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {}; }

 private:
  IRModule mod_;
  Buffer buffer_;
  IndexMap pad_value_;
  DataType pad_value_dtype_;
};

class TransformationPaddingExpressionError : public ScheduleError {
 public:
  static void Check(IRModule mod, Buffer buffer, IndexMap pad_value) {
    Visitor visitor(buffer);
    ICHECK_EQ(pad_value->final_indices.size(), 1)
        << "Internal error: Should be caught by ScheduleError checks prior to this point";
    visitor(pad_value->final_indices[0]);
    if (visitor.illegal_load) {
      throw TransformationPaddingExpressionError(mod, buffer, pad_value,
                                                 visitor.illegal_load.value());
    }
  }

 private:
  struct Visitor : ExprVisitor {
    explicit Visitor(const Buffer& buffer) : buffer_(buffer) {}

    void VisitExpr_(const BufferLoadNode* op) final {
      if (!op->buffer.same_as(buffer_)) {
        illegal_load = GetRef<BufferLoad>(op);
      }
      ExprVisitor::VisitExpr_(op);
    }

    const Buffer& buffer_;
    Optional<BufferLoad> illegal_load;
  };

  TransformationPaddingExpressionError(IRModule mod, Buffer buffer, IndexMap pad_value,
                                       BufferLoad illegal_load)
      : mod_(mod), buffer_(buffer), pad_value_(pad_value), illegal_load_(illegal_load) {}

  String FastErrorString() const final {
    std::ostringstream ss;
    ss << "ScheduleError: Pad value may not contain load load from " << illegal_load_->buffer->name;
    return ss.str();
  }

  String DetailRenderTemplate() const final {
    std::ostringstream ss;
    ss << "ScheduleError: Pad value may only contain BufferLoad from the transformed buffer "
       << buffer_->name << ", but pad_value " << pad_value_ << " contains expression "
       << illegal_load_;
    return ss.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {}; }

  IRModule mod_;
  Buffer buffer_;
  IndexMap pad_value_;
  BufferLoad illegal_load_;
};

class TransformationIntroducesPaddingError : public ScheduleError {
 public:
  TransformationIntroducesPaddingError(IRModule mod, Buffer buffer, IndexMap index_map,
                                       PrimExpr padding_predicate)
      : mod_(std::move(mod)),
        buffer_(std::move(buffer)),
        index_map_(std::move(index_map)),
        padding_predicate_(std::move(padding_predicate)) {}

  String FastErrorString() const final {
    std::ostringstream ss;
    ss << "ScheduleError: Transformation would introduce padding at " << padding_predicate_ << ".";
    return ss.str();
  }

  String DetailRenderTemplate() const final {
    auto new_shape = index_map_->MapShape(buffer_->shape);
    std::ostringstream os;
    os << "The transformation " << index_map_ << " applied on buffer " << buffer_->name
       << " of shape " << buffer_->shape << " would result in shape " << new_shape
       << ".  However, this would introduce padding wherever " << padding_predicate_ << " is true.";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {}; }

 private:
  IRModule mod_;
  Buffer buffer_;
  IndexMap index_map_;
  PrimExpr padding_predicate_;
};

void TransformLayout(ScheduleState self, const StmtSRef& block_sref, int buffer_index,
                     BufferIndexType buffer_index_type, const IndexMap& index_map,
                     const Optional<IndexMap>& pad_value) {
  // Step 1: Input handling and error checking
  const BlockNode* block_ptr = TVM_SREF_TO_BLOCK(block_sref);
  Buffer old_buffer =
      GetNthAccessBuffer(self, GetRef<Block>(block_ptr), buffer_index, buffer_index_type);
  auto [defining_site_sref, is_alloc] = GetBufferDefiningSite(block_sref, old_buffer);
  if (defining_site_sref.defined() && !is_alloc) {
    throw BufferIsSubregionError(self->mod, old_buffer);
  }
  if (pad_value) {
    if (pad_value.value()->final_indices.size() != 1) {
      throw TransformationPaddingIndexMapError(self->mod, pad_value.value());
    }
    if (pad_value.value()->final_indices[0]->dtype != old_buffer->dtype) {
      throw TransformationPaddingTypeError(self->mod, old_buffer, pad_value.value());
    }

    TransformationPaddingExpressionError::Check(self->mod, old_buffer, pad_value.value());
  }

  StmtSRef scope_sref = defining_site_sref.defined()
                            ? defining_site_sref.value()
                            : GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false);
  const BlockNode* scope_block = TVM_SREF_TO_BLOCK(scope_sref);

  auto [inverse, padding_predicate] = [&]() {
    Array<Range> region;
    for (const auto& dim : old_buffer->shape) {
      region.push_back(Range::FromMinExtent(make_zero(dim.dtype()), dim));
    }
    return index_map.NonSurjectiveInverse(region);
  }();

  bool has_padding = !is_zero(padding_predicate);
  if (has_padding && !pad_value.defined()) {
    throw TransformationIntroducesPaddingError(self->mod, old_buffer, index_map, padding_predicate);
  }

  // Step 2: Infer the shape of the new buffer
  Buffer new_buffer = old_buffer;
  new_buffer.CopyOnWrite()->shape = index_map->MapShape(old_buffer->shape);

  // Step 3: Rewrite BufferLoad/BufferStore access indices, block read/write regions, and block
  // alloc_buffers.
  auto [new_stmt, block_sref_reuse] =
      TransformLayoutRewriter::Rewrite(GetRef<Block>(scope_block), old_buffer, new_buffer,
                                       index_map, inverse, padding_predicate, pad_value);
  Block new_scope_block = Downcast<Block>(new_stmt);

  // Step 4: Rewrite buffer_map of the PrimFunc if necessary.
  if (!defining_site_sref.defined()) {
    GlobalVar g_var;
    GetRootPrimFunc(self->mod, scope_block, &g_var);
    IRModuleNode* new_mod = self->mod.CopyOnWrite();
    MapNode* new_map = new_mod->functions.CopyOnWrite();
    PrimFunc ref_new_func = Downcast<PrimFunc>(std::move(new_map->at(g_var)));
    PrimFuncNode* new_func = ref_new_func.CopyOnWrite();
    MapNode* new_buffer_map = new_func->buffer_map.CopyOnWrite();
    for (auto it = new_buffer_map->begin(); it != new_buffer_map->end(); ++it) {
      if ((*it).second.same_as(old_buffer)) {
        (*it).second = new_buffer;
      }
    }
    new_map->at(g_var) = std::move(ref_new_func);
  }

  // Step 4: Replace the scope block with the new block
  self->Replace(scope_sref, new_scope_block, block_sref_reuse);
}

/*!
 * \brief Detect the block iter type assoicated with the expression
 *
 * This function collects block iters in the expression and check if the block iters have the same
 * iter type. The detected iter type is the iter type of the block iters in the expression
 * if they have the same iter type, otherwise the detected iter type will be kOpaque.
 *
 * \param expr The expression
 * \param block_iter_type_map The mapping from block iter to iter type
 * \return The detected block iter type
 */
IterVarType DetectNewBlockIterType(
    const PrimExpr& expr,
    const std::unordered_map<const VarNode*, IterVarType>& block_iter_type_map) {
  IterVarType result{kOpaque};
  bool found = false;
  PostOrderVisit(expr, [&](const ObjectRef& obj) {
    if (const VarNode* var = obj.as<VarNode>()) {
      auto it = block_iter_type_map.find(var);
      if (it != block_iter_type_map.end()) {
        if (!found) {
          found = true;
          result = it->second;
        } else if (result != it->second) {
          result = kOpaque;
          return false;
        }
      }
    }
    return true;
  });
  return result;
}

class NotBijectiveAffineIndexMapError : public ScheduleError {
 public:
  NotBijectiveAffineIndexMapError(IRModule mod, IndexMap index_map)
      : mod_(std::move(mod)), index_map_(std::move(index_map)) {}
  String FastErrorString() const final {
    return "ScheduleError: The index map is not bijective affine.";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The index map " << index_map_->ToPythonString() << " is not bijective affine.";
    return os.str();
  }

  IRModule mod() const final { return mod_; }

  Array<ObjectRef> LocationsOfInterest() const final { return {}; }

 private:
  IRModule mod_;
  IndexMap index_map_;
};

class IndexMapNotApplicableToBlockIterError : public ScheduleError {
 public:
  static void Check(const IRModule mod, const Block& block, const IndexMap& index_map) {
    if (index_map->initial_indices.size() != block->iter_vars.size()) {
      throw IndexMapNotApplicableToBlockIterError(mod, block, index_map);
    }
  }
  explicit IndexMapNotApplicableToBlockIterError(IRModule mod, Block block, IndexMap index_map)
      : mod_(std::move(mod)), block_(std::move(block)), index_map_(std::move(index_map)) {}

  String FastErrorString() const final {
    return "ScheduleError: The index map can't be applied to block iters because the number of "
           "parameters mismatch.";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The index map " << index_map_->ToPythonString()
       << " can't be applied to block iters of {0} because the number of parameters mismatch. "
          "Expected: "
       << index_map_->initial_indices.size() << ", actual: " << block_->iter_vars.size();
    return os.str();
  }

  IRModule mod() const final { return mod_; }

  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

 private:
  IRModule mod_;
  Block block_;
  IndexMap index_map_;
};

class OpaqueNewIterTypeError : public ScheduleError {
 public:
  explicit OpaqueNewIterTypeError(IRModule mod, Block block, PrimExpr iter_value)
      : mod_(std::move(mod)), block_(std::move(block)), iter_value_(std::move(iter_value)) {}

  String FastErrorString() const final {
    return "ScheduleError: Cannot detect the new block iter type because it contains more than one "
           "type of original iter vars.";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "Cannot detect the block iter type for new iter value " << PrettyPrint(iter_value_)
       << " in {0} because it contains more than one type of original iter vars.";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

 private:
  IRModule mod_;
  Block block_;
  PrimExpr iter_value_;
};

void TransformBlockLayout(ScheduleState self, const StmtSRef& block_sref,
                          const IndexMap& index_map) {
  const BlockNode* block_ptr = TVM_SREF_TO_BLOCK(block_sref);
  const Block& block = GetRef<Block>(block_ptr);
  arith::Analyzer analyzer;

  // Step 1: Collect outer loops and loop vars
  Array<StmtSRef> loops = GetLoops(block_sref);  // outer loops of the block
  std::unordered_set<const VarNode*> loop_vars;  // loop vars of the outer loops
  for (const StmtSRef& loop_sref : loops) {
    CheckLoopStartsWithZero(self, loop_sref, &analyzer);
    loop_vars.emplace(loop_sref->StmtAs<ForNode>()->loop_var.get());
  }

  // Step 2: Check the all outer loops have a single child and the block bindings are trivial (all
  // binding values are loop vars)
  StmtSRef scope_sref{nullptr};  // the scope statement for replacement
  if (!loops.empty()) {
    scope_sref = loops.front();
    CheckGetSingleChildBlockRealizeOnSRefTree(self, loops.front());
  } else {
    scope_sref = block_sref;
  }

  BlockRealize block_realize = GetBlockRealize(self, block_sref);
  CheckBlockHasTrivialBinding(self, block_sref);

  // Step 3: Collect information of block iter vars
  Array<PrimExpr> block_vars;      // iter_var->var of each block iter
  Map<Var, Range> block_iter_dom;  // domain of block iter
  std::unordered_map<const VarNode*, IterVarType> block_iter_type;  // iter type of block iter

  Array<PrimExpr>
      block_iter_range_array;  // array of block iter extents in the same order as block iters
  for (const auto& iter_var : block->iter_vars) {
    block_vars.push_back(iter_var->var);
    block_iter_dom.Set(iter_var->var, iter_var->dom);
    block_iter_type[iter_var->var.get()] = iter_var->iter_type;
    ICHECK(is_zero(iter_var->dom->min));
    block_iter_range_array.push_back(iter_var->dom->extent);
  }

  // Step 4: Apply the IndexMap to block iters.
  IndexMapNotApplicableToBlockIterError::Check(self->mod, block, index_map);
  Array<PrimExpr> transformed_block_iters = index_map->MapIndices(block_vars);
  Array<PrimExpr> new_block_iter_range = index_map->MapShape(block_iter_range_array);

  // Step 5: Create the new block after transformation.

  // Step 5.1: Create new block iters. After applying the IndexMap f to block iters ax_0, ..., ax_n,
  // create block iter each expression in f(ax_0, ..., ax_n).
  Array<IterVar> new_block_iters;  // new block iters
  Array<PrimExpr> new_block_vars;  // iter_var->var of new block iters
  for (size_t i = 0; i < transformed_block_iters.size(); ++i) {
    Var new_block_var{"v" + std::to_string(i), transformed_block_iters[i]->dtype};
    new_block_vars.push_back(new_block_var);
    IterVarType iter_type = DetectNewBlockIterType(transformed_block_iters[i], block_iter_type);
    if (iter_type == kOpaque) {
      throw OpaqueNewIterTypeError(self->mod, GetRef<Block>(block_ptr), transformed_block_iters[i]);
    }
    auto dtype = new_block_var.dtype();
    new_block_iters.push_back(IterVar(
        /*dom=*/Range::FromMinExtent(make_zero(dtype), new_block_iter_range[i]),
        /*var=*/std::move(new_block_var), /*iter_type=*/iter_type));
  }

  // Step 5.2: Update the block body. Use the inverse map f^{-1} to replace the original block iters
  // in the body.
  Map<Var, PrimExpr> inverse_subst_map;
  // Construct the inverse map
  {
    Array<Range> initial_ranges;
    for (const PrimExpr& extent : block_iter_range_array) {
      initial_ranges.push_back(Range::FromMinExtent(make_const(extent.dtype(), 0), extent));
    }
    IndexMap inverse_index_map{nullptr};
    try {
      inverse_index_map = index_map.Inverse(initial_ranges);
    } catch (...) {
      throw NotBijectiveAffineIndexMapError(self->mod, index_map);
    }

    Array<PrimExpr> inversed_new_block_vars = inverse_index_map->MapIndices(
        new_block_vars);  // old block vars written in terms of new block vars

    for (int i = 0, n = block_vars.size(); i < n; ++i) {
      inverse_subst_map.Set(Downcast<Var>(block_vars[i]), inversed_new_block_vars[i]);
    }
  }
  Block new_block = Downcast<Block>(Substitute(GetRef<Block>(block_ptr), inverse_subst_map));
  new_block.CopyOnWrite()->iter_vars = new_block_iters;
  new_block = Downcast<Block>(BlockBufferAccessSimplifier::Simplify(new_block, &analyzer));

  // Step 5.3: Create outer loops for each new block iter.

  // Make new loop vars
  Array<PrimExpr> new_loop_vars;
  for (int i = 0; i < static_cast<int>(new_block_iters.size()); ++i) {
    new_loop_vars.push_back(Var("ax" + std::to_string(i), new_block_iters[i]->var.dtype()));
  }

  // Make new block realize
  BlockRealizeNode* new_block_realize = block_realize.CopyOnWrite();
  new_block_realize->iter_values = new_loop_vars;
  new_block_realize->block = new_block;

  // Generate outer loops
  Stmt body = GetRef<Stmt>(new_block_realize);
  for (int i = static_cast<int>(new_loop_vars.size()) - 1; i >= 0; --i) {
    body = For(Downcast<Var>(new_loop_vars[i]), 0, new_block_iter_range[i], ForKind::kSerial,
               std::move(body));
  }

  // Step 6: Do the actual replacement
  self->Replace(scope_sref, body, {{block, new_block}});
}

class BufferAxisSeparatorMutator : private ReplaceBufferMutator {
 public:
  static Block Mutate(const Block& scope_block, const Buffer& old_buffer, Buffer new_buffer,
                      Map<Block, Block>* block_sref_reuse) {
    BufferAxisSeparatorMutator mutator(old_buffer, std::move(new_buffer), block_sref_reuse);
    return Downcast<Block>(mutator.VisitStmt(scope_block));
  }

 private:
  BufferAxisSeparatorMutator(const Buffer& old_buffer, Buffer new_buffer,
                             Map<Block, Block>* block_sref_reuse)
      : ReplaceBufferMutator(old_buffer, new_buffer, block_sref_reuse) {}

  MatchBufferRegion VisitMatchBufferRegion(const MatchBufferRegion& match_buffer) final {
    auto it = buffer_var_map_.find(match_buffer->source->buffer->data.get());
    if (it != buffer_var_map_.end()) {
      const Buffer& new_source_buffer = it->second;
      Buffer new_target_buffer = match_buffer->buffer;
      new_target_buffer.CopyOnWrite()->axis_separators = new_source_buffer->axis_separators;
      if (new_target_buffer->shape.size() != new_source_buffer->shape.size()) {
        LOG(WARNING)
            << "Target buffer in match_buffer doesn't have the same dimensionality as its source "
               "buffer. `axis_separators` for the target buffer might be incorrect.";
      }
      buffer_var_map_[new_target_buffer->data.get()] = new_target_buffer;
      return MatchBufferRegion(new_target_buffer,
                               BufferRegion(new_source_buffer, match_buffer->source->region));
    }
    return match_buffer;
  }
};

void SetAxisSeparator(ScheduleState self, const StmtSRef& block_sref, int buffer_index,
                      BufferIndexType buffer_index_type, const Array<IntImm>& axis_separators) {
  const BlockNode* block_ptr = TVM_SREF_TO_BLOCK(block_sref);
  Buffer old_buffer =
      GetNthAccessBuffer(self, GetRef<Block>(block_ptr), buffer_index, buffer_index_type);
  auto [defining_site_sref, is_alloc] = GetBufferDefiningSite(block_sref, old_buffer);
  if (defining_site_sref.defined() && !is_alloc) {
    throw BufferIsSubregionError(self->mod, old_buffer);
  }

  StmtSRef scope_sref = defining_site_sref.defined()
                            ? defining_site_sref.value()
                            : GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false);
  const BlockNode* scope_block = TVM_SREF_TO_BLOCK(scope_sref);

  // Step 1: Check and update axis_separators of the buffer.
  Buffer new_buffer = old_buffer;
  new_buffer.CopyOnWrite()->axis_separators = axis_separators;

  Map<Block, Block> block_sref_reuse;

  // Step 2: Rewrite alloc_buffer of the block or buffer_map of the PrimFunc.
  Block new_scope_block = BufferAxisSeparatorMutator::Mutate(GetRef<Block>(scope_block), old_buffer,
                                                             new_buffer, &block_sref_reuse);
  if (!defining_site_sref.defined()) {
    // mutate buffer_map of the PrimFunc
    GlobalVar g_var;
    GetRootPrimFunc(self->mod, scope_block, &g_var);
    IRModuleNode* new_mod = self->mod.CopyOnWrite();
    MapNode* new_map = new_mod->functions.CopyOnWrite();
    PrimFunc ref_new_func = Downcast<PrimFunc>(std::move(new_map->at(g_var)));
    PrimFuncNode* new_func = ref_new_func.CopyOnWrite();
    MapNode* new_buffer_map = new_func->buffer_map.CopyOnWrite();
    for (auto it = new_buffer_map->begin(); it != new_buffer_map->end(); ++it) {
      if ((*it).second.same_as(old_buffer)) {
        (*it).second = new_buffer;
      }
    }
    new_map->at(g_var) = std::move(ref_new_func);
  }

  // Step 4: Replace the scope block with the new block
  self->Replace(scope_sref, new_scope_block, block_sref_reuse);
}

/******** InstructionKind Registration ********/

struct TransformLayoutTraits : public UnpackedInstTraits<TransformLayoutTraits> {
  static constexpr const char* kName = "TransformLayout";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 4;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv, Integer buffer_index,
                                      Integer buffer_index_type, IndexMap index_map,
                                      Optional<IndexMap> pad_value) {
    return sch->TransformLayout(block_rv, buffer_index.IntValue(),
                                static_cast<BufferIndexType>(buffer_index_type->value), index_map,
                                pad_value);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv, Integer buffer_index,
                                 Integer buffer_index_type, IndexMap index_map,
                                 Optional<IndexMap> pad_value) {
    PythonAPICall py("transform_layout");
    py.Input("block", block_rv);

    std::ostringstream os;
    os << "(\"" << BufferIndexType2Str(static_cast<BufferIndexType>(buffer_index_type->value))
       << "\", " << buffer_index << ")";
    py.Input("buffer", os.str());

    py.Input("index_map", index_map->ToPythonString());
    py.Input("pad_value", pad_value ? pad_value.value()->ToPythonString() : "None");

    return py.Str();
  }

 public:
  static ObjectRef AttrsAsJSON(const Array<ObjectRef>& attrs) {
    Array<ObjectRef> attrs_record;
    attrs_record.reserve(kNumAttrs);
    attrs_record.push_back(attrs[0]);
    attrs_record.push_back(attrs[1]);
    attrs_record.push_back(String(::tvm::SaveJSON(attrs[2])));
    attrs_record.push_back(attrs[3]);
    return std::move(attrs_record);
  }

  static Array<ObjectRef> AttrsFromJSON(const ObjectRef& attrs_record_) {
    Array<ObjectRef> attrs_record = Downcast<Array<ObjectRef>>(attrs_record_);
    Array<ObjectRef> attrs;
    attrs.push_back(attrs_record[0]);
    attrs.push_back(attrs_record[1]);
    attrs.push_back(::tvm::LoadJSON(Downcast<String>(attrs_record[2])));
    attrs.push_back(attrs_record[3]);
    return attrs;
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct TransformBlockLayoutTraits : public UnpackedInstTraits<TransformBlockLayoutTraits> {
  static constexpr const char* kName = "TransformBlockLayout";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv, IndexMap index_map) {
    return sch->TransformBlockLayout(block_rv, index_map);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv, IndexMap index_map) {
    PythonAPICall py("transform_block_layout");
    py.Input("block", block_rv);
    py.Input("index_map", index_map->ToPythonString());
    return py.Str();
  }

 public:
  static ObjectRef AttrsAsJSON(const Array<ObjectRef>& attrs) {
    Array<ObjectRef> attrs_record;
    attrs_record.reserve(kNumAttrs);
    attrs_record.push_back(String(::tvm::SaveJSON(attrs[0])));
    return std::move(attrs_record);
  }

  static Array<ObjectRef> AttrsFromJSON(const ObjectRef& attrs_record_) {
    Array<ObjectRef> attrs_record = Downcast<Array<ObjectRef>>(attrs_record_);
    Array<ObjectRef> attrs;
    attrs.push_back(::tvm::LoadJSON(Downcast<String>(attrs_record[0])));
    return attrs;
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct SetAxisSeparatorTraits : public UnpackedInstTraits<SetAxisSeparatorTraits> {
  static constexpr const char* kName = "SetAxisSeparator";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 3;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv, Integer buffer_index,
                                      Integer buffer_index_type, Array<IntImm> axis_separators) {
    return sch->SetAxisSeparator(block_rv, buffer_index.IntValue(),
                                 static_cast<BufferIndexType>(buffer_index_type->value),
                                 axis_separators);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv, Integer buffer_index,
                                 Integer buffer_index_type, Array<IntImm> axis_separators) {
    PythonAPICall py("set_axis_separator");
    py.Input("block", block_rv);

    std::ostringstream os;
    os << "(\"" << BufferIndexType2Str(static_cast<BufferIndexType>(buffer_index_type->value))
       << "\", " << buffer_index << ")";
    py.Input("buffer", os.str());

    py.Input("axis_separators", axis_separators);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(TransformLayoutTraits);
TVM_REGISTER_INST_KIND_TRAITS(TransformBlockLayoutTraits);
TVM_REGISTER_INST_KIND_TRAITS(SetAxisSeparatorTraits);

}  // namespace tir
}  // namespace tvm
