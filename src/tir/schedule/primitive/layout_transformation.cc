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

class LayoutTransformPlanner : private StmtExprVisitor {
 public:
  // Statement to be inserted prior to the analyzed block
  struct ProloguePlan {
    Stmt prologue;
  };


  // The block to be inserted, along with the location at which it
  // should be inserted.  The location will be either a For or a
  // Block, and will be after all writes the transformed buffer.
  struct EpiloguePlan {
    Stmt insert_after;
    Stmt new_block;
  };

  struct NoPaddingRequired {};

  using TransformPlan = std::variant<ProloguePlan, EpiloguePlan, NoPaddingRequired>;
  static TransformPlan Plan(Block block, Buffer old_buffer, Buffer new_buffer, IndexMap index_map,
                            IndexMap inverse, PrimExpr padding_predicate,
                            Optional<PrimExpr> pad_value) {
    LayoutTransformPlanner visitor(old_buffer);
    visitor(block);
    return visitor.Finalize(new_buffer, index_map, inverse, padding_predicate, pad_value);
  }

 private:
  LayoutTransformPlanner(Buffer old_buffer) : old_buffer_(old_buffer) {}

  void VisitStmt_(const ForNode* op) override {
    BindLoopVar context(this, GetRef<For>(op));
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
  TransformPlan Finalize(Buffer new_buffer, IndexMap index_map, IndexMap inverse,
                         PrimExpr padding_predicate, Optional<PrimExpr> pad_value) const {
    if (auto prologue_plan =
            FinalizeProloguePlan(new_buffer, index_map, inverse, padding_predicate, pad_value);
        prologue_plan.has_value()) {
      return prologue_plan.value();
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
                                                   Optional<PrimExpr> pad_value) const {
    if (write_info_.size() || is_zero(padding_predicate) || !pad_value.defined()) {
      return std::nullopt;
    }

    Array<PrimExpr> indices;
    for (const auto& var : inverse->initial_indices) {
      indices.push_back(var);
    }

    PrimExpr expr = (!padding_predicate) || (BufferLoad(new_buffer, indices) == pad_value.value());
    Stmt stmt = Evaluate(Call(DataType::Bool(), builtin::assume(), {expr}));

    std::stringstream block_name;
    block_name << "buffer_" << new_buffer->name << "_assumptions";
    auto read_region = BufferRegion::FromPoint(new_buffer, indices);
    stmt = BlockRealize({}, Bool(true), Block({}, {read_region}, {}, block_name.str(), stmt));

    for (size_t rev_i = 0; rev_i < inverse->initial_indices.size(); rev_i++) {
      size_t i = (inverse->initial_indices.size() - 1) - rev_i;
      Var loop_var = inverse->initial_indices[i];
      PrimExpr extent = new_buffer->shape[i];
      stmt = For(loop_var, 0, extent, ForKind::kSerial, stmt);
    }
    return ProloguePlan{stmt};
  }

  std::optional<EpiloguePlan> FinalizeEpiloguePlan(Buffer new_buffer, IndexMap index_map,
                                                   IndexMap inverse, PrimExpr padding_predicate,
                                                   Optional<PrimExpr> pad_value) const {
    if (write_info_.empty() || is_zero(padding_predicate) || !pad_value.defined()) {
      return std::nullopt;
    }

    Array<PrimExpr> indices;
    for (const auto& var : inverse->initial_indices) {
      indices.push_back(var);
    }

    Stmt stmt = BufferStore(new_buffer, pad_value.value(), indices);

    std::stringstream block_name;
    block_name << "buffer_" << new_buffer->name << "_padding";
    auto write_region = BufferRegion::FromPoint(new_buffer, indices);
    stmt =
        BlockRealize({}, padding_predicate, Block({}, {}, {write_region}, block_name.str(), stmt));

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
    BindLoopVar(LayoutTransformPlanner* self, For for_node)
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

    LayoutTransformPlanner* self_{nullptr};
    Var var_;
  };

  struct BindBlockRealize {
    BindBlockRealize(LayoutTransformPlanner* self, BlockRealize block_realize) : self_(self) {
      cache_ = std::move(block_realize);
      std::swap(self_->innermost_block_realize_, cache_);
    }
    ~BindBlockRealize() { std::swap(self_->innermost_block_realize_, cache_); }
    BindBlockRealize(const BindBlockRealize&) = delete;
    BindBlockRealize& operator=(const BindBlockRealize&) = delete;
    BindBlockRealize(BindBlockRealize&&) = delete;
    BindBlockRealize& operator=(BindBlockRealize&&) = delete;

    LayoutTransformPlanner* self_{nullptr};
    Optional<BlockRealize> cache_;
  };

  struct WriteInfo {
    // The BufferStore object
    BufferStore store;

    // The block realize that contains the store, if any.
    Optional<BlockRealize> innermost_block_realize;

    // The nested loops whose values contribute to the indices used in
    // the store.  Not all loop variables in the loopnest need to
    // contribute, but the first and last must.
    std::vector<For> dependent_loopnest;
  };

  std::vector<WriteInfo> write_info_;
  std::vector<For> active_loops_;
  std::unordered_map<const VarNode*, std::pair<size_t, size_t>> loop_depth_lookup_;
  Optional<BlockRealize> innermost_block_realize_{NullOpt};

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
      const Optional<PrimExpr>& pad_value) {
    auto plan = LayoutTransformPlanner::Plan(scope_stmt, old_buffer, new_buffer, index_map, inverse,
                                             padding_predicate, pad_value);

    arith::Analyzer analyzer;
    TransformLayoutRewriter rewriter(old_buffer, new_buffer, index_map, plan, &analyzer);
    Block result = Downcast<Block>(rewriter(scope_stmt));
    if (auto plan_ptr = std::get_if<LayoutTransformPlanner::ProloguePlan>(&plan)) {
      auto write_ptr = result.CopyOnWrite();
      write_ptr->body = SeqStmt({plan_ptr->prologue, write_ptr->body});
    }
    return {result, rewriter.block_sref_reuse_};
  }

 private:
  TransformLayoutRewriter(const Buffer& old_buffer, const Buffer& new_buffer,
                          const IndexMap& index_map,
                          const LayoutTransformPlanner::TransformPlan& plan,
                          arith::Analyzer* analyzer)
      : IRMutatorWithAnalyzer(analyzer),
        old_buffer_(old_buffer),
        new_buffer_(new_buffer),
        index_map_(index_map),
        plan_(plan),
        buffer_data_to_buffer_{{new_buffer->data, new_buffer}} {}

  void RewriteBufferAccess(Buffer* buffer, Array<PrimExpr>* indices) {
    *buffer = new_buffer_;
    *indices = index_map_->MapIndices(*indices, analyzer_);
  }

  using Parent = arith::IRMutatorWithAnalyzer;
  using Parent::VisitExpr_;
  using Parent::VisitStmt_;

  Stmt VisitStmt(const Stmt& stmt) final {
    Stmt output = Parent::VisitStmt(stmt);
    if (auto plan_ptr = std::get_if<LayoutTransformPlanner::EpiloguePlan>(&plan_)) {
      if (plan_ptr->insert_after.same_as(stmt)) {
        return SeqStmt({output, plan_ptr->new_block});
      }
    }
    return output;
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
    block_sref_reuse_.Set(GetRef<Block>(op), block);
    return std::move(block);
  }

  const Buffer& old_buffer_;
  const Buffer& new_buffer_;
  const IndexMap& index_map_;
  const LayoutTransformPlanner::TransformPlan& plan_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  Map<Block, Block> block_sref_reuse_;
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

class TransformationPaddingTypeError : public ScheduleError {
 public:
  TransformationPaddingTypeError(IRModule mod, Buffer buffer, PrimExpr pad_value)
      : mod_(mod), buffer_(buffer), pad_value_(pad_value) {}

  String FastErrorString() const final {
    std::ostringstream ss;
    ss << "ScheduleError: Type mismatch " << buffer_->dtype << " vs " << pad_value_->dtype;
    return ss.str();
  }

  String DetailRenderTemplate() const final {
    std::ostringstream ss;
    ss << "ScheduleError: Buffer " << buffer_->name << " has elements of type " << buffer_->dtype
       << ", but the transformation fills padding with " << pad_value_ << ", which is of type "
       << pad_value_->dtype;
    return ss.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {}; }

 private:
  IRModule mod_;
  Buffer buffer_;
  PrimExpr pad_value_;
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
                     const Optional<PrimExpr>& pad_value) {
  // Step 1: Input handling and error checking
  const BlockNode* block_ptr = TVM_SREF_TO_BLOCK(block_sref);
  Buffer old_buffer =
      GetNthAccessBuffer(self, GetRef<Block>(block_ptr), buffer_index, buffer_index_type);
  auto [defining_site_sref, is_alloc] = GetBufferDefiningSite(block_sref, old_buffer);
  if (defining_site_sref.defined() && !is_alloc) {
    throw BufferIsSubregionError(self->mod, old_buffer);
  }
  if (pad_value && pad_value.value()->dtype != old_buffer->dtype) {
    throw TransformationPaddingTypeError(self->mod, old_buffer, pad_value.value());
  }

  StmtSRef scope_sref = defining_site_sref.defined()
                            ? defining_site_sref.value()
                            : GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false);
  const BlockNode* scope_block = TVM_SREF_TO_BLOCK(scope_sref);

  auto [inverse, padding_predicate] = [&]() {
    Array<Range> region;
    for (const auto& dim : old_buffer->shape) {
      region.push_back(Range::FromMinExtent(0, dim));
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

  Array<ObjectRef> LocationsOfInterest() const final { return {}; }

 private:
  IRModule mod_;
  Block block_;
  IndexMap index_map_;
};

class NotTrivialBindingError : public ScheduleError {
 public:
  explicit NotTrivialBindingError(IRModule mod, Block block)
      : mod_(std::move(mod)), block_(std::move(block)) {}

  static void CheckBlockHasTrivialBinding(const IRModule& mod, const BlockRealize& block_realize,
                                          std::unordered_set<const VarNode*> outer_loop_vars) {
    // Step 2: Check all the binding values are loops vars
    for (const PrimExpr& iter_value : block_realize->iter_values) {
      const VarNode* loop_var = iter_value.as<VarNode>();
      if (!loop_var || !outer_loop_vars.count(loop_var)) {
        throw NotTrivialBindingError(mod, block_realize->block);
      }
    }
  }

  String FastErrorString() const final {
    return "ScheduleError: The binding values of the block are not variables of outer loops.";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The binding values of the {0} are not variables of outer loops.";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

 private:
  IRModule mod_;
  Block block_;
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
  NotTrivialBindingError::CheckBlockHasTrivialBinding(self->mod, block_realize, loop_vars);

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

  auto iter_map = arith::DetectIterMap(
      /*indices=*/transformed_block_iters, /*input_iters=*/block_iter_dom, /*predicate=*/Bool(true),
      /*check_level=*/arith::IterMapLevel::Bijective, &analyzer,
      /*simplify_trivial_iterators=*/true);
  if (iter_map->indices.empty()) {
    throw NotBijectiveAffineIndexMapError(self->mod, index_map);
  }

  // Step 5: Create the new block after transformation.

  // Step 5.1: Create new block iters. After applying the IndexMap f to block iters ax_0, ..., ax_n,
  // create block iter each expression in f(ax_0, ..., ax_n).
  Array<IterVar> new_block_iters;  // new block iters
  Array<PrimExpr> new_block_vars;  // iter_var->var of new block iters
  for (size_t i = 0; i < index_map->final_indices.size(); ++i) {
    Var new_block_var{"v" + std::to_string(i), DataType::Int(32)};
    new_block_vars.push_back(new_block_var);
    IterVarType iter_type = DetectNewBlockIterType(transformed_block_iters[i], block_iter_type);
    if (iter_type == kOpaque) {
      throw OpaqueNewIterTypeError(self->mod, GetRef<Block>(block_ptr), transformed_block_iters[i]);
    }
    new_block_iters.push_back(IterVar(/*dom=*/Range::FromMinExtent(0, new_block_iter_range[i]),
                                      /*var=*/std::move(new_block_var), /*iter_type=*/iter_type));
  }

  // Step 5.2: Update the block body. Use the inverse map f^{-1} to replace the original block iters
  // in the body.

  auto inverse_map = arith::InverseAffineIterMap(iter_map->indices, new_block_vars);
  // Trivial block iters will be simplified in DetectIterMap, they should be mapped to constant
  // zero.
  for (const auto& iter_var : block_ptr->iter_vars) {
    if (inverse_map.find(iter_var->var) == inverse_map.end()) {
      ICHECK(is_one(iter_var->dom->extent));
      inverse_map.Set(iter_var->var, 0);
    }
  }

  Block new_block = Downcast<Block>(Substitute(GetRef<Block>(block_ptr), inverse_map));
  new_block.CopyOnWrite()->iter_vars = new_block_iters;
  new_block = Downcast<Block>(BlockBufferAccessSimplifier::Simplify(new_block, &analyzer));

  // Step 5.3: Create outer loops for each new block iter.

  // Make new loop vars
  Array<PrimExpr> new_loop_vars;
  for (int i = 0; i < static_cast<int>(new_block_iters.size()); ++i) {
    new_loop_vars.push_back(Var("ax" + std::to_string(i), DataType::Int(32)));
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
                                      Optional<PrimExpr> pad_value) {
    return sch->TransformLayout(block_rv, buffer_index.IntValue(),
                                static_cast<BufferIndexType>(buffer_index_type->value), index_map,
                                pad_value);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv, Integer buffer_index,
                                 Integer buffer_index_type, IndexMap index_map,
                                 Optional<PrimExpr> pad_value) {
    PythonAPICall py("transform_layout");
    py.Input("block", block_rv);

    std::ostringstream os;
    os << "(\"" << BufferIndexType2Str(static_cast<BufferIndexType>(buffer_index_type->value))
       << "\", " << buffer_index << ")";
    py.Input("buffer", os.str());

    py.Input("index_map", index_map->ToPythonString());
    py.Input("pad_value", pad_value);

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
