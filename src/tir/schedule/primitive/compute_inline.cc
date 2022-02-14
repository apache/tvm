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

static const char kErrBodyInline[] = R"(The body of the inlined block should be in form of
    'A[i, j, k, ...] = f(i, j, k, ...)',
where the indices on the left are distinct atomic variables,
and there should not no variables other than the index variables)";

static const char kErrBodyReverseInline[] = R"(The body of the inlined block should be in form of
    `B[...] = g(i, j, k, A[i, j, k, ...] ...)`,
where A is the only buffer the block consumes, whose indices are distinct atomic variables,
and there should not no variables other than the index variables)";

class NotSingleReadWriteBuffer : public ScheduleError {
 public:
  explicit NotSingleReadWriteBuffer(IRModule mod, bool is_read, Block block)
      : mod_(mod), is_read_(is_read), block_(std::move(block)) {}

  String FastErrorString() const final {
    return is_read_ ? "ScheduleError: The block is allowed to read only a single buffer region"
                    : "ScheduleError: The block is allowed to write only a single buffer region";
  }

  String DetailRenderTemplate() const final {
    if (is_read_) {
      int k = block_->reads.size();
      return "The block is only allowed to read a single buffer region, but it reads " +
             std::to_string(k) + " region(s): {0}";
    } else {
      int k = block_->writes.size();
      return "The block is only allowed to write a single buffer region, but it writes " +
             std::to_string(k) + " region(s): {0}";
    }
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

  IRModule mod_;
  bool is_read_;
  Block block_;

  static Buffer GetSingleRead(const ScheduleState& self, const Block& block,
                              const StmtSRef& scope_root_sref) {
    const std::unordered_map<Buffer, Array<StmtSRef>, ObjectPtrHash, ObjectPtrEqual>&
        buffer_writers = self->block_info.at(scope_root_sref).scope->buffer_writers;
    const BufferNode* read_buffer = nullptr;
    for (const BufferRegion& read_region : block->reads) {
      const BufferNode* buffer = read_region->buffer.get();
      if (buffer == read_buffer) {
        continue;
      }
      if (buffer_writers.count(GetRef<Buffer>(buffer)) > 0) {
        if (read_buffer != nullptr) {
          throw NotSingleReadWriteBuffer(self->mod, true, block);
        }
        read_buffer = buffer;
      }
    }
    if (read_buffer == nullptr) {
      throw NotSingleReadWriteBuffer(self->mod, true, block);
    }
    return GetRef<Buffer>(read_buffer);
  }

  static Buffer GetSingleWrite(const ScheduleState& self, const Block& block) {
    if (block->writes.size() != 1) {
      throw NotSingleReadWriteBuffer(self->mod, false, block);
    }
    return block->writes[0]->buffer;
  }
};

class BodyAnalysisError : public ScheduleError {
 public:
  explicit BodyAnalysisError(bool is_reverse, IRModule mod, Block block)
      : is_reverse_(is_reverse), mod_(mod), block_(std::move(block)) {}

  String FastErrorString() const final {
    return "ScheduleError: The block cannot be inlined because its body pattern does not meet the "
           "condition for inlining";
  }

  String DetailRenderTemplate() const final {
    return is_reverse_ ? kErrBodyReverseInline : kErrBodyInline;
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

  bool is_reverse_;
  IRModule mod_;
  Block block_;
};

class NonSingleProducerError : public ScheduleError {
 public:
  explicit NonSingleProducerError(IRModule mod, Block block)
      : mod_(mod), block_(std::move(block)) {}

  String FastErrorString() const final {
    return "ScheduleError: The consumer block to be inlined is required to have only a single "
           "producer block, and the producer block should be a complete block who has only a "
           "single consumer";
  }

  String DetailRenderTemplate() const final {
    return "The consumer block {0} to be inlined is required to have only a single "
           "producer block, and the producer block should be a complete block who has only a "
           "single consumer";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

  IRModule mod_;
  Block block_;

  static void Check(const ScheduleState& self, const StmtSRef& consumer_block_sref,
                    const StmtSRef& scope_root_sref) {
    BlockScope scope = self->GetBlockScope(scope_root_sref);
    Array<Dependency> producers = scope->GetDepsByDst(consumer_block_sref);
    if (producers.size() == 1 && producers[0]->kind == DepKind::kRAW) {
      const StmtSRef& producer_block_sref = producers[0]->src;
      if (IsCompleteBlock(self, producer_block_sref, scope_root_sref)) {
        Array<Dependency> consumers = scope->GetDepsBySrc(producer_block_sref);
        if (consumers.size() == 1) {
          return;
        }
      }
    }
    const BlockNode* block = TVM_SREF_TO_BLOCK(block, consumer_block_sref);
    throw NonSingleProducerError(self->mod, GetRef<Block>(block));
  }
};

class OpaqueAccessError : public ScheduleError {
 public:
  explicit OpaqueAccessError(IRModule mod, StmtSRef scope_root_sref)
      : mod_(mod), scope_root_(nullptr) {
    const BlockNode* scope_root = TVM_SREF_TO_BLOCK(scope_root, scope_root_sref);
    this->scope_root_ = GetRef<Block>(scope_root);
  }

  String FastErrorString() const final {
    return "ScheduleError: The buffer to be inlined has opaque access (e.g. `B.data`), or its "
           "subregion is matched into other blocks";
  }

  String DetailRenderTemplate() const final {
    return "The buffer to be inlined has opaque access (e.g. `B.data`), or its "
           "subregion is matched into other blocks: {0}";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {scope_root_}; }

  IRModule mod_;
  Block scope_root_;
};

/*!
 * \brief The base class of the inliner, which handles:
 * 1) Substitute a subtree with the specific block being inlined
 * 2) Update the block signature to reflect the changes of read/write/allocated buffers
 * 3) Maintain a list of index variables and their substitution of the buffer being inlined
 */
class BaseInliner : public StmtExprMutator {
 protected:
  explicit BaseInliner(const Buffer& inlined_buffer, const Block& inlined_block,
                       const StmtSRef& scope_root_sref)
      : inlined_buffer_(inlined_buffer),
        inlined_store_(inlined_block->body.as<BufferStoreNode>()),
        scope_root_sref_(scope_root_sref) {
    AddBuffersInBlockSignature(inlined_block.get());
  }

  PrimExpr VisitExpr_(const VarNode* var) final {
    CheckOpaqueAccess(var);
    return StmtExprMutator::VisitExpr_(var);
  }

  PrimExpr VisitExpr_(const LoadNode* load) final {
    CheckOpaqueAccess(load->buffer_var.get());
    return StmtExprMutator::VisitExpr_(load);
  }

  Stmt VisitStmt_(const StoreNode* store) final {
    CheckOpaqueAccess(store->buffer_var.get());
    return StmtExprMutator::VisitStmt_(store);
  }

  Stmt VisitStmt_(const ForNode* loop) final {
    if (src_stmt.get() == loop) {
      loop = tgt_stmt.as<ForNode>();
      ICHECK(loop != nullptr);
    }
    return StmtExprMutator::VisitStmt_(loop);
  }

  Stmt VisitStmt_(const BlockNode* block) final {
    CheckMatchBufferRegion(block);
    AddBuffersInBlockSignature(block);
    Block src_block = GetRef<Block>(block);
    if (src_block.same_as(src_stmt)) {
      block = tgt_stmt.as<BlockNode>();
      ICHECK(block != nullptr);
    }
    Block tgt_block = Downcast<Block>(StmtExprMutator::VisitStmt_(block));
    bool is_scope_root = src_block.get() == scope_root_sref_->stmt;
    tgt_block = UpdateBuffersInBlockSignature(std::move(tgt_block), is_scope_root);
    block_reuse.Set(src_block, tgt_block);
    return std::move(tgt_block);
  }

  /*!
   * \brief Check if the indices are atomic distinct variables and the access is n-dimensional.
   * If so, set `self->idx_vars_` properly.
   * \param indices The indices to be extracted
   * \param expected_ndim The expected ndim of the access
   * \return A boolean flag indicating if the check is successful
   */
  bool UpdateAndCheckIndexVars(const Array<PrimExpr>& indices, int expected_ndim) {
    int n = indices.size();
    if (n != expected_ndim) {
      // Failure: dimension mismatch
      return false;
    }
    std::vector<const VarNode*> result;
    result.reserve(n);
    for (const PrimExpr& i : indices) {
      if (const auto* var = i.as<VarNode>()) {
        result.push_back(var);
      } else {
        // Failure: indexing expression is not a variable
        return false;
      }
    }
    using DistinctSet = std::unordered_set<const VarNode*>;
    int n_distinct = DistinctSet(result.begin(), result.end()).size();
    if (n != n_distinct) {
      // Failure: indexing variables are not distinct
      return false;
    }
    if (idx_vars_.empty()) {
      idx_vars_ = std::move(result);
    } else if (!support::ArrayWithSameContent(idx_vars_, result)) {
      // Failure: indexing variables are not consitent in different BufferLoads
      return false;
    }
    return true;
  }

  /*!
   * \brief Set the mapping of index substitution `self->idx_sub_`
   * \param indices The expressions that the corresponding index variables are replaced to
   */
  void SetIndexSubstitution(const Array<PrimExpr>& indices) {
    ICHECK_EQ(indices.size(), idx_vars_.size());
    int n = idx_vars_.size();
    idx_sub_.reserve(n);
    for (int i = 0; i < n; ++i) {
      idx_sub_[idx_vars_[i]] = indices[i];
    }
  }

 private:
  /*!
   * \brief Add the buffers in the block signature to the `buffer_var_map_`,
   * which is used for auto-completion of a block's read/write region
   * \param block The block whose signature to be added
   */
  void AddBuffersInBlockSignature(const BlockNode* block) {
    for (const BufferRegion& buffer_region : block->reads) {
      const Buffer& buffer = buffer_region->buffer;
      buffer_var_map_.Set(buffer->data, buffer);
    }
    for (const BufferRegion& buffer_region : block->writes) {
      const Buffer& buffer = buffer_region->buffer;
      buffer_var_map_.Set(buffer->data, buffer);
    }
    for (const Buffer& buffer : block->alloc_buffers) {
      buffer_var_map_.Set(buffer->data, buffer);
    }
  }

  /*!
   * \brief Update the following block signature:
   * 1) T.alloc_buffer, if the block is scope root
   * 2) T.reads, if the block is not scope root
   * 3) T.writes, if the block is not scope root
   * \param block The block to be updated
   * \param is_scope_root A flag indicating if a block is the scope root of the block to be inlined
   * \return The updated block
   */
  Block UpdateBuffersInBlockSignature(Block block, bool is_scope_root) {
    // Step 1. Update `BlockNode::alloc_buffers`
    Array<Buffer> alloc_buffers;
    if (is_scope_root) {
      alloc_buffers.reserve(block->alloc_buffers.size());
      for (const Buffer& alloc_buffer : block->alloc_buffers) {
        if (!alloc_buffer.same_as(inlined_buffer_)) {
          alloc_buffers.push_back(alloc_buffer);
        }
      }
    } else {
      alloc_buffers = std::move(block->alloc_buffers);
    }
    // Step 2. Update `BlockNode::reads` and `BlockNode::writes`
    Array<BufferRegion> reads = std::move(block->reads);
    Array<BufferRegion> writes = std::move(block->writes);
    auto f_access_inline_buffer = [this](const BufferRegion& access) {
      return access->buffer.same_as(this->inlined_buffer_);
    };
    if (!is_scope_root && (std::any_of(reads.begin(), reads.end(), f_access_inline_buffer) ||
                           std::any_of(writes.begin(), writes.end(), f_access_inline_buffer))) {
      Array<Array<BufferRegion>> inspected = GetBlockReadWriteRegion(block, buffer_var_map_);
      reads = std::move(inspected[0]);
      writes = std::move(inspected[1]);
    }
    // Step 3. Assemble the result
    BlockNode* n = block.CopyOnWrite();
    n->reads = std::move(reads);
    n->writes = std::move(writes);
    n->alloc_buffers = std::move(alloc_buffers);
    return block;
  }

  /*!
   * \brief Opaque access to the buffer to be inlined is disallowed.
   * This method checks if a buffer var belongs to the buffer
   * \param buffer_var The buffer var to be checked
   */
  void CheckOpaqueAccess(const VarNode* buffer_var) {
    if (inlined_buffer_->data.get() == buffer_var) {
      this->has_opaque_access = true;
    }
  }

  /*!
   * \brief The buffer to be inlined is not allowed to be region matched.
   * This method checks if a block has the disallowed behavior of buffer region match.
   * \param block The block to be checked
   */
  void CheckMatchBufferRegion(const BlockNode* block) {
    for (const MatchBufferRegion& match_buffer_region : block->match_buffers) {
      const Buffer& matched = match_buffer_region->source->buffer;
      if (matched.same_as(inlined_buffer_)) {
        this->has_opaque_access = true;
      }
    }
  }

 protected:
  /*! \brief The buffer to be inlined */
  Buffer inlined_buffer_{nullptr};
  /*! \brief The body of the block to be inlined */
  const BufferStoreNode* inlined_store_{nullptr};
  /*! \brief The scope root */
  StmtSRef scope_root_sref_{nullptr};
  /*! \brief Maps a buffer's data field to itself */
  Map<Var, Buffer> buffer_var_map_;
  /*! \brief The indices used for indexing the buffer to be inlined */
  std::vector<const VarNode*> idx_vars_;
  /*! \brief The mapping to substitute index variables to PrimExprs */
  std::unordered_map<const VarNode*, PrimExpr> idx_sub_;

 public:
  /*!
   * \brief The Stmt to be replaced when removing the leaf block
   * \note The pair (src_stmt, tgt_stmt) are produced by LeafBlockRemovalPlan to indicate a
   * transformation on top of the input AST. We take this approach to avoid changing the AST twice
   */
  Stmt src_stmt{nullptr};
  /*! \brief The Stmt to be replaced to when removing the leaf block */
  Stmt tgt_stmt{nullptr};
  /*! \brief The reuse mapping of block srefs */
  Map<Block, Block> block_reuse;
  /*! \brief Indicates if there is any opaque access of the inlined buffer */
  bool has_opaque_access{false};
};

/*!
 * \brief Helper to inline the producer block into its consumer(s)
 * The derived class implements the following functionalities:
 * 1) Substitute `BufferLoad` on the buffer to be inlined
 * to its value calculation in the producer block
 * 2) Analyze the producer block to determine the remapping of index variables
 */
class ComputeInliner : public BaseInliner {
 public:
  explicit ComputeInliner(const Buffer& inlined_buffer, const Block& producer_block,
                          const StmtSRef& scope_root_sref)
      : BaseInliner(inlined_buffer, producer_block, scope_root_sref) {}

  bool BodyPatternAllowInline(const Block& producer_block) {
    if (inlined_store_ == nullptr) {
      return false;
    }
    int n_vars = UndefinedVars(GetRef<Stmt>(inlined_store_), {}).size();
    if (!UpdateAndCheckIndexVars(inlined_store_->indices, n_vars)) {
      return false;
    }
    return true;
  }

 private:
  using BaseInliner::VisitExpr_;
  using BaseInliner::VisitStmt_;

  PrimExpr VisitExpr_(const BufferLoadNode* _load) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(_load));
    if (!load->buffer.same_as(inlined_buffer_)) {
      return std::move(load);
    }
    return ReplaceInlinedBuffer(std::move(load));
  }

  PrimExpr ReplaceInlinedBuffer(BufferLoad load) {
    SetIndexSubstitution(load->indices);
    return Substitute(inlined_store_->value, idx_sub_);
  }
};

/*!
 * \brief Helper to inline the consumer block into its producer
 * The derived class implements the following functionalities:
 * 1) Analyze the consumer block to determine the remapping of index variables
 * 2) Substitute `BufferStore` of the buffer to be inlined,
 * replacing it with direct writing to the buffer that consumer writes
 */
class ReverseComputeInliner : public BaseInliner {
  class Substituter : public StmtExprMutator {
   public:
    explicit Substituter(ReverseComputeInliner* self) : self_(self) {}

   private:
    PrimExpr VisitExpr_(const VarNode* var) final {
      auto it = self_->idx_sub_.find(var);
      ICHECK(it != self_->idx_sub_.end());
      return (*it).second;
    }

    PrimExpr VisitExpr_(const BufferLoadNode* _load) final {
      BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(_load));
      return load->buffer.same_as(self_->inlined_buffer_) ? self_->producer_rhs_ : load;
    }

    ReverseComputeInliner* self_;
  };

 public:
  explicit ReverseComputeInliner(const Buffer& inlined_buffer, const Block& consumer_block,
                                 const StmtSRef& scope_root_sref)
      : BaseInliner(inlined_buffer, consumer_block, scope_root_sref) {}

  bool BodyPatternAllowInline(const Block& consumer_block) {
    if (inlined_store_ == nullptr) {
      // Failure: block body is not BufferStore
      return false;
    }
    std::vector<const BufferLoadNode*> loads = ExtractBufferLoad(inlined_buffer_, inlined_store_);
    if (loads.size() == 0) {
      // Failure: no BufferLoad from the `inlined_buffer_`
      return false;
    }
    int n_vars = UndefinedVars(GetRef<BufferStore>(inlined_store_), {}).size();
    for (const BufferLoadNode* load : loads) {
      if (!UpdateAndCheckIndexVars(load->indices, n_vars)) {
        // Failure: incorrect of inconsistent index vars
        return false;
      }
    }
    return true;
  }

 private:
  using BaseInliner::VisitExpr_;
  using BaseInliner::VisitStmt_;

  Stmt VisitStmt_(const BufferStoreNode* _store) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(_store));
    if (!store->buffer.same_as(inlined_buffer_)) {
      return std::move(store);
    }
    return ReplaceInlinedBuffer(std::move(store));
  }

  Stmt ReplaceInlinedBuffer(BufferStore producer) {
    SetIndexSubstitution(producer->indices);
    producer_rhs_ = producer->value;
    return Substituter(this)(GetRef<BufferStore>(inlined_store_));
  }

  /*!
   * \brief Extracts expressions that loads a specific buffer
   * \param buffer The buffer to be loaded from
   * \param from The BufferStore statement to be extracted from
   * \return A list of `BufferLoad` expressions
   */
  static std::vector<const BufferLoadNode*> ExtractBufferLoad(const Buffer& buffer,
                                                              const BufferStoreNode* from) {
    struct Extractor : public ExprVisitor {
      void VisitExpr_(const BufferLoadNode* load) final {
        if (load->buffer.get() == buffer) {
          result.push_back(load);
        }
        ExprVisitor::VisitExpr_(load);
      }
      const BufferNode* buffer;
      std::vector<const BufferLoadNode*> result;
    } extractor;
    extractor.buffer = buffer.get();
    for (const PrimExpr& expr : from->indices) {
      extractor(expr);
    }
    extractor(from->value);
    return std::move(extractor.result);
  }

  /*! \brief The RHS value of the producer's BufferStore statement */
  PrimExpr producer_rhs_{nullptr};
};

void ComputeInlineImpl(ScheduleState self, const StmtSRef& producer_block_sref,
                       bool check_only = false) {
  const BlockNode* _producer_block = TVM_SREF_TO_BLOCK(_producer_block, producer_block_sref);
  Block producer_block = GetRef<Block>(_producer_block);
  Buffer inlined_buffer = NotSingleReadWriteBuffer::GetSingleWrite(self, producer_block);
  // Step 1. Get the scope block
  StmtSRef scope_root_sref = GetScopeRoot(self, producer_block_sref,
                                          /*require_stage_pipeline=*/true);
  // Step 2. Check completeness
  CheckNotOutputBlock(self, producer_block_sref, scope_root_sref);
  CheckCompleteBlock(self, producer_block_sref, scope_root_sref);
  // Step 3. Analyze the block body
  ComputeInliner inliner(inlined_buffer, producer_block, scope_root_sref);
  if (!inliner.BodyPatternAllowInline(producer_block)) {
    throw BodyAnalysisError(false, self->mod, producer_block);
  }
  // Step 4. Create a plan that removes the leaf block to be inlined
  LeafBlockRemovalPlan(self, producer_block_sref, &inliner.src_stmt, &inliner.tgt_stmt);
  // Step 5. Create an AST where the leaf `producer_block_sref` points to is removed,
  // and update other blocks who read from the removed block
  Stmt tgt_stmt = inliner(GetRef<Stmt>(scope_root_sref->stmt));
  if (inliner.has_opaque_access) {
    throw OpaqueAccessError(self->mod, scope_root_sref);
  }
  // Step 6. Do the real mutation on the AST and the sref tree in the schedule state
  if (check_only) {
    return;
  }
  self->Replace(scope_root_sref, tgt_stmt, inliner.block_reuse);
}

void ComputeInline(ScheduleState self, const StmtSRef& producer_block_sref) {
  ComputeInlineImpl(self, producer_block_sref);
}

bool CanComputeInline(const ScheduleState& self, const StmtSRef& producer_block_sref) {
  try {
    ComputeInlineImpl(self, producer_block_sref, true);
  } catch (const tvm::runtime::Error& e) {
    return false;
  }
  return true;
}

void ReverseComputeInlineImpl(ScheduleState self, const StmtSRef& consumer_block_sref,
                              bool check_only = false) {
  const BlockNode* _consumer_block = TVM_SREF_TO_BLOCK(_consumer_block, consumer_block_sref);
  Block consumer_block = GetRef<Block>(_consumer_block);
  // Step 1. Get the scope block
  StmtSRef scope_root_sref = GetScopeRoot(self, consumer_block_sref,  //
                                          /*require_stage_pipeline=*/true);
  Buffer inlined_buffer =
      NotSingleReadWriteBuffer::GetSingleRead(self, consumer_block, scope_root_sref);
  // Step 2. Check completeness
  CheckCompleteBlock(self, consumer_block_sref, scope_root_sref);
  // Step 3. Check if the consumer has a single complete producer
  NonSingleProducerError::Check(self, consumer_block_sref, scope_root_sref);
  // Step 4. Analyze the block body
  ReverseComputeInliner inliner(inlined_buffer, consumer_block, scope_root_sref);
  if (!inliner.BodyPatternAllowInline(consumer_block)) {
    throw BodyAnalysisError(true, self->mod, consumer_block);
  }
  // Step 5. Create a plan that removes the leaf block to be inlined
  LeafBlockRemovalPlan(self, consumer_block_sref, &inliner.src_stmt, &inliner.tgt_stmt);
  // Step 6. Create an AST where the leaf `consumer_block_sref` points to is removed,
  // and update other blocks who read from the removed block
  Stmt tgt_stmt = inliner(GetRef<Stmt>(scope_root_sref->stmt));
  if (inliner.has_opaque_access) {
    throw OpaqueAccessError(self->mod, scope_root_sref);
  }
  // Step 7. Do the real mutation on the AST and the sref tree in the schedule state
  if (check_only) {
    return;
  }
  self->Replace(scope_root_sref, tgt_stmt, inliner.block_reuse);
}

bool CanReverseComputeInline(const ScheduleState& self, const StmtSRef& block_sref) {
  try {
    ReverseComputeInlineImpl(self, block_sref, true);
  } catch (const tvm::runtime::Error& e) {
    return false;
  }
  return true;
}

void ReverseComputeInline(ScheduleState self, const StmtSRef& consumer_block_sref) {
  ReverseComputeInlineImpl(self, consumer_block_sref);
}

/******** InstructionKind Registration ********/

struct ComputeInlineTraits : public UnpackedInstTraits<ComputeInlineTraits> {
  static constexpr const char* kName = "ComputeInline";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv) {
    return sch->ComputeInline(block_rv);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv) {
    PythonAPICall py("compute_inline");
    py.Input("block", block_rv);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct ReverseComputeInlineTraits : public UnpackedInstTraits<ReverseComputeInlineTraits> {
  static constexpr const char* kName = "ReverseComputeInline";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv) {
    return sch->ReverseComputeInline(block_rv);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv) {
    PythonAPICall py("reverse_compute_inline");
    py.Input("block", block_rv);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(ComputeInlineTraits);
TVM_REGISTER_INST_KIND_TRAITS(ReverseComputeInlineTraits);

}  // namespace tir
}  // namespace tvm
