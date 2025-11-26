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
    'A[f(i, j, k, ...)] = g(i, j, k, ...)',
where the store indices mapping f on the left are bijective affine.)";

static const char kErrBodyReverseInline[] = R"(The body of the inlined block should be in form of
    `B[...] = g(i, j, k, A[f(i, j, k, ...)] ...)`,
where A is the only buffer the block consumes, whose indices are distinct atomic variables,
and there should be no variables other than the index variables), and f is a bijective affine
mapping and there should not be predicates in the inlined block. The iter domains of the inlined
block should be covered by the producer block.)";

class HasInitBlock : public ScheduleError {
 public:
  explicit HasInitBlock(IRModule mod, Block block) : mod_(mod), block_(block) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: The block has init statement";
  }

  ffi::String DetailRenderTemplate() const final {
    return "ScheduleError: The block has init statement: {0}";
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

  static void Check(const IRModule& mod, const Block& block) {
    if (block->init.defined()) {
      throw HasInitBlock(mod, block);
    }
  }

 private:
  IRModule mod_;
  Block block_;
};

class NotSingleReadWriteBuffer : public ScheduleError {
 public:
  explicit NotSingleReadWriteBuffer(IRModule mod, bool is_read, Block block)
      : mod_(mod), is_read_(is_read), block_(std::move(block)) {}

  ffi::String FastErrorString() const final {
    return is_read_ ? "ScheduleError: The block is allowed to read only a single buffer region"
                    : "ScheduleError: The block is allowed to write only a single buffer region";
  }

  ffi::String DetailRenderTemplate() const final {
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
  ffi::Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

  IRModule mod_;
  bool is_read_;
  Block block_;

  static Buffer GetSingleRead(const ScheduleState& self, const Block& block,
                              const StmtSRef& scope_root_sref) {
    const std::unordered_map<Buffer, ffi::Array<StmtSRef>, ObjectPtrHash, ObjectPtrEqual>&
        buffer_writers = self->block_info.at(scope_root_sref).scope->buffer_writers;
    const BufferNode* read_buffer = nullptr;
    for (const BufferRegion& read_region : block->reads) {
      const BufferNode* buffer = read_region->buffer.get();
      if (buffer == read_buffer) {
        continue;
      }
      if (buffer_writers.count(ffi::GetRef<Buffer>(buffer)) > 0) {
        if (read_buffer != nullptr) {
          throw NotSingleReadWriteBuffer(self->mod, true, block);
        }
        read_buffer = buffer;
      }
    }
    if (read_buffer == nullptr) {
      throw NotSingleReadWriteBuffer(self->mod, true, block);
    }
    return ffi::GetRef<Buffer>(read_buffer);
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

  ffi::String FastErrorString() const final {
    return "ScheduleError: The block cannot be inlined because its body pattern does not meet the "
           "condition for inlining";
  }

  ffi::String DetailRenderTemplate() const final {
    return is_reverse_ ? kErrBodyReverseInline : kErrBodyInline;
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

  bool is_reverse_;
  IRModule mod_;
  Block block_;
};

class NonSingleProducerError : public ScheduleError {
 public:
  explicit NonSingleProducerError(IRModule mod, Block block)
      : mod_(mod), block_(std::move(block)) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: The consumer block to be inlined is required to have only a single "
           "producer block, and the producer block should be a complete block who has only a "
           "single consumer";
  }

  ffi::String DetailRenderTemplate() const final {
    return "The consumer block {0} to be inlined is required to have only a single "
           "producer block, and the producer block should be a complete block who has only a "
           "single consumer";
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

  IRModule mod_;
  Block block_;

  /*!
   * \brief Check if the block has a single producer.
   * \param self The schedule state
   * \param block_sref The sref of the block to be checked
   * \param scope_root_sref The sref of the scope root
   * \return The sref of the producer block if the block has a single producer
   * \throw ScheduleError if the block does not have a single producer
   */
  static StmtSRef Check(const ScheduleState& self, const StmtSRef& consumer_block_sref,
                        const StmtSRef& scope_root_sref) {
    const BlockNode* scope_block = TVM_SREF_TO_BLOCK(scope_root_sref);
    const BlockNode* consumer_block = TVM_SREF_TO_BLOCK(consumer_block_sref);
    Buffer consumer_buffer = NotSingleReadWriteBuffer::GetSingleRead(
        self, ffi::GetRef<Block>(consumer_block), scope_root_sref);
    class ProducerFinder : public StmtVisitor {
     public:
      static std::vector<Block> GetProducer(const ScheduleState& self,
                                            const StmtSRef& scope_root_sref, const Buffer& buffer,
                                            const Block& scope_block) {
        ProducerFinder finder(self, scope_root_sref, buffer);
        finder(scope_block);
        return finder.producer_across_scope_.back();
      }

     private:
      explicit ProducerFinder(const ScheduleState& self, const StmtSRef& scope_root_sref,
                              const Buffer& buffer)
          : self_(self), scope_root_sref_(scope_root_sref), buffer_(buffer) {
        producer_across_scope_.push_back({});
      }

      void VisitStmt_(const BlockNode* node) final {
        producer_across_scope_.push_back({});
        StmtVisitor::VisitStmt_(node);
        // not a leaf block
        if (!producer_across_scope_.back().empty()) {
          auto producer_under_block = producer_across_scope_.back();
          producer_across_scope_.pop_back();
          producer_across_scope_.back().insert(producer_across_scope_.back().end(),
                                               producer_under_block.begin(),
                                               producer_under_block.end());
          return;
        }
        // leaf block
        producer_across_scope_.pop_back();
        for (const auto& write : node->writes) {
          if (write->buffer.same_as(buffer_)) {
            // Check if the producer block is a complete block
            StmtSRef producer_block_sref = self_->stmt2ref.at(node);
            if (!IsCompleteBlock(self_, producer_block_sref, scope_root_sref_)) {
              throw NonSingleProducerError(self_->mod, ffi::GetRef<Block>(node));
            }
            producer_across_scope_.back().push_back(ffi::GetRef<Block>(node));
            break;
          }
        }
      }
      ScheduleState self_;
      StmtSRef scope_root_sref_;
      Buffer buffer_;
      std::vector<std::vector<Block>> producer_across_scope_;
    };
    std::vector<Block> producer_across_scope = ProducerFinder::GetProducer(
        self, scope_root_sref, consumer_buffer, ffi::GetRef<Block>(scope_block));
    if (producer_across_scope.size() != 1) {
      throw NonSingleProducerError(self->mod, ffi::GetRef<Block>(consumer_block));
    }
    return self->stmt2ref.at(producer_across_scope[0].get());
  }
};

class OpaqueAccessError : public ScheduleError {
 public:
  explicit OpaqueAccessError(IRModule mod, StmtSRef scope_root_sref)
      : mod_(mod), scope_root_(nullptr) {
    const BlockNode* scope_root = TVM_SREF_TO_BLOCK(scope_root_sref);
    this->scope_root_ = ffi::GetRef<Block>(scope_root);
  }

  ffi::String FastErrorString() const final {
    return "ScheduleError: The buffer to be inlined has opaque access (e.g. `B.data`), or its "
           "subregion is matched into other blocks";
  }

  ffi::String DetailRenderTemplate() const final {
    return "The buffer to be inlined has opaque access (e.g. `B.data`), or its "
           "subregion is matched into other blocks: {0}";
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ObjectRef> LocationsOfInterest() const final { return {scope_root_}; }

  IRModule mod_;
  Block scope_root_;
};

class ProducerHasNonTrivialPredicateError : public ScheduleError {
 public:
  explicit ProducerHasNonTrivialPredicateError(IRModule mod, BlockRealize producer,
                                               PrimExpr new_predicate)
      : mod_(mod), producer_(producer), new_predicate_(new_predicate) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: The producer block has a non-trivial predicate.";
  }

  ffi::String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "ScheduleError: The producer block {0} has a non-trivial predicate "
       << producer_->predicate << " that cannot be implied by the synthesized predicate "
       << new_predicate_ << " of the new inlined block.";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ObjectRef> LocationsOfInterest() const final { return {producer_}; }

  IRModule mod_;
  BlockRealize producer_;
  PrimExpr new_predicate_;
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

  Stmt VisitStmt_(const ForNode* loop) final {
    if (src_stmt.get() == loop) {
      loop = tgt_stmt.as<ForNode>();
      ICHECK(loop != nullptr);
    }
    return StmtExprMutator::VisitStmt_(loop);
  }

  Stmt VisitStmt_(const BlockNode* block) {
    CheckMatchBufferRegion(block);
    AddBuffersInBlockSignature(block);
    Block src_block = ffi::GetRef<Block>(block);
    if (src_block.same_as(src_stmt)) {
      block = tgt_stmt.as<BlockNode>();
      ICHECK(block != nullptr);
    }
    Block tgt_block = Downcast<Block>(StmtExprMutator::VisitStmt_(block));
    bool is_scope_root = src_block.get() == scope_root_sref_->stmt;
    tgt_block = UpdateBuffersInBlockSignature(std::move(tgt_block), is_scope_root);
    block_reuse.Set(src_block, tgt_block);
    return tgt_block;
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
    ffi::Array<Buffer> alloc_buffers;
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
    ffi::Array<BufferRegion> reads = std::move(block->reads);
    ffi::Array<BufferRegion> writes = std::move(block->writes);
    auto f_access_inline_buffer = [this](const BufferRegion& access) {
      return access->buffer.same_as(this->inlined_buffer_);
    };
    if (!is_scope_root && (std::any_of(reads.begin(), reads.end(), f_access_inline_buffer) ||
                           std::any_of(writes.begin(), writes.end(), f_access_inline_buffer))) {
      ffi::Array<ffi::Array<BufferRegion>> inspected =
          GetBlockReadWriteRegion(block, buffer_var_map_);
      reads = inspected[0];
      writes = inspected[1];
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
  ffi::Map<Var, Buffer> buffer_var_map_;
  /*! \brief The indices used for indexing the buffer to be inlined */
  std::vector<Var> idx_vars_;
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
  ffi::Map<Block, Block> block_reuse;
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

    // Fast path on trivial case:
    // Check the store indices are same with the block iters;
    store_value_ = inlined_store_->value;
    size_t num_iters = producer_block->iter_vars.size();
    size_t buffer_ndim = inlined_store_->indices.size();
    if (num_iters == buffer_ndim) {
      std::vector<Var> idx_vars;
      idx_vars.reserve(num_iters);
      for (size_t i = 0; i < num_iters; ++i) {
        const IterVar& iter = producer_block->iter_vars[i];
        const PrimExpr& e = inlined_store_->indices[i];
        if (e.same_as(iter->var) ||
            (analyzer_.CanProveEqual(e, 0) && analyzer_.CanProveEqual(iter->dom->min, 0) &&
             analyzer_.CanProveEqual(iter->dom->extent, 1))) {
          idx_vars.push_back(iter->var);
        } else {
          break;
        }
      }
      if (idx_vars.size() == num_iters) {
        // match success
        idx_vars_ = std::move(idx_vars);
        return true;
      }
    }

    // If the mapping for store indices is non-trivial
    // check bijective mapping from producer iter var to store indices
    ffi::Map<Var, Range> producer_iter_doms;
    for (const auto& iter : producer_block->iter_vars) {
      producer_iter_doms.Set(iter->var, iter->dom);
    }
    arith::IterMapResult res = arith::DetectIterMap(
        /*indices=*/inlined_store_->indices,
        /*input_iters=*/producer_iter_doms,
        /*predicate=*/true,
        /*check_level=*/arith::IterMapLevel::Bijective,
        /*analyzer=*/&analyzer_,
        /*simplify_trivial_iterators=*/false);
    if (!res->errors.empty()) {
      // Failure: indices of BufferStore are not bijective affine
      return false;
    }
    idx_vars_.resize(buffer_ndim);
    for (size_t i = 0; i < idx_vars_.size(); ++i) {
      idx_vars_[i] = Var("ph_" + std::to_string(i), inlined_store_->indices[i].dtype());
    }
    auto inverse_iter_map = arith::InverseAffineIterMap(
        res->indices, ffi::Array<PrimExpr>(idx_vars_.begin(), idx_vars_.end()));
    for (const auto& iter : producer_block->iter_vars) {
      if (is_const_int(iter->dom->min) && analyzer_.CanProveEqual(iter->dom->extent, 1)) {
        // fallback mapping for constant iters
        inverse_iter_map.Set(iter->var, iter->dom->min);
      }
    }
    store_value_ = Substitute(store_value_, inverse_iter_map);
    return true;
  }

 private:
  using BaseInliner::VisitExpr_;
  using BaseInliner::VisitStmt_;

  PrimExpr VisitExpr_(const BufferLoadNode* _load) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(_load));
    if (!load->buffer.same_as(inlined_buffer_)) {
      return load;
    }
    return ReplaceInlinedBuffer(std::move(load));
  }

  PrimExpr ReplaceInlinedBuffer(BufferLoad load) {
    SetIndexSubstitution(load->indices);
    return Substitute(store_value_, idx_sub_);
  }

  /*!
   * \brief Set the mapping of index substitution `self->idx_sub_`
   * \param indices The expressions that the corresponding index variables are replaced to
   */
  void SetIndexSubstitution(const ffi::Array<PrimExpr>& indices) {
    ICHECK_EQ(indices.size(), idx_vars_.size());
    int n = idx_vars_.size();
    for (int i = 0; i < n; ++i) {
      idx_sub_[idx_vars_[i].get()] = indices[i];
    }
  }

  /*! \brief The arithmetic analyzer */
  arith::Analyzer analyzer_;
  /*! \brief The store value for inlinement. If the producer
   store indices are trivial, it is wrt the producer block iter var,
   otherwise it is wrt to the placeholder vars of store indices. */
  PrimExpr store_value_;
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
      if (it == self_->idx_sub_.end()) {
        return ffi::GetRef<Var>(var);
      }
      return (*it).second;
    }

    PrimExpr VisitExpr_(const BufferLoadNode* _load) final {
      BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(_load));
      return load->buffer.same_as(self_->inlined_buffer_) ? self_->producer_rhs_ : load;
    }

    ReverseComputeInliner* self_;
  };

  class RecursionResolver : public StmtExprMutator {
   public:
    explicit RecursionResolver(ReverseComputeInliner* self) : self_(self) {}

   private:
    PrimExpr VisitExpr_(const VarNode* var) final {
      auto it = self_->idx_sub_.find(var);
      if (it == self_->idx_sub_.end()) {
        return ffi::GetRef<Var>(var);
      }
      return (*it).second;
    }

    PrimExpr VisitExpr_(const BufferLoadNode* _load) final {
      BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(_load));
      return load->buffer.same_as(self_->inlined_buffer_)
                 ? StmtExprMutator::VisitExpr(
                       BufferLoad(self_->inlined_store_->buffer, self_->inlined_store_->indices))
                 : load;
    }

    ReverseComputeInliner* self_;
  };

 public:
  explicit ReverseComputeInliner(const Buffer& inlined_buffer, const BlockNode* producer_block,
                                 const BlockRealize& consumer_block_realize,
                                 const StmtSRef& scope_root_sref, const IRModule& mod)
      : BaseInliner(inlined_buffer, consumer_block_realize->block, scope_root_sref),
        producer_block_(producer_block),
        consumer_block_(consumer_block_realize->block.get()) {
    // Initialize the predicates to ensure consumer block iters are in-bound
    consumer_iter_in_bound_ = Bool(true);
    for (const IterVar& iter : consumer_block_realize->block->iter_vars) {
      consumer_iter_in_bound_ =
          consumer_iter_in_bound_ &&
          (iter->var >= iter->dom->min && iter->var < iter->dom->min + iter->dom->extent);
    }
  }

  bool BodyPatternAllowInline(const BlockRealize& consumer_block_realize) {
    const Block& consumer_block = consumer_block_realize->block;

    if (!is_one(consumer_block_realize->predicate)) {
      // Failure: Predicate is the consumer block is not supported
      return false;
    }
    if (inlined_store_ == nullptr) {
      // Failure: block body is not BufferStore
      return false;
    }
    std::vector<const BufferLoadNode*> loads = ExtractBufferLoad(inlined_buffer_, inlined_store_);
    if (loads.size() == 0) {
      // Failure: no BufferLoad from the `inlined_buffer_`
      return false;
    }

    // Collect block iter domains and update the substition map
    ffi::Map<Var, Range> consumer_iter_doms;
    for (const auto& iter_var : consumer_block->iter_vars) {
      consumer_iter_doms.Set(iter_var->var, iter_var->dom);
      // Set default mapping for unit iters
      if (is_const_int(iter_var->dom->extent, 1) && is_const_int(iter_var->dom->min)) {
        idx_sub_[iter_var->var.get()] = iter_var->dom->min;
      }
    }

    for (const BufferLoadNode* load : loads) {
      if (!UpdateAndCheckIndexExprs(load->indices)) {
        return false;
      }
    }

    arith::IterMapResult res = arith::DetectIterMap(
        /*indices=*/buffer_load_indices_,
        /*input_iters=*/consumer_iter_doms,
        /*predicate=*/true,
        /*check_level=*/arith::IterMapLevel::NoCheck,
        /*analyzer=*/&analyzer_,
        /*simplify_trivial_iterators=*/false);
    buffer_load_iter_map_ = res->indices;
    if (buffer_load_iter_map_.empty()) {
      // Failure: indices of BufferLoad are not bijective affine
      return false;
    }

    const BufferStoreNode* producer_store = nullptr;
    if (const auto* producer_if = producer_block_->body.as<tir::IfThenElseNode>()) {
      if (producer_if->else_case.defined()) {
        return false;
      }
      producer_store = producer_if->then_case.as<BufferStoreNode>();
    } else {
      producer_store = producer_block_->body.as<BufferStoreNode>();
      if (producer_block_->annotations.count(tir::attr::auto_copy) != 0) {
        const ForNode* producer_inner_loop = producer_block_->body.as<ForNode>();
        while (producer_inner_loop->body.as<ForNode>()) {
          producer_inner_loop = producer_inner_loop->body.as<ForNode>();
        }
        producer_store = producer_inner_loop->body.as<BufferStoreNode>();
      }
    }
    if (producer_store == nullptr) {
      // Failure: producer block body is not BufferStore
      return false;
    }
    CreateInverseMapping(producer_store->indices);
    if (!CheckConsumerCovered()) {
      // Failure: consumer block iter domains are not covered by the producer block
      return false;
    }

    return true;
  }

 private:
  using BaseInliner::VisitExpr_;
  using BaseInliner::VisitStmt_;

  /*! \brief Generate the predicate after inlining based on the consumer predicate */
  BlockRealize BuildInlinedConsumerPredicate(BlockRealize producer_block_realize) {
    // Bind the producer block iter domains for simplification
    ffi::Map<Var, PrimExpr> subst_map;
    Block producer_block = producer_block_realize->block;
    for (int i = 0, n = producer_block->iter_vars.size(); i < n; ++i) {
      const IterVar& iter = producer_block->iter_vars[i];
      const PrimExpr& binding = producer_block_realize->iter_values[i];
      subst_map.Set(iter->var, binding);
      analyzer_.Bind(iter->var, Range::FromMinExtent(iter->dom->min, iter->dom->extent));
    }
    if (producer_block->annotations.count(tir::attr::auto_copy) != 0) {
      auto bind = [&](const ForNode* loop) {
        analyzer_.Bind(loop->loop_var,
                       Range::FromMinExtent(make_zero(loop->extent->dtype), loop->extent));
      };
      const ForNode* producer_inner_loop = producer_block->body.as<ForNode>();
      while (producer_inner_loop->body.as<ForNode>()) {
        bind(producer_inner_loop);
        producer_inner_loop = producer_inner_loop->body.as<ForNode>();
      }
      bind(producer_inner_loop);
    }
    // Substitute the consumer block iters with the corresponding iters in the producer blocks
    PrimExpr predicate = Substituter(this)(consumer_iter_in_bound_);
    // Simplify the predicate using the producer block iter domains
    predicate = analyzer_.Simplify(predicate);
    if (is_one(predicate)) {
      return producer_block_realize;
    }
    if (const auto* if_ = producer_block->body.as<IfThenElseNode>()) {
      if (!if_->else_case.defined()) {
        PrimExpr if_predicate = analyzer_.Simplify(if_->condition);
        if (!StructuralEqual()(predicate, if_predicate)) {
          predicate = analyzer_.Simplify(predicate && if_->condition);
          producer_block.CopyOnWrite()->body = if_->then_case;
        }
      }
    }
    PrimExpr outer_predicate = Substitute(predicate, subst_map);
    auto n = producer_block_realize.CopyOnWrite();
    n->block = producer_block;
    n->predicate = analyzer_.Simplify(outer_predicate);
    return ffi::GetRef<BlockRealize>(n);
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    Block src_block = op->block;
    BlockRealize tgt_block_realize = Downcast<BlockRealize>(StmtMutator::VisitStmt_(op));
    if (src_block.get() == producer_block_) {
      tgt_block_realize = BuildInlinedConsumerPredicate(tgt_block_realize);
      block_reuse.Set(src_block, tgt_block_realize->block);
    }
    return tgt_block_realize;
  }

  Stmt VisitStmt_(const BufferStoreNode* _store) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(_store));
    if (!store->buffer.same_as(inlined_buffer_)) {
      return store;
    }
    return ReplaceInlinedBuffer(std::move(store));
  }

  /*!
   * \brief Check the consumer block iter domains are covered by the producer block iter domains
   * \return Whether the consumer block iter domains are covered
   */
  bool CheckConsumerCovered() {
    ffi::Map<IterVar, arith::IntSet> producer_iter_doms;
    for (const IterVar& iter_var : producer_block_->iter_vars) {
      producer_iter_doms.Set(iter_var, arith::IntSet::FromRange(iter_var->dom));
    }
    // For each block iter in the consumer block, find the corresponding expression in the producer
    for (const IterVar& iter : consumer_block_->iter_vars) {
      if (auto it = idx_sub_.find(iter->var.get()); it != idx_sub_.end()) {
        const PrimExpr& producer_iter = it->second;
        arith::IntSet producer_iter_range = arith::EvalSet(producer_iter, producer_iter_doms);
        if (analyzer_.CanProve(producer_iter_range.min() > iter->dom->min) ||
            analyzer_.CanProve(producer_iter_range.max() <
                               iter->dom->min + iter->dom->extent - 1)) {
          return false;
        }
      } else {
        return false;
      }
    }
    return true;
  }

  /*!
   * \brief Apply the inverse of `buffer_load_iter_map_` to producer indices. Update `idx_sub_` with
   *        the result. It will be later used to transform the BufferStore indices of the producer.
   * \param producer_indices The BufferStore indices of the producer.
   */
  void CreateInverseMapping(const ffi::Array<PrimExpr> producer_indices) {
    auto inverse_iter_map = arith::InverseAffineIterMap(buffer_load_iter_map_, producer_indices);
    for (const auto& pair : inverse_iter_map) {
      idx_sub_[pair.first.get()] = pair.second;
    }
  }

  Stmt ReplaceInlinedBuffer(BufferStore producer) {
    // "producer->value" may contain the buffer that is inlined in cases of reduction,
    // so we need to resolve the recursion first
    producer_rhs_ = RecursionResolver(this)(producer->value);
    return Substituter(this)(ffi::GetRef<BufferStore>(inlined_store_));
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

  /*!
   * \brief Update `buffer_load_indices_` with the given indices. If `buffer_load_indices_` is
   *        already non-empty, check it is consistent with the given indices.
   * \param indices The indices
   * \param expected_ndim The expected ndim of the access
   * \return A boolean flag indicating if the check is successful
   */
  bool UpdateAndCheckIndexExprs(const ffi::Array<PrimExpr>& indices) {
    if (buffer_load_indices_.empty()) {
      buffer_load_indices_ = indices;
    } else if (!std::equal(buffer_load_indices_.begin(), buffer_load_indices_.end(),
                           indices.begin(), indices.end(), ExprDeepEqual())) {
      // Failure: indices are not consistent in different BufferLoads
      return false;
    }
    return true;
  }

  /*! \brief The RHS value of the producer's BufferStore statement */
  PrimExpr producer_rhs_{nullptr};
  /*! \brief The indices of the consumer's BufferLoad */
  ffi::Array<PrimExpr> buffer_load_indices_;
  /*! \brief The IterMap representing the indices of the consumer's BufferLoad */
  ffi::Array<arith::IterSumExpr> buffer_load_iter_map_{nullptr};
  /*! \brief The producer block */
  const BlockNode* producer_block_{nullptr};
  /* \brief The consumer block */
  const BlockNode* consumer_block_{nullptr};
  /*! \brief The predicate to ensure the consumer block iters are in-bound. It will be inserted
   * as the predicate of the producer block after inlining.
   */
  PrimExpr consumer_iter_in_bound_{nullptr};
  /*! \brief The arithmetic analyzer */
  arith::Analyzer analyzer_;
};

void ComputeInlineImpl(ScheduleState self, const StmtSRef& producer_block_sref,
                       bool check_only = false) {
  const BlockNode* _producer_block = TVM_SREF_TO_BLOCK(producer_block_sref);
  Block producer_block = ffi::GetRef<Block>(_producer_block);
  HasInitBlock::Check(self->mod, producer_block);
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
  Stmt tgt_stmt = inliner(ffi::GetRef<Stmt>(scope_root_sref->stmt));
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
  const BlockNode* _consumer_block = TVM_SREF_TO_BLOCK(consumer_block_sref);
  Block consumer_block = ffi::GetRef<Block>(_consumer_block);
  BlockRealize consumer_block_realize = GetBlockRealize(self, consumer_block_sref);
  HasInitBlock::Check(self->mod, consumer_block);
  // Step 1. Get the scope block
  StmtSRef scope_root_sref = GetScopeRoot(self, consumer_block_sref,  //
                                          /*require_stage_pipeline=*/true);
  Buffer inlined_buffer =
      NotSingleReadWriteBuffer::GetSingleRead(self, consumer_block, scope_root_sref);
  // Step 2. Check completeness
  CheckCompleteBlock(self, consumer_block_sref, scope_root_sref);
  // Step 3. Check if the consumer has a single complete producer, and the producer is not an output
  // block
  StmtSRef producer_block_sref =
      NonSingleProducerError::Check(self, consumer_block_sref, scope_root_sref);
  CheckNotOutputBlock(self, producer_block_sref, scope_root_sref);
  // Step 4. Analyze the block body
  ReverseComputeInliner inliner(inlined_buffer, producer_block_sref->StmtAs<BlockNode>(),
                                consumer_block_realize, scope_root_sref, self->mod);
  if (!inliner.BodyPatternAllowInline(consumer_block_realize)) {
    throw BodyAnalysisError(true, self->mod, consumer_block);
  }
  // Step 5. Create a plan that removes the leaf block to be inlined
  LeafBlockRemovalPlan(self, consumer_block_sref, &inliner.src_stmt, &inliner.tgt_stmt);
  // Step 6. Create an AST where the leaf `consumer_block_sref` points to is removed,
  // and update other blocks who read from the removed block
  Stmt tgt_stmt = inliner(ffi::GetRef<Stmt>(scope_root_sref->stmt));
  if (inliner.has_opaque_access) {
    throw OpaqueAccessError(self->mod, scope_root_sref);
  }
  // Step 7. Do the real mutation on the AST and the sref tree in the schedule state
  if (check_only) {
    return;
  }
  self->Replace(scope_root_sref, tgt_stmt, inliner.block_reuse);
  // Step 8. Update the cached flags
  arith::Analyzer analyzer;
  BlockInfo& block_info = self->block_info[producer_block_sref];
  block_info.affine_binding = IsAffineBinding(
      /*realize=*/GetBlockRealize(self, producer_block_sref),
      /*loop_var_ranges=*/
      LoopDomainOfSRefTreePath(ffi::GetRef<StmtSRef>(producer_block_sref->parent)),
      /*analyzer=*/&analyzer);
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

/*!
 * \brief Helper to fuse epilogue block into reduction block
 * Analyzes epilogue pattern and transforms reduction init/update
 */
class ReductionEpilogueFuser : public BaseInliner {
 public:
  explicit ReductionEpilogueFuser(const Buffer& reduction_buffer, const BlockNode* reduction_block,
                                  const BlockRealize& epilogue_block_realize,
                                  const StmtSRef& scope_root_sref)
      : BaseInliner(reduction_buffer, epilogue_block_realize->block, scope_root_sref),
        reduction_block_(reduction_block),
        epilogue_block_(epilogue_block_realize->block.get()) {}

  bool BodyPatternAllowFusion(const BlockRealize& epilogue_block_realize);

  // Step 2: Create single fused reduction block
  Block CreateFusedReductionBlock(const BlockNode* reduction_block,
                                  const BlockRealizeNode* reduction_realize);

 private:
  bool AnalyzeEpiloguePattern(const PrimExpr& value);
  bool IsReductionBlock(const BlockNode* block);
  void ExtractEpilogueInfo();
  // Helper function to extract BufferLoad nodes from BufferStore
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

  const BlockNode* reduction_block_;
  const BlockNode* epilogue_block_;
  PrimExpr epilogue_addend_{nullptr};                      // C[vi, vj] in D = temp + C
  Buffer epilogue_output_buffer_{nullptr};                 // Output buffer D
  ffi::Array<PrimExpr> epilogue_output_indices_{nullptr};  // Indices of D[vi, vj]
  BufferRegion epilogue_output_region_{nullptr};           // Write region of D
  Buffer epilogue_addend_buffer_{nullptr};                 // Addend buffer C
  BufferRegion epilogue_addend_region_{nullptr};           // Read region of C
};

bool ReductionEpilogueFuser::BodyPatternAllowFusion(const BlockRealize& epilogue_block_realize) {
  // 1. Validate predicate
  if (!is_one(epilogue_block_realize->predicate)) {
    // Failure: Predicate in epilogue block is not supported
    return false;
  }

  // 2. Check if epilogue body is BufferStore
  if (inlined_store_ == nullptr) {
    // Failure: epilogue block body is not BufferStore
    return false;
  }

  // 3. Check if epilogue reads from reduction buffer
  std::vector<const BufferLoadNode*> loads = ExtractBufferLoad(inlined_buffer_, inlined_store_);
  if (loads.size() == 0) {
    // Failure: no BufferLoad from the reduction buffer
    return false;
  }

  // 4. Analyze epilogue pattern: D[i,j] = temp[i,j] + C[i,j]
  if (!AnalyzeEpiloguePattern(inlined_store_->value)) {
    // Failure: epilogue is not a simple addition pattern
    return false;
  }

  // 5. Check if producer is a reduction block
  if (!IsReductionBlock(reduction_block_)) {
    // Failure: producer is not a reduction block
    return false;
  }

  // 6. Extract epilogue information (output buffer, indices, regions, etc.)
  ExtractEpilogueInfo();

  return true;
}

bool ReductionEpilogueFuser::AnalyzeEpiloguePattern(const PrimExpr& value) {
  // Pattern: temp[i,j] + C[i,j] or C[i,j] + temp[i,j]
  if (const auto* add = value.as<AddNode>()) {
    const auto* load_a = add->a.as<BufferLoadNode>();
    const auto* load_b = add->b.as<BufferLoadNode>();

    bool a_is_target = load_a && load_a->buffer.same_as(inlined_buffer_);
    bool b_is_target = load_b && load_b->buffer.same_as(inlined_buffer_);

    // Ensure exactly one operand is from the reduction buffer
    if (a_is_target != b_is_target) {
      epilogue_addend_ = a_is_target ? add->b : add->a;
      return true;
    }
  }

  return false;
}

bool ReductionEpilogueFuser::IsReductionBlock(const BlockNode* block) {
  // Check if block has reduction iter vars
  for (const IterVar& iter : block->iter_vars) {
    if (iter->iter_type == kCommReduce) {
      return true;
    }
  }
  return false;
}

void ReductionEpilogueFuser::ExtractEpilogueInfo() {
  // Extract epilogue output buffer and indices
  epilogue_output_buffer_ = inlined_store_->buffer;
  epilogue_output_indices_ = inlined_store_->indices;

  // Extract epilogue output region from epilogue block writes
  for (const BufferRegion& write : epilogue_block_->writes) {
    if (write->buffer.same_as(epilogue_output_buffer_)) {
      epilogue_output_region_ = write;
      break;
    }
  }

  // Extract epilogue addend buffer and region from epilogue_addend_
  if (const auto* load = epilogue_addend_.as<BufferLoadNode>()) {
    epilogue_addend_buffer_ = load->buffer;
    // Find the read region from epilogue block reads
    for (const BufferRegion& read : epilogue_block_->reads) {
      if (read->buffer.same_as(epilogue_addend_buffer_)) {
        epilogue_addend_region_ = read;
        break;
      }
    }
  }
}

Block ReductionEpilogueFuser::CreateFusedReductionBlock(const BlockNode* reduction_block,
                                                        const BlockRealizeNode* reduction_realize) {
  ObjectPtr<BlockNode> new_block = ffi::make_object<BlockNode>(*reduction_block);

  // 1. Map epilogue block vars to reduction block vars
  std::vector<Var> reduction_data_vars;
  for (const IterVar& iter_var : reduction_block->iter_vars) {
    if (iter_var->iter_type == IterVarType::kDataPar) {
      reduction_data_vars.push_back(iter_var->var);
    }
  }
  std::vector<Var> epilogue_data_vars;
  for (const IterVar& iter_var : epilogue_block_->iter_vars) {
    if (iter_var->iter_type == IterVarType::kDataPar) {
      epilogue_data_vars.push_back(iter_var->var);
    }
  }

  ICHECK_EQ(reduction_data_vars.size(), epilogue_data_vars.size())
      << "ValueError: The number of data parallel iter vars must be the same in the reduction "
         "and epilogue blocks.";

  std::unordered_map<Var, Var> var_map;
  for (size_t i = 0; i < reduction_data_vars.size(); ++i) {
    var_map[epilogue_data_vars[i]] = reduction_data_vars[i];
  }

  // 2. Change init to epilogue value: D[vi, vj] = C[vi, vj]
  BufferStore new_init_store(epilogue_output_buffer_, Substitute(epilogue_addend_, var_map),
                             Substitute(epilogue_output_indices_, var_map));
  new_block->init = new_init_store;

  // 3. Replace output buffer from temp to D in body
  class BufferReplacer : public StmtExprMutator {
   public:
    BufferReplacer(Buffer old_buf, Buffer new_buf) : old_buffer_(old_buf), new_buffer_(new_buf) {}

    Stmt VisitStmt_(const BufferStoreNode* op) final {
      BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
      if (store->buffer.same_as(old_buffer_)) {
        return BufferStore(new_buffer_, store->value, store->indices);
      }
      return store;
    }

    PrimExpr VisitExpr_(const BufferLoadNode* op) final {
      BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
      if (load->buffer.same_as(old_buffer_)) {
        return BufferLoad(new_buffer_, load->indices);
      }
      return load;
    }

   private:
    Buffer old_buffer_;
    Buffer new_buffer_;
  };

  BufferReplacer replacer(inlined_buffer_, epilogue_output_buffer_);
  new_block->body = replacer(reduction_block->body);

  // 4. Update write regions
  ffi::Array<BufferRegion> new_writes;
  for (const BufferRegion& write : reduction_block->writes) {
    if (write->buffer.same_as(inlined_buffer_)) {
      new_writes.push_back(
          BufferRegion(epilogue_output_buffer_, Substitute(write->region, var_map)));
    } else {
      new_writes.push_back(write);
    }
  }
  new_block->writes = new_writes;

  // 5. Update read regions (C first, then A, B)
  ffi::Array<BufferRegion> new_reads;
  std::unordered_set<const BufferNode*> read_bufs;

  // Add C buffer read first (used in init)
  if (epilogue_addend_buffer_.defined()) {
    new_reads.push_back(BufferRegion(epilogue_addend_buffer_,
                                     Substitute(epilogue_addend_region_->region, var_map)));
    read_bufs.insert(epilogue_addend_buffer_.get());
  }

  // Add existing read regions (A, B, etc.)
  for (const BufferRegion& read : reduction_block->reads) {
    if (!read->buffer.same_as(inlined_buffer_)) {
      // Only add non-temp buffers
      if (read_bufs.find(read->buffer.get()) == read_bufs.end()) {
        new_reads.push_back(read);
        read_bufs.insert(read->buffer.get());
      }
    }
  }

  new_block->reads = new_reads;

  return Block(new_block);
}

/*!
 * \brief Check if a buffer is still referenced by other blocks in the scope
 */
static bool CheckBufferStillUsed(const Block& scope_root, const Buffer& buffer) {
  class BufferUsageChecker : public StmtVisitor {
   public:
    explicit BufferUsageChecker(const Buffer& buffer) : buffer_(buffer) {}

    bool CheckStmt(const Stmt& stmt) {
      found_usage_ = false;
      VisitStmt(stmt);
      return found_usage_;
    }

   private:
    void VisitStmt_(const BlockRealizeNode* op) final {
      if (found_usage_) return;

      if (!op || !op->block.defined()) {
        StmtVisitor::VisitStmt_(op);
        return;
      }

      const BlockNode* block = op->block.get();
      if (!block) {
        StmtVisitor::VisitStmt_(op);
        return;
      }

      // Check reads
      for (const BufferRegion& read : block->reads) {
        if (read->buffer.same_as(buffer_)) {
          found_usage_ = true;
          return;
        }
      }

      // Check writes
      for (const BufferRegion& write : block->writes) {
        if (write->buffer.same_as(buffer_)) {
          found_usage_ = true;
          return;
        }
      }

      // Continue visiting nested blocks
      StmtVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const BlockNode* op) final {
      if (found_usage_) return;
      if (!op) return;

      // Check alloc_buffers
      for (const Buffer& buf : op->alloc_buffers) {
        if (buf.same_as(buffer_)) {
          found_usage_ = true;
          return;
        }
      }

      StmtVisitor::VisitStmt_(op);
    }

    const Buffer& buffer_;
    bool found_usage_{false};
  };

  if (!scope_root->body.defined()) {
    return false;
  }

  BufferUsageChecker checker(buffer);
  return checker.CheckStmt(scope_root->body);
}

/*!
 * \brief Helper class to replace reduction and epilogue blocks with a single fused block
 */
class SingleBlockFusionReplacer : public StmtMutator {
 public:
  static Block Replace(Block old_scope_root, Block new_fused_block, Block old_reduction_block,
                       Block old_epilogue_block, Buffer reduction_buffer) {
    SingleBlockFusionReplacer replacer(std::move(new_fused_block), std::move(old_reduction_block),
                                       std::move(old_epilogue_block), std::move(reduction_buffer));
    Block result = Downcast<Block>(replacer(std::move(old_scope_root)));

    // Check if reduction_buffer is still referenced by other blocks
    bool buffer_still_used = CheckBufferStillUsed(result, reduction_buffer);

    // Remove intermediate temp buffer only if it's not used by other blocks
    if (!buffer_still_used) {
      BlockNode* p = result.CopyOnWrite();
      ffi::Array<Buffer> new_alloc_buffers;
      for (const Buffer& buf : p->alloc_buffers) {
        if (!buf.same_as(reduction_buffer)) {
          new_alloc_buffers.push_back(buf);
        }
      }
      p->alloc_buffers = new_alloc_buffers;
    }

    return result;
  }

 private:
  explicit SingleBlockFusionReplacer(Block new_fused_block, Block old_reduction_block,
                                     Block old_epilogue_block, Buffer reduction_buffer)
      : new_fused_block_(std::move(new_fused_block)),
        old_reduction_block_(std::move(old_reduction_block)),
        old_epilogue_block_(std::move(old_epilogue_block)),
        reduction_buffer_(std::move(reduction_buffer)) {}

  Stmt VisitStmt_(const ForNode* loop) final {
    Stmt mutated_body = StmtMutator::VisitStmt(loop->body);
    // Remove empty loops (containing only Evaluate(0))
    if (mutated_body.as<EvaluateNode>()) {
      return mutated_body;  // Return Evaluate(0) to be removed by SeqStmt
    }

    return For(loop->loop_var, loop->min, loop->extent, loop->kind, mutated_body,
               loop->thread_binding, loop->annotations);
  }

  Stmt VisitStmt_(const BlockRealizeNode* realize) final {
    if (realize->block.same_as(old_reduction_block_)) {
      // Replace reduction block with new fused block
      ObjectPtr<BlockRealizeNode> new_realize = ffi::make_object<BlockRealizeNode>(*realize);
      new_realize->block = new_fused_block_;
      return BlockRealize(new_realize);
    } else if (realize->block.same_as(old_epilogue_block_)) {
      // Remove epilogue block completely
      return Evaluate(0);
    }
    return StmtMutator::VisitStmt_(realize);
  }

  Stmt VisitStmt_(const SeqStmtNode* seq) final {
    ffi::Array<Stmt> new_stmts;
    for (const Stmt& stmt : seq->seq) {
      Stmt new_stmt = VisitStmt(stmt);
      // Remove Evaluate(0)
      if (!new_stmt.as<EvaluateNode>()) {
        new_stmts.push_back(new_stmt);
      }
    }
    return SeqStmt::Flatten(new_stmts);
  }

 private:
  Block new_fused_block_;
  Block old_reduction_block_;
  Block old_epilogue_block_;
  Buffer reduction_buffer_;
};

void FuseReductionEpilogueImpl(ScheduleState self, const StmtSRef& reduction_block_sref,
                               const StmtSRef& epilogue_block_sref, bool check_only = false) {
  const BlockNode* _reduction_block = TVM_SREF_TO_BLOCK(reduction_block_sref);
  const BlockNode* _epilogue_block = TVM_SREF_TO_BLOCK(epilogue_block_sref);

  Block reduction_block = ffi::GetRef<Block>(_reduction_block);
  Block epilogue_block = ffi::GetRef<Block>(_epilogue_block);
  BlockRealize epilogue_block_realize = GetBlockRealize(self, epilogue_block_sref);

  // Step 1. Get the scope block
  StmtSRef scope_root_sref =
      GetScopeRoot(self, epilogue_block_sref, /*require_stage_pipeline=*/true);

  // Step 2. Get the reduction buffer (intermediate buffer)
  Buffer reduction_buffer = NotSingleReadWriteBuffer::GetSingleWrite(self, reduction_block);

  // Step 3. Check completeness and reduction block properties
  CheckReductionBlock(self, reduction_block_sref, scope_root_sref);
  CheckCompleteBlock(self, epilogue_block_sref, scope_root_sref);
  CheckNotOutputBlock(self, reduction_block_sref, scope_root_sref);

  // Step 4. Analyze the epilogue pattern
  ReductionEpilogueFuser fuser(reduction_buffer, _reduction_block, epilogue_block_realize,
                               scope_root_sref);
  if (!fuser.BodyPatternAllowFusion(epilogue_block_realize)) {
    throw BodyAnalysisError(true, self->mod, epilogue_block);
  }

  if (check_only) {
    return;
  }

  // Step 5. Create single fused reduction block
  BlockRealize reduction_realize = GetBlockRealize(self, reduction_block_sref);
  Block fused_block = fuser.CreateFusedReductionBlock(_reduction_block, reduction_realize.get());

  // Step 6. Transform and replace IR
  const BlockNode* old_scope_root = TVM_SREF_TO_BLOCK(scope_root_sref);

  Block new_scope_root =
      SingleBlockFusionReplacer::Replace(ffi::GetRef<Block>(old_scope_root), fused_block,
                                         reduction_block, epilogue_block, reduction_buffer);

  // Step 7. Update schedule state
  ffi::Map<Block, Block> block_reuse;
  block_reuse.Set(ffi::GetRef<Block>(old_scope_root), new_scope_root);
  block_reuse.Set(reduction_block, fused_block);
  self->Replace(scope_root_sref, new_scope_root, block_reuse);

  // Step 8. Update BlockInfo
  self->UpdateScopeBlockInfo(GetBlockRealize(self, scope_root_sref));
}

void FuseReductionEpilogue(ScheduleState self, const StmtSRef& reduction_block_sref,
                           const StmtSRef& epilogue_block_sref) {
  FuseReductionEpilogueImpl(self, reduction_block_sref, epilogue_block_sref);
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

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block_rv) {
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

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block_rv) {
    PythonAPICall py("reverse_compute_inline");
    py.Input("block", block_rv);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(ComputeInlineTraits);
TVM_REGISTER_INST_KIND_TRAITS(ReverseComputeInlineTraits);

struct FuseReductionEpilogueTraits : public UnpackedInstTraits<FuseReductionEpilogueTraits> {
  static constexpr const char* kName = "FuseReductionEpilogue";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV reduction_block_rv,
                                      BlockRV epilogue_block_rv) {
    return sch->FuseReductionEpilogue(reduction_block_rv, epilogue_block_rv);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs,
                                      ffi::String reduction_block_rv,
                                      ffi::String epilogue_block_rv) {
    PythonAPICall py("fuse_reduction_epilogue");
    py.Input("reduction_block", reduction_block_rv);
    py.Input("epilogue_block", epilogue_block_rv);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(FuseReductionEpilogueTraits);

}  // namespace tir
}  // namespace tvm
