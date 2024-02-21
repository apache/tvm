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

  String FastErrorString() const final { return "ScheduleError: The block has init statement"; }

  String DetailRenderTemplate() const final {
    return "ScheduleError: The block has init statement: {0}";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

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
        self, GetRef<Block>(consumer_block), scope_root_sref);
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
              throw NonSingleProducerError(self_->mod, GetRef<Block>(node));
            }
            producer_across_scope_.back().push_back(GetRef<Block>(node));
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
        self, scope_root_sref, consumer_buffer, GetRef<Block>(scope_block));
    if (producer_across_scope.size() != 1) {
      throw NonSingleProducerError(self->mod, GetRef<Block>(consumer_block));
    }
    return self->stmt2ref.at(producer_across_scope[0].get());
  }
};

class OpaqueAccessError : public ScheduleError {
 public:
  explicit OpaqueAccessError(IRModule mod, StmtSRef scope_root_sref)
      : mod_(mod), scope_root_(nullptr) {
    const BlockNode* scope_root = TVM_SREF_TO_BLOCK(scope_root_sref);
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

class ProducerHasNonTrivialPredicateError : public ScheduleError {
 public:
  explicit ProducerHasNonTrivialPredicateError(IRModule mod, BlockRealize producer,
                                               PrimExpr new_predicate)
      : mod_(mod), producer_(producer), new_predicate_(new_predicate) {}

  String FastErrorString() const final {
    return "ScheduleError: The producer block has a non-trivial predicate.";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "ScheduleError: The producer block {0} has a non-trivial predicate "
       << producer_->predicate << " that cannot be implied by the synthesized predicate "
       << new_predicate_ << " of the new inlined block.";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {producer_}; }

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
    Map<Var, Range> producer_iter_doms;
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
        res->indices, Array<PrimExpr>(idx_vars_.begin(), idx_vars_.end()));
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
      return std::move(load);
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
  void SetIndexSubstitution(const Array<PrimExpr>& indices) {
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
        return GetRef<Var>(var);
      }
      return (*it).second;
    }

    PrimExpr VisitExpr_(const BufferLoadNode* _load) final {
      BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(_load));
      return load->buffer.same_as(self_->inlined_buffer_) ? self_->producer_rhs_ : load;
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
    Map<Var, Range> consumer_iter_doms;
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
  Block BuildInlinedConsumerPredicate(const BlockNode* producer_block) {
    // Bind the producer block iter domains for simplification
    Map<Var, PrimExpr> subst_map;
    for (int i = 0, n = producer_block->iter_vars.size(); i < n; ++i) {
      const IterVar& iter = producer_block->iter_vars[i];
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
    ObjectPtr<BlockNode> block = make_object<BlockNode>(*producer_block);
    if (is_one(predicate)) {
      return Block(block);
    }
    if (const auto* if_ = producer_block->body.as<tir::IfThenElseNode>()) {
      PrimExpr if_predicate = analyzer_.Simplify(if_->condition);
      if (!StructuralEqual()(predicate, if_predicate)) {
        predicate = analyzer_.Simplify(predicate && if_->condition);
      }
      block->body = IfThenElse(predicate, if_->then_case);
      return Block(block);
    }
    block->body = IfThenElse(predicate, block->body);
    return Block(block);
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    Block src_block = GetRef<Block>(op);
    Block tgt_block = Downcast<Block>(BaseInliner::VisitStmt_(op));
    if (op == producer_block_) {
      tgt_block = BuildInlinedConsumerPredicate(tgt_block.get());
      block_reuse.Set(src_block, tgt_block);
    }
    return std::move(tgt_block);
  }

  Stmt VisitStmt_(const BufferStoreNode* _store) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(_store));
    if (!store->buffer.same_as(inlined_buffer_)) {
      return std::move(store);
    }
    return ReplaceInlinedBuffer(std::move(store));
  }

  /*!
   * \brief Check the consumer block iter domains are covered by the producer block iter domains
   * \return Whether the consumer block iter domains are covered
   */
  bool CheckConsumerCovered() {
    Map<IterVar, arith::IntSet> producer_iter_doms;
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
  void CreateInverseMapping(const Array<PrimExpr> producer_indices) {
    auto inverse_iter_map = arith::InverseAffineIterMap(buffer_load_iter_map_, producer_indices);
    for (const auto& pair : inverse_iter_map) {
      idx_sub_[pair.first.get()] = pair.second;
    }
  }

  Stmt ReplaceInlinedBuffer(BufferStore producer) {
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

  /*!
   * \brief Update `buffer_load_indices_` with the given indices. If `buffer_load_indices_` is
   *        already non-empty, check it is consistent with the given indices.
   * \param indices The indices
   * \param expected_ndim The expected ndim of the access
   * \return A boolean flag indicating if the check is successful
   */
  bool UpdateAndCheckIndexExprs(const Array<PrimExpr>& indices) {
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
  Array<PrimExpr> buffer_load_indices_;
  /*! \brief The IterMap representing the indices of the consumer's BufferLoad */
  Array<arith::IterSumExpr> buffer_load_iter_map_{nullptr};
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
  Block producer_block = GetRef<Block>(_producer_block);
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
  const BlockNode* _consumer_block = TVM_SREF_TO_BLOCK(consumer_block_sref);
  Block consumer_block = GetRef<Block>(_consumer_block);
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
  Stmt tgt_stmt = inliner(GetRef<Stmt>(scope_root_sref->stmt));
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
      /*loop_var_ranges=*/LoopDomainOfSRefTreePath(GetRef<StmtSRef>(producer_block_sref->parent)),
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
