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

#include "../utils.h"

namespace tvm {
namespace tir {

/*! \brief The schedule error class when the padding size is invalid. */
class InvalidPaddingError : public ScheduleError {
 public:
  InvalidPaddingError(IRModule mod, Block block, Array<Integer> padding)
      : mod_(std::move(mod)), block_(std::move(block)), padding_(std::move(padding)) {}
  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }
  String FastErrorString() const final {
    return "ScheduleError: The padding size for the block is invalid.";
  }
  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The padding for the block {0} are invalid. It should be a list of "
       << block_->iter_vars.size() << " non-negative integers. Got " << padding_;
    return os.str();
  }

  static void Check(const ScheduleState& self, const Block& block, Array<Integer> padding) {
    if (padding.size() != block->iter_vars.size()) {
      throw InvalidPaddingError(self->mod, block, padding);
    }
    for (const auto& pad : padding) {
      if (pad->value < 0) {
        throw InvalidPaddingError(self->mod, block, padding);
      }
    }
  }

 private:
  IRModule mod_;
  Block block_;
  Array<Integer> padding_;
};

/*! \brief The schedule error class when the block body is not an Einsum pattern. */
class NonEinsumError : public ScheduleError {
 public:
  explicit NonEinsumError(IRModule mod, Block block)
      : mod_(std::move(mod)), block_(std::move(block)) {}

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }
  String FastErrorString() const final {
    return "ScheduleError: The block is not a computation of Einsum pattern.";
  }
  String DetailRenderTemplate() const final {
    return "The block {0} not a computation of Einsum pattern.";
  }

 private:
  IRModule mod_;
  Block block_;
};

/*! \brief Data structure that represents a Einsum computation. */
struct Einsum {
  // The output buffer
  Buffer output_buffer;
  // The indices of the output buffer
  Array<Var> output_indices;
  // The indices of the input buffers
  Map<Buffer, Array<Var>> input_indices;
};

class EinsumExtractor : public ExprVisitor {
 public:
  EinsumExtractor() = default;

  std::optional<Einsum> Extract(const Block& block) {
    const BufferStoreNode* update = block->body.as<BufferStoreNode>();
    // Step 1: Check the body is a BufferStore and the block has the init statement, and the
    // BufferStore and the init statement store have the same output buffer indices.
    if (update == nullptr || !block->init.defined()) {
      return std::nullopt;
    }

    if (Optional<Array<Var>> opt_indices = CheckTrivialBufferIndices(update);
        opt_indices.defined()) {
      ein_sum_.output_indices = std::move(opt_indices.value());
    } else {
      return std::nullopt;
    }
    ein_sum_.output_buffer = update->buffer;

    const BufferStoreNode* init = block->init.value().as<BufferStoreNode>();
    ICHECK(init != nullptr);
    if (!CompareBufferIndices(init->indices, ein_sum_.output_indices)) {
      return std::nullopt;
    }
    // Step 2: Check the BufferStore updates the output buffer and the input buffers indices are
    // block iter variables.
    CheckStoreValue(update->value);
    if (fail_) {
      return std::nullopt;
    }
    return std::move(ein_sum_);
  }

 private:
  void CheckStoreValue(const PrimExpr& update) {
    // Check the update part has the form:
    //   Output[output_indices] += Input_0[input_indices_0] op_0 Input_1[input_indices_1] op_1 ...
    // where output_indices and input_indices_i are the indices are arrays whose elements are the
    // block iter variables instead of composite PrimExpr, and op_i are the binary operations.

    // Check the value is Add and eithe LHS or RHS is the BufferLoad from the output buffer.
    const AddNode* add = update.as<AddNode>();
    if (add == nullptr) {
      fail_ = true;
      return;
    }
    const BufferLoadNode* lhs = add->a.as<BufferLoadNode>();
    const BufferLoadNode* rhs = add->b.as<BufferLoadNode>();
    if (lhs == nullptr && rhs != nullptr) {
      std::swap(lhs, rhs);
    }
    if (lhs == nullptr || !lhs->buffer.same_as(ein_sum_.output_buffer) ||
        !CompareBufferIndices(lhs->indices, ein_sum_.output_indices)) {
      fail_ = true;
      return;
    }
    VisitExpr(add->b);
  }

  void VisitExpr(const PrimExpr& n) final {
    if (n->IsInstance<BufferLoadNode>() || n->IsInstance<MulNode>() || n->IsInstance<CastNode>()) {
      ExprVisitor::VisitExpr(n);
    } else {
      fail_ = true;
      return;
    }
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    if (auto it = ein_sum_.input_indices.find(op->buffer);
        it != ein_sum_.input_indices.end() && !CompareBufferIndices(op->indices, (*it).second)) {
      fail_ = true;
      return;
    }
    if (Optional<Array<Var>> opt_indices = CheckTrivialBufferIndices(op); opt_indices.defined()) {
      ein_sum_.input_indices.Set(op->buffer, std::move(opt_indices.value()));
    } else {
      fail_ = true;
      return;
    }
  }

  void VisitExpr_(const CastNode* op) { VisitExpr(op->value); }

  bool Fail() { return fail_; }

  bool CompareBufferIndices(const Array<PrimExpr>& indices, const Array<Var>& other) {
    return std::equal(indices.begin(), indices.end(), other.begin(), other.end(),
                      [](const PrimExpr& a, const Var& b) { return a.same_as(b); });
  }

  Einsum ein_sum_;
  bool fail_{false};
};

Einsum ExtractEinsum(const ScheduleState& self, const Block& block) {
  EinsumExtractor extractor;
  std::optional<Einsum> einsum = extractor.Extract(block);
  if (!einsum.has_value()) {
    throw NonEinsumError(self->mod, block);
  }
  return einsum.value();
}

class BufferNotAllocatedInScopeError : public ScheduleError {
 public:
  explicit BufferNotAllocatedInScopeError(IRModule mod, Buffer buffer)
      : mod_(std::move(mod)), buffer_(std::move(buffer)) {}

  String FastErrorString() const final {
    return "ScheduleError: The buffer is not allocated as an intermediate buffer in current "
           "PrimFunc.";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The buffer " << buffer_->name
       << " is not allocated as an intermediate buffer in current PrimFunc.";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {}; }

 private:
  IRModule mod_;
  Buffer buffer_;
};

class PadEinsumRewriter : public ReplaceBufferMutator {
 public:
  PadEinsumRewriter(const std::unordered_map<const BlockNode*, PrimExpr> producer_predicate,
                    Map<Var, PrimExpr> padded_iter_extents, const Map<Buffer, Buffer>& buffer_remap,
                    Map<Block, Block>* block_sref_reuse, arith::Analyzer* analyzer)
      : ReplaceBufferMutator(buffer_remap, block_sref_reuse),
        producer_predicate_(producer_predicate),
        padded_iter_extents_(padded_iter_extents),
        analyzer_(analyzer) {}
  using ReplaceBufferMutator::VisitExpr_;
  using ReplaceBufferMutator::VisitStmt_;

  Stmt VisitStmt_(const ForNode* op) final {
    For new_for = Downcast<For>(ReplaceBufferMutator::VisitStmt_(op));
    if (padded_iter_extents_.count(new_for->loop_var)) {
      new_for.CopyOnWrite()->extent = padded_iter_extents_.at(new_for->loop_var);
    }
    return std::move(new_for);
  }

  Block PadProducerBlock(Block block, const PrimExpr& predicate) {
    BufferStore store = Downcast<BufferStore>(block->body);
    store.CopyOnWrite()->value =
        analyzer_->Simplify(if_then_else(predicate, store->value, make_zero(store->value.dtype())));
    block.CopyOnWrite()->body = std::move(store);
    return block;
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    Block old_block = GetRef<Block>(op);
    Block new_block = Downcast<Block>(ReplaceBufferMutator::VisitStmt_(op));
    if (auto it = producer_predicate_.find(op); it != producer_predicate_.end()) {
      new_block = PadProducerBlock(std::move(new_block), (*it).second);
    }

    // Mutate block iters
    Array<IterVar> new_iters;
    bool changed = false;
    for (const IterVar& iter : new_block->iter_vars) {
      if (auto it = padded_iter_extents_.find(iter->var); it != padded_iter_extents_.end()) {
        changed = true;
        new_iters.push_back(
            IterVar(Range::FromMinExtent(0, (*it).second), iter->var, iter->iter_type));
      } else {
        new_iters.push_back(iter);
      }
    }
    if (changed) {
      new_block.CopyOnWrite()->iter_vars = std::move(new_iters);
    }
    if (!old_block.same_as(new_block)) {
      block_sref_reuse_->Set(old_block, new_block);
    }
    return std::move(new_block);
  }

 private:
  const std::unordered_set<const BlockNode*> producer_blocks_;
  const std::unordered_map<const BlockNode*, PrimExpr> producer_predicate_;
  const Map<Var, PrimExpr> padded_iter_extents_;
  arith::Analyzer* analyzer_;
};

/*! \brief The schedule error class when the producer block cannot be padded. */
class InvalidProducerError : public ScheduleError {
 public:
  explicit InvalidProducerError(IRModule mod, Block producer)
      : mod_(std::move(mod)), producer_(std::move(producer)) {}

  String FastErrorString() const final {
    return "ScheduleError: The producer block cannot be padded.";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The producer block {0} cannot be padded. It should write to a single buffer and the "
          "body should be a BufferStore.";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {producer_}; }

 private:
  IRModule mod_;
  Buffer buffer_;
  Block producer_;
};

void PadEinsum(ScheduleState self, const StmtSRef& block_sref, const Array<Integer>& padding) {
  arith::Analyzer analyzer;
  // Step 1: Input checking and error handling
  const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
  BlockRealize realize = GetBlockRealize(self, block_sref);

  const StmtSRef& scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/true);
  InvalidPaddingError::Check(self, GetRef<Block>(block), padding);

  const Array<StmtSRef> producers = GetProducers(self, block_sref);
  {
    auto f_check_block_properties = [&](const StmtSRef& block_sref, bool is_producer) {
      CheckBlockHasTrivialBinding(self, block_sref);
      if (is_producer) {
        CheckCompleteBlock(self, block_sref, scope_sref);
      } else {
        CheckReductionBlock(self, block_sref, scope_sref);
      }
      Array loops = GetLoops(block_sref);
      ICHECK(!loops.empty());
      CheckGetSingleChildBlockRealizeOnSRefTree(self, loops.front());
    };

    // Check block properties of the computation block
    f_check_block_properties(block_sref, false);

    // Check block properties of the producer block
    for (const StmtSRef& producer_sref : producers) {
      f_check_block_properties(producer_sref, true);
    }
  }

  Einsum einsum = ExtractEinsum(self, GetRef<Block>(block));

  // Check input and output buffers are all allocated in the current scope.
  {
    auto f_check_buffer_allocated = [&](const Buffer& buffer) {
      auto [defining_site_sref, is_allocate] = GetBufferDefiningSite(block_sref, buffer);
      if (!defining_site_sref.defined() || !is_allocate) {
        throw BufferNotAllocatedInScopeError(self->mod, buffer);
      }
    };
    f_check_buffer_allocated(einsum.output_buffer);
    for (const auto& buffer_indices_pair : einsum.input_indices) {
      f_check_buffer_allocated(buffer_indices_pair.first);
    }
  }

  // Step 2: Prepare buffer and variable remapping. Infer the new shape of the input and the output
  // buffers. Infer the new extent of the block iters of the computation block and the producer
  // block.

  Map<Var, PrimExpr> padded_iter_extents;  // The new extents of both the block iters and loop vars

  // Convert the input padding array to a map from variables to the padded extents
  for (int i = 0, n = padding.size(); i < n; ++i) {
    const IterVar& iter = block->iter_vars[i];
    PrimExpr new_extent =
        IntImm(iter->var->dtype, Downcast<Integer>(iter->dom->extent)->value + padding[i]->value);
    padded_iter_extents.Set(iter->var, new_extent);
    padded_iter_extents.Set(Downcast<Var>(realize->iter_values[i]), new_extent);
  }

  Map<Buffer, Buffer> buffer_remap;  // mapping from buffers to new buffers with padded shapes

  // Utility function to pad a buffer with the new shape
  auto f_pad_buffer = [&padded_iter_extents](Buffer buffer, const Array<Var>& indices) -> Buffer {
    Array<PrimExpr> new_shape;
    for (const Var& index : indices) {
      new_shape.push_back(padded_iter_extents.at(index));
    }
    ICHECK_EQ(buffer->shape.size(), new_shape.size());
    buffer.CopyOnWrite()->shape = std::move(new_shape);
    return buffer;
  };

  buffer_remap.Set(einsum.output_buffer, f_pad_buffer(einsum.output_buffer, einsum.output_indices));

  std::unordered_map<const BlockNode*, PrimExpr> producer_predicate;

  // Different from the output block, the padding for the producer block is not directly specified
  // as the input argument. Instead, it is inferred from indices of the producer buffer accessed in
  // the output block.
  // We will find the indices (which are block iters) in BufferStore to the producer buffer
  // and infer the new extents of the block iters and the corresponding loop vars.
  for (const StmtSRef& producer_sref : producers) {
    const BlockNode* producer_block = TVM_SREF_TO_BLOCK(producer_sref);
    const BufferStoreNode* buffer_store = producer_block->body.as<BufferStoreNode>();
    Optional<Array<Var>> producer_store_indices;
    if (!buffer_store || producer_block->writes.size() != 1 ||
        !(producer_store_indices = CheckTrivialBufferIndices(buffer_store)).defined()) {
      throw InvalidProducerError(self->mod, GetRef<Block>(producer_block));
    }
    BlockRealize producer_realize = GetBlockRealize(self, producer_sref);

    const Buffer& old_buffer = producer_block->writes[0]->buffer;
    Buffer new_buffer = f_pad_buffer(old_buffer, einsum.input_indices.at(old_buffer));
    buffer_remap.Set(old_buffer, new_buffer);

    // The predicate to ensure the producer block is in the original bound before padding
    PrimExpr predicate = Bool(true);
    Map<Var, PrimExpr> indices_to_padded_extents;  // buffer indices to padded extents
    for (int i = 0, n = producer_store_indices.value().size(); i < n; ++i) {
      const Var& index = producer_store_indices.value()[i];
      PrimExpr padded_extent = new_buffer->shape[i];
      if (!analyzer.CanProveEqual(padded_extent, old_buffer->shape[i])) {
        predicate = predicate && (index < old_buffer->shape[i]);
      }
      indices_to_padded_extents.Set(index, padded_extent);
    }

    for (int i = 0, n = producer_block->iter_vars.size(); i < n; ++i) {
      const IterVar& iter = producer_block->iter_vars[i];
      if (auto it = indices_to_padded_extents.find(iter->var);
          it != indices_to_padded_extents.end()) {
        const PrimExpr& padded_extent = (*it).second;
        padded_iter_extents.Set(iter->var, padded_extent);
        padded_iter_extents.Set(Downcast<Var>(producer_realize->iter_values[i]), padded_extent);
      } else if (!is_one(iter->dom->extent)) {
        throw InvalidProducerError(self->mod, GetRef<Block>(producer_block));
      }
    }
    producer_predicate[producer_block] = predicate;
  }

  // Step 3: Mutate the AST subtree with the new buffers and the new block iter extents.
  Map<Block, Block> block_sref_reuse;
  PadEinsumRewriter rewriter(producer_predicate, padded_iter_extents, buffer_remap,
                             &block_sref_reuse, &analyzer);
  const BlockNode* scope_block = TVM_SREF_TO_BLOCK(scope_sref);
  Stmt new_scope_block = rewriter(GetRef<Block>(scope_block));

  // Step 4: Do the actual replacement.
  self->Replace(scope_sref, new_scope_block, block_sref_reuse);
}

/******** Instruction Registration ********/

struct PadEinsumTraits : public UnpackedInstTraits<PadEinsumTraits> {
  static constexpr const char* kName = "PadEinsum";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block, Array<Integer> padding) {
    sch->PadEinsum(block, padding);
  }

  static String UnpackedAsPython(Array<String> outputs, String block, Array<Integer> padding) {
    PythonAPICall py("pad_einsum");
    py.Input("block", block);
    py.Input("padding", padding);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(PadEinsumTraits);

}  // namespace tir
}  // namespace tvm
