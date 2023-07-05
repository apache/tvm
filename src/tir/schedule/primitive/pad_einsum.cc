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

#include <tvm/tir/op.h>

#include "../utils.h"

namespace tvm {
namespace tir {

/*!
 * \brief Check if buffer indices are all Vars and expr
 * \param buffer_access The BufferLoad or BufferStore
 * \return The indices if the indices are all Vars, otherwise NullOpt
 */
Optional<Array<Var>> CheckTrivialBufferIndices(const Array<PrimExpr>& buffer_access) {
  Array<Var> indices;
  for (const PrimExpr& index : buffer_access) {
    if (index->IsInstance<IntImmNode>()) {
      continue;
    }
    const VarNode* var = index.as<VarNode>();
    if (var == nullptr) {
      return NullOpt;
    }
    indices.push_back(GetRef<Var>(var));
  }
  return indices;
}

Optional<Array<Var>> CheckTrivialBufferAccess(const BufferRegion& buffer_region) {
  Array<Var> indices;
  indices.reserve(buffer_region->region.size());
  for (const Range& range : buffer_region->region) {
    if (!tir::is_one(range->extent)) {
      return NullOpt;
    }
    if (range->min->IsInstance<IntImmNode>()) {
      continue;
    }
    if (const auto* var = range->min.as<VarNode>()) {
      indices.push_back(GetRef<Var>(var));
    } else {
      return NullOpt;
    }
  }
  return indices;
}

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
       << block_->iter_vars.size() << " positive integers. Got " << padding_;
    return os.str();
  }

  static void Check(const ScheduleState& self, const Block& block, Array<Integer> padding) {
    if (padding.size() != block->iter_vars.size()) {
      throw InvalidPaddingError(self->mod, block, padding);
    }
    for (const auto& pad : padding) {
      if (pad->value <= 0) {
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
  Array<Buffer> output_buffers;
  // The indices of the output buffer
  Map<Buffer, Array<Var>> output_indices;
  // The input buffers
  Array<Buffer> input_buffers;
  // The indices of the input buffers
  Map<Buffer, Array<Var>> input_indices;
};

struct BufferPadding {
  Buffer buffer;
  Buffer padded_buffer;

  static BufferPadding FromBufferRegion(const BufferRegion& buffer_region,
                                        const Map<Var, PrimExpr>& iter_extents) {
    BufferPadding result;
    result.buffer = buffer_region->buffer;
    Array<PrimExpr> shape;
    shape.reserve(buffer_region->region.size());
    int ndim = buffer_region->region.size();
    for (int i = 0; i < ndim; ++i) {
      PrimExpr pos = buffer_region->region[i]->min;
      ICHECK(pos->IsInstance<IntImmNode>() || pos->IsInstance<VarNode>());
      if (pos->IsInstance<IntImmNode>()) {
        shape.push_back(IntImm(pos->dtype, 1));
      } else if (Optional<PrimExpr> extent = iter_extents.Get(Downcast<Var>(pos))) {
        shape.push_back(extent.value());
      } else {
        shape.push_back(buffer_region->buffer->shape[i]);
      }
    }
    result.padded_buffer = decl_buffer(shape, result.buffer->dtype, result.buffer->name + "_pad",
                                       result.buffer.scope());
    return result;
  }

  Stmt MakeCopyBlock(bool is_read, Array<Block>* blocks, arith::Analyzer* analyzer) {
    Array<Var> loop_vars;
    Array<Range> loop_doms;
    Array<IterVar> iter_vars;
    Array<Range> instance_dom;
    Array<PrimExpr> indices;
    int ndim = buffer->shape.size();
    for (int i = 0; i < ndim; ++i) {
      PrimExpr dim{nullptr};
      if (is_read) {
        dim = padded_buffer->shape[i];
      } else {
        dim = buffer->shape[i];
      }
      Range dom = Range::FromMinExtent(IntImm(dim->dtype, 0), dim);
      loop_vars.push_back(Var("i" + std::to_string(i), dim->dtype));
      loop_doms.push_back(dom);
      IterVar iter_var(dom, Var("v" + std::to_string(i), dim->dtype), kDataPar);
      instance_dom.push_back(Range::FromMinExtent(iter_var->var, IntImm(dim->dtype, 1)));
      iter_vars.push_back(iter_var);
      indices.push_back(iter_var->var);
    }
    Stmt body{nullptr};
    if (is_read) {
      PrimExpr predicate = Bool(true);
      for (int i = 0; i < ndim; ++i) {
        if (!analyzer->CanProveEqual(buffer->shape[i], padded_buffer->shape[i])) {
          predicate = predicate && (indices[i] < buffer->shape[i]);
        }
      }
      PrimExpr rhs = BufferLoad(buffer, indices);
      body =
          BufferStore(padded_buffer, if_then_else(predicate, rhs, make_zero(rhs->dtype)), indices);
    } else {
      body = BufferStore(buffer, BufferLoad(padded_buffer, indices), indices);
    }
    BufferRegion read_region(buffer, instance_dom);
    BufferRegion write_region(padded_buffer, instance_dom);
    if (!is_read) {
      std::swap(read_region, write_region);
    }
    Block new_block(iter_vars, {read_region}, {write_region}, padded_buffer->name, std::move(body));
    blocks->push_back(new_block);
    body = BlockRealize(Array<PrimExpr>{loop_vars.begin(), loop_vars.end()}, Bool(true), new_block);
    for (int i = ndim - 1; i >= 0; --i) {
      body = For(loop_vars[i], loop_doms[i]->min, loop_doms[i]->extent, ForKind::kSerial,
                 std::move(body));
    }
    return body;
  }
};

Einsum ExtractEinsum(const ScheduleState& self, const Block& block) {
  Einsum result;
  std::unordered_set<const BufferNode*> buffer_used;
  int n_reads = block->reads.size();
  for (int i = 0; i < n_reads; ++i) {
    const Buffer& buffer = block->reads[i]->buffer;
    if (buffer_used.count(buffer.get()) != 0) {
      throw NonEinsumError(self->mod, block);
    }
    buffer_used.insert(buffer.get());
    if (Optional<Array<Var>> opt_indices = CheckTrivialBufferAccess(block->reads[i])) {
      result.input_buffers.push_back(buffer);
      result.input_indices.Set(buffer, opt_indices.value());
    } else {
      throw NonEinsumError(self->mod, block);
    }
  }
  int n_writes = block->writes.size();
  for (int i = 0; i < n_writes; ++i) {
    const Buffer& buffer = block->writes[i]->buffer;
    if (buffer_used.count(buffer.get()) != 0) {
      throw NonEinsumError(self->mod, block);
    }
    buffer_used.insert(buffer.get());
    if (Optional<Array<Var>> opt_indices = CheckTrivialBufferAccess(block->writes[i])) {
      result.output_buffers.push_back(buffer);
      result.output_indices.Set(buffer, opt_indices.value());
    } else {
      throw NonEinsumError(self->mod, block);
    }
  }
  return result;
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

class PadEinsumBufferReplacer : public StmtExprMutator {
 public:
  Stmt VisitStmt_(const BlockNode* old_block_ptr) final {
    Block old_block = GetRef<Block>(old_block_ptr);
    Block block = Downcast<Block>(StmtMutator::VisitStmt_(old_block_ptr));
    Array<IterVar> iter_vars;
    iter_vars.reserve(block->iter_vars.size());
    for (const IterVar& iter_var : block->iter_vars) {
      if (Optional<PrimExpr> new_dom = iter2padded_extents.Get(iter_var->var)) {
        ObjectPtr<IterVarNode> new_iter_var = make_object<IterVarNode>(*iter_var.get());
        new_iter_var->dom = Range::FromMinExtent(iter_var->dom->min, new_dom.value());
        iter_vars.push_back(IterVar(new_iter_var));
      } else {
        iter_vars.push_back(iter_var);
      }
    }
    Array<BufferRegion> reads;
    reads.reserve(block->reads.size());
    for (const BufferRegion& read : block->reads) {
      if (Optional<Buffer> buffer = buffer_map_.Get(read->buffer)) {
        reads.push_back(BufferRegion(buffer.value(), read->region));
      } else {
        reads.push_back(read);
      }
    }
    Array<BufferRegion> writes;
    writes.reserve(block->writes.size());
    for (const BufferRegion& write : block->writes) {
      if (Optional<Buffer> buffer = buffer_map_.Get(write->buffer)) {
        writes.push_back(BufferRegion(buffer.value(), write->region));
      } else {
        writes.push_back(write);
      }
    }
    Block new_block =
        Block(iter_vars, reads, writes, block->name_hint, block->body, block->init,
              /*alloc_buffers=*/{}, /*match_buffers=*/{}, /*annotations=*/block->annotations);
    block_sref_reuse_.Set(old_block, new_block);
    return new_block;
  }

  Stmt VisitStmt_(const ForNode* old_for_ptr) final {
    For old_for = GetRef<For>(old_for_ptr);
    For new_for = Downcast<For>(StmtMutator::VisitStmt_(old_for_ptr));
    if (Optional<PrimExpr> new_extent = loop_var2padded_extent.Get(new_for->loop_var)) {
      ObjectPtr<ForNode> new_for_ptr = make_object<ForNode>(*new_for.get());
      new_for_ptr->extent = new_extent.value();
      new_for = For(new_for_ptr);
    }
    return new_for;
  }

  Stmt VisitStmt_(const BufferStoreNode* old_store_ptr) final {
    BufferStore store = Downcast<BufferStore>(StmtMutator::VisitStmt_(old_store_ptr));
    if (Optional<Buffer> buffer = buffer_map_.Get(store->buffer)) {
      return BufferStore(buffer.value(), store->value, store->indices);
    } else {
      return store;
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* old_load_ptr) final {
    BufferLoad load = Downcast<BufferLoad>(ExprMutator::VisitExpr_(old_load_ptr));
    if (Optional<Buffer> buffer = buffer_map_.Get(load->buffer)) {
      return BufferLoad(buffer.value(), load->indices);
    } else {
      return load;
    }
  }

  Map<Var, PrimExpr> iter2padded_extents;
  Map<Var, PrimExpr> loop_var2padded_extent;
  Map<Buffer, Buffer> buffer_map_;
  Map<Block, Block> block_sref_reuse_;
};

void PadEinsum(ScheduleState self, const StmtSRef& block_sref, const Array<Integer>& padding) {
  arith::Analyzer analyzer;
  // Step 1: Input checking and error handling
  const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
  BlockRealize realize = GetBlockRealize(self, block_sref);
  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/true);
  const BlockNode* scope_block = TVM_SREF_TO_BLOCK(scope_sref);
  InvalidPaddingError::Check(self, GetRef<Block>(block), padding);
  // Step 2. Extract the Einsum pattern
  ExtractEinsum(self, GetRef<Block>(block));
  // Step 3. Figure out the padding needed
  PadEinsumBufferReplacer replacer;
  for (int i = 0, n = padding.size(); i < n; ++i) {
    const IterVar& iter = block->iter_vars[i];
    PrimExpr dom = iter->dom->extent;
    PrimExpr new_dom = analyzer.Simplify(ceildiv(dom, padding[i]) * padding[i]);
    if (!analyzer.CanProveEqual(new_dom, dom)) {
      replacer.iter2padded_extents.Set(iter->var, new_dom);
      if (const auto* loop_var = realize->iter_values[i].as<VarNode>()) {
        replacer.iter2padded_extents.Set(GetRef<Var>(loop_var), new_dom);
        replacer.loop_var2padded_extent.Set(GetRef<Var>(loop_var), new_dom);
      }
    }
  }
  auto f_needs_padding = [&replacer](const Array<Range>& region) {
    for (const Range& range : region) {
      if (const auto* var = range->min.as<VarNode>()) {
        if (replacer.iter2padded_extents.count(GetRef<Var>(var))) {
          return true;
        }
      }
    }
    return false;
  };
  // Step 3. Convert the subtree under the scope root
  Array<Stmt> scope_body;
  if (const auto* seq_stmt = scope_block->body.as<SeqStmtNode>()) {
    scope_body = seq_stmt->seq;
  } else {
    scope_body.push_back(scope_block->body);
  }
  // Step 4. Find out the block of our interest
  int pos = -1;
  for (int i = 0; i < static_cast<int>(scope_body.size()); ++i) {
    bool found = false;
    PostOrderVisit(scope_body[i], [&found, &block](const ObjectRef& node) {
      if (node.get() == block) {
        found = true;
      }
    });
    if (found) {
      pos = i;
      break;
    }
  }
  ICHECK_NE(pos, -1);
  // Step 5. For each buffer, if it needs padding, create a new buffer and a new block
  Array<Stmt> read_blocks;
  Array<Stmt> write_blocks;
  Array<Block> new_copy_blocks;
  Array<Buffer> alloc_buffers;
  for (const BufferRegion& buffer_region : block->reads) {
    if (f_needs_padding(buffer_region->region)) {
      BufferPadding bp =
          BufferPadding::FromBufferRegion(buffer_region, replacer.iter2padded_extents);
      replacer.buffer_map_.Set(bp.buffer, bp.padded_buffer);
      read_blocks.push_back(bp.MakeCopyBlock(true, &new_copy_blocks, &analyzer));
      alloc_buffers.push_back(bp.padded_buffer);
    }
  }
  for (const BufferRegion& buffer_region : block->writes) {
    if (f_needs_padding(buffer_region->region)) {
      BufferPadding bp =
          BufferPadding::FromBufferRegion(buffer_region, replacer.iter2padded_extents);
      replacer.buffer_map_.Set(bp.buffer, bp.padded_buffer);
      write_blocks.push_back(bp.MakeCopyBlock(false, &new_copy_blocks, &analyzer));
      alloc_buffers.push_back(bp.padded_buffer);
    }
  }
  // Step 6. Create new scope body
  Array<Stmt> new_scope_body;
  for (int i = 0; i < static_cast<int>(scope_body.size()); ++i) {
    if (i != pos) {
      new_scope_body.push_back(scope_body[i]);
      continue;
    }
    new_scope_body.insert(new_scope_body.end(), read_blocks.begin(), read_blocks.end());
    new_scope_body.push_back(replacer(scope_body[i]));
    new_scope_body.insert(new_scope_body.end(), write_blocks.begin(), write_blocks.end());
  }
  // Step 7. Create new scope
  Block new_scope_block{nullptr};
  {
    ObjectPtr<BlockNode> n = make_object<BlockNode>(*scope_block);
    n->body = SeqStmt::Flatten(new_scope_body);
    n->alloc_buffers.insert(n->alloc_buffers.end(), alloc_buffers.begin(), alloc_buffers.end());
    new_scope_block = Block(n);
  }
  replacer.block_sref_reuse_.Set(GetRef<Block>(scope_block), new_scope_block);
  // Step 8. Do replacement and update flags
  self->Replace(scope_sref, new_scope_block, replacer.block_sref_reuse_);
  for (const Block& block : new_copy_blocks) {
    StmtSRef block_sref = self->stmt2ref.at(block.get());
    BlockInfo& block_info = self->block_info[block_sref];
    block_info.affine_binding = true;
    block_info.region_cover = true;
    block_info.stage_pipeline = true;
  }
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
