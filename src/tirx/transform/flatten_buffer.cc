/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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

/*!
 * \file flatten_buffer.cc
 */

#include <tvm/arith/iter_affine_map.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/type.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_set>

#include "../../arith/ir_mutator_with_analyzer.h"
#include "ir_utils.h"

namespace tvm {
namespace tirx {

/*!
 * \brief Transform multi-dimension BufferLoad/BufferStore into device-supported dimension
 *        for the TIR not contains opaque block.
 */
class BufferFlattener : public arith::IRMutatorWithAnalyzer {
 public:
  static PrimFunc Flatten(PrimFunc func) {
    arith::Analyzer ana;
    auto pass = BufferFlattener(ana);
    pass.MarkBufferMapShapes(func);
    auto body = pass.VisitStmt(func->body);

    // The buffers in func->buffer_map are deliberately left
    // unflattened, as they are used for validation of user-provided
    // arguments.  The flattened buffers used in the updated
    // function body alias the argument buffers.
    for (size_t i = func->params.size(); i > 0; i--) {
      auto handle = func->params[i - 1];
      if (auto opt = func->buffer_map.Get(handle)) {
        auto old_buf = opt.value();
        if (pass.buffers_used_.count(old_buf)) {
          auto new_buf = pass.GetFlattenedBuffer(old_buf);
          if (!old_buf.same_as(new_buf)) {
            body = SeqStmt::Flatten(DeclBuffer(new_buf, old_buf.data()), std::move(body));
          }
        }
      }
    }

    if (!body.same_as(func->body)) {
      func.CopyOnWrite()->body = std::move(body);
    }
    return func;
  }

 private:
  using IRMutatorWithAnalyzer::VisitExpr;
  using IRMutatorWithAnalyzer::VisitExpr_;
  using IRMutatorWithAnalyzer::VisitStmt;
  using IRMutatorWithAnalyzer::VisitStmt_;

  explicit BufferFlattener(const arith::Analyzer& ana) : IRMutatorWithAnalyzer(ana) {}

  Stmt VisitStmt_(const SBlockNode* op) final {
    TVM_FFI_ICHECK_EQ(op->match_buffers.size(), 0)
        << "Unexpected MatchBufferRegion found during tirx.transform.FlattenBuffer.  "
        << "All MatchBufferRegion should be removed in tirx.transform.LowerMatchBuffer.";

    SBlock block = ffi::GetRef<SBlock>(op);

    ffi::Array<BufferVar> alloc_buffers = op->alloc_buffers;
    alloc_buffers.MutateByApply([this](BufferVar buf) { return GetFlattenedBuffer(buf); });
    if (!alloc_buffers.same_as(op->alloc_buffers)) {
      block.CopyOnWrite()->alloc_buffers = alloc_buffers;
    }

    ffi::Array<BufferRegion> reads = op->reads;
    reads.MutateByApply([this](BufferRegion region) { return MutateBufferRegion(region); });
    if (!reads.same_as(op->reads)) {
      block.CopyOnWrite()->reads = reads;
    }

    ffi::Array<BufferRegion> writes = op->writes;
    writes.MutateByApply([this](BufferRegion region) { return MutateBufferRegion(region); });
    if (!writes.same_as(op->writes)) {
      block.CopyOnWrite()->writes = writes;
    }

    return StmtExprMutator::VisitStmt_(block.get());
  }

  Stmt VisitStmt_(const AllocBufferNode* op) final {
    auto node = StmtExprMutator::VisitStmt_(op).as_or_throw<AllocBuffer>();

    auto new_buf = GetFlattenedBuffer(node->buffer);
    if (!node->buffer.same_as(new_buf)) {
      node.CopyOnWrite()->buffer = new_buf;
    }

    return std::move(node);
  }

  Stmt VisitStmt_(const DeclBufferNode* op) final {
    auto node = StmtExprMutator::VisitStmt_(op).as_or_throw<DeclBuffer>();

    auto new_buf = GetFlattenedBuffer(node->buffer);
    if (!node->buffer.same_as(new_buf)) {
      node.CopyOnWrite()->buffer = new_buf;
    }

    return std::move(node);
  }

  BufferVar GetFlattenedBuffer(BufferVar buf) {
    if (auto remapped = buffer_remap_.Get(buf)) {
      return remapped.value();
    }
    auto flattened = buf.GetFlattenedBuffer();
    ffi::ObjectPtr<BufferTypeNode> type = CopyBufferType(flattened);

    // canonicalize shape
    for (size_t i = 0; i < flattened->shape.size(); ++i) {
      type->shape.Set(i, analyzer_->canonical_simplify(flattened->shape[i]));
    }
    type->layout = std::nullopt;
    flattened = RebuildBufferVar(flattened, std::move(type));

    buffer_remap_.Set(buf, flattened);
    return flattened;
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferVar original_buffer = op->buffer;
    BufferStore store = StmtExprMutator::VisitStmt_(op).as_or_throw<BufferStore>();
    store = VisitBufferAccess(store, original_buffer);
    return store;
  }

  Expr VisitExpr_(const BufferLoadNode* op) final {
    BufferVar original_buffer = op->buffer;
    BufferLoad load = StmtExprMutator::VisitExpr_(op).as_or_throw<BufferLoad>();
    load = VisitBufferAccess(load, original_buffer);
    return load;
  }

  ffi::Array<PrimExpr> GetSimplifiedElemOffset(const BufferVar& buffer,
                                               const ffi::Array<PrimExpr>& indices) {
    auto flattened_indices = buffer->ElemOffset(indices);
    return this->IterMapSimplifyWithContext(flattened_indices, false);
  }

  template <typename Node>
  Node VisitBufferAccess(Node node, const BufferVar& original_buffer) {
    TVM_FFI_ICHECK(node->buffer.defined());
    buffers_used_.insert(original_buffer);
    auto flattened_indices = GetSimplifiedElemOffset(original_buffer, node->indices);
    BufferVar flattened_buffer = GetFlattenedBuffer(original_buffer);

    auto writer = node.CopyOnWrite();
    writer->buffer = flattened_buffer;
    writer->indices = flattened_indices;
    return node;
  }

  BufferRegion MutateBufferRegion(BufferRegion region) {
    BufferVar orig_buf = region->buffer;
    BufferVar flattened_buf = GetFlattenedBuffer(orig_buf);
    if (flattened_buf.same_as(orig_buf)) {
      return region;
    }

    ffi::Array<PrimExpr> min_values;
    ffi::Array<PrimExpr> max_values;
    for (const auto& range : region->region) {
      min_values.push_back(range->min);
      max_values.push_back(range->min + range->extent - 1);
    }

    ffi::Array<PrimExpr> flattened_min = GetSimplifiedElemOffset(orig_buf, min_values);
    ffi::Array<PrimExpr> flattened_max = GetSimplifiedElemOffset(orig_buf, max_values);

    ffi::Array<Range> flattened_ranges;
    TVM_FFI_ICHECK_EQ(flattened_min.size(), flattened_max.size());
    for (size_t i = 0; i < flattened_min.size(); i++) {
      flattened_ranges.push_back(Range(flattened_min[i], flattened_max[i] + 1));
    }

    return BufferRegion(flattened_buf, flattened_ranges);
  }

  /*! \brief Set of buffers accessed during visitation (used to emit DeclBuffer for param buffers).
   */
  std::unordered_set<BufferVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> buffers_used_;

  /*! \brief The updated external buffer map. */
  ffi::Map<Var, BufferVar> updated_extern_buffer_map_;
};

PrimFunc FlattenBuffer(PrimFunc f) { return BufferFlattener::Flatten(f); }

namespace transform {

Pass FlattenBuffer() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return FlattenBuffer(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.FlattenBuffer", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.FlattenBuffer", FlattenBuffer);
}
}  // namespace transform

}  // namespace tirx
}  // namespace tvm
