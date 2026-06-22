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

/*!
 * \file lower_tirx_cleanup.cc
 * \brief Final cleanup stage for TIRx lowering.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/tirx_op.h>
#include <tvm/tirx/transform.h>

#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../arith/ir_mutator_with_analyzer.h"

namespace tvm {
namespace tirx {

class LayoutApplier : public arith::IRMutatorWithAnalyzer {
 public:
  static std::pair<Stmt, ffi::Map<Var, Buffer>> Flatten(
      const Stmt& stmt, const ffi::Map<tirx::Var, Buffer> buffer_map, const Target& target) {
    arith::Analyzer ana;
    LayoutApplier storage_lower(ana, target);
    std::unordered_map<Var, Buffer> new_buffer_map;
    std::vector<Buffer> param_flattened_buffers;
    for (const auto& kv : buffer_map) {
      if (kv.second->layout.defined()) {
        param_flattened_buffers.push_back(storage_lower.GetFlattenedBuffer(kv.second));
        Buffer buffer = kv.second;
        auto* writer = buffer.CopyOnWrite();
        writer->layout = std::nullopt;
        new_buffer_map[kv.first] = buffer;
      } else {
        new_buffer_map[kv.first] = kv.second;
      }
    }
    auto new_stmt = storage_lower(stmt);
    for (const auto& buf : param_flattened_buffers) {
      new_stmt = SeqStmt::Flatten(DeclBuffer(buf), std::move(new_stmt));
    }
    return std::make_pair(new_stmt, ffi::Map<Var, Buffer>(new_buffer_map));
  }

 protected:
  using IRMutatorWithAnalyzer::VisitExpr_;
  using IRMutatorWithAnalyzer::VisitStmt_;

  explicit LayoutApplier(const arith::Analyzer& analyzer, const Target& target)
      : arith::IRMutatorWithAnalyzer(analyzer), target_(target) {}

  ffi::Any VisitAny(const ffi::Any& any) {
    if (any == nullptr) {
      return any;
    }
    if (auto buffer = any.as<Buffer>()) {
      return GetFlattenedBuffer(buffer.value());
    } else if (auto prim_expr = any.as<PrimExpr>()) {
      return VisitExpr(prim_expr.value());
    } else if (auto stmt = any.as<Stmt>()) {
      return VisitStmt(stmt.value());
    }
    return any;
  }

  Stmt VisitStmt_(const AllocBufferNode* op) final {
    auto mutate = [this](Buffer buf) {
      if (target_->kind->name == "trn" && !buf->layout.defined()) {
        return buf;
      }
      return GetFlattenedBuffer(buf, /*is_alloc=*/true);
    };
    auto buffer = mutate(op->buffer);
    if (buffer.same_as(op->buffer)) {
      return ffi::GetRef<Stmt>(op);
    }
    auto n = CopyOnWrite(op);
    n->buffer = buffer;
    return Stmt(n);
  }

  Stmt VisitStmt_(const DeclBufferNode* op) final {
    auto buffer = GetFlattenedBuffer(op->buffer);
    if (buffer.same_as(op->buffer)) {
      return ffi::GetRef<Stmt>(op);
    }
    auto n = CopyOnWrite(op);
    n->buffer = buffer;
    return Stmt(n);
  }

  Buffer GetFlattenedBuffer(Buffer buf, bool is_alloc = false) {
    auto it = buffer_remap_.find(buf);
    if (it != buffer_remap_.end()) {
      return it->second;
    }
    auto trn_layout = buf->layout.as<TileLayoutNode>();
    Buffer flattened;
    tirx::BufferNode* writer;
    if (trn_layout && trn_layout->IsTrainium()) {
      ffi::Array<PrimExpr> new_shape =
          buf.scope() == "trn.psum" ? ffi::Array<PrimExpr>{trn_layout->GetSpan(ffi::String("Bank")),
                                                           trn_layout->GetSize(ffi::String("P")),
                                                           trn_layout->GetSpan(ffi::String("F"))}
                                    : ffi::Array<PrimExpr>{trn_layout->GetSize(ffi::String("P")),
                                                           trn_layout->GetSpan(ffi::String("F"))};
      flattened = buf;
      writer = flattened.CopyOnWrite();
      writer->shape = new_shape;
      writer->strides = {};
      writer->axis_separators = {};
    } else if (is_alloc) {
      if (auto tile_layout = buf->layout.as<TileLayoutNode>();
          tile_layout && tile_layout->HasThreadAxis()) {
        // Logical alloc_buffer with thread axes: physical shape = memory-axis span
        arith::Analyzer ana;
        PrimExpr mem_span = IntImm::Int32(1);
        for (const auto& iter : tile_layout->shard) {
          if (iter->axis->IsMemoryAxis()) {
            mem_span = mem_span + (iter->extent - 1) * iter->stride;
          }
        }
        for (const auto& iter : tile_layout->replica) {
          if (iter->axis->IsMemoryAxis()) {
            mem_span = mem_span + (iter->extent - 1) * iter->stride;
          }
        }
        for (const auto& [axis, off] : tile_layout->offset) {
          if (axis->IsMemoryAxis()) {
            mem_span = mem_span + off;
          }
        }
        flattened = buf;
        writer = flattened.CopyOnWrite();
        writer->shape = {ana->Simplify(mem_span)};
        writer->strides = {};
        writer->axis_separators = {};
      } else {
        flattened = buf.GetFlattenedBuffer();
        writer = flattened.CopyOnWrite();
      }
    } else {
      flattened = buf.GetFlattenedBuffer();
      writer = flattened.CopyOnWrite();
    }
    // TODO(Lunderberg): Move the handling of boolean into a
    // dedicated pass.
    if (flattened->dtype == DataType::Bool()) {
      writer->dtype = DataType::Int(8);
    }
    // canonicalize shape
    for (size_t i = 0; i < flattened->shape.size(); ++i) {
      writer->shape.Set(i, analyzer_->canonical_simplify(flattened->shape[i]));
    }
    writer->layout = std::nullopt;
    writer->elem_offset = StmtExprMutator::VisitExpr(buf->elem_offset);

    buffer_remap_[buf] = flattened;
    return flattened;
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore store = StmtExprMutator::VisitStmt_(op).as_or_throw<BufferStore>();
    bool store_returns_bool = (op->value.dtype() == DataType::Bool());
    store = VisitBufferAccess(store);

    // Handle casts from the value's dtype to the dtype of the
    // backing array.
    // TODO(Lunderberg): Move the handling of boolean into a
    // dedicated pass.
    if (store_returns_bool) {
      TVM_FFI_ICHECK_EQ(store->buffer->dtype, DataType::Int(8))
          << "Expected int8 backing array for boolean tensor";
      auto writer = store.CopyOnWrite();
      writer->value = tvm::cast(DataType::Int(8), store->value);
      return std::move(store);
    }
    return std::move(store);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    bool load_returns_bool = (op->dtype == DataType::Bool());
    BufferLoad load = StmtExprMutator::VisitExpr_(op).as_or_throw<BufferLoad>();
    load = VisitBufferAccess(load);
    // Handle casts from dtype of the backing array to value's dtype.
    // TODO(Lunderberg): Move the handling of boolean into a
    // dedicated pass.
    if (load_returns_bool) {
      TVM_FFI_ICHECK_EQ(load->buffer->dtype, DataType::Int(8))
          << "Expected int8 backing array for boolean tensor";
      load.CopyOnWrite()->dtype = DataType::Int(8);
      return tvm::cast(DataType::Bool(), load);
    } else {
      return std::move(load);
    }
  }

  Stmt VisitStmt_(const tirx::TilePrimitiveCallNode* op) final {
    ffi::Array<ffi::Any> args = op->args;
    args.MutateByApply([this](ffi::Any arg) -> ffi::Any { return VisitAny(arg); });
    if (args.same_as(op->args)) {
      return ffi::GetRef<Stmt>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->args = std::move(args);
      return Stmt(n);
    }
  }

  ffi::Array<PrimExpr> GetSimplifiedElemOffset(const Buffer& buffer,
                                               const ffi::Array<PrimExpr>& indices) {
    if (buffer->layout.defined()) {
      auto tile_layout = buffer->layout.value().as<TileLayoutNode>();
      if (tile_layout && tile_layout->IsTrainium()) {
        auto coord = buffer->layout.value()->Apply(indices, buffer->shape);
        std::vector<PrimExpr> res;
        for (const auto& axis : buffer.scope() == "trn.psum"
                                    ? ffi::Array<ffi::String>{"Bank", "P", "F"}
                                    : ffi::Array<ffi::String>{"P", "F"}) {
          auto it = coord.find(ffi::String(axis));
          if (it != coord.end()) {
            res.push_back(analyzer_->Simplify((*it).second));
          } else {
            res.push_back(0);
          }
        }
        return res;
      }
      if (auto tile = buffer->layout.value().as<TileLayoutNode>(); tile && tile->HasThreadAxis()) {
        LOG(FATAL) << "Cannot lower direct BufferLoad/BufferStore on a buffer with thread-axis "
                   << "layout: unable to verify that the coordinate matches the current thread. "
                   << "Use .view() + .local() to decompose thread and memory axes.";
      }
      auto res = buffer->layout.value()->Canonicalize()->Apply(indices, buffer->shape);
      TVM_FFI_ICHECK_EQ(res.size(), 1) << "Expected a single element offset";
      return {analyzer_->Simplify((*res.begin()).second)};
    }
    auto flattened_indices = buffer->ElemOffset(indices, true);
    TVM_FFI_ICHECK_EQ(flattened_indices.size(), 1) << "Expected a single element offset";
    return {analyzer_->Simplify(flattened_indices[0])};
  }

  template <typename Node>
  Node VisitBufferAccess(Node node) {
    TVM_FFI_ICHECK(node->buffer.defined());
    if (target_->kind->name == "trn" && !node->buffer->layout.defined()) {
      return node;
    }
    auto flattened_indices = GetSimplifiedElemOffset(node->buffer, node->indices);
    Buffer flattened_buffer = GetFlattenedBuffer(node->buffer);
    auto writer = node.CopyOnWrite();
    writer->buffer = flattened_buffer;
    writer->indices = flattened_indices;
    return node;
  }

  /*! \brief Map of buffers being remapped. */
  std::unordered_map<Buffer, Buffer, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> buffer_remap_;
  const Target& target_;
};

class BufferOffsetRemover : public StmtExprMutator {
 public:
  static Stmt Remove(const Stmt& stmt) { return BufferOffsetRemover()(stmt); }

 private:
  PrimExpr VisitExpr_(const tirx::CallNode* call) final {
    if (call->op.same_as(tirx::builtin::buffer_offset())) {
      auto buffer_load = call->args[0].as_or_throw<BufferLoad>();
      TVM_FFI_ICHECK_EQ(buffer_load->indices.size(), 1) << "Expected a single index";
      return buffer_load->indices[0];
    }
    return StmtExprMutator::VisitExpr_(call);
  }

  Stmt VisitStmt_(const DeclBufferNode* op) {
    auto buffer = op->buffer;
    auto elem_offset = this->VisitExpr(buffer->elem_offset);
    if (elem_offset.same_as(buffer->elem_offset)) {
      return StmtExprMutator::VisitStmt_(op);
    } else {
      auto n_buffer = buffer.CopyOnWrite();
      n_buffer->elem_offset = std::move(elem_offset);
      buffer_remap_[op->buffer] = buffer;
      auto n = CopyOnWrite(op);
      n->buffer = ffi::GetRef<Buffer>(n_buffer);
      return Stmt(n);
    }
  }

  using StmtExprMutator::VisitExpr_;
  using StmtExprMutator::VisitStmt_;

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore store = StmtExprMutator::VisitStmt_(op).as_or_throw<BufferStore>();
    store = VisitBufferAccess(store);
    return std::move(store);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    BufferLoad load = StmtExprMutator::VisitExpr_(op).as_or_throw<BufferLoad>();
    load = VisitBufferAccess(load);
    return std::move(load);
  }

  template <typename Node>
  Node VisitBufferAccess(Node node) {
    TVM_FFI_ICHECK(node->buffer.defined());
    auto it = buffer_remap_.find(node->buffer);
    if (it != buffer_remap_.end()) {
      auto writer = node.CopyOnWrite();
      writer->buffer = it->second;
      return node;
    }
    return node;
  }

  std::unordered_map<Buffer, Buffer, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> buffer_remap_;
};

namespace {
Target ResolveTarget(const PrimFunc& f) {
  auto target = f->GetAttr<Target>(tvm::attr::kTarget);
  if (!target.defined()) {
    target = Target::Current(false);
  }
  return target.value();
}
}  // namespace

namespace transform {

Pass LowerTIRxCleanup() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    Target target = ResolveTarget(f);
    auto* n = f.CopyOnWrite();
    std::tie(n->body, n->buffer_map) = LayoutApplier::Flatten(n->body, n->buffer_map, target);
    n->body = BufferOffsetRemover::Remove(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.LowerTIRxCleanup", {});
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
