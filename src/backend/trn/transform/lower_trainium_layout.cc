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
 * \file lower_trainium_layout.cc
 * \brief Trainium-specific TIRx layout lowering.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/tirx_op.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../../arith/ir_mutator_with_analyzer.h"

namespace tvm {
namespace tirx {

static bool IsTrainiumLayout(const TileLayoutNode* layout) {
  if (layout == nullptr) {
    return false;
  }
  return !std::any_of(layout->shard.begin(), layout->shard.end(), [](const Iter& iter) {
    return iter->axis->IsMemoryAxis() && !iter->axis.same_as(Axis::Get("F")) &&
           !iter->axis.same_as(Axis::Get("P")) && !iter->axis.same_as(Axis::Get("Bank"));
  });
}

class TrainiumLayoutApplier : public arith::IRMutatorWithAnalyzer {
 public:
  static std::pair<Stmt, ffi::Map<Var, BufferVar>> Lower(
      const Stmt& stmt, const ffi::Map<tirx::Var, BufferVar> buffer_map) {
    arith::Analyzer ana;
    TrainiumLayoutApplier storage_lower(ana);
    std::unordered_map<Var, BufferVar> new_buffer_map;
    std::vector<std::pair<BufferVar, BufferVar>> param_flattened_buffers;
    for (const auto& kv : buffer_map) {
      if (kv.second->layout.has_value()) {
        BufferVar flattened = storage_lower.GetFlattenedBuffer(kv.second);
        auto type = CopyBufferType(kv.second);
        type->layout = std::nullopt;
        BufferVar buffer = RebuildBufferVar(kv.second, std::move(type));
        param_flattened_buffers.emplace_back(flattened, buffer);
        new_buffer_map[kv.first] = buffer;
      } else {
        new_buffer_map[kv.first] = kv.second;
      }
    }
    auto new_stmt = storage_lower(stmt);
    for (const auto& [buf, source] : param_flattened_buffers) {
      new_stmt = SeqStmt::Flatten(DeclBuffer(buf, source.data()), std::move(new_stmt));
    }
    return std::make_pair(new_stmt, ffi::Map<Var, BufferVar>(new_buffer_map));
  }

 protected:
  using IRMutatorWithAnalyzer::VisitExpr_;
  using IRMutatorWithAnalyzer::VisitStmt_;

  explicit TrainiumLayoutApplier(const arith::Analyzer& analyzer)
      : arith::IRMutatorWithAnalyzer(analyzer) {}

  ffi::Any VisitAny(const ffi::Any& any) {
    if (any == nullptr) {
      return any;
    }
    if (auto buffer = any.as<BufferVar>()) {
      return GetFlattenedBuffer(buffer.value());
    } else if (auto prim_expr = any.as<PrimExpr>()) {
      return VisitPrimExpr(prim_expr.value());
    } else if (auto stmt = any.as<Stmt>()) {
      return VisitStmt(stmt.value());
    }
    return any;
  }

  Stmt VisitStmt_(const AllocBufferNode* op) final {
    if (!op->buffer->layout.has_value()) {
      return ffi::GetRef<Stmt>(op);
    }
    auto buffer = GetFlattenedBuffer(op->buffer, /*is_alloc=*/true);
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

  BufferVar GetFlattenedBuffer(BufferVar buf, bool is_alloc = false) {
    auto it = buffer_remap_.find(buf);
    if (it != buffer_remap_.end()) {
      return it->second;
    }
    auto trn_layout = buf->layout.as<TileLayoutNode>();
    BufferVar flattened;
    ffi::ObjectPtr<BufferTypeNode> type;
    if (IsTrainiumLayout(trn_layout)) {
      ffi::Array<PrimExpr> new_shape =
          buf.scope() == "trn.psum" ? ffi::Array<PrimExpr>{trn_layout->GetSpan(ffi::String("Bank")),
                                                           trn_layout->GetSize(ffi::String("P")),
                                                           trn_layout->GetSpan(ffi::String("F"))}
                                    : ffi::Array<PrimExpr>{trn_layout->GetSize(ffi::String("P")),
                                                           trn_layout->GetSpan(ffi::String("F"))};
      flattened = buf;
      type = CopyBufferType(flattened);
      type->shape = new_shape;
      type->strides = {};
    } else if (is_alloc) {
      if (auto tile_layout = buf->layout.as<TileLayoutNode>();
          tile_layout && tile_layout->HasThreadAxis()) {
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
        type = CopyBufferType(flattened);
        type->shape = {ana->Simplify(mem_span)};
        type->strides = {};
      } else {
        flattened = buf.GetFlattenedBuffer();
        type = CopyBufferType(flattened);
      }
    } else {
      flattened = buf.GetFlattenedBuffer();
      type = CopyBufferType(flattened);
    }
    if (flattened->dtype->dtype == DLDataType{kDLBool, 8, 1}) {
      type->dtype = PrimType::Int(8);
      type->data_pointer_type = PointerType(PrimType::Int(8), flattened.scope());
    }
    for (size_t i = 0; i < flattened->shape.size(); ++i) {
      type->shape.Set(i, analyzer_->canonical_simplify(flattened->shape[i]));
    }
    type->layout = std::nullopt;
    type->elem_offset = StmtExprMutator::VisitPrimExpr(buf->elem_offset);
    flattened = RebuildBufferVar(flattened, std::move(type));

    buffer_remap_[buf] = flattened;
    return flattened;
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore store = StmtExprMutator::VisitStmt_(op).as_or_throw<BufferStore>();
    PrimType store_value_ty = op->value.ty();
    bool store_returns_bool = store_value_ty.MatchesCode(DLDataTypeCode::kDLBool);
    store = VisitBufferAccess(store);

    if (store_returns_bool) {
      TVM_FFI_ICHECK_EQ(store->buffer->dtype->dtype, (DLDataType{kDLInt, 8, 1}))
          << "Expected int8 backing array for boolean tensor";
      auto writer = store.CopyOnWrite();
      writer->value = tvm::cast(PrimType::Int(8), store->value);
      return std::move(store);
    }
    return std::move(store);
  }

  Expr VisitExpr_(const BufferLoadNode* op) final {
    PrimType load_ty = op->ty.as_or_throw<PrimType>();
    bool load_returns_bool = load_ty.MatchesCode(DLDataTypeCode::kDLBool);
    BufferLoad load = StmtExprMutator::VisitExpr_(op).as_or_throw<BufferLoad>();
    load = VisitBufferAccess(load);
    if (load_returns_bool) {
      TVM_FFI_ICHECK_EQ(load->buffer->dtype->dtype, (DLDataType{kDLInt, 8, 1}))
          << "Expected int8 backing array for boolean tensor";
      load.CopyOnWrite()->ExprNode::ty = PrimType::Int(8);
      return tvm::cast(PrimType::Bool(), load);
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

  ffi::Array<PrimExpr> GetSimplifiedElemOffset(const BufferVar& buffer,
                                               const ffi::Array<PrimExpr>& indices) {
    if (buffer->layout.has_value()) {
      auto tile_layout = buffer->layout.value().as<TileLayoutNode>();
      if (IsTrainiumLayout(tile_layout)) {
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
      if (tile_layout && tile_layout->HasThreadAxis()) {
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
    if (!node->buffer->layout.has_value()) {
      return node;
    }
    auto flattened_indices = GetSimplifiedElemOffset(node->buffer, node->indices);
    BufferVar flattened_buffer = GetFlattenedBuffer(node->buffer);
    auto writer = node.CopyOnWrite();
    writer->buffer = flattened_buffer;
    writer->indices = flattened_indices;
    return node;
  }

  std::unordered_map<BufferVar, BufferVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> buffer_remap_;
};

class TrainiumBufferOffsetRemover : public StmtExprMutator {
 public:
  static Stmt Remove(const Stmt& stmt) { return TrainiumBufferOffsetRemover()(stmt); }

 private:
  Expr VisitExpr_(const CallNode* call) final {
    if (call->op.same_as(tirx::builtin::buffer_offset())) {
      auto buffer_load = call->args[0].as_or_throw<BufferLoad>();
      TVM_FFI_ICHECK_EQ(buffer_load->indices.size(), 1) << "Expected a single index";
      return buffer_load->indices[0];
    }
    return StmtExprMutator::VisitExpr_(call);
  }

  Stmt VisitStmt_(const DeclBufferNode* op) {
    auto buffer = op->buffer;
    auto elem_offset = this->VisitPrimExpr(buffer->elem_offset);
    if (elem_offset.same_as(buffer->elem_offset)) {
      return StmtExprMutator::VisitStmt_(op);
    } else {
      auto type = CopyBufferType(buffer);
      type->elem_offset = std::move(elem_offset);
      buffer = RebuildBufferVar(buffer, std::move(type));
      buffer_remap_[op->buffer] = buffer;
      auto n = CopyOnWrite(op);
      n->buffer = buffer;
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

  Expr VisitExpr_(const BufferLoadNode* op) final {
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

  std::unordered_map<BufferVar, BufferVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> buffer_remap_;
};

namespace transform {

Pass LowerTrainiumLayout() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    std::tie(n->body, n->buffer_map) = TrainiumLayoutApplier::Lower(n->body, n->buffer_map);
    n->body = TrainiumBufferOffsetRemover::Remove(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.backend.trn.LowerTrainiumLayout", {});
}

void RegisterTRNTransforms() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.backend.trn.transform.LowerTrainiumLayout", LowerTrainiumLayout);
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
