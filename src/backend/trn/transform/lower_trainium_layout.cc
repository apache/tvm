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

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/runtime/logging.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/tile_primitive.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <tuple>
#include <utility>
#include <vector>

#include "../../../tirx/ir/ir_mutator_with_analyzer.h"

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

class TrainiumLayoutApplier : public tirx::IRMutatorWithAnalyzer {
 public:
  static std::pair<Stmt, ffi::Array<Var>> Lower(const Stmt& stmt, const ffi::Array<Var>& params) {
    sym::Analyzer ana;
    auto storage_lower = ffi::make_object<TrainiumLayoutApplier>(ana);
    ffi::Array<Var> new_params;
    new_params.reserve(params.size());
    std::vector<std::pair<BufferVar, BufferVar>> param_flattened_buffers;
    for (const Var& param : params) {
      auto buffer = param.as<BufferVar>();
      if (!buffer) {
        new_params.push_back(param);
        continue;
      }
      if (buffer.value()->layout.has_value()) {
        BufferVar flattened = storage_lower->GetFlattenedBuffer(buffer.value());
        auto type = CopyBufferType(buffer.value());
        type->layout = std::nullopt;
        BufferVar source = RebuildBufferVar(buffer.value(), std::move(type));
        param_flattened_buffers.emplace_back(flattened, source);
        new_params.push_back(source.var());
      } else {
        new_params.push_back(buffer.value().var());
      }
    }
    auto new_stmt = storage_lower->Mutate(stmt, InplaceMode::kDisallow).ValueOrUnchanged(stmt);
    for (const auto& [buf, source] : param_flattened_buffers) {
      new_stmt = SeqStmt::Flatten(DeclBuffer(buf, source.data()), std::move(new_stmt));
    }
    return std::make_pair(new_stmt, new_params);
  }

  explicit TrainiumLayoutApplier(const sym::Analyzer& analyzer)
      : tirx::IRMutatorWithAnalyzer(analyzer) {}

 protected:
  using IRMutatorWithAnalyzer::Mutate_;

  ffi::Any MutateTileArgument(const ffi::Any& any) {
    if (auto buffer = any.as<BufferVar>()) {
      return GetFlattenedBuffer(buffer.value());
    }
    if (auto expr = any.as<PrimExpr>()) {
      return Mutate(expr.value()).ValueOrUnchanged(expr.value());
    }
    if (auto stmt = any.as<Stmt>()) {
      return Mutate(stmt.value()).ValueOrUnchanged(stmt.value());
    }
    return any;
  }

  UnchangedOr<Stmt> Mutate_(const AllocBufferNode* op, InplaceMode inplace_mode) final {
    if (!op->buffer->layout.has_value()) {
      return ffi::Unchanged();
    }
    auto buffer = GetFlattenedBuffer(op->buffer, /*is_alloc=*/true);
    if (buffer.same_as(op->buffer)) {
      return ffi::Unchanged();
    }
    if (inplace_mode == InplaceMode::kAllow) {
      const_cast<AllocBufferNode*>(op)->buffer = std::move(buffer);
      return ffi::Unchanged();
    }
    auto n = ffi::make_object<AllocBufferNode>(*op);
    n->buffer = std::move(buffer);
    return Stmt(n);
  }

  UnchangedOr<Stmt> Mutate_(const DeclBufferNode* op, InplaceMode inplace_mode) final {
    auto data_update = Mutate(op->data, inplace_mode);
    bool data_unchanged = data_update.UnchangedOrSameAs(op->data);
    Expr data = std::move(data_update).ValueOrUnchanged(op->data);
    auto buffer = GetFlattenedBuffer(op->buffer);
    if (buffer.same_as(op->buffer) && data_unchanged) {
      return ffi::Unchanged();
    }
    return DeclBuffer(buffer, std::move(data), op->span);
  }

  BufferVar GetFlattenedBuffer(BufferVar buf, bool is_alloc = false) {
    ffi::Any mapped = VarRemapGet(buf);
    if (mapped.type_index() != ffi::TypeIndex::kTVMFFINone) {
      return std::move(mapped).as_or_throw<UnchangedOr<BufferVar>>().ValueOrUnchanged(buf);
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
        sym::Analyzer ana;
        PrimExpr mem_span = prim::IntImm::Int32(1);
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
    }
    for (size_t i = 0; i < flattened->shape.size(); ++i) {
      type->shape.Set(i, analyzer_->canonical_simplify(flattened->shape[i]));
    }
    type->layout = std::nullopt;
    type->elem_offset = StmtExprMutator::Mutate(buf->elem_offset, InplaceMode::kDisallow)
                            .ValueOrUnchanged(buf->elem_offset);
    flattened = RebuildBufferVar(flattened, std::move(type));

    VarRemapSet(buf, flattened);
    return flattened;
  }

  UnchangedOr<Stmt> Mutate_(const BufferStoreNode* op, InplaceMode inplace_mode) final {
    // Index conversion needs the original logical layout after the parent remaps the buffer.
    BufferVar logical_buffer = op->buffer;
    BufferStore store = StmtExprMutator::Mutate_(op, inplace_mode)
                            .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                            .as_or_throw<BufferStore>();
    PrimType store_value_ty = op->value.ty();
    bool store_returns_bool = store_value_ty.MatchesCode(DLDataTypeCode::kDLBool);
    store = VisitBufferAccess(store, logical_buffer);

    if (store_returns_bool) {
      TVM_FFI_ICHECK_EQ(store->buffer->dtype->dtype, (DLDataType{kDLInt, 8, 1}))
          << "Expected int8 backing array for boolean tensor";
      auto writer = store.CopyOnWrite();
      writer->value = tvm::prim::cast(PrimType::Int(8), store->value);
      return std::move(store);
    }
    return std::move(store);
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* op, InplaceMode inplace_mode) final {
    BufferVar logical_buffer = op->source.as_or_throw<BufferVar>();
    PrimType load_ty = op->ty.as_or_throw<PrimType>();
    bool load_returns_bool = load_ty.MatchesCode(DLDataTypeCode::kDLBool);
    TensorLoad load = StmtExprMutator::Mutate_(op, inplace_mode)
                          .ValueOrUnchanged(ffi::GetRef<PrimExpr>(op))
                          .as_or_throw<TensorLoad>();
    load = VisitBufferAccess(load, logical_buffer);
    if (load_returns_bool) {
      TVM_FFI_ICHECK_EQ(load->source.as_or_throw<tvm::tirx::BufferVar>()->dtype->dtype,
                        (DLDataType{kDLInt, 8, 1}))
          << "Expected int8 backing array for boolean tensor";
      load.CopyOnWrite()->ExprNode::ty = PrimType::Int(8);
      return tvm::prim::cast(PrimType::Bool(), load);
    } else {
      return std::move(load);
    }
  }

  UnchangedOr<Stmt> Mutate_(const tirx::TilePrimitiveCallNode* op, InplaceMode inplace_mode) final {
    auto args = op->args.Map([this](const ffi::Any& arg) { return MutateTileArgument(arg); });
    if (args.same_as(op->args)) {
      return ffi::Unchanged();
    } else {
      if (inplace_mode == InplaceMode::kAllow) {
        const_cast<TilePrimitiveCallNode*>(op)->args = std::move(args);
        return ffi::Unchanged();
      }
      auto n = ffi::make_object<TilePrimitiveCallNode>(*op);
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

  BufferStore VisitBufferAccess(BufferStore node, const BufferVar& logical_buffer) {
    TVM_FFI_ICHECK(logical_buffer.defined());
    if (!logical_buffer->layout.has_value()) {
      return node;
    }
    auto flattened_indices = GetSimplifiedElemOffset(logical_buffer, node->indices);
    BufferVar flattened_buffer = GetFlattenedBuffer(logical_buffer);
    auto writer = node.CopyOnWrite();
    writer->buffer = flattened_buffer;
    writer->indices = flattened_indices;
    return node;
  }

  TensorLoad VisitBufferAccess(TensorLoad node, const BufferVar& logical_buffer) {
    TVM_FFI_ICHECK(logical_buffer.defined());
    if (!logical_buffer->layout.has_value()) {
      if (node->source.same_as(logical_buffer.var())) return node;
      return BufferLoad(node->source.as_or_throw<BufferVar>(), node->indices, node->span);
    }
    return BufferLoad(GetFlattenedBuffer(logical_buffer),
                      GetSimplifiedElemOffset(logical_buffer, node->indices), node->span);
  }
};

class TrainiumBufferOffsetRemover : public StmtExprMutator {
 public:
  static Stmt Remove(const Stmt& stmt) {
    return ffi::make_object<TrainiumBufferOffsetRemover>()->Mutate(stmt).ValueOrUnchanged(stmt);
  }

 private:
  UnchangedOr<Expr> Mutate_(const CallNode* call, InplaceMode inplace_mode) final {
    if (call->op.same_as(tirx::builtin::buffer_offset())) {
      auto buffer_load = call->args[0].as_or_throw<TensorLoad>();
      TVM_FFI_ICHECK_EQ(buffer_load->indices.size(), 1) << "Expected a single index";
      return buffer_load->indices[0];
    }
    return StmtExprMutator::Mutate_(call, inplace_mode);
  }
};

namespace transform {

Pass LowerTrainiumLayout() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    auto [body, params] = TrainiumLayoutApplier::Lower(n->body, n->params);
    n->body = std::move(body);
    n->params = std::move(params);
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
