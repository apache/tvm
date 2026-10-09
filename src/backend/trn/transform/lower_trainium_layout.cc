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
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
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
    std::vector<std::pair<TensorVar, TensorVar>> param_flattened_buffers;
    for (const Var& param : params) {
      auto buffer = param.as<TensorVar>();
      if (!buffer) {
        new_params.push_back(param);
        continue;
      }
      if (buffer.value()->layout.has_value()) {
        TensorVar flattened = storage_lower->GetFlattenedTensor(buffer.value());
        auto type = CopyTensorType(buffer.value());
        type->layout = std::nullopt;
        TensorVar source = RebuildTensorVar(buffer.value(), std::move(type));
        param_flattened_buffers.emplace_back(flattened, source);
        new_params.push_back(source.var());
      } else {
        new_params.push_back(buffer.value().var());
      }
    }
    auto new_stmt = storage_lower->Mutate(stmt, InplaceMode::kDisallow).ValueOrUnchanged(stmt);
    for (const auto& [buf, source] : param_flattened_buffers) {
      new_stmt = SeqStmt({Bind(buf, Call(buf.type(), tirx::decl_tensor_op(),
                                         {source.data(), tvm::Tuple(buf->shape),
                                          DataTypeImm(buf->dtype->dtype), StringImm(buf.scope())},
                                         {})),
                          std::move(new_stmt)});
    }
    return std::make_pair(new_stmt, new_params);
  }

  explicit TrainiumLayoutApplier(const sym::Analyzer& analyzer)
      : tirx::IRMutatorWithAnalyzer(analyzer) {}

 protected:
  using IRMutatorWithAnalyzer::Mutate_;

  ffi::Any MutateTileArgument(const ffi::Any& any) {
    if (auto buffer = any.as<TensorVar>()) {
      return GetFlattenedTensor(buffer.value());
    }
    if (auto expr = any.as<PrimExpr>()) {
      return Mutate(expr.value()).ValueOrUnchanged(expr.value());
    }
    if (auto stmt = any.as<Stmt>()) {
      return Mutate(stmt.value()).ValueOrUnchanged(stmt.value());
    }
    return any;
  }

  UnchangedOr<Stmt> Mutate_(const BindNode* op, InplaceMode inplace_mode) final {
    if (const auto* call = op->value.as<CallNode>();
        call && call->op.same_as(tirx::alloc_tensor_op())) {
      TensorVar original_buffer = op->var.as_or_throw<TensorVar>();
      if (!original_buffer->layout.has_value()) {
        return ffi::Unchanged();
      }
      auto buffer = GetFlattenedTensor(original_buffer, /*is_alloc=*/true);
      if (buffer.same_as(original_buffer)) {
        return ffi::Unchanged();
      }
      ffi::Array<Expr> args = call->args;
      args.Set(0, tvm::Tuple(buffer->shape, call->args[0]->span));
      args.Set(1, DataTypeImm(buffer->dtype->dtype, call->args[1]->span));
      args.Set(2, StringImm(buffer.scope(), call->args[2]->span));
      return Bind(buffer.var(),
                  Call(buffer.type(), tirx::alloc_tensor_op(), args, call->attrs, call->ty_args,
                       call->span),
                  op->span);
    }
    if (const auto* call = op->value.as<CallNode>();
        call && call->op.same_as(tirx::decl_tensor_op())) {
      TensorVar original_buffer = op->var.as_or_throw<TensorVar>();
      Expr original_data = call->args[0];
      auto data_update = Mutate(original_data, inplace_mode);
      bool data_unchanged = data_update.UnchangedOrSameAs(original_data);
      Expr data = std::move(data_update).ValueOrUnchanged(original_data);
      auto buffer = GetFlattenedTensor(original_buffer);
      if (buffer.same_as(original_buffer) && data_unchanged) {
        return ffi::Unchanged();
      }
      return Bind(buffer,
                  Call(buffer.type(), tirx::decl_tensor_op(),
                       {std::move(data), tvm::Tuple(buffer->shape),
                        DataTypeImm(buffer->dtype->dtype), StringImm(buffer.scope())},
                       call->attrs, call->ty_args, call->span),
                  op->span);
    }
    return IRMutatorWithAnalyzer::Mutate_(op, inplace_mode);
  }

  TensorVar GetFlattenedTensor(TensorVar buf, bool is_alloc = false) {
    ffi::Any mapped = VarRemapGet(buf);
    if (mapped.type_index() != ffi::TypeIndex::kTVMFFINone) {
      return std::move(mapped).as_or_throw<UnchangedOr<TensorVar>>().ValueOrUnchanged(buf);
    }
    auto trn_layout = buf->layout.as<TileLayoutNode>();
    TensorVar flattened = buf;
    ffi::ObjectPtr<TensorTypeNode> type;
    if (IsTrainiumLayout(trn_layout)) {
      ffi::Array<PrimExpr> new_shape =
          buf.scope() == "trn.psum" ? ffi::Array<PrimExpr>{trn_layout->GetSpan(ffi::String("Bank")),
                                                           trn_layout->GetSize(ffi::String("P")),
                                                           trn_layout->GetSpan(ffi::String("F"))}
                                    : ffi::Array<PrimExpr>{trn_layout->GetSize(ffi::String("P")),
                                                           trn_layout->GetSpan(ffi::String("F"))};
      flattened = buf;
      type = CopyTensorType(flattened);
      type->shape = new_shape;
      type->strides = {};
    } else if (is_alloc) {
      if (auto tile_layout = buf->layout.as<TileLayoutNode>();
          tile_layout && tile_layout->HasThreadAxis()) {
        sym::Analyzer ana;
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
        type = CopyTensorType(flattened);
        type->shape = {ana->Simplify(mem_span)};
        type->strides = {};
      } else {
        flattened = buf.GetFlattenedTensor();
        type = CopyTensorType(flattened);
      }
    } else {
      flattened = buf.GetFlattenedTensor();
      type = CopyTensorType(flattened);
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
    flattened = RebuildTensorVar(flattened, std::move(type));

    VarRemapSet(buf, flattened);
    return flattened;
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreNode* op, InplaceMode inplace_mode) final {
    // Index conversion needs the original logical layout after the parent remaps the buffer.
    TensorVar logical_buffer = op->dest.as_or_throw<TensorVar>();
    TensorStore store = StmtExprMutator::Mutate_(op, inplace_mode)
                            .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                            .as_or_throw<TensorStore>();
    PrimType store_value_ty = op->value.ty();
    bool store_returns_bool = store_value_ty.MatchesCode(DLDataTypeCode::kDLBool);
    store = VisitBufferAccess(store, logical_buffer);

    if (store_returns_bool) {
      TVM_FFI_ICHECK_EQ(store->dest.as_or_throw<TensorVar>()->dtype->dtype,
                        (DLDataType{kDLInt, 8, 1}))
          << "Expected int8 backing array for boolean tensor";
      auto writer = store.CopyOnWrite();
      writer->value = tvm::prim::cast(PrimType::Int(8), store->value);
      return std::move(store);
    }
    return std::move(store);
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* op, InplaceMode inplace_mode) final {
    TensorVar logical_buffer = op->source.as_or_throw<TensorVar>();
    PrimType load_ty = op->ty.as_or_throw<PrimType>();
    bool load_returns_bool = load_ty.MatchesCode(DLDataTypeCode::kDLBool);
    TensorLoad load = StmtExprMutator::Mutate_(op, inplace_mode)
                          .ValueOrUnchanged(ffi::GetRef<PrimExpr>(op))
                          .as_or_throw<TensorLoad>();
    load = VisitBufferAccess(load, logical_buffer);
    if (load_returns_bool) {
      TVM_FFI_ICHECK_EQ(load->source.as_or_throw<tvm::tirx::TensorVar>()->dtype->dtype,
                        (DLDataType{kDLInt, 8, 1}))
          << "Expected int8 backing array for boolean tensor";
      load.CopyOnWrite()->ExprNode::ty = PrimType::Int(8);
      return tvm::prim::cast(PrimType::Bool(), load);
    } else {
      return std::move(load);
    }
  }

  ffi::Array<PrimExpr> GetSimplifiedElemOffset(const TensorVar& buffer,
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
        LOG(FATAL) << "Cannot lower direct TensorLoad/TensorStore on a buffer with thread-axis "
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

  TensorStore VisitBufferAccess(TensorStore node, const TensorVar& logical_buffer) {
    TVM_FFI_ICHECK(logical_buffer.defined());
    if (!logical_buffer->layout.has_value()) {
      return node;
    }
    auto flattened_indices = GetSimplifiedElemOffset(logical_buffer, node->indices);
    TensorVar flattened_buffer = GetFlattenedTensor(logical_buffer);
    auto writer = node.CopyOnWrite();
    writer->dest = flattened_buffer;
    writer->indices = flattened_indices;
    return node;
  }

  TensorLoad VisitBufferAccess(TensorLoad node, const TensorVar& logical_buffer) {
    TVM_FFI_ICHECK(logical_buffer.defined());
    if (!logical_buffer->layout.has_value()) {
      if (node->source.same_as(logical_buffer.var())) return node;
      return MakeTensorLoad(node->source.as_or_throw<TensorVar>(), node->indices, node->span);
    }
    return MakeTensorLoad(GetFlattenedTensor(logical_buffer),
                          GetSimplifiedElemOffset(logical_buffer, node->indices), node->span);
  }
};

namespace transform {

Pass LowerTrainiumLayout() {
  auto pass_func = [](Function f, IRModule m, PassContext ctx) {
    if (!f->body.has_value()) return f;
    auto* n = f.CopyOnWrite();
    auto [body, params] = TrainiumLayoutApplier::Lower(n->body.value(), n->params);
    n->body = std::move(body);
    n->params = std::move(params);
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "tirx.backend.trn.LowerTrainiumLayout");
}

void RegisterTRNTransforms() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.backend.trn.transform.LowerTrainiumLayout", LowerTrainiumLayout);
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
