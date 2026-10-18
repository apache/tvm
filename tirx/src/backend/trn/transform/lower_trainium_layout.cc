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
#include <tvm/runtime/memory_scope.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/stmt_functor_with_analyzer.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <tuple>
#include <utility>
#include <vector>

#include "../../../tirx/transform/flatten_tensor_utils.h"

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

class TrainiumLayoutApplier : public tirx::StmtExprMutatorWithAnalyzer {
 public:
  static std::pair<Stmt, ffi::Array<Var>> Lower(const Stmt& stmt, const ffi::Array<Var>& params) {
    sym::Analyzer ana;
    auto storage_lower = ffi::make_object<TrainiumLayoutApplier>(ana);
    ffi::Array<Var> new_params;
    new_params.reserve(params.size());
    std::vector<std::pair<TensorVar, TensorVar>> param_flattened_tensors;
    for (const Var& param : params) {
      auto tensor = param.as<TensorVar>();
      if (!tensor) {
        new_params.push_back(param);
        continue;
      }
      if (tensor.value().ty()->layout.has_value() ||
          tensor.value().scope() == runtime::MemoryScope::Global()) {
        TensorVar flattened = storage_lower->GetFlattenedTensor(tensor.value());
        auto type = ffi::make_object<TensorTypeNode>(*tensor.value().ty().get());
        if (const auto* tile = type->layout.as<TileLayoutNode>()) {
          // Trainium's physical axes are embedded in the lowered indices. Ordinary
          // memory strides still define the parameter ABI, even when they are static.
          // Retain symbolic physical geometry that may define parameter variables.
          bool static_geometry =
              IsTrainiumLayout(tile) && tile->replica.empty() &&
              std::any_of(tile->shard.begin(), tile->shard.end(),
                          [](const Iter& iter) { return iter->axis->IsMemoryAxis(); });
          for (const auto& iter : tile->shard) {
            static_geometry &= iter->extent.as<IntImmNode>() && iter->stride.as<IntImmNode>();
          }
          bool lower_layout = static_geometry || type->HasDefaultLayout(ana);
          for (const auto& [axis, offset] : tile->offset) {
            lower_layout &= ana->CanProveEqual(offset, 0);
          }
          if (lower_layout) type->layout = std::nullopt;
        }
        TensorVar source =
            TensorVar(tensor.value()->name, TensorType(std::move(type)), tensor.value()->loc);
        param_flattened_tensors.emplace_back(flattened, source);
        new_params.push_back(source);
      } else {
        new_params.push_back(tensor.value());
      }
    }
    auto new_stmt = storage_lower->Mutate(stmt, InplaceMode::kDisallow).ValueOrUnchanged(stmt);
    for (const auto& [buf, source] : param_flattened_tensors) {
      new_stmt = SeqStmt(
          {Bind(buf, Call(buf.ty(), tirx::decl_tensor_op(), {source.data()}, {}, {buf.ty()})),
           std::move(new_stmt)});
    }
    return std::make_pair(new_stmt, new_params);
  }

  explicit TrainiumLayoutApplier(const sym::Analyzer& analyzer)
      : tirx::StmtExprMutatorWithAnalyzer(analyzer) {}

 protected:
  using StmtExprMutatorWithAnalyzer::Mutate_;

  ffi::Any MutateTileArgument(const ffi::Any& any) {
    if (auto tensor = any.as<TensorVar>()) {
      return GetFlattenedTensor(tensor.value());
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
      TensorVar original_tensor = op->var.as_or_throw<TensorVar>();
      ffi::Array<Expr> args = call->args;
      if (!args.empty()) {
        args.Set(0, Mutate(args[0], inplace_mode).ValueOrUnchanged(args[0]));
      }
      auto tensor = (original_tensor.ty()->layout.has_value() ||
                     original_tensor.scope() == runtime::MemoryScope::Global())
                        ? GetFlattenedTensor(original_tensor, /*is_alloc=*/true)
                        : original_tensor;
      if (tensor.same_as(original_tensor) && args.same_as(call->args)) {
        return ffi::Unchanged();
      }

      return Bind(
          tensor,
          Call(tensor.ty(), tirx::alloc_tensor_op(), args, call->attrs, {tensor.ty()}, call->loc),
          op->loc);
    }
    if (const auto* call = op->value.as<CallNode>();
        call && call->op.same_as(tirx::decl_tensor_op())) {
      TensorVar original_tensor = op->var.as_or_throw<TensorVar>();
      Expr original_data = call->args[0];
      auto data_update = Mutate(original_data, inplace_mode);
      bool data_unchanged = data_update.UnchangedOrSameAs(original_data);
      Expr data = std::move(data_update).ValueOrUnchanged(original_data);
      auto tensor = GetFlattenedTensor(original_tensor);
      if (tensor.same_as(original_tensor) && data_unchanged) {
        return ffi::Unchanged();
      }
      return Bind(tensor,
                  Call(tensor.ty(), tirx::decl_tensor_op(), {std::move(data)}, call->attrs,
                       {tensor.ty()}, call->loc),
                  op->loc);
    }
    return StmtExprMutatorWithAnalyzer::Mutate_(op, inplace_mode);
  }

  TensorVar GetFlattenedTensor(TensorVar buf, bool is_alloc = false) {
    static const auto scope_trn_psum = runtime::MemoryScope::Get("trn.psum");
    ffi::Any mapped = VarRemapGet(buf);
    if (mapped.type_index() != ffi::TypeIndex::kTVMFFINone) {
      return std::move(mapped).as_or_throw<UnchangedOr<TensorVar>>().ValueOrUnchanged(buf);
    }
    auto trn_layout = buf.ty()->layout.as<TileLayoutNode>();
    TensorVar flattened = buf;
    ffi::ObjectPtr<TensorTypeNode> type;
    if (IsTrainiumLayout(trn_layout)) {
      ffi::Array<PrimExpr> new_shape =
          buf.scope() == scope_trn_psum
              ? ffi::Array<PrimExpr>{trn_layout->GetSpan(ffi::String("Bank")),
                                     trn_layout->GetSize(ffi::String("P")),
                                     trn_layout->GetSpan(ffi::String("F"))}
              : ffi::Array<PrimExpr>{trn_layout->GetSize(ffi::String("P")),
                                     trn_layout->GetSpan(ffi::String("F"))};
      flattened = buf;
      type = ffi::make_object<TensorTypeNode>(*flattened.ty().get());
      type->shape = new_shape;
      type->layout = std::nullopt;
    } else if (is_alloc) {
      if (auto tile_layout = buf.ty()->layout.as<TileLayoutNode>();
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
        type = ffi::make_object<TensorTypeNode>(*flattened.ty().get());
        type->shape = {ana->Simplify(mem_span)};
        type->layout = std::nullopt;
      } else {
        flattened = FlattenedTensor(buf, ffi::GetRef<sym::Analyzer>(analyzer_));
        type = ffi::make_object<TensorTypeNode>(*flattened.ty().get());
      }
    } else {
      flattened = FlattenedTensor(buf, ffi::GetRef<sym::Analyzer>(analyzer_), /*fold_origin=*/true);
      type = ffi::make_object<TensorTypeNode>(*flattened.ty().get());
    }
    if (flattened.dtype()->dtype == DLDataType{kDLBool, 8, 1}) {
      type->dtype = PrimType::Int(8);
    }
    for (size_t i = 0; i < type->shape.size(); ++i) {
      type->shape.Set(i, analyzer_->canonical_simplify(type->shape[i]));
    }
    if (!IsTrainiumLayout(trn_layout)) type->layout = std::nullopt;
    flattened = TensorVar(flattened->name, TensorType(std::move(type)), flattened->loc);

    VarRemapSet(buf, flattened);
    return flattened;
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreNode* op, InplaceMode inplace_mode) final {
    // Index conversion needs the original logical layout after the parent remaps the tensor.
    TensorVar logical_tensor = op->dest.as_or_throw<TensorVar>();
    TensorStore store = StmtExprMutator::Mutate_(op, inplace_mode)
                            .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                            .as_or_throw<TensorStore>();
    PrimType store_value_ty = op->value.ty();
    bool store_returns_bool = store_value_ty.MatchesCode(DLDataTypeCode::kDLBool);
    store = VisitTensorAccess(store, logical_tensor);

    if (store_returns_bool) {
      TVM_FFI_ICHECK_EQ(store->dest.as_or_throw<TensorVar>().dtype()->dtype,
                        (DLDataType{kDLInt, 8, 1}))
          << "Expected int8 backing array for boolean tensor";
      auto writer = store.CopyOnWrite();
      writer->value = tvm::prim::cast(store->value, PrimType::Int(8));
      return std::move(store);
    }
    return std::move(store);
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* op, InplaceMode inplace_mode) final {
    TensorVar logical_tensor = op->source.as_or_throw<TensorVar>();
    PrimType load_ty = op->ty.as_or_throw<PrimType>();
    bool load_returns_bool = load_ty.MatchesCode(DLDataTypeCode::kDLBool);
    TensorLoad load = StmtExprMutator::Mutate_(op, inplace_mode)
                          .ValueOrUnchanged(ffi::GetRef<PrimExpr>(op))
                          .as_or_throw<TensorLoad>();
    load = VisitTensorAccess(load, logical_tensor);
    if (load_returns_bool) {
      TVM_FFI_ICHECK_EQ(load->source.as_or_throw<tvm::tirx::TensorVar>().dtype()->dtype,
                        (DLDataType{kDLInt, 8, 1}))
          << "Expected int8 backing array for boolean tensor";
      load.CopyOnWrite()->ExprNode::ty = PrimType::Int(8);
      return tvm::prim::cast(load, PrimType::Bool());
    } else {
      return std::move(load);
    }
  }

  ffi::Array<PrimExpr> GetSimplifiedElemOffset(const TensorVar& tensor,
                                               const ffi::Array<PrimExpr>& indices) {
    static const auto scope_trn_psum = runtime::MemoryScope::Get("trn.psum");
    if (tensor.ty()->layout.has_value()) {
      auto layout = tensor.ty()->layout.value();
      layout = StmtExprMutator::Mutate(ffi::AnyView(layout), InplaceMode::kDisallow)
                   .ValueOrUnchanged(layout)
                   .as_or_throw<Layout>();
      auto tile_layout = layout.as<TileLayoutNode>();
      if (IsTrainiumLayout(tile_layout)) {
        auto coord = layout->Apply(indices, tensor.shape());
        std::vector<PrimExpr> res;
        for (const auto& axis : tensor.scope() == scope_trn_psum
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
        LOG(FATAL) << "Cannot lower direct TensorLoad/TensorStore on a tensor with thread-axis "
                   << "layout: unable to verify that the coordinate matches the current thread. "
                   << "Use .view() + .local() to decompose thread and memory axes.";
      }
      auto res = layout->Canonicalize()->Apply(indices, tensor.shape());
      TVM_FFI_ICHECK_EQ(res.size(), 1) << "Expected a single element offset";
      return {analyzer_->Simplify((*res.begin()).second)};
    }
    auto flattened_indices = tensor.ty()->ElemOffset(indices);
    TVM_FFI_ICHECK_EQ(flattened_indices.size(), 1) << "Expected a single element offset";
    return {analyzer_->Simplify(flattened_indices[0])};
  }

  TensorStore VisitTensorAccess(TensorStore node, const TensorVar& logical_tensor) {
    TVM_FFI_ICHECK(logical_tensor.defined());
    if (!logical_tensor.ty()->layout.has_value() &&
        logical_tensor.scope() != runtime::MemoryScope::Global()) {
      return node;
    }
    auto flattened_indices = GetSimplifiedElemOffset(logical_tensor, node->indices);
    TensorVar flattened_tensor = GetFlattenedTensor(logical_tensor);
    auto writer = node.CopyOnWrite();
    writer->dest = flattened_tensor;
    writer->indices = flattened_indices;
    return node;
  }

  TensorLoad VisitTensorAccess(TensorLoad node, const TensorVar& logical_tensor) {
    TVM_FFI_ICHECK(logical_tensor.defined());
    if (!logical_tensor.ty()->layout.has_value() &&
        logical_tensor.scope() != runtime::MemoryScope::Global()) {
      if (node->source.same_as(logical_tensor)) return node;
      return TensorLoad(node->source.as_or_throw<TensorVar>(), node->indices, node->loc);
    }
    return TensorLoad(GetFlattenedTensor(logical_tensor),
                      GetSimplifiedElemOffset(logical_tensor, node->indices), node->loc);
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
