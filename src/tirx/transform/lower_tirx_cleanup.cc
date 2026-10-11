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

#include <tvm/ir/function.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/sym/analyzer.h>
#include <tvm/target/target.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../ir/ir_mutator_with_analyzer.h"
#include "flattened_tensor.h"
#include "ir_utils.h"

namespace tvm {
namespace tirx {

class LayoutApplier : public IRMutatorWithAnalyzer {
 public:
  using IRMutatorWithAnalyzer::Mutate;
  using IRMutatorWithAnalyzer::Mutate_;
  static std::pair<Stmt, ffi::Array<Var>> Flatten(const Stmt& stmt, const ffi::Array<Var>& params,
                                                  const Target& target) {
    sym::Analyzer ana;
    auto storage_lower = ffi::make_object<LayoutApplier>(ana, target);
    ffi::Array<Var> new_params;
    new_params.reserve(params.size());
    std::vector<std::pair<TensorVar, TensorVar>> param_flattened_tensors;
    for (const Var& param : params) {
      auto tensor = param.as<TensorVar>();
      if (!tensor) {
        new_params.push_back(param);
        continue;
      }
      storage_lower->tensor_aliases_.Set(tensor.value().var(), tensor.value().var());
      TensorVar flattened = storage_lower->GetFlattenedTensor(tensor.value());
      TensorVar source = tensor.value();
      if (source->layout.has_value()) {
        auto type = CopyTensorType(source);
        type->layout = std::nullopt;
        source = RebuildTensorVar(source, std::move(type));
      }
      if (!flattened.same_as(source)) {
        param_flattened_tensors.emplace_back(flattened, source);
      }
      new_params.push_back(source.var());
    }
    auto new_stmt = storage_lower->Mutate(stmt, InplaceMode::kAllow).ValueOrUnchanged(stmt);
    for (const auto& [buf, source] : param_flattened_tensors) {
      new_stmt =
          SeqStmt({Bind(buf, Call(buf.type(), decl_tensor_op(), {source.data()}, {}, {buf.type()})),
                   std::move(new_stmt)});
    }
    return std::make_pair(new_stmt, new_params);
  }

 public:
  explicit LayoutApplier(const sym::Analyzer& analyzer, const Target& target)
      : IRMutatorWithAnalyzer(analyzer), target_(target) {}

 protected:
  ffi::Any VisitAny(const ffi::Any& any) {
    if (any == nullptr) {
      return any;
    }
    if (auto tensor = any.as<TensorVar>()) {
      return GetFlattenedTensor(tensor.value());
    } else if (auto prim_expr = any.as<PrimExpr>()) {
      return Mutate(prim_expr.value(), InplaceMode::kDisallow).ValueOrUnchanged(prim_expr.value());
    } else if (auto stmt = any.as<Stmt>()) {
      return Mutate(stmt.value(), InplaceMode::kDisallow).ValueOrUnchanged(stmt.value());
    }
    return any;
  }

  UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) final {
    if (op->op.same_as(tensor_data_ptr_op()) && op->args.size() == 1) {
      if (auto var = op->args[0].as<Var>();
          var.has_value() && var.value()->ty.as<TensorTypeNode>()) {
        auto root_opt = tensor_aliases_.Get(var.value());
        TVM_FFI_ICHECK(root_opt.has_value())
            << "tensor_data_ptr projects " << var.value()->name
            << ", which has no visible definition "
            << "(AllocTensor/DeclTensor/Function parameter) at this point";
        Var root = root_opt.value();
        if (auto mapped = VarRemapGet(root); mapped != nullptr) {
          root = mapped.as_or_throw<Var>();
        }
        return root.as_or_throw<TensorVar>().data();
      }
    }
    return IRMutatorWithAnalyzer::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_(const BindNode* op, InplaceMode inplace_mode) final {
    if (auto source = op->value.as<TensorVar>(); source.has_value()) {
      auto tensor = op->var.as_or_throw<TensorVar>();
      Call declaration(tensor.type(), decl_tensor_op(), {source.value().data()}, {},
                       {tensor.type()});
      Bind binding(op->var, declaration, op->loc);
      return MutateDeclTensor(binding.get(), declaration.get(), inplace_mode)
          .ValueOrUnchanged(binding);
    }
    if (const auto* call = op->value.as<CallNode>(); call) {
      if (call->op.same_as(alloc_tensor_op())) return MutateAllocTensor(op, call, inplace_mode);
      if (call->op.same_as(decl_tensor_op())) return MutateDeclTensor(op, call, inplace_mode);
      if (call->op.same_as(Op::Get("tirx.cuda.decl_tmem"))) {
        auto tensor = op->var.as_or_throw<TensorVar>();
        // TMEM is already allocated by the pool.  Retain an explicit backing pointer so
        // ordinary data projections also work for views and function parameters.
        Call declaration(tensor.type(), decl_tensor_op(),
                         {Call(tensor->DataPtrType(), reinterpret_op(), {call->args[0]})}, {},
                         {tensor.type()});
        Bind binding(op->var, declaration, op->loc);
        return MutateDeclTensor(binding.get(), declaration.get(), inplace_mode)
            .ValueOrUnchanged(binding);
      }
    }
    return IRMutatorWithAnalyzer::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> MutateAllocTensor(const BindNode* op, const CallNode* tensor_call,
                                      InplaceMode inplace_mode) {
    tensor_aliases_.Set(op->var, op->var);
    auto mutate = [this](TensorVar buf) { return GetFlattenedTensor(buf, /*is_alloc=*/true); };
    auto tensor = mutate(op->var.as_or_throw<TensorVar>());
    ffi::Array<Expr> args = tensor_call->args;

    if (!args.empty()) {
      args.Set(0, Mutate(args[0], inplace_mode).ValueOrUnchanged(args[0]));
    }
    return Bind(tensor.var(),
                Call(tensor.type(), tirx::alloc_tensor_op(), args, tensor_call->attrs,
                     {tensor.type()}, tensor_call->loc),
                op->loc);
  }

  UnchangedOr<Stmt> MutateDeclTensor(const BindNode* op, const CallNode* tensor_call,
                                     InplaceMode inplace_mode) {
    RegisterTensorAlias(op->var.as_or_throw<TensorVar>(), tensor_call->args[0]);
    auto data_result = Mutate(tensor_call->args[0], inplace_mode);
    bool data_unchanged = data_result.UnchangedOrSameAs(tensor_call->args[0]);
    Expr data = std::move(data_result).ValueOrUnchanged(tensor_call->args[0]);
    auto tensor = GetFlattenedTensor(op->var.as_or_throw<TensorVar>());
    if (tensor.same_as(op->var.as_or_throw<TensorVar>()) && data_unchanged) {
      return ffi::Unchanged();
    }
    return Bind(tensor,
                Call(tensor.type(), decl_tensor_op(), {std::move(data)}, tensor_call->attrs,
                     {tensor.type()}, tensor_call->loc),
                op->loc);
  }

  TensorVar GetFlattenedTensor(TensorVar buf, bool is_alloc = false) {
    if (auto mapped = VarRemapGet(buf); mapped != nullptr) {
      return mapped.as_or_throw<TensorVar>();
    }
    TensorVar flattened = buf;
    ffi::ObjectPtr<TensorTypeNode> type;
    if (is_alloc) {
      if (auto tile_layout = ffi::AnyView(buf->layout).as<TileLayoutNode>();
          tile_layout && tile_layout->HasThreadAxis()) {
        // Logical alloc_tensor with thread axes: physical shape = memory-axis span
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
        flattened = FlattenedTensor(buf);
        type = CopyTensorType(flattened);
      }
    } else {
      flattened = FlattenedTensor(buf);
      type = CopyTensorType(flattened);
    }
    // Remap variables the pass has already rebuilt (a shape may load from
    // another local tensor), then canonicalize.
    for (size_t i = 0; i < type->shape.size(); ++i) {
      type->shape.Set(
          i, analyzer_->canonical_simplify(
                 StmtExprMutator::Mutate(ffi::AnyView(type->shape[i]), InplaceMode::kDisallow)
                     .ValueOrUnchanged(type->shape[i])
                     .as_or_throw<PrimExpr>()));
    }
    for (size_t i = 0; i < type->strides.size(); ++i) {
      type->strides.Set(
          i, StmtExprMutator::Mutate(ffi::AnyView(type->strides[i]), InplaceMode::kDisallow)
                 .ValueOrUnchanged(type->strides[i])
                 .as_or_throw<PrimExpr>());
    }
    type->layout = std::nullopt;
    type->elem_offset =
        StmtExprMutator::Mutate(ffi::AnyView(buf->elem_offset), InplaceMode::kDisallow)
            .ValueOrUnchanged(buf->elem_offset)
            .as_or_throw<PrimExpr>();
    if (ffi::StructuralEqual()(buf.type(), TensorType(type))) {
      return buf;
    }
    flattened = RebuildTensorVar(flattened, std::move(type));

    VarRemapSet(buf, flattened);
    return flattened;
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreNode* op, InplaceMode inplace_mode) final {
    // Preserve the logical tensor until VisitTensorAccess linearizes its indices.
    auto value = Mutate(op->value, inplace_mode);
    auto indices =
        Mutate(op->indices, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorStore store = ffi::GetRef<TensorStore>(op);
    if (!value.UnchangedOrSameAs(op->value) || !indices.UnchangedOrSameAs(op->indices)) {
      auto* n = store.CopyOnWrite();
      n->value = std::move(value).ValueOrUnchanged(op->value);
      n->indices = std::move(indices).ValueOrUnchanged(op->indices);
    }
    return VisitTensorAccess(std::move(store));
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* op, InplaceMode inplace_mode) final {
    auto indices =
        Mutate(op->indices, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorLoad load = ffi::GetRef<TensorLoad>(op);
    if (!indices.UnchangedOrSameAs(op->indices)) {
      load.CopyOnWrite()->indices = std::move(indices).ValueUnchecked();
    }
    return VisitTensorAccess(std::move(load));
  }

  ffi::Array<PrimExpr> GetSimplifiedElemOffset(const TensorVar& tensor,
                                               const ffi::Array<PrimExpr>& indices) {
    if (tensor->layout.has_value()) {
      if (auto tile = tensor->layout.value().as<TileLayoutNode>(); tile && tile->HasThreadAxis()) {
        LOG(FATAL) << "Cannot lower direct TensorLoad/TensorStore on a tensor with thread-axis "
                   << "layout: unable to verify that the coordinate matches the current thread. "
                   << "Use .view() + .local() to decompose thread and memory axes.";
      }
      auto res = tensor->layout.value()->Canonicalize()->Apply(indices, tensor->shape);
      TVM_FFI_ICHECK_EQ(res.size(), 1) << "Expected a single element offset";
      return {analyzer_->Simplify((*res.begin()).second)};
    }
    auto flattened_indices = tensor->ElemOffset(indices, true);
    TVM_FFI_ICHECK_EQ(flattened_indices.size(), 1) << "Expected a single element offset";
    return {analyzer_->Simplify(flattened_indices[0])};
  }

  template <typename Node>
  Node VisitTensorAccess(Node node) {
    TVM_FFI_ICHECK(node->dest.template as_or_throw<TensorVar>().defined());
    auto flattened_indices =
        GetSimplifiedElemOffset(node->dest.template as_or_throw<TensorVar>(), node->indices);
    TensorVar flattened_tensor = GetFlattenedTensor(node->dest.template as_or_throw<TensorVar>());
    auto writer = node.CopyOnWrite();
    writer->dest = flattened_tensor;
    writer->indices = flattened_indices;
    return node;
  }

  TensorLoad VisitTensorAccess(TensorLoad node) {
    TensorVar tensor = node->source.as_or_throw<tvm::tirx::TensorVar>();
    TVM_FFI_ICHECK(tensor.defined());
    return MakeTensorLoad(GetFlattenedTensor(tensor),
                          GetSimplifiedElemOffset(tensor, node->indices), node->loc);
  }

  /*! \brief Map of variables being remapped, including tensor variables. */

 private:
  void RegisterTensorAlias(TensorVar tensor, const Expr& data) {
    Var root = tensor.var();
    if (const auto* call = data.as<CallNode>();
        call && call->op.same_as(tensor_data_ptr_op()) && call->args.size() == 1) {
      if (auto source = call->args[0].as<Var>();
          source.has_value() && source.value()->ty.as<TensorTypeNode>()) {
        auto source_root = tensor_aliases_.Get(source.value());
        TVM_FFI_ICHECK(source_root.has_value())
            << "Tensor alias source " << source.value()->name
            << " must be registered before its DeclTensor alias";
        root = source_root.value();
      }
    }
    tensor_aliases_.Set(tensor.var(), root);
  }

  /*! \brief Physical roots of tensor aliases, flattened at each declaration. */
  ffi::Map<Var, Var> tensor_aliases_;
  const Target& target_;
};

namespace {
Target ResolveTarget(const Function& f) {
  auto target = f->GetAttr<Target>(tvm::attr::kTarget);
  if (!target.has_value()) {
    target = Target::Current(false).value();
  }
  return target.value();
}
}  // namespace

namespace transform {

Pass LowerTIRxCleanup() {
  auto pass_func = [](Function f, IRModule m, PassContext ctx) {
    if (!f->body.has_value()) return f;
    Target target = ResolveTarget(f);
    auto* n = f.CopyOnWrite();
    auto [body, params] = LayoutApplier::Flatten(n->body.value(), n->params, target);
    n->body = std::move(body);
    n->params = std::move(params);
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "tirx.LowerTIRxCleanup");
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
