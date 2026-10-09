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
    std::vector<std::pair<TensorVar, TensorVar>> param_flattened_buffers;
    for (const Var& param : params) {
      auto buffer = param.as<TensorVar>();
      if (!buffer) {
        new_params.push_back(param);
        continue;
      }
      storage_lower->buffer_aliases_.Set(buffer.value().var(), buffer.value().var());
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
    auto new_stmt = storage_lower->Mutate(stmt, InplaceMode::kAllow).ValueOrUnchanged(stmt);
    for (const auto& [buf, source] : param_flattened_buffers) {
      new_stmt = SeqStmt({Bind(buf, Call(buf.type(), decl_tensor_op(),
                                         {source.data(), tvm::Tuple(buf->shape),
                                          DataTypeImm(buf->dtype->dtype), StringImm(buf.scope())},
                                         {})),
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
    if (auto buffer = any.as<TensorVar>()) {
      return GetFlattenedTensor(buffer.value());
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
        auto root_opt = buffer_aliases_.Get(var.value());
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
    if (const auto* call = op->value.as<CallNode>(); call) {
      if (call->op.same_as(alloc_tensor_op())) return MutateAllocTensor(op, call, inplace_mode);
      if (call->op.same_as(decl_tensor_op())) return MutateDeclTensor(op, call, inplace_mode);
    }
    return IRMutatorWithAnalyzer::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> MutateAllocTensor(const BindNode* op, const CallNode* buffer_call,
                                      InplaceMode inplace_mode) {
    buffer_aliases_.Set(op->var, op->var);
    auto mutate = [this](TensorVar buf) {
      if (target_->kind->name == "trn" && !buf->layout.has_value()) {
        return buf;
      }
      return GetFlattenedTensor(buf, /*is_alloc=*/true);
    };
    auto buffer = mutate(op->var.as_or_throw<TensorVar>());
    if (buffer.same_as(op->var.as_or_throw<TensorVar>())) {
      return ffi::Unchanged();
    }
    return Bind(buffer.var(),
                Call(buffer.type(), tirx::alloc_tensor_op(),
                     {tvm::Tuple(buffer->shape, buffer_call->args[0]->span),
                      DataTypeImm(buffer->dtype->dtype, buffer_call->args[1]->span),
                      StringImm(buffer.scope(), buffer_call->args[2]->span)},
                     buffer_call->attrs, buffer_call->ty_args, buffer_call->span),
                op->span);
  }

  UnchangedOr<Stmt> MutateDeclTensor(const BindNode* op, const CallNode* buffer_call,
                                     InplaceMode inplace_mode) {
    RegisterBufferAlias(op->var.as_or_throw<TensorVar>(), buffer_call->args[0]);
    auto data_result = Mutate(buffer_call->args[0], inplace_mode);
    bool data_unchanged = data_result.UnchangedOrSameAs(buffer_call->args[0]);
    Expr data = std::move(data_result).ValueOrUnchanged(buffer_call->args[0]);
    auto buffer = GetFlattenedTensor(op->var.as_or_throw<TensorVar>());
    if (buffer.same_as(op->var.as_or_throw<TensorVar>()) && data_unchanged) {
      return ffi::Unchanged();
    }
    return Bind(buffer,
                Call(buffer.type(), decl_tensor_op(),
                     {std::move(data), tvm::Tuple(buffer->shape), DataTypeImm(buffer->dtype->dtype),
                      StringImm(buffer.scope())},
                     buffer_call->attrs, buffer_call->ty_args, buffer_call->span),
                op->span);
  }

  TensorVar GetFlattenedTensor(TensorVar buf, bool is_alloc = false) {
    if (auto mapped = VarRemapGet(buf); mapped != nullptr) {
      return mapped.as_or_throw<TensorVar>();
    }
    auto trn_layout = buf->layout.as<TileLayoutNode>();
    TensorVar flattened = buf;
    ffi::ObjectPtr<TensorTypeNode> type;
    if (trn_layout && trn_layout->IsTrainium()) {
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
        flattened = buf.GetFlattenedTensor();
        type = CopyTensorType(flattened);
      }
    } else {
      flattened = buf.GetFlattenedTensor();
      type = CopyTensorType(flattened);
    }
    // Remap variables the pass has already rebuilt (a shape may load from
    // another local buffer), then canonicalize.
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
    // TMEM addresses may load from an allocation whose variable was rebuilt
    // above. Keep the type metadata in sync with the declaration's pointer.
    for (size_t i = 0; i < type->allocated_addr.size(); ++i) {
      type->allocated_addr.Set(
          i, StmtExprMutator::Mutate(ffi::AnyView(type->allocated_addr[i]), InplaceMode::kDisallow)
                 .ValueOrUnchanged(type->allocated_addr[i])
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
    // Preserve the logical buffer until VisitBufferAccess linearizes its indices.
    auto value = Mutate(op->value, inplace_mode);
    auto indices =
        Mutate(op->indices, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorStore store = ffi::GetRef<TensorStore>(op);
    if (!value.UnchangedOrSameAs(op->value) || !indices.UnchangedOrSameAs(op->indices)) {
      auto* n = store.CopyOnWrite();
      n->value = std::move(value).ValueOrUnchanged(op->value);
      n->indices = std::move(indices).ValueOrUnchanged(op->indices);
    }
    return VisitBufferAccess(std::move(store));
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* op, InplaceMode inplace_mode) final {
    auto indices =
        Mutate(op->indices, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorLoad load = ffi::GetRef<TensorLoad>(op);
    if (!indices.UnchangedOrSameAs(op->indices)) {
      load.CopyOnWrite()->indices = std::move(indices).ValueUnchecked();
    }
    return VisitBufferAccess(std::move(load));
  }

  UnchangedOr<Stmt> Mutate_(const tirx::TileOpCallNode* op, InplaceMode inplace_mode) final {
    ffi::Array<Expr> args = op->args;
    args.MutateByApply([this](const Expr& arg) { return VisitAny(arg).as_or_throw<Expr>(); });
    if (args.same_as(op->args)) {
      return ffi::Unchanged();
    } else {
      if (inplace_mode == InplaceMode::kAllow) {
        auto* n = const_cast<tirx::TileOpCallNode*>(op);
        n->args = std::move(args);
        return ffi::Unchanged();
      }
      auto n = ffi::make_object<tirx::TileOpCallNode>(*op);
      n->args = std::move(args);
      return Stmt(n);
    }
  }

  ffi::Array<PrimExpr> GetSimplifiedElemOffset(const TensorVar& buffer,
                                               const ffi::Array<PrimExpr>& indices) {
    if (buffer->layout.has_value()) {
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

  template <typename Node>
  Node VisitBufferAccess(Node node) {
    TVM_FFI_ICHECK(node->buffer.defined());
    if (target_->kind->name == "trn" && !node->buffer->layout.has_value()) {
      return node;
    }
    auto flattened_indices = GetSimplifiedElemOffset(node->buffer, node->indices);
    TensorVar flattened_buffer = GetFlattenedTensor(node->buffer);
    auto writer = node.CopyOnWrite();
    writer->buffer = flattened_buffer;
    writer->indices = flattened_indices;
    return node;
  }

  TensorLoad VisitBufferAccess(TensorLoad node) {
    TensorVar buffer = node->source.as_or_throw<tvm::tirx::TensorVar>();
    TVM_FFI_ICHECK(buffer.defined());
    if (target_->kind->name == "trn" && !buffer->layout.has_value()) return node;
    return MakeTensorLoad(GetFlattenedTensor(buffer),
                          GetSimplifiedElemOffset(buffer, node->indices), node->span);
  }

  /*! \brief Map of variables being remapped, including buffer variables. */

 private:
  void RegisterBufferAlias(TensorVar buffer, const Expr& data) {
    Var root = buffer.var();
    if (const auto* call = data.as<CallNode>();
        call && call->op.same_as(tensor_data_ptr_op()) && call->args.size() == 1) {
      if (auto source = call->args[0].as<Var>();
          source.has_value() && source.value()->ty.as<TensorTypeNode>()) {
        auto source_root = buffer_aliases_.Get(source.value());
        TVM_FFI_ICHECK(source_root.has_value())
            << "Buffer alias source " << source.value()->name
            << " must be registered before its DeclTensor alias";
        root = source_root.value();
      }
    }
    buffer_aliases_.Set(buffer.var(), root);
  }

  /*! \brief Physical roots of buffer aliases, flattened at each declaration. */
  ffi::Map<Var, Var> buffer_aliases_;
  const Target& target_;
};

class BufferOffsetRemover : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  static Stmt Remove(const Stmt& stmt) {
    return ffi::make_object<BufferOffsetRemover>()
        ->Mutate(stmt, InplaceMode::kAllow)
        .ValueOrUnchanged(stmt);
  }

 private:
  UnchangedOr<Expr> Mutate_(const CallNode* call, InplaceMode inplace_mode) final {
    if (call->op.same_as(tirx::buffer_offset_op())) {
      auto buffer_load = call->args[0].as_or_throw<TensorLoad>();
      TVM_FFI_ICHECK_EQ(buffer_load->indices.size(), 1) << "Expected a single index";
      return buffer_load->indices[0];
    }
    return StmtExprMutator::Mutate_(call, inplace_mode);
  }
};

namespace {
Target ResolveTarget(const Function& f) {
  auto target = f->GetAttr<Target>(tvm::attr::kTarget);
  if (!target.has_value()) {
    target = Target::Current(false);
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
    n->body = BufferOffsetRemover::Remove(n->body.value());
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "tirx.LowerTIRxCleanup", {});
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
