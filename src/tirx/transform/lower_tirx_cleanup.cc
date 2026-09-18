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

#include <tvm/ir/prim/expr.h>
#include <tvm/runtime/logging.h>
#include <tvm/sym/analyzer.h>
#include <tvm/target/target.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/tile_primitive.h>
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
    std::vector<std::pair<BufferVar, BufferVar>> param_flattened_buffers;
    for (const Var& param : params) {
      auto buffer = param.as<BufferVar>();
      if (!buffer) {
        new_params.push_back(param);
        continue;
      }
      storage_lower->buffer_aliases_.Set(buffer.value().var(), buffer.value().var());
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
    auto new_stmt = storage_lower->Mutate(stmt, InplaceMode::kAllow).ValueOrUnchanged(stmt);
    for (const auto& [buf, source] : param_flattened_buffers) {
      new_stmt = SeqStmt::Flatten(DeclBuffer(buf, source.data()), std::move(new_stmt));
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
    if (auto buffer = any.as<BufferVar>()) {
      return GetFlattenedBuffer(buffer.value());
    } else if (auto prim_expr = any.as<PrimExpr>()) {
      return Mutate(prim_expr.value(), InplaceMode::kDisallow).ValueOrUnchanged(prim_expr.value());
    } else if (auto stmt = any.as<Stmt>()) {
      return Mutate(stmt.value(), InplaceMode::kDisallow).ValueOrUnchanged(stmt.value());
    }
    return any;
  }

  UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) final {
    if (op->op.same_as(builtin::buffer_data()) && op->args.size() == 1) {
      if (auto var = op->args[0].as<Var>();
          var.has_value() && var.value()->ty.as<BufferTypeNode>()) {
        auto root_opt = buffer_aliases_.Get(var.value());
        TVM_FFI_ICHECK(root_opt.has_value())
            << "buffer_data projects " << var.value()->name << ", which has no visible definition "
            << "(AllocBuffer/DeclBuffer/PrimFunc parameter) at this point";
        Var root = root_opt.value();
        if (auto mapped = VarRemapGet(root); mapped != nullptr) {
          root = mapped.as_or_throw<Var>();
        }
        return BufferVar(root).data();
      }
    }
    return IRMutatorWithAnalyzer::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_(const AllocBufferNode* op, InplaceMode inplace_mode) final {
    buffer_aliases_.Set(op->buffer.var(), op->buffer.var());
    auto mutate = [this](BufferVar buf) {
      if (target_->kind->name == "trn" && !buf->layout.has_value()) {
        return buf;
      }
      return GetFlattenedBuffer(buf, /*is_alloc=*/true);
    };
    auto buffer = mutate(op->buffer);
    if (buffer.same_as(op->buffer)) {
      return ffi::Unchanged();
    }
    if (inplace_mode == InplaceMode::kAllow) {
      auto* n = const_cast<AllocBufferNode*>(op);
      n->buffer = buffer;
      return ffi::Unchanged();
    }
    auto n = ffi::make_object<AllocBufferNode>(*op);
    n->buffer = buffer;
    return Stmt(n);
  }

  UnchangedOr<Stmt> Mutate_(const DeclBufferNode* op, InplaceMode inplace_mode) final {
    RegisterBufferAlias(op->buffer, op->data);
    auto data_result = Mutate(op->data, inplace_mode);
    bool data_unchanged = data_result.UnchangedOrSameAs(op->data);
    Expr data = std::move(data_result).ValueOrUnchanged(op->data);
    auto buffer = GetFlattenedBuffer(op->buffer);
    if (buffer.same_as(op->buffer) && data_unchanged) {
      return ffi::Unchanged();
    }
    return DeclBuffer(buffer, std::move(data), op->span);
  }

  BufferVar GetFlattenedBuffer(BufferVar buf, bool is_alloc = false) {
    if (auto mapped = VarRemapGet(buf); mapped != nullptr) {
      return mapped.as_or_throw<BufferVar>();
    }
    auto trn_layout = buf->layout.as<TileLayoutNode>();
    BufferVar flattened;
    ffi::ObjectPtr<BufferTypeNode> type;
    if (trn_layout && trn_layout->IsTrainium()) {
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
        // Logical alloc_buffer with thread axes: physical shape = memory-axis span
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
    type->layout = std::nullopt;
    type->elem_offset =
        StmtExprMutator::Mutate(ffi::AnyView(buf->elem_offset), InplaceMode::kDisallow)
            .ValueOrUnchanged(buf->elem_offset)
            .as_or_throw<PrimExpr>();
    if (ffi::StructuralEqual()(buf.type(), BufferType(type))) {
      return buf;
    }
    flattened = RebuildBufferVar(flattened, std::move(type));

    VarRemapSet(buf, flattened);
    return flattened;
  }

  UnchangedOr<Stmt> Mutate_(const BufferStoreNode* op, InplaceMode inplace_mode) final {
    // Preserve the logical buffer until VisitBufferAccess linearizes its indices.
    auto value = Mutate(op->value, inplace_mode);
    auto indices =
        Mutate(op->indices, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    BufferStore store = ffi::GetRef<BufferStore>(op);
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

  UnchangedOr<Stmt> Mutate_(const tirx::TilePrimitiveCallNode* op, InplaceMode inplace_mode) final {
    ffi::Array<ffi::Any> args = op->args;
    args.MutateByApply([this](ffi::Any arg) -> ffi::Any { return VisitAny(arg); });
    if (args.same_as(op->args)) {
      return ffi::Unchanged();
    } else {
      if (inplace_mode == InplaceMode::kAllow) {
        auto* n = const_cast<tirx::TilePrimitiveCallNode*>(op);
        n->args = std::move(args);
        return ffi::Unchanged();
      }
      auto n = ffi::make_object<tirx::TilePrimitiveCallNode>(*op);
      n->args = std::move(args);
      return Stmt(n);
    }
  }

  ffi::Array<PrimExpr> GetSimplifiedElemOffset(const BufferVar& buffer,
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
    if (target_->kind->name == "trn" && !node->buffer->layout.has_value()) {
      return node;
    }
    auto flattened_indices = GetSimplifiedElemOffset(node->buffer, node->indices);
    BufferVar flattened_buffer = GetFlattenedBuffer(node->buffer);
    auto writer = node.CopyOnWrite();
    writer->buffer = flattened_buffer;
    writer->indices = flattened_indices;
    return node;
  }

  TensorLoad VisitBufferAccess(TensorLoad node) {
    BufferVar buffer = node->source.as_or_throw<tvm::tirx::BufferVar>();
    TVM_FFI_ICHECK(buffer.defined());
    if (target_->kind->name == "trn" && !buffer->layout.has_value()) return node;
    return BufferLoad(GetFlattenedBuffer(buffer), GetSimplifiedElemOffset(buffer, node->indices),
                      node->span);
  }

  /*! \brief Map of variables being remapped, including buffer variables. */

 private:
  void RegisterBufferAlias(BufferVar buffer, const Expr& data) {
    Var root = buffer.var();
    if (const auto* call = data.as<CallNode>();
        call && call->op.same_as(builtin::buffer_data()) && call->args.size() == 1) {
      if (auto source = call->args[0].as<Var>();
          source.has_value() && source.value()->ty.as<BufferTypeNode>()) {
        auto source_root = buffer_aliases_.Get(source.value());
        TVM_FFI_ICHECK(source_root.has_value())
            << "Buffer alias source " << source.value()->name
            << " must be registered before its DeclBuffer alias";
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
    if (call->op.same_as(tirx::builtin::buffer_offset())) {
      auto buffer_load = call->args[0].as_or_throw<TensorLoad>();
      TVM_FFI_ICHECK_EQ(buffer_load->indices.size(), 1) << "Expected a single index";
      return buffer_load->indices[0];
    }
    return StmtExprMutator::Mutate_(call, inplace_mode);
  }
};

namespace {
Target ResolveTarget(const PrimFunc& f) {
  auto target = f->GetAttr<Target>(tvm::attr::kTarget);
  if (!target.has_value()) {
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
    auto [body, params] = LayoutApplier::Flatten(n->body, n->params, target);
    n->body = std::move(body);
    n->params = std::move(params);
    n->body = BufferOffsetRemover::Remove(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.LowerTIRxCleanup", {});
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
