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
#include <tvm/tirx/tile_primitive.h>
#include <tvm/tirx/transform.h>

#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../arith/ir_mutator_with_analyzer.h"
#include "ir_utils.h"

namespace tvm {
namespace tirx {

class LayoutApplier : public arith::IRMutatorWithAnalyzer {
 public:
  static std::pair<Stmt, ffi::Map<Var, BufferVar>> Flatten(
      const Stmt& stmt, const ffi::Array<Var>& params,
      const ffi::Map<tirx::Var, BufferVar> buffer_map, const Target& target) {
    arith::Analyzer ana;
    LayoutApplier storage_lower(ana, target);
    for (const Var& param : params) {
      if (param->ty.as<BufferTypeNode>()) {
        storage_lower.buffer_aliases_.Set(param, param);
      }
    }
    std::unordered_map<Var, BufferVar> new_buffer_map;
    std::vector<std::pair<BufferVar, BufferVar>> param_flattened_buffers;
    for (const auto& kv : buffer_map) {
      storage_lower.buffer_aliases_.Set(kv.second.var(), kv.second.var());
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

  explicit LayoutApplier(const arith::Analyzer& analyzer, const Target& target)
      : arith::IRMutatorWithAnalyzer(analyzer), target_(target) {}

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

  Expr VisitExpr_(const VarNode* op) final {
    Var var = ffi::GetRef<Var>(op);
    if (auto it = var_remap_.find(var); it != var_remap_.end()) {
      return it->second;
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  Expr VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::buffer_data()) && op->args.size() == 1) {
      if (auto var = op->args[0].as<Var>();
          var.has_value() && var.value()->ty.as<BufferTypeNode>()) {
        auto root_opt = buffer_aliases_.Get(var.value());
        TVM_FFI_ICHECK(root_opt.has_value())
            << "buffer_data projects " << var.value()->name << ", which has no visible definition "
            << "(AllocBuffer/DeclBuffer/PrimFunc parameter) at this point";
        Var root = root_opt.value();
        if (auto it = var_remap_.find(root); it != var_remap_.end()) {
          root = it->second;
        }
        return BufferVar(root).data();
      }
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  Stmt VisitStmt_(const AllocBufferNode* op) final {
    buffer_aliases_.Set(op->buffer.var(), op->buffer.var());
    auto mutate = [this](BufferVar buf) {
      if (target_->kind->name == "trn" && !buf->layout.has_value()) {
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
    RegisterBufferAlias(op->buffer, op->data);
    Expr data = VisitExpr(op->data);
    auto buffer = GetFlattenedBuffer(op->buffer);
    if (buffer.same_as(op->buffer) && data.same_as(op->data)) {
      return ffi::GetRef<Stmt>(op);
    }
    return DeclBuffer(buffer, std::move(data), op->span);
  }

  BufferVar GetFlattenedBuffer(BufferVar buf, bool is_alloc = false) {
    auto it = var_remap_.find(buf.var());
    if (it != var_remap_.end()) {
      return BufferVar(it->second);
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
    // Remap variables the pass has already rebuilt (a shape may load from
    // another local buffer), then canonicalize.
    for (size_t i = 0; i < type->shape.size(); ++i) {
      type->shape.Set(
          i, analyzer_->canonical_simplify(StmtExprMutator::VisitPrimExpr(type->shape[i])));
    }
    for (size_t i = 0; i < type->strides.size(); ++i) {
      type->strides.Set(i, StmtExprMutator::VisitPrimExpr(type->strides[i]));
    }
    type->layout = std::nullopt;
    type->elem_offset = StmtExprMutator::VisitPrimExpr(buf->elem_offset);
    if (ffi::StructuralEqual()(buf.type(), BufferType(type))) {
      return buf;
    }
    flattened = RebuildBufferVar(flattened, std::move(type));

    var_remap_[buf.var()] = flattened.var();
    return flattened;
  }

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

  /*! \brief Map of variables being remapped, including buffer variables. */
  std::unordered_map<Var, Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> var_remap_;

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
  static Stmt Remove(const Stmt& stmt) { return BufferOffsetRemover()(stmt); }

 private:
  Expr VisitExpr_(const VarNode* op) final {
    Var var = ffi::GetRef<Var>(op);
    if (auto it = var_remap_.find(var); it != var_remap_.end()) {
      return it->second;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

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
    Expr data = VisitExpr(op->data);
    auto elem_offset = this->VisitPrimExpr(buffer->elem_offset);
    if (!elem_offset.same_as(buffer->elem_offset)) {
      auto type = CopyBufferType(buffer);
      type->elem_offset = std::move(elem_offset);
      buffer = RebuildBufferVar(buffer, std::move(type));
      var_remap_[op->buffer.var()] = buffer.var();
    }
    if (buffer.same_as(op->buffer) && data.same_as(op->data)) {
      return ffi::GetRef<Stmt>(op);
    }
    return DeclBuffer(buffer, std::move(data), op->span);
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
    auto it = var_remap_.find(node->buffer.var());
    if (it != var_remap_.end()) {
      auto writer = node.CopyOnWrite();
      writer->buffer = BufferVar(it->second);
      return node;
    }
    return node;
  }

  std::unordered_map<Var, Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> var_remap_;
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
    std::tie(n->body, n->buffer_map) =
        LayoutApplier::Flatten(n->body, n->params, n->buffer_map, target);
    n->body = BufferOffsetRemover::Remove(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.LowerTIRxCleanup", {});
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
