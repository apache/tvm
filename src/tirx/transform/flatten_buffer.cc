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
 * \brief Flatten each n-d buffer ``buf`` into a 1-d storage view ``buf'``,
 *        rewriting every access ``buf[x]`` into ``buf'[f(x)]``.
 *
 *  The invariant: ``f(x) = layout.apply(x, shape) + elem_offset`` is fully
 *  determined by ``buf``'s geometry, and ``buf'`` is only a storage husk —
 *  same data origin, dtype, alignment and scope; no layout, no elem_offset.
 *
 *  The pass walks the AST top-down. At each buffer definition point
 *  (AllocBuffer/DeclBuffer; PrimFunc params are seeded up front) it derives,
 *  exactly once:
 *    - the fold view: the original geometry with its expression fields
 *      (runtime elem_offset, symbolic shapes/strides, layout iters) rewritten
 *      by the pass — the folded indices live in the rewritten program, so
 *      ``f``'s coefficients must reference rebuilt buffers; and
 *    - ``buf'``, the flattened storage husk.
 *  Every use site then only looks the pair up; a use before its definition is
 *  a hard error instead of a silently stale reference.
 */
class BufferFlattener : public arith::IRMutatorWithAnalyzer {
 public:
  static PrimFunc Flatten(PrimFunc func) {
    arith::Analyzer ana;
    auto pass = BufferFlattener(ana);
    pass.MarkBufferParamShapes(func);
    for (const Var& param : func->params) {
      if (auto buffer = param.as<BufferVar>()) {
        pass.extern_buffers_.insert(buffer.value());
        pass.Define(buffer.value());
      }
    }
    auto body = pass.VisitStmt(func->body);

    // Buffer parameters are deliberately left unflattened, as they are used
    // for validation of user-provided arguments.  The flattened buffers used
    // in the updated function body alias the argument buffers.
    for (size_t i = func->params.size(); i > 0; i--) {
      if (auto old_buf = func->params[i - 1].as<BufferVar>()) {
        if (pass.buffers_used_.count(old_buf.value())) {
          auto new_buf = pass.Lookup(old_buf.value()).flattened;
          if (!old_buf.value().same_as(new_buf)) {
            body = SeqStmt::Flatten(DeclBuffer(new_buf, old_buf.value().data()), std::move(body));
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

  struct FlatInfo {
    /*! \brief Original geometry with rewritten expression fields; the source
     *   of ``f``. Only used to fold indices, never emitted into the IR. */
    BufferVar fold_view;
    /*! \brief The 1-d storage husk ``buf'``. */
    BufferVar flattened;
  };

  /*! \brief Derive {fold view, flattened husk} for ``buf`` at its definition
   *   point. Idempotent so params can be seeded up front. */
  const FlatInfo& Define(const BufferVar& buf) {
    if (auto it = flat_map_.find(buf.var()); it != flat_map_.end()) {
      return it->second;
    }

    // Fold view: rewrite the geometry's expression leaves.
    auto view_type = CopyBufferType(buf);
    for (size_t i = 0; i < view_type->shape.size(); ++i) {
      view_type->shape.Set(i, this->VisitPrimExpr(view_type->shape[i]));
    }
    for (size_t i = 0; i < view_type->strides.size(); ++i) {
      view_type->strides.Set(i, this->VisitPrimExpr(view_type->strides[i]));
    }
    if (view_type->elem_offset.defined()) {
      view_type->elem_offset = this->VisitPrimExpr(view_type->elem_offset);
    }
    if (auto tile = view_type->layout.as<TileLayoutNode>()) {
      auto remap_iter = [this](const Iter& iter) {
        PrimExpr extent = this->VisitPrimExpr(iter->extent);
        PrimExpr stride = this->VisitPrimExpr(iter->stride);
        if (extent.same_as(iter->extent) && stride.same_as(iter->stride)) {
          return iter;
        }
        return Iter(extent, stride, iter->axis);
      };
      auto shard = tile->shard.Map(remap_iter);
      auto replica = tile->replica.Map(remap_iter);
      if (!shard.same_as(tile->shard) || !replica.same_as(tile->replica)) {
        view_type->layout = TileLayout(shard, replica, tile->offset);
      }
    }
    BufferVar fold_view = RebuildBufferVar(buf, std::move(view_type));

    // buf': the storage husk. The linearized indices carry layout and
    // elem_offset, so the husk keeps neither.
    auto flat = fold_view.GetFlattenedBuffer();
    auto type = CopyBufferType(flat);
    for (size_t i = 0; i < type->shape.size(); ++i) {
      type->shape.Set(i, analyzer_->canonical_simplify(type->shape[i]));
    }
    type->layout = std::nullopt;
    if (type->elem_offset.defined() && !is_zero(type->elem_offset)) {
      type->elem_offset = IntImm(type->elem_offset.ty().as_or_throw<PrimType>(), 0);
    }
    // Body-local buffers keep their identity when flattening changes nothing.
    // PrimFunc-parameter buffers always rebuild: the epilogue aliases the
    // rebuilt view onto the argument buffer with an explicit DeclBuffer, and
    // downstream s_tir passes pin that shape.
    BufferVar flattened =
        (!extern_buffers_.count(buf) && ffi::StructuralEqual()(BufferType(type), buf.type()))
            ? buf
            : RebuildBufferVar(buf, std::move(type));

    // Feed the base mutator's remap so stray buffer-var expressions follow.
    buffer_remap_.Set(buf, flattened);
    auto [it, inserted] = flat_map_.emplace(buf.var(), FlatInfo{fold_view, flattened});
    return it->second;
  }

  const FlatInfo& Lookup(const BufferVar& buf) {
    auto it = flat_map_.find(buf.var());
    TVM_FFI_ICHECK(it != flat_map_.end())
        << "Buffer " << buf.name()
        << " is used before its definition (AllocBuffer/DeclBuffer/PrimFunc param)";
    return it->second;
  }

  Stmt VisitStmt_(const SBlockNode* op) final {
    TVM_FFI_ICHECK_EQ(op->match_buffers.size(), 0)
        << "Unexpected MatchBufferRegion found during tirx.transform.FlattenBuffer.  "
        << "All MatchBufferRegion should be removed in tirx.transform.LowerMatchBuffer.";

    SBlock block = ffi::GetRef<SBlock>(op);

    ffi::Array<BufferVar> alloc_buffers = op->alloc_buffers;
    alloc_buffers.MutateByApply([this](BufferVar buf) { return Define(buf).flattened; });
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
    const FlatInfo& info = Define(op->buffer);
    if (info.flattened.same_as(op->buffer)) {
      return ffi::GetRef<Stmt>(op);
    }
    auto n = CopyOnWrite(op);
    n->buffer = info.flattened;
    return Stmt(n);
  }

  Stmt VisitStmt_(const DeclBufferNode* op) final {
    Expr data = op->data;
    bool is_extern_buffer_source = false;
    if (const auto* call = op->data.as<CallNode>();
        call && call->op.same_as(builtin::buffer_data()) && call->args.size() == 1) {
      if (const auto* var = call->args[0].as<VarNode>(); var && var->ty.as<BufferTypeNode>()) {
        is_extern_buffer_source = extern_buffers_.count(BufferVar(ffi::GetRef<Var>(var)));
      }
    }
    if (!is_extern_buffer_source) {
      data = VisitExpr(op->data);
    }
    const FlatInfo& info = Define(op->buffer);
    if (info.flattened.same_as(op->buffer) && data.same_as(op->data)) {
      return ffi::GetRef<Stmt>(op);
    }
    return DeclBuffer(info.flattened, std::move(data), op->span);
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

  Expr VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::masked_load()) || op->op.same_as(builtin::masked_store())) {
      bool is_load = op->op.same_as(builtin::masked_load());
      BufferVar original(op->args[0].as_or_throw<Var>());
      ffi::Array<PrimExpr> indices;
      for (size_t i = is_load ? 1 : 2; i + 1 < op->args.size(); ++i) {
        indices.push_back(this->VisitPrimExpr(op->args[i].as_or_throw<PrimExpr>()));
      }
      buffers_used_.insert(original);
      const FlatInfo& info = Lookup(original);
      ffi::Array<Expr> args{info.flattened.var()};
      if (!is_load) args.push_back(this->VisitExpr(op->args[1]));
      for (const PrimExpr& index : FoldIndices(info, indices)) args.push_back(index);
      args.push_back(this->VisitExpr(op->args.back()));
      return Call(op->ty, op->op, args, op->attrs, op->ty_args, op->span);
    }
    if (op->op.same_as(builtin::buffer_data()) && op->args.size() == 1) {
      if (auto var = op->args[0].as<Var>()) {
        if (var.value()->ty.as<BufferTypeNode>()) {
          BufferVar original(var.value());
          buffers_used_.insert(original);
          return Lookup(original).flattened.data();
        }
      }
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  ffi::Array<PrimExpr> FoldIndices(const FlatInfo& info, const ffi::Array<PrimExpr>& indices) {
    auto flattened_indices = info.fold_view->ElemOffset(indices);
    return this->IterMapSimplifyWithContext(flattened_indices, false);
  }

  template <typename Node>
  Node VisitBufferAccess(Node node, const BufferVar& original_buffer) {
    TVM_FFI_ICHECK(node->buffer.defined());
    buffers_used_.insert(original_buffer);
    const FlatInfo& info = Lookup(original_buffer);
    auto flattened_indices = FoldIndices(info, node->indices);

    auto writer = node.CopyOnWrite();
    writer->buffer = info.flattened;
    writer->indices = flattened_indices;
    return node;
  }

  BufferRegion MutateBufferRegion(BufferRegion region) {
    const FlatInfo& info = Lookup(region->buffer);
    if (info.flattened.same_as(region->buffer)) {
      return region;
    }

    ffi::Array<PrimExpr> min_values;
    ffi::Array<PrimExpr> max_values;
    for (const auto& range : region->region) {
      min_values.push_back(range->min);
      max_values.push_back(range->min + range->extent - 1);
    }

    ffi::Array<PrimExpr> flattened_min = FoldIndices(info, min_values);
    ffi::Array<PrimExpr> flattened_max = FoldIndices(info, max_values);

    ffi::Array<Range> flattened_ranges;
    TVM_FFI_ICHECK_EQ(flattened_min.size(), flattened_max.size());
    for (size_t i = 0; i < flattened_min.size(); i++) {
      flattened_ranges.push_back(Range(flattened_min[i], flattened_max[i] + 1));
    }

    return BufferRegion(info.flattened, flattened_ranges);
  }

  /*! \brief Set of buffers accessed during visitation (used to emit DeclBuffer for param buffers).
   */
  std::unordered_set<BufferVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> buffers_used_;

  /*! \brief Buffers whose storage is supplied by a PrimFunc parameter. */
  std::unordered_set<BufferVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> extern_buffers_;

  /*! \brief Per-buffer {fold view, flattened husk}, derived at definition points. */
  std::unordered_map<Var, FlatInfo, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> flat_map_;
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
