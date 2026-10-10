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
 * \file flatten_tensor.cc
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/type.h>
#include <tvm/sym/iter_affine_map.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_set>

#include "../ir/ir_mutator_with_analyzer.h"
#include "flattened_tensor.h"
#include "ir_utils.h"

namespace tvm {
namespace tirx {
using namespace tvm::prim;

/*!
 * \brief Flatten each n-d tensor ``buf`` into a 1-d storage view ``buf'``,
 *        rewriting every access ``buf[x]`` into ``buf'[f(x)]``.
 *
 *  The invariant: ``f(x) = layout.apply(x, shape) + elem_offset`` is fully
 *  determined by ``buf``'s geometry, and ``buf'`` is only a storage husk —
 *  same data origin, dtype, alignment and scope; no layout, no elem_offset.
 *
 *  The pass walks the AST top-down. At each tensor definition point
 *  (AllocTensor/DeclTensor; Function params are seeded up front) it derives,
 *  exactly once:
 *    - the fold view: the original geometry with its expression fields
 *      (runtime elem_offset, symbolic shapes/strides, layout iters) rewritten
 *      by the pass — the folded indices live in the rewritten program, so
 *      ``f``'s coefficients must reference rebuilt tensors; and
 *    - ``buf'``, the flattened storage husk.
 *  Every use site then only looks the pair up; a use before its definition is
 *  a hard error instead of a silently stale reference.
 */
class TensorFlattener : public IRMutatorWithAnalyzer {
 public:
  using IRMutatorWithAnalyzer::Mutate;
  using IRMutatorWithAnalyzer::Mutate_;
  static Function Flatten(Function func) {
    if (!func->body.has_value()) return func;
    sym::Analyzer ana;
    auto pass = ffi::make_object<TensorFlattener>(ana);
    pass->MarkTensorParamShapes(func);
    for (const Var& param : func->params) {
      if (auto tensor = param.as<TensorVar>()) {
        pass->extern_tensors_.insert(tensor.value());
        pass->Define(tensor.value());
      }
    }
    auto body_result = pass->Mutate(func->body, InplaceMode::kDisallow);
    bool body_unchanged = body_result.UnchangedOrSameAs(func->body);
    auto body = std::move(body_result).ValueOrUnchanged(func->body);

    // Tensor parameters are deliberately left unflattened, as they are used
    // for validation of user-provided arguments.  The flattened tensors used
    // in the updated function body alias the argument tensors.
    for (size_t i = func->params.size(); i > 0; i--) {
      if (auto old_buf = func->params[i - 1].as<TensorVar>()) {
        if (pass->tensors_used_.count(old_buf.value())) {
          auto new_buf = pass->Lookup(old_buf.value()).flattened;
          if (!old_buf.value().same_as(new_buf)) {
            body = SeqStmt({Bind(new_buf, Call(new_buf.type(), decl_tensor_op(),
                                               {old_buf.value().data(), tvm::Tuple(new_buf->shape),
                                                DataTypeImm(new_buf->dtype->dtype),
                                                StringImm(new_buf.scope())},
                                               {})),
                            std::move(body).value()});
            body_unchanged = false;
          }
        }
      }
    }

    if (!body_unchanged) {
      func.CopyOnWrite()->body = std::move(body);
    }
    return func;
  }

 public:
  explicit TensorFlattener(const sym::Analyzer& ana) : IRMutatorWithAnalyzer(ana) {}

 private:
  struct FlatInfo {
    /*! \brief Original geometry with rewritten expression fields; the source
     *   of ``f``. Only used to fold indices, never emitted into the IR. */
    TensorVar fold_view;
    /*! \brief The 1-d storage husk ``buf'``. */
    TensorVar flattened;
  };

  /*! \brief Derive {fold view, flattened husk} for ``buf`` at its definition
   *   point. Idempotent so params can be seeded up front. */
  const FlatInfo& Define(const TensorVar& buf) {
    if (auto it = flat_map_.find(buf.var()); it != flat_map_.end()) {
      return it->second;
    }

    // Fold view: rewrite the geometry's expression leaves.
    auto view_type = CopyTensorType(buf);
    auto mutate_expr = [this](const PrimExpr& expr) { return Mutate(expr).ValueOrUnchanged(expr); };
    view_type->shape = view_type->shape.Map(mutate_expr);
    view_type->strides = view_type->strides.Map(mutate_expr);
    if (view_type->elem_offset.defined()) {
      view_type->elem_offset = this->Mutate(view_type->elem_offset, InplaceMode::kDisallow)
                                   .ValueOrUnchanged(view_type->elem_offset);
    }
    if (auto tile = view_type->layout.as<TileLayoutNode>()) {
      auto remap_iter = [this](const Iter& iter) {
        PrimExpr extent =
            this->Mutate(iter->extent, InplaceMode::kDisallow).ValueOrUnchanged(iter->extent);
        PrimExpr stride =
            this->Mutate(iter->stride, InplaceMode::kDisallow).ValueOrUnchanged(iter->stride);
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
    TensorVar fold_view = RebuildTensorVar(buf, std::move(view_type));

    // buf': the storage husk. The linearized indices carry layout and
    // elem_offset, so the husk keeps neither.
    auto flat = FlattenedTensor(fold_view);
    auto type = CopyTensorType(flat);
    for (size_t i = 0; i < type->shape.size(); ++i) {
      type->shape.Set(i, analyzer_->canonical_simplify(type->shape[i]));
    }
    type->layout = std::nullopt;
    if (type->elem_offset.defined() && !IsZero(type->elem_offset)) {
      type->elem_offset = IntImm(type->elem_offset.ty().as_or_throw<PrimType>(), 0);
    }
    // Body-local tensors keep their identity when flattening changes nothing.
    // Function-parameter tensors always rebuild: the epilogue aliases the
    // rebuilt view onto the argument tensor with an explicit DeclTensor, and
    // downstream s_tir passes pin that shape.
    TensorVar flattened =
        (!extern_tensors_.count(buf) && ffi::StructuralEqual()(TensorType(type), buf.type()))
            ? buf
            : RebuildTensorVar(buf, std::move(type));

    // Feed the base mutator's remap so stray tensor-var expressions follow.
    VarRemapSet(buf, flattened);
    auto [it, inserted] = flat_map_.emplace(buf.var(), FlatInfo{fold_view, flattened});
    return it->second;
  }

  const FlatInfo& Lookup(const TensorVar& buf) {
    auto it = flat_map_.find(buf.var());
    TVM_FFI_ICHECK(it != flat_map_.end())
        << "Tensor " << buf.name()
        << " is used before its definition (AllocTensor/DeclTensor/Function param)";
    return it->second;
  }

  UnchangedOr<Stmt> Mutate_(const BindNode* op, InplaceMode inplace_mode) final {
    if (const auto* call = op->value.as<CallNode>(); call) {
      if (call->op.same_as(alloc_tensor_op())) return MutateAllocTensor(op, call, inplace_mode);
      if (call->op.same_as(decl_tensor_op())) return MutateDeclTensor(op, call, inplace_mode);
    }
    return IRMutatorWithAnalyzer::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> MutateAllocTensor(const BindNode* op, const CallNode* tensor_call,
                                      InplaceMode inplace_mode) {
    const FlatInfo& info = Define(op->var.as_or_throw<TensorVar>());
    ffi::Array<Expr> args = tensor_call->args;
    if (args.size() == 4) {
      args.Set(3, Mutate(args[3], inplace_mode).ValueOrUnchanged(args[3]));
    }
    if (info.flattened.same_as(op->var.as_or_throw<TensorVar>()) &&
        args.same_as(tensor_call->args)) {
      return ffi::Unchanged();
    }
    args.Set(0, tvm::Tuple(info.flattened->shape, tensor_call->args[0]->loc));
    args.Set(1, DataTypeImm(info.flattened->dtype->dtype, tensor_call->args[1]->loc));
    args.Set(2, StringImm(info.flattened.scope(), tensor_call->args[2]->loc));
    return Bind(info.flattened.var(),
                Call(info.flattened.type(), tirx::alloc_tensor_op(), args, tensor_call->attrs,
                     tensor_call->ty_args, tensor_call->loc),
                op->loc);
  }

  UnchangedOr<Stmt> MutateDeclTensor(const BindNode* op, const CallNode* tensor_call,
                                     InplaceMode inplace_mode) {
    Expr data = tensor_call->args[0];
    bool is_extern_tensor_source = false;
    if (const auto* call = tensor_call->args[0].as<CallNode>();
        call && call->op.same_as(tensor_data_ptr_op()) && call->args.size() == 1) {
      if (const auto* var = call->args[0].as<VarNode>(); var && var->ty.as<TensorTypeNode>()) {
        is_extern_tensor_source =
            extern_tensors_.count(ffi::GetRef<Var>(var).as_or_throw<TensorVar>());
      }
    }
    if (!is_extern_tensor_source) {
      data = Mutate(tensor_call->args[0], inplace_mode).ValueOrUnchanged(tensor_call->args[0]);
    }
    const FlatInfo& info = Define(op->var.as_or_throw<TensorVar>());
    if (info.flattened.same_as(op->var.as_or_throw<TensorVar>()) &&
        data.same_as(tensor_call->args[0])) {
      return ffi::Unchanged();
    }
    return Bind(info.flattened,
                Call(info.flattened.type(), decl_tensor_op(),
                     {std::move(data), tvm::Tuple(info.flattened->shape),
                      DataTypeImm(info.flattened->dtype->dtype), StringImm(info.flattened.scope())},
                     tensor_call->attrs, tensor_call->ty_args, tensor_call->loc),
                op->loc);
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreNode* op, InplaceMode inplace_mode) final {
    // The tensor and its indices must be flattened together by VisitTensorAccess.
    auto value = Mutate(op->value, inplace_mode);
    auto indices =
        Mutate(op->indices, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorStore store = ffi::GetRef<TensorStore>(op);
    if (!value.UnchangedOrSameAs(op->value) || !indices.UnchangedOrSameAs(op->indices)) {
      auto* n = store.CopyOnWrite();
      n->value = std::move(value).ValueOrUnchanged(op->value);
      n->indices = std::move(indices).ValueOrUnchanged(op->indices);
    }
    return VisitTensorAccess(std::move(store), op->dest.as_or_throw<TensorVar>());
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* op, InplaceMode inplace_mode) final {
    auto indices =
        Mutate(op->indices, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorLoad load = ffi::GetRef<TensorLoad>(op);
    if (!indices.UnchangedOrSameAs(op->indices)) {
      load.CopyOnWrite()->indices = std::move(indices).ValueUnchecked();
    }
    return VisitTensorAccess(std::move(load), op->source.as_or_throw<TensorVar>());
  }

  UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) final {
    if (op->op.same_as(masked_load_op()) || op->op.same_as(masked_store_op())) {
      bool is_load = op->op.same_as(masked_load_op());
      TensorVar original = op->args[0].as_or_throw<TensorVar>();
      ffi::Array<PrimExpr> indices;
      for (size_t i = is_load ? 1 : 2; i + 1 < op->args.size(); ++i) {
        indices.push_back(
            this->Mutate(op->args[i]).ValueOrUnchanged(op->args[i]).as_or_throw<PrimExpr>());
      }
      tensors_used_.insert(original);
      const FlatInfo& info = Lookup(original);
      ffi::Array<Expr> args{info.flattened.var()};
      if (!is_load)
        args.push_back(this->Mutate(op->args[1]).ValueOrUnchanged(op->args[1]).as_or_throw<Expr>());
      for (const PrimExpr& index : FoldIndices(info, indices)) args.push_back(index);
      args.push_back(this->Mutate(op->args[op->args.size() - 1])
                         .ValueOrUnchanged(op->args[op->args.size() - 1])
                         .as_or_throw<Expr>());
      return Call(op->ty, op->op, args, op->attrs, op->ty_args, op->loc);
    }
    if (op->op.same_as(tensor_data_ptr_op()) && op->args.size() == 1) {
      if (auto var = op->args[0].as<Var>()) {
        if (var.value()->ty.as<TensorTypeNode>()) {
          TensorVar original = var.value().as_or_throw<TensorVar>();
          tensors_used_.insert(original);
          return Lookup(original).flattened.data();
        }
      }
    }
    return IRMutatorWithAnalyzer::Mutate_(op, inplace_mode);
  }

  ffi::Array<PrimExpr> FoldIndices(const FlatInfo& info, const ffi::Array<PrimExpr>& indices) {
    auto flattened_indices = info.fold_view->ElemOffset(indices);
    return this->IterMapSimplifyWithContext(flattened_indices, false);
  }

  template <typename Node>
  Node VisitTensorAccess(Node node, const TensorVar& original_tensor) {
    TVM_FFI_ICHECK(node->dest.template as_or_throw<TensorVar>().defined());
    tensors_used_.insert(original_tensor);
    const FlatInfo& info = Lookup(original_tensor);
    auto flattened_indices = FoldIndices(info, node->indices);

    auto writer = node.CopyOnWrite();
    writer->dest = info.flattened;
    writer->indices = flattened_indices;
    return node;
  }

  TensorLoad VisitTensorAccess(TensorLoad node, const TensorVar& original_tensor) {
    tensors_used_.insert(original_tensor);
    const FlatInfo& info = Lookup(original_tensor);
    return MakeTensorLoad(info.flattened, FoldIndices(info, node->indices), node->loc);
  }

  /*! \brief Set of tensors accessed during visitation (used to emit DeclTensor for param tensors).
   */
  std::unordered_set<TensorVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> tensors_used_;

  /*! \brief Tensors whose storage is supplied by a Function parameter. */
  std::unordered_set<TensorVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> extern_tensors_;

  /*! \brief Per-tensor {fold view, flattened husk}, derived at definition points. */
  std::unordered_map<Var, FlatInfo, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> flat_map_;
};

Function FlattenTensor(Function f) { return TensorFlattener::Flatten(f); }

namespace transform {

Pass FlattenTensor() {
  auto pass_func = [=](Function f, IRModule m, PassContext ctx) {
    return FlattenTensor(std::move(f));
  };
  return CreateFunctionPass(pass_func, 0, "tirx.FlattenTensor");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.FlattenTensor", FlattenTensor);
}
}  // namespace transform

}  // namespace tirx
}  // namespace tvm
