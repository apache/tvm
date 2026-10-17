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
#include <tvm/tirx/stmt_functor_with_analyzer.h>
#include <tvm/tirx/transform.h>

#include <unordered_set>

#include "flattened_tensor.h"
#include "ir_utils.h"

namespace tvm {
namespace tirx {
using namespace tvm::prim;

/*!
 * \brief Flatten each n-d tensor ``buf`` into a 1-d storage view ``buf'``,
 *        rewriting every access ``buf[x]`` into ``buf'[f(x)]``.
 *
 *  The invariant: ``f(x) = layout.apply(x, shape)`` is fully
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
class TensorFlattener : public StmtExprMutatorWithAnalyzer {
 public:
  using StmtExprMutatorWithAnalyzer::Mutate;
  using StmtExprMutatorWithAnalyzer::Mutate_;
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
            body = SeqStmt({Bind(new_buf, Call(new_buf.ty(), decl_tensor_op(),
                                               {old_buf.value().data()}, {}, {new_buf.ty()})),
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
  explicit TensorFlattener(const sym::Analyzer& ana) : StmtExprMutatorWithAnalyzer(ana) {}

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
    if (auto it = flat_map_.find(buf); it != flat_map_.end()) {
      return it->second;
    }

    // Fold view: rewrite the geometry's expression leaves.
    auto view_type = ffi::make_object<TensorTypeNode>(*buf.ty().get());
    auto mutate_expr = [this](const PrimExpr& expr) { return Mutate(expr).ValueOrUnchanged(expr); };
    view_type->shape = view_type->shape.Map(mutate_expr);
    if (view_type->layout.has_value()) {
      view_type->layout = this->Mutate(ffi::AnyView(view_type->layout), InplaceMode::kDisallow)
                              .ValueOrUnchanged(view_type->layout)
                              .cast<ffi::Optional<Layout>>();
    }
    TensorVar fold_view = TensorVar(buf->name, TensorType(std::move(view_type)), buf->loc);

    // buf': the storage husk. The linearized indices carry layout and
    // elem_offset, so the husk keeps neither.
    auto flat = FlattenedTensor(fold_view);
    auto type = ffi::make_object<TensorTypeNode>(*flat.ty().get());
    for (size_t i = 0; i < type->shape.size(); ++i) {
      type->shape.Set(i, analyzer_->canonical_simplify(type->shape[i]));
    }
    type->layout = std::nullopt;
    // Body-local tensors keep their identity when flattening changes nothing.
    // Function-parameter tensors always rebuild: the epilogue aliases the
    // rebuilt view onto the argument tensor with an explicit DeclTensor, and
    // downstream s_tir passes pin that shape.
    TensorVar flattened =
        (!extern_tensors_.count(buf) && ffi::StructuralEqual()(TensorType(type), buf.ty()))
            ? buf
            : TensorVar(buf->name, TensorType(std::move(type)), buf->loc);

    // Feed the base mutator's remap so stray tensor-var expressions follow.
    VarRemapSet(buf, flattened);
    auto [it, inserted] = flat_map_.emplace(buf, FlatInfo{fold_view, flattened});
    return it->second;
  }

  const FlatInfo& Lookup(const TensorVar& buf) {
    auto it = flat_map_.find(buf);
    TVM_FFI_ICHECK(it != flat_map_.end())
        << "Tensor " << buf->name
        << " is used before its definition (AllocTensor/DeclTensor/Function param)";
    return it->second;
  }

  UnchangedOr<Stmt> Mutate_(const BindNode* op, InplaceMode inplace_mode) final {
    if (const auto* call = op->value.as<CallNode>(); call) {
      if (call->op.same_as(alloc_tensor_op())) return MutateAllocTensor(op, call, inplace_mode);
      if (call->op.same_as(decl_tensor_op())) return MutateDeclTensor(op, call, inplace_mode);
    }
    return StmtExprMutatorWithAnalyzer::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> MutateAllocTensor(const BindNode* op, const CallNode* tensor_call,
                                      InplaceMode inplace_mode) {
    const FlatInfo& info = Define(op->var.as_or_throw<TensorVar>());
    ffi::Array<Expr> args = tensor_call->args;
    if (!args.empty()) {
      args.Set(0, Mutate(args[0], inplace_mode).ValueOrUnchanged(args[0]));
    }
    if (info.flattened.same_as(op->var.as_or_throw<TensorVar>()) &&
        args.same_as(tensor_call->args)) {
      return ffi::Unchanged();
    }

    return Bind(info.flattened,
                Call(info.flattened.ty(), tirx::alloc_tensor_op(), args, tensor_call->attrs,
                     {info.flattened.ty()}, tensor_call->loc),
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
                Call(info.flattened.ty(), decl_tensor_op(), {std::move(data)}, tensor_call->attrs,
                     {info.flattened.ty()}, tensor_call->loc),
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
      ffi::Array<Expr> args{info.flattened};
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
    return StmtExprMutatorWithAnalyzer::Mutate_(op, inplace_mode);
  }

  ffi::Array<PrimExpr> FoldIndices(const FlatInfo& info, const ffi::Array<PrimExpr>& indices) {
    auto flattened_indices = info.fold_view.ty()->ElemOffset(indices);
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
    return TensorLoad(info.flattened, FoldIndices(info, node->indices), node->loc);
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
