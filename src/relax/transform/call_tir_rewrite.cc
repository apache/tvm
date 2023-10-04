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
 * \file src/relax/transform/call_tir_rewrite.cc
 * \brief Perform explicit tensor allocation for call_tir.
 */
#include <tvm/relax/attrs/op.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/tir/op.h>

#include "../../relay/transforms/pattern_utils.h"

namespace tvm {
namespace relax {

// ==================
// CallTIRMutator
// Perform explicit tensor allocation for call_tir, call_tir_inplace, or call_dps_packed.
// Example:
// lv0: Tensor(n, m) = rx.call_tir(func, (x), (n, m), dtype="float32")
// -->
// gv0 = rx.call("relax.builtin.alloc_tensor", [n, m], dtype="float32")
// rx.call_packed(func, x, gv0)

class CallTIRMutator : public ExprMutator {
 public:
  using ExprMutator::VisitExpr_;
  Expr VisitExpr_(const CallNode* call) override {
    // post-order mutation
    Expr expr = VisitExprPostOrder_(call);
    call = expr.as<CallNode>();

    static const Op& call_tir_op = Op::Get("relax.call_tir");
    static const Op& call_tir_inplace_op = Op::Get("relax.call_tir_inplace");
    static const Op& call_dps_packed_op = Op::Get("relax.call_dps_packed");
    static const Op& alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");
    static const Op& call_tir_dyn_op = Op::Get("relax.vm.call_tir_dyn");

    if (call->op == call_tir_op || call->op == call_tir_inplace_op ||
        call->op == call_dps_packed_op) {
      bool is_inplace_op = (call->op == call_tir_inplace_op);
      const auto* inplace_attrs = call->attrs.as<CallTIRInplaceAttrs>();
      Array<Expr> outs;
      if (const auto& _tensor_sinfo = MatchStructInfo<TensorStructInfo>(expr)) {
        // single output case
        const TensorStructInfo& tensor_sinfo = _tensor_sinfo.value();
        ICHECK(tensor_sinfo->shape.defined())
            << "the TensorStructInfo shape of call_tir has not populated";
        if (!is_inplace_op) {
          outs.push_back(
              builder_->Emit(Call(alloc_tensor_op,  //
                                  {Downcast<ShapeExpr>(tensor_sinfo->shape.value()),
                                   DataTypeImm(tensor_sinfo->dtype), PrimValue::Int64(0)},  //
                                  Attrs()),
                             "alloc"));
        } else {
          // if there is only one output, it must be an in-place argument, but check anyway
          ICHECK(inplace_attrs->inplace_indices[0].IntValue() != -1)
              << "If calling call_tir_inplace and there is one output, its in-place index must not"
                 " be -1.";
          outs.push_back(
              Downcast<Tuple>(call->args[1])->fields[inplace_attrs->inplace_indices[0].IntValue()]);
        }
      } else if (const auto& _tuple_sinfo = MatchStructInfo<TupleStructInfo>(expr)) {
        // multiple output case
        const TupleStructInfo& tuple_sinfo = _tuple_sinfo.value();
        for (size_t i = 0; i < tuple_sinfo->fields.size(); ++i) {
          const auto& field = tuple_sinfo->fields[i];

          ICHECK(field->IsInstance<TensorStructInfoNode>())
              << "call_tir expects Tuple of TensorStructInfo, but got " << field
              << " as an element of TupleStructInfo";
          const auto& field_tensor = Downcast<TensorStructInfo>(field);
          ICHECK(field_tensor->shape.defined())
              << "call_tir expects all TensorStructInfo has shape, but got " << field_tensor
              << " as an element of TupleStructInfo";
          if (!is_inplace_op || inplace_attrs->inplace_indices[i].IntValue() == -1) {
            outs.push_back(
                builder_->Emit(Call(alloc_tensor_op,
                                    {Downcast<ShapeExpr>(field_tensor->shape.value()),
                                     DataTypeImm(field_tensor->dtype), PrimValue::Int64(0)},
                                    Attrs()),
                               "alloc"));
          } else {
            outs.push_back(Downcast<Tuple>(call->args[1])
                               ->fields[inplace_attrs->inplace_indices[i].IntValue()]);
          }
        }
      } else {
        LOG(FATAL) << "TypeError: The struct info of call_tir expects to be TensorStructInfo or "
                      "TupleStructInfo, but got"
                   << expr->struct_info_;
      }

      Expr callee = call->args[0];
      Expr arg_tuple = call->args[1];
      Optional<Expr> shape_tuple_of_tir_args = NullOpt;
      if (call->args.size() > 2) {
        shape_tuple_of_tir_args = call->args[2];
      }

      while (true) {
        auto as_var = arg_tuple.as<Var>();
        if (!as_var) break;

        auto bound_expr = LookupBinding(as_var.value());
        if (!bound_expr) break;

        arg_tuple = bound_expr.value();
      }

      Array<Expr> args = [&]() {
        if (auto ptr = arg_tuple.as<TupleNode>()) {
          return ptr->fields;
        } else if (auto ptr = arg_tuple->struct_info_.as<TupleStructInfoNode>()) {
          size_t n_args = ptr->fields.size();
          Array<Expr> args;
          for (size_t i = 0; i < n_args; i++) {
            args.push_back(TupleGetItem(arg_tuple, i));
          }
          return args;
        } else {
          LOG(FATAL) << "Lowering of " << call
                     << " requires knowing how many arguments are passed to the function.  "
                     << "However, the tuple of arguments " << arg_tuple
                     << " is not itself a tuple, "
                     << "nor does its struct info " << GetStructInfo(arg_tuple)
                     << " define the number of arguments.";
        }
      }();

      for (size_t i = 0; i < outs.size(); i++) {
        bool output_is_inplace =
            is_inplace_op && inplace_attrs->inplace_indices[i].IntValue() != -1;
        if (!output_is_inplace) {
          args.push_back(outs[i]);
        }
      }

      if (shape_tuple_of_tir_args) {
        args.push_back(shape_tuple_of_tir_args.value());
      }

      Expr new_call = [&]() {
        if (shape_tuple_of_tir_args) {
          return Call(call_tir_dyn_op, {callee, Tuple(args)});
        } else {
          return Call(callee, args);
        }
      }();
      builder_->Emit(new_call, "_");

      if (outs.size() == 1) {
        return outs[0];
      } else {
        return Tuple(outs);
      }
    }

    return GetRef<Expr>(call);
  }
};

Expr CallTIRRewrite(const Expr& e) { return CallTIRMutator().VisitExpr(e); }

namespace transform {

Pass CallTIRRewrite() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) { return Downcast<Function>(CallTIRRewrite(f)); };
  return CreateFunctionPass(pass_func, 0, "CallTIRRewrite", {});
}

TVM_REGISTER_GLOBAL("relax.transform.CallTIRRewrite").set_body_typed(CallTIRRewrite);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
