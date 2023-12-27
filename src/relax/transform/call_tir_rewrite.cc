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
 * \brief Perform explicit tensor allocation for call_tir,
 *        call_tir_inplace, and call_dps_packed.
 */
#include <tvm/relax/attrs/op.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/tir/op.h>

#include "../../relay/transforms/pattern_utils.h"
#include "utils.h"

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
  explicit CallTIRMutator(const IRModule& mod) : ExprMutator(mod), mod_(std::move(mod)) {}

  IRModule Run() {
    for (const auto& [gv, func] : mod_->functions) {
      if (func->IsInstance<FunctionNode>()) {
        auto updated_func = Downcast<Function>(this->VisitExpr(func));
        builder_->UpdateFunction(gv, Downcast<BaseFunc>(updated_func));
      }
    }
    return builder_->GetContextIRModule();
  }

 private:
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
      bool is_inplace = (call->op == call_tir_inplace_op);
      const auto* inplace_attrs = call->attrs.as<CallTIRInplaceAttrs>();
      Array<Expr> outs;
      if (const auto& _tensor_sinfo = MatchStructInfo<TensorStructInfo>(expr)) {
        // single output case
        const TensorStructInfo& tensor_sinfo = _tensor_sinfo.value();
        ICHECK(tensor_sinfo->shape.defined())
            << "the TensorStructInfo shape of call_tir has not populated";
        int dev_index = 0;
        if (tensor_sinfo->vdevice.defined()) {
          dev_index = GetDeviceIndex(mod_, tensor_sinfo->vdevice.value());
        }
        if (!is_inplace) {
          outs.push_back(
              builder_->Emit(Call(alloc_tensor_op,
                                  {Downcast<ShapeExpr>(tensor_sinfo->shape.value()),
                                   DataTypeImm(tensor_sinfo->dtype), PrimValue::Int64(dev_index)},
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
          if (!is_inplace || inplace_attrs->inplace_indices[i].IntValue() == -1) {
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

      Array<Expr> args;
      if (call->args[1].as<TupleNode>()) {
        args = Downcast<Tuple>(call->args[1])->fields;
        // for call_tir_inplace, don't reinsert in-place args, only the newly allocated ones
        if (!is_inplace) {
          args.insert(args.end(), outs.begin(), outs.end());
        } else {
          for (size_t i = 0; i < outs.size(); i++) {
            if (inplace_attrs->inplace_indices[i].IntValue() == -1) {
              args.push_back(outs[i]);
            }
          }
        }

        if (call->args.size() == 2) {
          builder_->Emit(Call(call->args[0], args), "_");
        } else {
          // unpack semantics
          args.push_back(call->args[2]);
          builder_->Emit(Call(call_tir_dyn_op, {call->args[0], Tuple(args)}), "_");
        }
      } else {
        if (!is_inplace) {
          args = outs;
          args.insert(args.begin(), call->args[1]);
        } else {
          args.push_back(call->args[1]);
        }
        builder_->Emit(Call(call->args[0], args), "_");
      }

      if (outs.size() == 1) {
        return outs[0];
      }
      return std::move(Tuple(outs));
    }

    return GetRef<Expr>(call);
  }

  /*! \brief The context IRModule. */
  IRModule mod_;
};

namespace transform {

Pass CallTIRRewrite() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) { return CallTIRMutator(mod).Run(); };
  return CreateModulePass(/*pass_function=*/pass_func,
                          /*opt_level=*/0,
                          /*pass_name=*/"CallTIRRewrite",
                          /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.CallTIRRewrite").set_body_typed(CallTIRRewrite);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
