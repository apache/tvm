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
// Perform explicit tensor allocation for call_tir.
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
    static const Op& alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");
    static const Op& call_tir_dyn_op = Op::Get("relax.vm.call_tir_dyn");

    if (call->op == call_tir_op) {
      Array<Expr> outs;
      if (const auto& _tensor_sinfo = MatchStructInfo<TensorStructInfo>(expr)) {
        // single output case
        const TensorStructInfo& tensor_sinfo = _tensor_sinfo.value();
        ICHECK(tensor_sinfo->shape.defined())
            << "the TensorStructInfo shape of call_tir has not populated";
        outs.push_back(
            builder_->Emit(Call(alloc_tensor_op,  //
                                {Downcast<ShapeExpr>(tensor_sinfo->shape.value()),
                                 DataTypeImm(tensor_sinfo->dtype), PrimValue::Int64(0)},  //
                                Attrs()),
                           "alloc"));
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
          outs.push_back(
              builder_->Emit(Call(alloc_tensor_op,
                                  {Downcast<ShapeExpr>(field_tensor->shape.value()),
                                   DataTypeImm(field_tensor->dtype), PrimValue::Int64(0)},
                                  Attrs()),
                             "alloc"));
        }
      } else {
        LOG(FATAL) << "TypeError: The struct info of call_tir expects to be TensorStructInfo or "
                      "TupleStructInfo, but got"
                   << expr->struct_info_;
      }

      Array<Expr> args;
      if (call->args[1].as<TupleNode>()) {
        args = Downcast<Tuple>(call->args[1])->fields;
        args.insert(args.end(), outs.begin(), outs.end());

        if (call->args.size() == 2) {
          builder_->Emit(Call(call->args[0], args), "_");
        } else {
          // unpack semantics
          args.push_back(call->args[2]);
          builder_->Emit(Call(call_tir_dyn_op, {call->args[0], Tuple(args)}), "_");
        }
      } else {
        args = outs;
        args.insert(args.begin(), call->args[1]);
        builder_->Emit(Call(call->args[0], args), "_");
      }

      if (outs.size() == 1) {
        return outs[0];
      }
      return std::move(Tuple(outs));
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
