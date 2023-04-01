/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/utils.h>
#include <tvm/relay/op.h>

#include "op_common.h"

namespace tvm {
namespace relax {

// print

RELAY_REGISTER_OP("relax.print")
    .set_num_inputs(-1)
    .add_argument("vals", "Array<Expr>",
                  "The first value is Python-style format string to use to print. The others "
                  "are values to print")
    .set_attr<FCallPacked>("FCallPacked", "relax.run.print")
    .set_attr<FInferStructInfo>("FInferStructInfo", [](const Call& call, const BlockBuilder& ctx) {
      return TupleStructInfo(Array<StructInfo>());
    });

TVM_REGISTER_GLOBAL("relax.op.print").set_body_typed([](Array<Expr> vals, StringImm format) {
  Array<Expr> params;
  params.push_back(format);
  for (const auto val : vals) {
    params.push_back(val);
  }
  static const Op& op = Op::Get("relax.print");
  return Call(op, params);
});

// assert_op

// can't actually name it assert or else Python will consider it a syntax error

RELAY_REGISTER_OP("relax.assert_op")
    .set_num_inputs(-1)
    .add_argument("vals", "Array<Expr>",
                  "The first value is used as the assertion condition. The second value is "
                  "Python-style format string to use for displaying an error message, if the "
                  "assert fails. The others are used as format arguments if there is an error.")
    .set_attr<FCallPacked>("FCallPacked", "relax.run.assert_op")
    .set_attr<FInferStructInfo>("FInferStructInfo", [](const Call& call, const BlockBuilder& ctx) {
      // Ensure that the condition argument is a boolean scalar.
      // Also permitted is a tensor with unknown shape and unknown dtype
      // (checked dynamically in that case). Returns void.
      if (call->args.size() < 1) {
        ctx->ReportFatal(Diagnostic::Error(call)
                         << "Assert must have at least one argument (the condition).");
      }
      StructInfo arg_struct_info = GetStructInfo(call->args[0]);
      if (!IsBoolStructInfo(arg_struct_info)) {
        ctx->ReportFatal(Diagnostic::Error(call)
                         << "The argument to assert must be a boolean scalar, but received "
                         << arg_struct_info);
      }
      return TupleStructInfo(Array<StructInfo>());
    });

TVM_REGISTER_GLOBAL("relax.op.assert_op")
    .set_body_typed([](Expr condition, Array<Expr> vals, StringImm format) {
      static const Op& op = Op::Get("relax.assert_op");
      Array<Expr> args = {condition};
      args.push_back(format);
      for (auto val : vals) {
        args.push_back(val);
      }
      return Call(op, args);
    });

// tensor_to_shape

RELAY_REGISTER_OP("relax.tensor_to_shape")
    .set_num_inputs(1)
    .add_argument("input", "Expr", "The input expression")
    .set_attr<FInferStructInfo>("FInferStructInfo", [](const Call& call, const BlockBuilder& ctx) {
      ICHECK(call->args.size() == 1);
      ICHECK(call->args[0]->struct_info_.defined());
      const auto* info = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
      ICHECK(info && info->shape.defined());
      ShapeExpr shape_expr = Downcast<ShapeExpr>(info->shape.value());
      ICHECK(shape_expr->values.size() == 1);
      const IntImmNode* ndim = shape_expr->values[0].as<IntImmNode>();
      ICHECK(ndim);
      return ShapeStructInfo(ndim->value);
    });

TVM_REGISTER_GLOBAL("relax.op.tensor_to_shape").set_body_typed([](Expr expr) {
  static const Op& op = Op::Get("relax.tensor_to_shape");
  return Call(op, {expr}, {}, {});
});

}  // namespace relax
}  // namespace tvm
