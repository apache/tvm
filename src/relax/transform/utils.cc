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

#include "utils.h"

#include <tvm/relax/analysis.h>

namespace tvm {
namespace relax {

bool IsScalarTensor(const StructInfo& sinfo) {
  if (!sinfo->IsInstance<TensorStructInfoNode>()) {
    return false;
  }
  TensorStructInfo tensor_sinfo = Downcast<TensorStructInfo>(sinfo);
  if (!tensor_sinfo->shape.defined() || !tensor_sinfo->shape->IsInstance<ShapeExprNode>()) {
    return false;
  }
  return tensor_sinfo->shape.as<ShapeExprNode>()->values.size() == 0;
}

bool IsScalarTensor(const Expr& expr) { return IsScalarTensor(GetStructInfo(expr)); }

bool IsNestedTensor(const StructInfo& sinfo) {
  return IsNestedTensorConditioned(sinfo, [](const TensorStructInfo& sinfo) { return true; });
}

bool IsNestedTensor(const Expr& expr) { return IsNestedTensor(GetStructInfo(expr)); }

Function ComposeFunctions(Function func_a, Function func_b) {
  Array<Binding> bindings;

  Var func_a_output("func_a_output", func_a->ret_struct_info);

  bindings.push_back(VarBinding(func_a_output, func_a->body));

  auto func_a_outputs = [&]() -> Array<Expr> {
    if (auto func_a_output_tuple = func_a->ret_struct_info.as<TupleStructInfoNode>()) {
      Array<Expr> outputs;
      for (size_t i = 0; i < func_a_output_tuple->fields.size(); i++) {
        outputs.push_back(TupleGetItem(func_a_output, i));
      }
      return outputs;
    } else {
      return {func_a_output};
    }
  }();

  if (func_b->params.size() == 1 && func_b->params[0]->struct_info_.as<TupleStructInfoNode>()) {
    // Special case where the output of the first function is a tuple
    // that should be provided as-is to the second function, and
    // should not be unpacked into individual elements.
    auto param = func_b->params[0];
    bindings.push_back(MatchCast(param, func_a_output, GetStructInfo(param)));
  } else {
    CHECK_EQ(func_a_outputs.size(), func_b->params.size())
        << "ValueError: "
        << "Cannot compose functions together.  "
        << "First function produces " << func_a_outputs.size() << " values, "
        << "but second function expects " << func_b->params.size() << " parameters as input";
    for (size_t i = 0; i < func_a_outputs.size(); i++) {
      auto param = func_b->params[i];
      bindings.push_back(MatchCast(param, func_a_outputs[i], GetStructInfo(param)));
    }
  }

  auto new_body = SeqExpr({BindingBlock(bindings)}, func_b->body);

  auto new_function = Function(func_a->params, new_body, func_b->ret_struct_info,
                               func_a->is_pure && func_b->is_pure, func_a->attrs);

  new_function = CopyWithNewVars(new_function);
  new_function = Downcast<Function>(CanonicalizeBindings(new_function));
  new_function = Downcast<Function>(RemoveAllUnused(new_function));

  return new_function;
}

}  // namespace relax
}  // namespace tvm
