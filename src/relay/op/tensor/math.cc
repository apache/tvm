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
 * \file math.cc
 * \brief Math operators.
 */
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/topi/einsum.h>

#include "../make_op.h"
#include "../op_common.h"
#include "../type_relations.h"

namespace tvm {
namespace relay {

// relay.einsum
TVM_REGISTER_NODE_TYPE(EinsumAttrs);

bool EinsumRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  // Check attrs
  const EinsumAttrs* param = attrs.as<EinsumAttrs>();
  if (param == nullptr) {
    reporter->GetDiagCtx().EmitFatal(Diagnostic::Error(reporter->GetSpan())
                                     << "the call attributes are not defined");
    return false;
  }

  // types: [data, result]
  ICHECK_EQ(types.size(), 2) << "the arity of einsum is 2, not " << types.size();

  // Check input type is a tuple.
  const auto* tensor_tuple = types[0].as<TupleTypeNode>();
  if (tensor_tuple == nullptr) {
    reporter->GetDiagCtx().EmitFatal(
        Diagnostic::Error(reporter->GetSpan())
        << "einsum requires a tuple of tensors as the first argument, found "
        << PrettyPrint(types[0]));
    return false;
  }

  // Check the input tuple consists of tensors with consistent dtype.
  if (tensor_tuple->fields[0].as<IncompleteTypeNode>()) {
    return false;
  }
  ICHECK(tensor_tuple->fields[0].as<TensorTypeNode>());
  const auto& first = Downcast<TensorType>(tensor_tuple->fields[0]);
  const DataType dtype = first->dtype;
  std::vector<Array<PrimExpr>> input_shapes;
  for (const Type& ele : tensor_tuple->fields) {
    if (ele.as<IncompleteTypeNode>()) {
      return false;
    }

    const auto& e = Downcast<TensorType>(ele);

    const DataType& e_dtype = e->dtype;
    if (e_dtype != dtype) {
      throw Error("relay.einsum requires all tensors have the same dtype");
    }
    input_shapes.push_back(e->shape);
  }

  // Calculate output shape
  Array<IndexExpr> oshape = topi::InferEinsumShape(param->equation, input_shapes);

  auto rtype = TensorType(oshape, dtype);
  reporter->Assign(types[1], rtype);
  return true;
}

Array<te::Tensor> EinsumCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                const Type& out_type) {
  const EinsumAttrs* param = attrs.as<EinsumAttrs>();
  ICHECK(param != nullptr);
  return Array<te::Tensor>{topi::einsum(param->equation, inputs)};
}

Expr MakeEinsum(Expr data, String equation) {
  auto attrs = make_object<EinsumAttrs>();
  attrs->equation = std::move(equation);
  static const Op& op = Op::Get("einsum");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.einsum").set_body_typed(MakeEinsum);

RELAY_REGISTER_OP("einsum")
    .describe(R"doc(Evaluates the Einstein summation convention
on the operands)doc" TVM_ADD_FILELINE)
    .set_attrs_type<EinsumAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tuple of Tensors", "The input list of tensors.")
    .set_support_level(11)
    .add_type_rel("Einsum", EinsumRel)
    .set_attr<FTVMCompute>("FTVMCompute", EinsumCompute)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

}  // namespace relay
}  // namespace tvm
