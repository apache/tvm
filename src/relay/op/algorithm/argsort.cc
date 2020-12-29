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
 * \file argsort.cc
 * \brief Argsort operators
 */
#include <tvm/relay/attrs/algorithm.h>
#include <tvm/relay/op.h>

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(ArgsortAttrs);

bool ArgsortRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                const TypeReporter& reporter) {
  // `types` contains: [data, result]
  const ArgsortAttrs* param = attrs.as<ArgsortAttrs>();
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "Argsort: expect input type to be TensorType but get " << types[0];
    return false;
  }
  reporter->Assign(types[1], TensorType(data->shape, param->dtype));
  return true;
}

Expr MakeArgsort(Expr data, int axis, bool is_ascend, DataType dtype) {
  auto attrs = make_object<ArgsortAttrs>();
  attrs->axis = axis;
  attrs->is_ascend = is_ascend;
  attrs->dtype = dtype;
  static const Op& op = Op::Get("argsort");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.argsort").set_body_typed(MakeArgsort);

RELAY_REGISTER_OP("argsort")
    .describe(R"doc(Returns the indices that would sort an
input array along the given axis.
)doc" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .set_attrs_type<ArgsortAttrs>()
    .add_argument("data", "Tensor", "Input data.")
    .set_support_level(6)
    .add_type_rel("Argsort", ArgsortRel);

}  // namespace relay
}  // namespace tvm
