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
 *  Copyright (c) 2019 by Contributors
 * \file argwhere.cc
 * \brief Argwhere operators
 */
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/algorithm.h>
#include <tvm/relay/op_attr_types.h>

namespace tvm {
namespace relay {

// ArgWhere
bool ArgWhereRel(const Array<Type>& types,
                  int num_inputs,
                  const Attrs& attrs,
                  const TypeReporter& reporter) {
  CHECK_EQ(num_inputs, 1);
  auto tt = types[0].as<TensorTypeNode>();
  CHECK(tt != nullptr);
  const auto& input_shape = tt->shape;
  const auto& input_rank = input_shape.size();
  std::vector<IndexExpr> result_shape;
  result_shape.push_back(Any::make());
  result_shape.push_back(IntImm::make(Int(32), input_rank));
  reporter->Assign(types[1], TensorTypeNode::make(result_shape, Int(32)));
  return true;
}

TVM_REGISTER_API("relay.op._make.argwhere")
.set_body_typed<Expr(Expr)>([](Expr data) {
  static const Op& op = Op::Get("argwhere");
  auto attrs = make_node<ArgWhereAttrs>();
  return CallNode::make(op, {data}, Attrs(attrs), {});
});

RELAY_REGISTER_OP("argwhere")
.describe(R"doc(Find the indices of elements of a tensor that are
non-zero)doc" TVM_ADD_FILELINE)
.set_num_inputs(1)
.set_attrs_type_key("relay.attrs.ArgWhereAttrs")
.add_argument("condition", "Tensor", "The input condition tensor.")
.add_type_rel("ArgWhere", ArgWhereRel)
.set_attr<TOpIsStateful>("TOpIsStateful", false)
.set_attr<TOpPattern>("TOpPattern", kOpaque)
.set_support_level(10);

}  // namespace relay
}  // namespace tvm
