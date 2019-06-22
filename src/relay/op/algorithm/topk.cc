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
 * \file topk.cc
 * \brief TopK operators
 */
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/algorithm.h>

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(TopKAttrs);

bool TopKRel(const Array<Type>& types,
             int num_inputs,
             const Attrs& attrs,
             const TypeReporter& reporter) {
  // `types` contains: [data, result]
  const TopKAttrs* param = attrs.as<TopKAttrs>();
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  CHECK(data);
  int ndim = data->shape.size();
  int axis = param->axis;
  if (axis < 0) {
    axis += ndim;
  }
  CHECK(axis >= 0 && axis < ndim);
  Array<IndexExpr> out_shape;
  for (int i = 0; i < ndim; ++i) {
    if (i != axis || param->k < 1) {
      out_shape.push_back(data->shape[i]);
    } else {
      out_shape.push_back(param->k);
    }
  }
  auto values_ty = TensorTypeNode::make(out_shape, data->dtype);
  auto indices_ty = TensorTypeNode::make(out_shape, param->dtype);
  if (param->ret_type == "both") {
    reporter->Assign(types[1], TupleTypeNode::make({values_ty, indices_ty}));
  } else if (param->ret_type == "values") {
    reporter->Assign(types[1], values_ty);
  } else if (param->ret_type == "indices") {
    reporter->Assign(types[1], indices_ty);
  } else {
    LOG(FATAL) << "Unsupported ret type: " << param->ret_type;
  }
  return true;
}

Expr MakeTopK(Expr data,
              int k,
              int axis,
              std::string ret_type,
              bool is_ascend,
              DataType dtype) {
  auto attrs = make_node<TopKAttrs>();
  attrs->k = k;
  attrs->axis = axis;
  attrs->ret_type = ret_type;
  attrs->is_ascend = is_ascend;
  attrs->dtype = dtype;
  static const Op& op = Op::Get("topk");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op._make.topk")
.set_body_typed(MakeTopK);

RELAY_REGISTER_OP("topk")
.describe(R"doc(Get the top k elements in an input tensor along the given axis.
)doc" TVM_ADD_FILELINE)
.set_num_inputs(1)
.set_attrs_type_key("relay.attrs.TopKAttrs")
.add_argument("data", "Tensor", "Input data.")
.set_support_level(6)
.add_type_rel("TopK", TopKRel);

}  // namespace relay
}  // namespace tvm

