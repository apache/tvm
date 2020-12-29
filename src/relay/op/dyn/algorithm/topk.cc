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
 * \file topk.cc
 * \brief TopK operators
 */
#include <tvm/relay/attrs/algorithm.h>
#include <tvm/relay/op.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace relay {
namespace dyn {

bool TopKRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
             const TypeReporter& reporter) {
  // `types` contains: [data, k, result]
  const TopKAttrs* param = attrs.as<TopKAttrs>();
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* k = types[1].as<TensorTypeNode>();
  if (data == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "tile: expect input type to be TensorType but get " << types[0];
    return false;
  }
  if (k == nullptr) {
    ICHECK(types[1].as<IncompleteTypeNode>())
        << "tile: expect input type to be TensorType but get " << types[1];
    return false;
  }
  ICHECK(k->shape.size() <= 1) << "Parameter k must be a Scalar or a Tensor of shape (1, )";
  if (k->shape.size() == 1) {
    const IntImmNode* k_shape = k->shape[0].as<IntImmNode>();
    ICHECK(k_shape) << "Parameter k must have static shape";
    ICHECK_EQ(k_shape->value, 1) << "Parameter k must be a Scalar or a Tensor of shape (1, )";
  }
  int ndim = data->shape.size();
  int axis = param->axis;
  if (axis < 0) {
    axis += ndim;
  }
  ICHECK(axis >= 0 && axis < ndim);
  Array<IndexExpr> out_shape;
  for (int i = 0; i < ndim; ++i) {
    if (i != axis) {
      out_shape.push_back(data->shape[i]);
    } else {
      out_shape.push_back(Any());
    }
  }
  auto values_ty = TensorType(out_shape, data->dtype);
  auto indices_ty = TensorType(out_shape, param->dtype);
  if (param->ret_type == "both") {
    reporter->Assign(types[2], TupleType({values_ty, indices_ty}));
  } else if (param->ret_type == "values") {
    reporter->Assign(types[2], values_ty);
  } else if (param->ret_type == "indices") {
    reporter->Assign(types[2], indices_ty);
  } else {
    LOG(FATAL) << "Unsupported ret type: " << param->ret_type;
  }
  return true;
}

Expr MakeTopK(Expr data, Expr k, int axis, String ret_type, bool is_ascend, DataType dtype) {
  auto attrs = make_object<TopKAttrs>();
  attrs->axis = axis;
  attrs->ret_type = ret_type;
  attrs->is_ascend = is_ascend;
  attrs->dtype = dtype;
  static const Op& op = Op::Get("dyn.topk");
  return Call(op, {data, k}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.dyn._make.topk").set_body_typed(MakeTopK);

RELAY_REGISTER_OP("dyn.topk")
    .describe(R"doc(Get the top k elements in an input tensor along the given axis.
)doc" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .set_attrs_type<TopKAttrs>()
    .add_argument("data", "Tensor", "Input data.")
    .add_argument("k", "Tensor", "Number of top elements.")
    .set_support_level(6)
    .add_type_rel("DynTopK", TopKRel);

}  // namespace dyn
}  // namespace relay
}  // namespace tvm
