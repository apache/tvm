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
#include <tvm/tir/data_layout.h>
#include <tvm/tir/op.h>

#include "../../transforms/infer_layout_utils.h"

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(TopKAttrs);

InferCorrectLayoutOutput TopKInferCorrectLayout(const Attrs& attrs,
                                                const Array<Layout>& new_in_layouts,
                                                const Array<Layout>& old_in_layouts,
                                                const Array<tvm::relay::Type>& old_in_types) {
  const auto* attrs_ptr = attrs.as<TopKAttrs>();
  ICHECK(attrs_ptr);
  ObjectPtr<TopKAttrs> param = make_object<TopKAttrs>(*attrs_ptr);

  Array<Array<IndexExpr>> old_in_shapes;
  for (auto old_in_t : old_in_types) {
    ICHECK(old_in_t.as<TensorTypeNode>());
    old_in_shapes.push_back(old_in_t.as<TensorTypeNode>()->shape);
  }

  size_t axis =
      param->axis < 0 ? param->axis + old_in_shapes[0].size() : static_cast<size_t>(param->axis);

  Layout ret = Layout::Undef();

  // If new_in_layouts are defined, this code tries to modify the layout.
  if (new_in_layouts.defined() && old_in_layouts.defined()) {
    const auto& sp_dim = old_in_layouts[0][axis];
    auto new_index = new_in_layouts[0].IndexOf(sp_dim);
    param->axis = new_index;
    ret = new_in_layouts[0];
  } else if (old_in_layouts.defined()) {
    ret = old_in_layouts[0];
  }

  // TopK has 2 outputs, Values and Indices
  return InferCorrectLayoutOutput({ret}, {ret, ret}, Attrs(param));
}

bool TopKRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
             const TypeReporter& reporter) {
  // `types` contains: [data, result]
  const TopKAttrs* param = attrs.as<TopKAttrs>();
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
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
      const Integer& ck = param->k.value();
      if (ck->value < 1) {
        out_shape.push_back(data->shape[i]);
      } else {
        out_shape.push_back(ck);
      }
    }
  }
  auto values_ty = TensorType(out_shape, data->dtype);
  auto indices_ty = TensorType(out_shape, param->dtype);
  if (param->ret_type == "both") {
    reporter->Assign(types[1], TupleType({values_ty, indices_ty}));
  } else if (param->ret_type == "values") {
    reporter->Assign(types[1], values_ty);
  } else if (param->ret_type == "indices") {
    reporter->Assign(types[1], indices_ty);
  } else {
    LOG(FATAL) << "Unsupported ret type: " << param->ret_type;
  }
  return true;
}

Expr MakeTopK(Expr data, int k, int axis, String ret_type, bool is_ascend, DataType dtype) {
  auto attrs = make_object<TopKAttrs>();
  attrs->k = Integer(k);
  attrs->axis = axis;
  attrs->ret_type = ret_type;
  attrs->is_ascend = is_ascend;
  attrs->dtype = dtype;
  static const Op& op = Op::Get("topk");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.topk").set_body_typed(MakeTopK);

RELAY_REGISTER_OP("topk")
    .describe(R"doc(Get the top k elements in an input tensor along the given axis.
)doc" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .set_attrs_type<TopKAttrs>()
    .add_argument("data", "Tensor", "Input data.")
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", TopKInferCorrectLayout)
    .set_support_level(6)
    .add_type_rel("TopK", TopKRel);

}  // namespace relay
}  // namespace tvm
