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
 * \file col2im.cc
 * \brief Rearrange column to image operator
 */
#include <tvm/relay/attrs/image.h>

#include "../op_common.h"

namespace tvm {
namespace relay {

// relay.image.col2im
TVM_REGISTER_NODE_TYPE(Col2ImAttrs);

bool Col2ImRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  ICHECK_EQ(num_inputs, 3);
  ICHECK_EQ(types.size(), 4);
  auto data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    return false;
  }
  auto image_shape = types[1].as<TensorTypeNode>();
  if (image_shape == nullptr) {
    return false;
  }
  auto block_shape = types[2].as<TensorTypeNode>();
  if (block_shape == nullptr) {
    return false;
  }

  const auto* param = attrs.as<Col2ImAttrs>();
  ICHECK(param != nullptr);

  // TODO(vvchernov): Construct correct output shape from data shape and image_shape
  std::vector<IndexExpr> oshape;
  for (size_t i = 0; i < data->shape.size(); ++i) {
    oshape.push_back(data->shape[i]);
  }
  reporter->Assign(types[3], TensorType(oshape, data->dtype));
  return true;
}

Expr MakeCol2Im(Expr data, Expr image_shape, Expr block_shape,
    Array<Integer> dilation, Array<Integer> pads, Array<Integer> strides) {
  auto attrs = make_object<Col2ImAttrs>();
  attrs->dilation = std::move(dilation);
  attrs->pads = std::move(pads);
  attrs->strides = std::move(strides);
  static const Op& op = Op::Get("image.col2im");
  return Call(op, {data, image_shape, block_shape}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.image._make.col2im").set_body_typed(MakeCol2Im);

RELAY_REGISTER_OP("col2im")
    .describe(
        R"code(Rearrange column blocks back into a multidimensional image.)code" TVM_ADD_FILELINE)
    .set_attrs_type<Col2ImAttrs>()
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("image_shape", "Tensor",
        "The shape of the spatial dimensions of the image after rearranging the column blocks.")
    .add_argument("block_shape", "Tensor", "The shape of the block to apply on the input.")
    .add_type_rel("Col2Im", Col2ImRel)
    // TODO(vvchernov): correct pattern
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_support_level(10)

}  // namespace relay
}  // namespace tvm
