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
 * \file src/relay/op/contrib/gemmini/max_pool2d.cc
 * \brief 2D max pool operator definition for Gemmini.
 * \author Federico Peccia <https://fPecc.github.io/>
 */
#include <tvm/relay/op.h>

#include "../../../qnn/utils.h"
#include "../../op_common.h"
//#include "common.h"

namespace tvm {
namespace relay {
namespace op {
namespace contrib {
namespace gemmini {

/*! \brief Attributes used by the Gemmini GEMM operators */
struct GemminiMaxPool2DAttrs : public tvm::AttrsNode<GemminiMaxPool2DAttrs> {
  Array<IndexExpr> pool_size;
  Array<IndexExpr> pool_strides;
  Array<IndexExpr> pool_dilation;
  Array<IndexExpr> pool_padding;
  Array<PrimExpr> shape;

  TVM_DECLARE_ATTRS(GemminiMaxPool2DAttrs, "relay.attrs.GemminiMaxPool2DAttrs") {
    TVM_ATTR_FIELD(pool_size).describe("Pooling window size");
    TVM_ATTR_FIELD(pool_strides).describe("Pooling window strides");
    TVM_ATTR_FIELD(pool_dilation).describe("Pooling window dilation");
    TVM_ATTR_FIELD(pool_padding).describe("Pooling padding");
    TVM_ATTR_FIELD(shape).describe("Input shape");
  }
};

TVM_REGISTER_NODE_TYPE(GemminiMaxPool2DAttrs);

bool GemminiMaxPool2DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                         const TypeReporter& reporter) {
  const int data_index = 0;
  const int result_index = 2;

  const auto* data = types[data_index].as<TensorTypeNode>();
  if (data == nullptr) return false;

  const auto* params = attrs.as<GemminiMaxPool2DAttrs>();
  ICHECK(params != nullptr) << "GemminiMaxPool2DAttrs cannot be nullptr.";

  DataType ofm_dtype = DataType::Int(8);

  // Assign ofm type
  PrimExpr max_pool2d_h = ((data->shape[1] + (params->pool_padding[0] + params->pool_padding[2]) -
                            params->pool_size[0]) /
                           params->pool_strides[0]) +
                          1;
  PrimExpr max_pool2d_w = ((data->shape[2] + (params->pool_padding[1] + params->pool_padding[3]) -
                            params->pool_size[1]) /
                           params->pool_strides[1]) +
                          1;
  Array<IndexExpr> ofm_shape({data->shape[0], max_pool2d_h, max_pool2d_w, data->shape[3]});
  reporter->Assign(types[result_index], TensorType(ofm_shape, ofm_dtype));
  return true;
}

Expr MakeGemminiMaxPool2D(Expr data, Array<IndexExpr> pool_size, Array<IndexExpr> pool_strides,
                          Array<IndexExpr> pool_dilation, Array<IndexExpr> pool_padding,
                          Array<PrimExpr> shape) {
  auto attrs = make_object<GemminiMaxPool2DAttrs>();
  attrs->pool_size = std::move(pool_size);
  attrs->pool_strides = std::move(pool_strides);
  attrs->pool_dilation = std::move(pool_dilation);
  attrs->pool_padding = std::move(pool_padding);
  attrs->shape = std::move(shape);

  static const Op& op = Op::Get("contrib.gemmini.max_pool2d");

  // Trick to be able to accelerate the max pooling operation using the dw convolution function of
  // Gemmini ;)
  auto weights =
      Full(MakeConstantScalar(DataType::Int(8), 1), {attrs->shape[3], 1, 1}, DataType::Int(8));

  auto max_pool2d_output = Call(op, {data, weights}, Attrs(attrs), {});

  return max_pool2d_output;
}

TVM_REGISTER_GLOBAL("relay.op._make.gemmini_max_pool2d").set_body_typed(MakeGemminiMaxPool2D);

RELAY_REGISTER_OP("contrib.gemmini.max_pool2d")
    .describe("Gemmini 2D max pooling operator")
    .set_attrs_type<GemminiMaxPool2DAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The Input Feature Map tensor.")
    .add_argument("weights", "Tensor", "The Weights dummy tensor.")
    .set_support_level(11)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .add_type_rel("GemminiMaxPool2D", GemminiMaxPool2DRel);

}  // namespace gemmini
}  // namespace contrib
}  // namespace op
}  // namespace relay
}  // namespace tvm
