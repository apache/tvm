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
 *  Copyright (c) 2018 by Contributors
 * \file resize.cc
 * \brief Image operators
 */
#include <tvm/data_layout.h>
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/image.h>
#include <topi/elemwise.h>
#include <topi/image/resize.h>
#include "../op_common.h"

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(ResizeAttrs);

bool ResizeRel(const Array<Type>& types,
               int num_inputs,
               const Attrs& attrs,
               const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  static const Layout kNCHW("NCHW");

  const ResizeAttrs* param = attrs.as<ResizeAttrs>();
  CHECK(param != nullptr);
  const Layout in_layout(param->layout);
  auto layout_converter = BijectiveLayoutNode::make(in_layout, kNCHW);
  CHECK(layout_converter.defined())
    << "Resize only support input layouts that are convertible from NCHW."
    << " But got " << in_layout;

  auto oshape = layout_converter.ForwardShape(data->shape);
  oshape.Set(2, param->size[0]);
  oshape.Set(3, param->size[1]);

  // assign output type
  reporter->Assign(types[1],
                   TensorTypeNode::make(layout_converter.BackwardShape(oshape),
                                        data->dtype));
  return true;
}

Array<Tensor> ResizeCompute(const Attrs& attrs,
                            const Array<Tensor>& inputs,
                            const Type& out_type,
                            const Target& target) {
  const auto* param = attrs.as<ResizeAttrs>();
  CHECK(param != nullptr);
  CHECK(param->layout == "NCHW" || param->layout == "NHWC");
  const auto* out_ttype = out_type.as<TensorTypeNode>();
  CHECK(out_ttype != nullptr);
  Array<IndexExpr> oshape;
  if (param->layout == "NCHW") {
    oshape.push_back(out_ttype->shape[2]);
    oshape.push_back(out_ttype->shape[3]);
  } else if (param->layout == "NHWC") {
    oshape.push_back(out_ttype->shape[1]);
    oshape.push_back(out_ttype->shape[2]);
  }
  return Array<Tensor>{ topi::image::resize(inputs[0],
                                            oshape,
                                            param->layout,
                                            param->align_corners,
                                            param->method) };
}

// Positional relay function to create image operator
// used by frontend FFI.
Expr MakeResize(Expr data,
                Array<IndexExpr> size,
                std::string layout,
                std::string method,
                bool align_corners) {
  auto attrs = make_node<ResizeAttrs>();
  attrs->size = std::move(size);
  attrs->layout = std::move(layout);
  attrs->method = std::move(method);
  attrs->align_corners = align_corners;
  static const Op& op = Op::Get("image.resize");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op.image._make.resize")
.set_body_typed(MakeResize);


RELAY_REGISTER_OP("image.resize")
.describe(R"code(Perform resize to input array with nearest neighbour or bilinear interpolation.

- **data**: data is 4D array of shape
            (batch_size, channels, in_height, in_width) for NCHW
            (batch_size, in_height, in_width, channels) for NHWC

- **out**: Output is 4D array of shape
           for layout NCHW
           (batch_size, channels, size[0], size[1])

           for layout NHWC
           (batch_size, size[0], size[1], channels)
)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.ResizeAttrs")
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(5)
.add_type_rel("Resize", ResizeRel)
.set_attr<FTVMCompute>("FTVMCompute", ResizeCompute)
.set_attr<TOpPattern>("TOpPattern", kInjective);

}  // namespace relay
}  // namespace tvm
